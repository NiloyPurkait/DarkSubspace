#!/usr/bin/env python3
"""standard_mia_probe_decomposition.py.

SAE patching decomposition with three standard MIA probes (loss attack,
MIN-K%=20, zlib) on Pythia-6.9B fine-tuned for 5 epochs, to test whether
the layer-local effect appears at the LM head.

Used in the standard-probes appendix.
Reproduce:
    env/bin/python3 scripts/memcirc/standard_mia_probe_decomposition.py \
        --model-path runs/controlled_ft/run_20260306_055225/ft_epoch5/model \
        --sae-path runs/sae/<trained_sae>/sae_final.pt \
        --layer 16 \
        --member-texts data/memcirc_ctrl_ft/member.jsonl \
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \
        --output-dir runs/memcirc/standard_mia_probes/p69_dark_subspace_replication

For each of three patching conditions (h_orig, h_recon, h_residual) and three
standard probes (loss attack, MIN-K%=20, zlib), compute the AUROC on
1000 member + 1000 nonmember texts. The patch is installed as a forward hook
on layer 16 of the model. The SAE is applied to every token's residual-stream
activation, producing the recon and the residual.

Probes.
- loss        : per-text mean cross-entropy (Yeom 2018, Carlini 2022).
- min_k_20    : mean of bottom 20 percent token log-probabilities (Shi 2023).
- zlib        : per-text mean cross-entropy divided by zlib-compressed length
                (Carlini 2021).

Patching conditions.
- h_orig      : no patching (control, sanity check vs unpatched run).
- h_recon     : layer 16 residual stream replaced with SAE.decode(SAE.encode(h)).
- h_residual  : layer 16 residual stream replaced with h - SAE.reconstruct(h).

Hypothesis. The dark-subspace pattern (h_recon below h_orig, h_residual above
h_recon) replicates across the three published probes.
"""

import _bootstrap  # noqa: F401
import argparse
import json
import logging
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from sae_mia_audit.models.wrapper import load_model_and_tokenizer
from sae_mia_audit.utils.hf import HFModelSpec
from sae_mia_audit.utils.seed import set_global_seed, SeedConfig
from sae_mia_audit.utils.logging import setup_logging, get_logger
from sae_mia_audit.sae.io import load_sae_checkpoint_any

setup_logging(level=logging.INFO)
log = get_logger(__name__)


def auroc_bi(y_true: np.ndarray, y_score: np.ndarray):
    finite = np.isfinite(y_score)
    if not finite.all():
        med = float(np.median(y_score[finite])) if finite.any() else 0.0
        y_score = np.where(finite, y_score, med)
    a_pos = float(roc_auc_score(y_true, y_score))
    a_neg = float(roc_auc_score(y_true, -y_score))
    if a_neg > a_pos:
        return a_neg, -1
    return a_pos, +1


def load_jsonl(path):
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(obj.get("text", ""))
    return out


def find_layer_module(model, layer_idx: int):
    """Return the layer-`layer_idx` transformer block whose output we patch."""
    for path in [
        ("gpt_neox", "layers"),
        ("transformer", "h"),
        ("model", "layers"),
    ]:
        cur = model
        ok = True
        for p in path:
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok:
            return cur[layer_idx]
    raise RuntimeError(f"Could not locate layer module at index {layer_idx}")


def sae_reconstruct(sae, h: torch.Tensor) -> torch.Tensor:
    """Encode + decode (B, T, D) -> (B, T, D) reconstruction."""
    flat = h.reshape(-1, h.shape[-1])
    z = sae.encode(flat)
    r = sae.decode(z)
    return r.reshape(h.shape)


def install_patch_hook(layer_module, sae, mode: str, device, dtype):
    """Install a forward hook that rewrites the layer output.

    mode:
      - 'orig'     : passthrough (no patch).
      - 'recon'    : output -> SAE.decode(SAE.encode(output)).
      - 'residual' : output -> output - SAE.decode(SAE.encode(output)).
    """
    def hook(module, inputs, output):
        if mode == "orig":
            return output
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        h_in = h.to(dtype=dtype)
        with torch.no_grad():
            recon = sae_reconstruct(sae, h_in)
        if mode == "recon":
            new_h = recon.to(h.dtype)
        elif mode == "residual":
            new_h = (h_in - recon).to(h.dtype)
        else:
            raise ValueError(mode)
        if rest is None:
            return new_h
        return (new_h,) + rest
    return layer_module.register_forward_hook(hook)


def per_token_logprobs(model, tokenizer, text: str, device, seq_len: int):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return None
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0, :-1, :]
    targets = input_ids[0, 1:]
    log_probs_full = F.log_softmax(logits.float(), dim=-1)
    tok_lp = log_probs_full.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp.detach().cpu().numpy()


def probe_metrics_from_lps(per_text_lps, texts):
    """Compute (loss, min_k_20, zlib) score arrays from lists of per-token log-probs."""
    loss_scores = []
    mink20_scores = []
    zlib_scores = []
    for lps, text in zip(per_text_lps, texts):
        if lps is None or len(lps) == 0:
            loss_scores.append(np.nan)
            mink20_scores.append(np.nan)
            zlib_scores.append(np.nan)
            continue
        nll = -lps
        loss = float(nll.mean())
        loss_scores.append(loss)
        k = max(1, int(0.2 * len(lps)))
        bottom = np.sort(lps)[:k]
        mink20_scores.append(float(bottom.mean()))
        zbytes = len(zlib.compress(text.encode("utf-8", errors="replace")))
        zlib_scores.append(float(loss / max(1.0, zbytes)))
    return {
        "loss": np.array(loss_scores),
        "min_k_20": np.array(mink20_scores),
        "zlib": np.array(zlib_scores),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--sae-path", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--member-texts", required=True)
    p.add_argument("--nonmember-texts", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-tag", default="p69")
    args = p.parse_args()

    set_global_seed(SeedConfig(seed=args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading model {args.model_path}")
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype="bfloat16", device=args.device)
    wrapper = load_model_and_tokenizer(spec)
    model, tokenizer = wrapper.model, wrapper.tokenizer
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    log.info(f"Loading SAE {args.sae_path}")
    sae_module = load_sae_checkpoint_any(args.sae_path, device=device)
    sae_module.eval()
    for p_ in sae_module.parameters():
        p_.requires_grad_(False)

    layer_mod = find_layer_module(model, args.layer)
    log.info(f"Patching at layer {args.layer}: {type(layer_mod).__name__}")

    member_texts = load_jsonl(args.member_texts)
    nonmember_texts = load_jsonl(args.nonmember_texts)
    if args.limit > 0:
        member_texts = member_texts[: args.limit]
        nonmember_texts = nonmember_texts[: args.limit]
    texts = member_texts + nonmember_texts
    labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts))
    log.info(f"Eval pool: {len(member_texts)} members + {len(nonmember_texts)} non-members")

    conditions = ["orig", "recon", "residual"]
    results = {
        "model_tag": args.model_tag,
        "model_path": args.model_path,
        "sae_path": args.sae_path,
        "layer": args.layer,
        "n_member": len(member_texts),
        "n_nonmember": len(nonmember_texts),
        "seq_len": args.seq_len,
        "seed": args.seed,
        "conditions": {},
        "schema_version": "standard_mia_probe_decomposition_v1",
    }

    for cond in conditions:
        log.info(f"=== Condition: {cond} ===")
        handle = install_patch_hook(layer_mod, sae_module, cond, device, dtype)
        try:
            per_text_lps = []
            for t in tqdm(texts, desc=f"forward[{cond}]"):
                per_text_lps.append(per_token_logprobs(model, tokenizer, t, device, args.seq_len))
            scores_by_probe = probe_metrics_from_lps(per_text_lps, texts)
        finally:
            handle.remove()

        cond_block = {}
        for probe_name, scores in scores_by_probe.items():
            auroc, sign = auroc_bi(labels, scores)
            cond_block[probe_name] = {
                "auroc": float(auroc),
                "orientation_sign": int(sign),
                "scores": scores.tolist(),
            }
            log.info(f"  [{cond}/{probe_name}] AUROC = {auroc:.4f} (sign {sign:+d})")
        results["conditions"][cond] = cond_block

    # Summary deltas: drop = AUROC(orig) - AUROC(recon); residual_recovery = AUROC(residual) - AUROC(recon).
    summary = {}
    for probe_name in ["loss", "min_k_20", "zlib"]:
        a_orig = results["conditions"]["orig"][probe_name]["auroc"]
        a_recon = results["conditions"]["recon"][probe_name]["auroc"]
        a_resid = results["conditions"]["residual"][probe_name]["auroc"]
        summary[probe_name] = {
            "auroc_orig": a_orig,
            "auroc_recon": a_recon,
            "auroc_residual": a_resid,
            "drop_orig_minus_recon": a_orig - a_recon,
            "residual_recovery_residual_minus_recon": a_resid - a_recon,
            "dark_subspace_replicates": bool((a_orig - a_recon) > 0.05 and a_resid > 0.55),
        }
    results["summary"] = summary

    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
