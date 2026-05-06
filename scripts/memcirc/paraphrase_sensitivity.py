#!/usr/bin/env python3
"""paraphrase_sensitivity.py.

Word-order-shuffled paraphrase audit on Pythia-6.9B, Qwen2-7B, and
Pythia-12B. Reports fixed-orientation versus bidirectional AUROC and the
orientation-flip diagnostic. Probes the paraphrase confound from Duan
et al. COLM 2024.

Used in the paraphrase audit appendix.
Reproduce:
  env/bin/python3 scripts/memcirc/paraphrase_sensitivity.py \\
      --model-tag p69 \\
      --model-path runs/controlled_ft/run_20260306_055225/ft_epoch5/model \\
      --bcd-dir runs/memcirc/behavioral_channels/p69_epoch5 \\
      --sae-path runs/sae/<trained_sae>/sae_final.pt \\
      --layer 16 \\
      --member-texts data/memcirc_ctrl_ft/member.jsonl \\
      --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \\
      --output-dir runs/memcirc/paraphrase_sensitivity \\
      --mode word_shuffle

Tests whether the residual-direction membership signal is preserved under
paraphrase of member texts. If the signal is truly geometric (lives in an
abstract content-feature subspace), it should survive paraphrase. If it is
surface memorization (verbatim token identity), it should collapse toward
chance under paraphrase.

Paraphrase modes.
  --mode t5_paraphrase
      Loads Vamsi/T5_Paraphrase_Paws (released alongside Google PAWS-QQP).
      Per-member text yields 1 paraphrase (beam=4). GPU required, around
      30 min for 1000 texts. Higher quality but depends on the
      tokenizer/model being downloadable.

  --mode word_shuffle       (default, CPU, deterministic, works offline)
      Deterministic syntactic perturbation. For each sentence, shuffle
      tokens within a rolling window of size 3, keeping every 4th token as
      an anchor. This is NOT a true paraphrase, it is a minimum-viable
      stress test that preserves the word-identity multiset while
      scrambling local order. Reviewer-facing framing is "conservative
      syntactic perturbation, a stronger test would use a model-based
      paraphrase".

  --mode backtranslation_cached
      Only works if data/memcirc_ctrl_ft/member_paraphrased.jsonl already
      exists. Looks up per-text paraphrase from that file.

For each mode we score.
  - residual_d_K AUROC and TPR@1%FPR on ORIGINAL member vs original nonmember.
  - residual_d_K AUROC and TPR@1%FPR on PARAPHRASED member vs original nonmember.
  - delta (paraphrase - original) for each metric with paired bootstrap 95% CI.

Output. runs/memcirc/paraphrase_sensitivity/p69/results.json,
        runs/memcirc/paraphrase_sensitivity/p69/per_text_scores.json,
        runs/memcirc/paraphrase_sensitivity/p69/paraphrase_cache.json
        (mode=t5_paraphrase only).
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from sae_mia_audit.models.wrapper import load_model_and_tokenizer
from sae_mia_audit.utils.hf import HFModelSpec
from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
from sae_mia_audit.utils.logging import setup_logging, get_logger
from sae_mia_audit.sae.io import load_sae_checkpoint_any
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch

from sklearn.metrics import roc_auc_score, roc_curve

setup_logging(level=logging.INFO)
log = get_logger(__name__)


# ---------- paraphrase modes ----------

def perturb_word_shuffle(text: str, seed: int) -> str:
    """Conservative syntactic perturbation: rolling-window local shuffle.

    Splits on whitespace, slides a window of size 3, shuffles within the window
    deterministically (rng seeded on the text hash + the global seed).
    Every 4th token is kept as an anchor (not part of a shuffle group).
    This preserves the multiset of words (a weaker perturbation than
    synonym replacement) while scrambling local order.
    """
    rng = random.Random((hash(text) ^ seed) & 0xFFFFFFFF)
    toks = text.split()
    n = len(toks)
    out = list(toks)
    i = 0
    while i + 3 <= n:
        if i % 4 == 3:
            i += 1
            continue
        group = out[i:i + 3]
        rng.shuffle(group)
        out[i:i + 3] = group
        i += 3
    return " ".join(out)


def _load_t5_paraphraser(device: str):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tok = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    mdl = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(device)
    return tok, mdl


def paraphrase_t5(texts: List[str], device: str, batch: int = 8, max_len: int = 256) -> List[str]:
    tok, mdl = _load_t5_paraphraser(device)
    out = []
    for i in tqdm(range(0, len(texts), batch), desc="t5 paraphrase"):
        chunk = texts[i:i + batch]
        prompts = [f"paraphrase: {t} </s>" for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            gen = mdl.generate(
                **enc, max_length=max_len, num_beams=4, num_return_sequences=1,
                early_stopping=True, no_repeat_ngram_size=2,
            )
        for g in gen:
            out.append(tok.decode(g, skip_special_tokens=True))
    return out


def load_cached(path: Path, expected_n: int) -> List[str]:
    data = [json.loads(l)["text"] for l in path.read_text().splitlines() if l.strip()]
    if len(data) != expected_n:
        raise ValueError(f"cached paraphrase count {len(data)} != expected {expected_n}")
    return data


# ---------- metrics ----------

def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def auroc_bi(y_true, y_score):
    a_pos = float(roc_auc_score(y_true, y_score))
    a_neg = float(roc_auc_score(y_true, -y_score))
    if a_neg > a_pos:
        return a_neg, -1
    return a_pos, +1


def paired_bootstrap(y_true, s_a, s_b, fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            diffs.append(fn(y_true[idx], s_a[idx]) - fn(y_true[idx], s_b[idx]))
        except Exception:
            continue
    arr = np.array(diffs) if diffs else np.array([float("nan")])
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------- activation+residual scoring (duplicated from baseline_attacks_suite for modularity) ----------

def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


@torch.no_grad()
def collect_pooled_acts(wrapper, texts, layer, seq_len, batch_size, device):
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=False)
    acts = []
    for chunk in tqdm(list(_batched(texts, batch_size)), desc=f"pool layer-{layer}"):
        batch = tokenize_batch(wrapper.tokenizer, chunk, tok_cfg)
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        out = wrapper.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        h = out.hidden_states[layer]
        if attn is not None:
            m = attn.unsqueeze(-1).float()
            h_mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            h_mean = h.mean(dim=1)
        acts.append(h_mean.cpu().float().numpy())
    return np.concatenate(acts, axis=0)


def sae_reconstruct(sae_module, pooled, device, batch=32):
    outs = []
    t = torch.from_numpy(pooled).float().to(device)
    with torch.no_grad():
        for i in range(0, t.shape[0], batch):
            chunk = t[i:i + batch]
            rec = None
            for method in ("encode_decode", "__call__"):
                try:
                    if method == "encode_decode" and hasattr(sae_module, "encode_decode"):
                        rec = sae_module.encode_decode(chunk); break
                    if method == "__call__":
                        z = sae_module.encode(chunk) if hasattr(sae_module, "encode") else None
                        rec = sae_module.decode(z) if z is not None else sae_module(chunk)
                        if isinstance(rec, (tuple, list)):
                            rec = rec[0]
                        break
                except Exception:
                    continue
            if rec is None:
                raise RuntimeError("No compatible SAE forward API")
            outs.append(rec.detach().cpu().float().numpy())
    return np.concatenate(outs, axis=0)


def score_residual(acts, sae_module, d_K, global_mean, device):
    rec = sae_reconstruct(sae_module, acts, device)
    centered = acts - global_mean[None, :]
    rec_centered = rec - global_mean[None, :]
    residual = centered - rec_centered
    d_unit = d_K / (np.linalg.norm(d_K) + 1e-12)
    return residual @ d_unit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", default="p69")
    p.add_argument("--model-path", required=True)
    p.add_argument("--bcd-dir", required=True)
    p.add_argument("--sae-path", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--member-texts", required=True)
    p.add_argument("--nonmember-texts", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["word_shuffle", "t5_paraphrase", "backtranslation_cached"], default="word_shuffle")
    p.add_argument("--cached-paraphrase-file", default="")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    set_global_seed(SeedConfig(seed=args.seed))

    def _load(path):
        return [json.loads(l)["text"] for l in Path(path).read_text().splitlines() if l.strip()]

    members = _load(args.member_texts)
    nonmembers = _load(args.nonmember_texts)
    log.info(f"n_member={len(members)} n_nonmember={len(nonmembers)} mode={args.mode}")

    # Build paraphrased members
    if args.mode == "word_shuffle":
        paraphrased = [perturb_word_shuffle(t, args.seed) for t in members]
    elif args.mode == "t5_paraphrase":
        paraphrased = paraphrase_t5(members, args.device, batch=args.batch_size, max_len=args.seq_len)
    elif args.mode == "backtranslation_cached":
        if not args.cached_paraphrase_file:
            raise SystemExit("--cached-paraphrase-file required for mode=backtranslation_cached")
        paraphrased = load_cached(Path(args.cached_paraphrase_file), len(members))
    else:
        raise SystemExit(f"unknown mode {args.mode}")

    out_dir = Path(args.output_dir) / args.model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist paraphrase dump
    para_dump = out_dir / f"paraphrase_cache_{args.mode}.jsonl"
    with para_dump.open("w") as f:
        for orig, para in zip(members, paraphrased):
            f.write(json.dumps({"original": orig, "paraphrased": para}) + "\n")
    log.info(f"wrote paraphrase dump {para_dump}")

    # Load model + SAE + d_K
    log.info("loading model")
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype="float32", device=args.device)
    wrapper = load_model_and_tokenizer(spec)
    log.info("loading SAE")
    sae = load_sae_checkpoint_any(args.sae_path, device=args.device)
    sae_module = sae[0] if isinstance(sae, tuple) else sae
    sae_module.eval().to(args.device)
    bcd = np.load(Path(args.bcd_dir) / "directions.npz")
    # Per-layer BCD archives use d_K_layer{L}; flat "d_K" kept as legacy fallback.
    d_K_key = f"d_K_layer{args.layer}"
    if d_K_key in bcd.files:
        d_K = bcd[d_K_key]
    elif "d_K" in bcd.files:
        d_K = bcd["d_K"]
    else:
        raise KeyError(
            f"neither '{d_K_key}' nor 'd_K' in {args.bcd_dir}/directions.npz "
            f"(keys={list(bcd.files)})"
        )
    # Activations for 3 sets: orig members, paraphrased members, nonmembers
    acts_orig = collect_pooled_acts(wrapper, members, args.layer, args.seq_len, args.batch_size, args.device)
    acts_para = collect_pooled_acts(wrapper, paraphrased, args.layer, args.seq_len, args.batch_size, args.device)
    acts_non = collect_pooled_acts(wrapper, nonmembers, args.layer, args.seq_len, args.batch_size, args.device)

    # global_mean pooled over (orig members + nonmembers), to mirror the dark-subspace convention
    base_acts = np.concatenate([acts_orig, acts_non], axis=0)
    gm_key = f"global_mean_layer{args.layer}"
    if gm_key in bcd.files:
        global_mean = bcd[gm_key]
    elif "global_mean" in bcd.files:
        global_mean = bcd["global_mean"]
    else:
        global_mean = base_acts.mean(axis=0)

    # Residual scores
    s_orig = score_residual(acts_orig, sae_module, d_K, global_mean, args.device)
    s_para = score_residual(acts_para, sae_module, d_K, global_mean, args.device)
    s_non = score_residual(acts_non, sae_module, d_K, global_mean, args.device)

    # Original: orig members vs nonmembers
    y_orig = np.array([1] * len(s_orig) + [0] * len(s_non), dtype=int)
    scores_orig = np.concatenate([s_orig, s_non])
    auroc_o, sign_o = auroc_bi(y_orig, scores_orig)
    tpr1_o = tpr_at_fpr(y_orig, sign_o * scores_orig, 0.01)
    tpr5_o = tpr_at_fpr(y_orig, sign_o * scores_orig, 0.05)

    # Paraphrased: paraphrased members vs same nonmembers
    y_para = np.array([1] * len(s_para) + [0] * len(s_non), dtype=int)
    scores_para = np.concatenate([s_para, s_non])
    auroc_p, sign_p = auroc_bi(y_para, scores_para)
    tpr1_p = tpr_at_fpr(y_para, sign_p * scores_para, 0.01)
    tpr5_p = tpr_at_fpr(y_para, sign_p * scores_para, 0.05)

    # Also measure per-member shift: cosine(acts_orig[i], acts_para[i]) and delta score_residual
    cos_shift = []
    n = acts_orig.shape[0]
    for i in range(n):
        a, b = acts_orig[i], acts_para[i]
        den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        cos_shift.append(float(a @ b / den))
    score_shift_per_member = (s_para - s_orig).tolist()

    # Paired bootstrap deltas on the full (member, nonmember) set
    d_auroc, d_lo, d_hi = paired_bootstrap(
        y_orig, scores_orig, scores_para,
        lambda y, x: float(roc_auc_score(y, x)),
        n_boot=1000, seed=args.seed,
    )
    d_tpr1, d_tpr1_lo, d_tpr1_hi = paired_bootstrap(
        y_orig, sign_o * scores_orig, sign_o * scores_para,
        lambda y, x: tpr_at_fpr(y, x, 0.01),
        n_boot=1000, seed=args.seed,
    )

    result = {
        "model_tag": args.model_tag,
        "mode": args.mode,
        "layer": args.layer,
        "n_member": int(len(members)),
        "n_nonmember": int(len(nonmembers)),
        "original": {
            "residual_d_K_auroc": auroc_o,
            "residual_d_K_tpr_at_1pct_fpr": tpr1_o,
            "residual_d_K_tpr_at_5pct_fpr": tpr5_o,
            "orientation_sign": sign_o,
        },
        "paraphrased": {
            "residual_d_K_auroc": auroc_p,
            "residual_d_K_tpr_at_1pct_fpr": tpr1_p,
            "residual_d_K_tpr_at_5pct_fpr": tpr5_p,
            "orientation_sign": sign_p,
        },
        "delta_paraphrased_minus_original": {
            "auroc_mean": d_auroc, "auroc_ci95_lo": d_lo, "auroc_ci95_hi": d_hi,
            "tpr1_mean": d_tpr1, "tpr1_ci95_lo": d_tpr1_lo, "tpr1_ci95_hi": d_tpr1_hi,
        },
        "per_member_cos_shift": {
            "mean": float(np.mean(cos_shift)),
            "p05": float(np.percentile(cos_shift, 5)),
            "p50": float(np.percentile(cos_shift, 50)),
            "p95": float(np.percentile(cos_shift, 95)),
        },
        "per_member_score_shift": {
            "mean": float(np.mean(score_shift_per_member)),
            "p05": float(np.percentile(score_shift_per_member, 5)),
            "p50": float(np.percentile(score_shift_per_member, 50)),
            "p95": float(np.percentile(score_shift_per_member, 95)),
        },
        "paraphrase_cache_file": str(para_dump),
        "script": "paraphrase_sensitivity.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "seed": args.seed,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    (out_dir / "per_text_scores.json").write_text(json.dumps({
        "labels_orig": y_orig.tolist(),
        "scores_orig_vs_nonmember": scores_orig.tolist(),
        "labels_para": y_para.tolist(),
        "scores_para_vs_nonmember": scores_para.tolist(),
        "cos_shift_per_member": cos_shift,
        "score_shift_per_member": score_shift_per_member,
    }))
    log.info(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
