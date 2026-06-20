#!/usr/bin/env python3
"""baseline_attacks_suite.py.

Standard MIA baseline suite (loss, zlib, MIN-K%=20, MIN-K%=10, MIN-K%++=20)
plus the residual d_K attack on the four cohort models. Emits per-text scores
plus paired bootstrap AUROC, TPR@1%FPR, and TPR@5%FPR.

Used in Appendix app:tpr_paraphrase and the per-method results paragraph of
the paper. Reproduce per-roster:
    .venv/bin/python scripts/dark_subspace/baseline_attacks_suite.py \
        --roster scripts/dark_subspace/configs/oc_roster.json \
        --gate-passing-only \
        --member-texts data/memcirc_ctrl_ft/member.jsonl \
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \
        --output-dir runs/dark_subspace/baseline_attacks

Background.
PETAL (He et al., USENIX Sec 2025, arXiv 2502.18943) has no public reference
implementation in this repository, so this script runs an equivalent-purpose
stack of standard probability-based MIA baselines from the literature and
reports them paired with the orthogonal-complement residual-direction attack.

Baselines.
    1. loss       per-text mean cross-entropy (Yeom et al. 2018).
    2. zlib       loss divided by zlib-compressed length (Carlini et al. 2021).
    3. min_k_20   mean of bottom 20 percent token log-probs (Shi et al. 2023).
    4. min_k_10   mean of bottom 10 percent token log-probs.
    5. min_k_pp_20  Min-K%++ (Zhang et al. 2024) at k=0.2, normalized by
                    token log-prob mean and std over the vocabulary, with
                    bidirectional orientation.

Method scores.
    6. residual_d_K  projection of (h minus SAE.reconstruct(h)) onto the
                     channel-decomposition leading knowledge direction d_K.
                     Same residual_score_K_auroc as the dark-subspace and
                     orthogonal-complement pipeline, persisted per text so
                     TPR@1%FPR and paired bootstrap CIs are available.
    7. original_d_K  projection of h onto d_K (baseline reference).

All attacks are scored on AUROC, TPR@1%FPR, and TPR@5%FPR. Paired bootstrap
95 percent CIs (n_boot=10000, resample 2000 test items) are emitted per
method and paired (residual minus baseline).

Note on controlled-FT ceiling.
On Pythia-6.9B, Pythia-12B, and Qwen2-7B at epoch 5 of controlled FT, the
loss signal saturates at AUROC=1.0 because the fine-tune has effectively
memorised the 1000 member texts. On GPT-Neo-2.7B the loss signal sits near
0.50 because Neo's fine-tune barely moved. The paper Methods and Limitations
sections discuss this caveat.

Single-model usage:
    .venv/bin/python scripts/dark_subspace/baseline_attacks_suite.py \
        --model-tag p69 \
        --model-path <ft_model_dir> \
        --bcd-dir runs/dark_subspace/behavioral_channels/p69_epoch5 \
        --sae-path <sae_final.pt> \
        --layer 16 \
        --member-texts data/memcirc_ctrl_ft/member.jsonl \
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \
        --output-dir runs/dark_subspace/baseline_attacks/p69
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ---------- metrics ----------

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def auroc_bi(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, int]:
    """Bidirectional AUROC; returns (auroc, sign) where sign=+1 if higher=member, else -1."""
    finite = np.isfinite(y_score)
    if not finite.all():
        y_score = np.where(finite, y_score, float(np.median(y_score[finite])) if finite.any() else 0.0)
    a_pos = float(roc_auc_score(y_true, y_score))
    a_neg = float(roc_auc_score(y_true, -y_score))
    if a_neg > a_pos:
        return a_neg, -1
    return a_pos, +1


def mia_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    auroc, sign = auroc_bi(y_true, y_score)
    oriented = sign * y_score
    return {
        "auroc": auroc,
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, oriented, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(y_true, oriented, 0.05),
        "orientation_sign": sign,
    }


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            values.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_bootstrap_ci(y_true: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            va = metric_fn(y_true[idx], s_a[idx])
            vb = metric_fn(y_true[idx], s_b[idx])
            diffs.append(va - vb)
        except Exception:
            continue
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(diffs)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------- attack score computations ----------

def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


@torch.no_grad()
def collect_scores_and_activations(wrapper, texts, layer, seq_len, batch_size, device) -> Dict[str, np.ndarray]:
    """Returns per-text losses, min-k scores, min-k++ scores, zlib lengths, and
    mean-pooled layer-L activations. One forward pass per batch."""
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=False)
    losses = []
    minkprob_20 = []
    minkprob_10 = []
    minkpp_20 = []
    zlib_lens = []
    acts = []

    for chunk in tqdm(list(_batched(texts, batch_size)), desc=f"layer-{layer} scoring"):
        batch = tokenize_batch(wrapper.tokenizer, chunk, tok_cfg)
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        out = wrapper.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)

        # activations
        h = out.hidden_states[layer]  # (B, T, d)
        if attn is not None:
            mask = attn.unsqueeze(-1).float()
            h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            h_mean = h.mean(dim=1)
        acts.append(h_mean.cpu().float().numpy())

        # per-token log-probs
        logits = out.logits[:, :-1, :].contiguous()  # (B, T-1, V)
        labels = input_ids[:, 1:].contiguous()  # (B, T-1)
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)

        for b in range(logits.shape[0]):
            if attn is not None:
                m = attn[b, 1:].bool()
            else:
                m = torch.ones_like(labels[b], dtype=torch.bool)
            if m.sum() == 0:
                losses.append(float("nan"))
                minkprob_20.append(float("nan"))
                minkprob_10.append(float("nan"))
                minkpp_20.append(float("nan"))
                zlib_lens.append(0)
                continue
            token_lp = log_probs_all[b][range(labels.shape[1]), labels[b]]
            token_lp_valid = token_lp[m]

            mean_loss = float(-token_lp_valid.mean().item())
            losses.append(mean_loss)

            # Min-K% prob: mean of bottom k% token log-probs (smaller is more suspicious member if k is top, here we take the smallest log-prob tokens and average, which for MEMBERS tends to be HIGHER (less suspicious) than NONMEMBERS, with signs resolved by bidirectional auroc).
            n_valid = int(token_lp_valid.numel())
            sorted_lp, _ = torch.sort(token_lp_valid)
            k20 = max(1, int(0.20 * n_valid))
            k10 = max(1, int(0.10 * n_valid))
            minkprob_20.append(float(sorted_lp[:k20].mean().item()))
            minkprob_10.append(float(sorted_lp[:k10].mean().item()))

            # Min-K%++ (Zhang et al. 2024): (log p(x_t|<t) - mu_t) / sigma_t
            # where mu_t, sigma_t are mean/std over the vocabulary at position t.
            # Take mean of bottom 20% normalized.
            lp_t = log_probs_all[b]  # (T-1, V)
            mu_t = lp_t.mean(dim=-1)  # (T-1,)
            sigma_t = lp_t.std(dim=-1) + 1e-8  # (T-1,)
            norm_lp = (token_lp - mu_t) / sigma_t  # (T-1,)
            norm_lp_valid = norm_lp[m]
            sorted_nlp, _ = torch.sort(norm_lp_valid)
            minkpp_20.append(float(sorted_nlp[:k20].mean().item()))

            zlib_lens.append(len(zlib.compress(chunk[b].encode("utf-8"))))

    return {
        "loss": np.array(losses),
        "minkprob_20": np.array(minkprob_20),
        "minkprob_10": np.array(minkprob_10),
        "minkpp_20": np.array(minkpp_20),
        "zlib_len": np.array(zlib_lens),
        "activations": np.concatenate(acts, axis=0),
    }


def _sae_reconstruct_pooled(sae_module, pooled: np.ndarray, device: str, batch: int = 32) -> np.ndarray:
    """Encode+decode mean-pooled activations through the SAE. Returns reconstructed activations."""
    outs = []
    t = torch.from_numpy(pooled).float().to(device)
    with torch.no_grad():
        for i in range(0, t.shape[0], batch):
            chunk = t[i : i + batch]
            # Try a range of SAE interface possibilities
            rec = None
            for method in ("encode_decode", "__call__"):
                try:
                    if method == "encode_decode" and hasattr(sae_module, "encode_decode"):
                        rec = sae_module.encode_decode(chunk)
                        break
                    if method == "__call__":
                        z = sae_module.encode(chunk) if hasattr(sae_module, "encode") else None
                        if z is None:
                            r = sae_module(chunk)
                            rec = r[0] if isinstance(r, (tuple, list)) else r
                        else:
                            rec = sae_module.decode(z)
                        break
                except Exception:
                    continue
            if rec is None:
                raise RuntimeError("No compatible SAE forward API found")
            outs.append(rec.detach().cpu().float().numpy())
    return np.concatenate(outs, axis=0)


def run_model(
    model_tag: str,
    model_path: str,
    bcd_dir: str,
    sae_path: str,
    layer: int,
    member_texts: List[str],
    nonmember_texts: List[str],
    out_dir: Path,
    seq_len: int = 256,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
    min_recon_cos: float = 0.85,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(SeedConfig(seed=seed))

    log.info(f"[{model_tag}] loading model")
    spec = HFModelSpec(name_or_path=model_path, torch_dtype="float32", device=device)
    wrapper = load_model_and_tokenizer(spec)

    log.info(f"[{model_tag}] loading SAE {sae_path}")
    sae = load_sae_checkpoint_any(sae_path, device=device)
    sae_module = sae[0] if isinstance(sae, tuple) else sae
    sae_module.eval().to(device)

    log.info(f"[{model_tag}] loading channel-decomposition directions {bcd_dir}")
    bcd_npz = np.load(Path(bcd_dir) / "directions.npz")
    # directions.npz stores per-layer keys (d_K_layer{L}); fall back to flat "d_K"
    # for legacy archives that only encode a single layer.
    d_K_key = f"d_K_layer{layer}"
    if d_K_key in bcd_npz.files:
        d_K = bcd_npz[d_K_key]
    elif "d_K" in bcd_npz.files:
        d_K = bcd_npz["d_K"]
    else:
        raise KeyError(
            f"[{model_tag}] neither '{d_K_key}' nor 'd_K' found in "
            f"{bcd_dir}/directions.npz (keys={list(bcd_npz.files)})"
        )
    gm_key = f"global_mean_layer{layer}"
    if gm_key in bcd_npz.files:
        global_mean = bcd_npz[gm_key]
    elif "global_mean" in bcd_npz.files:
        global_mean = bcd_npz["global_mean"]
    else:
        global_mean = None

    n_m = len(member_texts)
    n_n = len(nonmember_texts)
    labels = np.array([1] * n_m + [0] * n_n, dtype=int)

    all_texts = member_texts + nonmember_texts
    log.info(f"[{model_tag}] collecting scores + activations for {len(all_texts)} texts")
    outs = collect_scores_and_activations(wrapper, all_texts, layer, seq_len, batch_size, device)

    # zlib ratio = loss / zlib_len
    zlib_ratio = outs["loss"] / np.maximum(outs["zlib_len"], 1)

    activations = outs["activations"]
    if global_mean is None:
        global_mean = activations.mean(axis=0)

    # SAE reconstruction (on pooled activations, mirroring the dark-subspace and OC convention)
    log.info(f"[{model_tag}] reconstructing {activations.shape[0]} pooled activations")
    reconstructed = _sae_reconstruct_pooled(sae_module, activations, device)

    # Reconstruction cosine sanity
    centered = activations - global_mean[None, :]
    recon_centered = reconstructed - global_mean[None, :]
    residual_centered = centered - recon_centered

    def _cos_row(a, b):
        nab = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        nab = np.maximum(nab, 1e-12)
        return float(np.mean(np.sum(a * b, axis=1) / nab))

    recon_cos = _cos_row(activations, reconstructed)
    log.info(f"[{model_tag}] recon_cos (uncentered mean cosine) = {recon_cos:.4f}")

    # d_K direction normalized
    d_K_unit = d_K / (np.linalg.norm(d_K) + 1e-12)

    # Activation-based scores
    score_orig = centered @ d_K_unit
    score_resid = residual_centered @ d_K_unit

    scores = {
        "loss": outs["loss"],
        "zlib_ratio": zlib_ratio,
        "minkprob_20": outs["minkprob_20"],
        "minkprob_10": outs["minkprob_10"],
        "minkpp_20": outs["minkpp_20"],
        "original_d_K": score_orig,
        "residual_d_K": score_resid,
    }

    metrics: Dict[str, Dict] = {}
    for name, s in scores.items():
        m = mia_metrics(labels, s)
        # Bootstrap CIs (AUROC and TPR@1%FPR)
        auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(
            labels, m["orientation_sign"] * s,
            lambda y, x: float(roc_auc_score(y, x)),
            n_boot=10000, seed=seed,
        )
        tpr1_mean, tpr1_lo, tpr1_hi = bootstrap_ci(
            labels, m["orientation_sign"] * s,
            lambda y, x: tpr_at_fpr(y, x, 0.01),
            n_boot=10000, seed=seed,
        )
        tpr5_mean, tpr5_lo, tpr5_hi = bootstrap_ci(
            labels, m["orientation_sign"] * s,
            lambda y, x: tpr_at_fpr(y, x, 0.05),
            n_boot=10000, seed=seed,
        )
        metrics[name] = {
            **m,
            "auroc_boot_mean": auroc_mean, "auroc_ci95_lo": auroc_lo, "auroc_ci95_hi": auroc_hi,
            "tpr1_boot_mean": tpr1_mean, "tpr1_ci95_lo": tpr1_lo, "tpr1_ci95_hi": tpr1_hi,
            "tpr5_boot_mean": tpr5_mean, "tpr5_ci95_lo": tpr5_lo, "tpr5_ci95_hi": tpr5_hi,
        }

    # Paired diffs (residual_d_K minus each baseline, for TPR@1%FPR)
    paired = {}
    ref_s = scores["residual_d_K"] * metrics["residual_d_K"]["orientation_sign"]
    for name in ("loss", "zlib_ratio", "minkprob_20", "minkprob_10", "minkpp_20", "original_d_K"):
        cmp_s = scores[name] * metrics[name]["orientation_sign"]
        d_mean, d_lo, d_hi = paired_bootstrap_ci(
            labels, ref_s, cmp_s,
            lambda y, x: tpr_at_fpr(y, x, 0.01),
            n_boot=10000, seed=seed,
        )
        paired[f"tpr1_residual_minus_{name}"] = {"mean": d_mean, "ci95_lo": d_lo, "ci95_hi": d_hi}
        d_mean, d_lo, d_hi = paired_bootstrap_ci(
            labels, ref_s, cmp_s,
            lambda y, x: float(roc_auc_score(y, x)),
            n_boot=10000, seed=seed,
        )
        paired[f"auroc_residual_minus_{name}"] = {"mean": d_mean, "ci95_lo": d_lo, "ci95_hi": d_hi}

    per_text_scores = {k: v.tolist() for k, v in scores.items()}
    per_text_scores["labels"] = labels.tolist()

    results = {
        "model_tag": model_tag,
        "model_path": model_path,
        "bcd_dir": bcd_dir,
        "sae_path": sae_path,
        "layer": layer,
        "n_member": int(n_m),
        "n_nonmember": int(n_n),
        "seq_len": seq_len,
        "reconstruction_cosine": float(recon_cos),
        "reconstruction_cosine_gate_passed": bool(recon_cos >= min_recon_cos),
        "methods": {name: {k: v for k, v in m.items() if k != "orientation_sign"} for name, m in metrics.items()},
        "orientations": {name: m["orientation_sign"] for name, m in metrics.items()},
        "paired_bootstrap": paired,
        "script": "baseline_attacks_suite.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "seed": seed,
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Separate per-text dump to avoid bloating results.json
    (out_dir / "per_text_scores.json").write_text(json.dumps(per_text_scores))

    log.info(f"[{model_tag}] wrote {out_dir / 'results.json'}")
    del wrapper, sae_module
    torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", type=str)
    p.add_argument("--model-path", type=str)
    p.add_argument("--bcd-dir", type=str)
    p.add_argument("--sae-path", type=str)
    p.add_argument("--layer", type=int)
    p.add_argument("--roster", type=str)
    p.add_argument("--gate-passing-only", action="store_true",
                   help="When using --roster, restrict to the 4 OC-gate-passing models (p69, p12b, neo, qwen2).")
    p.add_argument("--member-texts", type=str, required=True)
    p.add_argument("--nonmember-texts", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--continue-on-fail", action="store_true")
    args = p.parse_args()

    def _load_texts(path):
        out = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)["text"])
        return out

    member_texts = _load_texts(args.member_texts)
    nonmember_texts = _load_texts(args.nonmember_texts)
    log.info(f"n_member={len(member_texts)} n_nonmember={len(nonmember_texts)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.roster:
        roster = json.loads(Path(args.roster).read_text())
        if args.gate_passing_only:
            keep = {"p69", "p12b", "neo", "qwen2"}
            roster = [r for r in roster if r["model_tag"] in keep]
            log.info(f"Filtered roster to gate-passing {[r['model_tag'] for r in roster]}")
        summary = []
        failures = []
        for entry in roster:
            tag = entry["model_tag"]
            out_sub = output_dir / tag
            try:
                r = run_model(
                    model_tag=tag,
                    model_path=entry["model_path"],
                    bcd_dir=entry["bcd_dir"],
                    sae_path=entry["sae_path"],
                    layer=entry["layer"],
                    member_texts=member_texts,
                    nonmember_texts=nonmember_texts,
                    out_dir=out_sub,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    device=args.device,
                )
                summary.append({"model_tag": tag, "methods": r["methods"], "paired_bootstrap": r["paired_bootstrap"]})
            except Exception as e:
                log.error(f"[{tag}] FAILED: {e}")
                failures.append({"model_tag": tag, "error": str(e)})
                if not args.continue_on_fail:
                    raise
        (output_dir / "aggregate.json").write_text(json.dumps({
            "roster_size": len(roster),
            "n_success": len(summary),
            "n_failures": len(failures),
            "rows": summary,
            "failures": failures,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }, indent=2))
        log.info(f"Wrote aggregate to {output_dir / 'aggregate.json'}")
    else:
        required = ["model_tag", "model_path", "bcd_dir", "sae_path", "layer"]
        missing = [k for k in required if getattr(args, k.replace("-", "_")) in (None, "")]
        if missing:
            raise SystemExit(f"Missing single-model args: {missing}")
        run_model(
            model_tag=args.model_tag,
            model_path=args.model_path,
            bcd_dir=args.bcd_dir,
            sae_path=args.sae_path,
            layer=args.layer,
            member_texts=member_texts,
            nonmember_texts=nonmember_texts,
            out_dir=output_dir / args.model_tag,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
