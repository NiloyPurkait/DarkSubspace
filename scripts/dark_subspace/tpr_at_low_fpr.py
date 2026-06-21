#!/usr/bin/env python3
"""tpr_at_low_fpr.py.

Computes TPR-at-0.1%-FPR with n_boot=10000 on the residual d_K stream from
the per-text scores written by the baseline-attacks suite, with paired
bootstrap CIs.

Used in the per-method results paragraph and Appendix app:tpr_paraphrase of
the paper. Reproduce:
    .venv/bin/python scripts/dark_subspace/tpr_at_low_fpr.py --all

CPU only. No GPU dependency.

Inputs.
Reads cached per_text_scores.json artifacts produced by
baseline_attacks_suite.py and computes TPR at a user-specified FPR target
(default 0.001) for the residual_d_K channel, with bootstrap 95 percent CIs
at n_boot=10000, seed=12345.

Output schema (tpr_at_0.1pct_fpr.json).
    model              str (e.g. "p69")
    fpr_target         float (e.g. 0.001)
    n_boot             int (10000)
    seed               int (12345)
    method             str ("residual_d_K" by default)
    orientation_sign   int (+1 or -1, matches results.json)
    tpr_point          float (observed TPR at the target FPR)
    tpr_boot_mean      float
    tpr_ci_lo          float (2.5th percentile)
    tpr_ci_hi          float (97.5th percentile)
    n_member           int
    n_nonmember        int
    source             str (absolute path to per_text_scores.json read)

Single-model usage:
    .venv/bin/python scripts/dark_subspace/tpr_at_low_fpr.py --model p69
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


log = logging.getLogger("tpr_at_low_fpr")


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT / "runs" / "dark_subspace" / "baseline_attacks"

DEFAULT_MODELS = ("p69", "p12b", "neo", "qwen2")


def _resolve_orientation(per_text: dict, method: str) -> int:
    """Replicates the bidirectional AUROC orientation logic from
    `baseline_attacks_suite.auroc_bi`. Returns +1 or -1."""
    labels = np.array(per_text["labels"], dtype=int)
    s = np.array(per_text[method], dtype=np.float64)
    finite = np.isfinite(s)
    if not finite.all():
        med = float(np.median(s[finite])) if finite.any() else 0.0
        s = np.where(finite, s, med)
    a_pos = float(roc_auc_score(labels, s))
    a_neg = float(roc_auc_score(labels, -s))
    return -1 if a_neg > a_pos else +1


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def compute_tpr_with_boot(
    per_text: dict,
    method: str,
    fpr_target: float,
    n_boot: int,
    seed: int,
) -> dict:
    labels = np.array(per_text["labels"], dtype=int)
    raw_scores = np.array(per_text[method], dtype=np.float64)
    finite = np.isfinite(raw_scores)
    if not finite.all():
        med = float(np.median(raw_scores[finite])) if finite.any() else 0.0
        raw_scores = np.where(finite, raw_scores, med)

    sign = _resolve_orientation(per_text, method)
    oriented = sign * raw_scores

    tpr_point = _tpr_at_fpr(labels, oriented, fpr_target)

    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = labels[idx]
        ys = oriented[idx]
        if yt.min() == yt.max():
            # extremely rare with n=2000 balanced classes; resample once
            idx = rng.integers(0, n, size=n)
            yt = labels[idx]
            ys = oriented[idx]
        boot_vals[b] = _tpr_at_fpr(yt, ys, fpr_target)

    return {
        "orientation_sign": int(sign),
        "tpr_point": float(tpr_point),
        "tpr_boot_mean": float(np.mean(boot_vals)),
        "tpr_ci_lo": float(np.percentile(boot_vals, 2.5)),
        "tpr_ci_hi": float(np.percentile(boot_vals, 97.5)),
        "n_member": int((labels == 1).sum()),
        "n_nonmember": int((labels == 0).sum()),
    }


def run_for_model(
    model: str,
    fpr_target: float,
    n_boot: int,
    seed: int,
    method: str,
) -> dict:
    src = BASELINE_ROOT / model / "per_text_scores.json"
    if not src.exists():
        raise FileNotFoundError(f"missing artifact: {src}")
    log.info(f"[{model}] reading {src}")
    per_text = json.loads(src.read_text())

    if method not in per_text:
        raise KeyError(
            f"method '{method}' not in per_text_scores keys "
            f"{list(per_text.keys())}"
        )

    t0 = time.time()
    boot = compute_tpr_with_boot(per_text, method, fpr_target, n_boot, seed)
    dt = time.time() - t0

    out = {
        "model": model,
        "fpr_target": float(fpr_target),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "method": method,
        **boot,
        "source": str(src),
        "script": "scripts/dark_subspace/tpr_at_low_fpr.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "elapsed_seconds": float(dt),
    }
    log.info(
        f"[{model}] tpr_point={out['tpr_point']:.4f} "
        f"CI95=[{out['tpr_ci_lo']:.4f}, {out['tpr_ci_hi']:.4f}] "
        f"orientation={out['orientation_sign']} ({dt:.1f}s)"
    )
    return out


def write_output(model: str, payload: dict, out_name: str) -> Path:
    out_dir = BASELINE_ROOT / model
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(f"[{model}] wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=DEFAULT_MODELS, default=None,
                    help="run a single model")
    ap.add_argument("--all", action="store_true",
                    help="run all four (p69, p12b, neo, qwen2)")
    ap.add_argument("--method", type=str, default="residual_d_K",
                    help="per_text_scores key to score (default residual_d_K)")
    ap.add_argument("--fpr-target", type=float, default=0.001,
                    help="target FPR (default 0.001 = 0.1%)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-name", type=str, default=None,
                    help="filename for output (default tpr_at_<pct>pct_fpr.json)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.model and not args.all:
        ap.error("must specify --model or --all")

    if args.out_name is None:
        # 0.001 -> 0.1pct, 0.01 -> 1pct etc
        pct = args.fpr_target * 100
        if pct == int(pct):
            tag = f"{int(pct)}pct"
        else:
            tag = f"{pct:g}pct".replace(".", "p")
        out_name = f"tpr_at_{tag}_fpr.json"
    else:
        out_name = args.out_name

    targets = DEFAULT_MODELS if args.all else (args.model,)
    summary = []
    for m in targets:
        payload = run_for_model(m, args.fpr_target, args.n_boot, args.seed, args.method)
        write_output(m, payload, out_name)
        summary.append({
            "model": m,
            "tpr_point": payload["tpr_point"],
            "tpr_ci_lo": payload["tpr_ci_lo"],
            "tpr_ci_hi": payload["tpr_ci_hi"],
        })

    print("\n--- summary ---")
    for row in summary:
        print(
            f"{row['model']:>6}  TPR={row['tpr_point']:.4f}  "
            f"CI95=[{row['tpr_ci_lo']:.4f}, {row['tpr_ci_hi']:.4f}]"
        )


if __name__ == "__main__":
    main()
