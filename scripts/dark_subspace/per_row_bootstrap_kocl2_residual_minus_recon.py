#!/usr/bin/env python3
"""
per_row_bootstrap_kocl2_residual_minus_recon.py.

Sibling per-row bootstrap with margin = residual_AUROC - reconstruction_AUROC,
harmonising the appendix per-row table with the main-text sign test which
anchors to residual > reconstruction.

Used in the appendix per-row bootstrap table.
Reproduce:
    .venv/bin/python scripts/dark_subspace/per_row_bootstrap_kocl2_residual_minus_recon.py

Inputs.
  Same nine cohort rows as `per_row_bootstrap_kocl2.py` (seven cohort plus two
  Qwen2 secondary). Each results.json must expose
  `per_text_scores.score_K_recon` of length 2000 alongside `score_K_original`
  and `score_K_residual`.

Outputs (JSON).
  Per-row margin_mean, margin_CI_low, margin_CI_high (95 percent), CI
  excludes-zero flags, plus AUROCs for original, reconstruction, residual for
  cross-checking. Sign-test on the five inverting rows is computed for
  reference under the new margin definition.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))


# Same nine rows as the original kocl2 bootstrap (seven cohort + two anchor +
# two Qwen2 secondary). Mirrors per_row_bootstrap_kocl2.py exactly.
COHORT_ROWS: List[Tuple[str, str, bool]] = [
    ("P1B mult4 layer14 seed42",
     "runs/dark_subspace/sae_dark_subspace/p1b_mixed_sae_layer14_seed42/results.json", True),
    ("P2.8B mult4 layer20 seed42",
     "runs/dark_subspace/sae_dark_subspace/p2_8b_mixed_sae_seed42/results.json", True),
    ("Qwen2 mult4 seed42",
     "runs/dark_subspace/sae_dark_subspace/qwen2_mixed_seed42/results.json", True),
    ("Qwen2 mult8 seed42",
     "runs/dark_subspace/sae_dark_subspace/qwen2_mixed_mult8_seed42/results.json", True),
    ("Neo mult8 seed42",
     "runs/dark_subspace/sae_dark_subspace/neo_mixed_mult8_seed42/results.json", True),
    ("P69 N=6 anchor seed42_postfix",
     "runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed42_postfix/results.json", False),
    ("P12B fresh-init seed47",
     "runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed47/results.json", False),
]

QWEN2_MULT4_SECONDARY_SEEDS: List[Tuple[str, str]] = [
    ("Qwen2 mult4 seed43 (secondary)",
     "runs/dark_subspace/sae_dark_subspace/qwen2_mixed_seed43/results.json"),
    ("Qwen2 mult4 seed44 (secondary)",
     "runs/dark_subspace/sae_dark_subspace/qwen2_mixed_seed44/results.json"),
]


def _load_per_text_scores(
    results_json_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (labels, score_K_original, score_K_recon, score_K_residual) from results.json."""
    with open(results_json_path) as f:
        r = json.load(f)
    pts = r.get("per_text_scores")
    if not isinstance(pts, dict):
        raise ValueError(f"{results_json_path}: per_text_scores missing or not a dict")
    for required in ("labels", "score_K_original", "score_K_recon", "score_K_residual"):
        if required not in pts:
            raise ValueError(f"{results_json_path}: per_text_scores missing key {required}")
        if not isinstance(pts[required], list):
            raise ValueError(f"{results_json_path}: per_text_scores[{required}] is not a list")
    labels = np.array(pts["labels"], dtype=np.int64)
    sk_orig = np.array(pts["score_K_original"], dtype=np.float64)
    sk_recon = np.array(pts["score_K_recon"], dtype=np.float64)
    sk_resid = np.array(pts["score_K_residual"], dtype=np.float64)
    if not (len(labels) == len(sk_orig) == len(sk_recon) == len(sk_resid)):
        raise ValueError(
            f"{results_json_path}: length mismatch labels={len(labels)} "
            f"orig={len(sk_orig)} recon={len(sk_recon)} resid={len(sk_resid)}"
        )
    if labels.sum() not in (1000,):
        print(
            f"[warn] {results_json_path}: members={int(labels.sum())} (expected 1000)",
            file=sys.stderr,
        )
    return labels, sk_orig, sk_recon, sk_resid


def _bidirectional_auroc(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, int]:
    """AUROC, taking max of (score, -score) directions. Returns (auroc, sign)."""
    a = roc_auc_score(labels, scores)
    b = roc_auc_score(labels, -scores)
    if a >= b:
        return float(a), +1
    return float(b), -1


def paired_bootstrap_residual_minus_recon(
    labels: np.ndarray,
    sk_orig: np.ndarray,
    sk_recon: np.ndarray,
    sk_resid: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap on margin (residual_AUROC - reconstruction_AUROC).

    Each iteration resamples (text, score_*, label) tuples with replacement,
    recomputes both AUROCs, and records the margin. AUROC sign is locked to the
    full-sample direction for each score column independently (matches the
    original script's convention).

    Includes original AUROC for cross-reporting (no resampling needed since
    margin lives in residual vs recon space; orig is reported on full sample
    only).
    """
    n = len(labels)

    auroc_orig_full, sign_orig = _bidirectional_auroc(labels, sk_orig)
    auroc_recon_full, sign_recon = _bidirectional_auroc(labels, sk_recon)
    auroc_resid_full, sign_resid = _bidirectional_auroc(labels, sk_resid)
    sk_recon_signed = sign_recon * sk_recon
    sk_resid_signed = sign_resid * sk_resid
    margin_observed = auroc_resid_full - auroc_recon_full

    rng = np.random.default_rng(seed)
    margins = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs_labels = labels[idx]
        s = bs_labels.sum()
        if s == 0 or s == n:
            margins[i] = np.nan
            continue
        try:
            a_recon = roc_auc_score(bs_labels, sk_recon_signed[idx])
            a_resid = roc_auc_score(bs_labels, sk_resid_signed[idx])
        except ValueError:
            margins[i] = np.nan
            continue
        margins[i] = a_resid - a_recon

    margins_clean = margins[~np.isnan(margins)]
    n_valid = int(len(margins_clean))
    if n_valid < int(0.95 * n_boot):
        print(
            f"[warn] only {n_valid}/{n_boot} valid bootstrap iterations",
            file=sys.stderr,
        )
    return dict(
        margin_observed=float(margin_observed),
        margin_mean=float(np.mean(margins_clean)),
        margin_std=float(np.std(margins_clean)),
        margin_ci_low=float(np.quantile(margins_clean, 0.025)),
        margin_ci_high=float(np.quantile(margins_clean, 0.975)),
        n_boot=int(n_boot),
        n_valid=n_valid,
        auroc_orig_full=float(auroc_orig_full),
        auroc_recon_full=float(auroc_recon_full),
        auroc_resid_full=float(auroc_resid_full),
        sign_orig=int(sign_orig),
        sign_recon=int(sign_recon),
        sign_resid=int(sign_resid),
    )


def _ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    return ci_low > 0.0 or ci_high < 0.0


def main():
    ap = argparse.ArgumentParser(
        description="K-OC-2 cohort per-row bootstrap on residual minus reconstruction margin"
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output",
        default="results/dark_subspace/paper_claims/k_oc2_bootstrap_residual_minus_reconstruction_2026-05-05.json",
        help="Output JSON path (relative to repo root)",
    )
    args = ap.parse_args()

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows_out: List[Dict] = []

    print("=" * 78)
    print("Per-row paired bootstrap on residual minus reconstruction margin")
    print(f"  n_boot={args.n_boot}  seed={args.seed}")
    print("=" * 78)

    for label, rel_path, is_inverting in COHORT_ROWS:
        abs_path = REPO_ROOT / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"K-OC-2 cohort row missing: {abs_path}")
        labels_arr, sk_orig, sk_recon, sk_resid = _load_per_text_scores(abs_path)
        stats = paired_bootstrap_residual_minus_recon(
            labels=labels_arr,
            sk_orig=sk_orig,
            sk_recon=sk_recon,
            sk_resid=sk_resid,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        excludes = _ci_excludes_zero(stats["margin_ci_low"], stats["margin_ci_high"])
        row = {
            "label": label,
            "results_json": rel_path,
            "in_inverting_cohort": is_inverting,
            "n_member": int(labels_arr.sum()),
            "n_nonmember": int(len(labels_arr) - labels_arr.sum()),
            **stats,
            "ci_excludes_zero": bool(excludes),
        }
        rows_out.append(row)
        marker = "***" if excludes else "   "
        print(
            f"  {marker} {label}. margin_resid_minus_recon={stats['margin_observed']:+.4f} "
            f"95%CI=[{stats['margin_ci_low']:+.4f}, {stats['margin_ci_high']:+.4f}] "
            f"AUROC orig={stats['auroc_orig_full']:.4f} recon={stats['auroc_recon_full']:.4f} "
            f"resid={stats['auroc_resid_full']:.4f}"
        )

    secondary_rows: List[Dict] = []
    for label, rel_path in QWEN2_MULT4_SECONDARY_SEEDS:
        abs_path = REPO_ROOT / rel_path
        if not abs_path.exists():
            print(f"[warn] secondary row missing: {abs_path}", file=sys.stderr)
            continue
        labels_arr, sk_orig, sk_recon, sk_resid = _load_per_text_scores(abs_path)
        stats = paired_bootstrap_residual_minus_recon(
            labels=labels_arr,
            sk_orig=sk_orig,
            sk_recon=sk_recon,
            sk_resid=sk_resid,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        excludes = _ci_excludes_zero(stats["margin_ci_low"], stats["margin_ci_high"])
        secondary_rows.append({
            "label": label, "results_json": rel_path, **stats,
            "ci_excludes_zero": bool(excludes),
        })

    inverting_observed = [
        (row["label"], row["margin_observed"], row["ci_excludes_zero"])
        for row in rows_out if row["in_inverting_cohort"]
    ]
    n_inverting = len(inverting_observed)
    n_positive = sum(1 for _, m, _ in inverting_observed if m > 0)
    n_ci_excluding_zero_inverting = sum(1 for _, _, excl in inverting_observed if excl)
    binom = binomtest(n_positive, n_inverting, p=0.5, alternative="greater")
    sign_test_p_one_sided = float(binom.pvalue)

    if n_ci_excluding_zero_inverting >= 3 and sign_test_p_one_sided < 0.05:
        decision = "ACCEPT"
    elif n_ci_excluding_zero_inverting >= 2 and sign_test_p_one_sided < 0.05:
        decision = "NUANCED"
    else:
        decision = "NULL"

    # Across all 9 rows (cohort + secondary): how many have residual > recon?
    all_rows = rows_out + secondary_rows
    n_positive_all9 = sum(1 for r in all_rows if r["margin_observed"] > 0)

    print()
    print("-" * 78)
    print("Sign-test on the 5 K-OC-2 inverting rows (residual minus recon margin):")
    print(f"  n_positive_margin = {n_positive} / {n_inverting}")
    print(f"  CI excludes zero  = {n_ci_excluding_zero_inverting} / {n_inverting}")
    print(f"  p_one_sided       = {sign_test_p_one_sided:.6g}")
    print(f"  Pre-reg decision  = {decision}")
    print()
    print(f"Across all 9 rows: {n_positive_all9} / {len(all_rows)} have margin > 0")
    print("-" * 78)

    # The decision rule is the pre-registered ACCEPT / NUANCED / NULL rule
    # printed verbatim under sign_test.decision_rule below. The rule itself
    # was not modified during harmonisation; only the margin definition
    # changed from residual-minus-original to residual-minus-reconstruction
    # (see harmonisation_note). The pre_reg_decision field below records
    # what the unchanged pre-registered rule says when applied to the new
    # margin definition. The sister script per_row_bootstrap_kocl2.py uses
    # the same field name on the original-margin variant.
    output = {
        "experiment": "K-OC-2 cohort per-row paired bootstrap on residual minus reconstruction margin",
        "supersedes_in_appendix": "results/dark_subspace/paper_claims/k_oc2_bootstrap_2026-05-02.json (residual minus original margin)",
        "harmonisation_note": (
            "The main-text binomial sign test anchors to residual > reconstruction. "
            "This file recomputes the appendix per-row CIs on the same 9 cohort rows "
            "under the new margin definition so app:koc2_bootstrap supports the "
            "main-text claim. The pre-registered decision rule is unchanged; only "
            "the margin definition was updated."
        ),
        "n_boot": int(args.n_boot),
        "bootstrap_seed": int(args.seed),
        "margin_definition": "auroc(residual) - auroc(reconstruction)",
        "cohort_rows": rows_out,
        "qwen2_mult4_secondary_seeds": secondary_rows,
        "sign_test": {
            "n_inverting_cohort_rows": n_inverting,
            "n_positive_margin": n_positive,
            "n_ci_excludes_zero": n_ci_excluding_zero_inverting,
            "p_one_sided_binomial_05": sign_test_p_one_sided,
            "pre_reg_decision": decision,
            "decision_rule": (
                "ACCEPT  if >=3 of 5 inverting rows have 95% CI excluding zero AND sign test p < 0.05; "
                "NUANCED if  =2 of 5 with CI excluding zero AND sign test p < 0.05; "
                "NULL    otherwise."
            ),
        },
        "across_all_9_rows": {
            "n_total_rows": len(all_rows),
            "n_margin_positive": n_positive_all9,
        },
        "elapsed_sec": float(time.time() - t0),
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "script": "scripts/dark_subspace/per_row_bootstrap_kocl2_residual_minus_recon.py",
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"  wrote {out_path}")
    print(f"  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
