#!/usr/bin/env python3
"""
per_row_bootstrap_kocl2.py.

Runs paired bootstrap 95% CIs (n_boot=10000) on the per-row residual-minus-original
margin across the K-OC-2 cohort and reports a one-sided binomial sign test on the
5-of-7 inversion.

Used in Methods note on cohort statistics, Introduction headline, and the
appendix per-row bootstrap table.
Reproduce:
    .venv/bin/python scripts/dark_subspace/per_row_bootstrap_kocl2.py

Inputs.
  Seven K-OC-2 cohort rows (five inverting plus two anchor) with per_text_scores
  (labels, score_K_original, score_K_residual) of length 2000 each. Bootstrap
  n_boot = 10000 with random seed = 0.

Outputs (JSON).
  Per-row margin_mean, margin_CI_low, margin_CI_high (95 percent), the binomial
  sign-test p-value across the five inverting rows against H0=0.5, per-row 95
  percent CI excludes-zero counts, plus the two anchor rows for reference.

Pre-registered decision rule.
  ACCEPT  if at least three of five inverting rows have 95 percent CI excluding
          zero AND sign-test p < 0.05.
  NUANCED if two of five with CI excluding zero AND sign-test p < 0.05.
  NULL    if zero or one with CI excluding zero, or sign-test p >= 0.05.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))


# Seven-row K-OC-2 cohort plus the two anchor rows.
# Each entry: (label, results.json relative path, in_cohort_inverting flag)
#   in_cohort_inverting=True  -> counted in the 5-row sign test
#   in_cohort_inverting=False -> reported as anchor row (residual < original)
COHORT_ROWS: List[Tuple[str, str, bool]] = [
    # Five inverting rows (residual > original margin, K-OC-2 cohort headline)
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
    # Two anchor rows (residual < original by design, not part of the sign test)
    ("P69 N=6 anchor seed42_postfix",
     "runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed42_postfix/results.json", False),
    ("P12B fresh-init seed47",
     "runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed47/results.json", False),
]

# Qwen2 mult4 has seeds 42/43/44 on disk. To avoid pseudo-replication we include
# only seed 42 in the headline 5-row sign test. Seeds 43 and 44 are reported in
# a secondary block as a robustness check.
QWEN2_MULT4_SECONDARY_SEEDS: List[Tuple[str, str]] = [
    ("Qwen2 mult4 seed43 (secondary)",
     "runs/dark_subspace/sae_dark_subspace/qwen2_mixed_seed43/results.json"),
    ("Qwen2 mult4 seed44 (secondary)",
     "runs/dark_subspace/sae_dark_subspace/qwen2_mixed_seed44/results.json"),
]


def _load_per_text_scores(results_json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (labels, score_K_original, score_K_residual) from a results.json.

    Raises if any required field is missing or wrong length.
    """
    with open(results_json_path) as f:
        r = json.load(f)
    pts = r.get("per_text_scores")
    if not isinstance(pts, dict):
        raise ValueError(f"{results_json_path}: per_text_scores is missing or not a dict")
    for required in ("labels", "score_K_original", "score_K_residual"):
        if required not in pts:
            raise ValueError(f"{results_json_path}: per_text_scores missing key {required}")
        if not isinstance(pts[required], list):
            raise ValueError(f"{results_json_path}: per_text_scores[{required}] is not a list")
    labels = np.array(pts["labels"], dtype=np.int64)
    sk_orig = np.array(pts["score_K_original"], dtype=np.float64)
    sk_resid = np.array(pts["score_K_residual"], dtype=np.float64)
    if not (len(labels) == len(sk_orig) == len(sk_resid)):
        raise ValueError(
            f"{results_json_path}: length mismatch labels={len(labels)} "
            f"orig={len(sk_orig)} resid={len(sk_resid)}"
        )
    if labels.sum() not in (1000,):
        # Sanity check: 1000 members + 1000 nonmembers (canonical schema).
        # Print a warning but continue; some cohort rows may differ legitimately.
        print(
            f"[warn] {results_json_path}: members={int(labels.sum())} (expected 1000)",
            file=sys.stderr,
        )
    return labels, sk_orig, sk_resid


def _bidirectional_auroc(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, int]:
    """Compute AUROC, taking the max of (score, -score) directions.

    Returns (auroc, sign) where sign in {+1, -1} indicates which direction was used.
    Mirrors `bidirectional_auroc` in scripts/dark_subspace/sae_dark_subspace.py.
    """
    a = roc_auc_score(labels, scores)
    b = roc_auc_score(labels, -scores)
    if a >= b:
        return float(a), +1
    return float(b), -1


def paired_bootstrap_auroc_margin(
    labels: np.ndarray,
    sk_orig: np.ndarray,
    sk_resid: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap on margin (residual_AUROC - original_AUROC).

    Each bootstrap iteration resamples (text, score_orig, score_resid, label) tuples
    *with replacement* from the joint distribution, recomputes both AUROCs, and
    records the margin. AUROC sign is fixed at the original direction (taken from
    the full-sample AUROC computation) so per-row sign flips during resampling
    cannot mask the margin.

    Returns dict with margin_mean, margin_observed, margin_ci_low (2.5%),
    margin_ci_high (97.5%), n_boot.
    """
    n = len(labels)

    # Lock the AUROC sign convention from the full-sample AUROC. Both score_K_original
    # and score_K_residual are oriented members > nonmembers by the eval pipeline,
    # re-confirmed here.
    auroc_orig_full, sign_orig = _bidirectional_auroc(labels, sk_orig)
    auroc_resid_full, sign_resid = _bidirectional_auroc(labels, sk_resid)
    sk_orig_signed = sign_orig * sk_orig
    sk_resid_signed = sign_resid * sk_resid
    margin_observed = auroc_resid_full - auroc_orig_full

    rng = np.random.default_rng(seed)
    margins = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs_labels = labels[idx]
        # Bootstrap can produce a sample with all members or all nonmembers.guard.
        s = bs_labels.sum()
        if s == 0 or s == n:
            margins[i] = np.nan
            continue
        try:
            a_orig = roc_auc_score(bs_labels, sk_orig_signed[idx])
            a_resid = roc_auc_score(bs_labels, sk_resid_signed[idx])
        except ValueError:
            margins[i] = np.nan
            continue
        margins[i] = a_resid - a_orig

    # Drop NaNs (rare, only single-class bootstrap samples).
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
        auroc_resid_full=float(auroc_resid_full),
        sign_orig=int(sign_orig),
        sign_resid=int(sign_resid),
    )


def _ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    return ci_low > 0.0 or ci_high < 0.0


def main():
    ap = argparse.ArgumentParser(description="K-OC-2 cohort per-row bootstrap")
    ap.add_argument("--n-boot", type=int, default=10_000,
                    help="Number of paired bootstrap resamples per row (default 10000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for bootstrap (default 0)")
    ap.add_argument("--output",
                    default="results/dark_subspace/paper_claims/k_oc2_bootstrap_2026-05-02.json",
                    help="Output JSON path (relative to repo root)")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows_out: List[Dict] = []

    print("=" * 78)
    print("Per-row paired bootstrap on K-OC-2 cohort")
    print(f"  n_boot={args.n_boot}  seed={args.seed}")
    print("=" * 78)

    # 7 cohort rows (5 inverting + 2 anchor)
    for label, rel_path, is_inverting in COHORT_ROWS:
        abs_path = REPO_ROOT / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"K-OC-2 cohort row missing: {abs_path}")
        labels_arr, sk_orig, sk_resid = _load_per_text_scores(abs_path)
        stats = paired_bootstrap_auroc_margin(
            labels=labels_arr,
            sk_orig=sk_orig,
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
            f"  {marker} {label}: margin={stats['margin_observed']:+.4f} "
            f"95%CI=[{stats['margin_ci_low']:+.4f}, {stats['margin_ci_high']:+.4f}] "
            f"AUROC orig={stats['auroc_orig_full']:.4f} resid={stats['auroc_resid_full']:.4f}"
        )

    # Secondary Qwen2 mult4 seeds (pseudo-replication robustness check; not in headline test).
    secondary_rows: List[Dict] = []
    for label, rel_path in QWEN2_MULT4_SECONDARY_SEEDS:
        abs_path = REPO_ROOT / rel_path
        if not abs_path.exists():
            print(f"[warn] secondary row missing: {abs_path}", file=sys.stderr)
            continue
        labels_arr, sk_orig, sk_resid = _load_per_text_scores(abs_path)
        stats = paired_bootstrap_auroc_margin(
            labels=labels_arr, sk_orig=sk_orig, sk_resid=sk_resid,
            n_boot=args.n_boot, seed=args.seed,
        )
        excludes = _ci_excludes_zero(stats["margin_ci_low"], stats["margin_ci_high"])
        secondary_rows.append({
            "label": label, "results_json": rel_path, **stats,
            "ci_excludes_zero": bool(excludes),
        })

    # Sign-test on the 5 inverting rows.
    inverting_observed = [
        (row["label"], row["margin_observed"], row["ci_excludes_zero"])
        for row in rows_out if row["in_inverting_cohort"]
    ]
    n_inverting = len(inverting_observed)
    n_positive = sum(1 for _, m, _ in inverting_observed if m > 0)
    n_ci_excluding_zero_inverting = sum(1 for _, _, excl in inverting_observed if excl)
    binom = binomtest(n_positive, n_inverting, p=0.5, alternative="greater")
    sign_test_p_one_sided = float(binom.pvalue)

    # Pre-registered decision (mechanical)
    if n_ci_excluding_zero_inverting >= 3 and sign_test_p_one_sided < 0.05:
        decision = "ACCEPT"
    elif n_ci_excluding_zero_inverting >= 2 and sign_test_p_one_sided < 0.05:
        decision = "NUANCED"
    else:
        decision = "NULL"

    print()
    print("-" * 78)
    print("Sign-test on the 5 K-OC-2 inverting rows (binomial against H0=0.5):")
    print(f"  n_positive_margin = {n_positive} / {n_inverting}")
    print(f"  CI excludes zero  = {n_ci_excluding_zero_inverting} / {n_inverting}")
    print(f"  p_one_sided       = {sign_test_p_one_sided:.6g}")
    print(f"  Pre-reg decision  = {decision}")
    print("-" * 78)

    output = {
        "experiment": "K-OC-2 cohort per-row paired bootstrap CIs + sign test",
        "n_boot": int(args.n_boot),
        "bootstrap_seed": int(args.seed),
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
        "elapsed_sec": float(time.time() - t0),
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "script": "scripts/dark_subspace/per_row_bootstrap_kocl2.py",
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"  wrote {out_path}")
    print(f"  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
