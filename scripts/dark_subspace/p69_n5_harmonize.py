#!/usr/bin/env python3
"""
p69_n5_harmonize.py.

Harmonisation utility for the Pythia-6.9B (mixed) row of `tab:dark_subspace`.

Aggregates the canonical mult=4 / L1=5e-4 Pythia-6.9B mixed-data SAE cohort
at
  runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed{42_postfix,43,44,45,46,47}/
into the N=5 cluster reported in `tab:dark_subspace`, matching the Pythia-1B
seed list of {42, 43, 44, 45, 46}. The selection keeps seeds 42_postfix, 43,
44, 45, 46 and drops seed 47, giving a uniform "five SAE seeds" reporting
basis across the multi-seed Pythia rows of `tab:dark_subspace`.

Reproduce.
    .venv/bin/python scripts/dark_subspace/p69_n5_harmonize.py
    .venv/bin/python scripts/dark_subspace/p69_n5_harmonize.py --json

Outputs.
  Stdout. Human-readable per-seed and cluster summary tables for the harmonised
  N=5 set, alongside the underlying N=6 cluster summary, with per-metric deltas.

  Disk. JSON file at
    results/dark_subspace/paper_claims/p69_n5_harmonized_2026-05-06.json
  with schema {seeds_kept, seeds_dropped, rows[per-seed metrics],
  cluster_summary{original_score_K_auroc/reconstructed_score_K_auroc/
  residual_score_K_auroc/recon_cos/drop_original_minus_reconstructed each
  n/mean/std/min/max/values}, n6_reference{summary fields},
  n5_minus_n6_delta{per-metric mean/std deltas},
  materiality{verdict NEGLIGIBLE|MATERIAL, threshold_per_metric, abs_deltas},
  notes}.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]

# N=6 cohort. Pattern A keeps the first 5; drops seed 47 (the cohort tail).
P69_SEED_DIRS_FULL = [
    "p69_mixed_sae_seed42_postfix",
    "p69_mixed_sae_seed43",
    "p69_mixed_sae_seed44",
    "p69_mixed_sae_seed45",
    "p69_mixed_sae_seed46",
    "p69_mixed_sae_seed47",
]

P69_SEED_LABELS_FULL = ["42_postfix", "43", "44", "45", "46", "47"]

# Pattern A. Drop seed 47, keep five seeds matching P1B multi-seed.
P69_SEED_DIRS_KEEP = P69_SEED_DIRS_FULL[:5]
P69_SEED_LABELS_KEEP = P69_SEED_LABELS_FULL[:5]
P69_SEED_DIRS_DROP = P69_SEED_DIRS_FULL[5:]
P69_SEED_LABELS_DROP = P69_SEED_LABELS_FULL[5:]

# Materiality thresholds. Per user spec.
DELTA_THRESHOLD = 0.005  # |Δ| < 0.005 across every metric => NEGLIGIBLE.

METRICS_TO_REPORT = [
    "original_score_K_auroc",
    "reconstructed_score_K_auroc",
    "residual_score_K_auroc",
    "drop_original_minus_reconstructed",
    "recon_cos",
]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _summary(values: List[Optional[float]]) -> Dict[str, Any]:
    a = [v for v in values if v is not None]
    if not a:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0, "values": []}
    n = len(a)
    mean = sum(a) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in a) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(min(a)),
        "max": float(max(a)),
        "n": n,
        "values": [float(v) for v in a],
    }


def _extract_row(seed_dir: str, seed_label: str, base: Path) -> Dict[str, Any]:
    """Pull a single seed's canonical metrics from results.json."""
    res_path = base / seed_dir / "results.json"
    if not res_path.exists():
        return {"seed_dir": seed_dir, "seed_label": seed_label, "MISSING": True}
    res = _load(res_path)
    orig = res.get("original", {}).get("score_K_auroc")
    recon = res.get("sae_reconstructed", {}).get("score_K_auroc")
    resid = res.get("residual", {}).get("score_K_auroc")
    recon_cos = res.get("sae_quality", {}).get("reconstruction_cosine")
    drop = (orig - recon) if (orig is not None and recon is not None) else None
    return {
        "seed_dir": seed_dir,
        "seed_label": seed_label,
        "results_path": str(res_path),
        "original_score_K_auroc": orig,
        "reconstructed_score_K_auroc": recon,
        "residual_score_K_auroc": resid,
        "recon_cos": recon_cos,
        "drop_original_minus_reconstructed": drop,
    }


def _compute_cluster(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for f in METRICS_TO_REPORT:
        vals = [r.get(f) for r in rows if r.get(f) is not None]
        out[f] = _summary(vals)
    return out


def _delta_table(n5: Dict[str, Dict[str, Any]], n6: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for f in METRICS_TO_REPORT:
        n5m = n5[f].get("mean")
        n6m = n6[f].get("mean")
        n5s = n5[f].get("std")
        n6s = n6[f].get("std")
        out[f] = {
            "n5_mean": n5m,
            "n6_mean": n6m,
            "delta_mean_n5_minus_n6": (n5m - n6m) if (n5m is not None and n6m is not None) else None,
            "abs_delta_mean": (abs(n5m - n6m) if (n5m is not None and n6m is not None) else None),
            "n5_std": n5s,
            "n6_std": n6s,
            "delta_std_n5_minus_n6": (n5s - n6s) if (n5s is not None and n6s is not None) else None,
        }
    return out


def _materiality_verdict(delta: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    fails = []
    abs_deltas = {}
    for f in METRICS_TO_REPORT:
        ad = delta[f].get("abs_delta_mean")
        abs_deltas[f] = ad
        if ad is None:
            continue
        if ad >= DELTA_THRESHOLD:
            fails.append((f, ad))
    if fails:
        verdict = "MATERIAL"
        reason = (
            "abs delta mean exceeds " + str(DELTA_THRESHOLD)
            + " on metric(s) " + ", ".join(f"{name} (|Δ|={ad:.4f})" for name, ad in fails)
        )
    else:
        verdict = "NEGLIGIBLE"
        reason = (
            "all per-metric |Δ_mean| < " + str(DELTA_THRESHOLD)
            + " across " + str(len(METRICS_TO_REPORT)) + " metrics"
        )
    return {
        "verdict": verdict,
        "threshold": DELTA_THRESHOLD,
        "abs_delta_mean_per_metric": abs_deltas,
        "reason": reason,
        "metrics_checked": METRICS_TO_REPORT,
    }


def _print_table(label: str, summary: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{label}")
    print("  " + "-" * 78)
    print("  metric".ljust(40) + "n".rjust(4) + "mean".rjust(11) + "std".rjust(11) + "min".rjust(8) + "max".rjust(8))
    for f in METRICS_TO_REPORT:
        s = summary[f]
        n = s.get("n")
        mean = s.get("mean")
        std = s.get("std")
        mn = s.get("min")
        mx = s.get("max")
        m_s = f"{mean:.6f}" if mean is not None else "  n/a   "
        s_s = f"{std:.6f}" if std is not None else "  n/a   "
        mn_s = f"{mn:.4f}" if mn is not None else " n/a "
        mx_s = f"{mx:.4f}" if mx is not None else " n/a "
        print("  " + f.ljust(40) + str(n).rjust(4) + m_s.rjust(11) + s_s.rjust(11) + mn_s.rjust(8) + mx_s.rjust(8))


def _print_per_seed(rows: List[Dict[str, Any]]) -> None:
    print("\nPer-seed N=5 (Pattern A: keep [42_postfix, 43, 44, 45, 46]; drop [47]):")
    print("  " + "-" * 100)
    print("  seed_label".ljust(14) + "orig".rjust(10) + "recon".rjust(10)
          + "residual".rjust(10) + "drop".rjust(10) + "recon_cos".rjust(12) + "  results_path")
    for r in rows:
        print(
            "  "
            + str(r.get("seed_label")).ljust(14)
            + (f"{r.get('original_score_K_auroc'):.6f}" if r.get("original_score_K_auroc") is not None else "n/a").rjust(10)
            + (f"{r.get('reconstructed_score_K_auroc'):.6f}" if r.get("reconstructed_score_K_auroc") is not None else "n/a").rjust(10)
            + (f"{r.get('residual_score_K_auroc'):.6f}" if r.get("residual_score_K_auroc") is not None else "n/a").rjust(10)
            + (f"{r.get('drop_original_minus_reconstructed'):.6f}" if r.get("drop_original_minus_reconstructed") is not None else "n/a").rjust(10)
            + (f"{r.get('recon_cos'):.6f}" if r.get("recon_cos") is not None else "n/a").rjust(12)
            + "  " + (r.get("results_path") or "")
        )


def _print_delta(delta: Dict[str, Dict[str, float]]) -> None:
    print("\nN=5 minus N=6 delta (per metric):")
    print("  " + "-" * 88)
    print("  metric".ljust(40) + "N5_mean".rjust(11) + "N6_mean".rjust(11) + "Δmean".rjust(12) + "|Δ|".rjust(11))
    for f in METRICS_TO_REPORT:
        d = delta[f]
        n5m = d.get("n5_mean")
        n6m = d.get("n6_mean")
        dm = d.get("delta_mean_n5_minus_n6")
        ad = d.get("abs_delta_mean")
        n5_s = f"{n5m:.6f}" if n5m is not None else "  n/a   "
        n6_s = f"{n6m:.6f}" if n6m is not None else "  n/a   "
        d_s = f"{dm:+.6f}" if dm is not None else "   n/a   "
        ad_s = f"{ad:.6f}" if ad is not None else "  n/a   "
        print("  " + f.ljust(40) + n5_s.rjust(11) + n6_s.rjust(11) + d_s.rjust(12) + ad_s.rjust(11))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Harmonise the P69 mixed N=6 cohort to N=5 (Pattern A: drop seed 47, keep [42_postfix..46])."
    )
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--json", action="store_true", help="Print full JSON payload to stdout (alongside human table).")
    args = ap.parse_args()

    base = args.repo_root / "runs/dark_subspace/sae_dark_subspace"
    n6_path = args.repo_root / "runs/dark_subspace/sae_noise_floor/p69_aggregate.json"

    rows_keep: List[Dict[str, Any]] = []
    for d, lab in zip(P69_SEED_DIRS_KEEP, P69_SEED_LABELS_KEEP):
        rows_keep.append(_extract_row(d, lab, base))

    rows_drop: List[Dict[str, Any]] = []
    for d, lab in zip(P69_SEED_DIRS_DROP, P69_SEED_LABELS_DROP):
        rows_drop.append(_extract_row(d, lab, base))

    n5_summary = _compute_cluster(rows_keep)

    # N=6 reference. Re-derive from the same per-seed reads, NOT from the
    # cached aggregate, to remove any chance of drift. Also pull the cached
    # aggregate for cross-checking.
    rows_full = rows_keep + rows_drop
    n6_summary_recomputed = _compute_cluster(rows_full)

    n6_cached = None
    if n6_path.exists():
        n6_cached = _load(n6_path)

    delta = _delta_table(n5_summary, n6_summary_recomputed)
    materiality = _materiality_verdict(delta)

    payload = {
        "harmonisation_pattern": "A",
        "harmonisation_pattern_description": (
            "Pattern A: drop seed 47, keep [42_postfix, 43, 44, 45, 46]. "
            "Aligns the Pythia-6.9B mixed cohort to the same five seed "
            "labels as the Pythia-1B multi-seed cohort. The seed_label "
            "'42_postfix' is canonically labelled '42' in the cohort."
        ),
        "seeds_kept": P69_SEED_LABELS_KEEP,
        "seeds_dropped": P69_SEED_LABELS_DROP,
        "rows_kept": rows_keep,
        "rows_dropped": rows_drop,
        "cluster_summary_n5": n5_summary,
        "n6_reference": {
            "rows": rows_full,
            "summary_recomputed": n6_summary_recomputed,
            "cached_aggregate_path": str(n6_path) if n6_cached else None,
            "cached_aggregate_summary": (n6_cached or {}).get("summary"),
        },
        "n5_minus_n6_delta": delta,
        "materiality": materiality,
        "thresholds": {"delta_threshold": DELTA_THRESHOLD},
        "notes": [
            "All 6 P69 SAEs share identical HP (layer=16, d_mult=4, L1=5e-4, 200M tokens, dead-feat resample).",
            "Different random init per SAE training run.",
            "Eval seed=42 on every cohort member; score computation deterministic given SAE.",
            "Cohort mirrors the noise-floor aggregator at runs/dark_subspace/sae_noise_floor/p69_aggregate.json.",
            "Pattern A drops seed 47 (chronological tail of the cohort, training timestamp 20260416_155229) "
            "and aligns to the P1B multi-seed cohort seed list {42, 43, 44, 45, 46}.",
        ],
    }

    # Pretty print.
    print("=" * 88)
    print("Pythia-6.9B (mixed) N=5 harmonisation — Pattern A")
    print("=" * 88)
    print(f"  base dir                        : {base}")
    print(f"  N=6 noise_floor cached aggregate: {n6_path}")
    print(f"  seeds kept                      : {P69_SEED_LABELS_KEEP}")
    print(f"  seeds dropped                   : {P69_SEED_LABELS_DROP}")
    _print_per_seed(rows_keep)
    _print_table("N=5 cluster summary (harmonised)", n5_summary)
    _print_table("N=6 cluster summary (full cohort, recomputed)", n6_summary_recomputed)
    if n6_cached:
        cached = {
            f: {"mean": (n6_cached.get("summary", {}).get(f) or {}).get("mean"),
                 "std":  (n6_cached.get("summary", {}).get(f) or {}).get("std"),
                 "min":  (n6_cached.get("summary", {}).get(f) or {}).get("min"),
                 "max":  (n6_cached.get("summary", {}).get(f) or {}).get("max"),
                 "n":    (n6_cached.get("summary", {}).get(f) or {}).get("n")}
            for f in METRICS_TO_REPORT
        }
        _print_table("N=6 cached aggregate (sanity check)", cached)
    _print_delta(delta)
    print(f"\nMateriality verdict: **{materiality['verdict']}**")
    print(f"  reason: {materiality['reason']}")
    print(f"  threshold: |Δ_mean| < {materiality['threshold']} across all metrics")

    out_dir = args.repo_root / "results/dark_subspace/paper_claims"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "p69_n5_harmonized_2026-05-06.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")

    if args.json:
        print("\nFULL JSON PAYLOAD:")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
