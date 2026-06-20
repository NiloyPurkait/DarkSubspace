#!/usr/bin/env python3
"""
aggregate_topk_scope.py.

Aggregates the 15 per-seed ``results.json`` files produced by the
TopK SAE scope test (K in {32, 64, 128} x seeds {42..46}) into a single
``cluster_summary.json``.

Schema (output JSON):
  experiment, n_per_K, per_K, per_seed, cross_K_summary, cluster_verdict.

Reproduce::

    .venv/bin/python scripts/dark_subspace/aggregate_topk_scope.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAE_BASE = REPO / "runs/dark_subspace/sae_dark_subspace"
OUT_PATH = REPO / "results/dark_subspace/generated/topk_scope/cluster_summary.json"

KS = [32, 64, 128]
SEEDS = [42, 43, 44, 45, 46]


def extract_metrics(rj_path: Path) -> dict:
    """Read canonical metrics from a dark-subspace ``results.json``."""
    with open(rj_path) as f:
        d = json.load(f)
    orig = d.get("original", {}).get("score_K_auroc")
    recon = d.get("sae_reconstructed", {}).get("score_K_auroc")
    res = d.get("residual", {}).get("score_K_auroc")
    res_norm = d.get("residual", {}).get("norm_auroc")
    sq = d.get("sae_quality", {})
    rcos = sq.get("reconstruction_cosine")
    dse = d.get("dark_subspace_effect", {})
    drop = dse.get("auroc_drop_from_recon")
    mean_active = sq.get("mean_active_features")
    return {
        "original_score_K_auroc": orig,
        "sae_recon_score_K_auroc": recon,
        "residual_score_K_auroc": res,
        "residual_norm_auroc": res_norm,
        "recon_cosine": rcos,
        "delta_recon": drop,
        "mean_active_features": mean_active,
    }


def cluster_stats(values: list[float]) -> dict:
    """Mean, std, min, max over a list of floats."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": sum(vals) / len(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def main():
    per_seed_records: list[dict] = []
    missing: list[tuple[int, int]] = []

    for K in KS:
        for seed in SEEDS:
            rj_dir = SAE_BASE / f"topk_p69_topk{K}_seed{seed}"
            rj_path = rj_dir / "results.json"
            cfg_path = rj_dir / "config.json"
            if not rj_path.exists():
                missing.append((K, seed))
                continue
            metrics = extract_metrics(rj_path)
            rec = {
                "K": K,
                "seed": seed,
                "results_json": str(rj_path.relative_to(REPO)),
                "config_json": str(cfg_path.relative_to(REPO)) if cfg_path.exists() else None,
                **metrics,
            }
            # Sign indicator: did residual_score_K beat sae_recon_score_K?
            r = metrics["residual_score_K_auroc"]
            s = metrics["sae_recon_score_K_auroc"]
            rec["residual_gt_recon"] = bool(r is not None and s is not None and r > s)
            rec["cosine_ge_0p90"] = bool(
                metrics["recon_cosine"] is not None and metrics["recon_cosine"] >= 0.90
            )
            per_seed_records.append(rec)

    if missing:
        print(f"MISSING ({len(missing)}/15): {missing}", file=sys.stderr)
        if len(missing) == 15:
            print("ABORT: no per-seed results yet. Wait for job to land.", file=sys.stderr)
            sys.exit(2)

    METRIC_KEYS = [
        "original_score_K_auroc",
        "sae_recon_score_K_auroc",
        "residual_score_K_auroc",
        "residual_norm_auroc",
        "recon_cosine",
        "delta_recon",
        "mean_active_features",
    ]

    per_K: dict[str, dict] = {}
    for K in KS:
        rows = [r for r in per_seed_records if r["K"] == K]
        n_avail = len(rows)
        sign_test_n = sum(1 for r in rows if r["residual_gt_recon"])
        cosine_pass_n = sum(1 for r in rows if r["cosine_ge_0p90"])
        block = {
            "n_available": n_avail,
            "n_expected": len(SEEDS),
            "sign_test_residual_gt_recon": f"{sign_test_n}/{n_avail}",
            "cosine_ge_0p90_pass": f"{cosine_pass_n}/{n_avail}",
        }
        for mk in METRIC_KEYS:
            block[mk] = cluster_stats([r[mk] for r in rows])
        per_K[str(K)] = block

    # Cross-K (all 15 seeds) summary for the headline finding.
    all_sign = sum(1 for r in per_seed_records if r["residual_gt_recon"])
    all_cos = sum(1 for r in per_seed_records if r["cosine_ge_0p90"])
    cross_K_summary = {
        "n_total": len(per_seed_records),
        "n_expected": len(KS) * len(SEEDS),
        "sign_test_residual_gt_recon": f"{all_sign}/{len(per_seed_records)}",
        "cosine_ge_0p90_pass": f"{all_cos}/{len(per_seed_records)}",
        "residual_score_K_auroc_mean": cluster_stats(
            [r["residual_score_K_auroc"] for r in per_seed_records]
        )["mean"],
        "sae_recon_score_K_auroc_mean": cluster_stats(
            [r["sae_recon_score_K_auroc"] for r in per_seed_records]
        )["mean"],
        "delta_recon_mean": cluster_stats(
            [r["delta_recon"] for r in per_seed_records]
        )["mean"],
        "recon_cosine_mean": cluster_stats(
            [r["recon_cosine"] for r in per_seed_records]
        )["mean"],
    }

    # Honest cluster verdict — checks per-K and cross-K patterns.
    verdicts = []
    for K in KS:
        block = per_K[str(K)]
        st = block["sign_test_residual_gt_recon"]
        if st == "5/5":
            verdicts.append(f"K={K}: residual > recon clean (5/5)")
        else:
            verdicts.append(f"K={K}: PARTIAL/FLIPPED sign test ({st})")
    overall = (
        f"residual > recon survives in TopK SAE ({cross_K_summary['sign_test_residual_gt_recon']} sign test); "
        f"cosine gate ≥ 0.90 passes {cross_K_summary['cosine_ge_0p90_pass']}"
    )
    cluster_verdict = {
        "overall": overall,
        "per_K_signs": verdicts,
    }

    out = {
        "experiment": "topk_scope",
        "experiment_label": "TopK SAE scope test — Pythia-6.9B, layer 16, mult=4",
        "n_per_K": len(SEEDS),
        "Ks": KS,
        "seeds": SEEDS,
        "missing": missing,
        "per_K": per_K,
        "per_seed": per_seed_records,
        "cross_K_summary": cross_K_summary,
        "cluster_verdict": cluster_verdict,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"Cross-K sign test residual > recon: {cross_K_summary['sign_test_residual_gt_recon']}")
    print(f"Cross-K cosine >= 0.90: {cross_K_summary['cosine_ge_0p90_pass']}")
    print(f"Verdict: {overall}")
    for v in verdicts:
        print(f"  {v}")


if __name__ == "__main__":
    main()
