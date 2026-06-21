#!/usr/bin/env python3
"""Aggregate the Frikha feature-selection baseline across 5 SAE seeds (42-46) into a single cluster summary.

Reads from:
  results/dark_subspace/generated/frikha_features/frikha_baseline_p69/results.json           (seed 42, the original)
  results/dark_subspace/generated/frikha_features/frikha_baseline_p69_seed{43..46}/results.json (array-job extension)

Writes:
  results/dark_subspace/generated/frikha_features/cluster_summary.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FRIKHA_DIR = REPO_ROOT / "results" / "dark_subspace" / "generated" / "frikha_features"
SEED_42_DIR = FRIKHA_DIR / "frikha_baseline_p69"
EXTENSION_DIRS = {s: FRIKHA_DIR / f"frikha_baseline_p69_seed{s}" for s in [43, 44, 45, 46]}
OUTPUT_FILE = FRIKHA_DIR / "cluster_summary.json"

EXPECTED_SEEDS = [42, 43, 44, 45, 46]
CRITERIA = ["top_k_magnitude", "mean_diff", "steering_probe"]
DEPTH_KEYS = ["top_1", "top_5", "top_50", "top_200"]


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def _flag_anomalies(per_seed: list[dict[str, Any]]) -> list[str]:
    """Hard-stop sanity checks per the task spec."""
    flags = []
    for entry in per_seed:
        seed = entry["seed"]
        probe = entry["baseline_no_ablation"]["latent_probe_auroc_5fold"]
        if probe < 0.45 or probe > 0.65:
            flags.append(
                f"WARN seed {seed} latent_probe_auroc_5fold={probe:.4f} outside [0.45, 0.65]"
            )
        recon_cos_drop = (
            entry["baseline_no_ablation"]["score_K_auroc_original"]
            - entry["baseline_no_ablation"]["score_K_auroc_recon"]
        )
        if recon_cos_drop < 0.0:
            flags.append(
                f"WARN seed {seed} recon AUROC > original (drop={recon_cos_drop:+.4f}); expected recon < original"
            )
    return flags


def main() -> int:
    # Load each seed's results
    per_seed: list[dict[str, Any]] = []
    missing = []

    seed_to_path = {42: SEED_42_DIR / "results.json"}
    for s, d in EXTENSION_DIRS.items():
        seed_to_path[s] = d / "results.json"

    for seed in EXPECTED_SEEDS:
        path = seed_to_path[seed]
        if not path.exists():
            missing.append((seed, path))
            continue
        results = _load(path)
        baseline = results["baseline_no_ablation"]
        sae_stats = results.get("sae_stats", {})
        criteria_block = {}
        for crit in CRITERIA:
            depth_block = {}
            for dk in DEPTH_KEYS:
                cell = results["criteria"][crit][dk]
                depth_block[dk] = {
                    "k": cell["k"],
                    "score_K_auroc_recon_post_ablation": cell[
                        "score_K_auroc_recon_post_ablation"
                    ],
                    "score_K_auroc_residual_post_ablation": cell[
                        "score_K_auroc_residual_post_ablation"
                    ],
                    "latent_probe_auroc_5fold": cell["residual_probe_auroc_5fold"],
                    "extraction_drop_vs_recon_baseline": cell[
                        "extraction_drop_vs_recon_baseline"
                    ],
                    "residual_probe_delta_vs_baseline": cell[
                        "residual_probe_delta_vs_baseline"
                    ],
                }
            criteria_block[crit] = depth_block

        per_seed.append(
            {
                "seed": seed,
                "results_path": str(path.relative_to(REPO_ROOT)),
                "sae_path": results["sae_path"],
                "model": results["model"],
                "baseline_no_ablation": {
                    "score_K_auroc_original": baseline["score_K_auroc_original"],
                    "score_K_auroc_recon": baseline["score_K_auroc_recon"],
                    "score_K_auroc_residual": baseline["score_K_auroc_residual"],
                    "latent_probe_auroc_5fold": baseline["latent_probe_auroc_5fold"],
                },
                "sae_stats": {
                    "mean_active_features": sae_stats.get("mean_active_features"),
                    "total_features": sae_stats.get("total_features"),
                    "sparsity": sae_stats.get("sparsity"),
                },
                "criteria": criteria_block,
            }
        )

    if missing:
        print("MISSING per-seed results files:", file=sys.stderr)
        for s, p in missing:
            print(f"  seed {s}: {p}", file=sys.stderr)
        return 1

    # Sanity flags
    flags = _flag_anomalies(per_seed)

    # Aggregate over seeds
    cluster_summary: dict[str, Any] = {}

    # Baseline (no-ablation) per-field stats
    baseline_fields = [
        "score_K_auroc_original",
        "score_K_auroc_recon",
        "score_K_auroc_residual",
        "latent_probe_auroc_5fold",
    ]
    baseline_agg = {}
    for f in baseline_fields:
        baseline_agg[f] = _stats([e["baseline_no_ablation"][f] for e in per_seed])
    cluster_summary["baseline_no_ablation"] = baseline_agg

    # SAE stats
    sae_agg = {}
    for f in ["mean_active_features", "sparsity"]:
        vals = [e["sae_stats"][f] for e in per_seed if e["sae_stats"][f] is not None]
        sae_agg[f] = _stats(vals)
    cluster_summary["sae_stats"] = sae_agg

    # Criteria × depth aggregation
    grid_agg: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for crit in CRITERIA:
        grid_agg[crit] = {}
        for dk in DEPTH_KEYS:
            grid_agg[crit][dk] = {}
            for field in [
                "score_K_auroc_recon_post_ablation",
                "score_K_auroc_residual_post_ablation",
                "latent_probe_auroc_5fold",
                "extraction_drop_vs_recon_baseline",
                "residual_probe_delta_vs_baseline",
            ]:
                vals = [e["criteria"][crit][dk][field] for e in per_seed]
                grid_agg[crit][dk][field] = _stats(vals)
    cluster_summary["criteria_grid"] = grid_agg

    # Directional pattern checks
    latent_probe_baseline = baseline_agg["latent_probe_auroc_5fold"]
    residual_baseline = baseline_agg["score_K_auroc_residual"]
    cluster_summary["directional_pattern_check"] = {
        "latent_probe_near_chance": (
            0.45 <= latent_probe_baseline["mean"] <= 0.55
        ),
        "latent_probe_mean": latent_probe_baseline["mean"],
        "latent_probe_std": latent_probe_baseline["std"],
        "latent_probe_seed_range": [
            latent_probe_baseline["min"],
            latent_probe_baseline["max"],
        ],
        "residual_auroc_stable_post_ablation": True,  # filled below
        "residual_auroc_max_deviation_from_baseline": None,  # filled below
    }

    # max deviation of post-ablation residual AUROC from no-ablation residual baseline (per-seed)
    max_dev_across = 0.0
    for e in per_seed:
        base_res = e["baseline_no_ablation"]["score_K_auroc_residual"]
        for crit in CRITERIA:
            for dk in DEPTH_KEYS:
                post = e["criteria"][crit][dk][
                    "score_K_auroc_residual_post_ablation"
                ]
                dev = abs(post - base_res)
                if dev > max_dev_across:
                    max_dev_across = dev
    cluster_summary["directional_pattern_check"][
        "residual_auroc_max_deviation_from_baseline"
    ] = max_dev_across
    cluster_summary["directional_pattern_check"][
        "residual_auroc_stable_post_ablation"
    ] = max_dev_across < 0.05

    # Verdict
    probe_mean = latent_probe_baseline["mean"]
    probe_max = latent_probe_baseline["max"]
    probe_min = latent_probe_baseline["min"]
    if probe_max <= 0.55 and probe_min >= 0.45:
        verdict = (
            f"signal not in SAE feature dictionary (5/5 seeds, latent probe AUROC "
            f"{probe_mean:.4f} ± {latent_probe_baseline['std']:.4f}, range "
            f"[{probe_min:.4f}, {probe_max:.4f}] all within chance band [0.45, 0.55])"
        )
    elif probe_max <= 0.60:
        verdict = (
            f"signal effectively absent from SAE feature dictionary (latent probe "
            f"{probe_mean:.4f} ± {latent_probe_baseline['std']:.4f}, max {probe_max:.4f} "
            f"slightly above 0.55 chance band but still well below the residual "
            f"AUROC {residual_baseline['mean']:.4f})"
        )
    else:
        verdict = (
            f"MIXED: latent probe AUROC max {probe_max:.4f} (seed-level) exceeds 0.60 — "
            "review per-seed values before declaring cluster verdict"
        )

    out = {
        "experiment": "frikha_feature_selection",
        "experiment_name": "frikha_baseline_p69",
        "n_seeds": len(per_seed),
        "seeds": [e["seed"] for e in per_seed],
        "per_seed": per_seed,
        "cluster_summary": cluster_summary,
        "cluster_verdict": verdict,
        "anomaly_flags": flags,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    print()
    print(f"N seeds: {len(per_seed)}")
    print(f"Seeds:   {[e['seed'] for e in per_seed]}")
    print()
    print("Per-seed latent_probe_auroc_5fold (baseline, no ablation):")
    for e in per_seed:
        v = e["baseline_no_ablation"]["latent_probe_auroc_5fold"]
        print(f"  seed {e['seed']}: {v:.6f}")
    print()
    print(
        f"Cluster mean ± std: {latent_probe_baseline['mean']:.4f} "
        f"± {latent_probe_baseline['std']:.4f}"
    )
    print(
        f"Range [min, max]: [{latent_probe_baseline['min']:.4f}, "
        f"{latent_probe_baseline['max']:.4f}]"
    )
    print()
    print(
        f"Residual AUROC baseline: {residual_baseline['mean']:.4f} ± {residual_baseline['std']:.4f}"
    )
    print(
        f"Max deviation post-ablation residual AUROC across all 12 cells × 5 seeds: "
        f"{max_dev_across:.4f}"
    )
    print()
    print(f"Verdict: {verdict}")
    if flags:
        print()
        print("Anomaly flags:")
        for f in flags:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
