#!/usr/bin/env python3
"""
sae_noise_floor_aggregate.py.

Aggregates the six independently trained Pythia-6.9B mixed-data SAE seeds into
the cross-seed noise floor (drop_mean 0.209, drop_std 0.003, n_sigma 62) and
writes ``runs/dark_subspace/sae_noise_floor/p69_aggregate.json``.

Used in the Introduction headline, Methods note on noise floor, and the
appendix six-seed cohort table.
Reproduce:
    env/bin/python3 scripts/dark_subspace/sae_noise_floor_aggregate.py

Per-seed inputs (CPU-only post-processing of existing data, no GPU needed).
  ``runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed{42_postfix,43..47}/results.json``

Fields extracted per seed.
  original.score_K_auroc, sae_reconstructed.score_K_auroc,
  residual.score_K_auroc, sae_quality.reconstruction_cosine,
  sae_quality.L0 (from config, if absent skip),
  drop = original - reconstructed.

Output ``runs/dark_subspace/sae_noise_floor/p69_aggregate.json`` contains per-seed
rows, mean, std, min, max for each field, cross-seed CV (std over |mean|),
and the noise-floor table used in the appendix.
"""

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

P69_SEED_DIRS = [
    "p69_mixed_sae_seed42_postfix",
    "p69_mixed_sae_seed43",
    "p69_mixed_sae_seed44",
    "p69_mixed_sae_seed45",
    "p69_mixed_sae_seed46",
    "p69_mixed_sae_seed47",
]


def _load(p):
    return json.loads(p.read_text())


def _summary(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate the six P6.9B mixed-SAE seeds into the cross-seed noise floor."
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: two levels above this script).",
    )
    args = ap.parse_args()
    base = args.repo_root / "runs/dark_subspace/sae_dark_subspace"
    out_dir = args.repo_root / "runs/dark_subspace/sae_noise_floor"

    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for d in P69_SEED_DIRS:
        res_path = base / d / "results.json"
        cfg_path = base / d / "config.json"
        if not res_path.exists():
            print(f"MISSING {res_path}")
            continue
        res = _load(res_path)
        cfg = _load(cfg_path) if cfg_path.exists() else {}
        row = {
            "seed_dir": d,
            "seed_label": d.replace("p69_mixed_sae_seed", ""),
            "model_id": res.get("model"),
            "sae_path": res.get("sae_path"),
            "original_score_K_auroc": res.get("original", {}).get("score_K_auroc"),
            "reconstructed_score_K_auroc": res.get("sae_reconstructed", {}).get("score_K_auroc"),
            "residual_score_K_auroc": res.get("residual", {}).get("score_K_auroc"),
            "recon_cos": res.get("sae_quality", {}).get("reconstruction_cosine"),
            "L0": res.get("sae_quality", {}).get("L0"),
            "drop_original_minus_reconstructed": (
                res.get("original", {}).get("score_K_auroc", 0.0)
                - res.get("sae_reconstructed", {}).get("score_K_auroc", 0.0)
                if res.get("original", {}).get("score_K_auroc") is not None
                and res.get("sae_reconstructed", {}).get("score_K_auroc") is not None
                else None
            ),
        }
        per_seed.append(row)

    agg_fields = [
        "original_score_K_auroc",
        "reconstructed_score_K_auroc",
        "residual_score_K_auroc",
        "recon_cos",
        "L0",
        "drop_original_minus_reconstructed",
    ]
    summary = {
        f: _summary([r.get(f) for r in per_seed]) for f in agg_fields
    }

    # Noise-floor interpretation:
    #   If the DROP metric has std sigma across seeds, then the single-seed DROP
    #   is significant only if > z_0.95 * sigma (one-sided). We report N-sigma for
    #   the mean DROP relative to the cross-seed noise floor.
    drop_mean = summary["drop_original_minus_reconstructed"]["mean"]
    drop_std = summary["drop_original_minus_reconstructed"]["std"]
    n_sigma = (drop_mean / drop_std) if (drop_mean is not None and drop_std is not None and drop_std > 0) else None

    # Residual AUROC noise floor: the std of residual_score_K_auroc gives the
    # empirical floor for the geometric-separability claim
    res_std = summary["residual_score_K_auroc"]["std"]
    res_mean = summary["residual_score_K_auroc"]["mean"]

    out = {
        "model_tag": "p69",
        "roster_size": len(per_seed),
        "expected": 6,
        "source": "aggregation of six existing Pythia-6.9B mixed-data SAE seeds",
        "per_seed": per_seed,
        "summary": summary,
        "noise_floor": {
            "drop_mean": drop_mean,
            "drop_std_cross_seed": drop_std,
            "drop_n_sigma": n_sigma,
            "residual_AUROC_mean": res_mean,
            "residual_AUROC_std_cross_seed": res_std,
            "description": (
                "Cross-seed std of the drop (original-reconstructed score_K_AUROC) establishes the "
                "empirical noise floor. A measured drop of drop_mean at n_sigma cross-seed-std "
                "above zero supports the separability claim. The residual_AUROC cross-seed std "
                "gives the noise floor on the membership signal that survives into the SAE-residual "
                "space, used to frame the residual-probe claim."
            ),
        },
        "notes": [
            "All six SAEs share identical hyperparameters (layer=16, d_mult=4, L1=5e-4, 200M tokens, dead-feature resample).",
            "Different random initialisation per SAE training run.",
            "Evaluation seed 42 on all seeds, score computation deterministic given SAE.",
        ],
    }
    out_path = out_dir / "p69_aggregate.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps({"summary": summary, "noise_floor": out["noise_floor"]}, indent=2))


if __name__ == "__main__":
    main()
