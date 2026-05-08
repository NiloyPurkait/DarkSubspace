#!/usr/bin/env python3
"""
p12b_multiseed_query.py.

Computes the Pythia-12B 5-init cluster summary (mean drop 0.152, std 0.016)
over ``runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed{47..51}/results.json``
and the comparison against the prior 3-init estimate.

Used in Appendix (Pythia-12B replication, ``app:p12b_replication``,
A:1146-1150) of the paper.

Reproduce::

    env/bin/python3 scripts/dark_subspace/p12b_multiseed_query.py [--json] \\
        [--seeds 47,48,49,50,51] [--include-anchor]

Reads only the canonical schema keys: ``original``, ``sae_reconstructed``,
``residual``, ``sae_quality``, ``dark_subspace_effect``.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_TPL = "runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed{seed}/results.json"
CONFIG_TPL = "runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed{seed}/config.json"

# Single-seed anchor (seed 42) used for context with --include-anchor.
ANCHOR_SEED = 42
ANCHOR_RESULTS = "runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed42/results.json"

# Pre-N=5 cluster summary recorded in project notes (seeds 47, 48, 49):
PRIOR_N3_DROP_MEAN = 0.156
PRIOR_N3_DROP_STD = 0.015


def extract(seed: int) -> dict | None:
    """Extract canonical metrics for one seed.

    Parameters
    ----------
    seed : int
        SAE training seed.

    Returns
    -------
    dict or None
        Flat dict of metrics with derived ``computed.*`` fields, or ``None``
        if the per-seed ``results.json`` is missing.
    """
    p = ROOT / RESULTS_TPL.format(seed=seed)
    cfg_p = ROOT / CONFIG_TPL.format(seed=seed)
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    cfg = None
    if cfg_p.exists():
        with open(cfg_p) as fc:
            cfg = json.load(fc)
    out = {"seed": seed, "results_path": str(p.relative_to(ROOT))}
    out["config_path"] = str(cfg_p.relative_to(ROOT)) if cfg_p.exists() else None
    out["layer"] = cfg.get("layer") if cfg else None
    out["model_path"] = cfg.get("model_path") if cfg else None
    orig = d.get("original", {}) or {}
    recon = d.get("sae_reconstructed", {}) or {}
    resid = d.get("residual", {}) or {}
    sq = d.get("sae_quality", {}) or {}
    eff = d.get("dark_subspace_effect", {}) or {}
    out["original.score_K_auroc"] = orig.get("score_K_auroc")
    out["sae_reconstructed.score_K_auroc"] = recon.get("score_K_auroc")
    out["residual.score_K_auroc"] = resid.get("score_K_auroc")
    out["dark_subspace_effect.auroc_drop_from_recon"] = eff.get("auroc_drop_from_recon")
    out["dark_subspace_effect.residual_score_K_auroc"] = eff.get("residual_score_K_auroc")
    out["dark_subspace_effect.memorization_is_dark"] = eff.get("memorization_is_dark")
    out["dark_subspace_effect.d_K_coverage_cosine"] = eff.get("d_K_coverage_cosine")
    out["sae_quality.reconstruction_cosine"] = sq.get("reconstruction_cosine")
    out["sae_quality.mean_l0"] = sq.get("mean_l0")
    out["sae_quality.mean_active_features"] = sq.get("mean_active_features")
    out["sae_quality.l0_pct"] = sq.get("l0_pct")
    out["sae_quality.dictionary_size"] = sq.get("dictionary_size")
    o_k = out["original.score_K_auroc"]
    r_k = out["residual.score_K_auroc"]
    out["computed.residual_minus_original"] = (
        r_k - o_k if (o_k is not None and r_k is not None) else None
    )
    rc = out["sae_quality.reconstruction_cosine"]
    out["computed.recon_cos_gate_pass"] = (rc is not None and rc >= 0.85)
    return out


def fmt_v(v) -> str:
    """Pretty-print a metric value (4-decimal float or stringified scalar)."""
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def cluster_stats(values: list[float], label: str) -> dict:
    """Compute n, mean, std, min, max for a flat list of values.

    Parameters
    ----------
    values : list of float
        Per-seed metric values.
    label : str
        Prefix used in the returned dict keys.

    Returns
    -------
    dict
        Five summary keys, prefixed with ``label``.
    """
    if not values:
        return {label + ".n": 0}
    return {
        label + ".n": len(values),
        label + ".mean": statistics.mean(values),
        label + ".std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        label + ".min": min(values),
        label + ".max": max(values),
        label + ".values": values,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--seeds",
        type=str,
        default="47,48,49,50,51",
        help="Comma-separated seed list (default: P12B 5-init cluster)",
    )
    parser.add_argument(
        "--include-anchor",
        action="store_true",
        help="Also extract the seed-42 single-seed anchor for context",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows = []
    missing = []
    for s in seeds:
        ex = extract(s)
        if ex is None:
            missing.append(s)
        else:
            rows.append(ex)

    anchor = None
    if args.include_anchor:
        anchor_p = ROOT / ANCHOR_RESULTS
        if anchor_p.exists():
            anchor = extract(ANCHOR_SEED)

    drops = [r["dark_subspace_effect.auroc_drop_from_recon"] for r in rows
             if r.get("dark_subspace_effect.auroc_drop_from_recon") is not None]
    cosines = [r["sae_quality.reconstruction_cosine"] for r in rows
               if r.get("sae_quality.reconstruction_cosine") is not None]
    residuals = [r["residual.score_K_auroc"] for r in rows
                 if r.get("residual.score_K_auroc") is not None]
    originals = [r["original.score_K_auroc"] for r in rows
                 if r.get("original.score_K_auroc") is not None]
    recons = [r["sae_reconstructed.score_K_auroc"] for r in rows
              if r.get("sae_reconstructed.score_K_auroc") is not None]

    cluster = {}
    cluster.update(cluster_stats(drops, "drop"))
    cluster.update(cluster_stats(cosines, "recon_cos"))
    cluster.update(cluster_stats(residuals, "residual_K"))
    cluster.update(cluster_stats(originals, "original_K"))
    cluster.update(cluster_stats(recons, "recon_K"))

    prior = {
        "n3.drop.mean": PRIOR_N3_DROP_MEAN,
        "n3.drop.std": PRIOR_N3_DROP_STD,
    }
    if drops:
        n5_mean = cluster["drop.mean"]
        n5_std = cluster["drop.std"]
        prior["n5_minus_n3.drop.mean.delta"] = n5_mean - PRIOR_N3_DROP_MEAN
        prior["n5_minus_n3.drop.std.delta"] = n5_std - PRIOR_N3_DROP_STD
        prior["n5_drop_within_2sigma_of_n3"] = (
            abs(n5_mean - PRIOR_N3_DROP_MEAN) <= 2 * PRIOR_N3_DROP_STD
        )
        prior["std_tightened"] = (n5_std < PRIOR_N3_DROP_STD)
        outliers = []
        for r in rows:
            d = r["dark_subspace_effect.auroc_drop_from_recon"]
            if d is None:
                continue
            z = abs(d - n5_mean) / n5_std if n5_std > 0 else 0.0
            if z > 2.0:
                outliers.append({"seed": r["seed"], "drop": d, "z": z})
        prior["n5_outliers_2sigma"] = outliers

    if args.json:
        payload = {
            "seeds_requested": seeds,
            "missing": missing,
            "rows": rows,
            "cluster_summary": cluster,
            "prior_comparison": prior,
            "anchor_seed42": anchor,
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    print(f"\nP12B mixed-SAE multiseed query")
    print(f"Source: {ROOT}/runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed{{47..51}}/results.json")
    print()
    if missing:
        print(f"MISSING seeds: {missing}")
        print()
    for r in rows:
        print(f"=== seed {r['seed']} ===")
        print(f"  results: {r['results_path']}")
        print(f"  config:  {r['config_path']}")
        print(f"  layer:   {r['layer']}")
        for k in ("original.score_K_auroc",
                  "sae_reconstructed.score_K_auroc",
                  "residual.score_K_auroc",
                  "dark_subspace_effect.auroc_drop_from_recon",
                  "dark_subspace_effect.residual_score_K_auroc",
                  "dark_subspace_effect.d_K_coverage_cosine",
                  "dark_subspace_effect.memorization_is_dark",
                  "sae_quality.reconstruction_cosine",
                  "sae_quality.mean_l0",
                  "sae_quality.mean_active_features",
                  "sae_quality.l0_pct",
                  "sae_quality.dictionary_size",
                  "computed.residual_minus_original",
                  "computed.recon_cos_gate_pass"):
            print(f"    {k:<58} = {fmt_v(r.get(k))}")
        print()

    if drops:
        print("N=5 cluster summary")
        print(f"  drop:        n={cluster['drop.n']}, mean={cluster['drop.mean']:.4f}, "
              f"std={cluster['drop.std']:.4f}, range=[{cluster['drop.min']:.4f}, {cluster['drop.max']:.4f}]")
        print(f"  recon_cos:   n={cluster['recon_cos.n']}, mean={cluster['recon_cos.mean']:.4f}, "
              f"std={cluster['recon_cos.std']:.4f}, range=[{cluster['recon_cos.min']:.4f}, {cluster['recon_cos.max']:.4f}]")
        print(f"  residual_K:  n={cluster['residual_K.n']}, mean={cluster['residual_K.mean']:.4f}, "
              f"std={cluster['residual_K.std']:.4f}, range=[{cluster['residual_K.min']:.4f}, {cluster['residual_K.max']:.4f}]")
        print(f"  original_K:  n={cluster['original_K.n']}, mean={cluster['original_K.mean']:.4f}, "
              f"std={cluster['original_K.std']:.4f}, range=[{cluster['original_K.min']:.4f}, {cluster['original_K.max']:.4f}]")
        print(f"  recon_K:     n={cluster['recon_K.n']}, mean={cluster['recon_K.mean']:.4f}, "
              f"std={cluster['recon_K.std']:.4f}, range=[{cluster['recon_K.min']:.4f}, {cluster['recon_K.max']:.4f}]")
        print()

        print("Comparison vs prior N=3")
        print(f"  Prior N=3 (seeds 47,48,49):    drop_mean={PRIOR_N3_DROP_MEAN:.4f}, drop_std={PRIOR_N3_DROP_STD:.4f}")
        print(f"  Current N=5 (seeds 47-51):     drop_mean={cluster['drop.mean']:.4f}, drop_std={cluster['drop.std']:.4f}")
        print(f"  Delta mean (N=5 - N=3):        {prior['n5_minus_n3.drop.mean.delta']:+.4f}")
        print(f"  Delta std  (N=5 - N=3):        {prior['n5_minus_n3.drop.std.delta']:+.4f}")
        print(f"  N=5 mean within 2 sigma of N=3:    {prior['n5_drop_within_2sigma_of_n3']}")
        print(f"  Std tightened (N=5 < N=3):     {prior['std_tightened']}")
        print(f"  N=5 outliers (|z|>2):          {prior['n5_outliers_2sigma']}")

    if anchor is not None:
        print()
        print("Anchor (seed 42)")
        for k in ("original.score_K_auroc",
                  "sae_reconstructed.score_K_auroc",
                  "residual.score_K_auroc",
                  "dark_subspace_effect.auroc_drop_from_recon",
                  "sae_quality.reconstruction_cosine",
                  "sae_quality.mean_l0"):
            print(f"  {k:<58} = {fmt_v(anchor.get(k))}")


if __name__ == "__main__":
    main()
