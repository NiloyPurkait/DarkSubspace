#!/usr/bin/env python3
"""
aggregate_multiseed.py.

Reads per-seed ``results.json`` for the cross-architecture multiseed cohort
(P1B, GPT-Neo mult=8, Qwen2 mult=8, Falcon, OPT-6.7B, Gemma-2-2B) and computes
cluster mean, std, and range for the five quantities reported in
``tab:dark_subspace``.

Used in Appendix (cross-architecture multiseed cluster, A:1162-1164) of the
paper.

Reproduce::

    .venv/bin/python scripts/dark_subspace/aggregate_multiseed.py --model all
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAE_BASE = REPO / "runs/dark_subspace/sae_dark_subspace"

# Per-row spec: model_id -> (display_label, expected results.json paths, seeds).
# Each entry's path list is relative to SAE_BASE.
ROW_SPECS = {
    "p1b": {
        "label": "Pythia-1B (mixed, mult=4, layer 14)",
        "dir_pattern": "p1b_mixed_sae_seed{seed}",
        "fallback_dirs_for_existing": ["p1b_mixed_sae_layer14_seed42"],
        "seeds": [42, 43, 44, 45, 46],
        "expected_n": 5,
    },
    "neo_mult8": {
        "label": "GPT-Neo-2.7B (mixed, mult=8, layer 16)",
        "dir_pattern": "neo_mixed_mult8_seed{seed}",
        "fallback_dirs_for_existing": [],
        "seeds": [42, 43, 44, 45, 46],
        "expected_n": 5,
    },
    "qwen2_mult8": {
        "label": "Qwen2-7B (mixed, mult=8, layer 16)",
        "dir_pattern": "qwen2_mixed_mult8_seed{seed}",
        "fallback_dirs_for_existing": [],
        "seeds": [42, 43, 44, 45, 46],
        "expected_n": 5,
    },
    "falcon": {
        "label": "Falcon-7B (mixed, mult=4, layer 16)",
        "dir_pattern": "falcon_mixed_sae_seed{seed}",
        "fallback_dirs_for_existing": [],
        "seeds": [42, 43, 44, 45, 46],
        "expected_n": 5,
    },
    "opt67": {
        "label": "OPT-6.7B (mixed, mult=4, layer 24)",
        "dir_pattern": "opt67_mixed_sae_seed{seed}",
        "fallback_dirs_for_existing": [],
        "seeds": [42, 43, 44, 45, 46],
        "expected_n": 5,
    },
    "gemma2_2b": {
        "label": "Gemma-2-2B (mixed, mult=4, layer 16)",
        "dir_pattern": "gemma2_2b_mixed_sae_seed{seed}",
        "fallback_dirs_for_existing": ["gemma2_2b_mixed_sae"],
        "seeds": [42, 43, 44, 45, 46],
        "expected_n": 5,
    },
}


def find_results(spec, seed):
    """Locate ``results.json`` for a (spec, seed) pair.

    Parameters
    ----------
    spec : dict
        One ``ROW_SPECS`` entry.
    seed : int
        SAE training seed.

    Returns
    -------
    tuple[Path, bool]
        ``(path, found)``. The canonical pattern is tried first, then any
        ``fallback_dirs_for_existing`` entry for the legacy single-seed-42
        directory layout.
    """
    primary = SAE_BASE / spec["dir_pattern"].format(seed=seed) / "results.json"
    if primary.exists():
        return primary, True
    if seed == 42 and spec.get("fallback_dirs_for_existing"):
        for fb in spec["fallback_dirs_for_existing"]:
            cand = SAE_BASE / fb / "results.json"
            if cand.exists():
                return cand, True
    return primary, False


def extract_metrics(rj_path: Path) -> dict:
    """Read canonical metrics from a dark-subspace ``results.json``.

    Parameters
    ----------
    rj_path : Path
        Path to a per-seed ``results.json`` with the canonical schema
        (``original``, ``sae_reconstructed``, ``residual``, ``sae_quality``,
        ``dark_subspace_effect``).

    Returns
    -------
    dict
        Subset of metrics used by downstream cluster aggregation.
    """
    with open(rj_path) as f:
        d = json.load(f)
    orig = d.get("original", {}).get("score_K_auroc")
    recon = d.get("sae_reconstructed", {}).get("score_K_auroc")
    res = d.get("residual", {}).get("score_K_auroc")
    sq = d.get("sae_quality", {})
    rcos = sq.get("reconstruction_cosine")
    dse = d.get("dark_subspace_effect", {})
    drop = dse.get("auroc_drop_from_recon")
    mean_active = sq.get("mean_active_features")
    n_member = d.get("n_member")
    n_nonmember = d.get("n_nonmember")
    return {
        "orig_auroc": orig,
        "recon_auroc": recon,
        "residual_auroc": res,
        "drop": drop,
        "recon_cos": rcos,
        "mean_active_features": mean_active,
        "n_member": n_member,
        "n_nonmember": n_nonmember,
    }


def aggregate_one(model_key: str, spec: dict) -> dict:
    """Aggregate one model row across its seeds.

    Parameters
    ----------
    model_key : str
        Key from ``ROW_SPECS``.
    spec : dict
        Matching ``ROW_SPECS`` entry.

    Returns
    -------
    dict
        Per-seed and cluster-level summary. Cluster stats are computed only
        over seeds whose ``results.json`` exists on disk.
    """
    per_seed = {}
    missing_seeds = []
    for seed in spec["seeds"]:
        rj_path, found = find_results(spec, seed)
        if not found:
            missing_seeds.append(seed)
            per_seed[str(seed)] = {
                "results_json": str(rj_path),
                "found": False,
            }
            continue
        m = extract_metrics(rj_path)
        m["results_json"] = str(rj_path)
        m["found"] = True
        per_seed[str(seed)] = m

    found_seeds = [s for s in spec["seeds"] if str(s) in per_seed and per_seed[str(s)]["found"]]
    n_available = len(found_seeds)

    cluster = {
        "n_available": n_available,
        "expected_n": spec["expected_n"],
        "complete": n_available == spec["expected_n"],
        "missing_seeds": missing_seeds,
        "label": spec["label"],
    }
    for metric in ["orig_auroc", "recon_auroc", "residual_auroc", "drop", "recon_cos"]:
        vals = [per_seed[str(s)][metric] for s in found_seeds if per_seed[str(s)][metric] is not None]
        if not vals:
            cluster[f"{metric}_mean"] = None
            cluster[f"{metric}_stdev"] = None
            cluster[f"{metric}_min"] = None
            cluster[f"{metric}_max"] = None
            continue
        cluster[f"{metric}_mean"] = sum(vals) / len(vals)
        cluster[f"{metric}_stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cluster[f"{metric}_min"] = min(vals)
        cluster[f"{metric}_max"] = max(vals)

    return {
        "model_key": model_key,
        "spec": {k: v for k, v in spec.items() if k != "fallback_dirs_for_existing"},
        "per_seed": per_seed,
        "cluster": cluster,
    }


def print_human(model_key: str, agg: dict):
    """Pretty-print an aggregated row to stdout.

    Parameters
    ----------
    model_key : str
        Row key.
    agg : dict
        Output of ``aggregate_one``.
    """
    c = agg["cluster"]
    label = c["label"]
    n = c["n_available"]
    nexp = c["expected_n"]
    print(f"\n=== {model_key}: {label} ===")
    print(f"  N: {n}/{nexp} (complete: {c['complete']}, missing seeds: {c['missing_seeds']})")
    if n == 0:
        print("  NO results; all seeds are absent from the configured output directories.")
        return
    for metric in ["orig_auroc", "recon_auroc", "residual_auroc", "drop", "recon_cos"]:
        m = c[f"{metric}_mean"]
        s = c[f"{metric}_stdev"]
        mn = c[f"{metric}_min"]
        mx = c[f"{metric}_max"]
        if m is None:
            print(f"  {metric}: NO data")
        else:
            print(f"  {metric}: mean={m:.4f}  std={s:.4f}  range=[{mn:.4f}, {mx:.4f}]  (N={n})")
    print(f"  per-seed:")
    for seed in agg["spec"]["seeds"]:
        ps = agg["per_seed"][str(seed)]
        if ps["found"]:
            print(f"    seed={seed:3d}  drop={ps['drop']:.4f}  recon_cos={ps['recon_cos']:.4f}  resid_auroc={ps['residual_auroc']:.4f}")
        else:
            print(f"    seed={seed:3d}  PENDING ({ps['results_json']})")


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=list(ROW_SPECS) + ["all"])
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output JSON path. If --model all and --out unspecified, writes "
            "results/dark_subspace/paper_claims/multiseed_<model>.json per model and a combined "
            "results/dark_subspace/paper_claims/multiseed_cluster.json."
        ),
    )
    args = ap.parse_args()

    out_dir = REPO / "results/dark_subspace/paper_claims"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = list(ROW_SPECS)
    else:
        models = [args.model]

    combined = {}
    for mk in models:
        agg = aggregate_one(mk, ROW_SPECS[mk])
        combined[mk] = agg
        print_human(mk, agg)
        if args.model == "all":
            per_model_out = out_dir / f"multiseed_{mk}.json"
        elif args.out:
            per_model_out = Path(args.out)
        else:
            per_model_out = out_dir / f"multiseed_{mk}.json"
        with open(per_model_out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"  -> wrote {per_model_out}")

    if args.model == "all":
        combined_out = out_dir / "multiseed_cluster.json"
        with open(combined_out, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n=== Wrote combined cluster JSON: {combined_out} ===")


if __name__ == "__main__":
    main()
