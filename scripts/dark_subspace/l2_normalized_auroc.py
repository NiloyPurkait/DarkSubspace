#!/usr/bin/env python3
"""l2_normalized_auroc.py.

Re-evaluates membership AUROC after L2-normalising each residual vector
to unit length. Rules out residual-norm artefact as the explanation.

Used in Section "Results" (R:109-110) and Appendix `tab:l2_normalized`
(A:957) of the paper.
Reproduce: env/bin/python3 scripts/dark_subspace/l2_normalized_auroc.py [--json]

For each model with dark subspace results, loads per-text scores,
L2-normalises residual vectors (using saved norms), and recomputes AUROC
on projection scores that factor out the norm confound.

The existing `all_models_l2_bootstrap.json` already has these results.
This script reads from that file and from results.json files to produce
a clean comparison table.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DS_BASE = ROOT / "runs" / "dark_subspace" / "sae_dark_subspace"


def canonical_auroc(labels, scores):
    """AUROC with sign correction: max(auroc, 1-auroc).

    The channel-decomposition scoring direction may vary by model.
    The canonical convention is to report the sign that gives AUROC >= 0.5.
    """
    raw = float(roc_auc_score(labels, scores))
    return max(raw, 1.0 - raw)


def compute_norm_auroc(norms, labels):
    """AUROC of using raw norms as membership score."""
    return canonical_auroc(labels, norms)


def compute_l2_normalized_auroc_from_scores(scores, norms, labels):
    """Normalize scores by their corresponding norms, compute AUROC.

    For residual: score_K_residual / norm_residual gives the cosine-like
    projection score on a unit-length residual.
    """
    norms = np.array(norms)
    scores = np.array(scores)
    # Avoid division by zero
    safe_norms = np.where(norms > 1e-10, norms, 1e-10)
    normalized = scores / safe_norms
    return canonical_auroc(labels, normalized)


def main():
    parser = argparse.ArgumentParser(description="L2-Normalized Residual AUROC Comparison")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # v2 models with per-text scores
    v2_models = [
        ("p1b", "p1b_epoch5_v2"),
        ("neo", "neo_epoch5_v2"),
        ("p69", "p69_epoch5_v2"),
        ("opt67", "opt67_epoch5_v2"),
        ("p12b", "p12b_epoch5_v2"),
        ("mistral", "mistral_epoch5_v2"),
        ("qwen2", "qwen2_epoch5_v2"),
    ]

    # Also check pre-computed L2 bootstrap file
    l2_bootstrap_path = DS_BASE / "all_models_l2_bootstrap.json"
    l2_bootstrap = {}
    if l2_bootstrap_path.exists():
        with open(l2_bootstrap_path) as f:
            l2_bootstrap = json.load(f)

    all_results = []
    for short_name, v2_dir_name in v2_models:
        v2_path = DS_BASE / v2_dir_name / "results.json"
        if not v2_path.exists():
            print(f"SKIP {short_name}: no v2 results", file=sys.stderr)
            continue

        with open(v2_path) as f:
            r = json.load(f)

        pts = r["per_text_scores"]
        labels = np.array(pts["labels"])
        score_orig = np.array(pts["score_K_original"])
        score_recon = np.array(pts["score_K_recon"])
        score_resid = np.array(pts["score_K_residual"])
        norm_orig = np.array(pts["norm_original"])
        norm_recon = np.array(pts["norm_recon"])
        norm_resid = np.array(pts["norm_residual"])

        # Raw AUROCs (with sign correction)
        raw_orig = canonical_auroc(labels, score_orig)
        raw_recon = canonical_auroc(labels, score_recon)
        raw_resid = canonical_auroc(labels, score_resid)

        # Norm AUROCs (the confound)
        norm_auroc_orig = compute_norm_auroc(norm_orig, labels)
        norm_auroc_recon = compute_norm_auroc(norm_recon, labels)
        norm_auroc_resid = compute_norm_auroc(norm_resid, labels)

        # L2-normalized AUROCs
        l2_orig = compute_l2_normalized_auroc_from_scores(score_orig, norm_orig, labels)
        l2_recon = compute_l2_normalized_auroc_from_scores(score_recon, norm_recon, labels)
        l2_resid = compute_l2_normalized_auroc_from_scores(score_resid, norm_resid, labels)

        result = {
            "model": short_name,
            "source": str(v2_path),
            "n_texts": len(labels),
            "raw": {
                "original": raw_orig,
                "recon": raw_recon,
                "residual": raw_resid,
            },
            "norm_auroc": {
                "original": norm_auroc_orig,
                "recon": norm_auroc_recon,
                "residual": norm_auroc_resid,
            },
            "l2_normalized": {
                "original": l2_orig,
                "recon": l2_recon,
                "residual": l2_resid,
            },
            "delta_resid_raw_minus_l2": raw_resid - l2_resid,
        }

        # Cross-check against pre-computed bootstrap file
        if short_name in l2_bootstrap:
            lb = l2_bootstrap[short_name]
            result["cross_check_l2_bootstrap"] = {
                "original_l2": lb["Original"]["l2"],
                "recon_l2": lb["SAE-Recon"]["l2"],
                "residual_l2": lb["Residual"]["l2"],
                "residual_l2_ci": lb["Residual"].get("l2_ci", None),
            }

        all_results.append(result)

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print("\n" + "=" * 110)
        print("L2-NORMALIZED RESIDUAL AUROC COMPARISON")
        print("=" * 110)

        # Summary table
        print(f"\n{'Model':<10} {'Raw Orig':>10} {'Raw Recon':>10} {'Raw Resid':>10} {'Norm Resid':>11} {'L2 Orig':>10} {'L2 Recon':>10} {'L2 Resid':>10} {'Delta':>8}")
        print("-" * 110)
        for r in all_results:
            delta = r["delta_resid_raw_minus_l2"]
            print(f"{r['model']:<10} {r['raw']['original']:>10.4f} {r['raw']['recon']:>10.4f} {r['raw']['residual']:>10.4f} {r['norm_auroc']['residual']:>11.4f} {r['l2_normalized']['original']:>10.4f} {r['l2_normalized']['recon']:>10.4f} {r['l2_normalized']['residual']:>10.4f} {delta:>+8.4f}")

        # Dark subspace effect comparison
        print(f"\n{'Model':<10} {'Raw Drop':>10} {'L2 Drop':>10} {'Dark Sub?':>10}")
        print("-" * 50)
        for r in all_results:
            raw_drop = r["raw"]["recon"] - r["raw"]["original"]
            l2_drop = r["l2_normalized"]["recon"] - r["l2_normalized"]["original"]
            dark = "YES" if l2_drop < -0.02 else "NO"
            print(f"{r['model']:<10} {raw_drop:>+10.4f} {l2_drop:>+10.4f} {dark:>10}")

        # Source files
        print("\nSOURCE FILES:")
        for r in all_results:
            print(f"  {r['model']}: {r['source']}")

        if l2_bootstrap_path.exists():
            print(f"\nPre-computed L2 bootstrap: {l2_bootstrap_path}")


if __name__ == "__main__":
    main()
