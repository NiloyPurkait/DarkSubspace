#!/usr/bin/env python3
"""
recall_label_validation.py.

Validates the loss-median recall label against ROUGE-L extraction with
Spearman rank correlation and partial correlation controlling for loss,
across eight models.

Used in Methods (recall label validation), Appendix BCD details, and the
adversarial-robustness recall-label discussion.
Reproduce:
    env/bin/python3 scripts/memcirc/recall_label_validation.py

Reads pre-computed extractability predictor results to report Pearson and
Spearman correlations between per-text loss and per-text ROUGE-L scores.
Also reads recall_proxy_validation results where available for additional
correlation data.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Recall Label Validation: loss vs ROUGE-L correlation")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Source 1: bcd_extractability (9 models, 1000 texts each)
    ext_base = ROOT / "runs" / "memcirc" / "bcd_extractability"
    # Source 2: recall_proxy_validation (subset of models)
    rpv_base = ROOT / "runs" / "memcirc" / "recall_proxy_validation"

    all_results = []

    # Read from bcd_extractability (primary source)
    if ext_base.exists():
        for model_dir in sorted(ext_base.iterdir()):
            fp = model_dir / "extractability_predictor.json"
            if not fp.exists():
                continue
            with open(fp) as f:
                r = json.load(f)

            corrs = r.get("correlations", {})
            loss_rouge = corrs.get("loss_vs_rouge", {})
            sk_rouge = corrs.get("score_K_vs_rouge", {})
            sr_rouge = corrs.get("score_R_vs_rouge", {})

            partial = r.get("partial_correlations", {})
            sk_partial = partial.get("score_K_vs_rouge_controlling_loss", {})
            sr_partial = partial.get("score_R_vs_rouge_controlling_loss", {})

            stats = r.get("summary_stats", {})

            result = {
                "model": model_dir.name,
                "source": str(fp),
                "n_texts": r.get("n_texts", r.get("n_texts_valid", "?")),
                "loss_vs_rouge_l": {
                    "spearman_rho": loss_rouge.get("spearman_rho"),
                    "spearman_p": loss_rouge.get("p"),
                },
                "score_K_vs_rouge_l": {
                    "spearman_rho": sk_rouge.get("spearman_rho"),
                    "spearman_p": sk_rouge.get("p"),
                },
                "score_R_vs_rouge_l": {
                    "spearman_rho": sr_rouge.get("spearman_rho"),
                    "spearman_p": sr_rouge.get("p"),
                },
                "partial_score_K_vs_rouge_controlling_loss": {
                    "rho": sk_partial.get("rho"),
                    "p": sk_partial.get("p"),
                },
                "partial_score_R_vs_rouge_controlling_loss": {
                    "rho": sr_partial.get("rho"),
                    "p": sr_partial.get("p"),
                },
                "summary_stats": stats,
            }
            all_results.append(result)

    # Also read recall_proxy_validation for Pearson correlations
    rpv_results = []
    if rpv_base.exists():
        for model_dir in sorted(rpv_base.iterdir()):
            fp = model_dir / "recall_proxy_validation.json"
            if not fp.exists():
                continue
            with open(fp) as f:
                r = json.load(f)

            corr = r.get("correlation", {})
            rpv_results.append({
                "model": model_dir.name,
                "source": str(fp),
                "n_texts": r.get("n_texts_valid", r.get("n_texts_total", "?")),
                "spearman_rho": corr.get("spearman_rho"),
                "spearman_p": corr.get("spearman_pvalue"),
                "pearson_r": corr.get("pearson_r"),
                "pearson_p": corr.get("pearson_pvalue"),
                "agreement_rate": r.get("classification_agreement", {}).get("agreement_rate"),
            })

    if args.json:
        print(json.dumps({"bcd_extractability": all_results, "recall_proxy_validation": rpv_results}, indent=2))
    else:
        print("\n" + "=" * 120)
        print("RECALL LABEL VALIDATION: Loss vs ROUGE-L Correlation")
        print("=" * 120)

        # Primary table: loss vs ROUGE-L (Spearman)
        print(f"\n{'Model':<18} {'n':>6} {'Loss-ROUGE rho':>15} {'p-value':>12} {'score_K-ROUGE':>15} {'score_R-ROUGE':>15}")
        print("-" * 90)
        for r in all_results:
            lr = r["loss_vs_rouge_l"]
            sk = r["score_K_vs_rouge_l"]
            sr = r["score_R_vs_rouge_l"]
            rho = f"{lr['spearman_rho']:.4f}" if lr['spearman_rho'] is not None else "N/A"
            pv = f"{lr['spearman_p']:.2e}" if lr['spearman_p'] is not None else "N/A"
            sk_rho = f"{sk['spearman_rho']:.4f}" if sk['spearman_rho'] is not None else "N/A"
            sr_rho = f"{sr['spearman_rho']:.4f}" if sr['spearman_rho'] is not None else "N/A"
            print(f"{r['model']:<18} {r['n_texts']:>6} {rho:>15} {pv:>12} {sk_rho:>15} {sr_rho:>15}")

        # Recall proxy validation (Pearson + Spearman)
        if rpv_results:
            print(f"\n--- Recall Proxy Validation (member-only texts) ---")
            print(f"{'Model':<18} {'n':>6} {'Spearman rho':>14} {'Pearson r':>12} {'Agreement':>12}")
            print("-" * 70)
            for r in rpv_results:
                sp = f"{r['spearman_rho']:.4f}" if r['spearman_rho'] is not None else "N/A"
                pr = f"{r['pearson_r']:.4f}" if r['pearson_r'] is not None else "N/A"
                ag = f"{r['agreement_rate']:.3f}" if r['agreement_rate'] is not None else "N/A"
                print(f"{r['model']:<18} {r['n_texts']:>6} {sp:>14} {pr:>12} {ag:>12}")

        # Source files
        print("\nSOURCE FILES (bcd_extractability):")
        for r in all_results:
            print(f"  {r['source']}")
        if rpv_results:
            print("SOURCE FILES (recall_proxy_validation):")
            for r in rpv_results:
                print(f"  {r['source']}")


if __name__ == "__main__":
    main()
