#!/usr/bin/env python3
"""plot_score_distributions.py.

Generates ``figures/score_distributions_simple.pdf`` and the four-panel full
version for the Pythia-6.9B mixed-data SAE pooled score distribution comparison.

Used in Results (main body) and Appendix of the paper.
Reproduce: ``env/bin/python3 scripts/dark_subspace/plot_score_distributions.py``
(writes PDF and PNG, 300 DPI, to ``manuscript/figures/``).

Outputs:
  manuscript/figures/score_distributions.pdf         2x3 full version (appendix)
  manuscript/figures/score_distributions_simple.pdf  1x3 simple version (main body)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import os
ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
MEMBER_ONLY = ROOT / "runs/dark_subspace/sae_dark_subspace/p69_epoch5_v2/results.json"
# Mixed-data SAE single-seed source (seed 42, recon_cos=0.977). Per-seed
# AUROCs: orig=0.803, recon=0.591, resid=0.783; these are the closest single
# seed to the N=6 aggregate means 0.803, 0.594, 0.779 stored at
# runs/dark_subspace/sae_noise_floor/p69_aggregate.json.
MIXED_SAE   = ROOT / "runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed42_postfix/results.json"
# Simple figure pools all 6 mixed-data SAE seeds for the cohort-level KDE
# and reports the N=6 mean AUROC.
MIXED_SAE_SEEDS = [
    ROOT / f"runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed{s}/results.json"
    for s in ["42_postfix", "43", "44", "45", "46", "47"]
]
OUT_DIR     = ROOT / "manuscript/figures"

BLUE      = "#0072B2"   # Okabe-Ito blue  (member)
VERMILLION = "#D55E00"  # Okabe-Ito vermillion (nonmember)

COL_HEADERS = [
    r"Original $\mathbf{h}$",
    r"SAE Reconstruction $\mathbf{\hat{h}}$",
    r"Residual $\mathbf{h} - \mathbf{\hat{h}}$",
]

SCORE_KEYS = ["score_K_original", "score_K_recon", "score_K_residual"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path):
    with open(path) as f:
        d = json.load(f)
    pts = d["per_text_scores"]
    labels = np.array(pts["labels"])
    scores = {k: np.array(pts[k]) for k in SCORE_KEYS}
    return labels, scores


def load_data_pooled(paths):
    """Concatenate per-text scores across seeds.

    Returns the per-seed mean AUROCs alongside pooled labels and scores
    for KDE plotting.
    """
    all_labels = []
    pooled = {k: [] for k in SCORE_KEYS}
    per_seed_aurocs = {k: [] for k in SCORE_KEYS}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        pts = d["per_text_scores"]
        labels = np.array(pts["labels"])
        all_labels.append(labels)
        for k in SCORE_KEYS:
            arr = np.array(pts[k])
            pooled[k].append(arr)
            per_seed_aurocs[k].append(roc_auc_score(labels, arr))
    labels_pooled = np.concatenate(all_labels)
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    mean_aurocs = {k: float(np.mean(per_seed_aurocs[k])) for k in SCORE_KEYS}
    std_aurocs  = {k: float(np.std(per_seed_aurocs[k], ddof=1)) for k in SCORE_KEYS}
    return labels_pooled, pooled, mean_aurocs, std_aurocs


def compute_auroc(labels, scores):
    return roc_auc_score(labels, scores)


def optimal_threshold(labels, scores):
    """Find threshold where member and nonmember density curves cross.
    Uses a simple approach: sweep thresholds and find where
    P(score > t | member) - P(score > t | nonmember) changes sign minimally,
    approximated by the threshold that maximises Youden's J."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx]


def style_ax(ax):
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def draw_panel(ax, labels, scores, title=None, show_ylabel=True, xlim=None,
               auroc_override=None):
    """Draw a single density panel.

    If ``auroc_override`` is provided, use that value in the annotation instead
    of the AUROC computed from the supplied (possibly pooled) labels/scores.
    Threshold and KDEs are still computed from the supplied data.
    """
    mem = scores[labels == 1]
    non = scores[labels == 0]
    auroc = compute_auroc(labels, scores) if auroc_override is None else auroc_override
    thresh = optimal_threshold(labels, scores)

    # KDE densities
    sns.kdeplot(mem, ax=ax, color=BLUE, fill=True, alpha=0.35,
                label="Member", linewidth=1.4, bw_adjust=0.8)
    sns.kdeplot(non, ax=ax, color=VERMILLION, fill=True, alpha=0.35,
                label="Nonmember", linewidth=1.4, bw_adjust=0.8)

    # Optimal threshold line
    ax.axvline(thresh, color="#555555", ls="--", lw=0.9, alpha=0.7)

    # AUROC annotation.top-right corner
    ax.text(0.96, 0.94, f"AUROC = {auroc:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc",
                      alpha=0.85))

    if title:
        ax.set_title(title, fontsize=10, pad=6)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=10)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("")
    if xlim is not None:
        ax.set_xlim(xlim)
    style_ax(ax)


# ---------------------------------------------------------------------------
# Figure 1: Full 2x3 (appendix)
# ---------------------------------------------------------------------------

def make_full_figure():
    labels_mo, scores_mo = load_data(MEMBER_ONLY)
    labels_mx, scores_mx = load_data(MIXED_SAE)

    # Compute shared x-limits across all panels
    all_vals = np.concatenate([scores_mo[k] for k in SCORE_KEYS] +
                              [scores_mx[k] for k in SCORE_KEYS])
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    margin = (hi - lo) * 0.08
    xlim = (lo - margin, hi + margin)

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0),
                             sharex=True, sharey="row")

    row_labels = ["Member-only SAE", "Mixed-data SAE"]
    datasets = [(labels_mo, scores_mo), (labels_mx, scores_mx)]

    for row, ((labels, scores), row_label) in enumerate(zip(datasets, row_labels)):
        for col, key in enumerate(SCORE_KEYS):
            ax = axes[row, col]
            title = COL_HEADERS[col] if row == 0 else None
            draw_panel(ax, labels, scores[key], title=title,
                       show_ylabel=(col == 0), xlim=xlim)
            # X-axis label only on bottom row
            if row == 1:
                ax.set_xlabel("Score", fontsize=10)

        # Row label on leftmost panel
        axes[row, 0].annotate(
            row_label, xy=(-0.38, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", rotation=90,
            ha="center", va="center")

    # Single legend at the bottom
    handles, lab = axes[0, 0].get_legend_handles_labels()
    # Remove per-panel legends
    for axrow in axes:
        for ax in axrow:
            leg = ax.get_legend()
            if leg:
                leg.remove()
    fig.legend(handles, lab, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0.06, 0.04, 1.0, 1.0])
    fig.subplots_adjust(hspace=0.28, wspace=0.18)

    # Save
    fig.savefig(OUT_DIR / "score_distributions.pdf",
                bbox_inches="tight", dpi=300,
                metadata={"Creator": "plot_score_distributions.py"})
    fig.savefig(OUT_DIR / "score_distributions.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'score_distributions.pdf'}")
    print(f"Saved: {OUT_DIR / 'score_distributions.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Simple 1x3 (main body).mixed-data SAE only
# ---------------------------------------------------------------------------

def make_simple_figure():
    # Pool all 6 Pythia-6.9B mixed-data SAE seeds for the KDE and report the
    # N=6 per-seed mean AUROC in the annotation (orig 0.803, recon 0.594,
    # resid 0.779 in the canonical aggregate).
    labels, scores, mean_aurocs, std_aurocs = load_data_pooled(MIXED_SAE_SEEDS)

    print("N=6 per-seed AUROC means (std):")
    for k in SCORE_KEYS:
        print(f"  {k}: {mean_aurocs[k]:.4f} ({std_aurocs[k]:.4f})")

    # X-limits from pooled data
    all_vals = np.concatenate([scores[k] for k in SCORE_KEYS])
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    margin = (hi - lo) * 0.08
    xlim = (lo - margin, hi + margin)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5),
                             sharex=True, sharey=True)

    for col, key in enumerate(SCORE_KEYS):
        ax = axes[col]
        draw_panel(ax, labels, scores[key], title=COL_HEADERS[col],
                   show_ylabel=(col == 0), xlim=xlim,
                   auroc_override=mean_aurocs[key])
        ax.set_xlabel("Score", fontsize=10)

    # Single legend
    handles, lab = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()
    fig.legend(handles, lab, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    fig.subplots_adjust(wspace=0.12)

    fig.savefig(OUT_DIR / "score_distributions_simple.pdf",
                bbox_inches="tight", dpi=300,
                metadata={"Creator": "plot_score_distributions.py"})
    fig.savefig(OUT_DIR / "score_distributions_simple.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'score_distributions_simple.pdf'}")
    print(f"Saved: {OUT_DIR / 'score_distributions_simple.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Global matplotlib settings
    plt.rcParams.update({
        "pdf.fonttype": 42,        # TrueType fonts in PDF
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    })

    make_full_figure()
    make_simple_figure()
    print("\nDone.")
