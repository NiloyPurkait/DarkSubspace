#!/usr/bin/env python3
"""plot_privacy_aware_comparison.py.

Generates ``figures/privacy_aware_comparison.pdf`` showing standard vs
``$\\dK$``-penalised SAE on Pythia-6.9B as a grouped bar chart of membership
detection AUROC across four SAE conditions (Member-only, Mixed-data,
Privacy-aware lambda=0.1, Privacy-aware lambda=1.0), each with
Original, SAE-Recon, and Residual bars.

Used in Results of the paper.
Reproduce: ``env/bin/python3 scripts/dark_subspace/plot_privacy_aware_comparison.py``
(writes PDF and PNG, 300 DPI, to ``outputs/figures/`` unless ``FIGDIR`` is set).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data (verified from disk) ──────────────────────────────────────────────
CONDITIONS = [
    "Member-only\nSAE",
    "Mixed-data\nSAE",
    "Privacy-aware\n(\u03bb=0.1)",
    "Privacy-aware\n(\u03bb=1.0)",
]

# Each row: (original, sae_recon, residual)
AUROC = np.array([
    [0.803, 0.593, 0.805],  # Member-only
    [0.803, 0.649, 0.807],  # Mixed-data
    [0.803, 0.803, 0.520],  # Privacy-aware lambda=0.1
    [0.803, 0.802, 0.540],  # Privacy-aware lambda=1.0
])

# 95% bootstrap CI for SAE-Recon bars only: (lower, upper)
RECON_CI = np.array([
    [0.568, 0.618],
    [0.624, 0.672],
    [0.783, 0.822],
    [0.782, 0.821],
])

BAR_LABELS = ["Original", "SAE-Recon", "Residual"]

# ── Style ──────────────────────────────────────────────────────────────────
# Colorblind-friendly palette (Wong 2011, adapted)
COLOR_ORIGINAL = "#888888"   # Gray
COLOR_RECON    = "#0072B2"   # Blue (deuteranope-safe)
COLOR_RESIDUAL = "#E69F00"   # Orange (deuteranope-safe)
COLORS = [COLOR_ORIGINAL, COLOR_RECON, COLOR_RESIDUAL]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Layout geometry ────────────────────────────────────────────────────────
n_groups = len(CONDITIONS)
n_bars = 3
bar_width = 0.22
group_gap = 0.15  # extra space between groups

# Compute x positions: groups are spaced by (n_bars * bar_width + group_gap)
group_width = n_bars * bar_width + group_gap
group_centers = np.arange(n_groups) * group_width
offsets = np.array([-bar_width, 0, bar_width])

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.5))

for j in range(n_bars):
    x_positions = group_centers + offsets[j]
    values = AUROC[:, j]

    # Error bars only on SAE-Recon (j == 1)
    if j == 1:
        yerr_lower = values - RECON_CI[:, 0]
        yerr_upper = RECON_CI[:, 1] - values
        yerr = np.array([yerr_lower, yerr_upper])
    else:
        yerr = None

    bars = ax.bar(
        x_positions, values,
        width=bar_width,
        color=COLORS[j],
        edgecolor="white",
        linewidth=0.5,
        label=BAR_LABELS[j],
        yerr=yerr,
        capsize=3,
        error_kw={"linewidth": 1.0, "capthick": 1.0, "color": "#333333"},
        zorder=3,
    )

# ── Annotations: drop in pp above each SAE-Recon bar ──────────────────────
recon_x = group_centers + offsets[1]
for i in range(n_groups):
    drop_pp = round((AUROC[i, 1] - AUROC[i, 0]) * 100)
    label = f"{drop_pp:+d}pp" if drop_pp != 0 else "0pp"

    # Position annotation above the CI upper bound (or the bar top)
    y_top = RECON_CI[i, 1] + 0.012
    ax.annotate(
        label,
        xy=(recon_x[i], y_top),
        ha="center", va="bottom",
        fontsize=7.5, fontweight="bold",
        color="#333333",
    )

# ── Reference line at chance ───────────────────────────────────────────────
ax.axhline(y=0.5, color="#999999", linewidth=0.8, linestyle="--", zorder=1,
           label="Chance (0.5)")

# ── Axes ───────────────────────────────────────────────────────────────────
ax.set_ylabel("Membership Detection AUROC")
ax.set_ylim(0.40, 0.90)
ax.set_yticks(np.arange(0.4, 0.95, 0.1))
ax.set_xticks(group_centers)
ax.set_xticklabels(CONDITIONS, linespacing=1.15)

# Light horizontal grid only
ax.yaxis.grid(True, linewidth=0.4, color="#d0d0d0", zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)

# ── Legend ──────────────────────────────────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
# Reorder: Original, SAE-Recon, Residual, Chance
order = [0, 1, 2, 3]
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
    fancybox=False,
)

# ── Save ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
out_dir = Path(os.environ.get("FIGDIR", ROOT / "outputs" / "figures"))
out_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(out_dir / "privacy_aware_comparison.pdf")
fig.savefig(out_dir / "privacy_aware_comparison.png")
plt.close(fig)

print(f"Saved: {out_dir / 'privacy_aware_comparison.pdf'}")
print(f"Saved: {out_dir / 'privacy_aware_comparison.png'}")
