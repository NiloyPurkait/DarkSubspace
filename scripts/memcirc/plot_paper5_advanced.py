#!/usr/bin/env python3
"""plot_paper5_advanced.py.

Generates the higher-density supplementary figures (architecture scatter
overlays, layer heatmaps with annotations) from JSON via ``figure_data_loader``.

Used in the Appendix of the paper (advanced and supplementary figure variants).
Reproduce: ``env/bin/python3 scripts/memcirc/plot_paper5_advanced.py``
(writes PDF and PNG, 300 DPI, to ``paper5/figures/``).

All numeric values are sourced via ``figure_data_loader``. Style choices
(IBM Design palette, figure sizes, label offsets, ellipse policy) remain
hardcoded.

The Pythia-12B scaling-curve AUROC may be provisional. When the canonical
aggregate at ``runs/sae_array/p12b_freshinit/aggregate.json`` is not present, the
loader returns a placeholder with the current best-layer estimate from
``behavioral_channels`` and ``is_placeholder = True``. The scaling-curve panel
then draws an open-face marker and skips its CI band.

Figures.
  1. Dark Subspace Heatmap. AUROC across models x components.
  2. SAE Quality vs AUROC Drop. Scatter showing the AUROC gap is decoupled
     from reconstruction quality.
  3. Scaling Curve with Bootstrap CIs. Membership signal across model size.
  4. Architecture Scatter with Family Ellipses. Norm vs BCD by family.
  5. Layer Trajectory Heatmap. Membership AUROC across models x layers
     (loaded from disk).
"""

import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.colors import TwoSlopeNorm

from figure_data_loader import (
    REPO_ROOT,
    MODEL_REGISTRY,
    get_dark_subspace_table,
    get_norm_table,
    get_scaling_curve_data,
    load_bootstrap_for_model,
    load_behavioral_layers,
)

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,        # TrueType
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

ROOT = pathlib.Path(os.environ.get("REPO_ROOT", REPO_ROOT))
FIGDIR = pathlib.Path(os.environ.get("FIGDIR", ROOT / "paper5" / "figures"))
FIGDIR.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (IBM Design).STYLE ONLY
BLUE   = "#648FFF"
ORANGE = "#FE6100"
GREEN  = "#785EF0"
GREY   = "#888888"
RED    = "#DC267F"

GPT_COLOR   = "#648FFF"
LLAMA_COLOR = "#FE6100"
CTRL_COLOR  = "#785EF0"


def save(fig, stem):
    """Save figure as PDF + PNG."""
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIGDIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print(f"  -> {FIGDIR / stem}.pdf / .png")


# ======================================================================
# FIGURE 1: Dark Subspace Heatmap
# ======================================================================
def fig1_dark_subspace_heatmap():
    print("Figure 1: Dark Subspace Heatmap")

    model_order = [
        ("Pythia-6.9B",         "Pythia-6.9B"),
        ("Pythia-6.9B (mixed)", "Pythia-6.9B\n(mixed)"),
        ("Pythia-1B",           "Pythia-1B"),
        ("GPT-Neo-2.7B",        "GPT-Neo"),
        ("OPT-6.7B",            "OPT-6.7B"),
        ("Pythia-12B",          "Pythia-12B"),
        ("Mistral-7B",          "Mistral-7B"),
        ("Qwen2-7B",            "Qwen2-7B"),
        ("Falcon-7B",           r"Falcon-7B$^\dagger$"),
    ]
    keys   = [k for k, _ in model_order]
    models = [d for _, d in model_order]
    components = ["Original", "SAE Recon.", "Residual"]

    table = get_dark_subspace_table(keys)
    data = np.array([
        [table[k]["orig"], table[k]["recon"], table[k]["resid"]] for k in keys
    ])

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    norm = TwoSlopeNorm(vmin=0.50, vcenter=0.65, vmax=1.0)
    cmap = sns.diverging_palette(10, 220, s=80, l=55, as_cmap=True)

    sns.heatmap(
        data, annot=True, fmt=".3f",
        xticklabels=components, yticklabels=models,
        cmap=cmap, norm=norm,
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Membership AUROC", "shrink": 0.85},
        annot_kws={"fontsize": 9, "fontweight": "medium"},
        ax=ax,
    )

    # Black border around the "mixed" control row (row index 1)
    ax.add_patch(plt.Rectangle((0, 1), 3, 1,
                                fill=False, edgecolor="black",
                                linewidth=2.2, clip_on=False))

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    save(fig, "dark_subspace_heatmap")


# ======================================================================
# FIGURE 2: SAE Quality vs AUROC Drop
# ======================================================================
def fig2_sae_quality_scatter():
    print("Figure 2: SAE Quality vs AUROC Drop")

    # x = reconstruction cosine, y = original AUROC - reconstructed AUROC
    model_order = [
        ("Pythia-6.9B",        "Pythia-6.9B",         "GPT"),
        ("Pythia-1B",          "Pythia-1B",           "GPT"),
        ("GPT-Neo-2.7B",       "GPT-Neo",             "GPT"),
        ("OPT-6.7B",           "OPT-6.7B",            "GPT"),
        ("Pythia-12B",         "Pythia-12B",          "GPT"),
        ("Mistral-7B",         "Mistral-7B",          "LLaMA"),
        ("Qwen2-7B",           "Qwen2-7B",            "LLaMA"),
        ("Falcon-7B",          "Falcon-7B",           "GPT"),
        ("Pythia-6.9B (mixed)","Pythia-6.9B\n(mixed)","CTRL"),
    ]
    keys = [m[0] for m in model_order]
    table = get_dark_subspace_table(keys)

    fig, ax = plt.subplots(figsize=(3.25, 3.25))

    xs, ys, names = [], [], []
    for key, disp, fam in model_order:
        x = table[key]["recon_cos"]
        y = table[key]["drop"]
        xs.append(x); ys.append(y); names.append(disp)
        if fam == "GPT":
            c, m = GPT_COLOR, "o"
        elif fam == "LLaMA":
            c, m = LLAMA_COLOR, "s"
        else:
            c, m = CTRL_COLOR, "D"

        if disp == "GPT-Neo":
            ax.scatter(x, y, c=c, marker="*", s=180, zorder=5,
                       edgecolors="black", linewidths=0.8)
        elif disp == "Falcon-7B":
            ax.scatter(x, y, c=c, marker="^", s=80, zorder=5,
                       edgecolors="black", linewidths=0.8)
        else:
            ax.scatter(x, y, c=c, marker=m, s=60, zorder=5,
                       edgecolors="black", linewidths=0.5)

        offsets_ha = {
            "Pythia-6.9B":         (6, 4, "left"),
            "Pythia-1B":           (-6, 8, "right"),
            "GPT-Neo":             (6, -10, "left"),
            "OPT-6.7B":            (-6, 6, "right"),
            "Pythia-12B":          (6, -10, "left"),
            "Mistral-7B":          (6, 4, "left"),
            "Qwen2-7B":            (6, -10, "left"),
            "Falcon-7B":           (6, -10, "left"),
            "Pythia-6.9B\n(mixed)":(-6, -18, "right"),
        }
        dx, dy, ha = offsets_ha.get(disp, (5, 5, "left"))
        ax.annotate(disp, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, color="#333333", ha=ha)

    xs_arr, ys_arr = np.array(xs), np.array(ys)
    slope, intercept, r, _, _ = stats.linregress(xs_arr, ys_arr)
    x_line = np.linspace(0.5, 1.02, 50)
    ax.plot(x_line, slope * x_line + intercept, "--", color=GREY, alpha=0.6,
            linewidth=1.2, zorder=2)
    ax.text(0.52, 0.38, f"$R^2 = {r**2:.2f}$", fontsize=8.5, color=GREY)

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_xlim(0.48, 1.03)
    ax.set_ylim(-0.06, 0.42)
    ax.set_xlabel("Reconstruction cosine similarity")
    ax.set_ylabel("AUROC drop after SAE reconstruction")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GPT_COLOR,
                    markeredgecolor="black", markersize=7, label="GPT-family"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=LLAMA_COLOR,
                    markeredgecolor="black", markersize=7, label="LLaMA-family"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=CTRL_COLOR,
                    markeredgecolor="black", markersize=7, label="Mixed control"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.85,
              edgecolor="#cccccc", fontsize=7.5)

    fig.tight_layout()
    save(fig, "sae_quality_vs_drop")


# ======================================================================
# FIGURE 3: Scaling Curve with Bootstrap CIs
# ======================================================================
def fig3_scaling_curve():
    print("Figure 3: Scaling Curve v2")

    keys = ["Pythia-70M","Pythia-160M","Pythia-410M",
            "Pythia-1B","Pythia-2.8B","Pythia-6.9B","Pythia-12B"]
    sc = get_scaling_curve_data(keys)
    sizes  = np.array(sc["params"])
    aurocs = np.array(sc["aurocs"])
    placeholders = np.array(sc["placeholders"], dtype=bool)
    labels = ["70M","160M","410M","1B","2.8B","6.9B","12B"]

    # Per-model bootstrap CIs from canonical CIs file. behavioral_channels
    # AUROCs (best-layer membership) are not in the bootstrap-CI tables; the
    # closest available CI source is the dark_subspace original-AUROC CI for
    # P1B / P69 / P12B. For models without a CI entry we draw without a band.
    ci_lo = np.full_like(aurocs, np.nan)
    ci_hi = np.full_like(aurocs, np.nan)
    n_boot = None
    n_boot_path = None
    for i, k in enumerate(keys):
        if placeholders[i]:
            continue  # skip CI band when the value is a placeholder
        boot = load_bootstrap_for_model(k)
        if boot is None:
            continue
        # Use the original-AUROC CI half-width as a proxy when the
        # behavioral_channels best-layer AUROC differs from the bootstrap
        # original-AUROC. This avoids mis-reporting CI ranges for points that
        # don't have a direct bootstrap distribution. Mark them by recentering
        # the half-width on the plotted AUROC.
        orig = boot["original"]
        boot_auroc = float(orig["auroc"])
        half_lo = boot_auroc - float(orig["ci_lo"])
        half_hi = float(orig["ci_hi"]) - boot_auroc
        ci_lo[i] = aurocs[i] - half_lo
        ci_hi[i] = aurocs[i] + half_hi
        if n_boot is None:
            n_boot = boot["n_boot"]
            n_boot_path = boot["source"]

    # Tag the source of bootstrap CIs in the printed log so the manifest
    # captures provenance.
    if n_boot_path is not None:
        print(f"  bootstrap CIs: n_boot={n_boot} source={n_boot_path}")

    fig, ax = plt.subplots(figsize=(3.25, 3.0))

    ax.axhspan(0.45, 0.55, color="#f0f0f0", zorder=0)
    ax.axhline(0.50, color=GREY, linewidth=0.7, linestyle="--", alpha=0.6,
               label="Chance level")

    # CI band.only across consecutive points that both have CIs
    have_ci = ~np.isnan(ci_lo)
    if have_ci.any():
        ax.fill_between(
            sizes, np.where(have_ci, ci_lo, aurocs),
            np.where(have_ci, ci_hi, aurocs),
            where=have_ci,
            alpha=0.18, color=BLUE, zorder=2,
        )

    # Line + points
    ax.plot(sizes, aurocs, "-", color=BLUE, linewidth=1.8, zorder=4)
    for i, (xs, ys, ph) in enumerate(zip(sizes, aurocs, placeholders)):
        if ph:
            ax.scatter([xs],[ys], s=70, facecolors="white",
                       edgecolors=BLUE, linewidths=1.4, zorder=5)
        else:
            ax.scatter([xs],[ys], s=55, facecolors=BLUE,
                       edgecolors="white", linewidths=1.0, zorder=5)

    # Emergence zone shading
    ax.axvspan(160e6, 410e6, color="#FFD700", alpha=0.10, zorder=1)
    ax.annotate("emergence\nzone", xy=(270e6, 0.48), fontsize=7,
                color="#B8860B", ha="center", style="italic")

    # P12B annotation
    idx = labels.index("12B")
    note = "canonical aggregate\npending" if placeholders[idx] else "capacity\nsaturation?"
    ax.annotate(note,
                xy=(sizes[idx], aurocs[idx]), xytext=(7e9, 0.72),
                fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555",
                                connectionstyle="arc3,rad=0.2"),
                ha="center")

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(labels, fontsize=8)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlim(5e7, 2e10)
    ax.set_ylim(0.45, 0.95)
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("BCD membership AUROC")

    fig.tight_layout()
    save(fig, "scaling_curve_v2")


# ======================================================================
# FIGURE 4: Architecture Scatter with Family Ellipses
# ======================================================================
def fig4_architecture_scatter():
    print("Figure 4: Architecture Scatter with Ellipses")

    gpt_keys   = ["Pythia-1B","GPT-Neo-2.7B","Pythia-6.9B",
                  "OPT-6.7B","Pythia-12B","Falcon-7B"]
    llama_keys = ["Mistral-7B","Llama-3-8B","Qwen2-7B"]
    display = {
        "Pythia-1B":"Pythia-1B","GPT-Neo-2.7B":"GPT-Neo",
        "Pythia-6.9B":"Pythia-6.9B","OPT-6.7B":"OPT-6.7B",
        "Pythia-12B":"Pythia-12B","Falcon-7B":"Falcon-7B",
        "Mistral-7B":"Mistral-7B","Llama-3-8B":"Llama-3","Qwen2-7B":"Qwen2-7B",
    }
    table_gpt   = get_norm_table(gpt_keys)
    table_llama = get_norm_table(llama_keys)
    gpt_models   = {display[k]: (table_gpt[k]["norm_auroc"],   table_gpt[k]["bcd_auroc"])   for k in gpt_keys}
    llama_models = {display[k]: (table_llama[k]["norm_auroc"], table_llama[k]["bcd_auroc"]) for k in llama_keys}

    fig, ax = plt.subplots(figsize=(3.25, 3.25))

    # Diagonal reference
    ax.plot([0.45, 1.0], [0.45, 1.0], "--", color=GREY, linewidth=0.8,
            alpha=0.5, zorder=1, label="Norm = BCD")

    for name, (x, y) in gpt_models.items():
        ax.scatter(x, y, c=GPT_COLOR, marker="o", s=55, zorder=5,
                   edgecolors="black", linewidths=0.5)
        offsets_gpt = {
            "Pythia-1B": (5, 5, "left"), "GPT-Neo": (5, -12, "left"),
            "Pythia-6.9B": (5, 5, "left"), "OPT-6.7B": (-5, 5, "right"),
            "Pythia-12B": (5, -10, "left"), "Falcon-7B": (5, -10, "left"),
        }
        dx, dy, ha = offsets_gpt.get(name, (5, 5, "left"))
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, color="#333333", ha=ha)

    for name, (x, y) in llama_models.items():
        ax.scatter(x, y, c=LLAMA_COLOR, marker="s", s=55, zorder=5,
                   edgecolors="black", linewidths=0.5)
        offsets_llama = {
            "Mistral-7B": (5, 5, "left"), "Llama-3": (5, -10, "left"),
            "Qwen2-7B": (-5, -12, "right"),
        }
        dx, dy, ha = offsets_llama.get(name, (5, 5, "left"))
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, color="#333333", ha=ha)

    # 95% confidence ellipses (style)
    def draw_ellipse(points, color):
        pts = np.array(points)
        cx, cy = pts.mean(axis=0)
        if len(pts) < 3:
            w, h, angle = 0.12, 0.12, 0
        else:
            cov = np.cov(pts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            chi2_95 = 5.991
            w = 2 * np.sqrt(eigvals[0] * chi2_95)
            h = 2 * np.sqrt(max(eigvals[1], 1e-6) * chi2_95)
        ell = mpatches.Ellipse((cx, cy), w, h, angle=angle,
                                fill=False, edgecolor=color,
                                linewidth=1.5, linestyle="--",
                                alpha=0.6, zorder=3)
        ax.add_patch(ell)

    draw_ellipse(list(gpt_models.values()),   GPT_COLOR)
    draw_ellipse(list(llama_models.values()), LLAMA_COLOR)

    # Annotation arrows for gaps
    p69x, p69y = gpt_models["Pythia-6.9B"]
    ax.annotate("", xy=(p69x, p69y), xytext=(p69x, p69y - 0.25),
                arrowprops=dict(arrowstyle="<->", color=GPT_COLOR,
                                linewidth=1.3, shrinkA=4, shrinkB=4))
    gap_p69 = round((p69y - p69x) * 100)
    ax.text(p69x - 0.064, (p69y + p69y - 0.25) / 2, f"+{gap_p69}pp",
            fontsize=7, color=GPT_COLOR, fontweight="bold", rotation=90, va="center")

    mx, my = llama_models["Mistral-7B"]
    ax.annotate("", xy=(mx, my), xytext=(mx, my - 0.042),
                arrowprops=dict(arrowstyle="<->", color=LLAMA_COLOR,
                                linewidth=1.3, shrinkA=4, shrinkB=4))
    gap_m = round((my - mx) * 100)
    ax.text(mx + 0.018, my - 0.022, f"+{gap_m}pp",
            fontsize=7, color=LLAMA_COLOR, fontweight="bold")

    ax.set_xlim(0.46, 0.95)
    ax.set_ylim(0.55, 0.98)
    ax.set_xlabel("Activation norm AUROC")
    ax.set_ylabel("BCD score$_K$ AUROC")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GPT_COLOR,
                    markeredgecolor="black", markersize=7, label="GPT-family"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=LLAMA_COLOR,
                    markeredgecolor="black", markersize=7, label="LLaMA-family"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.85, edgecolor="#cccccc", fontsize=7.5)

    fig.tight_layout()
    save(fig, "architecture_scatter")


# ======================================================================
# FIGURE 5: Layer Trajectory Heatmap (from disk)
# ======================================================================
def fig5_layer_heatmap():
    print("Figure 5: Layer Trajectory Heatmap")

    # (display_name, registry_label).order matters
    model_specs = [
        ("Pythia-70M",  "Pythia-70M"),
        ("Pythia-160M", "Pythia-160M"),
        ("Pythia-410M", "Pythia-410M"),
        ("Pythia-1B",   "Pythia-1B"),
        ("GPT-Neo-2.7B","GPT-Neo-2.7B"),
        ("Pythia-2.8B", "Pythia-2.8B"),
        ("Pythia-6.9B", "Pythia-6.9B"),
        ("OPT-6.7B",    "OPT-6.7B"),
        ("Pythia-12B",  "Pythia-12B"),
        ("Falcon-7B",   "Falcon-7B"),
        # LLaMA-family
        ("Mistral-7B",  "Mistral-7B"),
        ("Llama-3-8B",  "Llama-3-8B"),
        ("Qwen2-7B",    "Qwen2-7B"),
    ]
    n_bins = 20
    matrix = np.full((len(model_specs), n_bins), np.nan)
    model_labels = []

    for i, (display, key) in enumerate(model_specs):
        model_labels.append(display)
        bl = load_behavioral_layers(key)
        layers = bl["layers"]
        aurocs = bl["aurocs"]
        n_layers = bl["n_layers"]
        for l, auroc in zip(layers, aurocs):
            frac = l / max(n_layers - 1, 1)
            bin_idx = min(int(frac * n_bins), n_bins - 1)
            if np.isnan(matrix[i, bin_idx]):
                matrix[i, bin_idx] = auroc
            else:
                matrix[i, bin_idx] = max(matrix[i, bin_idx], auroc)

    bin_labels = [f"{int(100 * b / n_bins)}%" for b in range(n_bins)]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    cmap = sns.color_palette("mako", as_cmap=True)
    cmap.set_bad(color="#f5f5f5")

    sns.heatmap(
        matrix, cmap=cmap, vmin=0.47, vmax=1.0,
        xticklabels=bin_labels, yticklabels=model_labels,
        cbar_kws={"label": "Membership AUROC", "shrink": 0.85},
        linewidths=0.3, linecolor="white",
        ax=ax, mask=np.isnan(matrix),
    )

    ax.axhline(10, color="black", linewidth=1.5, linestyle="-")

    ax.set_xlabel("Relative model depth")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)

    ytick_labels = ax.get_yticklabels()
    for idx, label in enumerate(ytick_labels):
        if idx < 10:
            label.set_color(GPT_COLOR)
        else:
            label.set_color(LLAMA_COLOR)
        label.set_fontweight("medium")

    for idx, label in enumerate(ax.get_xticklabels()):
        if idx % 3 != 0:
            label.set_visible(False)

    fig.tight_layout()
    save(fig, "layer_heatmap")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Generating paper5 advanced figures")
    print("=" * 60)
    fig1_dark_subspace_heatmap()
    fig2_sae_quality_scatter()
    fig3_scaling_curve()
    fig4_architecture_scatter()
    fig5_layer_heatmap()
    print("=" * 60)
    print("All figures complete.")
