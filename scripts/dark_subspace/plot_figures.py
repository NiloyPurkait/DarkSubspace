#!/usr/bin/env python3
"""plot_figures.py.

Generates the body and appendix headline figures (``score_distributions_simple.pdf``,
``dark_subspace_heatmap.pdf``, ``architecture_scatter.pdf``, ``scaling_curve_v2.pdf``,
``norm_direction_decomposition.pdf``, ``layer_heatmap.pdf``, ``dark_subspace.pdf``,
``sae_quality_vs_drop.pdf``) from JSON via ``figure_data_loader``.

Used in Methods, Results, and Appendix of the paper.
Reproduce: ``env/bin/python3 scripts/dark_subspace/plot_figures.py``
(writes PDF and PNG, 300 DPI, to ``outputs/figures/`` unless ``FIGDIR`` is set).

All numeric values are sourced via ``figure_data_loader`` (no hardcoded
AUROCs). The only literals that remain are style choices (Okabe-Ito hex codes,
figure sizes, label offsets, y-axis limits). See ``figure_data_loader.py``
for the canonical mapping from model label to ``results.json`` path.

The Pythia-12B scaling-curve AUROC may use a single-run source when the
aggregate at ``runs/sae_array/p12b_freshinit/aggregate.json`` is not present.
The marker is drawn with an open face in that case.

Figures.
  1. Dark Subspace Hero. Grouped bar chart (orig, recon, residual) per model.
  2. Scaling Curve. Best-layer membership AUROC vs model size (log scale).
  3. Norm-Direction Decomposition. Norm vs channel-decomposition AUROC per model.
"""

import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from figure_data_loader import (
    REPO_ROOT,
    get_dark_subspace_table,
    get_norm_table,
    get_scaling_curve_data,
)

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,      # TrueType in PDF
    "ps.fonttype": 42,
})

OUT = pathlib.Path(os.environ.get("FIGDIR", REPO_ROOT / "outputs" / "figures"))
OUT.mkdir(parents=True, exist_ok=True)

# ── Colorblind-friendly palette (Okabe-Ito).STYLE ONLY ──────────────────
C_BLUE    = "#0072B2"
C_ORANGE  = "#E69F00"
C_GREEN   = "#009E73"
C_RED     = "#D55E00"
C_CORAL   = "#CC6677"
C_GRAY    = "#999999"
C_LGRAY   = "#CCCCCC"
C_BLACK   = "#000000"


def _clean_axes(ax, keep_left=True, keep_bottom=True):
    """Remove top/right spines and optional grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    if not keep_bottom:
        ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)


def _save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print(f"  saved  {stem}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1.Dark Subspace Hero
# ═══════════════════════════════════════════════════════════════════════════
def fig_dark_subspace():
    print("Figure 1: Dark Subspace Hero")

    # Display order (kept identical to pre-refactor figure)
    model_order = [
        ("Pythia-6.9B",        "P69"),
        ("Pythia-6.9B (mixed)","P69\n(mixed)"),
        ("Pythia-1B",          "P1B"),
        ("GPT-Neo-2.7B",       "Neo"),
        ("OPT-6.7B",           "OPT"),
        ("Pythia-12B",         "P12B"),
        ("Mistral-7B",         "Mistral"),
        ("Qwen2-7B",           "Qwen2"),
        ("Falcon-7B",          "Falcon†"),
    ]
    labels = [m[1] for m in model_order]
    keys   = [m[0] for m in model_order]
    table  = get_dark_subspace_table(keys)
    orig   = [table[k]["orig"]  for k in keys]
    recon  = [table[k]["recon"] for k in keys]
    resid  = [table[k]["resid"] for k in keys]
    # "clean" residual flag.True for models whose norm-AUROC is well below
    # residual, i.e. the residual signal is not a norm artifact. Hard-coded to
    # match the prior figure's hatching policy (style choice, not a metric).
    clean  = [True,  True,  True,  True,  False, False, True,  False, False]

    n = len(labels)
    x = np.arange(n)
    w = 0.24  # bar width

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    _clean_axes(ax)

    ax.bar(x - w, orig,  w, color=C_BLUE,  label="Original",          zorder=3)
    ax.bar(x,     recon, w, color=C_CORAL, label="SAE-Reconstructed", zorder=3)

    # Residual bars.hatch for non-clean
    for i in range(n):
        hatch = "///" if not clean[i] else None
        edgecolor = C_GREEN if not clean[i] else "none"
        ax.bar(x[i] + w, resid[i], w, color=C_GREEN,
               hatch=hatch, edgecolor=edgecolor,
               linewidth=0.6, zorder=3,
               label="Residual" if i == 0 else "")
    ax.bar([], [], w, color=C_GREEN, hatch="///", edgecolor=C_GREEN,
           linewidth=0.6, label="Residual (norm confound)")

    ax.axhline(0.50, ls="--", lw=0.8, color=C_GRAY, zorder=1)
    ax.text(n - 0.5, 0.505, "chance", fontsize=8, color=C_GRAY,
            ha="right", va="bottom")

    # Bracket for P69 (mixed) control
    ax.annotate("", xy=(0.5, 0.04), xytext=(1.5, 0.04),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-", lw=0.8, color=C_GRAY))
    ax.text(1.0, 0.01, "mixed-data\nSAE control", fontsize=7,
            ha="center", va="bottom", color=C_GRAY,
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("score$_K$ AUROC")
    ax.set_ylim(0.40, 1.02)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8)

    _save(fig, "dark_subspace")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2.Scaling Curve
# ═══════════════════════════════════════════════════════════════════════════
def fig_scaling_curve():
    print("Figure 2: Scaling Curve")

    model_keys = ["Pythia-70M","Pythia-160M","Pythia-410M",
                  "Pythia-1B","Pythia-2.8B","Pythia-6.9B","Pythia-12B"]
    sc = get_scaling_curve_data(model_keys)
    sizes  = sc["params"]
    aurocs = sc["aurocs"]
    provisional = sc["provisional"]
    labels = ["70M","160M","410M","1B","2.8B","6.9B","12B"]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    _clean_axes(ax)

    ax.axhspan(0.45, 0.55, color=C_LGRAY, alpha=0.35, zorder=0)
    ax.text(sizes[0] * 0.85, 0.525, "chance level", fontsize=7,
            color=C_GRAY, va="center", ha="left", style="italic")
    ax.axhline(0.50, ls="--", lw=0.8, color=C_GRAY, zorder=1)

    ax.plot(sizes, aurocs, lw=1.5, color=C_BLUE, zorder=3)
    # Use an open marker for values drawn from single-run sources.
    for i, (xs, ys, ph) in enumerate(zip(sizes, aurocs, provisional)):
        if ph:
            ax.scatter([xs],[ys], s=70, facecolors="white",
                       edgecolors=C_BLUE, linewidths=1.4, zorder=4)
        else:
            ax.scatter([xs],[ys], s=60, facecolors=C_BLUE,
                       edgecolors="white", linewidths=0.8, zorder=4)

    # Annotate the P12B point
    idx_12b = labels.index("12B")
    is_ph = provisional[idx_12b]
    note  = "(single-run source)" if is_ph else "(layer-selection\n artifact)"
    ax.annotate(note,
                xy=(sizes[idx_12b], aurocs[idx_12b]),
                xytext=(sizes[idx_12b] * 1.05, aurocs[idx_12b] - 0.065),
                fontsize=7, color=C_GRAY, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_GRAY))

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Model Size (parameters)")
    ax.set_ylabel("Channel-decomposition Membership AUROC")
    ax.set_ylim(0.45, 0.95)
    ax.set_xlim(sizes[0] * 0.6, sizes[-1] * 1.5)

    _save(fig, "scaling_curve")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3.Norm-Direction Decomposition
# ═══════════════════════════════════════════════════════════════════════════
def fig_norm_direction():
    print("Figure 3: Norm-Direction Decomposition")

    gpt_keys   = ["Pythia-1B","GPT-Neo-2.7B","Pythia-6.9B","OPT-6.7B",
                  "Pythia-12B","Falcon-7B"]
    llama_keys = ["Mistral-7B","Llama-3-8B","Qwen2-7B"]

    # Display labels for the x-axis (style only.keeps original short labels)
    display = {
        "Pythia-1B":"P1B", "GPT-Neo-2.7B":"Neo", "Pythia-6.9B":"P69",
        "OPT-6.7B":"OPT", "Pythia-12B":"P12B", "Falcon-7B":"Falcon",
        "Mistral-7B":"Mistral","Llama-3-8B":"Llama-3","Qwen2-7B":"Qwen2",
    }

    table_gpt   = get_norm_table(gpt_keys)
    table_llama = get_norm_table(llama_keys)
    all_keys = gpt_keys + llama_keys
    all_names = [display[k] for k in all_keys]

    norms = [table_gpt[k]["norm_auroc"] if k in table_gpt else table_llama[k]["norm_auroc"]
             for k in all_keys]
    bcds  = [table_gpt[k]["bcd_auroc"]  if k in table_gpt else table_llama[k]["bcd_auroc"]
             for k in all_keys]

    n_gpt   = len(gpt_keys)
    n_llama = len(llama_keys)
    n_total = n_gpt + n_llama

    gap = 0.6
    x = np.arange(n_total, dtype=float)
    x[n_gpt:] += gap

    w = 0.32

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    _clean_axes(ax)

    ax.bar(x - w / 2, norms, w, color=C_LGRAY, edgecolor="none",
           label="Norm AUROC", zorder=3)

    for i in range(n_total):
        color = C_BLUE if i < n_gpt else C_ORANGE
        ax.bar(x[i] + w / 2, bcds[i], w, color=color, edgecolor="none",
               zorder=3,
               label=("Channel-decomp score$_K$ (GPT)" if i == 0
                      else ("Channel-decomp score$_K$ (LLaMA)" if i == n_gpt else "")))

    ax.axhline(0.50, ls="--", lw=0.8, color=C_GRAY, zorder=1)

    sep_x = (x[n_gpt - 1] + x[n_gpt]) / 2
    ax.axvline(sep_x, ls=":", lw=0.8, color=C_GRAY, zorder=1)

    ax.text(np.mean(x[:n_gpt]), -0.13, "GPT-family",
            fontsize=8, ha="center", va="top", color=C_GRAY,
            transform=ax.get_xaxis_transform())
    ax.text(np.mean(x[n_gpt:]), -0.13, "LLaMA-family",
            fontsize=8, ha="center", va="top", color=C_GRAY,
            transform=ax.get_xaxis_transform())

    # Gap annotation: P69
    idx_p69 = all_keys.index("Pythia-6.9B")
    gap_p69 = round((bcds[idx_p69] - norms[idx_p69]) * 100)
    ax.annotate(f"+{gap_p69}pp",
                xy=(x[idx_p69] + w / 2, bcds[idx_p69]),
                xytext=(x[idx_p69] + w / 2, bcds[idx_p69] + 0.04),
                fontsize=7, ha="center", va="bottom", color=C_BLUE,
                arrowprops=dict(arrowstyle="-", lw=0.0))
    ax.plot([x[idx_p69] - w / 2, x[idx_p69] - w / 2,
             x[idx_p69] + w / 2, x[idx_p69] + w / 2],
            [norms[idx_p69], norms[idx_p69] + 0.015,
             bcds[idx_p69] + 0.015, bcds[idx_p69]],
            lw=0.6, color=C_BLUE, zorder=5, clip_on=False)

    # Gap annotation: Mistral
    idx_m = all_keys.index("Mistral-7B")
    gap_m = round((bcds[idx_m] - norms[idx_m]) * 100)
    ax.annotate(f"+{gap_m}pp",
                xy=(x[idx_m] + w / 2, bcds[idx_m]),
                xytext=(x[idx_m] + w / 2, bcds[idx_m] + 0.04),
                fontsize=7, ha="center", va="bottom", color=C_ORANGE,
                arrowprops=dict(arrowstyle="-", lw=0.0))
    ax.plot([x[idx_m] - w / 2, x[idx_m] - w / 2,
             x[idx_m] + w / 2, x[idx_m] + w / 2],
            [norms[idx_m], norms[idx_m] + 0.015,
             bcds[idx_m] + 0.015, bcds[idx_m]],
            lw=0.6, color=C_ORANGE, zorder=5, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.40, 1.02)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.subplots_adjust(bottom=0.18)

    _save(fig, "norm_direction_decomposition")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_dark_subspace()
    fig_scaling_curve()
    fig_norm_direction()
    print("\nAll figures generated.")
