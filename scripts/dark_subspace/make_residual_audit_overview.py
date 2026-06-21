#!/usr/bin/env python3
"""Figure 2 for the ICML camera-ready: the SAE residual audit.

A sober, single-column concept figure in the ICML/PMLR house style: a serif
(STIX, Times-metric) typeface to match the body text, soft rounded boxes with
thin rules, a muted greyscale-plus-one-accent palette, and no drop shadows or
poster styling. It shows the residual audit on one activation and the membership
AUROC ordering it produces.

  Fine-tuned activation h
    -> SAE encode / decode
         -> reconstruction  h_hat
         -> residual  r = h - h_hat
  a single score_K membership detector reads all three views (the original h
  travels down the left rail, the reconstruction and residual from the split)
    -> three AUROC bars: Original 0.803, SAE-Recon 0.594, Residual 0.781

The three AUROC values are the exact tab:dark_subspace numbers for the
Pythia-6.9B (mixed-data) row, read from manuscript/results.tex and corroborated
by the fig:score_distributions caption. Nothing here is fabricated.

In this cell the residual (0.781) sits just below the original (0.803), so the
figure states the honest universal claim (the residual stays above the
reconstruction), not residual >= original.

Run:  .venv/bin/python scripts/dark_subspace/make_residual_audit_overview.py
Writes to assets/figures/ by default (set FIGDIR to redirect):
  residual_audit_overview.pdf
  residual_audit_overview.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- style ----
# Match the paper: a Times-metric serif and CM-style math. Greyscale plus one
# restrained accent for the residual. No shadows, soft rounded boxes.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

EDGE = "#3a3a3a"      # box and axis rules
INK = "#1a1a1a"       # text
ORIG = "#555555"      # original bar (dark grey)
RECON = "#a8a8a8"     # reconstruction bar (mid grey)
ACCENT = "#35608c"    # residual accent (muted slate blue)
ACC_FILL = "#eaf0f6"  # faint accent wash for the residual box
GREY_FILL = "#f4f4f4"  # faint neutral fill

# the only numerals in the figure, read from manuscript/results.tex
# (Pythia-6.9B mixed-data row of tab:dark_subspace) -- verified on disk.
AUROC_ORIGINAL = 0.803
AUROC_RECON = 0.594
AUROC_RESIDUAL = 0.781


def build():
    # aspect close to 1 so the rounded corners read as circular, not elliptical
    fig = plt.figure(figsize=(3.35, 3.15), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(cx, cy, w, h, fc="white", ec=EDGE, lw=1.0, round_size=0.022):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=f"round,pad=0,rounding_size={round_size}",
            mutation_aspect=1.06,
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))

    def arrow(x0, y0, x1, y1, color=EDGE, lw=1.0):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
            color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=3))

    def text(x, y, s, fs=8.2, color=INK, weight="normal", ha="center", va="center"):
        ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
                fontweight=weight, zorder=5)

    BH = 0.085  # standard box height

    # ---- row 1: fine-tuned activation ----
    box(0.52, 0.935, 0.46, BH)
    text(0.52, 0.935, r"Fine-tuned activation $\mathbf{h}$", fs=8.4)

    # ---- row 2: SAE ----
    arrow(0.52, 0.892, 0.52, 0.847)
    box(0.52, 0.805, 0.46, BH)
    text(0.52, 0.805, "SAE encode and decode", fs=8.4)

    # ---- row 3: the two views from the SAE split (single-line labels) ----
    yv = 0.635
    box(0.325, yv, 0.40, BH)
    text(0.325, yv, r"Reconstruction $\hat{\mathbf{h}}$", fs=8.2)
    box(0.745, yv, 0.40, BH, fc=ACC_FILL, ec=ACCENT, lw=1.1)
    text(0.745, yv, r"Residual $\mathbf{r}{=}\mathbf{h}{-}\hat{\mathbf{h}}$",
         fs=8.0, color=ACCENT)
    arrow(0.46, 0.762, 0.37, yv + 0.043)
    arrow(0.58, 0.762, 0.70, yv + 0.043, color=ACCENT)

    # ---- row 4: a single detector reads all three views ----
    yd = 0.485
    box(0.52, yd, 0.92, 0.066, fc=GREY_FILL, round_size=0.018)
    text(0.52, yd, r"single $\mathrm{score}_K$ membership detector", fs=8.0)
    arrow(0.325, yv - 0.043, 0.325, yd + 0.033)
    arrow(0.745, yv - 0.043, 0.745, yd + 0.033, color=ACCENT)

    # ---- the original activation h also reaches the detector via a left rail ----
    rx = 0.05
    ax.add_patch(FancyArrowPatch(
        (0.29, 0.935), (rx, 0.935), arrowstyle="-", color=ORIG, lw=1.0,
        shrinkA=0, shrinkB=0, zorder=1))
    ax.add_patch(FancyArrowPatch(
        (rx, 0.935), (rx, yd), arrowstyle="-", color=ORIG, lw=1.0,
        shrinkA=0, shrinkB=0, zorder=1))
    arrow(rx, yd, 0.072, yd, color=ORIG)
    text(rx + 0.026, 0.715, r"original $\mathbf{h}$", fs=7.6, color=ORIG, ha="left")

    # ---- caption line for the bar panel ----
    text(0.52, 0.405, "membership AUROC (Pythia-6.9B, mixed-data SAE)",
         fs=7.2, color="#666666")

    # ---- row 5: AUROC bars on a [0.50, 0.85] axis ----
    bars = [("Original", AUROC_ORIGINAL, ORIG),
            ("SAE-Recon", AUROC_RECON, RECON),
            ("Residual", AUROC_RESIDUAL, ACCENT)]
    A0, A1 = 0.50, 0.85
    base_y, top_y = 0.075, 0.355
    cxs = [0.24, 0.52, 0.80]
    bw = 0.155

    def bar_h(a):
        return (a - A0) / (A1 - A0) * (top_y - base_y)

    ax.plot([0.07, 0.93], [base_y, base_y], color=EDGE, lw=1.0, zorder=2)
    for (name, a, col), cx in zip(bars, cxs):
        h = bar_h(a)
        ax.add_patch(Rectangle((cx - bw / 2, base_y), bw, h,
                               facecolor=col, edgecolor="none", zorder=2))
        text(cx, base_y + h + 0.028, f"{a:.3f}", fs=8.6, weight="bold", color=col)
        text(cx, base_y - 0.032, name, fs=7.8)
    return fig


def main():
    fig = build()
    repo_root = os.path.dirname(os.path.dirname(HERE))
    outdir = os.environ.get("FIGDIR", os.path.join(repo_root, "assets", "figures"))
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, "residual_audit_overview.pdf")
    png = os.path.join(outdir, "residual_audit_overview.png")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)


if __name__ == "__main__":
    main()
