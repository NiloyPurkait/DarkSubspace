#!/usr/bin/env python3
"""figure_style.py.

Shared Matplotlib style tokens (font, palette, axis style) used by every paper
figure script.

Used in all main-body and appendix figures of the paper.

Reproduce::

    # imported by plotting scripts:
    from figure_style import apply_style, MODEL_COLORS, LINEWIDTH
    apply_style()

Double-column format dimensions::

    linewidth   approx. 7.0 in (figure* environments)
    columnwidth approx. 3.35 in (figure environments)

All figures use serif fonts (STIXGeneral / Times) to match the LaTeX body.
Colour palette is Okabe Ito (colourblind safe).
"""

import matplotlib
import matplotlib.pyplot as plt

# Dimensions
LINEWIDTH  = 7.0    # inches, full-width figure (figure*)
COLUMNWIDTH = 3.35  # inches, single-column figure

# Okabe Ito palette (colourblind safe)
OI_VERMILION = "#D55E00"
OI_BLUE      = "#0072B2"
OI_AMBER     = "#E69F00"
OI_SKY       = "#56B4E9"
OI_GREEN     = "#009E73"
OI_PINK      = "#CC79A7"
OI_GREY      = "#999999"
OI_BLACK     = "#000000"

# Semantic aliases
C_MEMBER     = OI_VERMILION
C_NONMEMBER  = OI_BLUE
C_PROMOTE    = OI_GREEN
C_SUPPRESS   = OI_PINK
C_NEUTRAL    = "#DDDDDD"
C_BG         = "#F7F7F7"

# Model palette (4 models)
MODEL_COLORS = {
    "pythia-1b":    OI_BLUE,
    "pythia-6.9b":  OI_VERMILION,
    "gpt-neo-2.7B": OI_GREEN,
    "opt-6.7b":     OI_AMBER,
}
MODEL_MARKERS = {
    "pythia-1b":    "o",
    "pythia-6.9b":  "s",
    "gpt-neo-2.7B": "D",
    "opt-6.7b":     "^",
}
MODEL_LABELS = {
    "pythia-1b":    "Pythia-1B",
    "pythia-6.9b":  "Pythia-6.9B",
    "gpt-neo-2.7B": "GPT-Neo-2.7B",
    "opt-6.7b":     "OPT-6.7B",
}


def apply_style():
    """Apply global rcParams. Call once at script start."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        # Font, serif to match the LaTeX body
        "font.family":       "serif",
        "font.serif":        ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":  "stix",
        # Base sizes (set for 7.0-in figure, scaled ~1.15x)
        "font.size":         9,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   7.5,
        # Axes
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.6,
        "axes.grid":         False,
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        # Output
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.04,
        # Ticks
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        # No LaTeX (faster, more portable)
        "text.usetex":       False,
    })


def add_panel_labels(axes, x=-0.12, y=1.06, fontsize=10.5, **kwargs):
    """Add bold (a), (b), ... labels to a list of axes."""
    for ax, letter in zip(axes, "abcdefghij"):
        ax.text(x, y, f"({letter})", transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="top", **kwargs)
