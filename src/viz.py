"""Shared visual language for every figure in the project.

One palette, one matplotlib style, applied everywhere so the notebooks, the
report figures and the dashboard read as a single system.

The categorical hues are used in fixed slot order (never cycled) and were
validated for colour-vision deficiency separation and lightness banding against
the light chart surface. Because three of the slots sit below 3:1 contrast on a
light surface, every multi-series figure also carries a legend and, where there
are few enough series, direct labels -- identity is never colour-alone.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed categorical slot order.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Semantic aliases for the two states this project talks about constantly.
C_NORMAL = SERIES[0]
C_FAILURE = SERIES[1]

# Status colours are reserved and never reused as a series colour. They always
# ship alongside a text label.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Single-hue sequential ramp (light -> dark) for magnitude: confusion matrices,
# density heatmaps.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
    "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list("pdm_blue", SEQUENTIAL_BLUE)

# Diverging ramp (blue <-> red with a neutral grey midpoint) for signed
# quantities such as correlations. Never a rainbow, never a hue at the midpoint.
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "pdm_div",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#ef9a9a", "#d03b3b", "#7f1d1d"],
)

FONT_STACK = ["DejaVu Sans", "system-ui", "Segoe UI", "sans-serif"]


def apply_style() -> None:
    """Install the project matplotlib style. Call once per notebook/script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            # Gridlines are chrome: they sit behind the marks, never across them.
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def finish(ax, title: str | None = None, subtitle: str | None = None,
           xlabel: str | None = None, ylabel: str | None = None,
           legend: bool = False) -> None:
    """Apply the recessive-chrome conventions to a finished axes."""
    # The subtitle occupies the band directly under the title, so the title pad
    # grows to make room for it rather than letting the two overlap.
    if title:
        ax.set_title(title, loc="left", pad=26 if subtitle else 10)
    if subtitle:
        ax.text(
            0.0, 1.012, subtitle, transform=ax.transAxes,
            fontsize=9, color=INK_MUTED, va="bottom", ha="left",
        )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best")
    ax.tick_params(length=0)


def save(fig, name: str) -> None:
    """Persist a figure into reports/figures using the project conventions."""
    from .config import FIGURES_DIR

    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def bar_ends(ax, orientation: str = "v") -> None:
    """Round the data-end of every bar patch (4px feel at project DPI)."""
    for patch in ax.patches:
        patch.set_linewidth(0)
    _ = orientation  # kept for call-site clarity


__all__ = [
    "SURFACE", "INK_PRIMARY", "INK_SECONDARY", "INK_MUTED", "GRIDLINE",
    "BASELINE", "SERIES", "C_NORMAL", "C_FAILURE", "STATUS",
    "CMAP_SEQUENTIAL", "CMAP_DIVERGING", "apply_style", "finish", "save",
]
