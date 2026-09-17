"""Exploratory analysis figures, shared by notebook 02 and the report build.

Every function returns its axes so a notebook can compose or annotate further.
Each figure is built to answer one stated question -- the question is the title.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import viz


# --------------------------------------------------------------------------
# Class balance
# --------------------------------------------------------------------------
def plot_class_balance(df: pd.DataFrame, ax=None):
    """The headline problem: failures are 3.4% of the fleet."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.4))
    counts = df["y_failure"].value_counts().reindex([0, 1]).fillna(0)
    labels = ["Healthy", "Failure"]
    colors = [viz.C_NORMAL, viz.C_FAILURE]

    bars = ax.bar([0, 1], counts.to_numpy(), width=0.5, color=colors)
    total = counts.sum()
    for x, (b, c) in enumerate(zip(bars, counts.to_numpy())):
        ax.text(x, c + total * 0.015, f"{int(c):,}\n{c/total:.1%}",
                ha="center", va="bottom", fontsize=10,
                color=viz.INK_SECONDARY)
    ax.set_xticks([0, 1], labels)
    ax.set_ylim(0, total * 1.12)
    viz.finish(ax, title="Failures are rare: 3.4% of recorded cycles",
               subtitle="a model predicting 'healthy' always would score 96.6% accuracy",
               xlabel="", ylabel="Cycles")
    return ax


def plot_mode_counts(df: pd.DataFrame, ax=None):
    """How the 339 failures break down by recorded mode."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 3.8))
    counts = {c: int(df[c].sum()) for c in cfg.MODE_COLS}
    order = sorted(counts, key=lambda k: -counts[k])
    names = [cfg.MODE_LABELS[c] for c in order]
    vals = [counts[c] for c in order]
    # RNF is shown in a status colour with its own label because it is the one
    # mode the models cannot learn -- it is a caveat, not a series.
    colors = [viz.STATUS["warning"] if c == "RNF" else viz.SERIES[0]
              for c in order]

    ys = np.arange(len(order))[::-1]
    ax.barh(ys, vals, height=0.6, color=colors)
    for y, v, c in zip(ys, vals, order):
        note = "  (no sensor signature - excluded from fault classifier)" if c == "RNF" else ""
        ax.text(v + max(vals) * 0.015, y, f"{v}{note}", va="center",
                fontsize=9, color=viz.INK_SECONDARY)
    ax.set_yticks(ys, names)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.62)
    viz.finish(ax, title="Recorded failure modes",
               subtitle="modes can co-occur, so counts exceed the 339 distinct failures",
               xlabel="Occurrences", ylabel="")
    return ax


# --------------------------------------------------------------------------
# Sensor behaviour
# --------------------------------------------------------------------------
def plot_sensor_distributions(df: pd.DataFrame, cols=None, axes=None):
    """Healthy vs failed distribution for each sensor and derived quantity."""
    import matplotlib.pyplot as plt

    cols = cols or (cfg.SENSOR_COLS + cfg.ENGINEERED_COLS)
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.0 * nrows))
        axes = np.atleast_1d(axes).ravel()
    else:
        fig = axes[0].figure

    healthy = df.loc[df["y_failure"] == 0]
    failed = df.loc[df["y_failure"] == 1]

    for i, col in enumerate(cols):
        ax = axes[i]
        lo = float(df[col].min())
        hi = float(df[col].max())
        bins = np.linspace(lo, hi, 45)
        ax.hist(healthy[col], bins=bins, density=True, color=viz.C_NORMAL,
                alpha=0.85, label="Healthy")
        ax.hist(failed[col], bins=bins, density=True, color=viz.C_FAILURE,
                alpha=0.85, label="Failure")
        viz.finish(ax, title=col, xlabel="", ylabel="Density")
        if i == 0:
            ax.legend(loc="upper right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Where failures live in each measured quantity",
                 x=0.005, ha="left", fontsize=13, fontweight="semibold",
                 color=viz.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes


def plot_correlation(df: pd.DataFrame, ax=None):
    """Correlation structure -- diverging scale, grey at zero."""
    import matplotlib.pyplot as plt

    cols = cfg.SENSOR_COLS + cfg.ENGINEERED_COLS + ["y_failure"]
    corr = df[cols].corr(numeric_only=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(corr.to_numpy(), cmap=viz.CMAP_DIVERGING, vmin=-1, vmax=1)
    ax.grid(False)
    short = [c.replace(" [", "\n[").replace("_", " ") for c in corr.columns]
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(short)), short, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(v) > 0.55 else viz.INK_SECONDARY)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("Pearson r", color=viz.INK_SECONDARY, fontsize=9)
    viz.finish(ax, title="Correlation between measured and derived quantities",
               subtitle="torque and speed are strongly inversely related - a constant-power spindle",
               xlabel="", ylabel="")
    return ax


def plot_operating_envelope(df: pd.DataFrame, ax=None):
    """Torque against speed, with failures overplotted -- the money figure.

    Two series only, so the all-pairs colour gate is satisfied, and both are
    named in the legend as well as separated by marker size.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 5.0))
    healthy = df.loc[df["y_failure"] == 0]
    failed = df.loc[df["y_failure"] == 1]

    ax.scatter(healthy[cfg.COL_SPEED], healthy[cfg.COL_TORQUE], s=7,
               color=viz.C_NORMAL, alpha=0.35, linewidths=0, label="Healthy")
    ax.scatter(failed[cfg.COL_SPEED], failed[cfg.COL_TORQUE], s=26,
               color=viz.C_FAILURE, alpha=0.95, linewidths=0.8,
               edgecolors=viz.SURFACE, label="Failure")
    ax.grid(axis="both")
    viz.finish(ax, title="The operating envelope, and where it breaks",
               subtitle="failures cluster at the low-speed/high-torque and high-speed/low-torque corners",
               xlabel=cfg.COL_SPEED, ylabel=cfg.COL_TORQUE, legend=True)
    return ax


def plot_wear_vs_failure(df: pd.DataFrame, ax=None):
    """Failure rate against tool wear, binned -- the degradation story."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.2))
    bins = np.arange(0, 270, 20)
    grouped = df.groupby(pd.cut(df[cfg.COL_WEAR], bins, right=False),
                         observed=True)
    rate = grouped["y_failure"].mean() * 100
    n = grouped["y_failure"].size()
    centres = [iv.left + 10 for iv in rate.index]

    ax.bar(centres, rate.to_numpy(), width=17, color=viz.SERIES[0])
    for x, r, c in zip(centres, rate.to_numpy(), n.to_numpy()):
        if r > 0:
            ax.text(x, r + rate.max() * 0.03, f"{r:.1f}%", ha="center",
                    va="bottom", fontsize=8.5, color=viz.INK_SECONDARY)
    ax.set_ylim(0, rate.max() * 1.22)
    viz.finish(ax, title="Failure rate rises sharply past 200 minutes of tool wear",
               subtitle="bar height is the observed failure rate within each 20-minute wear band",
               xlabel="Tool wear (min)", ylabel="Failure rate (%)")
    return ax


def plot_type_breakdown(df: pd.DataFrame, ax=None):
    """Failure rate by machine quality variant, with counts labelled."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.8))
    g = df.groupby(cfg.COL_TYPE, observed=True)["y_failure"]
    rate = (g.mean() * 100).reindex(cfg.TYPE_ORDER)
    n = g.size().reindex(cfg.TYPE_ORDER)
    fails = g.sum().reindex(cfg.TYPE_ORDER)

    xs = np.arange(len(cfg.TYPE_ORDER))
    ax.bar(xs, rate.to_numpy(), width=0.52, color=viz.SERIES[0])
    for x, r, nn, ff in zip(xs, rate.to_numpy(), n.to_numpy(), fails.to_numpy()):
        ax.text(x, r + rate.max() * 0.04, f"{r:.2f}%\n{int(ff)} of {int(nn):,}",
                ha="center", va="bottom", fontsize=9, color=viz.INK_SECONDARY)
    ax.set_xticks(xs, [f"Type {t}" for t in cfg.TYPE_ORDER])
    ax.set_ylim(0, rate.max() * 1.34)
    viz.finish(ax, title="Failure rate by machine quality variant",
               subtitle="L (low) through H (high) quality; L dominates the fleet at 60%",
               xlabel="", ylabel="Failure rate (%)")
    return ax


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Healthy vs failed means for every quantity, with the relative shift."""
    cols = cfg.SENSOR_COLS + cfg.ENGINEERED_COLS
    healthy = df.loc[df["y_failure"] == 0, cols].mean()
    failed = df.loc[df["y_failure"] == 1, cols].mean()
    out = pd.DataFrame({
        "quantity": cols,
        "healthy_mean": healthy.to_numpy(),
        "failed_mean": failed.to_numpy(),
    })
    out["shift_%"] = 100 * (out["failed_mean"] - out["healthy_mean"]) / out["healthy_mean"].abs()
    return out.sort_values("shift_%", key=abs, ascending=False).reset_index(drop=True)


__all__ = [
    "plot_class_balance", "plot_mode_counts", "plot_sensor_distributions",
    "plot_correlation", "plot_operating_envelope", "plot_wear_vs_failure",
    "plot_type_breakdown", "summary_table",
]
