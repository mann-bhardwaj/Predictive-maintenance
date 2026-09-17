"""SHAP explainability -- answering "why did the system flag this machine?"

Two levels of explanation, and they answer different questions:

* **Global** (mean |SHAP| across the fleet): which physical quantities drive
  failure risk overall. This is what goes in a report.
* **Local** (one row's SHAP values): why *this* machine, right now, crossed the
  alarm threshold. This is what a maintenance technician needs, and it is what
  the dashboard renders live.

A note on units, because getting this wrong is a common and visible error:
``TreeExplainer`` reports contributions in whatever space the model's raw output
lives in. For a random forest that is **probability** (the forest averages leaf
class frequencies); for gradient boosting it is **log-odds** (the additive
margin before the logistic link). The explainer detects which and labels every
figure accordingly, so a contribution of ``+0.20`` is never silently read as the
wrong quantity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config as cfg
from . import features as feat
from . import viz


class PipelineExplainer:
    """SHAP wrapper that keeps the sklearn pipeline and the explainer in sync.

    The preprocessor is applied first, then a ``TreeExplainer`` runs on the
    transformed matrix, so the feature names shown to the user are the
    post-encoding ones (including the one-hot machine-type columns).
    """

    def __init__(self, pipeline, background: pd.DataFrame | None = None):
        self.pipeline = pipeline
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["clf"]
        self.names = feat.feature_names(self.prep)
        self._background = background
        self.explainer = shap.TreeExplainer(self.model)
        self.units = self._detect_units()

    def _detect_units(self) -> str:
        """Which space the SHAP contributions are expressed in."""
        name = type(self.model).__name__
        if "XGB" in name or "GradientBoosting" in name or "LGBM" in name:
            return "log-odds"
        return "probability"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.prep.transform(X)
        return pd.DataFrame(arr, columns=self.names, index=X.index)

    def shap_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values for every row, aligned to readable feature names."""
        Xt = self.transform(X)
        vals = self.explainer.shap_values(Xt)
        vals = np.asarray(vals)
        # Binary tree models may return (n, f) or (n, f, 2) depending on version.
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return pd.DataFrame(vals, columns=self.names, index=X.index)

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        sv = self.shap_frame(X)
        imp = sv.abs().mean().sort_values(ascending=False)
        signed = sv.mean()
        return pd.DataFrame({
            "feature": imp.index,
            "mean_abs_shap": imp.to_numpy(),
            "mean_signed_shap": signed.reindex(imp.index).to_numpy(),
        }).reset_index(drop=True)

    def explain_row(self, X_row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
        """The top drivers for a single machine, largest contribution first."""
        sv = self.shap_frame(X_row).iloc[0]
        Xt = self.transform(X_row).iloc[0]
        frame = pd.DataFrame({
            "feature": sv.index,
            "value": Xt.reindex(sv.index).to_numpy(),
            "shap": sv.to_numpy(),
        })
        frame["abs_shap"] = frame["shap"].abs()
        frame = frame.sort_values("abs_shap", ascending=False).head(top_n)
        frame["direction"] = np.where(frame["shap"] > 0,
                                      "raises risk", "lowers risk")
        return frame.drop(columns="abs_shap").reset_index(drop=True)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_global_importance(importance: pd.DataFrame, top_n: int = 12,
                           title="What drives failure risk across the fleet",
                           units: str = "probability", ax=None):
    """Horizontal bars, single hue, directly labelled -- magnitude only."""
    import matplotlib.pyplot as plt

    top = importance.head(top_n).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 0.42 * len(top) + 1.4))

    ys = np.arange(len(top))
    ax.barh(ys, top["mean_abs_shap"], height=0.62, color=viz.SERIES[0])
    ax.set_yticks(ys, top["feature"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    span = top["mean_abs_shap"].max()
    for y, v in zip(ys, top["mean_abs_shap"]):
        ax.text(v + span * 0.015, y, f"{v:.3f}", va="center", fontsize=9,
                color=viz.INK_SECONDARY)
    ax.set_xlim(0, span * 1.16)
    viz.finish(ax, title=title,
               subtitle=f"mean |SHAP| in {units} units; magnitude of influence, not direction",
               xlabel=f"Mean |SHAP| ({units})", ylabel="")
    return ax


def plot_local_explanation(row_expl: pd.DataFrame, probability: float,
                           threshold: float,
                           title="Why this machine was flagged",
                           units: str = "probability", ax=None):
    """Signed contributions for one machine: diverging colour, always labelled."""
    import matplotlib.pyplot as plt

    top = row_expl.iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 0.46 * len(top) + 1.6))

    ys = np.arange(len(top))
    colors = [viz.STATUS["critical"] if s > 0 else viz.SERIES[0]
              for s in top["shap"]]
    ax.barh(ys, top["shap"], height=0.6, color=colors)
    ax.axvline(0, color=viz.BASELINE, linewidth=1.2)
    ax.set_yticks(ys, top["feature"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    span = max(top["shap"].abs().max(), 1e-6)
    for y, s in zip(ys, top["shap"]):
        off = span * 0.03 * (1 if s > 0 else -1)
        ax.text(s + off, y, f"{s:+.2f}", va="center",
                ha="left" if s > 0 else "right", fontsize=9,
                color=viz.INK_SECONDARY)
    ax.set_xlim(-span * 1.35, span * 1.35)

    verdict = "ALARM" if probability >= threshold else "within limits"
    viz.finish(
        ax, title=title,
        subtitle=(f"failure probability {probability:.1%} vs threshold "
                  f"{threshold:.1%} -- {verdict}. Red raises risk, blue lowers it."),
        xlabel=f"SHAP contribution ({units})", ylabel="",
    )
    return ax


def plot_beeswarm(explainer: PipelineExplainer, X: pd.DataFrame,
                  max_display: int = 10, title="SHAP value distribution"):
    """SHAP's own beeswarm, restyled to the project palette."""
    import matplotlib.pyplot as plt

    Xt = explainer.transform(X)
    sv = explainer.shap_frame(X)
    expl = shap.Explanation(values=sv.to_numpy(), data=Xt.to_numpy(),
                            feature_names=list(sv.columns))
    shap.plots.beeswarm(expl, max_display=max_display, show=False,
                        color=viz.CMAP_DIVERGING)
    fig = plt.gcf()
    fig.set_size_inches(7.4, 0.42 * max_display + 1.6)
    ax = plt.gca()
    ax.set_title(title, loc="left", color=viz.INK_PRIMARY, fontsize=12,
                 fontweight="semibold")
    ax.set_facecolor(viz.SURFACE)
    fig.set_facecolor(viz.SURFACE)
    return fig


def narrate(row_expl: pd.DataFrame, probability: float, threshold: float,
            mode_probs: dict | None = None) -> str:
    """Plain-language explanation -- the sentence a technician actually reads."""
    raisers = row_expl[row_expl["shap"] > 0].head(3)
    verdict = ("FLAGGED for inspection" if probability >= threshold
               else "no action required")
    lines = [
        f"Assessment: {verdict} "
        f"(failure probability {probability:.1%}, alarm threshold {threshold:.1%})."
    ]
    if len(raisers):
        drivers = ", ".join(
            f"{r.feature} (contribution {r.shap:+.2f})" for r in raisers.itertuples()
        )
        lines.append(f"Principal risk drivers: {drivers}.")
    if mode_probs:
        ranked = sorted(mode_probs.items(), key=lambda kv: -kv[1])
        top = ranked[0]
        lines.append(
            f"Most likely fault mode: {cfg.MODE_LABELS[top[0]]} "
            f"({top[1]:.1%} confidence)."
        )
        others = [f"{cfg.MODE_LABELS[k]} {v:.0%}" for k, v in ranked[1:] if v >= 0.20]
        if others:
            lines.append("Also indicated: " + ", ".join(others) + ".")
    return " ".join(lines)


__all__ = [
    "PipelineExplainer", "plot_global_importance", "plot_local_explanation",
    "plot_beeswarm", "narrate",
]
