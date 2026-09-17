"""Unsupervised anomaly detection: the safety net for faults never labelled.

The supervised classifier can only recognise the four fault modes present in
the training data. A real plant also suffers novel faults -- a new tool
supplier, a degraded coolant pump, a miscalibrated sensor -- for which no
labelled examples exist. An Isolation Forest fitted on **healthy operation
only** gives a second, label-free opinion: "this machine is operating unlike
anything in its own normal history."

Fitting on healthy rows only is what makes this a novelty detector rather than
a repackaged classifier. Labels are used strictly to *evaluate* it afterwards,
never to fit it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import features as feat
from . import viz


def build_anomaly_detector(contamination: float | None = None) -> Pipeline:
    """Isolation Forest over the scaled physical feature space."""
    contamination = (cfg.ANOMALY_CONTAMINATION if contamination is None
                     else contamination)
    return Pipeline([
        ("prep", feat.build_preprocessor(scale_numeric=True)),
        ("iso", IsolationForest(
            n_estimators=400,
            max_samples="auto",
            contamination=contamination,
            bootstrap=False,
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        )),
    ])


def fit_on_healthy(detector: Pipeline, X_train: pd.DataFrame, y_train) -> Pipeline:
    """Fit using only the rows labelled healthy."""
    y_train = np.asarray(y_train).astype(int)
    detector.fit(X_train.loc[y_train == 0])
    return detector


def anomaly_scores(detector: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher score = more anomalous.

    ``IsolationForest.score_samples`` returns higher values for *more normal*
    points, so the sign is flipped to keep the intuitive direction everywhere
    else in the project.
    """
    return -detector.named_steps["iso"].score_samples(
        detector.named_steps["prep"].transform(X)
    )


def evaluate_detector(scores: np.ndarray, y_true, top_k_fractions=(0.01, 0.034, 0.05, 0.10)) -> dict:
    """How much of the labelled failure population the detector surfaces.

    ``lift`` answers the operational question directly: if maintenance
    inspects only the k% most anomalous machines, how many times more failures
    do they find than by inspecting k% at random?
    """
    y_true = np.asarray(y_true).astype(int)
    base_rate = float(y_true.mean())
    out = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "base_failure_rate": base_rate,
        "top_k": [],
    }
    order = np.argsort(-scores)
    for frac in top_k_fractions:
        k = max(int(round(frac * len(scores))), 1)
        flagged = y_true[order[:k]]
        captured = float(flagged.sum() / max(y_true.sum(), 1))
        precision = float(flagged.mean())
        out["top_k"].append({
            "fraction_inspected": frac,
            "machines_inspected": k,
            "failures_found": int(flagged.sum()),
            "recall": captured,
            "precision": precision,
            "lift_vs_random": float(precision / base_rate) if base_rate else 0.0,
        })
    return out


def plot_score_distribution(scores: np.ndarray, y_true,
                            title="Anomaly score: healthy vs failed", ax=None):
    """Two labelled, directly-annotated distributions on one axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 4.2))
    y_true = np.asarray(y_true).astype(int)
    bins = np.linspace(scores.min(), scores.max(), 60)

    ax.hist(scores[y_true == 0], bins=bins, density=True, color=viz.C_NORMAL,
            alpha=0.85, label=f"Healthy (n={int((y_true == 0).sum()):,})")
    ax.hist(scores[y_true == 1], bins=bins, density=True, color=viz.C_FAILURE,
            alpha=0.85, label=f"Failed (n={int(y_true.sum()):,})")
    viz.finish(ax, title=title,
               subtitle="detector fitted on healthy rows only; labels used for evaluation",
               xlabel="Anomaly score (higher = more unusual)",
               ylabel="Density", legend=True)
    return ax


def plot_top_k_lift(evaluation: dict, title="Inspection efficiency", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.2))
    rows = evaluation["top_k"]
    labels = [f"{r['fraction_inspected']*100:.1f}%" for r in rows]
    lifts = [r["lift_vs_random"] for r in rows]
    xs = np.arange(len(rows))

    ax.bar(xs, lifts, width=0.56, color=viz.SERIES[0])
    ax.axhline(1.0, color=viz.INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(len(rows) - 0.5, 1.06, "random inspection", ha="right",
            color=viz.INK_MUTED, fontsize=8.5)
    for x, r in zip(xs, rows):
        ax.text(x, r["lift_vs_random"] + max(lifts) * 0.03,
                f"{r['lift_vs_random']:.1f}x\n{r['failures_found']} found",
                ha="center", va="bottom", fontsize=9, color=viz.INK_SECONDARY)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, max(lifts) * 1.28)
    viz.finish(ax, title=title,
               subtitle="failures found per machine inspected, vs inspecting at random",
               xlabel="Share of fleet inspected (most anomalous first)",
               ylabel="Lift over random")
    return ax


__all__ = [
    "build_anomaly_detector", "fit_on_healthy", "anomaly_scores",
    "evaluate_detector", "plot_score_distribution", "plot_top_k_lift",
]
