"""Evaluation for a rare-event problem: metrics, cost-driven thresholds, plots.

Accuracy is deliberately never the headline. With a 3.4% failure rate, a model
that predicts "healthy" for every machine scores 96.6% accuracy and prevents
zero breakdowns. The headline metric here is **PR-AUC** (average precision),
which is sensitive to performance on the rare class, supported by recall at a
fixed precision and by the expected maintenance cost at the chosen operating
point.
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config as cfg
from . import viz


# --------------------------------------------------------------------------
# Scalar metrics
# --------------------------------------------------------------------------
def expected_cost(y_true, y_pred, c_fn: float | None = None,
                  c_fp: float | None = None) -> float:
    """Total maintenance cost of a set of decisions, in inspection-units.

    One unit = the cost of one unnecessary inspection. A missed failure costs
    ``c_fn`` units. True positives and true negatives are treated as free: a
    caught failure still needs the same inspection that a false alarm does, and
    that cost is common to both so it cancels out of the comparison.
    """
    c_fn = cfg.COST_FALSE_NEGATIVE if c_fn is None else c_fn
    c_fp = cfg.COST_FALSE_POSITIVE if c_fp is None else c_fp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(c_fn * fn + c_fp * fp)


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    b2 = beta ** 2
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


def classification_report_at(y_true, y_prob, threshold: float,
                             c_fn: float | None = None,
                             c_fp: float | None = None) -> dict:
    """Full metric bundle for one operating point."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    n_failures = int(y_true.sum())
    baseline_cost = (cfg.COST_FALSE_NEGATIVE if c_fn is None else c_fn) * n_failures
    model_cost = expected_cost(y_true, y_pred, c_fn, c_fp)

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": fbeta(precision, recall, beta=1.0),
        "f2": fbeta(precision, recall, beta=2.0),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost_units": model_cost,
        "cost_do_nothing": float(baseline_cost),
        "cost_saved_vs_do_nothing": float(baseline_cost - model_cost),
        "cost_saved_pct": float(
            100.0 * (baseline_cost - model_cost) / baseline_cost
        ) if baseline_cost else 0.0,
    }


def recall_at_precision(y_true, y_prob, min_precision: float = 0.90) -> dict:
    """Best achievable recall subject to a precision floor.

    This is the metric a maintenance planner actually cares about: "if I agree
    to act on at most one false alarm in ten, how many breakdowns do I catch?"
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have one more element than thresholds.
    ok = precision[:-1] >= min_precision
    if not ok.any():
        return {"min_precision": min_precision, "recall": 0.0, "threshold": None,
                "achievable": False}
    idx = int(np.argmax(np.where(ok, recall[:-1], -1)))
    return {
        "min_precision": float(min_precision),
        "recall": float(recall[idx]),
        "precision": float(precision[idx]),
        "threshold": float(thresholds[idx]),
        "achievable": True,
    }


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------
def tune_threshold_by_cost(y_true, y_prob, c_fn: float | None = None,
                           c_fp: float | None = None,
                           grid: Sequence[float] | None = None) -> dict:
    """Choose the decision threshold that minimises expected maintenance cost.

    Selection is always performed on out-of-fold predictions from the training
    data, never on the held-out test set -- the threshold is a fitted parameter
    like any other.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    grid = np.linspace(0.001, 0.999, 999) if grid is None else np.asarray(grid)

    costs = np.array([
        expected_cost(y_true, (y_prob >= t).astype(int), c_fn, c_fp) for t in grid
    ])
    best_i = int(np.argmin(costs))
    return {
        "threshold": float(grid[best_i]),
        "cost_units": float(costs[best_i]),
        "grid": grid.tolist(),
        "costs": costs.tolist(),
        "c_fn": float(cfg.COST_FALSE_NEGATIVE if c_fn is None else c_fn),
        "c_fp": float(cfg.COST_FALSE_POSITIVE if c_fp is None else c_fp),
    }


def threshold_cost_sweep(y_true, y_prob,
                         ratios: Iterable[float] | None = None) -> pd.DataFrame:
    """How the operating point moves as the cost assumption changes.

    The chosen 10:1 ratio is a business input rather than a fact about the data,
    so its influence is made explicit instead of buried.
    """
    ratios = cfg.COST_RATIO_SWEEP if ratios is None else ratios
    rows = []
    for r in ratios:
        tuned = tune_threshold_by_cost(y_true, y_prob, c_fn=r, c_fp=1.0)
        rep = classification_report_at(y_true, y_prob, tuned["threshold"],
                                       c_fn=r, c_fp=1.0)
        rows.append({
            "cost_ratio_FN_to_FP": f"{int(r)}:1",
            "threshold": round(rep["threshold"], 4),
            "precision": round(rep["precision"], 4),
            "recall": round(rep["recall"], 4),
            "f2": round(rep["f2"], 4),
            "false_alarms": rep["fp"],
            "missed_failures": rep["fn"],
            "cost_units": round(rep["cost_units"], 1),
            "cost_saved_pct": round(rep["cost_saved_pct"], 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_pr_curves(results: dict, y_true, title="Precision-recall by model",
                   ax=None):
    """PR curves for several models on one axes, with a baseline reference."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.6))
    base = float(np.mean(np.asarray(y_true).astype(int)))

    for i, (name, y_prob) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=viz.SERIES[i % len(viz.SERIES)],
                label=f"{name} (PR-AUC {ap:.3f})")

    ax.axhline(base, color=viz.INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(0.01, base + 0.02, f"no-skill baseline = failure rate {base:.3f}",
            color=viz.INK_MUTED, fontsize=8.5, va="bottom")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    viz.finish(ax, title=title, xlabel="Recall (share of failures caught)",
               ylabel="Precision (share of alarms that were real)", legend=True)
    return ax


def plot_roc_curves(results: dict, y_true, title="ROC by model", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, (name, y_prob) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=viz.SERIES[i % len(viz.SERIES)],
                label=f"{name} (ROC-AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, linestyle=(0, (4, 3)),
            linewidth=1.2, label="chance")
    viz.finish(ax, title=title, xlabel="False positive rate",
               ylabel="True positive rate", legend=True)
    return ax


def plot_confusion(y_true, y_pred, labels=("Healthy", "Failure"),
                   title="Confusion matrix", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4.0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax.imshow(cm, cmap=viz.CMAP_SEQUENTIAL, aspect="equal")
    ax.grid(False)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=13, fontweight="semibold",
                    color="#ffffff" if val > 0.55 * vmax else viz.INK_PRIMARY)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    viz.finish(ax, title=title, xlabel="Predicted", ylabel="Actual")
    return ax


def plot_calibration(y_true, y_prob, n_bins=10,
                     title="Calibration of failure probability", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.4))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                             strategy="quantile")
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, linestyle=(0, (4, 3)),
            linewidth=1.2, label="perfectly calibrated")
    ax.plot(prob_pred, prob_true, color=viz.SERIES[0], marker="o",
            label="model")
    viz.finish(ax, title=title, xlabel="Predicted probability",
               ylabel="Observed failure rate", legend=True)
    return ax


def plot_threshold_cost(tuned: dict, title="Maintenance cost vs decision threshold",
                        ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))
    grid = np.asarray(tuned["grid"])
    costs = np.asarray(tuned["costs"])
    ax.plot(grid, costs, color=viz.SERIES[0], label="expected cost")
    ax.axvline(tuned["threshold"], color=viz.STATUS["critical"], linewidth=1.6,
               linestyle=(0, (3, 2)))
    ax.annotate(
        f"chosen threshold {tuned['threshold']:.3f}\ncost {tuned['cost_units']:.0f} units",
        xy=(tuned["threshold"], tuned["cost_units"]),
        xytext=(min(tuned["threshold"] + 0.12, 0.62), costs.max() * 0.62),
        color=viz.INK_SECONDARY, fontsize=9,
        arrowprops=dict(arrowstyle="-", color=viz.BASELINE, linewidth=1),
    )
    viz.finish(ax, title=title,
               subtitle=f"cost of a missed failure = {tuned['c_fn']:.0f}x an unnecessary inspection",
               xlabel="Decision threshold", ylabel="Cost (inspection-units)")
    return ax


def save_json(obj, path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        raise TypeError(f"not JSON serialisable: {type(o)}")

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)


__all__ = [
    "expected_cost", "fbeta", "classification_report_at", "recall_at_precision",
    "tune_threshold_by_cost", "threshold_cost_sweep", "plot_pr_curves",
    "plot_roc_curves", "plot_confusion", "plot_calibration",
    "plot_threshold_cost", "save_json",
]
