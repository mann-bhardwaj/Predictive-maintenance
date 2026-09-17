"""Model definitions and cross-validated training.

Three baselines, in deliberate order of sophistication, so each one's
contribution is visible:

1. **Logistic regression** -- a linear, fully interpretable reference. Its
   coefficients say which direction each physical quantity pushes risk.
2. **Random forest** -- captures the non-monotonic structure (both too little
   and too much spindle power is dangerous) that a linear model cannot.
3. **XGBoost** -- gradient boosting, usually the strongest on tabular data of
   this size, and the model the dashboard ships by default if it wins.

Class imbalance is handled with class weights rather than synthetic
oversampling: no invented sensor readings to justify, and nothing that can leak
across a validation fold. The decision threshold is then tuned separately on
out-of-fold predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config as cfg
from . import features as feat


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
def build_models(pos_weight: float) -> dict[str, Pipeline]:
    """Pipelines keyed by display name.

    ``pos_weight`` is the negative/positive ratio in the training data, passed
    to XGBoost as ``scale_pos_weight``; the sklearn models use their own
    ``class_weight='balanced'`` machinery, which computes the same idea.
    """
    return {
        "Logistic regression": Pipeline([
            ("prep", feat.build_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=4000,
                C=1.0,
                solver="lbfgs",
                random_state=cfg.RANDOM_STATE,
            )),
        ]),
        "Random forest": Pipeline([
            ("prep", feat.build_preprocessor(scale_numeric=False)),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=cfg.RANDOM_STATE,
            )),
        ]),
        "XGBoost": Pipeline([
            ("prep", feat.build_preprocessor(scale_numeric=False)),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=1.0,
                reg_lambda=1.0,
                scale_pos_weight=pos_weight,
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                n_jobs=-1,
                random_state=cfg.RANDOM_STATE,
            )),
        ]),
    }


def positive_weight(y) -> float:
    y = np.asarray(y).astype(int)
    n_pos = max(int(y.sum()), 1)
    return float((len(y) - n_pos) / n_pos)


# --------------------------------------------------------------------------
# Cross-validation producing out-of-fold probabilities
# --------------------------------------------------------------------------
def cross_val_oof(models: dict[str, Pipeline], X: pd.DataFrame, y,
                  n_splits: int | None = None,
                  random_state: int | None = None) -> dict[str, np.ndarray]:
    """Stratified k-fold out-of-fold probabilities for every model.

    Out-of-fold predictions serve two purposes: an honest estimate of
    generalisation for model selection, and a clean sample on which to tune the
    decision threshold without ever touching the held-out test set.

    The whole pipeline -- imputation, scaling, one-hot encoding -- is refitted
    inside each fold, so no statistic from a validation fold reaches its own
    training fold.
    """
    n_splits = cfg.N_SPLITS if n_splits is None else n_splits
    random_state = cfg.RANDOM_STATE if random_state is None else random_state

    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    oof = {name: np.zeros(len(y), dtype=float) for name in models}
    fold_scores = {name: [] for name in models}

    from sklearn.base import clone
    from sklearn.metrics import average_precision_score

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        for name, model in models.items():
            m = clone(model)
            if name == "XGBoost":
                m.set_params(clf__scale_pos_weight=positive_weight(y_tr))
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_va)[:, 1]
            oof[name][va_idx] = p
            fold_scores[name].append(average_precision_score(y_va, p))

    return oof, {k: np.asarray(v) for k, v in fold_scores.items()}


def cv_summary(fold_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean +/- std PR-AUC per model across folds."""
    rows = [
        {
            "model": name,
            "cv_pr_auc_mean": float(np.mean(s)),
            "cv_pr_auc_std": float(np.std(s)),
            "folds": [round(float(x), 4) for x in s],
        }
        for name, s in fold_scores.items()
    ]
    return pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False)


# --------------------------------------------------------------------------
# Fault-mode classifier (multi-label)
# --------------------------------------------------------------------------
def build_mode_classifier(pos_weights: dict[str, float] | None = None):
    """One binary head per physical fault mode, wrapped as a multi-label model.

    Modes co-occur -- 23 of the usable failures carry two or more flags, most
    often power together with overstrain -- so independent binary heads are the
    honest formulation. Taking the highest-probability head still yields a
    single headline diagnosis when one is wanted.
    """
    from sklearn.multioutput import MultiOutputClassifier

    base = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
    )
    return Pipeline([
        ("prep", feat.build_preprocessor(scale_numeric=False)),
        ("clf", MultiOutputClassifier(base, n_jobs=1)),
    ])


def mode_predict_proba(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-mode probabilities as a tidy frame with the mode codes as columns."""
    probas = model.predict_proba(X)
    cols = {}
    for code, p in zip(cfg.MODE_COLS_MODELLED, probas):
        # MultiOutputClassifier returns one (n, 2) array per label; if a label
        # had a single class in training, guard the index.
        cols[code] = p[:, 1] if p.shape[1] == 2 else np.zeros(len(X))
    return pd.DataFrame(cols, index=X.index)


__all__ = [
    "build_models", "positive_weight", "cross_val_oof", "cv_summary",
    "build_mode_classifier", "mode_predict_proba",
]
