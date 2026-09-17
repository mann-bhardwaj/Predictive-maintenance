"""Loading, integrity auditing and splitting of the AI4I 2020 dataset.

Design decisions encoded here (all deliberate, all documented in the README):

* ``RNF`` (random failure) stays in the binary target but is excluded from the
  fault-mode target -- it has no sensor signature, so it is an unlearnable class
  and including it would depress macro metrics for a reason unrelated to model
  quality.
* Nine rows carry ``Machine failure == 1`` with no mode flag at all. They remain
  in the binary problem (a real stoppage was recorded) and are excluded from the
  fault-mode problem. Together with RNF these rows form an irreducible noise
  floor, quantified by :func:`label_noise_floor`.
* ``UDI`` and ``Product ID`` are identifiers, never features: ``UDI`` is a row
  counter and ``Product ID`` encodes the machine type, which is already a
  feature. Feeding either one would leak ordering information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV exactly as published (BOM-tolerant)."""
    path = path or cfg.RAW_CSV
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = {
        cfg.COL_UDI, cfg.COL_PRODUCT_ID, cfg.COL_TYPE, cfg.COL_AIR_TEMP,
        cfg.COL_PROC_TEMP, cfg.COL_SPEED, cfg.COL_TORQUE, cfg.COL_WEAR,
        cfg.TARGET_BINARY, *cfg.MODE_COLS,
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing)}")
    return df


# --------------------------------------------------------------------------
# Integrity audit
# --------------------------------------------------------------------------
@dataclass
class LabelAudit:
    n_rows: int
    n_missing: int
    n_duplicate_ids: int
    n_failures: int
    failure_rate: float
    mode_counts: dict
    n_failure_no_mode: int
    n_mode_without_failure: int
    n_multi_mode: int
    multi_mode_combos: dict
    n_rnf_flagged: int
    n_rnf_and_failure: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def audit_labels(df: pd.DataFrame) -> LabelAudit:
    """Quantify every label inconsistency in the dataset.

    Being explicit about these is the difference between a model whose recall
    ceiling is understood and one whose ceiling looks like a bug.
    """
    modes = cfg.MODE_COLS_MODELLED
    n_modes = df[modes].sum(axis=1)
    failed = df[cfg.TARGET_BINARY] == 1

    combos = (
        df.loc[failed & (n_modes > 1), modes]
        .apply(lambda row: "+".join(m for m in modes if row[m] == 1), axis=1)
        .value_counts()
        .to_dict()
    )

    return LabelAudit(
        n_rows=int(len(df)),
        n_missing=int(df.isna().sum().sum()),
        n_duplicate_ids=int(df[cfg.COL_UDI].duplicated().sum()),
        n_failures=int(failed.sum()),
        failure_rate=float(failed.mean()),
        mode_counts={c: int(df[c].sum()) for c in cfg.MODE_COLS},
        n_failure_no_mode=int((failed & (df[cfg.MODE_COLS].sum(axis=1) == 0)).sum()),
        n_mode_without_failure=int(((~failed) & (n_modes > 0)).sum()),
        n_multi_mode=int((failed & (n_modes > 1)).sum()),
        multi_mode_combos=combos,
        n_rnf_flagged=int(df["RNF"].sum()),
        n_rnf_and_failure=int((failed & (df["RNF"] == 1)).sum()),
    )


def label_noise_floor(df: pd.DataFrame) -> dict:
    """Share of recorded failures that no sensor-driven model can explain.

    A failure is 'unexplainable' when no physical mode flag accompanies it: the
    only flag is RNF, or there is no flag at all. These rows put a hard ceiling
    on the precision any honest model can reach, because identical sensor
    readings elsewhere in the data are labelled normal.
    """
    failed = df[cfg.TARGET_BINARY] == 1
    physical = df[cfg.MODE_COLS_MODELLED].sum(axis=1) > 0
    unexplained = failed & ~physical
    return {
        "n_failures": int(failed.sum()),
        "n_unexplained_failures": int(unexplained.sum()),
        "unexplained_share_of_failures": float(unexplained.sum() / max(failed.sum(), 1)),
        "n_explainable_failures": int((failed & physical).sum()),
    }


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the modelling targets without mutating the caller's frame."""
    out = df.copy()
    out["y_failure"] = out[cfg.TARGET_BINARY].astype(int)
    out["n_physical_modes"] = out[cfg.MODE_COLS_MODELLED].sum(axis=1).astype(int)
    # Eligible for the fault-mode problem: a recorded failure with at least one
    # physically-driven mode flag.
    out["mode_eligible"] = (
        (out["y_failure"] == 1) & (out["n_physical_modes"] > 0)
    )
    return out


def mode_target_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows and multi-label target matrix for the fault-mode classifier."""
    subset = df.loc[df["mode_eligible"]].copy()
    Y = subset[cfg.MODE_COLS_MODELLED].astype(int)
    return subset, Y


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, test_size: float | None = None,
                     random_state: int | None = None):
    """Hold back a stratified test set that is scored exactly once.

    Stratifying on the composite (failure, mode-signature) key keeps the rare
    fault modes represented on both sides of the split, which matters when the
    smallest mode has only 46 examples in total.
    """
    test_size = cfg.TEST_SIZE if test_size is None else test_size
    random_state = cfg.RANDOM_STATE if random_state is None else random_state

    signature = df[cfg.MODE_COLS_MODELLED].astype(str).agg("".join, axis=1)
    strat = df["y_failure"].astype(str) + "_" + signature

    # Collapse signatures too rare to stratify on. A mode combination seen only
    # once -- the single TWF+PWF+OSF row, for instance -- cannot appear on both
    # sides of a split, so those rows are absorbed into the most populous
    # signature sharing their binary label. Repeated until every stratum has at
    # least two members, so the split degrades gracefully instead of failing.
    for _ in range(5):
        counts = strat.value_counts()
        too_rare = counts[counts < 2].index
        if len(too_rare) == 0:
            break
        for y_val in df["y_failure"].unique():
            same_class = strat[df["y_failure"] == y_val]
            viable = same_class.value_counts()
            viable = viable[~viable.index.isin(too_rare)]
            if viable.empty:
                continue
            host = viable.index[0]
            rows = (df["y_failure"] == y_val) & strat.isin(too_rare)
            strat = strat.where(~rows, host)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def describe_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Small table proving the split preserved the rare-event structure."""
    rows = []
    for name, part in (("train", train_df), ("test", test_df)):
        row = {
            "split": name,
            "rows": len(part),
            "failures": int(part["y_failure"].sum()),
            "failure_rate_%": round(100 * part["y_failure"].mean(), 3),
        }
        for m in cfg.MODE_COLS_MODELLED:
            row[m] = int(part[m].sum())
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "load_raw", "audit_labels", "label_noise_floor", "add_targets",
    "mode_target_matrix", "stratified_split", "describe_split", "LabelAudit",
]
