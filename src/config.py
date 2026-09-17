"""Central configuration: paths, column names, domain constants, and the
maintenance-cost assumptions that drive threshold selection.

Everything tunable lives here so notebooks, the pipeline and the dashboard all
read the same source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RAW_CSV = DATA_RAW / "ai4i2020.csv"

for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20

# --------------------------------------------------------------------------
# Raw schema (AI4I 2020 Predictive Maintenance Dataset)
# --------------------------------------------------------------------------
COL_UDI = "UDI"
COL_PRODUCT_ID = "Product ID"
COL_TYPE = "Type"
COL_AIR_TEMP = "Air temperature [K]"
COL_PROC_TEMP = "Process temperature [K]"
COL_SPEED = "Rotational speed [rpm]"
COL_TORQUE = "Torque [Nm]"
COL_WEAR = "Tool wear [min]"

TARGET_BINARY = "Machine failure"

# The five recorded failure modes.
MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# RNF ("random failure") has no physical driver in the sensors, so it is kept in
# the binary target -- operationally a failure is a failure -- but excluded from
# the fault-mode classifier, where it would be an unlearnable class.
MODE_COLS_MODELLED = ["TWF", "HDF", "PWF", "OSF"]

MODE_LABELS = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}

SENSOR_COLS = [COL_AIR_TEMP, COL_PROC_TEMP, COL_SPEED, COL_TORQUE, COL_WEAR]

# --------------------------------------------------------------------------
# Engineered feature names (see features.py)
# --------------------------------------------------------------------------
FEAT_POWER = "mechanical_power_W"
FEAT_TEMP_DIFF = "temp_differential_K"
FEAT_STRAIN = "wear_torque_strain_minNm"
FEAT_TORQUE_PER_SPEED = "torque_per_rpm"
FEAT_COOLING_INDEX = "cooling_load_index"
FEAT_TYPE_ORD = "type_quality_ordinal"

ENGINEERED_COLS = [
    FEAT_POWER,
    FEAT_TEMP_DIFF,
    FEAT_STRAIN,
    FEAT_TORQUE_PER_SPEED,
    FEAT_COOLING_INDEX,
]

# Numeric columns fed to the models.
NUMERIC_FEATURES = SENSOR_COLS + ENGINEERED_COLS
CATEGORICAL_FEATURES = [COL_TYPE]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Machine-quality variants and their share of the fleet.
TYPE_ORDER = ["L", "M", "H"]
TYPE_TO_ORDINAL = {"L": 0, "M": 1, "H": 2}

# --------------------------------------------------------------------------
# Maintenance economics
# --------------------------------------------------------------------------
# A missed failure (unplanned breakdown: lost production, collateral damage,
# emergency labour) is assumed to cost 10x an unnecessary inspection. This is a
# business input, not a property of the data -- evaluate.py sweeps it so the
# sensitivity of the operating point is explicit.
COST_FALSE_NEGATIVE = 10.0
COST_FALSE_POSITIVE = 1.0
COST_RATIO_SWEEP = [5.0, 10.0, 20.0, 50.0]

# Anomaly detector: expected share of abnormal operating conditions. Set from
# the observed failure base rate rather than tuned on labels.
ANOMALY_CONTAMINATION = 0.034

# --------------------------------------------------------------------------
# Artefact filenames
# --------------------------------------------------------------------------
ARTIFACT_BINARY_MODEL = MODELS_DIR / "failure_classifier.joblib"
ARTIFACT_MODE_MODEL = MODELS_DIR / "fault_mode_classifier.joblib"
ARTIFACT_ANOMALY_MODEL = MODELS_DIR / "anomaly_detector.joblib"
ARTIFACT_METADATA = MODELS_DIR / "model_metadata.json"
ARTIFACT_SHAP_BACKGROUND = MODELS_DIR / "shap_background.joblib"

METRICS_CV = METRICS_DIR / "cv_results.json"
METRICS_TEST = METRICS_DIR / "test_results.json"
METRICS_MODE = METRICS_DIR / "fault_mode_results.json"
METRICS_ANOMALY = METRICS_DIR / "anomaly_results.json"
METRICS_THRESHOLD_SWEEP = METRICS_DIR / "threshold_sweep.json"
