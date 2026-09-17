"""Industrial AI-based predictive maintenance and process fault detection.

Package layout
--------------
``config``    paths, schema constants, maintenance-cost assumptions
``data``      loading, label auditing, target construction, splitting
``features``  physics-informed feature engineering and the preprocessor
``models``    baseline model zoo, cross-validation, fault-mode classifier
``anomaly``   unsupervised novelty detection (Isolation Forest)
``evaluate``  rare-event metrics, cost-driven threshold tuning, figures
``explain``   SHAP global and local explanations
``viz``       the shared palette and matplotlib style
"""

__version__ = "1.0.0"

__all__ = [
    "config", "data", "features", "models", "anomaly", "evaluate", "explain",
    "viz",
]
