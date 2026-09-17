#!/usr/bin/env python3
"""End-to-end training and evaluation run.

    python run_pipeline.py

Executes all six stages, persists the fitted models for the dashboard, writes
every metric to ``reports/metrics/*.json`` and every figure to
``reports/figures/*.png``.

Discipline enforced here:

* The held-out test set is scored **once**, at the end, using a threshold that
  was chosen entirely from out-of-fold training predictions.
* The whole preprocessing pipeline is refitted inside every CV fold.
* The anomaly detector is fitted on healthy training rows only.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import anomaly, config as cfg, data, eda, evaluate, explain, features, models, viz

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def main() -> int:
    t0 = time.time()
    viz.apply_style()

    # ==================================================================
    # STAGE 1 -- Dataset and understanding
    # ==================================================================
    banner("STAGE 1  Dataset, schema and label integrity")
    df_raw = data.load_raw()
    audit = data.audit_labels(df_raw)
    noise = data.label_noise_floor(df_raw)

    print(f"rows: {audit.n_rows:,}   missing values: {audit.n_missing}   "
          f"duplicate IDs: {audit.n_duplicate_ids}")
    print(f"failures: {audit.n_failures} ({audit.failure_rate:.2%})")
    print(f"mode counts: {audit.mode_counts}")
    print(f"failures with no mode flag at all: {audit.n_failure_no_mode}")
    print(f"failures with 2+ modes: {audit.n_multi_mode}  {audit.multi_mode_combos}")
    print(f"RNF flagged rows: {audit.n_rnf_flagged} "
          f"(of which actual failures: {audit.n_rnf_and_failure})")
    print(f"NOISE FLOOR: {noise['n_unexplained_failures']} of "
          f"{noise['n_failures']} failures "
          f"({noise['unexplained_share_of_failures']:.1%}) have no physical "
          f"signature - no sensor-driven model can explain them.")

    evaluate.save_json({"audit": audit.as_dict(), "noise_floor": noise},
                       cfg.METRICS_DIR / "data_audit.json")

    # ==================================================================
    # STAGE 2 -- Feature engineering and EDA
    # ==================================================================
    banner("STAGE 2  Physics-informed features and exploratory analysis")
    df = data.add_targets(df_raw)
    df = features.add_physics_features(df)
    print(f"engineered: {', '.join(cfg.ENGINEERED_COLS)}")
    print("\nHealthy vs failed means (largest relative shift first):")
    shift = eda.summary_table(df)
    print(shift.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    shift.to_csv(cfg.METRICS_DIR / "feature_shift.csv", index=False)

    for name, fn in [
        ("01_class_balance", eda.plot_class_balance),
        ("02_failure_modes", eda.plot_mode_counts),
        ("04_correlation", eda.plot_correlation),
        ("05_operating_envelope", eda.plot_operating_envelope),
        ("06_wear_vs_failure", eda.plot_wear_vs_failure),
        ("07_machine_type", eda.plot_type_breakdown),
    ]:
        ax = fn(df)
        viz.save(ax.figure, name)
        plt.close(ax.figure)
    fig, _ = eda.plot_sensor_distributions(df)
    viz.save(fig, "03_sensor_distributions")
    plt.close(fig)
    print(f"figures written to {cfg.FIGURES_DIR}")

    # ==================================================================
    # Split -- the test set is not touched again until the very end
    # ==================================================================
    banner("Train / test split (stratified, test set sealed until the end)")
    train_df, test_df = data.stratified_split(df)
    print(data.describe_split(train_df, test_df).to_string(index=False))

    X_train = features.X_frame(train_df)
    y_train = train_df["y_failure"].to_numpy()
    X_test = features.X_frame(test_df)
    y_test = test_df["y_failure"].to_numpy()

    # ==================================================================
    # STAGE 3 -- Baseline supervised models
    # ==================================================================
    banner("STAGE 3  Baseline models, 5-fold stratified cross-validation")
    zoo = models.build_models(pos_weight=models.positive_weight(y_train))
    oof, fold_scores = models.cross_val_oof(zoo, X_train, y_train)
    cv_table = models.cv_summary(fold_scores)
    print(cv_table.to_string(index=False))

    best_name = cv_table.iloc[0]["model"]
    print(f"\nselected by CV PR-AUC: {best_name}")

    ax = evaluate.plot_pr_curves(oof, y_train,
                                 title="Precision-recall, out-of-fold on training data")
    viz.save(ax.figure, "08_pr_curves_cv")
    plt.close(ax.figure)
    ax = evaluate.plot_roc_curves(oof, y_train,
                                  title="ROC, out-of-fold on training data")
    viz.save(ax.figure, "09_roc_curves_cv")
    plt.close(ax.figure)

    cv_payload = {
        "table": cv_table.drop(columns="folds").to_dict(orient="records"),
        "folds": {k: v.tolist() for k, v in fold_scores.items()},
        "selected_model": best_name,
        "oof_pr_auc": {k: float(np.mean(v)) for k, v in fold_scores.items()},
    }
    evaluate.save_json(cv_payload, cfg.METRICS_CV)

    # ------------------------------------------------------------------
    # Threshold selection -- on out-of-fold predictions only
    # ------------------------------------------------------------------
    banner("Operating point: cost-driven threshold (chosen out-of-fold)")
    oof_best = oof[best_name]
    tuned = evaluate.tune_threshold_by_cost(y_train, oof_best)
    threshold = tuned["threshold"]
    print(f"cost assumption: a missed failure = {tuned['c_fn']:.0f}x an "
          f"unnecessary inspection")
    print(f"chosen threshold: {threshold:.4f}  "
          f"(out-of-fold cost {tuned['cost_units']:.0f} inspection-units)")

    ax = evaluate.plot_threshold_cost(tuned)
    viz.save(ax.figure, "10_threshold_cost")
    plt.close(ax.figure)

    sweep = evaluate.threshold_cost_sweep(y_train, oof_best)
    print("\nSensitivity of the operating point to the cost assumption:")
    print(sweep.to_string(index=False))
    sweep.to_csv(cfg.METRICS_DIR / "threshold_sweep.csv", index=False)
    evaluate.save_json({"chosen": {k: v for k, v in tuned.items()
                                   if k not in ("grid", "costs")},
                        "sweep": sweep.to_dict(orient="records")},
                       cfg.METRICS_THRESHOLD_SWEEP)

    rap = evaluate.recall_at_precision(y_train, oof_best, 0.90)
    print(f"\nout-of-fold recall at >=90% precision: {rap['recall']:.3f}")

    # ------------------------------------------------------------------
    # Fit the selected model on all training data
    # ------------------------------------------------------------------
    best_model = zoo[best_name]
    if best_name == "XGBoost":
        best_model.set_params(clf__scale_pos_weight=models.positive_weight(y_train))
    best_model.fit(X_train, y_train)

    # ==================================================================
    # STAGE 4 -- Unsupervised anomaly detection
    # ==================================================================
    banner("STAGE 4  Anomaly detection (Isolation Forest, healthy rows only)")
    detector = anomaly.build_anomaly_detector()
    anomaly.fit_on_healthy(detector, X_train, y_train)
    scores_test = anomaly.anomaly_scores(detector, X_test)
    anom_eval = anomaly.evaluate_detector(scores_test, y_test)

    print(f"PR-AUC {anom_eval['pr_auc']:.3f}   ROC-AUC {anom_eval['roc_auc']:.3f}   "
          f"(label-free detector, evaluated against labels)")
    print(pd.DataFrame(anom_eval["top_k"]).to_string(index=False,
                                                     float_format=lambda v: f"{v:,.3f}"))
    evaluate.save_json(anom_eval, cfg.METRICS_ANOMALY)

    ax = anomaly.plot_score_distribution(scores_test, y_test)
    viz.save(ax.figure, "11_anomaly_scores")
    plt.close(ax.figure)
    ax = anomaly.plot_top_k_lift(anom_eval)
    viz.save(ax.figure, "12_anomaly_lift")
    plt.close(ax.figure)

    # ==================================================================
    # STAGE 5 -- Predictive maintenance: fault-mode diagnosis
    # ==================================================================
    banner("STAGE 5  Fault-mode diagnosis (multi-label, conditional on failure)")
    train_modes, Y_train_modes = data.mode_target_matrix(train_df)
    test_modes, Y_test_modes = data.mode_target_matrix(test_df)
    Xm_train = features.X_frame(train_modes)
    Xm_test = features.X_frame(test_modes)
    print(f"mode-eligible rows: {len(train_modes)} train / {len(test_modes)} test")
    print("per-mode positives (train):",
          {c: int(Y_train_modes[c].sum()) for c in cfg.MODE_COLS_MODELLED})

    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.base import clone

    mode_rows = []
    for code in cfg.MODE_COLS_MODELLED:
        y_m = Y_train_modes[code].to_numpy()
        n_pos = int(y_m.sum())
        head = clone(models.build_mode_classifier())
        # Single-label head for a clean stratified CV estimate per mode.
        from xgboost import XGBClassifier
        from sklearn.pipeline import Pipeline
        single = Pipeline([
            ("prep", features.build_preprocessor(scale_numeric=False)),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=-1, random_state=cfg.RANDOM_STATE)),
        ])
        folds = min(5, n_pos) if n_pos >= 2 else 0
        if folds >= 2:
            skf = StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=cfg.RANDOM_STATE)
            p_oof = cross_val_predict(single, Xm_train, y_m, cv=skf,
                                      method="predict_proba")[:, 1]
            cv_ap = float(average_precision_score(y_m, p_oof))
            cv_auc = float(roc_auc_score(y_m, p_oof))
        else:
            cv_ap = cv_auc = float("nan")
        mode_rows.append({
            "mode": code,
            "label": cfg.MODE_LABELS[code],
            "train_positives": n_pos,
            "test_positives": int(Y_test_modes[code].sum()),
            "cv_pr_auc": round(cv_ap, 4),
            "cv_roc_auc": round(cv_auc, 4),
        })

    mode_model = models.build_mode_classifier()
    mode_model.fit(Xm_train, Y_train_modes)
    P_test_modes = models.mode_predict_proba(mode_model, Xm_test)

    for row in mode_rows:
        code = row["mode"]
        y_m = Y_test_modes[code].to_numpy()
        if y_m.sum() >= 1 and y_m.sum() < len(y_m):
            row["test_pr_auc"] = round(float(average_precision_score(y_m, P_test_modes[code])), 4)
            row["test_roc_auc"] = round(float(roc_auc_score(y_m, P_test_modes[code])), 4)
        else:
            row["test_pr_auc"] = row["test_roc_auc"] = None

    mode_table = pd.DataFrame(mode_rows)
    print("\n" + mode_table.to_string(index=False))

    # Top-1 agreement: does the highest-probability head name a true mode?
    top1 = P_test_modes.idxmax(axis=1)
    hit = np.array([Y_test_modes.iloc[i][top1.iloc[i]] == 1
                    for i in range(len(top1))])
    top1_acc = float(hit.mean())
    print(f"\ntop-1 diagnosis names a genuinely present mode in "
          f"{top1_acc:.1%} of held-out failures ({int(hit.sum())}/{len(hit)})")

    evaluate.save_json({
        "per_mode": mode_rows,
        "top1_accuracy": top1_acc,
        "n_train_rows": int(len(train_modes)),
        "n_test_rows": int(len(test_modes)),
        "note": ("Conditional model: trained and evaluated only on rows with a "
                 "recorded failure carrying at least one physical mode flag. "
                 "RNF and mode-less failures are excluded by design."),
    }, cfg.METRICS_MODE)

    # Per-mode CV PR-AUC figure.
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    xs = np.arange(len(mode_table))
    ax.bar(xs, mode_table["cv_pr_auc"], width=0.52, color=viz.SERIES[0])
    for x, v, n in zip(xs, mode_table["cv_pr_auc"], mode_table["train_positives"]):
        ax.text(x, v + 0.03, f"{v:.3f}\n(n={n})", ha="center", va="bottom",
                fontsize=9, color=viz.INK_SECONDARY)
    ax.set_xticks(xs, [cfg.MODE_LABELS[c].replace(" failure", "")
                       for c in mode_table["mode"]])
    ax.set_ylim(0, 1.18)
    viz.finish(ax, title="Fault-mode diagnosis quality by mode",
               subtitle="cross-validated PR-AUC on the failure population; n = training positives",
               xlabel="", ylabel="PR-AUC")
    viz.save(fig, "13_fault_mode_quality")
    plt.close(fig)

    # ==================================================================
    # STAGE 6 -- Explainability
    # ==================================================================
    banner("STAGE 6  SHAP explainability")
    if best_name == "Logistic regression":
        print("selected model is linear; using its coefficients plus a tree "
              "surrogate for SHAP")
        surrogate = zoo["XGBoost"]
        surrogate.set_params(clf__scale_pos_weight=models.positive_weight(y_train))
        surrogate.fit(X_train, y_train)
        explainer = explain.PipelineExplainer(surrogate)
    else:
        explainer = explain.PipelineExplainer(best_model)

    shap_sample = X_train.sample(n=min(2000, len(X_train)),
                                 random_state=cfg.RANDOM_STATE)
    importance = explainer.global_importance(shap_sample)
    print(importance.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    importance.to_csv(cfg.METRICS_DIR / "shap_global_importance.csv", index=False)

    print(f"SHAP contributions are in {explainer.units} units "
          f"(model: {best_name})")
    ax = explain.plot_global_importance(importance, units=explainer.units)
    viz.save(ax.figure, "14_shap_global")
    plt.close(ax.figure)

    try:
        fig = explain.plot_beeswarm(explainer, shap_sample.sample(
            n=min(600, len(shap_sample)), random_state=cfg.RANDOM_STATE))
        viz.save(fig, "15_shap_beeswarm")
        plt.close(fig)
    except Exception as exc:  # beeswarm is cosmetic; never fail the run for it
        print(f"(beeswarm skipped: {exc})")

    # ==================================================================
    # FINAL -- score the sealed test set exactly once
    # ==================================================================
    banner("FINAL EVALUATION  held-out test set, scored once")
    p_test = best_model.predict_proba(X_test)[:, 1]
    report = evaluate.classification_report_at(y_test, p_test, threshold)
    print(json.dumps(report, indent=2))

    rap_test = evaluate.recall_at_precision(y_test, p_test, 0.90)
    report["recall_at_90_precision"] = rap_test
    report["model"] = best_name
    report["n_test"] = int(len(y_test))
    evaluate.save_json(report, cfg.METRICS_TEST)

    test_curves = {best_name: p_test}
    ax = evaluate.plot_pr_curves(test_curves, y_test,
                                 title="Precision-recall on the held-out test set")
    viz.save(ax.figure, "16_pr_curve_test")
    plt.close(ax.figure)

    ax = evaluate.plot_confusion(y_test, (p_test >= threshold).astype(int),
                                 title=f"Held-out confusion matrix at threshold {threshold:.3f}")
    viz.save(ax.figure, "17_confusion_test")
    plt.close(ax.figure)

    ax = evaluate.plot_calibration(y_test, p_test)
    viz.save(ax.figure, "18_calibration_test")
    plt.close(ax.figure)

    # A worked local explanation for the highest-risk held-out machine.
    worst_i = int(np.argmax(p_test))
    row = X_test.iloc[[worst_i]]
    row_expl = explainer.explain_row(row)
    mode_p = models.mode_predict_proba(mode_model, row).iloc[0].to_dict()
    ax = explain.plot_local_explanation(row_expl, float(p_test[worst_i]), threshold,
                                       units=explainer.units)
    viz.save(ax.figure, "19_local_explanation")
    plt.close(ax.figure)
    print("\nWorked example -- highest-risk machine in the held-out set:")
    print(explain.narrate(row_expl, float(p_test[worst_i]), threshold, mode_p))
    print(f"actual label: {'FAILURE' if y_test[worst_i] == 1 else 'healthy'}")

    # ==================================================================
    # Persist artefacts for the dashboard
    # ==================================================================
    banner("Persisting artefacts")
    joblib.dump(best_model, cfg.ARTIFACT_BINARY_MODEL)
    joblib.dump(mode_model, cfg.ARTIFACT_MODE_MODEL)
    joblib.dump(detector, cfg.ARTIFACT_ANOMALY_MODEL)
    joblib.dump(shap_sample.head(200), cfg.ARTIFACT_SHAP_BACKGROUND)

    metadata = {
        "version": "1.0.0",
        "selected_model": best_name,
        "decision_threshold": threshold,
        "threshold_rule": ("minimises expected maintenance cost on out-of-fold "
                           f"training predictions, with a missed failure costing "
                           f"{cfg.COST_FALSE_NEGATIVE:.0f}x an unnecessary inspection"),
        "cost_false_negative": cfg.COST_FALSE_NEGATIVE,
        "cost_false_positive": cfg.COST_FALSE_POSITIVE,
        "feature_cols": cfg.FEATURE_COLS,
        "engineered_cols": cfg.ENGINEERED_COLS,
        "shap_units": explainer.units,
        "mode_cols": cfg.MODE_COLS_MODELLED,
        "anomaly_score_reference": {
            "p50": float(np.percentile(scores_test, 50)),
            "p90": float(np.percentile(scores_test, 90)),
            "p99": float(np.percentile(scores_test, 99)),
        },
        "cv_pr_auc": {k: float(np.mean(v)) for k, v in fold_scores.items()},
        "test_metrics": {k: report[k] for k in
                         ("pr_auc", "roc_auc", "precision", "recall", "f2",
                          "tp", "fp", "fn", "tn", "cost_saved_pct")},
        "fault_mode_top1_accuracy": top1_acc,
        "noise_floor": noise,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    with open(cfg.ARTIFACT_METADATA, "w") as fh:
        json.dump(metadata, fh, indent=2)

    train_df.to_parquet(cfg.DATA_PROCESSED / "train.parquet", index=False)
    test_df.to_parquet(cfg.DATA_PROCESSED / "test.parquet", index=False)

    print(f"models     -> {cfg.MODELS_DIR}")
    print(f"metrics    -> {cfg.METRICS_DIR}")
    print(f"figures    -> {cfg.FIGURES_DIR}")
    print(f"\ncompleted in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
