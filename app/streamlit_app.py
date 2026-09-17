"""Predictive maintenance dashboard.

    streamlit run app/streamlit_app.py

Three things an operator can do here:

1. **Score one machine** from live sensor values and see why it was flagged.
2. **Score a whole fleet** from a CSV and get an inspection-priority list.
3. **Read the model card** -- what the system was trained on, how it performs,
   and where it is known to be blind.

Everything the app shows comes from the artefacts written by
``run_pipeline.py``; the app never trains anything itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import anomaly, config as cfg, explain, features, models, viz

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="•",
    layout="wide",
)

viz.apply_style()


# --------------------------------------------------------------------------
# Artefact loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    missing = [p.name for p in (cfg.ARTIFACT_BINARY_MODEL, cfg.ARTIFACT_MODE_MODEL,
                                cfg.ARTIFACT_ANOMALY_MODEL, cfg.ARTIFACT_METADATA)
               if not p.exists()]
    if missing:
        return None, None, None, None, missing

    clf = joblib.load(cfg.ARTIFACT_BINARY_MODEL)
    mode_clf = joblib.load(cfg.ARTIFACT_MODE_MODEL)
    detector = joblib.load(cfg.ARTIFACT_ANOMALY_MODEL)
    with open(cfg.ARTIFACT_METADATA) as fh:
        meta = json.load(fh)
    return clf, mode_clf, detector, meta, []


@st.cache_resource(show_spinner="Preparing explainer...")
def load_explainer(_clf):
    return explain.PipelineExplainer(_clf)


clf, mode_clf, detector, meta, missing = load_artifacts()

if missing:
    st.title("Industrial predictive maintenance")
    st.error(
        "Model artefacts are missing: " + ", ".join(missing) +
        "\n\nRun the training pipeline first:\n\n    python run_pipeline.py"
    )
    st.stop()

explainer = load_explainer(clf)
THRESHOLD = float(meta["decision_threshold"])
ANOM_REF = meta.get("anomaly_score_reference", {})


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def build_row(machine_type: str, air_t: float, proc_t: float, speed: float,
              torque: float, wear: float) -> pd.DataFrame:
    raw = pd.DataFrame([{
        cfg.COL_TYPE: machine_type,
        cfg.COL_AIR_TEMP: air_t,
        cfg.COL_PROC_TEMP: proc_t,
        cfg.COL_SPEED: speed,
        cfg.COL_TORQUE: torque,
        cfg.COL_WEAR: wear,
    }])
    return features.add_physics_features(raw)


def risk_band(prob: float) -> tuple[str, str]:
    """Status label and colour for a probability, relative to the threshold."""
    if prob >= max(0.75, THRESHOLD * 2):
        return "CRITICAL", viz.STATUS["critical"]
    if prob >= THRESHOLD:
        return "ALARM", viz.STATUS["serious"]
    if prob >= THRESHOLD * 0.5:
        return "WATCH", viz.STATUS["warning"]
    return "HEALTHY", viz.STATUS["good"]


def anomaly_percentile(score: float) -> str:
    if not ANOM_REF:
        return "n/a"
    if score >= ANOM_REF.get("p99", np.inf):
        return "top 1% most unusual"
    if score >= ANOM_REF.get("p90", np.inf):
        return "top 10% most unusual"
    if score >= ANOM_REF.get("p50", np.inf):
        return "above median unusualness"
    return "typical operating conditions"


def mode_bar(mode_probs: dict):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ranked = sorted(mode_probs.items(), key=lambda kv: kv[1])
    ys = np.arange(len(ranked))
    vals = [v for _, v in ranked]
    ax.barh(ys, vals, height=0.6, color=viz.SERIES[0])
    ax.set_yticks(ys, [cfg.MODE_LABELS[k].replace(" failure", "")
                       for k, _ in ranked])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for y, v in zip(ys, vals):
        ax.text(v + 0.02, y, f"{v:.0%}", va="center", fontsize=9,
                color=viz.INK_SECONDARY)
    ax.set_xlim(0, 1.16)
    viz.finish(ax, title="Which fault is indicated",
               subtitle="independent per-mode probabilities; faults can co-occur",
               xlabel="Probability", ylabel="")
    return fig


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Industrial predictive maintenance and fault detection")
st.caption(
    f"{meta['selected_model']} classifier · alarm threshold "
    f"{THRESHOLD:.1%} · held-out PR-AUC "
    f"{meta['test_metrics']['pr_auc']:.3f} · recall "
    f"{meta['test_metrics']['recall']:.1%} at precision "
    f"{meta['test_metrics']['precision']:.1%}"
)

tab_single, tab_fleet, tab_card = st.tabs(
    ["Single machine", "Fleet scoring", "Model card"]
)

# ==========================================================================
# TAB 1 -- single machine
# ==========================================================================
with tab_single:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Sensor readings")
        machine_type = st.selectbox(
            "Machine quality variant", cfg.TYPE_ORDER, index=0,
            help="L = low, M = medium, H = high quality. 60% of the fleet is L.",
        )
        air_t = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
        proc_t = st.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
        speed = st.slider("Rotational speed (rpm)", 1160, 2890, 1500, 5)
        torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.1)
        wear = st.slider("Tool wear (min)", 0, 255, 100, 1)

        st.markdown("**Preset scenarios**")
        preset = st.radio(
            "Load a scenario", [
                "None",
                "Healthy nominal",
                "Heat dissipation risk",
                "Power failure risk",
                "Overstrain risk",
            ],
            label_visibility="collapsed",
        )

    presets = {
        "Healthy nominal": ("L", 298.5, 308.6, 1500, 40.0, 60),
        "Heat dissipation risk": ("L", 302.5, 310.8, 1320, 52.0, 180),
        "Power failure risk": ("L", 300.0, 310.0, 2700, 12.0, 100),
        "Overstrain risk": ("L", 300.0, 310.0, 1300, 62.0, 225),
    }
    if preset != "None":
        machine_type, air_t, proc_t, speed, torque, wear = presets[preset]

    row = build_row(machine_type, air_t, proc_t, speed, torque, wear)
    X_row = features.X_frame(row)

    prob = float(clf.predict_proba(X_row)[:, 1][0])
    band, band_color = risk_band(prob)
    mode_probs = models.mode_predict_proba(mode_clf, X_row).iloc[0].to_dict()
    a_score = float(anomaly.anomaly_scores(detector, X_row)[0])
    row_expl = explainer.explain_row(X_row, top_n=6)

    with right:
        st.subheader("Assessment")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Failure probability", f"{prob:.1%}")
        c2.metric("Alarm threshold", f"{THRESHOLD:.1%}")
        c3.metric("Status", band)
        c4.metric("Anomaly score", f"{a_score:+.3f}")

        st.markdown(
            f"<div style='border-left:4px solid {band_color};"
            f"padding:10px 14px;background:#f6f6f4;border-radius:4px;'>"
            f"<strong>{band}</strong> &nbsp;·&nbsp; "
            f"{anomaly_percentile(a_score)}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Status bands are labelled, not colour-only: ALARM means the "
            "probability crossed the cost-optimal threshold; WATCH means it is "
            "within half a threshold of doing so."
        )

        st.markdown("#### " + ("Recommended action: inspect" if prob >= THRESHOLD
                               else "Recommended action: none"))
        st.write(explain.narrate(row_expl, prob, THRESHOLD, mode_probs))

        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            fig = explain.plot_local_explanation(
                row_expl, prob, THRESHOLD, units=explainer.units)
            st.pyplot(fig.figure, clear_figure=True)
        with g2:
            st.pyplot(mode_bar(mode_probs), clear_figure=True)

        with st.expander("Derived physical quantities"):
            derived = row[cfg.ENGINEERED_COLS].T.rename(columns={0: "value"})
            derived["value"] = derived["value"].round(3)
            st.dataframe(derived, width="stretch")

        with st.expander("Contribution table (same numbers as the chart)"):
            st.dataframe(
                row_expl.assign(shap=lambda d: d["shap"].round(4),
                                value=lambda d: d["value"].round(3)),
                width="stretch", hide_index=True,
            )

# ==========================================================================
# TAB 2 -- fleet scoring
# ==========================================================================
with tab_fleet:
    st.subheader("Score a fleet from CSV")
    st.write(
        "Upload a CSV with the raw sensor columns and every row is scored, "
        "ranked by risk, and returned as an inspection priority list. "
        f"Required columns: `{cfg.COL_TYPE}`, `{cfg.COL_AIR_TEMP}`, "
        f"`{cfg.COL_PROC_TEMP}`, `{cfg.COL_SPEED}`, `{cfg.COL_TORQUE}`, "
        f"`{cfg.COL_WEAR}`."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    use_sample = st.checkbox(
        "Use the held-out test split instead", value=uploaded is None,
        help="Scores the 2,000 machines the model never saw during training.",
    )

    fleet = None
    truth = None
    if uploaded is not None:
        fleet = pd.read_csv(uploaded, encoding="utf-8-sig")
    elif use_sample:
        sample_path = cfg.DATA_PROCESSED / "test.parquet"
        if sample_path.exists():
            fleet = pd.read_parquet(sample_path)
            truth = fleet["y_failure"].to_numpy()
        else:
            st.info("Run `python run_pipeline.py` to generate the held-out split.")

    if fleet is not None:
        required = [cfg.COL_TYPE] + cfg.SENSOR_COLS
        absent = [c for c in required if c not in fleet.columns]
        if absent:
            st.error(f"Missing required columns: {absent}")
        else:
            scored = features.add_physics_features(fleet)
            Xf = features.X_frame(scored)
            probs = clf.predict_proba(Xf)[:, 1]
            scores = anomaly.anomaly_scores(detector, Xf)
            mode_p = models.mode_predict_proba(mode_clf, Xf)

            out = fleet.copy()
            out["failure_probability"] = probs.round(4)
            out["status"] = [risk_band(p)[0] for p in probs]
            out["anomaly_score"] = scores.round(4)
            out["likely_fault"] = [
                cfg.MODE_LABELS[c] for c in mode_p.idxmax(axis=1)
            ]
            out["fault_confidence"] = mode_p.max(axis=1).to_numpy().round(3)
            # A machine only carries a diagnosis if it was flagged at all.
            healthy = out["failure_probability"] < THRESHOLD
            out.loc[healthy, ["likely_fault", "fault_confidence"]] = ["-", np.nan]

            out = out.sort_values("failure_probability", ascending=False)

            n_alarm = int((probs >= THRESHOLD).sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Machines scored", f"{len(out):,}")
            m2.metric("Flagged for inspection", f"{n_alarm:,}",
                      f"{n_alarm/len(out):.1%} of fleet")
            m3.metric("Mean failure probability", f"{probs.mean():.2%}")
            m4.metric("Highest risk", f"{probs.max():.1%}")

            if truth is not None:
                tp = int(((probs >= THRESHOLD) & (truth == 1)).sum())
                fn = int(((probs < THRESHOLD) & (truth == 1)).sum())
                fp = int(((probs >= THRESHOLD) & (truth == 0)).sum())
                st.success(
                    f"Ground truth available for this sample: "
                    f"{tp} failures caught, {fn} missed, {fp} false alarms "
                    f"({tp}/{tp+fn} = {tp/max(tp+fn,1):.1%} recall)."
                )

            st.dataframe(
                out.head(200), width="stretch", hide_index=True,
                column_config={
                    "failure_probability": st.column_config.ProgressColumn(
                        "Failure probability", min_value=0.0, max_value=1.0,
                        format="%.1f%%",
                    ),
                },
            )
            st.download_button(
                "Download the full scored list (CSV)",
                out.to_csv(index=False).encode(),
                file_name="inspection_priority.csv",
                mime="text/csv",
            )

# ==========================================================================
# TAB 3 -- model card
# ==========================================================================
with tab_card:
    st.subheader("Model card")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training**")
        st.write(
            f"- Dataset: AI4I 2020 Predictive Maintenance, "
            f"{meta['training_rows']:,} training / {meta['test_rows']:,} "
            f"held-out rows\n"
            f"- Selected model: {meta['selected_model']} "
            f"(chosen on cross-validated PR-AUC)\n"
            f"- Features: 5 raw sensors, 5 derived physical quantities, "
            f"machine quality variant\n"
            f"- Imbalance: class weighting, no synthetic oversampling\n"
            f"- SHAP contributions are in {meta.get('shap_units', 'probability')} units"
        )
        st.markdown("**Cross-validated PR-AUC**")
        cv = pd.DataFrame(
            [{"model": k, "PR-AUC": round(v, 4)}
             for k, v in meta["cv_pr_auc"].items()]
        ).sort_values("PR-AUC", ascending=False)
        st.dataframe(cv, hide_index=True, width="stretch")

    with c2:
        st.markdown("**Held-out performance** (test set scored once)")
        tm = meta["test_metrics"]
        st.dataframe(
            # Values are formatted to strings so the column stays a single
            # dtype -- mixing floats and a percentage string breaks the Arrow
            # conversion Streamlit uses to render tables.
            pd.DataFrame([
                {"metric": "PR-AUC", "value": f"{tm['pr_auc']:.4f}"},
                {"metric": "ROC-AUC", "value": f"{tm['roc_auc']:.4f}"},
                {"metric": "Precision", "value": f"{tm['precision']:.1%}"},
                {"metric": "Recall", "value": f"{tm['recall']:.1%}"},
                {"metric": "F2", "value": f"{tm['f2']:.4f}"},
                {"metric": "Failures caught", "value": f"{tm['tp']}"},
                {"metric": "Failures missed", "value": f"{tm['fn']}"},
                {"metric": "False alarms", "value": f"{tm['fp']}"},
                {"metric": "Maintenance cost avoided",
                 "value": f"{tm['cost_saved_pct']:.1f}%"},
            ]),
            hide_index=True, width="stretch",
        )
        st.markdown("**Operating point**")
        st.write(meta["threshold_rule"])

    st.divider()
    st.markdown("**Known limitations** — read these before trusting a number")
    nf = meta["noise_floor"]
    st.write(
        f"- {nf['n_unexplained_failures']} of {nf['n_failures']} recorded "
        f"failures ({nf['unexplained_share_of_failures']:.1%}) carry no "
        "physical fault flag: random failures, or failures with no mode "
        "recorded at all. No sensor-driven model can predict these, so they "
        "set a hard ceiling on achievable recall.\n"
        f"- Fault-mode diagnosis is **conditional on a failure having been "
        f"flagged** and covers only the four physically-driven modes. It "
        f"names a genuinely present mode in "
        f"{meta['fault_mode_top1_accuracy']:.1%} of held-out failures.\n"
        "- The dataset is a single snapshot per machine cycle, not a "
        "continuous time series, so this system predicts *failure state*, not "
        "remaining useful life. Any RUL claim would require run-to-failure "
        "histories such as NASA C-MAPSS.\n"
        "- Features are physical quantities only. The dataset's labels were "
        "generated by threshold rules over these same quantities, so the "
        "fault-mode stage scores very high by construction; the binary stage "
        "is the honest difficulty measure.\n"
        "- The anomaly detector is unsupervised and fitted on healthy rows "
        "only. It exists to catch fault types the classifier has never seen, "
        "and its precision is correspondingly lower."
    )
