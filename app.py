"""
app.py
======
Streamlit dashboard — AI Tax + Healthcare Fraud Detection.

Tabs
────
  Pipeline   : run the 4-step ML pipeline
  Analytics  : ROC curves, PR curves, confusion-matrix heatmaps,
               score distributions, feature importance
  Predict    : live single-record fraud probability scoring
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluate_model import evaluate_model
from src.feature_engineering import (
    build_healthcare_features, build_tax_features,
    load_healthcare_claims, load_tax_records,
    split_healthcare, split_tax,
)
from src.generate_synthetic_data import (
    generate_healthcare_claims, generate_tax_records,
    save_healthcare_claims, save_tax_records,
)
from src.train_model import train_model

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
REPORTS   = ROOT / "reports"
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
TARGET    = "is_fraud"
ID_COLS   = {"tax_id", "claim_id", "provider_id", TARGET}

# ── Colour palette ─────────────────────────────────────────────────────────────
C_LEGIT = "#2196F3"   # blue
C_FRAUD = "#F44336"   # red
C_OK    = "#4CAF50"   # green


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_model(name: str):
    p = MODEL_DIR / f"{name}.joblib"
    return joblib.load(p) if p.exists() else None


def _predict(model, df_raw: pd.DataFrame) -> np.ndarray:
    feat_cols = [c for c in df_raw.columns if c not in ID_COLS]
    return model.predict_proba(df_raw[feat_cols])[:, 1]


# ── Pipeline tab ──────────────────────────────────────────────────────────────

def _tab_pipeline() -> None:
    st.header("ML Pipeline")
    st.caption("Run each step in order. Steps are safe to re-run at any time.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("① Generate Data", use_container_width=True):
            with st.spinner("Generating 100,000 synthetic records …"):
                save_tax_records(generate_tax_records())
                save_healthcare_claims(generate_healthcare_claims())
            st.success("tax_records.csv + healthcare_claims.csv saved.")

    with c2:
        if st.button("② Feature Engineering", use_container_width=True):
            raw_tax = ROOT / "data" / "raw" / "tax_records.csv"
            raw_hc  = ROOT / "data" / "raw" / "healthcare_claims.csv"
            if not raw_tax.exists() or not raw_hc.exists():
                st.error("Run Step ① first.")
            else:
                with st.spinner("Engineering features …"):
                    split_tax(build_tax_features(load_tax_records()))
                    split_healthcare(build_healthcare_features(load_healthcare_claims()))
                st.success("train/test CSVs saved to data/processed/.")

    with c3:
        if st.button("③ Train Models", use_container_width=True):
            if not (PROC_DIR / "tax_train.csv").exists():
                st.error("Run Step ② first.")
            else:
                with st.spinner("Training — this takes ~30 s …"):
                    train_model(PROC_DIR / "tax_train.csv",        "tax_fraud_model")
                    train_model(PROC_DIR / "healthcare_train.csv", "healthcare_fraud_model")
                st.success("Both models saved to models/.")

    with c4:
        if st.button("④ Evaluate Models", use_container_width=True):
            if not (MODEL_DIR / "tax_fraud_model.joblib").exists():
                st.error("Run Step ③ first.")
            else:
                with st.spinner("Evaluating …"):
                    evaluate_model("tax_fraud_model",        PROC_DIR / "tax_test.csv")
                    evaluate_model("healthcare_fraud_model", PROC_DIR / "healthcare_test.csv")
                st.success("Evaluation reports saved.")

    # ── Quick KPI cards ────────────────────────────────────────────────────
    st.divider()
    for label, name in [("Tax Fraud Model", "tax_fraud_model"),
                        ("Healthcare Fraud Model", "healthcare_fraud_model")]:
        ev = _load_json(REPORTS / f"{name}_eval_metrics.json")
        if not ev:
            st.info(f"{label}: no evaluation report yet.")
            continue
        st.subheader(label)
        cm = ev.pop("confusion_matrix", {})
        kpi_keys = ["roc_auc", "pr_auc", "precision", "recall", "f1",
                    "test_samples", "fraud_rate"]
        cols = st.columns(len(kpi_keys))
        for col, k in zip(cols, kpi_keys):
            v = ev.get(k, "—")
            col.metric(k.replace("_", " ").upper(),
                       f"{v:.4f}" if isinstance(v, float) else str(v))


# ── Analytics tab ─────────────────────────────────────────────────────────────

def _roc_pr_chart(model_name: str, test_csv: Path, colour: str, label: str):
    model = _load_model(model_name)
    if model is None or not test_csv.exists():
        st.warning(f"Model or test data not found for {label}.")
        return
    df   = pd.read_csv(test_csv)
    feat = [c for c in df.columns if c not in ID_COLS]
    y    = df[TARGET]
    yp   = model.predict_proba(df[feat])[:, 1]

    fpr, tpr, _ = roc_curve(y, yp)
    roc_auc = roc_auc_score(y, yp)
    pr_auc = average_precision_score(y, yp)

    # ROC
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"ROC (AUC={roc_auc:.3f})", line=dict(color=colour, width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="grey"), showlegend=False))
    fig_roc.update_layout(title=f"{label} — ROC Curve",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          height=340, margin=dict(t=40, b=30))
    st.plotly_chart(fig_roc, use_container_width=True)

    # PR
    prec, rec, _ = precision_recall_curve(y, yp)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                name=f"PR (AP={pr_auc:.3f})", line=dict(color=colour, width=2)))
    fig_pr.update_layout(title=f"{label} — Precision-Recall Curve",
                         xaxis_title="Recall", yaxis_title="Precision",
                         height=340, margin=dict(t=40, b=30))
    st.plotly_chart(fig_pr, use_container_width=True)


def _confusion_heatmap(model_name: str):
    ev = _load_json(REPORTS / f"{model_name}_eval_metrics.json")
    if not ev or "confusion_matrix" not in ev:
        return
    cm = ev["confusion_matrix"]
    z  = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = px.imshow(
        z,
        text_auto=True,
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        color_continuous_scale=[[0, "#E3F2FD"], [1, C_FRAUD]],
        title=f"{model_name.replace('_', ' ').title()} — Confusion Matrix",
    )
    fig.update_layout(height=300, margin=dict(t=50, b=10),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def _score_distribution(model_name: str, test_csv: Path, label: str):
    model = _load_model(model_name)
    if model is None or not test_csv.exists():
        return
    df   = pd.read_csv(test_csv)
    feat = [c for c in df.columns if c not in ID_COLS]
    df["fraud_score"] = model.predict_proba(df[feat])[:, 1]
    df["label"] = df[TARGET].map({0: "Legit", 1: "Fraud"})

    fig = px.histogram(
        df, x="fraud_score", color="label",
        color_discrete_map={"Legit": C_LEGIT, "Fraud": C_FRAUD},
        nbins=60, barmode="overlay", opacity=0.70,
        title=f"{label} — Fraud Score Distribution",
        labels={"fraud_score": "Fraud Probability", "count": "Records"},
    )
    fig.update_layout(height=320, margin=dict(t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)


def _feature_importance(model_name: str, test_csv: Path, label: str, top_n: int = 15):
    model = _load_model(model_name)
    if model is None or not test_csv.exists():
        return
    df        = pd.read_csv(test_csv)
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    rf        = model.named_steps["model"]
    imp       = pd.Series(rf.feature_importances_, index=feat_cols)
    imp       = imp.nlargest(top_n).sort_values()

    fig = px.bar(
        imp, x=imp.values, y=imp.index, orientation="h",
        title=f"{label} — Top {top_n} Feature Importances",
        labels={"x": "Importance", "y": "Feature"},
        color=imp.values,
        color_continuous_scale=[[0, "#E3F2FD"], [1, C_FRAUD]],
    )
    fig.update_layout(height=420, margin=dict(t=50, b=30),
                      coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def _tab_analytics() -> None:
    st.header("Analytics & Model Insights")

    datasets = {
        "Tax Fraud Model":        ("tax_fraud_model",        PROC_DIR / "tax_test.csv",         C_OK),
        "Healthcare Fraud Model": ("healthcare_fraud_model", PROC_DIR / "healthcare_test.csv",  C_FRAUD),
    }

    tab_tax, tab_hc = st.tabs(list(datasets.keys()))
    tabs = [tab_tax, tab_hc]

    for tab, (label, (model_name, test_csv, colour)) in zip(tabs, datasets.items()):
        with tab:
            c_left, c_right = st.columns(2)
            with c_left:
                _roc_pr_chart(model_name, test_csv, colour, label)
            with c_right:
                _confusion_heatmap(model_name)
                _score_distribution(model_name, test_csv, label)
            _feature_importance(model_name, test_csv, label)


# ── Predict tab ───────────────────────────────────────────────────────────────

def _show_shap(model, record: pd.DataFrame, feat_cols: list) -> None:
    """Show top-10 SHAP feature contributions for one record."""
    try:
        scaler = model.named_steps["scaler"]
        clf    = model.named_steps["model"]
        X_scaled = scaler.transform(record[feat_cols])
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_scaled)
        # XGBoost returns ndarray (n_samples, n_features); RF returns list[class0, class1]
        vals: np.ndarray = sv[1][0] if isinstance(sv, list) else sv[0]
        shap_s = pd.Series(vals, index=feat_cols)
        top10 = shap_s.reindex(shap_s.abs().nlargest(10).index)[::-1]
        colours = [C_FRAUD if v > 0 else C_LEGIT for v in top10.values]
        fig = go.Figure(go.Bar(
            x=top10.values, y=top10.index, orientation="h",
            marker_color=colours,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(
            title="Top-10 Feature Contributions (SHAP)",
            xaxis_title="SHAP value  (red = pushes toward Fraud, blue = pushes toward Legit)",
            yaxis_title="",
            height=360, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.caption(f"SHAP not available: {exc}")


def _predict_tab_tax(model, threshold: float = 0.50) -> None:
    st.subheader("Tax Filing — Enter Values")
    c1, c2, c3 = st.columns(3)
    with c1:
        reported_income      = st.number_input("Reported Income ($)", 0, 10_000_000, 65_000, step=1000)
        w2_income            = st.number_input("W-2 Income ($)",       0, 10_000_000, 70_000, step=1000)
        se_income            = st.number_input("Self-Employ. Income ($)", 0, 2_000_000, 0, step=500)
        investment_income    = st.number_input("Investment Income ($)", 0, 2_000_000, 0, step=500)
    with c2:
        total_deductions     = st.number_input("Total Deductions ($)",  0, 5_000_000, 12_000, step=500)
        business_deductions  = st.number_input("Business Deductions ($)", 0, 2_000_000, 5_000, step=500)
        charitable_deductions= st.number_input("Charitable Deductions ($)", 0, 500_000, 1_000, step=200)
        medical_deductions   = st.number_input("Medical Deductions ($)", 0, 500_000, 2_000, step=200)
    with c3:
        refund_amount        = st.number_input("Claimed Refund ($)",  0, 500_000, 1_500, step=100)
        claimed_dependents   = st.slider("Claimed Dependents", 0, 6, 1)
        num_prior_audits     = st.slider("Prior Audits", 0, 3, 0)
        num_amendments       = st.slider("Prior Amendments", 0, 3, 0)
        days_to_file         = st.slider("Days to File", 1, 289, 90)
        taxpayer_age         = st.slider("Taxpayer Age", 18, 84, 40)
        filing_year          = st.selectbox("Filing Year", [2020,2021,2022,2023,2024,2025], index=4)
        occ_risk             = st.slider("Occupation Risk Score", 0.0, 1.0, 0.20, step=0.01)
        round_refund         = int(refund_amount % 500 < 10)

    inc = max(reported_income, 1)
    record = pd.DataFrame([{
        "filing_year": filing_year, "taxpayer_age": taxpayer_age,
        "days_to_file": days_to_file,
        "w2_income": w2_income, "self_employment_income": se_income,
        "investment_income": investment_income, "reported_income": reported_income,
        "business_deductions": business_deductions,
        "charitable_deductions": charitable_deductions,
        "medical_deductions": medical_deductions,
        "total_deductions": total_deductions,
        "claimed_dependents": claimed_dependents,
        "refund_amount": refund_amount, "round_number_refund": round_refund,
        "num_prior_audits": num_prior_audits, "num_amendments": num_amendments,
        "occupation_risk_score": occ_risk,
        "deduction_to_income_ratio": total_deductions / inc,
        "refund_to_income_ratio":   refund_amount / inc,
        "se_income_share":          se_income / inc,
        "income_gap":               max(w2_income + se_income - reported_income, 0),
        "income_gap_ratio":         max(w2_income + se_income - reported_income, 0) / inc,
        "high_charitable_deduction": int(charitable_deductions / inc > 0.10),
        "late_filer":               int(days_to_file > 240),
        "multiple_amendments":      int(num_amendments >= 2),
        "repeat_audit_history":     int(num_prior_audits >= 2),
        "filing_status_code":       1,   # default married_jointly
    }])

    feat_cols = list(record.columns)
    score = model.predict_proba(record)[:, 1][0]
    _show_score(score, "Tax Filing", threshold)
    _show_shap(model, record, feat_cols)


def _predict_tab_hc(model, threshold: float = 0.50) -> None:
    st.subheader("Healthcare Claim — Enter Values")
    c1, c2, c3 = st.columns(3)
    with c1:
        claim_amount         = st.number_input("Claim Amount ($)",   0, 200_000, 2_500, step=100)
        allowed_amount       = st.number_input("Allowed Amount ($)", 0, 200_000, 1_800, step=100)
        upcoding_score       = st.slider("Upcoding Score",    0.0, 1.0, 0.10, step=0.01)
        unbundling_score     = st.slider("Unbundling Score",  0.0, 1.0, 0.05, step=0.01)
    with c2:
        procedure_code_count = st.slider("# Procedure Codes",  1, 8, 2)
        diagnosis_code_count = st.slider("# Diagnosis Codes",  1, 12, 3)
        procedure_risk_score = st.slider("Procedure Risk Score", 0.0, 1.0, 0.30, step=0.01)
        diagnosis_complexity = st.slider("Diagnosis Complexity", 0.0, 1.0, 0.40, step=0.01)
    with c3:
        patient_age          = st.slider("Patient Age",         0, 94, 55)
        provider_tenure      = st.slider("Provider Tenure (yrs)", 0, 40, 10)
        claim_velocity       = st.slider("Provider Claim Velocity", 1, 60, 12)
        claim_hour           = st.slider("Claim Hour (0-23)",   0, 23, 10)
        is_weekend           = st.checkbox("Weekend Claim")
        days_since           = st.slider("Days Since Last Claim", 0, 180, 30)
        prior_claims         = st.slider("Prior Claims YTD",    0, 30, 4)
        duplicate_flag       = st.checkbox("Duplicate Claim Flag")
        spc_risk             = st.slider("Specialty Risk Score", 0.0, 1.0, 0.40, step=0.01)

    record = pd.DataFrame([{
        "patient_age": patient_age,
        "provider_specialty_risk": spc_risk,
        "provider_tenure_years": provider_tenure,
        "provider_claim_velocity": claim_velocity,
        "claim_amount": claim_amount,
        "allowed_amount": allowed_amount,
        "diagnosis_code_count": diagnosis_code_count,
        "procedure_code_count": procedure_code_count,
        "procedure_risk_score": procedure_risk_score,
        "diagnosis_complexity": diagnosis_complexity,
        "claim_hour": claim_hour,
        "is_weekend": int(is_weekend),
        "days_since_last_claim": days_since,
        "num_prior_claims_ytd": prior_claims,
        "duplicate_claim_flag": int(duplicate_flag),
        "unbundling_score": unbundling_score,
        "upcoding_score": upcoding_score,
        "amount_to_allowed_ratio":   claim_amount / max(allowed_amount, 1),
        "amount_per_procedure":      claim_amount / max(procedure_code_count, 1),
        "diagnosis_procedure_ratio": diagnosis_code_count / max(procedure_code_count, 1),
        "night_claim":               int(claim_hour <= 5 or claim_hour >= 22),
        "rapid_reclaim":             int(days_since <= 3),
        "high_velocity_provider":    int(claim_velocity >= 20),
        "velocity_x_amount":         claim_velocity * claim_amount,
        "elderly_patient":           int(patient_age >= 70),
        "gender_code":               1,
    }])

    feat_cols = list(record.columns)
    score = model.predict_proba(record)[:, 1][0]
    _show_score(score, "Healthcare Claim", threshold)
    _show_shap(model, record, feat_cols)


def _show_score(score: float, domain: str, threshold: float = 0.50) -> None:
    st.divider()
    colour = C_FRAUD if score >= threshold else C_OK
    verdict = "⚠️ HIGH FRAUD RISK" if score >= threshold else "✅ LOW FRAUD RISK"
    st.markdown(
        f"<h2 style='color:{colour};text-align:center'>{verdict}</h2>"
        f"<h3 style='text-align:center'>Fraud Probability: {score:.1%}</h3>",
        unsafe_allow_html=True,
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": colour},
            "steps": [
                {"range": [0,  40], "color": "#E8F5E9"},
                {"range": [40, 60], "color": "#FFF9C4"},
                {"range": [60, 100],"color": "#FFEBEE"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
        },
        title={"text": f"{domain} Fraud Score"},
    ))
    fig.update_layout(height=300, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _tab_predict() -> None:
    st.header("Live Fraud Prediction")
    st.caption("Adjust sliders and inputs — the gauge updates on every change.")

    tax_model = _load_model("tax_fraud_model")
    hc_model  = _load_model("healthcare_fraud_model")

    if tax_model is None or hc_model is None:
        st.warning("Run the full pipeline (Pipeline tab) before using predictions.")
        return

    threshold = st.slider(
        "Decision threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower = catch more fraud (more false alarms). Higher = only flag very likely fraud.",
    )

    tab_tax, tab_hc = st.tabs(["Tax Filing", "Healthcare Claim"])
    with tab_tax:
        _predict_tab_tax(tax_model, threshold)
    with tab_hc:
        _predict_tab_hc(hc_model, threshold)


# ── App entry point ───────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="AI Fraud Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("🔍 AI Tax + Healthcare Fraud Detection")
    st.caption("50,000 tax records · 50,000 healthcare claims · XGBoost · Plotly analytics")

    pipeline_tab, analytics_tab, predict_tab = st.tabs(
        ["🔧 Pipeline", "📊 Analytics", "🎯 Predict"]
    )

    with pipeline_tab:
        _tab_pipeline()
    with analytics_tab:
        _tab_analytics()
    with predict_tab:
        _tab_predict()


if __name__ == "__main__":
    main()
