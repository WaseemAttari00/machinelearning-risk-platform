"""
Streamlit Dashboard — Intelligent Risk Analytics Platform.

What this dashboard provides:
  1. Interactive prediction forms for both domains
  2. Model performance metrics and evaluation charts
  3. SHAP explainability visualizations
  4. Side-by-side model comparison

Why Streamlit?
  Streamlit lets you build a data app in pure Python — no HTML/CSS/JavaScript needed.
  Every time the user changes a widget (slider, dropdown), Streamlit re-runs the
  script from top to bottom and re-renders the page.
  This "reactive" model is perfect for ML demos: you adjust an input and instantly
  see the prediction change.

  For a portfolio project, Streamlit demonstrates:
    - You can build end-to-end applications (not just notebooks)
    - You understand how to present model outputs to non-technical users
    - You can connect a frontend to a backend API

Architecture:
  This Streamlit app calls the FastAPI backend at API_URL.
  It does NOT directly import the model — that's intentional.
  In a real deployment, the frontend and backend are separate services.
"""

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
# In Docker, this should be the API container's address.
# Locally, it points to localhost.
API_URL = "http://localhost:8000"

# Path to saved evaluation reports (for metrics display)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Risk Analytics Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "💳 Credit Risk Prediction",
        "🌐 Network Intrusion Detection",
        "📊 Model Performance",
        "🔍 SHAP Explainability",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tech Stack**: XGBoost · SHAP · MLflow · FastAPI · Streamlit · Docker"
)


# ── Helper functions ──────────────────────────────────────────────────────────

def check_api_health() -> dict:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "unavailable", "models_loaded": []}


def load_evaluation_report(domain: str) -> dict | None:
    """Load evaluation report JSON from disk (or return None if not found)."""
    report_path = PROJECT_ROOT / "models" / domain / "evaluation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return None


def predict(endpoint: str, payload: dict) -> dict | None:
    """Call the prediction API and return the result."""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict/{endpoint}",
            json=payload,
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. Make sure it's running:\n"
            "`uvicorn api.main:app --reload --port 8000`"
        )
        return None


# ── API Status Banner ─────────────────────────────────────────────────────────
health = check_api_health()
if health["status"] == "healthy":
    models_ready = health.get("models_ready", False)
    if models_ready:
        st.sidebar.success("✅ API Online · Models Ready")
    else:
        st.sidebar.warning("⚠️ API Online · Models Not Trained")
else:
    st.sidebar.error("❌ API Offline")


# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("🛡️ Intelligent Risk Analytics Platform")
    st.markdown(
        """
        An end-to-end machine learning platform demonstrating the complete ML lifecycle:
        from raw data to deployed predictions with full explainability.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💳 Credit Risk")
        st.markdown(
            """
            **Dataset**: Give Me Some Credit (Kaggle)
            **Task**: Predict probability of loan default
            **Model**: XGBoost with Optuna hyperparameter tuning
            **Key Challenge**: Severe class imbalance (~7% positive)
            """
        )

    with col2:
        st.subheader("🌐 Network Intrusion")
        st.markdown(
            """
            **Dataset**: NSL-KDD
            **Task**: Detect malicious vs. benign network traffic
            **Model**: XGBoost with Optuna hyperparameter tuning
            **Key Challenge**: Mixed numeric + categorical features, multi-type attacks
            """
        )

    st.markdown("---")
    st.subheader("ML Pipeline")
    st.markdown(
        """
        ```
        Raw Data → Validation → Feature Engineering → Baseline Model
             ↓                                               ↓
        Optuna Tuning → XGBoost → Evaluation → SHAP Explainability
             ↓
        MLflow Tracking → FastAPI → Streamlit (this dashboard)
        ```
        """
    )


elif page == "💳 Credit Risk Prediction":
    st.title("💳 Credit Risk Prediction")
    st.markdown(
        "Adjust the applicant's financial features to predict their default probability."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financial Features")
        revolving_util = st.slider(
            "Revolving Utilization of Unsecured Lines",
            min_value=0.0, max_value=1.5, value=0.3, step=0.01,
            help="Total credit card balance / total credit limit. Above 1.0 means over-limit.",
        )
        age = st.slider("Age", min_value=18, max_value=100, value=45)
        debt_ratio = st.slider(
            "Debt Ratio",
            min_value=0.0, max_value=5.0, value=0.35, step=0.01,
            help="Monthly debt payments / gross monthly income.",
        )
        monthly_income = st.number_input(
            "Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0
        )

    with col2:
        st.subheader("Credit History")
        times_30_59 = st.slider("Times 30-59 Days Past Due", 0, 20, 0)
        times_60_89 = st.slider("Times 60-89 Days Past Due", 0, 20, 0)
        times_90 = st.slider("Times 90+ Days Late", 0, 20, 0)
        open_credit = st.slider("Open Credit Lines & Loans", 0, 50, 8)
        real_estate_loans = st.slider("Real Estate Loans or Lines", 0, 20, 1)
        dependents = st.slider("Number of Dependents", 0, 10, 0)

    if st.button("🔮 Predict Default Risk", type="primary"):
        payload = {
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": times_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit,
            "NumberOfTimes90DaysLate": times_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": times_60_89,
            "NumberOfDependents": float(dependents),
        }

        with st.spinner("Running prediction..."):
            result = predict("credit-risk", payload)

        if result:
            st.markdown("---")
            col_pred, col_prob = st.columns(2)

            with col_pred:
                if result["prediction"] == 1:
                    st.error(f"### ⚠️ {result['risk_label']}")
                else:
                    st.success(f"### ✅ {result['risk_label']}")

            with col_prob:
                prob_pct = result["probability"] * 100
                st.metric(
                    label="Default Probability",
                    value=f"{prob_pct:.1f}%",
                    delta=f"Threshold: {result['threshold_used']}",
                )

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Default Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 60], "color": "yellow"},
                        {"range": [60, 100], "color": "lightcoral"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": result["threshold_used"] * 100,
                    },
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


elif page == "🌐 Network Intrusion Detection":
    st.title("🌐 Network Intrusion Detection")
    st.markdown("Analyze network traffic features to detect potential intrusions.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Connection Properties")
        protocol_type = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
        service = st.selectbox(
            "Service",
            ["http", "ftp", "smtp", "ssh", "telnet", "ftp_data", "domain_u", "other"],
        )
        flag = st.selectbox("Connection Flag", ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1"])
        duration = st.number_input("Duration (seconds)", min_value=0.0, value=0.0)
        logged_in = st.selectbox("Logged In", [1, 0], index=0)

    with col2:
        st.subheader("Traffic Volume")
        src_bytes = st.number_input("Source Bytes", min_value=0.0, value=215.0, step=10.0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0.0, value=45076.0, step=100.0)
        count = st.slider("Same-host Connections (2s window)", 0.0, 512.0, 1.0)
        srv_count = st.slider("Same-service Connections (2s window)", 0.0, 512.0, 1.0)
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0, step=0.01)
        serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0, step=0.01)

    if st.button("🔮 Analyze Traffic", type="primary"):
        payload = {
            "duration": duration,
            "protocol_type": protocol_type,
            "service": service,
            "flag": flag,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "logged_in": logged_in,
            "count": count,
            "srv_count": srv_count,
            "same_srv_rate": same_srv_rate,
            "diff_srv_rate": 0.0,
            "dst_host_count": 9.0,
            "dst_host_srv_count": 9.0,
            "serror_rate": serror_rate,
            "rerror_rate": 0.0,
        }

        with st.spinner("Analyzing traffic..."):
            result = predict("network-intrusion", payload)

        if result:
            st.markdown("---")
            if result["prediction"] == 1:
                st.error(f"### 🚨 {result['risk_label']}")
                st.metric("Attack Probability", f"{result['probability']*100:.1f}%")
            else:
                st.success(f"### ✅ {result['risk_label']}")
                st.metric("Benign Probability", f"{(1-result['probability'])*100:.1f}%")


elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    domain = st.selectbox("Select Domain", ["credit_risk", "network_intrusion"])
    report = load_evaluation_report(domain)

    if report is None:
        st.warning(
            f"No evaluation report found for '{domain}'.\n\n"
            f"Run training first:\n```\npython -m src.models.train --domain {domain}\n```"
        )
    else:
        metrics = report["scalar_metrics"]

        st.subheader("Scalar Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")
        col4.metric("F1 Score", f"{metrics['f1']:.4f}")
        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

        st.markdown("---")

        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            cm = report["confusion_matrix"]
            labels = ["Negative", "Positive"]
            fig = px.imshow(
                cm,
                x=["Predicted Neg", "Predicted Pos"],
                y=["Actual Neg", "Actual Pos"],
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_fi:
            st.subheader("Top Feature Importances")
            fi = report.get("feature_importance", {})
            if fi:
                top_features = dict(list(fi.items())[:15])
                fig = px.bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation="h",
                    title="Feature Importance (XGBoost)",
                    labels={"x": "Importance Score", "y": "Feature"},
                )
                fig.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Classification Report")
        st.code(report["classification_report"])
        st.info(
            f"**Optimal Threshold**: {metrics.get('optimal_threshold', 'N/A')} "
            f"(F1 = {metrics.get('optimal_f1', 'N/A')})"
        )


elif page == "🔍 SHAP Explainability":
    st.title("🔍 SHAP Feature Explainability")
    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) explains how each feature contributes
        to individual predictions. The beeswarm plot shows:
        - **X-axis**: SHAP value (positive = pushes toward risky, negative = pushes toward safe)
        - **Color**: Red = high feature value, Blue = low feature value
        - **Row order**: Features sorted by mean absolute SHAP value (most important at top)
        """
    )

    domain = st.selectbox("Select Domain", ["credit_risk", "network_intrusion"])

    shap_path = PROJECT_ROOT / "models" / domain / "shap_summary.png"
    if shap_path.exists():
        st.subheader("SHAP Summary Plot (Global Feature Importance)")
        st.image(str(shap_path), use_column_width=True)

        # Show individual waterfall plots
        st.subheader("SHAP Waterfall Plots (Local Explanations)")
        st.markdown(
            "Each waterfall plot explains a single prediction: "
            "how much each feature pushed the model's output up or down."
        )
        for i in range(3):
            waterfall_path = PROJECT_ROOT / "models" / domain / f"shap_waterfall_{i}.png"
            if waterfall_path.exists():
                st.image(str(waterfall_path), caption=f"Sample {i} Explanation")
    else:
        st.warning(
            f"SHAP plots not found for '{domain}'.\n\n"
            "Run the SHAP analysis after training:\n"
            "```python\n"
            "from src.explainability.shap_analysis import run_shap_analysis\n"
            "# (called automatically from train.py after training)\n"
            "```"
        )
