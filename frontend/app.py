"""Streamlit dashboard for the Intelligent Risk Analytics Platform."""

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


st.set_page_config(
    page_title="Risk Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def check_api_health() -> dict:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "unavailable", "models_loaded": []}


def load_evaluation_report(domain: str) -> dict | None:
    """Load evaluation report JSON from disk."""
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


health = check_api_health()
if health["status"] == "healthy":
    models_ready = health.get("models_ready", False)
    if models_ready:
        st.sidebar.success("✅ API Online · Models Ready")
    else:
        st.sidebar.warning("⚠️ API Online · Models Not Trained")
else:
    st.sidebar.error("❌ API Offline")


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
            **Dataset**: UCI Default of Credit Card Clients
            **Task**: Predict probability of credit card default
            **Model**: XGBoost with Optuna hyperparameter tuning
            **Key Challenge**: Class imbalance (~22% positive class)
            """
        )

    with col2:
        st.subheader("🌐 Network Intrusion")
        st.markdown(
            """
            **Dataset**: NSL-KDD
            **Task**: Detect malicious vs. benign network traffic
            **Model**: XGBoost with Optuna hyperparameter tuning
            **Key Challenge**: Mixed numeric + categorical features, novel attack types in test set
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
        st.subheader("Account Details")
        limit_bal = st.number_input("Credit Limit (NTD)", min_value=0.0, value=200000.0, step=10000.0)
        age = st.slider("Age", min_value=18, max_value=100, value=35)
        sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox(
            "Education",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Other"}[x],
        )
        marriage = st.selectbox(
            "Marital Status",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
        )

    with col2:
        st.subheader("Repayment History")
        pay_0 = st.slider("Repayment Status (Sep)", min_value=-2, max_value=9, value=-1,
                          help="-2=no consumption, -1=paid duly, 1+ = months delayed")
        pay_2 = st.slider("Repayment Status (Aug)", min_value=-2, max_value=9, value=-1)
        pay_3 = st.slider("Repayment Status (Jul)", min_value=-2, max_value=9, value=-1)
        bill_amt1 = st.number_input("Bill Amount Sep (NTD)", min_value=0.0, value=3913.0, step=100.0)
        pay_amt1 = st.number_input("Payment Amount Sep (NTD)", min_value=0.0, value=0.0, step=100.0)

    if st.button("🔮 Predict Default Risk", type="primary"):
        payload = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            "PAY_0": pay_0,
            "PAY_2": pay_2,
            "PAY_3": pay_3,
            "PAY_4": -1,
            "PAY_5": -1,
            "PAY_6": -1,
            "BILL_AMT1": bill_amt1,
            "BILL_AMT2": 3102.0,
            "BILL_AMT3": 689.0,
            "BILL_AMT4": 0.0,
            "BILL_AMT5": 0.0,
            "BILL_AMT6": 0.0,
            "PAY_AMT1": pay_amt1,
            "PAY_AMT2": 689.0,
            "PAY_AMT3": 0.0,
            "PAY_AMT4": 0.0,
            "PAY_AMT5": 0.0,
            "PAY_AMT6": 0.0,
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
            "Run training first:\n"
            "```\npython -m src.models.train --domain {domain}\n```"
        )
