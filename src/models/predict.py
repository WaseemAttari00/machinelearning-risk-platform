"""
Inference module — load trained models and run predictions on single records.

Models and pipelines are loaded once at module import time and cached for the
lifetime of the process to avoid per-request disk I/O overhead.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = get_project_root()

_model_cache: dict[str, tuple] = {}


def load_model_and_pipeline(domain: str) -> tuple:
    """
    Load (or retrieve from cache) the model and preprocessing pipeline for a domain.

    Args:
        domain: "credit_risk" or "network_intrusion"

    Returns:
        (model, pipeline, feature_names)
    """
    if domain in _model_cache:
        return _model_cache[domain]

    model_dir = PROJECT_ROOT / "models" / domain
    model_path = model_dir / "xgboost_model.joblib"
    pipeline_path = model_dir / "preprocessing_pipeline.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            f"Run training first: python -m src.models.train --domain {domain}"
        )
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline not found at {pipeline_path}.\n"
            f"Run training first: python -m src.models.train --domain {domain}"
        )

    logger.info("Loading model from {path}", path=str(model_path))
    model = joblib.load(model_path)

    logger.info("Loading pipeline from {path}", path=str(pipeline_path))
    pipeline = joblib.load(pipeline_path)

    cfg = load_config(domain)
    if domain == "credit_risk":
        feature_names = cfg["features"]["numeric_features"]
    else:
        feature_names = (
            cfg["features"]["numeric_features"] +
            cfg["features"]["categorical_features"]
        )

    _model_cache[domain] = (model, pipeline, feature_names)
    logger.info("Model and pipeline cached for domain: {domain}", domain=domain)
    return _model_cache[domain]


def predict_single(domain: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Run inference on a single input record.

    Args:
        domain: "credit_risk" or "network_intrusion"
        input_data: Dict mapping feature names to their values.

    Returns:
        Dict with prediction (0/1), probability, risk_label, domain, threshold_used.
    """
    model, pipeline, feature_names = load_model_and_pipeline(domain)

    input_df = pd.DataFrame([input_data])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[feature_names]

    X_processed = pipeline.transform(input_df)
    probability = float(model.predict_proba(X_processed)[0, 1])

    threshold = _get_optimal_threshold(domain)
    prediction = int(probability >= threshold)

    if domain == "credit_risk":
        risk_label = "High Risk (Likely Default)" if prediction == 1 else "Low Risk (Likely No Default)"
    else:
        risk_label = "Attack / Malicious" if prediction == 1 else "Normal / Benign"

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_label": risk_label,
        "domain": domain,
        "threshold_used": round(threshold, 3),
    }


def _get_optimal_threshold(domain: str) -> float:
    """Read the optimal decision threshold from the evaluation report. Defaults to 0.5."""
    report_path = PROJECT_ROOT / "models" / domain / "evaluation_report.json"
    if report_path.exists():
        import json
        with open(report_path) as f:
            report = json.load(f)
        return report.get("optimal_threshold", 0.5)
    return 0.5
