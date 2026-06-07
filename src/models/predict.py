"""
Prediction module.

This module is the inference path: given raw input data, run it through the
preprocessing pipeline and return a prediction with a confidence score.

It is used by:
  - The FastAPI endpoint (real-time single-record prediction)
  - The Streamlit frontend (via the API)

Design: load model and pipeline once on startup, reuse for every request.
This avoids the 200ms+ overhead of loading a model file on every request.
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

# Module-level cache: models are loaded once when this module is imported.
# The dict maps domain name → (model, pipeline, feature_names).
_model_cache: dict[str, tuple] = {}


def load_model_and_pipeline(domain: str) -> tuple:
    """
    Load (or retrieve from cache) the model and preprocessing pipeline for a domain.

    Why cache?
      Loading a joblib file involves disk I/O and deserialization — typically 50-200ms.
      For an API serving hundreds of requests per second, doing this per-request would
      be unacceptable. The cache loads once and serves all subsequent requests instantly.

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
        # For network intrusion, feature names include one-hot expanded categoricals.
        # We return the pre-encoding names here; the pipeline handles expansion.
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
                    Example: {"age": 35, "DebtRatio": 0.21, ...}

    Returns:
        {
          "prediction": 0 or 1,
          "probability": float (0.0 to 1.0),
          "risk_label": "Low Risk" / "High Risk",
          "domain": str
        }
    """
    model, pipeline, feature_names = load_model_and_pipeline(domain)

    # Convert the input dict to a single-row DataFrame.
    # The pipeline expects a DataFrame with named columns — not a raw array.
    input_df = pd.DataFrame([input_data])

    # Ensure all expected feature columns are present.
    # Fill missing features with NaN — the imputer in the pipeline handles them.
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # Keep only the feature columns in the correct order
    input_df = input_df[feature_names]

    # Apply the preprocessing pipeline (same transforms as during training)
    X_processed = pipeline.transform(input_df)

    # Get probability of positive class (default=1 or attack=1)
    probability = float(model.predict_proba(X_processed)[0, 1])

    # Load optimal threshold from evaluation report if available
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
    """
    Read the optimal decision threshold from the saved evaluation report.
    Falls back to 0.5 if the report doesn't exist yet.
    """
    report_path = PROJECT_ROOT / "models" / domain / "evaluation_report.json"
    if report_path.exists():
        import json
        with open(report_path) as f:
            report = json.load(f)
        return report.get("optimal_threshold", 0.5)
    return 0.5
