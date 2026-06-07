"""
Model evaluation for binary classification.

Computes accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
classification report, optimal decision threshold, and feature importances.
All results are returned as a structured dict suitable for MLflow logging,
JSON persistence, and Streamlit display.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = get_project_root()


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    domain: str,
    model_name: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute the full evaluation suite for a binary classifier.

    Args:
        model: Fitted sklearn-compatible model with predict_proba().
        X_test: Preprocessed test feature matrix.
        y_test: True binary labels (0/1).
        feature_names: Column names for X_test.
        domain: "credit_risk" or "network_intrusion".
        model_name: Human-readable identifier for logging.
        threshold: Decision threshold (default 0.5).

    Returns:
        Dict with scalar_metrics, confusion_matrix, classification_report,
        optimal_threshold, feature_importance, and sample counts.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    scalar_metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "threshold": threshold,
    }

    logger.info(
        "{model} evaluation — Acc={acc:.4f} | Prec={p:.4f} | Rec={r:.4f} | F1={f:.4f} | AUC={auc:.4f}",
        model=model_name,
        acc=accuracy,
        p=precision,
        r=recall,
        f=f1,
        auc=roc_auc,
    )

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    logger.info(
        "Confusion matrix — TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        tn=tn, fp=fp, fn=fn, tp=tp,
    )

    optimal_threshold, optimal_f1 = _find_optimal_threshold(y_test, y_prob)
    scalar_metrics["optimal_threshold"] = round(float(optimal_threshold), 3)
    scalar_metrics["optimal_f1"] = round(float(optimal_f1), 4)
    logger.info(
        "Optimal threshold: {t:.3f} (F1={f:.4f})",
        t=optimal_threshold,
        f=optimal_f1,
    )

    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            feature_importance = dict(zip(feature_names, importances.tolist()))
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

    cls_report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "domain": domain,
        "model_name": model_name,
        "scalar_metrics": scalar_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "optimal_threshold": optimal_threshold,
        "feature_importance": feature_importance,
        "n_test_samples": len(y_test),
        "n_positive": int(y_test.sum()),
    }


def _find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = None,
) -> tuple[float, float]:
    """
    Sweep thresholds from 0.05 to 0.95 and return the one that maximizes F1.

    Returns:
        (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    best_threshold = 0.5
    best_f1 = 0.0

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def save_evaluation_report(metrics: dict[str, Any], domain: str) -> None:
    """Save the evaluation report as JSON to models/<domain>/evaluation_report.json."""
    report_path = PROJECT_ROOT / "models" / domain / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation report saved to {path}", path=str(report_path))


def load_evaluation_report(domain: str) -> dict[str, Any]:
    """Load a previously saved evaluation report."""
    report_path = PROJECT_ROOT / "models" / domain / "evaluation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found for domain '{domain}'.\n"
            f"Run training first: python -m src.models.train --domain {domain}"
        )
    with open(report_path) as f:
        return json.load(f)
