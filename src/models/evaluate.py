"""
Model evaluation module.

Computes a comprehensive suite of metrics for binary classification:
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - Confusion matrix
  - Classification report (per-class precision/recall/F1)
  - Optimal threshold search (best F1 threshold)

All results are returned as a structured dictionary so they can be:
  - Logged to MLflow
  - Saved as a JSON report
  - Displayed in Streamlit

Why so many metrics?
  No single metric tells the whole story for imbalanced binary classification.
  A model that always predicts "no default" achieves ~93% accuracy on the credit
  risk dataset — but catches 0 actual defaults. You need Recall to catch this.
  Precision and F1 balance the tradeoff between catching true positives and
  avoiding false positives.
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
        model: Any fitted sklearn-compatible model with predict_proba().
        X_test: Preprocessed test feature matrix.
        y_test: True binary labels (0/1).
        feature_names: Column names for X_test (used in reports).
        domain: "credit_risk" or "network_intrusion".
        model_name: Human-readable name for logging and reporting.
        threshold: Decision threshold (default 0.5).
                   Predictions above this → positive class.

    Returns:
        Dictionary with:
          - scalar_metrics: flat dict suitable for MLflow.log_metrics()
          - confusion_matrix: 2×2 list [[TN, FP], [FN, TP]]
          - classification_report: full sklearn report as string
          - optimal_threshold: threshold that maximizes F1 on test set
          - feature_importance: dict of {feature_name: importance_score} if available
    """
    # Get probability scores for the positive class (class 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Apply the threshold to get binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # ── Core metrics ──────────────────────────────────────────────────────────
    # zero_division=0 means if a class has no predictions, return 0 instead of warning
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

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    # cm[0][0]=TN, cm[0][1]=FP, cm[1][0]=FN, cm[1][1]=TP
    tn, fp, fn, tp = cm.ravel()
    logger.info(
        "Confusion matrix — TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        tn=tn, fp=fp, fn=fn, tp=tp,
    )

    # ── Optimal threshold search ──────────────────────────────────────────────
    # The default 0.5 threshold is rarely optimal for imbalanced problems.
    # We sweep thresholds and find the one that maximizes F1 score.
    optimal_threshold, optimal_f1 = _find_optimal_threshold(y_test, y_prob)
    scalar_metrics["optimal_threshold"] = round(float(optimal_threshold), 3)
    scalar_metrics["optimal_f1"] = round(float(optimal_f1), 4)
    logger.info(
        "Optimal threshold: {t:.3f} (F1={f:.4f})",
        t=optimal_threshold,
        f=optimal_f1,
    )

    # ── Feature importance (tree models only) ─────────────────────────────────
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            feature_importance = dict(zip(feature_names, importances.tolist()))
            # Sort descending for easy reading
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

    # ── Full classification report ────────────────────────────────────────────
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
    Find the decision threshold that maximizes F1 score.

    We sweep from 0.05 to 0.95 in steps of 0.01 and pick the threshold
    with the highest F1. In production you might optimize for different
    metrics depending on business requirements (e.g., maximize recall
    if missing a fraud is worse than a false alarm).

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
    """
    Save the evaluation report as a JSON file in models/<domain>/.

    JSON is human-readable and easily loaded by Streamlit for display.
    """
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
