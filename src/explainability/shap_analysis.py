"""
SHAP (SHapley Additive exPlanations) analysis module.

What is SHAP?
  SHAP is a method for explaining the output of any machine learning model.
  It answers the question: "How much did each feature contribute to THIS prediction?"

  The math behind it comes from game theory — Shapley values fairly distribute the
  "credit" (prediction contribution) among all features by considering every possible
  combination of features.

Why use SHAP instead of just feature importances?
  Built-in feature importances (model.feature_importances_) tell you which features
  are most used across all predictions, but they:
    - Don't tell you the direction of the effect (does higher age → more risk or less?)
    - Don't tell you how each feature affected a specific prediction
    - Can be misleading for correlated features

  SHAP gives you:
    - Global importance: which features matter most overall
    - Direction: does high DebtRatio push predictions up or down?
    - Local explanations: why did THIS specific customer get a high-risk prediction?

Why does this matter for interviews?
  "Explainability" is now a compliance requirement in finance (GDPR Article 22) and
  healthcare. Being able to explain a model's decision to a non-technical stakeholder
  is a real job skill. Interviewers will ask about this.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — required for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = get_project_root()


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    sample_size: Optional[int] = 1000,
) -> shap.Explanation:
    """
    Compute SHAP values for a dataset using TreeExplainer.

    Args:
        model: A fitted tree-based model (XGBoost, Random Forest, etc.)
        X: Preprocessed feature matrix (numpy array).
        feature_names: Column names for the features.
        sample_size: Subsample size for speed. SHAP is O(n) but 1000 samples
                     is enough for reliable global summaries.

    Returns:
        shap.Explanation object containing SHAP values and base values.

    Why TreeExplainer?
      SHAP has model-specific explainers:
        - TreeExplainer  : for tree-based models (XGBoost, LightGBM, RF)
        - LinearExplainer: for linear models (Logistic Regression)
        - KernelExplainer: model-agnostic (much slower)

      TreeExplainer is ~100x faster than KernelExplainer because it exploits
      the tree structure to compute exact SHAP values efficiently.
      Since we use XGBoost, we always use TreeExplainer.
    """
    if sample_size and len(X) > sample_size:
        # Reproducible random subsample
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X[idx]
        logger.info(
            "Subsampled {n} rows for SHAP (from {total})",
            n=sample_size,
            total=len(X),
        )
    else:
        X_sample = X

    logger.info("Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Attach feature names so plots are labeled correctly
    if hasattr(shap_values, "feature_names"):
        shap_values.feature_names = feature_names

    logger.info(
        "SHAP values computed — shape: {shape}",
        shape=shap_values.values.shape,
    )
    return shap_values, X_sample


def plot_summary(
    shap_values: shap.Explanation,
    X_sample: np.ndarray,
    feature_names: list[str],
    domain: str,
    max_display: int = 15,
    save: bool = True,
) -> plt.Figure:
    """
    Generate a SHAP beeswarm summary plot.

    What does the summary plot show?
      - Each row is one feature
      - Each dot is one prediction (one row of data)
      - X-axis: SHAP value (positive = pushes prediction toward class 1,
                            negative = pushes toward class 0)
      - Color: red = high feature value, blue = low feature value
      - Features are sorted top-to-bottom by mean absolute SHAP value

    How to read it:
      "High RevolvingUtilization (red dots) pushes predictions toward default (positive SHAP).
       Low RevolvingUtilization (blue dots) reduces default probability."

    Args:
        shap_values: SHAP Explanation object from compute_shap_values.
        X_sample: The data subset used for SHAP computation.
        feature_names: Feature column names.
        domain: For save path and title.
        max_display: Number of top features to show.
        save: If True, save the plot to models/<domain>/shap_summary.png.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values.values,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )

    plt.title(f"SHAP Feature Importance — {domain.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout()

    if save:
        output_path = PROJECT_ROOT / "models" / domain / "shap_summary.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved to {path}", path=str(output_path))

    return fig


def plot_waterfall(
    model,
    X_single: np.ndarray,
    feature_names: list[str],
    domain: str,
    sample_index: int = 0,
    save: bool = True,
) -> plt.Figure:
    """
    Generate a SHAP waterfall plot for a single prediction.

    What does the waterfall plot show?
      It explains one specific prediction by showing:
        - The baseline value (average model output across all training data)
        - Each feature's contribution: how much it pushed the prediction up or down
        - The final prediction value

    This is the "local explanation" — useful for explaining to a loan officer
    WHY the model classified a specific applicant as high risk.

    Args:
        model: Fitted model.
        X_single: Single-row feature matrix (1 × n_features array).
        feature_names: Feature names.
        domain: For save path.
        sample_index: Index label for the saved filename.
        save: If True, save the plot.

    Returns:
        matplotlib Figure object.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_single)

    # Attach feature names
    shap_values.feature_names = feature_names

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_values[0], max_display=15, show=False)
    plt.title(f"SHAP Explanation — Sample {sample_index}", fontsize=13)
    plt.tight_layout()

    if save:
        output_path = PROJECT_ROOT / "models" / domain / f"shap_waterfall_{sample_index}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Waterfall plot saved to {path}", path=str(output_path))

    return fig


def run_shap_analysis(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    domain: str,
    sample_size: int = 1000,
) -> None:
    """
    Run the full SHAP analysis pipeline: compute values + generate all plots.

    Called from train.py after model training is complete.
    """
    logger.info("Starting SHAP analysis for domain: {domain}", domain=domain)

    shap_values, X_sample = compute_shap_values(
        model, X_test, feature_names, sample_size
    )

    # Summary plot (global feature importance)
    plot_summary(shap_values, X_sample, feature_names, domain)

    # Waterfall plot for the first 3 test samples (local explanations)
    for i in range(min(3, len(X_sample))):
        plot_waterfall(
            model,
            X_sample[i:i+1],
            feature_names,
            domain,
            sample_index=i,
        )

    logger.info("SHAP analysis complete for domain: {domain}", domain=domain)
