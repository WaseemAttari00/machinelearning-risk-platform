"""
SHAP (SHapley Additive exPlanations) analysis module.

What is SHAP?
  SHAP answers: "How much did each feature contribute to THIS prediction?"
  The math comes from cooperative game theory — Shapley values fairly distribute
  the 'credit' (prediction contribution) among all features.

Why SHAP instead of built-in feature importances?
  Built-in importances (model.feature_importances_) tell you which features are
  used most across all predictions, but NOT:
    - Direction: does higher PAY_0 → more risk or less?
    - Local: why did THIS specific customer get flagged?
    - Reliable importance for correlated features

  SHAP gives you all three. It is also model-agnostic — same API for XGBoost,
  Random Forests, and neural networks.

Why does this matter for interviews?
  Explainability is now a legal requirement in finance (GDPR Article 22) and
  healthcare. Being able to explain a decision to a non-technical stakeholder
  (loan officer, security analyst) is a real job skill.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — required for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
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
) -> tuple:
    """
    Compute SHAP values for a dataset using TreeExplainer.

    Why TreeExplainer?
      SHAP has model-specific explainers:
        - TreeExplainer  : for tree-based models (XGBoost, LightGBM, RF) — exact, fast
        - LinearExplainer: for linear models (Logistic Regression)
        - KernelExplainer: model-agnostic, ~100x slower

      We always use TreeExplainer since our primary model is XGBoost.

    Args:
        model: A fitted tree-based model (XGBoost).
        X: Preprocessed feature matrix (numpy array).
        feature_names: Column names for the features.
        sample_size: Subsample size. 1000 samples gives reliable global summaries
                     while keeping computation fast (SHAP is O(n·depth)).

    Returns:
        (shap_values_array, X_sample, explainer)
        - shap_values_array: 2D numpy array of shape (n_samples, n_features)
        - X_sample: The subsampled input data
        - explainer: The fitted TreeExplainer (needed for waterfall plots)
    """
    if sample_size and len(X) > sample_size:
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

    # In SHAP 0.45+, calling explainer(X) returns an Explanation object.
    # .values gives the raw 2D shap value array.
    # For binary classification, XGBoost returns values for class 1 (positive class).
    shap_explanation = explainer(X_sample)

    # shap_explanation.values can be 3D for multi-output models: (n, features, classes)
    # For binary XGBoost with tree_method="hist", it's (n, features) for the positive class.
    sv = shap_explanation.values
    if sv.ndim == 3:
        # Take the positive class (index 1) if 3D
        sv = sv[:, :, 1]

    logger.info("SHAP values computed — shape: {shape}", shape=sv.shape)
    return sv, X_sample, explainer


def plot_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list[str],
    domain: str,
    max_display: int = 15,
    save: bool = True,
) -> plt.Figure:
    """
    Generate a SHAP beeswarm summary plot.

    How to read this plot:
      - Each row = one feature (sorted by mean |SHAP| — most important at top)
      - Each dot = one prediction (one data sample)
      - X-axis = SHAP value: positive → pushes prediction toward class 1 (risky)
                              negative → pushes toward class 0 (safe)
      - Color: red = high feature value, blue = low feature value

    Example interpretation:
      "High PAY_0 (red dots) pushes predictions strongly toward Default.
       Low PAY_0 (blue dots) reduces default probability."

    This plot is the first thing to show in a portfolio demo. It instantly
    communicates both importance AND direction in one visualization.
    """
    fig, _ = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )

    title = domain.replace("_", " ").title()
    plt.title(f"SHAP Feature Importance — {title}", fontsize=14, pad=15)
    plt.tight_layout()

    if save:
        output_path = PROJECT_ROOT / "models" / domain / "shap_summary.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved to {path}", path=str(output_path))

    return fig


def plot_waterfall(
    explainer,
    X_single: np.ndarray,
    feature_names: list[str],
    domain: str,
    sample_index: int = 0,
    save: bool = True,
) -> plt.Figure:
    """
    Generate a SHAP waterfall plot for a single prediction.

    The waterfall plot is the "local explanation":
      - Starts at the base value (average model output across all training data)
      - Each bar shows how much one feature pushed the prediction up or down
      - Ends at the final prediction for this specific sample

    Use case: "Mr. Smith was flagged as high risk. The model says his PAY_0=3
    (3 months late) contributed +0.42 to the log-odds. His LIMIT_BAL was low,
    contributing +0.18. His age contributed -0.05 (slightly protective)."

    This is what you show a loan officer or security analyst.
    """
    # Compute SHAP for this single sample
    shap_explanation = explainer(X_single)
    sv = shap_explanation.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    # Rebuild Explanation object with correct feature names for the waterfall plot
    exp_single = shap.Explanation(
        values=sv[0],
        base_values=float(shap_explanation.base_values[0])
            if hasattr(shap_explanation.base_values, "__len__")
            else float(shap_explanation.base_values),
        data=X_single[0],
        feature_names=feature_names,
    )

    fig, _ = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(exp_single, max_display=15, show=False)
    plt.title(f"SHAP Local Explanation — Sample {sample_index}", fontsize=12, pad=10)
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
    Run the full SHAP analysis: compute values → summary plot → 3 waterfall plots.
    Called from train.py after training is complete.
    """
    logger.info("Starting SHAP analysis for domain: {domain}", domain=domain)

    shap_values, X_sample, explainer = compute_shap_values(
        model, X_test, feature_names, sample_size
    )

    # Global: beeswarm summary
    plt.close("all")
    plot_summary(shap_values, X_sample, feature_names, domain)
    plt.close("all")

    # Local: waterfall for first 3 samples
    for i in range(min(3, len(X_sample))):
        plot_waterfall(
            explainer,
            X_sample[i:i+1],
            feature_names,
            domain,
            sample_index=i,
        )
        plt.close("all")

    logger.info("SHAP analysis complete for domain: {domain}", domain=domain)
