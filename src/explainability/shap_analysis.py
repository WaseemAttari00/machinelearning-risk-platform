"""
SHAP explainability module.

Generates global feature importance (beeswarm summary) and local per-prediction
explanations (waterfall plots) for the trained XGBoost models, saved as PNG files
to models/<domain>/.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
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
    Compute SHAP values using TreeExplainer.

    Args:
        model: Fitted tree-based model (XGBoost).
        X: Preprocessed feature matrix.
        feature_names: Column names for X.
        sample_size: Number of rows to subsample for efficiency. None uses all rows.

    Returns:
        (shap_values_2d, X_sample, explainer)
    """
    if sample_size and len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X[idx]
        logger.info("Subsampled {n} rows for SHAP (from {total})", n=sample_size, total=len(X))
    else:
        X_sample = X

    logger.info("Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_sample)

    sv = shap_explanation.values
    if sv.ndim == 3:
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
    Generate a SHAP beeswarm summary plot showing global feature importance.

    Each dot is one sample; position on x-axis is the SHAP value (contribution
    to the prediction); color indicates feature value magnitude.

    Args:
        shap_values: 2D array (n_samples, n_features).
        X_sample: Corresponding input data for color encoding.
        feature_names: Column names.
        domain: Used for plot title and output path.
        max_display: Number of top features to display.
        save: If True, saves PNG to models/<domain>/shap_summary.png.
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

    Shows how each feature pushed the prediction up or down from the base value.

    Args:
        explainer: Fitted TreeExplainer.
        X_single: Single-row feature array.
        feature_names: Column names.
        domain: Used for output path.
        sample_index: Index label for the output filename.
        save: If True, saves PNG to models/<domain>/shap_waterfall_<n>.png.
    """
    shap_explanation = explainer(X_single)
    sv = shap_explanation.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

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
    Run full SHAP analysis: one summary plot and three waterfall plots.

    Called from train.py after training completes. Output PNGs are saved to
    models/<domain>/.
    """
    logger.info("Starting SHAP analysis for domain: {domain}", domain=domain)

    shap_values, X_sample, explainer = compute_shap_values(
        model, X_test, feature_names, sample_size
    )

    plt.close("all")
    plot_summary(shap_values, X_sample, feature_names, domain)
    plt.close("all")

    for i in range(min(3, len(X_sample))):
        plot_waterfall(explainer, X_sample[i:i+1], feature_names, domain, sample_index=i)
        plt.close("all")

    logger.info("SHAP analysis complete for domain: {domain}", domain=domain)
