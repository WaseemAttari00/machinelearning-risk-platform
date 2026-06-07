"""Domain-agnostic preprocessing utilities: splitting, pipeline I/O."""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into feature matrix X and target series y.

    Args:
        df: Full DataFrame including target column.
        target_column: Name of the column to predict.
        drop_columns: Additional columns to exclude from features (e.g., ID columns).

    Returns:
        (X, y) — feature DataFrame and target Series.
    """
    cols_to_drop = [target_column]
    if drop_columns:
        cols_to_drop.extend(drop_columns)

    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df[target_column]

    logger.info(
        "Split: X shape={shape}, target='{target}', class counts={counts}",
        shape=X.shape,
        target=target_column,
        counts=y.value_counts().to_dict(),
    )
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and labels into train and test sets.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Fraction of data held out for testing (default 20%).
        random_state: Seed for reproducibility.
        stratify: If True, preserves class proportions in both splits.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    stratify_labels = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    logger.info(
        "Train set: {tr} rows | Test set: {te} rows | Stratified: {s}",
        tr=len(X_train),
        te=len(X_test),
        s=stratify,
    )
    logger.info(
        "Train class distribution: {dist}",
        dist=y_train.value_counts(normalize=True).round(3).to_dict(),
    )
    logger.info(
        "Test class distribution: {dist}",
        dist=y_test.value_counts(normalize=True).round(3).to_dict(),
    )

    return X_train, X_test, y_train, y_test


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Serialize a fitted sklearn Pipeline to disk using joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Pipeline saved to {path}", path=path)


def load_pipeline(path: str) -> Pipeline:
    """Load a previously saved sklearn Pipeline from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Pipeline not found at: {path}")
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from {path}", path=path)
    return pipeline
