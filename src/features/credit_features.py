"""
Credit risk feature engineering pipeline.

Pipeline: SimpleImputer(median) → OutlierClipper(99th pct) → StandardScaler.
All transformations are fit on training data only; test data is only transformed.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips feature values above a given percentile learned from training data.

    Inherits from BaseEstimator and TransformerMixin for sklearn Pipeline
    compatibility (get_params/set_params, fit_transform).
    """

    def __init__(self, upper_percentile: float = 99.0):
        self.upper_percentile = upper_percentile
        self.clip_bounds_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col in X.columns:
            self.clip_bounds_[col] = np.percentile(X[col].dropna(), self.upper_percentile)

        logger.info(
            "OutlierClipper fitted on {n} columns at {p}th percentile",
            n=len(self.clip_bounds_),
            p=self.upper_percentile,
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X = X.copy()
        for col, upper in self.clip_bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(upper=upper)

        return X


def build_credit_pipeline() -> Pipeline:
    """
    Build the credit risk preprocessing Pipeline.

    Steps:
      1. SimpleImputer(strategy="median") — handles missing values
      2. OutlierClipper(upper_percentile=99) — caps extreme values
      3. StandardScaler() — normalizes to zero mean, unit variance

    Returns:
        Unfitted sklearn Pipeline.
    """
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper(upper_percentile=99)),
        ("scaler", StandardScaler()),
    ])
    logger.info("Credit risk pipeline built: imputer → clipper → scaler")
    return pipeline


def get_feature_names(config: dict) -> list[str]:
    """Return the ordered feature column names from config."""
    return config["features"]["numeric_features"]
