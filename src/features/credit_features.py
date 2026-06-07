"""
Credit Risk Feature Engineering.

This module builds a scikit-learn Pipeline that transforms raw credit risk
features into a form suitable for model training.

Key design principle: EVERYTHING is a Pipeline step.
  This means the exact same transformations are applied to:
    - Training data (pipeline.fit_transform)
    - Test data     (pipeline.transform — no fitting, no leakage)
    - API requests  (pipeline.transform — real-time, single-row)

If you did transformations outside a Pipeline (e.g., computing medians from the
full dataset and using them on test data), you would be leaking information from
the test set into the training process.

Pipeline structure:
    Raw features
        → Impute missing values (median strategy)
        → Clip extreme outliers
        → Scale to zero mean, unit variance
    Processed features (ready for model)
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
    Custom scikit-learn transformer that clips values above a given percentile.

    Why clip outliers?
      Tree-based models (XGBoost) are robust to outliers. But some features in
      credit data have pathological outliers: MonthlyIncome can be 3,008,750 when
      the 99th percentile is ~25,000. These extreme values can dominate feature
      importance calculations and make SHAP plots unreadable.

    Why a custom transformer instead of just calling df.clip()?
      By wrapping it in a sklearn-compatible transformer (with fit/transform),
      it becomes a Pipeline step. The .fit() method learns the percentile bounds
      from the TRAINING data. The .transform() method applies those same bounds
      to test data — never re-computing from test data.

    Inheriting from BaseEstimator, TransformerMixin gives us:
      - get_params() / set_params() for Optuna and GridSearchCV compatibility
      - fit_transform() for free (calls fit then transform)
    """

    def __init__(self, upper_percentile: float = 99.0):
        """
        Args:
            upper_percentile: Values above this percentile are clipped.
                              99.0 means the top 1% of values are capped.
        """
        self.upper_percentile = upper_percentile
        self.clip_bounds_: dict = {}  # Learned during fit — underscore suffix is sklearn convention

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn the upper clip bound for each column from the training data.

        The 'y=None' signature is required by sklearn's API even though we
        don't use labels — this is an unsupervised transformation.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col in X.columns:
            self.clip_bounds_[col] = np.percentile(X[col].dropna(), self.upper_percentile)

        logger.info(
            "OutlierClipper fitted on {n} columns at {p}th percentile",
            n=len(self.clip_bounds_),
            p=self.upper_percentile,
        )
        return self  # sklearn transformers must return self from fit()

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Apply the learned clip bounds to new data.

        Columns not seen during fit are left unchanged (defensive default).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X = X.copy()  # Never modify the input in-place
        for col, upper in self.clip_bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(upper=upper)

        return X


def build_credit_pipeline() -> Pipeline:
    """
    Build the credit risk preprocessing Pipeline.

    Pipeline steps (in order):
      1. SimpleImputer(strategy="median")
           Replaces NaN values with the column median computed on train data.
           Median is preferred over mean for financial data because it is
           robust to the extreme outliers that are common in income/debt features.

      2. OutlierClipper(upper_percentile=99)
           Caps values above the 99th percentile. We clip AFTER imputing because
           imputed values might themselves be extreme if the median is near an outlier.

      3. StandardScaler()
           Scales each feature to zero mean and unit variance.
           Why scale? Logistic Regression is sensitive to feature scale — a feature
           with range [0, 1,000,000] will dominate a feature with range [0, 1] unless
           both are scaled. XGBoost doesn't need scaling, but including it makes the
           pipeline compatible with any model type.

    Returns:
        An unfitted sklearn Pipeline. Call pipeline.fit_transform(X_train)
        to fit it and transform training data, then pipeline.transform(X_test)
        for test data.
    """
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper(upper_percentile=99)),
        ("scaler", StandardScaler()),
    ])
    logger.info("Credit risk pipeline built: imputer → clipper → scaler")
    return pipeline


def get_feature_names(config: dict) -> list[str]:
    """
    Return the ordered list of feature column names from config.

    This is used by SHAP and evaluation code to map column indices back
    to human-readable names after pipeline transformation.
    """
    return config["features"]["numeric_features"]
