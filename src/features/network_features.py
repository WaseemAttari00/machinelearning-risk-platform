"""
Network intrusion feature engineering pipeline.

NSL-KDD has mixed feature types: continuous numeric, binary numeric, and
categorical (protocol_type, service, flag). A ColumnTransformer applies
separate sub-pipelines to each subset, then concatenates the results.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger

logger = get_logger(__name__)


def binarize_labels(series: pd.Series, label_map: dict[str, int]) -> pd.Series:
    """
    Convert NSL-KDD string labels to binary 0/1.

    Maps "normal" → 0; all attack types → 1.

    Args:
        series: Raw label column (e.g., "normal", "neptune", "smurf").
        label_map: From config — {"normal": 0, "attack": 1}.

    Returns:
        Binary integer Series (0 = normal, 1 = attack).
    """
    normal_label = [k for k, v in label_map.items() if v == 0]
    if not normal_label:
        raise ValueError("label_map must contain an entry mapping to 0 (the negative class).")
    normal_label = normal_label[0]

    binary = series.apply(lambda x: 0 if str(x).strip() == normal_label else 1)
    n_normal = (binary == 0).sum()
    n_attack = (binary == 1).sum()
    logger.info(
        "Label binarization: {n} normal, {a} attack ({pct:.1f}% attack)",
        n=n_normal,
        a=n_attack,
        pct=n_attack / len(binary) * 100,
    )
    return binary


def build_network_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build the network intrusion preprocessing Pipeline.

    Numeric sub-pipeline: SimpleImputer(median) → StandardScaler
    Categorical sub-pipeline: SimpleImputer(most_frequent) →
        OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        Unfitted sklearn Pipeline.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
    ])

    logger.info(
        "Network pipeline built: {n} numeric + {c} categorical features",
        n=len(numeric_features),
        c=len(categorical_features),
    )
    return pipeline


def get_feature_names_after_encoding(
    pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """
    Reconstruct output column names after OneHotEncoder expansion.

    Args:
        pipeline: A fitted pipeline.
        numeric_features: Original numeric column names (passed through unchanged).
        categorical_features: Original categorical column names.

    Returns:
        Ordered list of feature names matching pipeline output columns.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    categorical_names = list(encoder.get_feature_names_out(categorical_features))
    return list(numeric_features) + categorical_names
