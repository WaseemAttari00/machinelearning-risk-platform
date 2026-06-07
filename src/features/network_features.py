"""
Network Intrusion Feature Engineering.

NSL-KDD has a mix of feature types:
  - Continuous numeric features (duration, src_bytes, dst_bytes, rates...)
  - Categorical features (protocol_type: tcp/udp/icmp, service: http/ftp/..., flag: SF/S0/...)
  - The label column: strings like "normal", "neptune", "smurf" → binarized to 0/1

Pipeline structure:
    Categorical features → OneHotEncoder
    Numeric features     → StandardScaler
    [Combined via ColumnTransformer]

Why ColumnTransformer?
  Different columns need different transformations. ColumnTransformer lets us
  apply different pipelines to different column subsets and then concatenate
  the results into a single output matrix. This is the standard sklearn pattern
  for heterogeneous (mixed-type) data.
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

    NSL-KDD labels look like: "normal", "neptune", "smurf", "back", "portsweep", ...
    We map "normal" → 0 (benign) and everything else → 1 (attack).

    Args:
        series: The raw label column with string values.
        label_map: From config — {"normal": 0, "attack": 1}.
                   "attack" is a special key meaning "everything not in the map".

    Returns:
        Binary integer Series (0 = normal, 1 = attack).

    Why binarize here instead of in the pipeline?
      Label binarization transforms y (the target), not X (the features).
      sklearn Pipelines only transform X. We handle y separately.
    """
    # Map "normal" → 0; everything else → 1
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

    Two sub-pipelines are combined with ColumnTransformer:

    Numeric sub-pipeline:
        SimpleImputer(median) → StandardScaler
        (NSL-KDD has no missing values, but the imputer is defensive)

    Categorical sub-pipeline:
        SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')

        Why OneHotEncoder?
          Categorical features like protocol_type (tcp/udp/icmp) have no natural
          numeric ordering. If we encoded tcp=0, udp=1, icmp=2, the model would
          incorrectly assume udp is "twice as large" as tcp.
          One-hot encoding creates a separate binary column for each category.

        Why handle_unknown='ignore'?
          The test set might contain a service or flag value not seen during training.
          'ignore' maps unknown categories to an all-zeros vector rather than raising
          an error. This is essential for production robustness.

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
            handle_unknown="ignore",    # Unknown categories → all-zero vector
            sparse_output=False,        # Return dense array, not sparse matrix
        )),
    ])

    # ColumnTransformer applies the right sub-pipeline to each column subset
    # remainder="drop" means any column not listed is dropped from the output
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
    Get human-readable feature names after OneHotEncoder expansion.

    OneHotEncoder turns 1 categorical column with N unique values into N binary
    columns. This function reconstructs the full ordered list of output column
    names so SHAP plots are labeled correctly.

    Example:
        protocol_type had [tcp, udp, icmp] →
        ["protocol_type_tcp", "protocol_type_udp", "protocol_type_icmp"]

    Args:
        pipeline: A FITTED pipeline (must be fitted to get encoder categories).
        numeric_features: Original numeric column names (passed through unchanged).
        categorical_features: Original categorical column names.

    Returns:
        List of feature names in the order they appear in the pipeline output.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]

    # get_feature_names_out returns arrays like ["protocol_type_tcp", ...]
    categorical_names = list(encoder.get_feature_names_out(categorical_features))

    return list(numeric_features) + categorical_names
