"""
Data ingestion module.

Responsibility:
  Load raw data from disk into a pandas DataFrame. Nothing else.
  No cleaning, no transformation — pure loading.

Why separate ingestion from preprocessing?
  Separation of concerns. If the raw data format changes (e.g., CSV → Parquet),
  you only change this file. The downstream pipeline doesn't need to know or care.
  It also makes unit testing easy: mock this function and test preprocessing in isolation.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_credit_risk(config: Optional[dict] = None) -> pd.DataFrame:
    """
    Load the raw Give Me Some Credit CSV.

    The Kaggle competition provides:
      - cs-training.csv : labeled data we use for training and evaluation
      - cs-test.csv     : unlabeled holdout (we ignore this for our purposes)

    Args:
        config: Optional pre-loaded config dict. If None, loads from YAML.

    Returns:
        Raw DataFrame with all original columns including the target.
    """
    if config is None:
        config = load_config("credit_risk")

    raw_path = config["data"]["raw_path"]
    logger.info("Loading credit risk data from {path}", path=raw_path)

    if not Path(raw_path).exists():
        raise FileNotFoundError(
            f"Raw data not found at: {raw_path}\n\n"
            "Download instructions:\n"
            "  1. Go to https://www.kaggle.com/c/GiveMeSomeCredit/data\n"
            "  2. Download 'cs-training.csv'\n"
            "  3. Place it at: data/raw/credit_risk/cs-training.csv"
        )

    df = pd.read_csv(raw_path)
    logger.info(
        "Loaded {rows} rows, {cols} columns from credit risk dataset",
        rows=len(df),
        cols=len(df.columns),
    )
    return df


def load_network_intrusion(config: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the NSL-KDD train and test sets.

    NSL-KDD specifics:
      - No header row in the file — we supply column names from the config
      - The last column is 'difficulty_level' which we drop (it's a meta column)
      - Labels are strings like "normal", "neptune", "smurf" — we binarize later

    Args:
        config: Optional pre-loaded config dict. If None, loads from YAML.

    Returns:
        Tuple of (train_df, test_df) — both have the same columns.
    """
    if config is None:
        config = load_config("network_intrusion")

    train_path = config["data"]["raw_train_path"]
    test_path = config["data"]["raw_test_path"]
    column_names = config["data"]["column_names"]

    for path in [train_path, test_path]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"NSL-KDD data not found at: {path}\n\n"
                "Download instructions:\n"
                "  1. Go to https://www.unb.ca/cic/datasets/nsl.html\n"
                "  2. Download 'NSL-KDD.zip'\n"
                "  3. Extract KDDTrain+.txt and KDDTest+.txt\n"
                "  4. Place them at: data/raw/network_intrusion/"
            )

    logger.info("Loading NSL-KDD training data from {path}", path=train_path)
    train_df = pd.read_csv(train_path, header=None, names=column_names)

    logger.info("Loading NSL-KDD test data from {path}", path=test_path)
    test_df = pd.read_csv(test_path, header=None, names=column_names)

    logger.info(
        "Loaded NSL-KDD — train: {tr} rows, test: {te} rows",
        tr=len(train_df),
        te=len(test_df),
    )
    return train_df, test_df


def load_processed(path: str) -> pd.DataFrame:
    """
    Load a processed (feature-engineered) dataset from a Parquet file.

    Why Parquet instead of CSV for processed data?
      - Parquet preserves column data types (dtypes). CSV loses this — every column
        comes back as object/string and you have to re-cast manually.
      - Parquet is ~5x faster to read and ~3x smaller on disk.
      - It supports columnar reads: if you only need 5 of 40 columns, it only
        reads those 5 from disk.

    Args:
        path: Absolute or relative path to the .parquet file.

    Returns:
        DataFrame loaded from Parquet.
    """
    logger.info("Loading processed data from {path}", path=path)
    return pd.read_parquet(path)


def save_processed(df: pd.DataFrame, path: str) -> None:
    """
    Save a processed DataFrame to Parquet format.

    Creates parent directories automatically so callers don't need to.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Saved {rows} rows to {path}",
        rows=len(df),
        path=path,
    )
