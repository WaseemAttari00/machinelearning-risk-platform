"""Data ingestion — load raw datasets from disk into DataFrames."""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_credit_risk(config: Optional[dict] = None) -> pd.DataFrame:
    """
    Load the UCI Default of Credit Card Clients CSV.

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
            "  1. Go to https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients\n"
            "  2. Download 'default of credit card clients.xls'\n"
            "  3. Convert to CSV and place at: data/raw/credit_risk/credit_risk.csv"
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

    The files have no header row; column names are supplied from config.
    The 'difficulty_level' column (last column) is dropped after loading.

    Args:
        config: Optional pre-loaded config dict. If None, loads from YAML.

    Returns:
        Tuple of (train_df, test_df).
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
    """Load a processed dataset from a Parquet file."""
    logger.info("Loading processed data from {path}", path=path)
    return pd.read_parquet(path)


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save a processed DataFrame to Parquet, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved {rows} rows to {path}", rows=len(df), path=path)
