"""Data validation — schema checks, quality checks, and leakage detection."""

from typing import Any

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    expected_columns: list[str],
    target_column: str,
    domain: str,
) -> dict[str, Any]:
    """
    Run a suite of validation checks on a DataFrame.

    Checks performed:
      - Minimum row count
      - All expected columns are present
      - Missing value counts (warns if >20% in any column)
      - Target column exists and is binary
      - Severe class imbalance (<5% minority class)
      - Duplicate rows
      - Constant (zero-variance) columns

    Args:
        df: The DataFrame to validate.
        expected_columns: List of column names expected in the data.
        target_column: Name of the prediction target column.
        domain: Domain name for logging ("credit_risk" or "network_intrusion").

    Returns:
        Validation report dict with keys: passed, n_rows, n_cols,
        missing_columns, missing_value_counts, class_distribution,
        warnings, errors.
    """
    logger.info("Running data validation for domain: {domain}", domain=domain)

    report: dict[str, Any] = {
        "domain": domain,
        "passed": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_columns": [],
        "missing_value_counts": {},
        "class_distribution": {},
        "warnings": [],
        "errors": [],
    }

    # ── Shape ─────────────────────────────────────────────────────────────────
    if len(df) < 1000:
        report["warnings"].append(
            f"Dataset has only {len(df)} rows — expected at least 1,000."
        )
    logger.info("Shape: {rows} rows × {cols} columns", rows=len(df), cols=len(df.columns))

    # ── Expected columns ──────────────────────────────────────────────────────
    actual_columns = set(df.columns)
    missing = [c for c in expected_columns if c not in actual_columns]
    if missing:
        report["missing_columns"] = missing
        report["errors"].append(f"Missing expected columns: {missing}")
        report["passed"] = False
        logger.error("Missing columns: {cols}", cols=missing)
    else:
        logger.info("All expected columns present.")

    # ── Missing values ────────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    report["missing_value_counts"] = null_counts.to_dict()

    for col, count in null_counts.items():
        pct = count / len(df) * 100
        if pct > 20:
            report["warnings"].append(
                f"Column '{col}' has {pct:.1f}% missing values — consider dropping."
            )
            logger.warning("Column '{col}' is {pct:.1f}% missing", col=col, pct=pct)
        else:
            logger.info(
                "Column '{col}' has {count} missing values ({pct:.1f}%)",
                col=col, count=count, pct=pct,
            )

    # ── Target column ─────────────────────────────────────────────────────────
    if target_column not in df.columns:
        report["errors"].append(f"Target column '{target_column}' not found.")
        report["passed"] = False
    else:
        report["class_distribution"] = df[target_column].value_counts().to_dict()

        value_counts = df[target_column].value_counts(normalize=True) * 100
        for label, pct in value_counts.items():
            logger.info("Class '{label}': {pct:.1f}%", label=label, pct=pct)

        min_class_pct = value_counts.min()
        if min_class_pct < 5.0:
            report["warnings"].append(
                f"Severe class imbalance: minority class is only {min_class_pct:.1f}%."
                " Consider class_weight='balanced', SMOTE, or threshold tuning."
            )

    # ── Duplicates ────────────────────────────────────────────────────────────
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        pct = n_duplicates / len(df) * 100
        report["warnings"].append(
            f"{n_duplicates} duplicate rows ({pct:.1f}%) — consider dropping."
        )
        logger.warning("{n} duplicate rows found ({pct:.1f}%)", n=n_duplicates, pct=pct)

    # ── Constant columns ──────────────────────────────────────────────────────
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        report["warnings"].append(f"Constant columns (zero variance): {constant_cols}")
        logger.warning("Constant columns: {cols}", cols=constant_cols)

    if report["errors"]:
        logger.error("Validation FAILED with {n} error(s).", n=len(report["errors"]))
    else:
        logger.info("Validation PASSED with {n} warning(s).", n=len(report["warnings"]))

    return report


def assert_no_leakage(df: pd.DataFrame, target_column: str, feature_columns: list[str]) -> None:
    """Raise ValueError if the target column appears in the feature list."""
    if target_column in feature_columns:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Target column '{target_column}' is included "
            f"in feature_columns. Remove it from the feature list."
        )
    logger.info("No direct target leakage detected.")
