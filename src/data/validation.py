"""
Data validation module.

Why validate data at all?
  In production ML systems, data is the most common source of silent failures.
  A model that was 95% accurate on last month's data might silently degrade to 70%
  if the input distribution shifts — and you won't know unless you check.

  Validation catches three categories of problems:
    1. Schema issues   — wrong column names, wrong dtypes, missing columns
    2. Quality issues  — unexpected nulls, impossible values, distribution drift
    3. Leakage risk    — target column appearing in features

This module produces a validation report (dict) and raises an error if critical
checks fail. Non-critical issues are logged as warnings.
"""

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

    Args:
        df: The DataFrame to validate.
        expected_columns: List of column names we expect to see.
        target_column: Name of the prediction target column.
        domain: Domain name for logging ("credit_risk" or "network_intrusion").

    Returns:
        A validation report dictionary with keys:
          - passed (bool): True if all critical checks passed
          - n_rows, n_cols: Shape of the data
          - missing_columns: Columns we expected but didn't find
          - missing_value_counts: {column: count} for columns with nulls
          - class_distribution: {label: count} for the target column
          - warnings: List of non-critical issue descriptions
          - errors: List of critical issue descriptions
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

    # ── Check 1: Shape ────────────────────────────────────────────────────────
    # If the dataset is suspiciously small it may have been truncated.
    if len(df) < 1000:
        report["warnings"].append(
            f"Dataset has only {len(df)} rows — expected at least 1,000."
        )
    logger.info("Shape: {rows} rows × {cols} columns", rows=len(df), cols=len(df.columns))

    # ── Check 2: Expected columns ─────────────────────────────────────────────
    # We check which expected columns are missing (could be a download error or
    # the user placed a different file in the raw directory).
    actual_columns = set(df.columns)
    missing = [c for c in expected_columns if c not in actual_columns]
    if missing:
        report["missing_columns"] = missing
        report["errors"].append(
            f"Missing expected columns: {missing}"
        )
        report["passed"] = False
        logger.error("Missing columns: {cols}", cols=missing)
    else:
        logger.info("All expected columns present.")

    # ── Check 3: Missing values ───────────────────────────────────────────────
    # Count nulls per column and flag columns where >20% of values are missing.
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]  # Only columns with any nulls
    report["missing_value_counts"] = null_counts.to_dict()

    for col, count in null_counts.items():
        pct = count / len(df) * 100
        if pct > 20:
            report["warnings"].append(
                f"Column '{col}' has {pct:.1f}% missing values — consider dropping."
            )
            logger.warning(
                "Column '{col}' is {pct:.1f}% missing", col=col, pct=pct
            )
        else:
            logger.info(
                "Column '{col}' has {count} missing values ({pct:.1f}%)",
                col=col, count=count, pct=pct,
            )

    # ── Check 4: Target column exists and is binary ───────────────────────────
    if target_column not in df.columns:
        report["errors"].append(f"Target column '{target_column}' not found.")
        report["passed"] = False
    else:
        unique_values = df[target_column].dropna().unique()
        report["class_distribution"] = df[target_column].value_counts().to_dict()

        # Log the class balance — this is critical for imbalanced classification
        value_counts = df[target_column].value_counts(normalize=True) * 100
        for label, pct in value_counts.items():
            logger.info("Class '{label}': {pct:.1f}%", label=label, pct=pct)

        # Warn about severe imbalance (< 5% positive class)
        min_class_pct = value_counts.min()
        if min_class_pct < 5.0:
            report["warnings"].append(
                f"Severe class imbalance: minority class is only {min_class_pct:.1f}%."
                " Consider class_weight='balanced', SMOTE, or threshold tuning."
            )

    # ── Check 5: Duplicate rows ───────────────────────────────────────────────
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        pct = n_duplicates / len(df) * 100
        report["warnings"].append(
            f"{n_duplicates} duplicate rows ({pct:.1f}%) — consider dropping."
        )
        logger.warning("{n} duplicate rows found ({pct:.1f}%)", n=n_duplicates, pct=pct)

    # ── Check 6: Constant columns ─────────────────────────────────────────────
    # A column with only one unique value carries zero information.
    constant_cols = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]
    if constant_cols:
        report["warnings"].append(
            f"Constant columns (zero variance): {constant_cols}"
        )
        logger.warning("Constant columns: {cols}", cols=constant_cols)

    # ── Summary ───────────────────────────────────────────────────────────────
    if report["errors"]:
        logger.error(
            "Validation FAILED with {n} error(s).", n=len(report["errors"])
        )
    else:
        logger.info(
            "Validation PASSED with {n} warning(s).", n=len(report["warnings"])
        )

    return report


def assert_no_leakage(df: pd.DataFrame, target_column: str, feature_columns: list[str]) -> None:
    """
    Assert that the target column is not also listed as a feature.

    Data leakage is the #1 cause of "too good to be true" model performance.
    If the target accidentally appears in the feature matrix, the model learns
    a trivial identity function and achieves near-perfect accuracy on train/test
    — but fails completely in production.

    This check catches the most common form: directly including the target as input.
    """
    if target_column in feature_columns:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Target column '{target_column}' is included "
            f"in feature_columns. Remove it from the feature list."
        )
    logger.info("No direct target leakage detected.")
