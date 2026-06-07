"""
Unit tests for data ingestion and validation.

Why write tests for an ML project?
  Most ML tutorials skip testing. This is a mistake for portfolios because:
  1. Tests demonstrate engineering discipline that interviewers value
  2. They catch regressions when you modify preprocessing logic
  3. The test suite serves as documentation of expected behavior

Testing strategy:
  - We don't test against real datasets (they're large and not in the repo)
  - We create synthetic mini-datasets that mirror the schema of real data
  - This makes tests fast (milliseconds) and self-contained

Run all tests:
  pytest tests/ -v

Run with coverage:
  pytest tests/ --cov=src --cov-report=term-missing
"""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import assert_no_leakage, validate_dataframe
from src.data.preprocessing import split_features_target, make_train_test_split


# ── Fixtures ──────────────────────────────────────────────────────────────────
# pytest fixtures are reusable setup functions.
# Any test function that declares a fixture name as a parameter gets it injected.

@pytest.fixture
def credit_risk_sample_df():
    """Synthetic credit risk DataFrame mimicking the Give Me Some Credit schema."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": np.random.uniform(0, 1.5, n),
        "age": np.random.randint(18, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": np.random.randint(0, 5, n),
        "DebtRatio": np.random.uniform(0, 2, n),
        "MonthlyIncome": np.where(
            np.random.random(n) < 0.05, np.nan, np.random.uniform(1000, 20000, n)
        ),
        "NumberOfOpenCreditLinesAndLoans": np.random.randint(0, 20, n),
        "NumberOfTimes90DaysLate": np.random.randint(0, 3, n),
        "NumberRealEstateLoansOrLines": np.random.randint(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse": np.random.randint(0, 3, n),
        "NumberOfDependents": np.where(
            np.random.random(n) < 0.03, np.nan, np.random.randint(0, 6, n).astype(float)
        ),
        "SeriousDlqin2yrs": np.random.choice([0, 1], n, p=[0.93, 0.07]),
    })


@pytest.fixture
def network_intrusion_sample_df():
    """Synthetic network intrusion DataFrame mimicking NSL-KDD schema."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "duration": np.random.exponential(10, n),
        "protocol_type": np.random.choice(["tcp", "udp", "icmp"], n),
        "service": np.random.choice(["http", "ftp", "smtp", "ssh"], n),
        "flag": np.random.choice(["SF", "S0", "REJ"], n),
        "src_bytes": np.random.exponential(1000, n),
        "dst_bytes": np.random.exponential(5000, n),
        "logged_in": np.random.choice([0, 1], n),
        "count": np.random.randint(1, 512, n).astype(float),
        "label": np.random.choice(["normal", "neptune", "smurf"], n, p=[0.5, 0.3, 0.2]),
    })


# ── Validation tests ──────────────────────────────────────────────────────────

class TestValidation:

    def test_validate_passes_on_clean_data(self, credit_risk_sample_df):
        """A clean DataFrame should pass validation with no errors."""
        expected_cols = [
            "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
            "MonthlyIncome", "SeriousDlqin2yrs"
        ]
        report = validate_dataframe(
            df=credit_risk_sample_df,
            expected_columns=expected_cols,
            target_column="SeriousDlqin2yrs",
            domain="credit_risk",
        )
        assert report["passed"] is True
        assert len(report["errors"]) == 0

    def test_validate_fails_on_missing_column(self, credit_risk_sample_df):
        """Missing a required column should set passed=False and populate errors."""
        df_missing = credit_risk_sample_df.drop(columns=["age"])
        report = validate_dataframe(
            df=df_missing,
            expected_columns=["age", "DebtRatio", "SeriousDlqin2yrs"],
            target_column="SeriousDlqin2yrs",
            domain="credit_risk",
        )
        assert report["passed"] is False
        assert "age" in str(report["errors"])

    def test_validate_detects_class_imbalance(self, credit_risk_sample_df):
        """Severe class imbalance should produce a warning (not an error)."""
        # Create extreme imbalance: 1% positive
        df = credit_risk_sample_df.copy()
        df["SeriousDlqin2yrs"] = np.random.choice([0, 1], len(df), p=[0.99, 0.01])
        report = validate_dataframe(
            df=df,
            expected_columns=["SeriousDlqin2yrs"],
            target_column="SeriousDlqin2yrs",
            domain="credit_risk",
        )
        assert report["passed"] is True  # Imbalance is a warning, not an error
        assert any("imbalance" in w.lower() for w in report["warnings"])

    def test_no_leakage_raises_on_target_in_features(self, credit_risk_sample_df):
        """Including the target in feature_columns should raise ValueError."""
        with pytest.raises(ValueError, match="DATA LEAKAGE"):
            assert_no_leakage(
                df=credit_risk_sample_df,
                target_column="SeriousDlqin2yrs",
                feature_columns=["age", "DebtRatio", "SeriousDlqin2yrs"],
            )

    def test_no_leakage_passes_when_clean(self, credit_risk_sample_df):
        """Clean feature list should not raise."""
        assert_no_leakage(
            df=credit_risk_sample_df,
            target_column="SeriousDlqin2yrs",
            feature_columns=["age", "DebtRatio"],
        )


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:

    def test_split_features_target_correct_shapes(self, credit_risk_sample_df):
        """X and y should have correct shapes after splitting."""
        X, y = split_features_target(
            credit_risk_sample_df,
            target_column="SeriousDlqin2yrs",
        )
        assert len(X) == len(credit_risk_sample_df)
        assert len(y) == len(credit_risk_sample_df)
        assert "SeriousDlqin2yrs" not in X.columns

    def test_split_features_target_drops_extra_columns(self, credit_risk_sample_df):
        """Extra columns listed in drop_columns should be removed from X."""
        df = credit_risk_sample_df.copy()
        df["Unnamed: 0"] = range(len(df))

        X, y = split_features_target(
            df,
            target_column="SeriousDlqin2yrs",
            drop_columns=["Unnamed: 0"],
        )
        assert "Unnamed: 0" not in X.columns

    def test_train_test_split_preserves_class_ratio(self, credit_risk_sample_df):
        """Stratified split should maintain approximately the same positive class rate."""
        X, y = split_features_target(credit_risk_sample_df, "SeriousDlqin2yrs")
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, stratify=True)

        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        overall_pos_rate = y.mean()

        # Train and test positive rates should each be within 2% of the overall rate
        assert abs(train_pos_rate - overall_pos_rate) < 0.02
        assert abs(test_pos_rate - overall_pos_rate) < 0.02

    def test_train_test_split_sizes(self, credit_risk_sample_df):
        """Default test_size=0.2 should give an 80/20 split."""
        X, y = split_features_target(credit_risk_sample_df, "SeriousDlqin2yrs")
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2)

        total = len(X_train) + len(X_test)
        assert total == len(X)
        # Allow ±5 rows for rounding in stratified split
        assert abs(len(X_test) - int(0.2 * len(X))) <= 5
