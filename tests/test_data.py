"""Unit tests for data ingestion and validation."""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import assert_no_leakage, validate_dataframe
from src.data.preprocessing import split_features_target, make_train_test_split


@pytest.fixture
def credit_risk_sample_df():
    """Synthetic DataFrame matching the UCI credit risk feature schema."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "LIMIT_BAL": np.random.uniform(10000, 800000, n),
        "SEX": np.random.choice([1, 2], n),
        "EDUCATION": np.random.choice([1, 2, 3, 4], n),
        "MARRIAGE": np.random.choice([1, 2, 3], n),
        "AGE": np.random.randint(18, 75, n),
        "PAY_0": np.random.randint(-2, 4, n),
        "PAY_2": np.random.randint(-2, 4, n),
        "PAY_3": np.random.randint(-2, 4, n),
        "PAY_4": np.random.randint(-2, 4, n),
        "PAY_5": np.random.randint(-2, 4, n),
        "PAY_6": np.random.randint(-2, 4, n),
        "BILL_AMT1": np.random.uniform(0, 50000, n),
        "BILL_AMT2": np.random.uniform(0, 50000, n),
        "BILL_AMT3": np.random.uniform(0, 50000, n),
        "BILL_AMT4": np.random.uniform(0, 50000, n),
        "BILL_AMT5": np.random.uniform(0, 50000, n),
        "BILL_AMT6": np.random.uniform(0, 50000, n),
        "PAY_AMT1": np.random.uniform(0, 10000, n),
        "PAY_AMT2": np.random.uniform(0, 10000, n),
        "PAY_AMT3": np.random.uniform(0, 10000, n),
        "PAY_AMT4": np.random.uniform(0, 10000, n),
        "PAY_AMT5": np.random.uniform(0, 10000, n),
        "PAY_AMT6": np.random.uniform(0, 10000, n),
        "default": np.random.choice([0, 1], n, p=[0.78, 0.22]),
    })


@pytest.fixture
def network_intrusion_sample_df():
    """Synthetic DataFrame matching the NSL-KDD feature schema."""
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


class TestValidation:

    def test_validate_passes_on_clean_data(self, credit_risk_sample_df):
        """A clean DataFrame should pass validation with no errors."""
        expected_cols = ["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "default"]
        report = validate_dataframe(
            df=credit_risk_sample_df,
            expected_columns=expected_cols,
            target_column="default",
            domain="credit_risk",
        )
        assert report["passed"] is True
        assert len(report["errors"]) == 0

    def test_validate_fails_on_missing_column(self, credit_risk_sample_df):
        """Missing a required column should set passed=False and populate errors."""
        df_missing = credit_risk_sample_df.drop(columns=["AGE"])
        report = validate_dataframe(
            df=df_missing,
            expected_columns=["AGE", "LIMIT_BAL", "default"],
            target_column="default",
            domain="credit_risk",
        )
        assert report["passed"] is False
        assert "AGE" in str(report["errors"])

    def test_validate_detects_class_imbalance(self, credit_risk_sample_df):
        """Severe class imbalance should produce a warning (not an error)."""
        df = credit_risk_sample_df.copy()
        df["default"] = np.random.choice([0, 1], len(df), p=[0.99, 0.01])
        report = validate_dataframe(
            df=df,
            expected_columns=["default"],
            target_column="default",
            domain="credit_risk",
        )
        assert report["passed"] is True
        assert any("imbalance" in w.lower() for w in report["warnings"])

    def test_no_leakage_raises_on_target_in_features(self, credit_risk_sample_df):
        """Including the target in feature_columns should raise ValueError."""
        with pytest.raises(ValueError, match="DATA LEAKAGE"):
            assert_no_leakage(
                df=credit_risk_sample_df,
                target_column="default",
                feature_columns=["LIMIT_BAL", "AGE", "default"],
            )

    def test_no_leakage_passes_when_clean(self, credit_risk_sample_df):
        """Clean feature list should not raise."""
        assert_no_leakage(
            df=credit_risk_sample_df,
            target_column="default",
            feature_columns=["LIMIT_BAL", "AGE"],
        )


class TestPreprocessing:

    def test_split_features_target_correct_shapes(self, credit_risk_sample_df):
        """X and y should have correct shapes after splitting."""
        X, y = split_features_target(credit_risk_sample_df, target_column="default")
        assert len(X) == len(credit_risk_sample_df)
        assert len(y) == len(credit_risk_sample_df)
        assert "default" not in X.columns

    def test_split_features_target_drops_extra_columns(self, credit_risk_sample_df):
        """Columns listed in drop_columns should be removed from X."""
        df = credit_risk_sample_df.copy()
        df["ID"] = range(len(df))

        X, y = split_features_target(df, target_column="default", drop_columns=["ID"])
        assert "ID" not in X.columns

    def test_train_test_split_preserves_class_ratio(self, credit_risk_sample_df):
        """Stratified split should maintain approximately the same positive class rate."""
        X, y = split_features_target(credit_risk_sample_df, "default")
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, stratify=True)

        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        overall_pos_rate = y.mean()

        assert abs(train_pos_rate - overall_pos_rate) < 0.02
        assert abs(test_pos_rate - overall_pos_rate) < 0.02

    def test_train_test_split_sizes(self, credit_risk_sample_df):
        """Default test_size=0.2 should give an 80/20 split."""
        X, y = split_features_target(credit_risk_sample_df, "default")
        X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2)

        total = len(X_train) + len(X_test)
        assert total == len(X)
        assert abs(len(X_test) - int(0.2 * len(X))) <= 5
