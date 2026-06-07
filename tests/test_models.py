"""Unit tests for feature engineering and model evaluation components."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.credit_features import OutlierClipper, build_credit_pipeline
from src.features.network_features import binarize_labels, build_network_pipeline
from src.models.evaluate import _find_optimal_threshold, evaluate_model


class TestOutlierClipper:

    def test_fit_learns_bounds(self):
        """fit() should compute percentile bounds from training data."""
        X = pd.DataFrame({"a": [1, 2, 3, 100], "b": [0.1, 0.2, 0.3, 10.0]})
        clipper = OutlierClipper(upper_percentile=75)
        clipper.fit(X)
        assert "a" in clipper.clip_bounds_
        assert "b" in clipper.clip_bounds_
        assert clipper.clip_bounds_["a"] < 100

    def test_transform_clips_outliers(self):
        """transform() should cap values above the learned 99th percentile."""
        np.random.seed(42)
        X_train = pd.DataFrame({"a": np.random.normal(0, 1, 100)})
        X_test = pd.DataFrame({"a": [0.0, 1.0, 999.0]})

        clipper = OutlierClipper(upper_percentile=99)
        clipper.fit(X_train)
        X_transformed = clipper.transform(X_test)

        assert X_transformed["a"].max() < 999

    def test_transform_does_not_modify_input(self):
        """transform() should not modify the input DataFrame in place."""
        X = pd.DataFrame({"a": [1.0, 2.0, 100.0]})
        X_original = X.copy()
        clipper = OutlierClipper(upper_percentile=75)
        clipper.fit(X)
        _ = clipper.transform(X)
        pd.testing.assert_frame_equal(X, X_original)


class TestCreditPipeline:

    def test_pipeline_fit_transform_no_error(self):
        """The credit pipeline should fit and transform without exceptions."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            "LIMIT_BAL": np.random.uniform(10000, 800000, n),
            "AGE": np.random.randint(18, 80, n).astype(float),
            "PAY_0": np.random.randint(-2, 4, n).astype(float),
            "BILL_AMT1": np.where(np.random.random(n) < 0.05, np.nan, np.random.uniform(0, 50000, n)),
        })
        pipeline = build_credit_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == (n, len(X.columns))

    def test_pipeline_no_nans_after_transform(self):
        """Pipeline should produce no NaN values (imputer handles missing values)."""
        X = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, 2.0, 3.0],
        })
        pipeline = build_credit_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert not np.isnan(X_transformed).any()


class TestBinarizeLabels:

    def test_normal_maps_to_zero(self):
        """'normal' label should map to 0."""
        series = pd.Series(["normal", "normal", "normal"])
        label_map = {"normal": 0, "attack": 1}
        result = binarize_labels(series, label_map)
        assert (result == 0).all()

    def test_attacks_map_to_one(self):
        """Non-normal labels should map to 1."""
        series = pd.Series(["neptune", "smurf", "back", "portsweep"])
        label_map = {"normal": 0, "attack": 1}
        result = binarize_labels(series, label_map)
        assert (result == 1).all()

    def test_mixed_labels(self):
        """Mixed series should have correct binary values."""
        series = pd.Series(["normal", "neptune", "normal", "smurf"])
        label_map = {"normal": 0, "attack": 1}
        result = binarize_labels(series, label_map)
        expected = pd.Series([0, 1, 0, 1])
        pd.testing.assert_series_equal(result, expected)


class TestNetworkPipeline:

    def test_pipeline_handles_categorical_features(self):
        """Network pipeline should one-hot encode categorical features."""
        X = pd.DataFrame({
            "duration": [0.0, 1.0, 2.0],
            "src_bytes": [100.0, 200.0, 300.0],
            "protocol_type": ["tcp", "udp", "icmp"],
        })
        numeric = ["duration", "src_bytes"]
        categorical = ["protocol_type"]
        pipeline = build_network_pipeline(numeric, categorical)
        X_transformed = pipeline.fit_transform(X)

        # 2 numeric + 3 one-hot columns (tcp, udp, icmp) = 5 columns
        assert X_transformed.shape[1] == 5


class TestEvaluation:

    def test_evaluate_model_returns_all_metrics(self):
        """evaluate_model should return all required metric keys."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.choice([0, 1], 200, p=[0.7, 0.3])

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        result = evaluate_model(
            model=model,
            X_test=X,
            y_test=y,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            domain="test",
            model_name="test_lr",
        )

        required_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert required_keys.issubset(result["scalar_metrics"].keys())

    def test_evaluate_model_auc_is_valid(self):
        """ROC-AUC should be between 0 and 1."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.choice([0, 1], 200)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        result = evaluate_model(model, X, y, ["f1", "f2", "f3", "f4", "f5"], "test", "lr")
        auc = result["scalar_metrics"]["roc_auc"]
        assert 0.0 <= auc <= 1.0

    def test_find_optimal_threshold_returns_valid_range(self):
        """Optimal threshold should be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_prob = np.random.uniform(0, 1, 100)

        threshold, f1 = _find_optimal_threshold(y_true, y_prob)
        assert 0.0 < threshold < 1.0
        assert 0.0 <= f1 <= 1.0
