"""Integration tests for the FastAPI prediction endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.utils.config import get_project_root

PROJECT_ROOT = get_project_root()

client = TestClient(app)


class TestHealthEndpoint:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_expected_keys(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "models_available" in data


# Minimal valid UCI credit risk payload (for reuse across tests)
_VALID_CREDIT_PAYLOAD = {
    "LIMIT_BAL": 200000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 3913.0, "BILL_AMT2": 3102.0, "BILL_AMT3": 689.0,
    "BILL_AMT4": 0.0, "BILL_AMT5": 0.0, "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0, "PAY_AMT2": 689.0, "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0,
}


class TestCreditRiskSchema:

    def test_invalid_payload_returns_422(self):
        """Missing all required fields should return HTTP 422."""
        response = client.post(
            "/api/v1/predict/credit-risk",
            json={"AGE": "not_a_number"},
        )
        assert response.status_code == 422

    def test_invalid_age_returns_422(self):
        """AGE below 18 should fail Pydantic validation."""
        payload = {**_VALID_CREDIT_PAYLOAD, "AGE": 10}
        response = client.post("/api/v1/predict/credit-risk", json=payload)
        assert response.status_code == 422

    def test_invalid_sex_returns_422(self):
        """SEX outside [1, 2] should fail Pydantic validation."""
        payload = {**_VALID_CREDIT_PAYLOAD, "SEX": 5}
        response = client.post("/api/v1/predict/credit-risk", json=payload)
        assert response.status_code == 422

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "models" / "credit_risk" / "xgboost_model.joblib").exists(),
        reason="Model not trained yet — run: python -m src.models.train --domain credit_risk",
    )
    def test_valid_prediction_returns_200(self):
        """A valid request should return 200 with all required prediction fields."""
        response = client.post("/api/v1/predict/credit-risk", json=_VALID_CREDIT_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0


class TestNetworkIntrusionSchema:

    def test_invalid_protocol_returns_422(self):
        """Invalid protocol_type should fail Pydantic validation."""
        payload = {
            "duration": 0.0,
            "protocol_type": "invalid_protocol",
            "service": "http",
            "flag": "SF",
            "src_bytes": 215.0,
            "dst_bytes": 45076.0,
            "logged_in": 1,
            "count": 1.0,
            "srv_count": 1.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "dst_host_count": 9.0,
            "dst_host_srv_count": 9.0,
            "serror_rate": 0.0,
            "rerror_rate": 0.0,
        }
        response = client.post("/api/v1/predict/network-intrusion", json=payload)
        assert response.status_code == 422

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "models" / "network_intrusion" / "xgboost_model.joblib").exists(),
        reason="Model not trained yet — run: python -m src.models.train --domain network_intrusion",
    )
    def test_valid_prediction_returns_200(self):
        """A valid request should return 200 with prediction fields."""
        payload = {
            "duration": 0.0,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 215.0,
            "dst_bytes": 45076.0,
            "logged_in": 1,
            "count": 1.0,
            "srv_count": 1.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "dst_host_count": 9.0,
            "dst_host_srv_count": 9.0,
            "serror_rate": 0.0,
            "rerror_rate": 0.0,
        }
        response = client.post("/api/v1/predict/network-intrusion", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
