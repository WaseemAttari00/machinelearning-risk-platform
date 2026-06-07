"""
Integration tests for the FastAPI endpoints.

These tests use FastAPI's TestClient (powered by httpx) to test the API
without starting a real server. The TestClient creates an in-process ASGI
app and calls it directly — much faster than spinning up a real server.

Note: Tests that call prediction endpoints will fail if models aren't trained.
We mark those with @pytest.mark.skipif to avoid false failures in CI.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.utils.config import get_project_root

PROJECT_ROOT = get_project_root()

# TestClient is a synchronous wrapper around the FastAPI app.
# It handles startup/shutdown events automatically.
client = TestClient(app)


class TestHealthEndpoint:

    def test_root_returns_200(self):
        """The root endpoint should return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_returns_200(self):
        """The health endpoint should always return HTTP 200 (even if models aren't loaded)."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_expected_keys(self):
        """Health response should contain required keys."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "models_available" in data


class TestCreditRiskSchema:

    def test_invalid_payload_returns_422(self):
        """Missing required fields should return HTTP 422 (Unprocessable Entity)."""
        response = client.post(
            "/api/v1/predict/credit-risk",
            json={"age": "not_a_number"},  # Wrong type + missing required fields
        )
        assert response.status_code == 422

    def test_invalid_age_returns_422(self):
        """Age below 18 should fail Pydantic validation."""
        payload = {
            "RevolvingUtilizationOfUnsecuredLines": 0.5,
            "age": 10,  # Below minimum of 18
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.3,
            "MonthlyIncome": 5000.0,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 0,
        }
        response = client.post("/api/v1/predict/credit-risk", json=payload)
        assert response.status_code == 422


class TestNetworkIntrusionSchema:

    def test_invalid_protocol_returns_422(self):
        """Invalid protocol_type should fail Pydantic validation."""
        payload = {
            "duration": 0.0,
            "protocol_type": "invalid_protocol",  # Not tcp/udp/icmp
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
