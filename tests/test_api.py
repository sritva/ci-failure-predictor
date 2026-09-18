"""Tests for FastAPI CI failure predictor inference service."""

import pytest
from fastapi.testclient import TestClient

from src.service.main import app, ensure_loaded


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient with loaded model artifacts."""
    ensure_loaded()
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Test GET / returns basic metadata and documentation routes."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ci-failure-predictor"
    assert "model_version" in data
    assert data["operating_threshold"] == 0.73
    assert data["predict_url"] == "/predict"


def test_health_endpoint(client: TestClient):
    """Test GET /health returns 200 with ok status and model_version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "1.0.0-xgb"


def test_predict_valid_request(client: TestClient):
    """Test valid POST /predict returns 200 and schema-compliant response."""
    payload = {
        "diff_size": 250,
        "files_changed": 8,
        "test_files_changed": 0,
        "touches_test_files": False,
        "time_of_day": 23,
        "day_of_week": 5,
        "is_weekend": True,
        "author_past_run_count": 3,
        "author_past_failure_rate": 0.6,
        "repo_recent_failure_rate": 0.4,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "failure_risk_score" in data
    assert 0.0 <= data["failure_risk_score"] <= 1.0
    assert data["risk_label"] in ["low", "medium", "high"]
    assert data["model_version"] == "1.0.0-xgb"

    # Top risk factors validation
    top_factors = data["top_risk_factors"]
    assert isinstance(top_factors, list)
    assert len(top_factors) == 3
    for factor in top_factors:
        assert "feature" in factor
        assert "contribution" in factor
        assert isinstance(factor["feature"], str)
        assert isinstance(factor["contribution"], (int, float))


def test_predict_out_of_range_failure_rate(client: TestClient):
    """Test author_past_failure_rate > 1.0 triggers 422 Unprocessable Entity."""
    payload = {
        "diff_size": 100,
        "files_changed": 2,
        "test_files_changed": 1,
        "touches_test_files": True,
        "time_of_day": 14,
        "day_of_week": 2,
        "is_weekend": False,
        "author_past_run_count": 5,
        "author_past_failure_rate": 1.5,  # Invalid: > 1.0
        "repo_recent_failure_rate": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_out_of_range_time_of_day(client: TestClient):
    """Test time_of_day out of [0, 23] triggers 422 Unprocessable Entity."""
    payload = {
        "diff_size": 100,
        "files_changed": 2,
        "test_files_changed": 1,
        "touches_test_files": True,
        "time_of_day": 25,  # Invalid: > 23
        "day_of_week": 2,
        "is_weekend": False,
        "author_past_run_count": 5,
        "author_past_failure_rate": 0.1,
        "repo_recent_failure_rate": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_out_of_range_day_of_week(client: TestClient):
    """Test day_of_week out of [0, 6] triggers 422 Unprocessable Entity."""
    payload = {
        "diff_size": 100,
        "files_changed": 2,
        "test_files_changed": 1,
        "touches_test_files": True,
        "time_of_day": 12,
        "day_of_week": 7,  # Invalid: > 6
        "is_weekend": False,
        "author_past_run_count": 5,
        "author_past_failure_rate": 0.1,
        "repo_recent_failure_rate": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field(client: TestClient):
    """Test omitting a required field triggers 422 Unprocessable Entity."""
    payload = {
        "diff_size": 100,
        "files_changed": 2,
        # test_files_changed is missing
        "touches_test_files": True,
        "time_of_day": 12,
        "day_of_week": 3,
        "is_weekend": False,
        "author_past_run_count": 5,
        "author_past_failure_rate": 0.1,
        "repo_recent_failure_rate": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_high_risk_scores_higher_than_low_risk(client: TestClient):
    """Sanity-checks the real model: high-risk PR inputs score higher than low-risk PR inputs."""
    # Benign PR: small changes, includes tests, low author/repo failure history
    low_risk_payload = {
        "diff_size": 15,
        "files_changed": 1,
        "test_files_changed": 1,
        "touches_test_files": True,
        "time_of_day": 11,
        "day_of_week": 2,
        "is_weekend": False,
        "author_past_run_count": 40,
        "author_past_failure_rate": 0.02,
        "repo_recent_failure_rate": 0.05,
    }

    # Risky PR: massive diff, no tests, high author failure rate, high recent repo failure rate
    high_risk_payload = {
        "diff_size": 4500,
        "files_changed": 50,
        "test_files_changed": 0,
        "touches_test_files": False,
        "time_of_day": 23,
        "day_of_week": 6,
        "is_weekend": True,
        "author_past_run_count": 2,
        "author_past_failure_rate": 0.85,
        "repo_recent_failure_rate": 0.65,
    }

    low_resp = client.post("/predict", json=low_risk_payload)
    high_resp = client.post("/predict", json=high_risk_payload)

    assert low_resp.status_code == 200
    assert high_resp.status_code == 200

    low_score = low_resp.json()["failure_risk_score"]
    high_score = high_resp.json()["failure_risk_score"]

    assert high_score > low_score, (
        f"High risk score ({high_score}) must be greater than low risk score ({low_score})"
    )
    assert low_resp.json()["risk_label"] == "low"
    assert high_resp.json()["risk_label"] == "high"


def test_risk_label_boundary_bucketing(client: TestClient, monkeypatch):
    """Test risk_label boundary behavior around operating_threshold (0.73) and half-threshold (0.365)."""
    from unittest.mock import MagicMock
    import numpy as np
    from src.service.main import app_state

    # Verify boundary logic for low (<0.365), medium ([0.365, 0.73)), high (>=0.73)
    mock_model = MagicMock()
    original_model = app_state["model"]

    payload = {
        "diff_size": 100,
        "files_changed": 2,
        "test_files_changed": 1,
        "touches_test_files": True,
        "time_of_day": 12,
        "day_of_week": 3,
        "is_weekend": False,
        "author_past_run_count": 5,
        "author_past_failure_rate": 0.1,
        "repo_recent_failure_rate": 0.2,
    }

    test_cases = [
        (0.7300, "high"),    # Exact operating threshold
        (0.7301, "high"),    # Just above operating threshold
        (0.7299, "medium"),  # Just below operating threshold
        (0.3650, "medium"),  # Exact half-threshold
        (0.3651, "medium"),  # Just above half-threshold
        (0.3649, "low"),     # Just below half-threshold
        (0.0000, "low"),     # Lower bound
        (1.0000, "high"),    # Upper bound
    ]

    try:
        app_state["model"] = mock_model
        for prob, expected_label in test_cases:
            mock_model.predict_proba.return_value = np.array([[1.0 - prob, prob]])
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["risk_label"] == expected_label, (
                f"Prob {prob} expected label '{expected_label}', got '{data['risk_label']}'"
            )
    finally:
        app_state["model"] = original_model
