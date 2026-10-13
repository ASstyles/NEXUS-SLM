"""Integration tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "hardware" in data


def test_version_endpoint():
    resp = client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["system"] == "NEXUS-SLM"
    assert data["taxonomy_categories"] == 14
    assert data["taxonomy_languages"] == 3


def test_labels_endpoint():
    resp = client.get("/labels")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["categories"]) == 14
    assert "payment_failure" in data["categories"]
    assert "fraud_report" in data["categories"]


def test_predict_hinglish_ticket():
    resp = client.post("/predict", json={
        "text": "Mera payment kat gaya but order confirm nahi hua."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "payment_failure"
    assert data["language"] == "hinglish"
    assert data["confidence"] is None or data["confidence"] > 0.0
    assert data["confidence_status"] in ("uncalibrated_heuristic", "unavailable")
    assert "latency_ms" in data


def test_predict_hindi_ticket():
    resp = client.post("/predict", json={
        "text": "मेरा payment कट गया लेकिन order confirm नहीं हुआ।"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "payment_failure"
    assert data["language"] == "hindi"


def test_predict_english_fraud_ticket():
    resp = client.post("/predict", json={
        "text": "Urgent: Unauthorized charges of $450 detected on my card!"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "fraud_report"
    assert data["priority"] == "critical"
    assert data["requires_human"] is True


def test_metrics_telemetry():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert data["total_requests"] >= 1


def test_benchmark_matrix():
    resp = client.get("/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert "columns" in data
    assert "rows" in data


def test_ready_endpoint():
    """Verify Kubernetes readiness probe endpoint."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["ready"] is True
    assert "active_model_version" in data


def test_predict_returns_raw_fields():
    """Verify predict response includes observable raw output fields."""
    resp = client.post("/predict", json={
        "ticket": "Mera payment kat gaya but order confirm nahi hua."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "raw_category" in data
    assert "raw_priority" in data
    assert "raw_language" in data
    assert data["raw_category"] is not None

