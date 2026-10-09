"""Tests for model versioning, registry, rollback, and data drift monitoring."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_model_info_endpoint():
    resp = client.get("/model")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_version" in data
    assert "base_model" in data
    assert "model_hash" in data


def test_model_registry_list():
    resp = client.get("/registry")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)
    versions = [m["version"] for m in models]
    assert "nexus-qlora-v1" in versions


def test_rollback_to_registered_version():
    # 1. Switch to baseline-smollm-135m
    resp = client.post("/model/rollback?target_version=baseline-smollm-135m")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["active_version"] == "baseline-smollm-135m"

    # 2. Verify health reflects rollback
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    assert h_resp.json()["model_version"] == "baseline-smollm-135m"

    # 3. Verify inference works under rolled back state
    pred_resp = client.post("/predict", json={"ticket": "Mera order deliver nahi hua abhi tak."})
    assert pred_resp.status_code == 200
    assert pred_resp.json()["model_version"] == "baseline-smollm-135m"

    # 4. Switch back to production candidate nexus-qlora-v1
    resp_back = client.post("/model/rollback?target_version=nexus-qlora-v1")
    assert resp_back.status_code == 200
    assert resp_back.json()["active_version"] == "nexus-qlora-v1"

    # 5. Verify restored
    h_restored = client.get("/health")
    assert h_restored.json()["model_version"] == "nexus-qlora-v1"


def test_rollback_nonexistent_version_rejected():
    resp = client.post("/model/rollback?target_version=nonexistent-v999")
    assert resp.status_code == 404
    assert "not registered" in resp.json()["detail"].lower()


def test_data_drift_endpoint():
    resp = client.get("/monitoring/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "language_psi" in data
    assert "category_psi" in data
    assert "drift_detected" in data


def test_monitoring_alerts_endpoint():
    resp = client.get("/monitoring/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_alerts" in data
    assert "status" in data
