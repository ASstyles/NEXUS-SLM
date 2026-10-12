"""Rigorous tests for the strict inference API contract and safety escalation policy."""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

client = TestClient(app)


def test_contract_input_with_ticket_field():
    """Validates that input using the strict 'ticket' key works as specified in contract."""
    resp = client.post("/predict", json={
        "ticket": "My bank account was debited twice for order #9821."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "category" in data
    assert "priority" in data
    assert "language" in data
    assert "requires_human" in data
    assert "confidence" in data
    assert data["category"] in CATEGORIES
    assert data["priority"] in PRIORITIES
    assert data["language"] in LANGUAGES
    assert data["confidence"] is None or (0.0 <= data["confidence"] <= 1.0)
    assert data["confidence_status"] in ("uncalibrated_heuristic", "unavailable")


def test_contract_rejects_empty_ticket():
    resp = client.post("/predict", json={"ticket": ""})
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data or "detail" in data


def test_contract_rejects_whitespace_ticket():
    resp = client.post("/predict", json={"ticket": "    \n\t  "})
    assert resp.status_code == 422


def test_contract_rejects_missing_field():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_contract_rejects_non_string_ticket():
    resp = client.post("/predict", json={"ticket": 98765})
    assert resp.status_code == 422


def test_contract_rejects_oversized_ticket():
    huge_text = "This is a long ticket. " * 300  # > 6000 chars
    resp = client.post("/predict", json={"ticket": huge_text})
    assert resp.status_code == 422


def test_contract_supports_native_hindi():
    resp = client.post("/predict", json={
        "ticket": "मेरा खाता ब्लॉक हो गया है, कृपया अनब्लॉक करें।"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["language"] == "hindi"
    assert data["category"] in CATEGORIES


def test_contract_supports_hinglish():
    resp = client.post("/predict", json={
        "ticket": "Bhai mere paise kat gaye lekin recharge nahi hua."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["language"] == "hinglish"
    assert data["category"] in CATEGORIES


def test_contract_supports_roman_hindi():
    resp = client.post("/predict", json={
        "ticket": "Mujhe mera refund kab milega? Do din ho gaye."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] in CATEGORIES


def test_contract_supports_unicode_and_emojis():
    resp = client.post("/predict", json={
        "ticket": "Package arrived damaged! 📦💥 Broken glass everywhere 😡"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] in CATEGORIES


def test_escalation_fraud_ticket():
    resp = client.post("/predict", json={
        "ticket": "Someone hacked into my account and transferred $2,000 without my authorization!"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "fraud_report"
    assert data["priority"] == "critical"
    assert data["requires_human"] is True
    assert "fraud" in data.get("escalation_reason", "")


def test_escalation_legal_and_account_closure():
    resp = client.post("/predict", json={
        "ticket": "Please delete my account and erase all my personal data under GDPR."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["requires_human"] is True


def test_escalation_short_ticket():
    resp = client.post("/predict", json={
        "ticket": "Payment failed"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["requires_human"] is True
    assert "short_input" in data.get("escalation_reason", "")
