"""End-to-end integration tests for local demonstration fixtures, safety policies, and rollback."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

client = TestClient(app)


def test_demo_fixtures_endpoint():
    """Verifies that the API exposes the full synthetic fixtures catalog."""
    resp = client.get("/demo/fixtures")
    assert resp.status_code == 200
    data = resp.json()
    assert "fixtures" in data
    fixtures = data["fixtures"]
    assert len(fixtures) >= 30

    # Verify all 10 categories are represented
    found_categories = {f["category"] for f in fixtures}
    for expected_cat in [
        "payment_failure", "refund_request", "account_access", "password_reset",
        "subscription", "delivery_issue", "technical_issue", "fraud_report",
        "account_closure", "general_query"
    ]:
        assert expected_cat in found_categories

    # Verify all 3 language modalities are represented
    found_languages = {f["language"] for f in fixtures}
    assert {"english", "hinglish", "hindi"}.issubset(found_languages)


def test_native_devanagari_hindi_mandatory_escalation():
    """
    CRITICAL SAFETY REQUIREMENT:
    All native-script Devanagari Hindi tickets MUST deterministically trigger human review
    due to BPE subword fragmentation risks, regardless of model confidence score.
    """
    hindi_samples = [
        "मेरे खाते से पैसे कट गए हैं लेकिन ऑर्डर अभी तक कन्फर्म नहीं हुआ है।",
        "मैंने सामान वापस भेज दिया है, कृपया मेरे रिफंड की स्थिति की जानकारी दें।",
        "मेरा अकाउंट लॉक हो गया है और लॉगिन करने पर ओटीपी प्राप्त नहीं हो रहा है।",
        "मैं अपना पासवर्ड भूल गया हूँ, कृपया पासवर्ड रीसेट करने का लिंक भेजें।",
        "डिलीवरी वाले ने टूटा हुआ पार्सल दिया है और अंदर का सामान गायब है।",
        "क्या ग्राहक सहायता सेवा सप्ताहांत पर भी उपलब्ध रहती है?",
    ]

    for sample in hindi_samples:
        resp = client.post("/predict", json={"ticket": sample})
        assert resp.status_code == 200, f"Failed on sample: {sample}"
        data = resp.json()
        assert data["language"] == "hindi"
        assert data["requires_human"] is True, f"Devanagari sample failed to escalate: {sample}"
        assert data["escalation_reason"] in (
            "native_hindi_devanagari_unreliable_tokenization",
            "safety_critical_fraud_detection",
        )
        assert data["confidence_status"] in ("uncalibrated_heuristic", "unavailable")


def test_fraud_report_critical_escalation():
    """Fraud reports must always escalate to human review and be mapped to critical priority."""
    fraud_samples = [
        "Urgent: Unauthorized charges of $450 detected on my credit card without my OTP or permission!",
        "Fraud alert: Kisi aur ne unauthorized card charge kar liya hai mere account se bina kisi otp ke!",
        "धोखाधड़ी की सूचना: मेरे कार्ड से अनधिकृत लेन-देन हुआ है, तुरंत मेरा खाता सुरक्षित करें।",
    ]

    for sample in fraud_samples:
        resp = client.post("/predict", json={"ticket": sample})
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "fraud_report"
        assert data["priority"] == "critical"
        assert data["requires_human"] is True
        assert data["escalation_reason"] == "safety_critical_fraud_detection"


def test_all_ten_demo_categories_execute_safely():
    """Verifies that all 10 synthetic demo categories process with valid JSON schema output."""
    fixtures_file = Path("data/demo_fixtures.json")
    assert fixtures_file.exists(), "data/demo_fixtures.json fixture file missing"

    with open(fixtures_file, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    for item in catalog["fixtures"]:
        resp = client.post("/predict", json={"ticket": item["text"]})
        assert resp.status_code == 200, f"Prediction failed on fixture {item['id']}"
        data = resp.json()
        assert data["category"] in CATEGORIES
        assert data["priority"] in PRIORITIES
        assert data["language"] in LANGUAGES
        assert isinstance(data["requires_human"], bool)
        assert data["confidence_status"] in ("uncalibrated_heuristic", "unavailable")
        assert data["latency_ms"] is not None
        assert data["model_version"] is not None


def test_zero_downtime_rollback_e2e_cycle():
    """Tests cycling through active models via POST /model/rollback."""
    # 1. Switch to baseline-smollm-135m
    resp_base = client.post("/model/rollback?target_version=baseline-smollm-135m")
    assert resp_base.status_code == 200
    data_base = resp_base.json()
    assert data_base["active_version"] == "baseline-smollm-135m"

    # Verify predict responds with updated model version
    pred_base = client.post("/predict", json={"ticket": "Mera payment kat gaya par confirmation nahi aayi."})
    assert pred_base.status_code == 200
    assert pred_base.json()["model_version"] == "baseline-smollm-135m"

    # 2. Switch back to production nexus-qlora-v1
    resp_v1 = client.post("/model/rollback?target_version=nexus-qlora-v1")
    assert resp_v1.status_code == 200
    assert resp_v1.json()["active_version"] == "nexus-qlora-v1"

    # Verify predict restored
    pred_v1 = client.post("/predict", json={"ticket": "Mera payment kat gaya par confirmation nahi aayi."})
    assert pred_v1.status_code == 200
    assert pred_v1.json()["model_version"] == "nexus-qlora-v1"
