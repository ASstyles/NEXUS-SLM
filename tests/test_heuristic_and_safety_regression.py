"""Mandatory regression test suite for heuristic fallback and safety escalation (Sprint B)."""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.inference.rules import (
    detect_language,
    classify_heuristic_category_and_priority,
    get_fallback_prediction,
)
from src.inference.policy import HumanEscalationPolicy

client = TestClient(app)


class TestLanguageDetectionRegression:
    """Verifies elimination of substring matching false positives in language detection."""

    def test_camera_is_not_hinglish(self):
        """'camera' contains substring 'mera' but is pure English."""
        assert detect_language("camera") == "english"
        assert detect_language("The camera has a problem.") == "english"
        assert detect_language("My camera is broken and does not work.") == "english"

    def test_chair_is_not_hinglish(self):
        """'chair' contains substring 'hai' but is pure English."""
        assert detect_language("chair") == "english"
        assert detect_language("The office chair is damaged on delivery.") == "english"
        assert detect_language("Wheelchair access request for the venue.") == "english"

    def test_genuine_hinglish_detected(self):
        """Genuine Romanized Hindi tokens are correctly identified as Hinglish."""
        assert detect_language("Mera payment kat gaya par order confirm nahi hua.") == "hinglish"
        assert detect_language("Order cancel karo bhai please.") == "hinglish"
        assert detect_language("Paise wapas kab tak aayenge?") == "hinglish"

    def test_native_devanagari_detected_as_hindi(self):
        """Tickets containing Devanagari script are classified as Hindi."""
        assert detect_language("मेरा ऑर्डर अभी तक नहीं मिला।") == "hindi"
        assert detect_language("पेमेंट कट गया है लेकिन रसीद नहीं मिली।") == "hindi"


class TestHackathonAndFraudRegression:
    """Verifies that non-fraud compounds like hack-a-thon do not trigger fraud escalation."""

    def test_hackathon_not_fraud_or_critical(self):
        """'I am participating in a hack-a-thon' must NOT be classified as fraud or critical."""
        text = "I am participating in a hack-a-thon and need API credentials."
        cat, prio, req_human = classify_heuristic_category_and_priority(text)
        assert cat != "fraud_report"
        assert prio != "critical"

        # End-to-end API test
        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] != "fraud_report"
        assert data["priority"] != "critical"

    def test_hackathon_compound_variants(self):
        """Tests variants of hackathon words."""
        for phrase in [
            "Hackathon submission issue with my project",
            "Attending a hack-a-thon this weekend",
            "My Hackathons team needs sponsorship info",
            "Doing a quick life-hack for settings",
        ]:
            cat, prio, _ = classify_heuristic_category_and_priority(phrase)
            assert cat != "fraud_report", f"Failed on '{phrase}'"
            assert prio != "critical", f"Failed on '{phrase}'"

    def test_genuine_account_hack_triggers_fraud_and_critical(self):
        """'Someone hacked my account' must trigger fraud_report and critical priority."""
        text = "Someone hacked my account and made unauthorized transactions!"
        cat, prio, req_human = classify_heuristic_category_and_priority(text)
        assert cat == "fraud_report"
        assert prio == "critical"
        assert req_human is True

        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "fraud_report"
        assert data["priority"] == "critical"
        assert data["requires_human"] is True

    def test_hacked_and_stolen_account_policy_escalation(self):
        """Verifies policy escalation on account theft."""
        req_human, reason, _, cat, prio = HumanEscalationPolicy.evaluate_escalation(
            text="I was hacked and someone stole my account credentials",
            category="fraud_report",
            priority="critical",
            language="english",
            inference_mode="heuristic_fallback",
        )
        assert req_human is True
        assert reason == "safety_critical_fraud_detection"
        assert cat == "fraud_report"
        assert prio == "critical"


class TestCategoryPrecedenceRegression:
    """Verifies explicit precedence between subscription, order cancellation, and product complaints."""

    def test_cancel_subscription_classified_as_subscription(self):
        """'Cancel my subscription' must be categorized under subscription, NOT order_cancel."""
        text = "Cancel my subscription immediately before the next auto-renew date."
        cat, prio, _ = classify_heuristic_category_and_priority(text)
        assert cat == "subscription"

        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == 200
        assert resp.json()["category"] == "subscription"

    def test_cancel_order_classified_as_order_cancel(self):
        """'Cancel my order' must be categorized under order_cancel."""
        text = "Cancel my order #89212 placed yesterday."
        cat, prio, _ = classify_heuristic_category_and_priority(text)
        assert cat == "order_cancel"

        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == 200
        assert resp.json()["category"] == "order_cancel"

    def test_camera_broken_classified_as_product_complaint(self):
        """'My camera is broken' must be categorized as product_complaint in English."""
        text = "My camera is broken and the lens is shattered."
        cat, prio, _ = classify_heuristic_category_and_priority(text)
        assert cat == "product_complaint"

        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "product_complaint"
        assert data["language"] == "english"

    def test_camera_has_a_problem_not_fraud(self):
        """'The camera has a problem' must NOT trigger fraud escalation."""
        text = "The camera has a problem with focus."
        cat, prio, _ = classify_heuristic_category_and_priority(text)
        assert cat != "fraud_report"
        assert prio != "critical"

        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == 200
        assert resp.json()["category"] != "fraud_report"


class TestEscalationPrecisionAudit:
    """Verifies that routine non-risky tickets containing ambiguous words are not over-escalated."""

    def test_routine_billing_query_not_escalated_as_financial_dispute(self):
        """Routine question about a monthly invoice should not be escalated as a high-impact dispute."""
        req_human, reason, _, _, _ = HumanEscalationPolicy.evaluate_escalation(
            text="Could you please email me a copy of my tax invoice for last month?",
            category="billing",
            priority="low",
            language="english",
            raw_confidence=0.95,
            inference_mode="neural",
        )
        assert req_human is False
        assert reason == "automated_routing_permitted"

    def test_routine_subscription_inquiry_not_escalated_as_dispute(self):
        """Standard subscription renewal notice query should not escalate as a dispute."""
        req_human, reason, _, _, _ = HumanEscalationPolicy.evaluate_escalation(
            text="How do I change the renewal payment method for my annual subscription?",
            category="subscription",
            priority="medium",
            language="english",
            raw_confidence=0.92,
            inference_mode="neural",
        )
        assert req_human is False
        assert reason == "automated_routing_permitted"
