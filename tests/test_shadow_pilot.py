"""Comprehensive tests for the shadow pilot pipeline, privacy redaction, and evaluation protocol."""

import asyncio
import json
import pytest
from pathlib import Path
from src.pilot.privacy_redactor import PrivacyRedactor
from src.pilot.shadow_pipeline import ShadowPipeline
from src.pilot.evaluation_protocol import PilotEvaluationProtocol, compute_wilson_ci


# ==============================================================================
# 1. SAFETY & ZERO CUSTOMER ACTION ENFORCEMENT
# ==============================================================================

def test_customer_facing_action_prevention():
    """Verify that shadow pipeline strictly blocks any execution of customer-impacting actions."""
    pipeline = ShadowPipeline()
    with pytest.raises(PermissionError) as exc_info:
        pipeline.execute_customer_action("refund", ticket_id="TKT-100", amount=50.0)
    assert "CRITICAL SAFETY VIOLATION" in str(exc_info.value)
    assert "shadow mode" in str(exc_info.value)


# ==============================================================================
# 2. PRIVACY & PII REDACTION TESTS
# ==============================================================================

def test_pii_credit_card_and_cvv_redaction():
    """Verify payment cards and CVVs are sanitized before shadow processing."""
    text = "My visa card 4532 1122 3344 5566 with cvv: 492 was charged unexpectedly."
    sanitized, counts = PrivacyRedactor.redact_sensitive_data(text)
    assert "4532 1122 3344 5566" not in sanitized
    assert "492" not in sanitized
    assert "[CARD_REDACTED_ENDING_5566]" in sanitized
    assert "[CVV_REDACTED]" in sanitized
    assert counts["credit_card"] == 1
    assert counts["cvv"] == 1


def test_pii_email_and_phone_redaction():
    """Verify customer contact information is sanitized."""
    text = "Contact me at customer.support@example.com or mobile +1-555-839-2019."
    sanitized, counts = PrivacyRedactor.redact_sensitive_data(text)
    assert "customer.support@example.com" not in sanitized
    assert "555-839-2019" not in sanitized
    assert "[EMAIL_REDACTED]" in sanitized
    assert "[PHONE_REDACTED]" in sanitized
    assert counts["email"] == 1
    assert counts["phone"] == 1


def test_secrets_and_bearer_token_redaction():
    """Verify API tokens and secrets are stripped."""
    text = "Here is my authorization token: bearer sk_live_992019283019283019283 for debugging."
    sanitized, counts = PrivacyRedactor.redact_sensitive_data(text)
    assert "sk_live_992019283019283019283" not in sanitized
    assert "[SECRET_TOKEN_REDACTED]" in sanitized
    assert counts["api_key"] == 1


# ==============================================================================
# 3. PIPELINE RESILIENCE & DEDUPLICATION TESTS
# ==============================================================================

def test_deduplication_of_replayed_events():
    """Verify duplicate webhooks and replay attacks are dropped."""
    pipeline = ShadowPipeline(max_queue_size=10)
    status1, ok1 = pipeline.ingest_ticket("TKT-001", "Order package is delayed.")
    assert status1 == "ENQUEUED"
    assert ok1 is True

    # Same ticket content and ID replayed
    status2, ok2 = pipeline.ingest_ticket("TKT-001", "Order package is delayed.")
    assert status2 == "DUPLICATE_DROPPED"
    assert ok2 is False
    assert pipeline.metrics["duplicates_dropped"] == 1


def test_queue_overflow_backpressure_drop():
    """Verify backpressure drop when shadow queue fills up to protect system."""
    pipeline = ShadowPipeline(max_queue_size=2)
    pipeline.ingest_ticket("TKT-01", "First ticket.")
    pipeline.ingest_ticket("TKT-02", "Second ticket.")

    # Third ticket exceeds capacity
    status3, ok3 = pipeline.ingest_ticket("TKT-03", "Third ticket overflowing.")
    assert status3 == "BACKPRESSURE_DROPPED"
    assert ok3 is False
    assert pipeline.metrics["backpressure_drops"] == 1


@pytest.mark.asyncio
async def test_authoritative_workflow_unaffected_by_model_timeout():
    """Verify model hang/timeout does not crash worker and records timeout cleanly."""
    def hanging_predictor(text: str):
        import time
        time.sleep(1.0)  # Simulates slow model
        return None

    # Pipeline with 0.1s timeout threshold
    pipeline = ShadowPipeline(predictor_fn=hanging_predictor, timeout_seconds=0.1)
    payload = {
        "ticket_id": "TKT-TIMEOUT-1",
        "redacted_text": "Slow inference ticket inquiry.",
    }

    result = await pipeline.process_item(payload)
    assert result["status"] == "TIMEOUT"
    assert result["requires_human"] is True
    assert result["escalation_reason"] == "inference_timeout_isolation"
    assert pipeline.metrics["timeouts"] == 1


@pytest.mark.asyncio
async def test_escalation_audit_record_creation(tmp_path):
    """Verify every requires_human=True prediction generates an audit record."""
    audit_file = tmp_path / "escalations.jsonl"
    pred_file = tmp_path / "predictions.jsonl"

    class FakePred:
        def __init__(self):
            self.category = "fraud_report"
            self.priority = "critical"
            self.language = "english"
            self.confidence = 0.99
            self.latency_ms = 45.0

    pipeline = ShadowPipeline(
        predictor_fn=lambda t: FakePred(),
        audit_log_path=str(audit_file),
        predictions_log_path=str(pred_file),
    )

    payload = {
        "ticket_id": "TKT-FRAUD-99",
        "redacted_text": "Someone stole my wallet and made unauthorized transactions.",
    }

    res = await pipeline.process_item(payload)
    assert res["requires_human"] is True
    assert res["escalation_reason"] == "safety_critical_fraud_detection"
    assert audit_file.exists()

    with open(audit_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticket_id"] == "TKT-FRAUD-99"
    assert record["requires_human"] is True


# ==============================================================================
# 4. DOUBLE-BLIND EVALUATION PROTOCOL TESTS
# ==============================================================================

def test_double_blind_labeling_and_ground_truth_freeze(tmp_path):
    """Verify reviewers record blinded labels and freeze ground truth with SHA256 manifest."""
    blind_path = tmp_path / "blinded.jsonl"
    frozen_path = tmp_path / "frozen.jsonl"

    protocol = PilotEvaluationProtocol(
        blinded_labels_path=str(blind_path),
        frozen_labels_path=str(frozen_path),
    )

    # 1. Reviewer records label without model output
    r1 = protocol.record_blinded_human_label(
        ticket_id="TKT-001",
        sanitized_text="Where is my shipment?",
        human_category="order_tracking",
        human_priority="medium",
        human_requires_human=False,
        reviewer_id="human_expert_01",
    )
    assert r1["human_category"] == "order_tracking"

    # 2. Freeze labels
    manifest = protocol.freeze_pilot_ground_truth()
    assert manifest["frozen_samples_count"] == 1
    assert "sha256_hash" in manifest
    assert frozen_path.exists()

    # 3. Evaluate Concordance
    gt_records = [r1]
    shadow_preds = {
        "TKT-001": {
            "predicted_category": "order_tracking",
            "requires_human": False,
            "confidence": 0.95,
        }
    }
    eval_res = protocol.evaluate_shadow_concordance(gt_records, shadow_preds)
    assert eval_res["classification_concordance"]["category_agreement_rate"] == 1.0
    assert eval_res["escalation_safety_concordance"]["sev1_safety_escapes_count"] == 0


def test_wilson_confidence_interval_math():
    """Verify Wilson confidence interval computation."""
    low, high = compute_wilson_ci(k=50, n=100)
    assert 0.40 <= low <= 0.45
    assert 0.58 <= high <= 0.62

    # Edge cases
    low_zero, high_zero = compute_wilson_ci(k=0, n=50)
    assert low_zero == 0.0
    assert high_zero > 0.0
