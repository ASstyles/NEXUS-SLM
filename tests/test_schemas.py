"""Unit tests for Pydantic data schemas and validation rules."""

import pytest
from pydantic import ValidationError
from src.data.schemas import TicketRecord, TriagePrediction


def test_ticket_record_valid():
    rec = TicketRecord(
        id="ticket_000001",
        text="Mera payment kat gaya but order confirm nahi hua.",
        language="hinglish",
        category="payment_failure",
        priority="high",
        requires_human=True,
        source="synthetic",
        validated=True,
    )
    assert rec.id == "ticket_000001"
    assert rec.category == "payment_failure"
    assert rec.language == "hinglish"
    assert rec.requires_human is True


def test_ticket_record_empty_text():
    with pytest.raises(ValidationError):
        TicketRecord(
            id="ticket_000002",
            text="   ",
            language="english",
            category="refund_request",
            priority="low",
            requires_human=False,
        )


def test_ticket_record_invalid_category():
    with pytest.raises(ValidationError):
        TicketRecord(
            id="ticket_000003",
            text="Valid text",
            language="english",
            category="unknown_category_xxx",  # Invalid
            priority="low",
            requires_human=False,
        )


def test_triage_prediction_schema():
    pred = TriagePrediction(
        category="fraud_report",
        priority="critical",
        language="hindi",
        requires_human=True,
        confidence=0.98,
        latency_ms=120.5,
    )
    assert pred.category == "fraud_report"
    assert pred.priority == "critical"
    assert pred.confidence == 0.98
    assert pred.is_valid is True
    assert pred.error_message is None


def test_ticket_record_normalizes_roman_hindi():
    """Verify TicketRecord normalizes roman_hindi to hinglish without validation error."""
    rec = TicketRecord(
        id="ticket_v2_001",
        text="Mera order deliver nahi hua bhai.",
        language="roman_hindi",
        category="delivery_issue",
        priority="medium",
        requires_human=False,
    )
    assert rec.language == "hinglish"


def test_triage_prediction_invalid_output_contract():
    """Verify TriagePrediction supports is_valid=False with nullable confidence."""
    pred = TriagePrediction(
        category="general_query",
        priority="medium",
        language="english",
        requires_human=True,
        confidence=None,
        is_valid=False,
        error_message="JSON syntax error: Unterminated string",
    )
    assert pred.is_valid is False
    assert pred.confidence is None
    assert "JSON syntax error" in pred.error_message

