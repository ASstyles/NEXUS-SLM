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
