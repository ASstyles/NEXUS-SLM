"""Unit tests for Data Quality, Deduplication, and Leakage Detection."""

import pytest
from src.data.deduplication import Deduplicator
from src.data.leakage_detector import LeakageDetector
from src.data.schemas import TicketRecord
from src.data.splitter import DatasetSplitter
from src.data.validator import DataValidator


def test_validator_normalization():
    raw = "   Mera   payment \t\t kat gaya...   "
    clean = DataValidator.normalize_text(raw)
    assert clean == "Mera payment kat gaya..."


def test_validator_rejects_missing_field():
    validator = DataValidator()
    ticket, res = validator.validate_record({
        "id": "t1",
        "text": "Valid text",
        "category": "payment_failure",
        # missing priority, language, requires_human
    })
    assert res.valid is False
    assert len(res.errors) >= 3


def test_deduplicator_exact_and_near():
    dedup = Deduplicator(similarity_threshold=0.80)
    records = [
        TicketRecord(id="t1", text="Payment failed at bank gateway.", language="english", category="payment_failure", priority="high", requires_human=True),
        TicketRecord(id="t2", text="Payment failed at bank gateway.", language="english", category="payment_failure", priority="high", requires_human=True),  # Exact duplicate
        TicketRecord(id="t3", text="Payment failed at bank gateway!", language="english", category="payment_failure", priority="high", requires_human=True),  # Near duplicate
        TicketRecord(id="t4", text="Where is my parcel? Need tracking update.", language="english", category="order_tracking", priority="low", requires_human=False),
    ]

    unique, dup_log = dedup.deduplicate(records)
    assert len(unique) == 2  # Retains t1 and t4
    assert len(dup_log) == 2


def test_leakage_detector_zero_tolerance():
    detector = LeakageDetector(similarity_threshold=0.80)
    train_records = [
        TicketRecord(id="tr1", text="My payment failed and money debited.", language="english", category="payment_failure", priority="high", requires_human=True),
    ]
    val_records = []
    test_records = [
        TicketRecord(id="te1", text="My payment failed and money debited.", language="english", category="payment_failure", priority="high", requires_human=True),
    ]

    report = detector.detect_leakage(train_records, val_records, test_records)
    assert report.leakage_detected is True
    assert report.exact_duplicates_across_splits == 1

    # Sanitization purge test
    sanitized, purged = detector.sanitize_train_against_locked_test(train_records, test_records)
    assert purged == 1
    assert len(sanitized) == 0


def test_splitter_ratios():
    splitter = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    records = [
        TicketRecord(
            id=f"t_{i}",
            text=f"Sample text number {i} for testing splits",
            language="english" if i % 2 == 0 else "hinglish",
            category="payment_failure" if i % 2 == 0 else "refund_request",
            priority="medium",
            requires_human=False,
        )
        for i in range(100)
    ]
    train, val, test = splitter.split(records)
    assert len(train) + len(val) + len(test) == 100
    assert len(train) >= 75
    assert len(test) >= 8
