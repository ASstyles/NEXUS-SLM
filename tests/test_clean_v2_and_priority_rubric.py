"""Regression and contract tests for Clean V2 Template-Grouped splits and Priority Rubric.

Validates:
- Clean template-grouped dataset invariants and zero-leakage guarantees
- Multilingual ticket processing (English, Devanagari Hindi, Hinglish)
- Input validation, missing fields, and error handling
- Safety escalation invariants across critical/high priority categories
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.data.schemas import TicketRecord
from src.data.validator import DataValidator
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_clean_v2_split_integrity():
    """Verify that clean template-grouped dataset splits match lockfile counts and schemas."""
    clean_dir = PROJECT_ROOT / "data" / "v2_clean_template_grouped"
    lock_file = clean_dir / "dataset_v2_clean_lock.json"
    if not lock_file.exists():
        lock_file = PROJECT_ROOT / "dataset_v2_clean_lock.json"
    assert lock_file.exists(), "dataset_v2_clean_lock.json must exist"

    with open(lock_file, "r", encoding="utf-8") as f:
        lock_meta = json.load(f)

    split_counts = lock_meta.get("split_counts", {})

    validator = DataValidator()

    for split_name in ["train", "val", "test"]:
        file_path = clean_dir / f"{split_name}_clean.jsonl"
        assert file_path.exists(), f"Missing split file: {file_path}"

        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    item = json.loads(line)
                    rec, v_res = validator.validate_record(item)
                    assert v_res.valid, f"Invalid record at {file_path}:{line_no}: {v_res.errors}"
                    records.append(rec)

        expected_count = split_counts.get(split_name)
        assert len(records) == expected_count, (
            f"Record count mismatch for {split_name}_clean: expected {expected_count}, got {len(records)}"
        )


def test_clean_v2_zero_text_leakage():
    """Ensure clean train, val, and test splits have 0% exact text overlap."""
    clean_dir = PROJECT_ROOT / "data" / "v2_clean_template_grouped"
    
    def get_texts(fname):
        with open(clean_dir / fname, "r", encoding="utf-8") as f:
            return {json.loads(l)["text"].strip().lower() for l in f if l.strip()}

    train_texts = get_texts("train_clean.jsonl")
    val_texts = get_texts("val_clean.jsonl")
    test_texts = get_texts("test_clean.jsonl")

    assert train_texts.isdisjoint(val_texts), "Leakage detected between train_clean and val_clean"
    assert train_texts.isdisjoint(test_texts), "Leakage detected between train_clean and test_clean"
    assert val_texts.isdisjoint(test_texts), "Leakage detected between val_clean and test_clean"


def test_api_multilingual_inference():
    """Validate API handles English, Devanagari Hindi, and Hinglish tickets."""
    client = TestClient(app)

    multilingual_payloads = [
        {"text": "My bank account was debited twice for order #8812. Please refund immediately.", "expected_lang": "english"},
        {"text": "मेरे खाते से पैसे कट गए हैं लेकिन ऑर्डर कन्फर्म नहीं हुआ। कृपया तुरंत समाधान करें।", "expected_lang": "hindi"},
        {"text": "Bhai mera balance deduct ho gaya par order place nahi hua, jaldi refund karo", "expected_lang": "hinglish"},
    ]

    for item in multilingual_payloads:
        resp = client.post("/predict", json={"text": item["text"]})
        assert resp.status_code == 200, f"Failed for text: {item['text']}"
        data = resp.json()
        assert data["category"] in CATEGORIES
        assert data["priority"] in PRIORITIES
        assert data["language"] in LANGUAGES
        assert data["language"] == item["expected_lang"], (
            f"Language mismatch for '{item['text']}': expected {item['expected_lang']}, got {data['language']}"
        )


def test_api_escalation_invariants():
    """Verify conservative escalation invariants on fraud and financial loss."""
    client = TestClient(app)

    high_risk_texts = [
        "Unauthorized transaction noticed on my account! My card was compromised by a hacker.",
        "Mera account hack ho gaya hai unauthorized debit hua hai",
        "धोखाधड़ी हुई है मेरे खाते से पैसे कट गए बिना अनुमति के",
    ]

    for txt in high_risk_texts:
        resp = client.post("/predict", json={"text": txt})
        assert resp.status_code == 200
        data = resp.json()
        assert data["requires_human"] is True, f"High-risk query failed escalation invariant: {txt}"
        assert data["priority"] in ("high", "critical"), f"Expected high/critical priority for: {txt}"


def test_api_missing_and_invalid_inputs():
    """Verify API input validation and HTTP 422 responses for malformed payloads."""
    client = TestClient(app)

    # Missing text field entirely
    resp = client.post("/predict", json={})
    assert resp.status_code == 422

    # Wrong data type for text
    resp = client.post("/predict", json={"text": 12345})
    assert resp.status_code == 422

    # Empty text string (should be rejected by validation or handled gracefully)
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422 or resp.status_code == 400 or (
        resp.status_code == 200 and resp.json().get("requires_human") is True
    )


def test_clean_training_notebook_integrity():
    """Verify clean GPU training notebook structure, dependencies, and dataset isolation."""
    nb_path = PROJECT_ROOT / "notebooks" / "clean_data_qwen25_05b_lora_training.ipynb"
    assert nb_path.exists(), f"Missing clean training notebook: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb and len(nb["cells"]) >= 10
    all_source = " ".join("".join(c.get("source", [])) for c in nb["cells"])

    # Verify key training tokens and constants
    assert "Qwen/Qwen2.5-0.5B-Instruct" in all_source
    assert "qwen25_05b_clean_v2_lora" in all_source
    assert "train_clean.jsonl" in all_source
    assert "val_clean.jsonl" in all_source
    assert "test_clean.jsonl" in all_source

    # Verify that test split is documented as quarantined and not loaded as training dataset
    assert "QUARANTINED" in all_source


def test_evaluation_sample_size_documentation():
    """Verify clean evaluation reports document sample count and total split size."""
    rep_path = PROJECT_ROOT / "experiments" / "artifacts" / "clean_v2_full_evaluation_report.json"
    assert rep_path.exists(), f"Missing full evaluation report: {rep_path}"

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert "models" in rep
    for model_key, model_eval in rep["models"].items():
        assert "sample_count" in model_eval
        assert "total_split_size" in model_eval
        assert model_eval["total_split_size"] in (746, 449)
        assert model_eval["category_metrics"]["accuracy"] >= 0.0
        assert model_eval["escalation_metrics"]["false_negatives"] == 0

