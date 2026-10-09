"""Extended audit validation test suite for NEXUS-SLM.
Covers:
- Malformed model responses and strict schema validation
- Adversarial prompt injection defense execution
- FastAPI error status codes (422 for invalid payloads)
- Model registry integrity and cryptographic hashing
- Hardware detection sanity
- Robustness benchmark subsets
"""

import json
import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.data.deduplication import Deduplicator
from src.data.leakage_detector import LeakageDetector
from src.data.schemas import TicketRecord
from src.evaluation.adversarial import AdversarialBenchmark
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.robustness import RobustnessBenchmark
from src.inference.parser import OutputParser
from src.models.registry import ModelRegistry
from src.utils.hardware import get_hardware_info

client = TestClient(app)


# ==========================================
# 1. MALFORMED JSON & SCHEMA VALIDATION TESTS
# ==========================================

def test_parser_unterminated_json():
    """Test malformed JSON with unterminated brace."""
    raw = '{"category": "payment_failure", "priority": "high", "language": "hindi"'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is False
    assert data is None
    assert "No JSON object found" in err or "syntax error" in err


def test_parser_trailing_comma_json():
    """Test malformed JSON with trailing comma."""
    raw = '{"category": "payment_failure", "priority": "high", "language": "hindi", "requires_human": true,}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is False
    assert "JSON syntax error" in err


def test_parser_json_array_instead_of_dict():
    """Test model generating a JSON array instead of an object."""
    raw = '["payment_failure", "high", "hindi", true]'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is False
    assert "Expected JSON dictionary" in err


def test_parser_boolean_string_conversion():
    """Test parser converts stringified 'true'/'false' to native booleans."""
    raw = '{"category": "payment_failure", "priority": "high", "language": "hindi", "requires_human": "true"}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is True
    assert data["requires_human"] is True

    raw2 = '{"category": "refund_request", "priority": "low", "language": "english", "requires_human": "false"}'
    valid2, data2, err2 = parser.parse_and_validate(raw2)
    assert valid2 is True
    assert data2["requires_human"] is False


def test_parser_invalid_boolean_string():
    """Test parser rejects non-boolean requires_human value."""
    raw = '{"category": "payment_failure", "priority": "high", "language": "hindi", "requires_human": "maybe"}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is False
    assert "must be boolean" in err


def test_parser_extra_fields_ignored_safely():
    """Test that extra fields in JSON do not break required taxonomy fields."""
    raw = '{"category": "payment_failure", "priority": "high", "language": "hindi", "requires_human": true, "extra_notes": "foo"}'
    parser = OutputParser()
    valid, data, err = parser.parse_and_validate(raw)
    assert valid is True
    assert data["category"] == "payment_failure"
    assert data["requires_human"] is True


# ==========================================
# 2. FASTAPI EDGE CASES & STATUS CODES
# ==========================================

def test_api_predict_empty_text():
    """Empty text must return 422 Unprocessable Entity."""
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_api_predict_missing_text_field():
    """Missing text key must return 422 Unprocessable Entity."""
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_api_predict_oversized_text():
    """Text exceeding 4000 characters must return 422 Unprocessable Entity."""
    oversized = "a" * 4001
    resp = client.post("/predict", json={"text": oversized})
    assert resp.status_code == 422


def test_api_predict_returns_inference_mode():
    """Predict response must declare inference_mode."""
    resp = client.post("/predict", json={"text": "Refund my order please"})
    assert resp.status_code == 200
    data = resp.json()
    assert "inference_mode" in data
    assert data["inference_mode"] in ["neural", "heuristic_fallback"]


# ==========================================
# 3. ADVERSARIAL & ROBUSTNESS EXECUTION
# ==========================================

def test_adversarial_benchmark_runs():
    """Verify that adversarial benchmark executes attacks and reports metrics."""
    def dummy_predictor(text: str):
        class Pred:
            category = "general_query"
            raw_output = '{"category": "general_query", "priority": "medium", "language": "english", "requires_human": true}'
            confidence = 0.90
        return Pred()

    adv = AdversarialBenchmark()
    report = adv.run_benchmark(dummy_predictor)
    assert report["total_attacks"] == 6
    assert "resistance_rate" in report
    assert "findings_summary" in report


def test_robustness_benchmark_subsets():
    """Verify that robustness benchmark runs all 9 language/noise subsets."""
    def dummy_predictor(text: str):
        class Pred:
            category = "payment_failure"
            raw_output = '{"category": "payment_failure", "priority": "high", "language": "hinglish", "requires_human": true}'
            confidence = 0.95
        return Pred()

    rob = RobustnessBenchmark()
    report = rob.run_benchmark(dummy_predictor)
    assert "subsets" in report
    assert len(report["subsets"]) == 9
    assert "english_canonical" in report["subsets"]
    assert "hindi_devanagari" in report["subsets"]
    assert "hinglish_roman" in report["subsets"]
    assert "mixed_script_code_switched" in report["subsets"]


# ==========================================
# 4. MODEL REGISTRY & HARDWARE
# ==========================================

def test_model_registry_operations():
    """Verify registering, retrieving, and listing models."""
    reg = ModelRegistry()
    test_ver = "test-nexus-v0.9"
    rec = reg.register_model(
        version=test_ver,
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path="./models/test",
        quantization="4bit_nf4",
        evaluation_results={"macro_f1": 0.88},
        deployment_status="STAGING",
    )
    assert rec["version"] == test_ver
    assert "model_hash" in rec
    assert rec["model_hash"].startswith("sha256_")

    fetched = reg.get_model(test_ver)
    assert fetched is not None
    assert fetched["deployment_status"] == "STAGING"

    models = reg.list_models()
    assert any(m["version"] == test_ver for m in models)


def test_hardware_diagnostics():
    """Verify hardware info structure."""
    hw = get_hardware_info()
    assert "os" in hw
    assert "cpu_count" in hw
    assert "cuda_available" in hw
    assert "ram_total_gb" in hw
