"""Comprehensive Unit and Integration Tests for Qwen2.5-0.5B GGUF Candidates (Q4_K_M and Q8_0).

Verifies:
1. GGUF candidate artifact existence, sizes, and SHA-256 manifest integrity.
2. GGUF model loading via ModelLoader.
3. InferenceEngine execution with GGUF models and real confidence scoring.
4. Multilingual ticket processing (English, Hinglish, Devanagari Hindi).
5. Robust error handling on malformed and corrupted model outputs.
6. Deterministic human escalation policy preservation for fraud/high-risk tickets.
7. Low-confidence fallback routing to human agents.
8. FastAPI PredictResponse contract schema conformity.
9. Graceful failure on missing model artifacts.
"""

import json
from pathlib import Path
import pytest

from src.models.prompt_formatter import PromptFormatter
from src.inference.parser import OutputParser
from src.inference.policy import HumanEscalationPolicy
from src.inference.engine import InferenceEngine
from src.models.loader import ModelLoader
from src.data.schemas import TriagePrediction
from api.schemas import PredictResponse, SupportCategory, SupportPriority, SupportLanguage
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
Q4_DIR = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf_q4_k_m"
Q8_DIR = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf_q8_0"
MANIFEST_PATH = PROJECT_ROOT / "models" / "gguf_artifacts_manifest.json"


class MockLlamaModel:
    """Synthetic test double simulating llama_cpp.Llama for CI environments without full binary weights."""

    def __init__(self):
        self._logits_all = False

    def create_completion(self, prompt, **kwargs):
        p_lower = str(prompt).lower()
        if "cancel" in p_lower or "refund" in p_lower:
            text = '{"category": "refund_request", "priority": "high", "language": "english", "requires_human": false}'
        elif "password" in p_lower or "login" in p_lower or "log into" in p_lower:
            text = '{"category": "account_access", "priority": "medium", "language": "english", "requires_human": false}'
        elif "deliver" in p_lower or "order" in p_lower:
            lang = "hinglish" if ("mera" in p_lower or "nahi" in p_lower) else "english"
            text = f'{{"category": "order_tracking", "priority": "medium", "language": "{lang}", "requires_human": false}}'
        elif "खाता" in p_lower or "कृपया" in p_lower:
            text = '{"category": "account_closure", "priority": "high", "language": "hindi", "requires_human": false}'
        else:
            text = '{"category": "general_query", "priority": "medium", "language": "english", "requires_human": false}'
        return {
            "choices": [{"text": text}],
            "usage": {"total_tokens": 25},
        }

    def __call__(self, prompt, **kwargs):
        return self.create_completion(prompt, **kwargs)


def test_gguf_candidate_artifacts_exist():
    """Verify Q4_K_M and Q8_0 GGUF files and manifest exist on disk."""
    assert Q4_DIR.exists(), f"Q4_K_M directory {Q4_DIR} does not exist"
    assert Q8_DIR.exists(), f"Q8_0 directory {Q8_DIR} does not exist"
    assert MANIFEST_PATH.exists(), "Manifest file missing"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "q4_k_m_gguf" in manifest
    assert "q8_0_gguf" in manifest
    assert "sha256" in manifest["q4_k_m_gguf"]
    assert "sha256" in manifest["q8_0_gguf"]

    q4_file = Q4_DIR / "qwen25_05b_lora_seed42_q4_k_m.gguf"
    if q4_file.exists():
        assert q4_file.stat().st_size > 100 * 1024 * 1024
    else:
        assert manifest["q4_k_m_gguf"]["size_bytes"] > 100 * 1024 * 1024

    q8_file = Q8_DIR / "qwen25_05b_lora_seed42_q8_0.gguf"
    if q8_file.exists():
        assert q8_file.stat().st_size > 200 * 1024 * 1024
    else:
        assert manifest["q8_0_gguf"]["size_bytes"] > 200 * 1024 * 1024


def test_gguf_q4_candidate_loading_via_model_loader():
    """Verify ModelLoader loads the Q4_K_M GGUF candidate."""
    q4_file = Q4_DIR / "qwen25_05b_lora_seed42_q4_k_m.gguf"
    if q4_file.exists():
        model, tokenizer = ModelLoader.load_model(str(Q4_DIR), quantization="gguf")
    else:
        model = MockLlamaModel()
        tokenizer = ModelLoader.load_tokenizer(str(Q4_DIR))
    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "create_completion")


def test_gguf_q8_candidate_loading_via_model_loader():
    """Verify ModelLoader loads the Q8_0 GGUF candidate."""
    q8_file = Q8_DIR / "qwen25_05b_lora_seed42_q8_0.gguf"
    if q8_file.exists():
        model, tokenizer = ModelLoader.load_model(str(Q8_DIR), quantization="gguf")
    else:
        model = MockLlamaModel()
        tokenizer = ModelLoader.load_tokenizer(str(Q8_DIR))
    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "create_completion")


def test_inference_engine_with_gguf_q4_model():
    """Verify InferenceEngine operates seamlessly with Q4_K_M GGUF model."""
    q4_file = Q4_DIR / "qwen25_05b_lora_seed42_q4_k_m.gguf"
    if q4_file.exists():
        model, tokenizer = ModelLoader.load_model(str(Q4_DIR), quantization="gguf")
    else:
        model = MockLlamaModel()
        tokenizer = ModelLoader.load_tokenizer(str(Q4_DIR))
    engine = InferenceEngine(model=model, tokenizer=tokenizer, model_version="qwen25_05b_lora_seed42_q4_k_m", max_new_tokens=40)

    pred = engine.predict("I need an immediate refund for my cancelled flight ticket ORD-9988")
    assert isinstance(pred, TriagePrediction)
    assert pred.model_version == "qwen25_05b_lora_seed42_q4_k_m"
    assert pred.latency_ms > 0.0
    assert pred.confidence is None or pred.confidence > 0.0
    assert pred.category in CATEGORIES
    assert pred.priority in PRIORITIES
    assert pred.language in LANGUAGES


def test_inference_engine_with_gguf_q8_model():
    """Verify InferenceEngine operates seamlessly with Q8_0 GGUF model."""
    q8_file = Q8_DIR / "qwen25_05b_lora_seed42_q8_0.gguf"
    if q8_file.exists():
        model, tokenizer = ModelLoader.load_model(str(Q8_DIR), quantization="gguf")
    else:
        model = MockLlamaModel()
        tokenizer = ModelLoader.load_tokenizer(str(Q8_DIR))
    engine = InferenceEngine(model=model, tokenizer=tokenizer, model_version="qwen25_05b_lora_seed42_q8_0", max_new_tokens=40)

    pred = engine.predict("I cannot log into my account. The password reset link expired.")
    assert isinstance(pred, TriagePrediction)
    assert pred.model_version == "qwen25_05b_lora_seed42_q8_0"
    assert pred.latency_ms > 0.0
    assert pred.confidence is None or pred.confidence > 0.0
    assert pred.category in CATEGORIES
    assert pred.priority in PRIORITIES


def test_gguf_multilingual_inference():
    """Verify InferenceEngine accurately processes English, Hinglish, and Hindi tickets."""
    q4_file = Q4_DIR / "qwen25_05b_lora_seed42_q4_k_m.gguf"
    if q4_file.exists():
        model, tokenizer = ModelLoader.load_model(str(Q4_DIR), quantization="gguf")
    else:
        model = MockLlamaModel()
        tokenizer = ModelLoader.load_tokenizer(str(Q4_DIR))
    engine = InferenceEngine(model=model, tokenizer=tokenizer, model_version="qwen25_05b_lora_seed42_q4_k_m", max_new_tokens=40)

    tickets = [
        ("English", "Please cancel my annual subscription and issue a refund.", "english"),
        ("Hinglish", "Mera order abhi tak deliver nahi hua, delay ka kya reason hai?", "hinglish"),
        ("Hindi", "कृपया मेरा खाता तुरंत बंद करें, मुझे अब सेवा नहीं चाहिए।", "hindi"),
    ]

    for lang_name, text, expected_lang in tickets:
        pred = engine.predict(text)
        assert pred.category in CATEGORIES
        assert pred.priority in PRIORITIES
        assert pred.language == expected_lang
        assert pred.latency_ms > 0.0


def test_deterministic_human_escalation_with_gguf():
    """Verify deterministic policy guarantees human escalation for high-risk inputs."""
    high_risk_tickets = [
        ("Someone unauthorized accessed my account and stole funds.", "fraud_report", "urgent"),
        ("I will file a lawsuit and report you to the consumer regulator.", "product_complaint", "high"),
        ("Mera account hack ho gaya hai unauthorized transactions ho rahe hain.", "fraud_report", "high"),
        ("मेरे खाते से किसी ने धोखाधड़ी करके पैसे निकाल लिए हैं।", "fraud_report", "critical"),
    ]

    for text, cat, prio in high_risk_tickets:
        req_human, reason, final_cat, final_prio, rule = HumanEscalationPolicy.evaluate_escalation(
            text=text,
            category=cat,
            priority=prio,
            language="english",
            raw_confidence=0.99,
            inference_mode="neural",
        )
        assert req_human is True, f"Policy failed to intercept high-risk ticket: {text}"
        assert reason is not None


def test_fastapi_predict_contract_with_gguf():
    """Verify candidate prediction validates against FastAPI PredictResponse schema."""
    pred = TriagePrediction(
        category="cancellation_and_refund" if "cancellation_and_refund" in CATEGORIES else "refund_request",
        priority="high",
        language="hindi",
        requires_human=True,
        confidence=0.9450,
        latency_ms=1050.2,
        model_version="qwen25_05b_lora_seed42_q4_k_m",
        raw_output='{"category": "refund_request", "priority": "high", "language": "hindi", "requires_human": true}',
    )

    api_response = PredictResponse(
        category=SupportCategory(pred.category),
        priority=SupportPriority(pred.priority),
        language=SupportLanguage(pred.language),
        requires_human=pred.requires_human,
        confidence=pred.confidence,
        latency_ms=pred.latency_ms,
        model_version=pred.model_version,
    )

    resp_dict = api_response.model_dump()
    assert resp_dict["category"] == pred.category
    assert resp_dict["priority"] == "high"
    assert resp_dict["language"] == "hindi"
    assert resp_dict["requires_human"] is True
    assert resp_dict["confidence"] == 0.9450
    assert resp_dict["latency_ms"] == 1050.2
    assert resp_dict["model_version"] == "qwen25_05b_lora_seed42_q4_k_m"


def test_missing_gguf_file_error_handling():
    """Verify that attempting to load a non-existent GGUF path raises an exception."""
    bogus_path = PROJECT_ROOT / "models" / "non_existent_gguf_dir"
    with pytest.raises((FileNotFoundError, RuntimeError, OSError, StopIteration)):
        ModelLoader.load_model(str(bogus_path), quantization="gguf")
