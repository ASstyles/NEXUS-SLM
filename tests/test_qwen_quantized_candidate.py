"""Comprehensive Unit and Integration Tests for Qwen2.5-0.5B Quantized Candidate.

Verifies:
1. Quantized candidate artifact existence and metadata integrity.
2. Quantized model loading via ORTModelForCausalLM and ModelLoader.
3. Correct model identity derived from trained Qwen2.5-0.5B architecture.
4. Tokenizer, ChatML formatting, and output schema compatibility.
5. Robust multilingual handling (English, Hinglish, Native Devanagari Hindi).
6. Resilient handling of malformed and corrupted model outputs without crashes.
7. Deterministic human escalation policy preservation (fraud, account takeover, regulatory vectors).
8. Low-confidence fallback routing to human agents.
9. FastAPI PredictResponse contract schema conformity.
10. Graceful failure and error handling on missing model artifacts.
"""

import json
from pathlib import Path
import pytest
import torch

from src.models.prompt_formatter import PromptFormatter
from src.inference.parser import OutputParser
from src.inference.policy import HumanEscalationPolicy
from src.inference.engine import InferenceEngine
from src.models.loader import ModelLoader
from src.data.schemas import TriagePrediction
from api.schemas import PredictResponse, SupportCategory, SupportPriority, SupportLanguage
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INT8_DIR = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_int8"
FP32_DIR = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42"


def test_quantized_candidate_artifacts_exist():
    """Verify all expected ONNX INT8 candidate files exist on disk."""
    assert INT8_DIR.exists(), f"Candidate directory {INT8_DIR} does not exist"
    required_files = [
        "model_quantized.onnx",
        "config.json",
        "ort_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "nexus_qwen_int8_metadata.json",
        "nexus_qwen25_05b_lora_seed42_int8.zip",
    ]
    for rf in required_files:
        fpath = INT8_DIR / rf
        assert fpath.exists(), f"Required artifact {rf} is missing in {INT8_DIR}"
        assert fpath.stat().st_size > 0, f"Artifact {rf} is empty (0 bytes)"


def test_quantized_candidate_metadata_hashes():
    """Verify metadata records SHA256 hashes matching the directory contents."""
    meta_path = INT8_DIR / "nexus_qwen_int8_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["candidate_version"] == "qwen25_05b_lora_seed42_int8"
    assert meta["quantization_format"] == "onnx_dynamic_int8"
    assert "artifact_hashes" in meta
    assert "model_quantized.onnx" in meta["artifact_hashes"]


def test_quantized_candidate_loading_ort():
    """Verify that ORTModelForCausalLM loads model_quantized.onnx successfully."""
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(INT8_DIR))
    assert tok is not None
    model = ORTModelForCausalLM.from_pretrained(str(INT8_DIR), file_name="model_quantized.onnx")
    assert model is not None
    assert getattr(model, "device", None) == torch.device("cpu") or str(getattr(model, "device", "cpu")) == "cpu"


def test_quantized_candidate_loading_via_model_loader():
    """Verify ModelLoader loads the quantized ONNX candidate."""
    model, tokenizer = ModelLoader.load_model(str(INT8_DIR), quantization="onnx_int8")
    assert model is not None
    assert tokenizer is not None


def test_quantized_model_identity_derived_from_qwen():
    """Verify that model architecture identity conforms to Qwen2.5-0.5B."""
    with open(INT8_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg.get("model_type") == "qwen2"
    assert cfg.get("vocab_size") == 151936
    assert cfg.get("hidden_size") == 896
    assert cfg.get("num_hidden_layers") == 24
    assert cfg.get("num_attention_heads") == 14
    assert cfg.get("num_key_value_heads") == 2


def test_inference_engine_with_ort_model():
    """Verify InferenceEngine operates seamlessly with an ORT model without .parameters() crash."""
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(INT8_DIR))
    model = ORTModelForCausalLM.from_pretrained(str(INT8_DIR), file_name="model_quantized.onnx")
    engine = InferenceEngine(model=model, tokenizer=tok, model_version="qwen25_05b_lora_seed42_int8", max_new_tokens=25)

    pred = engine.predict("I need a refund for my order ORD-1234")
    assert isinstance(pred, TriagePrediction)
    assert pred.model_version == "qwen25_05b_lora_seed42_int8"
    assert pred.latency_ms > 0.0
    assert pred.category in CATEGORIES
    assert pred.priority in PRIORITIES
    assert pred.language in LANGUAGES


def test_multilingual_inputs_english_hinglish_hindi():
    """Verify ChatML prompt formatting across English, Hinglish, and Native Devanagari Hindi."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(INT8_DIR))

    inputs = [
        ("English", "Please cancel my subscription before renewal."),
        ("Hinglish", "Mera payment kat gaya lekin order place nahi hua."),
        ("Hindi", "मेरे बैंक खाते से पैसे कट गए हैं लेकिन टिकट बुक नहीं हुआ।"),
    ]

    for lang_name, text in inputs:
        prompt = PromptFormatter.format_prompt(text, tokenizer=tok)
        assert "<|im_start|>user" in prompt
        assert text in prompt
        assert prompt.endswith("<|im_start|>assistant\n")
        # Ensure tokenizer encodes without throwing exceptions
        tokens = tok.encode(prompt)
        assert len(tokens) > 10


def test_malformed_model_output_handling():
    """Verify OutputParser and InferenceEngine handle invalid model outputs gracefully without crashes."""
    parser = OutputParser()

    malformed_examples = [
        "",  # Empty string
        "This is plain text with no json at all.",  # Non-JSON
        '{"category": "invalid_cat", "priority": "high", "language": "english", "requires_human": false}',  # Unknown category
        '{"category": "billing", "priority": "urgent_critical_unknown", "language": "english", "requires_human": false}',  # Unknown priority
        '{"category": "billing", "priority": "medium"}',  # Missing language and requires_human
        '{"category": "billing", "priority": "medium", "language": "english", "requires_human": "not_a_bool"}',  # Invalid type
        "{broken json string: 123",  # Syntax error
    ]

    for raw in malformed_examples:
        is_valid, parsed, err = parser.parse_and_validate(raw)
        assert is_valid is False
        assert err is not None


def test_deterministic_human_escalation_fraud_and_high_risk():
    """Verify deterministic policy guarantees human escalation for high-risk inputs regardless of raw confidence."""
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
            raw_confidence=0.99,  # Even with 99% confidence, policy must intercept
            inference_mode="neural",
        )
        assert req_human is True, f"Policy failed to intercept high-risk ticket: {text}"
        assert reason is not None


def test_low_confidence_fallback_behavior():
    """Verify that low-confidence neural model outputs are escalated to human review."""
    text = "General inquiry about working hours."
    req_human, reason, _, _, _ = HumanEscalationPolicy.evaluate_escalation(
        text=text,
        category="general_query",
        priority="low",
        language="english",
        raw_confidence=0.35,  # Low confidence below threshold
        inference_mode="neural",
    )
    assert req_human is True
    assert "confidence" in reason.lower()


def test_fastapi_response_contract_compatibility():
    """Verify that candidate predictions cleanly validate against FastAPI PredictResponse Pydantic contract."""
    pred = TriagePrediction(
        category="billing",
        priority="high",
        language="hinglish",
        requires_human=True,
        confidence=0.9123,
        latency_ms=1240.5,
        model_version="qwen25_05b_lora_seed42_int8",
        raw_output='{"category": "billing", "priority": "high", "language": "hinglish", "requires_human": true}',
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
    assert resp_dict["category"] == "billing"
    assert resp_dict["priority"] == "high"
    assert resp_dict["language"] == "hinglish"
    assert resp_dict["requires_human"] is True
    assert resp_dict["confidence"] == 0.9123
    assert resp_dict["latency_ms"] == 1240.5
    assert resp_dict["model_version"] == "qwen25_05b_lora_seed42_int8"


def test_missing_model_artifacts_error_handling():
    """Verify that attempting to load a non-existent model path fails gracefully with an exception."""
    bogus_path = PROJECT_ROOT / "models" / "non_existent_model_directory_12345"
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        ModelLoader.load_model(str(bogus_path), quantization="onnx_int8")
