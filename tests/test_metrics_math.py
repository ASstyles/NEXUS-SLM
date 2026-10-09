"""Mathematical and formula validation tests for evaluation metrics and prompt formatting."""

import pytest
import numpy as np
from src.evaluation.metrics import EvaluationMetrics
from src.models.prompt_formatter import PromptFormatter


def test_macro_f1_mathematical_definition():
    """
    Verifies that macro-F1 is the unweighted mean of per-class F1 scores.
    Class A: 2 TP, 0 FP, 0 FN -> F1 = 1.0
    Class B: 1 TP, 1 FP, 1 FN -> Precision = 0.5, Recall = 0.5 -> F1 = 0.5
    Expected Macro-F1 = (1.0 + 0.5) / 2 = 0.75
    """
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "A", "B", "A"]  # One B misclassified as A
    # For A: TP=2, FP=1, FN=0 -> Prec=2/3 (~0.667), Rec=1.0 -> F1=4/5 = 0.80
    # For B: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5 -> F1=2*(1*0.5)/(1+0.5) = 2/3 (~0.667)
    # Expected Macro-F1 = (0.80 + 0.6667) / 2 = 0.7333

    metrics = EvaluationMetrics.calculate_classification_metrics(y_true, y_pred, labels=["A", "B"])
    assert metrics["macro_f1"] == pytest.approx(0.7333, abs=1e-3)
    assert metrics["accuracy"] == 0.75
    assert "per_class" in metrics
    assert "A" in metrics["per_class"]
    assert "B" in metrics["per_class"]
    assert metrics["per_class"]["A"]["f1"] == pytest.approx(0.80, abs=1e-2)
    assert metrics["per_class"]["B"]["f1"] == pytest.approx(0.6667, abs=1e-2)


def test_structured_metrics_mathematical_definition():
    """
    Verifies field-level accuracy, exact-match accuracy, and individual field accuracies.
    Sample 1: all 4 match -> 4/4
    Sample 2: cat & lang match, prio & human mismatch -> 2/4
    Total field accuracy = (4 + 2) / 8 = 0.75
    Exact match accuracy = 1 / 2 = 0.50
    """
    gts = [
        {"category": "billing", "priority": "low", "language": "english", "requires_human": False},
        {"category": "fraud_report", "priority": "critical", "language": "hindi", "requires_human": True},
    ]
    preds = [
        {"category": "billing", "priority": "low", "language": "english", "requires_human": False},
        {"category": "fraud_report", "priority": "high", "language": "hindi", "requires_human": False},
    ]
    valid_flags = [True, True]

    res = EvaluationMetrics.calculate_structured_metrics(gts, preds, valid_flags)
    assert res["json_validity_rate"] == 1.0
    assert res["category_accuracy"] == 1.0
    assert res["priority_accuracy"] == 0.5
    assert res["language_accuracy"] == 1.0
    assert res["human_escalation_accuracy"] == 0.5
    assert res["field_level_accuracy"] == 0.75
    assert res["exact_match_accuracy"] == 0.50


def test_performance_latency_percentiles():
    """Verifies P50, P95, P99 and throughput calculations."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    perf = EvaluationMetrics.calculate_performance_metrics(latencies, ram_mb=1024.0, vram_mb=512.0)

    # 10 samples: p50 should be around 55.0 ms
    assert perf["latency_p50_ms"] == pytest.approx(55.0, abs=1.0)
    assert perf["latency_p95_ms"] > perf["latency_p50_ms"]
    assert perf["latency_p99_ms"] >= perf["latency_p95_ms"]
    assert perf["mean_latency_ms"] == 55.0
    assert perf["throughput_req_per_sec"] == pytest.approx(1000.0 / 55.0, abs=0.1)
    assert perf["ram_mb"] == 1024.0
    assert perf["vram_mb"] == 512.0


def test_prompt_formatter_structure():
    """Verifies ChatML structure and JSON instructions in PromptFormatter."""
    prompt = PromptFormatter.format_prompt("Mera payment kat gaya", tokenizer=None)
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    assert "Mera payment kat gaya" in prompt
    assert "category" in prompt
    assert "priority" in prompt
    assert "requires_human" in prompt
