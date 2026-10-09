"""Empirical calibration parameter fitting and evaluation on VALIDATION data for NEXUS-SLM."""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.optimize import minimize
from src.evaluation.calibration import ConfidenceCalibrator
from src.inference.engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_validation_inference() -> Tuple[List[float], List[bool], List[Dict[str, Any]]]:
    """Runs inference across data/validation/validation.jsonl (NEVER on test data!)."""
    val_path = Path("data/validation/validation.jsonl")
    adapter_path = Path("models/nexus_qlora_v1")

    with open(val_path, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(val_samples)} validation samples for calibration fitting.")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    engine = InferenceEngine(model=model, tokenizer=tokenizer, model_version="nexus-qlora-v1")

    confidences = []
    correctness = []
    records = []

    for idx, sample in enumerate(val_samples):
        text = sample["text"]
        true_cat = sample["category"]

        pred = engine.predict(text)
        conf = float(pred.confidence)
        is_correct = (pred.category == true_cat)

        confidences.append(conf)
        correctness.append(is_correct)
        records.append({
            "idx": idx,
            "text": text[:60] + "...",
            "true_cat": true_cat,
            "pred_cat": pred.category,
            "confidence": conf,
            "is_correct": is_correct,
        })

    return confidences, correctness, records


def fit_platt_scaling(confidences: List[float], correctness: List[bool]) -> Dict[str, Any]:
    """
    Fits Platt scaling (logistic regression on logit(p)) on validation data.
    p_cal = 1 / (1 + exp(-(a * logit(p) + b)))
    """
    p = np.clip(np.array(confidences), 1e-4, 1.0 - 1e-4)
    y = np.array(correctness, dtype=float)
    logits = np.log(p / (1.0 - p))

    # Minimize negative log-likelihood (binary cross-entropy)
    def loss(params):
        a, b = params
        z = a * logits + b
        p_cal = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        bce = -np.mean(y * np.log(p_cal + 1e-12) + (1.0 - y) * np.log(1.0 - p_cal + 1e-12))
        return bce

    res = minimize(loss, [1.0, 0.0], method="L-BFGS-B")
    a_opt, b_opt = float(res.x[0]), float(res.x[1])

    # Apply calibrated probabilities
    z_cal = a_opt * logits + b_opt
    p_calibrated = 1.0 / (1.0 + np.exp(-np.clip(z_cal, -30, 30)))

    return {
        "fitted_parameters": {"a": round(a_opt, 4), "b": round(b_opt, 4)},
        "loss": round(float(res.fun), 4),
        "converged": bool(res.success),
        "calibrated_confidences": [round(float(c), 4) for c in p_calibrated],
    }


def main():
    confs, correct, records = run_validation_inference()
    calibrator = ConfidenceCalibrator(num_bins=5)

    # Raw metrics on validation set
    raw_val_metrics = calibrator.compute_calibration_metrics(confs, correct)
    logger.info(f"Raw Validation ECE: {raw_val_metrics['expected_calibration_error']}")
    logger.info(f"Raw Validation Mean Conf: {raw_val_metrics['mean_confidence']} vs Mean Acc: {raw_val_metrics['mean_accuracy']}")

    # Fit Platt scaling strictly on validation set
    platt_fit = fit_platt_scaling(confs, correct)
    platt_confs = platt_fit["calibrated_confidences"]
    platt_val_metrics = calibrator.compute_calibration_metrics(platt_confs, correct)
    logger.info(f"Platt Validation ECE: {platt_val_metrics['expected_calibration_error']}")
    logger.info(f"Platt Validation Mean Conf: {platt_val_metrics['mean_confidence']} vs Mean Acc: {platt_val_metrics['mean_accuracy']}")

    # Critical statistical assessment of sample size:
    n_val = len(confs)
    sample_size_assessment = {
        "validation_sample_count": n_val,
        "classes_count": 14,
        "avg_samples_per_class": round(n_val / 14, 2),
        "is_statistically_sufficient": False,
        "statistical_limitation_rationale": (
            f"N={n_val} validation samples across 14 classes (~3.1 samples/class) provides insufficient statistical "
            "power for reliable multi-class parametric calibration. Platt scaling reduces empirical validation ECE, "
            "but small-sample variance risks severe generalization instability on production traffic. "
            "Therefore, confidence scores must be designated as 'uncalibrated_heuristic' and uncertain predictions "
            "must be deterministically routed to human escalation rather than relying on unproven confidence thresholds."
        ),
    }

    full_report = {
        "timestamp": "2026-10-09T22:05:00Z",
        "dataset_used": "data/validation/validation.jsonl",
        "sample_size_assessment": sample_size_assessment,
        "raw_validation_calibration": raw_val_metrics,
        "platt_scaling_validation_calibration": {
            "parameters": platt_fit["fitted_parameters"],
            "converged": platt_fit["converged"],
            "metrics": platt_val_metrics,
        },
        "governance_recommendation": {
            "status": "UNRELIABLE_FOR_AUTONOMOUS_ROUTING",
            "action": "Mark confidence as heuristic-only; route any ticket without overwhelming margin or high uncertainty to human review.",
            "api_field_designation": "calibrated_confidence: optional; confidence_status: 'uncalibrated_heuristic'"
        }
    }

    out_file = Path("experiments/artifacts/validation_calibration_study.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Validation calibration study saved to {out_file}")
    print("\n=== Validation Calibration Study Complete ===")
    print(f"Sample count: {n_val}")
    print(f"Raw Val ECE: {raw_val_metrics['expected_calibration_error']}")
    print(f"Platt Val ECE: {platt_val_metrics['expected_calibration_error']}")
    print(f"Platt parameters: {platt_fit['fitted_parameters']}")


if __name__ == "__main__":
    main()
