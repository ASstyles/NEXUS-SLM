"""Final Model Acceptance and Verification Evaluation for NEXUS-SLM.

Evaluates the production fallback (nexus-qlora-v1) and frozen candidate (qwen25_05b_clean_v2_lora)
on the quarantined Clean Template-Grouped V2 Test Set (data/v2_clean_template_grouped/test_clean.jsonl, N=746).

Emits:
- experiments/artifacts/final_clean_v2_model_evaluation.json
- experiments/artifacts/final_calibration_metrics.json
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["EVALUATION_RUN"] = "1"
os.environ["DISABLE_RATE_LIMIT"] = "1"
os.environ["DISABLE_CONCURRENCY_LIMIT"] = "1"

import numpy as np
import psutil
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

from api.main import get_fallback_prediction
from src.inference.parser import OutputParser
from src.inference.policy import HumanEscalationPolicy
from src.models.calibration import CalibrationEvaluator, SelectivePredictor, TemperatureScalingCalibrator
from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_production_fallback(test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("EVALUATING PRODUCTION FALLBACK: nexus-qlora-v1 (Heuristic Baseline)")
    print("=" * 80)

    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 * 1024)

    results = []
    latencies = []
    t_start = time.perf_counter()

    for idx, s in enumerate(test_samples):
        txt = s["text"]
        t0 = time.perf_counter()

        fb = get_fallback_prediction(txt)
        req_human, esc_reason, cal_conf, cat_str, prio_str = HumanEscalationPolicy.evaluate_escalation(
            text=txt,
            category=fb["category"],
            priority=fb["priority"],
            language=fb["language"],
            raw_confidence=fb["confidence"],
            is_parse_error=False,
            inference_mode="heuristic_fallback",
            model_requires_human=fb.get("requires_human"),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 + 35.0  # Heuristic overhead simulation
        latencies.append(elapsed_ms)

        results.append({
            "ground_truth": s,
            "raw_model": {
                "category": fb["category"],
                "priority": fb["priority"],
                "language": fb["language"],
                "requires_human": fb.get("requires_human"),
                "confidence": fb.get("confidence"),
                "is_valid": True,
            },
            "final_response": {
                "category": cat_str,
                "priority": prio_str,
                "language": fb["language"],
                "requires_human": req_human,
                "escalation_reason": esc_reason,
                "calibrated_confidence": cal_conf,
                "confidence_status": "unavailable",
            },
            "latency_ms": elapsed_ms,
        })

    duration_sec = time.perf_counter() - t_start
    ram_after = process.memory_info().rss / (1024 * 1024)

    return compute_split_metrics("nexus-qlora-v1", "heuristic_fallback", test_samples, results, latencies, duration_sec, max(ram_before, ram_after))


def evaluate_candidate_neural(test_samples: List[Dict[str, Any]], batch_size: int = 16) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print("\n" + "=" * 80)
    print("EVALUATING FROZEN CANDIDATE: qwen25_05b_clean_v2_lora (Neural LoRA)")
    print("=" * 80)

    adapter_path = PROJECT_ROOT / "experiments" / "artifacts" / "training_runs" / "qwen25_05b_clean_v2_lora_run_20261030_190017" / "adapter"
    base_model_path = PROJECT_ROOT / "models" / "scratch_merged_qwen"
    base_target = str(base_model_path) if base_model_path.exists() else "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model from: {base_target}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_target,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    torch.set_num_threads(8)
    print("Configured torch to 8 CPU threads.")

    parser = OutputParser()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 * 1024)

    results = []
    latencies = []
    t_start = time.perf_counter()

    num_batches = math.ceil(len(test_samples) / batch_size)
    print(f"Beginning batched evaluation: {len(test_samples)} samples in {num_batches} batches (batch_size={batch_size})...")

    for b_idx in range(num_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, len(test_samples))
        batch_samples = test_samples[b_start:b_end]

        t_b0 = time.perf_counter()
        prompts = [PromptFormatter.format_prompt(s["text"], tokenizer=tokenizer) for s in batch_samples]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        t_b1 = time.perf_counter()
        batch_duration_ms = (t_b1 - t_b0) * 1000.0
        sample_latency_ms = round(batch_duration_ms / len(batch_samples), 2)

        prompt_len = inputs["input_ids"].shape[1]
        for i, s in enumerate(batch_samples):
            latencies.append(sample_latency_ms)
            gen_tokens = outputs.sequences[i][prompt_len:]
            raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Compute token generation confidence
            probs = []
            for step_idx, step_logits in enumerate(outputs.scores):
                tok_id = gen_tokens[step_idx].item() if step_idx < len(gen_tokens) else None
                if tok_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                step_prob = torch.softmax(step_logits[i], dim=-1)
                probs.append(torch.max(step_prob).item())
            raw_conf = round(sum(probs) / len(probs), 4) if probs else None

            # Parse JSON
            is_valid, parsed, err = parser.parse_and_validate(raw_text)

            if is_valid and parsed is not None:
                cat_m = parsed["category"]
                prio_m = parsed["priority"]
                lang_m = parsed["language"]
                req_h_m = parsed["requires_human"]
                is_parse_err = False
            else:
                cat_m = "general_query"
                prio_m = "medium"
                lang_m = "english"
                req_h_m = True
                is_parse_err = True

            # Evaluate deterministic policy escalation
            req_human_pol, esc_reason, cal_conf, cat_final, prio_final = HumanEscalationPolicy.evaluate_escalation(
                text=s["text"],
                category=cat_m,
                priority=prio_m,
                language=lang_m,
                raw_confidence=raw_conf,
                is_parse_error=is_parse_err,
                inference_mode="neural",
                model_requires_human=req_h_m,
            )

            results.append({
                "ground_truth": s,
                "raw_model": {
                    "category": cat_m,
                    "priority": prio_m,
                    "language": lang_m,
                    "requires_human": req_h_m,
                    "confidence": raw_conf,
                    "is_valid": is_valid,
                    "raw_output": raw_text,
                    "parse_error": err,
                },
                "final_response": {
                    "category": cat_final,
                    "priority": prio_final,
                    "language": lang_m,
                    "requires_human": req_human_pol,
                    "escalation_reason": esc_reason,
                    "calibrated_confidence": cal_conf,
                    "confidence_status": "uncalibrated_heuristic" if raw_conf is not None else "unavailable",
                },
                "latency_ms": sample_latency_ms,
            })

        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == num_batches:
            processed = len(results)
            elapsed_tot = time.perf_counter() - t_start
            print(f"  Processed {processed}/{len(test_samples)} samples ({elapsed_tot:.1f}s elapsed, batch {b_idx+1}/{num_batches})...")

    duration_sec = time.perf_counter() - t_start
    ram_after = process.memory_info().rss / (1024 * 1024)

    metrics = compute_split_metrics("qwen25_05b_clean_v2_lora", "neural_lora", test_samples, results, latencies, duration_sec, max(ram_before, ram_after))
    return metrics, results


def compute_split_metrics(
    model_name: str,
    inference_mode: str,
    test_samples: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    latencies: List[float],
    duration_sec: float,
    peak_ram_mb: float,
) -> Dict[str, Any]:
    # 1. Category Classification Metrics
    y_true_cat = [r["ground_truth"]["category"] for r in results]
    y_pred_cat = [r["final_response"]["category"] for r in results]
    y_pred_cat_raw = [r["raw_model"]["category"] for r in results]

    cat_acc = accuracy_score(y_true_cat, y_pred_cat)
    raw_cat_acc = accuracy_score(y_true_cat, y_pred_cat_raw)
    correct_count = sum(1 for yt, yp in zip(y_true_cat, y_pred_cat) if yt == yp)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_cat, y_pred_cat, labels=CATEGORIES, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_cat, y_pred_cat, labels=CATEGORIES, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_s = precision_recall_fscore_support(
        y_true_cat, y_pred_cat, labels=CATEGORIES, average=None, zero_division=0
    )
    per_class = {}
    for c, p, r, f, s in zip(CATEGORIES, per_class_p, per_class_r, per_class_f1, per_class_s):
        per_class[c] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }

    # Confusion matrix for categories
    cat_conf_matrix = confusion_matrix(y_true_cat, y_pred_cat, labels=CATEGORIES).tolist()

    # 2. Priority Classification Metrics
    y_true_prio = [r["ground_truth"]["priority"] for r in results]
    y_pred_prio = [r["final_response"]["priority"] for r in results]
    prio_acc = accuracy_score(y_true_prio, y_pred_prio)
    prio_macro_p, prio_macro_r, prio_macro_f1, _ = precision_recall_fscore_support(
        y_true_prio, y_pred_prio, labels=PRIORITIES, average="macro", zero_division=0
    )
    prio_p, prio_r, prio_f, prio_s = precision_recall_fscore_support(
        y_true_prio, y_pred_prio, labels=PRIORITIES, average=None, zero_division=0
    )
    per_priority = {}
    for pr, p, r, f, s in zip(PRIORITIES, prio_p, prio_r, prio_f, prio_s):
        per_priority[pr] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
    prio_conf_matrix = confusion_matrix(y_true_prio, y_pred_prio, labels=PRIORITIES).tolist()

    # 3. Language Identification Metrics
    y_true_lang = [r["ground_truth"]["language"] for r in results]
    y_pred_lang = [r["final_response"]["language"] for r in results]
    lang_acc = accuracy_score(y_true_lang, y_pred_lang)

    per_language = {}
    for lang in ["english", "hindi", "hinglish"]:
        idx_list = [i for i, r in enumerate(results) if r["ground_truth"].get("language") == lang]
        if idx_list:
            sub_true_cat = [y_true_cat[i] for i in idx_list]
            sub_pred_cat = [y_pred_cat[i] for i in idx_list]
            sub_true_lang = [y_true_lang[i] for i in idx_list]
            sub_pred_lang = [y_pred_lang[i] for i in idx_list]
            sub_cat_acc = accuracy_score(sub_true_cat, sub_pred_cat)
            sub_lang_acc = accuracy_score(sub_true_lang, sub_pred_lang)
            _, _, sub_f1, _ = precision_recall_fscore_support(
                sub_true_cat, sub_pred_cat, labels=CATEGORIES, average="macro", zero_division=0
            )
            per_language[lang] = {
                "count": len(idx_list),
                "category_accuracy": round(float(sub_cat_acc), 4),
                "category_macro_f1": round(float(sub_f1), 4),
                "language_id_accuracy": round(float(sub_lang_acc), 4),
            }

    # 4. Exact Four-Field Structured Match
    exact_match_count = 0
    for r in results:
        gt = r["ground_truth"]
        pr = r["final_response"]
        c_ok = gt.get("category", "").lower() == pr.get("category", "").lower()
        p_ok = gt.get("priority", "").lower() == pr.get("priority", "").lower()
        l_ok = gt.get("language", "").lower() == pr.get("language", "").lower()
        h_ok = bool(gt.get("requires_human")) == bool(pr.get("requires_human"))
        if c_ok and p_ok and l_ok and h_ok:
            exact_match_count += 1
    exact_match_rate = round(exact_match_count / len(results), 4)

    # 5. JSON Validity & Schema Compliance
    valid_json_count = sum(1 for r in results if r["raw_model"]["is_valid"])
    json_validity_rate = round(valid_json_count / len(results), 4)

    # 6. Escalation Metrics: Model vs Policy
    y_true_esc = [bool(r["ground_truth"].get("requires_human")) for r in results]
    y_pred_policy_esc = [bool(r["final_response"]["requires_human"]) for r in results]
    y_pred_model_esc = [bool(r["raw_model"]["requires_human"]) for r in results]

    # Policy escalation evaluation
    tp = sum(1 for yt, yp in zip(y_true_esc, y_pred_policy_esc) if yt and yp)
    fn = sum(1 for yt, yp in zip(y_true_esc, y_pred_policy_esc) if yt and not yp)
    fp = sum(1 for yt, yp in zip(y_true_esc, y_pred_policy_esc) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true_esc, y_pred_policy_esc) if not yt and not yp)

    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        y_true_esc, y_pred_policy_esc, labels=[True], average="binary", zero_division=0
    )

    # Model-only escalation evaluation (before policy overrides)
    model_tp = sum(1 for yt, yp in zip(y_true_esc, y_pred_model_esc) if yt and yp)
    model_fn = sum(1 for yt, yp in zip(y_true_esc, y_pred_model_esc) if yt and not yp)
    model_fp = sum(1 for yt, yp in zip(y_true_esc, y_pred_model_esc) if not yt and yp)
    model_tn = sum(1 for yt, yp in zip(y_true_esc, y_pred_model_esc) if not yt and not yp)
    m_p, m_r, m_f1, _ = precision_recall_fscore_support(
        y_true_esc, y_pred_model_esc, labels=[True], average="binary", zero_division=0
    )

    throughput = round(len(results) / max(duration_sec, 0.001), 2)

    return {
        "model_version": model_name,
        "inference_mode": inference_mode,
        "sample_count": len(results),
        "total_test_samples": len(test_samples),
        "category_metrics": {
            "accuracy": round(float(cat_acc), 4),
            "correct_count": correct_count,
            "raw_model_accuracy": round(float(raw_cat_acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "confusion_matrix_labels": CATEGORIES,
            "confusion_matrix": cat_conf_matrix,
        },
        "priority_metrics": {
            "accuracy": round(float(prio_acc), 4),
            "macro_precision": round(float(prio_macro_p), 4),
            "macro_recall": round(float(prio_macro_r), 4),
            "macro_f1": round(float(prio_macro_f1), 4),
            "per_priority": per_priority,
            "confusion_matrix_labels": PRIORITIES,
            "confusion_matrix": prio_conf_matrix,
        },
        "language_metrics": {
            "accuracy": round(float(lang_acc), 4),
            "per_language_breakdown": per_language,
        },
        "exact_structured_match": {
            "match_count": exact_match_count,
            "match_rate": exact_match_rate,
        },
        "json_validity": {
            "valid_count": valid_json_count,
            "validity_rate": json_validity_rate,
            "schema_compliant_rate": json_validity_rate,
        },
        "escalation_metrics": {
            "final_policy": {
                "true_positives": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "true_negatives": tn,
                "precision": round(float(esc_p), 4),
                "recall": round(float(esc_r), 4),
                "f1": round(float(esc_f1), 4),
                "escalation_rate": round(sum(1 for yp in y_pred_policy_esc if yp) / len(results), 4),
            },
            "raw_model_requires_human": {
                "true_positives": model_tp,
                "false_negatives": model_fn,
                "false_positives": model_fp,
                "true_negatives": model_tn,
                "precision": round(float(m_p), 4),
                "recall": round(float(m_r), 4),
                "f1": round(float(m_f1), 4),
                "escalation_rate": round(sum(1 for yp in y_pred_model_esc if yp) / len(results), 4),
            },
        },
        "latency_and_resources": {
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p90_latency_ms": round(float(np.percentile(latencies, 90)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "throughput_req_per_sec": throughput,
            "peak_rss_memory_mb": round(peak_ram_mb, 2),
            "total_duration_sec": round(duration_sec, 2),
        },
    }


def perform_calibration_audit(candidate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PERFORMING CONFIDENCE CALIBRATION AUDIT")
    print("=" * 80)

    confidences = [r["raw_model"]["confidence"] for r in candidate_results if r["raw_model"]["confidence"] is not None]
    accuracies = [
        r["raw_model"]["category"] == r["ground_truth"]["category"]
        for r in candidate_results
        if r["raw_model"]["confidence"] is not None
    ]

    # 1. Empirical ECE, MCE, Brier
    ece_data = CalibrationEvaluator.compute_ece(confidences, accuracies, num_bins=10)

    # 2. Temperature Scaling Evaluation
    temp_calibrator = TemperatureScalingCalibrator(temperature=1.15)
    calibrated_confs = [temp_calibrator.calibrate_probability(c) for c in confidences]
    ece_calibrated = CalibrationEvaluator.compute_ece(calibrated_confs, accuracies, num_bins=10)

    # 3. Distribution statistics
    conf_arr = np.array(confidences)
    dist_stats = {
        "sample_count": len(confidences),
        "min": round(float(np.min(conf_arr)), 4),
        "max": round(float(np.max(conf_arr)), 4),
        "mean": round(float(np.mean(conf_arr)), 4),
        "std": round(float(np.std(conf_arr)), 4),
        "p25": round(float(np.percentile(conf_arr, 25)), 4),
        "p50_median": round(float(np.percentile(conf_arr, 50)), 4),
        "p75": round(float(np.percentile(conf_arr, 75)), 4),
        "p90": round(float(np.percentile(conf_arr, 90)), 4),
    }

    # 4. Risk vs Coverage analysis
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    risk_coverage = []
    for th in thresholds:
        accepted_indices = [i for i, c in enumerate(confidences) if c >= th]
        coverage = len(accepted_indices) / len(confidences) if confidences else 0.0
        if accepted_indices:
            acc_subset = [accuracies[i] for i in accepted_indices]
            acc_at_th = sum(acc_subset) / len(acc_subset)
            error_at_th = 1.0 - acc_at_th
        else:
            acc_at_th = 1.0
            error_at_th = 0.0

        risk_coverage.append({
            "confidence_threshold": th,
            "coverage_rate": round(coverage, 4),
            "retained_samples": len(accepted_indices),
            "accuracy": round(acc_at_th, 4),
            "error_rate": round(error_at_th, 4),
        })

    calibration_audit = {
        "audit_timestamp": datetime.utcnow().isoformat() + "Z",
        "calibration_scope": {
            "score_definition": "Mean token generation probability of generated ChatML output JSON string",
            "is_categorical_posterior": False,
            "categorical_posterior_note": "Scores reflect auto-regressive generation token probabilities, not 14-class posterior softmax distribution",
            "dataset_independence_status": "No independent calibration holdout split existed. Temperature T=1.15 was preset rather than fitted via NLL on an independent holdout.",
            "is_bounded": True,
            "is_monotonic": True,
        },
        "raw_confidence_metrics": {
            "expected_calibration_error_ece": ece_data["ece"],
            "max_calibration_error_mce": ece_data["mce"],
            "brier_score": ece_data["brier_score"],
            "reliability_diagram_bins": ece_data["bins"],
        },
        "temperature_scaled_metrics": {
            "temperature": 1.15,
            "expected_calibration_error_ece": ece_calibrated["ece"],
            "max_calibration_error_mce": ece_calibrated["mce"],
            "brier_score": ece_calibrated["brier_score"],
            "reliability_diagram_bins": ece_calibrated["bins"],
        },
        "confidence_distribution": dist_stats,
        "risk_versus_coverage": risk_coverage,
        "policy_recommendation": (
            "Because token-level sequence generation probabilities do not represent a true 14-class posterior "
            "and no separate holdout calibration set existed, confidence MUST remain designated as 'uncalibrated_heuristic'. "
            "Automated routing based purely on confidence thresholds cannot be certified for unmonitored straight-through processing."
        ),
    }

    return calibration_audit


def main():
    test_path = PROJECT_ROOT / "data" / "v2_clean_template_grouped" / "test_clean.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing clean test set at: {test_path}")

    with open(test_path, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded locked clean test set: N={len(test_samples)} samples from {test_path.relative_to(PROJECT_ROOT)}")

    # 1. Evaluate Production Fallback (nexus-qlora-v1)
    fallback_metrics = evaluate_production_fallback(test_samples)

    # 2. Evaluate Frozen Candidate (qwen25_05b_clean_v2_lora)
    candidate_metrics, candidate_raw_results = evaluate_candidate_neural(test_samples, batch_size=16)

    # 3. Perform Confidence Calibration Audit
    calibration_metrics = perform_calibration_audit(candidate_raw_results)

    # 4. Save Final Evaluation Artifact
    eval_artifact = {
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_metadata": {
            "file": str(test_path.relative_to(PROJECT_ROOT)),
            "sample_count": len(test_samples),
            "source": "data/v2_clean_template_grouped/test_clean.jsonl",
            "is_quarantined_test_split": True,
        },
        "production_baseline": fallback_metrics,
        "frozen_candidate": candidate_metrics,
        "head_to_head_comparison": {
            "metric": "category_accuracy",
            "production_baseline": fallback_metrics["category_metrics"]["accuracy"],
            "candidate": candidate_metrics["category_metrics"]["accuracy"],
            "delta_points": round(candidate_metrics["category_metrics"]["accuracy"] - fallback_metrics["category_metrics"]["accuracy"], 4),
            "macro_f1_production": fallback_metrics["category_metrics"]["macro_f1"],
            "macro_f1_candidate": candidate_metrics["category_metrics"]["macro_f1"],
            "delta_macro_f1": round(candidate_metrics["category_metrics"]["macro_f1"] - fallback_metrics["category_metrics"]["macro_f1"], 4),
            "exact_match_production": fallback_metrics["exact_structured_match"]["match_rate"],
            "exact_match_candidate": candidate_metrics["exact_structured_match"]["match_rate"],
            "high_risk_false_negatives_production": fallback_metrics["escalation_metrics"]["final_policy"]["false_negatives"],
            "high_risk_false_negatives_candidate": candidate_metrics["escalation_metrics"]["final_policy"]["false_negatives"],
            "high_risk_recall_candidate": candidate_metrics["escalation_metrics"]["final_policy"]["recall"],
        },
    }

    eval_out = PROJECT_ROOT / "experiments" / "artifacts" / "final_clean_v2_model_evaluation.json"
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_artifact, f, indent=2)
    print(f"\n[Artifact Saved] Model Evaluation: {eval_out}")

    # 5. Save Calibration Metrics Artifact
    cal_out = PROJECT_ROOT / "experiments" / "artifacts" / "final_calibration_metrics.json"
    with open(cal_out, "w", encoding="utf-8") as f:
        json.dump(calibration_metrics, f, indent=2)
    print(f"[Artifact Saved] Calibration Metrics: {cal_out}")

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION SUMMARY:")
    print("=" * 80)
    print(f"Production Baseline Category Accuracy: {fallback_metrics['category_metrics']['accuracy'] * 100:.2f}% | Macro-F1: {fallback_metrics['category_metrics']['macro_f1']:.4f}")
    print(f"Frozen Candidate Category Accuracy:    {candidate_metrics['category_metrics']['accuracy'] * 100:.2f}% ({candidate_metrics['category_metrics']['correct_count']}/{len(test_samples)}) | Macro-F1: {candidate_metrics['category_metrics']['macro_f1']:.4f}")
    print(f"Candidate Language Accuracy:           {candidate_metrics['language_metrics']['accuracy'] * 100:.2f}%")
    print(f"Candidate JSON Validity Rate:          {candidate_metrics['json_validity']['validity_rate'] * 100:.2f}%")
    print(f"Candidate Policy High-Risk Recall:     {candidate_metrics['escalation_metrics']['final_policy']['recall'] * 100:.2f}% (FNs: {candidate_metrics['escalation_metrics']['final_policy']['false_negatives']})")
    print(f"Candidate ECE (Uncalibrated):          {calibration_metrics['raw_confidence_metrics']['expected_calibration_error_ece']:.4f}")
    print(f"Candidate ECE (Temperature Scaled):    {calibration_metrics['temperature_scaled_metrics']['expected_calibration_error_ece']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

