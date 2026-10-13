"""NEXUS-SLM Complete Backend Performance Audit Runner.

Executes a factual, fully reproducible audit of the current NEXUS-SLM backend:
1. Production model (nexus-qlora-v1) in active runtime condition (heuristic fallback due to recovered state).
2. Experimental candidates:
   - qwen25_05b_lora_seed42_q8_0 (GGUF INT8)
   - qwen25_05b_lora_seed42_q4_k_m (GGUF Q4_K_M)
   - qwen25_05b_lora_seed42_int8 (ONNX Dynamic INT8)
3. Evaluates on:
   - Held-out validation split (data/validation/validation.jsonl, N=44)
   - Held-out test split (data/test/test.jsonl, N=42)
   - High-risk challenge set (data/evaluation/high_risk_challenge_set.jsonl, N=70)
   - Local benchmark demo suite (data/demo_fixtures.json, N=30)
4. Calculates all classification metrics, confusion matrices, priority, per-language slices,
   exact structured matches, JSON validity, escalation precision/recall/FN/FP, and latency percentiles.
5. Emits:
   - experiments/artifacts/backend_performance_audit_results.json
   - experiments/artifacts/backend_performance_audit_metrics.csv
"""

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import psutil
from fastapi.testclient import TestClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import api.main as api_module
from api.main import app, load_model_instance
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_environment_info() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "python_version": platform.python_version(),
    }


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / max(total, 1)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {}
    for lbl, p, r, f, s in zip(labels, per_class_p, per_class_r, per_class_f1, per_class_supp):
        per_class[lbl] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,
        },
    }


def evaluate_split_with_client(
    client: TestClient,
    samples: List[Dict[str, Any]],
    split_name: str,
    model_version: str,
) -> Dict[str, Any]:
    results = []
    latencies = []
    model_latencies = []

    for item in samples:
        ticket_text = item["text"]
        t0 = time.perf_counter()
        resp = client.post("/predict", json={"text": ticket_text})
        t1 = time.perf_counter()
        elapsed_ms = round((t1 - t0) * 1000, 2)
        latencies.append(elapsed_ms)

        if resp.status_code == 200:
            data = resp.json()
            model_latencies.append(data.get("latency_ms", elapsed_ms))
            results.append({
                "ground_truth": item,
                "prediction": data,
                "status_code": 200,
                "success": True,
                "elapsed_ms": elapsed_ms,
            })
        else:
            results.append({
                "ground_truth": item,
                "prediction": None,
                "status_code": resp.status_code,
                "error": resp.text,
                "success": False,
                "elapsed_ms": elapsed_ms,
            })

    # Category classification
    y_true_cat = [r["ground_truth"]["category"] for r in results]
    y_pred_cat = [
        r["prediction"]["category"] if r["success"] and r["prediction"] else "unknown"
        for r in results
    ]
    cat_metrics = compute_classification_metrics(y_true_cat, y_pred_cat, CATEGORIES)

    # Priority classification
    y_true_prio = [r["ground_truth"]["priority"] for r in results]
    y_pred_prio = [
        r["prediction"]["priority"] if r["success"] and r["prediction"] else "unknown"
        for r in results
    ]
    prio_metrics = compute_classification_metrics(y_true_prio, y_pred_prio, PRIORITIES)

    # Language classification
    y_true_lang = [r["ground_truth"]["language"] for r in results]
    y_pred_lang = [
        r["prediction"]["language"] if r["success"] and r["prediction"] else "unknown"
        for r in results
    ]
    lang_metrics = compute_classification_metrics(y_true_lang, y_pred_lang, LANGUAGES)

    # Per-language category breakdown
    per_language_results = {}
    for lang in ["english", "hindi", "hinglish"]:
        idx_list = [i for i, r in enumerate(results) if r["ground_truth"].get("language") == lang]
        if idx_list:
            sub_true = [y_true_cat[i] for i in idx_list]
            sub_pred = [y_pred_cat[i] for i in idx_list]
            sub_correct = sum(1 for yt, yp in zip(sub_true, sub_pred) if yt == yp)
            sub_acc = sub_correct / max(len(idx_list), 1)
            _, _, sub_macro_f1, _ = precision_recall_fscore_support(
                sub_true, sub_pred, labels=CATEGORIES, average="macro", zero_division=0
            )
            per_language_results[lang] = {
                "sample_count": len(idx_list),
                "correct": sub_correct,
                "accuracy": round(float(sub_acc), 4),
                "macro_f1": round(float(sub_macro_f1), 4),
            }

    # JSON validity & Schema compliance
    valid_json_count = sum(1 for r in results if r["success"])
    json_validity_rate = round(valid_json_count / max(len(results), 1), 4)

    # Exact Structured Prediction Match
    exact_match_count = 0
    for r in results:
        if not r["success"]:
            continue
        gt = r["ground_truth"]
        pr = r["prediction"]
        c_match = gt.get("category", "").lower() == pr.get("category", "").lower()
        p_match = gt.get("priority", "").lower() == pr.get("priority", "").lower()
        l_match = gt.get("language", "").lower() == pr.get("language", "").lower()
        h_match = bool(gt.get("requires_human")) == bool(pr.get("requires_human"))
        if c_match and p_match and l_match and h_match:
            exact_match_count += 1

    exact_match_rate = round(exact_match_count / max(len(results), 1), 4)

    # Escalation & Confidence Evaluation
    y_true_esc = [bool(r["ground_truth"].get("requires_human")) for r in results]
    y_pred_esc = [
        bool(r["prediction"].get("requires_human")) if r["success"] and r["prediction"] else True
        for r in results
    ]

    # True Positive: GT=True, Pred=True (Correct human review)
    # False Negative: GT=True, Pred=False (SAFETY ESCAPE!)
    # False Positive: GT=False, Pred=True (Unnecessary human escalation)
    # True Negative: GT=False, Pred=False (Correct autonomous resolution)
    tp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt and yp)
    fn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt and not yp)
    fp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if not yt and not yp)

    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        y_true_esc, y_pred_esc, labels=[True], average="binary", zero_division=0
    )

    escalated_count = sum(1 for yp in y_pred_esc if yp)
    automated_count = sum(1 for yp in y_pred_esc if not yp)
    escalation_rate = round(escalated_count / max(len(results), 1), 4)
    automation_rate = round(automated_count / max(len(results), 1), 4)

    # Confidence check
    conf_values = [
        r["prediction"].get("confidence")
        for r in results
        if r["success"] and r["prediction"] and r["prediction"].get("confidence") is not None
    ]
    conf_available_count = len(conf_values)

    return {
        "split_name": split_name,
        "model_version": model_version,
        "sample_count": len(samples),
        "successful_requests": valid_json_count,
        "failed_requests": len(results) - valid_json_count,
        "category_metrics": cat_metrics,
        "priority_metrics": prio_metrics,
        "language_metrics": lang_metrics,
        "per_language_breakdown": per_language_results,
        "json_validity": {
            "valid_count": valid_json_count,
            "total_count": len(results),
            "validity_rate": json_validity_rate,
            "malformed_response_rate": round(1.0 - json_validity_rate, 4),
        },
        "exact_structured_match": {
            "match_count": exact_match_count,
            "total_count": len(results),
            "match_rate": exact_match_rate,
        },
        "escalation_metrics": {
            "ground_truth_escalated_count": sum(1 for yt in y_true_esc if yt),
            "predicted_escalated_count": escalated_count,
            "eligible_for_automation_count": automated_count,
            "escalation_rate": escalation_rate,
            "automation_rate": automation_rate,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "escalation_precision": round(float(esc_p), 4),
            "escalation_recall": round(float(esc_r), 4),
            "escalation_f1": round(float(esc_f1), 4),
        },
        "confidence_metrics": {
            "confidence_available_count": conf_available_count,
            "confidence_available_rate": round(conf_available_count / max(len(results), 1), 4),
            "confidence_status": "unavailable" if conf_available_count == 0 else "available",
            "mean_confidence": round(float(np.mean(conf_values)), 4) if conf_values else None,
        },
        "latency_summary": {
            "mean_api_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "p50_api_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
            "p95_api_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
            "mean_model_ms": round(float(np.mean(model_latencies)), 2) if model_latencies else 0.0,
        },
    }


def benchmark_latency_and_throughput(
    client: TestClient,
    fixtures: List[Dict[str, Any]],
    model_version: str,
    concurrency_test: bool = True,
) -> Dict[str, Any]:
    # Warmup
    for i in range(3):
        client.post("/predict", json={"text": "Warmup request for latency benchmark."})

    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024**2)

    api_latencies = []
    model_latencies = []
    success_count = 0
    error_count = 0

    t_start = time.perf_counter()
    for item in fixtures:
        txt = item["text"]
        t0 = time.perf_counter()
        resp = client.post("/predict", json={"text": txt})
        t1 = time.perf_counter()
        elapsed_ms = round((t1 - t0) * 1000, 2)
        api_latencies.append(elapsed_ms)

        if resp.status_code == 200:
            success_count += 1
            data = resp.json()
            model_latencies.append(data.get("latency_ms", elapsed_ms))
        else:
            error_count += 1

    total_duration_sec = time.perf_counter() - t_start
    mem_after_mb = process.memory_info().rss / (1024**2)
    peak_mem_mb = max(mem_before_mb, mem_after_mb)

    throughput = round(len(fixtures) / max(total_duration_sec, 0.001), 2)

    return {
        "model_version": model_version,
        "sample_size": len(fixtures),
        "success_count": success_count,
        "error_count": error_count,
        "error_rate": round(error_count / max(len(fixtures), 1), 4),
        "total_duration_sec": round(total_duration_sec, 2),
        "throughput_req_per_sec": throughput,
        "peak_rss_memory_mb": round(peak_mem_mb, 2),
        "api_latency_ms": {
            "min": round(float(np.min(api_latencies)), 2),
            "p50": round(float(np.percentile(api_latencies, 50)), 2),
            "p90": round(float(np.percentile(api_latencies, 90)), 2),
            "p95": round(float(np.percentile(api_latencies, 95)), 2),
            "p99": round(float(np.percentile(api_latencies, 99)), 2),
            "max": round(float(np.max(api_latencies)), 2),
            "mean": round(float(np.mean(api_latencies)), 2),
        },
        "model_latency_ms": {
            "min": round(float(np.min(model_latencies)), 2) if model_latencies else 0.0,
            "p50": round(float(np.percentile(model_latencies, 50)), 2) if model_latencies else 0.0,
            "p90": round(float(np.percentile(model_latencies, 90)), 2) if model_latencies else 0.0,
            "p95": round(float(np.percentile(model_latencies, 95)), 2) if model_latencies else 0.0,
            "p99": round(float(np.percentile(model_latencies, 99)), 2) if model_latencies else 0.0,
            "max": round(float(np.max(model_latencies)), 2) if model_latencies else 0.0,
            "mean": round(float(np.mean(model_latencies)), 2) if model_latencies else 0.0,
        },
    }


def evaluate_high_risk_challenge(
    client: TestClient,
    challenge_samples: List[Dict[str, Any]],
    model_version: str,
) -> Dict[str, Any]:
    high_risk_total = 0
    high_risk_escalated = 0
    control_total = 0
    control_escalated = 0

    for item in challenge_samples:
        txt = item["text"]
        gt_esc = bool(item.get("requires_human", True))
        is_control = item.get("risk_type") == "control_benign"

        resp = client.post("/predict", json={"text": txt})
        if resp.status_code == 200:
            pred_esc = bool(resp.json().get("requires_human", True))
        else:
            pred_esc = True

        if is_control:
            control_total += 1
            if pred_esc:
                control_escalated += 1
        else:
            high_risk_total += 1
            if pred_esc:
                high_risk_escalated += 1

    high_risk_recall = round(high_risk_escalated / max(high_risk_total, 1), 4)
    false_negatives = high_risk_total - high_risk_escalated
    control_fpr = round(control_escalated / max(control_total, 1), 4)

    return {
        "model_version": model_version,
        "total_challenge_samples": len(challenge_samples),
        "high_risk_samples": high_risk_total,
        "high_risk_escalated": high_risk_escalated,
        "high_risk_recall": high_risk_recall,
        "false_negatives": false_negatives,
        "control_samples": control_total,
        "control_escalated": control_escalated,
        "control_false_positive_rate": control_fpr,
    }


def main():
    print("=" * 80)
    print("NEXUS-SLM: COMPLETE FACTUAL BACKEND PERFORMANCE AUDIT")
    print("=" * 80)

    env_info = get_environment_info()
    print("\n[Environment Information]")
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    # Load datasets
    val_path = PROJECT_ROOT / "data" / "validation" / "validation.jsonl"
    test_path = PROJECT_ROOT / "data" / "test" / "test.jsonl"
    challenge_path = PROJECT_ROOT / "data" / "evaluation" / "high_risk_challenge_set.jsonl"
    demo_path = PROJECT_ROOT / "data" / "demo_fixtures.json"

    val_samples = [json.loads(l) for l in open(val_path, encoding="utf-8") if l.strip()]
    test_samples = [json.loads(l) for l in open(test_path, encoding="utf-8") if l.strip()]
    challenge_samples = [json.loads(l) for l in open(challenge_path, encoding="utf-8") if l.strip()]
    demo_data = json.loads(open(demo_path, encoding="utf-8").read())
    demo_fixtures = demo_data.get("fixtures", demo_data)

    print(f"\nLoaded Datasets:")
    print(f"  Validation split (held-out): N={len(val_samples)}")
    print(f"  Test split (locked held-out): N={len(test_samples)}")
    print(f"  High-risk challenge set: N={len(challenge_samples)}")
    print(f"  Benchmark demo fixtures: N={len(demo_fixtures)}")

    models_to_audit = [
        {
            "version": "nexus-qlora-v1",
            "type": "production",
            "description": "Active production service runtime (operates in heuristic fallback due to recovered state)",
            "artifact_size_mb": 3.6,  # tokenizer + configs
        },
        {
            "version": "qwen25_05b_lora_seed42_q8_0",
            "type": "experimental",
            "description": "Experimental Qwen2.5-0.5B LoRA GGUF Q8_0 (8-bit quantization)",
            "artifact_size_mb": 506.47,
        },
        {
            "version": "qwen25_05b_lora_seed42_q4_k_m",
            "type": "experimental",
            "description": "Experimental Qwen2.5-0.5B LoRA GGUF Q4_K_M (4-bit quantization)",
            "artifact_size_mb": 379.38,
        },
        {
            "version": "qwen25_05b_lora_seed42_int8",
            "type": "experimental",
            "description": "Experimental Qwen2.5-0.5B LoRA ONNX Dynamic INT8",
            "artifact_size_mb": 473.04,
        },
    ]

    audit_results: Dict[str, Any] = {
        "audit_timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": env_info,
        "models_evaluated": {},
        "historical_registered_production_baseline": {
            "model_version": "nexus-qlora-v1 (historical intact weights)",
            "base_model": "HuggingFaceTB/SmolLM-135M-Instruct",
            "adapter_type": "QLoRA FP32",
            "validation_n44": {
                "category_accuracy": 0.4545,
                "macro_f1": 0.4412,
                "json_validity_rate": 0.9545,
            },
            "test_locked_n42": {
                "category_accuracy": 0.4286,
                "macro_f1": 0.4089,
                "weighted_f1": 0.4089,
                "priority_accuracy": 0.4762,
                "language_accuracy": 0.9524,
                "json_validity_rate": 0.9762,
                "exact_match_accuracy": 0.0714,
                "human_escalation_f1": 0.5952,
                "p50_latency_ms": 1204.46,
                "p95_latency_ms": 1306.49,
                "peak_ram_mb": 516.65,
            },
            "source": "models/registry.json (canonical record prior to adapter file loss)",
        },
    }

    flat_csv_rows = []

    for m in models_to_audit:
        ver = m["version"]
        print("\n" + "-" * 75)
        print(f"AUDITING MODEL: {ver} ({m['type'].upper()})")
        print(f"Description: {m['description']}")
        print("-" * 75)

        engine = load_model_instance(ver)
        api_module.inference_engine = engine
        api_module.ACTIVE_VERSION = ver
        client = TestClient(app)

        # 1. Evaluate on Validation Set (N=44)
        print(f"  [1/4] Running evaluation on held-out validation split (N={len(val_samples)})...")
        val_eval = evaluate_split_with_client(client, val_samples, "validation_v1", ver)
        print(f"    Val Category Acc: {val_eval['category_metrics']['correct']}/{val_eval['category_metrics']['total']} ({val_eval['category_metrics']['accuracy']}) | Macro-F1: {val_eval['category_metrics']['macro_f1']}")
        print(f"    Val Exact Match: {val_eval['exact_structured_match']['match_count']}/{val_eval['exact_structured_match']['total_count']} ({val_eval['exact_structured_match']['match_rate']}) | JSON Valid: {val_eval['json_validity']['validity_rate']}")

        # 2. Evaluate on Test Set (N=42)
        print(f"  [2/4] Running evaluation on held-out locked test split (N={len(test_samples)})...")
        test_eval = evaluate_split_with_client(client, test_samples, "test_v1_locked", ver)
        print(f"    Test Category Acc: {test_eval['category_metrics']['correct']}/{test_eval['category_metrics']['total']} ({test_eval['category_metrics']['accuracy']}) | Macro-F1: {test_eval['category_metrics']['macro_f1']}")
        print(f"    Test Exact Match: {test_eval['exact_structured_match']['match_count']}/{test_eval['exact_structured_match']['total_count']} ({test_eval['exact_structured_match']['match_rate']}) | JSON Valid: {test_eval['json_validity']['validity_rate']}")

        # 3. Evaluate High-Risk Challenge Set (N=70)
        print(f"  [3/4] Running evaluation on high-risk challenge set (N={len(challenge_samples)})...")
        chal_eval = evaluate_high_risk_challenge(client, challenge_samples, ver)
        print(f"    High-Risk Recall: {chal_eval['high_risk_escalated']}/{chal_eval['high_risk_samples']} ({chal_eval['high_risk_recall']}) | False Negatives: {chal_eval['false_negatives']}")

        # 4. Benchmark Latency and Throughput (N=30)
        print(f"  [4/4] Running latency & throughput benchmark on demo fixtures (N={len(demo_fixtures)})...")
        bench_eval = benchmark_latency_and_throughput(client, demo_fixtures, ver)
        print(f"    P50 API: {bench_eval['api_latency_ms']['p50']} ms | P95 API: {bench_eval['api_latency_ms']['p95']} ms | Throughput: {bench_eval['throughput_req_per_sec']} req/s | Peak RAM: {bench_eval['peak_rss_memory_mb']} MB")

        model_audit_entry = {
            "metadata": m,
            "validation_split": val_eval,
            "test_split": test_eval,
            "high_risk_challenge": chal_eval,
            "benchmark_performance": bench_eval,
        }
        audit_results["models_evaluated"][ver] = model_audit_entry

        # Add to CSV flat summary
        flat_csv_rows.append({
            "model_version": ver,
            "role": m["type"],
            "val_cat_acc": f"{val_eval['category_metrics']['correct']}/{val_eval['category_metrics']['total']} ({val_eval['category_metrics']['accuracy']})",
            "val_macro_f1": val_eval["category_metrics"]["macro_f1"],
            "val_exact_match": f"{val_eval['exact_structured_match']['match_count']}/{val_eval['exact_structured_match']['total_count']} ({val_eval['exact_structured_match']['match_rate']})",
            "val_json_validity": val_eval["json_validity"]["validity_rate"],
            "test_cat_acc": f"{test_eval['category_metrics']['correct']}/{test_eval['category_metrics']['total']} ({test_eval['category_metrics']['accuracy']})",
            "test_macro_f1": test_eval["category_metrics"]["macro_f1"],
            "test_exact_match": f"{test_eval['exact_structured_match']['match_count']}/{test_eval['exact_structured_match']['total_count']} ({test_eval['exact_structured_match']['match_rate']})",
            "test_json_validity": test_eval["json_validity"]["validity_rate"],
            "high_risk_recall": f"{chal_eval['high_risk_escalated']}/{chal_eval['high_risk_samples']} ({chal_eval['high_risk_recall']})",
            "false_negatives": chal_eval["false_negatives"],
            "p50_latency_ms": bench_eval["api_latency_ms"]["p50"],
            "p95_latency_ms": bench_eval["api_latency_ms"]["p95"],
            "throughput_rps": bench_eval["throughput_req_per_sec"],
            "peak_ram_mb": bench_eval["peak_rss_memory_mb"],
            "artifact_size_mb": m["artifact_size_mb"],
        })

    # Save artifacts
    artifacts_dir = PROJECT_ROOT / "experiments" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    json_path = artifacts_dir / "backend_performance_audit_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[Artifact Saved] Machine-readable audit results: {json_path}")

    csv_path = artifacts_dir / "backend_performance_audit_metrics.csv"
    if flat_csv_rows:
        fieldnames = list(flat_csv_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_csv_rows)
        print(f"[Artifact Saved] Tabular metrics summary: {csv_path}")

    print("\n" + "=" * 80)
    print("AUDIT EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
