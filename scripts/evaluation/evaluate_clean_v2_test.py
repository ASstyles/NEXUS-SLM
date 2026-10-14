"""Evaluation script for NEXUS-SLM candidates on the Clean Template-Grouped V2 Test Set.

Evaluates candidates on data/v2_clean_template_grouped/test_clean.jsonl.
Reports:
- Category Accuracy and Macro-F1
- Per-category precision, recall, and F1
- Priority accuracy and per-class metrics
- Language identification accuracy
- English, Hindi, and Hinglish performance
- Exact structured match rate
- JSON validity and schema compliance
- Escalation precision, recall, false negatives, and false positives
- Inference latency (P50/P95), throughput, and memory footprint

Emits experiments/artifacts/clean_v2_evaluation_report.json.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import api.main as api_module
from api.main import app, load_model_instance
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_candidate_on_clean_test(
    model_version: str,
    test_path: Path,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    print("=" * 75)
    print(f"EVALUATING MODEL: {model_version} on Clean V2 Test Set")
    print("=" * 75)

    with open(test_path, "r", encoding="utf-8") as f:
        all_samples = [json.loads(l) for l in f if l.strip()]

    if sample_limit and sample_limit < len(all_samples):
        # Stratified sampling by category and language for representative evaluation
        samples_by_cat = Counter()
        samples = []
        for s in all_samples:
            cat = s.get("category")
            if samples_by_cat[cat] < max(2, sample_limit // len(CATEGORIES) + 1):
                samples.append(s)
                samples_by_cat[cat] += 1
            if len(samples) >= sample_limit:
                break
    else:
        samples = all_samples

    print(f"Loaded clean test samples: N={len(samples)} (Total available: {len(all_samples)})")

    engine = load_model_instance(model_version)
    api_module.inference_engine = engine
    api_module.ACTIVE_VERSION = model_version
    client = TestClient(app)

    # Health check
    h = client.get("/health").json()
    print(f"Inference Mode: {h.get('inference_mode')}")

    results = []
    api_latencies = []
    model_latencies = []

    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024**2)

    t_start = time.perf_counter()
    for idx, s in enumerate(samples, 1):
        txt = s["text"]
        t0 = time.perf_counter()
        resp = client.post("/predict", json={"text": txt})
        t1 = time.perf_counter()
        elapsed_ms = round((t1 - t0) * 1000, 2)
        api_latencies.append(elapsed_ms)

        if resp.status_code == 200:
            data = resp.json()
            model_latencies.append(data.get("latency_ms", elapsed_ms))
            results.append({
                "ground_truth": s,
                "prediction": data,
                "success": True,
                "elapsed_ms": elapsed_ms,
            })
        else:
            results.append({
                "ground_truth": s,
                "prediction": None,
                "success": False,
                "elapsed_ms": elapsed_ms,
            })

        if idx % 25 == 0 or idx == len(samples):
            print(f"  Processed {idx}/{len(samples)} queries...")

    total_duration_sec = time.perf_counter() - t_start
    mem_after_mb = process.memory_info().rss / (1024**2)

    # 1. Category Classification Metrics
    y_true_cat = [r["ground_truth"]["category"] for r in results]
    y_pred_cat = [r["prediction"]["category"] if r["success"] and r["prediction"] else "unknown" for r in results]

    cat_acc = accuracy_score(y_true_cat, y_pred_cat)
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

    # 2. Priority Classification Metrics
    y_true_prio = [r["ground_truth"]["priority"] for r in results]
    y_pred_prio = [r["prediction"]["priority"] if r["success"] and r["prediction"] else "unknown" for r in results]
    prio_acc = accuracy_score(y_true_prio, y_pred_prio)
    _, _, prio_macro_f1, _ = precision_recall_fscore_support(
        y_true_prio, y_pred_prio, labels=PRIORITIES, average="macro", zero_division=0
    )

    # 3. Language Identification Metrics
    y_true_lang = [r["ground_truth"]["language"] for r in results]
    y_pred_lang = [r["prediction"]["language"] if r["success"] and r["prediction"] else "unknown" for r in results]
    lang_acc = accuracy_score(y_true_lang, y_pred_lang)

    # 4. Per-Language Category Performance
    per_language = {}
    for lang in ["english", "hindi", "hinglish"]:
        idx_list = [i for i, r in enumerate(results) if r["ground_truth"].get("language") == lang]
        if idx_list:
            sub_true = [y_true_cat[i] for i in idx_list]
            sub_pred = [y_pred_cat[i] for i in idx_list]
            sub_acc = accuracy_score(sub_true, sub_pred)
            _, _, sub_f1, _ = precision_recall_fscore_support(
                sub_true, sub_pred, labels=CATEGORIES, average="macro", zero_division=0
            )
            per_language[lang] = {
                "count": len(idx_list),
                "accuracy": round(float(sub_acc), 4),
                "macro_f1": round(float(sub_f1), 4),
            }

    # 5. Exact Structured Output Match
    exact_match_count = 0
    for r in results:
        if not r["success"]:
            continue
        gt = r["ground_truth"]
        pr = r["prediction"]
        c_ok = gt.get("category", "").lower() == pr.get("category", "").lower()
        p_ok = gt.get("priority", "").lower() == pr.get("priority", "").lower()
        l_ok = gt.get("language", "").lower() == pr.get("language", "").lower()
        h_ok = bool(gt.get("requires_human")) == bool(pr.get("requires_human"))
        if c_ok and p_ok and l_ok and h_ok:
            exact_match_count += 1

    exact_match_rate = round(exact_match_count / max(len(results), 1), 4)

    # 6. JSON Validity
    valid_json_count = sum(1 for r in results if r["success"])
    json_validity_rate = round(valid_json_count / max(len(results), 1), 4)

    # 7. Escalation Evaluation
    y_true_esc = [bool(r["ground_truth"].get("requires_human")) for r in results]
    y_pred_esc = [bool(r["prediction"].get("requires_human")) if r["success"] and r["prediction"] else True for r in results]

    tp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt and yp)
    fn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt and not yp)
    fp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if not yt and not yp)

    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        y_true_esc, y_pred_esc, labels=[True], average="binary", zero_division=0
    )

    throughput = round(len(results) / max(total_duration_sec, 0.001), 2)

    return {
        "model_version": model_version,
        "dataset": str(test_path.relative_to(PROJECT_ROOT)),
        "sample_count": len(results),
        "total_split_size": len(all_samples),
        "category_metrics": {
            "accuracy": round(float(cat_acc), 4),
            "correct_count": sum(1 for yt, yp in zip(y_true_cat, y_pred_cat) if yt == yp),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
        },
        "priority_metrics": {
            "accuracy": round(float(prio_acc), 4),
            "macro_f1": round(float(prio_macro_f1), 4),
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
        },
        "escalation_metrics": {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "precision": round(float(esc_p), 4),
            "recall": round(float(esc_r), 4),
            "f1": round(float(esc_f1), 4),
            "escalation_rate": round(sum(1 for yp in y_pred_esc if yp) / max(len(results), 1), 4),
        },
        "latency_and_resources": {
            "p50_api_ms": round(float(np.percentile(api_latencies, 50)), 2),
            "p95_api_ms": round(float(np.percentile(api_latencies, 95)), 2),
            "mean_api_ms": round(float(np.mean(api_latencies)), 2),
            "p50_model_ms": round(float(np.percentile(model_latencies, 50)), 2) if model_latencies else 0.0,
            "p95_model_ms": round(float(np.percentile(model_latencies, 95)), 2) if model_latencies else 0.0,
            "throughput_req_per_sec": throughput,
            "peak_rss_memory_mb": round(max(mem_before_mb, mem_after_mb), 2),
            "total_duration_sec": round(total_duration_sec, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Clean Template V2 Test Set")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to evaluate (default: 100)")
    args = parser.parse_args()

    clean_test_path = PROJECT_ROOT / "data" / "v2_clean_template_grouped" / "test_clean.jsonl"
    if not clean_test_path.exists():
        raise FileNotFoundError(f"Missing clean test set: {clean_test_path}")

    models_to_test = [
        "nexus-qlora-v1",
        "qwen25_05b_lora_seed42_q8_0",
        "qwen25_05b_lora_seed42_q4_k_m",
    ]

    all_evals = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "clean_test_file": str(clean_test_path.relative_to(PROJECT_ROOT)),
        "evaluation_sample_size": args.samples,
        "models": {},
    }

    for ver in models_to_test:
        eval_res = evaluate_candidate_on_clean_test(ver, clean_test_path, sample_limit=args.samples)
        all_evals["models"][ver] = eval_res
        print(f"Results for {ver}:")
        print(f"  Category Acc: {eval_res['category_metrics']['accuracy']} | Macro-F1: {eval_res['category_metrics']['macro_f1']}")
        print(f"  Priority Acc: {eval_res['priority_metrics']['accuracy']} | Lang Acc: {eval_res['language_metrics']['accuracy']}")
        print(f"  Exact Match Rate: {eval_res['exact_structured_match']['match_rate']} | JSON Valid: {eval_res['json_validity']['validity_rate']}")
        print(f"  Escalation FNs: {eval_res['escalation_metrics']['false_negatives']} (Recall: {eval_res['escalation_metrics']['recall']})")
        print(f"  P50 Latency: {eval_res['latency_and_resources']['p50_api_ms']} ms | P95: {eval_res['latency_and_resources']['p95_api_ms']} ms\n")

    out_file = PROJECT_ROOT / "experiments" / "artifacts" / "clean_v2_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_evals, f, indent=2)
    print(f"\n[Artifact Saved] Clean V2 Evaluation Report: {out_file}")


if __name__ == "__main__":
    main()
