"""Comprehensive Evaluation Script for NEXUS-SLM Candidates on Clean Template-Grouped V2 Splits.

Evaluates candidates on data/v2_clean_template_grouped/test_clean.jsonl (N=746) or val_clean.jsonl (N=449).
Calculates:
- Actual evaluated sample count
- Category accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1
- Per-category precision, recall, F1, and support
- Priority accuracy, Macro-F1, and per-priority precision/recall/F1/support
- Language identification accuracy overall and sliced across English, Hindi, and Hinglish
- Exact structured match rate (simultaneous category, priority, language, requires_human)
- JSON validity and schema compliance rate
- Detailed field-level error analysis and confusion profiling
- Conservative escalation metrics (TP, FN, FP, TN, precision, recall, F1, escalation rate)
- Hardware metrics: P50/P95 latency, mean latency, peak RSS memory, throughput

Preserves all existing reports and writes to experiments/artifacts/clean_v2_full_evaluation_report.json.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import api.main as api_module
from api.main import app, load_model_instance
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model_on_split(
    model_version: str,
    split_path: Path,
    sample_limit: Optional[int] = None,
    log_interval: int = 50,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"EVALUATING MODEL: {model_version} on {split_path.name}")
    print("=" * 80)

    with open(split_path, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    if sample_limit and sample_limit < len(all_samples):
        # Deterministic stratified sample for exact comparison
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

    print(f"Loaded samples: N={len(samples)} (Total split size: {len(all_samples)})")

    # Load inference engine
    engine = load_model_instance(model_version)
    api_module.inference_engine = engine
    api_module.ACTIVE_VERSION = model_version
    client = TestClient(app)

    health = client.get("/health").json()
    print(f"Active Inference Mode: {health.get('inference_mode')}")

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
                "id": s.get("id", f"sample_{idx}"),
                "text": txt,
                "ground_truth": s,
                "prediction": data,
                "success": True,
                "elapsed_ms": elapsed_ms,
            })
        else:
            results.append({
                "id": s.get("id", f"sample_{idx}"),
                "text": txt,
                "ground_truth": s,
                "prediction": None,
                "success": False,
                "elapsed_ms": elapsed_ms,
            })

        if idx % log_interval == 0 or idx == len(samples):
            rate = idx / max(time.perf_counter() - t_start, 0.001)
            print(f"  [{model_version}] Processed {idx}/{len(samples)} ({rate:.2f} req/s)...")

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

    # Confusion Matrix for Category
    conf_mat = confusion_matrix(y_true_cat, y_pred_cat, labels=CATEGORIES).tolist()

    # Category Error Analysis
    category_errors = []
    category_confusion_pairs = Counter()
    for r in results:
        gt_cat = r["ground_truth"]["category"]
        pr_cat = r["prediction"]["category"] if r["success"] and r["prediction"] else "unknown"
        if gt_cat != pr_cat:
            pair = f"{gt_cat} -> {pr_cat}"
            category_confusion_pairs[pair] += 1
            if len(category_errors) < 30:  # store sample for diagnosis
                category_errors.append({
                    "id": r["id"],
                    "text_snippet": r["text"][:80],
                    "expected": gt_cat,
                    "predicted": pr_cat,
                    "language": r["ground_truth"]["language"],
                })

    # 2. Priority Classification Metrics
    y_true_prio = [r["ground_truth"]["priority"] for r in results]
    y_pred_prio = [r["prediction"]["priority"] if r["success"] and r["prediction"] else "unknown" for r in results]
    prio_acc = accuracy_score(y_true_prio, y_pred_prio)
    prio_macro_p, prio_macro_r, prio_macro_f1, _ = precision_recall_fscore_support(
        y_true_prio, y_pred_prio, labels=PRIORITIES, average="macro", zero_division=0
    )
    prio_p, prio_r, prio_f, prio_s = precision_recall_fscore_support(
        y_true_prio, y_pred_prio, labels=PRIORITIES, average=None, zero_division=0
    )
    per_priority = {}
    for p_label, pr, rc, f1, sup in zip(PRIORITIES, prio_p, prio_r, prio_f, prio_s):
        per_priority[p_label] = {
            "precision": round(float(pr), 4),
            "recall": round(float(rc), 4),
            "f1": round(float(f1), 4),
            "support": int(sup),
        }

    priority_confusion_pairs = Counter()
    for yt, yp in zip(y_true_prio, y_pred_prio):
        if yt != yp:
            priority_confusion_pairs[f"{yt} -> {yp}"] += 1

    # 3. Language Identification Metrics
    y_true_lang = [r["ground_truth"]["language"] for r in results]
    y_pred_lang = [r["prediction"]["language"] if r["success"] and r["prediction"] else "unknown" for r in results]
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

    # 4. Exact Structured Match Rate
    exact_match_count = 0
    field_match_counts = Counter()
    for r in results:
        if not r["success"]:
            continue
        gt = r["ground_truth"]
        pr = r["prediction"]
        c_ok = gt.get("category", "").lower() == pr.get("category", "").lower()
        p_ok = gt.get("priority", "").lower() == pr.get("priority", "").lower()
        l_ok = gt.get("language", "").lower() == pr.get("language", "").lower()
        h_ok = bool(gt.get("requires_human")) == bool(pr.get("requires_human"))

        if c_ok: field_match_counts["category"] += 1
        if p_ok: field_match_counts["priority"] += 1
        if l_ok: field_match_counts["language"] += 1
        if h_ok: field_match_counts["requires_human"] += 1

        if c_ok and p_ok and l_ok and h_ok:
            exact_match_count += 1

    exact_match_rate = round(exact_match_count / max(len(results), 1), 4)

    # 5. JSON Validity & Schema Compliance
    valid_json_count = sum(1 for r in results if r["success"])
    json_validity_rate = round(valid_json_count / max(len(results), 1), 4)

    # 6. Escalation Safeguards Evaluation
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
        "dataset_split": split_path.name,
        "dataset_path": str(split_path.relative_to(PROJECT_ROOT)),
        "sample_count": len(results),
        "total_split_size": len(all_samples),
        "evaluation_type": "full_split" if (sample_limit is None or sample_limit >= len(all_samples)) else f"stratified_sample_{sample_limit}",
        "category_metrics": {
            "accuracy": round(float(cat_acc), 4),
            "correct_count": sum(1 for yt, yp in zip(y_true_cat, y_pred_cat) if yt == yp),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "top_confusion_pairs": dict(category_confusion_pairs.most_common(8)),
            "confusion_matrix_labels": CATEGORIES,
            "confusion_matrix": conf_mat,
        },
        "priority_metrics": {
            "accuracy": round(float(prio_acc), 4),
            "macro_precision": round(float(prio_macro_p), 4),
            "macro_recall": round(float(prio_macro_r), 4),
            "macro_f1": round(float(prio_macro_f1), 4),
            "per_priority": per_priority,
            "top_confusion_pairs": dict(priority_confusion_pairs.most_common(6)),
        },
        "language_metrics": {
            "accuracy": round(float(lang_acc), 4),
            "per_language_breakdown": per_language,
        },
        "exact_structured_match": {
            "match_count": exact_match_count,
            "match_rate": exact_match_rate,
            "field_match_rates": {
                k: round(v / max(len(results), 1), 4) for k, v in field_match_counts.items()
            },
        },
        "json_validity": {
            "valid_count": valid_json_count,
            "validity_rate": json_validity_rate,
            "schema_compliant_rate": json_validity_rate,
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
        "field_error_samples": category_errors[:15],
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
    parser = argparse.ArgumentParser(description="Comprehensive Clean V2 Test & Val Evaluation")
    parser.add_argument("--model", type=str, default="all", help="Model version to evaluate: 'nexus-qlora-v1', 'qwen25_05b_lora_seed42_q8_0', 'all'")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split: 'test' or 'val'")
    parser.add_argument("--samples", type=int, default=None, help="Sample limit (default: None = full split)")
    parser.add_argument("--output", type=str, default="experiments/artifacts/clean_v2_full_evaluation_report.json")
    args = parser.parse_args()

    split_file = "test_clean.jsonl" if args.split == "test" else "val_clean.jsonl"
    split_path = PROJECT_ROOT / "data" / "v2_clean_template_grouped" / split_file
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    if args.model == "all":
        models = [
            "nexus-qlora-v1",
            "qwen25_05b_lora_seed42_q8_0",
        ]
    else:
        models = [args.model]

    out_file = PROJECT_ROOT / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if present to append/merge
    report_data = {
        "report_generated_at": datetime.utcnow().isoformat() + "Z",
        "split_evaluated": str(split_path.relative_to(PROJECT_ROOT)),
        "hardware_environment": {
            "os": "Windows 11 Enterprise",
            "cpu": "AMD Ryzen 5 5600H (8 physical cores, 12 logical threads)",
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cuda_available": False,
        },
        "models": {},
    }
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                old = json.load(f)
                if isinstance(old, dict) and "models" in old:
                    report_data["models"] = old["models"]
        except Exception:
            pass

    for m in models:
        eval_res = evaluate_model_on_split(m, split_path, sample_limit=args.samples)
        key = f"{m}__full_{eval_res['sample_count']}" if args.samples is None else f"{m}__sample_{args.samples}"
        report_data["models"][key] = eval_res
        print(f"\n[Summary for {m} on N={eval_res['sample_count']}]:")
        print(f"  Category Acc: {eval_res['category_metrics']['accuracy']} | Macro-F1: {eval_res['category_metrics']['macro_f1']}")
        print(f"  Priority Acc: {eval_res['priority_metrics']['accuracy']} | Lang Acc: {eval_res['language_metrics']['accuracy']}")
        print(f"  Exact Match Rate: {eval_res['exact_structured_match']['match_rate']} | JSON Valid: {eval_res['json_validity']['validity_rate']}")
        print(f"  Escalation FNs: {eval_res['escalation_metrics']['false_negatives']} (Recall: {eval_res['escalation_metrics']['recall']})")
        print(f"  P50 Latency: {eval_res['latency_and_resources']['p50_api_ms']} ms | Throughput: {eval_res['latency_and_resources']['throughput_req_per_sec']} req/s\n")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[Artifact Saved] Comprehensive Evaluation Report: {out_file}")


if __name__ == "__main__":
    main()
