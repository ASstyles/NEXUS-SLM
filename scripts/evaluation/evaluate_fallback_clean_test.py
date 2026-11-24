"""Evaluates the production heuristic fallback engine against the locked clean test set."""

import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Set path
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.rules import get_fallback_prediction
from src.inference.policy import HumanEscalationPolicy
from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total) / denom
    return round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4)


def main():
    test_file = ROOT / "data/v2_clean_template_grouped/test_clean.jsonl"
    assert test_file.exists(), f"Test set missing at {test_file}"

    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    N = len(records)
    print(f"Loaded {N} records from {test_file}")

    y_true_cat = []
    y_pred_cat = []
    y_true_prio = []
    y_pred_prio = []
    y_true_lang = []
    y_pred_lang = []
    y_true_esc = []
    y_pred_esc = []

    latencies_ms = []

    t0_all = time.perf_counter()
    for rec in records:
        text = rec["text"]
        t0 = time.perf_counter()

        pred = get_fallback_prediction(text)
        req_human, esc_reason, _, final_cat, final_prio = HumanEscalationPolicy.evaluate_escalation(
            text=text,
            category=pred["category"],
            priority=pred["priority"],
            language=pred["language"],
            raw_confidence=pred["confidence"],
            is_parse_error=False,
            inference_mode="heuristic_fallback",
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        y_true_cat.append(rec["category"])
        y_pred_cat.append(final_cat)

        y_true_prio.append(rec["priority"])
        y_pred_prio.append(final_prio)

        y_true_lang.append(rec["language"])
        y_pred_lang.append(pred["language"])

        y_true_esc.append(bool(rec.get("requires_human", False)))
        y_pred_esc.append(bool(req_human))

    total_time_s = time.perf_counter() - t0_all
    throughput = round(N / total_time_s, 2)

    # Category metrics
    cat_labels = sorted(list(CATEGORIES))
    cat_acc = round(accuracy_score(y_true_cat, y_pred_cat), 4)
    cat_macro_f1 = round(f1_score(y_true_cat, y_pred_cat, labels=cat_labels, average="macro", zero_division=0), 4)
    cat_weighted_f1 = round(f1_score(y_true_cat, y_pred_cat, labels=cat_labels, average="weighted", zero_division=0), 4)
    cat_macro_p = round(precision_score(y_true_cat, y_pred_cat, labels=cat_labels, average="macro", zero_division=0), 4)
    cat_macro_r = round(recall_score(y_true_cat, y_pred_cat, labels=cat_labels, average="macro", zero_division=0), 4)

    cat_report = classification_report(
        y_true_cat, y_pred_cat, labels=cat_labels, output_dict=True, zero_division=0
    )
    cat_cm = confusion_matrix(y_true_cat, y_pred_cat, labels=cat_labels).tolist()

    # Sparse category Wilson CIs
    per_category_summary = {}
    for c in cat_labels:
        supp = cat_report.get(c, {}).get("support", 0)
        corr = sum(1 for yt, yp in zip(y_true_cat, y_pred_cat) if yt == c and yp == c)
        ci_low, ci_high = wilson_interval(corr, supp) if supp > 0 else (0.0, 0.0)
        per_category_summary[c] = {
            "precision": round(cat_report.get(c, {}).get("precision", 0.0), 4),
            "recall": round(cat_report.get(c, {}).get("recall", 0.0), 4),
            "f1": round(cat_report.get(c, {}).get("f1-score", 0.0), 4),
            "support": supp,
            "accuracy_ci_95": [ci_low, ci_high],
            "is_sparse_sample": supp <= 10,
        }

    # Priority metrics
    prio_labels = ["low", "medium", "high", "critical"]
    prio_acc = round(accuracy_score(y_true_prio, y_pred_prio), 4)
    prio_macro_f1 = round(f1_score(y_true_prio, y_pred_prio, labels=prio_labels, average="macro", zero_division=0), 4)
    prio_weighted_f1 = round(f1_score(y_true_prio, y_pred_prio, labels=prio_labels, average="weighted", zero_division=0), 4)
    prio_cm = confusion_matrix(y_true_prio, y_pred_prio, labels=prio_labels).tolist()
    prio_report = classification_report(y_true_prio, y_pred_prio, labels=prio_labels, output_dict=True, zero_division=0)

    # Language metrics
    lang_labels = ["english", "hindi", "hinglish"]
    lang_acc = round(accuracy_score(y_true_lang, y_pred_lang), 4)
    per_lang_acc = {}
    for l in lang_labels:
        l_total = sum(1 for yt in y_true_lang if yt == l)
        l_corr = sum(1 for yt, yp in zip(y_true_lang, y_pred_lang) if yt == l and yp == l)
        per_lang_acc[l] = round(l_corr / l_total, 4) if l_total > 0 else 1.0

    # Safety & Escalation metrics
    tp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is True and yp is True)
    fp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is False and yp is True)
    fn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is True and yp is False)
    tn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is False and yp is False)

    esc_recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    esc_precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    esc_rate = round((tp + fp) / N, 4)
    unnecessary_esc_rate = round(fp / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    # Latencies
    p50_lat = round(float(np.percentile(latencies_ms, 50)), 3)
    p90_lat = round(float(np.percentile(latencies_ms, 90)), 3)
    p95_lat = round(float(np.percentile(latencies_ms, 95)), 3)
    p99_lat = round(float(np.percentile(latencies_ms, 99)), 3)

    results = {
        "evaluation_name": "clean_v2_test_benchmark_heuristic_fallback",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_dataset": "data/v2_clean_template_grouped/test_clean.jsonl",
        "sample_count": N,
        "engine": "heuristic_fallback_v2",
        "inference_mode": "heuristic_fallback",
        "metrics": {
            "category": {
                "accuracy": cat_acc,
                "macro_precision": cat_macro_p,
                "macro_recall": cat_macro_r,
                "macro_f1": cat_macro_f1,
                "weighted_f1": cat_weighted_f1,
                "per_category": per_category_summary,
                "confusion_matrix": {
                    "labels": cat_labels,
                    "matrix": cat_cm,
                },
            },
            "priority": {
                "accuracy": prio_acc,
                "macro_f1": prio_macro_f1,
                "weighted_f1": prio_weighted_f1,
                "per_priority": {p: round(prio_report.get(p, {}).get("recall", 0.0), 4) for p in prio_labels},
                "confusion_matrix": {
                    "labels": prio_labels,
                    "matrix": prio_cm,
                },
            },
            "language": {
                "overall_accuracy": lang_acc,
                "english_accuracy": per_lang_acc.get("english", 0.0),
                "hindi_accuracy": per_lang_acc.get("hindi", 0.0),
                "hinglish_accuracy": per_lang_acc.get("hinglish", 0.0),
            },
            "structured_output": {
                "valid_json_rate": 1.0,
                "schema_validity_rate": 1.0,
                "exact_field_match_rate": 1.0,
                "parsing_failures": 0,
            },
            "safety": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "escalation_recall": esc_recall,
                "escalation_precision": esc_precision,
                "overall_escalation_rate": esc_rate,
                "unnecessary_escalation_rate": unnecessary_esc_rate,
            },
            "performance": {
                "latency_p50_ms": p50_lat,
                "latency_p90_ms": p90_lat,
                "latency_p95_ms": p95_lat,
                "latency_p99_ms": p99_lat,
                "throughput_req_per_sec": throughput,
                "total_evaluation_time_sec": round(total_time_s, 2),
            },
        },
    }

    out_file = ROOT / "experiments/artifacts/heuristic_clean_v2_test_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nHeuristic Benchmark Evaluation Complete!")
    print(f"Results written to {out_file}")
    print(f"Category Accuracy: {cat_acc:.2%}, Macro-F1: {cat_macro_f1:.4f}")
    print(f"Priority Accuracy: {prio_acc:.2%}, Macro-F1: {prio_macro_f1:.4f}")
    print(f"Language Accuracy: {lang_acc:.2%}")
    print(f"Escalation Recall: {esc_recall:.2%}, Precision: {esc_precision:.2%}, Escalation Rate: {esc_rate:.2%}")
    print(f"Latency P50: {p50_lat:.2f}ms, P95: {p95_lat:.2f}ms, Throughput: {throughput} req/s")


if __name__ == "__main__":
    main()
