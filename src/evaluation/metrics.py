"""Comprehensive evaluation metrics for NEXUS-SLM."""

import json
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


class EvaluationMetrics:
    """Calculates classification, structured JSON validity, and system performance metrics."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Calculates accuracy, macro/weighted precision, recall, F1, and per-class metrics."""
        if not y_true or not y_pred:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "per_class": {},
            }

        all_labels = labels or sorted(list(set(y_true).union(set(y_pred))))

        # Macro and weighted metrics
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0, labels=all_labels
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0, labels=all_labels
        )
        acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)

        # Per-class metrics
        prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=all_labels
        )
        per_class: Dict[str, Dict[str, float]] = {}
        for lbl, p, r, f, s in zip(all_labels, prec_c, rec_c, f1_c, sup_c):
            per_class[lbl] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_labels).tolist()

        return {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_class": per_class,
            "labels": all_labels,
            "confusion_matrix": cm,
        }

    @staticmethod
    def calculate_structured_metrics(
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        json_valid_flags: List[bool],
    ) -> Dict[str, Any]:
        """
        Calculates JSON validity, field-level accuracy, and exact-match accuracy.
        """
        n = len(ground_truths)
        if n == 0:
            return {}

        valid_json_count = sum(1 for v in json_valid_flags if v)
        json_validity_rate = round(valid_json_count / n, 4)

        cat_correct = 0
        prio_correct = 0
        lang_correct = 0
        human_correct = 0
        exact_matches = 0

        for gt, pred in zip(ground_truths, predictions):
            c_ok = (str(gt.get("category", "")).lower() == str(pred.get("category", "")).lower())
            p_ok = (str(gt.get("priority", "")).lower() == str(pred.get("priority", "")).lower())
            l_ok = (str(gt.get("language", "")).lower() == str(pred.get("language", "")).lower())
            h_ok = (bool(gt.get("requires_human")) == bool(pred.get("requires_human")))

            if c_ok:
                cat_correct += 1
            if p_ok:
                prio_correct += 1
            if l_ok:
                lang_correct += 1
            if h_ok:
                human_correct += 1

            if c_ok and p_ok and l_ok and h_ok:
                exact_matches += 1

        field_accuracy = round((cat_correct + prio_correct + lang_correct + human_correct) / (4 * n), 4)

        return {
            "json_validity_rate": json_validity_rate,
            "category_accuracy": round(cat_correct / n, 4),
            "priority_accuracy": round(prio_correct / n, 4),
            "language_accuracy": round(lang_correct / n, 4),
            "human_escalation_accuracy": round(human_correct / n, 4),
            "field_level_accuracy": field_accuracy,
            "exact_match_accuracy": round(exact_matches / n, 4),
            "total_evaluated": n,
        }

    @staticmethod
    def calculate_performance_metrics(
        latencies_ms: List[float],
        ram_mb: Optional[float] = None,
        vram_mb: Optional[float] = None,
        model_size_mb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculates latency percentiles, throughput, and hardware memory consumption."""
        if not latencies_ms:
            return {
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "mean_latency_ms": 0.0,
                "throughput_req_per_sec": 0.0,
            }

        arr = np.array(latencies_ms)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean_lat = float(np.mean(arr))
        throughput = round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0

        return {
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "throughput_req_per_sec": throughput,
            "ram_mb": ram_mb,
            "vram_mb": vram_mb,
            "model_size_mb": model_size_mb,
        }
