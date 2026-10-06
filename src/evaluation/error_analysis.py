"""Error analysis and failure mode classification for NEXUS-SLM."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter
from src.utils.constants import ERROR_TYPES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorAnalyzer:
    """Classifies mispredictions and identifies systematic failure patterns."""

    SIMILAR_CATEGORY_PAIRS = {
        frozenset(["billing", "payment_failure"]),
        frozenset(["order_cancel", "refund_request"]),
        frozenset(["order_tracking", "delivery_issue"]),
        frozenset(["account_access", "password_reset"]),
        frozenset(["product_complaint", "refund_request"]),
        frozenset(["general_query", "subscription"]),
    }

    def __init__(self, output_dir: str = "experiments/artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def classify_error_type(
        self,
        ground_truth: Dict[str, Any],
        predicted: Dict[str, Any],
        text: str,
        confidence: float,
    ) -> str:
        """Classifies a specific misprediction into our standardized error taxonomy."""
        gt_cat = str(ground_truth.get("category", "")).lower()
        pred_cat = str(predicted.get("category", "")).lower()
        gt_prio = str(ground_truth.get("priority", "")).lower()
        pred_prio = str(predicted.get("priority", "")).lower()
        gt_lang = str(ground_truth.get("language", "")).lower()
        pred_lang = str(predicted.get("language", "")).lower()

        # 1. Invalid JSON or parse failure
        if confidence == 0.0 or pred_cat == "unknown":
            return "invalid_json"

        # 2. Similar Category Confusion
        if frozenset([gt_cat, pred_cat]) in self.SIMILAR_CATEGORY_PAIRS:
            return "similar_categories"

        # 3. Language Ambiguity / Hinglish Ambiguity
        if gt_lang != pred_lang:
            if gt_lang == "hinglish" or pred_lang == "hinglish":
                return "hinglish_ambiguity"
            return "language_error"

        # 4. Input Length extremes
        words = text.split()
        if len(words) <= 3:
            return "short_input"
        elif len(words) >= 40:
            return "long_input"

        # 5. Priority Mismatch (when category was right)
        if gt_cat == pred_cat and gt_prio != pred_prio:
            return "priority_error"

        # 6. Typo or informal markers in text
        if any(marker in text.lower() for marker in ["plz", "nhi", "paymnt", "refnd", "bro", "yaar"]):
            return "typos"

        # 7. Rare class (e.g. account_closure, fraud_report)
        if gt_cat in ["account_closure", "fraud_report"]:
            return "rare_class"

        return "other"

    def analyze_predictions(
        self,
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        texts: List[str],
        confidences: List[float],
    ) -> Dict[str, Any]:
        """Collects, classifies, and aggregates errors across test predictions."""
        logger.info(f"Analyzing errors across {len(ground_truths)} evaluated predictions...")

        failures: List[Dict[str, Any]] = []
        category_errors: Counter = Counter()
        language_errors: Counter = Counter()
        error_type_counter: Counter = Counter()

        for gt, pred, text, conf in zip(ground_truths, predictions, texts, confidences):
            c_match = (str(gt.get("category", "")).lower() == str(pred.get("category", "")).lower())
            p_match = (str(gt.get("priority", "")).lower() == str(pred.get("priority", "")).lower())
            l_match = (str(gt.get("language", "")).lower() == str(pred.get("language", "")).lower())
            h_match = (bool(gt.get("requires_human")) == bool(pred.get("requires_human")))

            # Record failure if any field mismatched
            if not (c_match and p_match and l_match and h_match):
                err_type = self.classify_error_type(gt, pred, text, conf)
                error_type_counter[err_type] += 1
                category_errors[gt["category"]] += 1
                language_errors[gt["language"]] += 1

                failures.append({
                    "id": gt.get("id", "unknown"),
                    "input_text": text,
                    "expected": {
                        "category": gt["category"],
                        "priority": gt["priority"],
                        "language": gt["language"],
                        "requires_human": gt.get("requires_human", False),
                    },
                    "predicted": {
                        "category": pred["category"],
                        "priority": pred["priority"],
                        "language": pred["language"],
                        "requires_human": pred.get("requires_human", False),
                    },
                    "confidence": conf,
                    "error_type": err_type,
                })

        total = len(ground_truths)
        fail_count = len(failures)
        accuracy = round((total - fail_count) / total, 4) if total else 1.0

        hardest_categories = [
            {"category": cat, "error_count": cnt}
            for cat, cnt in category_errors.most_common(5)
        ]
        hardest_languages = [
            {"language": lang, "error_count": cnt}
            for lang, cnt in language_errors.most_common(3)
        ]

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_evaluated": total,
            "total_failures": fail_count,
            "overall_accuracy": accuracy,
            "error_distribution": dict(error_type_counter),
            "hardest_categories": hardest_categories,
            "hardest_languages": hardest_languages,
            "representative_failures": failures[:20],  # Sample 20 representative failures
        }

        with open(self.output_dir / "error_analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Error Analysis complete: {fail_count} failures identified ({accuracy * 100:.1f}% accuracy).")
        return report
