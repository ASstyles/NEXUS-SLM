"""Double-blind prospective evaluation protocol and concordance validator for NEXUS-SLM pilot."""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return max(0.0, round(lower_bound, 4)), min(1.0, round(upper_bound, 4))


class PilotEvaluationProtocol:
    """Manages blinded human ground-truth ingestion, freezing, and concordance scoring."""

    def __init__(
        self,
        blinded_labels_path: str = "data/pilot/blinded_human_labels.jsonl",
        frozen_labels_path: str = "data/pilot/frozen_pilot_ground_truth.jsonl",
    ):
        self.blinded_labels_path = Path(blinded_labels_path)
        self.frozen_labels_path = Path(frozen_labels_path)
        self.blinded_labels_path.parent.mkdir(parents=True, exist_ok=True)

    def record_blinded_human_label(
        self,
        ticket_id: str,
        sanitized_text: str,
        human_category: str,
        human_priority: str,
        human_requires_human: bool,
        reviewer_id: str,
        risk_type: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Records human judgment created without seeing model predictions (Double-Blind).
        Validates taxonomy membership before saving.
        """
        if human_category not in CATEGORIES:
            raise ValueError(f"Invalid category '{human_category}'. Must be in {CATEGORIES}")
        if human_priority not in PRIORITIES:
            raise ValueError(f"Invalid priority '{human_priority}'. Must be in {PRIORITIES}")

        record = {
            "ticket_id": ticket_id,
            "sanitized_text": sanitized_text,
            "human_category": human_category,
            "human_priority": human_priority,
            "human_requires_human": human_requires_human,
            "risk_type": risk_type or ("high_risk" if human_requires_human else "routine_safe"),
            "reviewer_id": reviewer_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "notes": notes,
        }

        with open(self.blinded_labels_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Recorded blinded human judgment for {ticket_id} by {reviewer_id}")
        return record

    def freeze_pilot_ground_truth(self) -> Dict[str, Any]:
        """
        Freezes accumulated blinded human labels into immutable ground-truth file.
        Computes SHA256 manifest hash to guarantee labels cannot be retroactively modified.
        """
        if not self.blinded_labels_path.exists():
            raise FileNotFoundError(f"Blinded labels file not found at {self.blinded_labels_path}")

        records = []
        with open(self.blinded_labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Write to frozen file
        with open(self.frozen_labels_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Compute SHA256
        h = hashlib.sha256()
        with open(self.frozen_labels_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        manifest_hash = h.hexdigest()

        manifest = {
            "freeze_timestamp": datetime.utcnow().isoformat() + "Z",
            "frozen_samples_count": len(records),
            "frozen_file_path": str(self.frozen_labels_path),
            "sha256_hash": manifest_hash,
            "status": "FROZEN_FOR_CONCORDANCE_EVALUATION",
        }

        manifest_file = self.frozen_labels_path.parent / "ground_truth_freeze_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Frozen {len(records)} human labels at {self.frozen_labels_path} (Hash: {manifest_hash[:16]}...)")
        return manifest

    @classmethod
    def evaluate_shadow_concordance(
        cls,
        ground_truth_records: List[Dict[str, Any]],
        shadow_predictions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates shadow predictions against frozen blinded human labels.
        Separates classification quality and escalation safety.
        """
        y_true_cat = []
        y_pred_cat = []
        y_true_esc = []
        y_pred_esc = []

        fn_safety_escapes = []
        fp_over_escalations = []
        matched_records = 0

        for gt in ground_truth_records:
            tid = gt["ticket_id"]
            if tid not in shadow_predictions:
                continue

            pred = shadow_predictions[tid]
            matched_records += 1

            yt_c = gt["human_category"]
            yp_c = pred.get("predicted_category", "unknown")
            y_true_cat.append(yt_c)
            y_pred_cat.append(yp_c)

            yt_e = gt["human_requires_human"]
            yp_e = pred.get("requires_human", False)
            y_true_esc.append(yt_e)
            y_pred_esc.append(yp_e)

            if yt_e and not yp_e:
                fn_safety_escapes.append({
                    "ticket_id": tid,
                    "text": gt.get("sanitized_text", "")[:80],
                    "risk_type": gt.get("risk_type", "unknown"),
                    "human_category": yt_c,
                    "model_category": yp_c,
                    "model_confidence": pred.get("confidence", 0.0),
                })
            elif not yt_e and yp_e:
                fp_over_escalations.append({
                    "ticket_id": tid,
                    "text": gt.get("sanitized_text", "")[:80],
                    "escalation_reason": pred.get("escalation_reason", "unknown"),
                })

        if matched_records == 0:
            return {"status": "NO_MATCHED_PREDICTIONS"}

        # 1. Classification Metrics
        acc = accuracy_score(y_true_cat, y_pred_cat)
        correct_cat_count = sum(yt == yp for yt, yp in zip(y_true_cat, y_pred_cat))
        acc_ci_low, acc_ci_high = compute_wilson_ci(correct_cat_count, matched_records)

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_cat, y_pred_cat, labels=CATEGORIES, average="macro", zero_division=0
        )
        weight_p, weight_r, weight_f1, _ = precision_recall_fscore_support(
            y_true_cat, y_pred_cat, labels=CATEGORIES, average="weighted", zero_division=0
        )

        # 2. Escalation Safety Metrics
        tp_esc = sum(yt and yp for yt, yp in zip(y_true_esc, y_pred_esc))
        fn_esc = len(fn_safety_escapes)
        fp_esc = len(fp_over_escalations)
        tn_esc = sum(not yt and not yp for yt, yp in zip(y_true_esc, y_pred_esc))

        esc_recall = tp_esc / (tp_esc + fn_esc) if (tp_esc + fn_esc) > 0 else 0.0
        esc_precision = tp_esc / (tp_esc + fp_esc) if (tp_esc + fp_esc) > 0 else 0.0
        esc_ci_low, esc_ci_high = compute_wilson_ci(tp_esc, tp_esc + fn_esc)

        return {
            "matched_eval_samples": matched_records,
            "classification_concordance": {
                "category_agreement_rate": round(acc, 4),
                "agreement_count": correct_cat_count,
                "confidence_interval_95": [acc_ci_low, acc_ci_high],
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weight_f1, 4),
                "macro_precision": round(macro_p, 4),
                "macro_recall": round(macro_r, 4),
            },
            "escalation_safety_concordance": {
                "escalation_recall": round(esc_recall, 4),
                "confidence_interval_95": [esc_ci_low, esc_ci_high],
                "escalation_precision": round(esc_precision, 4),
                "true_positives": tp_esc,
                "false_negatives": fn_esc,
                "false_positives": fp_esc,
                "true_negatives": tn_esc,
                "sev1_safety_escapes_count": fn_esc,
                "false_negative_escapes": fn_safety_escapes,
                "sample_false_positives": fp_over_escalations[:5],
            },
        }
