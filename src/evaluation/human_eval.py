"""Human-in-the-loop evaluation and audit workflow for NEXUS-SLM."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HumanEvaluator:
    """Manages human audit sampling, recording ratings, and computing human-vs-model concordance."""

    def __init__(self, review_storage_path: str = "experiments/human_eval/reviews.json"):
        self.storage_path = Path(review_storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        if not self.storage_path.exists():
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def load_reviews(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def record_review(
        self,
        ticket_id: str,
        text: str,
        predicted: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]],
        human_judgment: str,  # "Correct", "Incorrect", "Ambiguous"
        reviewer_id: str = "reviewer_01",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Records a human assessment record."""
        valid_judgments = {"Correct", "Incorrect", "Ambiguous"}
        if human_judgment not in valid_judgments:
            raise ValueError(f"Invalid judgment '{human_judgment}'. Must be one of {valid_judgments}")

        reviews = self.load_reviews()
        record = {
            "review_id": f"rev_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{ticket_id}",
            "ticket_id": ticket_id,
            "text": text,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "human_judgment": human_judgment,
            "reviewer_id": reviewer_id,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        }

        reviews.append(record)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=2)

        logger.info(f"Recorded human review for ticket {ticket_id}: {human_judgment}")
        return record

    def compute_human_agreement(self) -> Dict[str, Any]:
        """Calculates concordance between automated model evaluation and human judgment."""
        reviews = self.load_reviews()
        if not reviews:
            return {
                "total_reviews": 0,
                "human_accuracy_rate": 0.0,
                "concordance_rate": 0.0,
                "ambiguity_rate": 0.0,
                "status": "NO_REVIEWS_RECORDED",
            }

        correct_count = sum(1 for r in reviews if r["human_judgment"] == "Correct")
        incorrect_count = sum(1 for r in reviews if r["human_judgment"] == "Incorrect")
        ambiguous_count = sum(1 for r in reviews if r["human_judgment"] == "Ambiguous")
        total = len(reviews)

        # Agreement between automated metric (e.g. predicted == ground_truth) and human mark
        agreement_count = 0
        for r in reviews:
            gt = r.get("ground_truth")
            pred = r.get("predicted")
            if gt and pred:
                auto_match = (str(gt.get("category")).lower() == str(pred.get("category")).lower())
                human_ok = (r["human_judgment"] == "Correct")
                if auto_match == human_ok:
                    agreement_count += 1

        return {
            "total_reviews": total,
            "human_correct_count": correct_count,
            "human_incorrect_count": incorrect_count,
            "ambiguous_count": ambiguous_count,
            "human_accuracy_rate": round(correct_count / total, 4),
            "ambiguity_rate": round(ambiguous_count / total, 4),
            "human_model_concordance_rate": round(agreement_count / total, 4) if total else 0.0,
        }
