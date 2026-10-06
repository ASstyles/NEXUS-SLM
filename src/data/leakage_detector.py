"""Data leakage detector across train, validation, and test splits."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from src.data.schemas import LeakageReport, TicketRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LeakageDetector:
    """Audits splits for data leakage, template overlap, and high cross-split similarity."""

    def __init__(self, similarity_threshold: float = 0.80, ngram_size: int = 3):
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        # Remove numbers/order IDs to detect parameterized template leakage
        text = re.sub(r"\b\d+\b", "<NUM>", text)
        text = re.sub(r"[^\w\s<>]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _get_ngrams(self, text: str) -> Set[str]:
        norm = self._normalize(text)
        if len(norm) < self.ngram_size:
            return {norm}
        return {
            norm[i : i + self.ngram_size]
            for i in range(len(norm) - self.ngram_size + 1)
        }

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a.intersection(b)) / len(a.union(b))

    def detect_leakage(
        self,
        train_records: List[TicketRecord],
        val_records: List[TicketRecord],
        test_records: List[TicketRecord],
        dataset_version: str = "dataset_v1",
    ) -> LeakageReport:
        """
        Inspects cross-split overlap between Train and Test/Val.
        Guarantees that test records have no identical or near-identical counterparts in train.
        """
        logger.info(
            f"Analyzing leakage across Train ({len(train_records)}), "
            f"Val ({len(val_records)}), Test ({len(test_records)})..."
        )

        train_norm_map: Dict[str, str] = {self._normalize(r.text): r.id for r in train_records}
        train_ngrams = [(r.id, self._get_ngrams(r.text), r.text) for r in train_records]

        exact_dups = 0
        near_dups = 0
        max_sim = 0.0
        template_leaks: List[Dict[str, Any]] = []

        # Check Test vs Train (Strict Locked Test Set Policy)
        for test_r in test_records:
            test_norm = self._normalize(test_r.text)

            # Check 1: Exact normalized / template match
            if test_norm in train_norm_map:
                exact_dups += 1
                template_leaks.append({
                    "test_id": test_r.id,
                    "train_id": train_norm_map[test_norm],
                    "test_text": test_r.text,
                    "type": "exact_or_template_match",
                    "similarity": 1.0,
                })
                max_sim = 1.0
                continue

            # Check 2: High n-gram similarity match
            test_ngrams = self._get_ngrams(test_r.text)
            for tr_id, tr_ng, tr_text in train_ngrams:
                sim = self._jaccard(test_ngrams, tr_ng)
                if sim > max_sim:
                    max_sim = sim

                if sim >= self.similarity_threshold:
                    near_dups += 1
                    template_leaks.append({
                        "test_id": test_r.id,
                        "train_id": tr_id,
                        "test_text": test_r.text,
                        "train_text": tr_text,
                        "type": "near_duplicate_similarity",
                        "similarity": round(sim, 3),
                    })
                    break

        has_leakage = (exact_dups > 0 or near_dups > 0)
        status = "PASSED_ZERO_LEAKAGE" if not has_leakage else "LEAKAGE_DETECTED"

        report = LeakageReport(
            dataset_version=dataset_version,
            train_count=len(train_records),
            val_count=len(val_records),
            test_count=len(test_records),
            exact_duplicates_across_splits=exact_dups,
            near_duplicates_flagged=near_dups,
            max_similarity_score=round(max_sim, 4),
            template_leakages=template_leaks[:50],  # sample up to 50
            leakage_detected=has_leakage,
            status=status,
        )

        logger.info(
            f"Leakage Audit Completed: Status={status}, Exact Dups={exact_dups}, "
            f"Near Dups={near_dups}, Max Similarity={max_sim:.3f}"
        )
        return report

    def sanitize_train_against_locked_test(
        self,
        train_records: List[TicketRecord],
        test_records: List[TicketRecord],
    ) -> Tuple[List[TicketRecord], int]:
        """
        Enforces strict zero-leakage: purges any training record that encroaches
        on the locked test set. The locked test set is NEVER altered.
        """
        test_norm_map = {self._normalize(r.text): r.id for r in test_records}
        test_ngrams = [(r.id, self._get_ngrams(r.text)) for r in test_records]

        sanitized_train: List[TicketRecord] = []
        purged_count = 0

        for tr in train_records:
            tr_norm = self._normalize(tr.text)
            if tr_norm in test_norm_map:
                purged_count += 1
                continue

            tr_ng = self._get_ngrams(tr.text)
            leak = False
            for _, test_ng in test_ngrams:
                if self._jaccard(tr_ng, test_ng) >= self.similarity_threshold:
                    leak = True
                    break
            if leak:
                purged_count += 1
            else:
                sanitized_train.append(tr)

        logger.info(f"Sanitization complete: Purged {purged_count} leaky training instances to protect locked test set.")
        return sanitized_train, purged_count
