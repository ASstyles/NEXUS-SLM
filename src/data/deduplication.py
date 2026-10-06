"""Deduplication and near-duplicate detection for NEXUS-SLM."""

import re
from typing import Dict, List, Set, Tuple
from src.data.schemas import TicketRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """Detects and removes exact and near-duplicate support ticket records."""

    def __init__(self, similarity_threshold: float = 0.85, ngram_size: int = 3):
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size

    @staticmethod
    def _canonicalize(text: str) -> str:
        """Strips punctuation, lowercases, and compresses whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _get_char_ngrams(self, text: str) -> Set[str]:
        """Extracts character n-grams from canonicalized text."""
        canonical = self._canonicalize(text)
        if len(canonical) < self.ngram_size:
            return {canonical}
        return {
            canonical[i : i + self.ngram_size]
            for i in range(len(canonical) - self.ngram_size + 1)
        }

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Calculates Jaccard similarity index between two sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def deduplicate(
        self, records: List[TicketRecord]
    ) -> Tuple[List[TicketRecord], List[Dict[str, str]]]:
        """
        Removes exact and near-duplicates while preserving high-quality instances.
        Returns:
            (unique_records, duplicate_log)
        """
        unique_records: List[TicketRecord] = []
        duplicate_log: List[Dict[str, str]] = []
        seen_canonical: Dict[str, str] = {}
        seen_ngrams: List[Tuple[str, Set[str], str]] = []  # (id, ngrams, text)

        for record in records:
            canon = self._canonicalize(record.text)

            # 1. Exact canonical duplicate
            if canon in seen_canonical:
                duplicate_log.append({
                    "duplicate_id": record.id,
                    "matched_id": seen_canonical[canon],
                    "duplicate_text": record.text,
                    "reason": "exact_canonical_match",
                    "similarity": "1.00",
                })
                continue

            # 2. Near-duplicate similarity check
            ngrams = self._get_char_ngrams(record.text)
            is_near_dup = False

            for prev_id, prev_ngrams, prev_text in seen_ngrams:
                sim = self._jaccard_similarity(ngrams, prev_ngrams)
                if sim >= self.similarity_threshold:
                    duplicate_log.append({
                        "duplicate_id": record.id,
                        "matched_id": prev_id,
                        "duplicate_text": record.text,
                        "matched_text": prev_text,
                        "reason": f"near_duplicate_jaccard_{sim:.2f}",
                        "similarity": f"{sim:.2f}",
                    })
                    is_near_dup = True
                    break

            if not is_near_dup:
                seen_canonical[canon] = record.id
                seen_ngrams.append((record.id, ngrams, record.text))
                unique_records.append(record)

        logger.info(
            f"Deduplication finished. Input: {len(records)}, Retained: {len(unique_records)}, "
            f"Removed: {len(duplicate_log)}"
        )
        return unique_records, duplicate_log
