"""Deterministic stratified splitting for NEXUS-SLM."""

import random
from typing import Dict, List, Tuple
from collections import defaultdict
from src.data.schemas import TicketRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """Performs deterministic stratified splitting of support tickets."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self, records: List[TicketRecord]
    ) -> Tuple[List[TicketRecord], List[TicketRecord], List[TicketRecord]]:
        """
        Splits records into (train, val, test) using stratified bucketing
        by (category, language) to maintain class and language distributions.
        """
        rng = random.Random(self.seed)

        # Group records by stratification key: (category, language)
        strata: Dict[Tuple[str, str], List[TicketRecord]] = defaultdict(list)
        for record in records:
            strata[(record.category, record.language)].append(record)

        train_records: List[TicketRecord] = []
        val_records: List[TicketRecord] = []
        test_records: List[TicketRecord] = []

        for (cat, lang), items in strata.items():
            # Deterministic shuffle within stratum
            items_copy = list(items)
            rng.shuffle(items_copy)

            n_total = len(items_copy)
            if n_total < 3:
                # Handle tiny strata gracefully
                train_records.extend(items_copy)
                continue

            n_test = max(1, int(round(n_total * self.test_ratio)))
            n_val = max(1, int(round(n_total * self.val_ratio)))
            n_train = n_total - n_test - n_val

            if n_train <= 0:
                n_train = max(1, n_total - 2)
                n_val = 1
                n_test = n_total - n_train - n_val

            test_records.extend(items_copy[:n_test])
            val_records.extend(items_copy[n_test : n_test + n_val])
            train_records.extend(items_copy[n_test + n_val :])

        # Final deterministic shuffle of resulting splits
        rng.shuffle(train_records)
        rng.shuffle(val_records)
        rng.shuffle(test_records)

        logger.info(
            f"Stratified split completed with seed={self.seed}: "
            f"Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}"
        )
        return train_records, val_records, test_records
