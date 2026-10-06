"""Complete end-to-end dataset pipeline for NEXUS-SLM."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.data.curated_test_set import get_curated_locked_test_records
from src.data.deduplication import Deduplicator
from src.data.leakage_detector import LeakageDetector
from src.data.schemas import DatasetQualityReport, LeakageReport, TicketRecord
from src.data.splitter import DatasetSplitter
from src.data.synthetic_generator import SyntheticTicketGenerator
from src.data.validator import DataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """Orchestrates data ingestion, validation, deduplication, splitting, and leakage audit."""

    def __init__(
        self,
        version: str = "dataset_v1",
        output_dir: str = "data",
        seed: int = 42,
    ):
        self.version = version
        self.base_dir = Path(output_dir)
        self.seed = seed

        self.validator = DataValidator()
        self.deduplicator = Deduplicator(similarity_threshold=0.85)
        self.splitter = DatasetSplitter(train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=seed)
        self.leakage_detector = LeakageDetector(similarity_threshold=0.80)

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for sub in ["raw", "interim", "processed", "train", "validation", "test", "metadata"]:
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)

    def run_pipeline(
        self,
        num_synthetic_samples: int = 1200,
        include_curated_locked_test: bool = True,
    ) -> Dict[str, Any]:
        """Runs the complete data preparation lifecycle."""
        logger.info(f"Starting NEXUS-SLM Data Pipeline for {self.version}...")

        # 1. Generate / Ingest Synthetic Data
        generator = SyntheticTicketGenerator(seed=self.seed)
        raw_synthetic = generator.generate_dataset(num_samples=num_synthetic_samples, source="synthetic")

        # Save raw interim
        raw_path = self.base_dir / "raw" / f"raw_synthetic_{self.version}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in raw_synthetic:
                f.write(r.model_dump_json() + "\n")

        # 2. Validation & Quality Filtering
        valid_records: List[TicketRecord] = []
        invalid_count = 0
        for rec in raw_synthetic:
            validated_rec, res = self.validator.validate_record(rec.model_dump())
            if res.valid and validated_rec:
                valid_records.append(validated_rec)
            else:
                invalid_count += 1

        logger.info(f"Validation phase: {len(valid_records)} valid, {invalid_count} rejected.")

        # 3. Deduplication
        unique_records, dup_log = self.deduplicator.deduplicate(valid_records)
        dup_count = len(dup_log)

        # 4. Locked Test Set & Splitting Strategy
        # To guarantee the test set is NOT from the same synthetic template generator,
        # we load our independent curated locked test set!
        if include_curated_locked_test:
            locked_test = get_curated_locked_test_records()
            # Split the synthetic data into train and validation
            train_records, val_records, _ = self.splitter.split(unique_records)
            test_records = locked_test
        else:
            train_records, val_records, test_records = self.splitter.split(unique_records)

        # 5. Data Leakage Detection & Protection of Locked Test Set
        leakage_report: LeakageReport = self.leakage_detector.detect_leakage(
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            dataset_version=self.version,
        )

        # Enforce zero-leakage by sanitizing train if any overlap was flagged
        if leakage_report.leakage_detected:
            train_records, purged = self.leakage_detector.sanitize_train_against_locked_test(
                train_records, test_records
            )
            # Re-verify
            leakage_report = self.leakage_detector.detect_leakage(
                train_records, val_records, test_records, self.version
            )

        # 6. Save Splits
        splits = {
            "train": (self.base_dir / "train" / "train.jsonl", train_records),
            "validation": (self.base_dir / "validation" / "validation.jsonl", val_records),
            "test": (self.base_dir / "test" / "test.jsonl", test_records),
        }

        for split_name, (filepath, records) in splits.items():
            with open(filepath, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(r.model_dump_json() + "\n")
            logger.info(f"Saved {split_name} split to {filepath} ({len(records)} records)")

        # 7. Compute Statistical Quality Report
        all_records = train_records + val_records + test_records
        cat_dist: Dict[str, int] = {}
        lang_dist: Dict[str, int] = {}
        prio_dist: Dict[str, int] = {}
        lengths = [len(r.text) for r in all_records]
        synth_count = sum(1 for r in all_records if r.source == "synthetic")
        human_count = sum(1 for r in all_records if r.source in ["human_reviewed", "curated"])

        for r in all_records:
            cat_dist[r.category] = cat_dist.get(r.category, 0) + 1
            lang_dist[r.language] = lang_dist.get(r.language, 0) + 1
            prio_dist[r.priority] = prio_dist.get(r.priority, 0) + 1

        total = len(all_records)
        quality_report = DatasetQualityReport(
            dataset_version=self.version,
            total_examples=total,
            examples_per_category=cat_dist,
            examples_per_language=lang_dist,
            examples_per_priority=prio_dist,
            synthetic_count=synth_count,
            synthetic_percentage=round((synth_count / total) * 100, 2) if total else 0.0,
            human_reviewed_count=human_count,
            human_reviewed_percentage=round((human_count / total) * 100, 2) if total else 0.0,
            curated_count=sum(1 for r in all_records if r.source == "curated"),
            duplicate_count=dup_count,
            invalid_record_count=invalid_count,
            mean_char_length=round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
            min_char_length=min(lengths) if lengths else 0,
            max_char_length=max(lengths) if lengths else 0,
        )

        # Save metadata reports
        qual_path = self.base_dir / "metadata" / "dataset_quality_report.json"
        with open(qual_path, "w", encoding="utf-8") as f:
            f.write(quality_report.model_dump_json(indent=2))

        leak_path = self.base_dir / "metadata" / "leakage_report.json"
        with open(leak_path, "w", encoding="utf-8") as f:
            f.write(leakage_report.model_dump_json(indent=2))

        logger.info(f"Dataset Pipeline Completed Successfully for {self.version}!")
        return {
            "quality_report": quality_report.model_dump(),
            "leakage_report": leakage_report.model_dump(),
            "train_size": len(train_records),
            "val_size": len(val_records),
            "test_size": len(test_records),
        }


def main():
    pipeline = DataPipeline(version="dataset_v1")
    summary = pipeline.run_pipeline(num_synthetic_samples=1200)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
