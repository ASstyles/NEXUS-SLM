"""Evaluator module for executing rigorous benchmark evaluations."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import psutil
import torch
from src.evaluation.metrics import EvaluationMetrics
from src.inference.parser import OutputParser
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NexusEvaluator:
    """Evaluates NEXUS-SLM or baseline models against the locked test benchmark."""

    def __init__(
        self,
        test_dataset_path: str = "data/test/test.jsonl",
        output_dir: str = "experiments/evaluations",
    ):
        self.test_dataset_path = Path(test_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = OutputParser()

    def load_test_data(self) -> List[Dict[str, Any]]:
        """Loads locked test dataset records."""
        if not self.test_dataset_path.exists():
            raise FileNotFoundError(f"Locked test dataset not found at: {self.test_dataset_path}")
        records = []
        with open(self.test_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def evaluate(
        self,
        predictor_fn: Callable[[str], Any],
        model_name: str = "nexus-v1.0",
        is_baseline: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes benchmark evaluation on the locked test set.
        Args:
            predictor_fn: Callable taking text string and returning TriagePrediction
            model_name: Name/version of model being evaluated
            is_baseline: If True, saves to experiments/baseline/baseline_results.json
        """
        records = self.load_test_data()
        logger.info(f"Evaluating {model_name} on {len(records)} locked test examples...")

        ground_truths = []
        predictions = []
        latencies_ms = []
        json_valid_flags = []
        raw_outputs = []

        # Memory snapshot before
        ram_before = psutil.virtual_memory().used / (1024**2)
        vram_before = (torch.cuda.memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

        for idx, rec in enumerate(records, start=1):
            text = rec["text"]
            t0 = time.perf_counter()
            pred = predictor_fn(text)
            dt = (time.perf_counter() - t0) * 1000

            # Use internal latency if measured by engine, else exterior timer
            lat = getattr(pred, "latency_ms", None) or dt
            latencies_ms.append(lat)

            # Check JSON validity from raw generation using strict parser
            raw_gen = getattr(pred, "raw_output", None)
            if raw_gen:
                is_v, _, _ = self.parser.parse_and_validate(raw_gen)
                json_valid_flags.append(is_v)
            else:
                confidence = getattr(pred, "confidence", 0.0)
                json_valid_flags.append(confidence > 0.0)

            p_dict = {
                "category": getattr(pred, "category", "unknown"),
                "priority": getattr(pred, "priority", "unknown"),
                "language": getattr(pred, "language", "unknown"),
                "requires_human": getattr(pred, "requires_human", False),
            }
            predictions.append(p_dict)
            ground_truths.append(rec)
            raw_outputs.append(getattr(pred, "raw_output", ""))

        ram_after = psutil.virtual_memory().used / (1024**2)
        vram_after = (torch.cuda.memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

        # Compute Metrics
        # 1. Classification Metrics (Category)
        y_true_cat = [r["category"] for r in ground_truths]
        y_pred_cat = [p["category"] for p in predictions]
        cat_metrics = EvaluationMetrics.calculate_classification_metrics(
            y_true=y_true_cat,
            y_pred=y_pred_cat,
            labels=CATEGORIES,
        )

        # 2. Priority & Language & Human Classification Metrics
        prio_metrics = EvaluationMetrics.calculate_classification_metrics(
            y_true=[r["priority"] for r in ground_truths],
            y_pred=[p["priority"] for p in predictions],
            labels=PRIORITIES,
        )
        lang_metrics = EvaluationMetrics.calculate_classification_metrics(
            y_true=[r["language"] for r in ground_truths],
            y_pred=[p["language"] for p in predictions],
            labels=LANGUAGES,
        )

        # 3. Structured Output Metrics
        struct_metrics = EvaluationMetrics.calculate_structured_metrics(
            ground_truths=ground_truths,
            predictions=predictions,
            json_valid_flags=json_valid_flags,
        )

        # 4. Performance Metrics
        perf_metrics = EvaluationMetrics.calculate_performance_metrics(
            latencies_ms=latencies_ms,
            ram_mb=round(ram_after, 2),
            vram_mb=round(vram_after, 2),
        )

        eval_result = {
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "test_sample_count": len(records),
            "macro_f1": cat_metrics["macro_f1"],
            "weighted_f1": cat_metrics["weighted_f1"],
            "category_accuracy": struct_metrics["category_accuracy"],
            "priority_accuracy": struct_metrics["priority_accuracy"],
            "language_accuracy": struct_metrics["language_accuracy"],
            "human_escalation_accuracy": struct_metrics["human_escalation_accuracy"],
            "json_validity_rate": struct_metrics["json_validity_rate"],
            "field_level_accuracy": struct_metrics["field_level_accuracy"],
            "exact_match_accuracy": struct_metrics["exact_match_accuracy"],
            "latency_p50_ms": perf_metrics["latency_p50_ms"],
            "latency_p95_ms": perf_metrics["latency_p95_ms"],
            "latency_p99_ms": perf_metrics["latency_p99_ms"],
            "throughput_req_per_sec": perf_metrics["throughput_req_per_sec"],
            "ram_mb": perf_metrics["ram_mb"],
            "vram_mb": perf_metrics["vram_mb"],
            "detailed_category_metrics": cat_metrics,
            "priority_metrics": prio_metrics,
            "language_metrics": lang_metrics,
        }

        # Save Report
        tag = "baseline" if is_baseline else f"eval_{model_name.replace('/', '_')}"
        out_file = self.output_dir / f"{tag}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2)

        if is_baseline:
            base_dir = Path("experiments/baseline")
            base_dir.mkdir(parents=True, exist_ok=True)
            with open(base_dir / "baseline_results.json", "w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=2)

        logger.info(
            f"Evaluation Complete for {model_name}: "
            f"Macro-F1={eval_result['macro_f1']}, JSON-Valid={eval_result['json_validity_rate']}, "
            f"P50 Latency={eval_result['latency_p50_ms']}ms"
        )
        return eval_result
