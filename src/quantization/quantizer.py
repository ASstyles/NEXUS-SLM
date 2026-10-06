"""Quantization optimizer and precision trade-off benchmarking for NEXUS-SLM."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import psutil
import torch
from src.evaluation.metrics import EvaluationMetrics
from src.utils.hardware import get_hardware_info
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelQuantizer:
    """
    Manages and benchmarks precision configurations:
      - FP16 (Half precision)
      - INT8 (8-bit quantization)
      - INT4 (4-bit NF4 with double quantization)
    """

    SUPPORTED_PRECISIONS = ["fp16", "int8", "int4"]

    def __init__(self, output_dir: str = "experiments/artifacts/quantization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hw_info = get_hardware_info()

    def get_quantization_configs(self) -> Dict[str, Any]:
        """Returns BitsAndBytesConfig dictionary for each precision mode."""
        configs = {
            "fp16": {"torch_dtype": "float16", "load_in_4bit": False, "load_in_8bit": False},
            "int8": {"torch_dtype": "float16", "load_in_4bit": False, "load_in_8bit": True},
            "int4": {
                "torch_dtype": "float16",
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        }
        return configs

    def evaluate_precision_tradeoffs(
        self,
        precision_eval_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Takes evaluation metrics from each precision mode and produces
        the relative trade-off matrix (accuracy retained, memory saved, speedup).
        """
        base_fp16 = precision_eval_results.get("fp16", {})
        fp16_f1 = base_fp16.get("macro_f1", 1.0)
        fp16_lat = base_fp16.get("latency_p50_ms", 1.0)
        fp16_vram = base_fp16.get("vram_mb", 1.0)

        tradeoffs = {}
        for prec, res in precision_eval_results.items():
            f1 = res.get("macro_f1", 0.0)
            lat = res.get("latency_p50_ms", 0.0)
            vram = res.get("vram_mb", 0.0)

            retention = round((f1 / fp16_f1) * 100, 2) if fp16_f1 > 0 else 100.0
            vram_reduction = round(((fp16_vram - vram) / fp16_vram) * 100, 2) if fp16_vram > 0 else 0.0
            latency_delta = round(((lat - fp16_lat) / fp16_lat) * 100, 2) if fp16_lat > 0 else 0.0

            tradeoffs[prec] = {
                "macro_f1": f1,
                "f1_retention_pct": retention,
                "json_validity_rate": res.get("json_validity_rate", 0.0),
                "latency_p50_ms": lat,
                "latency_p95_ms": res.get("latency_p95_ms", 0.0),
                "throughput_req_per_sec": res.get("throughput_req_per_sec", 0.0),
                "vram_mb": vram,
                "vram_reduction_pct": vram_reduction,
                "latency_delta_pct": latency_delta,
            }

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "hardware": self.hw_info,
            "tradeoffs": tradeoffs,
        }

        with open(self.output_dir / "quantization_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
