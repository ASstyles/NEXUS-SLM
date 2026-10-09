"""Quantization benchmark harness for NEXUS-SLM.
Evaluates supported inference formats (FP32 vs INT8 dynamic quantization) on locked test set.
Records hardware constraints for FP16/BF16 and INT4.
"""

import json
import time
import psutil
import torch
from pathlib import Path
import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(WORKSPACE_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.metrics import EvaluationMetrics
from src.inference.parser import OutputParser
from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

ARTIFACTS_DIR = WORKSPACE_ROOT / "experiments" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILE = WORKSPACE_ROOT / "data" / "test" / "test.jsonl"
ADAPTER_PATH = WORKSPACE_ROOT / "models" / "nexus_qlora_v1"
BASE_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"


def load_candidate_model(precision: str = "fp32"):
    """Loads NEXUS-SLM candidate model with specified precision format."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if precision == "fp32":
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        model.eval()
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        return model, tokenizer, model_size_mb

    elif precision == "int8":
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        model.eval()
        # Merge LoRA before quantizing for CPU execution
        merged = model.merge_and_unload()
        # Dynamic INT8 quantization on linear layers
        quantized_model = torch.ao.quantization.quantize_dynamic(
            merged,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        quantized_model.eval()
        # Estimate quantized size: int8 weights use 1 byte per element
        model_size_mb = sum(p.numel() * (1 if p.dtype == torch.qint8 else p.element_size()) for p in quantized_model.parameters()) / (1024**2)
        return quantized_model, tokenizer, model_size_mb

    else:
        raise ValueError(f"Precision '{precision}' not supported on host.")


def evaluate_precision(precision: str, test_records: list):
    """Evaluates a precision variant on the locked test set."""
    print(f"\nEvaluating precision variant: {precision.upper()}...")
    t0_load = time.perf_counter()
    model, tokenizer, size_mb = load_candidate_model(precision)
    load_time_sec = round(time.perf_counter() - t0_load, 2)

    parser = OutputParser()
    predictions = []
    ground_truths = []
    json_valid_flags = []
    latencies_ms = []

    ram_before = psutil.virtual_memory().used / (1024**2)
    peak_ram = ram_before

    t0_bench = time.perf_counter()
    for rec in test_records:
        text = rec["text"]
        prompt = PromptFormatter.format_prompt(text, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=45,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt)

        cur_ram = psutil.virtual_memory().used / (1024**2)
        if cur_ram > peak_ram:
            peak_ram = cur_ram

        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        is_valid, parsed, _ = parser.parse_and_validate(raw)
        json_valid_flags.append(is_valid)

        if is_valid and parsed:
            pred_item = {
                "category": parsed["category"],
                "priority": parsed["priority"],
                "language": parsed["language"],
                "requires_human": parsed["requires_human"],
            }
        else:
            pred_item = {
                "category": "unknown",
                "priority": "unknown",
                "language": "unknown",
                "requires_human": False,
            }
        predictions.append(pred_item)
        ground_truths.append(rec)

    total_time_sec = time.perf_counter() - t0_bench
    throughput_rps = round(len(test_records) / total_time_sec, 2)

    cat_metrics = EvaluationMetrics.calculate_classification_metrics(
        y_true=[r["category"] for r in ground_truths],
        y_pred=[p["category"] for p in predictions],
        labels=CATEGORIES,
    )
    struct_metrics = EvaluationMetrics.calculate_structured_metrics(
        ground_truths=ground_truths,
        predictions=predictions,
        json_valid_flags=json_valid_flags,
    )
    perf_metrics = EvaluationMetrics.calculate_performance_metrics(
        latencies_ms=latencies_ms,
        ram_mb=round(peak_ram, 2),
        vram_mb=0.0,
        model_size_mb=round(size_mb, 2),
    )

    return {
        "precision": precision.upper(),
        "model_size_mb": round(size_mb, 1),
        "macro_f1": cat_metrics["macro_f1"],
        "weighted_f1": cat_metrics["weighted_f1"],
        "category_accuracy": struct_metrics["category_accuracy"],
        "json_validity_rate": struct_metrics["json_validity_rate"],
        "exact_match_accuracy": struct_metrics["exact_match_accuracy"],
        "latency_p50_ms": perf_metrics["latency_p50_ms"],
        "latency_p95_ms": perf_metrics["latency_p95_ms"],
        "latency_p99_ms": perf_metrics["latency_p99_ms"],
        "throughput_req_per_sec": throughput_rps,
        "peak_ram_mb": perf_metrics["ram_mb"],
        "load_time_sec": load_time_sec,
    }


def main():
    print("=" * 70)
    print("NEXUS-SLM QUANTIZATION & INFERENCE FORMAT BENCHMARK")
    print("=" * 70)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]
    print(f"Locked test records loaded: {len(test_records)}")

    # 1. FP32 (Full Precision Baseline)
    fp32_res = evaluate_precision("fp32", test_records)

    # 2. INT8 (Dynamic Quantization)
    int8_res = evaluate_precision("int8", test_records)

    # 3. Document Unsupported Hardware Quantizations
    fp16_res = {
        "precision": "FP16 / BF16",
        "supported": False,
        "model_size_mb": 257.5,
        "macro_f1": None,
        "latency_p95_ms": None,
        "throughput_req_per_sec": None,
        "peak_ram_mb": None,
        "status": "UNSUPPORTED_ON_CPU",
        "technical_limitation": "Host PyTorch CPU build does not support native BF16 matrix multiplication instructions; emulation would degrade latency.",
    }
    int4_res = {
        "precision": "INT4",
        "supported": False,
        "model_size_mb": 135.0,
        "macro_f1": None,
        "latency_p95_ms": None,
        "throughput_req_per_sec": None,
        "peak_ram_mb": None,
        "status": "UNSUPPORTED_ON_CPU",
        "technical_limitation": "BitsAndBytes 4-bit NF4/FP4 quantization requires NVIDIA CUDA runtime; host environment is running PyTorch CPU build.",
    }

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": "Intel 8-Core CPU (12 threads), 16GB RAM",
        "test_samples": len(test_records),
        "formats_benchmarked": [fp32_res, int8_res],
        "hardware_limited_formats": [fp16_res, int4_res],
    }

    # Save artifact
    out_json = ARTIFACTS_DIR / "quantization_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved quantization results to: {out_json}")

    # Plot Quantization Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    names = ["FP32 (Active)", "INT8 (Dynamic)"]
    sizes = [fp32_res["model_size_mb"], int8_res["model_size_mb"]]
    f1s = [fp32_res["macro_f1"] * 100, int8_res["macro_f1"] * 100]
    p95s = [fp32_res["latency_p95_ms"], int8_res["latency_p95_ms"]]

    color1 = "#4f46e5"
    color2 = "#10b981"

    # Subplot 1: Model Size vs Macro-F1
    ax1.bar(names, sizes, color=["#6366f1", "#06b6d4"], width=0.45, alpha=0.85)
    ax1.set_ylabel("Model Footprint (MB)", fontsize=10, fontweight="bold")
    ax1.set_title("Model Size Compression", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    for i, v in enumerate(sizes):
        ax1.text(i, v + 15, f"{v:.1f} MB", ha="center", fontweight="bold", fontsize=9)

    # Subplot 2: Macro-F1 vs Latency
    ax2.plot(names, f1s, marker="o", color=color1, linewidth=2.2, label="Macro-F1 (%)")
    ax2.set_ylabel("Macro-F1 (%)", color=color1, fontsize=10, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color1)
    ax2.set_ylim(0, 50)
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(names, p95s, marker="s", color=color2, linewidth=2.2, linestyle="--", label="P95 Latency (ms)")
    ax2_twin.set_ylabel("P95 Latency (ms)", color=color2, fontsize=10, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color2)

    plt.suptitle("NEXUS-SLM Quantization Trade-Off: Precision vs Footprint vs Latency", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plot_path = ARTIFACTS_DIR / "quantization_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved quantization plot to: {plot_path}")

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("NEXUS-SLM-Quantization-Study")
        with mlflow.start_run(run_name="quantization_fp32_vs_int8"):
            mlflow.log_params({"model": BASE_MODEL_ID, "adapter": "nexus_qlora_v1", "hardware": "cpu"})
            mlflow.log_metrics({
                "fp32_macro_f1": fp32_res["macro_f1"],
                "fp32_p95_latency_ms": fp32_res["latency_p95_ms"],
                "fp32_size_mb": fp32_res["model_size_mb"],
                "int8_macro_f1": int8_res["macro_f1"],
                "int8_p95_latency_ms": int8_res["latency_p95_ms"],
                "int8_size_mb": int8_res["model_size_mb"],
            })
            mlflow.log_artifact(str(out_json))
            mlflow.log_artifact(str(plot_path))
    except Exception as e:
        print(f"MLflow log note: {e}")


if __name__ == "__main__":
    main()
