"""Detailed latency profiling harness for NEXUS-SLM.
Decomposes total request latency into:
1. Tokenization time
2. Autoregressive token generation time
3. Post-processing / JSON parsing & validation time
Measures P50, P95, P99, Mean, Std Dev, and Throughput.
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(WORKSPACE_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.inference.parser import OutputParser
from src.models.prompt_formatter import PromptFormatter

ARTIFACTS_DIR = WORKSPACE_ROOT / "experiments" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILE = WORKSPACE_ROOT / "data" / "test" / "test.jsonl"
ADAPTER_PATH = WORKSPACE_ROOT / "models" / "nexus_qlora_v1"
BASE_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"


def run_latency_profile():
    print("=" * 70)
    print("NEXUS-SLM DETAILED INFERENCE LATENCY PROFILING")
    print("Decomposing tokenization, generation, and output parsing latency")
    print("=" * 70)

    # 1. Measure one-time cold load time
    t0_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()
    cold_load_time_ms = round((time.perf_counter() - t0_load) * 1000, 2)
    print(f"One-Time Model Cold Load Time: {cold_load_time_ms:.1f} ms ({cold_load_time_ms/1000:.2f} s)")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    # Warm-up run (1 query, excluded from stats)
    warmup_text = "Mera payment fail ho gaya."
    w_prompt = PromptFormatter.format_prompt(warmup_text, tokenizer=tokenizer)
    w_inputs = tokenizer(w_prompt, return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(**w_inputs, max_new_tokens=20, do_sample=False)
    print("Warm-up pass completed.\n")

    parser = OutputParser()
    tokenization_times = []
    generation_times = []
    parsing_times = []
    total_request_times = []
    output_token_counts = []

    print(f"Profiling across {len(test_records)} locked test queries...")
    for idx, rec in enumerate(test_records, 1):
        text = rec["text"]

        # Phase 1: Prompt formatting & Tokenization
        t0_tok = time.perf_counter()
        prompt = PromptFormatter.format_prompt(text, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        dt_tok = (time.perf_counter() - t0_tok) * 1000
        tokenization_times.append(dt_tok)

        # Phase 2: Neural Generation
        t0_gen = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=45,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt_gen = (time.perf_counter() - t0_gen) * 1000
        generation_times.append(dt_gen)

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        output_token_counts.append(len(new_tokens))

        # Phase 3: Post-processing / JSON parsing
        t0_parse = time.perf_counter()
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        is_valid, parsed, _ = parser.parse_and_validate(raw_output)
        dt_parse = (time.perf_counter() - t0_parse) * 1000
        parsing_times.append(dt_parse)

        total_dt = dt_tok + dt_gen + dt_parse
        total_request_times.append(total_dt)

    def _stats(arr):
        return {
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "std_dev_ms": round(float(np.std(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
        }

    total_stats = _stats(total_request_times)
    tok_stats = _stats(tokenization_times)
    gen_stats = _stats(generation_times)
    parse_stats = _stats(parsing_times)

    total_bench_time_sec = sum(total_request_times) / 1000.0
    throughput_rps = round(len(test_records) / total_bench_time_sec, 2)
    avg_gen_tokens = round(float(np.mean(output_token_counts)), 1)
    tok_per_sec = round(sum(output_token_counts) / (sum(generation_times) / 1000.0), 2)

    profile_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": "Intel 8-Core CPU (12 threads), 16GB RAM",
        "sample_count": len(test_records),
        "cold_model_load_ms": cold_load_time_ms,
        "throughput_req_per_sec": throughput_rps,
        "generation_tokens_per_sec": tok_per_sec,
        "average_tokens_generated": avg_gen_tokens,
        "total_request_latency": total_stats,
        "phase_breakdown": {
            "tokenization": tok_stats,
            "generation": gen_stats,
            "parsing_validation": parse_stats,
        },
        "phase_percentages": {
            "tokenization_pct": round((tok_stats["mean_ms"] / total_stats["mean_ms"]) * 100, 2),
            "generation_pct": round((gen_stats["mean_ms"] / total_stats["mean_ms"]) * 100, 2),
            "parsing_pct": round((parse_stats["mean_ms"] / total_stats["mean_ms"]) * 100, 2),
        },
    }

    out_json = ARTIFACTS_DIR / "latency_breakdown.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profile_report, f, indent=2)
    print(f"Saved latency breakdown report to: {out_json}")

    # Plot Latency Breakdown & Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot 1: Phase Latency Breakdown (Stacked / Bar)
    phases = ["Tokenization", "Generation", "JSON Parsing"]
    means = [tok_stats["mean_ms"], gen_stats["mean_ms"], parse_stats["mean_ms"]]
    colors = ["#3b82f6", "#6366f1", "#10b981"]

    ax1.bar(phases, means, color=colors, width=0.45, alpha=0.85)
    ax1.set_ylabel("Mean Latency (ms)", fontsize=10, fontweight="bold")
    ax1.set_title("Latency Breakdown per Request Phase", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    for i, v in enumerate(means):
        ax1.text(i, v + 25, f"{v:.1f} ms", ha="center", fontweight="bold", fontsize=9)

    # Subplot 2: Total Latency Percentiles (P50, Mean, P95, P99)
    pcts = ["P50", "Mean", "P95", "P99"]
    pct_vals = [total_stats["p50_ms"], total_stats["mean_ms"], total_stats["p95_ms"], total_stats["p99_ms"]]
    ax2.bar(pcts, pct_vals, color=["#10b981", "#6366f1", "#f59e0b", "#ef4444"], width=0.45, alpha=0.85)
    ax2.set_ylabel("Latency (ms)", fontsize=10, fontweight="bold")
    ax2.set_title("Request Latency Percentile Distribution", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    for i, v in enumerate(pct_vals):
        ax2.text(i, v + 25, f"{v:.0f} ms", ha="center", fontweight="bold", fontsize=9)

    plt.suptitle("NEXUS-SLM Latency Characteristics: Execution Profile on CPU", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plot_path = ARTIFACTS_DIR / "latency_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved latency distribution plot to: {plot_path}")

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("NEXUS-SLM-Latency-Benchmark")
        with mlflow.start_run(run_name="latency_profile"):
            mlflow.log_params({"model": BASE_MODEL_ID, "adapter": "nexus_qlora_v1", "hardware": "cpu"})
            mlflow.log_metrics({
                "latency_p50_ms": total_stats["p50_ms"],
                "latency_p95_ms": total_stats["p95_ms"],
                "latency_p99_ms": total_stats["p99_ms"],
                "latency_mean_ms": total_stats["mean_ms"],
                "latency_std_dev_ms": total_stats["std_dev_ms"],
                "cold_load_time_ms": cold_load_time_ms,
                "throughput_rps": throughput_rps,
                "generation_tok_per_sec": tok_per_sec,
            })
            mlflow.log_artifact(str(out_json))
            mlflow.log_artifact(str(plot_path))
    except Exception as e:
        print(f"MLflow note: {e}")

    return profile_report


if __name__ == "__main__":
    run_latency_profile()
