"""Executes Step 4: Real zero-shot evaluation of the base SLM on locked dataset_v1 test benchmark."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import psutil
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.metrics import EvaluationMetrics
from src.inference.parser import OutputParser
from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
TEST_FILE = Path("data/test/test.jsonl")
OUT_DIR = Path("experiments/exp_001_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== STEP 4: RUNNING BASELINE EVALUATION ({MODEL_NAME}) ===")

# 1. Load locked test samples
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_records = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(test_records)} locked test benchmark samples.")

# 2. Load model and tokenizer
print("Loading base SLM in Float32 on CPU...")
t_load_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
load_time = time.time() - t_load_start
num_params = sum(p.numel() for p in model.parameters())
model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

print(f"Base model loaded in {load_time:.2f}s | Params: {num_params:,} | Size: {model_size_mb:.2f} MB")

# 3. Evaluate each sample
parser = OutputParser()
predictions_log = []
ground_truths = []
predictions = []
latencies_ms = []
json_valid_flags = []

ram_before = psutil.virtual_memory().used / (1024**2)
peak_ram = ram_before

for idx, rec in enumerate(test_records, start=1):
    text = rec["text"]
    prompt = PromptFormatter.format_prompt(text, tokenizer=tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    dt_ms = (time.perf_counter() - t0) * 1000
    latencies_ms.append(round(dt_ms, 2))

    cur_ram = psutil.virtual_memory().used / (1024**2)
    if cur_ram > peak_ram:
        peak_ram = cur_ram

    generated_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    is_valid, parsed_data, err = parser.parse_and_validate(raw_output)
    json_valid_flags.append(is_valid)

    if is_valid and parsed_data:
        pred_cat = parsed_data["category"]
        pred_prio = parsed_data["priority"]
        pred_lang = parsed_data["language"]
        pred_human = parsed_data["requires_human"]
    else:
        # If output failed JSON formatting, record unknown/failed
        pred_cat = "unknown"
        pred_prio = "unknown"
        pred_lang = "unknown"
        pred_human = False

    pred_item = {
        "category": pred_cat,
        "priority": pred_prio,
        "language": pred_lang,
        "requires_human": pred_human,
    }
    predictions.append(pred_item)
    ground_truths.append(rec)

    predictions_log.append({
        "sample_id": rec.get("id", f"sample_{idx}"),
        "text": text,
        "ground_truth": {
            "category": rec["category"],
            "priority": rec["priority"],
            "language": rec["language"],
            "requires_human": rec["requires_human"],
        },
        "raw_output": raw_output,
        "json_valid": is_valid,
        "error": err,
        "predicted": pred_item,
        "latency_ms": round(dt_ms, 2),
    })

    if idx % 10 == 0 or idx == len(test_records):
        print(f"Processed {idx}/{len(test_records)} samples... (latest latency: {dt_ms:.1f}ms, valid: {is_valid})")

# 4. Calculate metrics
cat_metrics = EvaluationMetrics.calculate_classification_metrics(
    y_true=[r["category"] for r in ground_truths],
    y_pred=[p["category"] for p in predictions],
    labels=CATEGORIES,
)
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
struct_metrics = EvaluationMetrics.calculate_structured_metrics(
    ground_truths=ground_truths,
    predictions=predictions,
    json_valid_flags=json_valid_flags,
)
perf_metrics = EvaluationMetrics.calculate_performance_metrics(
    latencies_ms=latencies_ms,
    ram_mb=round(peak_ram, 2),
    vram_mb=0.0,
    model_size_mb=round(model_size_mb, 2),
)

final_metrics = {
    "experiment_id": "exp_001_baseline",
    "timestamp": datetime.utcnow().isoformat(),
    "model_name": MODEL_NAME,
    "model_type": "Zero-Shot Base SLM",
    "num_parameters": num_params,
    "test_sample_count": len(test_records),
    "accuracy": cat_metrics["accuracy"],
    "macro_precision": cat_metrics["macro_precision"],
    "macro_recall": cat_metrics["macro_recall"],
    "macro_f1": cat_metrics["macro_f1"],
    "weighted_f1": cat_metrics["weighted_f1"],
    "per_class_f1": {k: v["f1"] for k, v in cat_metrics["per_class"].items()},
    "json_validity_rate": struct_metrics["json_validity_rate"],
    "category_accuracy": struct_metrics["category_accuracy"],
    "priority_accuracy": struct_metrics["priority_accuracy"],
    "language_accuracy": struct_metrics["language_accuracy"],
    "human_escalation_accuracy": struct_metrics["human_escalation_accuracy"],
    "field_level_accuracy": struct_metrics["field_level_accuracy"],
    "exact_match_accuracy": struct_metrics["exact_match_accuracy"],
    "latency_p50_ms": perf_metrics["latency_p50_ms"],
    "latency_p95_ms": perf_metrics["latency_p95_ms"],
    "latency_p99_ms": perf_metrics["latency_p99_ms"],
    "mean_latency_ms": perf_metrics["mean_latency_ms"],
    "throughput_req_per_sec": perf_metrics["throughput_req_per_sec"],
    "peak_ram_mb": perf_metrics["ram_mb"],
    "peak_vram_mb": 0.0,
    "model_size_mb": perf_metrics["model_size_mb"],
}

# 5. Save all artifacts
with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2)

with open(OUT_DIR / "predictions.jsonl", "w", encoding="utf-8") as f:
    for p in predictions_log:
        f.write(json.dumps(p) + "\n")

config_data = {
    "experiment_id": "exp_001_baseline",
    "model_name_or_path": MODEL_NAME,
    "quantization": "fp32_cpu",
    "prompt_style": "chat_template_instructions",
    "dataset": "data/test/test.jsonl",
    "dataset_version": "dataset_v1",
}
with open(OUT_DIR / "config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config_data, f)

# Also save to experiments/baseline/baseline_results.json for comparison table
base_artifact_path = Path("experiments/baseline/baseline_results.json")
base_artifact_path.parent.mkdir(parents=True, exist_ok=True)
with open(base_artifact_path, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2)

report_md = f"""# Experiment 001: Base SLM Baseline Evaluation

- **Model**: `{MODEL_NAME}` ({num_params:,} parameters)
- **Execution Mode**: Zero-Shot Base Model (No Fine-Tuning)
- **Dataset**: `dataset_v1` Locked Test Set ({len(test_records)} examples)
- **Date**: {final_metrics['timestamp']}

## Measured Metrics

| Metric | Measured Value |
|---|---:|
| **Macro-F1** | {final_metrics['macro_f1']:.4f} |
| **Weighted-F1** | {final_metrics['weighted_f1']:.4f} |
| **Category Accuracy** | {final_metrics['category_accuracy'] * 100:.2f}% |
| **JSON Validity Rate** | {final_metrics['json_validity_rate'] * 100:.2f}% |
| **Exact Match Accuracy** | {final_metrics['exact_match_accuracy'] * 100:.2f}% |
| **P50 Latency** | {final_metrics['latency_p50_ms']} ms |
| **P95 Latency** | {final_metrics['latency_p95_ms']} ms |
| **Throughput** | {final_metrics['throughput_req_per_sec']} req/s |
| **Peak RAM** | {final_metrics['peak_ram_mb']} MB |
| **Model Size** | {final_metrics['model_size_mb']} MB |

## Empirical Observations
1. In zero-shot mode without fine-tuning, the base SLM struggles significantly with adhering to structured JSON outputs.
2. The model frequently emits conversational text or python code snippets (`def support_ticket(...)`) rather than valid JSON.
"""
with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
    f.write(report_md)

print("\n" + "="*60)
print("BASE SLM BASELINE RESULTS COMPLETE:")
print(f"  Macro-F1: {final_metrics['macro_f1']:.4f}")
print(f"  JSON Validity Rate: {final_metrics['json_validity_rate']*100:.1f}%")
print(f"  Category Accuracy: {final_metrics['category_accuracy']*100:.1f}%")
print(f"  P50 Latency: {final_metrics['latency_p50_ms']} ms")
print(f"  Peak RAM: {final_metrics['peak_ram_mb']} MB")
print("="*60)
