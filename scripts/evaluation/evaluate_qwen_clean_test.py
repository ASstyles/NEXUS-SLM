"""Evaluates the clean V2 Qwen2.5-0.5B LoRA candidate against the locked clean test benchmark."""

import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import psutil
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.parser import OutputParser
from src.inference.policy import HumanEscalationPolicy
from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total) / denom
    return round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4)


def main():
    test_file = ROOT / "data/v2_clean_template_grouped/test_clean.jsonl"
    adapter_path = ROOT / "experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter"
    base_target = ROOT / "models/scratch_merged_qwen"

    assert test_file.exists(), f"Test set missing at {test_file}"
    assert adapter_path.exists(), f"Adapter missing at {adapter_path}"
    assert base_target.exists(), f"Base model missing at {base_target}"

    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    N = len(records)
    print(f"Loaded {N} records from {test_file}")

    # Set threads for optimal multi-core CPU throughput
    cpu_cores = os.cpu_count() or 8
    torch.set_num_threads(cpu_cores)

    # 1. Measure Cold Start Time
    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024**2)

    t_load_start = time.perf_counter()
    print("Loading tokenizer and base Qwen model...")
    tok = AutoTokenizer.from_pretrained(str(adapter_path))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_target),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print("Applying clean V2 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    cold_start_time_ms = round((time.perf_counter() - t_load_start) * 1000, 2)
    ram_after_mb = process.memory_info().rss / (1024**2)
    model_ram_mb = round(ram_after_mb - ram_before_mb, 2)
    print(f"Model loaded in {cold_start_time_ms:.1f}ms. RAM Footprint: {model_ram_mb:.1f} MB")

    parser = OutputParser()
    eos_ids = [tok.eos_token_id, 151645, 151643]

    batch_size = 16
    raw_outputs = []
    latencies_per_sample_ms = []

    t_eval_start = time.perf_counter()
    print(f"Running formal evaluation over {N} records in batches of {batch_size}...")

    for i in range(0, N, batch_size):
        batch_records = records[i : i + batch_size]
        prompts = [PromptFormatter.format_prompt(r["text"], tokenizer=tok) for r in batch_records]

        t_b0 = time.perf_counter()
        inputs = tok(prompts, return_tensors="pt", padding=True)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids,
            )

        batch_time_ms = (time.perf_counter() - t_b0) * 1000
        per_sample_ms = batch_time_ms / len(batch_records)

        for out_seq in outputs:
            gen_text = tok.decode(out_seq[prompt_len:], skip_special_tokens=True).strip()
            raw_outputs.append(gen_text)
            latencies_per_sample_ms.append(per_sample_ms)

        if (i // batch_size + 1) % 5 == 0 or (i + batch_size >= N):
            done = min(i + batch_size, N)
            pct = (done / N) * 100
            print(f"  Processed {done}/{N} ({pct:.1f}%) [batch latency: {batch_time_ms:.1f}ms, {per_sample_ms:.1f}ms/sample]")

    total_eval_time_s = time.perf_counter() - t_eval_start
    throughput = round(N / total_eval_time_s, 2)
    print(f"Generation complete in {total_eval_time_s:.2f}s ({throughput} req/s). Analyzing predictions...")

    # Parse and validate predictions
    y_true_cat, y_pred_cat = [], []
    y_true_prio, y_pred_prio = [], []
    y_true_lang, y_pred_lang = [], []
    y_true_esc, y_pred_esc = [], []

    valid_json_count = 0
    schema_valid_count = 0
    exact_match_count = 0
    parsing_failures = []
    false_negative_examples = []

    for idx, (rec, raw_text) in enumerate(zip(records, raw_outputs)):
        is_valid, parsed_data, err = parser.parse_and_validate(raw_text)

        if is_valid and parsed_data:
            valid_json_count += 1
            schema_valid_count += 1

            cat = parsed_data["category"]
            prio = parsed_data["priority"]
            lang = parsed_data["language"]
            model_req_human = parsed_data["requires_human"]

            # Exact field match check
            if (
                cat == rec["category"]
                and prio == rec["priority"]
                and lang == rec["language"]
                and model_req_human == rec["requires_human"]
            ):
                exact_match_count += 1
        else:
            parsing_failures.append({"index": idx, "id": rec["id"], "raw_output": raw_text, "error": err})
            cat = "general_query"
            prio = "medium"
            lang = "english"
            model_req_human = True

        # Human Escalation Policy
        req_human, esc_reason, _, final_cat, final_prio = HumanEscalationPolicy.evaluate_escalation(
            text=rec["text"],
            category=cat,
            priority=prio,
            language=lang,
            raw_confidence=0.90 if is_valid else None,  # Model confidence estimate
            is_parse_error=(not is_valid),
            inference_mode="neural",
            model_requires_human=model_req_human,
        )

        y_true_cat.append(rec["category"])
        y_pred_cat.append(cat)

        y_true_prio.append(rec["priority"])
        y_pred_prio.append(prio)

        y_true_lang.append(rec["language"])
        y_pred_lang.append(lang)

        true_esc = bool(rec.get("requires_human", False))
        pred_esc = bool(req_human)
        y_true_esc.append(true_esc)
        y_pred_esc.append(pred_esc)

        if true_esc and not pred_esc:
            false_negative_examples.append({
                "id": rec["id"],
                "text": rec["text"],
                "true_category": rec["category"],
                "pred_category": cat,
                "reason": esc_reason,
            })

    # Metrics computation
    cat_labels = sorted(list(CATEGORIES))
    cat_acc = round(accuracy_score(y_true_cat, y_pred_cat), 4)
    cat_macro_f1 = round(f1_score(y_true_cat, y_pred_cat, labels=cat_labels, average="macro", zero_division=0), 4)
    cat_weighted_f1 = round(f1_score(y_true_cat, y_pred_cat, labels=cat_labels, average="weighted", zero_division=0), 4)
    cat_macro_p = round(precision_score(y_true_cat, y_pred_cat, labels=cat_labels, average="macro", zero_division=0), 4)
    cat_macro_r = round(recall_score(y_true_cat, y_pred_cat, labels=cat_labels, average="macro", zero_division=0), 4)

    cat_report = classification_report(y_true_cat, y_pred_cat, labels=cat_labels, output_dict=True, zero_division=0)
    cat_cm = confusion_matrix(y_true_cat, y_pred_cat, labels=cat_labels).tolist()

    per_category_summary = {}
    for c in cat_labels:
        supp = cat_report.get(c, {}).get("support", 0)
        corr = sum(1 for yt, yp in zip(y_true_cat, y_pred_cat) if yt == c and yp == c)
        ci_low, ci_high = wilson_interval(corr, supp) if supp > 0 else (0.0, 0.0)
        per_category_summary[c] = {
            "precision": round(cat_report.get(c, {}).get("precision", 0.0), 4),
            "recall": round(cat_report.get(c, {}).get("recall", 0.0), 4),
            "f1": round(cat_report.get(c, {}).get("f1-score", 0.0), 4),
            "support": supp,
            "accuracy_ci_95": [ci_low, ci_high],
            "is_sparse_sample": supp <= 10,
        }

    prio_labels = ["low", "medium", "high", "critical"]
    prio_acc = round(accuracy_score(y_true_prio, y_pred_prio), 4)
    prio_macro_f1 = round(f1_score(y_true_prio, y_pred_prio, labels=prio_labels, average="macro", zero_division=0), 4)
    prio_weighted_f1 = round(f1_score(y_true_prio, y_pred_prio, labels=prio_labels, average="weighted", zero_division=0), 4)
    prio_cm = confusion_matrix(y_true_prio, y_pred_prio, labels=prio_labels).tolist()
    prio_report = classification_report(y_true_prio, y_pred_prio, labels=prio_labels, output_dict=True, zero_division=0)

    lang_labels = ["english", "hindi", "hinglish"]
    lang_acc = round(accuracy_score(y_true_lang, y_pred_lang), 4)
    per_lang_acc = {}
    for l in lang_labels:
        l_total = sum(1 for yt in y_true_lang if yt == l)
        l_corr = sum(1 for yt, yp in zip(y_true_lang, y_pred_lang) if yt == l and yp == l)
        per_lang_acc[l] = round(l_corr / l_total, 4) if l_total > 0 else 1.0

    tp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is True and yp is True)
    fp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is False and yp is True)
    fn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is True and yp is False)
    tn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is False and yp is False)

    esc_recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    esc_precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    esc_rate = round((tp + fp) / N, 4)
    unnecessary_esc_rate = round(fp / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    p50_lat = round(float(np.percentile(latencies_per_sample_ms, 50)), 2)
    p90_lat = round(float(np.percentile(latencies_per_sample_ms, 90)), 2)
    p95_lat = round(float(np.percentile(latencies_per_sample_ms, 95)), 2)
    p99_lat = round(float(np.percentile(latencies_per_sample_ms, 99)), 2)

    results = {
        "evaluation_name": "clean_v2_test_benchmark_qwen25_05b_lora",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_dataset": "data/v2_clean_template_grouped/test_clean.jsonl",
        "sample_count": N,
        "engine": "qwen25_05b_clean_v2_lora",
        "inference_mode": "neural",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": "experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter",
        "adapter_model_sha256": "9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2",
        "metrics": {
            "category": {
                "accuracy": cat_acc,
                "macro_precision": cat_macro_p,
                "macro_recall": cat_macro_r,
                "macro_f1": cat_macro_f1,
                "weighted_f1": cat_weighted_f1,
                "per_category": per_category_summary,
                "confusion_matrix": {
                    "labels": cat_labels,
                    "matrix": cat_cm,
                },
            },
            "priority": {
                "accuracy": prio_acc,
                "macro_f1": prio_macro_f1,
                "weighted_f1": prio_weighted_f1,
                "per_priority": {p: round(prio_report.get(p, {}).get("recall", 0.0), 4) for p in prio_labels},
                "confusion_matrix": {
                    "labels": prio_labels,
                    "matrix": prio_cm,
                },
            },
            "language": {
                "overall_accuracy": lang_acc,
                "english_accuracy": per_lang_acc.get("english", 0.0),
                "hindi_accuracy": per_lang_acc.get("hindi", 0.0),
                "hinglish_accuracy": per_lang_acc.get("hinglish", 0.0),
            },
            "structured_output": {
                "valid_json_rate": round(valid_json_count / N, 4),
                "schema_validity_rate": round(schema_valid_count / N, 4),
                "exact_field_match_rate": round(exact_match_count / N, 4),
                "parsing_failures_count": len(parsing_failures),
            },
            "safety": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "escalation_recall": esc_recall,
                "escalation_precision": esc_precision,
                "overall_escalation_rate": esc_rate,
                "unnecessary_escalation_rate": unnecessary_esc_rate,
                "false_negative_count": len(false_negative_examples),
            },
            "performance": {
                "cold_start_time_ms": cold_start_time_ms,
                "model_ram_footprint_mb": model_ram_mb,
                "latency_p50_ms": p50_lat,
                "latency_p90_ms": p90_lat,
                "latency_p95_ms": p95_lat,
                "latency_p99_ms": p99_lat,
                "throughput_req_per_sec": throughput,
                "total_evaluation_time_sec": round(total_eval_time_s, 2),
            },
        },
    }

    out_file = ROOT / "experiments/artifacts/qwen25_05b_clean_v2_lora_test_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("QWEN2.5-0.5B LoRA CLEAN TEST BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset: Locked Clean Test Set (N={N})")
    print(f"Category Accuracy: {cat_acc:.2%} | Macro-F1: {cat_macro_f1:.4f} | Weighted-F1: {cat_weighted_f1:.4f}")
    print(f"Priority Accuracy: {prio_acc:.2%} | Macro-F1: {prio_macro_f1:.4f}")
    print(f"Language Accuracy: {lang_acc:.2%} (EN: {per_lang_acc.get('english',0):.2%}, HI: {per_lang_acc.get('hindi',0):.2%}, Hinglish: {per_lang_acc.get('hinglish',0):.2%})")
    print(f"JSON Validity: {valid_json_count/N:.2%} | Exact Field Match: {exact_match_count/N:.2%}")
    print(f"Escalation Recall: {esc_recall:.2%} | Precision: {esc_precision:.2%} | Rate: {esc_rate:.2%}")
    print(f"Latency P50: {p50_lat}ms | P95: {p95_lat}ms | Throughput: {throughput} req/s")
    print(f"Results written to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
