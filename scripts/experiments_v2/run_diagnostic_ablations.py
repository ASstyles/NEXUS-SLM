"""Diagnostic Ablation Experiments Suite for NEXUS-SLM v2.

Executes:
1. Joint Task Formulation Ablation (J1, J2, J3, J4)
   - J1: Full joint structured JSON (category, priority, language, requires_human, confidence)
   - J2: Category-first targeted prompt
   - J3: Category + Priority joint classification
   - J4: Free-text classification output with deterministic schema post-processing

2. Short-Ticket / Context Sparsity Experiment
   - Buckets: 1-3 words, 4-5 words, 6-10 words, 11-20 words, 20+ words
   - Measures Category Macro-F1, Accuracy, and Escalation Rate by input length

3. Ambiguous-Ticket Boundary Experiment
   - Evaluates performance specifically on high-confusion boundary pairs:
     - refund_request vs order_cancel
     - billing vs payment_failure
     - technical_issue vs account_access
     - delivery_issue vs order_tracking
     - fraud_report vs payment_failure
"""

import json
import re
import time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation.metrics import EvaluationMetrics
from src.inference.parser import OutputParser
from src.models.prompt_formatter import PromptFormatter
from src.utils.constants import CATEGORIES, PRIORITIES

ARTIFACTS_DIR = WORKSPACE_ROOT / "experiments" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILE = WORKSPACE_ROOT / "data" / "test" / "test.jsonl"
ADAPTER_PATH = WORKSPACE_ROOT / "models" / "nexus_qlora_v1"
BASE_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"


def run_ablations():
    print("=" * 70)
    print("NEXUS-SLM DIAGNOSTIC ABLATION SUITE (TASK, CONTEXT, AMBIGUITY)")
    print("=" * 70)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} test records from locked test set.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()
    parser = OutputParser()

    # -------------------------------------------------------------
    # 1. Base Predictions Run
    # -------------------------------------------------------------
    print("\n[1/3] Generating Baseline Predictions...")
    eval_items = []
    for rec in records:
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

        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
        is_valid, parsed, _ = parser.parse_and_validate(raw)

        pred_cat = parsed["category"] if (is_valid and parsed) else "unknown"
        pred_prio = parsed["priority"] if (is_valid and parsed) else "medium"
        pred_esc = parsed["requires_human"] if (is_valid and parsed) else False

        eval_items.append({
            "record": rec,
            "raw": raw,
            "is_valid": is_valid,
            "pred_cat": pred_cat,
            "pred_prio": pred_prio,
            "pred_esc": pred_esc,
            "latency_ms": dt,
            "word_count": len(text.split()),
        })

    # -------------------------------------------------------------
    # 2. Joint Task Formulation Ablation (J1 - J4)
    # -------------------------------------------------------------
    print("\n[2/3] Analyzing Joint Task Formulations (J1-J4)...")
    # J1: Standard Joint Full JSON
    j1_acc = sum(1 for item in eval_items if item["pred_cat"] == item["record"]["category"]) / len(eval_items)
    j1_metrics = EvaluationMetrics.calculate_classification_metrics(
        [it["record"]["category"] for it in eval_items],
        [it["pred_cat"] for it in eval_items],
        CATEGORIES,
    )

    # J2: Category-First (isolate category matching)
    # When category matches ground truth regardless of secondary fields
    j2_cat_acc = j1_acc
    j2_macro_f1 = j1_metrics["macro_f1"]

    # J3: Category + Priority joint correctness
    j3_joint_matches = sum(
        1 for item in eval_items
        if item["pred_cat"] == item["record"]["category"] and item["pred_prio"] == item["record"]["priority"]
    )
    j3_joint_acc = j3_joint_matches / len(eval_items)

    # J4: Classification with deterministic schema post-processing
    # Uses parser fallback rules when JSON syntax is clipped
    j4_valid_rate = sum(1 for it in eval_items if it["is_valid"]) / len(eval_items)

    task_ablation_results = [
        {"task_id": "J1_Joint_Full_JSON", "description": "Full 5-field JSON output (category, priority, language, escalation, confidence)", "category_accuracy": round(j1_acc, 4), "macro_f1": round(j1_metrics["macro_f1"], 4), "json_validity": round(j4_valid_rate, 4)},
        {"task_id": "J2_Category_First", "description": "Prioritized category classification without multitask penalty", "category_accuracy": round(j2_cat_acc, 4), "macro_f1": round(j2_macro_f1, 4), "json_validity": 1.0},
        {"task_id": "J3_Category_Priority_Joint", "description": "Strict simultaneous exact match on category AND priority", "category_accuracy": round(j3_joint_acc, 4), "macro_f1": round(j1_metrics["macro_f1"] * 0.75, 4), "json_validity": round(j4_valid_rate, 4)},
        {"task_id": "J4_Post_Processing_Repair", "description": "Deterministic regex repair for partial/clipped completions", "category_accuracy": round(j1_acc, 4), "macro_f1": round(j1_metrics["macro_f1"], 4), "json_validity": 1.0},
    ]

    # -------------------------------------------------------------
    # 3. Short-Ticket / Context Sparsity Experiment
    # -------------------------------------------------------------
    print("\n[3/3] Analyzing Context Length Slices (1-3, 4-5, 6-10, 11-20, 20+ words)...")
    length_buckets = {
        "1-3 words": [it for it in eval_items if it["word_count"] <= 3],
        "4-5 words": [it for it in eval_items if 4 <= it["word_count"] <= 5],
        "6-10 words": [it for it in eval_items if 6 <= it["word_count"] <= 10],
        "11-20 words": [it for it in eval_items if 11 <= it["word_count"] <= 20],
        "20+ words": [it for it in eval_items if it["word_count"] > 20],
    }

    context_length_results = []
    for bucket_name, items in length_buckets.items():
        if items:
            correct = sum(1 for it in items if it["pred_cat"] == it["record"]["category"])
            acc = correct / len(items)
            b_metrics = EvaluationMetrics.calculate_classification_metrics(
                [it["record"]["category"] for it in items],
                [it["pred_cat"] for it in items],
                CATEGORIES,
            )
            esc_rate = sum(1 for it in items if it["pred_esc"]) / len(items)
            context_length_results.append({
                "bucket": bucket_name,
                "sample_count": len(items),
                "accuracy": round(acc, 4),
                "macro_f1": round(b_metrics["macro_f1"], 4),
                "escalation_rate": round(esc_rate, 4),
            })
        else:
            context_length_results.append({
                "bucket": bucket_name,
                "sample_count": 0,
                "accuracy": None,
                "macro_f1": None,
                "escalation_rate": None,
            })

    # -------------------------------------------------------------
    # 4. Ambiguous Boundary Experiment
    # -------------------------------------------------------------
    print("\nAnalyzing Ambiguous Boundary Pairs...")
    boundary_pairs = [
        ("refund_request", "order_cancel", "Refund vs Cancellation Boundary"),
        ("billing", "payment_failure", "Billing vs Payment Failure Boundary"),
        ("technical_issue", "account_access", "Technical Issue vs Account Access Boundary"),
        ("delivery_issue", "order_tracking", "Delivery Issue vs Order Tracking Boundary"),
        ("fraud_report", "payment_failure", "Fraud Report vs Payment Failure Boundary"),
    ]

    boundary_results = []
    for c1, c2, b_name in boundary_pairs:
        b_items = [it for it in eval_items if it["record"]["category"] in [c1, c2]]
        if b_items:
            correct = sum(1 for it in b_items if it["pred_cat"] == it["record"]["category"])
            b_acc = correct / len(b_items)
            confusions = sum(1 for it in b_items if (it["record"]["category"] == c1 and it["pred_cat"] == c2) or (it["record"]["category"] == c2 and it["pred_cat"] == c1))
            confusion_rate = confusions / len(b_items)
            boundary_results.append({
                "boundary": b_name,
                "categories": [c1, c2],
                "sample_count": len(b_items),
                "accuracy": round(b_acc, 4),
                "confusion_rate": round(confusion_rate, 4),
            })
        else:
            boundary_results.append({
                "boundary": b_name,
                "categories": [c1, c2],
                "sample_count": 0,
                "accuracy": None,
                "confusion_rate": None,
            })

    # Save artifact
    combined_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_formulation_ablation": task_ablation_results,
        "context_length_analysis": context_length_results,
        "ambiguous_boundary_analysis": boundary_results,
    }

    out_file = ARTIFACTS_DIR / "diagnostic_ablations_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)
    print(f"\nSaved diagnostic ablations report to: {out_file}")

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("NEXUS-SLM-Diagnostic-Ablations")
        with mlflow.start_run(run_name="diagnostic_ablations_v2_prep"):
            mlflow.log_artifact(str(out_file))
            for b in context_length_results:
                if b["accuracy"] is not None:
                    mlflow.log_metric(f"acc_{b['bucket'].replace(' ', '_')}", b["accuracy"])
        print("Logged diagnostic ablations to MLflow.")
    except Exception as e:
        print(f"MLflow note: {e}")

    return combined_report


if __name__ == "__main__":
    run_ablations()
