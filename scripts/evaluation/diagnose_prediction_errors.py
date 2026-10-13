"""Detailed prediction error diagnosis and failure taxonomy script for NEXUS-SLM.

Evaluates candidates on the held-out validation set (N=44) and held-out test set (N=42).
Deconstructs all mismatches by failure field:
- category_mismatch
- priority_mismatch
- language_mismatch
- escalation_mismatch (FN vs FP)
- formatting_error (unparseable / raw text)
- missing_field (omitted key)
- invalid_taxonomy_label (hallucinated label)
Emits experiments/artifacts/prediction_error_diagnosis.json.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
import api.main as api_module
from api.main import app, load_model_instance
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


def diagnose_model_on_split(client: TestClient, split_path: Path, split_name: str, model_version: str) -> Dict[str, Any]:
    with open(split_path, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    failure_records = []
    category_errors = 0
    priority_errors = 0
    language_errors = 0
    escalation_fn_errors = 0  # Safety escape: Ground truth True, predicted False
    escalation_fp_errors = 0  # Over-escalation: Ground truth False, predicted True
    formatting_errors = 0
    missing_field_errors = 0
    invalid_label_errors = 0
    exact_matches = 0

    for s in samples:
        resp = client.post("/predict", json={"text": s["text"]})
        if resp.status_code != 200:
            formatting_errors += 1
            failure_records.append({
                "ticket_id": s.get("id"),
                "text": s.get("text"),
                "error_type": "HTTP_500_SERVER_ERROR",
                "details": resp.text,
            })
            continue

        pred = resp.json()
        gt_cat = s.get("category", "").lower()
        gt_prio = s.get("priority", "").lower()
        gt_lang = s.get("language", "").lower()
        gt_esc = bool(s.get("requires_human", False))

        pr_cat = pred.get("category", "").lower()
        pr_prio = pred.get("priority", "").lower()
        pr_lang = pred.get("language", "").lower()
        pr_esc = bool(pred.get("requires_human", False))

        # Check raw validity
        is_parse_error = pred.get("inference_mode") == "heuristic_fallback" and pred.get("confidence") is None and "parse error" in pred.get("escalation_reason", "").lower()
        raw_out = pred.get("raw_output", "")

        c_match = (gt_cat == pr_cat)
        p_match = (gt_prio == pr_prio)
        l_match = (gt_lang == pr_lang)
        e_match = (gt_esc == pr_esc)

        if c_match and p_match and l_match and e_match:
            exact_matches += 1
        else:
            mismatches = []
            if not c_match:
                category_errors += 1
                mismatches.append({"field": "category", "ground_truth": gt_cat, "predicted": pr_cat})
            if not p_match:
                priority_errors += 1
                mismatches.append({"field": "priority", "ground_truth": gt_prio, "predicted": pr_prio})
            if not l_match:
                language_errors += 1
                mismatches.append({"field": "language", "ground_truth": gt_lang, "predicted": pr_lang})
            if not e_match:
                if gt_esc and not pr_esc:
                    escalation_fn_errors += 1
                    mismatches.append({"field": "escalation", "type": "FALSE_NEGATIVE_SAFETY_ESCAPE", "ground_truth": gt_esc, "predicted": pr_esc})
                else:
                    escalation_fp_errors += 1
                    mismatches.append({"field": "escalation", "type": "FALSE_POSITIVE_OVER_ESCALATION", "ground_truth": gt_esc, "predicted": pr_esc})

            # Check raw output for syntax issues
            raw_err_type = None
            if raw_out:
                try:
                    parsed = json.loads(raw_out)
                    missing = [k for k in ["category", "priority", "language", "requires_human"] if k not in parsed]
                    if missing:
                        missing_field_errors += 1
                        raw_err_type = f"missing_fields: {missing}"
                    if parsed.get("category") not in CATEGORIES:
                        invalid_label_errors += 1
                        raw_err_type = f"invalid_category: {parsed.get('category')}"
                except Exception:
                    formatting_errors += 1
                    raw_err_type = "invalid_json_syntax"

            failure_records.append({
                "ticket_id": s.get("id"),
                "text": s.get("text"),
                "language": gt_lang,
                "mismatches": mismatches,
                "raw_output": raw_out[:120] if raw_out else None,
                "raw_issue": raw_err_type,
            })

    total = len(samples)
    return {
        "model_version": model_version,
        "split_name": split_name,
        "total_samples": total,
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / total, 4),
        "error_distribution_by_field": {
            "category_mismatches": category_errors,
            "category_accuracy": round((total - category_errors) / total, 4),
            "priority_mismatches": priority_errors,
            "priority_accuracy": round((total - priority_errors) / total, 4),
            "language_mismatches": language_errors,
            "language_accuracy": round((total - language_errors) / total, 4),
            "escalation_false_negatives": escalation_fn_errors,
            "escalation_false_positives": escalation_fp_errors,
            "raw_formatting_errors": formatting_errors,
            "raw_missing_fields": missing_field_errors,
            "raw_invalid_taxonomy_labels": invalid_label_errors,
        },
        "detailed_failures": failure_records,
    }


def main():
    print("=" * 75)
    print("NEXUS-SLM: DETAILED PREDICTION ERROR DIAGNOSIS")
    print("=" * 75)

    val_path = PROJECT_ROOT / "data" / "validation" / "validation.jsonl"
    test_path = PROJECT_ROOT / "data" / "test" / "test.jsonl"

    models = [
        "nexus-qlora-v1",
        "qwen25_05b_lora_seed42_q8_0",
        "qwen25_05b_lora_seed42_q4_k_m",
        "qwen25_05b_lora_seed42_int8",
    ]

    all_diagnostics = {}

    for ver in models:
        print(f"\nDiagnosing model: {ver}...")
        engine = load_model_instance(ver)
        api_module.inference_engine = engine
        api_module.ACTIVE_VERSION = ver
        client = TestClient(app)

        val_diag = diagnose_model_on_split(client, val_path, "validation", ver)
        test_diag = diagnose_model_on_split(client, test_path, "test_locked", ver)

        print(f"  Validation Exact Matches: {val_diag['exact_matches']}/{val_diag['total_samples']} ({val_diag['exact_match_rate']})")
        print(f"     Category Errors: {val_diag['error_distribution_by_field']['category_mismatches']}")
        print(f"     Priority Errors: {val_diag['error_distribution_by_field']['priority_mismatches']}")
        print(f"     Language Errors: {val_diag['error_distribution_by_field']['language_mismatches']}")
        print(f"     Escalation FNs: {val_diag['error_distribution_by_field']['escalation_false_negatives']}, FPs: {val_diag['error_distribution_by_field']['escalation_false_positives']}")
        print(f"  Test Locked Exact Matches: {test_diag['exact_matches']}/{test_diag['total_samples']} ({test_diag['exact_match_rate']})")
        print(f"     Category Errors: {test_diag['error_distribution_by_field']['category_mismatches']}")
        print(f"     Priority Errors: {test_diag['error_distribution_by_field']['priority_mismatches']}")

        all_diagnostics[ver] = {
            "validation": val_diag,
            "test_locked": test_diag,
        }

    out_file = PROJECT_ROOT / "experiments" / "artifacts" / "prediction_error_diagnosis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_diagnostics, f, indent=2)
    print(f"\n[Artifact Saved] Full diagnostic report: {out_file}")


if __name__ == "__main__":
    main()
