"""Executes Step 8: Detailed error analysis on fine-tuned NEXUS-SLM predictions."""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.error_analysis import ErrorAnalyzer
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

PRED_FILE = Path("experiments/exp_002_qlora/predictions.jsonl")
OUT_FILE = Path("experiments/exp_002_qlora/error_analysis.json")
REPORT_MD = Path("experiments/exp_002_qlora/error_analysis_report.md")

print("=== STEP 8: ERROR ANALYSIS EXECUTION ===")

if not PRED_FILE.exists():
    raise FileNotFoundError(f"Predictions file not found at {PRED_FILE}")

records = []
with open(PRED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

analyzer = ErrorAnalyzer(output_dir="experiments/exp_002_qlora")

gts = [r["ground_truth"] for r in records]
preds = [r["predicted"] for r in records]
texts = [r["text"] for r in records]
confs = [1.0 if r.get("json_valid") else 0.0 for r in records]

report = analyzer.analyze_predictions(
    ground_truths=gts,
    predictions=preds,
    texts=texts,
    confidences=confs,
)

# 1. Confusion Matrix
cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
conf_matrix = [[0 for _ in range(len(CATEGORIES))] for _ in range(len(CATEGORIES))]

for r in records:
    gt_c = r["ground_truth"].get("category", "unknown")
    pr_c = r["predicted"].get("category", "unknown")
    if gt_c in cat_to_idx and pr_c in cat_to_idx:
        conf_matrix[cat_to_idx[gt_c]][cat_to_idx[pr_c]] += 1

# 2 & 3. Category Breakdown
cat_total = Counter(r["ground_truth"]["category"] for r in records)
cat_correct = Counter()
lang_total = Counter(r["ground_truth"]["language"] for r in records)
lang_correct = Counter()
prio_total = Counter(r["ground_truth"]["priority"] for r in records)
prio_correct = Counter()

for r in records:
    gt_cat = r["ground_truth"]["category"]
    pr_cat = r["predicted"]["category"]
    gt_lang = r["ground_truth"]["language"]
    pr_lang = r["predicted"]["language"]
    gt_prio = r["ground_truth"]["priority"]
    pr_prio = r["predicted"]["priority"]

    if gt_cat == pr_cat and r.get("json_valid"):
        cat_correct[gt_cat] += 1
    if gt_lang == pr_lang and r.get("json_valid"):
        lang_correct[gt_lang] += 1
    if gt_prio == pr_prio and r.get("json_valid"):
        prio_correct[gt_prio] += 1

cat_accs = {c: (cat_correct[c] / cat_total[c]) if cat_total[c] else 0.0 for c in cat_total}
sorted_cats = sorted(cat_accs.items(), key=lambda x: (x[1], x[0]))

worst_cats = sorted_cats[:5]
best_cats = sorted_cats[-5:]

# 4. Language Performance
lang_perf = {
    l: {
        "total": lang_total[l],
        "correct": lang_correct[l],
        "accuracy": (lang_correct[l] / lang_total[l]) if lang_total[l] else 0.0,
    }
    for l in sorted(lang_total.keys())
}

# 5. Priority Performance
prio_perf = {
    p: {
        "total": prio_total[p],
        "correct": prio_correct[p],
        "accuracy": (prio_correct[p] / prio_total[p]) if prio_total[p] else 0.0,
    }
    for p in sorted(prio_total.keys())
}

# 6. Invalid JSON Cases
invalid_json_records = [r for r in records if not r.get("json_valid")]
invalid_json_count = len(invalid_json_records)

# 7. Representative Errors (at least 20)
representative_errors = []
for i, r in enumerate(records):
    raw_gen = r.get("raw_output", "")
    gt = r["ground_truth"]
    pr = r["predicted"]
    
    # Analyze concrete failure cause
    if not r.get("json_valid"):
        if "{" not in raw_gen:
            cause = "Model failed to emit JSON braces entirely; output was conversational preamble or echoed prompt text."
        elif "}" not in raw_gen:
            cause = "Model generated opening brace but generation truncated before emitting closing brace."
        else:
            cause = f"Model emitted malformed text that failed json.loads: {r.get('error', 'Syntax error')}"
    elif gt["category"] != pr["category"]:
        cause = f"Model output valid JSON but misclassified category as '{pr['category']}' instead of '{gt['category']}'."
    else:
        cause = "Field mismatch in secondary fields (priority, language, or escalation)."

    representative_errors.append({
        "sample_index": i + 1,
        "input": r["text"],
        "expected": {
            "category": gt["category"],
            "priority": gt["priority"],
            "language": gt["language"],
            "requires_human": gt.get("requires_human", False),
        },
        "predicted": {
            "category": pr["category"],
            "priority": pr["priority"],
            "language": pr["language"],
            "requires_human": pr.get("requires_human", False),
        },
        "raw_output": raw_gen[:200],
        "json_valid": r.get("json_valid", False),
        "why_it_failed": cause,
    })

    if len(representative_errors) >= 25:
        break

detailed_report = {
    "total_evaluated": len(records),
    "total_failures": report["total_failures"],
    "overall_accuracy": report["overall_accuracy"],
    "categories": CATEGORIES,
    "confusion_matrix": conf_matrix,
    "worst_performing_categories": [
        {"category": c, "accuracy": round(a, 4), "total_samples": cat_total[c]}
        for c, a in worst_cats
    ],
    "best_performing_categories": [
        {"category": c, "accuracy": round(a, 4), "total_samples": cat_total[c]}
        for c, a in best_cats
    ],
    "language_performance": lang_perf,
    "priority_performance": prio_perf,
    "invalid_json_count": invalid_json_count,
    "invalid_json_rate": round(invalid_json_count / len(records), 4) if records else 0.0,
    "error_taxonomy_distribution": report.get("error_distribution", {}),
    "representative_errors_count": len(representative_errors),
    "representative_errors": representative_errors,
}

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(detailed_report, f, indent=2)

dashboard_err_path = Path("experiments/artifacts/error_analysis_report.json")
dashboard_err_path.parent.mkdir(parents=True, exist_ok=True)
with open(dashboard_err_path, "w", encoding="utf-8") as f:
    json.dump(detailed_report, f, indent=2)

print("\n" + "="*60)
print("ERROR ANALYSIS COMPLETED AND SAVED:")
print(f"  Total Evaluated:        {len(records)}")
print(f"  Total Failures:         {detailed_report['total_failures']}")
print(f"  Invalid JSON Count:     {invalid_json_count} / {len(records)} ({detailed_report['invalid_json_rate']*100:.1f}%)")
print(f"  Worst Categories:       {[c['category'] for c in detailed_report['worst_performing_categories'][:3]]}")
print(f"  Language Performance:   {json.dumps(lang_perf)}")
print(f"  Saved to:               {OUT_FILE} and {dashboard_err_path}")
print("="*60)
