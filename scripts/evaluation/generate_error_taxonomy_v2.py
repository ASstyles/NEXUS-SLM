"""Generate deep Workstream 3.1 error taxonomy from true-base Qwen2.5-0.5B predictions."""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


def main():
    preds_path = ROOT / "experiments/artifacts/qwen25_05b_clean_v2_true_base_predictions.json"
    summary_path = ROOT / "experiments/results/eval_qwen_clean_v2_true_base.json"
    stacked_path = ROOT / "experiments/artifacts/qwen25_05b_clean_v2_lora_test_results.json"

    preds = json.loads(preds_path.read_text(encoding="utf-8"))["predictions"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stacked = json.loads(stacked_path.read_text(encoding="utf-8"))

    # Category confusion pairs
    conf_pairs = Counter()
    by_gt_cat = defaultdict(Counter)
    examples_by_pair = defaultdict(list)

    prio_conf = Counter()
    prio_examples = defaultdict(list)

    lang_errors = defaultdict(Counter)
    lang_examples = defaultdict(list)

    for r in preds:
        gc, pc = r["gt_category"], r["raw_category"]
        gp, pp = r["gt_priority"], r["raw_priority"]
        gl, pl = r["gt_language"], r["raw_language"]

        by_gt_cat[gc][pc] += 1
        if gc != pc:
            conf_pairs[(gc, pc)] += 1
            if len(examples_by_pair[(gc, pc)]) < 3:
                examples_by_pair[(gc, pc)].append({
                    "ticket_id": r["ticket_id"],
                    "language": gl,
                    "gt_category": gc,
                    "pred_category": pc,
                    "gt_priority": gp,
                    "pred_priority": pp,
                    "text": r["text"],
                    "raw_output": r["raw_output"],
                })

        if gp != pp:
            prio_conf[(gp, pp)] += 1
            if len(prio_examples[(gp, pp)]) < 2:
                prio_examples[(gp, pp)].append({
                    "ticket_id": r["ticket_id"],
                    "language": gl,
                    "category": gc,
                    "gt_priority": gp,
                    "pred_priority": pp,
                    "text": r["text"],
                    "raw_output": r["raw_output"],
                })

        if gc != pc or gp != pp or gl != pl:
            if gc != pc:
                lang_errors[gl][f"{gc}->{pc}"] += 1
            if gl != pl:
                lang_errors[f"lang_id:{gl}"][pl] += 1
            if len(lang_examples[gl]) < 4 and gc != pc:
                lang_examples[gl].append({
                    "ticket_id": r["ticket_id"],
                    "gt": f"{gc}/{gp}/{gl}",
                    "pred": f"{pc}/{pp}/{pl}",
                    "text": r["text"],
                    "raw_output": r["raw_output"],
                })

    focus_classes = [
        "general_query",
        "product_complaint",
        "refund_request",
        "account_access",
        "payment_failure",
        "subscription",
        "fraud_report",
        "order_cancel",
    ]

    focus_breakdown = {}
    raw_per_class = summary["raw_model_metrics"]["category"]["per_class"]
    stacked_per_cat = stacked["metrics"]["category"]["per_category"]
    for c in focus_classes:
        focus_breakdown[c] = {
            "true_base_qwen": raw_per_class[c],
            "stacked_scratch_qwen": {
                "precision": stacked_per_cat[c]["precision"],
                "recall": stacked_per_cat[c]["recall"],
                "f1": stacked_per_cat[c]["f1"],
                "support": int(stacked_per_cat[c]["support"]),
            },
            "true_base_predicted_distribution": dict(by_gt_cat[c].most_common()),
        }

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "data/v2_clean_template_grouped/test_clean.jsonl",
        "sample_count": len(preds),
        "true_base_overall": {
            "category_accuracy": summary["raw_model_metrics"]["category"]["accuracy"],
            "category_macro_f1": summary["raw_model_metrics"]["category"]["macro_f1"],
            "priority_accuracy": summary["raw_model_metrics"]["priority"]["accuracy"],
            "priority_macro_f1": summary["raw_model_metrics"]["priority"]["macro_f1"],
            "language_accuracy": summary["raw_model_metrics"]["language"]["accuracy"],
            "json_validity_rate": summary["json_validity_rate"],
        },
        "stacked_scratch_overall": {
            "category_accuracy": stacked["metrics"]["category"]["accuracy"],
            "category_macro_f1": stacked["metrics"]["category"]["macro_f1"],
            "priority_accuracy": stacked["metrics"]["priority"]["accuracy"],
            "priority_macro_f1": stacked["metrics"]["priority"]["macro_f1"],
            "language_accuracy": stacked["metrics"]["language"]["overall_accuracy"],
        },
        "focus_class_comparison": focus_breakdown,
        "true_base_per_class_all_14": raw_per_class,
        "true_base_priority_per_class": summary["raw_model_metrics"]["priority"]["per_class"],
        "true_base_language_breakdown": summary["language_breakdown"],
        "top_category_confusions": [
            {
                "gt_category": gc,
                "pred_category": pc,
                "count": cnt,
                "examples": examples_by_pair[(gc, pc)],
            }
            for (gc, pc), cnt in conf_pairs.most_common(15)
        ],
        "top_priority_confusions": [
            {
                "gt_priority": gp,
                "pred_priority": pp,
                "count": cnt,
                "examples": prio_examples[(gp, pp)],
            }
            for (gp, pp), cnt in prio_conf.most_common(10)
        ],
        "language_error_patterns": {k: dict(v.most_common(8)) for k, v in lang_errors.items()},
        "language_failure_examples": dict(lang_examples),
    }

    out_path = ROOT / "experiments/artifacts/neural_error_taxonomy_v2.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== FOCUS CLASSES (True Base vs Stacked) ===")
    for c, v in focus_breakdown.items():
        tb = v["true_base_qwen"]
        st = v["stacked_scratch_qwen"]
        print(f"  {c:18s} sup={tb['support']:3d} | TrueBase F1={tb['f1']:.4f} (P={tb['precision']:.3f}, R={tb['recall']:.3f}) | Stacked F1={st['f1']:.4f} (P={st['precision']:.3f}, R={st['recall']:.3f}) | preds={v['true_base_predicted_distribution']}")
    print("\n=== PRIORITY PER CLASS (True Base) ===")
    for p, v in out["true_base_priority_per_class"].items():
        print(f"  {p:10s} sup={v['support']:3d} P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f}")
    print("\n=== BY LANGUAGE (True Base) ===")
    for l, v in out["true_base_language_breakdown"].items():
        print(f"  {l:10s} N={v['count']:3d} cat_acc={v['raw_category_accuracy']:.4f} cat_f1={v['raw_category_macro_f1']:.4f} prio_acc={v['raw_priority_accuracy']:.4f}")


if __name__ == "__main__":
    main()
