"""Evaluate Hardened Policy V3 (rules.py + policy.py) into versioned snapshots without altering V2 archives."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import EvaluationMetrics
from src.inference.policy import HumanEscalationPolicy
from src.inference.rules import get_fallback_prediction
from src.utils.constants import CATEGORIES, LANGUAGES, PRIORITIES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_policy_v3_dataset(path: Path, dataset_key: str, description: str) -> Dict[str, Any]:
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if any("adjudication" in r for r in records):
        records = [
            r
            for r in records
            if r.get("adjudication", {}).get("final_category") in CATEGORIES
            and r.get("adjudication", {}).get("status") != "PENDING"
        ]

    gt_cats, pred_cats = [], []
    gt_prios, pred_prios = [], []
    gt_langs, pred_langs = [], []
    gt_hums, pred_hums = [], []
    latencies_ms = []

    for rec in records:
        txt = rec.get("text") or rec.get("redacted_text") or rec.get("ticket") or ""
        t0 = time.perf_counter()
        fb = get_fallback_prediction(txt)
        req_h, _, _, final_cat, final_prio = HumanEscalationPolicy.evaluate_escalation(
            text=txt,
            category=fb["category"],
            priority=fb["priority"],
            language=fb["language"],
            raw_confidence=None,
            is_parse_error=False,
            inference_mode="heuristic_fallback",
            model_requires_human=fb.get("requires_human"),
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        adj = rec.get("adjudication") or {}
        gt_node = rec.get("ground_truth") or rec.get("human_annotation") or {}
        gt_cat = rec.get("category") or adj.get("final_category") or gt_node.get("category")
        gt_prio = rec.get("priority") or adj.get("final_priority") or gt_node.get("priority")
        gt_lang = rec.get("language") or gt_node.get("language")
        gt_hum = rec.get("requires_human")
        if gt_hum is None:
            gt_hum = adj.get("final_requires_human") if "final_requires_human" in adj else gt_node.get("requires_human")

        gt_cats.append(gt_cat)
        pred_cats.append(final_cat)
        gt_prios.append(gt_prio)
        pred_prios.append(final_prio)
        gt_langs.append(gt_lang)
        pred_langs.append(fb["language"])
        gt_hums.append(bool(gt_hum))
        pred_hums.append(bool(req_h))

    cat_m = EvaluationMetrics.calculate_classification_metrics(gt_cats, pred_cats, labels=CATEGORIES)
    prio_m = EvaluationMetrics.calculate_classification_metrics(gt_prios, pred_prios, labels=PRIORITIES)
    lang_m = EvaluationMetrics.calculate_classification_metrics(gt_langs, pred_langs, labels=LANGUAGES)

    tp = sum(1 for y, p in zip(gt_hums, pred_hums) if y and p)
    fn = sum(1 for y, p in zip(gt_hums, pred_hums) if y and not p)
    fp = sum(1 for y, p in zip(gt_hums, pred_hums) if not y and p)
    tn = sum(1 for y, p in zip(gt_hums, pred_hums) if not y and not p)
    crit_tp = sum(1 for y, p, pr in zip(gt_hums, pred_hums, gt_prios) if pr == "critical" and p)
    crit_fn = sum(1 for y, p, pr in zip(gt_hums, pred_hums, gt_prios) if pr == "critical" and not p)
    exact_4 = sum(
        1
        for gc, pc, gp, pp, gl, pl, gh, ph in zip(
            gt_cats, pred_cats, gt_prios, pred_prios, gt_langs, pred_langs, gt_hums, pred_hums
        )
        if (gc == pc and gp == pp and gl == pl and gh == ph)
    )

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "metadata": {
            "dataset_key": dataset_key,
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "policy_version": "hardened_policy_v3",
            "engine": "heuristic_fallback_v3_multilingual_hardened",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "description": description,
        },
        "sample_count": len(records),
        "category_metrics": {
            "accuracy": cat_m["accuracy"],
            "macro_precision": cat_m["macro_precision"],
            "macro_recall": cat_m["macro_recall"],
            "macro_f1": cat_m["macro_f1"],
            "weighted_f1": cat_m["weighted_f1"],
            "per_class": cat_m["per_class"],
        },
        "priority_metrics": {
            "accuracy": prio_m["accuracy"],
            "macro_f1": prio_m["macro_f1"],
            "weighted_f1": prio_m["weighted_f1"],
            "per_class_recall": {p: prio_m["per_class"][p]["recall"] for p in PRIORITIES},
        },
        "language_metrics": {
            "overall_accuracy": lang_m["accuracy"],
            "breakdown": {
                l: {"support": int(lang_m["per_class"][l]["support"]), "accuracy": lang_m["per_class"][l]["recall"]}
                for l in LANGUAGES
            },
        },
        "escalation_metrics": {
            "overall_escalation_rate": round((tp + fp) / len(records), 4),
            "unnecessary_escalation_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
            "high_risk_recall": round(tp / (tp + fn), 4) if (tp + fn) else 1.0,
            "high_risk_false_negatives": fn,
        },
        "critical_safety_metrics": {
            "critical_recall": round(crit_tp / (crit_tp + crit_fn), 4) if (crit_tp + crit_fn) else 1.0,
            "critical_false_negatives": crit_fn,
            "safety_violation_count": crit_fn,
        },
        "composite_metrics": {
            "exact_4field_match_rate": round(exact_4 / len(records), 4),
        },
        "latency_ms": {
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
        },
    }


def main():
    syn = evaluate_policy_v3_dataset(
        ROOT / "data/v2_clean_template_grouped/test_clean.jsonl",
        "v2_clean_template_grouped_test",
        "Hardened Policy V3 evaluated on locked synthetic test set (N=746)",
    )
    hum = evaluate_policy_v3_dataset(
        ROOT / "data/human_eval/annotation_template.jsonl",
        "human_eval_v1_dev_exposed",
        "Hardened Policy V3 evaluated on development-exposed V1 human scenarios (N=184)",
    )
    out_syn = ROOT / "experiments/results/eval_synthetic_test_v3_hardened.json"
    out_hum = ROOT / "experiments/results/eval_human_holdout_v3_hardened.json"
    out_syn.write_text(json.dumps(syn, indent=2), encoding="utf-8")
    out_hum.write_text(json.dumps(hum, indent=2), encoding="utf-8")
    print(json.dumps({"synthetic_v3": {k: v for k, v in syn.items() if k != "category_metrics"}, "synthetic_cat": {k: v for k, v in syn["category_metrics"].items() if k != "per_class"}, "human_v3_summary": {"cat_acc": hum["category_metrics"]["accuracy"], "cat_f1": hum["category_metrics"]["macro_f1"], "lang_acc": hum["language_metrics"]["overall_accuracy"], "esc_rate": hum["escalation_metrics"]["overall_escalation_rate"]}}, indent=2))


if __name__ == "__main__":
    main()
