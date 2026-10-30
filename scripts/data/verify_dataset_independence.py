"""Dataset Independence and Cross-Family Leakage Verification Script.

Inspects:
1. data/v2_clean_template_grouped/ (train_clean, val_clean, test_clean, dataset_v2_clean_lock.json)
2. Dataset V1 (train, validation, test)
3. Dataset V2 original (train, val, test_v2)
4. Checks SHA256 hashes, record counts, schemas, and label distributions.
5. Performs deep cross-family overlap investigation:
   - Exact string matches between V1 (train/val/test) and V2 (train/val/test)
   - Normalized template matches between V1 (val/test) and V2 train (data seen during experimental Qwen training)
   - Maximum Jaccard word-level overlap.

Emits experiments/artifacts/dataset_independence_verification.json.
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import TicketRecord
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


def compute_sha256(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def normalize_template(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r'[\$₹£€]\s*[\d\.,]+', '<AMOUNT>', t)
    t = re.sub(r'\b\d+\s*(?:rs|rupees|inr|dollars|usd)\b', '<AMOUNT>', t)
    t = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', t)
    t = re.sub(r'\b(?:order|ticket|tracking|txn|tx|ref|id)[\s#:-]*[a-z0-9_-]+\b', '<ORDER_ID>', t)
    t = re.sub(r'\b\d+\b', '<NUM>', t)
    t = re.sub(r'[^\w\s<>]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def word_jaccard(t1: str, t2: str) -> float:
    w1 = set(t1.lower().split())
    w2 = set(t2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def inspect_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Validate against TicketRecord schema
    valid_schema_count = 0
    schema_errors = []
    for idx, r in enumerate(records):
        try:
            TicketRecord(**r)
            valid_schema_count += 1
        except Exception as e:
            if len(schema_errors) < 5:
                schema_errors.append(f"Row {idx}: {e}")

    texts = [r["text"].strip() for r in records]
    templates = [normalize_template(r["text"]) for r in records]
    languages = Counter([r.get("language") for r in records])
    categories = Counter([r.get("category") for r in records])
    priorities = Counter([r.get("priority") for r in records])
    human_req = Counter([bool(r.get("requires_human")) for r in records])

    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": compute_sha256(path),
        "record_count": len(records),
        "valid_schema_count": valid_schema_count,
        "schema_compliant": valid_schema_count == len(records),
        "schema_errors_sample": schema_errors,
        "unique_raw_texts": len(set(texts)),
        "unique_templates": len(set(templates)),
        "language_distribution": dict(languages),
        "category_distribution": dict(categories),
        "priority_distribution": dict(priorities),
        "requires_human_distribution": {str(k): v for k, v in human_req.items()},
        "records": records,
        "texts": texts,
        "templates": templates,
    }


def compare_two_splits(s1: Dict[str, Any], s2: Dict[str, Any], name1: str, name2: str) -> Dict[str, Any]:
    s1_texts = set(s1["texts"])
    s2_texts = set(s2["texts"])
    exact_shared = s1_texts & s2_texts

    s1_tmpl = set(s1["templates"])
    s2_tmpl = set(s2["templates"])
    tmpl_shared = s1_tmpl & s2_tmpl

    # Find highest near-duplicate Jaccard pairs if no exact match
    max_jaccard = 0.0
    top_pair = None
    for t1 in s1["texts"][:100]:
        for t2 in s2["texts"][:100]:
            j = word_jaccard(t1, t2)
            if j > max_jaccard:
                max_jaccard = j
                top_pair = (t1[:60], t2[:60])

    s2_exact_count = sum(1 for t in s2["texts"] if t in s1_texts)
    s2_tmpl_count = sum(1 for t in s2["templates"] if t in s1_tmpl)

    return {
        "split_1": name1,
        "split_2": name2,
        "split_1_count": s1["record_count"],
        "split_2_count": s2["record_count"],
        "exact_duplicate_unique_strings": len(exact_shared),
        "split_2_records_with_exact_in_split_1": s2_exact_count,
        "split_2_exact_overlap_rate_pct": round(s2_exact_count / max(s2["record_count"], 1) * 100, 2),
        "template_duplicate_unique_patterns": len(tmpl_shared),
        "split_2_records_with_template_in_split_1": s2_tmpl_count,
        "split_2_template_overlap_rate_pct": round(s2_tmpl_count / max(s2["record_count"], 1) * 100, 2),
        "sample_max_jaccard_similarity": round(max_jaccard, 4),
        "sample_top_similar_pair": top_pair,
    }


def main():
    print("=" * 80)
    print("NEXUS-SLM: COMPREHENSIVE DATASET INDEPENDENCE & LEAKAGE VERIFICATION")
    print("=" * 80)

    files_to_inspect = {
        "v2_clean_train": PROJECT_ROOT / "data" / "v2_clean_template_grouped" / "train_clean.jsonl",
        "v2_clean_val": PROJECT_ROOT / "data" / "v2_clean_template_grouped" / "val_clean.jsonl",
        "v2_clean_test": PROJECT_ROOT / "data" / "v2_clean_template_grouped" / "test_clean.jsonl",
        "v1_train": PROJECT_ROOT / "data" / "train" / "train.jsonl",
        "v1_val": PROJECT_ROOT / "data" / "validation" / "validation.jsonl",
        "v1_test": PROJECT_ROOT / "data" / "test" / "test.jsonl",
        "v2_orig_train": PROJECT_ROOT / "data" / "v2" / "train.jsonl",
        "v2_orig_val": PROJECT_ROOT / "data" / "v2" / "val.jsonl",
        "v2_orig_test": PROJECT_ROOT / "data" / "v2" / "test_v2.jsonl",
        "challenge_set": PROJECT_ROOT / "data" / "evaluation" / "high_risk_challenge_set.jsonl",
    }

    inspections = {}
    for key, p in files_to_inspect.items():
        if p.exists():
            print(f"Inspecting: {p.relative_to(PROJECT_ROOT)}...")
            inspections[key] = inspect_file(p)
        else:
            print(f"MISSING: {p}")

    # Check lock file
    v2_clean_lock_path = PROJECT_ROOT / "data" / "v2_clean_template_grouped" / "dataset_v2_clean_lock.json"
    lock_data = json.loads(open(v2_clean_lock_path, "r", encoding="utf-8").read()) if v2_clean_lock_path.exists() else {}

    # Comparisons
    comparisons = []

    # 1. Clean V2 Internal Disjointness
    comparisons.append(compare_two_splits(inspections["v2_clean_train"], inspections["v2_clean_val"], "v2_clean_train", "v2_clean_val"))
    comparisons.append(compare_two_splits(inspections["v2_clean_train"], inspections["v2_clean_test"], "v2_clean_train", "v2_clean_test"))
    comparisons.append(compare_two_splits(inspections["v2_clean_val"], inspections["v2_clean_test"], "v2_clean_val", "v2_clean_test"))

    # 2. V1 Internal Disjointness
    comparisons.append(compare_two_splits(inspections["v1_train"], inspections["v1_val"], "v1_train", "v1_val"))
    comparisons.append(compare_two_splits(inspections["v1_train"], inspections["v1_test"], "v1_train", "v1_test"))
    comparisons.append(compare_two_splits(inspections["v1_val"], inspections["v1_test"], "v1_val", "v1_test"))

    # 3. Cross-Family Leakage: Did V2 Training See Dataset V1?
    comparisons.append(compare_two_splits(inspections["v2_orig_train"], inspections["v1_val"], "v2_orig_train (Qwen Training Data)", "v1_val (Held-out Benchmark)"))
    comparisons.append(compare_two_splits(inspections["v2_orig_train"], inspections["v1_test"], "v2_orig_train (Qwen Training Data)", "v1_test (Locked Benchmark)"))
    comparisons.append(compare_two_splits(inspections["v2_orig_train"], inspections["v1_train"], "v2_orig_train", "v1_train"))

    # Prepare output summary
    results_summary = {
        "audit_timestamp": "2026-10-13T16:15:00Z",
        "lock_file": {
            "path": str(v2_clean_lock_path.relative_to(PROJECT_ROOT)),
            "content": lock_data,
        },
        "file_inspections": {
            k: {k2: v2 for k2, v2 in v.items() if k2 not in ("records", "texts", "templates")}
            for k, v in inspections.items()
        },
        "comparisons": comparisons,
    }

    out_path = PROJECT_ROOT / "experiments" / "artifacts" / "dataset_independence_verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[Audit Saved] {out_path}")

    # Print Key Empirical Findings
    print("\n" + "=" * 80)
    print("KEY EMPIRICAL FINDINGS:")
    print("=" * 80)
    print("\n1. Clean V2 Template Grouping Verification:")
    for c in comparisons[:3]:
        print(f"   {c['split_1']} vs {c['split_2']}:")
        print(f"      Exact Overlap: {c['exact_duplicate_unique_strings']} ({c['split_2_exact_overlap_rate_pct']}%)")
        print(f"      Template Overlap: {c['template_duplicate_unique_patterns']} ({c['split_2_template_overlap_rate_pct']}%)")

    print("\n2. Cross-Family Overlap: Did Dataset V2 Training Data leak into Dataset V1?")
    for c in comparisons[6:]:
        print(f"   {c['split_1']} vs {c['split_2']}:")
        print(f"      Exact Strings Shared: {c['exact_duplicate_unique_strings']} ({c['split_2_exact_overlap_rate_pct']}%)")
        print(f"      Templates Shared: {c['template_duplicate_unique_patterns']} ({c['split_2_template_overlap_rate_pct']}%)")
        print(f"      Sample Top Jaccard: {c['sample_max_jaccard_similarity']}")


if __name__ == "__main__":
    main()
