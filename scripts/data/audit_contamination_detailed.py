"""Detailed dataset contamination and template leakage audit script for NEXUS-SLM.
Analyzes exact duplicates, near duplicates, and template overlap across V1, V2, and Challenge sets.
Also creates a versioned, template-grouped clean split under data/v2_clean_template_grouped/.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normalize_text_template(text: str) -> str:
    """Normalizes a ticket into a template skeleton by masking numbers, amounts, IDs, and punctuation."""
    t = text.lower().strip()
    # Mask currency amounts
    t = re.sub(r'[\$₹£€]\s*[\d\.,]+', '<AMOUNT>', t)
    t = re.sub(r'\b\d+\s*(?:rs|rupees|inr|dollars|usd)\b', '<AMOUNT>', t)
    # Mask emails
    t = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', t)
    # Mask order/ticket/tracking IDs
    t = re.sub(r'\b(?:order|ticket|tracking|txn|tx|ref|id)[\s#:-]*[a-z0-9_-]+\b', '<ORDER_ID>', t)
    # Mask isolated numeric sequences
    t = re.sub(r'\b\d+\b', '<NUM>', t)
    # Strip punctuation but keep mask tags
    t = re.sub(r'[^\w\s<>]', ' ', t)
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def compute_jaccard_similarity(s1: str, s2: str) -> float:
    w1 = set(s1.split())
    w2 = set(s2.split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def audit_dataset_family(family_name: str, split_paths: Dict[str, Path]) -> Dict[str, Any]:
    splits = {}
    for name, p in split_paths.items():
        with open(p, 'r', encoding='utf-8') as f:
            records = [json.loads(l) for l in f if l.strip()]
        splits[name] = records

    report: Dict[str, Any] = {
        "family": family_name,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "splits": {},
        "cross_split_leakage": {},
    }

    # Per split distribution
    for name, records in splits.items():
        exact_texts = [r["text"].strip() for r in records]
        templates = [normalize_text_template(r["text"]) for r in records]
        languages = Counter([r.get("language") for r in records])
        categories = Counter([r.get("category") for r in records])

        unique_texts = len(set(exact_texts))
        unique_templates = len(set(templates))

        report["splits"][name] = {
            "record_count": len(records),
            "unique_raw_texts": unique_texts,
            "unique_templates": unique_templates,
            "internal_duplicate_text_count": len(records) - unique_texts,
            "internal_duplicate_template_count": len(records) - unique_templates,
            "language_distribution": dict(languages),
            "category_distribution": dict(categories),
        }

    # Cross split analysis
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1_name = split_names[i]
            s2_name = split_names[j]
            s1_recs = splits[s1_name]
            s2_recs = splits[s2_name]

            s1_texts = set(r["text"].strip() for r in s1_recs)
            s2_texts = set(r["text"].strip() for r in s2_recs)
            exact_overlap = s1_texts & s2_texts

            s1_tmpl = set(normalize_text_template(r["text"]) for r in s1_recs)
            s2_tmpl = set(normalize_text_template(r["text"]) for r in s2_recs)
            template_overlap = s1_tmpl & s2_tmpl

            # Count records in s2 matching s1
            s2_records_with_exact_in_s1 = sum(1 for r in s2_recs if r["text"].strip() in s1_texts)
            s2_records_with_tmpl_in_s1 = sum(1 for r in s2_recs if normalize_text_template(r["text"]) in s1_tmpl)

            pair_key = f"{s1_name}_vs_{s2_name}"
            report["cross_split_leakage"][pair_key] = {
                f"{s1_name}_count": len(s1_recs),
                f"{s2_name}_count": len(s2_recs),
                "exact_duplicate_unique_strings": len(exact_overlap),
                f"{s2_name}_records_exactly_in_{s1_name}": s2_records_with_exact_in_s1,
                f"{s2_name}_exact_overlap_rate_pct": round(s2_records_with_exact_in_s1 / max(len(s2_recs), 1) * 100, 2),
                "template_duplicate_unique_patterns": len(template_overlap),
                f"{s2_name}_records_with_template_in_{s1_name}": s2_records_with_tmpl_in_s1,
                f"{s2_name}_template_overlap_rate_pct": round(s2_records_with_tmpl_in_s1 / max(len(s2_recs), 1) * 100, 2),
            }

    return report


def create_clean_template_grouped_dataset_v2():
    """Groups all V2 records by template skeleton and produces clean disjoint train/val/test splits."""
    v2_dir = PROJECT_ROOT / "data" / "v2"
    all_recs = []
    seen_ids = set()

    for p in [v2_dir / "train.jsonl", v2_dir / "val.jsonl", v2_dir / "test_v2.jsonl"]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        # Normalize language field 'roman_hindi' -> 'hinglish'
                        if r.get("language") == "roman_hindi":
                            r["language"] = "hinglish"
                        rid = r.get("id")
                        if rid not in seen_ids:
                            seen_ids.add(rid)
                            all_recs.append(r)

    # Group records by template
    template_groups = defaultdict(list)
    for r in all_recs:
        tmpl = normalize_text_template(r["text"])
        template_groups[tmpl].append(r)

    print(f"\n[Template Grouping V2] Total unique records: {len(all_recs)}, Total unique template clusters: {len(template_groups)}")

    # Deterministic split by template clusters (80% train, 10% val, 10% test)
    # Sort clusters by template hash for reproducibility
    sorted_templates = sorted(template_groups.keys(), key=lambda t: (len(template_groups[t]), t))

    # Greedy allocation to maintain balanced splits
    target_train = int(len(all_recs) * 0.80)
    target_val = int(len(all_recs) * 0.10)

    train_recs = []
    val_recs = []
    test_recs = []

    import hashlib
    for tmpl in sorted_templates:
        cluster = template_groups[tmpl]
        h = int(hashlib.md5(tmpl.encode("utf-8")).hexdigest(), 16) % 100
        if h < 80:
            train_recs.extend(cluster)
        elif h < 90:
            val_recs.extend(cluster)
        else:
            test_recs.extend(cluster)

    out_dir = PROJECT_ROOT / "data" / "v2_clean_template_grouped"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train_clean.jsonl", "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "val_clean.jsonl", "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "test_clean.jsonl", "w", encoding="utf-8") as f:
        for r in test_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "dataset_name": "dataset_v2_clean_template_grouped",
        "created_at": "2026-10-13T16:00:00Z",
        "source": "Re-clustered from data/v2/ with strict template disjointness",
        "total_records": len(all_recs),
        "total_template_clusters": len(template_groups),
        "split_counts": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
        },
        "guarantee": "Zero template overlap across train, val, and test splits (100% disjoint clusters)."
    }
    with open(out_dir / "dataset_v2_clean_lock.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Created Clean Split] Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    print(f"[Artifact Saved] {out_dir}")
    return meta


def main():
    print("=" * 75)
    print("NEXUS-SLM: DATASET CONTAMINATION & TEMPLATE LEAKAGE AUDIT")
    print("=" * 75)

    v1_paths = {
        "train": PROJECT_ROOT / "data" / "train" / "train.jsonl",
        "validation": PROJECT_ROOT / "data" / "validation" / "validation.jsonl",
        "test": PROJECT_ROOT / "data" / "test" / "test.jsonl",
    }
    v2_paths = {
        "train": PROJECT_ROOT / "data" / "v2" / "train.jsonl",
        "validation": PROJECT_ROOT / "data" / "v2" / "val.jsonl",
        "test": PROJECT_ROOT / "data" / "v2" / "test_v2.jsonl",
    }

    v1_audit = audit_dataset_family("Dataset V1", v1_paths)
    v2_audit = audit_dataset_family("Dataset V2", v2_paths)

    full_report = {
        "audit_timestamp": "2026-10-13T16:00:00Z",
        "dataset_v1": v1_audit,
        "dataset_v2": v2_audit,
    }

    out_file = PROJECT_ROOT / "experiments" / "artifacts" / "comprehensive_dataset_contamination_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[Audit Report Saved] {out_file}")

    # Generate the clean template-grouped split
    clean_meta = create_clean_template_grouped_dataset_v2()

    # Print summary findings
    print("\n" + "=" * 75)
    print("SUMMARY AUDIT FINDINGS:")
    print("=" * 75)
    print("1. DATASET V1:")
    for pair, leak in v1_audit["cross_split_leakage"].items():
        print(f"   {pair}: exact duplicate unique strings = {leak['exact_duplicate_unique_strings']}, template overlap = {leak['template_duplicate_unique_patterns']}")
    print("\n2. DATASET V2:")
    for pair, leak in v2_audit["cross_split_leakage"].items():
        print(f"   {pair}:")
        print(f"      Exact duplicate strings: {leak['exact_duplicate_unique_strings']}")
        print(f"      Overlap rate: {leak[list(leak.keys())[4]]}% exact, {leak[list(leak.keys())[7]]}% template")


if __name__ == "__main__":
    main()
