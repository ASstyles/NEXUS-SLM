"""Comprehensive quantitative audit of dataset quality, template overlap, and sparsity."""

import json
import os
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

SPLITS = {
    "train": ROOT / "data/v2_clean_template_grouped/train_clean.jsonl",
    "val": ROOT / "data/v2_clean_template_grouped/val_clean.jsonl",
    "test": ROOT / "data/v2_clean_template_grouped/test_clean.jsonl",
}


def normalize_text_template(text: str) -> str:
    """Masks dynamic entity slots (Order IDs, amounts, cities, phone numbers)."""
    t = text.strip().lower()
    # Mask order IDs: ORD-123456 or #123456
    t = re.sub(r"\b(?:ord-?\d+|\#\d+)\b", "<ORDER_ID>", t)
    # Mask monetary amounts: Rs 500, ₹500, $500, 500 rupees, 500 rs
    t = re.sub(r"[\$₹]\s*\d+(?:\.\d+)?|\b\d+\s*(?:rupees|rs|inr|usd)\b", "<AMOUNT>", t)
    # Mask phone / tracking numbers / dates
    t = re.sub(r"\b\d{10,12}\b", "<PHONE>", t)
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "<DATE>", t)
    # Mask common cities
    cities = [
        "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata",
        "pune", "ahmedabad", "jaipur", "surat", "lucknow", "kanpur", "nagpur", "indore",
        "bhopal", "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik", "varanasi"
    ]
    t = re.sub(r"\b(" + "|".join(cities) + r")\b", "<CITY>", t)
    # Normalize whitespaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_split(path: pathlib.Path) -> List[Dict[str, Any]]:
    assert path.exists(), f"Split file {path} not found"
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def audit_datasets() -> Dict[str, Any]:
    split_data = {}
    for name, p in SPLITS.items():
        split_data[name] = load_split(p)

    report: Dict[str, Any] = {
        "audit_name": "clean_v2_dataset_quality_and_generalisation_audit",
        "splits": {},
        "cross_split_leakage_and_overlap": {},
    }

    templates_by_split: Dict[str, Set[str]] = {}
    raw_texts_by_split: Dict[str, Set[str]] = {}

    for name, records in split_data.items():
        total_count = len(records)
        raw_texts = [r["text"].strip() for r in records]
        unique_raw = len(set(raw_texts))
        exact_duplicate_count = total_count - unique_raw

        norm_templates = [normalize_text_template(t) for t in raw_texts]
        unique_templates = len(set(norm_templates))
        template_duplicate_count = total_count - unique_templates

        templates_by_split[name] = set(norm_templates)
        raw_texts_by_split[name] = set(raw_texts)

        cat_counts = Counter(r["category"] for r in records)
        prio_counts = Counter(r["priority"] for r in records)
        lang_counts = Counter(r["language"] for r in records)
        esc_counts = Counter(bool(r.get("requires_human", False)) for r in records)

        lengths = [len(t.split()) for t in raw_texts]
        char_lengths = [len(t) for t in raw_texts]

        report["splits"][name] = {
            "sample_count": total_count,
            "unique_raw_texts": unique_raw,
            "exact_raw_duplicates": exact_duplicate_count,
            "unique_normalized_templates": unique_templates,
            "template_duplication_ratio": round(1.0 - (unique_templates / total_count), 4),
            "category_distribution": dict(cat_counts),
            "priority_distribution": dict(prio_counts),
            "language_distribution": dict(lang_counts),
            "requires_human_distribution": dict(esc_counts),
            "text_length_words": {
                "mean": round(sum(lengths) / len(lengths), 2),
                "min": min(lengths),
                "max": max(lengths),
            },
            "text_length_chars": {
                "mean": round(sum(char_lengths) / len(char_lengths), 2),
                "min": min(char_lengths),
                "max": max(char_lengths),
            },
            "sparse_categories": [cat for cat, cnt in cat_counts.items() if cnt <= 10],
        }

    # Cross-split overlap analysis
    train_templates = templates_by_split["train"]
    test_templates = templates_by_split["test"]
    val_templates = templates_by_split["val"]

    train_test_overlap = len(train_templates.intersection(test_templates))
    train_val_overlap = len(train_templates.intersection(val_templates))
    exact_train_test_leakage = len(raw_texts_by_split["train"].intersection(raw_texts_by_split["test"]))

    report["cross_split_leakage_and_overlap"] = {
        "exact_raw_train_test_leakage_count": exact_train_test_leakage,
        "template_train_test_overlap_count": train_test_overlap,
        "template_train_test_overlap_percentage": round((train_test_overlap / len(test_templates)) * 100, 2),
        "template_train_val_overlap_count": train_val_overlap,
        "template_train_val_overlap_percentage": round((train_val_overlap / len(val_templates)) * 100, 2),
        "findings": [
            f"Exact string leakage: {exact_train_test_leakage} instances.",
            f"Normalized template overlap: {train_test_overlap}/{len(test_templates)} test templates ({round((train_test_overlap / len(test_templates)) * 100, 1)}%) share identical sentence skeletons with training data after masking entity slots.",
            f"Category Sparsity: Test set contains only {split_data['test']} 'subscription' instances.",
            "Synthetic Bias: Slot substitution (varying order IDs, amounts, and cities) creates high nominal dataset size but limited lexical variance.",
        ],
    }

    out_file = ROOT / "experiments/artifacts/dataset_quality_audit_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDataset Audit Report:")
    for s_name, s_info in report["splits"].items():
        print(f"  {s_name.upper()}: {s_info['sample_count']} samples, {s_info['unique_normalized_templates']} unique templates, sparse: {s_info['sparse_categories']}")
    print(f"  Template Train-Test Overlap: {train_test_overlap}/{len(test_templates)} ({round((train_test_overlap / len(test_templates)) * 100, 1)}%)")
    print(f"  Audit report written to {out_file}")

    return report


if __name__ == "__main__":
    audit_datasets()
