"""Audits training examples for the four zero-recall canonical categories:
payment_failure, order_tracking, technical_issue, product_complaint.
"""

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TRAIN_FILE = Path("data/train/train.jsonl")
VAL_FILE = Path("data/validation/validation.jsonl")
TEST_FILE = Path("data/test/test.jsonl")

TARGET_CATEGORIES = ["payment_failure", "order_tracking", "technical_issue", "product_complaint"]


def audit():
    print("=" * 70)
    print("AUDIT OF WEAK CANONICAL CATEGORIES IN DATASET V1")
    print("=" * 70)

    for split_name, path in [("train", TRAIN_FILE), ("val", VAL_FILE), ("test", TEST_FILE)]:
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        
        print(f"\n--- Split: {split_name.upper()} (Total: {len(records)}) ---")
        for cat in TARGET_CATEGORIES:
            cat_records = [r for r in records if r["category"] == cat]
            langs = Counter(r["language"] for r in cat_records)
            prios = Counter(r["priority"] for r in cat_records)
            esc = Counter(r["requires_human"] for r in cat_records)
            print(f"Category: {cat} -> Count: {len(cat_records)}, Languages: {dict(langs)}, Priorities: {dict(prios)}, Escalation: {dict(esc)}")

    # Detailed inspection of train examples for targets
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]

    print("\n" + "=" * 70)
    print("SAMPLE TEXT AUDIT FOR TARGET CATEGORIES IN TRAIN")
    print("=" * 70)

    for cat in TARGET_CATEGORIES:
        sample_recs = [r for r in train_records if r["category"] == cat]
        print(f"\n>>> [{cat}] (Total in train: {len(sample_recs)})")
        for i, r in enumerate(sample_recs[:5]):
            print(f"  {i+1}. [{r['language']} / {r['priority']} / human={r['requires_human']}]")
            print(f"     \"{r['text']}\"")


if __name__ == "__main__":
    audit()
