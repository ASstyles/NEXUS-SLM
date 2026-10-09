"""Dataset V1 Detailed Audit Script.

Performs exhaustive audit of the V1 training dataset:
- Category distribution
- Language distribution (Devanagari Hindi, Roman Hindi, Hinglish, English)
- Priority distribution
- Escalation distribution
- Duplicate and near-duplicate analysis
- Conflicting label detection
- Length distribution (extremely short < 5 words, extremely long > 35 words)
- Rare categories identification (minimum count per category)
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_FILE = WORKSPACE_ROOT / "data" / "train" / "train.jsonl"
OUT_FILE = WORKSPACE_ROOT / "experiments" / "artifacts" / "dataset_v1_audit.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def jaccard_similarity(s1: str, s2: str) -> float:
    w1 = set(re.findall(r"\w+", s1.lower()))
    w2 = set(re.findall(r"\w+", s2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def is_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def run_audit():
    print("=" * 65)
    print("AUDITING NEXUS-SLM DATASET V1 (TRAINING SPLIT)")
    print("=" * 65)

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    print(f"Total training examples: {total}")

    # Category counts
    cat_counts = Counter(r["category"] for r in records)

    # Priority counts
    prio_counts = Counter(r["priority"] for r in records)

    # Escalation counts
    esc_counts = Counter(r["requires_human"] for r in records)

    # Language and script breakdown
    devanagari_count = 0
    roman_hindi_count = 0
    hinglish_count = 0
    english_count = 0
    raw_lang_counts = Counter(r["language"] for r in records)

    for r in records:
        text = r["text"]
        lang = r["language"].lower()
        has_dev = is_devanagari(text)

        if has_dev:
            devanagari_count += 1
        elif lang == "hindi":
            roman_hindi_count += 1
        elif lang == "hinglish":
            hinglish_count += 1
        else:
            english_count += 1

    total_hindi_related = devanagari_count + roman_hindi_count + hinglish_count

    # Length analysis
    short_examples = []
    long_examples = []
    word_lengths = []
    char_lengths = []

    for r in records:
        words = r["text"].split()
        n_words = len(words)
        n_chars = len(r["text"])
        word_lengths.append(n_words)
        char_lengths.append(n_chars)

        if n_words < 5:
            short_examples.append({
                "id": r["id"],
                "text": r["text"],
                "category": r["category"],
                "words": n_words,
            })
        elif n_words > 30:
            long_examples.append({
                "id": r["id"],
                "text": r["text"],
                "category": r["category"],
                "words": n_words,
            })

    # Exact Duplicates
    text_to_records = defaultdict(list)
    for r in records:
        text_norm = r["text"].strip().lower()
        text_to_records[text_norm].append(r)

    exact_duplicates = [recs for recs in text_to_records.values() if len(recs) > 1]
    exact_duplicate_count = sum(len(recs) - 1 for recs in exact_duplicates)

    # Conflicting Labels (same normalized text, different category or priority)
    conflicting_labels = []
    for text_norm, recs in text_to_records.items():
        if len(recs) > 1:
            cats = set(r["category"] for r in recs)
            prios = set(r["priority"] for r in recs)
            escs = set(r["requires_human"] for r in recs)
            if len(cats) > 1 or len(prios) > 1 or len(escs) > 1:
                conflicting_labels.append({
                    "text": recs[0]["text"],
                    "categories": list(cats),
                    "priorities": list(prios),
                    "escalations": list(escs),
                    "count": len(recs),
                })

    # Near Duplicates (pairwise Jaccard > 0.85 among non-exact duplicates)
    near_duplicates = []
    unique_texts = list(text_to_records.keys())
    for i in range(len(unique_texts)):
        for j in range(i + 1, min(i + 50, len(unique_texts))):
            s1 = unique_texts[i]
            s2 = unique_texts[j]
            sim = jaccard_similarity(s1, s2)
            if sim > 0.85:
                near_duplicates.append({
                    "text_1": text_to_records[s1][0]["text"],
                    "text_2": text_to_records[s2][0]["text"],
                    "category_1": text_to_records[s1][0]["category"],
                    "category_2": text_to_records[s2][0]["category"],
                    "similarity": round(sim, 3),
                })

    # Rare Categories Identification
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1])
    min_category_name, min_category_count = sorted_cats[0]
    max_category_name, max_category_count = sorted_cats[-1]

    rare_threshold = 20  # categories with fewer than 20 examples
    rare_categories = {k: v for k, v in sorted_cats if v < rare_threshold}

    audit_report = {
        "dataset_version": "dataset_v1",
        "split": "train",
        "total_examples": total,
        "category_distribution": dict(cat_counts),
        "priority_distribution": dict(prio_counts),
        "escalation_distribution": {str(k): v for k, v in esc_counts.items()},
        "raw_language_distribution": dict(raw_lang_counts),
        "multilingual_granularity": {
            "english": english_count,
            "hinglish": hinglish_count,
            "roman_hindi": roman_hindi_count,
            "devanagari_hindi": devanagari_count,
            "total_hindi_related": total_hindi_related,
        },
        "length_statistics": {
            "mean_words": round(sum(word_lengths) / total, 1),
            "min_words": min(word_lengths),
            "max_words": max(word_lengths),
            "mean_chars": round(sum(char_lengths) / total, 1),
            "extremely_short_count_under_5_words": len(short_examples),
            "extremely_long_count_over_30_words": len(long_examples),
        },
        "data_quality_issues": {
            "exact_duplicate_count": exact_duplicate_count,
            "near_duplicate_count": len(near_duplicates),
            "conflicting_label_count": len(conflicting_labels),
            "conflicting_cases": conflicting_labels,
        },
        "rare_categories": {
            "min_category": {"name": min_category_name, "count": min_category_count},
            "max_category": {"name": max_category_name, "count": max_category_count},
            "imbalance_ratio": round(max_category_count / min_category_count, 2),
            "underrepresented_categories_under_20_samples": rare_categories,
        },
        "sample_short_examples": short_examples[:5],
        "sample_near_duplicates": near_duplicates[:3],
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print("\n--- AUDIT SUMMARY ---")
    print(f"Total Examples: {total}")
    print(f"Categories ({len(cat_counts)}): {sorted_cats[:3]} ... {sorted_cats[-2:]}")
    print(f"Minimum Category Count: {min_category_name} ({min_category_count} samples)")
    print(f"Imbalance Ratio: {audit_report['rare_categories']['imbalance_ratio']}x")
    print(f"Multilingual Breakdown: English={english_count}, Hinglish={hinglish_count}, Roman Hindi={roman_hindi_count}, Devanagari Hindi={devanagari_count}")
    print(f"Extremely Short (<5 words): {len(short_examples)}")
    print(f"Extremely Long (>30 words): {len(long_examples)}")
    print(f"Exact Duplicates: {exact_duplicate_count}")
    print(f"Near Duplicates (Jaccard > 0.85): {len(near_duplicates)}")
    print(f"Conflicting Label Cases: {len(conflicting_labels)}")
    print(f"\nAudit report saved to: {OUT_FILE}")
    return audit_report


if __name__ == "__main__":
    run_audit()
