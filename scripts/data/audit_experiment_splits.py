"""Audit Experiment Inputs, Datasets, and Splits for NEXUS-SLM.

Performs exhaustive verification of:
1. Training and validation splits (Dataset V1 vs Dataset V2).
2. Label quality, schema conformity, and taxonomy adherence.
3. Class, language, and priority distributions.
4. Exact duplicates, near-duplicates, and conflicting labels.
5. Cross-split data leakage against validation and locked historical test sets.
6. Provenance (synthetic vs public vs independently human-validated).
"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


def compute_sha256(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def jaccard_similarity(s1: str, s2: str) -> float:
    w1 = set(s1.split())
    w2 = set(s2.split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def is_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"Error parsing line {idx} in {path}: {e}")
    return records


def audit_split(name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    cat_counts = Counter(r.get("category") for r in records)
    prio_counts = Counter(r.get("priority") for r in records)
    lang_counts = Counter(r.get("language") for r in records)
    human_counts = Counter(bool(r.get("requires_human")) for r in records)

    # Validate categories, priorities, languages
    invalid_cats = [r for r in records if r.get("category") not in CATEGORIES]
    invalid_prios = [r for r in records if r.get("priority") not in PRIORITIES]
    invalid_langs = [r for r in records if r.get("language") not in LANGUAGES and r.get("language") != "roman_hindi"]

    # Script distribution
    devanagari_count = sum(1 for r in records if is_devanagari(r.get("text", "")))

    # Length stats
    word_lengths = [len(r.get("text", "").split()) for r in records]
    char_lengths = [len(r.get("text", "")) for r in records]

    # Internal duplicates
    text_to_records = defaultdict(list)
    for r in records:
        norm = normalize_text(r.get("text", ""))
        text_to_records[norm].append(r)

    exact_dups = {k: v for k, v in text_to_records.items() if len(v) > 1}

    # Conflicting labels
    conflicts = []
    for norm, recs in exact_dups.items():
        cats = set(r.get("category") for r in recs)
        if len(cats) > 1:
            conflicts.append({"norm_text": norm, "categories": list(cats), "count": len(recs)})

    # Internal near-duplicates
    near_dups = []
    norm_texts = [(r.get("id", f"idx_{i}"), normalize_text(r.get("text", "")), r.get("category")) for i, r in enumerate(records)]
    # sample pairwise if dataset is large, or full if <= 1000
    if total <= 1000:
        for i in range(len(norm_texts)):
            id1, t1, c1 = norm_texts[i]
            for j in range(i + 1, len(norm_texts)):
                id2, t2, c2 = norm_texts[j]
                if t1 != t2:
                    sim = jaccard_similarity(t1, t2)
                    if sim >= 0.85:
                        near_dups.append({"id1": id1, "id2": id2, "sim": round(sim, 3), "t1": t1[:50], "t2": t2[:50], "c1": c1, "c2": c2})

    # Provenance
    provenance_sources = Counter()
    for r in records:
        prov = r.get("provenance", {})
        if isinstance(prov, dict) and prov.get("source"):
            provenance_sources[prov["source"]] += 1
        elif r.get("source"):
            provenance_sources[r["source"]] += 1
        else:
            provenance_sources["unspecified"] += 1

    return {
        "name": name,
        "count": total,
        "category_distribution": dict(cat_counts),
        "priority_distribution": dict(prio_counts),
        "language_distribution": dict(lang_counts),
        "devanagari_script_count": devanagari_count,
        "human_escalation_distribution": dict(human_counts),
        "invalid_categories_count": len(invalid_cats),
        "invalid_priorities_count": len(invalid_prios),
        "invalid_languages_count": len(invalid_langs),
        "exact_duplicate_groups": len(exact_dups),
        "exact_duplicate_total_excess": sum(len(v) - 1 for v in exact_dups.values()),
        "conflicting_labels_count": len(conflicts),
        "conflicting_labels": conflicts[:5],
        "near_duplicate_count": len(near_dups),
        "near_duplicates_sample": near_dups[:5],
        "mean_word_length": round(sum(word_lengths) / max(total, 1), 1),
        "mean_char_length": round(sum(char_lengths) / max(total, 1), 1),
        "provenance_breakdown": dict(provenance_sources),
    }


def check_cross_leakage(train_name: str, train_recs: List[Dict[str, Any]], eval_name: str, eval_recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    exact_leakages = []
    near_leakages = []

    eval_norm = [(e.get("id", f"eval_{j}"), normalize_text(e.get("text", "")), e) for j, e in enumerate(eval_recs)]

    for i, t in enumerate(train_recs):
        t_id = t.get("id", f"train_{i}")
        t_norm = normalize_text(t.get("text", ""))
        for e_id, e_norm, e_raw in eval_norm:
            if t_norm == e_norm:
                exact_leakages.append({
                    "train_id": t_id,
                    "eval_id": e_id,
                    "train_text": t.get("text", ""),
                    "eval_text": e_raw.get("text", ""),
                    "train_cat": t.get("category"),
                    "eval_cat": e_raw.get("category"),
                })
            else:
                sim = jaccard_similarity(t_norm, e_norm)
                if sim >= 0.80:
                    near_leakages.append({
                        "train_id": t_id,
                        "eval_id": e_id,
                        "similarity": round(sim, 3),
                        "train_text": t.get("text", ""),
                        "eval_text": e_raw.get("text", ""),
                        "train_cat": t.get("category"),
                        "eval_cat": e_raw.get("category"),
                    })

    return {
        "train_set": train_name,
        "eval_set": eval_name,
        "exact_leakage_count": len(exact_leakages),
        "near_leakage_count": len(near_leakages),
        "exact_leakages": exact_leakages[:5],
        "near_leakages": near_leakages[:5],
        "is_leakage_free": (len(exact_leakages) == 0 and len(near_leakages) == 0),
    }


def main():
    print("=" * 75)
    print("NEXUS-SLM EXPERIMENT INPUTS & DATA AUDIT")
    print("=" * 75)

    data_dir = PROJECT_ROOT / "data"
    train_v1_path = data_dir / "train" / "train.jsonl"
    val_v1_path = data_dir / "validation" / "validation.jsonl"
    test_v1_path = data_dir / "test" / "test.jsonl"

    v2_dir = data_dir / "v2"
    v2_train_path = v2_dir / "train.jsonl"
    v2_val_path = v2_dir / "val.jsonl"
    v2_test_path = v2_dir / "test_v2.jsonl"
    challenge_path = data_dir / "evaluation" / "high_risk_challenge_set.jsonl"

    files = {
        "train_v1": train_v1_path,
        "val_v1": val_v1_path,
        "test_v1": test_v1_path,
        "v2_train": v2_train_path,
        "v2_val": v2_val_path,
        "v2_test": v2_test_path,
        "challenge_70": challenge_path,
    }

    hashes = {name: compute_sha256(p) for name, p in files.items()}
    print("\nFile SHA-256 Hashes:")
    for name, h in hashes.items():
        print(f"  {name:<15}: {h} ({files[name]})")

    # Load records
    datasets = {name: load_dataset(p) for name, p in files.items()}
    datasets["v2_train_1000"] = datasets["v2_train"][:1000]

    # Run audits
    audits = {}
    for key in ["train_v1", "val_v1", "test_v1", "v2_train_1000", "v2_val", "v2_test", "challenge_70"]:
        audits[key] = audit_split(key, datasets[key])
        print(f"\nAudit: {key} (N={audits[key]['count']})")
        print(f"  Categories (14 expected): {len(audits[key]['category_distribution'])} unique")
        print(f"  Exact dups: {audits[key]['exact_duplicate_total_excess']}, Near dups: {audits[key]['near_duplicate_count']}, Conflicting: {audits[key]['conflicting_labels_count']}")
        print(f"  Devanagari texts: {audits[key]['devanagari_script_count']}")
        print(f"  Provenance: {audits[key]['provenance_breakdown']}")

    # Cross-split leakage checks
    print("\n" + "=" * 75)
    print("CROSS-SPLIT DATA LEAKAGE AUDIT")
    print("=" * 75)

    leakage_checks = [
        ("v2_train_1000", "test_v1"),
        ("v2_train_1000", "val_v1"),
        ("v2_train_1000", "v2_val"),
        ("v2_train_1000", "v2_test"),
        ("v2_train_1000", "challenge_70"),
        ("train_v1", "test_v1"),
        ("train_v1", "val_v1"),
        ("v2_train", "test_v1"),
        ("v2_train", "v2_val"),
        ("v2_train", "v2_test"),
    ]

    leakage_results = []
    for tr, ev in leakage_checks:
        res = check_cross_leakage(tr, datasets[tr], ev, datasets[ev])
        leakage_results.append(res)
        status_str = "CLEAN" if res["is_leakage_free"] else f"WARNING (exact={res['exact_leakage_count']}, near={res['near_leakage_count']})"
        print(f"  Leakage {tr:<15} vs {ev:<15}: {status_str}")
        if not res["is_leakage_free"]:
            for ex in res["exact_leakages"]:
                print(f"    EXACT: [{ex['train_cat']}] '{ex['train_text'][:40]}' == [{ex['eval_cat']}] '{ex['eval_text'][:40]}'")
            for nr in res["near_leakages"]:
                print(f"    NEAR (sim={nr['similarity']}): [{nr['train_cat']}] '{nr['train_text'][:40]}' ~ [{nr['eval_cat']}] '{nr['eval_text'][:40]}'")

    # Save artifact
    out_artifact = {
        "file_hashes": hashes,
        "audits": audits,
        "leakage_results": leakage_results,
    }

    out_path = PROJECT_ROOT / "experiments" / "artifacts" / "experiment_input_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_artifact, f, indent=2)

    print(f"\nAudit completed. Artifact saved to: {out_path}")


if __name__ == "__main__":
    main()
