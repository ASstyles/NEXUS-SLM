import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schemas import TicketRecord
from src.data.validator import DataValidator
from src.data.leakage_detector import LeakageDetector

train_p = Path("data/train/train.jsonl")
val_p = Path("data/validation/validation.jsonl")
test_p = Path("data/test/test.jsonl")

def load_split(p: Path):
    recs = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(TicketRecord(**json.loads(line)))
    return recs

train = load_split(train_p)
val = load_split(val_p)
test = load_split(test_p)
all_recs = train + val + test

print(f"Total examples: {len(all_recs)}")
print(f"  Training: {len(train)}")
print(f"  Validation: {len(val)}")
print(f"  Test (Locked): {len(test)}")

cat_counts = Counter(r.category for r in all_recs)
lang_counts = Counter(r.language for r in all_recs)
prio_counts = Counter(r.priority for r in all_recs)

print("\nExamples per Category:")
for k, v in sorted(cat_counts.items()):
    print(f"  {k}: {v}")

print("\nExamples per Language:")
for k, v in sorted(lang_counts.items()):
    print(f"  {k}: {v}")

print("\nExamples per Priority:")
for k, v in sorted(prio_counts.items()):
    print(f"  {k}: {v}")

synth_count = sum(1 for r in all_recs if r.source == "synthetic")
human_count = sum(1 for r in all_recs if r.source in ["curated", "human_reviewed"])

print(f"\nSynthetic Count: {synth_count} ({synth_count/len(all_recs)*100:.2f}%)")
print(f"Human/Curated Count: {human_count} ({human_count/len(all_recs)*100:.2f}%)")

# Check missing values and invalid labels
validator = DataValidator()
invalid_count = 0
for r in all_recs:
    rec, res = validator.validate_record(r.model_dump())
    if not res.valid:
        invalid_count += 1
print(f"Invalid / Missing Value Records: {invalid_count}")

# Check Leakage
detector = LeakageDetector()
leak_report = detector.detect_leakage(train, val, test, dataset_version="dataset_v1")
print(f"\nLeakage Detection Status: {leak_report.status}")
print(f"Exact Cross-Split Duplicates: {leak_report.exact_duplicates_across_splits}")
print(f"Near-Duplicates Flagged: {leak_report.near_duplicates_flagged}")
print(f"Max Similarity Score: {leak_report.max_similarity_score}")
