"""Create immutable versioned archive experiments/results/archive_v1/ with provenance manifest."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments" / "results" / "archive_v1"
ARCHIVE.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_canonical_lf(path: Path) -> str:
    """Line-ending-independent SHA-256 matching Git/POSIX LF checkouts."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


HISTORICAL_FILES = [
    ("experiments/artifacts/heuristic_clean_v2_test_results.json", "Heuristic Fallback V2 under Policy V1 (100% fallback escalation, 2026-11-24T13:47:19Z)"),
    ("experiments/artifacts/qwen25_05b_clean_v2_lora_test_results.json", "Qwen2.5-0.5B Clean V2 LoRA loaded on double-LoRA contaminated scratch_merged_qwen base with preset 0.90 confidence (2026-11-24T14:10:53Z)"),
    ("experiments/artifacts/final_clean_v2_model_evaluation.json", "Clean V2 comparison run on scratch_merged_qwen + simulated +35ms heuristic baseline (2026-10-30T20:02:04Z)"),
    ("experiments/artifacts/qwen25_05b_evaluation_report.json", "Early Qwen2.5-0.5B seed42 validation on V1 N=44 split (2026-10-11T18:32:23Z)"),
    ("experiments/artifacts/qwen25_05b_gguf_q4_k_m_evaluation_report.json", "Early Q4_K_M GGUF seed42 evaluation on V1 N=44 validation split (2026-10-12T15:19:01Z)"),
    ("experiments/artifacts/final_calibration_metrics.json", "Sequence token probability ECE/Brier baseline without class posteriors (2026-10-30T20:02:04Z)"),
    ("experiments/artifacts/escalation_safety_evaluation.json", "V1 escalation policy benchmark on V1 N=42 test & N=70 challenge set (2026-10-11T14:44:35Z)"),
    ("experiments/results/eval_synthetic_test_baseline_v1.json", "PR #4 Historical Baseline Policy V1 on test_clean.jsonl N=746 (commit 3e5a1ea / e290a69)"),
    ("experiments/results/eval_synthetic_test_refined_v2.json", "PR #4 Production Heuristic V2 + Refined Policy V2 on test_clean.jsonl N=746 (commit 3e5a1ea / e290a69)"),
    ("experiments/results/eval_human_holdout_baseline_v1.json", "PR #4 Historical Baseline Policy V1 on N=184 dev-exposed scenarios (commit 3e5a1ea / e290a69)"),
    ("experiments/results/eval_human_holdout_refined_v2.json", "PR #4 Production Heuristic V2 + Refined Policy V2 on N=184 dev-exposed scenarios (commit 3e5a1ea / e290a69)"),
]

manifest = {
    "archive_version": "v1_pre_independent_validation_sprint",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": "e290a69214834bf3e4d48bc34601ff15581790d5",
    "note": "Immutable historical baseline snapshots preserved for audit traceability. Do not overwrite or delete.",
    "artifacts": [],
}

for rel_path, provenance_note in HISTORICAL_FILES:
    src = ROOT / rel_path
    if not src.exists():
        manifest["artifacts"].append({"source_path": rel_path, "status": "MISSING"})
        continue
    dst = ARCHIVE / src.name
    shutil.copy2(src, dst)
    manifest["artifacts"].append({
        "archived_file": dst.name,
        "source_path": rel_path,
        "sha256": sha256_file(dst),
        "sha256_canonical_lf": sha256_canonical_lf(dst),
        "size_bytes": dst.stat().st_size,
        "provenance_note": provenance_note,
    })

(ARCHIVE / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Archived {len(manifest['artifacts'])} historical artifacts to {ARCHIVE}")
