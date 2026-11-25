"""Automated Artifact, Dataset Hash, Archive & Documentation Consistency Checker."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "experiments" / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive_v1"

EXPECTED_FILES = [
    "eval_human_holdout_baseline_v1.json",
    "eval_human_holdout_refined_v2.json",
    "eval_human_holdout_v3_hardened.json",
    "eval_synthetic_test_baseline_v1.json",
    "eval_synthetic_test_refined_v2.json",
    "eval_synthetic_test_v3_hardened.json",
    "eval_qwen_clean_v2_true_base.json",
    "reproduced_refined_v2_baseline.json",
]

CANONICAL_DATASETS = {
    "train_clean": (
        "data/v2_clean_template_grouped/train_clean.jsonl",
        4797,
        "f67b50c7724e8cd53d029058c688e712b99b9a3f1be3df0a92c756975220fe1c",
        "8e042d18120ec5e7f87d023856b141bbfd8c3b62cdc0319f7b9b8df18040f3e0",
    ),
    "val_clean": (
        "data/v2_clean_template_grouped/val_clean.jsonl",
        449,
        "faa98cf83cbe209736d8890a902f65fa382cb2ff31b96e55608d98d9e68fa29a",
        "1bfd056bb5b7e57edf6509c91a5cf08f745f5216386416fcbdbfdf6642eed279",
    ),
    "test_clean": (
        "data/v2_clean_template_grouped/test_clean.jsonl",
        746,
        "fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b",
        "7e32c5111284db5091caf7186f73ce09876980fddfc83aa540e142f1e1b6acd1",
    ),
    "human_eval_v1": (
        "data/human_eval/annotation_template.jsonl",
        200,
        "e26d4eec99ddd247d3037652466df0024100b0cc85b9ca48e0fa32ed300923d3",
        "c27374cbbc185e5b950dc2a1db15841affb290c8499f1eaa7af70ae68dafebfd",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_canonical_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_all() -> List[str]:
    errors: List[str] = []

    # 1. Verify evaluation artifacts exist and satisfy core invariants
    for fname in EXPECTED_FILES:
        fp = RESULTS_DIR / fname
        if not fp.exists():
            errors.append(f"Missing evaluation artifact: {fp}")
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        if fname.startswith("eval_human_holdout") or fname.startswith("eval_synthetic_test"):
            crit = data.get("critical_safety_metrics", {})
            if crit.get("critical_recall", 0.0) < 0.999 or crit.get("critical_false_negatives", -1) != 0:
                errors.append(f"{fname}: critical safety invariant violated ({crit})")
        elif fname == "eval_qwen_clean_v2_true_base.json":
            pol_esc = data.get("policy_applied_metrics", {}).get("escalation", {})
            if pol_esc.get("critical_false_negatives", -1) != 0:
                errors.append(f"{fname}: critical FN != 0 ({pol_esc})")

    # 2. Verify locked dataset counts and SHA-256 checksums (canonical LF invariant across Windows/Linux Git)
    for key, (rel_p, expected_n, expected_win_hash, expected_lf_hash) in CANONICAL_DATASETS.items():
        p = ROOT / rel_p
        if not p.exists():
            errors.append(f"Missing canonical dataset {key}: {rel_p}")
            continue
        actual_lf = sha256_canonical_lf(p)
        if actual_lf != expected_lf_hash:
            errors.append(f"{rel_p}: Canonical LF SHA-256 mismatch! expected {expected_lf_hash}, got {actual_lf}")
        n = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        if n != expected_n:
            errors.append(f"{rel_p}: line count mismatch! expected {expected_n}, got {n}")

    # 3. Verify immutable archive integrity (canonical LF check works on Windows CRLF and Ubuntu LF)
    manifest_p = ARCHIVE_DIR / "ARCHIVE_MANIFEST.json"
    if not manifest_p.exists():
        errors.append("Missing experiments/results/archive_v1/ARCHIVE_MANIFEST.json")
    else:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        for entry in manifest.get("artifacts", []):
            af = ARCHIVE_DIR / entry["archived_file"]
            if not af.exists():
                errors.append(f"Missing archived file: {af.name}")
            elif sha256_canonical_lf(af) != entry.get("sha256_canonical_lf", entry["sha256"]):
                errors.append(f"Archived file checksum tampered: {af.name}")

    # 4. Verify sync markers & block known stale values / mismatched hashes in active docs
    from scripts.evaluation.sync_results_from_artifacts import build_authoritative_summary_block

    expected_sync_block = build_authoritative_summary_block()
    for rel_doc in ("README.md", "docs/RESULTS.md"):
        doc_p = ROOT / rel_doc
        txt = doc_p.read_text(encoding="utf-8")
        if "<!-- SYNC:RESULTS_SUMMARY:START -->" not in txt or "<!-- SYNC:RESULTS_SUMMARY:END -->" not in txt:
            errors.append(f"{rel_doc}: missing SYNC:RESULTS_SUMMARY markers")
        else:
            actual_block = (
                "<!-- SYNC:RESULTS_SUMMARY:START -->"
                + txt.split("<!-- SYNC:RESULTS_SUMMARY:START -->")[1].split("<!-- SYNC:RESULTS_SUMMARY:END -->")[0]
                + "<!-- SYNC:RESULTS_SUMMARY:END -->"
            )
            if actual_block.strip() != expected_sync_block.strip():
                errors.append(f"{rel_doc}: synced results table is out of date with JSON artifacts; run scripts/evaluation/sync_results_from_artifacts.py")

    stale_doc_checks = {
        "README.md": ["(231/231)", "(232/232)", "(283)", "49.82%", "215 passed"],
        "docs/RESULTS.md": ["(231/231)", "(232/232)", "215 passed"],
        "docs/EXPERIMENT_TRACKING_STANDARD.md": ["88c3e8ec9ab5ee2a7fe018260a92548480a875a59334fe8923a4b7fcf7164cf3", "| 744 |"],
        "docs/EVALUATION_INTEGRITY_AUDIT.md": ["86cae5be"],
        "docs/HUMAN_EVALUATION_PROTOCOL.md": ["164 records"],
        "docs/CALIBRATION_ACCEPTANCE_REPORT.md": ["0.2894", "0.2781"],
        "docs/FINAL_10_OUT_OF_10_ACCEPTANCE_REPORT.md": ["has earned an authoritative, mathematically defensible **10/10"],
    }
    for rel_doc, forbidden_strings in stale_doc_checks.items():
        txt = (ROOT / rel_doc).read_text(encoding="utf-8")
        for bad in forbidden_strings:
            if bad in txt:
                errors.append(f"{rel_doc}: contains known stale/incorrect string '{bad}'")

    # Required authoritative values present
    std_txt = (ROOT / "docs/EXPERIMENT_TRACKING_STANDARD.md").read_text(encoding="utf-8")
    if "faa98cf83cbe209736d8890a902f65fa382cb2ff31b96e55608d98d9e68fa29a" not in std_txt:
        errors.append("docs/EXPERIMENT_TRACKING_STANDARD.md missing verified val_clean SHA-256")
    readme_txt = (ROOT / "README.md").read_text(encoding="utf-8")
    if "53.89%" not in readme_txt or "57.51%" not in readme_txt:
        errors.append("README.md missing reconciled true-base Qwen (53.89%) or Refined Policy V2 (57.51%) figures")

    return errors


def main():
    errs = check_all()
    if errs:
        print("DOCUMENTATION & ARTIFACT CONSISTENCY ERRORS:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print("PASS: All artifacts, dataset SHA-256 hashes, archives, and synchronized docs match cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    main()
