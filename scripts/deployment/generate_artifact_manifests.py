"""Generate cryptographic SHA-256 provenance manifests for all datasets and model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_canonical_lf(path: Path) -> str:
    """Normalized LF SHA-256 invariant across Windows CRLF and POSIX/Git LF checkouts."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_manifests() -> Dict[str, Any]:
    # 1. Dataset manifest
    dataset_entries = [
        ("v1_train", "data/train/train.jsonl", "Historical V1 training split (template-overlapping)"),
        ("v1_val", "data/val/val.jsonl", "Historical V1 validation split"),
        ("v1_test", "data/test/test.jsonl", "Historical V1 locked test split"),
        ("v2_clean_train", "data/v2_clean_template_grouped/train_clean.jsonl", "V2 clean template-grouped training split"),
        ("v2_clean_val", "data/v2_clean_template_grouped/val_clean.jsonl", "V2 clean template-grouped validation split"),
        ("v2_clean_test", "data/v2_clean_template_grouped/test_clean.jsonl", "V2 locked clean test split (N=746)"),
        ("human_eval_v1_exposed", "data/human_eval/annotation_template.jsonl", "Development-exposed V1 human scenarios (N=184 active + 16 pending)"),
        ("high_risk_challenge_70", "data/evaluation/high_risk_challenge_set.jsonl", "High-risk adversarial/safety challenge set (N=70)"),
        ("v3_diverse_train", "data/v3_diverse/train_v3.jsonl", "V3 syntactically diverse synthetic training split"),
        ("v3_diverse_val", "data/v3_diverse/val_v3.jsonl", "V3 syntactically diverse validation split"),
        ("v3_diverse_cal", "data/v3_diverse/cal_v3.jsonl", "V3 calibration holdout split"),
        ("v3_diverse_audit", "data/v3_diverse/dev_audit_v3.jsonl", "V3 unseen dev audit benchmark split"),
        ("human_eval_v2_plan", "data/human_eval_v2/allocation_plan_v2.json", "V2 independent human eval cohort scaffold (PENDING external reviewers)"),
        ("human_eval_v2_intake", "data/human_eval_v2/intake_collection_template.jsonl", "V2 400-slot intake template (320 sealed + 80 dev reserved)"),
    ]

    datasets_out = []
    for key, rel, desc in dataset_entries:
        p = ROOT / rel
        if not p.exists():
            continue
        recs = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) if p.suffix == ".jsonl" else None
        datasets_out.append({
            "key": key,
            "path": rel,
            "description": desc,
            "size_bytes": p.stat().st_size,
            "record_count": recs,
            "sha256": sha256_file(p),
            "sha256_canonical_lf": sha256_canonical_lf(p),
        })

    ds_manifest = {
        "schema_version": "nexus_dataset_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets_out,
    }
    (ROOT / "data/dataset_manifest.json").write_text(json.dumps(ds_manifest, indent=2), encoding="utf-8")

    # 2. Model weight artifact manifest (large weights excluded from git per .gitignore)
    weight_paths = [
        ("qwen25_05b_lora_seed42", "models/qwen25_05b_lora_seed42/adapter_model.safetensors", "V1 template-trained Qwen2.5-0.5B LoRA adapter (gitignored)"),
        ("qwen25_05b_clean_v2_lora", "experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter/adapter_model.safetensors", "Clean V2 Qwen2.5-0.5B LoRA adapter (gitignored, Experimental)"),
        ("qwen25_05b_seed42_f16_gguf", "models/qwen25_05b_lora_seed42_gguf/qwen25_05b_lora_seed42_f16.gguf", "F16 GGUF converted from seed42 merge (gitignored)"),
        ("qwen25_05b_seed42_q8_0_gguf", "models/qwen25_05b_lora_seed42_gguf_q8_0/qwen25_05b_lora_seed42_q8_0.gguf", "Q8_0 GGUF converted from seed42 merge (gitignored)"),
        ("qwen25_05b_seed42_q4_k_m_gguf", "models/qwen25_05b_lora_seed42_gguf_q4_k_m/qwen25_05b_lora_seed42_q4_k_m.gguf", "Q4_K_M GGUF converted from seed42 merge (gitignored)"),
        ("qwen25_05b_seed42_onnx_int8", "models/qwen25_05b_lora_seed42_int8/model_quantized.onnx", "Dynamic INT8 ONNX converted from seed42 merge (gitignored)"),
        ("nexus_v2_hybrid_multitask_head", "experiments/artifacts/v2_models/nexus_v2_hybrid_multitask_head.pt", "V2 PyTorch Hybrid Multitask + Ordinal head weights (regenerable via run_v2_architecture_experiments.py)"),
        ("nexus_v2_lexical_featurizer", "experiments/artifacts/v2_models/nexus_v2_lexical_featurizer.joblib", "V2 Multilingual TF-IDF + SVD featurizer"),
    ]
    models_out = []
    for key, rel, note in weight_paths:
        p = ROOT / rel
        models_out.append({
            "key": key,
            "path": rel,
            "present_on_disk": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha256_file(p) if p.exists() else None,
            "note": note,
        })

    # Include HF base snapshot hash if cached locally
    hf_snaps = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots").glob("*"))
    hf_info = None
    if hf_snaps and (hf_snaps[-1] / "model.safetensors").exists():
        ms = hf_snaps[-1] / "model.safetensors"
        hf_info = {
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision_dir": hf_snaps[-1].name,
            "size_bytes": ms.stat().st_size,
            "sha256": sha256_file(ms),
        }

    mod_manifest = {
        "schema_version": "nexus_model_artifact_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_active_engine": "heuristic_fallback_v2 (deterministic Python modules under src/inference/, 0 external binary weights required for clean clone)",
        "clean_clone_neural_reproducibility_note": (
            "Large binary weights (*.safetensors, *.gguf, *.onnx, *.pt) are excluded from Git via .gitignore. "
            "Clean clones run Heuristic Fallback out of the box and can deterministically train V2 multitask heads via "
            "scripts/experiments_v2/run_v2_architecture_experiments.py once HF base Qwen/Qwen2.5-0.5B-Instruct is available."
        ),
        "hf_base_snapshot_verified": hf_info,
        "local_weight_artifacts": models_out,
    }
    (ROOT / "models/artifact_manifest.json").write_text(json.dumps(mod_manifest, indent=2), encoding="utf-8")
    return {"datasets": len(datasets_out), "models": len(models_out)}


def verify_manifests() -> List[str]:
    errs = []
    ds_p = ROOT / "data/dataset_manifest.json"
    if not ds_p.exists():
        return ["Missing data/dataset_manifest.json"]
    ds = json.loads(ds_p.read_text(encoding="utf-8"))
    for entry in ds.get("datasets", []):
        p = ROOT / entry["path"]
        if not p.exists():
            errs.append(f"Missing dataset {entry['path']}")
        elif sha256_canonical_lf(p) != entry.get("sha256_canonical_lf", entry["sha256"]):
            errs.append(f"Dataset SHA-256 mismatch: {entry['path']}")
    mod_p = ROOT / "models/artifact_manifest.json"
    if not mod_p.exists():
        errs.append("Missing models/artifact_manifest.json")
    else:
        mod = json.loads(mod_p.read_text(encoding="utf-8"))
        for entry in mod.get("local_weight_artifacts", []):
            p = ROOT / entry["path"]
            # Large binary weights are gitignored in clean CI clones; verify hash whenever present locally
            if p.exists() and entry.get("sha256"):
                if sha256_file(p) != entry["sha256"]:
                    errs.append(f"Model artifact SHA-256 mismatch: {entry['path']}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        errs = verify_manifests()
        if errs:
            print("MANIFEST ERRORS:", errs)
            sys.exit(1)
        print("PASS: All dataset & model manifests verified.")
    else:
        res = build_manifests()
        print("Built manifests:", res)


if __name__ == "__main__":
    main()
