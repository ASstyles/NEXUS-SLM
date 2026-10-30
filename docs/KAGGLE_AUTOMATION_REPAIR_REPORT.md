# NEXUS-SLM — Kaggle Training Automation Repair & Security Audit Report

**Date**: 2026-10-30  
**Branch**: `sprint/clean-data-model-improvement`  
**Auditor / Engineering Lead**: NEXUS-SLM Core Agentic Team  
**Scope**: `scripts/training/submit_kaggle_gpu_job.py`, `tests/test_kaggle_automation_regression.py`, `notebooks/kaggle_clean_v2_bundle/`

---

## 1. Executive Summary

A comprehensive architectural and security audit of `scripts/training/submit_kaggle_gpu_job.py` revealed multiple high-severity defects that rendered remote GPU training unsafe and defective. Most critically, the legacy automation:
1. Hardcoded target output to `models/qwen25_05b_lora_seed42` and executed `shutil.rmtree(OUTPUT_DIR, ignore_errors=True)`, thereby risking destruction of existing model weights.
2. Selected the legacy notebook (`notebooks/nexus_qwen25_05b_lora_training.ipynb`) contaminated with legacy dataset splits instead of the template-grouped clean dataset pipeline.
3. Extracted untrusted remote zip archives using `shutil.unpack_archive` without path sanitization (vulnerable to Zip Slip path traversal) or CRC/integrity checks.
4. Bound evaluation to the obsolete script `evaluate_qwen_candidate.py` and silently swallowed evaluation failures, exiting with status code 0.

All nine defects have been corrected. Comprehensive regression tests were authored in `tests/test_kaggle_automation_regression.py` (9/9 tests passing), bringing the repository's automated test suite to **160/160 passing tests**.

---

## 2. Detailed Root-Cause & Remediation Matrix

| Defect ID | Severity | Root Cause | Implemented Resolution | Verified By |
|---|---|---|---|---|
| **KAG-01** | **CRITICAL** | `shutil.rmtree(OUTPUT_DIR)` targeted `models/qwen25_05b_lora_seed42`, destroying earlier model artifacts. | Removed `rmtree`. Created `PROTECTED_MODEL_DIRS` registry that raises `PermissionError` if any legacy model path or `models/` root is targeted. | `test_legacy_model_directories_remain_untouched` |
| **KAG-02** | **CRITICAL** | `NOTEBOOK_SRC` pointed to `notebooks/nexus_qwen25_05b_lora_training.ipynb` (legacy dataset). | Pointed `DEFAULT_NOTEBOOK_SRC` to clean notebook `notebooks/clean_data_qwen25_05b_lora_training_v2.ipynb` (embedding $N=4,797$ clean train and $N=449$ clean val). | `test_clean_notebook_is_selected` |
| **KAG-03** | **HIGH** | Arbitrary overwrite of prior download artifacts without versioning or run isolation. | Enforced dedicated output root `experiments/artifacts/training_runs/{candidate_name}_{run_id}`. Rejects non-empty target paths with `FileExistsError`. | `test_existing_artifacts_cannot_be_overwritten` |
| **KAG-04** | **HIGH** | Kernel slug collision (`nexus-qwen25-05b-lora-training`) and bundle overwrites in `notebooks/kaggle_job`. | Assigned unique kernel slug `nexus-qwen25-clean-v2-training` and isolated bundle directory `notebooks/kaggle_clean_v2_bundle/`. | `test_prepare_kernel_bundle_segregation` |
| **KAG-05** | **HIGH** | Unchecked zip extraction (`shutil.unpack_archive`) vulnerable to Zip Slip directory traversal. | Implemented `safe_extract_zip()` with canonical path resolution, rejecting members with `..`, absolute paths, or escaping destinations. | `test_safe_extract_zip_prevents_zip_slip` |
| **KAG-06** | **HIGH** | Missing archive integrity checks on corrupt or truncated downloads. | Integrated `zipfile.is_zipfile()` validation, 0-byte checks, and CRC header testing via `zf.testzip()`. | `test_safe_extract_zip_rejects_corrupted_archive` |
| **KAG-07** | **MEDIUM** | Incomplete artifact acceptance (missing configs, missing safetensors, or 0-byte weights). | Added `verify_adapter_artifacts()` checking `adapter_config.json`, `adapter_model.safetensors` size, and tokenizer files. | `test_verify_adapter_artifacts_rejects_incomplete` |
| **KAG-08** | **HIGH** | Evaluation pointed to legacy script `evaluate_qwen_candidate.py`. | Integrated `scripts/evaluation/evaluate_clean_v2_comprehensive.py` evaluating clean test/val splits against all quality gates. | Code inspection & regression tests |
| **KAG-09** | **HIGH** | Silent exit code 0 even when evaluation or downloads failed. | Propagated exit codes throughout the orchestrator; any step failure calls `sys.exit(1)`. | `test_evaluation_failure_causes_automation_failure` |

---

## 3. Security Implementation Details

### 3.1 Protected Directory Boundary Guard
```python
PROTECTED_MODEL_DIRS = [
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42",
    PROJECT_ROOT / "models" / "nexus_qlora_v1",
    PROJECT_ROOT / "models" / "nexus_qlora_v2",
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf",
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf_q4_k_m",
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf_q8_0",
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_int8",
    PROJECT_ROOT / "models",
]

def check_protected_path(target_dir: Path) -> None:
    resolved_target = target_dir.resolve()
    for prot in PROTECTED_MODEL_DIRS:
        if resolved_target == prot.resolve():
            raise PermissionError(f"SAFETY VIOLATION: Target directory '{target_dir}' is a protected directory!")
```

### 3.2 Zip Slip Path Traversal Protection
```python
def safe_extract_zip(zip_path: Path, target_dir: Path) -> List[Path]:
    target_dir = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        corrupted = zf.testzip()
        if corrupted is not None:
            raise ValueError(f"Corrupted zip archive: bad CRC on '{corrupted}'")
        for member in zf.infolist():
            normalized = member.filename.replace("\\", "/")
            if member.filename.startswith(("/", "\\")) or ".." in normalized.split("/"):
                raise ValueError(f"SECURITY ALERT: Zip Slip vulnerability detected: {member.filename}")
            dest = (target_dir / member.filename).resolve()
            if not str(dest).startswith(str(target_dir)):
                raise ValueError(f"SECURITY ALERT: Extraction path escapes destination: {dest}")
        zf.extractall(str(target_dir))
```

---

## 4. Automated Regression Test Suite

All safety invariants are continuously enforced via `tests/test_kaggle_automation_regression.py`:

```
tests/test_kaggle_automation_regression.py::test_clean_notebook_is_selected PASSED           [ 11%]
tests/test_kaggle_automation_regression.py::test_prepare_kernel_bundle_segregation PASSED   [ 22%]
tests/test_kaggle_automation_regression.py::test_legacy_model_directories_remain_untouched PASSED [ 33%]
tests/test_kaggle_automation_regression.py::test_existing_artifacts_cannot_be_overwritten PASSED  [ 44%]
tests/test_kaggle_automation_regression.py::test_safe_extract_zip_rejects_empty_archive PASSED    [ 55%]
tests/test_kaggle_automation_regression.py::test_safe_extract_zip_rejects_corrupted_archive PASSED[ 66%]
tests/test_kaggle_automation_regression.py::test_safe_extract_zip_prevents_zip_slip PASSED        [ 77%]
tests/test_kaggle_automation_regression.py::test_verify_adapter_artifacts_rejects_incomplete PASSED[ 88%]
tests/test_kaggle_automation_regression.py::test_evaluation_failure_causes_automation_failure PASSED[100%]
```

Execution time: **0.17s** (9 passed). Total repository test suite: **160 passed in 31.99s**.

---

## 5. Verification & Reproduction Instructions

To execute the Kaggle automation regression tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/test_kaggle_automation_regression.py
```

To run a safe dry-run / status check of the Kaggle GPU job:
```powershell
.\.venv\Scripts\python.exe scripts/training/submit_kaggle_gpu_job.py --check-only
```

To execute full remote GPU training (once Kaggle credentials are provided):
```powershell
.\.venv\Scripts\python.exe scripts/training/submit_kaggle_gpu_job.py
```

