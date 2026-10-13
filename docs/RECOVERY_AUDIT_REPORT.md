# NEXUS-SLM — Complete Emergency Recovery, Integrity Verification, and Project Continuation Report

**Audit Date**: October 13, 2026  
**Recovery Lead**: Autonomous Emergency Recovery & Verification Agent  
**Overall Recovery Status**: **COMPLETE & VERIFIED (100% RECOVERED)**  
**Target Repository**: `https://github.com/ASstyles/NEXUS-SLM.git`  

---

## Executive Summary

Following accidental local file deletion and project extraction, a full emergency recovery, integrity verification, and environment restoration workflow was conducted across all 10 designated phases. 

- **Git History**: Restored in an independent sibling clone (`C:\Users\avira\Desktop\NEXUS-SLM-RECOVERY-CLONE`) at remote commit `2a17eae4a4ca40c1ca2b516453f0b73455ecf1d4` (`main` branch).
- **Tracked Project Files**: All 316 tracked repository files are present and content-identical (0 substantive diffs, zero missing tracked files).
- **Missing Dataset Restored**: `data/v2/train.jsonl` (2,304,530 bytes, 5,000 JSONL records) was verified against `data/v2/dataset_v2_lock.json` and restored with exact SHA-256 match `db392a62cfa41eb4115db1af674a64d0c5ba6dd6dc8d21baabd213245e1b388a`.
- **Surviving Model Weights & Checksums**: All surviving large model weights (Q8_0 GGUF, Q4_K_M GGUF, F16 intermediate GGUF, Qwen LoRA adapter, INT8 ONNX) were verified on disk with 100% bitwise cryptographic checksum match.
- **Local Development Environment**: Virtual environment was cleanly rebuilt with Python 3.10.11, CPU-only PyTorch 2.14.1+cpu (zero CUDA), CI dependencies (`llama-cpp-python` CPU, `transformers`, `optimum`, `onnxruntime`, `fastapi`, `pytest`), and project editable install (`nexus-slm` 1.0.0). `pip check` confirmed zero dependency conflicts.
- **Test Suite & Verification**: 100% of the regression test suite passed (**134 passed / 134 collected**, 0 failures, 0 skips). Fresh local demo verification benchmark executed with linear percentile estimation (`numpy.percentile`), confirming P95 latency (1169.09 ms vs 2500.0 ms SLA) and human escalation safety invariants.
- **Active Production Model**: Preserved strictly as `nexus-qlora-v1` in `api/main.py` and `models/registry.json`. Q8_0 and Q4_K_M candidates remain `experimental`.
- **Operational Governance**: In accordance with non-negotiable safety rules, shadow pilot activation remains strictly blocked on missing operational evidence (BLK-01, BLK-02, BLK-03, BLK-05, BLK-06, BLK-07) without fake data or premature production changes.

---

## 1. Directory Paths & Snapshot Preservation Ledger

| Asset | Path | Status | Details |
| :--- | :--- | :---: | :--- |
| **Original Project Directory** | `C:\Users\avira\Desktop\NEXUS-SLM` | **PRESERVED & RESTORED** | Surviving files protected; 6 missing tracked files safely restored. |
| **Recovery Clone Destination** | `C:\Users\avira\Desktop\NEXUS-SLM-RECOVERY-CLONE` | **RESTORED & CLEAN** | Cloned from `https://github.com/ASstyles/NEXUS-SLM.git`, commit `2a17eae4a4ca40c1ca2b516453f0b73455ecf1d4`. |
| **Original Downloaded ZIP** | `C:\Users\avira\Downloads\NEXUS-SLM-main.zip` | **PRESERVED UNTOUCHED** | Size: 19,829,425 bytes, MD5/SHA preserved intact. |
| **Pre-Recovery Local Backup** | `C:\Users\avira\Desktop\NEXUS-SLM-PRE-RECOVERY-BACKUP` | **CREATED PRIOR TO WRITES** | 489 irreplaceable files (149.61 MB) backed up before any restoration. |
| **Available Disk Space** | Drive `C:\` | **VERIFIED ABUNDANT** | Total: 475.67 GB, Free: 284.72 GB before/after rebuild. |

---

## 2. Git History & Remote Verification Ledger

| Property | Value | Verification Finding |
| :--- | :--- | :--- |
| **Remote Repository URL** | `https://github.com/ASstyles/NEXUS-SLM.git` | Accessible via Git Credential Manager / HTTPS. |
| **Previously Audited Commit** | `2a17eae4a4ca40c1ca2b516453f0b73455ecf1d4` | Confirmed. |
| **Cloned HEAD Commit SHA** | `2a17eae4a4ca40c1ca2b516453f0b73455ecf1d4` | **EXACT MATCH** (`git rev-parse HEAD`). |
| **Cloned Active Branch** | `main` | Tracking `origin/main`. |
| **Commit History Depth** | Full Commit History | `git rev-parse --is-shallow-repository` returned `false`. |
| **Working Tree Status** | Clean | `working tree clean`, zero unstaged changes in clone. |
| **Git Internals Handling** | Preserved Separate | Recovery clone kept in sibling directory; no Git internals copied into damaged folder. |

---

## 3. Restored File Ledger (Audit of Changes to Original Project)

Exactly **6 missing tracked files** were identified as absent from the recovered directory but present in the remote repository HEAD. Each file was validated for source checksum before copying, written to its missing path without overwriting any surviving file, and re-verified post-copy:

| # | Relative File Path | Size (Bytes) | Cryptographic SHA-256 | Source Commit | Post-Copy Verification Status |
| :-: | :--- | :-: | :--- | :-: | :---: |
| 1 | `data/v2/train.jsonl` | 2,304,530 | `db392a62cfa41eb4115db1af674a64d0c5ba6dd6dc8d21baabd213245e1b388a` | `6e7d528` | **VERIFIED (5,000 JSONL lines valid)** |
| 2 | `models/nexus_qlora_v1/tokenizer.json` | 3,767,835 | `265d20773e77d78d934ddbf21cbd0ff83a0005928bdf6a6f8da84e0cee04cecc` | `6e7d528` | **VERIFIED (Valid JSON token map)** |
| 3 | `models/nexus_qlora_v2/tokenizer.json` | 3,767,938 | `0a61d67ca5f4c571e162e4c8fc4cc9a9674c374315dffebfd44db9500e04e6a0` | `6e7d528` | **VERIFIED (Valid JSON token map)** |
| 4 | `models/qwen25_05b_lora_seed42/merges.txt` | 1,823,241 | `b8e04af9c950e393054010dcef499c3f005e803e9b2da1a09b96a26c33266eb1` | `cfd60b1` | **VERIFIED (BPE merge table)** |
| 5 | `models/qwen25_05b_lora_seed42/vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | `cfd60b1` | **VERIFIED (Valid JSON vocabulary)** |
| 6 | `models/qwen25_05b_lora_seed42_gguf_q4_k_m/vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | `cfd60b1` | **VERIFIED (Valid JSON vocabulary)** |

*Total files restored to original project directory:* **6 files** (17,216,910 bytes / 16.42 MB).  
*Files overwritten or deleted:* **0 files**.

---

## 4. Model Weights & Dataset Cryptographic Verification

All surviving large model weights, quantization artifacts, and locked dataset splits were audited by computing SHA-256 hashes directly from the on-disk files and comparing them with historical project manifests:

| Component | Disk Path | Size | Reference Hash | Actual Calculated Hash | Audit Result |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Q8_0 GGUF Model** | `models/qwen25_05b_lora_seed42_gguf_q8_0/qwen25_05b_lora_seed42_q8_0.gguf` | 531,067,808 bytes | `b0c891dca80498f95e97d37ace60b574860fbfc3100574aeb76bae524e79f6d8` | `b0c891dca80498f95e97d37ace60b574860fbfc3100574aeb76bae524e79f6d8` | **MATCH (PRESENT)** |
| **Q4_K_M GGUF Model** | `models/qwen25_05b_lora_seed42_gguf_q4_k_m/qwen25_05b_lora_seed42_q4_k_m.gguf` | 397,807,520 bytes | `69a7c3d640566738cbeaafa2400ebc5d9e19d5d3034e045369bb018e86a0a22d` | `69a7c3d640566738cbeaafa2400ebc5d9e19d5d3034e045369bb018e86a0a22d` | **MATCH (PRESENT)** |
| **F16 GGUF Intermediate** | `models/qwen25_05b_lora_seed42_gguf/qwen25_05b_lora_seed42_f16.gguf` | 994,156,448 bytes | `03babe1c2645dc13faa7c8a1db97853d41b435274edcc678bbc70054a3499caf` | `03babe1c2645dc13faa7c8a1db97853d41b435274edcc678bbc70054a3499caf` | **MATCH (PRESENT)** |
| **Qwen LoRA Adapter** | `models/qwen25_05b_lora_seed42/adapter_model.safetensors` | 35,237,104 bytes | `8a30ceafc32c2180d7414a7bdaa5199162921f4960ecd7eef34eed3a49f91a0a` | `8a30ceafc32c2180d7414a7bdaa5199162921f4960ecd7eef34eed3a49f91a0a` | **MATCH (PRESENT)** |
| **INT8 ONNX Model** | `models/qwen25_05b_lora_seed42_int8/model_quantized.onnx` | 496,023,750 bytes | `d38a4d687101c2529da7ad38a00d9fdfc69d113902ec6609e92877cfef61080f` | `d38a4d687101c2529da7ad38a00d9fdfc69d113902ec6609e92877cfef61080f` | **MATCH (PRESENT)** |
| **V2 Train Dataset** | `data/v2/train.jsonl` | 2,304,530 bytes | `db392a62cfa41eb4115db1af674a64d0c5ba6dd6dc8d21baabd213245e1b388a` | `db392a62cfa41eb4115db1af674a64d0c5ba6dd6dc8d21baabd213245e1b388a` | **MATCH (RESTORED)** |
| **V2 Val Dataset** | `data/v2/val.jsonl` | 231,116 bytes | `8c6e2fff62163c8c9ccad9b220488b4d87dd783c0f413ed4df4f5a3117a4891a` | `8c6e2fff62163c8c9ccad9b220488b4d87dd783c0f413ed4df4f5a3117a4891a` | **MATCH (PRESENT)** |
| **V2 Test Dataset** | `data/v2/test_v2.jsonl` | 225,681 bytes | `bbb0c028c0954f6dcda809d0eacb6d8be286d2e2f8e0a1ae943bab40fea149a5` | `bbb0c028c0954f6dcda809d0eacb6d8be286d2e2f8e0a1ae943bab40fea149a5` | **MATCH (PRESENT)** |
| **V1 Test Dataset** | `data/test/test.jsonl` | 12,620 bytes | `edf3fa5cd949a149ad5d759d81553d9704467491877d0f3c8b7cc5a887a9eda4` | `edf3fa5cd949a149ad5d759d81553d9704467491877d0f3c8b7cc5a887a9eda4` | **MATCH (PRESENT)** |
| **High-Risk Challenge** | `data/evaluation/high_risk_challenge_set.jsonl` | 21,998 bytes | `70b43711fd2f515bdd27d6daa8e604698a109f6093cdaea7f85bd584339dbc5b` | `70b43711fd2f515bdd27d6daa8e604698a109f6093cdaea7f85bd584339dbc5b` | **MATCH (PRESENT)** |

---

## 5. Development Environment Rebuild & Dependency Audit

1. **Virtual Environment**: Reconstructed cleanly using `py -3.10 -m venv .venv --clear`.
2. **Python Runtime**: `3.10.11 (64-bit AMD64)` verified.
3. **PyTorch Engine**: Installed `torch==2.14.1+cpu` from `https://download.pytorch.org/whl/cpu`.
   - `torch.cuda.is_available()`: `False`
   - `torch.version.cuda`: `None` (Zero GPU/CUDA bloat, strictly CPU-only CI architecture).
4. **CI Dependencies**: Installed via `requirements-ci.txt` with `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
   - `llama-cpp-python`: `0.3.36`
   - `transformers`: `4.57.6`
   - `optimum`: `2.1.0`
   - `onnxruntime`: `1.23.2`
   - `scikit-learn`: `1.7.2`
   - `pandas`: `2.3.3`
   - `fastapi`: `0.143.0`
   - `pytest`: `9.1.1`
5. **Project Package**: Installed editable package `nexus-slm` 1.0.0 via `pip install --no-deps -e .`.
6. **Integrity Validation**: `python -m pip check` returned `No broken requirements found.`

---

## 6. Verification Test Suite & Benchmark Results

### 6.1 Regression Suite Execution
- **Command**: `pytest -v tests/`
- **Collected**: 134 test functions
- **Passed**: **134** (100%)
- **Failed**: 0
- **Errors**: 0
- **Skipped**: 0
- **Duration**: 62.79 seconds
- **Exit Code**: 0

### 6.2 Key Invariant & Routing Policy Verifications
1. **Nullable Confidence Handling**:
   - `test_unavailable_confidence_cannot_silently_permit_automation` (**PASSED**).
   - When raw confidence is `None` (uncomputed or `logits_all=False`), the system conservatively routes to human review with reason `confidence_unavailable_conservative_escalation`.
2. **Deterministic Anti-Weakening Policy**:
   - `test_model_cannot_weaken_deterministic_escalation_policy` (**PASSED**).
   - A model output of `requires_human=False` CANNOT weaken deterministic safety rules (fraud, account closure, native Devanagari Hindi, prompt injection).
3. **Model Precedence for Escalation**:
   - `test_model_predicted_escalation_honored_when_policy_would_permit` (**PASSED**).
   - When policy would otherwise permit automation, a model prediction of `requires_human=True` is strictly honored with reason `model_predicted_human_escalation`.
4. **Hinglish Financial Dispute Invariant**:
   - `test_hinglish_financial_cut_ho_gaya_triggers_escalation` & `test_api_hinglish_payment_dispute_escalates` (**PASSED**).
   - "Mera payment account se cut ho gaya..." triggers `requires_human=True`, category `payment_failure`, priority `high`, reason `high_impact_financial_dispute`.

### 6.3 Fresh Synthetic Local Demo Benchmark
- **Candidate Tested**: `qwen25_05b_lora_seed42_q8_0`
- **Percentile Estimation Method**: `numpy.percentile(method='linear') [Hyndman and Fan Method 7]`
- **Benchmark Fixtures**: $N=30$ representative queries across 10 categories and 3 languages (English, Hinglish, Hindi)
- **Execution**: 30/30 succeeded, 0 errors.
- **Measured Latency**:
  - Median Client Latency: **935.80 ms**
  - P95 Client Latency: **1,169.09 ms** (SLA target $\le 2500$ ms: **PASS**)
  - Peak RAM: within budget ($\le 1.5$ GB)
- **Escalation Invariant Audit**: **PASS**
- **Artifacts Saved**: Saved fresh to `experiments/artifacts/fresh_local_demo_verification_results.json` and report folder without overwriting historical report files.

---

## 7. Model Lifecycle & Operational Governance Status

| Model Version | Registry Lifecycle Stage | Configuration Status | Serving Traffic |
| :--- | :---: | :---: | :---: |
| `nexus-qlora-v1` | **production** | Active default in `api/main.py` & `models/registry.json` | 100% Active Production Traffic |
| `qwen25_05b_lora_seed42_q8_0` | **experimental** | Staging candidate (Qualified for pilot) | 0% (Shadow Staging Only) |
| `qwen25_05b_lora_seed42_q4_k_m` | **experimental** | Staging candidate | 0% |
| `qwen25_05b_lora_seed42_int8` | **experimental** | Staging candidate | 0% |
| `nexus-slm-int8-dynamic` | **experimental** | Historical evaluation | 0% |
| `baseline-smollm-135m` | **retired** | Baseline | 0% |

### Remaining Shadow Pilot Activation Blockers
As audited in `docs/operations/activation_blocker_tracker.md`, the shadow pilot remains strictly **BLOCKED** pending real external operational evidence:

- **BLK-01 (Data Access & CRM Authorization)**: `FAIL` — Unexecuted DPA; `CRM_WEBHOOK_SECRET` unset; webhook listener offline.
- **BLK-02 (Cloud Infrastructure)**: `FAIL` — Internal VPC cluster service unprovisioned; DNS lookup for `nexus-slm-shadow.internal` unresolvable.
- **BLK-03 (Human Review Workforce Panel)**: `FAIL` — Reviewer panel unrostered; ground truth freeze manifest `data/pilot/ground_truth_freeze_manifest.json` does not exist.
- **BLK-04 (Remote Linux Docker CI)**: **PASS** — GitHub Actions Run ID 38283527589 passed on `ubuntu-latest`.
- **BLK-05 (Standard Hardware Profiling)**: `FAIL` — Target 2-vCPU / 4-GB dedicated staging host profiling report unexecuted.
- **BLK-06 (Shadow Drift Monitoring)**: `FAIL` — Continuous telemetry store and PSI alerting unprovisioned.
- **BLK-07 (Human Review Delivery Endpoint)**: `FAIL` — Webhook destination `POST /webhooks/escalations` disconnected.

---

## 8. Phased Continuation Plan (Next Milestone)

### Phase A: Local Baseline Re-Verification & Documentation Freeze (Current — COMPLETE)
- [x] Verify complete recovery ledger and checksum integrity.
- [x] Maintain reproducible Python 3.10 virtual environment and passing 134-test regression suite.
- [x] Validate Q8_0 GGUF local demonstration and documented linear percentile calculation.
- [x] Affirm that unmeasured confidence enforces conservative human escalation.

### Phase B: Staging Node Hardware Profiling (Next Milestone)
1. **Target**: Provision an isolated 2-vCPU / 4-GB RAM Linux staging node to execute standard-hardware thermal profiling (`BLK-05`).
2. **Evaluation Protocol**: Run $N=100$ sequential and concurrent benchmark passes under steady-state thermal conditions to certify sustained P95 latency $\le 2,500$ ms and peak RAM $\le 2,048$ MB.
3. **Artifact Output**: Deposit `experiments/artifacts/staging_hardware_profiling_report.json`.

### Phase C: Operational Governance & Human Reviewer Onboarding
1. **DPO & Legal Review**: Execute formal Data Protection Agreement (DPA) Addendum (`BLK-01`).
2. **Reviewer Panel Onboarding**: Roster 5 qualified support specialists and execute initial double-blind pilot cohort labeling to freeze prospective ground truth (`BLK-03`).
3. **Infrastructure Provisioning**: Deploy shadow service container behind internal VPC proxy with live Prometheus metrics (`BLK-02`).
4. **CRM Dispatch Integration**: Configure webhook endpoint with authenticated signing secrets and DLQ retry policy (`BLK-07`).

### Phase D: Shadow Pilot Activation Gate
- Activate shadow mirroring **only after** all 7 blocker gates (BLK-01 through BLK-07) have verified real evidence.
- Maintain `nexus-qlora-v1` as the active customer-facing production model throughout shadow execution until formal executive promotion approval.
