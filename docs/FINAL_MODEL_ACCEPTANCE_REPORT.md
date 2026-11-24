# NEXUS-SLM — Final Model Acceptance and Evidence Reconciliation Report

**Date**: 2026-10-31  
**Branch**: `sprint/clean-data-model-improvement`  
**Evaluated Candidate**: `qwen25_05b_clean_v2_lora`  
**Base Architecture**: `Qwen/Qwen2.5-0.5B-Instruct` (494M parameters, 151,643 vocabulary)  
**Production Baseline**: `nexus-qlora-v1` (Heuristic Fallback + Policy Safeguards)  
**Dataset**: `data/v2_clean_template_grouped/test_clean.jsonl` ($N=746$) & `val_clean.jsonl` ($N=449$)  
**Status**: **FROZEN EXPERIMENTAL CANDIDATE (PROMOTION REJECTED — RETAIN BASELINE IN PRODUCTION)**  

---

## 1. Executive Summary & Verification Scope

This report delivers the authoritative, empirically grounded model acceptance evaluation and evidence reconciliation for `qwen25_05b_clean_v2_lora`, trained on an NVIDIA Tesla T4 cloud GPU. 

### Core Verification Findings:
1. **Cloud GPU Execution Independently Verified**: Confirmed execution of Kaggle kernel `aviralsangal12/nexus-qwen25-clean-v2-training` on an NVIDIA Tesla T4 GPU (14.56 GB VRAM). Training executed for exactly 1,200 optimization steps (2.0 epochs, 1,585.32s) achieving train loss `0.0731` and validation loss `0.06945`. The test dataset ($N=746$) was strictly quarantined (0% exposure to training or checkpoint selection).
2. **Adapter Cryptographic Integrity Confirmed**: Recomputed SHA-256 for `adapter_model.safetensors` (`9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2`, 35,237,104 bytes) and `adapter_config.json` (`7A236E4CF7521BFC6D64C4C01CB16879C7B4AF272434C7D64B7CAE017247A17C`). Live loading and multi-lingual inference verified on CPU.
3. **Threshold Reconciliation on Validation Set ($N=449$)**: The candidate achieved **79.96%** (359 / 449 correct) validation category accuracy. The strict machine-readable acceptance gate requires $\ge 80.0\%$. The candidate missed the threshold by **0.04%** (exactly 1 sample). Rather than artificially lowering the threshold, this failure is recorded transparently as **FAIL** against the strict 80.0% validation gate.
4. **Full Test Set Evaluation ($N=746$)**: Evaluated on the locked clean test set (`test_clean.jsonl`), the candidate achieved **69.17%** category accuracy (516 / 746 correct) and **0.6044** Macro-F1, outperforming the production baseline (61.80% accuracy, 0.5473 Macro-F1). However, 69.17% is substantially below the 80.0% production target.
5. **Safety & Escalation Analysis**: While the candidate achieved **100.0% JSON validity** and **96.78% language accuracy**, the deterministic policy escalation recall was **87.56%** with **26 high-risk false negatives**, failing the zero-false-negative gate required for autonomous unmonitored production triage.
6. **Confidence Calibration Audit**: ECE is **0.2851** (uncalibrated) and **0.2712** (temperature scaled $T=1.15$). All 746 generated sequences clustered in the 0.90–1.00 confidence bin (mean confidence 0.9768 vs accuracy 0.6917). Because sequence token probabilities do not represent a true 14-class posterior and no independent calibration holdout set existed, automated confidence-based routing **cannot be certified**.
7. **Production Gate Decision**: **RETAIN `nexus-qlora-v1` IN PRODUCTION**. The candidate is preserved as a verified experimental artifact, but rejected for production promotion.

---

## 2. Remote GPU Execution Verification

The Kaggle remote GPU training job was audited using raw execution telemetry, environment inspection, and build artifacts:

| Parameter | Reported Value | Verified Remote Value | Audit Verdict |
|---|---|---|:---:|
| **Kernel Slug** | `aviralsangal12/nexus-qwen25-clean-v2-training` | `aviralsangal12/nexus-qwen25-clean-v2-training` | **VERIFIED** |
| **Accelerator** | NVIDIA Tesla T4 | NVIDIA Tesla T4 (14.56 GB VRAM, sm_75) | **VERIFIED** |
| **Operating System** | Linux (Ubuntu) | Linux 6.18 / Python 3.13.15 / PyTorch 2.11.0+cu128 | **VERIFIED** |
| **Optimization Steps** | 1,200 | 1,200 steps (2.0 epochs, batch size 2, accum 4) | **VERIFIED** |
| **Compute Duration** | 1,585.32 seconds | 1,585.32 seconds (Total kernel wall clock: 2,680.8s) | **VERIFIED** |
| **Training Loss** | 0.0731 | 0.0731 (Initial loss: 2.842 -> 0.0731) | **VERIFIED** |
| **Validation Loss** | 0.06945 | 0.06945 (Step 1200 best checkpoint) | **VERIFIED** |
| **Dataset Isolation** | Test Set Quarantined | `test_clean.jsonl` quarantined (0% training exposure) | **VERIFIED** |
| **Remote Exit Status** | Success | Process exited code 0; zip archive generated | **VERIFIED** |

---

## 3. Downloaded Adapter Verification & Live Inference

The model artifacts downloaded from Kaggle into `experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter/` were inspected and cryptographically validated:

### 3.1 Cryptographic File Checksums
| File Name | File Size | Recomputed SHA-256 Checksum | Expected Checksum | Status |
|---|:---:|---|---|:---:|
| `adapter_model.safetensors` | 35,237,104 bytes | `9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2` | Exact match | **MATCH** |
| `adapter_config.json` | 1,159 bytes | `7A236E4CF7521BFC6D64C4C01CB16879C7B4AF272434C7D64B7CAE017247A17C` | Exact match | **MATCH** |
| `candidate_metadata.json` | 503 bytes | `5D092E6C4C824E564757C85390F57EB57422E1AC41C47053531DA1696E3D4822` | Exact match | **MATCH** |
| `tokenizer.json` | 11,421,896 bytes | `625BDD9E45E20EE4D69B4B8FEBE99EFA9A55C8E75C460C45C5162E0CEBD4E65D` | Valid Qwen2.5 | **MATCH** |

### 3.2 Live Inference Verification
The adapter was loaded into PyTorch `PeftModel` on base model `Qwen/Qwen2.5-0.5B-Instruct` and evaluated across multiple natural user requests:

1. **Hinglish Financial Dispute**:
   - Input: `"Mera payment cut ho gaya lekin course nahi mila"`
   - Output: `{"category": "payment_failure", "priority": "critical", "language": "hinglish", "requires_human": true}`
   - Result: Valid JSON, correct category, correctly marked for human escalation, confidence 0.9719.
2. **English Technical Query**:
   - Input: `"App keeps crashing on payment checkout screen after recent update."`
   - Output: `{"category": "technical_issue", "priority": "high", "language": "english", "requires_human": true}`
   - Result: Valid JSON, correct category, priority mapped accurately.
3. **Devanagari Hindi Refund Request**:
   - Input: `"मेरा रिफंड अभी तक बैंक खाते में नहीं आया है, कृपया सहायता करें।"`
   - Output: `{"category": "refund_request", "priority": "high", "language": "hindi", "requires_human": true}`
   - Result: Valid JSON, correct category, Hindi script recognized accurately.

---

## 4. Reconciled Acceptance Thresholds & Policy

The committed repository acceptance specification and promotion reports establish the following machine-readable quality gates:

| Acceptance Gate | Required Threshold | Validation Set Result ($N=449$) | Full Test Set Result ($N=746$) | Acceptance Gate Verdict |
|---|:---:|:---:|:---:|:---:|
| **Category Accuracy** | $\ge 80.0\%$ | **79.96%** (359 / 449) | **69.17%** (516 / 746) | **FAIL** (Missed on both splits) |
| **Category Macro-F1** | $\ge 0.7500$ | **0.8103** | **0.6044** | **CONDITIONAL PASS** (Val passed, Test failed) |
| **JSON Validity Rate** | $\ge 95.0\%$ | **100.0%** (449 / 449) | **100.0%** (746 / 746) | **PASS** |
| **Language Accuracy** | $\ge 95.0\%$ | **100.0%** (449 / 449) | **96.78%** (722 / 746) | **PASS** |
| **High-Risk Escalation Recall**| $100.0\%$ | **100.0%** | **87.56%** | **FAIL** (Test set had 26 FNs) |
| **High-Risk False Negatives** | **0 (Zero)** | **0** | **26** | **FAIL** (Zero-tolerance breached) |

### Evidence Reconciliation Note:
- The reported validation accuracy was **79.96%** (359/449). A threshold of 80.00% requires $\ge 360/449$ (80.18%).
- While missing by a single validation sample (0.04%), an engineering acceptance gate must be maintained objectively without retroactive modification.
- On the full test set ($N=746$), accuracy drops to **69.17%**, confirming that the model has not generalized to the 80.0% threshold across unseen templates.

---

## 5. Quarantined Clean Test Set ($N=746$) Empirical Evaluation

Both the production baseline (`nexus-qlora-v1`) and frozen candidate (`qwen25_05b_clean_v2_lora`) were evaluated against the exact same 746 records of `data/v2_clean_template_grouped/test_clean.jsonl`.

### 5.1 Head-to-Head Comparison Scorecard
| Metric Dimension | Production Baseline (`nexus-qlora-v1`) | Frozen Candidate (`qwen25_05b_clean_v2_lora`) | Delta ($\Delta$) |
|---|:---:|:---:|:---:|
| **Evaluated Sample Count** | 746 (100% Split) | 746 (100% Split) | Identical |
| **Category Accuracy** | **61.80%** (461 / 746) | **69.17%** (516 / 746) | **+7.37%** (+55 tickets) |
| **Category Macro-F1** | **0.5473** | **0.6044** | **+0.0571** |
| **Category Weighted-F1** | **0.6356** | **0.6630** | **+0.0274** |
| **Priority Accuracy** | **50.54%** | **45.44%** | -5.10% |
| **Priority Macro-F1** | **0.2974** | **0.3175** | +0.0201 |
| **Language Accuracy** | **99.87%** (745 / 746) | **96.78%** (722 / 746) | -3.09% |
| — English Accuracy | 100.0% (246 / 246) | 100.0% (246 / 246) | 0.0% |
| — Hindi (Devanagari) Accuracy | 100.0% (178 / 178) | 100.0% (178 / 178) | 0.0% |
| — Hinglish (Roman Hindi) Accuracy| 99.69% (321 / 322) | 92.55% (298 / 322) | -7.14% |
| **Exact 4-Field Match Rate** | 7.51% (56 / 746) | 16.89% (126 / 746) | **+9.38%** |
| **JSON Validity Rate** | **100.0%** (746 / 746) | **100.0%** (746 / 746) | 0.0% |
| **Policy Escalation Recall** | **100.0%** (209 / 209) | **87.56%** (183 / 209) | -12.44% |
| **High-Risk False Negatives** | **0 (Zero)** | **26** | **+26 FNs (Breach)** |
| **Latency P50** | **4.62 ms** (Heuristic) | **1,048.17 ms** (Neural CPU) | +1,043.55 ms |
| **Latency P95** | **5.86 ms** (Heuristic) | **1,386.15 ms** (Neural CPU) | +1,380.29 ms |
| **Throughput (CPU)** | **204.64 req/sec** | **0.93 req/sec** | -203.71 req/sec |
| **Peak RSS RAM** | **288.66 MB** | **2,671.21 MB** | +2,382.55 MB |

### 5.2 Per-Category Performance Breakdown ($N=746$)
| Category | Support ($N$) | Candidate Precision | Candidate Recall | Candidate F1 | Baseline F1 | $\Delta$ F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `billing` | 69 | **1.0000** | **0.9855** | **0.9927** | 1.0000 | -0.0073 |
| `password_reset` | 64 | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 0.0000 |
| `account_closure` | 55 | **1.0000** | **1.0000** | **1.0000** | 0.0000 | **+1.0000** |
| `order_tracking` | 76 | **1.0000** | **0.8158** | **0.8986** | 0.6667 | **+0.2319** |
| `technical_issue` | 51 | **1.0000** | **0.7647** | **0.8667** | 0.5000 | **+0.3667** |
| `delivery_issue` | 30 | **0.6383** | **1.0000** | **0.7792** | 0.6176 | **+0.1616** |
| `payment_failure` | 28 | **1.0000** | **0.4286** | **0.6000** | 0.0667 | **+0.5333** |
| `fraud_report` | 47 | **0.4087** | **1.0000** | **0.5802** | 1.0000 | -0.4198 |
| `order_cancel` | 110 | **1.0000** | **0.4091** | **0.5806** | 0.6452 | -0.0646 |
| `refund_request` | 78 | **0.3920** | **1.0000** | **0.5632** | 0.8271 | -0.2639 |
| `account_access` | 50 | **0.5833** | **0.2800** | **0.3784** | 0.6301 | -0.2517 |
| `subscription` | 2 | **0.1250** | **1.0000** | **0.2222** | 0.0000 | **+0.2222** |
| `product_complaint` | 44 | **0.0000** | **0.0000** | **0.0000** | 0.4348 | -0.4348 |
| `general_query` | 42 | **0.0000** | **0.0000** | **0.0000** | 0.2736 | -0.2736 |

---

## 6. Confidence Calibration & Reliability Audit

### 6.1 Audit Scope & Score Definition
- **Score Definition**: The candidate's reported confidence represents the **mean token generation probability** of the auto-regressive ChatML JSON output string.
- **Categorical Posterior Status**: The confidence score **does not represent a categorical posterior probability distribution** across the 14 support categories.
- **Holdout Calibration Set**: **None existed**. The temperature scaling parameter ($T=1.15$) was statically preset rather than fitted via negative log-likelihood on a held-out calibration split.

### 6.2 Empirical Calibration Metrics ($N=746$)
| Metric | Raw Confidence | Temperature Scaled ($T=1.15$) | Benchmark Standard |
|---|:---:|:---:|:---:|
| **Expected Calibration Error (ECE)** | **0.2851** | **0.2712** | $\le 0.1000$ (Severe Overconfidence) |
| **Maximum Calibration Error (MCE)** | **0.2851** | **0.2712** | $\le 0.1500$ |
| **Brier Score** | **0.2933** | **0.2852** | $\le 0.1500$ |
| **Mean Model Confidence** | **0.9768** | **0.9629** | Empirical Accuracy: **0.6917** |
| **Calibration Gap** | **+28.51%** | **+27.12%** | Overconfidence Bias |

### 6.3 Reliability Diagram & Bin Distribution (10 Bins)
| Confidence Bin | Sample Count | Mean Confidence | Empirical Accuracy | Calibration Gap |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 – 0.80 | 0 | — | — | 0.0000 |
| 0.80 – 0.90 | 0 | — | — | 0.0000 |
| **0.90 – 1.00** | **746 (100%)** | **0.9768** | **0.6917** | **+0.2851** |

### 6.4 Selective Prediction & Risk-Coverage Analysis
Evaluating accuracy under varying confidence acceptance thresholds:

| Confidence Threshold | Retained Samples | Coverage Rate | Filtered Accuracy | Error Rate |
|:---:|:---:|:---:|:---:|:---:|
| $\ge 0.50$ | 746 | 100.0% | 69.17% | 30.83% |
| $\ge 0.70$ | 746 | 100.0% | 69.17% | 30.83% |
| $\ge 0.85$ | 746 | 100.0% | 69.17% | 30.83% |
| $\ge 0.95$ | 746 | 100.0% | 69.17% | 30.83% |

**Audit Conclusion**: Because all 746 generated sequences have confidence scores $> 0.95$, threshold filtering achieves zero selective coverage separation. The confidence score cannot distinguish correct classifications from misclassifications. Therefore, confidence **must remain designated as `uncalibrated_heuristic`**, and automated straight-through routing cannot be certified.

---

## 7. Model Acceptance Verdict Matrix

| Dimension | Standard / Gate | Candidate Result | Final Acceptance Verdict |
|---|---|---|:---:|
| **1. GPU Training Authenticity** | NVIDIA Tesla T4, 1200 steps, logs verified | Fully verified from raw Kaggle execution telemetry | **PASS** |
| **2. Artifact & Cryptographic Integrity**| Exact SHA-256 matches, safe archive extraction | Bit-exact matching checksums; live loading verified | **PASS** |
| **3. Category Accuracy (Val $N=449$)** | $\ge 80.0\%$ | 79.96% (359 / 449 correct) | **FAIL** (-0.04% / -1 sample) |
| **4. Category Accuracy (Test $N=746$)**| $\ge 80.0\%$ | 69.17% (516 / 746 correct) | **FAIL** (-10.83%) |
| **5. Category Macro-F1 (Val)** | $\ge 0.7500$ | 0.8103 | **PASS** (+0.0603) |
| **6. Category Macro-F1 (Test)** | $\ge 0.7500$ | 0.6044 | **FAIL** (-0.1456) |
| **7. Structured Output Validity** | $\ge 95.0\%$ valid JSON schema | 100.0% (746 / 746 valid JSON) | **PASS** |
| **8. Multi-Lingual Identification** | $\ge 95.0\%$ language accuracy | 96.78% (100% EN, 100% HI, 92.55% Hinglish) | **PASS** |
| **9. Safety Escalation Recall** | $100.0\%$ recall on high-risk tickets | 87.56% (26 high-risk false negatives) | **FAIL** |
| **10. High-Risk False Negatives** | 0 (Zero Tolerance) | 26 false negatives | **FAIL** |
| **11. Confidence Calibration** | Calibrated posterior, ECE $\le 0.10$ | Sequence token probability; ECE 0.2851; uncalibrated | **CONDITIONAL PASS** (Requires heuristic flag) |
| **12. Latency & Resource Footprint** | P95 $\le 2,500$ ms, RAM $\le 4,096$ MB | P95 = 1,386 ms, Peak RAM = 2,671 MB | **PASS** |

### Final Model Acceptance Recommendation:
**RETAIN PRODUCTION BASELINE (`nexus-qlora-v1`)**. While `qwen25_05b_clean_v2_lora` provides a demonstrable classification improvement over the production baseline (+7.37% accuracy, +0.0571 Macro-F1, +1.0000 on account closure), it fails three non-negotiable production criteria:
1. Category accuracy on the test set is 69.17% (below the 80.0% threshold).
2. High-risk safety recall drops from 100.0% to 87.56% (26 false negatives).
3. Confidence scores are uncalibrated (ECE = 0.2851) and cannot safely govern straight-through automated processing.
The candidate is preserved in the experiment registry for research and quantization refinement, but blocked from production cutover.

