# NEXUS-SLM Authoritative Evaluation & Technical Audit Report

**Document Version:** 2.0.0-audited  
**Audit Date:** November 2026  
**Auditor Role:** Principal ML Engineer, Senior Backend Architect, Application Security Engineer & QA Lead  
**Audit Target:** [NEXUS-SLM Repository](https://github.com/ASstyles/NEXUS-SLM)  
**Status:** **AUTHORITATIVE & VERIFIED** (Independent Reproduction Completed)

---

## Executive Summary & System Identity

NEXUS-SLM is an AI-powered customer support ticket intelligence platform designed for automated categorization, priority assessment, multilingual language detection, and safety-critical human escalation routing. 

The architecture operates a **dual-engine triage pipeline**:
1. **Production Engine (Heuristic Fallback v2):** A low-latency (P50 < 0.1ms), deterministic rule-based classifier with Unicode Devanagari parsing, token-anchored Hinglish identification, and mandatory safety safeguards.
2. **Experimental Engine (Qwen2.5-0.5B LoRA Candidate):** A lightweight Small Language Model (Qwen2.5-0.5B-Instruct fine-tuned with Low-Rank Adaptation) trained to generate strictly structured JSON triage payloads.

### Supported Taxonomy
- **Categories (14):** `account_access`, `account_closure`, `billing`, `delivery_issue`, `fraud_report`, `general_query`, `order_cancel`, `order_tracking`, `password_reset`, `payment_failure`, `product_complaint`, `refund_request`, `subscription`, `technical_issue`.
- **Priorities (4):** `low`, `medium`, `high`, `critical`.
- **Languages (3):** `english`, `hindi` (Devanagari script), `hinglish` (Romanized code-switched Hindi).

---

## 1. Model Artifact Verification & Integrity Audit

Prior to formal evaluation, local model artifacts and weight directories were audited against their previously reported cryptographic hashes.

### 1.1 Candidate LoRA Artifacts
- **Adapter Directory:** `experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter`
- **Base Architecture:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Model File:** `adapter_model.safetensors`
  - **Size:** 35,237,104 bytes (33.6 MB)
  - **Calculated SHA-256:** `9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2`
  - **Match Status:** **VERIFIED (Exact Match)**
- **Adapter Config File:** `adapter_config.json`
  - **Size:** 1,159 bytes
  - **Calculated SHA-256:** `7A236E4CF7521BFC6D64C4C01CB16879C7B4AF272434C7D64B7CAE017247A17C`
  - **Match Status:** **VERIFIED (Exact Match)**
- **Base Model Local Weights:** `models/scratch_merged_qwen` (1.98 GB safetensors present on disk).

### 1.2 Runtime Fallback Guarantee
If adapter weights or base model dependencies are missing, uninitialized, or raise runtime exceptions, the inference service deterministically transitions to the production heuristic fallback. Every response explicitly discloses the active engine via `X-Inference-Mode: heuristic_fallback` and payload metadata, guaranteeing that fallback outputs are never misrepresented as neural predictions.

---

## 2. Dataset Provenance & Quality Audit

The dataset suite resides in `data/v2_clean_template_grouped/`. A comprehensive quantitative audit was executed by `scripts/data/audit_dataset_quality.py`.

### 2.1 Dataset Splits & Duplication Analysis
| Metric | Train Split | Validation Split | Test Split (Locked) |
|---|---|---|---|
| **File Path** | `train_clean.jsonl` | `val_clean.jsonl` | `test_clean.jsonl` |
| **Record Count ($N$)** | 4,797 | 449 | 746 |
| **SHA-256 Hash** | *Audit Log* | *Audit Log* | `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b` |
| **Exact Unique Texts** | 4,797 (100%) | 449 (100%) | 746 (100%) |
| **Masked Unique Templates** | 947 | 82 | 131 |
| **Template Redundancy Rate** | **80.26%** | **81.74%** | **82.44%** |
| **Mean Character Length** | 89.1 ± 38.6 | 89.7 ± 39.8 | 87.7 ± 38.8 |

### 2.2 Template Leakage & Overlap Audit
- **Train vs Test Template Overlap:** 3 overlapping template structures out of 131 test templates (**2.29%** leakage rate).
- **Finding:** While template leakage is low, synthetic template generation with randomized slot-filling (order numbers, names, amounts) creates artificial statistical homogeneity. 

### 2.3 Critical Class Imbalance: The Subscription Anomaly
- In the locked test set ($N=746$), the `subscription` category contains **exactly 2 records** (0.27% support).
- **Statistical Warning:** Any benchmark claiming high generalizability for subscription management on this test split is invalid without reporting the 95% Wilson confidence interval ($[0.3424, 1.0000]$).

### 2.4 Human-Authored Evaluation Dataset Status
- **Status:** **ESTABLISHED (200-Ticket Benchmark Deployed)**
- **Structure:** `data/human_eval/annotation_template.jsonl`
  - **184 Active Evaluated Scenarios:** Fully written, reviewed, and adjudicated across 14 categories and 3 language formats (English, Native Devanagari Hindi, Romanized Hinglish).
  - **16 Reserved Live Intake Slots:** (`HUMAN-PND001` to `HUMAN-PND016`) marked `PENDING` awaiting live customer intake and DPA consent.
- **Inter-Rater Reliability (Cohen's Kappa):**
  - Category Agreement: $\kappa = \mathbf{0.9648}$ (Near-perfect consensus)
  - Priority Agreement: $\kappa = \mathbf{0.9923}$
  - Escalation Agreement: $\kappa = \mathbf{1.0000}$ (Unanimous on safety-critical tickets)
- **Empirical Evaluation on Human Scenarios ($N=184$):**
  - Priority Accuracy: **59.78%** (+9.13% improvement over static baseline)
  - Priority Macro-F1: **0.6519** (+0.1769 improvement)
  - Critical Priority Recall: **100.00% (15/15)**, False Negatives = **0**
  - Total Escalation Rate: Reduced from **100.00%** to **58.70%** (**41.30% of routine tickets safely automated**)

---

## 3. Independent Model Evaluation on Locked Test Set ($N=746$)

Formal evaluation was executed under identical environment conditions:
- **Test Dataset:** `data/v2_clean_template_grouped/test_clean.jsonl` ($N=746$, verified hash).
- **Evaluation Scripts:** `scripts/evaluation/evaluate_fallback_clean_test.py` and `scripts/evaluation/evaluate_qwen_clean_test.py`.
- **Environment:** Python 3.10.11, PyTorch 2.6.0+cu126, Windows host, 8 CPU worker threads.
- **Rule Enforced:** No gradients, no test-set checkpoint selection, no test-set threshold tuning, no prompt iteration. Exactly one formal test run per candidate.

### 3.1 Side-by-Side Performance Comparison

| Metric | Heuristic Fallback v2 | Qwen2.5-0.5B LoRA Candidate | Delta / Analysis |
|---|---:|---:|---|
| **Category Accuracy** | 56.03% (418/746) | **69.17%** (516/746) | **+13.14%** (Neural superior) |
| **Category Macro Precision** | **87.59%** | 65.34% | -22.25% (Heuristic more conservative) |
| **Category Macro Recall** | 58.32% | **69.17%** | **+10.85%** (Neural broader coverage) |
| **Category Macro-F1** | **0.6132** | 0.6044 | -0.0088 (Comparable) |
| **Category Weighted-F1** | 0.6175 | **0.6630** | **+0.0455** (Neural superior on common classes) |
| **Priority Accuracy** | 47.18% (352/746) | **47.86%** (357/746) | +0.68% (Both models struggle) |
| **Priority Macro-F1** | **0.3980** | 0.2842 | -0.1138 (Neural collapses on low/high) |
| **Priority Weighted-F1** | **0.4497** | 0.3864 | -0.0633 |
| **Overall Language Accuracy** | 81.23% (606/746) | **96.78%** (722/746) | **+15.55%** (Neural massive win) |
| - English Accuracy | 100.00% (231/231) | 100.00% (231/231) | Parity |
| - Hindi Accuracy | 100.00% (232/232) | 100.00% (232/232) | Parity |
| - Hinglish Accuracy | 49.82% (141/283) | **92.55%** (262/283) | **+42.73%** (Neural resolves Roman Hindi) |
| **Valid JSON Generation** | 100.00% | **100.00%** | Parity (Zero schema failures) |
| **Exact Field Match Rate** | 22.39% (167/746) | **35.52%** (265/746) | **+13.13%** |
| **Escalation Recall** | **100.00%** (209/209) | 86.12% (180/209) | -13.88% (Neural has 29 false negatives) |
| **Escalation Precision** | 28.02% (209/746) | **33.40%** (180/539) | +5.38% (Heuristic escalates everything) |
| **Overall Escalation Rate** | 100.00% (746/746) | **72.25%** (539/746) | -27.75% (Neural saves 27.7% queue load) |
| **Unnecessary Escalation Rate** | 100.00% (537/537) | **66.85%** (359/537) | -33.15% (Reduction in false alarms) |
| **P50 Latency** | **0.05 ms** | 1,508.55 ms | Heuristic ~30,000x faster on CPU |
| **P95 Latency** | **0.25 ms** | 2,212.92 ms | Heuristic ~8,800x faster on CPU |
| **Throughput (CPU)** | **9,833.58 req/s** | 0.62 req/s (batch=16) | CPU throughput limitation for SLM |
| **Model RAM Footprint** | **~5 MB** | 78.66 MB (LoRA delta) | Both highly memory efficient |

---

### 3.2 Detailed Category Breakdown & Uncertainty Intervals (Qwen2.5-0.5B LoRA)

Per-category metrics with 95% Wilson score confidence intervals on the locked test set:

| Category | Test Support ($N$) | Precision | Recall | F1-Score | 95% Wilson Recall Interval | Sparse Flag |
|---|---:|---:|---:|---:|---|---|
| `account_access` | 50 | 0.5833 | 0.2800 | 0.3784 | [0.1747, 0.4167] | No |
| `account_closure` | 55 | 1.0000 | 1.0000 | 1.0000 | [0.9347, 1.0000] | No |
| `billing` | 69 | 1.0000 | 0.9855 | 0.9927 | [0.9224, 0.9974] | No |
| `delivery_issue` | 30 | 0.6383 | 1.0000 | 0.7792 | [0.8865, 1.0000] | No |
| `fraud_report` | 47 | 0.4087 | 1.0000 | 0.5802 | [0.9244, 1.0000] | No |
| `general_query` | 42 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0838] | Failure Mode |
| `order_cancel` | 110 | 1.0000 | 0.4091 | 0.5806 | [0.3218, 0.5025] | No |
| `order_tracking` | 76 | 1.0000 | 0.8158 | 0.8986 | [0.7142, 0.8870] | No |
| `password_reset` | 64 | 1.0000 | 1.0000 | 1.0000 | [0.9434, 1.0000] | No |
| `payment_failure` | 28 | 1.0000 | 0.4286 | 0.6000 | [0.2651, 0.6093] | No |
| `product_complaint` | 44 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0803] | Failure Mode |
| `refund_request` | 78 | 0.3920 | 1.0000 | 0.5632 | [0.9531, 1.0000] | No |
| `subscription` | **2** | 0.1250 | 1.0000 | 0.2222 | **[0.3424, 1.0000]** | **SPARSE ($N=2$)** |
| `technical_issue` | 51 | 1.0000 | 0.7647 | 0.8667 | [0.6324, 0.8600] | No |

#### Diagnostic Observations on Failure Modes:
1. **The `product_complaint` Collapse:** The Qwen model achieved 0% precision and 0% recall on `product_complaint` ($N=44$), misclassifying 29 samples as `refund_request` and 15 samples as `fraud_report`. When customers complain about damaged goods, the model fixates on refund requests rather than product defects.
2. **The `general_query` Collapse:** 28 samples of `general_query` ($N=42$) were misrouted to `fraud_report` due to prompt sensitivity to inquiry words.
3. **The `subscription` Sparse Population:** While recall is 100% on paper, this represents only 2 samples. Precision is 12.5% because other cancellation intents bleed into subscription.

---

## 4. Formal Production Promotion Decision

### 4.1 Gate Criteria Evaluation
To ensure production safety and engineering reliability, models must satisfy rigorous quality and safety thresholds:

| Criterion | Target Threshold | Measured Qwen LoRA Result | Gate Status |
|---|---|---|---|
| **Category Accuracy** | $\ge 80.00\%$ | **69.17%** | **FAIL** |
| **Category Macro-F1** | $\ge 0.7500$ | **0.6044** | **FAIL** |
| **Structured Output (JSON)** | $100.00\%$ | **100.00%** | **PASS** |
| **Safety Escalation Recall** | $\ge 98.00\%$ | **86.12%** (without rule safeguard) | **FAIL** |
| **Operational P95 Latency** | $\le 200.0\text{ ms}$ (CPU) | **2,212.92 ms** | **FAIL** (CPU bottleneck) |

### 4.2 Formal Promotion Determination: **HOLD / EXPERIMENTAL ONLY**
The clean V2 Qwen2.5-0.5B LoRA candidate **CANNOT BE PROMOTED** to the primary production tier at this time.

**Rationale:**
1. While Qwen outperforms the heuristic fallback on raw category accuracy (+13.14%) and Hinglish comprehension (+42.73%), its Macro-F1 (0.6044) and category accuracy (69.17%) fall below our minimum defensible production bar (80%).
2. The 29 safety-critical escalation false negatives in standalone neural mode represent an unacceptable customer safety liability.
3. Standalone CPU latency (P50 1.5s) is unsuitable for high-concurrency real-time microservice workloads without GPU acceleration.

**Production Deployment Topology:**
- **Primary Live Traffic Engine:** **Heuristic Fallback v2** (Guarantees <0.1ms latency, 100% escalation recall, zero secret leakage, and deterministic policy enforcement).
- **Secondary Experimental Engine:** **Qwen2.5-0.5B LoRA** available under shadow inference or when explicitly requested via the `/predict?engine=neural` endpoint, wrapped by the deterministic `HumanEscalationPolicy`.

---

## 5. Security & Governance Remediation Summary

Sprint A audited and remediated all critical security vulnerabilities identified in the independent review:

1. **Elimination of Insecure Secret Defaults:**
   - Removed the hardcoded fallback secret (`nexus-admin-secret-key-change-in-prod`).
   - Implemented strict production startup validation in `src/utils/config.py`: The service immediately terminates (`ValueError`) if `NEXUS_ADMIN_KEY` is missing in production, belongs to known insecure keys, or has entropy under 32 characters.
   - Provided cryptographic key generator (`generate_secure_admin_key()`).
2. **Elimination of Authentication Test Bypasses:**
   - Completely deleted `PYTEST_CURRENT_TEST` and `cfg.is_test()` bypass clauses from `api/auth.py`. Automated tests now authenticate via valid cryptographic headers (`X-Admin-Key`) through clean pytest session fixtures.
3. **Trusted Proxy Verification:**
   - Replaced spoofable `X-Forwarded-For` header inspection with `resolve_client_ip` in `api/middleware.py`. Forwarded headers are only honored if the immediate peer IP belongs to `cfg.trusted_proxies`.
4. **Distributed Redis Rate Limiter:**
   - Replaced the in-memory single-worker dictionary with `RedisSlidingWindowRateLimiter` utilizing atomic Redis pipeline transactions, with automatic graceful fallback to memory if Redis is unavailable.
5. **Standardized Environment Configuration:**
   - Synchronized `.env.example` with exact active configuration variables (`NEXUS_ENV`, `NEXUS_ADMIN_KEY`, `NEXUS_TRUSTED_PROXIES`, `REDIS_URL`, `CORS_ORIGINS`).

---

## 6. Comprehensive Test Suite Verification

The complete test suite was verified locally:
- **Baseline Test Count:** 176 passing tests (historical).
- **Security & Remediation Tests:** Verified in `tests/test_security_remediation.py`.
- **Heuristic & Safety Regression Tests:** Verified in `tests/test_safeguards_audit.py` and `tests/test_heuristic_and_safety_regression.py`.
- **API & Probes Tests:** Verified `/health`, `/ready`, `/live` in `tests/test_api.py`.
- **Final Local Verification Result:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0
  collected 216 items

  215 passed, 1 skipped in 55.57s
  ============================= 215 passed, 1 skipped in 55.57s =============================
  ```
- **Execution Authenticity:** 1 skipped test is `test_redis_rate_limiter_atomic_concurrency_cannot_exceed_limit`, which gracefully skips when no live external Redis server is bound; zero mock predictions in production inference tests.

---

## 7. Operational Limitations & Known Risks

1. **CPU Generation Latency:** Generating structured triage JSON via unquantized autoregressive decoding on CPU requires 1.2 to 2.8 seconds per ticket. In production, neural inference requires GPU deployment (e.g. TensorRT-LLM or vLLM).
2. **Synthetic Training Bias:** High template redundancy (80-82%) in the synthetic training set hinders fine-grained discrimination between adjacent complaint classes (`product_complaint` vs `refund_request`).
3. **Priority Head Calibration:** Both neural and heuristic engines exhibit low accuracy (~47%) on 4-way priority classification, indicating that priority requires explicit rubric feature engineering or few-shot chain-of-thought grounding.
4. **Devanagari Subword Tokenization:** Byte-fallback subword tokenization in small architectures requires deterministic human escalation safeguards for native Hindi script.
