# NEXUS-SLM — Final Independent Verification and Release Gate Audit

**Audit Date**: October 9, 2026  
**Auditor**: Independent Release Gate Audit & Verification Suite  
**Candidate Evaluated**: NEXUS-SLM-135M-LoRA (`nexus-qlora-v1`)  
**Base Architecture**: `HuggingFaceTB/SmolLM-135M-Instruct` (135M parameters)  
**Adapter Path**: `models/nexus_qlora_v1` (LoRA $r=16, \alpha=32$)  
**Target Taxonomy**: 14 Support Categories, 4 Priority Levels, 3 Language Modalities  

---

## 1. Executive Summary & Release Gate Decision

### Final Release Decision: **READY WITH CONDITIONS**

The independent verification and release-gate phase for **NEXUS-SLM** has concluded. Rather than relying on documented assertions, all verifiable claims, test suites, cryptographic manifests, evaluation math, safeguards, and performance benchmarks were independently executed and evaluated on the actual codebase.

The candidate **NEXUS-SLM-135M-LoRA (`nexus-qlora-v1`)** satisfies all core operational, cryptographic, and safety criteria. However, because Docker container execution on a Linux host remains pending remote CI execution (the local Windows host lacks a Docker daemon), and because native Devanagari Hindi tickets require mandatory human escalation due to byte-level token fragmentation, the system cannot be designated as unconditionally `READY`. It is certified as **READY WITH CONDITIONS**.

```
+-------------------------------------------------------------------------------+
|                           RELEASE GATE DECISION                               |
|                                                                               |
|                      >>>  READY WITH CONDITIONS  <<<                          |
|                                                                               |
|  - Automated Test Suite:       78 / 78 PASSED (100%)                          |
|  - Cryptographic Manifest:     7 / 7 SHA256 Hashes VERIFIED                   |
|  - Native Hindi Routing:       DETERMINISTIC FALLBACK IMPLEMENTED & VERIFIED  |
|  - Confidence Calibration:     UNRELIABLE HEURISTIC DECLARED & GATED          |
|  - Critical Safeguards:        DETERMINISTIC RULES TESTED & PROVEN            |
|  - Docker Linux CI Run:        PENDING (Awaiting GitHub Actions Execution)    |
|  - Concurrency Guideline:      1-2 workers per CPU replica (Horizontal Scale) |
+-------------------------------------------------------------------------------+
```

---

## 2. Evidence Verification Ledger

This audit strictly distinguishes **independently reproduced results** from claims that are **supported only by existing artifacts**.

| Verification Area | Status | Verification Method | Empirical Findings & Discrepancies |
|---|:---:|:---:|---|
| **Automated Test Suite** | **REPRODUCED** | Executed via pytest in `.venv` | **78 / 78 tests passed** (0 failures, 9 Starlette deprecation warnings). Covers API contracts, schemas, parsers, metrics math, rollback, data pipeline, and 17 dedicated safeguard audit tests. |
| **Candidate Manifest Hashes** | **REPRODUCED** | Calculated SHA256 against disk files | **7 / 7 hashes matched exactly** against `experiments/final_candidate.json`. Adapter safetensors (`643971...`), config, tokenizers, and dataset splits verified. |
| **Model Evaluation Metrics** | **REPRODUCED** | Recomputed from saved test predictions | Saved prediction snapshot yields **Accuracy = 0.3810** (16/42), **Macro-F1 = 0.3754**, **Weighted-F1 = 0.3754**. Documented delta: 2 samples lower than locked manifest run `exp_epoch_3` (Accuracy 0.4286, Macro-F1 0.4089). |
| **Confidence Calibration** | **REPRODUCED** | Recomputed ECE & sample analysis | Raw model mean confidence is **0.9689** vs accuracy **0.3810** (Overconfidence Gap: **+0.5879**). Constant scaling multiplier (0.75x) rejected as invalid calibration. Small validation size (N=44) documented. |
| **Native Hindi Fallback** | **REPRODUCED** | Implemented policy & tested all modalities | Devanagari Hindi and mixed-script tickets trigger deterministic human escalation (`requires_human=True`). High model confidence (0.999) cannot bypass. Hinglish, Roman Hindi, English pass without false escalation. |
| **Critical Routing Safeguards** | **REPRODUCED** | Unit & API integration tests | Tested fraud, account closure/GDPR, critical priority, short tickets (<=3 words), ambiguous multi-intent tickets, and parse failures. All rules deterministic. |
| **Live Rollback & Governance** | **REPRODUCED** | Live FastAPI client execution | Verified dynamic hot rollback between `nexus-qlora-v1` and `baseline-smollm-135m`. Health status, inference engine, and PSI drift tracking verified. |
| **API Performance Benchmarks** | **REPRODUCED** | Re-run live 40-sample + load benchmark | Cold start: **1270.56 ms**. Warm P50: **1472.49 ms**, P95: **1950.47 ms**, P99: **2056.58 ms**. Mean throughput: **0.67 req/sec**. 0 errors across 104 requests. |
| **V1 vs V2 Comparison** | **DOCUMENTED** | Inspected artifact `v2_comprehensive_evaluation.json` | V1 decisively superior to V2: Macro-F1 0.4089 vs 0.2224 (-45.6%), JSON validity 97.6% vs 83.3%, P95 latency 1.3s vs 2.5s. Retaining V1 verified as correct. |
| **Training Loss Convergence** | **DOCUMENTED** | Verified in `runs.json` and loss curves | `exp_epoch_3` converged with `train_loss=0.2077`, `eval_loss=0.1439` over 3 epochs (986.8s CPU training). Training was not re-run. |
| **Commercial LLM API Benchmark** | **DOCUMENTED** | Inspected `model_comparison_results.json` | Marked as "NOT RUN (API Key Missing)". Cost numbers are analytical projections based on token pricing, not live API predictions. |
| **Docker Linux Integration** | **PENDING** | Workflow configured in `.github/workflows/ci.yml` | Docker CLI/daemon absent on Windows host. CI pipeline upgraded to test build, env config, healthcheck, inference, and SIGTERM shutdown on Linux agent. |

---

## 3. Resolution of the Three Release Blockers

### Blocker 1: Docker and CI Validation
* **Status**: **PENDING REMOTE CI CONFIRMATION**
* **Host Limitation**: The audit host is Windows without an active Docker daemon (`docker : The term 'docker' is not recognized`). Production containers cannot be booted locally.
* **Implemented Mitigation**: The CI/CD workflow in `.github/workflows/ci.yml` has been updated with a full container lifecycle integration job on `ubuntu-latest`:
  1. Builds image using Docker Buildx and loads into local daemon (`tags: nexus-slm:ci-test`).
  2. Runs container with explicit environment configuration:
     `-e ACTIVE_MODEL_VERSION=nexus-qlora-v1 -e API_PORT=8000 -e API_HOST=0.0.0.0`.
  3. Polls `http://localhost:8000/health` up to 30 attempts until HTTP 200 and `"status": "healthy"` is returned.
  4. Dispatches live inference request to `POST /predict` and asserts structured JSON output with `category` and `requires_human`.
  5. Issues graceful SIGTERM via `docker stop --time 15 nexus-ci-container`, verifies clean exit, and captures container logs.
* **Release Gate Constraint**: Execution is marked as **PENDING** until the first Linux GitHub Actions run completes.

### Blocker 2: Native Devanagari Hindi Fallback
* **Status**: **RESOLVED & VERIFIED**
* **Root Cause**: SmolLM-135M uses a byte-fallback BPE tokenizer that fragments Devanagari characters into 3–4 tokens per character, resulting in a low accuracy of **14.3%** on native Hindi tickets.
* **Implemented Solution**:
  1. Implemented deterministic regex detection for Devanagari script (`[\u0900-\u097F\uA8E0-\uA8FF]`) in `src/inference/policy.py`.
  2. Any ticket containing Devanagari script (pure Hindi or mixed-script) immediately forces `requires_human = True` with `escalation_reason = "native_hindi_devanagari_unreliable_tokenization"`.
  3. **Non-bypassability**: This rule executes prior to confidence thresholding. A model predicting 0.999 confidence **cannot** silently bypass human review.
  4. **Modality Testing**: Verified across all 5 language modalities in `tests/test_safeguards_audit.py`:
     - Pure Devanagari Hindi (`"मेरा खाता ब्लॉक हो गया है..."`) $\to$ Escalated (`requires_human=True`).
     - Mixed-Script Devanagari (`"मेरा payment कट गया..."`) $\to$ Escalated (`requires_human=True`).
     - Hinglish in Latin script (`"Bhai mera package dispatch nahi hua..."`) $\to$ Automated routing permitted.
     - Roman Hindi in Latin script (`"Mujhe mera refund kab milega?..."`) $\to$ Automated routing permitted.
     - Standard English (`"I received a damaged phone charger..."`) $\to$ Automated routing permitted.

### Blocker 3: Confidence Calibration Handling
* **Status**: **RESOLVED & VERIFIED**
* **Flawed Baseline**: The previous implementation applied an arbitrary constant multiplier (`raw_confidence * 0.75`), which is statistically invalid and does not constitute proper calibration.
* **Statistical Assessment of Validation Data**:
  - The validation set (`data/validation/validation.jsonl`) contains only **N=44 samples** across 14 classes (~3.1 samples/class).
  - Fitting parametric Platt scaling on N=44 yields extreme variance ($a=0.3661, b=-2.5275$) that compresses all probabilities around ~0.38 without providing genuine posterior probabilities.
  - Test set Expected Calibration Error (ECE) is **0.5879** with an Overconfidence Gap of **+0.5879** (mean confidence 0.9689 vs accuracy 0.3810).
* **Implemented Policy**:
  1. Removed the unsupported constant multiplier claim.
  2. Declared confidence status in `PredictResponse` as `"confidence_status": "uncalibrated_heuristic"`.
  3. Set `calibrated_confidence: null` to avoid making unsupported calibration claims.
  4. Enforced conservative human-escalation gating: Any prediction with heuristic confidence $< 0.85$ is deterministically routed to human review (`escalation_reason = "unreliable_confidence_below_threshold"`).

---

## 4. Critical Routing Safeguards Audit

All routing safeguards were audited for determinism, test coverage, and escalation performance.

### 4.1 Safeguard Inventory & Determinism

| Safeguard | Trigger Condition | Output Override | Escalation Reason | Deterministic? |
|---|---|---|---|:---:|
| **Fraud Detection** | `category == "fraud_report"` or matches keywords (`fraud`, `unauthorized`, `stolen`, `hacked`, `धोखाधड़ी`, `हैक`, `चोरी`) | Force category `fraud_report`, priority `critical`, `requires_human = True` | `safety_critical_fraud_detection` | **YES** |
| **Account Closure & GDPR** | `category == "account_closure"` or matches keywords (`delete my account`, `gdpr`, `erase my data`, `legal action`, `lawyer`, `court`) | Force category `account_closure`, `requires_human = True` | `legal_or_account_closure_action` | **YES** |
| **Critical Priority** | `priority == "critical"` | `requires_human = True` | `critical_urgency_escalation` | **YES** |
| **Native Devanagari Hindi** | Contains Devanagari characters `[\u0900-\u097F\uA8E0-\uA8FF]` | `requires_human = True` | `native_hindi_devanagari_unreliable_tokenization` | **YES** |
| **Short Ticket Sparsity** | Input text word count $\le 3$ words | `requires_human = True` | `short_input_context_sparsity` | **YES** |
| **Ambiguous / Multi-Intent** | Ambiguity indicators (`not sure if`, `either or`) or conflicting intent pairs (`refund` + `cancel`, `delete` + `login`, `bill` + `crash`) | `requires_human = True` | `ambiguous_ticket_requires_human_clarification` | **YES** |
| **Parse / Schema Failure** | Output generation fails JSON parsing or contains invalid taxonomy values | `category = "general_query"`, `confidence = 0.0`, `requires_human = True` | `parse_or_schema_validation_failure` | **YES** |
| **Low / Uncertain Confidence** | Raw heuristic confidence $< 0.85$ | `requires_human = True` | `unreliable_confidence_below_threshold` | **YES** |
| **Heuristic Fallback Mode** | Model weights uninitialized and priority $\in$ `[high, critical]` or category $\in$ `[payment_failure, delivery_issue]` | `requires_human = True` | `heuristic_mode_high_impact_rule` | **YES** |

### 4.2 Empirical Escalation Behavior on Labeled Test Set (N=42)

Evaluating the policy on the locked test benchmark where ground-truth `requires_human` labels exist (17 True, 25 False):

```
+-------------------------------------------------------------+
|               CONFUSION MATRIX: HUMAN ESCALATION            |
|                                                             |
|                         Predicted Escalated   Predicted Auto|
|  Actual Human (17):           TP = 8              FN = 9    |
|  Actual Auto  (25):           FP = 12             TN = 13   |
+-------------------------------------------------------------+
```

* **True Positives (Correctly Escalated)**: **8** (Fraud, Legal, Devanagari Hindi)
* **False Positives (Conservatively Over-Escalated)**: **12** (Safe Devanagari Hindi and low-confidence tickets intentionally escalated for safety)
* **False Negatives (Missed Escalation)**: **9** (Subtle tickets labeled `requires_human=True` in ground truth but lacking explicit trigger keywords and predicted with high model confidence)
* **True Negatives (Correctly Automated)**: **13**
* **Escalation Sensitivity (Recall)**: **47.1%** (8 / 17)
* **Escalation Precision**: **40.0%** (8 / 20)
* **False Positive Rate (FPR)**: **48.0%** (12 / 25)
* **False Negative Rate (FNR)**: **52.9%** (9 / 17)

> [!WARNING]
> **No Claim of Perfect Protection**: While deterministic keyword and script safeguards eliminate misses for known high-risk vectors (fraud, account closure, native Devanagari), statistical false negatives (52.9%) still occur on nuanced tickets that rely solely on model classification. Human review teams must monitor automated triage streams.

---

## 5. Independent Model Evaluation Reproduction

### 5.1 Recomputed Evaluation Metrics from Saved Predictions

Recomputing metrics from the frozen candidate predictions (`experiments/artifacts/confidence_calibration.json`) across the locked 42-sample test set:

| Metric | Saved Prediction Snapshot | Candidate Manifest (`exp_epoch_3`) | Delta | Calculation Method |
|---|:---:|:---:|:---:|---|
| **Category Accuracy** | **0.3810** (16/42) | **0.4286** (18/42) | -0.0476 | $\frac{\text{Correct Predictions}}{\text{Total Samples}}$ ($N=42$) |
| **Macro Precision** | **0.4548** | — | — | Unweighted mean across 14 classes |
| **Macro Recall** | **0.3810** | — | — | Unweighted mean across 14 classes |
| **Macro F1** | **0.3754** | **0.4089** | -0.0335 | $\frac{1}{14} \sum_{i=1}^{14} F1_i$ |
| **Weighted F1** | **0.3754** | **0.4089** | -0.0335 | Weighted by class support ($S_i=3$ for all $i$) |
| **Micro F1** | **0.3810** | — | — | Global true positives / total |
| **JSON Validity Rate** | **100.0%** | **97.6%** | +0.024 | Parser validity on generated completions |
| **Exact Match Accuracy** | **0.1429** (6/42) | **0.0714** (3/42) | +0.0715 | Exact match on category, priority, language, human |

### 5.2 Per-Class Performance Breakdown

Each class in the locked test set has exactly **3 samples** ($N=42$ total):

| Category | Support | Precision | Recall | F1-Score | Qualitative Finding |
|---|:---:|:---:|:---:|:---:|---|
| `payment_failure` | 3 | 0.0000 | 0.0000 | 0.0000 | Confused with `account_access` and `order_tracking` |
| `refund_request` | 3 | 0.4000 | 0.6667 | 0.5000 | Moderate recall; confused with `order_cancel` |
| `account_access` | 3 | 0.1333 | 0.6667 | 0.2222 | Overpredicted by model on login-adjacent queries |
| `password_reset` | 3 | 1.0000 | 0.6667 | 0.8000 | High precision on credential queries |
| `order_cancel` | 3 | 0.6667 | 0.6667 | 0.6667 | Reliable intent discrimination |
| `order_tracking` | 3 | 0.0000 | 0.0000 | 0.0000 | Zero recall; confused with `delivery_issue` |
| `subscription` | 3 | 0.3333 | 0.3333 | 0.3333 | Moderate precision |
| `billing` | 3 | 1.0000 | 0.3333 | 0.5000 | High precision on invoice keywords |
| `technical_issue` | 3 | 0.0000 | 0.0000 | 0.0000 | **Zero recall; never predicted by candidate** |
| `fraud_report` | 3 | 1.0000 | 0.3333 | 0.5000 | High precision; safe escalation applied |
| `delivery_issue` | 3 | 0.6667 | 0.6667 | 0.6667 | Good recall on shipment complaints |
| `account_closure` | 3 | 0.6667 | 0.6667 | 0.6667 | High recall; legal keywords provide safety |
| `product_complaint`| 3 | 0.0000 | 0.0000 | 0.0000 | **Zero recall; never predicted by candidate** |
| `general_query` | 3 | 0.5000 | 0.3333 | 0.4000 | Catch-all class |

> [!NOTE]
> **Capacity Ceiling**: SmolLM-135M exhibits severe parameter saturation when discriminating 14 granular intent categories. Classes `technical_issue` and `product_complaint` have 0.0 F1 in this prediction snapshot.

---

## 6. Performance & Benchmark Claims Validation

### 6.1 Re-run API Benchmark Results (Hardware: Intel 8-core / 12-thread CPU, 16GB RAM)

Executing the live benchmark suite against the FastAPI service with active LoRA weights yielded:

* **Cold-Start Latency**: **1,270.56 ms** (First request initialization)
* **Warm P50 Latency**: **1,472.49 ms**
* **Warm P95 Latency**: **1,950.47 ms**
* **Warm P99 Latency**: **2,056.58 ms**
* **Mean Latency**: **1,498.72 ms**
* **Throughput**: **0.67 req/sec**
* **Error Rate**: **0.0%** (0 errors / 40 requests)

#### Component Latency Breakdown:
* API Overhead & Middleware Dispatch: `6.23 ms` (0.4%)
* Tokenization & ChatML Formatting: `15.00 ms` (1.0%)
* PyTorch Neural Model Execution: `1,467.49 ms` (97.9%)
* Parsing & Policy Enforcement: `10.00 ms` (0.7%)

### 6.2 Controlled Load Testing by Concurrency Tier

| Concurrency Tier | Requests | Duration | Throughput | Error Rate | P50 Latency | P95 Latency | CPU % | RAM (MB) | Stability |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 Worker** | 16 | 30.02 s | **0.53 req/s** | **0.0%** | 1,739.2 ms | 2,606.3 ms | 96.0% | 13,275 | **STABLE** |
| **2 Workers** | 16 | 23.87 s | **0.67 req/s** | **0.0%** | 2,836.1 ms | 4,862.2 ms | 91.7% | 13,344 | **STABLE** |
| **4 Workers** | 16 | 24.10 s | **0.66 req/s** | **0.0%** | 5,888.5 ms | 6,229.5 ms | 90.7% | 13,436 | **STABLE** |
| **8 Workers** | 16 | 23.76 s | **0.67 req/s** | **0.0%** | 10,385.3 ms | 14,074.8 ms | 89.4% | 13,442 | **STABLE** |

* **Sustainable Operating Point**: **1–2 workers per container instance**. A single container provides ~0.53–0.67 req/sec throughput. Additional concurrency does not increase CPU throughput because single-inference PyTorch execution saturates host CPU cores; it increases queue waiting time.
* **Horizontal Scaling Rule**: For production workloads exceeding 30 req/min, deploy a cluster of 3–5 container replicas behind a round-robin load balancer.

### 6.3 Economic Cost Estimates & Assumptions

Cost claims in `experiments/artifacts/cost_analysis.json` are **analytical projections**, not billing receipts:
* **NEXUS-SLM Self-Hosted**: Projected at **$0.0058 per 1,000 requests** based on an EC2 spot `c6i.xlarge` instance ($0.038/hr) delivering continuous 1.8 req/sec.
* **Commercial Cloud APIs**: Derived from published token pricing catalogs (GPT-4o-mini at $0.15/$0.60 per 1M tokens, yielding ~$0.045 per 1,000 requests).
* **Head-to-head External LLM Execution**: **Not executed**. No external API keys were provided.

---

## 7. Model-Quality Limitations & Remaining Risks

1. **Small Benchmark Sample Size**: The locked evaluation set contains 42 samples (3 samples/class). Statistical confidence intervals are wide ($\pm 12\%$).
2. **Rare Class Saturation**: At 135M parameters, the model cannot distinguish subtle semantic boundaries for rare classes (`technical_issue`, `product_complaint`, `order_tracking`).
3. **Devanagari Subword Fragmentation**: Native Devanagari Hindi causes byte fallback and low accuracy (14.3%). Gated by deterministic human review.
4. **Heuristic Confidence**: Confidence values are uncalibrated and must not be used as statistical probabilities of correctness.
5. **CPU Inference Latency**: Wall-clock CPU latency is ~1.5 seconds per request. Workloads requiring sub-200ms latency require GPU acceleration (TensorRT-LLM / vLLM).

---

## 8. Exact Deployment Prerequisites

Before promoting NEXUS-SLM to production cutover:

1. **Verify CI/CD Linux Docker Run**: Confirm green build and passing integration tests on GitHub Actions (`.github/workflows/ci.yml`).
2. **Horizontal Container Allocation**: Provision minimum 3 container instances with 2 vCPUs / 4GB RAM each behind an NGINX or ALB load balancer.
3. **Human Escalation Queue Integration**: Connect webhook or queue subscriber for all responses where `requires_human == true`.
4. **Data Drift & Alert Monitoring**: Monitor `/monitoring/drift` and trigger re-audit if Category or Language PSI $\ge 0.25$.
5. **Indic Script Safeguard Maintenance**: Preserve the Devanagari regex filter until the base architecture is upgraded to an Indic-native SLM (e.g. Qwen2.5-0.5B).

---

## 9. Sign-Off & Status Certification

| Role | Sign-Off Status | Notes |
|---|:---:|---|
| **Independent Verification Auditor** | **CERTIFIED** | All tests (78/78) passed, hashes verified, safeguards tested. |
| **Release Decision** | **READY WITH CONDITIONS** | Production deployment approved subject to prerequisites in Section 8. |
