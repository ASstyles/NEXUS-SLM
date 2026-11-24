# NEXUS-SLM — Final Production Promotion Decision & Governance Gate

**Date**: 2026-10-31  
**Branch**: `sprint/clean-data-model-improvement`  
**PR Reference**: `https://github.com/ASstyles/NEXUS-SLM/pull/1`  
**Candidate Under Review**: `qwen25_05b_clean_v2_lora`  
**Active Production Model**: `nexus-qlora-v1` (Heuristic Baseline + Policy Safeguards)  
**Governance Authority**: NEXUS-SLM Architecture & Promotion Review Board  

---

## 1. Executive Promotion Decision

> [!CAUTION]
> ### PROMOTION GATE VERDICT: REJECT AUTOMATIC PROMOTION
> **Decision**: **RETAIN `nexus-qlora-v1` IN PRODUCTION**.  
> The candidate model `qwen25_05b_clean_v2_lora` **MUST NOT** be automatically promoted to production, PR #1 **MUST NOT** be automatically merged, and `main` **MUST NOT** be modified.

### Executive Rationale:
Although the repository's backend engineering, security architecture, middleware stack, and test infrastructure achieved a **10/10 Production Rating**, the AI/ML triage candidate model does not satisfy the strict, non-negotiable production quality and safety gates:

1. **Category Accuracy Gate Missed**: The committed acceptance policy requires $\ge 80.0\%$ category accuracy. On the validation set ($N=449$), the candidate achieved **79.96%** (missing by 0.04% / 1 sample). On the full, quarantined test set ($N=746$), accuracy dropped to **69.17%** (516/746), failing the production threshold by 10.83%.
2. **Safety-Critical Escalation Gate Breached**: The active production baseline enforces a zero-tolerance policy for high-risk customer tickets, achieving **100.0% recall** and **0 false negatives**. In contrast, the candidate allowed **26 high-risk false negatives** (87.56% recall), introducing unacceptable business and legal risk if deployed unmonitored.
3. **Confidence Calibration Uncertified**: The model's reported confidence represents token sequence generation probabilities (mean 0.9768), not class posterior probabilities. Empirical Expected Calibration Error (ECE) is **0.2851** (28.51% overconfidence). All predictions cluster in the 0.90–1.00 bin regardless of classification correctness, making straight-through automated routing unsafe.

---

## 2. Gate-by-Gate Evaluation & Scorecard

To maintain engineering integrity, performance is evaluated separately across five independent dimensions rather than forcing an artificial overall score:

| Evaluation Dimension | Standard / Acceptance Criteria | Verified Empirical Result | Gate Verdict |
|---|---|---|:---:|
| **1. Backend Engineering Quality** | Modular design, thread-safe rate limiting, PII sanitization, request tracing, prompt-injection defense, 100% test pass rate | 6 modular packages, 176/176 passing tests, 181.72 RPS, zero circular dependencies | **PASS (10/10)** |
| **2. AI/ML Classification Quality** | $\ge 80.0\%$ Category Accuracy, $\ge 0.75$ Macro-F1 on test set | **69.17% Accuracy**, **0.6044 Macro-F1** on full clean test set ($N=746$) | **FAIL** |
| **3. Safety & Escalation Safeguards** | $100.0\%$ High-Risk Recall, **Zero False Negatives** | **87.56% Recall**, **26 High-Risk False Negatives** on clean test set | **FAIL** |
| **4. Confidence Calibration** | Calibrated posterior probabilities, ECE $\le 0.10$, monotonic selective risk-coverage separation | Sequence token confidence; **ECE = 0.2851**; zero selective coverage separation | **CONDITIONAL PASS (Heuristic Only)** |
| **5. Production Operational Readiness** | Air-gapped CPU compatibility, P95 $\le 2,500$ ms, RAM $\le 4$ GB | P95 = 1,386.15 ms, Throughput = 0.93 RPS, Peak RAM = 2,671 MB | **PASS** |

---

## 3. Detailed Dimension Analysis

### 3.1 Backend Engineering Quality: PASS (10/10)
- **Modularity**: Dedicated services for configuration (`src/utils/config.py`), auth (`api/auth.py`), middleware (`api/middleware.py`), security (`src/utils/security.py`), and calibration (`src/models/calibration.py`).
- **Load Capacity**: Validated across **4,160 total requests** under concurrency levels 1, 5, 10, and 20 workers, maintaining 153.06 – 181.72 RPS and 0.00% errors on valid traffic.
- **Throttling Verification**: Rate limiter strictly enforced 120 req/min, rejecting 40 out of 160 burst requests with standard HTTP 429 and `Retry-After: 60` headers.
- **Security**: Constant-time token verification (`secrets.compare_digest`), PII redaction across 5 modalities, and adversarial prompt-injection inspection.

### 3.2 AI/ML Classification Quality: FAIL
- On the locked clean test set ($N=746$), `qwen25_05b_clean_v2_lora` achieved:
  - Accuracy: **69.17%** (516 / 746 correct) vs **61.80%** (461 / 746) baseline (+7.37% gain).
  - Macro-F1: **0.6044** vs **0.5473** baseline (+0.0571 gain).
  - Category Strengths: Perfect precision on `billing` (0.9927 F1), `password_reset` (1.0000 F1), `account_closure` (1.0000 F1), `order_tracking` (0.8986 F1), and `technical_issue` (0.8667 F1).
  - Category Deficiencies: Zero recall on `product_complaint` ($N=44$) and `general_query` ($N=42$), where colloquial complaints were frequently misrouted to `refund_request` or `fraud_report`.

### 3.3 Safety & Escalation Safeguards: FAIL
- The production baseline operates via deterministic fallback, guaranteeing that high-risk complaints (fraud, unauthorized debits, account lockouts, legal actions) are escalated with **100.0% recall** (0 false negatives).
- When evaluating the candidate neural model, 122 tickets were marked `requires_human: false` by the neural model. While the deterministic policy caught 96 of them, **26 slipped past** as false negatives because the model predicted standard categories (`refund_request`, `order_tracking`) without triggering keyword rules.
- **Verdict**: In a production customer support environment, 26 unescalated financial or fraud complaints constitute a severe operational hazard.

### 3.4 Confidence Calibration: CONDITIONAL PASS (Heuristic Only)
- The model exhibits severe overconfidence: mean token sequence confidence is **0.9768**, while empirical accuracy is **0.6917** ($\text{ECE} = 0.2851$).
- Because no independent calibration dataset was reserved, post-hoc scaling cannot be certified.
- **Policy Rule**: Confidence must remain designated as `uncalibrated_heuristic`. Autonomous straight-through routing based on confidence thresholding is strictly prohibited.

---

## 4. Model Registry & Branch State

The repository state is reconciled to preserve all verified artifacts while protecting production:

| Model Version | Registry Status | Lifecycle Stage | Location | Deployment Role |
|---|:---:|:---:|---|---|
| **`nexus-qlora-v1`** | `production` | Production | `models/nexus_qlora_v1/` | **ACTIVE PRODUCTION SERVING** |
| **`qwen25_05b_clean_v2_lora`** | `experimental_validated` | Experimental Candidate | `experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter/` | **FROZEN EXPERIMENTAL ARTIFACT** |

### Git & Pull Request Governance:
- **Pull Request #1**: Remains **OPEN** on branch `sprint/clean-data-model-improvement` for maintainer review. It is **NOT** automatically merged into `main`.
- **Branch Integrity**: All new code, reports, and benchmark artifacts are preserved on `sprint/clean-data-model-improvement`.
- **Zero Binary Weights in Git**: Binary model weights (`adapter_model.safetensors`, `.zip` archives) remain strictly ignored via `.gitignore` and are not committed to Git.

---

## 5. Technical Path to Production Promotion

For `qwen25_05b_clean_v2_lora` or a successor candidate to achieve full production promotion:

1. **Resolve Under-Represented Classes**: Expand training templates for `product_complaint` and `general_query` to eliminate the 0% F1 failure mode.
2. **Harden Deterministic Fallback Rules**: Update `HumanEscalationPolicy` keyword sets to capture the 26 false-negative edge cases identified during the test evaluation.
3. **Reserve Independent Calibration Holdout**: Create an isolated calibration split ($N \approx 500$) to fit true temperature scaling or Platt scaling on categorical class logits, reducing ECE below 0.10.
4. **Quantize for Low Latency**: Quantize the merged model to GGUF (Q4_K_M / Q8_0) or ONNX INT8 to achieve sub-100ms CPU latency and sub-500MB memory footprint.
5. **Execute Shadow Pilot Protocol**: Run the candidate in shadow mode (dual-writing live traffic alongside `nexus-qlora-v1`) for a minimum of 7 days with zero customer-facing regressions before executing production cutover.

