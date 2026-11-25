# NEXUS-SLM — Neural EXecution for Unified Support Intelligence
### Efficient, Defensible Small Language Model Platform for Multilingual Ticket Triage

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Tests](https://img.shields.io/badge/tests-236%20passed%2C%201%20skipped-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Production Engine](https://img.shields.io/badge/production%20engine-HEURISTIC%20FALLBACK%20V2%2FV3-brightgreen.svg)](docs/RESULTS.md)
[![Neural Candidate](https://img.shields.io/badge/neural%20candidate-EXP_V2_5%20HYBRID%20(EXPERIMENTAL)-yellow.svg)](docs/NEURAL_MODEL_ACCEPTANCE_REPORT.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-009688.svg)](https://fastapi.tiangolo.com)

---

## 1. Project Overview

**NEXUS-SLM** is an enterprise small language model system engineered for low-latency, privacy-compliant, multilingual customer support ticket triage. Built for code-switched environments including English, Hindi (Devanagari script), and colloquial Hinglish (Romanized Hindi), NEXUS-SLM accepts unstructured customer messages and produces strictly validated structured JSON output across 14 categories, 4 priorities, and 3 languages.

### Dual-Engine Architecture
The platform operates a strict safety-first dual-engine architecture:
1. **Production Engine (Deterministic Heuristic Fallback V2/V3):** A deterministic, Unicode-boundary-safe rule-based classifier with Devanagari & Hinglish lexical rules, sub-millisecond direct latency (`0.07 ms` P50 / `0.22 ms` P95), **`100.0%` critical safety recall (`0` FN)** across synthetic `N=746`, dev-exposed `N=184`, and 8-domain stress audits, plus conservative `confidence: null` abstention. Active by default in production.
2. **Experimental Neural Candidates (`EXPERIMENTAL_NOT_PROMOTED`):**
   - **Legacy True-Base `Qwen/Qwen2.5-0.5B-Instruct` + Clean V2 LoRA:** Evaluated on uncontaminated HF weights at `53.89%` category accuracy (`0.5375` F1) with zero F1 on `general_query`/`product_complaint` (**Rejected** for production).
   - **V2 Multitask Hybrid Encoder (`exp_v2_5`, Frozen Qwen2.5-0.5B + Lexical SVD Residual + Ordinal Priority + Temp Scaling):** Reached **`89.14%` category accuracy (`0.8558` F1)** on `test_clean`, `0.0534` calibrated ECE, and **`288.78 ms` CPU P95 latency**, yet remains held in experimental status because priority accuracy (`48.66%` / `0.4290` F1) fails the `>= 80%` / `0.75` gate and independent Human Eval V2 (`N >= 300`) is **`PENDING`**.

### Inference Contract (`POST /predict`)
```json
// Request
{
  "text": "Mera payment kat gaya but order confirm nahi hua."
}

// Validated Response
{
  "category": "payment_failure",
  "priority": "high",
  "language": "hinglish",
  "requires_human": true,
  "confidence": null,
  "confidence_status": "unavailable",
  "latency_ms": 0.08,
  "model_version": "heuristic_fallback_v2",
  "inference_mode": "heuristic_fallback",
  "request_id": "9d023b34-d309-4ce8-b5e1-840250df7a69",
  "calibrated_confidence": null,
  "escalation_reason": "high_impact_financial_dispute",
  "raw_category": "payment_failure",
  "raw_priority": "high",
  "raw_language": "hinglish"
}
```

---

## 2. Authoritative Verification & Independent Evaluation ($N=746$)

Formal evaluation was executed across the entire locked clean test dataset (`data/v2_clean_template_grouped/test_clean.jsonl`, $N=746$, SHA-256: `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b`) without mocks or gradient leakage.

### Side-by-Side Benchmark Results

<!-- SYNC:RESULTS_SUMMARY:START -->
**Dataset:** Locked Synthetic Template-Grouped Clean Test (`data/v2_clean_template_grouped/test_clean.jsonl`, `N=746`, SHA-256 prefix `fc65071b1e9f3303...`)  
**Exact Language Support Counts:** English `246`, Hindi (Devanagari) `178`, Hinglish (Romanized) `322`  
**Archive Notice:** All historical snapshots (`Policy V1`, pre-PR#4 `13:47Z` run, and double-adapter stacked run) are preserved immutably under [`experiments/results/archive_v1/`](experiments/results/archive_v1/ARCHIVE_MANIFEST.json).

### Locked Synthetic Test Benchmark ($N=746$)

| Metric | Heuristic V2 (Policy V1 Archive) | Production Heuristic V2 (Refined Policy V2) | Qwen2.5-0.5B LoRA (True Uncontaminated HF Base) | Historical Qwen LoRA Stacked (`scratch_merged_qwen`, Contaminated) |
|---|---:|---:|---:|---:|
| **Category Accuracy** | 57.51% | **57.51%** | 53.89% | 69.17% |
| **Category Macro-F1** | 0.6516 | **0.6516** | 0.5375 | 0.6044 |
| **Priority Accuracy** | 35.79% | 36.46% | 44.24% | **47.86%** |
| **Priority Macro-F1** | 0.3816 | **0.3827** | 0.2491 | 0.2842 |
| **Language Accuracy** | 81.23% | 81.23% | 84.58% | **96.78%** |
| - English ($N=246$) | 100.00% | 100.00% | 93.90% cat | 100.00% lang |
| - Hindi Devanagari ($N=178$) | 100.00% | 100.00% | 20.79% cat | 100.00% lang |
| - Hinglish Roman ($N=322$) | 56.52% | 56.52% | 41.61% cat | 92.55% lang |
| **JSON Validity Rate** | 100.00% | 100.00% | 100.00% | 100.00% |
| **High-Risk Escalation Recall** | 100.00% (0 FN) | 84.69% (32 FN) | 100.00% (0 FN, conf=None) | 86.12% (29 FN) |
| **Critical Priority Recall** | 100.00% (0 FN) | **100.00% (0 FN)** | **100.00% (0 FN)** | — (unweighted critical in raw) |
| **Overall Escalation Rate** | 100.00% | **70.64%** | 100.00% (policy conservative) | 72.25% |
| **Direct CPU P50 Latency** | ~0.04 ms | **0.04 ms** (P95 0.18 ms) | — | 1508.55 ms (P95 2212.92 ms) |

### Development-Exposed Human Scenario Audit ($N=184$ Active + $16$ Pending)

| Metric | Heuristic V1 (Policy V1, $N=184$) | Production Heuristic V2 (Refined Policy V2, $N=184$) |
|---|---:|---:|
| **Category Accuracy / Macro-F1** | 56.52% / 0.5965 | **60.33% / 0.6278** |
| **Priority Accuracy / Macro-F1** | 51.09% / 0.5280 | **59.78% / 0.6519** |
| **Language Accuracy (EN 80 / HI 46 / HG 58)** | 97.28% | 97.28% |
| **Overall Escalation Rate** | 100.00% | **58.70%** |
| **High-Risk Recall / FN** | 100.00% (0 FN) | 85.71% (9 FN) |
| **Critical Safety Recall** | 100.00% (0 FN) | **100.00% (0 FN)** |

### Validation Sprint V3 Hardened Policy & Unpromoted Neural V2 Candidate

| System Version | Dataset (`test_clean` N=746 / `human_v1` N=184) | Cat Acc | Cat Macro-F1 | Prio Acc / F1 | Lang Acc (Hinglish ID) | High-Risk Recall (FN) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| **Hardened Policy V3** | `test_clean.jsonl` (`N=746`) | **69.57%** | **0.7365** | 39.14% / 0.4125 | 85.79% (67.08%) | 90.43% (20 FN) | Active Hardened Heuristic |
| **Hardened Policy V3** | `human_eval_v1` (`N=184`, dev-exposed) | **72.28%** | **0.7419** | 64.13% / 0.6875 | 97.83% (93.10%) | 87.30% (8 FN) | Active Hardened Heuristic |
| **`exp_v2_5_hybrid_qwen_lexical_fusion`** | `test_clean.jsonl` (`N=746`) | **89.14%** | **0.8558** | 48.66% / 0.4290 | 96.78% (92.55%) | 88.52% (24 FN, 0 crit FN) | `EXPERIMENTAL_NOT_PROMOTED` |
<!-- SYNC:RESULTS_SUMMARY:END -->

*For complete per-category matrices, Wilson confidence intervals, true-base vs. stacked contamination reconciliation, and version archives, see [`docs/RESULTS.md`](docs/RESULTS.md) and [`docs/FINAL_METRIC_CONSISTENCY_AUDIT.md`](docs/FINAL_METRIC_CONSISTENCY_AUDIT.md).*

---

## 3. Production Documentation Suite

| Document | Purpose & Description |
|---|---|
| [**Master Engineering Acceptance**](docs/FINAL_10_OUT_OF_10_ENGINEERING_ACCEPTANCE.md) | Independent technical review & honest score matrix across all 12 workstreams |
| [**Metric Consistency Audit**](docs/FINAL_METRIC_CONSISTENCY_AUDIT.md) | Forensic reconciliation of V1 archives, Refined V2, Hardened V3 & True-Base Qwen |
| [**Human Evaluation V2 Protocol**](docs/FINAL_HUMAN_EVALUATION_PROTOCOL.md) | Blind 320+80 cohort, Cohen's Kappa IRR, & adjudication scaffold (`PENDING` external review) |
| [**Independent Benchmark Results**](docs/INDEPENDENT_BENCHMARK_RESULTS.md) | Sealed `N>=300` status (`PENDING`), leakage guarantees & blocking dependencies |
| [**Neural Model V2 Research**](docs/NEURAL_MODEL_V2_RESEARCH.md) | Error taxonomy, V3 diverse dataset (`0` leakage) & Exp 1–5 multitask ablation study |
| [**Neural V2 Acceptance Report**](docs/NEURAL_MODEL_ACCEPTANCE_REPORT.md) | Predefined gate scorecard (`EXPERIMENTAL_NOT_PROMOTED`: category pass, priority fail) |
| [**Multilingual Acceptance Report**](docs/MULTILINGUAL_ACCEPTANCE_REPORT.md) | EN/Hindi/Hinglish Unicode word-boundary fixes & 48-case sub-regime evaluation |
| [**Final Safety Acceptance**](docs/FINAL_SAFETY_ACCEPTANCE_REPORT.md) | 8-domain safety taxonomy (`100%` recall, `0%` negative-control FPR) & Policy V3 |
| [**Final Calibration Acceptance**](docs/FINAL_CALIBRATION_ACCEPTANCE.md) | Disjoint `cal_v3` temperature scaling (`ECE 0.1343 -> 0.0534`) & `null` rule semantics |
| [**Final Inference Acceptance**](docs/FINAL_INFERENCE_ACCEPTANCE_REPORT.md) | Live CPU benchmarks: Heuristic (`0.22 ms`), Single-Forward Neural (`288.78 ms`), GGUF, ONNX |
| [**Distributed Reliability Report**](docs/FINAL_DISTRIBUTED_RELIABILITY_REPORT.md) | Live Uvicorn HTTP concurrency (`265.07 req/s`), 429 bursts, & Redis failover/recovery |
| [**Final Security Acceptance**](docs/FINAL_SECURITY_ACCEPTANCE_REPORT.md) | Secrets scan (`0` real findings), error hygiene, proxy XFF & prod startup checks |
| [**Final Dashboard Acceptance**](docs/FINAL_DASHBOARD_ACCEPTANCE.md) | Next.js 14 `/evaluation/summary` wiring, null-safe triage UI & `next build` verification |
| [**Final CI/CD Acceptance**](docs/FINAL_CI_ACCEPTANCE_REPORT.md) | GitHub Actions Redis service, manifest/doc consistency checks & `pytest` status |
| [**Reproducibility Guide**](docs/REPRODUCIBILITY_GUIDE.md) | Clean-clone reproduction sequence & verified SHA-256 dataset/weight manifest index |
| [**10/10 Capstone Acceptance Report (V1)**](docs/FINAL_10_OUT_OF_10_ACCEPTANCE_REPORT.md) | Historical PR #4 capstone evaluation snapshot (superseded by Master Engineering Acceptance) |
| [**Baseline Engineering Audit**](docs/BASELINE_ENGINEERING_AUDIT.md) | Immutable baseline engineering snapshot of commit `3e5a1ea` and test suites |
| [**Evaluation Integrity Audit**](docs/EVALUATION_INTEGRITY_AUDIT.md) | Forensic resolution of human eval, escalation rate, and template count discrepancies |
| [**Experiment Tracking Standard**](docs/EXPERIMENT_TRACKING_STANDARD.md) | Unified JSON evaluation schema, canonical hashes, and policy versioning standard |
| [**Human Evaluation Governance**](docs/HUMAN_EVALUATION_GOVERNANCE.md) | Dual annotation workflow, Cohen's Kappa IRR validation, and 6 adjudicated boundary cases |
| [**Independent Validation Protocol**](docs/INDEPENDENT_VALIDATION_PROTOCOL.md) | Formal scientific separation of exposed development data vs virgin $N \ge 300$ validation |
| [**Neural Model Error Analysis**](docs/NEURAL_MODEL_ERROR_ANALYSIS.md) | Deep failure mode analysis of Qwen2.5-0.5B LoRA on locked test set ($N=746$) |
| [**Multilingual Evaluation Report**](docs/MULTILINGUAL_EVALUATION_REPORT.md) | Native Hindi BPE tokenization, Rule 2 safety invariant, and Hinglish word boundaries |
| [**Safety Taxonomy & Acceptance**](docs/SAFETY_AND_ESCALATION_ACCEPTANCE.md) | 5-tier safety hierarchy, 100% critical recall invariant, and negative control testing |
| [**Confidence Calibration Report**](docs/CALIBRATION_ACCEPTANCE_REPORT.md) | Multiclass ECE, Brier score, autoregressive token caveats, and calibration invariants |
| [**Production Backend Acceptance**](docs/PRODUCTION_BACKEND_ACCEPTANCE.md) | Redis Lua atomic rate limiting, in-memory failover, timing-safe auth, and load stability |
| [**Inference Performance Acceptance**](docs/INFERENCE_PERFORMANCE_ACCEPTANCE.md) | CPU benchmarks for Heuristic V2, GGUF Q4_K_M, and ONNX dynamic INT8 collapse |
| [**Dataset Generalisation Audit**](docs/DATASET_GENERALISATION_AUDIT.md) | Lexical diversity, template compression, and zero cross-partition leakage |
| [**Dashboard Verification Report**](docs/DASHBOARD_VERIFICATION_REPORT.md) | Next.js 14 operator UI, real-time triage demo, model registry, and telemetry |
| [**CI/CD Verification Report**](docs/CI_CD_VERIFICATION_REPORT.md) | Automated GitHub Actions pipeline, Redis service container, and Docker health probes |
| [**Authoritative Results Report**](docs/RESULTS.md) | Verified empirical metrics, side-by-side comparison, uncertainty intervals, gate decision |
| [**Final Excellence Sprint Report**](docs/FINAL_EXCELLENCE_SPRINT_REPORT.md) | Comprehensive engineering sprint report across all priority dimensions |
| [**Human Evaluation Results**](docs/HUMAN_EVALUATION_RESULTS.md) | Side-by-side empirical evaluation of Baseline V1 vs Refined V2 on 184 human scenarios |
| [**Dataset Card**](docs/dataset_card.md) | Provenance, template redundancy analysis, class support counts, and sparsity flags |
| [**Model Card**](docs/model_card.md) | Specifications, parameters, SHA-256 hashes, and benchmark results for Qwen2.5-0.5B LoRA |
| [**Operational Limitations**](docs/limitations.md) | In-depth analysis of CPU generation latency, synthetic bias, and priority ambiguities |

---

## 4. Security & Hardening Highlights

Sprint A resolved all previously identified security and operational concerns:
- **Zero Default Secrets:** Hardcoded admin keys removed. Production mode strictly aborts startup if `NEXUS_ADMIN_KEY` is missing or below 32 characters entropy.
- **Zero Authentication Bypasses:** Completely eliminated test-runner auth bypasses (`PYTEST_CURRENT_TEST`). All tests authenticate via valid cryptographic headers.
- **Trusted Proxy Verification:** `resolve_client_ip` enforces that `X-Forwarded-For` is only respected if the immediate peer IP is listed in `cfg.trusted_proxies`.
- **Distributed Redis Rate Limiter:** Atomic Redis sliding window limiter with automatic cooldown-based local failover and `strict_reject` 503 policy.
- **Full Test Suite Verification:** **236 passing automated tests, 1 skipped locally** (real-Redis atomic concurrency test executes in CI `redis:7-alpine` service container).

---

## 5. Quickstart & Verification

### Step 1: Environment Setup
```powershell
# Create and activate virtual environment (Python 3.10 recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core and test dependencies
pip install -r requirements.txt
pip install -r requirements-ci.txt
```

### Step 2: Run Full Automated Test Suite (`236 passed, 1 skipped` locally without Redis daemon)
```powershell
python -m pytest -v
```

### Step 3: Launch Local Inference Service
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Test Triage Prediction via cURL
```powershell
# Heuristic Production Fallback Prediction
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"text": "Mera payment kat gaya but order confirm nahi hua."}'
```
