# NEXUS-SLM — Neural EXecution for Unified Support Intelligence
### Efficient, Defensible Small Language Model Platform for Multilingual Ticket Triage

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Tests](https://img.shields.io/badge/tests-215%20passed%2C%201%20skipped-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Production Engine](https://img.shields.io/badge/production%20engine-HEURISTIC%20FALLBACK%20v2-brightgreen.svg)](docs/RESULTS.md)
[![Neural Candidate](https://img.shields.io/badge/neural%20candidate-QWEN2.5--0.5B%20LoRA%20(HOLD)-yellow.svg)](docs/RESULTS.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-009688.svg)](https://fastapi.tiangolo.com)

---

## 1. Project Overview

**NEXUS-SLM** is an enterprise small language model system engineered for low-latency, privacy-compliant, multilingual customer support ticket triage. Built for code-switched environments including Indian English, Hindi (Devanagari script), and colloquial Hinglish (Roman Hindi), NEXUS-SLM accepts unstructured customer messages and produces strictly validated structured JSON output across 14 categories, 4 priorities, and 3 languages.

### Dual-Engine Architecture
The platform operates a defensible dual-engine architecture:
1. **Production Engine (Heuristic Fallback v2):** A deterministic, token-anchored rule-based classifier with Unicode Devanagari parsing, Hinglish stopword matching, sub-millisecond latency (P50 0.05ms), and 100% high-risk safety escalation recall. Active by default for production reliability.
2. **Experimental Engine (Qwen2.5-0.5B LoRA Candidate):** A 494M parameter causal language model (`Qwen/Qwen2.5-0.5B-Instruct` fine-tuned with LoRA) that generates structured JSON payloads with 100% schema validity and 92.55% Hinglish accuracy. Currently held in experimental status following independent gate evaluation.

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

| Metric | Heuristic Fallback v2 (Production) | Qwen2.5-0.5B LoRA (Experimental) | Delta / Analysis |
|---|:---:|:---:|---|
| **Category Accuracy** | 56.03% (418/746) | **69.17%** (516/746) | **+13.14%** (Neural superior) |
| **Category Macro Precision** | **87.59%** | 65.34% | -22.25% (Heuristic more conservative) |
| **Category Macro Recall** | 58.32% | **69.17%** | **+10.85%** (Neural broader coverage) |
| **Category Macro-F1** | **0.6132** | 0.6044 | -0.0088 (Comparable) |
| **Category Weighted-F1** | 0.6175 | **0.6630** | **+0.0455** (Neural superior on common classes) |
| **Priority Accuracy** | 47.18% (352/746) | **47.86%** (357/746) | +0.68% (Both models struggle) |
| **Overall Language Accuracy** | 81.23% (606/746) | **96.78%** (722/746) | **+15.55%** (Neural massive win) |
| - English Accuracy | 100.00% (231/231) | 100.00% (231/231) | Parity |
| - Hindi Accuracy | 100.00% (232/232) | 100.00% (232/232) | Parity |
| - Hinglish Accuracy | 49.82% (141/283) | **92.55%** (262/283) | **+42.73%** (Neural resolves Roman Hindi) |
| **Valid JSON Generation** | 100.00% | **100.00%** | Parity (Zero schema failures) |
| **Exact 4-Field Match** | 22.39% (167/746) | **35.52%** (265/746) | **+13.13%** |
| **Escalation Recall** | **100.00%** (209/209) | 86.12% (180/209) | -13.88% (Neural has 29 false negatives) |
| **Overall Escalation Rate** | 100.00% (746/746) | **72.25%** (539/746) | -27.75% (Neural saves 27.7% queue load) |
| **P50 Latency** | **0.05 ms** | 1,508.55 ms | Heuristic ~30,000x faster on CPU |
| **P95 Latency** | **0.25 ms** | 2,212.92 ms | Heuristic ~8,800x faster on CPU |
| **Throughput (CPU)** | **9,833.58 req/s** | 0.62 req/s (batch=16) | CPU throughput limit for SLM |
| **Model RAM Footprint** | **~5 MB** | 78.66 MB (LoRA delta) | Both highly memory efficient |

*For complete confusion matrices, Wilson score confidence intervals, and gate evaluations, consult [docs/RESULTS.md](docs/RESULTS.md).*

---

## 3. Production Documentation Suite

| Document | Purpose & Description |
|---|---|
| [**Final Excellence Sprint Report**](docs/FINAL_EXCELLENCE_SPRINT_REPORT.md) | Comprehensive engineering sprint report across all 9 priority dimensions |
| [**Authoritative Results Report**](docs/RESULTS.md) | Verified empirical metrics, side-by-side comparison, uncertainty intervals, gate decision |
| [**Human Evaluation Protocol**](docs/HUMAN_EVALUATION_PROTOCOL.md) | Sealed 200-ticket benchmark schema, dual annotation workflow, and IRR Cohen's Kappa |
| [**Human Evaluation Results**](docs/HUMAN_EVALUATION_RESULTS.md) | Empirical evaluation results on 184 adjudicated human scenarios |
| [**Priority Classification Research**](docs/PRIORITY_CLASSIFICATION_RESEARCH.md) | 4-tier objective operational impact rubric & empirical accuracy gains |
| [**Escalation Precision Research**](docs/ESCALATION_PRECISION_RESEARCH.md) | 7-tier escalation hierarchy, collision resolution, physical safety safeguards |
| [**Dataset Diversity Report**](docs/DATASET_DIVERSITY_REPORT.md) | Lexical redundancy, template compression (10:1), and n-gram containment audit |
| [**Dataset V3 Specification Protocol**](docs/DATASET_V3_PROTOCOL.md) | Zero-leakage hash partitioning, Levenshtein distance constraints, and DPA compliance |
| [**Confidence Calibration Report**](docs/CALIBRATION_VALIDATION_REPORT.md) | ECE, Brier score analysis, autoregressive confidence caveats, and routing policy |
| [**Distributed Backend Verification**](docs/DISTRIBUTED_BACKEND_VERIFICATION.md) | Redis Lua atomic script, failover to memory, concurrency benchmarks, and Kubernetes probes |
| [**Inference Optimisation Report**](docs/INFERENCE_OPTIMISATION_REPORT.md) | Comparative CPU benchmarks for GGUF Q4_K_M, ONNX INT8, and Heuristic Fallback V2 |
| [**Security Remediation Report**](docs/SECURITY_REMEDIATION_REPORT.md) | Audit and fixes for admin secrets, auth bypasses, proxy validation, and Redis rate limiting |
| [**Safety Evaluation Report**](docs/SAFETY_EVALUATION_REPORT.md) | Human escalation policy verification, compound words (`hackathon`), and regex safety rules |
| [**Dataset Card**](docs/dataset_card.md) | Provenance, template redundancy analysis, class support counts, and sparsity flags |
| [**Model Card**](docs/model_card.md) | Specifications, parameters, SHA-256 hashes, and benchmark results for Qwen2.5-0.5B LoRA |
| [**Operational Limitations**](docs/limitations.md) | In-depth analysis of CPU generation latency, synthetic bias, and priority ambiguities |

---

## 4. Security & Hardening Highlights

Sprint A resolved all previously identified security and operational concerns:
- **Zero Default Secrets:** Hardcoded admin keys removed. Production mode strictly aborts startup if `NEXUS_ADMIN_KEY` is missing or below 32 characters entropy.
- **Zero Authentication Bypasses:** Completely eliminated test-runner auth bypasses (`PYTEST_CURRENT_TEST`). All tests authenticate via valid cryptographic headers.
- **Trusted Proxy Verification:** `resolve_client_ip` enforces that `X-Forwarded-For` is only respected if the immediate peer IP is listed in `cfg.trusted_proxies`.
- **Distributed Redis Rate Limiter:** Atomic Redis sliding window limiter with automatic memory fallback.
- **Full Test Suite Verification:** **215 passing automated tests, 1 skipped** (100% pass rate).

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

### Step 2: Run Full Automated Test Suite (215 Passing Tests)
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
