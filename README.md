# NEXUS-SLM — Neural EXecution for Unified Support Intelligence
### Efficient Small Language Model for Multilingual Support-Ticket Triage

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Tests](https://img.shields.io/badge/tests-160%2F160%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Production Readiness](https://img.shields.io/badge/readiness-PRODUCTION%20HEURISTIC%20FALLBACK%20ACTIVE-yellow.svg)](docs/production_readiness_report.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-009688.svg)](https://fastapi.tiangolo.com)

---

## 1. Project Overview
**NEXUS-SLM** is an enterprise small language model system engineered for low-latency, privacy-compliant, multilingual customer support ticket triage. Built specifically for code-switched environments including Indian English, Hindi (Devanagari script), and colloquial Hinglish (Roman Hindi), NEXUS-SLM accepts unstructured customer messages and produces strictly validated structured JSON output across 14 categories, 4 priorities, and 3 languages.

### Production Active Mode & Heuristic Fallback
In air-gapped CPU deployments where heavy neural weights are decoupled or when running in offline mode, NEXUS-SLM activates its **verified deterministic heuristic fallback engine**:
- **Zero Fabricated Confidence**: When running under heuristic fallback, `confidence` is set to `null` and `confidence_status` is `"unavailable"`. The system never invents synthetic confidence scores.
- **Strict Safety Escalation**: Deterministic policies guarantee 0 false negatives on high-risk transactions (fraud reports, unauthorized debit, account compromise, GDPR data erasure).
- **Sub-5ms Latency**: Delivers P50 latency of ~4.6 ms on standard CPU hardware with 100% JSON validity.

### Inference Contract (`POST /predict`)
```json
// Input Request
{
  "text": "Mera payment kat gaya but order confirm nahi hua."
}

// Validated Output (Production Heuristic Fallback)
{
  "category": "payment_failure",
  "priority": "high",
  "language": "hinglish",
  "requires_human": true,
  "confidence": null,
  "confidence_status": "unavailable",
  "latency_ms": 4.62,
  "model_version": "nexus-qlora-v1",
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

## 2. Production Documentation Suite

| Document | Description |
|---|---|
| [Clean Data GPU Training Results](docs/CLEAN_DATA_GPU_TRAINING_RESULTS.md) | Validation audit, clean test results (N=746), and GPU candidate metrics |
| [Kaggle Automation Repair Report](docs/KAGGLE_AUTOMATION_REPAIR_REPORT.md) | Security audit, Zip Slip fixes, unique run isolation, and regression verification |
| [System Architecture](docs/architecture.md) | End-to-end pipeline diagrams, tokenizer, parser, and policy architecture |
| [Deployment Guide](docs/deployment.md) | Setup, Docker containerization, Compose orchestration, resource requirements |
| [Inference Contract](docs/inference.md) | Strict API schema, 14 categories, 4 priorities, 3 languages, curl examples |
| [Monitoring & Data Drift](docs/monitoring.md) | Telemetry format, PSI drift calculations, percentiles, degradation alerts |
| [Security Audit](docs/security.md) | Credential scanning, prompt injection defense, DoS bounds, threat model |
| [Model Card](docs/model_card.md) | Parameters, training methodology, empirical benchmark metrics, safety |
| [Clean Data Training & Validation](docs/CLEAN_DATA_TRAINING_AND_VALIDATION_REPORT.md) | Discrepancy analysis, template grouping isolation, sample size resolution |

---

## 3. Dataset V2 Clean-Split Governance & Benchmarks

The project enforces strict template-grouped dataset partitioning (`data/v2_clean_template_grouped/`) with zero text and zero template overlap across splits:
- **Training Split**: `train_clean.jsonl` ($N=4,797$, SHA-256: `f67b50c7...`)
- **Validation Split**: `val_clean.jsonl` ($N=449$, SHA-256: `faa98cf8...`)
- **Locked Test Split**: `test_clean.jsonl` ($N=746$, SHA-256: `fc65071b...`) — **Strictly quarantined** from training and tuning.

### Comprehensive Evaluation on Locked Clean Test Set ($N=746$)

| Metric | Production Fallback (`nexus-qlora-v1`) | GGUF Q8_0 Experimental (`qwen25_05b_lora_seed42_q8_0`)* |
|---|:---:|:---:|
| **Sample Count** | **746 (Full Split)** | 100 (Stratified Sample) |
| **Category Accuracy** | **61.80%** (461/746) | 90.91% |
| **Macro-F1** | **0.5473** | 0.8974 |
| **Weighted-F1** | **0.6356** | 0.9021 |
| **Priority Accuracy** | **50.54%** | 86.36% |
| **Language Accuracy** | **99.87%** (English: 100%, Hindi: 100%, Hinglish: 99.6%) | 93.33% |
| **JSON Validity Rate** | **100.0%** | 100.0% |
| **Exact Structured Match** | **38.07%** | 40.91% |
| **High-Risk False Negatives** | **0 (100% Recall)** | 0 (100% Recall) |
| **P50 Latency (ms)** | **4.62 ms** | 1157.0 ms |
| **P95 Latency (ms)** | **5.86 ms** | 1374.8 ms |
| **Throughput (req/s)** | **204.6 req/s** | 0.86 req/s |
| **Peak Memory (MB)** | **288.7 MB** | 1479.4 MB |

*\*Note: GGUF candidate evaluated on earlier validation slice; clean V2 training produces `qwen25_05b_clean_v2_lora`.*

---

## 4. Model Registry Status

| Version | Status | Lifecycle Stage | Base Model | Path | Notes |
|---|---|---|---|---|---|
| `nexus-qlora-v1` | **Active** | Production | `SmolLM-135M-Instruct` | `models/nexus_qlora_v1` | Active production service with verified deterministic heuristic fallback |
| `qwen25_05b_clean_v2_lora` | Proposed | Experimental | `Qwen2.5-0.5B-Instruct` | `experiments/artifacts/training_runs/...` | Clean-data LoRA candidate from cloud GPU training pipeline |
| `qwen25_05b_lora_seed42_q8_0` | Qualified | Experimental | `Qwen2.5-0.5B-Instruct` | `models/qwen25_05b_lora_seed42_gguf_q8_0/...` | Weight-only 8-bit GGUF candidate (passes quality gates) |
| `qwen25_05b_lora_seed42_q4_k_m` | Qualified | Experimental | `Qwen2.5-0.5B-Instruct` | `models/qwen25_05b_lora_seed42_gguf_q4_k_m/...` | Weight-only 4-bit GGUF candidate |
| `qwen25_05b_lora_seed42_int8` | Rejected | Experimental | `Qwen2.5-0.5B-Instruct` | `models/qwen25_05b_lora_seed42_int8` | FAILED quality gates (dynamic activation degradation) |
| `test-nexus-v0.9` | Unverified | Mock Staging | `Qwen2.5-1.5B-Instruct` | `models/test` (Missing) | Synthetic unit-test registry fixture; no physical weights on disk |

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

### Step 2: Run Full Automated Test Suite (160 Passing Tests)
```powershell
python -m pytest
```

### Step 3: Launch Local Inference Service
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Test Triage Prediction via cURL / PowerShell
```powershell
# Test cURL
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"text": "Mera account hack ho gaya hai unauthorized debit hua hai"}'

# Test Health Check
curl "http://localhost:8000/health"
```

### Step 5: Clean-Data GPU Training Automation
```powershell
# 1. Verify Kaggle authentication status safely
python scripts/training/submit_kaggle_gpu_job.py --check-only

# 2. Submit clean-data GPU training job (requires Kaggle credentials)
python scripts/training/submit_kaggle_gpu_job.py
```

---

## 6. License
This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
