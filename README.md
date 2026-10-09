# NEXUS-SLM — Neural EXecution for Unified Support Intelligence
### Efficient Small Language Model for Multilingual Support-Ticket Triage

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Tests](https://img.shields.io/badge/tests-61%2F61%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Production Readiness](https://img.shields.io/badge/readiness-READY%20WITH%20CONDITIONS-yellow.svg)](docs/production_readiness_report.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-009688.svg)](https://fastapi.tiangolo.com)

---

## 1. Project Overview
**NEXUS-SLM** is an enterprise small language model system engineered for low-latency, privacy-compliant, multilingual customer support ticket triage. Built specifically for code-switched environments including Indian English, Hindi (Devanagari script), and colloquial Hinglish (Roman Hindi), NEXUS-SLM accepts unstructured customer messages and produces strictly validated structured JSON output.

### Production Inference Contract (`POST /predict`)
```json
// Input Request
{
  "ticket": "Mera payment kat gaya but order confirm nahi hua."
}

// Strictly Validated Output
{
  "category": "payment_failure",
  "priority": "high",
  "language": "hinglish",
  "requires_human": false,
  "confidence": 0.7498,
  "latency_ms": 1824.5,
  "model_version": "nexus-qlora-v1",
  "inference_mode": "neural",
  "request_id": "9d023b34-d309-4ce8-b5e1-840250df7a69",
  "calibrated_confidence": 0.7498,
  "escalation_reason": "automated_routing_permitted"
}
```

---

## 2. Production Documentation Suite

| Document | Description |
|---|---|
| [System Architecture](docs/architecture.md) | End-to-end pipeline diagrams, tokenizer, parser, and policy architecture |
| [Deployment Guide](docs/deployment.md) | Setup, Docker containerization, Compose orchestration, resource requirements |
| [Inference Contract](docs/inference.md) | Strict API schema, 14 categories, 4 priorities, 3 languages, curl examples |
| [Monitoring & Data Drift](docs/monitoring.md) | Telemetry format, PSI drift calculations, percentiles, degradation alerts |
| [Security Audit](docs/security.md) | Credential scanning, prompt injection defense, DoS bounds, threat model |
| [Model Card](docs/model_card.md) | Parameters, training methodology, empirical benchmark metrics, safety |
| [Known Limitations](docs/limitations.md) | Devanagari subwords, short inputs, calibration gap, hardware boundaries |
| [Production Readiness Report](docs/production_readiness_report.md) | Full audit scorecard, load test results, final certification |

---

## 3. Final Candidate Lock & Empirical Results

The model candidate is permanently locked in `experiments/final_candidate.json`:
- **Model Version**: `nexus-qlora-v1`
- **Base Model**: `HuggingFaceTB/SmolLM-135M-Instruct` (135,436,608 parameters)
- **Adapter**: `models/nexus_qlora_v1` (LoRA $r=16, \alpha=32$, `q_proj`, `v_proj`)
- **Dataset**: `dataset_v1` (316 train, 44 val, 42 locked test)
- **Precision**: FP32 CPU Native Runtime (`torch.set_num_threads(12)`)
- **Adapter Hash**: `sha256_643971155ba3fe3a91379005fbe85d5ca95e73af90ec7cd65f871379a838983f`
- **Locked Test Hash**: `sha256_edf3fa5cd949a149ad5d759d81553d9704467491877d0f3c8b7cc5a887a9eda4`

### Benchmark Matrix (Locked 42-Sample Test Set)

| Metric | Base Model (SmolLM-135M) | NEXUS-SLM v1 (Production Candidate) | NEXUS-SLM v2 (Experimental) |
|---|:---:|:---:|:---:|
| **Macro F1** | 0.0000 | **0.4089** | 0.2224 |
| **Category Accuracy** | 0.0000 | **0.4286** | 0.2381 |
| **Priority F1** | 0.0000 | **0.2739** | 0.1421 |
| **Language Accuracy** | 0.0000 | **0.9524** | 0.6190 |
| **JSON Validity Rate** | 0.0000 | **0.9762** | 0.8333 |
| **P50 Latency (ms)** | 4888.22 | **1204.46** | 2326.77 |
| **P95 Latency (ms)** | 5396.91 | **1306.49** | 2551.76 |
| **RAM Footprint (MB)** | 13196.75 | **982.24** | 985.00 |

*Empirical Rationale*: V1 decisively outperforms V2 on Macro-F1 (+45.6% relative gain), Category Accuracy (+44.5%), and JSON Validity (+14.6%) while running nearly 2x faster.

---

## 4. Key Production Hardening Features

1. **Strict Inference Contract**: Accepts either `{"ticket": "..."}` or `{"text": "..."}`. Length enforced between 1 and 4,000 characters. Deterministic HTTP 422 structured rejections for empty, whitespace, or oversized payloads.
2. **Confidence Calibration**: Identified empirical Expected Calibration Error (ECE) of 0.588. Implemented conservative post-hoc calibration dampener (0.75x) and gated human review on confidence $< 0.65$.
3. **Deterministic Human Escalation**: Automatic routing to human agents for fraud reports, account deletions, GDPR requests, low confidence, and short inputs ($\le 3$ words).
4. **Adversarial Robustness**: Validated against 8 prompt injection attacks. 100% schema validity preserved; 0 unexpected fields leaked; 75% attacks triggered automated safety escalation.
5. **Observability & PSI Data Drift**: Structured JSONL logging with customer text omitted by default. Real-time PSI drift monitoring across language and category distributions.
6. **Zero-Downtime Rollback**: Instant model version switching via `POST /model/rollback?target_version=...` without container rebuilds.

---

## 5. Quickstart

### Step 1: Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

### Step 2: Run Full Automated Test Suite (61 Tests)
```powershell
pytest -v
```

### Step 3: Launch FastAPI Production Service
```powershell
python scripts/deployment/run_api.py
# Server binds to: http://localhost:8000
# OpenAPI Docs: http://localhost:8000/docs
```

### Step 4: Run Final End-to-End Validation
```powershell
python scripts/deployment/run_final_e2e_validation.py
```

---

## 6. License
This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
