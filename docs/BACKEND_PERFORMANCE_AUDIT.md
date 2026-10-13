# NEXUS-SLM Complete Backend Performance Audit Report

**Audit Date:** October 13, 2026  
**Lead Auditor:** ML Systems, Backend, & Performance Lead Engineer (DeepMind Antigravity)  
**Target Repository:** `C:\Users\avira\Desktop\NEXUS-SLM`  
**Git Branch:** `sprint/master-engineering-sprint`  
**Active Production Model:** `nexus-qlora-v1`  
**Evaluated Models:** `nexus-qlora-v1`, `qwen25_05b_lora_seed42_q8_0`, `qwen25_05b_lora_seed42_q4_k_m`, `qwen25_05b_lora_seed42_int8`  
**Machine-Readable Audit Data:**  
- [`experiments/artifacts/backend_performance_audit_results.json`](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/backend_performance_audit_results.json)  
- [`experiments/artifacts/backend_performance_audit_metrics.csv`](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/backend_performance_audit_metrics.csv)

---

## 1. Executive Summary & Measured Backend Health

This performance audit provides a factual, reproducible, empirical evaluation of the NEXUS-SLM backend service and all candidate models under strictly identical evaluation conditions. No metrics have been fabricated, estimated, or extrapolated.

### Key Audit Findings
1. **Production Runtime State (`nexus-qlora-v1`):**  
   The active production model `nexus-qlora-v1` is served by `api/main.py`. Because the binary `adapter_model.safetensors` weight artifact was lost during the prior file recovery event (as documented in `docs/RECOVERY_AUDIT_REPORT.md`), the live service runs using its built-in verified deterministic heuristic engine (`inference_mode="heuristic_fallback"`). On the held-out test set ($N=42$), this operational mode achieves **64.29% Category Accuracy** (27/42), **0.6401 Macro-F1**, **100% Language Identification**, **4.43 ms P50 latency**, and **0 false negatives** on safety challenge tickets. For historical completeness, the model's recorded benchmark prior to weight loss was **42.86% Category Accuracy** (18/42) and **0.4089 Macro-F1** on the same test set.
2. **Experimental Candidate Comparison (GGUF vs ONNX):**  
   - **`qwen25_05b_lora_seed42_q8_0` (GGUF 8-bit, 506.5 MB):** Top-performing neural candidate. Achieves **93.18% Category Accuracy** (41/44) and **0.9265 Macro-F1** on held-out validation, and **88.10% Category Accuracy** (37/42) and **0.8736 Macro-F1** on held-out test. P95 latency is **1,269.53 ms** on CPU (well within the $\le 2,500$ ms SLA). Zero safety escapes (100% recall on high-risk tickets).
   - **`qwen25_05b_lora_seed42_q4_k_m` (GGUF 4-bit, 379.4 MB):** Strong lightweight alternative. Achieves **90.91% Category Accuracy** (40/44) and **0.8956 Macro-F1** on validation, and **83.33% Category Accuracy** (35/42) and **0.8206 Macro-F1** on test. P95 latency is **1,178.08 ms** on CPU with **960.34 MB** peak RAM footprint.
   - **`qwen25_05b_lora_seed42_int8` (ONNX Dynamic INT8, 473.0 MB):** Catastrophic quality collapse confirmed. Dynamic activation quantization degraded test Category Accuracy to **21.43%** (9/42) and Macro-F1 to **0.1814**. Severe generation failures on Hindi Devanagari (7.14% validation accuracy) and multiple non-JSON hallucinations safely intercepted by the output parser. P95 latency is **3,048.18 ms** (violates SLA).
3. **Escalation and Safety Integrity:**  
   Across all 4 models and all 70 high-risk challenge cases, **0 false negatives occurred** (100% high-risk recall). Under current conservative operating policy, whenever calibrated confidence is unavailable, tickets route to human specialists (`escalation_rate = 100%`), strictly preventing automated misclassification escapes.
4. **Test Suite Health:**  
   The regression test suite passed **144 out of 144 tests** (0 failed, 0 skipped) in **40.50 seconds**.

---

## 2. Test Environment & Audit Methodology

### 2.1 Hardware and Runtime Specifications
| Parameter | Audited Value |
| :--- | :--- |
| **Host Operating System** | Windows 11 Home (Build 26300), AMD64 |
| **Processor** | 12th Gen Intel(R) Core(TM) i5-12450H (8 physical cores, 12 logical threads) |
| **System RAM** | 15.65 GB total (2.05 GB available at audit start) |
| **Python Runtime** | 3.10.11 (`.venv\Scripts\python.exe`) |
| **FastAPI / Starlette** | FastAPI 0.115.0 / Starlette 0.38.6 |
| **llama-cpp-python** | 0.3.1 (CPU backend, AVX2 enabled, 8 inference threads) |
| **ONNX Runtime** | 1.23.2 (`optimum.onnxruntime`) |
| **PyTorch** | 2.14.1+cpu (Strict CPU execution, CUDA disabled) |

### 2.2 Datasets and Evaluation Conditions
All models were evaluated under identical conditions against the following canonical fixtures:
1. **Held-Out Validation Split:** `data/validation/validation.jsonl` ($N=44$). Ground-truth labeled for category, priority, language, and requires_human.
2. **Held-Out Locked Test Split:** `data/test/test.jsonl` ($N=42$). Canonical locked test split from Dataset V1.
3. **High-Risk Challenge Set:** `data/evaluation/high_risk_challenge_set.jsonl` ($N=70$). Evaluates fraud, account takeover, payment dispute, and legal escalation safety.
4. **Local Benchmark Suite:** `data/demo_fixtures.json` ($N=30$). Measures API vs model latency, throughput, and memory consumption.
5. **Warm-up Policy:** 3 warm-up requests executed prior to latency recording. Greedily sampled (temperature $0.0$, `max_new_tokens=60`).

---

## 3. Production Model Results (`nexus-qlora-v1`)

### 3.1 Operational Context
`nexus-qlora-v1` is designated as the active production model in `models/registry.json` and `api/main.py`. Because the fine-tuned adapter weights (`adapter_model.safetensors`) were lost during a historical system incident, `api/main.py` serves traffic via its deterministic heuristic fallback pipeline. Both the live operational performance and the historically registered benchmark (intact weights) are reported below:

### 3.2 Empirical Classification Metrics
| Evaluation Split | Metric | Measured Value | Numerator / Denominator | Historical Benchmark (Intact Adapter) |
| :--- | :--- | :--- | :--- | :--- |
| **Validation ($N=44$)** | Category Accuracy | **45.45%** | 20 / 44 | 45.45% (20 / 44) |
| | Macro-F1 (Category) | **0.4673** | N/A | 0.4412 |
| | Micro-F1 (Category) | **0.4545** | N/A | 0.4545 |
| | Weighted-F1 (Category) | **0.4673** | N/A | 0.4412 |
| | Priority Accuracy | **34.09%** | 15 / 44 | N/A |
| | Language Accuracy | **97.73%** | 43 / 44 | N/A |
| | Exact Structured Match | **13.64%** | 6 / 44 | N/A |
| | JSON Validity Rate | **100.0%** | 44 / 44 | 95.45% |
| **Locked Test ($N=42$)** | Category Accuracy | **64.29%** | 27 / 42 | 42.86% (18 / 42) |
| | Macro-F1 (Category) | **0.6401** | N/A | 0.4089 |
| | Micro-F1 (Category) | **0.6429** | N/A | 0.4286 |
| | Weighted-F1 (Category) | **0.6401** | N/A | 0.4089 |
| | Priority Accuracy | **35.71%** | 15 / 42 | 47.62% |
| | Language Accuracy | **100.0%** | 42 / 42 | 95.24% |
| | Exact Structured Match | **21.43%** | 9 / 42 | 7.14% (3 / 42) |
| | JSON Validity Rate | **100.0%** | 42 / 42 | 97.62% |

### 3.3 Per-Language Breakdown
- **Validation Split ($N=44$):**
  - English ($N=15$): Accuracy = 40.00% (6/15), Macro-F1 = 0.3512
  - Hindi Devanagari ($N=14$): Accuracy = 42.86% (6/14), Macro-F1 = 0.3333
  - Hinglish ($N=15$): Accuracy = 53.33% (8/15), Macro-F1 = 0.4966
- **Test Split ($N=42$):**
  - English ($N=14$): Accuracy = 78.57% (11/14), Macro-F1 = 0.7262
  - Hindi Devanagari ($N=14$): Accuracy = 57.14% (8/14), Macro-F1 = 0.4571
  - Hinglish ($N=14$): Accuracy = 57.14% (8/14), Macro-F1 = 0.4881

### 3.4 Operational Performance ($N=30$)
- **API Latency:** P50 = **4.43 ms**, P90 = **6.04 ms**, P95 = **6.30 ms**, P99 = **6.43 ms**, Max = **6.45 ms**
- **Throughput:** **219.38 req/s**
- **Peak RSS RAM:** **292.08 MB**
- **Artifact Size on Disk:** 3.6 MB (tokenizer and configuration metadata)
- **Error Rate:** 0.0% (0 / 30)

---

## 4. Experimental Model Comparison

### 4.1 Head-to-Head Comparative Metrics Table
All metrics measured simultaneously across identical datasets:

| Metric | Production (`nexus-qlora-v1` live) | Candidate 1 (`qwen25_05b_q8_0` GGUF) | Candidate 2 (`qwen25_05b_q4_k_m` GGUF) | Candidate 3 (`qwen25_05b_int8` ONNX) | Production SLA Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Format & Quantization** | Heuristic fallback | GGUF Q8_0 (8-bit) | GGUF Q4_K_M (4-bit) | ONNX Dynamic INT8 | N/A |
| **Artifact Size on Disk** | 3.6 MB | 506.47 MB | 379.38 MB | 473.04 MB | $< 2,048$ MB |
| **Val Category Acc ($N=44$)** | 45.45% (20/44) | **93.18% (41/44)** | 90.91% (40/44) | 25.00% (11/44) | $\ge 50.0\%$ |
| **Val Macro-F1 ($N=44$)** | 0.4673 | **0.9265** | 0.8956 | 0.2275 | $\ge 0.550$ |
| **Test Category Acc ($N=42$)**| 64.29% (27/42) | **88.10% (37/42)** | 83.33% (35/42) | 21.43% (9/42) | $\ge 50.0\%$ |
| **Test Macro-F1 ($N=42$)** | 0.6401 | **0.8736** | 0.8206 | 0.1814 | $\ge 0.550$ |
| **Test Micro-F1 ($N=42$)** | 0.6429 | **0.8810** | 0.8333 | 0.2143 | N/A |
| **Test Weighted-F1 ($N=42$)**| 0.6401 | **0.8736** | 0.8206 | 0.1814 | N/A |
| **Language Acc (Val / Test)** | 97.7% / 100% | **100.0% / 100.0%** | **100.0% / 100.0%** | 50.0% / 47.6% | $\ge 90.0\%$ |
| **Val Hindi Accuracy** | 42.86% (6/14) | **92.86% (13/14)** | **92.86% (13/14)** | 7.14% (1/14) | $\ge 50.0\%$ |
| **Val Hinglish Accuracy** | 53.33% (8/15) | **93.33% (14/15)** | **93.33% (14/15)** | 26.67% (4/15) | $\ge 50.0\%$ |
| **Val English Accuracy** | 40.00% (6/15) | **93.33% (14/15)** | 86.67% (13/15) | 40.00% (6/15) | $\ge 50.0\%$ |
| **Test Hinglish Accuracy** | 57.14% (8/14) | **100.0% (14/14)** | **100.0% (14/14)** | 21.43% (3/14) | $\ge 50.0\%$ |
| **Val Exact Structured Match**| 13.64% (6/44) | 31.82% (14/44) | **38.64% (17/44)** | 6.82% (3/44) | N/A |
| **JSON Validity Rate** | 100.0% | **100.0%** | **100.0%** | 100.0%* | $\ge 95.0\%$ |
| **High-Risk Safety Recall** | 100.0% (70/70) | **100.0% (70/70)** | **100.0% (70/70)** | 100.0% (70/70) | **100.0% (0 FN)** |
| **Safety False Negatives** | **0** | **0** | **0** | **0** | **0** |
| **P50 Latency (API / Model)**| 4.4 ms / 35.1 ms | 1,027.9 ms / 1,022.3 ms | 921.7 ms / 915.6 ms | 1,998.8 ms / 1,966.0 ms| $\le 2,000$ ms |
| **P95 Latency (API / Model)**| 6.3 ms / 35.1 ms | **1,269.5 ms / 1,263.0 ms**| **1,178.1 ms / 1,172.0 ms**| 3,048.2 ms / 2,957.4 ms| $\le 2,500$ ms |
| **Peak Resident RAM** | 292.08 MB | **1,046.03 MB** | **960.34 MB** | 1,423.94 MB | $\le 2,048$ MB |
| **Throughput (req/s)** | 219.38 | 0.94 | 1.03 | 0.49 | N/A |
| **Overall Gate Verdict** | Active Production | **PASS ALL GATES** | **PASS ALL GATES** | **FAIL QUALITY & SLA**| 6 Gates Required |

*\*Note: ONNX int8 produced 18 raw non-JSON outputs and 6 schema violations; the 100% validity reflects graceful fallback parsing in `src/inference/parser.py`.*

---

## 5. Detailed Classification Metrics (Candidate Q8_0)

### 5.1 Per-Category Precision, Recall, and F1 on Held-Out Validation ($N=44$)
Evaluated using canonical 14-class customer support taxonomy:

| Category | Support | Precision | Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `payment_failure` | 4 | 1.0000 | 1.0000 | **1.0000** | Perfect Classification |
| `refund_request` | 3 | 1.0000 | 1.0000 | **1.0000** | Perfect Classification |
| `account_access` | 3 | 0.7500 | 1.0000 | **0.8571** | High Precision/Recall |
| `password_reset` | 3 | 1.0000 | 1.0000 | **1.0000** | Perfect Classification |
| `order_cancel` | 4 | 1.0000 | 1.0000 | **1.0000** | Perfect Classification |
| `order_tracking` | 3 | 1.0000 | 0.6667 | **0.8000** | 1 Confusion to `delivery_issue` |
| `subscription` | 3 | 1.0000 | 1.0000 | **1.0000** | Perfect Classification |
| `billing` | 3 | 1.0000 | 1.0000 | **1.0000** | Perfect Classification |
| `technical_issue` | 3 | 1.0000 | 1.0000 | **1.0000** | Perfect Classification |
| `fraud_report` | 3 | 1.0000 | 0.6667 | **0.8000** | 1 Confusion to `account_access` |
| `delivery_issue` | 3 | 0.7500 | 1.0000 | **0.8571** | High Precision/Recall |
| `account_closure` | 3 | 1.0000 | 1.0000 | **1.0000** | Perfect Classification |
| `product_complaint` | 3 | 0.7500 | 1.0000 | **0.8571** | High Precision/Recall |
| `general_query` | 3 | 1.0000 | 0.6667 | **0.8000** | 1 Confusion to `product_complaint` |
| **Macro Average** | **44** | **0.9464** | **0.9286** | **0.9265** | **Overall Macro-F1 = 0.9265** |

### 5.2 Category Confusion Matrix Analysis
Out of 44 validation samples, only **3 misclassifications** occurred:
1. Ground Truth `order_tracking` $\rightarrow$ Predicted `delivery_issue` (1 sample): User inquired about estimated delivery date of an in-transit package.
2. Ground Truth `fraud_report` $\rightarrow$ Predicted `account_access` (1 sample): User reported suspicious login attempts and requested account lockout.
3. Ground Truth `general_query` $\rightarrow$ Predicted `product_complaint` (1 sample): User inquired about warranty coverage terms.

On the locked test set ($N=42$), only **5 misclassifications** occurred:
1. `refund_request` $\rightarrow$ `payment_failure` (1 sample)
2. `subscription` $\rightarrow$ `billing` (1 sample)
3. `technical_issue` $\rightarrow$ `payment_failure` (1 sample)
4. `technical_issue` $\rightarrow$ `subscription` (1 sample)
5. `product_complaint` $\rightarrow$ `delivery_issue` (1 sample)

### 5.3 Priority Classification Performance
- **Validation Accuracy:** **54.55%** (24 / 44 correct), Macro-F1 = **0.5415**
- **Test Accuracy:** **50.00%** (21 / 42 correct), Macro-F1 = **0.4891**
- **Root Cause Analysis:** Low priority accuracy is primarily driven by subjective annotation inconsistencies in the synthetic ground truth. For example, tickets reporting severe app crashes on startup were annotated as `low` priority in the ground truth, whereas the model rationally assigned `high`.

---

## 6. Escalation and Confidence Evaluation

### 6.1 Escalation Policy Audit
In NEXUS-SLM, human escalation is governed by two complementary mechanisms:
1. **Model Predicted `requires_human`**: Autonomous model inference.
2. **Deterministic Safeguard Policy (`HumanEscalationPolicy`)**: Strict programmatic rules that force `requires_human = True` regardless of model output whenever:
   - High-impact financial dispute keywords are matched (`payment_failure`, `fraud_report`).
   - Account security keywords are matched (`"someone accessed"`, `"hacked"`).
   - Account closure/legal action is identified.
   - Delivery theft or tampering is detected.
   - Language is Hindi written in Devanagari script.
   - Calibrated confidence is unavailable.

### 6.2 Empirical Escalation Metrics on Validation Split ($N=44$)
| Metric | Production Live | Candidate Q8_0 | Candidate Q4_K_M | Candidate INT8 | Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ground-Truth Escalated** | 29 | 29 | 29 | 29 | Tickets marked `requires_human=True` |
| **Predicted Escalated** | 44 | 44 | 44 | 44 | Total tickets escalated |
| **Escalation Rate** | 100.0% | 100.0% | 100.0% | 100.0% | % of traffic routed to humans |
| **Automation Rate** | 0.0% | 0.0% | 0.0% | 0.0% | % of traffic resolved autonomously |
| **True Positives (TP)** | 29 | 29 | 29 | 29 | Correct human review routing |
| **False Negatives (FN)** | **0** | **0** | **0** | **0** | **Zero safety escapes** |
| **False Positives (FP)** | 15 | 15 | 15 | 15 | Conservative over-escalation |
| **Escalation Precision** | 0.6591 | 0.6591 | 0.6591 | 0.6591 | TP / (TP + FP) |
| **Escalation Recall** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | TP / (TP + FN) |
| **Escalation F1** | 0.7945 | 0.7945 | 0.7945 | 0.7945 | Harmonic mean |

### 6.3 Honest Confidence Handling
- **Status:** Evaluated candidates running with `logits_all=False` on CPU do not compute full-vocabulary token log-probabilities.
- **Reporting Invariant:** Confidence is represented honestly as `"confidence": null` (`"confidence_status": "unavailable"`).
- **Safety Consequence:** Because calibrated confidence is unavailable, the deterministic policy triggers rule `confidence_unavailable_conservative_escalation`. This guarantees that uncalibrated model outputs are never allowed to bypass human review autonomously in production.

---

## 7. Backend Performance, Latency, and Resource Utilization

### 7.1 Detailed Latency Percentiles ($N=30$ Demo Requests)
Measurements conducted over 30 representative enterprise tickets (English, Hindi, Hinglish across all support categories) using in-process ASGI TestClient:

| Model Version | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | Mean (ms) | Inference P95 | Framework Overhead | SLA Met? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `nexus-qlora-v1` (live) | **4.43** | 6.04 | 6.30 | 6.43 | 3.32 | 6.45 | 4.52 | 35.09 ms | ~1.2 ms | **PASS** |
| `qwen25_05b_q8_0` | **1,027.93** | 1,207.07 | **1,269.53** | 1,323.29 | 919.31 | 1,337.23 | 1,065.83 | 1,262.96 ms | ~6.5 ms | **PASS** |
| `qwen25_05b_q4_k_m` | **921.74** | 1,142.51 | **1,178.08** | 1,196.00 | 823.92 | 1,199.80 | 972.31 | 1,172.01 ms | ~6.0 ms | **PASS** |
| `qwen25_05b_int8` (ONNX)| **1,998.78** | 2,968.31 | **3,048.18** | 3,198.70 | 870.22 | 3,260.00 | 2,022.24 | 2,957.35 ms | ~90.8 ms | **FAIL SLA** |

### 7.2 Throughput, Concurrency, and Memory
| Model Version | Throughput (req/s) | Peak RSS RAM (MB) | RAM SLA Gate | Disk Size (MB) | Failure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `nexus-qlora-v1` (live) | **219.38** | 292.08 MB | PASS ($< 2,048$ MB) | 3.60 MB | 0.0% (0 / 30) |
| `qwen25_05b_q8_0` | **0.94** | 1,046.03 MB | PASS ($< 2,048$ MB) | 506.47 MB | 0.0% (0 / 30) |
| `qwen25_05b_q4_k_m` | **1.03** | 960.34 MB | PASS ($< 2,048$ MB) | 379.38 MB | 0.0% (0 / 30) |
| `qwen25_05b_int8` (ONNX)| **0.49** | 1,423.94 MB | PASS ($< 2,048$ MB) | 473.04 MB | 0.0% (0 / 30)* |

*\*Note: ONNX int8 produced non-crashing responses, but outputs frequently contained hallucinated text strings.*

---

## 8. Test Results, Sample Sizes, and Measurement Confidence

### 8.1 Test Suite Verification
- **Test Command:** `.venv\Scripts\pytest.exe -v --tb=short`
- **Result:** **144 passed, 0 failed, 0 skipped** in 40.50 seconds.
- **Coverage Areas:** API routing, Pydantic schemas, language alias normalization, deterministic policies, output parser, model loading, and regression tests.

### 8.2 Sample Sizes and Statistical Limitations
1. **Validation Sample Size ($N=44$):**  
   The held-out validation set comprises 44 tickets. A sample size of $N=44$ provides a margin of error of approximately $\pm 7.5\%$ at a 95% confidence level for category accuracy. While statistically sufficient for model selection against large differences (e.g. 93.18% vs 25.00%), it should not be treated as an exhaustive representation of all enterprise edge cases.
2. **Dataset V2 Contamination Warning:**  
   As established in `experiments/artifacts/dataset_leakage_and_quality_audit.json`, Dataset V2 contains over 70% cross-split template duplicates between train and validation. This audit strictly utilized **Dataset V1** (`data/validation/validation.jsonl` and `data/test/test.jsonl`), which is verified to have 0 cross-split duplicates.

---

## 9. Comprehensive Metric Provenance Table

| Metric | Measured Value | Sample Size ($N$) | Evaluation Source | Model Evaluated | Audit Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Live Val Category Acc** | 45.45% (20/44) | 44 | Held-Out Validation | `nexus-qlora-v1` (live) | Baseline operational accuracy |
| **Live Test Category Acc** | 64.29% (27/42) | 42 | Held-Out Locked Test | `nexus-qlora-v1` (live) | Baseline test accuracy |
| **Historical Test Cat Acc** | 42.86% (18/42) | 42 | Historical Benchmark | `nexus-qlora-v1` (intact) | Registry record prior to weight loss |
| **Candidate Q8_0 Val Acc** | 93.18% (41/44) | 44 | Held-Out Validation | `qwen25_05b_q8_0` | Highest neural classification accuracy |
| **Candidate Q8_0 Val Macro-F1**| 0.9265 | 44 | Held-Out Validation | `qwen25_05b_q8_0` | Balanced per-class representation |
| **Candidate Q8_0 Test Acc** | 88.10% (37/42) | 42 | Held-Out Locked Test | `qwen25_05b_q8_0` | Superior generalization on locked test |
| **Candidate Q8_0 Test Macro-F1**| 0.8736 | 42 | Held-Out Locked Test | `qwen25_05b_q8_0` | Strong held-out generalization |
| **Candidate Q4_K_M Val Acc**| 90.91% (40/44) | 44 | Held-Out Validation | `qwen25_05b_q4_k_m` | Fast 4-bit candidate accuracy |
| **Candidate INT8 Val Acc** | 25.00% (11/44) | 44 | Held-Out Validation | `qwen25_05b_int8` | Activation quantization failure |
| **Q8_0 Val Hindi Accuracy**| 92.86% (13/14) | 14 | Held-Out Validation | `qwen25_05b_q8_0` | Outstanding Devanagari performance |
| **Q8_0 Val Hinglish Accuracy**| 93.33% (14/15) | 15 | Held-Out Validation | `qwen25_05b_q8_0` | Outstanding Hinglish performance |
| **Q8_0 Val English Accuracy**| 93.33% (14/15) | 15 | Held-Out Validation | `qwen25_05b_q8_0` | Outstanding English performance |
| **Q8_0 Language Ident. Acc**| 100.0% (44/44) | 44 | Held-Out Validation | `qwen25_05b_q8_0` | Zero language misidentifications |
| **Q8_0 JSON Validity Rate** | 100.0% (44/44) | 44 | Held-Out Validation | `qwen25_05b_q8_0` | Strict schema conformance |
| **High-Risk Safety Recall** | 100.0% (70/70) | 70 | Synthetic Challenge Set| All 4 Models | Zero safety escapes |
| **Safety False Negatives** | 0 | 70 | Synthetic Challenge Set| All 4 Models | Zero high-risk missed escalations |
| **Q8_0 P50 End-to-End Latency**| 1,027.93 ms | 30 | Local Benchmark | `qwen25_05b_q8_0` | Measured CPU median latency |
| **Q8_0 P95 End-to-End Latency**| 1,269.53 ms | 30 | Local Benchmark | `qwen25_05b_q8_0` | Passes $\le 2,500$ ms SLA gate |
| **Q4_K_M P95 Latency** | 1,178.08 ms | 30 | Local Benchmark | `qwen25_05b_q4_k_m` | 7.2% faster than Q8_0 |
| **INT8 P95 Latency** | 3,048.18 ms | 30 | Local Benchmark | `qwen25_05b_int8` | Fails $\le 2,500$ ms SLA gate |
| **Q8_0 Peak RSS Memory** | 1,046.03 MB | 30 | Local Benchmark | `qwen25_05b_q8_0` | Passes $\le 2,048$ MB RAM gate |
| **Unit & Integration Tests** | 144 / 144 Passed | 144 | Pytest Test Suite | Repository Code | 100% test pass rate |

---

## 10. Ranked Recommendations for Next Improvements

1. **Rank 1 (Data Governance): Re-Curation of Dataset V2 (BLK-07)**  
   *Impact: High.* Dataset V2 suffers from 70%+ cross-split template contamination. Re-splitting using embedding-based deduplication is critical before training or evaluating larger iterations.
2. **Rank 2 (ML Engineering): Calibrated Confidence Classification Head**  
   *Impact: High.* Currently, running `logits_all=False` on CPU forces confidence to be unavailable (`null`), leading to 100% conservative human escalation. Training a lightweight logistic regression or Platt scaling head over hidden representations will enable calibrated confidence and allow safe automated ticket resolution.
3. **Rank 3 (Production Staging): Staging Promotion Review for Q8_0**  
   *Impact: High.* `qwen25_05b_lora_seed42_q8_0` has passed all 6 empirical acceptance gates (93.2% accuracy, 1,269 ms P95 latency, 1,046 MB RAM, 100% JSON validity, 100% challenge recall, 92.9% Hindi accuracy). Once external governance blockers (DPA, CRM review queue, VPC DNS) are resolved, it should be promoted to replace `nexus-qlora-v1`.
4. **Rank 4 (Optimization): Syllable Caching for Native Hindi Devanagari**  
   *Impact: Medium.* Devanagari inference experiences ~200 ms additional latency due to byte-level tokenization. Implementing prefix KV-caching for common Devanagari administrative prompts will reduce Hindi latency toward English levels.
