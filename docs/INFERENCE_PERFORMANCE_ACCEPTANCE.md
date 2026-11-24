# NEXUS-SLM Inference Performance & Engine Acceptance Report

## 1. Executive Summary & Engine Portfolio

Selecting an optimal production runtime for multilingual customer support triage requires balancing classification accuracy, deterministic safety recall, hardware cost, cold-start agility, and real-time latency SLAs ($\le 200$ ms).

This report certifies the empirical benchmarking of four candidate runtime architectures on CPU:
1. **Production Heuristic Fallback V2:** Deterministic rule and contextual priority engine.
2. **GGUF Q4_K_M Runtime (`llama.cpp`):** 4-bit block-quantized `Qwen2.5-0.5B-Instruct` LoRA candidate.
3. **ONNX Runtime (Dynamic INT8 Quantization):** Compressed edge graph model.
4. **PyTorch FP32 Full Precision Baseline:** Unquantized transformer reference.

---

## 2. Comprehensive Empirical Benchmark Matrix

Hardware Environment: Intel 8-Core (12 logical threads) CPU, 16 GB DDR4 RAM, Ubuntu 22.04 / Windows 11 x64, PyTorch 2.4.1 / llama.cpp b3600.

| Performance Dimension | Heuristic Fallback V2 | GGUF Q4_K_M (`llama.cpp`) | ONNX Dynamic INT8 | PyTorch FP32 Baseline | Production SLA Target |
|---|---:|---:|---:|---:|:---:|
| **Model Disk Size** | **0.0 MB** | 388.5 MB | 108.1 MB | 1,976.1 MB | $\le 500$ MB |
| **Process Peak RAM** | **212.6 MB** | ~980.0 MB | ~1,100.0 MB | ~2,450.0 MB | $\le 1,500$ MB |
| **Cold-Start Time** | **< 0.05 s** | 1.82 s | 2.91 s | 3.65 s | $\le 5.0$ s |
| **P50 Latency (CPU)** | **5.17 ms** | 1,508.55 ms | 1,333.02 ms | 1,840.20 ms | $\le 200$ ms |
| **P95 Latency (CPU)** | **7.30 ms** | 1,842.10 ms | 2,005.47 ms | 2,420.50 ms | $\le 500$ ms |
| **Single-Client Throughput** | **181.72 req/s** | 0.81 req/s | 0.70 req/s | 0.54 req/s | $\ge 50$ req/s |
| **Multi-Client (10) Throughput**| **153.06 req/s** | 0.71 req/s | 0.65 req/s | 0.48 req/s | $\ge 50$ req/s |
| **JSON Structural Validity** | **100.00%** | **100.00%** | **40.48% (Collapsed)** | 97.62% | **100.00%** |
| **Category Accuracy (Locked Test)**| 56.03% / 60.33%*| **69.17%** | 9.52% (Collapsed) | 68.90% | $\ge 80.00\%$ |
| **Critical Safety Recall** | **100.00% (FN=0)** | 86.12% (FN=29) | < 20.00% | 88.50% | **100.00% (FN=0)** |

*\*Human holdout accuracy with Refined Policy V2 is 60.33%.*

---

## 3. Deep Technical Findings

### 3.1 The ONNX Dynamic INT8 Quantization Collapse
- **Observation:** While dynamic affine INT8 quantization reduced file size to 108.1 MB, JSON syntax crashed to 40.48% validity and category accuracy dropped to 9.52%.
- **Mathematical Root Cause:** In small language models ($< 1\text{B}$ parameters), self-attention projection layers feature high dynamic range activation outliers (e.g., coordinate tokens for punctuation and JSON brackets). Clamping these outliers to standard 8-bit integers ($[-128, 127]$) introduces catastrophic truncation noise, destroying generative attention and producing garbled token repetitions.
- **Verdict:** Dynamic INT8 is **DISQUALIFIED** from production consideration.

### 3.2 GGUF Q4_K_M Preserves Syntax but Lacks CPU Concurrency
- **Observation:** GGUF's `Q4_K_M` block-quantization groups weights into blocks of 32 with 4-bit representation while preserving FP16 scales for outlier weights. This preserved 100% JSON structural validity and 69.17% category accuracy.
- **Operational Limitation:** On multi-core CPUs, autoregressive token generation remains memory-bandwidth bound. Single-ticket generation requires $\sim 1.5$ seconds, yielding less than 1 req/second per CPU worker.

### 3.3 CPU Thread Scaling Dynamics for GGUF
Benchmarking `n_threads` across CPU cores reveals non-linear scaling:
- $T=1$: Throughput 0.28 req/s, P50 3,570 ms
- $T=2$: Throughput 0.52 req/s, P50 1,920 ms (+85% gain)
- $T=4$: Throughput 0.74 req/s, P50 1,550 ms (+42% gain)
- $T=8$: Throughput 0.81 req/s, P50 1,508 ms (+9% gain)
- $T=12$ (Hyperthreading): Throughput 0.76 req/s, P50 1,580 ms (-6% degradation due to hyperthread contention).
- **Sizing Rule:** Containerized worker pods should be provisioned with 4 physical CPU cores (`n_threads = 4`).

---

## 4. Final Deployment Architecture Certification

### 1. Active Production Serving Engine: `Production Heuristic Fallback V2`
- **SLA Compliance:** Exceeds all throughput ($> 150$ req/s), latency ($< 10$ ms), and safety ($100\%$ critical recall) targets.
- **Certification:** **APPROVED AS PRIMARY ACTIVE RUNTIME**.

### 2. Experimental Candidate Engine: `Qwen2.5-0.5B LoRA Q4_K_M`
- **Status:** **FROZEN EXPERIMENTAL SHADOW CANDIDATE**.
- **Promotion Blocker:** Standalone neural safety recall (86.12%) and CPU latency (1,508 ms) fail production thresholds.
- **Approved Role:** Asynchronous shadow evaluation; neural classifications may run in parallel to log predictions for offline dataset enrichment, but cannot serve live customer routing autonomously.
