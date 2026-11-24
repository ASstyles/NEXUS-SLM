# NEXUS-SLM — Inference Optimisation & Engine Benchmark Report

## 1. Executive Summary

This report delivers a rigorous empirical benchmark and trade-off analysis across the three candidate inference runtimes evaluated for NEXUS-SLM:
1. **GGUF Runtime (`llama.cpp`) with 4-bit Quantization (`Q4_K_M`):** The experimental neural candidate running quantized weights of `Qwen2.5-0.5B-Instruct` with LoRA adapters.
2. **ONNX Runtime with Dynamic INT8 Quantization:** Compact neural runtime evaluated for edge CPU deployment.
3. **Heuristic Fallback V2 Engine:** High-throughput, deterministic pattern-matching engine executing contextual priority and safety rules.

---

## 2. Comparative Benchmark Matrix

All benchmarks were conducted on an Intel 8-Core (12 logical thread) CPU with 16GB RAM under standard CPU deployment parameters:

| Metric | Heuristic Fallback V2 | GGUF Q4_K_M (`llama.cpp`) | ONNX Dynamic INT8 | PyTorch FP32 Baseline |
|---|---:|---:|---:|---:|
| **Model Weight Size** | **0.0 MB** | 388.5 MB | 108.1 MB | 1,976.1 MB |
| **Process Peak RAM** | **212.6 MB** | ~980.0 MB | ~1,100.0 MB | ~2,450.0 MB |
| **Cold-Start Latency** | **< 0.05 s** | 1.82 s | 2.91 s | 3.65 s |
| **Inference Latency P50** | **5.17 ms** | 1,508.55 ms | 1,333.02 ms | 1,840.20 ms |
| **Inference Latency P95** | **7.30 ms** | 1,842.10 ms | 2,005.47 ms | 2,420.50 ms |
| **Throughput (1 client)** | **181.72 req/s** | 0.81 req/s | 0.70 req/s | 0.54 req/s |
| **Throughput (10 clients)** | **153.06 req/s** | 0.71 req/s | 0.65 req/s | 0.48 req/s |
| **JSON Structural Validity** | **100.00%** | **100.00%** | 40.48% (Severe failure) | 97.62% |
| **Category Accuracy (Test)** | 56.03% | 69.17% | 9.52% (Severe failure) | 68.90% |
| **Critical Safety Recall** | **100.00% (FN=0)** | 86.12% (FN=29) | < 20.00% | 88.50% |

---

## 3. Deep Engine Analysis & Findings

### 3.1 GGUF Q4_K_M (`llama.cpp`) Runtime
- **Strengths:**
  - Quantization from FP32 (1,976 MB) down to Q4_K_M (388.5 MB) achieves an **80.3% reduction in disk and memory footprint**.
  - Accuracy is preserved without structural degeneration: Category accuracy remains at 69.17%, with 100% JSON formatting validity.
  - Native Devanagari subword tokenization in Qwen2.5-0.5B completely eliminates byte-fallback tokens, enabling high Hindi/Hinglish accuracy (96.78% language detection).
- **Weaknesses:**
  - High CPU latency (P50 = 1.51 seconds).
  - High-risk false negatives: The model produced 29 false negatives on safety-critical tickets when evaluated without deterministic policy overrides.
  - Low concurrency throughput: Peak CPU utilization occurs at concurrency = 1 (~0.81 req/sec). Beyond 8 threads, CPU cache thrashing and context switching degrade throughput.

### 3.2 ONNX Dynamic INT8 Quantization Failure Mode
- **Critical Finding:** While ONNX INT8 reduced the model size to 108 MB, it suffered **catastrophic generation collapse**:
  - Category accuracy plummeted to **9.52%**.
  - JSON validity crashed to **40.48%**.
- **Root Cause:** Uniform dynamic affine INT8 quantization on autoregressive attention projections truncates activation outliers in self-attention heads. In Small Language Models ($< 1B$ parameters), attention outlier representations are critical for tracking syntax tokens and JSON brackets. Without FP16 mixed precision or SmoothQuant calibration, standard INT8 destroys generative SLM capabilities.

### 3.3 Heuristic Fallback V2 Engine
- **Strengths:**
  - **Sub-10ms Latency:** P50 of 5.17 ms and P95 of 7.30 ms enables instant responses for conversational support bots.
  - **Massive Concurrency:** Delivers over 150 requests/second on standard 8-core CPU hardware.
  - **Zero Safety Failure:** 100% Critical Recall with 0 False Negatives on both synthetic and sealed human holdouts.
  - **Zero Infrastructure Footprint:** Requires no GPU, zero neural weights, and boots instantaneously.
- **Weaknesses:**
  - Fixed lexical taxonomy: Relies on regular expressions and dictionary patterns, limiting flexibility on novel out-of-vocabulary human slang.

---

## 4. Multi-Thread Scaling Behavior on CPU

Benchmarking GGUF `n_threads` across CPU cores:
- **1 Thread:** Throughput: 0.28 req/s | P50: 3,570 ms
- **2 Threads:** Throughput: 0.52 req/s | P50: 1,920 ms (+85% speedup)
- **4 Threads:** Throughput: 0.74 req/s | P50: 1,550 ms (+42% speedup)
- **8 Threads:** Throughput: 0.81 req/s | P50: 1,508 ms (+9% speedup)
- **12 Threads (Hyperthreading):** Throughput: 0.76 req/s | P50: 1,580 ms (-6% slowdown due to hyperthread contention)

**Recommendation:** For containerized deployment, pin `n_threads = 4` per worker replica with 2 dedicated vCPUs to maximize core efficiency without context-switch penalties.

---

## 5. Production Deployment Architecture Decision

1. **Default Production Engine:** The **Production Heuristic Fallback V2** remains the primary active runtime. It satisfies all operational SLAs (sub-10ms latency, 150+ req/s throughput, 100% critical recall).
2. **Experimental Neural Engine:** The GGUF Q4_K_M Qwen candidate remains strictly designated as an **experimental shadow-pipeline candidate**. It must operate in an asynchronous shadow capacity until neural safety recall reaches 100% and CPU latency is amortized via speculative decoding or batching.
