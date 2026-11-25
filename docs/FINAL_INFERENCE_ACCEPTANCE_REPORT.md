# NEXUS-SLM — Final CPU Inference Optimization & Runtime Acceptance Report

> **Artifacts:** [runtime_inference_benchmarks_v2.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/runtime_inference_benchmarks_v2.json) | [benchmark_all_runtimes_v2.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/scripts/evaluation/benchmark_all_runtimes_v2.py) | [multitask_v2.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/src/models/multitask_v2.py)

---

## 1. Objective & Measurement Provenance

To resolve the historical CPU `2,212.92 ms` autoregressive Qwen bottleneck without fake latency adjustments (`+35.0 ms` heuristic offset removed in `api/main.py` and evaluation scripts), we measured **6 live execution runtimes** directly on Windows x64 CPU (12 logical CPUs) on a 28-sample category-stratified probe (`14`–`28` samples) from `test_clean.jsonl`:

- **Production Heuristic V3 (`src/inference/rules.py` + `policy.py`)**: Zero external weights.
- **Experimental Single-Forward Neural V2 (`exp_v2_5` Hybrid Head)**: Replaces autoregressive token-by-token JSON decoding with a single frozen Qwen2.5-0.5B encoder pass (`mean+last` pooling) + TF-IDF SVD + PyTorch residual head + deterministic JSON construction.
- **Quantized Generative Runtimes (`GGUF Q4_K_M`, `GGUF Q8_0`, `ONNX Dynamic INT8`)**: Note that the converted `.gguf` and `.onnx` binaries in `models/` originate from the V1 `qwen25_05b_lora_seed42` checkpoint and are measured strictly for CPU runtime throughput, RAM footprint, and quantization validity (**NOT** for clean-split accuracy claims).

---

## 2. Live CPU Benchmark Matrix

| Runtime Engine | Weight Disk Size | Cold Load | Peak Process RSS | JSON Validity | P50 Latency | **P95 Latency** | P99 Latency | Throughput | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`heuristic_fallback_v2_python` (V3)** | **0.0 MB** | **0.5 ms** | **269.94 MB** | **100.0%** | **0.09 ms** | **0.22 ms** | 4.45 ms | **4,937.51 req/s** | **PRODUCTION** |
| **`nexus_v2_hybrid_qwen05b_single_forward_head`** | **5.34 MB** (+base) | 2,356.23 ms | 2,429.93 MB | **100.0%** | **186.37 ms** | **288.78 ms** | 303.61 ms | **4.90 req/s** | **EXP_CANDIDATE** |
| `gguf_q4_k_m_t8` (Seed42 V1, 8 threads) | 379.38 MB | 917.59 ms | 901.96 MB | 100.0% | 1,315.57 ms | 1,572.39 ms | 1,732.38 ms | 0.78 req/s | Characterized (CPU) |
| `gguf_q4_k_m_t4` (Seed42 V1, 4 threads) | 379.38 MB | 1,270.91 ms | 900.46 MB | 100.0% | 1,403.32 ms | 1,630.21 ms | 1,781.39 ms | 0.73 req/s | Characterized (CPU) |
| `gguf_q8_0_t8` (Seed42 V1, 8 threads) | 506.47 MB | 1,109.15 ms | 997.97 MB | 100.0% | 1,669.33 ms | 2,508.07 ms | 3,000.67 ms | 0.56 req/s | Characterized (CPU) |
| `qwen25_05b_seed42_onnx_dynamic_int8` | 473.05 MB | 3,785.83 ms | 1,225.05 MB | **37.5%** | 2,752.68 ms | 5,003.32 ms | 5,066.80 ms | 0.30 req/s | **REJECTED** |
| *Legacy FP32 Qwen LoRA Autoregressive* (Baseline) | — | — | — | 100.0% | — | *2,212.92 ms* | — | — | Superseded |

---

## 3. Key Engineering Findings & Quantization Tradeoffs

1. **Single-Forward Encoder Architecture Eliminates Autoregressive Bottleneck (7.7x Faster P95):**
   - Autoregressive Qwen2.5-0.5B decoding generates ~35 tokens sequentially (`1.3–2.5 s` even quantized in GGUF `Q4_K_M`/`Q8_0`).
   - Switching to single-forward encoder pooling (`nexus_v2_hybrid_qwen05b_single_forward_head`) dropped CPU **P95 latency from `2,212.92 ms` to `288.78 ms`** while maintaining **100.0% JSON validity** by construction (`6.3x` faster than `0.78 req/s` GGUF Q4_K_M).
   - **Tradeoff:** Peak resident RSS rises to `2,429.93 MB` in FP32 PyTorch due to holding HF `Qwen2.5-0.5B-Instruct` weights alongside ORT/Python runtime allocations.
2. **Dynamic INT8 ONNX Linear Quantization Regression (`37.5%` JSON Validity):**
   - Just as observed historically on SmolLM-135M INT8 (`40.48%` validity), naive ONNX dynamic INT8 quantization on `Qwen2.5-0.5B` severely corrupts autoregressive decoding (`37.5%` valid JSON, `5,003.32 ms` P95 on CPU ORT) and is explicitly **REJECTED**.
3. **Production Runtime Recommendation:**
   - Continue serving **Heuristic Fallback V3 (`0.22 ms` direct P95, `269.94 MB` RSS, `0` external binaries)** in production while completing priority calibration and independent human evaluation for future single-forward neural candidates.
