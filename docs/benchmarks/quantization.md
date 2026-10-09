# Quantization & Precision Benchmark: NEXUS-SLM-135M

## 1. Executive Summary

This study benchmarks inference precision formats for the champion candidate **NEXUS-SLM-135M-LoRA** (`nexus-qlora-v1`) to determine the trade-off between memory footprint, computational latency, and classification accuracy.

Benchmarks were executed on the locked test set (`data/test/test.jsonl`, 42 instances) using identical evaluation metrics and deterministic validation.

---

## 2. Precision Comparison Matrix

| Precision Format | Model Disk Size | Macro-F1 | Category Acc | JSON Validity | P50 Latency | P95 Latency | Throughput | Peak RAM | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **FP32 (Baseline)** | **516.6 MB** | **0.4089** | **42.86%** | **97.62%** | **1,210 ms** | **1,325 ms** | **0.82 req/s** | **14.4 GB** | **PRODUCTION** |
| **INT8 (Dynamic)** | **108.1 MB** | **0.0944** | **9.52%** | **40.48%** | **1,333 ms** | **2,005 ms** | **0.70 req/s** | **14.5 GB** | **EXPERIMENTAL** |
| **FP16 / BF16** | *257.5 MB* | *None* | *None* | *None* | *None* | *None* | *None* | *None* | **UNSUPPORTED** |
| **INT4 (NF4/FP4)**| *135.0 MB* | *None* | *None* | *None* | *None* | *None* | *None* | *None* | **UNSUPPORTED** |

---

## 3. Analysis of Supported Quantization Formats

### FP32 (Full Precision Baseline)
- **Model Footprint:** 516.6 MB dedicated parameters.
- **Task Accuracy:** Achieves full Macro-F1 of **0.4089** (40.89%) and **97.62%** JSON schema compliance.
- **Latency Profile:** Predictable P50 of 1,210 ms, P95 of 1,325 ms.
- **Finding:** Recommended production runtime format on CPU architecture.

### INT8 (PyTorch Post-Training Dynamic Quantization)
- **Compression:** Achieved **4.78x reduction** in linear layer weight storage (down to 108.1 MB).
- **Severe Accuracy Degradation:** Macro-F1 dropped catastrophically from **0.4089 to 0.0944**; JSON validity plunged from **97.62% to 40.48%**.
- **Latency Regressions:** P95 latency increased by 51.4% (1,325 ms $\rightarrow$ 2,005 ms). Dynamic quantization on CPU incurs runtime dequantization/re-quantization overhead on small activation matrices that exceeds the memory bandwidth savings of a 135M parameter model.
- **Conclusion:** Post-training dynamic INT8 is **unsuitable** for production deployment without Quantization-Aware Training (QAT).

---

## 4. Hardware Limitations & Unsupported Formats

1. **FP16 / BF16 on CPU:**
   - *Technical Limitation:* Host PyTorch CPU build does not support native hardware accelerated BF16 matrix multiplication (`torch.cpu.is_bf16_supported() == False`). Running non-native FP16 emulation causes runtime conversion penalties without speedup.
2. **INT4 (BitsAndBytes NF4/FP4):**
   - *Technical Limitation:* `bitsandbytes` 4-bit CUDA kernels require active NVIDIA GPU compute with compatible CUDA drivers. Under the PyTorch CPU runtime, INT4 execution is unsupported.

---

## 5. Artifacts
- Benchmark JSON: `experiments/artifacts/quantization_benchmark.json`
- Visualization: `experiments/artifacts/quantization_comparison.png`
