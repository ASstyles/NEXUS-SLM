# NEXUS-SLM: Qwen2.5-0.5B Quantization, Optimization & Evaluation Evidence Report

**Document ID:** `NEXUS-DOC-OPT-2026-QWEN-001`  
**Evaluation Date:** October 12, 2026  
**Environment:** Local CPU Infrastructure (Windows 11, Intel 8-Core / 12 Threads, 16 GB RAM)  
**Evaluator:** Antigravity AI Autonomous Engineering Agent  
**Active Production Model:** `nexus-qlora-v1` (SmolLM-135M-Instruct) — **Preserved Unchanged**  
**Evaluated Candidates:**  
1. `qwen25_05b_lora_seed42` (Unquantized FP32 LoRA Baseline Candidate)  
2. `qwen25_05b_lora_seed42_int8` (ONNX Dynamic INT8 Quantized Candidate)  

---

## 1. Executive Summary

In accordance with strict evidence-driven machine learning governance, the Qwen2.5-0.5B-Instruct LoRA candidate (trained on Kaggle Tesla T4 GPU, Seed 42) was evaluated on local CPU hardware across full baseline reproduction, ONNX INT8 dynamic quantization, safety challenge stress testing, and regression suites.

### Key Governance Outcome:
- **`nexus-qlora-v1` remains the active production model.**
- **`qwen25_05b_lora_seed42_int8` is registered strictly as an EXPERIMENTAL candidate.**
- **Promotion Status:** **`EVALUATED_HELD_FOR_APPROVAL` (Rejected for production deployment).**
- **Quantization Result:** **Dynamic INT8 quantization compressed artifact size by 4.05x (488 MB) and reduced peak RAM to 1,294.7 MB (passing RAM limit), but caused catastrophic quality collapse in autoregressive generation (Macro-F1 collapsed from 0.8974 to 0.1823, JSON validity dropped to 40.91%, and Native Devanagari Hindi accuracy dropped to 0.0%). P95 CPU latency was 2,637.6 ms (exceeding the 2,500 ms SLA). Four out of six acceptance gates FAILED.**

---

## 2. Environment & Infrastructure Baseline

| Parameter | Specification | Verification Source |
|---|---|---|
| Operating System | Windows 11 Enterprise (10.0.26300) | `platform.platform()` |
| CPU Architecture | x86_64, Intel 8 Physical Cores (12 Logical Threads) | `psutil.cpu_count()` |
| Available System RAM | 15.65 GB Total (Peak RSS Budget: 2,048 MB) | `psutil.virtual_memory()` |
| Free Storage | 272.49 GB Free on Volume `C:` | `Get-PSDrive C` |
| Python Environment | Python 3.10.11 (`.venv\Scripts\python.exe`) | `sys.version` |
| PyTorch Version | 2.14.1+cpu (No CUDA acceleration) | `torch.__version__` |
| HuggingFace Transformers | 4.57.6 | `transformers.__version__` |
| HuggingFace Optimum | 2.1.0 | `optimum.__version__` |
| ONNX Runtime Engine | 1.23.2 | `onnxruntime.__version__` |
| PEFT Version | 0.21.2 | `peft.__version__` |
| Cloud / Paid Services | **None ($0.00 incurred)** | Free local CPU compute only |

---

## 3. Dataset Identifiers & Locked Checksums

All evaluations were executed against the locked, approved data splits. SHA-256 hashes were verified before execution:

| Dataset Split | File Path | Sample Count (N) | Verified SHA-256 Checksum |
|---|---|---|---|
| **Validation Split** | `data/validation/validation.jsonl` | 44 | `2e349ad3b1330aec224c266460d2e4189b94138a1d63146a10bd8f45ea65508e` |
| **Historical Test Split** | `data/test/test.jsonl` | 42 | `edf3fa5cd949a149ad5d759d81553d9704467491877d0f3c8b7cc5a887a9eda4` |
| **Safety Challenge Set** | `data/evaluation/high_risk_challenge_set.jsonl` | 70 | `70b43711fd2f515bdd27d6daa8e604698a109f6093cdaea7f85bd584339dbc5b` |

*Governance Notice: In accordance with Safety Rule 6, the locked test split (`test.jsonl`) was strictly quarantined and never used for quantization parameter selection, calibration, or model tuning.*

---

## 4. Candidate Artifact Verification & Mathematical Derivation Proof

### 4.1 Original LoRA Candidate (`models/qwen25_05b_lora_seed42`)
All 9 artifact files recorded in `nexus_qwen_metadata.json` were verified with exact SHA-256 matches:
- `adapter_model.safetensors`: `8a30ceafc32c2180d7414a7bdaa5199162921f4960ecd7eef34eed3a49f91a0a` (35,237,104 bytes)
- `adapter_config.json`: `51f5eaea09f409d7849d457fb4ac7ad09c65b6a8223a931f21ef44910c24363b`
- `tokenizer.json`: `9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa` (11,421,896 bytes)
- `vocab.json`: `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910`
- `merges.txt`: `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5`

### 4.2 Mathematical Derivation Proof (LoRA to Export Verification)
To guarantee that the exported ONNX model is derived directly from the trained LoRA candidate rather than evaluating the untouched foundation base model:
- Foundation base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Merged weights: `models/scratch_merged_qwen/model.safetensors`
- Adapter matrices: `lora_A`, `lora_B` with rank $r=16$, $\alpha=32$ ($\text{scaling} = 2.0$)
- **Measured Weight Delta:**
  $$\Delta W = W_{\text{merged}} - W_{\text{base}}$$
  $$\text{Expected } \Delta W = \frac{\alpha}{r} (B \cdot A)$$
  $$\max |\Delta W - \text{Expected } \Delta W| = 5.8797 \times 10^{-8}$$
  $$\max |W_{\text{merged}} - W_{\text{base}}| = 0.001774$$
- **Proof Outcome:** **VERIFIED.** The weights differ from the untouched base model and match the LoRA mathematical formulation with machine precision ($5.88 \times 10^{-8}$).

### 4.3 Quantized Candidate (`models/qwen25_05b_lora_seed42_int8`)
- Format: ONNX Dynamic INT8 (`avx512_vnni`, u8/s8 IntegerOps, per-channel=False)
- `model_quantized.onnx`: `d38a4d687101c2529da7ad38a00d9fdfc69d113902ec6609e92877cfef61080f` (496,023,750 bytes / 473.05 MB)
- Package Archive: `nexus_qwen25_05b_lora_seed42_int8.zip` (311.16 MB valid archive; resolved a packaging bug where recursive self-zipping had previously bloated disk usage to 10 GB).

---

## 5. Comprehensive Performance Comparison Matrix

The table below contrasts the active production baseline against the unquantized candidate and the quantized candidate under identical evaluation conditions on the local CPU environment:

| Dimension / Metric | Active Production (`nexus-qlora-v1`) | Unquantized Qwen Candidate (`qwen25_05b_lora_seed42`) | Quantized Qwen Candidate (`qwen25_05b_lora_seed42_int8`) | Target Acceptance SLA |
|---|---|---|---|---|
| **Base Model Architecture** | SmolLM-135M-Instruct | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B-Instruct | — |
| **Model Format** | PyTorch FP32 LoRA | PyTorch FP32 LoRA | ONNX Dynamic INT8 | Standalone CPU |
| **Artifact Size on Disk** | 134.5 MB | 116.2 MB (adapter) / 1,976 MB | **488.19 MB** | < 1,000 MB |
| **Model Startup / Load Time** | ~1.50 s | 2.81 s | 3.14 s | < 5.0 s |
| **Validation Examples (N)** | 44 | 44 | 44 | 44 |
| **Validation Category Accuracy** | 0.4545 (20/44) | **0.9091 (40/44)** | 0.1818 (8/44) | >= 0.5500 |
| **Validation Macro-F1** | 0.4412 | **0.8974** | 0.1823 | **>= 0.5500 (Gate 1)** |
| **Validation Weighted-F1** | N/A | **0.9021** | 0.2036 | — |
| **English Slice Accuracy** | 0.5714 (8/14) | **0.9333 (14/15)** | 0.3333 (5/15) | — |
| **Hinglish Slice Accuracy** | 0.5714 (8/14) | **0.8667 (13/15)** | 0.2000 (3/15) | — |
| **Native Hindi Accuracy** | 0.1429 (2/14) | **0.9286 (13/14)** | **0.0000 (0/14)** | **>= 0.5000 (Gate 2)** |
| **JSON Schema Validity Rate** | 0.9545 (42/44) | **1.0000 (44/44)** | **0.4091 (18/44)** | **>= 0.9500 (Gate 4)** |
| **Exact Structured Match Acc** | 0.0714 (3/42) | **0.4318 (19/44)** | 0.0682 (3/44) | — |
| **Challenge Set Recall (N=70)** | **1.0000 (0 FN)** | **1.0000 (0 FN)** | **1.0000 (0 FN)** | **1.0000 (Gate 3)** |
| **Challenge Confusion Matrix** | 60 TP / 0 FN / 10 TN | 60 TP / 0 FN / 10 TN | 60 TP / 0 FN / 10 TN | Zero False Negatives |
| **CPU Latency P50** | **1,204.5 ms** | 3,501.5 ms | 1,568.7 ms | — |
| **CPU Latency P95** | **1,306.5 ms** | 4,101.0 ms | **2,637.6 ms** | **<= 2,500.0 ms (Gate 5)** |
| **Peak Process RSS** | **516.65 MB** | 2,400.99 MB | **1,294.67 MB** | **<= 2,048.0 MB (Gate 5)** |
| **Regression Test Suite** | 104 Passed | 104 Passed | **116 Passed (100%)** | **All Pass (Gate 6)** |

---

## 6. Formal Acceptance Gate Audits

Every gate was evaluated independently against empirical measurements:

### Gate 1 — Validation Macro-F1
- **Threshold:** $\ge 0.5500$
- **Unquantized Candidate:** $0.8974$ (**PASS**)
- **Quantized Candidate:** $0.1823$ (**FAIL**)
- **Finding:** Dynamic INT8 caused a 79.7% drop in Macro-F1.

### Gate 2 — Native Devanagari Hindi Category Accuracy
- **Threshold:** $\ge 0.5000$
- **Unquantized Candidate:** $0.9286$ (13/14 correct) (**PASS**)
- **Quantized Candidate:** $0.0000$ (0/14 correct) (**FAIL**)
- **Finding:** Quantization noise completely destroyed subword representations for Devanagari Hindi tickets, resulting in 100% misclassification on Hindi inputs.

### Gate 3 — High-Risk Safety Challenge Recall
- **Threshold:** Recall $= 1.0000$ with **0 False Negatives** across $N=70$ challenge tickets
- **Unquantized Candidate:** Recall $= 1.0000$ ($60\text{ TP}, 0\text{ FN}, 0\text{ FP}, 10\text{ TN}$) (**PASS**)
- **Quantized Candidate:** Recall $= 1.0000$ ($60\text{ TP}, 0\text{ FN}, 0\text{ FP}, 10\text{ TN}$) (**PASS**)
- **Finding:** The deterministic `HumanEscalationPolicy` successfully intercepted 100% of high-risk fraud, account takeover, and regulatory vectors without exception. *Note: Zero false negatives on 70 challenge cases is evidence for this benchmark only, not proof of universal safety.*

### Gate 4 — JSON Schema Validity Rate
- **Threshold:** $\ge 95.0\%$ ($0.9500$)
- **Unquantized Candidate:** $1.0000$ (44/44 valid) (**PASS**)
- **Quantized Candidate:** $0.4091$ (18/44 valid, 26 parse failures) (**FAIL**)
- **Finding:** The model frequently hallucinated invalid priority keys (e.g. `"refund_required"`, `"account_access"`), repeated system prompt instructions, or truncated JSON objects mid-generation.

### Gate 5 — CPU System Resource Requirements
- **Thresholds:** CPU Inference P95 Latency $\le 2,500.0\text{ ms}$ **AND** Peak Process RSS $\le 2,048.0\text{ MB}$ (Both conditions mandatory)
- **Unquantized Candidate:** P95 $= 4,101.0\text{ ms}$ (> 2,500 ms), RSS $= 2,400.99\text{ MB}$ (> 2,048 MB) (**FAIL**)
- **Quantized Candidate:** P95 $= 2,637.6\text{ ms}$ (> 2,500 ms), RSS $= 1,294.67\text{ MB}$ ($\le 2,048\text{ MB}$) (**FAIL**)
- **Finding:** Quantization successfully reduced RAM footprint from 2,401 MB to 1,295 MB (passing the RAM SLA), but P95 latency (2,637.6 ms) narrowly missed the 2,500 ms SLA by 137.6 ms. Because both conditions are mandatory, Gate 5 fails.

### Gate 6 — Service Integration & Regression Suite
- **Threshold:** 100% pass rate across existing and newly added regression tests
- **Measured Result:** **116 passed in 24.36s (100% pass rate, 0 failures, 0 skips)** (**PASS**)
- **Finding:** Full compatibility verified with FastAPI `PredictResponse` schemas, `OutputParser`, `InferenceEngine`, `ModelLoader`, and `HumanEscalationPolicy`.

---

## 7. Deep Engineering Diagnosis: Root Cause of INT8 Degradation

Why did ONNX Dynamic INT8 degrade generation quality while FP32 achieved 89.7% Macro-F1?

1. **Activation Outlier Sensitivity:**
   Qwen2.5 utilizes SwiGLU activation functions and RMSNorm. In autoregressive transformer decoders, activations develop extreme outlier channels (features whose magnitudes are 20x to 100x larger than typical dimensions).
2. **Per-Tensor Dynamic Quantization Scaling:**
   The `AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)` configuration applies per-tensor dynamic quantization. The presence of outlier channels forces the dynamic scale factor to widen dramatically, quantizing all normal-magnitude weights and activations into 0 or $\pm 1$.
3. **Logit Scrambling at `lm_head`:**
   Because dynamic quantization was applied across all 195 MatMul nodes (including the language model head projecting to 151,936 vocabulary tokens), quantization noise severely shifted the top-1 argmax token selection, leading to catastrophic token repetition and invalid field values.
4. **Historical Confirmation:**
   This matches historical findings recorded in `models/registry.json` for SmolLM (`nexus-slm-int8-dynamic`), which similarly degraded Macro-F1 to 9.4% and JSON validity to 40.5% under dynamic linear quantization.

---

## 8. Test Suite & Verification Results

Command executed:
```bash
.\.venv\Scripts\python.exe -m pytest
```
Outcome:
- **Test Count:** 116 collected, 116 passed (0 failed, 0 errors, 0 skipped)
- **Execution Duration:** 24.36 seconds
- **Added Targeted Tests:** 12 tests in `tests/test_qwen_quantized_candidate.py`:
  - `test_quantized_candidate_artifacts_exist`: PASS
  - `test_quantized_candidate_metadata_hashes`: PASS
  - `test_quantized_candidate_loading_ort`: PASS
  - `test_quantized_candidate_loading_via_model_loader`: PASS
  - `test_quantized_model_identity_derived_from_qwen`: PASS
  - `test_inference_engine_with_ort_model`: PASS
  - `test_multilingual_inputs_english_hinglish_hindi`: PASS
  - `test_malformed_model_output_handling`: PASS
  - `test_deterministic_human_escalation_fraud_and_high_risk`: PASS
  - `test_low_confidence_fallback_behavior`: PASS
  - `test_fastapi_response_contract_compatibility`: PASS
  - `test_missing_model_artifacts_error_handling`: PASS

---

## 9. Governance Decision & Actionable Recommendation

### Governance Decision:
- **`nexus-qlora-v1` remains preserved as the active production model.**
- **`qwen25_05b_lora_seed42_int8` is held in the registry as an experimental artifact.**
- No production traffic, canaries, or automatic promotions are enabled.

### Evidence-Based Next Action:
**Do not use dynamic INT8 activation quantization for Qwen2.5 on CPU.**  
Instead, pursue **Weight-Only Quantization (INT4 / INT8)**:
1. Export the merged weights using **GGUF (Q4_K_M or Q8_0)** for execution via `llama-cpp-python` with AVX2 CPU vector extensions. Weight-only quantization leaves activation dynamic ranges untouched, preventing outlier feature distortion while reducing model memory to ~350 MB and achieving 25-35 tokens/sec CPU throughput (< 1,200 ms latency).
2. Alternatively, evaluate **AWQ (Activation-Aware Weight Quantization)** with INT4 weights, which selectively protects salient outlier channels from quantization.
