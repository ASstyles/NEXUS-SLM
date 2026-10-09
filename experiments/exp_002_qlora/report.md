# NEXUS-SLM Experiment Report: exp_002_qlora

## 1. Experiment Overview
- **Experiment ID**: `exp_002_qlora`
- **Model Architecture**: `HuggingFaceTB/SmolLM-135M-Instruct`
- **Method**: Parameter-Efficient Fine-Tuning (PEFT LoRA)
- **Dataset**: `dataset_v1` (Locked, SHA256 verified)
- **Target Hardware**: Intel Core i5 / x86_64 CPU (PyTorch 2.14.1+cpu)
- **Date / Timestamp**: 2026-10-06T12:19:18Z

---

## 2. Parameter & LoRA Configuration
- **Total Parameters**: 136,358,208
- **Frozen Parameters**: 134,515,008 (98.65%)
- **Trainable Parameters**: 1,843,200 (1.352%)
- **LoRA Rank ($r$)**: 16
- **LoRA Alpha ($\alpha$)**: 32
- **LoRA Dropout**: 0.05
- **Target Attention Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`

---

## 3. Training Dynamics
- **Epochs**: 1
- **Batch Size**: 4 (Gradient Accumulation Steps: 2, Effective Batch Size: 8)
- **Total Optimization Steps**: 40
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=1e-8$)
- **Learning Rate**: 5e-4 with Cosine Annealing
- **Initial Training Loss**: 4.1982
- **Final Training Loss**: 1.8142 (Average Epoch Loss: 2.8518)
- **Validation Loss**: 1.7351
- **Training Wallclock Duration**: 177.08 seconds (2.95 minutes)

---

## 4. Evaluation on Locked Test Benchmark ($N=42$)
Evaluated on the exact immutable test set (`data/metadata/dataset_v1_lock.json`):

| Metric | Measured Value |
|---|---:|
| Classification Accuracy | 0.00% |
| Macro-F1 | 0.0000 |
| Weighted-F1 | 0.0000 |
| JSON Validity Rate | 0.00% |
| Category Accuracy | 0.00% |
| Priority Accuracy | 0.00% |
| Language Accuracy | 0.00% |
| Human Escalation Accuracy | 59.52% |
| P50 Latency | 4,237.41 ms |
| P95 Latency | 4,675.85 ms |
| P99 Latency | 4,702.65 ms |
| Model Size (FP32) | 520.17 MB |
| Trainable Adapter Size | 7.37 MB |
| Peak System RAM | 11,899.40 MB |
| Peak VRAM | 0.00 MB (CPU execution) |

---

## 5. Empirical Diagnosis
1. **Convergence vs Format Synthesis**:
   The cross-entropy training loss decreased significantly from 4.198 to 1.814 over 40 steps, confirming that gradient updates successfully reached the LoRA matrices. However, 1 epoch of instruction tuning on a 135M parameter SLM was insufficient for the model to reliably emit end-to-end well-formed JSON syntax while closing curly brackets before the token budget cutoff.
2. **Failure Modes**:
   Output generations frequently repeated parts of the input prompt or omitted closing braces (`}`), causing 100% of responses to fail JSON validation and fallback to default unparsed state.
3. **Hardware Constraints**:
   PyTorch was installed without CUDA acceleration (`2.14.1+cpu`), forcing CPU-only computation with low throughput (0.23 req/sec).
