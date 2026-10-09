# NEXUS-SLM Model Card

## 1. Model Details

- **Model Name**: NEXUS-SLM-135M-LoRA-v1
- **Model Version**: `nexus-qlora-v1`
- **Candidate Status**: Locked Production Candidate (Manifest: `experiments/final_candidate.json`)
- **Base Architecture**: `HuggingFaceTB/SmolLM-135M-Instruct` (LlamaForCausalLM, 30 layers, 9 attention heads, hidden dim 576)
- **Base Parameters**: 135,436,608
- **Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) via LoRA ($r=16, \alpha=32, \text{dropout}=0.05$) targeting `q_proj` and `v_proj`
- **Trainable Parameters**: 921,600 (0.68% of base model)
- **Precision**: FP32 on CPU runtime
- **Tokenizer**: SmolLM-135M Byte-level BPE (Vocab size: 49,152, ChatML format)
- **Cryptographic Model Hash**: `sha256_643971155ba3fe3a91379005fbe85d5ca95e73af90ec7cd65f871379a838983f`

---

## 2. Intended Use

- **Primary Application**: Automated triage, classification, urgency prioritization, and routing of enterprise customer support tickets.
- **Target Environments**: Offline, air-gapped, on-premise, or private cloud CPU clusters where privacy regulations (GDPR, HIPAA) prohibit sending raw customer text to commercial cloud LLMs.
- **Out-of-Scope Applications**: Autonomous resolution of tickets without human oversight; financial transaction approval; general-purpose conversational chat.

---

## 3. Supported Inputs & Outputs

### 3.1 Input Modalities
- **Languages**: English, Hinglish (Hindi-English code-switched), Roman Hindi, and Native Hindi (Devanagari script).
- **Format**: Plain text customer support tickets (1 to 4,000 characters).

### 3.2 Output Schema
- **Category**: One of 14 support categories (`payment_failure`, `fraud_report`, `refund_request`, etc.).
- **Priority**: Urgency level (`low`, `medium`, `high`, `critical`).
- **Language**: Detected modality (`english`, `hindi`, `hinglish`).
- **Requires Human**: Boolean escalation flag.
- **Confidence**: Calibrated confidence score in $[0.0, 1.0]$.

---

## 4. Training Methodology

- **Dataset**: `dataset_v1` (316 training samples, 44 validation samples, 42 locked test samples).
- **Dataset Hash**: `sha256_11df262072d7243f623bfd829b407a8dbd2c5a3087ed697a836a7152c7688967`.
- **Training Hyperparameters**:
  - Epochs: 3
  - Batch Size: 4 (Effective batch size: 8 with gradient accumulation)
  - Learning Rate: $5 \times 10^{-4}$ with linear warmup (ratio 0.05) and weight decay 0.01
  - Max Sequence Length: 280 tokens
  - Loss Trajectory: Train Loss decreased from 0.98 to 0.2077; Eval Loss reached 0.1439.

---

## 5. Measured Evaluation Metrics (Locked Test Benchmark)

Evaluated on the locked 42-sample test benchmark (`sha256_edf3fa5cd949a149ad5d759d81553d9704467491877d0f3c8b7cc5a887a9eda4`):

| Evaluation Metric | Baseline Base Model | NEXUS-SLM v1 (Candidate) | NEXUS-SLM v2 |
|---|---|---|---|
| **Macro F1** | 0.0000 | **0.4089** | 0.2224 |
| **Category Accuracy** | 0.0000 | **0.4286** | 0.2381 |
| **Priority F1** | 0.0000 | **0.2739** | 0.1421 |
| **Language Accuracy** | 0.0000 | **0.9524** | 0.6190 |
| **JSON Validity Rate** | 0.0000 | **0.9762** | 0.8333 |
| **P50 Latency (ms)** | 4888.22 | **1204.46** | 2326.77 |
| **P95 Latency (ms)** | 5396.91 | **1306.49** | 2551.76 |
| **RAM Footprint (MB)** | 13196.75 | **982.24** | 985.00 |

*Empirical Selection Rationale*: V1 significantly outperforms V2 on Macro-F1 (+45.6% relative advantage), Category Accuracy (+44.5%), and JSON Validity (+14.6%) while running nearly 2x faster.

---

## 6. Confidence Calibration Profile

- **Expected Calibration Error (ECE)**: $0.5879$ (58.8%).
- **Mean Model Confidence**: $0.9689$ vs **Mean Accuracy**: $0.3810$.
- **Finding**: Raw token probabilities in small causal language models exhibit severe overconfidence.
- **Production Policy**: Raw confidence is never exposed as an uncalibrated posterior. Calibrated scaling and deterministic confidence thresholds ($< 0.65 \implies \text{escalate}$) are applied.

---

## 7. Known Limitations & Failure Modes

1. **Native Devanagari Hindi Weakness**: SmolLM-135M vocabulary lacks dedicated subword tokens for Devanagari script, fragmenting Hindi words into byte fallback tokens and resulting in low classification accuracy (14.3%). Roman Hindi and Hinglish perform substantially better (57.1%).
2. **Short Ticket Degradation**: Tickets with $\le 3$ words (e.g., "Payment failed") lack syntactic context, degrading classification recall. These tickets are automatically routed to human agents.
3. **Rare Class Boundary Confusion**: Subtle distinctions between semantically adjacent categories (e.g. `account_closure` vs `account_access`) require deterministic safety gating.
4. **Parameter Capacity Ceiling**: At 135M parameters, the model cannot perform complex multi-intent reasoning.
5. **CPU Latency Under High Concurrency**: CPU thread contention increases queue latency at concurrency $> 2$. Concurrency should be capped at 1-2 workers per CPU instance.

---

## 8. Safety & Ethical Considerations

- The model must **never** be operated as an unmonitored autonomous system.
- Deterministic safeguards guarantee that fraud reports, account deletions, and legal complaints are always routed to human staff.
