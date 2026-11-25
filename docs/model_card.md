# NEXUS-SLM Model Card: Qwen2.5-0.5B LoRA Clean V2

## 1. Model Details

- **Model Identifier**: `qwen25_05b_clean_v2_lora`
- **Model Version**: Clean V2 Candidate (Run ID: `20261030_190017`)
- **Lifecycle Stage**: **Experimental / Candidate on Hold** (Benchmarked against locked test set; see [RESULTS.md](RESULTS.md))
- **Base Architecture**: `Qwen/Qwen2.5-0.5B-Instruct` (Qwen2ForCausalLM, 24 layers, 14 attention heads, hidden dim 896)
- **Base Parameters**: ~494 Million
- **Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) via LoRA ($r=16, \alpha=32, \text{dropout}=0.05$) targeting attention projections (`q_proj`, `v_proj`)
- **Adapter Checkpoint Path**: `experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter`
- **Adapter Weights File**: `adapter_model.safetensors` (35,237,104 bytes)
  - **Verified SHA-256**: `9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2`
- **Adapter Config File**: `adapter_config.json` (1,159 bytes)
  - **Verified SHA-256**: `7A236E4CF7521BFC6D64C4C01CB16879C7B4AF272434C7D64B7CAE017247A17C`
- **Tokenizer**: Qwen2.5 BPE Tokenizer (Vocab size: 151,936, ChatML formatting)
- **Runtime Environment**: PyTorch 2.6.0 on CPU / CUDA

---

## 2. Intended Use & Boundaries

- **Primary Application**: Automated triage, classification, priority evaluation, and routing of multilingual customer support tickets into structured JSON.
- **Target Environments**: Enterprise private clouds, on-premise Kubernetes clusters, or edge inference environments handling Indian English, Native Hindi (Devanagari), and Hinglish (Roman Hindi).
- **Out-of-Scope & Prohibited Applications**:
  - Autonomous financial disbursements, chargebacks, or refunds without human adjudication.
  - Autonomous account deletion without compliance verification.
  - Unmonitored classification of safety-critical fraud reports without deterministic policy oversight.

---

## 3. Supported Taxonomy & Schema

- **Categories (14)**: `account_access`, `account_closure`, `billing`, `delivery_issue`, `fraud_report`, `general_query`, `order_cancel`, `order_tracking`, `password_reset`, `payment_failure`, `product_complaint`, `refund_request`, `subscription`, `technical_issue`.
- **Priorities (4)**: `low`, `medium`, `high`, `critical`.
- **Languages (3)**: `english`, `hindi`, `hinglish`.
- **Output Schema**:
  ```json
  {
    "category": "payment_failure",
    "priority": "high",
    "language": "hinglish",
    "requires_human": true
  }
  ```

---

## 4. Training & Validation Provenance

- **Dataset Root**: `data/v2_clean_template_grouped/`
- **Training Set**: `train_clean.jsonl` ($N=4,797$ samples, 947 masked templates)
- **Validation Set**: `val_clean.jsonl` ($N=449$ samples, 82 masked templates)
- **Locked Test Set**: `test_clean.jsonl` ($N=746$ samples, 131 masked templates, SHA-256 `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b`)
- **Training Strategy**: Cloud GPU fine-tuning under strict template-grouped isolation. Zero test set leakage during training.

---

## 5. Measured Independent Evaluation Benchmark (Locked Test Set, $N=746$)

Evaluated on the locked 746-sample clean test benchmark (`data/v2_clean_template_grouped/test_clean.jsonl`, EN 246 / HI 178 / Hinglish 322):

| Metric | Qwen2.5-0.5B True HF Base + Clean LoRA | Stacked Historical Run (`scratch_merged_qwen`, Contaminated) | Production Heuristic V2 (Refined Policy V2) |
|---|---:|---:|---:|
| **Category Accuracy** | **53.89%** (402/746) | 69.17% (516/746) | **57.51%** (429/746) |
| **Category Macro-F1** | **0.5375** | 0.6044 | **0.6516** |
| **Priority Accuracy** | **44.24%** (330/746) | 47.86% (357/746) | 36.46% (272/746) |
| **Priority Macro-F1** | **0.2491** | 0.2842 | **0.3827** |
| **Language Accuracy** | **84.58%** | 96.78% (722/746) | 81.23% (606/746) |
| - English ($N=246$) | 93.90% cat | 100.00% lang | 100.00% lang |
| - Hindi Devanagari ($N=178$) | 20.79% cat | 100.00% lang | 100.00% lang |
| - Hinglish Roman ($N=322$) | 41.61% cat | 92.55% lang | 56.52% lang |
| **Structured JSON Validity** | **100.00%** (746/746) | 100.00% (746/746) | 100.00% (746/746) |
| **High-Risk Escalation Recall** | **100.00% (0 FN, conf=None)** | 86.12% (180/209, 29 FN) | 84.69% (177/209, 32 FN) |
| **Critical Safety Recall** | **100.00% (0 FN)** | — | **100.00% (0 FN)** |
| **Overall Escalation Rate** | **100.00% (Rule 12 conservative)** | 72.25% (539/746) | **70.64%** (527/746) |
| **Latency P50 / P95 (CPU)** | CPU batch=16 (~1.1s/item) | 1,508.55 ms / 2,212.92 ms | **0.04 ms / 0.18 ms** (direct) |

> *Audit Correction:* Earlier drafts reported `69.17%` without flagging `scratch_merged_qwen` V1 `seed42` adapter leakage (+15.28 pp inflation over uncontaminated `Qwen/Qwen2.5-0.5B-Instruct` `53.89%`).

---

## 6. Known Failure Modes & Diagnostic Findings

1. **`product_complaint` vs `refund_request` Confusion**: The model achieved 0% recall on `product_complaint` ($N=44$), erroneously mapping damaged/defective product reports to `refund_request` ($29/44$) and `fraud_report` ($15/44$).
2. **`general_query` Dispersion**: 0% recall on `general_query` ($N=42$), with 28 samples misclassified as `fraud_report` due to inquiry sensitivity.
3. **Sparse Category Population**: The `subscription` category has only 2 samples in the locked test set. While recall is 100% on paper, precision is 12.5% and the 95% Wilson interval is $[0.3424, 1.0000]$.
4. **Standalone Safety Gap**: In standalone neural mode without policy enforcement, the model generated 29 escalation false negatives on high-risk tickets. Consequently, **deterministic wrapping by `HumanEscalationPolicy` is strictly mandatory**.

---

## 7. Production Gate Status

- **Status**: **HOLD / EXPERIMENTAL ONLY**
- **Reason**: Does not meet the $\ge 80.0\%$ category accuracy, $\ge 0.75$ Macro-F1, or standalone safety recall thresholds.
- **Deployment Topology**: Operates as a shadow or secondary candidate behind the **Heuristic Fallback v2** engine.
