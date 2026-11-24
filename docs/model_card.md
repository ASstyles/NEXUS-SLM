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

Evaluated on the locked 746-sample clean test benchmark with zero mock dependencies:

| Metric | Measured Clean-Test Value ($N=746$) | Baseline Fallback Comparison |
|---|---:|---:|
| **Category Accuracy** | **69.17%** (516/746) | 56.03% (418/746) |
| **Category Macro Precision** | **65.34%** | 87.59% |
| **Category Macro Recall** | **69.17%** | 58.32% |
| **Category Macro-F1** | **0.6044** | 0.6132 |
| **Category Weighted-F1** | **0.6630** | 0.6175 |
| **Priority Accuracy** | **47.86%** (357/746) | 47.18% (352/746) |
| **Priority Macro-F1** | **0.2842** | 0.3980 |
| **Priority Weighted-F1** | **0.3864** | 0.4497 |
| **Overall Language Accuracy** | **96.78%** (722/746) | 81.23% (606/746) |
| - English Language Accuracy | **100.00%** (231/231) | 100.00% (231/231) |
| - Hindi Language Accuracy | **100.00%** (232/232) | 100.00% (232/232) |
| - Hinglish Language Accuracy | **92.55%** (262/283) | 49.82% (141/283) |
| **Structured JSON Validity** | **100.00%** (746/746) | 100.00% (746/746) |
| **Exact 4-Field Match Rate** | **35.52%** (265/746) | 22.39% (167/746) |
| **Escalation Recall** | **86.12%** (180/209) | 100.00% (209/209) |
| **Escalation Precision** | **33.40%** (180/539) | 28.02% (209/746) |
| **Overall Escalation Rate** | **72.25%** (539/746) | 100.00% (746/746) |
| **Latency P50 (CPU, Batch 16)** | **1,508.55 ms** | 0.05 ms |
| **Latency P95 (CPU, Batch 16)** | **2,212.92 ms** | 0.25 ms |
| **Throughput (CPU)** | **0.62 req/s** | 9,833.58 req/s |
| **RAM Footprint (LoRA Delta)** | **78.66 MB** | ~5 MB |

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
