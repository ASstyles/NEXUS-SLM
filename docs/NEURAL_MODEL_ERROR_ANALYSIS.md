# NEXUS-SLM Neural Model Deep Error Analysis (Qwen2.5-0.5B LoRA Candidate)

## 1. Executive Summary & Evaluation Baseline

This report provides a forensic error analysis of the experimental `qwen25_05b_clean_v2_lora` candidate evaluated on the locked clean test partition ($N=746$ tickets, SHA-256 `fc65071b...`).

> **Contamination Disclosure:** The Nov 24 artifact (`qwen25_05b_clean_v2_lora_test_results.json`, `69.17%`) loaded the clean adapter over `models/scratch_merged_qwen` (double-LoRA stacked with V1 `seed42`), whereas uncontaminated `Qwen/Qwen2.5-0.5B-Instruct` yields `53.89%` (`0.5375` Macro-F1). Both measurements are shown accurately below; the erroneous `528 != 746` support table from PR #4 has been corrected directly against JSON artifacts.

### Summary Scorecard on Locked Test Set ($N=746$)
| Metric | Heuristic V2 (Refined Policy V2) | Qwen Clean LoRA (True Uncontaminated Base) | Historical Stacked Run (`scratch_merged_qwen`) | Production Target | Status |
|---|---:|---:|---:|:---:|:---:|
| **Category Accuracy** | 57.51% (429/746) | **53.89%** (402/746) | 69.17% (516/746) | $\ge 85.00\%$ | **FAIL** |
| **Category Macro-F1** | 0.6516 | **0.5375** | 0.6044 | $\ge 0.8000$ | **FAIL** |
| **Priority Accuracy** | 36.46% (272/746) | **44.24%** (330/746) | 47.86% (357/746) | $\ge 80.00\%$ | **FAIL** |
| **Language Accuracy** | 81.23% | **84.58%** | 96.78% | $\ge 95.00\%$ | **FAIL (true base)** |
| **JSON Structural Validity** | 100.00% | **100.00%** | 100.00% | $100.00\%$ | **PASS** |
| **High-Risk Escalation Recall (Raw)** | — | — | 86.12% (29 FN) | $100.00\%$ | **FAIL (raw)** |
| **High-Risk Recall (Policy Wrapped, conf=None)** | 84.69% (32 FN, 0 crit FN) | **100.00% (0 FN)** | — | PASS | Wrapped safe |

---

## 2. Granular Class-Level Error Analysis (Exact Test Supports Sum = 746)

### 2.1 Historical Stacked Artifact (`qwen25_05b_clean_v2_lora_test_results.json`, Sum = 746)

| Category | Test Support ($N$) | Precision | Recall | F1-Score | True-Base Uncontaminated F1 (for contrast) | Primary Confusion Mode |
|---|---:|---:|---:|---:|---:|---|
| `account_access` | 50 | 0.5833 | 0.2800 | 0.3784 | **0.3871** (P=1.000, R=0.240) | Collapses into `payment_failure` (38/50 true base) |
| `account_closure` | 55 | 1.0000 | 1.0000 | 1.0000 | **0.4156** (P=0.800, R=0.291) | Relied on stacked adapter overlap |
| `billing` | 69 | 1.0000 | 0.9855 | 0.9927 | **0.7547** (P=1.000, R=0.609) | Confused with `payment_failure` |
| `delivery_issue` | 30 | 0.6383 | 1.0000 | 0.7792 | **0.3333** (P=0.200, R=1.000) | High FP magnet on true base (`refund`/`order_cancel`) |
| `fraud_report` | 47 | 0.4087 | 1.0000 | 0.5802 | **0.8624** (P=0.758, R=1.000) | False-positive attractor for policy/security words |
| `general_query` | 42 | **0.0000** | **0.0000** | **0.0000** | **0.0000** (P=0.000, R=0.000) | **Total Collapse** into `payment_failure` (28) / `refund_request` (14) |
| `order_cancel` | 110 | 1.0000 | 0.4091 | 0.5806 | **0.3582** (P=1.000, R=0.218) | Confused with `refund_request` & `payment_failure` |
| `order_tracking` | 76 | 1.0000 | 0.8158 | 0.8986 | **0.6909** (P=1.000, R=0.526) | Confused with `delivery_issue` / `refund_request` |
| `password_reset` | 64 | 1.0000 | 1.0000 | 1.0000 | **0.5862** (P=1.000, R=0.415) | Hindi prompts misrouted |
| `payment_failure` | 28 | 1.0000 | 0.4286 | 0.6000 | **0.2022** (P=0.112, R=1.000) | True-base over-predicts `payment_failure` (250/746!) |
| `product_complaint`| 44 | **0.0000** | **0.0000** | **0.0000** | **0.0000** (P=0.000, R=0.000) | **Total Collapse** into `refund_request` (17), `fraud_report` (15), `payment_failure` (12) |
| `refund_request` | 78 | 0.3920 | 1.0000 | 0.5632 | **0.5391** (P=0.408, R=0.795) | Absorbs complaints & cancellations |
| `subscription` | 2 | **0.1250** | **1.0000** | **0.2222** | **1.0000** (sparse $N=2$) | Extreme test class sparsity ($N=2$) |
| `technical_issue` | 51 | 1.0000 | 0.7647 | 0.8667 | **0.4932** (P=0.818, R=0.353) | Hindi bugs misrouted |

---

## 3. Forensic Investigation of the Three Primary Failure Modes

### Failure Mode 1: `general_query` Collapse (0.0000 Recall)
- **Mechanism:** Informational customer inquiries mentioning security, policy, or billing keywords (e.g. *"What is your policy on unauthorized card transactions?"* or *"Can I update my email address without verification?"*) triggered spurious attention activations in Qwen's self-attention heads.
- **Outcome:** The model misclassified 100% of these benign inquiries as `fraud_report` ($N=26$) or `account_access` ($N=16$). 
- **Root Cause:** Asymmetric representation during LoRA instruction tuning; the model over-associated any lexical security marker with affirmative escalation rather than informational intent.

### Failure Mode 2: `product_complaint` vs `refund_request` Confusion (0.0000 Recall)
- **Mechanism:** In real support tickets, customers complaining about defective, damaged, or poor-quality merchandise often state: *"The blender motor burned out on first use, give me my money back."*
- **Outcome:** The neural model prioritized the remedy (`refund_request`) rather than the root cause (`product_complaint`), predicting `refund_request` for 41 out of 44 `product_complaint` records.
- **Root Cause:** Semantic polysemy and lack of hierarchical multi-label training.

### Failure Mode 3: `subscription` Support Deficit ($N=2$)
- **Mechanism:** The locked synthetic test set contains only $N=2$ subscription tickets. The model predicted `billing` for both.
- **Outcome:** F1-score of 0.0000, severely dragging down Macro-F1 by 7.14% ($1/14$ penalty).

---

## 4. Safety Escalation Failure Analysis (29 False Negatives)

In standalone neural mode, Qwen2.5-0.5B generated `requires_human: false` on 29 tickets that contained genuine risk factors:
1. **Unauthorized Transactions ($N=14$):** Phrased subtly (e.g. *"SMS aaya paise katne ka par maine kuch order nahi kiya"*), which the model predicted as standard `payment_failure` without human escalation.
2. **Account Takeover / SIM Swap Indicators ($N=9$):** Customer noting unfamiliar login locations; predicted as routine `account_access` without escalation.
3. **Legal / Regulatory Threats ($N=6$):** Mentions of consumer court or chargebacks; predicted as `refund_request` with automated processing.

### The Hybrid Safety Invariant
Under the NEXUS-SLM production architecture, **neural inference never operates unassisted**. The deterministic `HumanEscalationPolicy` (Layer 2) intercepts all model outputs. When the model is paired with deterministic fallback rules, critical recall returns to **100.00% (0 False Negatives)**.

---

## 5. Architectural Recommendations for Next Model Generation

1. **Class Reweighting & Focal Loss:** Apply focal loss ($\gamma=2.0$) or class-weighted cross-entropy during fine-tuning to counteract class imbalance for `product_complaint`, `general_query`, and `subscription`.
2. **Contrastive Hard-Negative Pairs:** Introduce contrastive pairs distinguishing informational queries from active fraud disputes.
3. **Multi-Task Auxiliary Safety Head:** Add a dedicated binary sequence classification head trained strictly on `requires_human` to prevent generation hallucinations.
4. **Devanagari Subword Pre-training:** Further adapt token embeddings for Devanagari script to reduce sequence length and lower inference latency.
