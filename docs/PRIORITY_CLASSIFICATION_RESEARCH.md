# NEXUS-SLM Priority Classification Research & Audit

**Document ID:** RESEARCH-PRIORITY-ML-V1  
**Author:** Principal ML Engineer, NLP Research Engineer & QA Auditor  
**Date:** 2026-11-24  
**Status:** Completed Audit | Architecture Evaluated | Evidence Published  

---

## 1. Executive Summary: The ~47% Accuracy Bottleneck

Across previous evaluation runs on the locked 746-ticket benchmark, priority classification exhibited an apparent performance ceiling:
- **Production Heuristic Fallback v2:** 47.18% Accuracy (Macro-F1 0.4410)
- **Experimental Qwen2.5-0.5B LoRA:** 47.86% Accuracy (Macro-F1 0.4286)

An in-depth independent audit of `data/v2_clean_template_grouped/` revealed that this 47% plateau is **not primarily a model capacity limitation**. Instead, it is driven by **severe ground-truth label noise, subjective assignment, and extreme central-tendency bias** in the synthetic dataset.

When evaluated against the newly constructed, dual-adjudicated human evaluation benchmark ($N=154$), an objective operational impact policy achieves **59.09% accuracy and 0.6135 Macro-F1**, demonstrating that clear operational boundaries significantly outperform synthetic label memorization.

---

## 2. Objective 4-Tier Operational Priority Rubric

To eliminate subjective sentiment bias (e.g., equating customer anger or punctuation with high priority), the project defines a formal 4-tier operational rubric based on **business impact, operational urgency, and safety risk**:

```
                       OPERATIONAL URGENCY & IMPACT MATRIX
┌──────────┬─────────────────────────────┬────────────────────────────────────┐
│ Priority │ Operational Criteria        │ Standard Category Realizations     │
├──────────┼─────────────────────────────┼────────────────────────────────────┤
│ CRITICAL │ Active financial/identity   │ • fraud_report (account takeover)  │
│          │ compromise; immediate severe│ • Unauthorized debit / card freeze │
│          │ physical or safety hazards; │ • Physical hazard (smoking device) │
│          │ legal notices / subpoenas.  │ • Consumer court / formal lawsuit  │
├──────────┼─────────────────────────────┼────────────────────────────────────┤
│ HIGH     │ Significant service/payment │ • payment_failure (gateway drop    │
│          │ disruption; account lockout;│   with bank debit)                 │
│          │ severe delivery failure;    │ • account_access (2FA lock/locked) │
│          │ immediate cancellation.     │ • delivery_issue (tampered seal)   │
├──────────┼─────────────────────────────┼────────────────────────────────────┤
│ MEDIUM   │ Standard service workflows; │ • order_tracking (status inquiry)  │
│          │ routine account updates;    │ • password_reset (routine link)    │
│          │ ordinary product returns;   │ • refund_request (standard return) │
│          │ subscription plan changes.  │ • subscription (plan upgrade/down) │
├──────────┼─────────────────────────────┼────────────────────────────────────┤
│ LOW      │ Routine informational       │ • general_query (store hours, FAQs)│
│          │ assistance; non-disputed tax│ • billing (annual tax invoice /    │
│          │ documentation; feedback.    │   GST receipt request)             │
└──────────┴─────────────────────────────┴────────────────────────────────────┘
```

### Critical Distinction: Sentiment vs. Severity
- *"I am very frustrated, please answer my question about store hours"* $\to$ **LOW** (No operational disruption).
- *"Someone accessed my account without permission"* $\to$ **CRITICAL** (Severe security breach, regardless of calm phrasing).

---

## 3. Ground-Truth Label Discrepancy Audit

An exhaustive audit of the 5,992 records in `data/v2_clean_template_grouped/` was executed across training ($N=4797$), validation ($N=449$), and testing ($N=746$).

### 3.1 Distribution & Central-Tendency Bias
Across all three splits, priority assignments exhibit extreme compression into `medium`:
- `train_clean.jsonl`: `medium` = 50.6% ($2,428$), `low` = 23.0% ($1,105$), `high` = 20.7% ($992$), `critical` = 5.7% ($272$)
- `test_clean.jsonl`: `medium` = 50.5% ($377$), `high` = 24.1% ($180$), `low` = 21.4% ($160$), `critical` = 3.9% ($29$)

### 3.2 Material Contradictions in Ground Truth
The audit uncovered severe label inconsistencies within the synthetic training set:

1. **Dangerous Down-labeling of Fraud Incidents:**
   - Ticket: *"I received an SMS that order ORD-597116 was placed from my account using my saved debit card. I never placed this order... Please freeze my account immediately."*
   - Synthetic Ground Truth: `priority: medium`, `requires_human: False` *(Dangerous error: account takeover ignored)*.
   - Frequency: 50 records in `train_clean.jsonl` categorized under `fraud_report` were assigned `medium` priority with `requires_human: False`.

2. **Inverted Priority on Benign Inquiries:**
   - Ticket: *"Kripya batayein ki aapki customer care team se baat karne ke liye helpline kab uplabdh rehti hai."* (Hindi helpline hours inquiry)
   - Synthetic Ground Truth: `priority: high`, `requires_human: True` *(False escalation: routine FAQ escalated)*.
   - Frequency: 34 records in `general_query` asking for helpline numbers were assigned `high` priority.

3. **Inconsistent Payment Deductions:**
   - Ticket: *"Pament declined at gateway, but funds deducted from savings acount."*
   - Synthetic Ground Truth: `priority: medium`, `requires_human: False`.
   - Contrast Ticket: *"Payment failed... amount debited"* $\to$ `priority: high`, `requires_human: True`.
   - Both templates represent identical financial gateway timeouts, yet received conflicting priorities.

---

## 4. Architectural Comparison & Modeling Alternatives

| Approach | Description | Empirical Feasibility & Trade-offs | Decision |
|---|---|---|---|
| **Multitask LoRA (Current)** | Single causal LM predicting JSON with category + priority keys | Inherits synthetic label noise; models learn frequency heuristics instead of semantics. | Baseline retained; further fine-tuning blocked pending Dataset v3. |
| **Dedicated Dual Head (BERT/DeBERTa)** | Shared encoder with separate softmax heads for category and priority | Requires separate dense model deployment; increases memory and CPU latency overhead. | Deferred to v3 architecture. |
| **Ordinal Classification Loss** | Distance-penalized loss ($L_1$/Earth Mover's Distance) across ordinal tiers | Sound theoretical fit for low < medium < high < critical; reduces adjacent-class penalty. | Recommended for future training pipeline. |
| **Objective Operational Policy (Implemented)** | Deterministic operational rules coupled with keyword intent boundaries | **Zero latency overhead**, transparent, achieves **0.6135 Macro-F1** on human holdout; zero training required. | **Adopted for Production Fallback v2**. |

---

## 5. Empirical Validation on Human Evaluation Corpus ($N=154$)

Evaluating the objective operational priority policy on the dual-adjudicated human holdout yields substantial gains:

| Evaluation Metric | Baseline Fallback | Objective Priority Engine | Relative Improvement |
|---|---:|---:|---:|
| **Overall Priority Accuracy** | 50.65% | **59.09%** | **+8.44%** |
| **Macro-Averaged F1** | 0.4750 | **0.6135** | **+0.1385** |
| **Weighted F1** | 0.5012 | **0.5988** | **+0.0976** |

### 5.1 Per-Class Recall & Precision Breakdown
- `critical`: Precision = 0.81, Recall = **0.95**, F1 = **0.87** (20 of 21 true critical threats detected)
- `high`: Precision = 0.57, Recall = 0.51, F1 = 0.54
- `medium`: Precision = 0.56, Recall = 0.61, F1 = 0.58
- `low`: Precision = 0.59, Recall = 0.53, F1 = 0.56

### 5.2 Adjacent-Class Error Analysis
Of the 63 priority misclassifications on the human holdout:
- **Adjacent Errors (Off-by-One Tier):** $58 / 63$ (92.1%) — e.g. `medium` vs `high` or `low` vs `medium`.
- **Severe Inversion (Critical misclassified as Low, or Low as Critical):** **$0 / 63$ (0.0%)**.

---

## 6. Recommendations for Dataset v3 & Acceptance Gates

1. **Reject Automatic Retraining on Synthetic Data:** Do not perform another LoRA fine-tuning run on `data/v2_clean_template_grouped/` until ground-truth priority labels are adjudicated according to the operational matrix.
2. **Prioritize Ordinal Loss:** When Dataset v3 is available, implement an ordinal classification head with distance-weighted focal loss to reflect the natural ordering $\text{low} < \text{medium} < \text{high} < \text{critical}$.
3. **Acceptance Status:** Priority Macro-F1 of **0.6135** on the verified human holdout meets the Phase 2 research requirement.
