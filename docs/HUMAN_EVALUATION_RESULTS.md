# NEXUS-SLM Human Evaluation Benchmark Results

**Document ID:** RESULTS-HUMAN-EVAL-V1  
**Evaluation Date:** 2026-11-24  
**Benchmark Partition:** Sealed Human-Authored Holdout (`data/human_eval/annotation_template.jsonl`)  
**Sample Count:** $N = 200$ Total ($N = 184$ Adjudicated Scenarios, $N = 16$ Pending Live Intake Slots)  
**Evaluator:** Principal QA Lead & Independent ML Auditor  
**Status:** Operational Infrastructure Verified | Baseline Measured | Real Customer Ingestion PENDING  

---

## 1. Executive Summary

This report documents the empirical evaluation of the NEXUS-SLM baseline triage engine against the newly expanded 200-ticket human-authored scenario evaluation corpus. 

The evaluation addresses the historical limitation of evaluating solely against synthetic template data. The corpus features natural human syntax, colloquial expressions, typographical errors, code-switching (Romanized Hinglish), native Hindi (Devanagari), multi-intent requests, and negative controls.

> [!IMPORTANT]
> **Data Provenance & Operational Classification:**
> The evaluated records represent **manually authored realistic scenario evaluation data** ($N=184$), created to evaluate out-of-distribution robustness. They are **not live customer interactions**.
> Formal live customer evaluation remains **PENDING** customer consent and live production intake under executed DPAs.

---

## 2. Inter-Rater Reliability (IRR) Metrics

All 184 adjudicated tickets underwent independent dual annotation (`annotator_alpha` paired with `annotator_beta` or `annotator_gamma`). Inter-annotator agreement was quantified using Cohen's Kappa ($\kappa$):

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

| Evaluation Dimension | Observed Agreement ($p_o$) | Cohen's Kappa ($\kappa$) | Interpretation |
|---|---|---|---|
| **Category Classification (14 classes)** | 96.74% | **0.9648** | Near-perfect inter-annotator consensus |
| **Priority Assessment (4 classes)** | 99.46% | **0.9923** | Near-perfect inter-annotator consensus |
| **Human Escalation Flag (`requires_human`)** | 100.00% | **1.0000** | Unanimous agreement on safety boundary |

Disputed tickets ($N=6$) involving multi-intent boundary cases (e.g. `order_cancel` vs `refund_request` for pre-dispatch cancellations, or `delivery_issue` vs `product_complaint` for courier transit damage) were formally adjudicated by the Lead QA Auditor with recorded rationales in `adjudication.adjudication_notes`.

---

## 3. Empirical Baseline Evaluation (Heuristic Fallback v2)

Evaluation was performed on the sealed $N=184$ adjudicated holdout using `src/inference/rules.py` and `src/inference/policy.py`.

### 3.1 Primary Metric Summary

| Metric | Synthetic Test ($N=746$) | Human-Authored Holdout ($N=184$) | Observed Delta |
|---|---:|---:|---|
| **Category Accuracy** | 56.03% | **58.15%** | +2.12% |
| **Category Macro-F1** | 0.6132 | **0.6131** | -0.0001 |
| **Category Weighted-F1** | 0.6175 | **0.6088** | -0.0087 |
| **Priority Accuracy** | 47.18% | **54.89%** | +7.71% |
| **Priority Macro-F1** | 0.4410 | **0.5230** | +0.0820 |
| **Priority Weighted-F1** | 0.4650 | **0.5222** | +0.0572 |
| **Language Accuracy** | 81.23% | **97.28%** | **+16.05%** |
| **Escalation Rate** | 100.00% | **100.00%** | 0.00% (Safety policy floor) |

---

## 4. Key Analytical Findings & Error Modes

### 4.1 Token-Anchored Language Detection Generalization
Language accuracy on the human-authored dataset reached **97.28%**, substantially higher than the synthetic test set (81.23%). The token-anchored Hinglish word-boundary regex (`src/inference/rules.py`) successfully eliminated false-positive substring matches (e.g. *"camera"* or *"chair"* triggering Hinglish) while robustly identifying authentic code-switched Hindi-English phrasing.

### 4.2 Priority Prediction Dynamics
Priority accuracy on the human holdout reached **54.89%** (Macro-F1 0.5230), outperforming the synthetic test set (47.18%). This is because human scenarios feature realistic operational signals (e.g. monetary deduction without order creation) that align with operational impact rubrics, whereas synthetic test data suffered from central-tendency compression and label noise.

### 4.3 Escalation Policy Conservative Floor
The heuristic fallback exhibits an escalation rate of **100.00%** on both the synthetic and human-authored holdout. This is caused by:
1. `HumanEscalationPolicy` line 188: Any ticket evaluated without calibrated confidence (`raw_confidence is None`) triggers mandatory fallback escalation (`confidence_unavailable_conservative_escalation`).
2. Broad keyword collision: Terms like `"tax invoice"` or `"renewal"` trigger financial dispute rules.
Addressing this without compromising safety is addressed in Priority Three (Contextual Escalation & Safety Precision).

---

## 5. Statistical Uncertainty & Sample Limitations

For $N=184$, 95% Wilson score confidence intervals for key metrics are:
- Category Accuracy: $58.15\%$ $[50.95\%, 65.04\%]$
- Priority Accuracy: $54.89\%$ $[47.70\%, 61.89\%]$
- Language Accuracy: $97.28\%$ $[93.75\%, 98.85\%]$

Given average support of $\sim 13$ samples per category, per-class metrics have moderate variance. As additional real customer interactions are collected under DPA agreements, the benchmark will expand toward $N=500+$.
