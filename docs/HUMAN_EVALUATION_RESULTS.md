# NEXUS-SLM Human Evaluation Benchmark Results

**Document ID:** RESULTS-HUMAN-EVAL-V2-RECONCILED  
**Evaluation Date:** 2026-11-24  
**Benchmark Partition:** Sealed Human-Authored Holdout (`data/human_eval/annotation_template.jsonl`)  
**Sample Count:** $N = 200$ Total ($N = 184$ Adjudicated Scenarios, $N = 16$ Reserved Live Intake Slots)  
**Dataset SHA-256:** `e26d4eec99ddd247d3037652466df0024100b0cc85b9ca48e0fa32ed300923d3`  
**Evaluator:** Principal QA Lead & Independent ML Auditor  
**Machine-Readable Artifacts:**  
- `experiments/results/eval_human_holdout_baseline_v1.json`
- `experiments/results/eval_human_holdout_refined_v2.json`
**Status:** Operational Infrastructure Verified | Baseline Measured | Policy V2 Validated | Real Customer Ingestion PENDING  

---

## 1. Executive Summary

This report documents the empirical evaluation of the NEXUS-SLM triage system against the 200-ticket human-authored scenario evaluation corpus. 

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

## 3. Policy Evolution & Side-by-Side Empirical Results

To eliminate past documentation ambiguities, results are reported for both policy generations on the exact same $N=184$ adjudicated records.

### 3.1 Primary Metric Comparison

| Metric | Baseline Policy V1 | Refined Policy V2 | Delta (V2 vs V1) | Synthetic Test (V2) |
|---|---:|---:|---|---:|
| **Category Accuracy** | 56.52% | **60.33%** | +3.81% | 57.51% |
| **Category Macro-F1** | 0.5965 | **0.6278** | +0.0313 | 0.6516 |
| **Category Weighted-F1** | 0.5898 | **0.6268** | +0.0370 | 0.6267 |
| **Priority Accuracy** | 51.09% | **59.78%** | **+8.69%** | 36.46% |
| **Priority Macro-F1** | 0.5280 | **0.6519** | **+0.1239** | 0.3827 |
| **Priority Weighted-F1** | 0.5222 | **0.5837** | +0.0615 | 0.3703 |
| **Language Accuracy** | 97.28% | **97.28%** | 0.00% | 81.23% |
| **Escalation Rate** | 100.00% | **58.70%** | **-41.30%** | 70.64% |
| **Critical Recall** | **100.00%** | **100.00%** | 0.00% (Safety Invariant) | 100.00% |
| **Critical False Negatives**| **0** | **0** | 0 (Zero Tolerance) | 0 |
| **Exact 4-Field Match** | 13.59% | **35.87%** | **+22.28%** | 14.21% |

---

## 4. Analytical Findings & Error Modes

### 4.1 Priority Accuracy Improvement via Contextual Rubric
Under Baseline Policy V1, priority was assigned via static 1:1 mapping from category. This failed on multi-intent tickets (e.g. general queries involving missing high-value shipments). Refined Policy V2 uses `determine_contextual_priority` to dynamically scale priority based on explicit urgency markers, monetary sums ($\ge \$100$), recurring failures, and security indicators, raising Priority Accuracy to **59.78%** and Priority Macro-F1 to **0.6519**.

### 4.2 Resolution of the Escalation Floor
In Baseline V1, Rule 12 (`confidence is None`) unconditionally escalated all heuristic classifications, forcing a **100.00%** escalation rate. Refined Policy V2 safely routes routine, unambiguous inquiries (e.g., standard shipping FAQ, password guidance) via `automated_routing_permitted`, reducing human holdout escalation to **58.70%** without a single critical safety slip (**100% Critical Recall, 0 False Negatives**).

### 4.3 Multilingual Token-Anchored Generalization
Language accuracy reached **97.28%** across human-authored scenarios (100% English, 100% Hindi Devanagari, 91.38% Hinglish). The token-anchored regex prevented false-positive substring matches on English tokens like *"camera"* or *"chair"*.

---

## 5. Statistical Uncertainty & Sample Limitations

For $N=184$, 95% Wilson score confidence intervals for key metrics are:
- Category Accuracy: $60.33\%$ $[53.12\%, 67.12\%]$
- Priority Accuracy: $59.78\%$ $[52.57\%, 66.60\%]$
- Language Accuracy: $97.28\%$ $[93.75\%, 98.85\%]$
- Escalation Rate: $58.70\%$ $[51.48\%, 65.57\%]$
- Critical Safety Recall: $100.00\%$ $[83.42\%, 100.00\%]$ ($N=20$ critical cases)
