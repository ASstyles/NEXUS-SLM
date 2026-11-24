# NEXUS-SLM Dataset Card: Clean V2 Multilingual Support Benchmark

**Dataset Name:** `v2_clean_template_grouped`  
**Dataset Version:** 2.0.0-Clean  
**Release Date:** 2026-10-13  
**Lock Hash (Test Set):** `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b`  
**Maintenance:** NEXUS-SLM AI Research & Engineering Team  

---

## 1. Dataset Summary & Purpose

The `v2_clean_template_grouped` dataset is a controlled, synthetic support-ticket triage benchmark designed to train and evaluate small language models (SLMs) on structured triage tasks across 14 support categories, 4 urgency priorities, and 3 languages (English, Native Devanagari Hindi, and Romanized Hinglish).

The dataset is partitioned into three locked splits:
- **Train:** 4,797 records (947 unique normalized templates)
- **Validation:** 449 records (82 unique normalized templates)
- **Test (Locked Acceptance Set):** 746 records (131 unique normalized templates)

---

## 2. Data Provenance & Generation Methodology

### 2.1 Synthetic vs. Human Proportions
- **Synthetic Component:** 100% of the historical locked benchmark splits ($N = 5,992$ across train/val/test) are synthetically generated.
- **Human Component:** Deployed and tracked under `data/human_eval/`:
  - Exactly **200 records** in `annotation_template.jsonl`.
  - **184 completed, independently adjudicated human scenarios** across all 14 categories and 3 language formats (English, Hindi Devanagari, Hinglish).
  - **16 reserved live customer intake slots** (`HUMAN-PND001` to `HUMAN-PND016`) marked `PENDING` awaiting live customer intake and DPA consent.
  - Inter-Rater Reliability (IRR): Category Cohen's $\kappa = 0.9648$, Priority $\kappa = 0.9923$, Escalation $\kappa = 1.0000$.

### 2.2 Generation Architecture
Tickets were produced using grammar-pattern slot synthesis:
1. **Syntactic Skeletons:** Seed templates encoding specific customer intents (e.g. `Order <ORDER_ID> kab tak deliver hoga <CITY> me?`).
2. **Entity Slot Injection:** Programmatic slot filling with randomized entity dictionaries (Order numbers, city names, amounts, item categories, courier names).
3. **Lexical Normalization:** Rule-based verification to ensure consistent punctuation and ChatML tokenization structure.

---

## 3. Quantitative Quality Audit Findings

An independent quantitative audit of the three splits (`experiments/artifacts/dataset_quality_audit_report.json`) identified key distributional characteristics:

### 3.1 Lexical & Template Redundancy
| Metric | Train ($N=4,797$) | Val ($N=449$) | Test ($N=746$) |
|---|---|---|---|
| **Unique Raw Texts** | 4,797 | 449 | 746 |
| **Exact Raw Duplicates** | 0 | 0 | 0 |
| **Unique Normalized Skeletons** | 947 | 82 | 131 |
| **Skeleton Duplication Ratio** | 80.26% | 81.74% | 82.44% |
| **Mean Word Count** | 10.4 words | 10.8 words | 10.9 words |
| **Exact Raw Test $\cap$ Train Leakage** | — | — | **0 instances** (0.0%) |
| **Template Overlap (Test vs. Train)** | — | — | **3 / 131 templates (2.3%)** |

### 3.2 Category Imbalance & Sparsity
- **Dense Categories:** `order_cancel` (110 test samples), `refund_request` (78 test samples), `order_tracking` (76 test samples).
- **Extremely Sparse Category:** `subscription` has only **2 samples** in the entire 746-sample test set.
- **Statistical Implication:** Generalization claims for `subscription` cannot be defended on the synthetic test set alone; confidence intervals must reflect $N=2$.

---

## 4. Label Methodology & Circularity Audit

1. **Initial Label Assignment:** Ground-truth labels (`category`, `priority`, `language`, `requires_human`) were assigned programmatically at template creation.
2. **Keyword Circularity Warning:** Earlier benchmark versions utilized identical keyword dictionaries for dataset label generation and production escalation routing, artificially producing a nominal 100% escalation recall at the expense of 38.6% precision (345 false positives).
3. **Remediation:** Label generation logic has been decoupled from production runtime policy. Independent human adjudication workflows have been established in `data/human_eval/`.

---

## 5. Known Limitations & Prohibited Conclusions

1. **No Real-World Generalization Guarantee:** Passing performance on synthetic template variations proves pattern matching over defined slots; it does **not** prove robustness against real-world customer noise, typos, code-mixing ambiguity, or adversarial inputs.
2. **Devanagari Subword Fragmentation:** Small language models exhibit tokenization fragility on complex conjunct Devanagari scripts, necessitating conservative safety routing for native Hindi.
3. **Prohibited Claims:**
   - It is prohibited to claim production readiness based solely on synthetic test accuracy.
   - It is prohibited to report human evaluation scores until genuine, consented human tickets are collected and adjudicated.
