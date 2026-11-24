# NEXUS-SLM — Dataset Diversity & Redundancy Audit Report

## 1. Executive Summary

This report documents the empirical audit of dataset redundancy, lexical diversity, template replication, and cross-split overlap across the NEXUS-SLM dataset families.

### Key Audit Findings
- **Synthetic Template Redundancy:** The current operational benchmark `data/v2_clean_template_grouped/` contains **5,992 total records** across train (4,797), validation (449), and test (746). However, the entire dataset is generated from only **648 underlying normalized template skeletons**.
- **Severe Compression in Synthetic Test Split:** In `test_clean.jsonl` ($N=746$), there are only **291 unique surface texts** and **70 unique template skeletons** (a compression ratio of $0.0938$). Over 60% of the synthetic test set consists of slotted variations of the same 70 sentence structures.
- **Cross-Split Template Disjointness:** In `data/v2_clean_template_grouped/`, cross-split template overlap is **strictly 0** (train-test overlap = 0, val-test overlap = 0). This confirms that the historical cluster-disjoint partitioning successfully removed exact template leakage present in unclustered v1/v2 datasets.
- **Human Holdout Contrast:** In stark contrast, the 200-ticket human evaluation benchmark exhibits a Type-Token Ratio (TTR) of **0.4337** (vs **0.0965** in the synthetic test split) and an average ticket length of **14.13 words** (vs **8.32 words**). The human benchmark introduces an open vocabulary with idioms, punctuation irregularities, and compound intents that cannot be captured by slot-substituted synthetic generators.

---

## 2. Quantitative Comparative Metrics

The following empirical measurements were computed via deterministic lexical and syntactic normalization:

| Split / Partition | Total Records | Unique Surface Texts | Unique Templates | Template Compression Ratio | Vocabulary Size | Total Tokens | Type-Token Ratio (TTR) | Mean Word Length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train (`train_clean.jsonl`)** | 4,797 | 1,506 | 521 | 0.1086 | 2,409 | 48,807 | 0.0494 | 10.17 |
| **Val (`val_clean.jsonl`)** | 449 | 163 | 57 | 0.1269 | 435 | 3,587 | 0.1213 | 7.99 |
| **Historical Test (`test_clean.jsonl`)** | 746 | 291 | 70 | 0.0938 | 599 | 6,205 | 0.0965 | 8.32 |
| **Human Holdout (`annotation_template.jsonl`)** | 200 | 200 | 185* | 0.9250 | 1,226 | 2,827 | **0.4337** | **14.13** |

*\*Note: For the human holdout, 16 records represent reserved pending intake slots sharing placeholder phrasing. The 184 completed human scenarios have a 100% unique template ratio (1.000).*

---

## 3. Lexical and N-Gram Overlap Analysis

To evaluate generalization vs memorization risk, we measured n-gram Jaccard similarity and test-containment ratios ($|Train \cap Test| / |Test|$) between the training partition and test holdouts:

### 3.1 Train vs Historical Synthetic Test ($N=746$)
- **Unigram Jaccard:** 0.1039 | **Test Containment:** **47.25%**
- **Bigram Jaccard:** 0.0339 | **Test Containment:** **21.88%**
- **Trigram Jaccard:** 0.0157 | **Test Containment:** **10.76%**
- **Interpretation:** While template skeletons are disjoint, nearly 48% of the test unigrams and 22% of the test bigrams appear in the training split. This is typical of domain-specific customer support vocabularies (e.g., "refund", "password", "order", "delivery").

### 3.2 Train vs Human Evaluation Holdout ($N=184$ Active Scenarios)
- **Unigram Human Containment:** **44.70%** (55.30% novel vocabulary unseen in synthetic training)
- **Bigram Human Containment:** **16.24%** (83.76% novel word pairs)
- **Trigram Human Containment:** **5.81%** (94.19% novel 3-word syntactic structures)
- **Interpretation:** The human benchmark presents a genuine generalization hurdle: 94.2% of its trigrams are entirely absent from the synthetic training corpus. A model relying on synthetic n-gram memorization experiences severe degradation when confronted with human syntax.

---

## 4. Entity-Slot Substitution Analysis

Synthetic ticket generation in NEXUS-SLM relied heavily on regularized slot substitution where template strings were populated with pseudo-random entities.

### Prevalence of Entity Slots Across Splits:
- **Order / Transaction IDs:** Present in 982 train tickets (20.5%), 83 val tickets (18.5%), 164 test tickets (22.0%), and 52 human tickets (26.0%).
- **Monetary Amounts:** Present in 47 train tickets (1.0%), 2 val tickets (0.4%), 17 test tickets (2.3%), and 14 human tickets (7.0%).
- **Arbitrary Numeric Tokens:** Present in 935 train tickets (19.5%), 120 val tickets (26.7%), 213 test tickets (28.6%), and 54 human tickets (27.0%).

### Common Template Frequency Distribution in Synthetic Test ($N=746$):
The top 5 most frequent template clusters in the synthetic test set account for **108 of the 746 tickets (14.5%)**:
1. `"sampark soochana"` (Contact info query) — 27 occurrences
2. `"तकनीकी समस्या है"` (Technical issue) — 21 occurrences
3. `"bhai maine password reset ka request daala tha par mail me jo link aaya wo expire dikha raha hai please ek fresh password reset link bhejo mere email <EMAIL> pe"` — 20 occurrences
4. `"need tax gst invoice with company registered details for <ORDER_ID>"` — 20 occurrences
5. `"कैंसिल करना है"` (Want to cancel) — 20 occurrences

---

## 5. Architectural Implications & Recommendations

1. **Synthetic Test Saturation:** The 746-ticket synthetic test set is saturated and repetitive. High performance on this benchmark (e.g. ~70% accuracy) does not guarantee real-world generalization because 70 template structures cannot adequately represent human communicative variability.
2. **Locked Historical Status:** As mandated by project rules, `data/v2_clean_template_grouped/test_clean.jsonl` must remain strictly locked as an invariant regression benchmark.
3. **Primary Benchmark Transition:** Future model promotions must be gated against the **Sealed Human Evaluation Benchmark** (`data/human_eval/`) as the definitive evaluation ground truth, with synthetic benchmarks relegated to regression testing.
