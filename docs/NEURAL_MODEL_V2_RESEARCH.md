# NEXUS-SLM — Neural Model V2 Deep Error Analysis, Diverse Data Architecture & Multitask Research

> **Status:** `EXPERIMENTAL_NOT_PROMOTED` (Validated locally; awaiting priority rubric resolution and independent human eval V2 `N >= 300`).
> **Artifacts:** [neural_error_taxonomy_v2.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/neural_error_taxonomy_v2.json) | [runs_v2.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/runs_v2.json) | [dataset_v3_manifest.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/data/v3_diverse/dataset_v3_manifest.json) | [v2_multiclass_calibration_report.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/v2_multiclass_calibration_report.json)

---

## 1. Workstream 3.1 — Deep Error Taxonomy on True Uncontaminated Qwen2.5-0.5B LoRA

Before designing any new architecture or dataset, we reproduced both the contaminated `scratch_merged_qwen` evaluation (`69.17%` category accuracy) and the **true uncontaminated Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` + `qwen25_05b_clean_v2_lora` evaluation** (`eval_qwen_clean_v2_true_base.json`) across all `N=746` rows of `data/v2_clean_template_grouped/test_clean.jsonl` (`sha256: fc65071b...`).

### 1.1 Baseline Collapse Comparison (Stacked Scratch vs True Base)

| Metric | Stacked `scratch_merged_qwen` | True Base `Qwen2.5-0.5B-Instruct` + Clean V2 LoRA | Diagnosis |
|---|---:|---:|---|
| **Category Accuracy** | 69.17% | **53.89%** | True generalization dropped -15.28 pp once V1 merge leak was removed |
| **Category Macro-F1** | 0.6044 | **0.5375** | Severe class collapse across rare and broad intent classes |
| **Priority Accuracy** | 47.86% | **44.24%** | Majority-class (`medium`/`high`) collapse |
| **Priority Macro-F1** | 0.2842 | **0.2491** | **0.00% recall** on `low` (`n=160`) and `critical` (`n=29`) |
| **Language Accuracy** | 96.78% | **84.58%** | Hindi script & Hinglish intent degradation |
| **English Cat Acc** | — | **93.90%** (`n=246`) | Strong English class separability (`macro_f1=0.5591`) |
| **Hindi Cat Acc** | — | **20.79%** (`n=178`, `macro_f1=0.0833`) | Catastrophic Devanagari BPE collapse into `payment_failure` |
| **Hinglish Cat Acc** | — | **41.61%** (`n=322`, `macro_f1=0.4942`) | Template shortcut confusion |

### 1.2 Focus Class & Concrete Error Examples (`neural_error_taxonomy_v2.json`)

| Focus Category | Support | True-Base F1 | Stacked F1 | Observed Failure Mode & Representative Exact Output |
|---|---:|---:|---:|---|
| `general_query` | 42 | **0.0000** | 0.0000 | Predicted as `payment_failure` (28) & `refund_request` (14). Ex: *"Sampark soochana"* -> `{"category": "payment_failure", "priority": "high", ...}` |
| `product_complaint` | 44 | **0.0000** | 0.0000 | Predicted as `refund_request` (17), `fraud_report` (15), `payment_failure` (12). Ex: *"wo catalog ke vivaran se bilkul mel nahi khata..."* -> `"refund_request"` |
| `technical_issue` | 51 | **0.0385** | — | Recall `1.96%` (50/51 misclassified as `payment_failure`). Ex: *"App kholte hi band ho ja raha hai, takneeki samasya hai."* -> `"payment_failure"` |
| `order_cancel` | 110 | **0.3582** | 0.5806 | Recall `21.82%` (42 -> `refund_request`, 27 -> `payment_failure`, 17 -> `delivery_issue`). Ex: *"Arrey bhai Order cancel please"* -> `"refund_request"` |
| `account_access` | 50 | **0.3871** | 0.3784 | Recall `24.00%` (38 -> `payment_failure`). Ex: *"OTP nahi aa raha"* -> `"payment_failure"` |
| `payment_failure` | 28 | **0.2022** | 0.6000 | Over-predicted as catch-all class (`precision = 11.24%`, `recall = 100.0%`) |
| `refund_request` | 78 | **0.5391** | 0.5632 | Precision `40.79%`, absorbs `order_cancel` & `product_complaint` |
| `fraud_report` | 47 | **0.8624** | 0.5802 | Precision `75.81%` (false positives from `product_complaint`), `recall = 100.0%` |
| `subscription` | 2 | **1.0000** | 0.2222 | Severe support scarcity in `test_clean` (`n=2` only) |

### 1.3 Root Causes Confirmed
1. **Template Shortcut Learning in V1/V2:** `train_clean.jsonl` (`4,797` rows) collapsed to only **~751 unique de-numbered text patterns**, causing repetitive `"Sir please App me..."` / `"Kripya aadesh sankhya ORD-..."` templates to dominate LoRA gradients.
2. **Stochastic Priority Labels in Legacy Synthetic Data:** In `test_clean.jsonl`, identical short texts (e.g., `"पासवर्ड भूल गया"` and `"OTP nahi aa raha"`) appear verbatim across `low`, `medium`, and `high` ground-truth labels, imposing a structural Bayes error ceiling near ~50–55% for single-text priority prediction on `test_clean`.
3. **Generative Tokenization Bottleneck on Devanagari:** Small autoregressive Qwen LoRA decoding degraded on Devanagari Hindi (`20.79%` category accuracy) when relying solely on causal next-token generation.

---

## 2. Workstream 3.2 — V3 Syntactically Diverse Dataset Protocol

We constructed `data/v3_diverse/` ([build_v3_diverse_dataset.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/scripts/data/build_v3_diverse_dataset.py)) strictly disjoint from all quarantined evaluation splits (`test_clean`, `val_clean`, `human_eval_v1`, `high_risk_challenge_set`, `v1 test`) using `LeakageDetector`:

| Split File | Samples | Unique De-numbered | EN / HI / Hinglish | SHA-256 (prefix) | Leakage Check vs 1,507 Quarantined |
|---|---:|---:|---:|---|---|
| [train_v3.jsonl](file:///C:/Users/avira/Desktop/NEXUS-SLM/data/v3_diverse/train_v3.jsonl) | 680 | 680 (`100%`) | 224 / 228 / 228 | `d72cba8975c1bb01...` | `0` overlaps |
| [val_v3.jsonl](file:///C:/Users/avira/Desktop/NEXUS-SLM/data/v3_diverse/val_v3.jsonl) | 252 | 252 (`100%`) | 84 / 84 / 84 | `51cf413c9e2e1d99...` | `0` overlaps |
| [cal_v3.jsonl](file:///C:/Users/avira/Desktop/NEXUS-SLM/data/v3_diverse/cal_v3.jsonl) | 84 | 84 (`100%`) | 28 / 28 / 28 | `af638a6e21c671f3...` | `0` overlaps |
| [dev_audit_v3.jsonl](file:///C:/Users/avira/Desktop/NEXUS-SLM/data/v3_diverse/dev_audit_v3.jsonl) | 84 | 84 (`100%`) | 28 / 28 / 28 | `0a28299713712c99...` | `0` overlaps |

Internal cross-split overlap (`train_vs_val`, `train_vs_cal`, `train_vs_audit`, `val_vs_cal`, `val_vs_audit`, `cal_vs_audit`) is strictly **`0`**. Equal coverage spans all 14 categories (`6` per class in `cal_v3`/`dev_audit_v3`, `18` per class in `val_v3`, `44–64` per class in `train_v3`), incorporating informal shorthand, typos, formal Hindi, Romanized Hinglish, multi-intent tickets, and negative controls. All V3 records remain explicitly labeled `synthetic_v3_diverse` (NOT real customer tickets).

---

## 3. Workstream 3.3 — Architecture & Data Ablation Experiments (`experiments/runs_v2.json`)

Implemented in [src/models/multitask_v2.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/src/models/multitask_v2.py) and executed in [run_v2_architecture_experiments.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/scripts/experiments_v2/run_v2_architecture_experiments.py) on Windows x64 CPU (`torch 2.9.0+cpu`, Python `3.10.9`). Model selection was performed **strictly on combined validation (`val_clean` + `val_v3`, `N=701`)** without ever selecting on `test_clean`.

### 3.1 Controlled Ablation Leaderboard

| Experiment ID | Architecture & Data | Val (`N=701`) Cat Acc / F1 | `dev_audit_v3` (`N=84`) Cat Acc / F1 | `dev_audit_v3` Prio Acc / F1 | `test_clean` (`N=746`) Cat Acc / F1 | `test_clean` Prio Acc / F1 | `test_clean` Lang Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| **Baseline (Prev)** | Qwen2.5-0.5B True-Base + Clean V2 LoRA (gen) | — | — | — | 53.89% / 0.5375 | 44.24% / 0.2491 | 84.58% |
| `exp_v2_1` (`2.37s`) | Char+Word TF-IDF + Balanced LogReg (`train_clean` raw only) | 83.45% / 0.8084 | 65.48% / 0.6431 | 33.33% / 0.3824 | 77.48% / 0.7696 | 45.17% / 0.4630 | 96.65% |
| `exp_v2_2` (`2.08s`) | Char+Word TF-IDF + Balanced LogReg (`dedup + 2x train_v3`) | 84.31% / 0.8236 | **85.71% / 0.8617** | 55.95% / 0.5652 | **91.55% / 0.8966** | 41.29% / 0.4226 | 96.78% |
| `exp_v2_3` (`11.67s`) | PyTorch Multitask Residual + Ordinal Head (SVD 384 + 8 script) | 87.02% / 0.8525 | 76.19% / 0.7580 | 57.14% / 0.5997 | 83.91% / 0.8032 | 48.39% / 0.4208 | 95.04% |
| `exp_v2_4` (`23.97s`) | Frozen HF `Qwen2.5-0.5B` (1792-d) + Multitask Ordinal Head | 73.32% / 0.7222 | 71.43% / 0.7180 | **76.19% / 0.7457** | 76.14% / 0.6903 | 44.64% / 0.3897 | 96.78% |
| **`exp_v2_5` (Selected)** | **Fusion (`Qwen2.5-0.5B 1792-d + TF-IDF SVD 384 + 8 script`) + Multitask + Ordinal + Temp Scaling** | **88.45% / 0.8767** | **80.95% / 0.8055** | 54.76% / 0.5656 | **89.14% / 0.8558** | **48.66% / 0.4290** | **96.78%** |

### 3.2 Key Scientific Findings Across Experiments
1. **Diverse Data Impact (`exp_v2_1 -> exp_v2_2`):** Replacing repetitive raw templates with skeleton-deduplicated + V3 diverse data raised diverse unseen `dev_audit_v3` performance by **+20.23 pp** (`65.48% -> 85.71%`), confirming that overfitting to slot templates was a primary defect.
2. **Complementary Signal of LLM Hidden States vs Lexical Char/Word Features (`exp_v2_3`, `exp_v2_4`, `exp_v2_5`):** Frozen HF `Qwen2.5-0.5B` contextual features achieved the highest priority macro-F1 on diverse data (**`0.7457`**, `76.19%` accuracy on `dev_audit_v3`), whereas lexical char+word `3–5` n-grams anchored cross-script Hinglish precision. Combining both in `exp_v2_5` yielded the highest validation score (`0.8845` cat acc, `0.8767` macro-F1).
3. **Eliminating Class Collapses in `exp_v2_5` on `test_clean`:**
   - `product_complaint`: `0.0000 -> 0.6154` F1 (`100%` recall)
   - `general_query`: `0.0000 -> 0.4789` F1 (`40.48%` recall)
   - `technical_issue`: `0.0385 -> 0.5634` F1 (`39.22%` recall)
   - `payment_failure`: `0.2022 -> 1.0000` F1 | `account_access`: `0.3871 -> 1.0000` F1 | `order_cancel`: `0.3582 -> 0.9862` F1 | `refund_request`: `0.5391 -> 1.0000` F1.
   - **Remaining Weakest Classes on `test_clean`:** `delivery_issue` (`0.4211` F1, `26.67%` recall, `n=30`), `general_query` (`0.4789` F1, `n=42`), `technical_issue` (`0.5634` F1, `n=51`).
   - **Language Parity on `test_clean`:** English `100.0%` cat acc, Hinglish `85.40%` (`0.8576` macro-F1), Hindi `80.90%` (+60.11 pp over True-Base Qwen LoRA's `20.79%`).
4. **Development-Exposed `human_eval_v1` (`N=184`):** On human-authored scenarios (tagged development-exposed; not used for test claims), `exp_v2_5` reached **`87.50%` category accuracy (`0.8866` Macro-F1)** with minimum per-class F1 `0.6957` (`general_query`), `100%` language detection, and `100%` critical recall (`0` critical FN), while priority remained `55.98%` (`0.5617` Macro-F1).
5. **Multiclass Probability Calibration (`cal_v3`, `N=84` -> `test_clean`):** Fitting a scalar temperature (`T = 0.5446`) reduced locked-test **ECE from `0.1343 -> 0.0534`**, **Brier score from `0.1833 -> 0.1510`**, and **NLL from `0.5007 -> 0.4137`**. Vector scaling (`cal_v3` `N=84` -> `ECE=0.1050`) overfit 14 class parameters on small `cal_v3`, confirming scalar temperature scaling is safer until independent `N >= 300` calibration data exists.

### 3.3 Checkpoint & Reproducibility Provenance
- **Hybrid Weights:** `experiments/artifacts/v2_models/nexus_v2_hybrid_multitask_head.pt` (gitignored binary; SHA-256: `29156d3904293c0fa60819144cd4fa87b8bed4093135fd24cea989772ba680b6`)
- **Regenerable Command:** `.\.venv\Scripts\python.exe scripts\experiments_v2\run_v2_architecture_experiments.py`
- **CPU Training Duration:** `25.45 s` (head optimization; one-time frozen Qwen feature pass on CPU)
- **Production Policy Guard:** Under Workstream 3.4 acceptance criteria, **`exp_v2_5` remains `EXPERIMENTAL_NOT_PROMOTED`** and production continues running deterministic Heuristic Fallback.
