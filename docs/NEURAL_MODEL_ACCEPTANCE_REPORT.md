# NEXUS-SLM — Neural Model V2 Acceptance Report

> **Promotion Decision:** `EXPERIMENTAL_NOT_PROMOTED` (Heuristic Fallback remains the sole production engine).
> **Evaluated Candidate:** `exp_v2_5_hybrid_qwen_lexical_fusion` (`Qwen/Qwen2.5-0.5B-Instruct` 1792-d + Multilingual TF-IDF SVD 384-d + 8 script dims -> Multitask Residual + Ordinal Priority + Temp Scaling `T=0.5446`).
> **Selection Criterion:** Strictly selected on combined validation split (`val_clean` + `val_v3`, `N=701`); locked test `test_clean` (`N=746`) was never used for checkpoint selection or hyperparameter tuning.

---

## 1. Predefined Workstream 3.4 Acceptance Scorecard

| Acceptance Gate | Predefined Target | Measured (`test_clean`, `N=746`) | Measured (`dev_audit_v3`, `N=84`) | Status |
|---|---:|---:|---:|---|
| **Category Accuracy** | >= 85.00% | **89.14%** | 80.95% | **PASS** (on `test_clean`) |
| **Category Macro-F1** | >= 0.8000 | **0.8558** | 0.8055 | **PASS** |
| **Priority Accuracy** | >= 80.00% | **48.66%** | 54.76% | **FAIL** |
| **Priority Macro-F1** | >= 0.7500 | **0.4290** | 0.5656 | **FAIL** |
| **Valid Structured Output** | 100.00% | **100.00%** | 100.00% | **PASS** (architectural guarantee) |
| **Independent Safety Acceptance** | Pass (`0` Crit FN + Sealed V2 Eval) | `0` Crit FN (Synthetic) | **`PENDING`** (`human_eval_v2`: 0/300 ready) | **FAIL / PENDING** |

---

## 2. Granular Comparison: Legacy V1 LoRA vs V2 Research Candidate

| System | Category Acc | Category Macro-F1 | Priority Acc | Priority Macro-F1 | Lang Acc | Hindi Cat Acc | Critical FN (Policy) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| True-Base `Qwen2.5-0.5B` + Clean V2 LoRA | 53.89% | 0.5375 | 44.24% | 0.2491 | 84.58% | 20.79% | 0 | **REJECTED** |
| `exp_v2_5` Hybrid Neural Head (Unpromoted) | **89.14%** | **0.8558** | 48.66% | 0.4290 | **96.78%** | **80.90%** | 0 | **EXPERIMENTAL_NOT_PROMOTED** |

### Exposed Development Human Evaluation Reference (`N=184`, Non-Sealed)
For diagnostic tracking only (tagged **development-exposed**, never cited as independent validation):
- Category Accuracy: **87.50%** | Category Macro-F1: **0.8866** (all 14 classes >= 0.6957 F1)
- Priority Accuracy: **55.98%** | Priority Macro-F1: **0.5617** (`low` 0.6087, `medium` 0.5952, `high` 0.4000, `critical` 0.6429)
- Critical Recall (Policy): **100.0%** (`0` critical FN)

---

## 3. Blocking Gaps Preventing Production Promotion

1. **Priority Accuracy & Macro-F1 Failure (`48.66%` / `0.4290` vs `>= 80%` / `0.75`):**
   - Root cause is dual: (a) high label entropy in legacy synthetic `test_clean` priority annotations where identical short utterances carry conflicting priorities, and (b) subtle SLA vs emotional severity boundaries (`medium` vs `high`) requiring deterministic business policy rules rather than text-only classification.
   - Note that standalone `exp_v2_4` reached `76.19%` / `0.7457` F1 on clean V3 rubric `dev_audit_v3`, proving the model architecture can learn priority when labels follow an unambiguous rubric, though still shy of `80% / 0.75` consistently across domains.
2. **Per-Class Category Weaknesses on `test_clean`:**
   - Despite eliminating `general_query` and `product_complaint` zero-F1 collapses, `delivery_issue` (`0.4211` F1), `general_query` (`0.4789` F1), and `technical_issue` (`0.5634` F1) require stronger semantic coverage before autonomous routing without human supervision.
3. **Independent Human Evaluation Pending (`data/human_eval_v2/`):**
   - Zero independently double-annotated, adjudicated external scenarios (`0 / 300`) are presently populated in `data/human_eval_v2/`. No neural model may be promoted to production before passing blind independent evaluation under [FINAL_HUMAN_EVALUATION_PROTOCOL.md](file:///C:/Users/avira/Desktop/NEXUS-SLM/docs/FINAL_HUMAN_EVALUATION_PROTOCOL.md).

---

## 4. Governance & Runtime Binding

- `models/registry.json` and `api/main.py` retain **Heuristic Fallback V2/V3** as the active production engine.
- `strict_neural_mode=True` explicitly returns HTTP `503 Service Unavailable` rather than silently downgrading to heuristics.
- Weight artifacts (`experiments/artifacts/v2_models/nexus_v2_hybrid_multitask_head.pt`) are tracked via SHA-256 in `models/artifact_manifest.json` (`29156d39...`) and excluded from Git commits via `.gitignore`.
