# NEXUS-SLM — Final Probability Calibration Acceptance Report

> **Artifacts:** [v2_multiclass_calibration_report.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/v2_multiclass_calibration_report.json) | [multitask_v2.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/src/models/multitask_v2.py) | [calibration.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/src/models/calibration.py) | [calibration_acceptance_report.md](file:///C:/Users/avira/Desktop/NEXUS-SLM/docs/CALIBRATION_ACCEPTANCE_REPORT.md)

---

## 1. Architectural Separation: Confidence Semantics Across Engines

| Engine Mode | Probability Head Available? | API `confidence` / `calibrated_confidence` Field | `confidence_status` Field | Rationale |
|---|---|---|---|---|
| **Heuristic Fallback V2 / V3** (Production) | No | `null` | `uncalibrated_heuristic_omitted` | Rule-based regex classification has no probabilistic class distribution; omitting `null` avoids synthetic confidence constants. |
| **Generative Qwen2.5-0.5B LoRA** (Legacy Exp) | No (autoregressive tokens) | `null` | `generative_token_probability_unavailable` | Autoregressive JSON token probabilities confuse surface syntax confidence (e.g. `{`, `"category"`) with semantic class uncertainty. |
| **V2 Neural Hybrid Multitask (`exp_v2_5`)** (Exp) | **Yes** (14-class softmax logits) | Softmax max after scalar `T=0.5446` | `temperature_scaled_multiclass` | Direct supervised classification logits across 14 categories, post-hoc calibrated on disjoint `cal_v3.jsonl`. |

---

## 2. True Multiclass Calibration Evaluation (`exp_v2_5`, Disjoint `cal_v3` `N=84` -> `test_clean` `N=746`)

Scalar temperature scaling (`T = 0.5446`) and penalized per-class vector scaling (`l2 = 0.05`) were fit exclusively on the held-out calibration split `data/v3_diverse/cal_v3.jsonl` (`N=84`, `sha256: af638a6e21c671f3...`) without ever exposing `test_clean.jsonl` during fitting.

### 2.1 Overall Comparison (`test_clean` `N=746` & `dev_audit_v3` `N=84`)

| Partition & Calibration State | Samples | Accuracy | Mean Conf | Overconf Gap | **ECE (10-bin)** | **MCE** | **Brier Score** | **NLL** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_clean` Uncalibrated Logits | 746 | 0.8914 | 0.7571 | -0.1343 | **0.1343** | 0.3670 | 0.1833 | 0.5007 |
| **`test_clean` Scalar Temp (`T=0.5446`)** | 746 | 0.8914 | 0.8812 | **-0.0102** | **0.0534** | 0.3812 | **0.1510** | **0.4137** |
| `test_clean` Vector Scaled (`cal_v3` N=84) | 746 | 0.8257 | 0.9140 | +0.0882 | 0.1050 | 0.6726 | 0.2634 | 0.6283 |
| `dev_audit_v3` Uncalibrated Logits | 84 | 0.8095 | 0.7177 | -0.0918 | 0.1258 | 0.8255 | 0.2447 | 0.5521 |
| **`dev_audit_v3` Scalar Temp (`T=0.5446`)** | 84 | 0.8095 | 0.8788 | +0.0693 | **0.0918** | 0.7668 | **0.2374** | **0.4513** |

**Key Engineering Finding:**
1. Scalar temperature scaling (`T=0.5446`) cut locked-test ECE by **60.2%** (`0.1343 -> 0.0534`), reduced mean calibration gap to `-0.0102`, and lowered multi-class NLL from `0.5007` to `0.4137`.
2. Per-class vector scaling (28 parameters on `N=84`) degraded locked-test accuracy (`89.14% -> 82.57%`) and increased ECE to `0.1050` due to overfitting class biases — validating our decision to **reject vector scaling** in favor of single-scalar `T`.

### 2.2 Per-Language Calibrated Reliability (`test_clean` `N=746`, `T=0.5446`)

| Language | Samples | Accuracy | Mean Conf | Overconf Gap | ECE | MCE | Brier Score | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English | 246 | 1.0000 | 0.9651 | -0.0349 | **0.0349** | 0.1759 | **0.0079** | **0.0374** |
| Hinglish | 322 | 0.8540 | 0.9046 | +0.0506 | **0.0635** | 0.8380 | 0.1639 | 0.4013 |
| Hindi (Devanagari) | 178 | 0.8090 | 0.7229 | -0.0861 | **0.1091** | 0.8196 | 0.3255 | 0.9564 |

*(Notice Hindi outputs still carry conservative uncertainty — `mean_conf = 0.7229` below the `0.85` escalation threshold — matching policy safeguards.)*

---

## 3. Risk–Coverage Selective Automation Tradeoff (`test_clean`, `T=0.5446`)

| Confidence Threshold ($\tau$) | Retained Auto-Candidates | Coverage Rate | Selective Category Accuracy |
|---:|---:|---:|---:|
| $\ge 0.50$ | 655 / 746 | 87.80% | **96.18%** |
| $\ge 0.65$ | 619 / 746 | 82.98% | **99.35%** |
| $\ge 0.75$ | 603 / 746 | 80.83% | **99.34%** |
| **$\ge 0.85$ (Policy Gate)** | **572 / 746** | **76.68%** | **99.83%** |
| $\ge 0.90$ | 571 / 746 | 76.54% | **100.00%** |
| $\ge 0.95$ | 499 / 746 | 66.89% | **100.00%** |

At the policy gate (`0.85`), 572/746 tickets above threshold achieve **99.83% category accuracy**, while remaining low-confidence predictions fall cleanly to human review. (Priority gating and safety rules always override confidence.)

---

## 4. Calibration Acceptance Verdict

- **Multiclass Neural Head Calibration (`exp_v2_5`):** **PASS** (verified improvement from `0.1343 -> 0.0534` ECE, honest `0.85` selective accuracy `99.83%`, strict out-of-sample `cal_v3` fitting).
- **Production Heuristic Engine:** Correctly abstains with `confidence: null` and `confidence_status: "uncalibrated_heuristic_omitted"`.
- **Remaining Limitation:** Calibration must be re-verified on the sealed `N >= 300` independent human benchmark (`data/human_eval_v2/`) before enabling any neural confidence threshold in production.
