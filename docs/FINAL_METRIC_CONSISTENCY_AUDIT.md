# NEXUS-SLM Final Metric & Documentation Consistency Audit

**Audit Date (UTC):** 2026-11-25  
**Branch:** `sprint/final-independent-validation`  
**Audited Commit Baseline:** `e290a69214834bf3e4d48bc34601ff15581790d5` (PR #4 merge)  
**Authoritative Archive:** [`experiments/results/archive_v1/ARCHIVE_MANIFEST.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/experiments/results/archive_v1/ARCHIVE_MANIFEST.json)

---

## 1. Executive Summary of Forensic Findings

A full forensic audit across code, model checkpoints, datasets, JSON evaluation artifacts, and repository Markdown documentation revealed **five root sources of metric inconsistency**:

1. **Pre-PR#4 (`13:47Z`) vs. PR #4 (`23:00Z`) Policy Drift in Heuristic Fallback:**
   - [`heuristic_clean_v2_test_results.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/heuristic_clean_v2_test_results.json) (`2026-11-24T13:47:19Z`) measured Heuristic Fallback V2 under Policy V1 (`56.03%` category accuracy, `47.18%` priority accuracy, `100.00%` forced escalation, `0.053 ms` direct-call P50).
   - Commit `3e5a1ea` / PR #4 `e290a69` (`2026-11-24T23:00:00Z`) updated `src/inference/rules.py` and `src/inference/policy.py` to **Refined Policy V2** ([`eval_synthetic_test_refined_v2.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/experiments/results/eval_synthetic_test_refined_v2.json)): `57.51%` category accuracy (`0.6516` Macro-F1), `36.46%` priority accuracy (`0.3827` Macro-F1), `70.64%` escalation rate, `84.69%` high-risk recall (`32` non-critical high-risk false negatives, `0` critical FN).
   - [`README.md`](file:///c:/Users/avira/Desktop/NEXUS-SLM/README.md) and [`docs/RESULTS.md`](file:///c:/Users/avira/Desktop/NEXUS-SLM/docs/RESULTS.md) previously displayed the superseded `13:47Z` heuristic metrics alongside PR #4 documents citing `23:00Z` Refined Policy V2 metrics.
2. **Double-LoRA Contamination (`models/scratch_merged_qwen`) Inflated Clean V2 Qwen Benchmark by +15.28 pp:**
   - Inspection of tensor weights proved that `models/scratch_merged_qwen/model.safetensors` differs from clean HuggingFace `Qwen/Qwen2.5-0.5B-Instruct` (`q_proj` max diff `0.001774`, mean diff `0.000186`) because `scripts/optimization/quantize_qwen_gguf.py` previously merged V1 `qwen25_05b_lora_seed42` (trained on V1 template-leaked splits) into that directory.
   - `api/main.py`, `scripts/evaluation/evaluate_qwen_clean_test.py`, and `scripts/evaluation/run_final_model_acceptance_evaluation.py` loaded `qwen25_05b_clean_v2_lora` **on top of `models/scratch_merged_qwen`**, producing `69.17%` category accuracy (`0.6044` Macro-F1) and `96.78%` language accuracy.
   - Re-evaluating the exact same `qwen25_05b_clean_v2_lora` adapter (`SHA256: 9b677b8d488ccd402505ccd7293074aa16104236eecf8b370e5fde15ac4161d2`) on the **true uncontaminated HF `Qwen/Qwen2.5-0.5B-Instruct` base** ([`eval_qwen_clean_v2_true_base.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/experiments/results/eval_qwen_clean_v2_true_base.json)) yields **53.89% category accuracy (`0.5375` Macro-F1), 44.24% priority accuracy (`0.2491` Macro-F1), and 84.58% language accuracy**, with severe collapse on Devanagari Hindi (`20.79%`), `general_query` (`0.00%` F1), `product_complaint` (`0.00%` F1), `low` priority (`0.00%`), and `critical` priority (`0.00%`).
   - All three loaders have now been fixed to never substitute `models/scratch_merged_qwen`.
3. **Synthetic `+ 35.0 ms` Latency Offset & Hardcoded `0.90` Confidence:**
   - `api/main.py:323` and `scripts/evaluation/run_final_model_acceptance_evaluation.py:75` added an artificial `+ 35.0 ms` offset to heuristic fallback latency, creating spurious `35.07 ms` telemetry P50 readings. Direct-call latency reproduced in [`reproduced_refined_v2_baseline.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/experiments/results/reproduced_refined_v2_baseline.json) is `0.04 ms` P50 / `0.18 ms` P95 on `test_clean.jsonl` (`N=746`).
   - `scripts/evaluation/evaluate_qwen_clean_test.py:180` passed `raw_confidence=0.90 if is_valid else None` into `HumanEscalationPolicy`, bypassing Rule 12 conservative escalation (`29` FN in the Nov 24 artifact vs `0` FN when `raw_confidence=None`).
4. **Prometheus Telemetry Key Mismatch in `api/main.py`:**
   - `/metrics?format=prometheus` read non-existent flat keys (`latency_p50_ms`, `latency_p95_ms`, `escalation_rate`) instead of `latency_percentiles.p50`, `latency_percentiles.p95`, and `human_escalation_rate`, emitting `0.0` despite real telemetry. Fixed in `api/main.py`.
5. **Untested Documentation Claims, Wrong Hashes, Wrong Support Counts & Premature 10/10 Claim:**
   - Existing `scripts/evaluation/check_documentation_consistency.py` and `tests/test_evaluation_consistency.py` only validated 4 JSON files without inspecting a single Markdown file, dataset SHA-256 hash, or README figure.

---

## 2. Authoritative Dataset Hashes & Support Counts (Verified)

| Dataset Artifact | Verified Record Count | Verified SHA-256 | Exact Breakdown Verified |
|---|---:|---|---|
| `data/v2_clean_template_grouped/train_clean.jsonl` | 4,797 | `f67b50c7724e8cd53d029058c688e712b99b9a3f1be3df0a92c756975220fe1c` | EN 1,488 / HI 1,173 / Hinglish 2,136 |
| `data/v2_clean_template_grouped/val_clean.jsonl` | 449 | `faa98cf83cbe209736d8890a902f65fa382cb2ff31b96e55608d98d9e68fa29a` | EN 117 / HI 115 / Hinglish 217 (`general_query`: 3) |
| `data/v2_clean_template_grouped/test_clean.jsonl` | 746 | `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b` | EN 246 / HI 178 / Hinglish 322 (`subscription`: 2; `req_human`: 209 True / 537 False) |
| `data/human_eval/annotation_template.jsonl` | 200 total (184 active, 16 pending) | `e26d4eec99ddd247d3037652466df0024100b0cc85b9ca48e0fa32ed300923d3` | Active N=184: EN 80 / HI 46 / Hinglish 58; `req_human`: 63 True / 121 False |
| `data/evaluation/high_risk_challenge_set.jsonl` | 70 | `70b43711fd2f515bdd27d6daa8e604698a109f6093cdaea7f85bd584339dbc5b` | 7 high-risk challenge categories (10 each) |

---

## 3. Itemized Discrepancy & Remediation Table

| Document | Detected Discrepancy / Stale Claim | Authoritative Artifact Truth | Action Taken |
|---|---|---|---|
| `README.md` | Test count contradictory (`221` badge vs `215` in Sec 4); stale Nov 24 `13:47Z` heuristic metrics (`56.03%`, `47.18%`, `100%` esc) alongside Qwen language counts `231/232/283` | `pytest`: `221 passed, 1 skipped` (Redis atomic concurrency skipped locally without Redis daemon); `test_clean.jsonl` EN 246 / HI 178 / Hinglish 322; Refined Policy V2 + Heuristic V2 `57.51%`/`36.46%`/`70.64%`; True-base Qwen `53.89%` | Reconciled & auto-synced via `sync_results_from_artifacts.py` |
| `docs/RESULTS.md` | Presented Nov 24 `13:47Z` Policy V1 metrics without Policy V2 reconciliation or true-base Qwen separation | Both `eval_synthetic_test_baseline_v1.json` (`100%` esc), `eval_synthetic_test_refined_v2.json` (`70.64%` esc), `qwen25_05b_clean_v2_lora_test_results.json` (stacked `69.17%`), and `eval_qwen_clean_v2_true_base.json` (`53.89%`) | Updated with machine-synced versioned tables |
| `docs/model_card.md` & `docs/limitations.md` | Language counts `231/232/283`, stale `56.03%`/`47.18%`, omitted `scratch_merged_qwen` finding | EN 246 / HI 178 / Hinglish 322; Refined Policy V2 & true-base Qwen documented | Corrected |
| `docs/EXPERIMENT_TRACKING_STANDARD.md` | `val_clean.jsonl` listed as 744 samples / hash `88c3e8ec...` | Actual `N=449`, SHA-256 `faa98cf83cbe209736d8890a902f65fa382cb2ff31b96e55608d98d9e68fa29a` | Corrected |
| `docs/EVALUATION_INTEGRITY_AUDIT.md` | `test_clean.jsonl` hash listed as `86cae5be...` | Actual SHA-256 `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b` | Corrected |
| `docs/HUMAN_EVALUATION_PROTOCOL.md` & `HUMAN_EVALUATION_RESULTS.md` | `HUMAN_EVALUATION_PROTOCOL` cited `164` (`154` adjudicated, `10` reserved); wrong V1 language counts | `200` total (`184` active adjudicated, `16` pending); EN 80 / HI 46 / Hinglish 58; explicitly tagged development-exposed | Corrected |
| `docs/FINAL_EXCELLENCE_SPRINT_REPORT.md` | Claimed human breakdown `70 EN / 60 Hinglish / 54 HI`, V1 Priority `50.65%`/`0.4750` | Actual N=184: EN 80 / HI 46 / Hinglish 58; `eval_human_holdout_baseline_v1.json` Priority `51.09%`/`0.5280` | Corrected |
| `docs/NEURAL_MODEL_ERROR_ANALYSIS.md` | Class supports summed to `528 != 746`; claimed `general_query` P=`0.2222` & `subscription` R=`0.0000` | Actual N=746 supports; stacked `general_query` P=`0.0000` R=`0.0000`, `subscription` P=`0.1250` R=`1.0000`; now adds true-base error breakdown | Corrected |
| `docs/MULTILINGUAL_EVALUATION_REPORT.md` | Claimed balanced `248/250/248` counts (`test_clean`) | Actual `246` EN / `178` HI / `322` Hinglish | Corrected |
| `docs/SAFETY_AND_ESCALATION_ACCEPTANCE.md` | Claimed `N=85` adversarial suite | Actual repo artifacts: `adversarial_report.json` (`N=6`) + `prompt_injection_evaluation.json` (`N=8`) = 14 unique payloads (note historical N=85 claim unverified) | Corrected |
| `docs/CALIBRATION_ACCEPTANCE_REPORT.md` | Cited Brier `0.2894` (raw) / `0.2781` (T=1.15) | Actual `final_calibration_metrics.json`: ECE `0.2851`/`0.2712`, Brier `0.2933`/`0.2852` | Corrected |
| `docs/FINAL_10_OUT_OF_10_ACCEPTANCE_REPORT.md` | Prematurely awarded 10/10 despite failing neural accuracy, calibration, and independent eval | Reclassified as historical PR #4 milestone superseded by honest Sprint Audit ([`docs/FINAL_10_OUT_OF_10_ENGINEERING_ACCEPTANCE.md`](file:///c:/Users/avira/Desktop/NEXUS-SLM/docs/FINAL_10_OUT_OF_10_ENGINEERING_ACCEPTANCE.md)) | Added explicit supersession notice |

---

## 4. Machine-Readable Verification

All table values in [`README.md`](file:///c:/Users/avira/Desktop/NEXUS-SLM/README.md) and [`docs/RESULTS.md`](file:///c:/Users/avira/Desktop/NEXUS-SLM/docs/RESULTS.md) are deterministically synchronized and enforced by:
- [`scripts/evaluation/sync_results_from_artifacts.py`](file:///c:/Users/avira/Desktop/NEXUS-SLM/scripts/evaluation/sync_results_from_artifacts.py)
- [`scripts/evaluation/check_documentation_consistency.py`](file:///c:/Users/avira/Desktop/NEXUS-SLM/scripts/evaluation/check_documentation_consistency.py)
- [`tests/test_evaluation_consistency.py`](file:///c:/Users/avira/Desktop/NEXUS-SLM/tests/test_evaluation_consistency.py)
