# NEXUS-SLM — Multilingual Intelligence Acceptance Report (English, Hindi, Hinglish)

> **Artifacts:** [multilingual_and_safety_audit_v3.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/multilingual_and_safety_audit_v3.json) | [eval_synthetic_test_v3_hardened.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/results/eval_synthetic_test_v3_hardened.json) | [eval_human_holdout_v3_hardened.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/results/eval_human_holdout_v3_hardened.json) | [rules.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/src/inference/rules.py) | [policy.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/src/inference/policy.py)

---

## 1. Multilingual Remediation Summary (Hardened Policy V3 + Neural V2)

Historical inspection identified three concrete defects in V2 heuristics and V1 Qwen LoRA:
1. **Python Unicode `\b` Boundary Bug on Devanagari Combining Marks:** Words ending in vowel matras or anusvara (e.g., `धुआं`, `डिलीवरी`, `कहाँ`, `टूटा`, `इनवॉइस`) failed to match `\b...\b` boundaries because Python `re` excludes `Mn`/`Mc` combining marks from `\w`. Fixed via Unicode-safe anchor pair `(?<![\w\u0900-\u097F])...(?![\w\u0900-\u097F])`.
2. **Hinglish Lexical Sparsity:** Locked test Hinglish language identification stood at only `56.52%` in `refined_policy_v2`. Expanding unambiguous Romanized tokens (`kshatigrast`, `sthiti`, `prapt`, `turant`, `kripya`, `tooti`, `gayab`, `bhul`, `rha`, `hua`, `krna`, etc.) improved `test_clean` Hinglish ID accuracy from `56.52% -> 67.08%` (`dev_exposed_v1` `93.10%`) while preserving `0` false positives on English substrings (`camera`, `chair`).
3. **Hindi Content Classification & Unmasked Verification:** Native Devanagari tickets are still safely escalated (`requires_human=True`) by policy, AND rules + `exp_v2_5` now classify underlying Hindi intent (`category` and `priority`) accurately prior to escalation.

---

## 2. Locked Benchmark Progression (Refined V2 -> Hardened V3)

| Dataset Split | Engine / Policy | Cat Acc | Cat Macro-F1 | Prio Acc | Prio Macro-F1 | Lang Acc | Hinglish ID | High-Risk Recall (FN) | Crit FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`test_clean.jsonl` (`N=746`)** | Heuristic `refined_policy_v2` | 57.51% | 0.6516 | 36.46% | 0.3827 | 81.23% | 56.52% | 84.69% (32) | 0 |
| **`test_clean.jsonl` (`N=746`)** | **Heuristic `hardened_policy_v3`** | **69.57%** | **0.7365** | **39.14%** | **0.4125** | **85.79%** | **67.08%** | **90.43% (20)** | **0** |
| **`test_clean.jsonl` (`N=746`)** | True-Base Qwen LoRA (Raw) | 53.89% | 0.5375 | 44.24% | 0.2491 | 84.58% | — | 100.0% (0, policy) | 0 |
| **`test_clean.jsonl` (`N=746`)** | `exp_v2_5` Hybrid Neural (Exp) | **89.14%** | **0.8558** | 48.66% | 0.4290 | **96.78%** | 92.55% ID / 85.40% Cat | 88.52% (24, policy) | 0 |
| `human_eval_v1` (`N=184`, dev-exp.) | Heuristic `refined_policy_v2` | 60.33% | 0.6278 | 59.78% | 0.6519 | 97.28% | — | — | 0 |
| `human_eval_v1` (`N=184`, dev-exp.) | **Heuristic `hardened_policy_v3`** | **72.28%** | **0.7419** | **64.13%** | **0.6875** | **97.83%** | **93.10%** | **87.30% (8)** | **0** |

---

## 3. Targeted 48-Case Sub-Regime Multilingual Suite (`run_multilingual_and_safety_audit.py`)

This regression suite (`16` English + `16` Devanagari Hindi + `16` Romanized Hinglish; completely separate from `test_clean` and sealed `human_eval_v2`) verifies formal, informal, shorthand/typos, transliterated code-switching, multi-intent, safety-critical, and negative-control messages:

### 3.1 Production Engine (`hardened_policy_v3` Heuristic)

| Language | Cases | Category Acc (95% Wilson CI) | Priority Acc | Lang ID Acc | Escalation Agreement | Single Observed Category Miss(es) |
|---|---:|---:|---:|---:|---:|---|
| **English** | 16 | **93.75%** (`[0.7167, 0.9889]`) | 93.75% | **100.0%** | 87.50% | `ml_en_09`: predicted `general_query` for sparking heater cord, yet **policy caught `physical_safety_hazard_detected` -> `critical` + `requires_human=True`**. |
| **Hindi (Devanagari)** | 16 | **100.0%** (`[0.8064, 1.0000]`) | **100.0%** | **100.0%** | **100.0%** | Zero category/priority/language errors; 14 routed via `native_hindi_devanagari_unreliable_tokenization`, 2 safety-critical caught upstream (`fraud` + `smoke`). |
| **Hinglish (Romanized)** | 16 | **93.75%** (`[0.7167, 0.9889]`) | 93.75% | **100.0%** | 93.75% | `ml_hg_16`: intentionally ambiguous *"cancel karu ya refund mangoo"* predicted `refund_request` vs `order_cancel`, immediately escalated via `ambiguous_ticket_requires_human_clarification`. |

### 3.2 Experimental Neural Candidate (`exp_v2_5` Hybrid)

| Language | Cases | Category Acc (95% Wilson CI) | Priority Acc | Lang ID Acc | Escalation Agreement |
|---|---:|---:|---:|---:|---:|
| **English** | 16 | **100.0%** (`[0.8064, 1.0000]`) | 93.75% | **100.0%** | 87.50% |
| **Hindi (Devanagari)** | 16 | **100.0%** (`[0.8064, 1.0000]`) | 93.75% | **100.0%** | 100.0% |
| **Hinglish (Romanized)** | 16 | **93.75%** (`[0.7167, 0.9889]`) | 75.00% | **100.0%** | 87.50% |

---

## 4. Residual Multilingual Limitations

1. **Hinglish Lexical Recall on Unseen Synthetic Variants (`test_clean` `67.08%`):** Pure rule matching cannot capture arbitrary out-of-vocabulary Romanized spelling variations; Hybrid Neural `exp_v2_5` reaches `92.55%` Hinglish ID recall and `85.40%` Hinglish category accuracy once promoted.
2. **Conservative Devanagari Escalation Policy:** Until independent blind human evaluation V2 (`data/human_eval_v2/`, `N >= 300`, currently **`PENDING`**) validates Hindi routing at zero critical FN, all native Devanagari inputs continue routing safely to human review.
3. **Code-Switched Hinglish + Devanagari Boundary:** Mixed-script tickets (e.g., `"भाई OTP नहीं आ रहा"`) route as `hindi` via Devanagari precedence and escalate safely.
