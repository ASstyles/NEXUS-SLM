# NEXUS-SLM Final Independent Human Evaluation Protocol (V2)

> **Governance Status:** `PENDING` (Infrastructure, blind dual-annotation packets, Cohen's Kappa calculator, deduplication/leakage detector, and adjudication tooling are implemented and validated; execution awaits two independent external human annotators and a third-party adjudicator.)

---

## 1. Separation from Development-Exposed Datasets

| Dataset | Role in Validation | Status | Usage Constraint |
|---|---|---|---|
| `data/human_eval/annotation_template.jsonl` (184 active + 16 reserved slots) | **Development-exposed** historical human audit set | `EXPOSED` | Used during Refined Policy V2 threshold tuning. **Must NOT** be counted toward or merged into the V2 independent benchmark. |
| `data/v2_clean_template_grouped/test_clean.jsonl` (N=746) | Historical locked synthetic template-grouped benchmark | `LOCKED_HISTORICAL` | Benchmark reference only; never modify labels or train on test. |
| `data/human_eval_v2/` (`sealed_final_eval`, target N >= 300; scaffold N=320 + `dev_calibration_reserved` N=80) | **Independent V2 evaluation & separate dev calibration cohort** | `PENDING_AUTHORING` | **Zero overlap** with all V1/V2 synthetic sets and the 184 dev-exposed scenarios. `sealed_final_eval` must NEVER be used for threshold tuning, prompt changes, rules, or model selection. |

---

## 2. Target Cohort Stratification (`data/human_eval_v2/allocation_plan_v2.json`)

- **Total Planned Slots:** 400 (`320` in `sealed_final_eval`, exceeding the `>= 300` minimum; `80` in `dev_calibration_reserved` for pre-evaluation threshold calibration).
- **14 Categories Covered (22–24 sealed slots each):** `account_access`, `account_closure`, `billing`, `delivery_issue`, `fraud_report`, `general_query`, `order_cancel`, `order_tracking`, `password_reset`, `payment_failure`, `product_complaint`, `refund_request`, `subscription`, `technical_issue`.
- **4 Priority Levels Covered:** `low` (70), `medium` (110), `high` (95), `critical` (45).
- **4 Language / Script Strata Covered in Governance:** `english` (100), `hindi` (Devanagari, 80), `hinglish` (Romanized Hindi-English, 90), and `mixed_language` (explicit Devanagari + Romanized code-switching, 50).
- **Mandatory Edge & Risk Challenge Strata:** `standard`, `mixed_language`, `ambiguous`, `negative_control` (safe queries containing sensitive keywords), `multi_intent`, `financial_security_risk`, `physical_safety_hazard`, and `typo_unusual_phrasing`.

---

## 3. Blind Dual-Annotation & Adjudication Protocol

Versioned rubric: [`data/human_eval_v2/annotation_guidelines_v2.0.0.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/data/human_eval_v2/annotation_guidelines_v2.0.0.json).

1. **Authoring Independence:** Scenarios must be freshly written or legally/consensually collected from independent intake streams. No LLM, neural model, or heuristic engine in this repository may generate scenario texts or pre-populate ground-truth labels.
2. **Blind Packet Generation:** Run `python scripts/data/human_eval_v2_manager.py prepare-blind-packets` to strip `target_category_hint`, `target_priority_hint`, author metadata, and model outputs before exporting `blind_annotator_a_packet.jsonl` and `blind_annotator_b_packet.jsonl`.
3. **Provenance & Blindness Verification:**
   - Every annotation must carry non-empty `annotator_id`, `annotated_at_utc`, and `guideline_version == "2.0.0"`.
   - `annotator_a.annotator_id != annotator_b.annotator_id` is enforced automatically; synthetic/model strings in reviewer IDs cause rejection.
4. **Inter-Annotator Agreement & Disagreement Adjudication:**
   - Run `python scripts/data/human_eval_v2_manager.py validate` to compute raw agreement rates and **Cohen's Kappa ($\kappa$)** for `category`, unweighted and linearly-weighted ordinal `priority`, `language`, and `requires_human`.
   - Any field-level disagreement queues into `disagreements_queue` and requires resolution by a distinct third adjudicator (`adjudicator_id not in {annotator_a, annotator_b}`) recorded in `data/human_eval_v2/adjudication_log_v2.jsonl`.
5. **Data Quality & Anti-Leakage Checks:**
   - Exact normalized duplicates (`exact_duplicate_pairs`) and near-duplicates (`token_jaccard >= 0.85` and `SequenceMatcher >= 0.85`) must be zero.
   - Cross-dataset overlap against `data/v2_clean_template_grouped/*`, `data/train|val|test/*`, `data/evaluation/high_risk_challenge_set.jsonl`, and the 184 dev-exposed `data/human_eval/annotation_template.jsonl` must be zero.

---

## 4. Live Customer Data Collection Programme (Separate Governance Gate)

Real customer tickets are **NOT** included in this repository. Before enabling live customer intake, all items in [`data/human_eval_v2/consent_and_privacy_checklist.json`](file:///c:/Users/avira/Desktop/NEXUS-SLM/data/human_eval_v2/consent_and_privacy_checklist.json) must be signed off:

1. Executed DPA / Customer Consent Notice (`DPA-01`).
2. Automated + manual PII redaction for account IDs, cards, phone numbers, and government IDs (`PII-02`).
3. Documented lawful basis under GDPR/DPDP (`LAW-03`).
4. Retention & deletion SLA (`RET-04`) and customer opt-out (`OPT-05`).

---

## 5. Reproducible Commands

```bash
# 1. Initialize or refresh V2 intake scaffold
.\.venv\Scripts\python.exe scripts\data\human_eval_v2_manager.py init-scaffold

# 2. Export blind dual-annotation packets once scenarios are authored
.\.venv\Scripts\python.exe scripts\data\human_eval_v2_manager.py prepare-blind-packets

# 3. Validate provenance, Kappa, duplicates, cross-split leakage, and support counts
.\.venv\Scripts\python.exe scripts\data\human_eval_v2_manager.py validate
```
