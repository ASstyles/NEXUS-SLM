# NEXUS-SLM Human Evaluation Benchmark Infrastructure

## 1. Operational Overview & Corpus Status

| Metric | Current Corpus Value |
|---|---|
| **Total Ingested Records** | 200 |
| **Adjudicated / Consensus Records** | 184 |
| **Pending Live Intake Slots** | 16 |
| **Dual-Annotated Records** | 184 |
| **Languages Represented** | English, Hindi (Devanagari), Romanized Hinglish |
| **Categories Represented** | All 14 discrete categories |
| **Priorities Represented** | `low`, `medium`, `high`, `critical` |
| **Operational Status** | Operational Infrastructure Active | Customer Intake PENDING |

> [!IMPORTANT]
> **Operational Data Classification:**
> The current 184 adjudicated tickets are **manually authored realistic scenario evaluation records** constructed to test out-of-distribution language, mixed scripts, typos, and negative control edge cases.
> They are strictly distinguished from **live collected customer conversations**, which remain designated as **PENDING** customer consent and live production intake.

---

## 2. Directory Artifacts

- **`annotation_schema.json` / `schema.json`:** JSON Schema (Draft 2020-12) defining rigorous validation rules for all ticket records, annotator blocks, provenance, and adjudication states.
- **`annotation_template.jsonl`:** The canonical line-delimited JSON benchmark holding 164 human-authored and intake scenario records.
- **`../../docs/HUMAN_EVALUATION_PROTOCOL.md`:** Comprehensive protocol detailing sampling, double-blind annotation, Inter-Rater Reliability (IRR), dispute adjudication, and zero-PII compliance.
- **`../../docs/HUMAN_EVALUATION_RESULTS.md`:** Empirical evaluation report documenting model performance and baseline metrics on this holdout cohort.

---

## 3. Privacy, Anonymization & Security Standards

In accordance with strict ethical guidelines, no raw customer PII is ever committed to the repository:
1. All records must have `pii_sanitization_status` set to `"manually_verified"`.
2. Direct identifiers (names, emails, phone numbers, addresses, account numbers, card numbers) are scrubbed and replaced with typed tokens (`<CUSTOMER_NAME>`, `<ORDER_ID>`, `<EMAIL>`, etc.).
3. Raw ticket text remains secured in air-gapped secret vaults represented as `[REDACTED_SECURE_STORAGE]`.

---

## 4. Multi-Annotator Workflow & Adjudication Protocol

1. **Independent Dual Review:** Each ticket is evaluated by at least two independent annotators (`annotator_alpha`, `annotator_beta`, or `annotator_gamma`).
2. **Consensus (`CONSENSUS_REACHED`):** When both annotators agree on category, priority, and human escalation requirement.
3. **Dispute Adjudication (`ADJUDICATED`):** When annotators differ (e.g. `billing` vs `subscription`), a Lead Adjudicator reviews the disagreement notes and assigns the authoritative ground truth with documented reasoning.
4. **Inter-Annotator Agreement:** Cohen's Kappa ($\kappa$) is tracked across category, priority, and escalation decisions.

---

## 5. Schema Validation

To validate that all records conform to the schema:
```bash
python -c "
import json, jsonschema
with open('data/human_eval/annotation_schema.json', 'r', encoding='utf-8') as sf:
    schema = json.load(sf)
with open('data/human_eval/annotation_template.jsonl', 'r', encoding='utf-8') as df:
    for idx, line in enumerate(df):
        if line.strip():
            jsonschema.validate(instance=json.loads(line), schema=schema)
print('All human eval records verified successfully.')
"
```
