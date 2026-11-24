# NEXUS-SLM Human-Written Evaluation Dataset Infrastructure

## 1. Operational Status: PENDING COLLECTION & ADJUDICATION

> [!IMPORTANT]
> **Status Notice:** Human evaluation metrics are explicitly marked as **PENDING**. Synthetic test data can demonstrate software invariants, but real-world generalization requires verified human-written customer tickets. In accordance with project integrity rules, human evaluation evidence will only be recorded once genuine, consented, and adjudicated records have been collected.

---

## 2. Dataset Objective & Scope

This directory contains the operational infrastructure, data schema, and annotation protocols for constructing a clean benchmark of 100–200 authentic customer support tickets across:
- **English:** Natural customer phrasing, slang, typos, ambiguous intent.
- **Native Hindi (Devanagari):** Real-world Hindi inquiries and formal complaint language.
- **Romanized Hinglish:** Natural code-switched Hindi-English ticket submissions.
- **Complex Scenarios:** Ambiguous multi-intent requests, subscription cancellations, financial disputes, account takeovers, and delivery tampering.

---

## 3. Privacy, Anonymization & PII Redaction Protocol

Customer privacy is strictly protected. Raw customer data is never committed to Git.

### Mandatory Redaction Rules:
1. **Direct Identifiers:** Names, usernames, physical addresses, email addresses, phone numbers, and Social Security / Aadhaar numbers are replaced with typed tokens (e.g. `<NAME>`, `<PHONE>`, `<EMAIL>`).
2. **Order & Transaction IDs:** Real enterprise order numbers and transaction hashes are replaced with standardized synthetic tokens (`<ORDER_ID>`, `<TRANSACTION_ID>`).
3. **Financial Data:** Credit card numbers, CVVs, UPI handles, and bank account numbers are scrubbed prior to export.
4. **Consent & DPA Requirement:** Every ticket must link to a valid customer consent or Data Processing Agreement (`provenance.license_or_dpa`).

---

## 4. Multi-Annotator Protocol & Adjudication

To eliminate annotator bias and ensure defensible ground truth:
1. **Independent Annotation:** Each safety-critical ticket is reviewed by at least **two independent annotators** without cross-communication.
2. **Disagreement Tracking:** If annotators disagree on category, priority, or human escalation routing, the ticket is flagged with status `DISPUTED`.
3. **Lead Adjudication:** A designated Senior Support Lead or QA Auditor reviews disputed instances, assigns final ground-truth labels, and records the reasoning in `adjudication.adjudication_notes`.
4. **Inter-Annotator Agreement:** Cohen's Kappa ($\kappa$) is computed and reported for multi-annotator subsets.

---

## 5. Artifact Reference
- **Schema Definition:** [`schema.json`](./schema.json)
- **Collection Template:** [`annotation_template.jsonl`](./annotation_template.jsonl)
- **Collection & Adjudication Guide:** [`docs/HUMAN_EVALUATION_DATASET_GUIDE.md`](../../docs/HUMAN_EVALUATION_DATASET_GUIDE.md)
