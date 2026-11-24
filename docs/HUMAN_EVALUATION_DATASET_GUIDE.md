# NEXUS-SLM Human Evaluation Dataset Guide & Annotation SOP

**Document ID:** SOP-DATA-HUMAN-01  
**Target:** 100–200 Consented, Anonymized, Human-Written Support Tickets  
**Status:** Operational Infrastructure Active | Evaluation Metrics PENDING  

---

## 1. Ethical Governance & Provenance

1. **Explicit Customer Consent:** Tickets must be collected solely from customer interactions where explicit consent or enterprise Data Processing Agreement (DPA) terms authorize use of anonymized interaction logs for ML evaluation.
2. **Zero Storage of Raw PII in Git:** Unredacted ticket text (`raw_text`) must remain encrypted inside an air-gapped secrets store. Only cryptographically verified, PII-scrubbed records (`redacted_text`) may be placed in benchmarking repositories.
3. **Audit Trail:** Every record contains a `provenance` block recording the intake channel, consent release identifier, anonymizer engine, and timestamp.

---

## 2. PII Redaction Standard Operating Procedure

Before any ticket enters the annotation queue, it must undergo automated PII scrubbing followed by human verification:
1. **Names & Entities:** Customer and agent names are replaced with `<CUSTOMER_NAME>` or `<AGENT_NAME>`.
2. **Contact Info:** Email addresses $\to$ `<EMAIL>`, Phone numbers $\to$ `<PHONE>`, Postal addresses $\to$ `<ADDRESS>`.
3. **Identifiers:** Order IDs $\to$ `<ORDER_ID>`, Transaction IDs $\to$ `<TX_ID>`, Tracking numbers $\to$ `<TRACKING_ID>`.
4. **Financial Credentials:** Credit card digits, bank account numbers, and UPI IDs $\to$ `<FINANCIAL_CREDENTIAL>`.
5. **Verification Checklist:** The privacy auditor must confirm that no residual identifiers remain before updating `pii_sanitization_status` to `"manually_verified"`.

---

## 3. Taxonomy & Annotation Rubric

### 3.1 Category Taxonomy (14 Discrete Classes)
- `order_tracking`: Where is my order, delivery timeline, courier location.
- `order_cancel`: Explicit request to cancel an order or scheduled package.
- `subscription`: Recurring plan management, subscription cancellation, membership fee disputes.
- `refund_request`: Demands for money return, refund status on returned items.
- `payment_failure`: Payment gateway drops, amount debited without order generation, UPI errors.
- `password_reset`: Forgotten passwords, password change links.
- `account_access`: Login problems, OTP delivery failures, two-factor authentication locks.
- `account_closure`: Account deletion requests, GDPR right-to-be-forgotten erasures.
- `billing`: Invoices, tax receipts, non-disputed charge explanations.
- `technical_issue`: Application crashes, HTTP 500 errors, website bugs.
- `product_complaint`: Defective, broken, damaged, counterfeit physical items.
- `delivery_issue`: Delayed couriers, wrong delivery address, missing parcel contents.
- `fraud_report`: Unauthorized transactions, stolen credentials, scam reports.
- `general_query`: Store hours, policy questions, general assistance.

### 3.2 Priority Rubric (4 Discrete Levels)
- `critical`: Active account takeovers, unauthorized financial deductions, legal notices, safety hazards.
- `high`: Payment failures, delivery failures, damaged items, cancellations needing immediate intervention.
- `medium`: General order tracking, password resets, standard refunds, subscription inquiries.
- `low`: Informational questions, invoices, general FAQs.

### 3.3 Human Escalation Flag (`requires_human`)
- `True`: Tickets involving safety-critical events (fraud, legal action), ambiguous multiple intents, high emotional distress, or financial disputes requiring manual authorization.
- `False`: Routine, unambiguous tickets suitable for immediate automated resolution.

---

## 4. Multi-Annotator Workflow & Adjudication

```
[Raw Intake] --> [Automated & Manual PII Scrubbing] --> [Queue Assignment]
                                                                |
                          +-------------------------------------+
                          |                                     |
                [Annotator Alpha]                     [Annotator Beta]
                          |                                     |
                          +-------------------------------------+
                                            |
                                 [Agreement Check]
                                   /           \
                           (Agreement)      (Disagreement)
                                 |                 |
                    [CONSENSUS_REACHED]     [Lead Adjudicator Review]
                                                   |
                                            [ADJUDICATED]
```

1. **Independent Evaluation:** Annotator Alpha and Annotator Beta evaluate tickets independently. Neither annotator sees the other's annotations or model predictions.
2. **Disagreement Criteria:** Any discrepancy in `category`, `priority`, or `requires_human` triggers mandatory adjudication.
3. **Lead Adjudication:** A Principal QA Auditor reviews disputed instances, documents the decision rationale in `adjudication.adjudication_notes`, and assigns the canonical ground truth.

---

## 5. Dataset Versioning & Quality Control

- **Format:** Line-delimited JSON (`.jsonl`).
- **Schema Validation:** Every record is validated against `data/human_eval/schema.json` via automated pre-commit hook.
- **Evaluation Status:** The repository treats human evaluation as **PENDING** until the target cohort of 100–200 records is fully collected, anonymized, and adjudicated.
