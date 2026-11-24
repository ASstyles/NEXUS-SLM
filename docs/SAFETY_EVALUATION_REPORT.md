# NEXUS-SLM Safety & Human Escalation Evaluation Report (Sprint B)

**Audit Date:** 2026-11-24  
**Focus:** Ground-Truth Circularity Audit, Contextual Escalation Policy, Precision Hardening, and False Positive Mitigation  

---

## 1. Audit of Label Circularity (SEC-SAFE-01)

An in-depth code audit was conducted comparing the synthetic dataset generation routines (`src/data/pipeline.py` and template rules) against `HumanEscalationPolicy` (`src/inference/policy.py`).

### 1.1 Findings
- **Keyword Overlap Circularity:** In synthetic dataset v1 and early v2 iterations, the ground-truth boolean label `requires_human` was assigned using keyword dictionaries (`FRAUD_KEYWORDS`, `FINANCIAL_DISPUTE_KEYWORDS`, `ACCOUNT_SECURITY_KEYWORDS`, `LEGAL_CLOSURE_KEYWORDS`) identical to those used by `HumanEscalationPolicy`.
- **False Impression of Safety:** Because the evaluation dataset was generated with the same rules that the policy used to escalate, historical benchmarks reported an artificial **100% escalation recall** (217 / 217).
- **Severe Operational Degradation:** On the 746-sample clean test benchmark:
  - **True Positives (TP):** 217
  - **False Positives (FP):** 345
  - **True Negatives (TN):** 184
  - **False Negatives (FN):** 0
  - **Escalation Precision:** $217 / (217 + 345) = 38.61\%$
  - **Overall Escalation Rate:** $(217 + 345) / 746 = 75.34\%$
- **Triage Queue Overload:** Escalating 75.3% of all customer support tickets renders an automated triage system ineffective in production, defeating the objective of autonomous first-tier resolution.

---

## 2. Context-Aware Escalation Architecture

To resolve the precision deficit without sacrificing mandatory safety protections, `HumanEscalationPolicy` was refactored into modular, contextual detection components:

### 2.1 Contextual Fraud vs. Non-Fraud Compounds
- **Vulnerability:** Unanchored matching of `"hack"` caused tickets like `"I am participating in a hack-a-thon"` to be classified as critical fraud and routed to the security incident team.
- **Remediation:** Added explicit negative compound screening (`NON_FRAUD_COMPOUND_REGEX`) covering `hackathon`, `hack-a-thon`, `hackerrank`, `life-hack`, etc.
- **Genuine Attack Recognition:** Full sentence context checking requires verbs and targets matching account compromise (e.g. `someone hacked my account`, `unauthorized debit without OTP`, `stolen card credentials`).

### 2.2 Financial Disputes vs. Routine Billing Inquiries
- **Vulnerability:** Matches on `"billed"` or `"auto-renew"` indiscriminately escalated routine subscription confirmations.
- **Remediation:** Confined financial dispute escalations to explicit double-charging (`charged twice`, `debited twice`, `do baar kat gaye`) or unfulfilled debits (`amount debited without order confirmation`, `gateway timeout par paise cut`).

### 2.3 Substring False Positive Elimination in Language Detection
- **Vulnerability:** `camera` was classified as Hinglish because it contains `mera`; `chair` was classified as Hinglish because it contains `hai`.
- **Remediation:** Replaced substring checking with strict word-boundary token matching (`\b(mera|hai|nahi|karo|paise)\b`) across curated Romanized Hindi lexicons.

---

## 3. Positive and Negative Test Case Validation

| Test Case Text | Expected Category | Expected Priority | Expected Human Escalation | Actual Result |
|---|---|---|---|---|
| `The camera has a problem with focus.` | `product_complaint` | `high` | `False` (Automated) | **PASS** |
| `Office chair arrived damaged on delivery.` | `product_complaint` | `high` | `True` (Delivery issue rule) | **PASS** |
| `I am participating in a hack-a-thon and need API credentials.` | `general_query` | `medium` | `False` (Automated) | **PASS** |
| `Someone hacked my account and made unauthorized transactions!` | `fraud_report` | `critical` | `True` (Mandatory safety) | **PASS** |
| `Cancel my subscription immediately.` | `subscription` | `medium` | `False` (Automated) | **PASS** |
| `Cancel my order #98212 placed yesterday.` | `order_cancel` | `high` | `False` (Automated) | **PASS** |
| `Could you please email me a copy of my tax invoice for last month?` | `billing` | `low` | `False` (Automated) | **PASS** |
| `My account was debited twice for order #4120!` | `payment_failure` | `high` | `True` (Financial dispute) | **PASS** |
| `Delete my account and erase all my personal data under GDPR.` | `account_closure` | `medium` | `True` (Legal / Privacy) | **PASS** |
