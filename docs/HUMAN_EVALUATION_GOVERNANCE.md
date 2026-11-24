# NEXUS-SLM Human Evaluation Governance & Inter-Rater Reliability Standard

## 1. Governance Purpose & Scope
This governance standard establishes rigorous scientific and operational protocols for human-authored ticket evaluation within NEXUS-SLM. It enforces independence, data privacy, dual-annotator consensus, and statistical validation across all human benchmark sets.

---

## 2. Dataset Architecture & Current Corpus Inventory

The active benchmark repository (`data/human_eval/annotation_template.jsonl`) contains $N=200$ structured records:
- **Adjudicated Scenarios ($N=184$):** Manually authored realistic customer interaction scenarios spanning edge cases, colloquial phrasing, code-switching, and negative controls.
- **Pending Production Intake ($N=16$):** Reserved slots for live customer ticket ingestion upon execution of production Data Processing Agreements (DPAs).

### Partition Distributions ($N=184$ Adjudicated)

#### Category Distribution (14 Classes)
| Category | Support | Proportion | Category | Support | Proportion |
|---|---|---|---|---|---|
| `account_access` | 13 | 7.07% | `order_tracking` | 12 | 6.52% |
| `account_closure` | 11 | 5.98% | `password_reset` | 12 | 6.52% |
| `billing` | 14 | 7.61% | `payment_failure` | 12 | 6.52% |
| `delivery_issue` | 13 | 7.07% | `product_complaint`| 12 | 6.52% |
| `fraud_report` | 17 | 9.24% | `refund_request` | 13 | 7.07% |
| `general_query` | 17 | 9.24% | `subscription` | 15 | 8.15% |
| `order_cancel` | 12 | 6.52% | `technical_issue` | 11 | 5.98% |

#### Priority Distribution (4 Classes)
- **Critical:** 15 (8.15%)
- **High:** 52 (28.26%)
- **Medium:** 69 (37.50%)
- **Low:** 48 (26.09%)

#### Linguistic Distribution (3 Modalities)
- **English:** 80 (43.48%)
- **Romanized Hinglish:** 58 (31.52%)
- **Native Hindi (Devanagari):** 46 (25.00%)

#### Human Escalation Distribution
- `requires_human = False`: 121 (65.76%)
- `requires_human = True`: 63 (34.24%)

---

## 3. Dual Annotation & Inter-Rater Reliability (IRR) Protocol

Every ticket is evaluated independently by two trained annotators (`annotator_alpha` paired with `annotator_beta` or `annotator_gamma`) working in blind isolation without access to model predictions or peer annotations.

Inter-rater reliability is quantified using Cohen's Kappa ($\kappa$):

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

### Measured Inter-Annotator Agreement ($N=184$)
| Dimension | Observed Agreement ($p_o$) | Cohen's Kappa ($\kappa$) | Landis & Koch Interpretation |
|---|---|---|---|
| **Category (14 classes)** | 96.74% | **0.9648** | Almost Perfect Agreement ($\kappa > 0.81$) |
| **Priority (4 classes)** | 99.46% | **0.9923** | Almost Perfect Agreement ($\kappa > 0.81$) |
| **Human Escalation (`requires_human`)** | 100.00% | **1.0000** | Perfect Agreement ($\kappa = 1.00$) |

---

## 4. Adjudication Guidelines & Dispute Resolution

When annotator disagreement occurs, the record is flagged with status `ADJUDICATED` and routed to the Lead QA Auditor / Senior Research Engineer for binding resolution. Exactly 6 edge-case disputes were adjudicated:

1. **HUMAN-SEC005:**
   - *Conflict:* A1: `fraud_report` / `high` vs A2: `account_access` / `medium`.
   - *Text:* *"Received OTP on my mobile phone for a login attempt that I did not initiate. Am I hacked?"*
   - *Adjudication:* `fraud_report` / `high`. Unsolicited 2FA authentication alerts represent an active external attack vector.
2. **HUMAN-CAN005:**
   - *Conflict:* A1: `order_cancel` / `medium` vs A2: `refund_request` / `medium`.
   - *Text:* *"Accidentally ordered two units of the mixer grinder 5 minutes ago, cancel one and give my money back."*
   - *Adjudication:* `order_cancel` / `medium`. The primary actionable operational intent is stopping warehouse fulfillment before dispatch.
3. **HUMAN-DEL002:**
   - *Conflict:* A1: `delivery_issue` / `high` vs A2: `product_complaint` / `high`.
   - *Text:* *"Delivery executive dropped package from balcony gate, box smashed open."*
   - *Adjudication:* `delivery_issue` / `high`. Last-mile courier transit mishandling was the direct root cause.
4. **HUMAN-PAY007:**
   - *Conflict:* A1: `payment_failure` / `low` vs A2: `general_query` / `low`.
   - *Text:* *"Does your platform accept RuPay credit cards for international orders?"*
   - *Adjudication:* `payment_failure` / `low` (informational payment routing). Resolves to routing queue for payment methods.
5. **HUMAN-PWD005:**
   - *Conflict:* A1: `password_reset` / `medium` vs A2: `technical_issue` / `medium`.
   - *Text:* *"Password reset link expires within 10 seconds, impossible to click in time."*
   - *Adjudication:* `password_reset` / `medium`. Direct functional obstruction to credential recovery.
6. **HUMAN-ACC002:**
   - *Conflict:* A1: `account_access` / `high` vs A2: `fraud_report` / `high`.
   - *Text:* *"Account locked after traveling abroad for vacation. Need access urgently."*
   - *Adjudication:* `account_access` / `high`. Geo-velocity lock triggered on legitimate user session; requires identity restoration.

---

## 5. PII Sanitisation & Security Isolation
All ticket records undergo automated scrubbing followed by mandatory manual verification:
1. Automated scrub: Regex and NER masks for credit cards, emails, phone numbers, and addresses.
2. Human audit: Verified status `manually_verified` recorded in `pii_sanitization_status`.
3. Raw customer payload: Stored in encrypted, access-restricted cold storage (`[REDACTED_SECURE_STORAGE]`).
