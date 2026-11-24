# NEXUS-SLM Human Evaluation Protocol & Annotation Governance

**Document ID:** PROTOCOL-HUMAN-EVAL-V1  
**Version:** 1.0.0  
**Effective Date:** 2026-11-24  
**Operational Status:** Operational Infrastructure Complete | Customer Ticket Intake PENDING  
**Author:** Principal QA Lead, NLP Research Engineer & Application Security Architect  

---

## 1. Executive Summary & Objective

The NEXUS-SLM platform previously relied primarily on synthetic template-generated evaluations ($N=746$). While synthetic evaluation validates structural stability, deterministic invariants, and API routing schemas, it exhibits severe benchmark limitations:
- **Lexical and Syntactic Monotony:** Template overlap reached 80.3% in training and 82.4% in testing.
- **Unrealistic Language Distribution:** Inability to capture slang, typos, multi-intent ambiguity, code-switching, and conversational nuances.
- **Circular Safety Evaluation:** Ground-truth risk labels in synthetic data were generated using the same heuristic keyword dictionaries as the triage policy, artificially inflating recall to 100%.

To establish real-world generalizability and compliance with independent model evaluation governance, this protocol defines a rigorous, dual-annotated human evaluation benchmark programme.

> [!IMPORTANT]
> **Operational Designation Notice:**
> Manually authored scenario tickets are classified as **human-authored scenario evaluation data**, strictly distinct from **collected live enterprise customer interactions**.
> The live customer collection pipeline is designated as **PENDING** until formal enterprise Data Processing Agreements (DPA) and production consent ingestion are finalized.

---

## 2. Benchmark Scope & Phrasing Diversity

The human evaluation dataset targets 100–200 independently authored tickets (current corpus: 200 tickets, with 184 adjudicated scenarios and 16 reserved live intake slots) balanced across:
1. **Linguistic Distributions:**
   - **English (US/UK/Global):** Natural customer syntax, typos, corporate jargon, terse complaints.
   - **Native Hindi (Devanagari):** Native script phrasing, regional formal complaints, consumer forum terminology.
   - **Romanized Hinglish:** Natural code-switched Hindi-English ticket submissions reflecting modern Indian conversational commerce.

2. **Categorical Coverage (14 Discrete Classes):**
   `account_access`, `account_closure`, `billing`, `delivery_issue`, `fraud_report`, `general_query`, `order_cancel`, `order_tracking`, `password_reset`, `payment_failure`, `product_complaint`, `refund_request`, `subscription`, `technical_issue`.

3. **Adversarial & Edge-Case Topologies:**
   - **Benign Contexts with Security/Fraud Lexemes (Negative Controls):** e.g., *"I am attending a hackathon next week"*, *"Sending tax invoice for GST"*, *"College cybersecurity seminar discount"*.
   - **Severe Safety Hazards & Active Exploits:** e.g., *"Someone accessed my account without permission"*, *"Card debited without OTP"*, *"Electric appliance smoking on first use"*.
   - **Multi-Intent & Compound Demands:** e.g., *"Cancel order <ORDER_ID> and refund my money immediately because delivery is late"*.
   - **Informal Orthography & Typos:** e.g., *"plzz cancel my subsription bro"*, *"mera paasword kaam nhi kr rha"*.

---

## 3. Data Governance, Privacy & Zero-PII Policy

In compliance with GDPR (Article 17/32), Indian Digital Personal Data Protection Act (DPDPA 2023), and ISO/IEC 27701:

1. **Air-Gapped Raw Storage:** Raw ticket text (`raw_text`) containing real personal identifiers is strictly quarantined in an encrypted vault (`[REDACTED_SECURE_STORAGE]`) and never committed to version control.
2. **Tokenized Redaction:** All public benchmarking files (`redacted_text`) undergo automated regex scrubbing followed by secondary human privacy audit:
   - Names $\to$ `<CUSTOMER_NAME>`, `<AGENT_NAME>`
   - Contact Info $\to$ `<EMAIL>`, `<PHONE>`, `<ADDRESS>`
   - Identifiers $\to$ `<ORDER_ID>`, `<TRACKING_ID>`, `<TX_ID>`
   - Financial Data $\to$ Credit cards, CVVs, and UPI IDs are scrubbed completely.
3. **Audit Provenance:** Every record tracks `provenance.source_channel`, `consent_confirmed`, `license_or_dpa`, and `anonymizer_agent`.
4. **Data Minimization:** No biometric, medical, political, or protected class information is collected.

---

## 4. Multi-Annotator Protocol & Inter-Rater Reliability (IRR)

To eliminate individual annotator bias and produce high-confidence ground truth:

```
[Raw Scenario / Ingested Ticket] 
               │
      [PII Sanitization & Audit]
               │
     ┌─────────┴─────────┐
     ▼                   ▼
[Annotator Alpha]   [Annotator Beta / Gamma]
     │                   │
     └─────────┬─────────┘
               ▼
       [Agreement Check]
         /           \
  (Unanimous)      (Discrepancy)
      │                  │
[CONSENSUS_REACHED] [Lead Adjudication: Lead QA / Principal ML]
                         │
                   [ADJUDICATED]
```

### 4.1 Independent Double Review
- Every safety-critical, ambiguous, or multi-intent ticket is reviewed by **two independent annotators** without cross-communication.
- Annotators assign:
  1. `category` (14 classes)
  2. `priority` (`low`, `medium`, `high`, `critical`)
  3. `requires_human` (`true`, `false`)
  4. `confidence_level` (`low`, `medium`, `high`)
  5. `notes` (qualitative justification)

### 4.2 Inter-Annotator Agreement Metrics
Inter-rater agreement is computed across multi-annotated records using Cohen's Kappa ($\kappa$):
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where $p_o$ is observed agreement and $p_e$ is hypothetical chance agreement.
- Target $\kappa_{\text{category}} \ge 0.85$ (Near-perfect agreement)
- Target $\kappa_{\text{priority}} \ge 0.75$ (Substantial agreement)
- Target $\kappa_{\text{escalation}} \ge 0.90$ (High safety alignment)

### 4.3 Adjudication Mechanism
When annotators diverge:
1. The record is flagged with status `DISPUTED` in the queue.
2. A Lead Adjudicator (Senior QA Lead or Principal MLE) reviews the discordant notes.
3. The Adjudicator assigns final canonical ground truth (`final_category`, `final_priority`, `final_requires_human`) and documents the operational rationale in `adjudication_notes`.
4. The ticket status updates to `ADJUDICATED`.

---

## 5. Development vs. Sealed Evaluation Partition Strategy

To prevent data contamination and subtle test-set overfitting:
1. **Sealed Holdout Partition:** The human evaluation dataset (`data/human_eval/annotation_template.jsonl`) is strictly sealed. It is forbidden to use these records for gradient updates, prompt tuning, heuristic regex mining, or threshold calibration.
2. **Development Partition:** If specific failure modes from human testing require heuristic or neural remediation, synthetic or separate non-holdout development tickets must be created to tune the system.
3. **Independent Hashes:** Normalized SHA-256 hashes are maintained to ensure zero cross-partition leakage between synthetic training sets and human evaluation sets.

---

## 6. Sample Size Limitations & Statistical Power

- The current benchmark consists of 164 records ($N=154$ adjudicated, $N=10$ reserved pending intake).
- **Statistical Uncertainty:** With $N \approx 150$, a 95% Clopper-Pearson confidence interval for an observed category accuracy of 75% spans $[67.3\%, 81.7\%]$.
- **Support per Class:** Across 14 categories, mean support is approximately 11 tickets per category. Statistical claims of per-category significance must be reported with explicit sample counts and standard errors.
- **Reporting Rule:** Never report overall accuracy without reporting per-class support and adjacent priority confusion matrices.
