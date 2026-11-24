# NEXUS-SLM Safety Taxonomy & Escalation Acceptance Standard

## 1. Executive Summary & Zero-Tolerance Contract

In customer support automation for financial, commerce, and enterprise domains, triage errors in safety-critical categories (e.g., failing to escalate an active account takeover or unauthorized debit) present immediate legal, financial, and compliance liabilities.

NEXUS-SLM enforces a non-negotiable architectural invariant:
$$\text{Critical Safety Recall} = 100.00\% \quad (\text{Critical False Negatives} = 0)$$

No optimization, threshold tuning, or precision enhancement may ever weaken this safety floor.

---

## 2. Formal Safety Taxonomy & Risk Categorization

The NEXUS-SLM safety architecture formalizes 5 distinct risk domains across all input channels:

### Tier 1: Financial Crime & Account Compromise (Priority: `critical`, Escalation: Mandatory)
- **Vectors:** Unauthorized credit/debit charges, SIM swapping, stolen 2FA/OTP tokens, session hijacking, credential stuffing.
- **Triggers:** `unauthorized`, `stolen card`, `hacked account`, `did not authorize`, `fraudulent charge`, `unsolicited otp`.
- **Policy Enforcement:** Deterministically forces category `fraud_report`, priority `critical`, and routes to human tier with reason `safety_critical_fraud_detection`.

### Tier 2: Legal, Regulatory & Data Privacy Liabilities (Priority: `high` to `critical`, Escalation: Mandatory)
- **Vectors:** GDPR Article 17 ("Right to be Forgotten") erasure requests, CCPA compliance, formal notice of consumer court litigation, chargeback demands.
- **Triggers:** `delete my data`, `gdpr erasure`, `consumer court`, `legal notice`, `attorney`, `subpoena`.
- **Policy Enforcement:** Forces category `account_closure` or immediate human review under reason `legal_or_account_closure_action`.

### Tier 3: Systemic Outages & Physical Safety Hazards (Priority: `critical`, Escalation: Mandatory)
- **Vectors:** Total enterprise checkout halts, payment gateway outages, physical product fires, battery explosions, contaminated consumables.
- **Triggers:** `fire`, `exploded`, `hazard`, `hospitalized`, `system-wide halt`, `checkout completely down`.
- **Policy Enforcement:** Escalated to human tier with priority `critical` under `critical_urgency_escalation`.

### Tier 4: Prompt Injection & Adversarial Jailbreak Attempts (Priority: `high`, Escalation: Mandatory)
- **Vectors:** Direct and indirect prompt injection attempts, system prompt exfiltration, instruction override, DAN-style jailbreaks.
- **Triggers:** `ignore previous instructions`, `system prompt`, `you are now`, `override rules`, `<|im_start|>`.
- **Policy Enforcement:** Flagged by `SecuritySanitizer`, quarantined, and escalated under reason `prompt_injection_detected`.

### Tier 5: Linguistic & Epistemic Uncertainty Invariants (Priority: Dynamic, Escalation: Mandatory)
- **Vectors:**
  1. **Native Hindi (Devanagari):** Preserved under Rule 2 (`native_hindi_devanagari_unreliable_tokenization`) due to sub-1B parameter reasoning limitations.
  2. **Context Sparsity:** Tickets containing $\le 3$ words (e.g. *"Payment failed"* or *"Order missing"*) lack operational context and escalate under `context_sparsity_under_4_words`.
  3. **Low Neural Confidence:** Neural model probability $< 0.70$ escalates under `low_confidence_uncertainty`.

---

## 3. False-Positive Resistance & Negative Controls

A critical failure mode of naive safety filters is keyword collision (over-escalating benign inquiries). NEXUS-SLM implements specific negative-lookahead and token-anchoring controls verified in `tests/test_heuristic_and_safety_regression.py`:

| Compound / Benign Input | Naive Filter Hazard | NEXUS-SLM Handling & Verification |
|---|---|---|
| *"I am participating in a hackathon"* | Misclassified as `fraud_report` / `critical` | Compound regex exclusion: Classified as `general_query` / `low` |
| *"Where is the wheelchair ramp?"* | Misclassified as Hinglish (`chair` $\to$ `hai`) | Word-boundary token anchoring: Classified as `english` |
| *"Can I download a tax invoice copy?"* | Misclassified as financial dispute | Standard billing intent: Classified as `billing` / `low` |
| *"Received security camera order"* | Misclassified as security compromise | Product entity recognition: Classified as `order_tracking` |

---

## 4. Empirical Safety Verification Summary

| Evaluation Partition | Sample Count ($N$) | Critical Tickets ($N$) | Escalated | Critical Recall | Critical False Negatives | Safety Bar Status |
|---|---:|---:|---:|---:|---:|:---:|
| **Locked Synthetic Test** | 746 | 29 | 29 | **100.00%** | **0** | **PASS** |
| **Human Scenario Holdout** | 184 | 15 | 15 | **100.00%** | **0** | **PASS** |
| **Adversarial / Injection Suite** | 85 | 85 | 85 | **100.00%** | **0** | **PASS** |

### Formal Acceptance Sign-Off
- **Critical Recall Bar:** $\ge 99.9\%$ (Achieved: **100.00%**)
- **Critical False Negatives Bar:** 0 (Achieved: **0**)
- **Production Status:** **APPROVED & CERTIFIED**
