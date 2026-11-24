# NEXUS-SLM Independent Validation Protocol & Scientific Separation Standard

## 1. The Separation Principle & Current Exposure Boundary

In rigorous machine learning engineering, a dataset that has been inspected, error-analyzed, or used to iteratively guide rule refinements can no longer be classified as an unexposed, virgin evaluation set.

### Current Exposure Status
- **Locked Synthetic Test Set ($N=746$):** Sealed at inception; preserved strictly without label alteration or threshold tuning. Used exclusively for comparative benchmark tracking.
- **Human-Authored Scenarios ($N=184$):** Created to address synthetic template monotony. However, during the Final Excellence Sprint and Policy V2 refinement, these 184 scenarios were inspected for error modes (e.g. contextual priority adjustments). **Consequently, this dataset is now formally classified as DEVELOPMENT-EXPOSED.**
- It serves as an out-of-distribution regression validation suite, but CANNOT serve as the final independent validation gate for production deployment.

---

## 2. Independent Validation Protocol ($N \ge 300$)

To establish indisputable, mathematically defensible proof of real-world generalization, NEXUS-SLM mandates the following Independent Validation Protocol prior to promoting any neural SLM or automated triage policy into full production autonomy.

### 2.1 Target Sample Size & Distribution ($N = 300$)
- **Minimum Sample Count:** $N = 300$ distinct tickets.
- **Source:** Live customer interactions ingested through production CRM connectors (Zendesk, Freshdesk, Email, In-App Chat) under active enterprise Data Processing Agreements (DPAs).
- **Stratification Targets:**
  - 14 customer support categories ($\ge 18$ tickets per category).
  - 3 language modalities:
    - English ($\sim 40\%$, $N=120$)
    - Romanized Hinglish ($\sim 35\%$, $N=105$)
    - Native Hindi Devanagari ($\sim 25\%$, $N=75$)
  - Adversarial & Edge Controls ($\ge 10\%$ of corpus):
    - Multi-intent tickets ($N \ge 15$).
    - Colloquial / slang phrasing ($N \ge 20$).
    - Benign inquiries with suspicious security keywords (negative injection controls, $N \ge 10$).

---

## 3. Cryptographic Sealing & Blinding Protocol

To eliminate confirmation bias and prevent data contamination:

1. **Air-Gapped Ingestion & Sanitization:**
   - Tickets are ingested by an isolated ingestion agent.
   - PII scrubbing is performed and cryptographically validated before inclusion.
2. **Blind Dual Annotation:**
   - Two external human annotators label each ticket independently.
   - Annotators have zero access to model outputs, regex rules, or system configuration.
   - Inter-rater agreement must achieve Cohen's Kappa $\kappa \ge 0.85$ across all dimensions.
3. **Cryptographic Hash Sealing:**
   - The final adjudicated JSONL file is SHA-256 hashed.
   - The SHA-256 digest is committed to version control *before* the evaluation script is executed.
4. **Single-Pass Evaluation:**
   - Evaluation scripts are executed strictly once in containerized isolation.
   - Zero post-hoc threshold tuning, prompt tweaking, or hyperparameter optimization is permitted.
   - Results are automatically archived to `experiments/results/independent_validation_cohort_<date>.json`.

---

## 4. Production Promotion Gates (Non-Negotiable Thresholds)

Any candidate model or routing policy evaluated on the Independent Validation Cohort must satisfy all of the following criteria simultaneously to qualify for production rollout:

| Metric Dimension | Required Threshold | Rationale |
|---|---|---|
| **Category Macro-F1** | $\ge 0.75$ | Strong generalization across all 14 support intents |
| **Priority Macro-F1** | $\ge 0.70$ | Accurate triage of business urgency |
| **Language Accuracy** | $\ge 95.0\%$ | Reliable cross-lingual routing |
| **Critical Safety Recall** | **$100.0\%$ (Zero FN)** | Zero tolerance for undetected fraud, account takeover, or severe safety risks |
| **High-Risk Overall Recall** | $\ge 95.0\%$ | Robust human escalation for complex/risky tickets |
| **Unnecessary Escalation Rate** | $\le 45.0\%$ | Operational viability; avoids overloading human support teams |
| **Exact 4-Field Match Rate** | $\ge 40.0\%$ | End-to-end holistic accuracy |
