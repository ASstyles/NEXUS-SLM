# NEXUS-SLM — Dataset V3 Specification & Partitioning Protocol

## 1. Scope & Objective

To eliminate synthetic template redundancy, lexical leakage, and artificial score inflation, this protocol defines the mandatory architecture and governance for future **Dataset V3** production releases.

Dataset V3 serves as the next-generation multilingual customer ticket benchmark for NEXUS-SLM, enforcing strict mathematical guarantees against data contamination and memorization shortcuts.

---

## 2. Mathematical Partitioning & Zero-Leakage Architecture

### 2.1 Normalized Template Clustering
Before any train/val/test assignment, every candidate ticket $x_i$ must be transformed into a canonical syntactic skeleton $\Phi(x_i)$:
1. **Entity Masking:** All named entities, order IDs, tracking codes, currency amounts, dates, and email addresses are replaced with standard token masks (`<ORDER_ID>`, `<AMOUNT>`, `<EMAIL>`, `<NUM>`).
2. **Orthographic Normalization:** Case-folding, punctuation strip, and whitespace collapse.
3. **Phonetic & Stemmed Normalization:** For Hinglish and Hindi, apply transliteration canonicalization to map phonetic spelling variations to uniform root keys.

### 2.2 Hash-Based Deterministic Partitioning
Every unique canonical template $T_k \in \mathcal{T}$ is assigned to a single split using cryptographic hash partitioning:

$$\text{hash}(T_k) = \text{SHA256}(T_k) \pmod{100}$$

- **Train Split (70%):** $\text{hash}(T_k) \in [0, 69]$
- **Validation Split (15%):** $\text{hash}(T_k) \in [70, 84]$
- **Test Split (15%):** $\text{hash}(T_k) \in [85, 99]$

**Invariant 1 (Cluster Disjointness):**
$$\mathcal{T}_{\text{train}} \cap \mathcal{T}_{\text{val}} = \emptyset, \quad \mathcal{T}_{\text{train}} \cap \mathcal{T}_{\text{test}} = \emptyset, \quad \mathcal{T}_{\text{val}} \cap \mathcal{T}_{\text{test}} = \emptyset$$

### 2.3 Cross-Split Levenshtein Distance Constraint
To prevent near-identical paraphrases from bypassing template normalization:
- For every template $T_{\text{test}} \in \mathcal{T}_{\text{test}}$ and $T_{\text{train}} \in \mathcal{T}_{\text{train}}$, the normalized Levenshtein distance must satisfy:
$$\text{Dist}_{\text{norm}}(T_{\text{test}}, T_{\text{train}}) \ge 0.35$$
- Any candidate test template with normalized distance $< 0.35$ to any training template is automatically quarantined or relocated to the training split.

---

## 3. Linguistic Diversity & Data Generation Strategies

Dataset V3 rejects single-template slot substitution in favor of 5 diverse authoring vectors:

1. **Human-Authored Real Scenarios (Target: 30% of total corpus):**
   - Independently collected via the human evaluation protocol across enterprise domains.
   - Dual-annotated with documented inter-rater reliability (Cohen's $\kappa \ge 0.85$).
2. **Back-Translation & Paraphrase Crowdsourcing:**
   - Source English/Hindi text back-translated through multiple intermediate languages (German, Japanese, Tamil) to inject syntactic variety.
3. **Code-Mixed Hinglish Injection:**
   - Incorporating realistic Romanized Hindi syntax, colloquialisms, phonetics, and grammatical switches.
4. **Contrastive & Adversarial Pairs (Minimum 10% of Test):**
   - *Negative Suspicion Pairs:* Benign tickets containing security keywords (e.g., *"I attended a cybersecurity webinar"* $\to$ `general_query` / `low`).
   - *Polite Severe Incidents:* Calm phrasing describing catastrophic failures (e.g., *"Thank you for your service, but my account balance was emptied without notice"* $\to$ `fraud_report` / `critical`).
   - *Prompt Injections:* Benign support requests combined with instruction overrides.
5. **Class Balance Requirements:**
   - Equal representation across all 14 categories.
   - Multi-tier priority distribution: Low (25%), Medium (35%), High (25%), Critical (15%).

---

## 4. Privacy, Governance, and DPA Compliance

1. **PII Sanitization Level 3:**
   - All text processed through the multi-stage regex + entity-scrubber pipeline before persistence.
   - Zero real names, telephone numbers, credit cards, CVVs, or street addresses permitted in repository commits.
2. **Consent & Provenance Metadata:**
   - Every record in Dataset V3 must contain immutable provenance metadata:
     ```json
     {
       "provenance": {
         "source_channel": "support_email | in_app_chat | human_scenario",
         "consent_confirmed": true,
         "license_or_dpa": "DPA-2026-NEXUS-XX",
         "anonymizer_agent": "pii_scrubber_v3_manual",
         "collection_batch": "v3_q1_2027"
       }
     }
     ```
3. **Dataset Lock & Immutability:**
   - Once finalized, Dataset V3 files will be locked with SHA-256 integrity checksums stored in `data/metadata/dataset_v3_lock.json`.
