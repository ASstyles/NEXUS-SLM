# Systematic Error & Failure Mode Analysis: NEXUS-SLM

## 1. Executive Summary

This document provides a systematic diagnostic audit of failure modes exhibited by **NEXUS-SLM-135M-LoRA** (`nexus-qlora-v1`) across 55 audited test and edge cases.

---

## 2. Failure Mode Taxonomy

```
+-----------------------------------------------------------------------+
| TOTAL AUDITED ERROR CASES (N = 55)                                   |
+-----------------------------------------------------------------------+
  |-- [45.5%] Semantic Category Overlap (Refund vs Cancellation)
  |-- [23.6%] Short Context / Intent Ambiguity (<5 words)
  |-- [18.2%] Devanagari Byte Token Fragmentation
  |-- [ 9.1%] Rare Class Underrepresentation (<15 samples in train)
  |-- [ 3.6%] Output Truncation & JSON Bracket Cutoff
```

---

## 3. Detailed Failure Case Patterns

### Case 1: Semantic Category Overlap (Refund vs Cancellation vs Payment)
- **Prevalence:** 45.5% of errors.
- **Example Input:** `"Mera order cancel kardo aur paise account me refund bhejo"`
- **Ground Truth:** `cancellation`
- **NEXUS-SLM Prediction:** `refund_request` or `payment_issue`
- **Root Cause:** Ticket contains high-frequency signals for multiple intents simultaneously ("cancel", "paise", "refund"). Without a multi-label classification objective, the model defaults to the dominant lexical token.
- **Production Mitigation:** Implement a secondary disambiguation layer or route compound intent tickets to human triage.

### Case 2: Short Input / Context Sparsity
- **Prevalence:** 23.6% of errors.
- **Example Input:** `"Delivery issue"` / `"kab aayega?"`
- **Ground Truth:** `delivery_delay`
- **NEXUS-SLM Prediction:** `general_query`
- **Root Cause:** Insufficient syntactic structure prevents the small 135M attention mechanism from grounding the priority or specific issue type.
- **Production Mitigation:** Enforce `requires_human: true` whenever prompt word count $< 6$.

### Case 3: Devanagari Byte-Pair Fragmentation
- **Prevalence:** 18.2% of errors.
- **Example Input:** `"मेरा रिफंड अभी तक बैंक में नहीं आया"`
- **Ground Truth:** `refund_request`
- **NEXUS-SLM Prediction:** `general_query` or invalid key
- **Root Cause:** SmolLM's tokenizer allocates minimal single tokens to Hindi Devanagari script, splitting each character into 3–4 byte tokens. This expands sequence length and dilutes attention heads.
- **Production Mitigation:** Integrate a transliteration pre-processor (Devanagari $\rightarrow$ Romanized Hinglish) before feeding the prompt to the model.

### Case 4: Rare Class Underrepresentation
- **Prevalence:** 9.1% of errors.
- **Example Input:** `"Account password reset link expired, please help"`
- **Ground Truth:** `account_access`
- **NEXUS-SLM Prediction:** `technical_support`
- **Root Cause:** `account_access` had fewer than 12 training instances in `dataset_v1`, causing the model to absorb it into the broader `technical_support` cluster.
- **Production Mitigation:** Augment training distribution with targeted synthetic data for under-represented categories in `dataset_v2`.

### Case 5: Generation Truncation (JSON Syntax Cutoff)
- **Prevalence:** 3.6% of errors.
- **Root Cause:** Occasionally the generation finishes at 45 tokens before outputting the final `}`.
- **Production Mitigation:** Handled by deterministic regex auto-completion in `src/inference/parser.py`.
