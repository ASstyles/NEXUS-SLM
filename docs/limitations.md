# NEXUS-SLM Known Limitations & Operational Boundaries

**Document Version:** 2.0.0-audited  
**Audit Reference:** [RESULTS.md](RESULTS.md)  
**Status:** **EMPIRICALLY VERIFIED**

This document provides a rigorous, unvarnished technical analysis of the empirical limitations, boundary conditions, and hardware constraints of NEXUS-SLM under both the production heuristic fallback and experimental neural engines.

---

## 1. Summary of Empirical Limitations

| Limitation Domain | Empirical Finding | Operational Consequence | Engineering Mitigation |
|---|---|---|---|
| **CPU Generation Latency** | Autoregressive decoding takes 1.2s to 2.8s per ticket on CPU (P50: 1,508ms) | Unsuitable for sub-second real-time API SLAs under high concurrency | Primary production routing utilizes **Heuristic Fallback v2** (<0.1ms); neural deployed to GPU/vLLM |
| **Synthetic Template Bias** | 80.3% to 82.4% template redundancy across dataset splits | Model overfits to template syntactic phrasing; confuses `product_complaint` with `refund_request` | Deployed human annotation infrastructure (`data/human_eval/`) for out-of-distribution evaluation |
| **Priority Classification Bottleneck** | ~47% priority accuracy for both heuristic (47.18%) and neural (47.86%) engines | Inability to reliably separate adjacent urgency classes (`medium` vs `high`) | Priority rubric feature engineering and rule-based escalation overrides |
| **Category Sparsity (`subscription`)** | Locked test set contains only $N=2$ samples for `subscription` (0.27%) | Claims of generalizability on subscription cancellation have wide 95% CI ($[0.3424, 1.0000]$) | Explicit sparsity tagging in evaluation telemetry; targeted dataset augmentation |
| **Standalone Neural Escalation Gap** | Standalone Qwen LoRA had 29 false negatives on high-risk tickets (86.12% recall) | Risk of financial loss or security breach if operated without deterministic safety guard | Mandatory wrapping via `HumanEscalationPolicy` guaranteeing 100% safety recall |
| **Native Hindi Subwords** | Devanagari script expands into multi-byte token sequences in small BPE vocabularies | Increased sequence length and latency for native Hindi | Deterministic Devanagari detection and routing policy |

---

## 2. In-Depth Technical Analysis

### 2.1 CPU Hardware Generation Latency & Concurrency Ceiling
- **Root Cause**: The fine-tuned Qwen2.5-0.5B model contains ~494 million parameters. Autoregressive token-by-token generation across 8 CPU worker threads consumes ~1.5 seconds per sample.
- **Operational Reality**: While small language models are lightweight compared to 70B models, CPU inference cannot sustain microservice concurrency exceeding 1–2 requests per container without queuing degradation.
- **Production Standard**: High-throughput customer support pipelines must either:
  1. Use the **Heuristic Fallback v2** engine for live traffic (P50 0.05ms, throughput >9,800 req/s on CPU).
  2. Deploy the neural candidate onto specialized GPU runtime engines (vLLM, TensorRT-LLM, or Triton).

### 2.2 Synthetic Template Homogeneity & Adjacent Class Bleed
- **Root Cause**: The synthetic data generator instantiated tickets by injecting variable slots (order numbers, customer names, refund amounts) into fixed syntactic sentence frames.
- **Empirical Evidence**: Quantitative audit revealed only 947 unique templates across 4,797 training samples (80.26% redundancy).
- **Failure Mode**: When customers report broken or defective products, the model fixates on words like "broken" or "not working" and predicts `refund_request` (29 samples) or `fraud_report` (15 samples), resulting in 0% recall for `product_complaint` ($N=44$).
- **Mitigation Strategy**: The synthetic dataset cannot substitute for real customer writing. The repository now includes the `data/human_eval/` protocol to ingest and evaluate genuine customer support tickets.

### 2.3 Priority Classification Ambiguity
- **Root Cause**: The dataset labels priority (`low`, `medium`, `high`, `critical`) based on subtle subjective urgency cues rather than explicit objective criteria.
- **Empirical Evidence**: Both heuristic rules (47.18% accuracy) and the neural model (47.86% accuracy) achieve roughly equivalent, mediocre priority performance.
- **Operational Rule**: Priority is never treated as a fully automated routing criterion. Safety-critical categories (`fraud_report`) deterministically force `critical` priority regardless of model predictions.

### 2.4 The Subscription Class Sample Size Deficit
- **Root Cause**: Severe class imbalance in the locked clean test split resulted in only 2 ground-truth `subscription` samples.
- **Scientific Caveat**: The measured 100% recall for `subscription` is statistically uninformative due to sample sparsity ($N=2$, 95% Wilson confidence interval $[0.3424, 1.0000]$). In contrast, precision was 12.5% because other cancellation intents were misattributed to subscription.

---

## 3. Boundary & Out-of-Scope Operational Scenarios

NEXUS-SLM is **explicitly out-of-scope** for:
1. **Fully Autonomous Financial Transactions**: The system does not possess authority to approve payments, execute refunds, or process credit adjustments.
2. **Autonomous Account Destruction**: Permanent GDPR account deletion requests must be reviewed by compliance personnel.
3. **Open-Ended Conversational Dialogue**: NEXUS-SLM is a specialized structured classification engine, not a generative chatbot. Freeform dialogue without JSON validation is strictly disallowed.
