# NEXUS-SLM Known Limitations & Operational Boundaries

This document provides an unvarnished analysis of the empirical limitations, boundary failure modes, and hardware constraints of NEXUS-SLM-135M-LoRA (`nexus-qlora-v1`).

---

## 1. Summary of Empirical Limitations

| Limitation Domain | Empirical Finding | Operational Impact | Mitigation Implemented |
|---|---|---|---|
| **Devanagari Hindi** | 14.3% accuracy on native script | Misclassifies native Hindi tickets | Deterministic keyword fallback + Hinglish preference |
| **Short Inputs ($\le 3$ words)** | Significant recall drop on context-sparse inputs | Higher classification uncertainty | Deterministic `short_input` human escalation |
| **Model Overconfidence** | Expected Calibration Error (ECE) = 0.588 | Raw model probabilities report ~0.97 confidence even when incorrect | Post-hoc confidence calibration + gating |
| **Small Evaluation Benchmark** | 42 locked test samples (3 per category) | Wide 95% confidence intervals on individual class F1 scores | Full test lock integrity; bootstrap evaluation |
| **Rare Class Ambiguity** | Low recall on `account_closure` vs `account_access` | Edge-case boundary confusion | Deterministic keywords force `account_closure` escalation |
| **Hardware Throughput** | ~0.50 req/sec on 12-thread CPU | Concurrency $> 2$ causes queuing latency spikes | Concurrency capped at 1-2 workers; horizontal scaling required |
| **Parameter Capacity** | 135M parameters represents an empirical ceiling | Cannot reliably handle multi-intent tickets | Architectural upgrade to Qwen2.5-0.5B recommended for future phases |

---

## 2. In-Depth Analysis

### 2.1 Native Devanagari Hindi Tokenization Penalty
- **Root Cause**: The base model tokenizer (`SmolLM-135M`) was pre-trained predominantly on English and code corpora. The vocabulary contains minimal native Devanagari subwords.
- **Consequence**: Native Hindi words are decomposed into individual UTF-8 bytes (byte fallback), inflating token sequence length by 3x–4x and diluting cross-attention representations.
- **Evidence**: On the locked benchmark, native Hindi category accuracy is 14.3%, compared to 57.1% on Hinglish and English.

### 2.2 Model Overconfidence & Confidence Calibration
- **Root Cause**: Cross-entropy fine-tuning on a small curated dataset produces highly peaked softmax distributions.
- **Consequence**: The model generates average token probabilities of 96.9% even when the predicted category is incorrect.
- **Mitigation**: NEXUS-SLM explicitly decouples raw generation probabilities from routing trust. The system applies a 0.75x conservative calibration dampener and routes any prediction with calibrated confidence $< 0.65$ to human auditors.

### 2.3 Short Ticket Context Sparsity
- **Root Cause**: Tickets like `"Payment failed"` or `"Help login"` lack grammatical structure, verbs, or details identifying the failure mechanism (gateway timeout vs wrong password vs card decline).
- **Mitigation**: The `HumanEscalationPolicy` inspects token count; tickets $\le 3$ words are flagged `short_input_context_sparsity` and routed for human assistance.

### 2.4 CPU Hardware Concurrency Limits
- **Root Cause**: Full FP32 transformer forward generation across 12 OpenMP threads utilizes nearly 100% of available CPU cores per request.
- **Consequence**: When multiple requests arrive simultaneously, OS thread scheduling creates queue contention. Under concurrency 4, P95 latency rises from 2.3s to 8.5s.
- **Deployment Rule**: A single CPU container must not be allocated more than 1-2 concurrent requests. High-traffic deployments must scale horizontally via multi-container replica pods behind an NGINX load balancer.

---

## 3. Boundary & Out-of-Scope Scenarios

NEXUS-SLM is **not designed** and **must not be used** for:
- Automated ticket resolution without human triage review.
- Automated debit/credit or refund transaction approvals.
- Open-domain multilingual chat outside the defined 14-category taxonomy.
