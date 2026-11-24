# NEXUS-SLM — Confidence Calibration & Validation Report

## 1. Executive Summary & Problem Definition

In autonomous ticket triage and routing systems, the reliability of a model's predicted confidence $p(\hat{y} \mid x)$ is as vital as its classification accuracy. 
- **Overconfident errors** lead to automated misrouting of urgent customer requests to incorrect resolution teams or improper self-service deflection.
- **Underconfident correct predictions** waste human agent capacity by triggering unnecessary escalations.

This audit evaluates the confidence properties of the NEXUS-SLM experimental candidate (`qwen25_05b_clean_v2_lora_run_20261030_190017`) and the legacy `nexus-qlora-v1` candidate against empirical outcomes across synthetic validation, synthetic test, and human holdout data.

---

## 2. Empirical Calibration Audit & Findings

### 2.1 The Semantic Mismatch in Autoregressive Confidence
In sequence-to-sequence small language models (such as Qwen2.5-0.5B Instruct), the confidence score is frequently extracted as the **mean token generation probability** of the output ChatML JSON string:

$$\bar{p}_{\text{seq}} = \frac{1}{T} \sum_{t=1}^T p(w_t \mid w_{<t}, x)$$

**Critical Finding:** $\bar{p}_{\text{seq}}$ does **not** represent a true 14-class categorical posterior $P(Y = c \mid x)$. Because the model has been instruction fine-tuned to produce syntactically valid JSON, its token generation probabilities are consistently elevated ($\sim 0.95 - 0.99$) even when the classification token itself is erroneous.

### 2.2 Calibration Metrics on Synthetic Test Split ($N=746$)

Audit of `experiments/artifacts/final_calibration_metrics.json`:

| Metric | Raw Autoregressive Confidence | Temperature Scaled ($T=1.15$) | Ideal Value |
|---|---:|---:|:---|
| **Mean Confidence** | 0.9768 (97.68%) | 0.9629 (96.29%) | $\approx$ Empirical Accuracy |
| **Empirical Category Accuracy** | 0.6917 (69.17%) | 0.6917 (69.17%) | — |
| **Overconfidence Gap** | **+28.51%** | **+27.12%** | **0.00%** |
| **Expected Calibration Error (ECE)** | **0.2851** | **0.2712** | $< 0.05$ |
| **Maximum Calibration Error (MCE)** | **0.2851** | **0.2712** | $< 0.10$ |
| **Brier Score** | 0.2933 | 0.2852 | $\to 0.00$ |

#### Reliability Diagram Breakdown (10 Equal Bins, $N=746$):
- **Bins 0.0 to 0.9:** Exactly **0 samples** (0%). The model never produced a confidence score below 0.90.
- **Bin 0.9 to 1.0:** **746 samples (100%)**, Mean Confidence: **0.9768**, Accuracy: **0.6917**, Calibration Gap: **0.2851**.

### 2.3 Validation Split Calibration Fitting ($N=44$, `data/validation/`)
Audit of `scripts/evaluation/calibrate_on_validation.py` and `experiments/artifacts/validation_calibration_study.json`:
- **Validation Sample Size:** $N = 44$ samples across 14 categories ($\approx 3.14$ samples per class).
- **Platt Scaling Logistic Fit:** $a = 0.3661, b = -2.5275$.
- **Validation ECE:** Reduced from $0.6051$ to $0.0580$ on the 44 validation samples.
- **Statistical Insufficiency:** A sample size of $N=44$ lacks the statistical power to parameterize a generalized calibration mapping across 14 multi-lingual classes. The empirical parameters suffer severe high variance under out-of-distribution human text.

---

## 3. Confidence Policy & Safety Invariants

Based on the empirical findings, NEXUS-SLM enforces the following mandatory operational policies:

### 3.1 Prohibited Practices
1. **No Raw Probability Routing:** Raw autoregressive model probabilities must NEVER be treated as Bayesian confidence or used directly for unescalated autonomous action.
2. **No Model-Confidence Safety Override:** Under no circumstances may a high confidence score ($p > 0.95$) bypass a Tier 1 deterministic safety rule (e.g. fraud detection, unauthorized debits, physical hazards, account lockout). Tier 1 deterministic checks take absolute precedence.
3. **No Uncalibrated Claims:** The API must not return raw generation logprobs labeled as calibrated probabilities. In `src/inference/engine.py` and `src/api/routes.py`, confidence is explicitly documented as heuristic generation confidence.

### 3.2 Automated Routing Thresholds (Neural Mode)
When running in neural inference mode:
- **Calibrated / High Confidence ($p \ge 0.85$ AND no safety triggers):** Automated queue routing permitted.
- **Moderate Confidence ($0.70 \le p < 0.85$):** Automated category tagging permitted; requires background asynchronous agent verification.
- **Low Confidence ($p < 0.70$ OR $p = \text{None}$):** Mandatory escalation (Rule 12 in `src/inference/policy.py`).

### 3.3 Heuristic Engine Policy
In production heuristic fallback mode (`inference_mode == "heuristic"`), the engine explicitly returns `confidence = None`. Escalation is governed deterministically by contextual risk, operational impact, and ambiguity detection, ensuring zero false-negative safety breaches and avoiding artificial score reliance.

---

## 4. Recommendations for Future Neural Retraining

1. **Dedicated Classification Head:** Move away from pure autoregressive text generation for classification. Train a dedicated linear classification head over the pooled transformer encoder representations to produce genuine categorical softmax distributions.
2. **Temperature Scaling on Held-Out Human Holdout:** Fit temperature scaling parameters $T$ via negative log-likelihood (NLL) optimization on a dedicated held-out partition of at least 500 diverse human-authored tickets.
3. **Conformal Prediction:** Implement split conformal prediction to output valid prediction sets with guaranteed coverage guarantees (e.g., $1 - \alpha = 95\%$) rather than point probability estimates.
