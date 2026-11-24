# NEXUS-SLM Model Evaluation Protocol

**Protocol Version:** 2.0-Formal  
**Effective Date:** 2026-11-24  
**Auditor Roles:** Principal ML Engineer, MLOps Lead, Independent Technical Auditor  

---

## 1. Non-Negotiable Integrity Constraints

1. **Zero Fabrication:** All reported evaluation numbers must derive strictly from programmatically executed inference over physical artifacts.
2. **Locked Benchmark Dataset:** The evaluation set `data/v2_clean_template_grouped/test_clean.jsonl` (SHA-256 `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b`, $N = 746$) must remain strictly immutable. It must never be modified, resampled, filtered, or used for gradient updates, hyperparameter selection, prompt tuning, or calibration.
3. **Single Formal Test Pass:** The formal candidate acceptance evaluation is executed against the test set exactly once. Repeated trial-and-error searching over the test partition is strictly prohibited.
4. **Engine Provenance:** Every sample output must record `inference_mode` (`neural` vs `heuristic_fallback`) and `model_version` to guarantee that heuristic fallbacks are never misrepresented as neural predictions.
5. **No Mock Weights:** Neural evaluations must execute the loaded weights in memory. If weights cannot be loaded, the run must be marked BLOCKED.

---

## 2. Benchmark Artifacts & Specifications

### 2.1 Candidate Neural Model
- **Identifier:** `qwen25_05b_clean_v2_lora`
- **Base Architecture:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Directory:** `experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter/`
- **Adapter Model SHA-256:** `9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2`
- **Adapter Config SHA-256:** `7A236E4CF7521BFC6D64C4C01CB16879C7B4AF272434C7D64B7CAE017247A17C`
- **LoRA Parameters:** $r=16, \alpha=32, \text{dropout}=0.05$, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Inference Precision:** `float32` on CPU / `float16` on GPU

### 2.2 Baseline Production Fallback
- **Identifier:** `heuristic_fallback_v2`
- **Engine:** Deterministic rule-based engine (`src/inference/rules.py` and `src/inference/policy.py`)
- **Execution Mode:** `heuristic_fallback`

---

## 3. Evaluation Metrics Suite

For both the experimental neural candidate and the production heuristic baseline, the following metrics are evaluated across all 746 test instances:

1. **Category Classification (14 classes):**
   - Overall Accuracy
   - Macro Precision, Macro Recall, Macro-F1
   - Weighted Precision, Weighted Recall, Weighted-F1
   - Per-class Precision, Recall, F1, and Support Count
   - $14 \times 14$ Confusion Matrix
   - Wilson 95% Confidence Intervals for sparse categories ($N \le 10$)
2. **Priority Classification (4 classes: low, medium, high, critical):**
   - Accuracy, Macro-F1, Weighted-F1
   - Per-class Recall and $4 \times 4$ Confusion Matrix
3. **Multilingual Classification (3 languages: english, hindi, hinglish):**
   - English Accuracy, Hindi Accuracy, Hinglish Accuracy, Overall Language Accuracy
4. **Structured JSON Output:**
   - Valid JSON Syntax Rate (%)
   - Schema Validity Rate (%)
   - Exact Field Match Rate (%)
   - Parsing Failures count
5. **Safety & Human Escalation Routing:**
   - True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
   - High-Risk Escalation Recall ($TP / (TP + FN)$)
   - High-Risk Escalation Precision ($TP / (TP + FP)$)
   - Overall Human Escalation Rate ($(TP + FP) / N$)
   - Unnecessary Escalation Rate ($FP / (TN + FP)$)
6. **Operational Latency & Systems Footprint:**
   - Cold-start load time (ms)
   - Inference Latency: P50, P90, P95, P99 (ms)
   - Mean Throughput (requests/sec)
   - Peak Inference RAM (MB)
   - GPU VRAM consumption (if available)

---

## 4. Production Promotion Gates

To achieve recommendation for production deployment, a candidate model must satisfy:
1. **Category Accuracy:** $\ge 80.0\%$
2. **Category Macro-F1:** $\ge 0.750$
3. **JSON Validity:** $100.0\%$ on formal test set
4. **High-Risk Safety Recall:** $100.0\%$ (Zero false negatives on designated high-risk acceptance tickets)
5. **Priority Macro-F1:** $\ge 0.500$
6. **Regression Invariant:** Must not exhibit regression on high-risk safety or operational stability compared to the heuristic baseline.
