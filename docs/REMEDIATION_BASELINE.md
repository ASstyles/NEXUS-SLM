# NEXUS-SLM Remediation Baseline Audit

**Audit Date:** 2026-11-24  
**Auditor Roles:** Principal ML Engineer, Senior Backend Architect, MLOps Engineer, AppSec Engineer, QA Lead  
**Workspace:** `C:\Users\avira\Desktop\NEXUS-SLM`  
**Git Branch:** `sprint/clean-data-model-improvement`  
**Commit:** `2d32786 docs: finalize model acceptance audit, backend verification, and production gate decision`  

---

## 1. Baseline Test Suite Reproduction

Before any code changes, the existing test suite was executed to independently establish reproduction metrics.

- **Exact Command:** `.venv\Scripts\pytest -q`
- **Python Version:** `Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr 5 2023)`
- **Core Dependencies:** `torch==2.14.1+cpu`, `transformers==4.57.6`, `peft==0.21.2`, `fastapi==0.143.0`, `pydantic==2.14.1`, `pytest==9.1.1`, `scikit-learn==1.7.2`, `starlette==1.7.0`, `anyio==4.15.1`, `httpx==0.28.1`
- **Execution Results:**
  - **Passed:** 176
  - **Failed:** 0
  - **Skipped:** 0
  - **Blocked:** 0
  - **Warnings:** 10 (Starlette deprecation warnings regarding HTTP_422 and testclient)
  - **Runtime:** 41.93s
- **Mock vs. Production Behavior Analysis:**
  - Total test definitions: 176
  - Tests using mocks or monkeypatching: 12 (in `test_kaggle_automation_regression.py`, `test_qwen_quantized_candidate.py`, `test_qwen_gguf_candidate.py`, `test_safeguards_audit.py`, `test_backend_and_intelligence.py`)
  - Tests executing production code paths: 164
  - **Critical Finding on Inference:** While 164 tests exercise production code paths, `/predict` integration tests ran against the **deterministic heuristic fallback engine**, because `nexus-qlora-v1` had missing adapter weights and fell back to rules. Therefore, historical passing test results cannot be cited as proof of neural model inference.

---

## 2. Model & Dataset Artifact Verification

### 2.1 Qwen2.5-0.5B LoRA Adapter Artifacts
- **Directory:** `experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter/`
- **Git Tracking Status:** Excluded from Git via `.gitignore` (`experiments/artifacts/training_runs/`), but **physically present on local disk**.
- **Cryptographic Hashes Verified:**
  - `adapter_model.safetensors`:
    - Measured SHA-256: `9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2`
    - Reported SHA-256: `9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2` (MATCH)
    - Size: 35,237,104 bytes
  - `adapter_config.json`:
    - Measured SHA-256: `7A236E4CF7521BFC6D64C4C01CB16879C7B4AF272434C7D64B7CAE017247A17C`
    - Reported SHA-256: `7A236E4CF7521BFC6D64C4C01CB16879C7B4AF272434C7D64B7CAE017247A17C` (MATCH)
    - Size: 1,159 bytes
  - Base Model ID: `Qwen/Qwen2.5-0.5B-Instruct`
  - Base Model Weights: Physically present in `models/scratch_merged_qwen/model.safetensors` (1,976,163,472 bytes).
  - LoRA Load Test: Verified in `.venv` using `PeftModel.from_pretrained(base_model, adapter_path)` — loaded cleanly in 4.2 seconds on CPU.

### 2.2 Clean Test Dataset
- **File:** `data/v2_clean_template_grouped/test_clean.jsonl`
- **Measured SHA-256:** `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b`
- **Reported SHA-256:** `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b` (MATCH)
- **Sample Count:** 746 records (MATCH)

---

## 3. Comprehensive Baseline Condition Matrix

| Component | Current condition | Evidence | Severity | Required fix |
|---|---|---|---|---|
| **Security: Admin Secret** | Insecure hardcoded default secret (`nexus-admin-secret-key-change-in-prod`) configured as default fallback in `src/utils/config.py`. | `src/utils/config.py:66` | **CRITICAL** | Remove default fallback for production; enforce strong random key requirement and fail startup in production if secret is missing or insecure default. |
| **Security: Auth Bypass in Tests** | `verify_admin_access` in `api/auth.py` checks `os.getenv("PYTEST_CURRENT_TEST")` or `cfg.is_test()` to allow unauthenticated pass-through. | `api/auth.py:25` | **HIGH** | Remove `PYTEST_CURRENT_TEST` bypass from production code. Use standard FastAPI dependency overrides in test fixtures. |
| **Security: Reverse Proxy Header Spoofing** | `RateLimitingMiddleware` blindly reads `X-Forwarded-For` from client headers without validating against configured trusted proxies. | `api/middleware.py:81-87` | **HIGH** | Enforce a trusted proxy policy (`TRUSTED_PROXIES`). Only trust forwarded client IP when the direct client socket is in the trusted proxy set. |
| **Security: Environment Template Discrepancies** | `.env.example` contains obsolete variable names (`ENV`, `API_PORT`, `API_HOST`, `RATE_LIMIT_PER_MINUTE`) while code reads `NEXUS_ENV`, `NEXUS_PORT`, `NEXUS_HOST`, `NEXUS_ADMIN_KEY`, etc. | `.env.example:8-17` vs `src/utils/config.py:49-86` | **MEDIUM** | Harmonize `.env.example` with all configuration parameters defined in `AppConfig`. Provide clear development and production example templates. |
| **ML Inference: Candidate Evaluation Status** | The clean V2 Qwen2.5-0.5B LoRA candidate had only reported 449-sample validation metrics (79.96% acc, 0.8103 Macro-F1). The 746-sample clean test benchmark was not independently executed. | `experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_results.json` | **CRITICAL** | Run formal evaluation of the actual Qwen LoRA candidate against all 746 test records exactly once. Measure complete category, priority, language, JSON validity, safety, and latency metrics. |
| **Safety: Escalation Precision & Rate** | Escalation policy yields ~75.3% escalation rate on synthetic clean test with only 38.6% precision (345 false positives), overwhelming human triage queues. | `docs/CLEAN_DATA_GPU_TRAINING_RESULTS.md` | **HIGH** | Refactor escalation logic with token-aware contextual checks; separate high-risk mandatory escalations from routine keyword matches. |
| **Safety: Evaluation Circularity** | Ground-truth `requires_human` labels in synthetic data were generated using overlapping keyword families as `HumanEscalationPolicy`. | `src/inference/policy.py:10-50` vs synthetic generation logic | **HIGH** | Audit and document keyword overlap circularity. Establish an independent human-review evaluation protocol and high-risk evaluation subset. |
| **Heuristic Fallback: Substring Matching Bugs** | `get_fallback_prediction` uses unanchored substring matching: `"mera"` in `"camera"`, `"hai"` in `"chair"` triggers Hinglish; `"hack"` in `"hack-a-thon"` triggers fraud. | `api/main.py:284, 300` | **HIGH** | Implement regex token boundary matching (`\b`), Unicode-aware script analysis, and explicit contextual rules for compounds. |
| **Heuristic Fallback: Category Precedence** | Confusion between subscription cancellation, order cancellation, and billing; subscription keywords override cancellation intents arbitrarily. | `api/main.py:293-335` | **MEDIUM** | Implement deterministic category precedence hierarchy with explicit intent discrimination (e.g. `cancel subscription` -> `subscription`, `cancel order` -> `order_cancel`). |
| **Backend: Rate Limiting Scalability** | `SlidingWindowRateLimiter` is in-memory process-local only; cannot enforce distributed limits across multiple API worker replicas. | `api/middleware.py:33-60` | **HIGH** | Implement Redis-backed atomic sliding-window rate limiting with in-memory fallback for single-node local development. |
| **Backend: Inference Concurrency & Async Loop** | Inference calls in `/predict` run synchronously within the async route handler, risking event loop blocking under heavy load. | `api/main.py:370` | **MEDIUM** | Offload synchronous model inference to a bounded thread pool (`anyio.to_thread.run_sync` or `asyncio.to_thread`). |
| **Confidence & Calibration** | Confidence was computed as mean generated-token softmax probability, conflating language fluency with classification certainty. Fallback returned `None`. | `src/inference/engine.py:32-49` | **HIGH** | Implement rigorous confidence semantics: multiclass temperature scaling on classification logits where available, Brier score, ECE, and preserve explicit uncalibrated/unavailable confidence states. |
| **Dataset: Quality & Human Data** | All training, validation, and test data are synthetic template-generated. Overlap analysis shows order-number variations inflating sample counts; test set has only 2 subscription samples. | `data/v2_clean_template_grouped/` | **MEDIUM** | Conduct comprehensive dataset audit (lexical duplication, slot overlap, class distributions). Build a real-world human evaluation workflow with schema, PII redaction, and annotation guidelines. |
| **Legal & Licensing** | `LICENSE` file contains only a 17-line standard Apache header rather than the complete canonical Apache License Version 2.0 text. | `LICENSE:1-17` | **MEDIUM** | Replace with the full canonical Apache License 2.0 text while preserving copyright attribution. |
| **Documentation Credibility** | Multiple documentation files contain unsupported self-awarded "10/10" ratings, outdated 42-sample metrics, and contradictory candidate promotion statuses. | `README.md`, `docs/10_10_*.md` | **HIGH** | Consolidate documentation into an authoritative `RESULTS.md`. Move historical self-assessment reports to an archive, eliminate promotional rhetoric, and present honest measured evidence. |

---

## 4. Verification Gate Criteria for Sprint Progression

Progress through subsequent remediation sprints requires:
1. **Sprint A (Security & Config):** Zero test auth bypasses; fail startup on default admin key; proxy-aware rate limiter; harmonized `.env.example`.
2. **Sprint B (Heuristics & Safety):** Elimination of substring false positives (`camera`, `chair`, `hack-a-thon`); reduced unnecessary escalations without safety degradation; 100% passing regression tests.
3. **Sprint C (Neural Evaluation):** Execution of Qwen LoRA candidate against all 746 test records; full classification, priority, language, JSON validity, safety, and latency metric extraction.
4. **Sprint D (Dataset Quality):** Quantitative audit of synthetic dataset; complete human evaluation collection infrastructure.
5. **Sprint E (Backend Hardening):** Redis rate limiting support; bounded thread pool inference offloading.
6. **Sprint F (Calibration & Safety):** Calibration evaluation (ECE, Brier score, selective prediction).
7. **Sprint G (Documentation & Validation):** Consolidate `RESULTS.md`, complete Apache 2.0 license, full regression suite execution.
