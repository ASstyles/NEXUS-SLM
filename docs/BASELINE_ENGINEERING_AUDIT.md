# NEXUS-SLM — Baseline Engineering Audit (10/10 Excellence Sprint)

**Document ID:** NEXUS-AUDIT-BASELINE-2026-11-24  
**Audit Timestamp:** 2026-11-24T23:04:00Z  
**Branch Audited:** `main` (Merge Commit: `3e5a1ea`)  
**Sprint Working Branch:** `sprint/10-10-engineering-excellence`  
**Auditor Collective:** Principal ML Engineer, Senior NLP Research Scientist, Senior Backend Architect, AppSec Engineer, MLOps Engineer, QA Lead  

---

## 1. Git & Repository Baseline

- **Current Commit SHA:** `3e5a1ea70e28f293b6e76cf0e35279be6a8bf8d9` (Merge pull request #3 from ASstyles/sprint/clean-data-model-improvement)
- **Active Working Branch:** `sprint/10-10-engineering-excellence` (Branched directly from `main` at `3e5a1ea`)
- **Recent Merged Pull Requests:**
  - PR #3 (`3e5a1ea`): Merged commit `c5eb4da` ("feat: complete final excellence sprint across human eval, priority intelligence, escalation precision, and MLOps")
  - PR #2 (`d668628`): Merged commit `279198e` ("fix(ci): inject ephemeral admin key and improve container startup diagnostics")
- **Remote Configuration:**
  - `origin https://github.com/ASstyles/NEXUS-SLM.git (fetch)`
  - `origin https://github.com/ASstyles/NEXUS-SLM.git (push)`

---

## 2. Test Suite & Coverage Baseline

- **Total Collected Tests:** **216 tests** across 17 test modules:
  - `tests/test_api.py`: 11 tests (including `/health`, `/ready`, `/live`, `/predict`)
  - `tests/test_backend_and_intelligence.py`: 15 tests (Tracing, Rate Limiter, Auth, PII, Injections, Calibration, Prometheus)
  - `tests/test_benchmarks.py`: 3 tests (Cost calculator, comparison table, LLM adapter)
  - `tests/test_clean_v2_and_priority_rubric.py`: 7 tests (Zero leakage, split integrity, rubric invariants)
  - `tests/test_data_pipeline.py`: 10 tests (Data validation, normalization, synthetic generator)
  - `tests/test_demo_integration.py`: 6 tests
  - `tests/test_extended.py`: 20 tests (JSON parser recovery, edge cases)
  - `tests/test_heuristic_and_safety_regression.py`: 14 tests (Hinglish, Devanagari, compound keywords)
  - `tests/test_inference_contract.py`: 14 tests (Strict schema, invalid input handling)
  - `tests/test_kaggle_automation_regression.py`: 9 tests (Training notebook generation integrity)
  - `tests/test_metrics_math.py`: 14 tests (F1, precision, recall math)
  - `tests/test_parser.py`: 7 tests
  - `tests/test_qwen_gguf_candidate.py`: 7 tests (GGUF loading, Q4/Q8 quantization, llama.cpp inference)
  - `tests/test_qwen_quantized_candidate.py`: 6 tests
  - `tests/test_qwen_tokenization_and_masking.py`: 8 tests
  - `tests/test_rollback_and_governance.py`: 7 tests
  - `tests/test_safeguards_audit.py`: 27 tests (High-risk escalation invariants)
  - `tests/test_schemas.py`: 7 tests
  - `tests/test_security_remediation.py`: 13 tests (Admin secret entropy, timing attack, trusted proxy, Redis failover)
  - `tests/test_shadow_pilot.py`: 11 tests
- **Baseline Test Execution Result:** **215 passed, 1 skipped (0 failures)**. Skipped test: `test_redis_rate_limiter_atomic_concurrency_cannot_exceed_limit` (gracefully skips when external Redis instance is absent).

---

## 3. Dataset Registry & Cryptographic Hashes

| Dataset Key | File Path | Record Count | Cryptographic SHA-256 Hash | Status / Role |
|---|---|---:|---|---|
| **Synthetic Test (Locked)** | `data/v2_clean_template_grouped/test_clean.jsonl` | 746 | `fc65071b1e9f3303facf64711e72f8bb0f31b450c7f6a3c381bd4a9f8408cf9b` | Locked historical benchmark |
| **Synthetic Train** | `data/v2_clean_template_grouped/train_clean.jsonl` | 4,797 | `f67b50c7724e8cd53d029058c688e712b99b9a3f1be3df0a92c756975220fe1c` | Model training set |
| **Synthetic Val** | `data/v2_clean_template_grouped/val_clean.jsonl` | 449 | `faa98cf83cbe209736d8890a902f65fa382cb2ff31b96e55608d98d9e68fa29a` | Model validation set |
| **Human Holdout** | `data/human_eval/annotation_template.jsonl` | 200 | `e26d4eec99ddd247d3037652466df0024100b0cc85b9ca48e0fa32ed300923d3` | 184 completed + 16 pending |
| **Challenge Set** | `data/evaluation/high_risk_challenge_set.jsonl` | 70 | `70b43711fd2f515bdd27d6daa8e604698a109f6093cdaea7f85bd584339dbc5b` | Adversarial / safety evaluation |

---

## 4. Models & Runtime Inference Architecture

### 4.1 Production Engine (Active Default)
- **Engine Name:** **Heuristic Fallback V2** (`inference_mode: "heuristic_fallback"`)
- **Core Logic:** `classify_heuristic_category_and_priority` in `src/inference/rules.py`
- **Contextual Priority:** `determine_contextual_priority` in `src/inference/rules.py`
- **Escalation Gating:** `HumanEscalationPolicy.evaluate_escalation` in `src/inference/policy.py`
- **Characteristics:** Sub-10ms CPU latency (P50 5.17 ms), 181 req/s throughput, 100% critical recall, zero neural weight dependencies.

### 4.2 Experimental Neural Candidate (Hold Status)
- **Model Name:** `qwen25_05b_clean_v2_lora` (`run_20261030_190017`)
- **Base Architecture:** `Qwen/Qwen2.5-0.5B-Instruct` (~494M parameters)
- **LoRA Adapter Safetensors:** `experiments/artifacts/training_runs/qwen25_05b_clean_v2_lora_run_20261030_190017/adapter/adapter_model.safetensors` (SHA-256: `9B677B8D488CCD402505CCD7293074AA16104236EECF8B370E5FDE15AC4161D2`)
- **GGUF Quantization:** `models/qwen25_05b_lora_seed42_gguf_q4_k_m/qwen25_05b_lora_seed42_q4_k_m.gguf` (388.5 MB)
- **Current Status:** **HELD BACK / EXPERIMENTAL ONLY** (86.12% standalone safety recall, 29 false negatives, 1.5s CPU latency).

---

## 5. Backend, Security & Infrastructure Baseline

- **FastAPI Framework:** 0.112+ with Pydantic v2 validation contracts.
- **API Endpoints:** `/predict`, `/health`, `/ready`, `/live`, `/model`, `/model/rollback`, `/metrics`, `/monitoring/drift`, `/monitoring/alerts`, `/version`, `/labels`, `/artifacts`.
- **Security Invariants:**
  - Mandatory 32-character admin key entropy (`NEXUS_ADMIN_KEY`) in production.
  - Zero test-runner bypasses (`PYTEST_CURRENT_TEST` removed).
  - Proxy-aware IP resolution (`resolve_client_ip` via `trusted_proxies`).
  - Regex PII scrubber (`sanitize_pii`) and prompt injection guard (`inspect_prompt_injection`).
- **Distributed Rate Limiting:**
  - Redis Lua script (`SLIDING_WINDOW_LUA_SCRIPT`) executing atomic ZSET sliding window.
  - Graceful failover to process-local `SlidingWindowRateLimiter` deque on Redis disconnect.
- **Docker Stack:** Multi-stage `Dockerfile` (Python 3.10 slim, unprivileged user, healthcheck). `docker-compose.yml` bundling API and Redis 7.
- **Dashboard:** Next.js 14 (`dashboard/`) with dynamic engine mode detection (`heuristic_fallback` vs `neural`).
