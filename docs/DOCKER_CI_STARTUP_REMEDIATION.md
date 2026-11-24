# NEXUS-SLM — Docker CI Startup Failure & Production Health Verification Report

**Date:** November 2026  
**Issue Type:** GitHub Actions CI Failure in Job `Container Build & Health Validation`  
**Target Workflow:** `.github/workflows/ci.yml`  
**Status:** **REMEDIATED & VERIFIED** (Test Suite: 214 Passed, 1 Skipped, 0 Failed)

---

## 1. Exact Root Cause & Startup Failure Explanation

### 1.1 The Failure Mechanism
In GitHub Actions workflow run `Sprint/clean data model improvement #17`, the job `Container Build & Health Validation` failed during the step `Verify Container Startup & Health Check`:
```text
Waiting for container startup and healthcheck...
Waiting for API to become ready (attempt 1/30)...
...
Waiting for API to become ready (attempt 30/30)...
Health response: 
Error: Process completed with exit code 1.
```

### 1.2 Underlying Root Cause
1. **Production Configuration Constraint:**
   - In `src/utils/config.py`, `AppConfig` defaults to `env="production"`.
   - When running in production, `AppConfig.validate()` strictly enforces:
     ```python
     if self.is_production():
         if not self.admin_api_key or not self.admin_api_key.strip():
             raise ValueError("Mandatory administrative secret (NEXUS_ADMIN_KEY) is missing in production.")
     ```
   - This check is triggered at module import time in `api/main.py` (`cfg = get_config()`) when Uvicorn loads the application.
2. **Missing Container Environment Variable:**
   - The CI workflow launched the container via:
     ```bash
     docker run -d \
       --name nexus-ci-container \
       -p 8000:8000 \
       -e ACTIVE_MODEL_VERSION=nexus-qlora-v1 \
       -e API_PORT=8000 \
       -e API_HOST=0.0.0.0 \
       nexus-slm:ci-test
     ```
   - Notice that `NEXUS_ADMIN_KEY` was **never provided**.
   - As a result, the Python runtime immediately raised an uncaught `ValueError: Mandatory administrative secret (NEXUS_ADMIN_KEY) is missing in production.` and terminated with exit code 1 immediately upon startup.
3. **Flawed Diagnostic Loop:**
   - The CI health check looped 30 times attempting `curl http://localhost:8000/health`, unaware that the container had already exited on attempt 1.
   - When the loop timed out, it printed an empty response and exited without displaying the container's exit code or crash logs.

---

## 2. Remediation Implementation

### 2.1 Ephemeral Secret Generation & Masking in CI (`.github/workflows/ci.yml`)
To exercise real production security without hardcoding or leaking secrets:
1. **Cryptographic Ephemeral Key:** Generated a 256-bit URL-safe secret via Python's `secrets.token_urlsafe(32)`.
2. **GitHub Actions Log Masking:** Masked the generated secret immediately with `::add-mask::$CI_ADMIN_KEY` so GitHub Actions automatically redacts it from all command outputs (`***`).
3. **Environment Injection:** Passed into `$GITHUB_ENV` and provided to `docker run`:
   ```bash
   docker run -d \
     --name nexus-ci-container \
     -p 8000:8000 \
     -e NEXUS_ENV=production \
     -e NEXUS_ENFORCE_ADMIN_AUTH=true \
     -e NEXUS_ADMIN_KEY="$CI_ADMIN_KEY" \
     -e ACTIVE_MODEL_VERSION=nexus-qlora-v1 \
     -e API_PORT=8000 \
     -e API_HOST=0.0.0.0 \
     nexus-slm:ci-test
   ```

### 2.2 Robust Container Startup Diagnostics
Replaced the blind retry loop with early-exit crash detection:
- **Instant Crash Detection:** Each iteration inspects `docker inspect --format='{{.State.Status}}'`. If the container is `exited` or `dead`, the loop terminates immediately, retrieves `{{.State.ExitCode}}`, prints the last 200 lines of container logs, and fails the job instantly.
- **Health Check Probe:** Probes `http://localhost:8000/health` and breaks as soon as HTTP 200 is returned.
- **Readiness Probe:** Probes `http://localhost:8000/ready` certifying inference traffic readiness.

### 2.3 Comprehensive Container Smoke Testing
Added automated validation steps against the live container:
1. **Prediction Contract Verification (`POST /predict`):**
   - Dispatches a genuine customer ticket (`{"ticket": "My payment failed and account got debited twice"}`).
   - Asserts HTTP 200.
   - Validates presence of strict response schema fields: `category`, `priority`, `language`, `requires_human`, `request_id`, `inference_mode`.
2. **Administrative Authentication Enforcement:**
   - Unauthenticated call to protected endpoint (`POST /model/rollback`) $\implies$ asserts HTTP 401.
   - Invalid bearer token $\implies$ asserts HTTP 403.
   - Valid `$CI_ADMIN_KEY` bearer token $\implies$ asserts HTTP 200.
3. **Graceful Shutdown & Secret Leak Audit:**
   - Issues `docker stop --time 15`.
   - Inspects container exit status.
   - Audits container logs to guarantee `$CI_ADMIN_KEY` was never printed or leaked.

### 2.4 Configuration Alignment Across Codebase
- **`src/utils/config.py`**:
  - `host` now checks `os.getenv("NEXUS_HOST", os.getenv("API_HOST", "0.0.0.0"))`.
  - `AppConfig.validate()` updated: In production, attempting to disable admin auth (`enforce_admin_auth=False`) is explicitly rejected with `ValueError`. Authentication cannot be silently disabled.
- **`scripts/deployment/run_api.py`**:
  - Now loads `host` and `port` directly from `AppConfig` via `get_config()`, ensuring runtime and configuration parity.
- **`docker-compose.yml`**:
  - Configured with `NEXUS_ENV=production` and `NEXUS_ADMIN_KEY=${NEXUS_ADMIN_KEY}`.

---

## 3. Dedicated Startup Configuration Tests

Added explicit security tests in `tests/test_security_remediation.py`:
1. **`test_production_without_admin_key_does_not_silently_disable_auth`**:
   - Asserts missing key in production raises `ValueError`.
   - Asserts attempting `enforce_admin_auth=False` in production raises `ValueError`.
2. **`test_production_docker_smoke_configuration_validation`**:
   - Simulates the exact container environment variables (`NEXUS_ENV=production`, `NEXUS_ADMIN_KEY=<generated-key>`, `API_PORT=8000`, `API_HOST=0.0.0.0`).
   - Asserts `AppConfig` initializes and validates successfully with `host="0.0.0.0"` and `port=8000`.

---

## 4. Verification Results

### 4.1 Security Remediation Test Suite
```powershell
.venv\Scripts\pytest -v tests/test_security_remediation.py
```
**Result:** `24 passed, 1 skipped, 1 warning in 11.20s` (100% pass rate).

### 4.2 Full Repository Test Suite
```powershell
.venv\Scripts\pytest -q
```
**Result:** `214 passed, 1 skipped, 10 warnings in 71.47s` (100% pass rate).

### 4.3 Static Checks & Pip Consistency
```powershell
.venv\Scripts\python -m compileall src/ api/ scripts/ tests/
.venv\Scripts\python -m pip check
```
**Result:** All modules compiled successfully with exit code 0; `No broken requirements found.`

---

## 5. Files Modified

| File | Changes Made |
|---|---|
| `.github/workflows/ci.yml` | Ephemeral secret generation, secret masking, instant crash diagnostics, `/health` and `/ready` validation, `/predict` schema smoke test, admin auth verification, and secret exclusion audit |
| `src/utils/config.py` | Fallback to `API_HOST` for host binding; strict rejection of `enforce_admin_auth=False` in production |
| `scripts/deployment/run_api.py` | Bound host and port directly to `AppConfig` |
| `docker-compose.yml` | Added `NEXUS_ENV=production` and `NEXUS_ADMIN_KEY=${NEXUS_ADMIN_KEY}` |
| `tests/test_security_remediation.py` | Added dedicated production startup and Docker configuration smoke tests |

---

## 6. Important Caveat & Awaiting Approval

- **Caveat:** Per the prompt's instructions: *Do not claim GitHub Actions passed until a new actual workflow run confirms it.* This report documents the local verification and configuration fixes. The GitHub Actions CI run will execute once this commit is pushed.
- **Awaiting Push Approval:** In compliance with the non-negotiable rule (*"Do not push, merge or deploy without explicit approval"*), all changes are currently staged locally on branch `sprint/clean-data-model-improvement`.
