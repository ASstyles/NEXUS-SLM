# NEXUS-SLM — Final CI/CD Acceptance & Verification Report

> **Artifacts:** [.github/workflows/ci.yml](file:///C:/Users/avira/Desktop/NEXUS-SLM/.github/workflows/ci.yml) | [generate_artifact_manifests.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/scripts/deployment/generate_artifact_manifests.py) | [check_documentation_consistency.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/scripts/evaluation/check_documentation_consistency.py) | [REPRODUCIBILITY_GUIDE.md](file:///C:/Users/avira/Desktop/NEXUS-SLM/docs/REPRODUCIBILITY_GUIDE.md)

---

## 1. CI Pipeline Hardening Summary (`ci.yml`)

The automated GitHub Actions pipeline runs 3 dependent jobs (`lint-and-tests`, `docker-build`, `build-dashboard`) triggered on pushes (`main`, `develop`, `sprint/final-independent-validation`) and pull requests (`main`):

1. **`lint-and-tests` (Ubuntu + Live `redis:7-alpine` Service Container):**
   - Verifies CPU-only PyTorch (`torch.version.cuda is None`), runs `pip check`, compiles Python bytecode across `src/ api/ scripts/ tests/`, and executes `pytest -v tests/` against live Redis so `test_redis_rate_limiter_atomic_concurrency_cannot_exceed_limit` runs natively in CI.
   - Runs **new governance & provenance gates** in every build:
     - `python scripts/evaluation/check_documentation_consistency.py`
     - `python scripts/deployment/generate_artifact_manifests.py --verify` (checks dataset SHA-256 checksums and optional on-disk model artifact hashes without requiring gitignored large weights on clean clones)
     - `python scripts/data/human_eval_v2_manager.py validate`
   - Enforces `.env` exclusion from Git tracking.
2. **`docker-build` (Container Production Smoke Suite):**
   - Builds `Dockerfile` (non-root `nexususer`, CPU-only stack, copying `src/`, `api/`, `data/`, `models/`, `experiments/`), boots with a masked 256-bit ephemeral `NEXUS_ADMIN_KEY`, and probes `/health`, `/ready`, `/live`, `POST /predict`, `/metrics?format=prometheus`, `/evaluation/summary`, `401`/`403` auth enforcement, and secret-free container logs on `SIGTERM`.
3. **`build-dashboard` (Node.js 20 Next.js Build):**
   - Installs `dashboard/` packages and verifies strict TypeScript compilation + static optimization (`npm run build`).

---

## 2. Local Verification Executed on `sprint/final-independent-validation`

| Verification Command | Outcome | Notes |
|---|---|---|
| `python -m compileall -q src api scripts tests` | **PASS** (exit 0) | Zero syntax/bytecode errors |
| `pytest -v tests\` | **236 passed, 1 skipped** (58.31s) | Only `test_redis_rate_limiter_atomic_concurrency_cannot_exceed_limit` skipped locally due to absent Windows Redis daemon (runs under CI `redis:7-alpine`) |
| `scripts\evaluation\check_documentation_consistency.py` | **PASS** (exit 0) | All canonical hashes, archives, and synced tables verified |
| `scripts\deployment\generate_artifact_manifests.py --verify` | **PASS** (exit 0) | 13 datasets + 8 weight manifest hashes verified |
| `scripts\data\human_eval_v2_manager.py validate` | **PASS** (`status: PENDING`) | Clean validation schema; `0/300` sealed scenarios pending reviewers |
| `scripts\deployment\run_security_audit.py` | **PASS** (exit 0) | Zero leaked secrets (`3` allowlisted non-prod test literals), PII/auth/proxy controls pass |
| `cd dashboard; npm run build` | **PASS** (exit 0) | Next.js `14.2.35` clean production build (`23.8 kB` route) |
| Remote GitHub Actions Trigger (`sprint/final-independent-validation`) | **NOT PUSHED** | Preserved local commit/branch governance; awaiting explicit user approval to push / open PR |
| Local Docker Container Build | **SKIPPED IN LOCAL WINDOWS SANDBOX** | Historical CI Run #23 passed cleanly on Ubuntu; local `docker-build` runs upon CI push |
