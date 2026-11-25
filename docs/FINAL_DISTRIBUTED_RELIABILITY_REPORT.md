# NEXUS-SLM — Final Distributed Reliability & Live HTTP Load Acceptance Report

> **Artifacts:** [distributed_reliability_live_audit.json](file:///C:/Users/avira/Desktop/NEXUS-SLM/experiments/artifacts/distributed_reliability_live_audit.json) | [api/middleware.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/api/middleware.py) | [api/main.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/api/main.py) | [test_distributed_reliability_and_safety.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/tests/test_distributed_reliability_and_safety.py)

---

## 1. Distributed Rate Limiting & Failover State Machine

[RedisSlidingWindowRateLimiter](file:///C:/Users/avira/Desktop/NEXUS-SLM/api/middleware.py#L140-L276) uses an atomic Redis Lua `ZSET` sliding window across replicas with two explicit, configurable failure modes (`NEXUS_RATE_LIMIT_FAILURE_MODE`):
- `degraded_local` (default): degrades cleanly to per-worker `SlidingWindowRateLimiter`, avoids dead-socket timeouts during `redis_recovery_cooldown_seconds` (`5.0 s`), automatically probes recovery once cooldown elapses, and restores `redis_distributed` state cleanly. Emits headers `X-RateLimit-Mode` and `X-RateLimit-Degraded: true`.
- `strict_reject`: fails closed by raising `RateLimiterUnavailableError`, returning HTTP `503 Service Unavailable` with `Retry-After: 5` and `X-RateLimit-Mode: strict_reject`.

---

## 2. Live Uvicorn HTTP Load & Burst 429 Results (`Port 63197`, Ephemeral Worker)

Executed against a real Uvicorn FastAPI HTTP server ([run_live_http_load_and_failover_audit.py](file:///C:/Users/avira/Desktop/NEXUS-SLM/scripts/deployment/run_live_http_load_and_failover_audit.py)) serving bilingual EN/Hindi/Hinglish POST `/predict` requests under `heuristic_fallback`:

| Concurrent HTTP Clients | Requests | Duration | Throughput | HTTP 200 | HTTP P50 Latency | **HTTP P95 Latency** | HTTP P99 Latency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150 | 0.632 s | 237.37 req/s | 150 / 150 | 3.70 ms | **5.46 ms** | 7.16 ms |
| 10 | 150 | 0.575 s | 260.68 req/s | 150 / 150 | 36.13 ms | **45.21 ms** | 47.40 ms |
| 25 | 150 | 0.566 s | **265.07 req/s** | 150 / 150 | 86.53 ms | **98.65 ms** | 102.70 ms |

*(Note on host measurement: `server_rss_mb: 4.21` in the raw JSON reflects Windows `.venv/Scripts/python.exe` launcher PID tracking rather than child uvicorn worker RAM; worker process RSS under heuristic mode was independently measured at `269.94 MB` in `runtime_inference_benchmarks_v2.json`.)*

### Live HTTP Burst `429` Verification (`limit=10 req/min`, 15 Requests)
- HTTP `200` allowed: **10 / 15**
- HTTP `429` rejected: **5 / 15** (`enforced_429_cleanly: true`)
- Verified response headers: `retry-after: 59`, `x-ratelimit-limit: 10`, `x-ratelimit-remaining: 0`, `x-ratelimit-mode: memory_local`.

### Live Contract & Prometheus `/metrics?format=prometheus` Check (`453` Requests Logged)
- `nexus_requests_total 453`
- `nexus_error_rate 0.0022` (1 deliberate HTTP 422 payload test)
- `nexus_latency_p50_ms 0.12` | `nexus_latency_p95_ms 0.25`
- `nexus_escalation_rate 0.7461`

---

## 3. Multi-Worker Shared Redis Failover Verification

Because a live local Redis 7+ daemon is **not installed/running in this local Windows sandbox** (`live_redis_daemon_local: false`), multi-worker coordination was verified via in-process simulated shared Lua `ZSET` state locally AND via a real `redis:7-alpine` service container in GitHub Actions `.github/workflows/ci.yml`:

1. **Two Workers, Shared Global Limit = 20 (`30` concurrent requests split across Worker 1 + Worker 2):**
   - Strictly `20` allowed + `10` HTTP `429` rejections (`strict_limit_respected: true`).
2. **Mid-Traffic Outage -> `degraded_local` Fallback:** `served_under_fallback: true`, `degraded_flag: true`.
3. **Cooldown Elapsed -> Automatic Recovery Probe:** `recovered_ok: true`, `mode_after_recovery: "redis_distributed"`, `degraded_after_recovery: false`, `recovery_successes: 1`.
4. **`strict_reject` Outage Behavior:** `raised_503_unavailable: true` (`mode: "strict_reject"`).

**Environment Disclosure:** Multi-worker real-Redis concurrency test (`test_redis_rate_limiter_atomic_concurrency_cannot_exceed_limit`) skips locally without a running `localhost:6379` Redis daemon and executes against live `redis:7-alpine` in GitHub Actions.
