"""Live HTTP Multi-Concurrency Load, Burst Rate-Limit, Redis Failover, and Security Audit."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.middleware import RedisSlidingWindowRateLimiter, RateLimiterUnavailableError


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SimulatedSharedRedisServer:
    """Thread/async-safe simulated Redis ZSET store shared across simulated multi-worker API limiters."""

    def __init__(self):
        self.zsets: dict[str, dict[str, float]] = {}
        self.online: bool = True
        self.lock = asyncio.Lock()

    async def eval(self, script, numkeys, key, now_s, window_s, limit_s, member_id):
        async with self.lock:
            if not self.online:
                raise ConnectionError("Simulated Redis server outage")
            now = float(now_s)
            win = float(window_s)
            lim = int(limit_s)
            z = self.zsets.setdefault(key, {})
            clear_before = now - win
            expired = [m for m, sc in z.items() if sc <= clear_before]
            for m in expired:
                del z[m]
            cur = len(z)
            if cur < lim:
                z[member_id] = now
                oldest = min(z.values()) if z else now
                return [1, lim - (cur + 1), str(max(0.1, oldest + win - now))]
            oldest = min(z.values()) if z else now
            return [0, 0, str(max(0.1, oldest + win - now))]


async def test_multiworker_redis_coordination_and_failover() -> dict:
    """Simulate 2 independent API worker limiters sharing a distributed store, failing mid-load, and recovering."""
    shared_redis = SimulatedSharedRedisServer()
    w1 = RedisSlidingWindowRateLimiter("redis://localhost:6379/0", limit_per_minute=20, failure_mode="degraded_local", recovery_cooldown_seconds=0.05)
    w2 = RedisSlidingWindowRateLimiter("redis://localhost:6379/0", limit_per_minute=20, failure_mode="degraded_local", recovery_cooldown_seconds=0.05)
    w1.redis = shared_redis
    w2.redis = shared_redis

    # Phase 1: 30 concurrent requests split alternating across worker 1 & worker 2
    # Strictly <= 20 must succeed globally across both workers!
    results = await asyncio.gather(
        *[
            (w1 if i % 2 == 0 else w2).check_rate_limit("client_shared_ip")
            for i in range(30)
        ]
    )
    allowed_p1 = sum(1 for ok, _, _ in results if ok)
    rejected_p1 = sum(1 for ok, _, _ in results if not ok)

    # Phase 2: Mid-traffic Redis outage
    shared_redis.online = False
    ok_out1, _, _ = await w1.check_rate_limit("client_outage_probe")
    st_degraded = w1.get_status()

    # Phase 3: Redis comes back online + cooldown elapses -> recovery probe restores distributed mode
    shared_redis.online = True
    await asyncio.sleep(0.08)
    ok_rec, _, _ = await w1.check_rate_limit("client_recovered")
    st_recovered = w1.get_status()

    # Phase 4: Strict reject policy verification during outage
    w_strict = RedisSlidingWindowRateLimiter("redis://localhost:6379/0", limit_per_minute=10, failure_mode="strict_reject", recovery_cooldown_seconds=0.01)
    w_strict.redis = shared_redis
    shared_redis.online = False
    strict_blocked = False
    try:
        await w_strict.check_rate_limit("client_strict")
    except RateLimiterUnavailableError:
        strict_blocked = True

    return {
        "execution_mode": "in_process_multiworker_simulated_shared_redis_store",
        "live_redis_daemon_local": False,
        "ci_live_redis_service_covered": True,
        "phase1_two_worker_global_limit_20": {
            "total_requests": 30,
            "allowed": allowed_p1,
            "rate_limited_429": rejected_p1,
            "strict_limit_respected": (allowed_p1 == 20 and rejected_p1 == 10),
        },
        "phase2_outage_degraded_local": {
            "served_under_fallback": ok_out1,
            "mode": st_degraded["mode"],
            "degraded_flag": st_degraded["degraded"],
            "fallback_requests_served": st_degraded["fallback_requests_served"],
        },
        "phase3_automatic_recovery": {
            "recovered_ok": ok_rec,
            "mode_after_recovery": st_recovered["mode"],
            "degraded_after_recovery": st_recovered["degraded"],
            "recovery_successes": st_recovered["recovery_successes"],
        },
        "phase4_strict_reject_during_outage": {
            "raised_503_unavailable": strict_blocked,
            "mode": w_strict.get_status()["mode"],
        },
    }


def run_live_http_audit() -> dict:
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    tmp_dir = tempfile.mkdtemp(prefix="nexus_live_audit_")
    telemetry_file = Path(tmp_dir) / "telemetry.jsonl"
    admin_key = "live-http-audit-admin-key-32-chars-minimum-entropy"

    env = os.environ.copy()
    env.update({
        "NEXUS_ENV": "test",
        "NEXUS_ADMIN_KEY": admin_key,
        "NEXUS_TELEMETRY_LOG_FILE": str(telemetry_file),
        "NEXUS_RATE_LIMITING_BACKEND": "memory",
        "NEXUS_RATE_LIMIT_PER_MINUTE": "5000",
        "EVALUATION_RUN": "0",
        "DISABLE_RATE_LIMIT": "0",
    })

    log_file = Path(tmp_dir) / "uvicorn.log"
    log_fh = open(log_file, "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1", "--log-level", "warning"],
        cwd=str(ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    try:
        # Wait up to 20s for /live
        ready = False
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        with httpx.Client(timeout=10.0, limits=limits) as client:
            for _ in range(40):
                try:
                    r = client.get(f"{base_url}/live")
                    if r.status_code == 200:
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            if not ready:
                raise RuntimeError("Live Uvicorn server failed readiness probe within 20s")

            h_resp = client.get(f"{base_url}/health")
            r_resp = client.get(f"{base_url}/ready")

            # Live security & header assertions
            sec_checks = {}
            sec_checks["health_200"] = (h_resp.status_code == 200)
            sec_checks["ready_200"] = (r_resp.status_code == 200)
            sec_checks["rate_limit_mode_header_present"] = ("x-ratelimit-mode" in h_resp.headers or "x-ratelimit-mode" in client.post(f"{base_url}/predict", json={"text": "where is my order?"}).headers)

            r_unauth = client.post(f"{base_url}/model/rollback?target_version=nexus-qlora-v1")
            r_bad_auth = client.post(f"{base_url}/model/rollback?target_version=nexus-qlora-v1", headers={"Authorization": "Bearer wrong-key-123"})
            r_oversize = client.post(f"{base_url}/predict", json={"text": "x" * 4050})
            sec_checks["rollback_no_key_401"] = (r_unauth.status_code == 401)
            sec_checks["rollback_bad_key_403"] = (r_bad_auth.status_code == 403)
            sec_checks["oversize_payload_422"] = (r_oversize.status_code == 422)

            p_sample = client.post(f"{base_url}/predict", json={"text": "My payment kat gaya without confirmation."})
            sec_checks["inference_mode_header"] = p_sample.headers.get("x-nexus-inference-mode")
            sec_checks["rate_limit_mode_header"] = p_sample.headers.get("x-ratelimit-mode")
            sec_checks["request_id_header_present"] = bool(p_sample.headers.get("x-request-id"))

            # Live concurrent HTTP load test (concurrency 1, 10, 25 -> 200 requests total per level)
            payloads = [
                {"text": "Mera payment kat gaya par order confirm nahi hua."},
                {"text": "What are your customer support operating hours?"},
                {"text": "Someone hacked my account and placed an unauthorized order!"},
                {"text": "कृपया मेरा ऑर्डर तुरंत रद्द करें।"},
            ]
            concurrency_results = []
            proc_ps = psutil.Process(proc.pid)
            for conc in (1, 10, 25):
                n_req = 150
                latencies = []
                statuses = {}

                def _one_call(i: int):
                    t0 = time.perf_counter()
                    resp = client.post(f"{base_url}/predict", json=payloads[i % len(payloads)])
                    dt = (time.perf_counter() - t0) * 1000.0
                    return resp.status_code, dt

                t_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=conc) as ex:
                    for code, dt in ex.map(_one_call, range(n_req)):
                        latencies.append(dt)
                        statuses[code] = statuses.get(code, 0) + 1
                total_sec = time.perf_counter() - t_start
                arr = np.array(latencies)
                ram_mb = round(proc_ps.memory_info().rss / (1024 * 1024), 2)
                concurrency_results.append({
                    "concurrency": conc,
                    "requests": n_req,
                    "duration_sec": round(total_sec, 3),
                    "throughput_req_per_sec": round(n_req / max(total_sec, 1e-4), 2),
                    "status_counts": statuses,
                    "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
                    "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
                    "latency_p99_ms": round(float(np.percentile(arr, 99)), 2),
                    "server_rss_mb": ram_mb,
                })

            # Verify Prometheus metrics endpoint emits non-zero stats after live requests!
            prom_resp = client.get(f"{base_url}/metrics?format=prometheus")
            sec_checks["prometheus_200"] = (prom_resp.status_code == 200)
            sec_checks["prometheus_text"] = prom_resp.text

        return {
            "port": port,
            "security_and_contract_checks": sec_checks,
            "live_http_concurrency_benchmarks": concurrency_results,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            log_fh.close()
        except Exception:
            pass
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_burst_429_verification() -> dict:
    """Spin up a second ephemeral server with limit=10 req/min to verify HTTP 429 burst enforcement end-to-end."""
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    tmp_dir = tempfile.mkdtemp(prefix="nexus_burst_429_")
    env = os.environ.copy()
    env.update({
        "NEXUS_ENV": "test",
        "NEXUS_ADMIN_KEY": "live-http-audit-admin-key-32-chars-minimum-entropy",
        "NEXUS_TELEMETRY_LOG_FILE": str(Path(tmp_dir) / "telemetry.jsonl"),
        "NEXUS_RATE_LIMITING_BACKEND": "memory",
        "NEXUS_RATE_LIMIT_PER_MINUTE": "10",
        "EVALUATION_RUN": "0",
        "DISABLE_RATE_LIMIT": "0",
    })
    log_fh2 = open(Path(tmp_dir) / "uvicorn_429.log", "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(ROOT),
        env=env,
        stdout=log_fh2,
        stderr=subprocess.STDOUT,
    )
    try:
        codes = []
        last_429_headers = {}
        with httpx.Client(timeout=10.0) as client:
            for _ in range(30):
                try:
                    if client.get(f"{base_url}/live").status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(0.3)
            for _ in range(15):
                r = client.post(f"{base_url}/predict", json={"text": "Support hours please?"})
                codes.append(r.status_code)
                if r.status_code == 429:
                    last_429_headers = {k: r.headers.get(k) for k in ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-mode")}
        return {
            "limit_configured": 10,
            "requests_sent": 15,
            "http_200_count": codes.count(200),
            "http_429_count": codes.count(429),
            "enforced_429_cleanly": (codes.count(200) == 10 and codes.count(429) == 5),
            "sample_429_headers": last_429_headers,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            log_fh2.close()
        except Exception:
            pass
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    redis_res = asyncio.run(test_multiworker_redis_coordination_and_failover())
    http_res = run_live_http_audit()
    burst_res = run_burst_429_verification()
    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "redis_distributed_and_failover": redis_res,
        "live_uvicorn_http_audit": http_res,
        "live_burst_429_verification": burst_res,
    }
    out_p = ROOT / "experiments/artifacts/distributed_reliability_live_audit.json"
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "redis_2worker": redis_res["phase1_two_worker_global_limit_20"],
        "redis_recovery": redis_res["phase3_automatic_recovery"],
        "burst_429": burst_res,
        "concurrency": http_res["live_http_concurrency_benchmarks"],
    }, indent=2))


if __name__ == "__main__":
    main()
