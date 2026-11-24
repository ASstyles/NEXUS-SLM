"""Concurrency Load & Stress Benchmark for NEXUS-SLM FastAPI Backend.

Evaluates latency percentiles (P50, P90, P95, P99), throughput (RPS),
peak RSS memory, CPU utilization, queueing, and rejection behavior across
increasing concurrency levels (1, 5, 10, 20 workers) with 1,000 requests per level.

Outputs:
- experiments/artifacts/concurrency_load_test_results.json
- experiments/artifacts/final_backend_concurrency_benchmark.json
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure evaluation mode does not suppress real concurrency middleware
os.environ["EVALUATION_RUN"] = "0"
os.environ["DISABLE_RATE_LIMIT"] = "0"

import numpy as np
import psutil
from fastapi.testclient import TestClient
from api.main import app

# Benchmark test payloads representing multilingual triage
TEST_PAYLOADS = [
    {"text": "Mera payment cut ho gaya lekin order confirm nahi hua."},
    {"text": "Urgent: Unauthorized transaction of $1,200 detected on credit card."},
    {"text": "मेरा रिफंड अभी तक बैंक खाते में नहीं आया है, कृपया सहायता करें।"},
    {"text": "How do I update my registered mobile number and KYC details?"},
    {"text": "Delivery executive misbehaved and the package seal was broken."},
    {"text": "App keeps crashing on payment checkout screen after recent update."},
    {"text": "Subscription renew ho gaya bina meri permission ke, cancel please."},
    {"text": "Account login nahi ho raha, OTP nahi aa raha mobile pe."},
]


def execute_request(client: TestClient, payload: Dict[str, str], client_ip: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    status_code = 0
    success = False
    headers = {"X-Forwarded-For": client_ip}
    try:
        resp = client.post("/predict", json=payload, headers=headers)
        t1 = time.perf_counter()
        status_code = resp.status_code
        latency_ms = (t1 - t0) * 1000.0
        success = (status_code == 200)
    except Exception as exc:
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        success = False

    return {
        "status_code": status_code,
        "latency_ms": latency_ms,
        "success": success,
    }


def run_load_test_for_concurrency(client: TestClient, concurrency: int, total_requests: int) -> Dict[str, Any]:
    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    # Prime CPU measurement
    psutil.cpu_percent(interval=None)

    start_wall = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(total_requests):
            payload = TEST_PAYLOADS[i % len(TEST_PAYLOADS)]
            worker_id = i % concurrency
            # Multi-session IP pool: each worker simulates 25 concurrent client sessions
            client_ip = f"10.{worker_id}.{(i // 40) % 25}.{(i % 25) + 1}"
            futures.append(executor.submit(execute_request, client, payload, client_ip))

        for f in futures:
            results.append(f.result())

    duration_sec = time.perf_counter() - start_wall
    ram_after_mb = process.memory_info().rss / (1024 * 1024)
    avg_cpu_percent = psutil.cpu_percent(interval=None)

    latencies = [r["latency_ms"] for r in results]
    successes = [r for r in results if r["success"]]
    status_counts = Counter(r["status_code"] for r in results)
    error_count = len(results) - len(successes)

    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))
    throughput_rps = round(total_requests / max(duration_sec, 0.001), 2)

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(successes),
        "failed_requests": error_count,
        "status_code_breakdown": dict(status_counts),
        "http_200_count": status_counts.get(200, 0),
        "http_429_rate_limited": status_counts.get(429, 0),
        "http_503_overload": status_counts.get(503, 0),
        "other_errors": error_count - status_counts.get(429, 0) - status_counts.get(503, 0),
        "error_rate": round(error_count / total_requests, 4),
        "duration_seconds": round(duration_sec, 3),
        "throughput_req_per_sec": throughput_rps,
        "latency_p50_ms": round(p50, 2),
        "latency_p90_ms": round(p90, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "latency_mean_ms": round(mean_lat, 2),
        "peak_ram_mb": round(max(ram_before_mb, ram_after_mb), 2),
        "cpu_utilization_percent": round(avg_cpu_percent, 1),
    }


def run_rate_limiting_rejection_test(client: TestClient) -> Dict[str, Any]:
    print("\nRunning Rate Limiter Rejection & Throttling Verification...")
    test_ip = "192.168.100.50"
    payload = {"text": "Rapid burst request to verify 429 throttling"}
    total_burst = 160  # Rate limit is 120/min

    results = []
    t0 = time.perf_counter()
    for i in range(total_burst):
        results.append(execute_request(client, payload, test_ip))
    dt = time.perf_counter() - t0

    status_counts = Counter(r["status_code"] for r in results)
    print(f"  Single IP Burst ({total_burst} requests in {dt:.3f}s): {dict(status_counts)}")

    return {
        "test_name": "single_ip_burst_throttling",
        "burst_requests": total_burst,
        "duration_seconds": round(dt, 3),
        "configured_limit_per_minute": 120,
        "accepted_200_count": status_counts.get(200, 0),
        "throttled_429_count": status_counts.get(429, 0),
        "rejection_rate": round(status_counts.get(429, 0) / total_burst, 4),
        "rejection_enforced": status_counts.get(429, 0) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="NEXUS-SLM Concurrency Load Benchmark")
    parser.add_argument("--requests", type=int, default=1000, help="Requests per concurrency level (default: 1000)")
    args = parser.parse_args()

    print("=" * 80)
    print("NEXUS-SLM CONCURRENCY LOAD & STRESS BENCHMARK")
    print(f"Requests per Concurrency Level: {args.requests}")
    print("=" * 80)

    client = TestClient(app)

    # Warm-up phase: 10 requests to prime routes & caches
    for _ in range(10):
        client.get("/health")

    concurrency_levels = [1, 5, 10, 20]
    total_requests_per_level = args.requests

    benchmarks_summary = []
    for c in concurrency_levels:
        print(f"\nEvaluating Concurrency = {c} (Total Requests: {total_requests_per_level})...")
        metrics = run_load_test_for_concurrency(client, c, total_requests_per_level)
        benchmarks_summary.append(metrics)
        print(f"  Throughput: {metrics['throughput_req_per_sec']} RPS | Error Rate: {metrics['error_rate'] * 100:.2f}%")
        print(f"  P50: {metrics['latency_p50_ms']} ms | P95: {metrics['latency_p95_ms']} ms | P99: {metrics['latency_p99_ms']} ms")
        print(f"  Status Codes: {metrics['status_code_breakdown']} | CPU: {metrics['cpu_utilization_percent']}% | Peak RAM: {metrics['peak_ram_mb']} MB")

    rejection_test = run_rate_limiting_rejection_test(client)

    artifact_dir = PROJECT_ROOT / "experiments" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "benchmark_timestamp": datetime.utcnow().isoformat() + "Z",
        "benchmark_configuration": {
            "requests_per_level": total_requests_per_level,
            "concurrency_levels": concurrency_levels,
            "total_load_requests": total_requests_per_level * len(concurrency_levels),
            "warmup_requests": 10,
            "workload_type": "multilingual_ticket_triage_payloads",
            "rate_limiter_architecture": "Process-local in-memory sliding window deque per client IP (threading.Lock). Not shared or distributed across multi-node or multi-process replicas without an external store such as Redis.",
            "concurrency_middleware": "ConcurrencyAndTimeoutMiddleware (Bounded semaphore limit: 50 concurrent requests, 15.0s per-request timeout)",
        },
        "system_info": {
            "os": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_physical_count": psutil.cpu_count(logical=False),
            "python_version": sys.version.split()[0],
            "total_physical_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "concurrency_benchmarks": benchmarks_summary,
        "rate_limiting_rejection_behavior": rejection_test,
    }

    out_file1 = artifact_dir / "concurrency_load_test_results.json"
    out_file2 = artifact_dir / "final_backend_concurrency_benchmark.json"

    with open(out_file1, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(out_file2, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Artifact Saved] Successfully written to: {out_file1}")
    print(f"[Artifact Saved] Successfully written to: {out_file2}")
    print("=" * 80)


if __name__ == "__main__":
    main()
