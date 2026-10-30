"""Concurrency Load & Stress Benchmark for NEXUS-SLM FastAPI Backend.

Evaluates latency percentiles (P50, P90, P95, P99), throughput (RPS),
peak RSS memory, and error rate across increasing concurrency levels.
Outputs machine-readable benchmark artifact: experiments/artifacts/concurrency_load_test_results.json.
"""

import asyncio
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def execute_request(client: TestClient, payload: Dict[str, str], worker_id: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    status_code = 0
    success = False
    headers = {"X-Forwarded-For": f"10.0.{worker_id}.1"}
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


def run_load_test_for_concurrency(concurrency: int, total_requests: int) -> Dict[str, Any]:
    client = TestClient(app)
    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    start_wall = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(total_requests):
            payload = TEST_PAYLOADS[i % len(TEST_PAYLOADS)]
            worker_id = i % concurrency
            futures.append(executor.submit(execute_request, client, payload, worker_id))

        for f in futures:
            results.append(f.result())

    duration_sec = time.perf_counter() - start_wall
    ram_after_mb = process.memory_info().rss / (1024 * 1024)

    latencies = [r["latency_ms"] for r in results]
    successes = [r for r in results if r["success"]]
    error_count = len(results) - len(successes)

    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))
    throughput_rps = round(total_requests / duration_sec, 2)

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(successes),
        "failed_requests": error_count,
        "error_rate": round(error_count / total_requests, 4),
        "duration_seconds": round(duration_sec, 3),
        "throughput_req_per_sec": throughput_rps,
        "latency_p50_ms": round(p50, 2),
        "latency_p90_ms": round(p90, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "latency_mean_ms": round(mean_lat, 2),
        "peak_ram_mb": round(max(ram_before_mb, ram_after_mb), 2),
    }


def main():
    print("=" * 80)
    print("NEXUS-SLM CONCURRENCY LOAD & LATENCY BENCHMARK")
    print("=" * 80)

    concurrency_levels = [1, 5, 10, 20]
    total_requests_per_level = 50

    benchmarks_summary = []
    for c in concurrency_levels:
        print(f"\nEvaluating Concurrency = {c} (Total Requests: {total_requests_per_level})...")
        metrics = run_load_test_for_concurrency(c, total_requests_per_level)
        benchmarks_summary.append(metrics)
        print(f"  Throughput: {metrics['throughput_req_per_sec']} RPS | Error Rate: {metrics['error_rate'] * 100}%")
        print(f"  P50: {metrics['latency_p50_ms']} ms | P95: {metrics['latency_p95_ms']} ms | P99: {metrics['latency_p99_ms']} ms")
        print(f"  Peak RSS: {metrics['peak_ram_mb']} MB")

    artifact_dir = PROJECT_ROOT / "experiments" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifact_dir / "concurrency_load_test_results.json"

    report = {
        "benchmark_timestamp": datetime.utcnow().isoformat() + "Z",
        "system_info": {
            "os": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "python_version": sys.version.split()[0],
        },
        "concurrency_benchmarks": benchmarks_summary,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Artifact Saved] Successfully written to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
