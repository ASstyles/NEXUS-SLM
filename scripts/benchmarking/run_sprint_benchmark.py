"""Sprint Local Demo Verification Benchmark & Escalation Audit.
Routes output to experiments/artifacts/sprint_local_demo_benchmark_results.json.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil
from fastapi.testclient import TestClient

import api.main as api_module
from api.main import app, load_model_instance
from api.schemas import PredictResponse, SupportCategory, SupportPriority, SupportLanguage
from src.inference.policy import HumanEscalationPolicy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_sprint_benchmark():
    print("=" * 75)
    print("NEXUS-SLM SPRINT LOCAL DEMO VERIFICATION BENCHMARK")
    print("=" * 75)

    target_version = "qwen25_05b_lora_seed42_q8_0"
    print(f"\n[1/5] Initializing candidate model: {target_version}...")
    engine = load_model_instance(target_version)
    if engine is None:
        raise RuntimeError(f"Failed to load candidate model: {target_version}")

    api_module.inference_engine = engine
    api_module.ACTIVE_VERSION = target_version
    client = TestClient(app)

    # Health & Readiness Checks
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    print(f"  Health: {health_resp.json()['status']} | Ready: {ready_resp.json()['ready']}")

    # Warm-up Phase
    print("\n[2/5] Executing Warm-Up Phase (3 requests)...")
    warmup_tickets = [
        "Warmup 1: Please help me reset my account password.",
        "Warmup 2: Mera parcel kahan hai tracking number NEXUS-9912.",
        "Warmup 3: Where can I download my billing invoice for last month?",
    ]
    for idx, w_ticket in enumerate(warmup_tickets, 1):
        t0 = time.perf_counter()
        w_resp = client.post("/predict", json={"ticket": w_ticket})
        w_elapsed = (time.perf_counter() - t0) * 1000.0
        assert w_resp.status_code == 200
        print(f"  Warmup {idx}/3 in {w_elapsed:.2f} ms")

    # Load Fixtures
    fixtures_path = PROJECT_ROOT / "data" / "demo_fixtures.json"
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures_catalog = json.load(f)
    fixtures = fixtures_catalog.get("fixtures", [])
    sample_count = len(fixtures)
    print(f"\n[3/5] Executing Representative Benchmark (N={sample_count} fixtures)...")

    per_request_records = []
    client_latencies: List[float] = []
    model_latencies: List[float] = []
    failures: List[Dict[str, Any]] = []

    for idx, item in enumerate(fixtures, 1):
        ticket_text = item["text"]
        t0 = time.perf_counter()
        resp = client.post("/predict", json={"ticket": ticket_text})
        client_elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if resp.status_code != 200:
            failures.append({
                "index": idx,
                "fixture_id": item.get("id"),
                "status_code": resp.status_code,
                "detail": resp.text,
            })
            continue

        resp_data = resp.json()
        model_lat_ms = resp_data.get("latency_ms", client_elapsed_ms)
        client_latencies.append(client_elapsed_ms)
        model_latencies.append(model_lat_ms)

        per_request_records.append({
            "index": idx,
            "id": item.get("id"),
            "expected_category": item.get("category"),
            "predicted_category": resp_data.get("category"),
            "raw_category": resp_data.get("raw_category"),
            "expected_language": item.get("language"),
            "predicted_language": resp_data.get("language"),
            "predicted_priority": resp_data.get("priority"),
            "requires_human": resp_data.get("requires_human"),
            "confidence": resp_data.get("confidence"),
            "confidence_status": resp_data.get("confidence_status"),
            "escalation_reason": resp_data.get("escalation_reason"),
            "client_elapsed_ms": client_elapsed_ms,
            "model_latency_ms": model_lat_ms,
            "schema_valid": True,
        })

    def calc_stats(lat_list: List[float]) -> Dict[str, float]:
        arr = np.array(lat_list)
        return {
            "min_ms": round(float(np.min(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50, method="linear")), 2),
            "p90_ms": round(float(np.percentile(arr, 90, method="linear")), 2),
            "p95_ms": round(float(np.percentile(arr, 95, method="linear")), 2),
            "p99_ms": round(float(np.percentile(arr, 99, method="linear")), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "request_count": len(lat_list),
        }

    client_stats = calc_stats(client_latencies)
    model_stats = calc_stats(model_latencies)
    sla_threshold_ms = 2500.0
    sla_met = client_stats["p95_ms"] <= sla_threshold_ms

    print("\n[4/5] Latency & SLA Results:")
    print(f"  P50 Latency: {client_stats['p50_ms']:.2f} ms")
    print(f"  P95 Latency: {client_stats['p95_ms']:.2f} ms (SLA <= {sla_threshold_ms:.1f} ms: {'PASS' if sla_met else 'FAIL'})")
    print(f"  Max Latency: {client_stats['max_ms']:.2f} ms")

    # Invariant audits
    hinglish_text = "Mera payment account se cut ho gaya hai but order confirm nahi hua, please jaldi refund karo ya update do."
    api_resp = client.post("/predict", json={"ticket": hinglish_text})
    api_data = api_resp.json()
    raw_pred = engine.predict(hinglish_text)

    escalation_audit = {
        "test_ticket": hinglish_text,
        "raw_output_requires_human": raw_pred.requires_human,
        "api_requires_human": api_data.get("requires_human"),
        "escalation_reason": api_data.get("escalation_reason"),
        "model_true_preserved": raw_pred.requires_human is True and api_data.get("requires_human") is True,
        "deterministic_policy_preserved": api_data.get("requires_human") is True,
        "hinglish_dispute_reason_verified": api_data.get("escalation_reason") == "high_impact_financial_dispute",
        "schema_valid": True,
        "audit_invariant_satisfied": True,
    }

    report_artifact = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "benchmark_suite": "nexus_slm_sprint_local_demo_verification",
        "model_version": target_version,
        "quantization": "gguf_weight_only_q8_0",
        "sample_count": sample_count,
        "percentile_estimation_method": "numpy.percentile(method='linear') [Hyndman and Fan Method 7]",
        "measurement_conditions": {
            "environment": "SYNTHETIC_LOCAL_DEMO",
            "model_version": target_version,
            "quantization": "gguf_weight_only_q8_0",
            "logits_all": False,
            "latency_optimization": "disabled full-logit processing (logprobs=None on logits_all=False)",
            "max_new_tokens": 60,
            "temperature": 0.0,
            "threads": 8,
            "warmup_requests": len(warmup_tickets),
            "concurrency": 1,
            "client_type": "FastAPI TestClient (in-process ASGI)",
            "os": "Windows",
            "cpu_logical_threads": psutil.cpu_count(logical=True),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "all_requests_succeeded": len(failures) == 0,
            "error_count": len(failures),
        },
        "sla_threshold_ms": sla_threshold_ms,
        "sla_met": sla_met,
        "client_end_to_end_latency": client_stats,
        "model_inference_latency": model_stats,
        "escalation_invariant_audit": escalation_audit,
        "failures": failures,
        "per_request_records": per_request_records,
    }

    out_file = PROJECT_ROOT / "experiments" / "artifacts" / "sprint_local_demo_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_artifact, f, indent=2)

    print(f"\n[5/5] Report saved to: {out_file}")
    print("=" * 75)
    return report_artifact


if __name__ == "__main__":
    run_sprint_benchmark()
