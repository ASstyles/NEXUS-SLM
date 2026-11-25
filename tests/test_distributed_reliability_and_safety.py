"""Tests for Redis rate-limiter operational modes, recovery, strict neural enforcement, Prometheus metrics, and V2 multitask components."""

import asyncio
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.middleware import RedisSlidingWindowRateLimiter, RateLimiterUnavailableError
from src.models.multitask_v2 import (
    MulticlassTemperatureScaler,
    MultilingualTextPreprocessor,
    NexusMultitaskNeuralHead,
    compute_multiclass_calibration_metrics,
    ordinal_probs_from_cum_logits,
    ordinal_targets_from_priority,
)
from src.utils.config import reload_config


class FakeRedisEval:
    """Deterministic fake async Redis client simulating healthy ZSET Lua eval and transient outages."""

    def __init__(self, limit: int = 3):
        self.limit = limit
        self.fail = False
        self.counts = {}
        self.eval_calls = 0

    async def eval(self, script, numkeys, key, now, window, limit_str, member_id):
        self.eval_calls += 1
        if self.fail:
            raise ConnectionError("Simulated Redis connection refused")
        c = self.counts.get(key, 0)
        lim = int(limit_str)
        if c < lim:
            self.counts[key] = c + 1
            return [1, lim - (c + 1), "59.0"]
        return [0, 0, "59.0"]


@pytest.mark.asyncio
async def test_redis_rate_limiter_failover_and_automatic_recovery():
    """Verify Redis -> degraded_local transition during outage and automatic recovery on probe."""
    limiter = RedisSlidingWindowRateLimiter(
        redis_url="redis://localhost:6379/0",
        limit_per_minute=2,
        failure_mode="degraded_local",
        recovery_cooldown_seconds=0.05,
    )
    fake = FakeRedisEval(limit=2)
    limiter.redis = fake

    # 1. Healthy distributed execution
    allowed, rem, _ = await limiter.check_rate_limit("10.0.0.1")
    assert allowed is True and rem == 1
    assert limiter.get_status()["mode"] == "redis_distributed"
    assert limiter.get_status()["healthy"] is True

    # 2. Simulated Redis outage -> transitions cleanly to degraded_local
    fake.fail = True
    allowed2, _, _ = await limiter.check_rate_limit("10.0.0.1")
    assert allowed2 is True
    st_down = limiter.get_status()
    assert st_down["mode"] == "degraded_local"
    assert st_down["degraded"] is True
    assert st_down["fallback_requests_served"] >= 1

    # 3. Restore Redis and wait past short cooldown -> automatic half-open probe recovers
    fake.fail = False
    await asyncio.sleep(0.08)
    allowed3, _, _ = await limiter.check_rate_limit("10.0.0.2")
    assert allowed3 is True
    st_rec = limiter.get_status()
    assert st_rec["mode"] == "redis_distributed"
    assert st_rec["healthy"] is True
    assert st_rec["degraded"] is False
    assert st_rec["recovery_successes"] == 1


@pytest.mark.asyncio
async def test_redis_rate_limiter_strict_reject_mode():
    """Verify strict_reject raises RateLimiterUnavailableError cleanly when Redis is down."""
    limiter = RedisSlidingWindowRateLimiter(
        redis_url="redis://localhost:6379/0",
        limit_per_minute=5,
        failure_mode="strict_reject",
        recovery_cooldown_seconds=0.01,
    )
    fake = FakeRedisEval(limit=5)
    fake.fail = True
    limiter.redis = fake

    with pytest.raises(RateLimiterUnavailableError):
        await limiter.check_rate_limit("10.0.0.9")

    st = limiter.get_status()
    assert st["mode"] == "strict_reject"
    assert st["strict_rejections"] == 1


def test_prometheus_metrics_export_real_non_zero_latencies_and_escalation(monkeypatch):
    """Verify /metrics?format=prometheus reads real nested percentiles and human_escalation_rate."""
    import api.main as api_main

    monkeypatch.setattr(
        api_main.telemetry,
        "get_summary_statistics",
        lambda: {
            "total_requests": 100,
            "error_count": 2,
            "error_rate": 0.02,
            "latency_percentiles": {"p50": 12.34, "p95": 45.67},
            "human_escalation_rate": 0.65,
        },
    )
    with TestClient(api_main.app) as client:
        r = client.get("/metrics?format=prometheus")
        assert r.status_code == 200
        assert "nexus_latency_p50_ms 12.34" in r.text
        assert "nexus_latency_p95_ms 45.67" in r.text
        assert "nexus_escalation_rate 0.65" in r.text


def test_strict_neural_mode_prevents_silent_heuristic_fallback(monkeypatch):
    """Ensure NEXUS_STRICT_NEURAL_MODE=true returns HTTP 503 instead of silent heuristic fallback."""
    import api.main as api_main

    monkeypatch.setenv("NEXUS_STRICT_NEURAL_MODE", "true")
    reload_config()
    orig_eng = api_main.inference_engine
    try:
        api_main.inference_engine = None
        with TestClient(api_main.app) as client:
            # Override any lifespan load failure test: force inference_engine None directly
            api_main.inference_engine = None
            r_pred = client.post("/predict", json={"text": "Where is my order tracking status?"})
            assert r_pred.status_code == 503
            r_ready = client.get("/ready")
            assert r_ready.status_code == 503
    finally:
        api_main.inference_engine = orig_eng
        monkeypatch.delenv("NEXUS_STRICT_NEURAL_MODE", raising=False)
        reload_config()


def test_v2_multitask_head_ordinal_and_calibration_math():
    """Verify ordinal cumulative probability conversion and scalar temperature NLL calibration."""
    targets = ordinal_targets_from_priority(torch.tensor([0, 1, 2, 3]))
    assert targets.tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]

    mod = NexusMultitaskNeuralHead(input_dim=32, hidden_dim=16)
    x = torch.randn(4, 32)
    out = mod(x)
    assert out["category_logits"].shape == (4, 14)
    p_probs = ordinal_probs_from_cum_logits(out["priority_cum_logits"])
    assert p_probs.shape == (4, 4)
    assert torch.allclose(p_probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    # Temperature scaler fit lowers NLL on overconfident synthetic logits
    rng = np.random.default_rng(7)
    y = rng.integers(0, 4, size=100)
    logits = np.zeros((100, 4), dtype=np.float32)
    for i in range(100):
        logits[i, y[i]] = 5.0 if i < 65 else -2.0
        logits[i, (y[i] + 1) % 4] = 0.0 if i < 65 else 4.0

    raw_e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    raw_probs = raw_e / raw_e.sum(axis=-1, keepdims=True)
    raw_m = compute_multiclass_calibration_metrics(raw_probs, y)

    scaler = MulticlassTemperatureScaler()
    T = scaler.fit_temperature(logits, y)
    assert T > 1.0
    cal_probs = scaler.predict_probs_temperature(logits)
    cal_m = compute_multiclass_calibration_metrics(cal_probs, y)
    assert cal_m["nll"] < raw_m["nll"]

    normed = MultilingualTextPreprocessor.normalize("https://example.com ORD-12345  मेरा\u200b ऑर्डर ")
    assert "<URL>" in normed and "<ORDER_ID>" in normed and "\u200b" not in normed
