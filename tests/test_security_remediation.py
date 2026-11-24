"""Comprehensive regression and verification tests for Security Remediation (Sprint A)."""

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.middleware import resolve_client_ip, SlidingWindowRateLimiter, RedisSlidingWindowRateLimiter
from src.utils.config import AppConfig, KNOWN_INSECURE_KEYS, generate_secure_admin_key

client = TestClient(app)


class TestAdminKeySecurity:
    """Verifies that insecure admin keys and missing production secrets are rejected."""

    def test_production_fails_on_missing_admin_key(self):
        """Production startup must abort if NEXUS_ADMIN_KEY is empty."""
        with pytest.raises(ValueError, match="Mandatory administrative secret .* is missing in production"):
            AppConfig(env="production", admin_api_key="", enforce_admin_auth=True)

    def test_production_without_admin_key_does_not_silently_disable_auth(self):
        """Production configuration must never silently disable admin authentication when key is missing."""
        # 1. Missing key in production fails
        with pytest.raises(ValueError, match="Mandatory administrative secret .* is missing in production"):
            AppConfig(env="production", admin_api_key="", enforce_admin_auth=True)

        # 2. Attempting to disable admin auth in production is explicitly rejected
        with pytest.raises(ValueError, match="Administrator authentication cannot be disabled in production"):
            AppConfig(env="production", admin_api_key="", enforce_admin_auth=False)

    @pytest.mark.parametrize("bad_key", list(KNOWN_INSECURE_KEYS))
    def test_production_fails_on_known_insecure_defaults(self, bad_key):
        """Production startup must abort if NEXUS_ADMIN_KEY matches known defaults."""
        with pytest.raises(ValueError, match="matches a known insecure or placeholder"):
            AppConfig(env="production", admin_api_key=bad_key, enforce_admin_auth=True)

    def test_production_fails_on_short_key(self):
        """Production startup must abort if secret has insufficient entropy (< 32 chars)."""
        with pytest.raises(ValueError, match="too short"):
            AppConfig(env="production", admin_api_key="short-secret-key-12345", enforce_admin_auth=True)

    def test_production_accepts_strong_random_key(self):
        """Production startup succeeds with cryptographically strong >= 32 char key."""
        strong_key = generate_secure_admin_key()
        assert len(strong_key) >= 32
        cfg = AppConfig(env="production", admin_api_key=strong_key, enforce_admin_auth=True)
        assert cfg.admin_api_key == strong_key

    def test_generate_secure_admin_key_entropy(self):
        """Verify secrets-based key generation utility produces unique, high-entropy tokens."""
        k1 = generate_secure_admin_key()
        k2 = generate_secure_admin_key()
        assert k1 != k2
        assert len(k1) >= 40  # 32 bytes base64-urlsafe gives 43 chars

    def test_production_docker_smoke_configuration_validation(self):
        """Simulates production Docker container startup with ephemeral admin key."""
        ephemeral_key = generate_secure_admin_key()
        with patch.dict(os.environ, {
            "NEXUS_ENV": "production",
            "NEXUS_ENFORCE_ADMIN_AUTH": "true",
            "NEXUS_ADMIN_KEY": ephemeral_key,
            "API_PORT": "8000",
            "API_HOST": "0.0.0.0",
        }):
            cfg = AppConfig()
            assert cfg.env == "production"
            assert cfg.enforce_admin_auth is True
            assert cfg.admin_api_key == ephemeral_key
            assert cfg.port == 8000
            assert cfg.host == "0.0.0.0"


class TestAuthenticationEnforcement:
    """Verifies strict auth enforcement on protected endpoints with NO test bypasses."""

    def test_admin_endpoint_rejects_missing_credentials(self):
        """Protected administrative endpoint must return 401 when no token is supplied."""
        resp = client.post("/model/rollback?target_version=baseline-smollm-135m")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers
        assert resp.headers["WWW-Authenticate"] == "Bearer"

    def test_admin_endpoint_rejects_invalid_credentials(self):
        """Protected administrative endpoint must return 403 when wrong token is supplied."""
        resp = client.post(
            "/model/rollback?target_version=baseline-smollm-135m",
            headers={"Authorization": "Bearer completely-wrong-admin-key"},
        )
        assert resp.status_code == 403
        assert "Forbidden" in resp.json()["detail"]

    def test_pytest_env_var_does_not_bypass_authentication(self):
        """Explicitly proves that PYTEST_CURRENT_TEST does NOT weaken or bypass auth."""
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "tests/test_something.py::test_case"}):
            resp = client.post(
                "/model/rollback?target_version=baseline-smollm-135m",
                headers={"Authorization": "Bearer invalid-key"},
            )
            assert resp.status_code == 403

            # And missing credentials still 401
            resp_missing = client.post("/model/rollback?target_version=baseline-smollm-135m")
            assert resp_missing.status_code == 401

    def test_admin_endpoint_accepts_valid_credentials(self):
        """Protected administrative endpoint succeeds with configured admin key."""
        from src.utils.config import get_config
        cfg = get_config()
        resp = client.post(
            "/model/rollback?target_version=nexus-qlora-v1",
            headers={"Authorization": f"Bearer {cfg.admin_api_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "SUCCESS"


class TestTrustedProxyAndRateLimiting:
    """Verifies proxy trust boundaries and distributed rate limiter mechanics."""

    def test_untrusted_proxy_header_is_ignored(self):
        """Spoofed X-Forwarded-For from untrusted IP must be discarded in favor of peer IP."""
        class DummyRequest:
            def __init__(self, host, xff):
                self.client = type("Client", (), {"host": host})()
                self.headers = {"X-Forwarded-For": xff} if xff else {}

        # Peer is 203.0.113.50 (an untrusted external client sending a spoofed header)
        req = DummyRequest("203.0.113.50", "198.51.100.1")
        resolved = resolve_client_ip(req, trusted_proxies=["10.0.0.1", "10.0.0.2"])
        assert resolved == "203.0.113.50"  # Spoofed IP ignored!

    def test_trusted_proxy_header_is_honored(self):
        """X-Forwarded-For from a verified trusted proxy must resolve to original client IP."""
        class DummyRequest:
            def __init__(self, host, xff):
                self.client = type("Client", (), {"host": host})()
                self.headers = {"X-Forwarded-For": xff} if xff else {}

        # Peer is 10.0.0.1 (trusted reverse proxy) forwarding for original client 198.51.100.1
        req = DummyRequest("10.0.0.1", "198.51.100.1, 10.0.0.1")
        resolved = resolve_client_ip(req, trusted_proxies=["10.0.0.1", "10.0.0.2"])
        assert resolved == "198.51.100.1"

    @pytest.mark.asyncio
    async def test_sliding_window_rate_limiter_enforces_limit_and_reset(self):
        """Verify in-memory sliding window allows up to limit and reports remaining requests."""
        limiter = SlidingWindowRateLimiter(limit_per_minute=3, window_seconds=60.0)
        client_id = "test-client-ip"

        # First 3 requests allowed
        ok1, rem1, _ = await limiter.check_rate_limit(client_id)
        assert ok1 is True
        assert rem1 == 2

        ok2, rem2, _ = await limiter.check_rate_limit(client_id)
        assert ok2 is True
        assert rem2 == 1

        ok3, rem3, _ = await limiter.check_rate_limit(client_id)
        assert ok3 is True
        assert rem3 == 0

        # 4th request blocked with retry-after > 0
        ok4, rem4, reset_in = await limiter.check_rate_limit(client_id)
        assert ok4 is False
        assert rem4 == 0
        assert reset_in > 0.0

    @pytest.mark.asyncio
    async def test_redis_rate_limiter_falls_back_gracefully_on_connection_error(self):
        """Redis rate limiter must gracefully fall back to local in-memory limiter if Redis is offline."""
        # Point to an unreachable port to simulate network failure
        limiter = RedisSlidingWindowRateLimiter(
            redis_url="redis://localhost:65534/0",
            limit_per_minute=5,
        )
        # Should not raise exception; instead falls back cleanly
        ok, rem, reset_in = await limiter.check_rate_limit("client-failover-test")
        assert ok is True
        assert rem == 4

    @pytest.mark.asyncio
    async def test_redis_rate_limiter_atomic_concurrency_cannot_exceed_limit(self):
        """
        Verifies that concurrent requests cannot exceed the configured limit
        under atomic sliding-window Redis Lua execution against a real Redis instance.
        """
        import asyncio
        import uuid
        import redis.asyncio as aioredis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = aioredis.from_url(redis_url, decode_responses=True)
        try:
            await client.ping()
        except Exception:
            pytest.skip("Real Redis instance not reachable at " + redis_url)

        limit = 10
        limiter = RedisSlidingWindowRateLimiter(
            redis_url=redis_url,
            limit_per_minute=limit,
            window_seconds=60.0,
            key_prefix="test_concurrent_rl:",
        )
        test_client_id = f"concurrent-client-{uuid.uuid4().hex[:8]}"

        # Clean key prior to test
        await client.delete(f"test_concurrent_rl:{test_client_id}")

        # Fire 25 concurrent requests simultaneously
        tasks = [limiter.check_rate_limit(test_client_id) for _ in range(25)]
        results = await asyncio.gather(*tasks)

        allowed_count = sum(1 for is_allowed, _, _ in results if is_allowed)
        denied_count = sum(1 for is_allowed, _, _ in results if not is_allowed)

        # Exactly `limit` requests must be allowed, and remaining must be denied
        assert allowed_count == limit, f"Expected exactly {limit} allowed, got {allowed_count}"
        assert denied_count == 15, f"Expected exactly 15 denied, got {denied_count}"

        # Clean up
        await client.delete(f"test_concurrent_rl:{test_client_id}")
        await client.aclose()
        await limiter.redis.aclose()
