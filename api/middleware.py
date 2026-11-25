"""Production middleware stack for NEXUS-SLM: Request Tracing, Rate Limiting, Timeout, and Concurrency Control."""

import abc
import asyncio
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_client_ip(request: Request, trusted_proxies: List[str]) -> str:
    """
    Securely resolves the client IP address.
    Only trusts X-Forwarded-For if the direct peer IP is explicitly listed in trusted_proxies.
    Prevents IP spoofing attacks designed to bypass rate limits or geofencing.
    """
    peer_ip = request.client.host if request.client else "127.0.0.1"
    trusted_set = set(trusted_proxies)

    # Local loopbacks are considered trusted for testing and internal reverse proxies
    is_trusted = (peer_ip in trusted_set) or (peer_ip in ("127.0.0.1", "::1", "localhost"))

    forwarded = request.headers.get("X-Forwarded-For")
    if is_trusted and forwarded:
        # Leftmost entry in X-Forwarded-For represents the original client IP
        client_ip = forwarded.split(",")[0].strip()
        if client_ip:
            return client_ip

    return peer_ip


class RateLimiterUnavailableError(RuntimeError):
    """Raised when distributed Redis rate limiter fails under strict_reject policy."""


class BaseRateLimiter(abc.ABC):
    """Abstract interface for rate limiting implementations."""

    @abc.abstractmethod
    async def check_rate_limit(self, client_id: str) -> Tuple[bool, int, float]:
        """Returns (is_allowed, remaining_requests, reset_time_seconds)."""
        pass

    @property
    @abc.abstractmethod
    def limit(self) -> int:
        pass

    def get_status(self) -> Dict[str, object]:
        """Returns operational backend mode and health diagnostics."""
        return {
            "backend": "memory",
            "mode": "memory_local",
            "last_request_mode": "memory_local",
            "healthy": True,
            "degraded": False,
        }


class SlidingWindowRateLimiter(BaseRateLimiter):
    """
    Thread-safe sliding-window rate limiter per client ID in local memory.
    Explicitly designated for local development, testing, and single-replica deployments.
    """

    def __init__(self, limit_per_minute: int = 120, window_seconds: float = 60.0):
        self._limit = limit_per_minute
        self.window = window_seconds
        self.history: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    async def check_rate_limit(self, client_id: str) -> Tuple[bool, int, float]:
        """Returns (is_allowed, remaining_requests, reset_time_seconds)."""
        now = time.time()
        with self.lock:
            q = self.history[client_id]
            # Evict timestamps older than sliding window
            while q and q[0] <= now - self.window:
                q.popleft()

            if len(q) < self._limit:
                q.append(now)
                remaining = self._limit - len(q)
                reset_in = round(self.window - (now - q[0]), 1) if q else self.window
                return True, remaining, max(0.1, reset_in)

            # Limit exceeded
            reset_in = round(self.window - (now - q[0]), 1)
            return False, 0, max(0.1, reset_in)


SLIDING_WINDOW_LUA_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member_id = ARGV[4]
local clear_before = now - window

-- 1. Remove expired timestamps outside sliding window
redis.call('ZREMRANGEBYSCORE', key, '-inf', clear_before)

-- 2. Count timestamps currently within window
local current_count = redis.call('ZCARD', key)

-- 3. Atomically check limit and insert if permitted
if current_count < limit then
    redis.call('ZADD', key, now, member_id)
    redis.call('EXPIRE', key, math.ceil(window) + 5)
    local remaining = limit - (current_count + 1)

    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local reset_in = window
    if oldest and #oldest >= 2 then
        reset_in = math.max(0.1, tonumber(oldest[2]) + window - now)
    end
    return {1, remaining, tostring(reset_in)}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local reset_in = window
    if oldest and #oldest >= 2 then
        reset_in = math.max(0.1, tonumber(oldest[2]) + window - now)
    end
    return {0, 0, tostring(reset_in)}
end
"""


class RedisSlidingWindowRateLimiter(BaseRateLimiter):
    """
    Distributed sliding-window rate limiter backed by Redis sorted sets (ZSET).
    Enforces atomic rate limiting across multiple API worker processes and container replicas
    via single-transaction Redis Lua script execution, with explicit operational failure
    modes ('degraded_local' or 'strict_reject') and automatic recovery probing.
    """

    def __init__(
        self,
        redis_url: str,
        limit_per_minute: int = 120,
        window_seconds: float = 60.0,
        key_prefix: str = "nexus_rl:",
        failure_mode: str = "degraded_local",
        recovery_cooldown_seconds: float = 5.0,
    ):
        self._limit = limit_per_minute
        self.window = window_seconds
        self.key_prefix = key_prefix
        self.redis_url = redis_url
        self.failure_mode = failure_mode if failure_mode in ("degraded_local", "strict_reject") else "degraded_local"
        self.recovery_cooldown_seconds = recovery_cooldown_seconds
        self._fallback_limiter = SlidingWindowRateLimiter(limit_per_minute, window_seconds)

        import redis.asyncio as aioredis
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self.mode: str = "redis_distributed"
        self._last_request_mode: str = "redis_distributed"
        self._degraded: bool = False
        self.degraded_since: Optional[float] = None
        self.last_error: Optional[str] = None
        self._last_probe_ts: float = 0.0
        self.recovery_attempts: int = 0
        self.recovery_successes: int = 0
        self.fallback_requests_served: int = 0
        self.strict_rejections: int = 0

    @property
    def limit(self) -> int:
        return self._limit

    def get_status(self) -> Dict[str, object]:
        return {
            "backend": "redis",
            "mode": self.mode,
            "last_request_mode": self._last_request_mode,
            "failure_policy": self.failure_mode,
            "healthy": not self._degraded,
            "degraded": self._degraded,
            "degraded_since_epoch": self.degraded_since,
            "last_error": self.last_error,
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
            "fallback_requests_served": self.fallback_requests_served,
            "strict_rejections": self.strict_rejections,
        }

    async def check_rate_limit(self, client_id: str) -> Tuple[bool, int, float]:
        now = time.time()
        key = f"{self.key_prefix}{client_id}"
        member_id = f"{now}:{uuid.uuid4().hex[:8]}"

        # Under degraded_local, avoid blocking every request on dead sockets during cooldown
        if (
            self._degraded
            and self.failure_mode == "degraded_local"
            and (now - self._last_probe_ts) < self.recovery_cooldown_seconds
        ):
            self.mode = "degraded_local"
            self._last_request_mode = "degraded_local"
            self.fallback_requests_served += 1
            return await self._fallback_limiter.check_rate_limit(client_id)

        try:
            if self._degraded:
                self.recovery_attempts += 1
                self._last_probe_ts = now

            # Atomic single-transaction sliding-window evaluation via Redis Lua script
            res = await self.redis.eval(
                SLIDING_WINDOW_LUA_SCRIPT,
                1,
                key,
                str(now),
                str(self.window),
                str(self._limit),
                member_id,
            )
            if self._degraded:
                logger.info(
                    f"Redis rate limiter recovered from {self.mode} to redis_distributed (key prefix '{self.key_prefix}')."
                )
                self._degraded = False
                self.degraded_since = None
                self.last_error = None
                self.recovery_successes += 1

            self.mode = "redis_distributed"
            self._last_request_mode = "redis_distributed"
            # res is [is_allowed_int, remaining_int, reset_in_str]
            is_allowed = bool(res[0] == 1)
            remaining = int(res[1])
            reset_in = round(float(res[2]), 1)
            return is_allowed, max(0, remaining), max(0.1, reset_in)

        except Exception as exc:
            self.last_error = str(exc)
            self._last_probe_ts = now
            if not self._degraded:
                self._degraded = True
                self.degraded_since = now

            if self.failure_mode == "strict_reject":
                self.mode = "strict_reject"
                self._last_request_mode = "strict_reject"
                self.strict_rejections += 1
                logger.error(
                    f"Redis rate limiter failure on key '{key}': {exc}. Failing closed under strict_reject policy."
                )
                raise RateLimiterUnavailableError(
                    f"Distributed rate limiting store unavailable: {exc}"
                ) from exc

            self.mode = "degraded_local"
            self._last_request_mode = "degraded_local"
            self.fallback_requests_served += 1
            logger.warning(
                f"Redis rate limiter failure on key '{key}': {exc}. Operating in degraded_local mode."
            )
            return await self._fallback_limiter.check_rate_limit(client_id)


_active_rate_limiter_instance: Optional[BaseRateLimiter] = None


def get_active_rate_limiter_status() -> Dict[str, object]:
    """Returns current operational status of active API rate limiter."""
    if _active_rate_limiter_instance is None:
        return {"backend": "uninitialized", "mode": "unknown", "healthy": True, "degraded": False}
    return _active_rate_limiter_instance.get_status()


def create_rate_limiter(cfg: Optional[AppConfig] = None) -> BaseRateLimiter:
    """Factory creating appropriate rate limiter based on configuration."""
    global _active_rate_limiter_instance
    cfg = cfg or get_config()
    limiter: BaseRateLimiter
    if cfg.rate_limiting_backend == "redis" and cfg.redis_url:
        try:
            limiter = RedisSlidingWindowRateLimiter(
                redis_url=cfg.redis_url,
                limit_per_minute=cfg.rate_limit_per_minute,
                failure_mode=cfg.rate_limit_failure_mode,
                recovery_cooldown_seconds=cfg.redis_recovery_cooldown_seconds,
            )
            _active_rate_limiter_instance = limiter
            return limiter
        except Exception as e:
            logger.warning(f"Failed to initialize Redis rate limiter: {e}. Defaulting to in-memory.")
    limiter = SlidingWindowRateLimiter(limit_per_minute=cfg.rate_limit_per_minute)
    _active_rate_limiter_instance = limiter
    return limiter


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Ensures every incoming request has an immutable trace request_id attached."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Enforces configurable client rate limits with standard HTTP 429/503 headers and proxy-aware IP resolution."""

    def __init__(self, app, limiter: Optional[BaseRateLimiter] = None):
        super().__init__(app)
        global _active_rate_limiter_instance
        cfg = get_config()
        self.cfg = cfg
        self.limiter = limiter or create_rate_limiter(cfg)
        _active_rate_limiter_instance = self.limiter
        self.exempt_paths = {"/health", "/ready", "/live", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if (
            request.url.path in self.exempt_paths
            or os.getenv("EVALUATION_RUN") == "1"
            or os.getenv("DISABLE_RATE_LIMIT") == "1"
        ):
            return await call_next(request)

        # Secure proxy-aware client IP resolution
        client_ip = resolve_client_ip(request, self.cfg.trusted_proxies)

        try:
            allowed, remaining, reset_in = await self.limiter.check_rate_limit(client_ip)
        except RateLimiterUnavailableError as exc:
            req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Rate limiting service unavailable",
                    "error_code": "RATE_LIMITER_UNAVAILABLE",
                    "detail": "Distributed rate limiter unreachable under strict_reject failure policy.",
                    "request_id": req_id,
                    "retry_after_seconds": 5,
                },
                headers={
                    "Retry-After": "5",
                    "X-RateLimit-Mode": "strict_reject",
                    "X-RateLimit-Degraded": "true",
                },
            )

        rl_status = self.limiter.get_status()
        rl_mode = str(rl_status.get("last_request_mode", rl_status.get("mode", "memory_local")))
        extra_headers = {"X-RateLimit-Mode": rl_mode}
        if rl_status.get("degraded"):
            extra_headers["X-RateLimit-Degraded"] = "true"

        if not allowed:
            req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            logger.warning(f"Rate limit exceeded for client {client_ip} on {request.url.path} [ID={req_id}]")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "detail": f"Too many requests. Limit is {self.limiter.limit} requests per minute.",
                    "request_id": req_id,
                    "retry_after_seconds": reset_in,
                },
                headers={
                    "Retry-After": str(int(reset_in)),
                    "X-RateLimit-Limit": str(self.limiter.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_in)),
                    **extra_headers,
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_in))
        for hk, hv in extra_headers.items():
            response.headers[hk] = hv
        return response


class ConcurrencyAndTimeoutMiddleware(BaseHTTPMiddleware):
    """Protects against connection pool exhaustion and enforces request timeouts."""

    def __init__(self, app, timeout_seconds: float = 30.0, max_concurrency: int = 25):
        super().__init__(app)
        self.timeout = timeout_seconds
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if (
            request.url.path in {"/health", "/ready", "/live", "/metrics"}
            or os.getenv("EVALUATION_RUN") == "1"
            or os.getenv("DISABLE_CONCURRENCY_LIMIT") == "1"
            or self.max_concurrency <= 0
        ):
            return await call_next(request)

        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        acquired = False
        sem = self._get_semaphore()
        try:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=2.0)
                acquired = True
            except asyncio.TimeoutError:
                logger.error(f"Concurrency queue full, rejected request [ID={req_id}]")
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "error": "Inference server busy",
                        "error_code": "CONCURRENCY_LIMIT_EXCEEDED",
                        "detail": "Maximum concurrent request capacity reached. Please retry shortly.",
                        "request_id": req_id,
                    },
                )

            try:
                response = await asyncio.wait_for(call_next(request), timeout=self.timeout)
                return response
            except asyncio.TimeoutError:
                logger.error(f"Request execution timed out after {self.timeout}s [ID={req_id}]")
                return JSONResponse(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    content={
                        "error": "Request timed out",
                        "error_code": "REQUEST_TIMEOUT",
                        "detail": f"Inference execution exceeded maximum threshold of {self.timeout} seconds.",
                        "request_id": req_id,
                    },
                )
        finally:
            if acquired:
                sem.release()
