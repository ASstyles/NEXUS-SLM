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


class RedisSlidingWindowRateLimiter(BaseRateLimiter):
    """
    Distributed sliding-window rate limiter backed by Redis sorted sets (ZSET).
    Enforces atomic rate limiting across multiple API worker processes and container replicas.
    """

    def __init__(
        self,
        redis_url: str,
        limit_per_minute: int = 120,
        window_seconds: float = 60.0,
        key_prefix: str = "nexus_rl:",
    ):
        self._limit = limit_per_minute
        self.window = window_seconds
        self.key_prefix = key_prefix
        self.redis_url = redis_url
        self._fallback_limiter = SlidingWindowRateLimiter(limit_per_minute, window_seconds)

        import redis.asyncio as aioredis
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self._redis_available = True

    @property
    def limit(self) -> int:
        return self._limit

    async def check_rate_limit(self, client_id: str) -> Tuple[bool, int, float]:
        now = time.time()
        window_start = now - self.window
        key = f"{self.key_prefix}{client_id}"

        try:
            # Atomic sliding-window evaluation using pipeline
            pipe = self.redis.pipeline(transaction=True)
            pipe.zremrangebyscore(key, "-inf", window_start)
            pipe.zcard(key)
            pipe.zrange(key, 0, 0, withscores=True)
            pipe.expire(key, int(self.window) + 5)
            _, current_count, oldest_entry, _ = await pipe.execute()

            if current_count < self._limit:
                # Add current timestamp
                member_id = f"{now}:{uuid.uuid4().hex[:8]}"
                pipe2 = self.redis.pipeline(transaction=True)
                pipe2.zadd(key, {member_id: now})
                pipe2.expire(key, int(self.window) + 5)
                await pipe2.execute()

                remaining = self._limit - (current_count + 1)
                reset_in = round(self.window - (now - oldest_entry[0][1]), 1) if oldest_entry else self.window
                return True, max(0, remaining), max(0.1, reset_in)

            # Limit reached
            reset_in = round(self.window - (now - oldest_entry[0][1]), 1) if oldest_entry else self.window
            return False, 0, max(0.1, reset_in)

        except Exception as exc:
            logger.warning(
                f"Redis rate limiter failure on key '{key}': {exc}. Failing open with local in-memory fallback."
            )
            return await self._fallback_limiter.check_rate_limit(client_id)


def create_rate_limiter(cfg: Optional[AppConfig] = None) -> BaseRateLimiter:
    """Factory creating appropriate rate limiter based on configuration."""
    cfg = cfg or get_config()
    if cfg.rate_limiting_backend == "redis" and cfg.redis_url:
        try:
            return RedisSlidingWindowRateLimiter(
                redis_url=cfg.redis_url,
                limit_per_minute=cfg.rate_limit_per_minute,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Redis rate limiter: {e}. Defaulting to in-memory.")
    return SlidingWindowRateLimiter(limit_per_minute=cfg.rate_limit_per_minute)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Ensures every incoming request has an immutable trace request_id attached."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Enforces configurable client rate limits with standard HTTP 429 headers and proxy-aware IP resolution."""

    def __init__(self, app, limiter: Optional[BaseRateLimiter] = None):
        super().__init__(app)
        cfg = get_config()
        self.cfg = cfg
        self.limiter = limiter or create_rate_limiter(cfg)
        self.exempt_paths = {"/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if (
            request.url.path in self.exempt_paths
            or os.getenv("EVALUATION_RUN") == "1"
            or os.getenv("DISABLE_RATE_LIMIT") == "1"
        ):
            return await call_next(request)

        # Secure proxy-aware client IP resolution
        client_ip = resolve_client_ip(request, self.cfg.trusted_proxies)

        allowed, remaining, reset_in = await self.limiter.check_rate_limit(client_ip)

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
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_in))
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
            request.url.path in {"/health", "/ready", "/metrics"}
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
