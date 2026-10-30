"""Production middleware stack for NEXUS-SLM: Request Tracing, Rate Limiting, Timeout, and Concurrency Control."""

import asyncio
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Dict, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Ensures every incoming request has an immutable trace request_id attached."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter per client IP."""

    def __init__(self, limit_per_minute: int = 120, window_seconds: float = 60.0):
        self.limit = limit_per_minute
        self.window = window_seconds
        self.history: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    async def check_rate_limit(self, client_ip: str) -> Tuple[bool, int, float]:
        """Returns (is_allowed, remaining_requests, reset_time_seconds)."""
        now = time.time()
        with self.lock:
            q = self.history[client_ip]
            # Evict timestamps older than sliding window
            while q and q[0] <= now - self.window:
                q.popleft()

            if len(q) < self.limit:
                q.append(now)
                remaining = self.limit - len(q)
                reset_in = round(self.window - (now - q[0]), 1) if q else self.window
                return True, remaining, max(0.1, reset_in)

            # Limit exceeded
            reset_in = round(self.window - (now - q[0]), 1)
            return False, 0, max(0.1, reset_in)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Enforces configurable client rate limits with standard HTTP 429 headers."""

    def __init__(self, app, limiter: Optional[SlidingWindowRateLimiter] = None):
        super().__init__(app)
        cfg = get_config()
        self.limiter = limiter or SlidingWindowRateLimiter(limit_per_minute=cfg.rate_limit_per_minute)
        # Paths excluded from rate limiting
        self.exempt_paths = {"/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if (
            request.url.path in self.exempt_paths
            or os.getenv("EVALUATION_RUN") == "1"
            or os.getenv("DISABLE_RATE_LIMIT") == "1"
        ):
            return await call_next(request)

        # Resolve client IP (support reverse proxies via X-Forwarded-For)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = "127.0.0.1"

        allowed, remaining, reset_in = await self.limiter.check_rate_limit(client_ip)

        if not allowed:
            req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            logger.warning(f"Rate limit exceeded for IP {client_ip} on {request.url.path} [ID={req_id}]")
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
        # Exempt health, ready, metrics, or evaluation runs
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
