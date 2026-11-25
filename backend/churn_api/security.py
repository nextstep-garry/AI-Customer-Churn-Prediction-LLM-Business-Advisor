"""API-key check and a small in-process rate limiter for /v1/* routes.

The limiter is per process: fine for one worker or a single container. Behind several
workers or replicas, enforce limits at the reverse proxy instead.
"""
from __future__ import annotations

import hmac
import threading
import time

from flask import Flask, request

from .errors import ApiError


class _FixedWindowLimiter:
    def __init__(self, per_minute: int):
        self.per_minute, self._hits, self._lock = per_minute, {}, threading.Lock()

    def check(self, key: str) -> int | None:
        """Return None if allowed, else the seconds until the window resets."""
        now = time.monotonic()
        with self._lock:
            start, count = self._hits.get(key, (now, 0))
            if now - start >= 60:
                start, count = now, 0
            if count >= self.per_minute:
                return max(1, int(60 - (now - start)))
            self._hits[key] = (start, count + 1)
            if len(self._hits) > 10_000:  # bound memory
                self._hits = {k: v for k, v in self._hits.items() if now - v[0] < 60}
        return None


def init_security(app: Flask, api_key: str, rate_limit_per_min: int) -> None:
    limiter = _FixedWindowLimiter(rate_limit_per_min) if rate_limit_per_min > 0 else None

    @app.before_request
    def _guard():
        if not request.path.startswith("/v1/"):
            return None
        supplied = request.headers.get("X-API-Key", "")
        if api_key and not hmac.compare_digest(supplied.encode(), api_key.encode()):
            raise ApiError(401, "unauthorized", "Missing or invalid X-API-Key header.")
        if limiter:
            wait = limiter.check(supplied or request.remote_addr or "anon")
            if wait is not None:
                raise ApiError(429, "rate_limited", f"Too many requests. Retry in {wait}s.",
                               headers={"Retry-After": str(wait)})
        return None
