"""
Per-endpoint rate limiting: fixed-window counters, in-memory.

In-memory and per-process on purpose - this app runs as a single uvicorn
worker (see DEPLOYMENT.md), so one shared counter is correct. Running with
multiple workers would give each its own counters, multiplying the
effective limit - worth knowing if that ever changes.

Memory footprint is one (window_start, count) tuple per distinct (bucket,
identity) pair ever seen, which is small enough at this app's scale not to
need active eviction.
"""
from __future__ import annotations

import time
from threading import Lock

from fastapi import Depends, HTTPException, Request

from . import auth
from .auth import User
from .config import settings

_lock = Lock()
_buckets: dict[tuple[str, str], tuple[float, int]] = {}


def client_ip(request: Request) -> str:
    """Best-effort caller identity for unauthenticated routes.

    Honors X-Forwarded-For (set by the Nginx reverse proxy in production -
    see DEPLOYMENT.md) so rate limiting keys on the real client, not the
    proxy, on every request; falls back to the direct connection otherwise.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check(bucket: str, identity: str, limit: int, window_seconds: int) -> None:
    key = (bucket, identity)
    now = time.monotonic()
    with _lock:
        window_start, count = _buckets.get(key, (now, 0))
        if now - window_start >= window_seconds:
            window_start, count = now, 0
        count += 1
        _buckets[key] = (window_start, count)
        over = count > limit
        retry_after = max(1, int(window_start + window_seconds - now))
    if over:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests - try again in {retry_after}s",
            headers={"Retry-After": str(retry_after)},
        )


def by_ip(bucket: str, limit: int, window_seconds: int = 60):
    """Dependency factory for routes with no signed-in user yet (login, register)."""

    def dependency(request: Request) -> None:
        _check(bucket, client_ip(request), limit, window_seconds)

    return dependency


def by_user(bucket: str, limit: int, window_seconds: int = 60):
    """Dependency factory for authenticated routes.

    Depends on auth.current_user itself (rather than requiring the route to
    pass a user in) so it 401s before ever counting toward the limit - a
    request that never authenticated shouldn't spend an authenticated user's
    quota, and FastAPI dedupes this against the route's own
    Depends(auth.current_user) within one request, so it's not a second
    session lookup.
    """

    def dependency(request: Request, user: User = Depends(auth.current_user)) -> None:
        _check(bucket, f"user:{user.id}", limit, window_seconds)

    return dependency
