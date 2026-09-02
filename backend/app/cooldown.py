"""
Shared per-provider cooldown, used by both llm.py's text chain and
vision.py's image chain.

Without this, a provider that just failed with a rate limit or an exhausted
quota gets tried again - and fails again - on every single subsequent
request, wasting a round-trip before falling through to the next provider
each time. This puts it in cooldown instead: skipped for
PROVIDER_COOLDOWN_SECONDS, then eligible again.

In-memory, per-process - same caveat as ratelimit.py: correct for the single
uvicorn worker this app runs as; multiple workers would each track their own
cooldowns independently.
"""
from __future__ import annotations

import time
from threading import Lock

from .config import settings

_lock = Lock()
_cooldowns: dict[str, float] = {}  # provider name -> monotonic time it's clear again

_COOLDOWN_TRIGGERS = ("rate limit", "credit", "quota")


def mark_failed(provider: str, reason: str) -> None:
    """Put `provider` in cooldown if `reason` (from llm.describe_failure)
    looks like something an immediate retry won't fix. A one-off timeout or
    a transient 500 isn't cooled down - only conditions that are likely
    still true on the very next request.
    """
    if not any(trigger in reason.lower() for trigger in _COOLDOWN_TRIGGERS):
        return
    with _lock:
        _cooldowns[provider] = time.monotonic() + settings.provider_cooldown_seconds


def filter_available(chain: list[str]) -> list[str]:
    """`chain` with any provider still in cooldown removed - unless that
    would empty it out entirely, in which case the original chain is
    returned unfiltered. Refusing to even try when every provider happens to
    be cooling down would turn a "probably still rate limited" guess into a
    hard outage; better to attempt and find out.
    """
    now = time.monotonic()
    with _lock:
        available = [p for p in chain if _cooldowns.get(p, 0.0) <= now]
    return available or chain
