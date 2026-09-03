"""Shared per-identity limiters and the typed 429 envelope they raise.

Two shapes, both keyed by a caller-supplied identity string:

  FixedWindow  counter that resets when its window rolls — "N per T seconds".
  InFlight     concurrency gauge — "N at once", acquire/release around work.

Both read their ceiling from `config` on EVERY call, which is what makes the
matching AdminConfig fields hot: an operator raising a limit on /settings sees
it apply to the next request with no restart and no bucket reset. A ceiling of
0 means unlimited and short-circuits before any bookkeeping, so a single-user
box pays nothing for machinery it does not want.

SINGLE PROCESS ONLY. All state is a module-level dict, so with
SERVER_WORKERS > 1 each worker enforces its own copy and every budget here is
effectively multiplied by the worker count. WITHIN a process the counters are
thread-safe (each instance guards its state with a threading.Lock), so a
`def` route or dependency running in the threadpool may call them alongside
the async callers. That is the same honest caveat the
hand-rolled limiter this module replaced carried: the threat model is "runaway
script / accidental double-click / one client starving the others", not a
motivated attacker spreading load across workers. Anything stronger needs
shared state (Redis) and does not belong in-process.

Imports are deliberately limited to stdlib + fastapi + config: captures_routes
and reports_routes import this module and `main` imports them, so importing
`main` here would close a cycle and break the app at startup.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from fastapi import HTTPException

from faster_whisper_backend import config as cfg


# Every limiter instance registers here so reset_all() below can clear the
# whole tree in one call — the single hook tests need (conftest calls it
# between cases; without it a limit tripped in one test 429s the next).
_ALL: list = []

# Ceiling on the number of distinct keys one FixedWindow retains. Reached only
# by a caller inventing identities (or a very large user base); the sweep in
# _prune keeps memory bounded either way.
_MAX_KEYS = 4096


def reset_all() -> None:
    """Drop all counters in every registered limiter."""
    for limiter in _ALL:
        limiter.clear()


class RateLimited(HTTPException):
    """429 carrying WHICH config field refused and when to retry.

    Subclasses HTTPException so an un-handled path still renders a sane 429
    with a Retry-After header through Starlette's built-in handler; main
    registers a handler for this exact type to emit body() instead.
    """

    def __init__(self, *, message: str, config_field: str,
                 retry_after: int) -> None:
        self.message = message
        self.config_field = config_field
        self.retry_after = int(retry_after)
        super().__init__(429, message,
                         headers={"Retry-After": str(self.retry_after)})

    def body(self) -> dict:
        """The JSON response body: an OpenAI-shaped `error` object naming the
        config field an admin would raise, plus a `detail` sibling.

        `detail` is NOT redundant. Several toast handlers in this tree
        (reports_routes.py, captures_routes.py, quick_config_routes.py) read
        `j.detail` off a failed response — they are written against
        FastAPI's default error shape, and without the sibling every one of
        them would degrade to showing a bare status code instead of the
        sentence explaining what to do about it.
        """
        return {
            "error": {
                "message": self.message,
                "type": "rate_limit_exceeded",
                "param": self.config_field,
                "retry_after": self.retry_after,
            },
            "detail": self.message,
        }


def identity_key(user: "dict[str, Any] | None", request: Any) -> str:
    """Who to charge a request to: user id, else API key id, else client host.

    The key_id rung matters for machine clients that authenticate with a key
    carrying no user, and the host rung is the last resort for anything with
    neither — several callers behind one NAT then share a bucket, which is the
    conservative direction. `request` may be a Starlette Request OR a
    WebSocket; only `.client` is touched, which both carry. Never returns ""
    (an empty key would merge every anonymous caller with a legitimate one
    whose id happened to be blank).
    """
    user = user or {}
    key = user.get("user_id") or user.get("key_id") or ""
    if not key:
        client = getattr(request, "client", None)
        key = getattr(client, "host", "") or ""
    return key or "<unknown>"


class FixedWindow:
    """N events per `window_s` seconds, per key. Counter resets on roll.

    Fixed window, not sliding: a caller can spend the budget at the end of one
    window and again at the start of the next. That burst is acceptable for
    every use here (these are flood guards, not quotas) and buys a two-tuple
    of state per key instead of a timestamp list.
    """

    def __init__(self, *, config_field: str, window_s: float,
                 default_max: int, message: str) -> None:
        self.config_field = config_field
        self.window_s = float(window_s)
        self.default_max = int(default_max)
        # Rendered with the LIVE limit and retry_after, so the sentence a user
        # sees always matches the config as it stands right now.
        self.message = message
        self._state: "dict[str, tuple[int, float]]" = {}
        # Guards _state: a `def` route/dependency runs in the threadpool, so
        # the read-modify-write below must not interleave with the loop's.
        self._lock = threading.Lock()
        _ALL.append(self)

    def limit(self) -> int:
        """Current ceiling, re-read from config on every call (hot field)."""
        return int(getattr(cfg, self.config_field, self.default_max))

    def hit(self, key: str) -> None:
        """Count one event and raise if that puts the key over the limit."""
        limit = self.limit()
        if limit <= 0:
            return
        key = key or "<unknown>"
        now = time.time()
        with self._lock:
            n, start = self._state.get(key, (0, now))
            if now - start > self.window_s:
                n, start = 0, now
            n += 1
            self._state[key] = (n, start)
            self._prune(key, now)
        # Raise OUTSIDE the lock: _reject only does arithmetic, and holding
        # the lock across an exception path buys nothing.
        if n > limit:
            raise self._reject(start, now, limit)

    def guard(self, key: str) -> None:
        """Raise if the key is ALREADY exhausted, without counting.

        For call sites that charge only failures (the login throttle): a
        legitimate caller must not be pushed further from recovery by the
        attempt that is about to succeed.
        """
        limit = self.limit()
        if limit <= 0:
            return
        key = key or "<unknown>"
        now = time.time()
        with self._lock:
            n, start = self._state.get(key, (0, now))
        if now - start > self.window_s:
            return
        if n >= limit:
            raise self._reject(start, now, limit)

    def penalize(self, key: str) -> bool:
        """Count one event without raising. Returns True when this call is the
        one that crossed the limit, so the caller can log the trip exactly
        once instead of on every subsequent attempt."""
        limit = self.limit()
        if limit <= 0:
            return False
        key = key or "<unknown>"
        now = time.time()
        with self._lock:
            n, start = self._state.get(key, (0, now))
            if now - start > self.window_s:
                n, start = 0, now
            n += 1
            self._state[key] = (n, start)
            self._prune(key, now)
        return n == limit

    def reset(self, key: str) -> None:
        """Forget a key's window entirely (a success clears its penalties)."""
        with self._lock:
            self._state.pop(key or "<unknown>", None)

    def clear(self) -> None:
        with self._lock:
            self._state.clear()

    def _reject(self, start: float, now: float, limit: int) -> RateLimited:
        retry_after = max(1, math.ceil(start + self.window_s - now))
        return RateLimited(
            message=self.message.format(limit=limit, retry_after=retry_after),
            config_field=self.config_field,
            retry_after=retry_after,
        )

    def _prune(self, live_key: str, now: float) -> None:
        """Bound the key dict. Runs AFTER the insert so the key just written is
        always present; it is excluded from both passes, since evicting the
        caller we are mid-decision about would silently reset their budget."""
        if len(self._state) <= _MAX_KEYS:
            return
        # Same shape as main._progress_set's sweep: drop everything whose
        # window has already rolled, then fall back to oldest-first.
        for k in [k for k, (_n, start) in self._state.items()
                  if k != live_key and now - start > self.window_s]:
            self._state.pop(k, None)
        while len(self._state) > _MAX_KEYS:
            oldest = min((k for k in self._state if k != live_key),
                         key=lambda k: self._state[k][1], default=None)
            if oldest is None:
                return
            self._state.pop(oldest, None)


class InFlight:
    """N concurrent holders per key: acquire before the work, release after.

    Unlike FixedWindow there is no deadline to report — a slot frees when
    whoever holds it finishes, which could be a second or a minute away.
    """

    # Advisory nudge only. There is no honest deadline for a concurrency cap,
    # and omitting Retry-After entirely makes well-behaved clients retry
    # immediately in a tight loop.
    RETRY_AFTER_S = 5

    def __init__(self, *, config_field: str, default_max: int,
                 message: str) -> None:
        self.config_field = config_field
        self.default_max = int(default_max)
        self.message = message
        self._counts: "dict[str, int]" = {}
        # Same contract as FixedWindow._lock: check-then-set must be atomic
        # or a threadpool caller over-admits a slot / loses a release.
        self._lock = threading.Lock()
        _ALL.append(self)

    def limit(self) -> int:
        """Current ceiling, re-read from config on every call (hot field)."""
        return int(getattr(cfg, self.config_field, self.default_max))

    def acquire(self, key: str) -> None:
        """Take a slot, or raise RateLimited when the key holds them all."""
        limit = self.limit()
        if limit <= 0:
            return
        key = key or "<unknown>"
        with self._lock:
            n = self._counts.get(key, 0)
            if n < limit:
                self._counts[key] = n + 1
                return
        # Compute the rejection under the lock, raise outside it.
        raise RateLimited(
            message=self.message.format(limit=limit,
                                        retry_after=self.RETRY_AFTER_S),
            config_field=self.config_field,
            retry_after=self.RETRY_AFTER_S,
        )

    def release(self, key: str) -> None:
        """Give a slot back. Deliberately defensive: a double release (or one
        after the limit was lowered to 0 mid-flight, where acquire was a
        no-op) must not drive a key negative and hand out free slots forever."""
        key = key or "<unknown>"
        with self._lock:
            n = self._counts.get(key, 0)
            if n <= 1:
                self._counts.pop(key, None)
                return
            self._counts[key] = max(0, n - 1)

    def count(self, key: str) -> int:
        with self._lock:
            return self._counts.get(key or "<unknown>", 0)

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()
