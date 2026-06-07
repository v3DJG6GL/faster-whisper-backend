"""
In-process request metrics for the /stats dashboard.

Bounded ring buffers + a few counters. No Prometheus, no external metrics
backend. All updates happen on the asyncio event loop (uvicorn runs
SERVER_WORKERS=1) so plain `Counter[k] += 1` and `deque.append` are safe
without explicit locking.

The "Recent transcriptions" widget on /stats is sourced from the durable
`transcriptions_store` SQLite database (same source as /quick-config's
trace panel) — see metrics_snapshot() below. Survives restart.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from typing import Any

logger = logging.getLogger("whisper-api")

START_TS = time.time()

_LATENCY_MAX = 200          # ring for p50/p95/p99
_ERROR_WINDOW_SEC = 15 * 60
_MODEL_LOAD_KEEP = 50       # bounded per-model history

# Long-lived stream paths whose duration would dominate latency stats.
SSE_PATHS = frozenset({"/logs/stream", "/stats/stream"})

req_count: Counter[str] = Counter()         # path -> total
err_count: Counter[str] = Counter()         # path -> 5xx total

# Bumped/dec'd by the transcribe handler with try/finally.
in_flight_transcriptions: int = 0

# Global latency ring (ms) used for p50/p95/p99.
_latency: deque[float] = deque(maxlen=_LATENCY_MAX)

# 5xx timestamps for rolling 1/5/15 min windows.
_errors_ts: deque[float] = deque()

# Cold-load durations per model name. Bounded to last _MODEL_LOAD_KEEP each.
model_loads: dict[str, list[float]] = {}


def record_request(path: str, status: int, duration_ms: float) -> None:
    """Called by the FastAPI middleware on every HTTP request."""
    req_count[path] += 1
    if status >= 500:
        err_count[path] += 1
        now = time.time()
        _errors_ts.append(now)
        cutoff = now - _ERROR_WINDOW_SEC
        while _errors_ts and _errors_ts[0] < cutoff:
            _errors_ts.popleft()
    if path not in SSE_PATHS:
        _latency.append(duration_ms)


def record_transcription(model: str, audio_dur: float, proc_dur: float,
                         status: str, words: int,
                         request_id: str | None = None,
                         user_id: str | None = None,
                         key_id: str | None = None) -> None:
    """Called from the transcribe handler's outer finally on every
    /transcribe request (both success and error paths). UPSERTs the
    timing half of the recent-transcriptions row keyed by request_id;
    record_trace() in quick_config_state has already inserted the rich
    half on the success path, so this call only patches timing fields
    in. On the error path it inserts a minimal row so /stats still
    counts the request.

    Also bumps the durable per-key/per-user usage rollup (usage_store),
    which — unlike recent_transcriptions — is never pruned to a rolling
    window, so it backs lifetime totals on /api-keys and /stats."""
    if not request_id:
        return
    try:
        import config as cfg
        import transcriptions_store
        transcriptions_store.record_timing(
            request_id=request_id,
            model=model,
            audio_dur_s=round(audio_dur, 3) if audio_dur else None,
            proc_dur_s=round(proc_dur, 3),
            status=status,
            words_count=int(words or 0),
            user_id=user_id,
            prune_every=int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PRUNE_EVERY", 50)),
            max_rows=int(getattr(cfg, "RECENT_TRANSCRIPTIONS_MAX", 500)),
            ttl_days=float(getattr(cfg, "RECENT_TRANSCRIPTIONS_TTL_DAYS", 30)),
        )
    except Exception as e:
        logger.warning("[metrics] record_transcription persist failed: %s", e)
    try:
        import usage_store
        usage_store.record_usage(
            key_id=key_id,
            user_id=user_id,
            audio_s=audio_dur or 0.0,
            words=int(words or 0),
            status=status,
        )
    except Exception as e:
        logger.warning("[metrics] usage rollup failed: %s", e)


def record_model_load(model: str, load_seconds: float) -> None:
    """Called once per WhisperModel(...) construction in _get_or_load_model."""
    bucket = model_loads.setdefault(model, [])
    bucket.append(load_seconds)
    # Preserve bucket[0] as the canonical first cold-load forever; trim
    # the middle so the bucket fits in _MODEL_LOAD_KEEP. The UI shows
    # first + last-N-avg, both of which depend on bucket[0] surviving.
    if len(bucket) > _MODEL_LOAD_KEEP:
        del bucket[1 : len(bucket) - _MODEL_LOAD_KEEP + 1]


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank quantile. Fine for N <= 200 and human display."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _errors_in(seconds: float) -> int:
    cutoff = time.time() - seconds
    # _errors_ts is append-ordered; iterate from newest. Bounded by
    # _ERROR_WINDOW_SEC of traffic so this stays O(window) at worst.
    n = 0
    for t in reversed(_errors_ts):
        if t < cutoff:
            break
        n += 1
    return n


def metrics_snapshot() -> dict[str, Any]:
    """Build the JSON payload returned by /stats/snapshot and /stats/stream."""
    durations = sorted(_latency)
    loads_summary = {}
    for m, v in model_loads.items():
        if not v:
            continue
        tail = v[-5:]
        loads_summary[m] = {
            "first": round(v[0], 2),
            "last5_avg": round(sum(tail) / len(tail), 2),
            "count": len(v),
        }
    try:
        import config as cfg
        import transcriptions_store
        limit = int(getattr(cfg, "STATS_RECENT_TRANSCRIPTIONS_COUNT", 20))
        rows = transcriptions_store.list_recent(limit=max(1, limit))
    except Exception as e:
        logger.warning("[metrics] list_recent failed: %s", e)
        rows = []
    # /stats is page-scoped {none, all} (no scope='own'); a non-admin
    # holder of pages.stats='all' must not be able to read other users'
    # transcripts via this widget. Project to the timing-only shape the
    # /stats JS actually renders and coerce nulls to numeric defaults so
    # `r.audio_dur.toFixed(1)` on error-path rows doesn't freeze the
    # live view.
    recent = [
        {
            "ts": r.get("ts"),
            "model": r.get("model") or "",
            "audio_dur": r.get("audio_dur") or 0.0,
            "proc_dur": r.get("proc_dur") or 0.0,
            "rtf": r.get("rtf"),
            "words": r.get("words") or 0,
            "status": r.get("status") or "error",
        }
        for r in rows
    ]
    return {
        "uptime_sec": round(time.time() - START_TS, 1),
        "in_flight_transcriptions": in_flight_transcriptions,
        "requests": dict(req_count),
        "errors_total": dict(err_count),
        "errors_window": {
            "1m": _errors_in(60),
            "5m": _errors_in(300),
            "15m": _errors_in(900),
        },
        "latency_ms": {
            "n": len(durations),
            "p50": round(_quantile(durations, 0.50), 1),
            "p95": round(_quantile(durations, 0.95), 1),
            "p99": round(_quantile(durations, 0.99), 1),
        },
        "recent_transcriptions": recent,
        "model_loads": loads_summary,
    }
