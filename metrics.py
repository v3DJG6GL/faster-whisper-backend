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

# Cap on distinct UNMATCHED-route keys (404 / pre-routing) in req_count.
# Matched routes carry a templated path (a small finite set) and are always
# counted under their own key; unmatched requests carry the raw client URL, so
# a flood of distinct nonexistent paths would otherwise grow these counters
# without bound — and inflate every /stats snapshot plus the 1 Hz SSE frame.
# The check compares len(req_count) — the TOTAL key count, matched keys
# included — so the effective unmatched budget is the cap minus however many
# matched routes have been hit (and the dict tops out at cap + #routes).
# Past it, further NEW unmatched paths fold into one sentinel key.
_MAX_UNMATCHED_KEYS = 512
_UNMATCHED_OVERFLOW = "(other)"

# Cap on the LENGTH of one such key. The key count alone doesn't bound the
# memory an unmatched flood can pin: the raw client URL is attacker-chosen and
# can be kilobytes long, and every retained key is re-serialised into each
# /stats snapshot and 1 Hz SSE frame. Applied before the cardinality check so
# what lands in req_count is always the truncated form. Comfortably longer
# than any real route.
_MAX_UNMATCHED_KEY_LEN = 120

# Statuses the durable usage rollup must NOT count as errors. A client
# cancel is not a server error (main preserves 'cancelled' into the
# recent-jobs row for the same reason); usage_store's only error test is
# `status == "ok"`, so the mapping happens here, before the hand-off.
_USAGE_NON_ERROR_STATUSES = frozenset(("ok", "cancelled"))

# Recent-jobs `kind` → usage-rollup kind. The recent-jobs vocabulary predates
# the per-kind statistics and is what /stats renders, so it stays; the rollup
# wants the desktop app's four words. Batch callers say file/url directly via
# `usage_kind` (their recent-jobs kind is None → "transcribe").
_USAGE_KIND_BY_JOB_KIND = {"dictate": "dictation", "translate": "text"}

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


def record_request(path: str, status: int, duration_ms: float,
                   *, unmatched: bool = False) -> None:
    """Called by the FastAPI middleware on every HTTP request.

    `unmatched=True` marks a request that matched no route (404 / pre-routing
    failure): its `path` is the raw client URL, so it is capped in both length
    and count to bound the counters against a distinct-path flood (see
    _MAX_UNMATCHED_KEY_LEN / _MAX_UNMATCHED_KEYS)."""
    if unmatched:
        path = path[:_MAX_UNMATCHED_KEY_LEN]
    if (unmatched and path not in req_count
            and len(req_count) >= _MAX_UNMATCHED_KEYS):
        path = _UNMATCHED_OVERFLOW
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
                         key_id: str | None = None,
                         kind: str | None = None,
                         stages: list | None = None,
                         key_label: str | None = None,
                         job_id: str | None = None,
                         usage_kind: str | None = None,
                         language: str | None = None) -> None:
    """Called from the transcribe handler's outer finally on every
    /transcribe request (both success and error paths). UPSERTs the
    timing half of the recent-transcriptions row keyed by request_id;
    record_trace() in quick_config_state has already inserted the rich
    half on the success path, so this call only patches timing fields
    in. On the error path it inserts a minimal row so /stats still
    counts the request.

    ``key_label`` is the API key's display label as the auth record already
    holds it (``user["key_label"]``), snapshotted at record time because
    labels are mutable and read-time resolution would rewrite history.
    When the caller passes none, it is looked up from ``key_id``.

    Also bumps the durable per-key/per-user usage rollup (usage_store),
    which — unlike recent_transcriptions — is never pruned to a rolling
    window, so it backs lifetime totals on /api-keys and /stats and the
    desktop app's statistics. ``job_id`` groups the utterances of one
    dictation session (and names a batch run by its client progress id) so
    the rollup counts sessions, not phrases; ``usage_kind`` is the rollup's
    own kind word (dictation/file/url/text) when the recent-jobs ``kind``
    does not map to one. Structured stage keys (``targets``, ``speakers``,
    ``retained``, ``kept_original``) on the ``stages`` dicts feed the
    per-stage statistics; their ``detail`` strings never do."""
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
            kind=kind,
            stages=stages,
            key_label=key_label,
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
            status="ok" if status in _USAGE_NON_ERROR_STATUSES else status,
            kind=usage_kind or _USAGE_KIND_BY_JOB_KIND.get(kind or ""),
            job_id=job_id or request_id,
            stages=stages,
            model=model or None,
            language=language,
            proc_s=proc_dur,
        )
    except Exception as e:
        logger.warning("[metrics] usage rollup failed: %s", e)


def record_download(model: str, seconds: float, bytes_done: int, *,
                    status: str = "ok") -> None:
    """Persist a model download that moved bytes as a 'download' recent-jobs
    row (called by download_progress.capture's exit). ``status`` is 'ok'
    for a finished transfer and 'error' for one that died mid-way — the
    GB / MB/s figures are still the bytes that moved, but the stage detail
    says the transfer did not complete. Minted request_id — downloads have
    no request of their own. No usage rollup: a download is server work,
    not user throughput."""
    try:
        import uuid

        import config as cfg
        import transcriptions_store
        mb_s = (bytes_done / (1 << 20)) / seconds if seconds > 0 else 0.0
        detail = f"{bytes_done / (1 << 30):.2f} GB · {mb_s:.1f} MB/s"
        if status != "ok":
            detail += " · aborted"
        transcriptions_store.record_timing(
            request_id=uuid.uuid4().hex,
            model=model,
            audio_dur_s=None,
            proc_dur_s=round(seconds, 3),
            status=status,
            words_count=0,
            kind="download",
            stages=[{"name": "download", "secs": round(seconds, 3),
                     "model": model,
                     "detail": detail,
                     "bytes": int(bytes_done)}],
            prune_every=int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PRUNE_EVERY", 50)),
            max_rows=int(getattr(cfg, "RECENT_TRANSCRIPTIONS_MAX", 500)),
            ttl_days=float(getattr(cfg, "RECENT_TRANSCRIPTIONS_TTL_DAYS", 30)),
        )
    except Exception as e:
        logger.warning("[metrics] record_download persist failed: %s", e)


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


def metrics_snapshot(*, include_identity: bool = False,
                     user_id: "str | None" = None) -> dict[str, Any]:
    """Build the JSON payload returned by /stats/snapshot and /stats/stream.

    ``include_identity`` follows jobs.jobs_snapshot's gate: admins (and
    own-scope viewers, whose rows are all their own) get the username /
    key_label of the recent jobs. ``user_id`` narrows the recent rows to
    that owner (the /stats "own" page scope); None = every user."""
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
        rows = transcriptions_store.list_recent(limit=max(1, limit),
                                                user_id_filter=user_id)
    except Exception as e:
        logger.warning("[metrics] list_recent failed: %s", e)
        rows = []
    # A non-admin holder of pages.stats='all' sees every user's rows and
    # must not be able to read other users' transcripts (or identities)
    # via this widget. Project to the timing-only shape the
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
            # Recent-jobs fields. kind: explicit column wins; legacy rows
            # resolve via source ('stream' = live dictation). username and
            # key_label are display identities, not transcript content —
            # the projection still carries no raw/final text — but they
            # are OTHER users' identities, so like jobs_snapshot they are
            # scrubbed unless the viewer is an admin (include_identity).
            "kind": r.get("kind")
                    or ("dictate" if r.get("source") == "stream"
                        else "transcribe"),
            "username": (r.get("username") or "") if include_identity else "",
            "key_label": (r.get("key_label") or "") if include_identity else "",
            "stages": r.get("stages") or [],
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
