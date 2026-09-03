"""1 Hz sampler behind the /stats "GPU busy" share and the range-mode
machine history.

Nothing on the server ticked without a client before this: /stats/stream
builds a payload only while an EventSource is open, so "how busy was the
GPU this afternoon" had no source. The lifespan runs loop(); each second
tick() appends 1/0 ("an inference slot is held") to metrics.busy_ring, and
every STATS_SYSTEM_METRICS_SAMPLE_S seconds it takes one machine sample (NVML +
psutil, blocking — hence to_thread) into a small queue that flush() writes
once a minute in one transaction to system_metrics_store (the
rolling operational DB, not the accounting one). prune() runs hourly with
STATS_SYSTEM_METRICS_RETENTION_DAYS.

Never per-tick writes: the ring is memory, samples are ~6 rows a minute.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from faster_whisper_backend import config as cfg
from faster_whisper_backend.stats import metrics
from faster_whisper_backend.runtime import system_stats
from faster_whisper_backend.stats import system_metrics_store

logger = logging.getLogger("whisper-api")

_pending: list[dict[str, Any]] = []
_ticks = 0
_last_flush = 0.0
_last_prune = 0.0
_last_warn = 0.0
FLUSH_EVERY_S = 60.0
PRUNE_EVERY_S = 3600.0


def sample_every() -> int:
    return max(1, int(getattr(cfg, "STATS_SYSTEM_METRICS_SAMPLE_S", 10) or 10))


def slot_busy_now() -> int:
    """1 when an inference slot is held (the GPU gate), else 0. Before the
    gate exists — no inference yet — the in-flight counter stands in."""
    gate = metrics.gpu_gate
    if gate is not None:
        return 1 if gate.held > 0 else 0
    return 1 if metrics.in_flight_transcriptions > 0 else 0


def tick(now: float | None = None) -> dict[str, Any] | None:
    """One second: append to the busy ring; on the sample cadence take a
    machine sample and queue it. Returns the sample when one was taken.
    Blocking (NVML / psutil) on sample ticks — run via to_thread."""
    global _ticks
    now = time.time() if now is None else float(now)
    metrics.busy_ring.append(slot_busy_now())
    _ticks += 1
    every = sample_every()
    if _ticks % every:
        return None
    s = sample(now)
    _pending.append(s)
    return s


def sample(now: float) -> dict[str, Any]:
    """One machine sample on the STATS_SYSTEM_METRICS_SAMPLE_S grid. NVML absent →
    the gpu fields are None; slot_busy is the busy share of the last
    sample window."""
    every = sample_every()
    gpu = None
    try:
        gpu = system_stats._build_gpu()
    except Exception:  # noqa: BLE001 — a missing GPU is a None sample
        gpu = None
    host: dict[str, Any] = {}
    try:
        host = system_stats._build_host() or {}
    except Exception:  # noqa: BLE001
        host = {}
    ring = list(metrics.busy_ring)
    window = ring[-every:]
    busy = (sum(window) / len(window)) if window else 0.0
    gpu = gpu or {}
    return {
        "ts": int(now) // every * every,
        "gpu_util": gpu.get("util_pct"),
        "gpu_mem_mb": gpu.get("mem_used_mb"),
        "gpu_temp": gpu.get("temp_c"),
        "cpu_pct": host.get("cpu_pct"),
        "ram_pct": host.get("ram_pct"),
        "slot_busy": round(busy, 3),
    }


def flush() -> int:
    """Write the queued samples in one transaction; returns the row count.
    A store failure keeps the rows for the next flush (bounded below)."""
    global _last_flush
    _last_flush = time.time()
    if not _pending:
        return 0
    rows = _pending[:]
    try:
        system_metrics_store.record(rows)
    except Exception as e:  # noqa: BLE001
        _warn("[stats-sampler] flush failed: %s", e)
        # Keep at most ten minutes of samples on a persistent failure.
        del _pending[:-60]
        return 0
    del _pending[:len(rows)]
    return len(rows)


def prune() -> int:
    global _last_prune
    _last_prune = time.time()
    days = int(getattr(cfg, "STATS_SYSTEM_METRICS_RETENTION_DAYS", 30) or 0)
    if days <= 0:
        return 0
    return system_metrics_store.prune(days)


def _warn(msg: str, *args: Any) -> None:
    """Rate-limited (one per minute) so a broken NVML cannot flood the log
    at 1 Hz."""
    global _last_warn
    now = time.time()
    if now - _last_warn < 60:
        return
    _last_warn = now
    logger.warning(msg, *args)


async def loop() -> None:
    """The lifespan task. Sampling and store work go through to_thread —
    NVML, psutil and SQLite block — so a slow probe never stalls the loop
    that serves every request. A failed tick is logged (rate-limited) and
    the loop goes on."""
    global _last_flush, _last_prune
    _last_flush = _last_prune = time.time()
    while True:
        await asyncio.sleep(1.0)
        now = time.time()
        try:
            await asyncio.to_thread(tick, now)
            if now - _last_flush >= FLUSH_EVERY_S:
                await asyncio.to_thread(flush)
            if now - _last_prune >= PRUNE_EVERY_S:
                await asyncio.to_thread(prune)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            _warn("[stats-sampler] tick failed: %s", e)


def _reset_for_tests() -> None:
    global _ticks, _last_flush, _last_prune, _last_warn
    _pending.clear()
    _ticks = 0
    _last_flush = _last_prune = _last_warn = 0.0
    metrics.busy_ring.clear()
