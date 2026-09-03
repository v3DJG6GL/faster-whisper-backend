"""SQLite store for the machine-load history behind /stats.

One row per STATS_SYSTEM_METRICS_SAMPLE_S grid second: GPU utilisation,
VRAM, GPU temperature, CPU, RAM and the busy decode-slot count, written in
batches by stats_sampler and read back downsampled by /stats/history. Rows
older than STATS_SYSTEM_METRICS_RETENTION_DAYS are pruned hourly.

Own file (STATS_SYSTEM_METRICS_DB) because this is rolling telemetry: it
shares neither the usage ledger's keep-for-years lifecycle nor the recent
transcriptions' plaintext-content sensitivity. Until 2026-09 the table lived
in the recent-transcriptions DB as `sys_samples`; adopt_legacy() moves those
rows over once.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Any

from faster_whisper_backend.core import store_common

logger = logging.getLogger("whisper-api")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS system_metrics (
  ts         INTEGER PRIMARY KEY,
  gpu_util   REAL,
  gpu_mem_mb REAL,
  gpu_temp   REAL,
  cpu_pct    REAL,
  ram_pct    REAL,
  slot_busy  REAL
);
"""

# Column names are interpolated into the history query; only these.
METRICS: frozenset[str] = frozenset(
    ("gpu_util", "gpu_mem_mb", "gpu_temp", "cpu_pct", "ram_pct", "slot_busy"))
_COLUMNS = ("ts", "gpu_util", "gpu_mem_mb", "gpu_temp", "cpu_pct", "ram_pct",
            "slot_busy")


def init_db(path: str) -> None:
    """Open (or create) the DB at `path` in WAL mode. Idempotent — call once
    on service startup before any other function in this module."""
    global _conn
    _conn = store_common.open_wal_db(path)
    _conn.executescript(_SCHEMA)
    store_common.secure_db_file(path)


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("system_metrics_store.init_db() was not called before use.")
    return _conn


def adopt_legacy(legacy: sqlite3.Connection) -> int:
    """One-time move of the pre-split `sys_samples` table out of another
    store's connection (the recent-transcriptions DB). Copies every row,
    then drops the table there so the copy never runs twice. Returns the
    row count moved; 0 when there is nothing to adopt."""
    has = legacy.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sys_samples'"
    ).fetchone()
    if not has:
        return 0
    rows = legacy.execute(
        "SELECT ts, gpu_util, gpu_mem_mb, gpu_temp, cpu_pct, ram_pct, slot_busy"
        " FROM sys_samples").fetchall()
    conn = _require_conn()
    with _lock:
        conn.executemany(
            "INSERT OR IGNORE INTO system_metrics"
            " (ts, gpu_util, gpu_mem_mb, gpu_temp, cpu_pct, ram_pct, slot_busy)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [tuple(r) for r in rows])
        conn.commit()
    legacy.execute("DROP TABLE sys_samples")
    legacy.commit()
    return len(rows)


def record(rows: list[dict[str, Any]]) -> int:
    """Insert (or replace on the same grid second) a batch of rows in one
    transaction. Returns the row count."""
    if not rows:
        return 0
    conn = _require_conn()
    with _lock:
        conn.executemany(
            "INSERT OR REPLACE INTO system_metrics"
            " (ts, gpu_util, gpu_mem_mb, gpu_temp, cpu_pct, ram_pct, slot_busy)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(int(r["ts"]), r.get("gpu_util"), r.get("gpu_mem_mb"),
              r.get("gpu_temp"), r.get("cpu_pct"), r.get("ram_pct"),
              r.get("slot_busy")) for r in rows],
        )
        conn.commit()
    return len(rows)


def list_series(*, metric: str, from_ts: float, to_ts: float,
                step_s: int) -> dict[str, list]:
    """`{t, avg, max}` of one metric between from_ts and to_ts, downsampled
    onto a `step_s` grid (AVG and MAX per bucket) so a week is ~2 000 points.
    Unknown metrics raise ValueError (the name is interpolated)."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric: {metric!r}")
    step = max(1, int(step_s))
    conn = _require_conn()
    cur = conn.execute(
        f"SELECT (ts / ?) * ? AS t, AVG({metric}) AS a, MAX({metric}) AS m"
        f" FROM system_metrics WHERE ts >= ? AND ts < ? AND {metric} IS NOT NULL"
        " GROUP BY t ORDER BY t",
        (step, step, int(from_ts), int(to_ts)),
    )
    t: list[int] = []
    avg: list[float] = []
    mx: list[float] = []
    for r in cur.fetchall():
        t.append(int(r["t"]))
        avg.append(round(float(r["a"]), 2))
        mx.append(round(float(r["m"]), 2))
    return {"t": t, "avg": avg, "max": mx}


def prune(retention_days: float) -> int:
    """Delete rows older than retention_days; returns the count. 0 = keep."""
    if not retention_days or retention_days <= 0:
        return 0
    conn = _require_conn()
    cutoff = int(time.time() - float(retention_days) * 86400)
    with _lock:
        cur = conn.execute("DELETE FROM system_metrics WHERE ts < ?", (cutoff,))
        conn.commit()
    return int(cur.rowcount or 0)
