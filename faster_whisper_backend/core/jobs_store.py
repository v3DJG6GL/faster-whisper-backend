"""Durable job resource for batch runs: status + the verbatim result, keyed
by the client's progress id, so a client that lost its HTTP connection (app
quit, laptop lid, network blip) can list, re-attach to and fetch the run it
started. Served through GET/DELETE /v1/jobs* (main.py).

Why a second table beside core/jobs.py: that registry is the LIVE picture
(one process-wide dict, gone with the handler); this store is the durable
one (one SQLite row per run, TTL-swept). recent_transcriptions keeps the
text-only trace for the trace panel and never the segments/translations/
speakers the desktop client needs to rebuild a record — this table keeps the
response body exactly as the POST would have returned it.

Only runs that arrived WITH a `progress_id` get a row: the id is the job id,
and a client that never asked for progress has no handle to come back with
(OpenAI-compatible callers are unaffected by construction).

Lifecycle per run:

  1. `start(...)` right after the progress entry is seeded — a `running` row.
  2. `finish(...)` in the handler's outer finally on EVERY path — `done` with
     the payload, `failed` with a client-safe error, or `cancelled`.
  3. `mark_running_as_failed("server restarted")` at lifespan start: a row
     still `running` when the process boots was interrupted (the work died
     with the process), so it must not read as in-flight forever.
  4. `prune(...)` lazily every Nth insert AND hourly (main._jobs_retention_loop):
     expired rows first, then the row cap (newest kept, running never
     evicted), then the byte cap (oldest finished results dropped until the
     stored result bytes fit).

Module-level connection: SQLite's WAL mode lets one connection be shared
across threads (`check_same_thread=False`); `_lock` serialises writers. The
handler writes `finish` off the loop thread (asyncio.to_thread) because a
result_json can be megabytes.

Do not log row content — result_json carries the transcript. Log lines
carry only counts and id prefixes.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from typing import Any

from faster_whisper_backend.core import store_common

logger = logging.getLogger("whisper-api")

_lock = threading.RLock()  # reentrant: start() holds it and may call prune()
_conn: sqlite3.Connection | None = None
_insert_counter = 0

STATES = ("running", "done", "failed", "cancelled")
KINDS = ("transcribe", "translate")
SOURCE_KINDS = ("file", "url", "text")

# Client-facing string caps. source_name is a file basename / URL host /
# "N segments → de,fr" summary; error is a curated HTTPException detail.
_CAP_NAME = 200
_CAP_ERROR = 200
_CAP_MODEL = 128
_CAP_SMALL = 32
# Server-built JSON side blobs (stage timings, plan snapshot) — defensive.
_CAP_SIDE_JSON = 64_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id          TEXT PRIMARY KEY,
  request_id      TEXT NOT NULL,
  kind            TEXT NOT NULL,
  user_id         TEXT,
  key_id          TEXT,
  state           TEXT NOT NULL DEFAULT 'running',
  created_ts      REAL NOT NULL,
  finished_ts     REAL,
  expires_ts      REAL NOT NULL,
  model           TEXT,
  source_kind     TEXT NOT NULL DEFAULT 'file',
  source_name     TEXT,
  task            TEXT,
  response_format TEXT,
  error           TEXT,
  result_json     TEXT,
  result_bytes    INTEGER NOT NULL DEFAULT 0,
  stages_json     TEXT,
  plan_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id, created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_key_created  ON jobs(key_id, created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_expires      ON jobs(expires_ts);
"""

# Additive columns for DBs created by an older build: (column, DDL suffix).
# Every column above that is NOT part of the first schema version belongs
# here too, so a pre-existing table grows into the current shape.
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("task", "TEXT"),
    ("response_format", "TEXT"),
    ("stages_json", "TEXT"),
    ("plan_json", "TEXT"),
)

# Every column except the result blob — what `get` / `list_jobs` return.
_META_COLS = (
    "job_id, request_id, kind, user_id, key_id, state, created_ts, "
    "finished_ts, expires_ts, model, source_kind, source_name, task, "
    "response_format, error, result_bytes, stages_json, plan_json"
)


def init_db(path: str) -> None:
    """Open (or create) the DB at `path` in WAL mode. Idempotent — call once
    on service startup before any other function in this module."""
    global _conn
    _conn = store_common.open_wal_db(path)
    _conn.execute("PRAGMA temp_store=MEMORY;")
    _conn.executescript(_SCHEMA)
    store_common.secure_db_file(path)
    cols = {r["name"] for r in _conn.execute("PRAGMA table_info(jobs)")}
    for col, ddl in _MIGRATIONS:
        if col not in cols:
            _conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ddl}")


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("jobs_store.init_db() was not called before use.")
    return _conn


def _reset_for_tests() -> None:
    """Drop the module connection (tests re-init onto a fresh temp file)."""
    global _conn, _insert_counter
    if _conn is not None:
        try:
            _conn.close()
        except Exception:  # noqa: BLE001
            pass
    _conn = None
    _insert_counter = 0


_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _clip(value: Any, cap: int) -> str | None:
    """Control characters out (a value can echo caller text), then cap."""
    if value is None:
        return None
    s = _CTRL_RE.sub("?", str(value))
    return s[:cap] if s else None


def _side_json(value: Any) -> str | None:
    """Serialise a server-built side blob (stages / plan); None when absent
    or over the defensive cap (the row still lands, just without it)."""
    if value is None:
        return None
    try:
        blob = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None
    return blob if len(blob) <= _CAP_SIDE_JSON else None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("stages_json", "plan_json"):
        raw = d.pop(key, None)
        name = key[:-5]
        if raw:
            try:
                d[name] = json.loads(raw)
            except (TypeError, ValueError):
                d[name] = None
        else:
            d[name] = None
    d["result_available"] = bool(d.get("state") == "done"
                                 and int(d.get("result_bytes") or 0) > 0)
    return d


def _lazy_prune_if_due(prune_every: int, *, ttl_s: float, max_rows: int,
                       max_bytes: int) -> None:
    """Bumps an in-process counter; calls prune() every Nth insert."""
    global _insert_counter
    _insert_counter += 1
    if prune_every <= 0 or _insert_counter % prune_every != 0:
        return
    try:
        prune(ttl_s=ttl_s, max_rows=max_rows, max_bytes=max_bytes)
    except Exception as e:  # noqa: BLE001 — a prune never fails a run
        logger.warning("[jobs] prune failed: %s", e)


def start(
    *,
    job_id: str,
    request_id: str,
    kind: str,
    user_id: str | None,
    key_id: str | None,
    model: str | None,
    source_kind: str,
    source_name: str | None,
    task: str | None = None,
    response_format: str | None = None,
    ttl_s: float,
    max_rows: int,
    max_bytes: int,
    prune_every: int = 20,
) -> None:
    """Insert the `running` row for a run that just seeded its progress
    entry. INSERT OR REPLACE: a finished row under a re-used id is
    superseded (an id still in flight never reaches here — the handler
    demotes duplicates to "no progress" before seeding)."""
    conn = _require_conn()
    now = time.time()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, request_id, kind, user_id, "
            "key_id, state, created_ts, finished_ts, expires_ts, model, "
            "source_kind, source_name, task, response_format, error, "
            "result_json, result_bytes, stages_json, plan_json) VALUES "
            "(?, ?, ?, ?, ?, 'running', ?, NULL, ?, ?, ?, ?, ?, ?, NULL, "
            "NULL, 0, NULL, NULL)",
            (
                job_id, request_id,
                kind if kind in KINDS else "transcribe",
                user_id or None, key_id or None,
                now, now + float(ttl_s),
                _clip(model, _CAP_MODEL),
                source_kind if source_kind in SOURCE_KINDS else "file",
                _clip(source_name, _CAP_NAME),
                _clip(task, _CAP_SMALL),
                _clip(response_format, _CAP_SMALL),
            ),
        )
        _lazy_prune_if_due(prune_every, ttl_s=ttl_s, max_rows=max_rows,
                           max_bytes=max_bytes)


def finish(
    *,
    job_id: str,
    state: str,
    error: str | None = None,
    result: Any = None,
    stages: Any = None,
    plan: Any = None,
    model: str | None = None,
    task: str | None = None,
    ttl_s: float,
) -> bool:
    """Stamp the terminal state (+ the verbatim response payload on `done`).
    `result` may be a dict (json / verbose_json) or a str (the `text`
    format) — json.dumps serialises either, and the result route hands the
    decoded value back, so the client receives what the POST would have.
    Returns False when the row is gone (pruned meanwhile)."""
    if state not in STATES or state == "running":
        raise ValueError(f"finish: not a terminal state: {state!r}")
    conn = _require_conn()
    now = time.time()
    blob: str | None = None
    if state == "done" and result is not None:
        blob = json.dumps(result, ensure_ascii=False)
    with _lock:
        cur = conn.execute(
            "UPDATE jobs SET state = ?, error = ?, result_json = ?, "
            "result_bytes = ?, stages_json = ?, plan_json = ?, "
            "model = COALESCE(?, model), task = COALESCE(?, task), "
            "finished_ts = ?, expires_ts = ? WHERE job_id = ?",
            (
                state,
                _clip(error, _CAP_ERROR) if state != "done" else None,
                blob,
                len(blob.encode("utf-8")) if blob is not None else 0,
                _side_json(stages), _side_json(plan),
                _clip(model, _CAP_MODEL), _clip(task, _CAP_SMALL),
                now, now + float(ttl_s), job_id,
            ),
        )
        return bool(cur.rowcount)


def get(job_id: str) -> dict[str, Any] | None:
    """Every column except the result blob (polls must not load MBs)."""
    conn = _require_conn()
    row = conn.execute(
        f"SELECT {_META_COLS} FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_result(job_id: str) -> Any:
    """The decoded result payload (dict or str), or None when absent."""
    conn = _require_conn()
    row = conn.execute(
        "SELECT result_json FROM jobs WHERE job_id = ? AND state = 'done'",
        (job_id,),
    ).fetchone()
    if not row or row["result_json"] is None:
        return None
    try:
        return json.loads(row["result_json"])
    except (TypeError, ValueError):
        return None


def _owner_clause(user_id: str | None, key_id: str | None,
                  all_users: bool) -> tuple[str, list[Any]]:
    """Ownership predicate mirroring main._progress_entry_for: a row belongs
    to the caller when its user_id matches, or — for a key without a user —
    when its key_id matches. A row with neither (open mode, no keys yet)
    belongs to callers with neither. `all_users` (admin) lifts the filter."""
    if all_users:
        return "1=1", []
    if not user_id and not key_id:
        return "(user_id IS NULL AND key_id IS NULL)", []
    return ("(user_id = ? OR (user_id IS NULL AND key_id = ?))",
            [user_id or "", key_id or ""])


def is_owner(row: dict[str, Any], *, user_id: str | None,
             key_id: str | None) -> bool:
    """Whether the caller identified by (user_id, key_id) owns `row` — the
    single-row twin of _owner_clause."""
    r_user = row.get("user_id")
    r_key = row.get("key_id")
    if not r_user and not r_key:
        return not user_id and not key_id
    if r_user:
        return bool(user_id) and r_user == user_id
    return bool(key_id) and r_key == key_id


def list_jobs(
    *,
    user_id: str | None,
    key_id: str | None,
    all_users: bool = False,
    state: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Newest first; own rows unless `all_users`; expired rows excluded."""
    conn = _require_conn()
    where, params = _owner_clause(user_id, key_id, all_users)
    clauses = [where, "expires_ts >= ?"]
    params.append(time.time())
    if state:
        clauses.append("state = ?")
        params.append(state)
    params.append(max(1, int(limit)))
    rows = conn.execute(
        f"SELECT {_META_COLS} FROM jobs WHERE " + " AND ".join(clauses)
        + " ORDER BY created_ts DESC LIMIT ?",
        params,
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete(job_id: str) -> bool:
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        return bool(cur.rowcount)


def mark_running_as_failed(error: str) -> int:
    """Startup: every row still `running` was interrupted by the previous
    process's death. Returns the count flipped to `failed`."""
    conn = _require_conn()
    now = time.time()
    with _lock:
        cur = conn.execute(
            "UPDATE jobs SET state = 'failed', error = ?, finished_ts = ? "
            "WHERE state = 'running'",
            (_clip(error, _CAP_ERROR), now),
        )
        return cur.rowcount or 0


def count() -> int:
    conn = _require_conn()
    row = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()
    return int(row["n"]) if row else 0


def total_result_bytes() -> int:
    conn = _require_conn()
    row = conn.execute(
        "SELECT COALESCE(SUM(result_bytes), 0) AS n FROM jobs").fetchone()
    return int(row["n"]) if row else 0


def prune(*, ttl_s: float, max_rows: int, max_bytes: int) -> int:
    """Three passes, each a single DELETE: expired rows (finished OR running
    past `ttl_s` — a run that outlives its own TTL is not a run any more);
    rows beyond `max_rows` newest, never a running one; then oldest finished
    rows until the stored result bytes fit `max_bytes`. 0 disables the row
    and byte caps. Returns the total deleted."""
    conn = _require_conn()
    deleted = 0
    now = time.time()
    with _lock:
        cur = conn.execute("DELETE FROM jobs WHERE expires_ts < ?", (now,))
        deleted += cur.rowcount or 0
        if max_rows > 0:
            cur = conn.execute(
                "DELETE FROM jobs WHERE state != 'running' AND job_id NOT IN "
                "(SELECT job_id FROM jobs ORDER BY created_ts DESC LIMIT ?)",
                (int(max_rows),),
            )
            deleted += cur.rowcount or 0
        if max_bytes > 0:
            total = total_result_bytes()
            if total > max_bytes:
                # Walk finished rows oldest-first, accumulating until the
                # remainder fits; delete that prefix in one statement.
                victims: list[str] = []
                for r in conn.execute(
                    "SELECT job_id, result_bytes FROM jobs WHERE "
                    "state != 'running' ORDER BY created_ts ASC"
                ):
                    if total <= max_bytes:
                        break
                    victims.append(r["job_id"])
                    total -= int(r["result_bytes"] or 0)
                if victims:
                    marks = ",".join("?" * len(victims))
                    cur = conn.execute(
                        f"DELETE FROM jobs WHERE job_id IN ({marks})", victims)
                    deleted += cur.rowcount or 0
    return deleted


def sweep_retention() -> int:
    """Hourly sweep entry point (main._jobs_retention_loop): reads the live
    config each call so a lowered knob applies on the next tick."""
    from faster_whisper_backend import config as cfg
    n = prune(
        ttl_s=float(getattr(cfg, "JOBS_TTL_S", 259_200)),
        max_rows=int(getattr(cfg, "JOBS_MAX_ROWS", 2000)),
        max_bytes=int(getattr(cfg, "JOBS_MAX_BYTES", 2_000_000_000)),
    )
    if n:
        logger.info("[jobs] retention sweep deleted %d row(s)", n)
    return n


def clear_all() -> int:
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM jobs")
        return cur.rowcount or 0
