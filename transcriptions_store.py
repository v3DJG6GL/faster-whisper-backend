"""Durable store for recent /transcribe traces.

SQLite (stdlib) in WAL mode — single-file, crash-safe, indexed. Lives at
cfg.RECENT_TRANSCRIPTIONS_DB (defaults to recent_transcriptions.local.sqlite3
alongside config.local.json).

Replaces the legacy in-memory ring buffers (`quick_config_state.recent_traces`
and `metrics.recent_tx`) so the /quick-config trace panel and /stats
dashboard "Recent transcriptions" widget survive service restarts and
scale beyond a 20-row cap.

Two upsert call sites per /transcribe request, both keyed by request_id:

  1. `record_trace(...)` (success path, inside the inner try) writes the
     rich payload: raw_text, final_text, steps_json, tokens_json,
     bigrams_json, model, language, user_id, username, created_ts.

  2. `record_timing(...)` (outer finally, always) writes proc_dur_s,
     audio_dur_s, words_count, status. On the error path it inserts a
     minimal row (no raw/final/steps) so /stats still counts the request.

Lazy pruning every `cfg.RECENT_TRANSCRIPTIONS_PRUNE_EVERY` inserts: a
single DELETE statement enforces both the row cap and the TTL. A cap of
0 disables the count clause; a TTL of 0 disables the age clause.

Module-level connection: SQLite's WAL mode lets us share one connection
across threads (`check_same_thread=False`). `_lock` serialises writers.

Do not log row content — entries carry literal dictation text, which
can be sensitive. Module log lines carry only counts and request_id
prefixes.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any

import store_common

logger = logging.getLogger("whisper-api")

_lock = threading.RLock()  # reentrant: record_trace/record_timing hold the lock and may call prune() which re-acquires it
_conn: sqlite3.Connection | None = None
_insert_counter = 0

# Generous field caps — raw/final are audio-driven and can be long
# (a 5-minute dictation block). Steps JSON is bounded by the existing
# reports_store cap so a trace pair (report + recent) doesn't surprise
# anyone with mismatched size limits.
_CAP_RAW = 50_000
_CAP_FINAL = 50_000
_CAP_STEPS_JSON = 200_000
# Hard row bound on the steps list, applied before the JSON byte cap.
_CAP_STEPS_ROWS = 500
_CAP_TOKEN_FIELD = 64
# The model id arrives verbatim from the request form field. A name outside the
# allowlist is rejected with a 400, but the outer `finally` in main.transcribe
# still records the attempt, so the rejected string reaches this table anyway.
# 96 matches config_store.ModelId's max_length — no legitimate id is affected.
_CAP_MODEL = 96
# The one client-originated column that had no cap. A streaming handshake's
# {"type":"config","language":...} is only length-bounded by the 1 MiB frame
# size, and on the fallback path (no fw_info) it lands here verbatim and is
# then served back through /quick-config/recent and the SSE replay. 32 is far
# above any BCP-47 tag.
_CAP_LANGUAGE = 32
# Recent-jobs extras. stages_json is server-built ([{name, secs, model?,
# detail?}, …] — a handful of pipeline stages), so the cap is a defensive
# bound, not a fight with an attacker. key_label mirrors api_keys' label cap.
_CAP_KIND = 16
_CAP_STAGES_JSON = 20_000
_CAP_KEY_LABEL = 120

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recent_transcriptions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id    TEXT NOT NULL UNIQUE,
  created_ts    REAL NOT NULL,
  user_id       TEXT,
  username      TEXT,
  model         TEXT NOT NULL,
  language      TEXT,
  source        TEXT NOT NULL DEFAULT 'file',
  status        TEXT NOT NULL DEFAULT 'ok',
  audio_dur_s   REAL,
  proc_dur_s    REAL,
  words_count   INTEGER,
  raw_text      TEXT,
  final_text    TEXT,
  steps_json    TEXT,
  tokens_json   TEXT,
  bigrams_json  TEXT,
  kind          TEXT,
  stages_json   TEXT,
  key_label     TEXT,
  wait_s        REAL,
  error_class   TEXT,
  error_stage   TEXT
);
CREATE INDEX IF NOT EXISTS idx_rt_created      ON recent_transcriptions(created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_rt_user_created ON recent_transcriptions(user_id, created_ts DESC);

-- Machine samples on the STATS_HISTORY_SAMPLE_S grid (stats_sampler): the
-- source of /stats/history. Lives here, in the rolling operational DB,
-- because it has that DB's lifecycle (30 days, pruned), not the usage
-- ledger's (kept for years).
CREATE TABLE IF NOT EXISTS sys_samples (
  ts         INTEGER PRIMARY KEY,
  gpu_util   REAL,
  gpu_mem_mb REAL,
  gpu_temp   REAL,
  cpu_pct    REAL,
  ram_pct    REAL,
  slot_busy  REAL
);
"""


def init_db(path: str) -> None:
    """Open (or create) the DB at `path` in WAL mode. Idempotent — call
    once on service startup before any other function in this module.
    Mirrors reports_store.init_db / captures_store.init pattern."""
    global _conn
    _conn = store_common.open_wal_db(path)
    _conn.execute("PRAGMA temp_store=MEMORY;")
    _conn.executescript(_SCHEMA)
    store_common.secure_db_file(path)
    # Migrate pre-existing DBs (created before the `source` column): add it with
    # the 'file' default so old rows read as batch/file-upload transcriptions.
    cols = {r["name"] for r in _conn.execute("PRAGMA table_info(recent_transcriptions)")}
    if "source" not in cols:
        _conn.execute(
            "ALTER TABLE recent_transcriptions ADD COLUMN source TEXT NOT NULL DEFAULT 'file'")
    # "Recent jobs" columns (nullable, additive): `kind` distinguishes
    # transcribe/translate/download rows (NULL reads as a transcription and
    # resolves via `source`), `stages_json` carries the per-stage timing list
    # ([{name, secs, model?, detail?}, …]) and `key_label` snapshots the API
    # key's display label at record time (labels are mutable/deletable in
    # api_keys, so resolving at read time would lie about history).
    # v2 stats columns (nullable, additive): `wait_s` = seconds the request
    # queued for a GPU slot, `error_class` / `error_stage` = why and where
    # a failed job failed (metrics.ERROR_CLASSES).
    for col, ddl in (
        ("kind", "ADD COLUMN kind TEXT"),
        ("stages_json", "ADD COLUMN stages_json TEXT"),
        ("key_label", "ADD COLUMN key_label TEXT"),
        ("wait_s", "ADD COLUMN wait_s REAL"),
        ("error_class", "ADD COLUMN error_class TEXT"),
        ("error_stage", "ADD COLUMN error_stage TEXT"),
    ):
        if col not in cols:
            _conn.execute(f"ALTER TABLE recent_transcriptions {ddl}")


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError(
            "transcriptions_store.init_db() was not called before use."
        )
    return _conn


def _truncate_steps(steps: list) -> list:
    """Same shape as reports_store._truncate_steps but with this module's
    own caps so the two stores can evolve their limits independently.
    Drops oldest pipeline stages first when over-cap; preserves the
    output-wrapper + terminal-trim trailers (the part admins care about)."""
    out: list = []
    for s in steps:
        if not isinstance(s, (list, tuple)) or len(s) < 3:
            continue
        out.append([str(s[0])[:512],
                    str(s[1])[:_CAP_RAW],
                    str(s[2])[:_CAP_RAW]])
    # Row bound first: the pipeline emits a few dozen stages, so 500 is
    # far above any legitimate trace, and it bounds the byte walk below
    # regardless of how many entries the incoming list carries.
    out = out[-_CAP_STEPS_ROWS:]
    # Walk newest-first accumulating each entry's own serialized length
    # instead of re-serializing the whole remaining list per drop (that
    # was quadratic). Budget the two enclosing brackets plus json.dumps'
    # ", " separator per entry, so the caller's single serialization is
    # guaranteed under the cap. Kept in lockstep with reports_store.
    budget = _CAP_STEPS_JSON - 2
    keep = 0
    for entry in reversed(out):
        need = len(json.dumps(entry, ensure_ascii=False)) + (2 if keep else 0)
        if need > budget:
            break
        budget -= need
        keep += 1
    return out[len(out) - keep:]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Materialize a row to the wire shape expected by /quick-config and
    /stats consumers. Decodes the JSON-bearing columns; computes the
    derived `rtf` (audio_dur_s / proc_dur_s) so callers don't repeat the
    formula. Keeps both `ts` (legacy /quick-config field name) and
    `created_ts` (the actual column) so existing JS keeps working."""
    d = dict(row)
    for col, key in (
        ("steps_json", "steps"),
        ("tokens_json", "tokens"),
        ("bigrams_json", "bigrams"),
        ("stages_json", "stages"),
    ):
        try:
            d[key] = json.loads(d.pop(col, "[]") or "[]")
        except (TypeError, ValueError):
            d[key] = []
        if not isinstance(d[key], list):
            d[key] = []
    audio = d.get("audio_dur_s")
    proc = d.get("proc_dur_s")
    d["rtf"] = round(audio / proc, 2) if (audio and proc and proc > 0) else None
    d["ts"] = d.get("created_ts")
    # Legacy /stats widget keys — kept for the existing dashboard JS.
    d["audio_dur"] = audio
    d["proc_dur"] = proc
    d["words"] = d.get("words_count")
    # /quick-config renderTrace + _buildReportForm read entry.raw / entry.final;
    # the live SSE event double-keys both names, hydrated rows must match.
    # Error-path rows (record_timing without record_trace) leave the text
    # columns NULL — coerce to '' so the JS string ops never see None.
    d["raw"] = d.get("raw_text") or ""
    d["final"] = d.get("final_text") or ""
    d["username"] = d.get("username") or ""
    d["language"] = d.get("language") or ""
    d["source"] = d.get("source") or "file"
    # Recent-jobs extras — NULL on rows written before the migration (the
    # `d.pop(col, "[]")` default in the loop above covers `stages` there).
    d["kind"] = d.get("kind") or None
    d["key_label"] = d.get("key_label") or ""
    d["wait_s"] = d.get("wait_s")
    d["error_class"] = d.get("error_class") or None
    d["error_stage"] = d.get("error_stage") or None
    return d


def _lazy_prune_if_due(prune_every: int, max_rows: int, ttl_days: float) -> None:
    """Bumps an in-process counter; calls prune() every Nth insert."""
    global _insert_counter
    _insert_counter += 1
    if prune_every <= 0:
        return
    if _insert_counter % prune_every != 0:
        return
    try:
        prune(max_rows=max_rows, ttl_days=ttl_days)
    except Exception as e:
        logger.warning("[recent-tx] prune failed: %s", e)


# Column names are interpolated into the history query; only these.
SYS_SAMPLE_METRICS: frozenset[str] = frozenset(
    ("gpu_util", "gpu_mem_mb", "gpu_temp", "cpu_pct", "ram_pct", "slot_busy"))


def record_sys_samples(rows: list[dict[str, Any]]) -> int:
    """Insert (or replace on the same grid second) a batch of samples in
    one transaction. Returns the row count."""
    if not rows:
        return 0
    conn = _require_conn()
    with _lock:
        conn.executemany(
            "INSERT OR REPLACE INTO sys_samples"
            " (ts, gpu_util, gpu_mem_mb, gpu_temp, cpu_pct, ram_pct, slot_busy)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(int(r["ts"]), r.get("gpu_util"), r.get("gpu_mem_mb"),
              r.get("gpu_temp"), r.get("cpu_pct"), r.get("ram_pct"),
              r.get("slot_busy")) for r in rows],
        )
        conn.commit()
    return len(rows)


def list_sys_samples(*, metric: str, from_ts: float, to_ts: float,
                     step_s: int) -> dict[str, list]:
    """`{t, avg, max}` of one metric between from_ts and to_ts, downsampled
    onto a `step_s` grid (AVG and MAX per bucket) so a week is ~2 000 points.
    Unknown metrics raise ValueError (the name is interpolated)."""
    if metric not in SYS_SAMPLE_METRICS:
        raise ValueError(f"unknown metric: {metric!r}")
    step = max(1, int(step_s))
    conn = _require_conn()
    cur = conn.execute(
        f"SELECT (ts / ?) * ? AS t, AVG({metric}) AS a, MAX({metric}) AS m"
        f" FROM sys_samples WHERE ts >= ? AND ts < ? AND {metric} IS NOT NULL"
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


def prune_sys_samples(ttl_days: float) -> int:
    """Delete samples older than ttl_days; returns the count."""
    if not ttl_days or ttl_days <= 0:
        return 0
    conn = _require_conn()
    cutoff = int(time.time() - float(ttl_days) * 86400)
    with _lock:
        cur = conn.execute("DELETE FROM sys_samples WHERE ts < ?", (cutoff,))
        conn.commit()
    return int(cur.rowcount or 0)



def record_trace(
    *,
    request_id: str,
    model: str,
    raw: str,
    final: str,
    steps: list | None = None,
    tokens: list | None = None,
    bigrams: list | None = None,
    language: str | None = None,
    source: str = "file",
    user_id: str | None = None,
    username: str | None = None,
    created_ts: float | None = None,
    prune_every: int = 50,
    max_rows: int = 500,
    ttl_days: float = 30.0,
) -> None:
    """Insert or update the rich half of a /transcribe row. Called on the
    success path (inside the inner try in main.py). UPSERTs by request_id
    so a later record_timing() call merges the timing fields in.

    All text fields are silently truncated at module caps; over-large
    `steps` lists shed leading entries (terminal trim + wrapper are the
    rows the admin actually wants — see _truncate_steps)."""
    if not request_id:
        return
    raw_s = (raw or "")[:_CAP_RAW]
    final_s = (final or "")[:_CAP_FINAL]
    model = (model or "")[:_CAP_MODEL]
    language = (language or "")[:_CAP_LANGUAGE] or None
    steps_blob = json.dumps(_truncate_steps(steps or []), ensure_ascii=False)
    tokens_blob = json.dumps([str(t)[:_CAP_TOKEN_FIELD] for t in (tokens or [])],
                             ensure_ascii=False)
    bigrams_blob = json.dumps([str(b)[:_CAP_TOKEN_FIELD * 2] for b in (bigrams or [])],
                              ensure_ascii=False)
    ts = float(created_ts) if created_ts else time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "INSERT INTO recent_transcriptions ("
            "  request_id, created_ts, user_id, username, model, language, source,"
            "  status, raw_text, final_text,"
            "  steps_json, tokens_json, bigrams_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?)"
            " ON CONFLICT(request_id) DO UPDATE SET"
            "  model        = excluded.model,"
            "  language     = COALESCE(excluded.language, recent_transcriptions.language),"
            "  source       = excluded.source,"
            "  user_id      = COALESCE(excluded.user_id, recent_transcriptions.user_id),"
            "  username     = COALESCE(excluded.username, recent_transcriptions.username),"
            "  raw_text     = excluded.raw_text,"
            "  final_text   = excluded.final_text,"
            "  steps_json   = excluded.steps_json,"
            "  tokens_json  = excluded.tokens_json,"
            "  bigrams_json = excluded.bigrams_json",
            (request_id, ts, user_id, username, model, language, source or "file",
             raw_s, final_s, steps_blob, tokens_blob, bigrams_blob),
        )
        _lazy_prune_if_due(prune_every, max_rows, ttl_days)


def record_timing(
    *,
    request_id: str,
    model: str,
    audio_dur_s: float | None,
    proc_dur_s: float,
    status: str,
    words_count: int,
    user_id: str | None = None,
    created_ts: float | None = None,
    kind: str | None = None,
    stages: list | None = None,
    key_label: str | None = None,
    prune_every: int = 50,
    max_rows: int = 500,
    ttl_days: float = 30.0,
    wait_s: float | None = None,
    error_class: str | None = None,
    error_stage: str | None = None,
) -> None:
    """Insert or update the timing half. Called in the outer finally so
    it runs on BOTH success (after record_trace) and error paths. UPSERT
    by request_id: on success it patches timing fields onto the row
    record_trace already wrote; on error it inserts a minimal row with
    no raw/final/steps.

    `kind` ('translate' / 'download' / NULL = a transcription, resolved
    via `source`), `stages` (per-stage timing dicts, JSON-encoded) and
    `key_label` are the additive "Recent jobs" fields — all optional,
    COALESCEd on conflict so a caller that omits them never blanks what
    an earlier write recorded."""
    if not request_id:
        return
    model = (model or "")[:_CAP_MODEL]
    kind = (kind or "")[:_CAP_KIND] or None
    key_label = (key_label or "")[:_CAP_KEY_LABEL] or None
    error_class = (error_class or "")[:32] or None
    error_stage = (error_stage or "")[:32] or None
    wait_s = None if wait_s is None else round(max(0.0, float(wait_s)), 3)
    stages_blob = None
    if stages:
        try:
            stages_blob = json.dumps(list(stages),
                                     ensure_ascii=False)[:_CAP_STAGES_JSON]
            json.loads(stages_blob)   # truncation may have cut mid-token
        except (TypeError, ValueError):
            stages_blob = None
    ts = float(created_ts) if created_ts else time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "INSERT INTO recent_transcriptions ("
            "  request_id, created_ts, user_id, model, status,"
            "  audio_dur_s, proc_dur_s, words_count,"
            "  kind, stages_json, key_label, wait_s, error_class, error_stage"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(request_id) DO UPDATE SET"
            "  status      = excluded.status,"
            "  audio_dur_s = excluded.audio_dur_s,"
            "  proc_dur_s  = excluded.proc_dur_s,"
            "  words_count = excluded.words_count,"
            "  model       = excluded.model,"
            "  kind        = COALESCE(excluded.kind, recent_transcriptions.kind),"
            "  stages_json = COALESCE(excluded.stages_json, recent_transcriptions.stages_json),"
            "  key_label   = COALESCE(excluded.key_label, recent_transcriptions.key_label),"
            "  wait_s      = COALESCE(excluded.wait_s, recent_transcriptions.wait_s),"
            "  error_class = COALESCE(excluded.error_class, recent_transcriptions.error_class),"
            "  error_stage = COALESCE(excluded.error_stage, recent_transcriptions.error_stage)",
            (request_id, ts, user_id, model, status,
             audio_dur_s, proc_dur_s, words_count,
             kind, stages_blob, key_label, wait_s, error_class, error_stage),
        )
        _lazy_prune_if_due(prune_every, max_rows, ttl_days)


def list_recent(
    *,
    before_ts: float | None = None,
    limit: int = 100,
    user_id_filter: str | None = None,
    query: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    slow_rtf: float | None = None,
) -> list[dict[str, Any]]:
    """Return up to `limit` rows newer than `before_ts` (or the newest
    `limit` rows when before_ts is None / 0), newest-first. When
    user_id_filter is set, returns only rows for that user — used by
    /quick-config scope='own' to keep one user's traces out of another
    user's view.

    When `query` is set, only rows whose raw_text OR final_text contain
    the substring are returned (case-insensitive substring match). This
    composes with both the before_ts cursor and user_id_filter, so the
    "Load older" pagination walks back through matches only. Note: SQLite
    LIKE is ASCII case-insensitive — non-ASCII (e.g. German umlauts) is
    matched case-sensitively; acceptable for this free-text search.

    The /stats jobs table adds `kind` (one recent-jobs kind), `status`
    ('ok' | 'error' | 'cancelled', or 'failed' = anything but ok) and
    `slow_rtf` (only jobs whose processing took more than that fraction of
    their audio: RTF = proc / audio > slow_rtf). All compose with the
    cursor and the user filter."""
    conn = _require_conn()
    where: list[str] = []
    params: list[Any] = []
    if before_ts and before_ts > 0:
        where.append("created_ts < ?")
        params.append(float(before_ts))
    if user_id_filter is not None:
        where.append("user_id = ?")
        params.append(user_id_filter)
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if status == "failed":
        where.append("status <> 'ok'")
    elif status:
        where.append("status = ?")
        params.append(status)
    if slow_rtf is not None and slow_rtf > 0:
        where.append("audio_dur_s > 0 AND proc_dur_s > ? * audio_dur_s")
        params.append(float(slow_rtf))
    if query:
        # Escape LIKE wildcards so a literal % or _ in the search text is
        # matched literally rather than as a wildcard.
        needle = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{needle}%"
        where.append(
            "(raw_text LIKE ? ESCAPE '\\' OR final_text LIKE ? ESCAPE '\\')"
        )
        params.append(like)
        params.append(like)
    sql = "SELECT * FROM recent_transcriptions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_ts DESC LIMIT ?"
    params.append(max(1, int(limit)))
    cur = conn.execute(sql, params)
    return [_row_to_dict(r) for r in cur.fetchall()]


def count() -> int:
    conn = _require_conn()
    row = conn.execute("SELECT COUNT(*) AS n FROM recent_transcriptions").fetchone()
    return int(row["n"]) if row else 0


def prune(*, max_rows: int, ttl_days: float) -> int:
    """Drop rows older than the TTL cutoff AND rows beyond the count cap.
    A max_rows or ttl_days of 0 disables that clause; both 0 makes prune
    a no-op."""
    if max_rows <= 0 and ttl_days <= 0:
        return 0
    conn = _require_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if max_rows > 0:
        clauses.append(
            "id NOT IN (SELECT id FROM recent_transcriptions"
            " ORDER BY created_ts DESC LIMIT ?)"
        )
        params.append(int(max_rows))
    if ttl_days > 0:
        clauses.append("created_ts < ?")
        params.append(time.time() - float(ttl_days) * 86400.0)
    sql = "DELETE FROM recent_transcriptions WHERE " + " OR ".join(clauses)
    with _lock:
        cur = conn.execute(sql, params)
        return cur.rowcount or 0


def clear_all() -> int:
    """Wipe every row. Returns the count deleted. Admin-triggered."""
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM recent_transcriptions")
        return cur.rowcount or 0
