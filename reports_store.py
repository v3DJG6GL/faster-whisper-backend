"""Durable store for user-submitted transcription error reports.

SQLite (stdlib) in WAL mode — single-file, crash-safe, indexed. Lives at
cfg.REPORTS_DB (defaults to reports.local.sqlite3 alongside config.local.json).

This is the only structured, query-able, end-user-editable durable
dictation-content surface produced by this app. The rotating text logger
is also durable content but format-locked; reports are user-curated for
triage. Plaintext on disk; whole-disk encryption is the deployment's
responsibility.

Do not log report content. The module's INFO/WARNING lines carry only
counts and report-id prefixes for forensics.

Module-level connection: SQLite's WAL mode lets us share one connection
across threads (with `check_same_thread=False`). All public functions
acquire `_lock` for writes; reads are concurrent and safe.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
import uuid
from typing import Any

import store_common
import text_corrections

logger = logging.getLogger("whisper-api")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

# Field caps — applied server-side before insert. raw/final get the
# generous cap because they're audio-driven and can legitimately be long
# (a 5-min dictation block); intended/comment/notes are human-typed.
_CAP_RAW = 50_000
_CAP_FINAL = 50_000
_CAP_STEPS_JSON = 200_000
# Hard row bound on the steps list, applied before the JSON byte cap.
_CAP_STEPS_ROWS = 500
_CAP_INTENDED = 2_000
_CAP_COMMENT = 4_000
_CAP_REQUEST_ID = 128
# Job provenance, added so a report about a translation identifies the
# translation. Same caps the recent-transcriptions store uses for the same
# two fields, so a value that fits one row fits the other.
_CAP_LANGUAGE = 32
_CAP_STAGES_JSON = 20_000
# Anything past this is not a plausible trace timestamp (year 2100). The point
# is not the upper bound but rejecting inf/NaN: sqlite stores them in a REAL
# column intact, and Starlette renders JSON with allow_nan=False, so one such
# row makes /reports/api/list raise on EVERY subsequent load.
_MAX_TRACE_TS = 4_102_444_800.0
_CAP_ADMIN_NOTES = 8_000
# Read ceiling for list_reports(). Matches the REPORTS_MAX eviction cap's
# default, so a store inside its own cap is unaffected; it bounds the response
# when eviction has not yet caught up or the cap was raised at runtime.
_LIST_LIMIT = 1000
# Public alias so the list route can tell the page when the ceiling was hit.
LIST_LIMIT = _LIST_LIMIT
_CAP_CORRECTIONS = text_corrections.CAP_CORRECTIONS
_CAP_CORRECTION_FIELD = text_corrections.CAP_CORRECTION_FIELD

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
  id              TEXT PRIMARY KEY,
  created_ts      REAL NOT NULL,
  trace_ts        REAL NOT NULL,
  request_id      TEXT,
  model           TEXT NOT NULL,
  raw_text        TEXT NOT NULL,
  final_text      TEXT NOT NULL,
  steps_json      TEXT NOT NULL,
  corrections_json TEXT NOT NULL DEFAULT '[]',
  intended_text   TEXT NOT NULL DEFAULT '',
  user_comment    TEXT NOT NULL DEFAULT '',
  reporter_role   TEXT NOT NULL,
  reporter_host   TEXT NOT NULL DEFAULT '',
  status          TEXT NOT NULL DEFAULT 'open',
  admin_notes     TEXT NOT NULL DEFAULT '',
  resolved_ts     REAL,
  user_id         TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_created    ON reports(created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_reports_status     ON reports(status);
CREATE INDEX IF NOT EXISTS idx_reports_request_id ON reports(request_id);
CREATE INDEX IF NOT EXISTS idx_reports_user_request
  ON reports(user_id, request_id) WHERE user_id IS NOT NULL;
"""

_VALID_STATUS = frozenset({"open", "resolved", "dismissed"})


def init_db(path: str) -> None:
    """Open (or create) the report DB at `path` in WAL mode. Idempotent:
    safe to call on every startup; the schema-CREATE statements use
    IF NOT EXISTS. Call before any other function in this module."""
    global _conn
    # Autocommit + WAL rationale lives on store_common.open_wal_db. _lock
    # serialises writers (so the COUNT inside _evict_to_cap sees a stable
    # total) but does not make the INSERT + DELETEs atomic vs crash.
    _conn = store_common.open_wal_db(path)
    # Column renames (2026-09): the transcript columns are raw_text /
    # final_text in every store. Before _SCHEMA so CREATE TABLE IF NOT
    # EXISTS never sees the old names as the live ones.
    have = {r["name"] for r in _conn.execute("PRAGMA table_info(reports)")}
    for old, new in (("raw", "raw_text"), ("final", "final_text")):
        if old in have and new not in have:
            _conn.execute(f"ALTER TABLE reports RENAME COLUMN {old} TO {new}")
    _conn.commit()
    _conn.executescript(_SCHEMA)
    _ensure_columns(_conn)
    store_common.secure_db_file(path)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Idempotent additive migrations for columns added after the initial
    release. `CREATE TABLE IF NOT EXISTS` never alters an existing table, so
    a new column must be ALTER-ed in here. Safe to run on every init.

    Same convention as api_keys_store._ensure_columns; this store had no such
    hook because it had never needed one."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(reports)")}
    for col, ddl in (
        # A report used to identify only the Whisper model, so "the French
        # translation is wrong" produced a row naming neither the language
        # nor the translation nor the model that made it.
        ("language", "ADD COLUMN language TEXT"),
        ("stages_json", "ADD COLUMN stages_json TEXT"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE reports {ddl}")


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError(
            "reports_store.init_db() was not called before use."
        )
    return _conn


def _clean_trace_ts(value: Any, now: float) -> float:
    """Coerce a caller-supplied trace timestamp to a finite, in-range float.

    inf/NaN are the load-bearing case: json.loads accepts the Infinity
    literal (and 1e400 overflows to inf under any parser), pydantic's bare
    float passes it through, and sqlite stores it in a REAL column intact.
    Starlette renders JSONResponse with allow_nan=False, so a single such row
    makes every later /reports/api/list raise — a stored, self-perpetuating
    500 on the admin triage page. Bound it before it reaches the column.
    """
    try:
        ts = float(value or 0.0)
    except (TypeError, ValueError):
        return now
    if not math.isfinite(ts):
        return now
    return max(0.0, min(ts, _MAX_TRACE_TS)) or now


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Materialize a row, decoding the JSON-bearing columns. Returns
    plain Python types ready for JSON serialization on the wire."""
    d = dict(row)
    # Columns are raw_text / final_text (every store agrees); the API and
    # the pages keep the established raw / final keys.
    if "raw_text" in d:
        d["raw"] = d.pop("raw_text")
    if "final_text" in d:
        d["final"] = d.pop("final_text")
    try:
        d["steps"] = json.loads(d.pop("steps_json", "[]") or "[]")
    except (TypeError, ValueError):
        d["steps"] = []
    try:
        d["corrections"] = json.loads(d.pop("corrections_json", "[]") or "[]")
    except (TypeError, ValueError):
        d["corrections"] = []
    # Absent on rows written before the provenance migration.
    try:
        d["stages"] = json.loads(d.pop("stages_json", None) or "[]")
        if not isinstance(d["stages"], list):
            d["stages"] = []
    except (TypeError, ValueError):
        d["stages"] = []
    d["language"] = d.get("language") or ""
    return d


# ---------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------

def _truncate_steps(steps: list) -> list:
    """Cap the steps list so its JSON serialization stays below
    _CAP_STEPS_JSON. Drops from the front (oldest pipeline stages first)
    so the last entries — output-wrapper + terminal-trim, which are what
    the admin actually wants to see — are preserved."""
    out: list = []
    for s in steps:
        if not isinstance(s, (list, tuple)) or len(s) < 3:
            continue
        out.append([str(s[0])[:_CAP_CORRECTION_FIELD],
                    str(s[1])[:_CAP_RAW],
                    str(s[2])[:_CAP_RAW]])
    # Row bound first: the pipeline emits a few dozen stages, so 500 is
    # far above any legitimate trace, and it bounds the byte walk below
    # regardless of how many entries the client posted.
    out = out[-_CAP_STEPS_ROWS:]
    # Walk newest-first accumulating each entry's own serialized length
    # instead of re-serializing the whole remaining list per drop (that
    # was quadratic on a large submitted steps array). Budget the two
    # enclosing brackets plus json.dumps' ", " separator per entry, so
    # the caller's single serialization is guaranteed under the cap.
    budget = _CAP_STEPS_JSON - 2
    keep = 0
    for entry in reversed(out):
        need = len(json.dumps(entry, ensure_ascii=False)) + (2 if keep else 0)
        if need > budget:
            break
        budget -= need
        keep += 1
    return out[len(out) - keep:]


# Delegates to text_corrections so /reports and /captures share one
# definition of the chip shape. Kept here as a module-level name for the
# external callers that already import it (reports_routes.submit_report).
_clean_corrections = text_corrections.clean_corrections


def find_by_request_user(
    request_id: "str | None", user_id: "str | None",
) -> "dict[str, Any] | None":
    """Return the most recent report row keyed on (user_id, request_id),
    or None. Used by upsert_report (so re-reporting the same trace
    updates the existing row instead of stacking duplicates) and by
    reports_routes.delete_my_report_api to target the caller's own row."""
    if not request_id or not user_id:
        return None
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM reports WHERE request_id = ? AND user_id = ?"
        " ORDER BY created_ts DESC LIMIT 1",
        (request_id, user_id),
    ).fetchone()
    return _row_to_dict(row) if row else None


# Resubmit merge reuses text_corrections.three_way_merge_corrections with an
# empty baseline — start from existing, overlay incoming on key match, dedupe
# anchorless on (wrong, correct). That helper is the single source of truth
# for chip-merge keying (anchored on (idx, idx_end), anchorless on
# (wrong, correct)) and the captures-routes group-PATCH path already uses it
# — keep the two stores in lockstep so future chip-merge tweaks land once.


def upsert_report(
    *,
    user_id: "str | None",
    request_id: "str | None",
    trace_ts: float,
    model: str,
    raw: str,
    final: str,
    steps: list,
    corrections: list,
    intended_text: str,
    user_comment: str,
    reporter_role: str,
    reporter_host: str,
    language: "str | None" = None,
    stages: "list | None" = None,
) -> "tuple[str, bool]":
    """Insert a new report or update the existing one keyed on
    (user_id, request_id). Returns (report_id, was_updated). The
    `was_updated` flag is True when an existing row was merged; False
    when a fresh row was inserted.

    On update: corrections go through three_way_merge_corrections —
    keyed on (idx, idx_end) for anchored chips and on (wrong, correct)
    for anchorless ones — intended_text and user_comment overwrite
    (latest submission supersedes), and created_ts bumps to "now" so
    the row re-floats to the top of /reports.
    """
    request_id = (request_id or None) and str(request_id)[:_CAP_REQUEST_ID]
    raw_t = (raw or "")[:_CAP_RAW]
    final_t = (final or "")[:_CAP_FINAL]
    steps_t = _truncate_steps(steps or [])
    corr_in = _clean_corrections(corrections or [])
    intended_t = (intended_text or "")[:_CAP_INTENDED]
    comment_t = (user_comment or "")[:_CAP_COMMENT]
    role_t = "admin" if reporter_role == "admin" else "user"
    now = time.time()
    trace_t = _clean_trace_ts(trace_ts, now)

    lang_t = (language or "")[:_CAP_LANGUAGE] or None
    # NULL, not "[]", when the blob cannot be stored: the UPDATE below
    # COALESCEs stages_json, so an oversized resubmission must keep what the
    # first submission recorded rather than blank it (transcriptions_store
    # .record does the same). _row_to_dict maps NULL to [] on read.
    stages_t: str | None = None
    if stages:
        try:
            cand = json.dumps(stages, ensure_ascii=False)[:_CAP_STAGES_JSON]
            json.loads(cand)   # truncation may have cut mid-token
            stages_t = cand
        except (TypeError, ValueError):
            stages_t = None

    conn = _require_conn()
    # Lookup and write share one lock span: with the lookup outside, two
    # concurrent submits of the same (user_id, request_id) could both see "no
    # existing row" and both INSERT — the index is not UNIQUE, so nothing else
    # deduplicates. find_by_request_user does not take _lock itself.
    with _lock:
        existing = find_by_request_user(request_id, user_id)
        if existing is not None:
            # Re-clean the union: three_way_merge_corrections returns current +
            # edited with no cap, so without this a caller resubmitting the same
            # request_id with fresh keys grows one row without bound. The captures
            # path already re-cleans via captures_store.update_capture.
            merged = _clean_corrections(text_corrections.three_way_merge_corrections(
                baseline=[], edited=corr_in, current=existing.get("corrections") or [],
            ))
            rid = existing["id"]
            conn.execute(
                "UPDATE reports SET"
                "  created_ts = ?, trace_ts = ?, model = ?, raw_text = ?, final_text = ?,"
                "  steps_json = ?, corrections_json = ?,"
                "  intended_text = ?, user_comment = ?,"
                "  reporter_role = ?, reporter_host = ?,"
                # COALESCE so a resubmission that omits the provenance never
                # blanks what the first submission recorded.
                "  language = COALESCE(?, language),"
                "  stages_json = COALESCE(?, stages_json),"
                "  status = 'open', resolved_ts = NULL"
                " WHERE id = ?",
                (
                    now, trace_t, model, raw_t, final_t,
                    json.dumps(steps_t, ensure_ascii=False),
                    json.dumps(merged, ensure_ascii=False),
                    intended_t, comment_t, role_t, reporter_host or "",
                    lang_t, stages_t,
                    rid,
                ),
            )
            logger.info("[reports] upsert-updated id=%s role=%s", rid[:8], role_t)
            return rid, True

        rid = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO reports ("
            " id, created_ts, trace_ts, request_id, model,"
            " raw_text, final_text, steps_json, corrections_json,"
            " intended_text, user_comment, reporter_role, reporter_host,"
            " status, admin_notes, resolved_ts,"
            " user_id, language, stages_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid, now, trace_t, request_id, model,
                raw_t, final_t,
                json.dumps(steps_t, ensure_ascii=False),
                json.dumps(corr_in, ensure_ascii=False),
                intended_t, comment_t, role_t, reporter_host or "",
                "open", "", None, user_id, lang_t, stages_t,
            ),
        )
        _evict_to_cap(conn)

    logger.info("[reports] created id=%s role=%s", rid[:8], role_t)
    return rid, False


def _evict_to_cap(conn: sqlite3.Connection) -> None:
    """Enforce REPORTS_MAX: when total > cap, delete oldest closed first,
    then oldest open. Two autocommit DELETEs (connection is in
    autocommit mode); logs only the count.

    Lazy-imports config so test harnesses that monkey-patch cfg pick up
    the current value. _lock is already held by the caller, so the COUNT
    sees a stable total even though the DELETEs are not transactionally
    atomic with the caller's INSERT."""
    try:
        import config as cfg
        cap = int(getattr(cfg, "REPORTS_MAX", 1000))
    except Exception:
        cap = 1000
    if cap < 1:
        return
    row = conn.execute("SELECT COUNT(*) FROM reports").fetchone()
    total = int(row[0]) if row else 0
    excess = total - cap
    if excess <= 0:
        return
    # Closed first: status != 'open', oldest by created_ts.
    closed = conn.execute(
        "DELETE FROM reports WHERE id IN ("
        "  SELECT id FROM reports WHERE status != 'open'"
        "  ORDER BY created_ts ASC LIMIT ?"
        ")",
        (excess,),
    ).rowcount
    remaining = excess - max(0, closed)
    if remaining > 0:
        open_deleted = conn.execute(
            "DELETE FROM reports WHERE id IN ("
            "  SELECT id FROM reports WHERE status = 'open'"
            "  ORDER BY created_ts ASC LIMIT ?"
            ")",
            (remaining,),
        ).rowcount
    else:
        open_deleted = 0
    if closed or open_deleted:
        logger.info(
            "[reports] evicted to cap: %d closed, %d open (cap=%d)",
            closed, open_deleted, cap,
        )


# ---------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------

def list_reports(user_id: str | None = None, *,
                 limit: "int | None" = _LIST_LIMIT) -> list[dict[str, Any]]:
    """Return reports newest-first. `user_id=None` means "no filter"
    (admin / scope=all context); a string narrows to a single owner.
    Client filters/searches in-page; the soft cap keeps the row count
    under what a browser can render.

    `limit=None` disables the read ceiling — the export path is the
    uncapped caller: REPORTS_MAX is operator-settable well above
    _LIST_LIMIT, and the "full JSON dump" must not silently drop
    everything past the newest 1000 rows.

    Symmetric to `captures_store.list_captures(user_id=...)`. The
    permission layer threads the right value via
    `Permissions.effective_user_id_for("reports", caller_uid)` so the
    "own vs all" decision lives in one place."""
    conn = _require_conn()
    # The soft cap the docstring relies on is an EVICTION cap, not a read cap:
    # nothing here bounded the row count, and each row decodes up to ~300 KB of
    # text plus a JSON parse of steps_json, inline in an async handler. A LIMIT
    # makes the documented ceiling real. _LIST_LIMIT is the eviction cap, so a
    # store at or under its own cap returns exactly what it returns today.
    # SQLite treats a negative LIMIT as "no limit".
    lim = -1 if limit is None else limit
    if user_id is None:
        cur = conn.execute(
            "SELECT * FROM reports ORDER BY created_ts DESC LIMIT ?",
            (lim,),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM reports WHERE user_id = ?"
            " ORDER BY created_ts DESC LIMIT ?",
            (user_id, lim),
        )
    return [_row_to_dict(r) for r in cur.fetchall()]


def get_report(rid: str) -> dict[str, Any] | None:
    conn = _require_conn()
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
    return _row_to_dict(row) if row else None


def recent_reports_for_user(
    user_id: str, limit: int = 100,
) -> list[dict[str, Any]]:
    """Return up to `limit` of the caller's most-recent open reports as
    full row dicts (includes corrections + intended_text). Newest-
    created-first. Feeds /quick-config so the page can re-render chips
    the user previously submitted, even after a hard reload.

    Filtered by user_id so other users' reports never leak into a
    different user's /quick-config view. status='open' is what the
    /quick-config UI treats as "still reported" (resolved/dismissed
    reports don't trigger the '✓ reported' badge either)."""
    if not user_id:
        return []
    conn = _require_conn()
    cur = conn.execute(
        "SELECT * FROM reports"
        " WHERE user_id = ? AND status = 'open' AND request_id IS NOT NULL"
        " ORDER BY created_ts DESC"
        " LIMIT ?",
        (user_id, int(limit)),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def counts_by_status(user_id: str | None = None) -> dict[str, int]:
    """Quick summary for the /reports page toolbar. `user_id=None` → all
    users (admin / scope=all); a string narrows to a single owner so a
    scope=own caller's toolbar counts match the rows they can see (and
    don't leak the global cross-user breakdown)."""
    conn = _require_conn()
    out = {"open": 0, "resolved": 0, "dismissed": 0}
    if user_id is None:
        cur = conn.execute(
            "SELECT status, COUNT(*) AS n FROM reports GROUP BY status"
        )
    else:
        cur = conn.execute(
            "SELECT status, COUNT(*) AS n FROM reports WHERE user_id = ?"
            " GROUP BY status", (user_id,),
        )
    for row in cur:
        if row["status"] in out:
            out[row["status"]] = int(row["n"])
    return out


# ---------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------

def update_report(rid: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    """Apply a partial update. Allowed fields: status, admin_notes.
    Returns the updated row dict or None if not found. Unknown fields
    are ignored silently (the route layer validates the shape; this is
    the last-line guard)."""
    if not patch:
        return get_report(rid)
    sets: list[str] = []
    params: list[Any] = []
    if "status" in patch:
        new_status = str(patch["status"] or "open")
        if new_status not in _VALID_STATUS:
            raise ValueError(f"invalid status: {new_status!r}")
        sets.append("status = ?")
        params.append(new_status)
        if new_status == "open":
            sets.append("resolved_ts = NULL")
        else:
            sets.append("resolved_ts = ?")
            params.append(time.time())
    if "admin_notes" in patch:
        notes = str(patch["admin_notes"] or "")[:_CAP_ADMIN_NOTES]
        sets.append("admin_notes = ?")
        params.append(notes)
    if not sets:
        return get_report(rid)
    params.append(rid)
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            f"UPDATE reports SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        if cur.rowcount == 0:
            return None
    return get_report(rid)


# ---------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------

def delete_report(rid: str) -> bool:
    """Delete a single report. Returns True if a row was removed."""
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM reports WHERE id = ?", (rid,))
        deleted = cur.rowcount > 0
    if deleted:
        logger.info("[reports] deleted id=%s", rid[:8])
    return deleted


def clear_all(reporter_host: str = "") -> int:
    """Wipe the entire table. Returns the count deleted. WARNING-logs
    the count + caller host for audit."""
    conn = _require_conn()
    with _lock:
        n = conn.execute("DELETE FROM reports").rowcount
    # VACUUM rewrites the entire DB file and blocks other writers — run
    # it outside _lock so unrelated writes (e.g. a fresh report submit
    # arriving during the wipe) aren't stalled. Connection is autocommit,
    # no transaction needed. Skip when nothing was deleted — VACUUM on an
    # already-empty table is pure I/O for zero space recovery.
    if n > 0:
        conn.execute("VACUUM")
    logger.warning(
        "[reports] admin from %s cleared %d reports",
        reporter_host or "<unknown>", n,
    )
    return n


# ---------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------

def sweep_retention() -> int:
    """Delete rows older than cfg.REPORTS_RETENTION_DAYS. Returns count
    deleted (0 when retention is disabled or nothing's old enough).
    Lazy-imports cfg so admin /settings edits take effect on next sweep."""
    try:
        import config as cfg
        days = int(getattr(cfg, "REPORTS_RETENTION_DAYS", 0))
    except Exception:
        return 0
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            "DELETE FROM reports WHERE created_ts < ?", (cutoff,)
        )
        n = cur.rowcount
    if n > 0:
        logger.warning(
            "[reports] retention sweep deleted %d rows older than %d days",
            n, days,
        )
    return n
