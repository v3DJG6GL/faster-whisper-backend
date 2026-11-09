"""Shared helpers for the persistent stores (SQLite and JSON files).

- `secure_db_file()` / `secure_file()` / `secure_dir()`: filesystem hardening.
- `log_safe()` / `LOG_FIELD_MAX`: cap + control-character screen for
  caller-supplied labels in log lines (used well beyond the stores).
- `LONE_SURROGATE_RE`: the lone-UTF-16-surrogate screen for client strings.
- `BUSY_TIMEOUT_S` / `open_wal_db()`: the one way a store opens its WAL DB;
  `close_quietly()` drops the previous one on a re-init; `transaction()`
  wraps an explicit BEGIN..COMMIT on it; `require_init()` is every store's
  "init_db() was not called" guard; `id_chunks()` splits an `IN (?,…)` id
  list into SQL_ID_CHUNK-sized pieces.
- `job_columns_ddl()` / `missing_job_columns()`: the job-label columns shared
  by usage_jobs and recent_transcriptions.
- `rename_columns()` / `add_missing_columns()`: the legacy column-rename and
  additive column migrations the stores run on init.
- `json_column()`: the tolerant decode of a JSON-bearing column;
  `json_object()`: the tolerant decode of client-sent JSON into a dict;
  `capped_json()`: a JSON blob under a byte cap, whole or not at all;
  `cfg_int()`: a live int knob read with a fallback.

Every store creates its DB at the process umask (0644 file / 0755 dir on a
typical Linux box) and nothing narrows it afterwards, yet those files hold
plaintext dictation, intended text, user comments, API-key hashes and
session-token hashes. `secure_db_file()` / `secure_dir()` tighten them to
owner-only right after init_db() has opened the connection.

Both are BEST-EFFORT: chmod is meaningless on Windows (the service runs there
too, see install-service.ps1) and unsupported on FAT/CIFS/9p mounts, and a
store must never fail to open over a file mode. OSError is swallowed.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator
from typing import Any

# Cap and control-character screen for a CALLER-supplied label that goes into a
# log line. Lives here rather than in main so the stores can reach it without an
# import cycle (main imports the stores, not the other way round).
LOG_FIELD_MAX = 120
# \x00-\x1f covers CR/LF and ESC (ANSI). Also screened: U+2028/U+2029, which
# the /logs viewer renders as a forced line break under `white-space: pre-wrap`
# (a forged record, like a bare LF); the BiDi overrides U+202A-U+202E /
# U+2066-U+2069, which reorder text within a line in a terminal and a browser;
# and C1/DEL (\x7f-\x9f), inert in every legitimate value. Deliberately NOT all
# of \p{Cf}: that would eat ZWNJ/ZWJ (U+200C/D) in real Persian and Indic
# filenames. Also lone UTF-16 surrogates (json.loads keeps "\ud800"): the
# utf-8 log file handler cannot encode one, so the WHOLE record would be
# dropped from the file /logs reads. Transcripts never pass through log_safe.
_LOG_UNSAFE_RE = re.compile("[\r\n\x00-\x1f\x7f-\x9f  ‪-‮⁦-⁩"
                            r"\ud800-\udfff]")


def log_safe(s, *, keep_end: bool = False) -> str:
    """Collapse control characters in a caller-supplied label and cap its
    length. A bare CR/LF would otherwise split one record into what the /logs
    viewer renders as extra, attacker-written lines — indistinguishable from
    genuine records, including their severity styling.

    `keep_end=True` keeps the LAST LOG_FIELD_MAX characters instead of the
    first: for a child's stderr tail, where the fatal line (yt-dlp's
    "ERROR: ...", ffmpeg's last line) comes at the end."""
    safe = _LOG_UNSAFE_RE.sub("?", s or "")
    return safe[-LOG_FIELD_MAX:] if keep_end else safe[:LOG_FIELD_MAX]


# A lone UTF-16 surrogate: json.loads keeps one ("\ud800"), but the tokenizer
# and UTF-8 encoding raise on it. Each client-string screen strips or rejects it.
LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# How long a statement waits for another connection's write lock before
# raising "database is locked". Within one process each store's _lock already
# serialises its writers; the wait matters when SERVER_WORKERS > 1 (sibling
# processes on the same file) and during WAL checkpoints. 5 s is the value
# Litestream and the SQLite docs recommend; pysqlite's own default happens to
# be the same, but the contract is spelled out here rather than inherited.
BUSY_TIMEOUT_S = 5.0


# Columns that describe ONE job and are written to BOTH per-job tables by the
# same call in metrics.record_transcription: usage_jobs (the long-lived
# numeric ledger, no text) and recent_transcriptions (the 30-day detail with
# the transcript). The labels are deliberately repeated in both — each table
# must answer "which model / how long / did it fail" on its own after the
# other has been pruned — so they are declared ONCE here and each schema
# renders them through job_columns_ddl(). A field added to one table and
# forgotten in the other fails test_job_columns.py.
#
# (name, SQLite type). Constraints (NOT NULL / DEFAULT) are per table: the
# ledger insists on values, the detail row tolerates NULL for an error-path
# insert that never reached the timing half.
JOB_COLUMNS: tuple[tuple[str, str], ...] = (
    ("user_id", "TEXT"),
    ("created_ts", "REAL"),
    ("kind", "TEXT"),
    ("status", "TEXT"),
    ("model", "TEXT"),
    ("language", "TEXT"),
    ("audio_s", "REAL"),
    ("processing_s", "REAL"),
    ("words", "INTEGER"),
    ("wait_s", "REAL"),
    ("error_class", "TEXT"),
    ("error_stage", "TEXT"),
)
JOB_COLUMN_NAMES: tuple[str, ...] = tuple(n for n, _ in JOB_COLUMNS)


def job_columns_ddl(constraints: dict[str, str] | None = None) -> str:
    """Render JOB_COLUMNS as CREATE TABLE lines, each ending in a comma, with
    the table's own constraint appended per column (e.g.
    {"user_id": "NOT NULL", "audio_s": "NOT NULL DEFAULT 0"}). Unknown keys
    in `constraints` raise: a typo there would silently drop a constraint."""
    constraints = dict(constraints or {})
    unknown = set(constraints) - set(JOB_COLUMN_NAMES)
    if unknown:
        raise ValueError(f"job_columns_ddl: not job columns: {sorted(unknown)}")
    lines = []
    for name, typ in JOB_COLUMNS:
        extra = constraints.get(name, "")
        lines.append(f"  {name:<13} {typ}{(' ' + extra) if extra else ''},")
    return "\n".join(lines)


def missing_job_columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """{column: reason} for every JOB_COLUMN the live table lacks or types
    differently. Empty means the table carries the full shared set."""
    have = {r["name"]: (r["type"] or "").upper()
            for r in conn.execute(f"PRAGMA table_info({table})")}
    out: dict[str, str] = {}
    for name, typ in JOB_COLUMNS:
        if name not in have:
            out[name] = "missing"
        elif have[name] != typ:
            out[name] = f"type {have[name]!r} != {typ!r}"
    return out


def rename_columns(conn: sqlite3.Connection, table: str,
                   pairs: Iterable[tuple[str, str]]) -> None:
    """Rename legacy columns in place (RENAME COLUMN keeps the data); a pair
    whose old name is gone or whose new name already exists is skipped. Run
    before CREATE TABLE IF NOT EXISTS so it never sees the old name as the
    live one."""
    have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    for old, new in pairs:
        if old in have and new not in have:
            conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
    conn.commit()


def add_missing_columns(conn: sqlite3.Connection, table: str,
                        columns: Iterable[tuple[str, str]]) -> None:
    """Additive migration: ALTER TABLE `table` ADD COLUMN for each
    (name, type-ddl) the live table lacks. CREATE TABLE IF NOT EXISTS never
    alters an existing table, so a column added later reaches an old DB only
    through here."""
    have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    for col, ddl in columns:
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def capped_json(value: Any, cap: int, *, allow_nan: bool = True,
                default: Any = None) -> str | None:
    """`value` as JSON, whole or not at all: None when it will not serialise
    or is over `cap`. The keywords pass straight to json.dumps."""
    try:
        blob = json.dumps(value, ensure_ascii=False, allow_nan=allow_nan, default=default)
    except (TypeError, ValueError):
        return None
    return blob if len(blob) <= cap else None


def json_column(raw: Any, empty: str) -> Any:
    """json.loads(raw or empty); a malformed or non-text value decodes as
    `empty` (a fresh container each call)."""
    try:
        return json.loads(raw or empty)
    except (TypeError, ValueError):
        return json.loads(empty)


def json_object(text: Any) -> "dict | None":
    """The dict client-sent `text` decodes to; None for malformed JSON or a
    non-object. RecursionError is what json raises on a deeply nested document
    (a ~400 KB "[[[[..." websocket frame fits ws_max_size), and it is a
    RuntimeError, not a ValueError: uncaught, a stream's blanket except
    Exception logs a full traceback with absolute paths, unthrottled, at a rate
    the client picks, and a batch request becomes a logged 500 and an err_count
    bump instead of "malformed → ignored"."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def cfg_int(name: str, default: int) -> int:
    """A live int knob; `default` when it is missing or unparseable. The
    config import stays lazy so a monkey-patched or /settings-edited value
    is read on every call."""
    try:
        from faster_whisper_backend.settings import config as cfg
        return int(getattr(cfg, name, default))
    except Exception:
        return default


def open_wal_db(path: str) -> sqlite3.Connection:
    """THE connection contract for the SQLite stores: create the parent dir,
    open `path` with a Row factory, a busy timeout, and switch it to WAL.
    Every store's init_db goes through here so a change to the contract lands
    in one place.

    isolation_level=None puts pysqlite in autocommit mode; every statement
    commits independently (each store's _lock serialises its writers). WAL
    gives crash-safety per statement; synchronous=NORMAL is the standard WAL
    recommendation (full durability against power loss is FULL, but NORMAL is
    fine against process crash and ~10x faster on small writes)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # timeout= is the busy timeout: pysqlite calls sqlite3_busy_timeout with
    # it, so no separate PRAGMA busy_timeout is needed.
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None,
                           timeout=BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def require_init[T](value: T | None, store: str) -> T:
    """Return `value`, or raise the shared RuntimeError naming `store` when
    its init_db() has not bound it yet (usage_routes maps it to a 503)."""
    if value is None:
        raise RuntimeError(f"{store}.init_db() was not called before use.")
    return value


# Ids per `WHERE id IN (?,?,…)` statement: under SQLite's classic 999
# bind-variable limit (a bulk request admits 1000 ids; a cap prune can drop
# tens of thousands of rows).
SQL_ID_CHUNK = 500


def id_chunks(ids: list[Any], size: int = SQL_ID_CHUNK) -> Iterator[tuple[list[Any], str]]:
    """Yield (chunk, "?,?,…") for an `IN (…)` over `ids`, `size` at a time,
    in order. The SQL around the marks stays at the caller."""
    for off in range(0, len(ids), size):
        chunk = ids[off:off + size]
        yield chunk, ",".join("?" * len(chunk))


def close_quietly(conn: sqlite3.Connection | None) -> None:
    """Close a store connection that a re-init is replacing; a failed close
    is ignored (the re-init must not leak the old handle plus its WAL/-shm
    files, nor fail over it). The caller still unbinds its global before
    opening the new one, so a failed open leaves no closed handle bound."""
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def unlink_quietly(path: str | None) -> None:
    """Best-effort os.unlink: no path is a no-op; any OSError (incl. already
    gone) is ignored."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection, begin: str = "BEGIN") -> Iterator[None]:
    """One explicit transaction on an autocommit (open_wal_db) connection,
    where `with conn:` would open none. COMMIT runs inside the guarded span: a
    failed COMMIT (SQLITE_FULL / IOERR / BUSY) can leave the transaction open
    on the shared connection, and every later BEGIN would then fail. ROLLBACK
    only while still in a transaction: SQLite may already have rolled back on
    its own, and a bare ROLLBACK would then raise and replace the real error.
    The caller holds its store lock(s) around the whole `with`."""
    conn.execute(begin)
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


# WAL keeps recently written rows in the -wal sidecar until a checkpoint, and
# -shm carries the index into it, so both need the same mode as the DB itself.
_WAL_SIDECARS = ("-wal", "-shm")


def secure_db_file(db_path: str) -> None:
    """chmod a store's DB file (and its WAL sidecars) to 0600 and the
    directory holding it to 0700."""
    _chmod(os.path.dirname(os.path.abspath(db_path)) or ".", 0o700)
    _chmod(db_path, 0o600)
    for suffix in _WAL_SIDECARS:
        _chmod(db_path + suffix, 0o600)


def secure_dir(path: str) -> None:
    """chmod a data directory (e.g. the raw capture WAV root) to 0700."""
    _chmod(path, 0o700)


def secure_file(path: str) -> None:
    """chmod a single data file to 0600.

    For non-SQLite surfaces that hold the same plaintext dictation or secrets
    as the stores: the rotating server log, config.local.json, the runtime
    caches, downloaded media and the merged / VAD-trimmed WAVs. In a
    tmp+os.replace swap call it on the tmp: os.replace carries the tmp inode's
    mode onto the destination.
    """
    _chmod(path, 0o600)


def _chmod(path: str, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass  # unsupported filesystem, foreign owner, or not yet created
