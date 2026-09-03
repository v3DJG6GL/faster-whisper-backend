"""Durable per-key / per-user usage rollup.

SQLite (stdlib) in WAL mode — single-file, crash-safe, indexed. Lives at
cfg.USAGE_DB (defaults to usage.local.sqlite3 alongside config.local.json).

Why a separate store: the recent-transcriptions table
(recent_transcriptions_store.py) is a pruned rolling window (row-cap + 30-day TTL),
so it cannot back lifetime usage totals. This store keeps compact HOURLY
ROLLUPS that are never aggressively pruned, so lifetime totals are a SUM over
hours.

Tables (every rollup is keyed by UTC epoch-hour, see below):

  usage_hourly            hour × key_id × kind — requests/errors/words/audio_s/
                          processing_s and `sessions` (jobs, counted once per job_id)
  usage_jobs              one row per job_id (client-minted): the per-job sums
                          plus the dictation OUTCOME the desktop app reports
                          afterwards (activation / delivery / app / translation)
  usage_job_stages        job_id × stage — what the optional stages cost
  usage_stage_hourly      hour × user × stage — stage runs, seconds, speakers…
  usage_target_hourly     hour × user × stage × target — translation targets
  usage_dictation_hourly  hour × user × activation × delivery × translation
  usage_app_hourly        hour × user × app_id — where dictations landed

`kind` is one of dictation / file / url / text. Rows written before the kind
column existed are folded into 'dictation' (nearly all of that history was
dictation, and a per-kind chart that hid it read as lost data); a write with
a kind the server does not know lands as 'unknown' and only ever contributes
to the all-kinds totals.

Each /transcribe request (and each dictation utterance, and each text
translation) bumps the rollups via record_usage, called from
metrics.record_transcription (which already runs inside a try/except on the
outer finally of the transcribe handler, on both success and error paths).
A dictation SESSION is many utterances under one job_id; `sessions` is bumped
only for the utterance that creates the job row, so "Runs" counts sessions,
not phrases.

**Bucketing is UTC epoch-hour** (`int(ts // 3600)`). Storing in UTC at hour
granularity lets every consumer reckon "days" in whatever timezone it wants by
summing the hours that fall inside that timezone's local day:

  - the desktop app's /v1/usage document reckons in the CALLER's IANA zone
    (`tz` param → zoneinfo), falling back to the server's local zone;
  - the admin /stats + /api-keys dashboards reckon in the SERVER's local
    timezone (the operator's perspective), via local_day_start_hour() and
    epoch_day_for().

`series()` aggregates hours into server-local days and returns days-since-epoch,
so `day * 86400` is still UTC midnight of that calendar date and the WebUI's
`new Date(day*86400*1000).toISOString().slice(0,10)` renders the right label.

Hour granularity is exact for whole-hour UTC offsets (CET/CEST = +1/+2). For
half-hour-offset zones (e.g. IST +5:30) a transcription in the partial hour at
a day boundary can land in the adjacent day — acceptable at this grain.

user_id is denormalised into every row (not resolved via JOIN) so revoking
a user or key never drops their historical usage, and aggregation needs no
join to the api-keys DB.

Module-level connection: WAL mode lets us share one connection across
threads (check_same_thread=False); _lock serialises writers. The connection
is in autocommit mode, so the multi-table writes below open an explicit
transaction — a job row without its rollup (or vice versa) would make the
session and utterance counts disagree forever.

Do not log row content — dictation data can be sensitive (an app_id names
the program a user dictated into). Module log lines carry only counts and id
prefixes.
"""
from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Sequence
import logging
import sqlite3
import threading
import time
import zoneinfo
from typing import Any

from faster_whisper_backend.core import store_common

logger = logging.getLogger("whisper-api")

_EPOCH = datetime.date(1970, 1, 1)

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

# Sentinel for rows we can't attribute to a real key/user. Kept as a literal
# id string so the NOT NULL columns stay satisfied and aggregation treats it
# as its own bucket. The UI renders it as a plain label.
OPEN_MODE_ID = "(open-mode)"

# ORDER BY column names are interpolated, so they MUST come from this set.
# processing_s (server processing seconds) and sessions (distinct dictation sessions
# / file runs, as opposed to HTTP requests) were written from day one and
# only read by document(); /stats reads them too since v2.
_METRICS: frozenset[str] = frozenset(
    ("requests", "errors", "words", "audio_s", "processing_s", "sessions"))

# Job kinds the per-kind breakdown knows. Anything else is folded into
# 'unknown' rather than rejected: a usage write must never fail a request.
KINDS: tuple[str, ...] = ("dictation", "file", "url", "text")
UNKNOWN_KIND = "unknown"

# Stage names as the batch handler emits them. The text-translation endpoint
# names its one stage 'translate'; it is the same work as the batch
# 'translating' stage and must land in the same row.
_STAGE_ALIASES = {"translate": "translating"}

# Which jobs a stage COULD have run on — the denominator of the "ran on N of
# M" meter. Translation applies to every kind (a dictation is translated via
# the text endpoint, a file inline); the audio stages only to batch inputs.
STAGE_APPLIES_TO: dict[str, tuple[str, ...]] = {
    "translating": KINDS,
    "diarizing": ("file", "url"),
    "separating": ("file", "url"),
    "vad": ("file", "url"),
}
# Every stage the rollup records, in pipeline order, with the kinds it can
# run on: the decode itself and the URL download included. document()
# lists only the optional stages (the desktop app's Statistics screen);
# document(all_stages=True) — the /stats console — lists all of them.
STAGE_ELIGIBLE: dict[str, tuple[str, ...]] = {
    "downloading": ("url",),
    "separating": ("file", "url"),
    "vad": ("file", "url"),
    "transcribing": ("file", "url", "dictation"),
    "diarizing": ("file", "url"),
    "translating": KINDS,
}

# The desktop app's outcome vocabulary. Validated at the route (pydantic
# enums); repeated here so the sweep's 'unreported' marker and the document's
# fixed bucket lists come from one place.
ACTIVATIONS: tuple[str, ...] = ("hold", "handsfree")
DELIVERIES: tuple[str, ...] = ("typed", "clipboard", "none", "unreported")
TRANSLATIONS: tuple[str, ...] = ("translated", "kept_original", "not_asked",
                                 "aborted", "unreported")
UNREPORTED = "unreported"

# Reading speed the "time saved" figure is measured against: a dictated
# word would have taken 1/40 min to type at a comfortable 40 wpm.
TYPING_WPM = 40.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_hourly (
  hour     INTEGER NOT NULL,
  key_id   TEXT    NOT NULL,
  user_id  TEXT    NOT NULL,
  kind     TEXT    NOT NULL DEFAULT 'unknown',
  requests INTEGER NOT NULL DEFAULT 0,
  errors   INTEGER NOT NULL DEFAULT 0,
  words    INTEGER NOT NULL DEFAULT 0,
  audio_s  REAL    NOT NULL DEFAULT 0,
  processing_s   REAL    NOT NULL DEFAULT 0,
  sessions INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, key_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_usage_user_hour ON usage_hourly(user_id, hour);
CREATE INDEX IF NOT EXISTS idx_usage_hour      ON usage_hourly(hour);

CREATE TABLE IF NOT EXISTS usage_jobs (
  job_id      TEXT    PRIMARY KEY,
  key_id      TEXT    NOT NULL,
""" + store_common.job_columns_ddl({
    "user_id": "NOT NULL", "created_ts": "NOT NULL", "kind": "NOT NULL",
    "status": "NOT NULL", "audio_s": "NOT NULL DEFAULT 0",
    "words": "NOT NULL DEFAULT 0", "processing_s": "NOT NULL DEFAULT 0",
    "wait_s": "NOT NULL DEFAULT 0",
}) + """
  utterances  INTEGER NOT NULL DEFAULT 0,
  activation  TEXT,
  delivery    TEXT,
  app_id      TEXT,
  translation TEXT,
  reported_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_usage_jobs_created ON usage_jobs(created_ts);
CREATE INDEX IF NOT EXISTS idx_usage_jobs_user_created ON usage_jobs(user_id, created_ts);

CREATE TABLE IF NOT EXISTS usage_job_stages (
  job_id   TEXT NOT NULL,
  stage    TEXT NOT NULL,
  secs     REAL NOT NULL DEFAULT 0,
  model    TEXT,
  targets  TEXT,
  speakers INTEGER,
  retained REAL,
  error    TEXT,
  PRIMARY KEY (job_id, stage)
);

CREATE TABLE IF NOT EXISTS usage_stage_hourly (
  hour          INTEGER NOT NULL,
  user_id       TEXT    NOT NULL,
  stage         TEXT    NOT NULL,
  runs          INTEGER NOT NULL DEFAULT 0,
  audio_s       REAL    NOT NULL DEFAULT 0,
  secs          REAL    NOT NULL DEFAULT 0,
  speakers      INTEGER NOT NULL DEFAULT 0,
  retained_sum  REAL    NOT NULL DEFAULT 0,
  kept_original INTEGER NOT NULL DEFAULT 0,
  errors        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, user_id, stage)
);

CREATE TABLE IF NOT EXISTS usage_target_hourly (
  hour    INTEGER NOT NULL,
  user_id TEXT    NOT NULL,
  stage   TEXT    NOT NULL,
  target  TEXT    NOT NULL,
  runs    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, user_id, stage, target)
);

CREATE TABLE IF NOT EXISTS usage_dictation_hourly (
  hour        INTEGER NOT NULL,
  user_id     TEXT    NOT NULL,
  activation  TEXT    NOT NULL,
  delivery    TEXT    NOT NULL,
  translation TEXT    NOT NULL,
  sessions    INTEGER NOT NULL DEFAULT 0,
  words       INTEGER NOT NULL DEFAULT 0,
  audio_s     REAL    NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, user_id, activation, delivery, translation)
);

CREATE TABLE IF NOT EXISTS usage_app_hourly (
  hour     INTEGER NOT NULL,
  user_id  TEXT    NOT NULL,
  app_id   TEXT    NOT NULL,
  sessions INTEGER NOT NULL DEFAULT 0,
  words    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, user_id, app_id)
);
"""


def init_db(path: str) -> None:
    """Open (or create) the DB at `path` in WAL mode. Idempotent — call
    once on service startup before any other function. Mirrors
    recent_transcriptions_store.init_db."""
    global _conn
    _conn = store_common.open_wal_db(path)
    _conn.execute("PRAGMA temp_store=MEMORY;")
    _park_legacy_hourly(_conn)
    _rename_columns(_conn)
    _conn.executescript(_SCHEMA)
    _migrate_columns(_conn)
    _fold_legacy_hourly(_conn)
    _reclassify_unknown_hourly(_conn)
    store_common.secure_db_file(path)


# Additive columns for DBs created before them. CREATE TABLE IF NOT EXISTS is
# a no-op against an existing table, so a column that joins the schema must
# also be listed here or it never reaches a live database. Same pattern as
# recent_transcriptions_store.init_db; idempotent (PRAGMA table_info first).
_COLUMN_MIGRATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "usage_jobs": (
        ("wait_s", "ADD COLUMN wait_s REAL NOT NULL DEFAULT 0"),
        ("error_class", "ADD COLUMN error_class TEXT"),
        ("error_stage", "ADD COLUMN error_stage TEXT"),
    ),
    "usage_job_stages": (
        ("error", "ADD COLUMN error TEXT"),
    ),
    "usage_stage_hourly": (
        ("errors", "ADD COLUMN errors INTEGER NOT NULL DEFAULT 0"),
    ),
}


def _migrate_columns(conn: sqlite3.Connection) -> None:
    for table, cols in _COLUMN_MIGRATIONS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, ddl in cols:
            if col not in have:
                conn.execute(f"ALTER TABLE {table} {ddl}")
    conn.commit()


def _rename_columns(conn: sqlite3.Connection) -> None:
    """Column renames (2026-09): processing_s spells its unit the way
    audio_s does and is what every other store now calls the same
    measurement. Runs before _SCHEMA (CREATE TABLE IF NOT EXISTS would
    otherwise leave the old name in place); RENAME COLUMN keeps the data."""
    for table in ("usage_hourly", "usage_jobs"):
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "proc_s" in have and "processing_s" not in have:
            conn.execute(f"ALTER TABLE {table} RENAME COLUMN proc_s TO processing_s")
    conn.commit()


def _park_legacy_hourly(conn: sqlite3.Connection) -> None:
    """First half of the pre-`kind` rebuild: rename the old usage_hourly
    aside. Its primary key grew a column, which ALTER TABLE cannot express,
    so _SCHEMA re-creates the table; the old indexes are dropped first
    because SQLite index names are database-global and CREATE INDEX IF NOT
    EXISTS would otherwise find them attached to the parked table."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(usage_hourly)")}
    if not cols or "kind" in cols:
        return
    conn.executescript(
        "DROP INDEX IF EXISTS idx_usage_user_hour;"
        "DROP INDEX IF EXISTS idx_usage_hour;"
        "ALTER TABLE usage_hourly RENAME TO usage_hourly_legacy;"
    )


def _fold_legacy_hourly(conn: sqlite3.Connection) -> None:
    """Second half: copy the parked rows into the new table as 'dictation'
    and drop the parking table. Old rows know no kind and no job, so every
    request counts as its own session — the truest reading of history that
    was recorded per request. Copy + drop share a transaction so a crash in
    between cannot leave the rows counted twice on the next start."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='usage_hourly_legacy'"
    ).fetchone()
    if exists is None:
        return
    with _lock:
        conn.execute("BEGIN")
        try:
            cur = conn.execute(
                "INSERT INTO usage_hourly"
                " (hour, key_id, user_id, kind, requests, errors, words,"
                "  audio_s, processing_s, sessions)"
                " SELECT hour, key_id, user_id, ?, requests, errors, words,"
                "  audio_s, 0, requests"
                " FROM usage_hourly_legacy",
                ("dictation",),
            )
            conn.execute("DROP TABLE usage_hourly_legacy")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    logger.info("[usage] migrated %d hourly rows to the per-kind rollup",
                cur.rowcount or 0)


def _reclassify_unknown_hourly(conn: sqlite3.Connection) -> None:
    """Fold 'unknown' hourly rows into 'dictation'. Servers that ran the
    first per-kind build folded their history as 'unknown', which the
    per-kind chart cannot draw; the history was dictation. Sums into an
    existing dictation row for the same hour×key (the hour of the update
    can hold both), then drops the unknown rows. Idempotent: a clean DB
    has nothing to fold."""
    n = conn.execute(
        "SELECT COUNT(*) FROM usage_hourly WHERE kind=?", (UNKNOWN_KIND,)
    ).fetchone()[0]
    if not n:
        return
    with _lock:
        conn.execute("BEGIN")
        try:
            conn.execute(
                "INSERT INTO usage_hourly"
                " (hour, key_id, user_id, kind, requests, errors, words,"
                "  audio_s, processing_s, sessions)"
                " SELECT hour, key_id, user_id, 'dictation', requests, errors,"
                "  words, audio_s, processing_s, sessions"
                " FROM usage_hourly WHERE kind=?"
                " ON CONFLICT(hour, key_id, kind) DO UPDATE SET"
                "  requests = requests + excluded.requests,"
                "  errors   = errors + excluded.errors,"
                "  words    = words + excluded.words,"
                "  audio_s  = audio_s + excluded.audio_s,"
                "  processing_s   = processing_s + excluded.processing_s,"
                "  sessions = sessions + excluded.sessions",
                (UNKNOWN_KIND,),
            )
            conn.execute("DELETE FROM usage_hourly WHERE kind=?", (UNKNOWN_KIND,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    logger.info("[usage] reclassified %d unknown-kind hourly rows as dictation", n)


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("usage_store.init_db() was not called before use.")
    return _conn


# --- time helpers ---------------------------------------------------------

def now_hour() -> int:
    return int(time.time() // 3600)


def hour_for_ts(ts: float) -> int:
    """UTC epoch-hour containing `ts`."""
    return int(float(ts) // 3600)


def epoch_day_for(ts: float) -> int:
    """Days-since-epoch of the SERVER-LOCAL calendar date containing `ts`.
    `date.fromtimestamp` resolves in local time (DST-safe — calendar dates,
    not fixed offsets). Used to roll hourly rows into server-local days."""
    return (datetime.date.fromtimestamp(ts) - _EPOCH).days


def local_day_start_hour(days_ago: int = 0) -> int:
    """UTC epoch-hour of SERVER-LOCAL midnight `days_ago` days back (0 =
    today). `datetime(date)` has no tzinfo → its .timestamp() interprets the
    naive value in local time, so this is the correct local-midnight instant
    even across DST. Used by the admin /stats + /api-keys windows.

    Clamped here rather than at each call site: stats_routes bounds `days` to
    3650 before calling, the /settings/api-keys usage window does not, and an
    unbounded value overflows the date arithmetic into an unhandled 500."""
    days_ago = max(0, min(int(days_ago), 3650))
    d = datetime.date.today() - datetime.timedelta(days=days_ago)
    midnight_ts = datetime.datetime(d.year, d.month, d.day).timestamp()
    return int(midnight_ts // 3600)


def resolve_tz(name: str | None) -> zoneinfo.ZoneInfo | None:
    """The caller's IANA zone, or None for "reckon in the server's local
    zone". Anything zoneinfo does not know — including an empty or absurdly
    long string — falls back rather than erroring: the zone only decides
    where the day boundaries fall, and a wrong-but-consistent boundary beats
    a broken statistics page."""
    if not name or len(name) > 64:
        return None
    try:
        return zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError, OSError):
        return None


def _date_of(ts: float, tz: zoneinfo.ZoneInfo | None) -> datetime.date:
    """Calendar date containing `ts` in `tz` (None = server-local)."""
    return datetime.datetime.fromtimestamp(ts, tz).date()


def _midnight_hour(day: datetime.date, tz: zoneinfo.ZoneInfo | None) -> int:
    """UTC epoch-hour of local midnight on `day` in `tz`. A naive datetime
    resolves in the server's local zone, an aware one in its own — both are
    calendar arithmetic, so DST transitions land where the zone says."""
    midnight = datetime.datetime(day.year, day.month, day.day, tzinfo=tz)
    return int(midnight.timestamp() // 3600)


# --- write path -----------------------------------------------------------

def _norm_kind(kind: str | None) -> str:
    return kind if kind in KINDS else UNKNOWN_KIND


def _stage_rows(stages: list | None) -> list[dict[str, Any]]:
    """Reduce the handler's stage timing dicts to what the rollup keeps.
    Only structured keys are read — the human `detail` string stays where
    it is (the recent-jobs receipt) and is never parsed back."""
    out: list[dict[str, Any]] = []
    for s in stages or []:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if not isinstance(name, str) or not name:
            continue
        targets = s.get("targets")
        speakers = s.get("speakers")
        retained = s.get("retained")
        kept = s.get("kept_original")
        error = s.get("error")
        out.append({
            "stage": _STAGE_ALIASES.get(name, name),
            "secs": float(s.get("secs") or 0.0),
            "model": s.get("model") if isinstance(s.get("model"), str) else None,
            # A soft-failed stage (the job went on without it) carries its
            # error class here; the job's own status stays ok.
            "error": (error[:32] if isinstance(error, str) and error else None),
            "targets": ([str(t) for t in targets if t]
                        if isinstance(targets, (list, tuple)) else []),
            "speakers": int(speakers) if isinstance(speakers, (int, float)) else None,
            "retained": float(retained) if isinstance(retained, (int, float)) else None,
            "kept_original": int(kept) if isinstance(kept, (int, float)) else 0,
        })
    return out


def record_usage(
    *,
    key_id: str | None,
    user_id: str | None,
    audio_s: float | None,
    words: int | None,
    status: str,
    hour: int | None = None,
    kind: str | None = None,
    job_id: str | None = None,
    stages: list | None = None,
    model: str | None = None,
    language: str | None = None,
    processing_s: float | None = None,
    wait_s: float | None = None,
    error_class: str | None = None,
    error_stage: str | None = None,
) -> None:
    """Record one transcription request (a batch run, a text translation, or
    ONE dictation utterance) into the rollups. Best-effort: any failure is
    logged, never raised — a usage write must not break a transcription.
    Falsy ids fall back to the open-mode sentinel so the NOT NULL columns
    stay valid.

    `wait_s` is the time this request spent queued for a GPU slot (summed
    per job like processing_s); `error_class` / `error_stage` classify a failed
    job (metrics.ERROR_CLASSES) and are kept once set — a later utterance
    of the same session never blanks them.

    `job_id` groups utterances into a session: the first record under an id
    creates the job row and counts the session; later ones only add to its
    sums. Without a job id every request is its own session. `hour` lets
    tests seed history; the job row is then stamped at that hour too."""
    try:
        kid = key_id or OPEN_MODE_ID
        uid = user_id or OPEN_MODE_ID
        h = now_hour() if hour is None else int(hour)
        created_ts = time.time() if hour is None else h * 3600
        err = 0 if status == "ok" else 1
        w = int(words or 0)
        a = float(audio_s or 0.0)
        p = float(processing_s or 0.0)
        wt = max(0.0, float(wait_s or 0.0))
        ecls = (error_class[:32] if isinstance(error_class, str) and error_class else None)
        estg = (error_stage[:32] if isinstance(error_stage, str) and error_stage else None)
        k = _norm_kind(kind)
        jid = job_id[:64] if isinstance(job_id, str) and job_id else None
        stage_rows = _stage_rows(stages)
        conn = _require_conn()
        with _lock:
            conn.execute("BEGIN")
            try:
                new_session = True
                if jid is not None:
                    owner = conn.execute(
                        "SELECT user_id FROM usage_jobs WHERE job_id = ?", (jid,)
                    ).fetchone()
                    if owner is not None and owner["user_id"] != uid:
                        # Client-minted ids can collide across users in
                        # theory: count the work, never merge it into
                        # someone else's job.
                        jid = None
                    else:
                        new_session = owner is None
                        conn.execute(
                            "INSERT INTO usage_jobs"
                            " (job_id, user_id, key_id, kind, created_ts, status,"
                            "  audio_s, words, processing_s, utterances, model, language,"
                            "  wait_s, error_class, error_stage)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)"
                            " ON CONFLICT(job_id) DO UPDATE SET"
                            "  utterances  = utterances + 1,"
                            "  audio_s     = audio_s + excluded.audio_s,"
                            "  words       = words + excluded.words,"
                            "  processing_s      = processing_s + excluded.processing_s,"
                            "  status      = excluded.status,"
                            "  model       = COALESCE(excluded.model, model),"
                            "  language    = COALESCE(excluded.language, language),"
                            "  wait_s      = wait_s + excluded.wait_s,"
                            "  error_class = COALESCE(error_class, excluded.error_class),"
                            "  error_stage = COALESCE(error_stage, excluded.error_stage)",
                            (jid, uid, kid, k, created_ts, status, a, w, p,
                             model or None, language or None, wt, ecls, estg),
                        )
                conn.execute(
                    "INSERT INTO usage_hourly"
                    " (hour, key_id, user_id, kind, requests, errors, words,"
                    "  audio_s, processing_s, sessions)"
                    " VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(hour, key_id, kind) DO UPDATE SET"
                    "  requests = requests + 1,"
                    "  errors   = errors + excluded.errors,"
                    "  words    = words  + excluded.words,"
                    "  audio_s  = audio_s + excluded.audio_s,"
                    "  processing_s   = processing_s + excluded.processing_s,"
                    "  sessions = sessions + excluded.sessions,"
                    "  user_id  = excluded.user_id",
                    (h, kid, uid, k, err, w, a, p, 1 if new_session else 0),
                )
                for st in stage_rows:
                    _record_stage(conn, h, uid, jid, a, st)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    except Exception as e:
        logger.warning("[usage] record_usage failed: %s", e)


def _record_stage(conn: sqlite3.Connection, hour: int, uid: str,
                  jid: str | None, audio_s: float, st: dict[str, Any]) -> None:
    """One stage of one request into the per-job detail row and the hourly
    stage/target rollups. A dictation's stage repeats per utterance, so the
    per-job row accumulates seconds instead of failing on the key."""
    err = st.get("error")
    if jid is not None:
        conn.execute(
            "INSERT INTO usage_job_stages"
            " (job_id, stage, secs, model, targets, speakers, retained, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(job_id, stage) DO UPDATE SET"
            "  secs     = secs + excluded.secs,"
            "  model    = COALESCE(excluded.model, model),"
            "  targets  = COALESCE(excluded.targets, targets),"
            "  speakers = COALESCE(excluded.speakers, speakers),"
            "  retained = COALESCE(excluded.retained, retained),"
            "  error    = COALESCE(error, excluded.error)",
            (jid, st["stage"], st["secs"], st["model"],
             ",".join(st["targets"]) or None, st["speakers"], st["retained"], err),
        )
    conn.execute(
        "INSERT INTO usage_stage_hourly"
        " (hour, user_id, stage, runs, audio_s, secs, speakers, retained_sum,"
        "  kept_original, errors)"
        " VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(hour, user_id, stage) DO UPDATE SET"
        "  runs          = runs + 1,"
        "  audio_s       = audio_s + excluded.audio_s,"
        "  secs          = secs + excluded.secs,"
        "  speakers      = speakers + excluded.speakers,"
        "  retained_sum  = retained_sum + excluded.retained_sum,"
        "  kept_original = kept_original + excluded.kept_original,"
        "  errors        = errors + excluded.errors",
        (hour, uid, st["stage"], audio_s, st["secs"], st["speakers"] or 0,
         st["retained"] or 0.0, st["kept_original"], 1 if err else 0),
    )
    for target in st["targets"]:
        conn.execute(
            "INSERT INTO usage_target_hourly (hour, user_id, stage, target, runs)"
            " VALUES (?, ?, ?, ?, 1)"
            " ON CONFLICT(hour, user_id, stage, target) DO UPDATE SET"
            "  runs = runs + 1",
            (hour, uid, st["stage"], target[:16]),
        )


def _roll_outcome(conn: sqlite3.Connection, job: sqlite3.Row,
                  activation: str, delivery: str, translation: str,
                  app_id: str | None) -> None:
    """Fold a reported (or sweep-marked) job into the dictation rollups.
    Bucketed by the job's OWN hour, not the report's: an outcome posted after
    midnight still belongs to the evening it was dictated."""
    hour = hour_for_ts(float(job["created_ts"]))
    uid = job["user_id"]
    words = int(job["words"] or 0)
    audio_s = float(job["audio_s"] or 0.0)
    conn.execute(
        "INSERT INTO usage_dictation_hourly"
        " (hour, user_id, activation, delivery, translation, sessions, words,"
        "  audio_s)"
        " VALUES (?, ?, ?, ?, ?, 1, ?, ?)"
        " ON CONFLICT(hour, user_id, activation, delivery, translation)"
        " DO UPDATE SET"
        "  sessions = sessions + 1,"
        "  words    = words + excluded.words,"
        "  audio_s  = audio_s + excluded.audio_s",
        (hour, uid, activation, delivery, translation, words, audio_s),
    )
    if app_id:
        conn.execute(
            "INSERT INTO usage_app_hourly (hour, user_id, app_id, sessions, words)"
            " VALUES (?, ?, ?, 1, ?)"
            " ON CONFLICT(hour, user_id, app_id) DO UPDATE SET"
            "  sessions = sessions + 1,"
            "  words    = words + excluded.words",
            (hour, uid, app_id, words),
        )


def record_outcome(
    *,
    user_id: str,
    job_id: str,
    activation: str,
    delivery: str,
    translation: str,
    app_id: str | None = None,
) -> str:
    """Attach the desktop app's outcome to one dictation job. Returns
    'accepted' on the first report and 'duplicate' for every later one (or
    for a job that belongs to someone else — indistinguishable on purpose).

    A job the server never saw (a session with no finished utterance, or
    one whose utterances raced the report) gets a stub row so the outcome
    still counts as a session; the stub carries no words or audio because
    the server only trusts its own utterance rows for those. Raises on
    store failure — the route maps that to a 5xx, unlike record_usage."""
    conn = _require_conn()
    now = time.time()
    with _lock:
        conn.execute("BEGIN")
        try:
            cur = conn.execute(
                "UPDATE usage_jobs SET activation = ?, delivery = ?,"
                " translation = ?, app_id = ?, reported_ts = ?"
                " WHERE job_id = ? AND user_id = ? AND reported_ts IS NULL",
                (activation, delivery, translation, app_id, now, job_id, user_id),
            )
            if cur.rowcount == 0:
                exists = conn.execute(
                    "SELECT 1 FROM usage_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if exists is not None:
                    conn.execute("COMMIT")
                    return "duplicate"
                conn.execute(
                    "INSERT INTO usage_jobs"
                    " (job_id, user_id, key_id, kind, created_ts, status,"
                    "  activation, delivery, translation, app_id, reported_ts)"
                    " VALUES (?, ?, ?, 'dictation', ?, 'ok', ?, ?, ?, ?, ?)",
                    (job_id, user_id, OPEN_MODE_ID, now, activation, delivery,
                     translation, app_id, now),
                )
            job = conn.execute(
                "SELECT user_id, created_ts, words, audio_s FROM usage_jobs"
                " WHERE job_id = ?", (job_id,),
            ).fetchone()
            _roll_outcome(conn, job, activation, delivery, translation, app_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return "accepted"


def sweep(
    *,
    unreported_after_h: int,
    jobs_retention_days: int,
    app_retention_days: int,
    hourly_retention_days: int,
) -> dict[str, int]:
    """Hourly maintenance: close out dictation jobs whose outcome never
    arrived, then prune. Returns counts for the log line.

    A dictation older than `unreported_after_h` with no outcome is rolled
    into the dictation buckets as 'unreported' (activation included, so the
    row's NOT NULL key is satisfied without inventing a mode) — the client
    was closed, crashed, or has the report turned off, and the session
    still happened. Only dictation jobs are eligible: batch and text jobs
    never report an outcome. Retention: 0 keeps everything."""
    conn = _require_conn()
    now = time.time()
    marked = pruned_jobs = pruned_apps = 0
    with _lock:
        conn.execute("BEGIN")
        try:
            stale = conn.execute(
                "SELECT job_id, user_id, created_ts, words, audio_s"
                " FROM usage_jobs"
                " WHERE kind = 'dictation' AND reported_ts IS NULL"
                "   AND created_ts < ?",
                (now - max(1, int(unreported_after_h)) * 3600,),
            ).fetchall()
            for job in stale:
                conn.execute(
                    "UPDATE usage_jobs SET activation = ?, delivery = ?,"
                    " translation = ?, reported_ts = ? WHERE job_id = ?",
                    (UNREPORTED, UNREPORTED, UNREPORTED, now, job["job_id"]),
                )
                _roll_outcome(conn, job, UNREPORTED, UNREPORTED, UNREPORTED, None)
                marked += 1
            if jobs_retention_days > 0:
                cutoff = now - int(jobs_retention_days) * 86400
                conn.execute(
                    "DELETE FROM usage_job_stages WHERE job_id IN"
                    " (SELECT job_id FROM usage_jobs WHERE created_ts < ?)",
                    (cutoff,),
                )
                pruned_jobs = conn.execute(
                    "DELETE FROM usage_jobs WHERE created_ts < ?", (cutoff,)
                ).rowcount or 0
            if app_retention_days > 0:
                pruned_apps = conn.execute(
                    "DELETE FROM usage_app_hourly WHERE hour < ?",
                    (now_hour() - int(app_retention_days) * 24,),
                ).rowcount or 0
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    pruned_hourly = prune(retention_days=hourly_retention_days)
    return {"marked": marked, "jobs": pruned_jobs, "apps": pruned_apps,
            "hourly": pruned_hourly}


# --- read path (admin dashboards) -----------------------------------------

def _window_clause(start_hour: int | None, end_hour: int | None,
                   ) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start_hour is not None:
        clauses.append("hour >= ?")
        params.append(int(start_hour))
    if end_hour is not None:
        clauses.append("hour <= ?")
        params.append(int(end_hour))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def totals_by_key(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Summed usage per key over an optional [start_hour, end_hour] window,
    optionally restricted to one user. Rows survive key/user revocation."""
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    if user_id is not None:
        where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
        params.append(user_id)
    cur = conn.execute(
        "SELECT key_id, user_id,"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_hourly" + where +
        " GROUP BY key_id, user_id"
        " ORDER BY audio_s DESC",
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def totals_by_user(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Summed usage per user over an optional window, keyed by user_id."""
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    cur = conn.execute(
        "SELECT user_id,"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_hourly" + where +
        " GROUP BY user_id",
        params,
    )
    return {r["user_id"]: dict(r) for r in cur.fetchall()}


def totals_for_user(
    user_id: str,
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
) -> dict[str, Any]:
    """Summed usage for one user over an optional [start_hour, end_hour]
    window, all kinds folded. Returns a zeros dict when the user has no rows
    (uses idx_usage_user_hour). Backs the per-user self-usage banner on
    /quick-config: pass start_hour = hour_for_ts(<viewer local midnight>)
    for a per-viewer-local 'today'."""
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
    params.append(user_id)
    row = conn.execute(
        "SELECT SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_hourly" + where,
        params,
    ).fetchone()
    return {
        "requests": int((row["requests"] if row else 0) or 0),
        "errors": int((row["errors"] if row else 0) or 0),
        "words": int((row["words"] if row else 0) or 0),
        "audio_s": float((row["audio_s"] if row else 0.0) or 0.0),
    }


def series(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
    bucket: str = "day",
    user_id: str | None = None,
    key_id: str | None = None,
) -> list[dict[str, Any]]:
    """Time-series of summed usage, one entry per SERVER-LOCAL day (the operator
    dashboard's perspective), ascending, all kinds folded. Hours are rolled
    into local days in Python via epoch_day_for(hour*3600) — DST-correct.
    `day` is days-since-epoch so `day*86400` is UTC midnight of that date
    (the client's label math). bucket='week' groups into 7-day blocks
    (day - day % 7). user_id / key_id None => global (all keys)."""
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    if user_id is not None:
        where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
        params.append(user_id)
    if key_id is not None:
        where = (where + " AND key_id = ?") if where else " WHERE key_id = ?"
        params.append(key_id)
    cur = conn.execute(
        "SELECT hour, requests, errors, words, audio_s, processing_s, sessions"
        " FROM usage_hourly" + where,
        params,
    )
    agg: dict[int, dict[str, Any]] = {}
    for r in cur.fetchall():
        day = epoch_day_for(int(r["hour"]) * 3600)
        if bucket == "week":
            day = day - (day % 7)
        cell = agg.get(day)
        if cell is None:
            cell = agg[day] = {"day": day, "requests": 0, "errors": 0,
                               "words": 0, "audio_s": 0.0, "processing_s": 0.0,
                               "sessions": 0}
        cell["requests"] += int(r["requests"] or 0)
        cell["errors"] += int(r["errors"] or 0)
        cell["words"] += int(r["words"] or 0)
        cell["audio_s"] += float(r["audio_s"] or 0.0)
        cell["processing_s"] += float(r["processing_s"] or 0.0)
        cell["sessions"] += int(r["sessions"] or 0)
    return [agg[d] for d in sorted(agg)]


def leaderboard(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
    by: str = "user",
    metric: str = "audio_s",
    limit: int = 50,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Top entities for a window, ranked by `metric`. by='user' groups by
    user_id; by='key' groups by key_id (carrying user_id). `metric` is
    validated against the column whitelist (it is interpolated into the
    ORDER BY). `user_id` narrows to one owner's rows (the /stats "own"
    scope ranks that user's keys); None = every user."""
    if metric not in _METRICS:
        metric = "audio_s"
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    if user_id is not None:
        where += (" AND" if where else " WHERE") + " user_id = ?"
        params = params + [user_id]
    if by == "key":
        group, cols = "key_id", "key_id, user_id"
    else:
        group, cols = "user_id", "user_id"
    cur = conn.execute(
        f"SELECT {cols},"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s,"
        " SUM(processing_s) AS processing_s, SUM(sessions) AS sessions"
        " FROM usage_hourly" + where +
        f" GROUP BY {group}"
        f" ORDER BY {metric} DESC"
        " LIMIT ?",
        params + [max(1, int(limit))],
    )
    return [dict(r) for r in cur.fetchall()]


def is_empty() -> bool:
    conn = _require_conn()
    row = conn.execute("SELECT 1 FROM usage_hourly LIMIT 1").fetchone()
    return row is None


# Every hour-keyed statistics table shares USAGE_RETENTION_DAYS: they are the
# same grain, and pruning one without the others would leave stage meters
# denominated over jobs that no longer exist. usage_app_hourly has its own,
# shorter clock (USAGE_APP_RETENTION_DAYS, applied by sweep()).
_HOURLY_TABLES = ("usage_hourly", "usage_stage_hourly", "usage_target_hourly",
                  "usage_dictation_hourly")


def prune(*, retention_days: int) -> int:
    """Drop rollup rows older than the retention cutoff. retention_days <= 0
    is a no-op (the rollup is tiny — unbounded is the default)."""
    if retention_days <= 0:
        return 0
    cutoff = now_hour() - int(retention_days) * 24
    conn = _require_conn()
    removed = 0
    with _lock:
        for table in _HOURLY_TABLES:
            cur = conn.execute(f"DELETE FROM {table} WHERE hour < ?", (cutoff,))
            removed += cur.rowcount or 0
    return removed


# --- read path (the desktop app's /v1/usage document) ---------------------

def _zero_cell() -> dict[str, Any]:
    return {"sessions": 0, "requests": 0, "errors": 0, "words": 0,
            "audio_s": 0.0, "processing_s": 0.0}


def _zero_split() -> dict[str, dict[str, Any]]:
    return {"all": _zero_cell(), **{k: _zero_cell() for k in KINDS}}


def _add_row(split: dict[str, dict[str, Any]], r: sqlite3.Row) -> None:
    """Add one usage_hourly row (or per-kind SUM row) to a split: always to
    `all`, and to its kind when the kind is one the client knows."""
    for cell in (split["all"], split.get(r["kind"])):
        if cell is None:
            continue
        cell["sessions"] += int(r["sessions"] or 0)
        cell["requests"] += int(r["requests"] or 0)
        cell["errors"] += int(r["errors"] or 0)
        cell["words"] += int(r["words"] or 0)
        cell["audio_s"] += float(r["audio_s"] or 0.0)
        cell["processing_s"] += float(r["processing_s"] or 0.0)


def _rounded(split: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for cell in split.values():
        cell["audio_s"] = round(cell["audio_s"], 3)
        cell["processing_s"] = round(cell["processing_s"], 3)
    return split


# The optional stages a caller may narrow the document to (`with=`). The
# decode and a URL download are the job, not a stage of it.
WITH_STAGES: tuple[str, ...] = tuple(STAGE_APPLIES_TO)

# Widest window the document will reckon: ten years of days.
MAX_WINDOW_DAYS = 3650


@dataclasses.dataclass(frozen=True)
class WindowSpec:
    """The validated window parameters /v1/usage and /stats/usage share:
    one parser, one 422 vocabulary. `tz` is the resolved zone (None =
    server-local, echoed as tz_name "local")."""
    tz: zoneinfo.ZoneInfo | None
    tz_name: str
    days: int | None
    from_day: int | None
    to_day: int | None
    all_time: bool
    with_stages: tuple[str, ...]

    @property
    def source(self) -> str:
        return "jobs" if self.with_stages else "rollups"


def parse_window_params(*, days: int | None = None, from_day: int | None = None,
                        to_day: int | None = None, all_time: bool = False,
                        with_: str | None = None, tz: str | None = None
                        ) -> WindowSpec:
    """Validate the raw query parameters of a usage window. `days` clamps to
    1..MAX_WINDOW_DAYS; `with_` is a comma list of optional stages; `tz` an
    IANA name (unknown → server-local). Raises ValueError with a message fit
    for a 422 body when a stage is unknown or `from` is after `to`."""
    zone = resolve_tz(tz)
    tz_name = str(tz) if zone is not None else "local"
    eff_days = None
    if days is not None:
        eff_days = max(1, min(int(days), MAX_WINDOW_DAYS))
    stages: tuple[str, ...] = ()
    if with_:
        stages = tuple(dict.fromkeys(
            s.strip() for s in with_.split(",") if s.strip()))
        unknown = [s for s in stages if s not in WITH_STAGES]
        if unknown:
            raise ValueError(f"unknown stage: {unknown[0]!r} (one of "
                             f"{', '.join(WITH_STAGES)})")
    if from_day is not None and to_day is not None and from_day > to_day:
        raise ValueError("'from' is after 'to'")
    return WindowSpec(tz=zone, tz_name=tz_name, days=eff_days,
                      from_day=from_day, to_day=to_day, all_time=bool(all_time),
                      with_stages=stages)


def empty_document(
    *,
    from_day: int,
    to_day: int,
    tz: str,
    source: str = "rollups",
    jobs_retention_days: int = 365,
    first_day: int | None = None,
) -> dict[str, Any]:
    """The zeroed shape the route serves when the store is unavailable, so
    the client renders an empty page instead of erroring. `from_day`/`to_day`
    are days-since-epoch of the (inclusive) window in the caller's zone."""
    return {
        "tz": tz,
        "range": {
            "from": int(from_day),
            "to": int(to_day),
            "days": int(to_day) - int(from_day) + 1,
            "first_day": first_day,
            "source": source,
            "jobs_retention_days": int(jobs_retention_days),
        },
        # An extra split per kind for the hour grid (words stay flat on the
        # slot for the desktop app; see _finish).

        "today": _zero_split(),
        "total": _zero_split(),
        "series": [],
        "stages": [],
        "dictation": {
            "sessions": 0, "words": 0, "audio_s": 0.0, "wpm": 0.0,
            "activation": {a: 0 for a in ACTIVATIONS},
            "delivery": {d: 0 for d in DELIVERIES},
            "translation": {t: 0 for t in TRANSLATIONS},
            "targets": [],
        },
        "apps": [],
        "calendar": [],
        "hours": [],
        "dom_hours": [],
        "streak": {"all": {"current": 0, "best": 0},
                   **{k: {"current": 0, "best": 0} for k in KINDS}},
        "time_saved_s": 0.0,
    }


def _epoch_day(d: datetime.date) -> int:
    return (d - _EPOCH).days


def _from_epoch_day(n: int) -> datetime.date:
    return _EPOCH + datetime.timedelta(days=int(n))


def resolve_window(
    *,
    today: datetime.date,
    days: int | None,
    from_day: int | None,
    to_day: int | None,
    all_time: bool,
    first_day: int | None,
) -> tuple[int, int]:
    """The inclusive [from, to] window in days-since-epoch. `to` defaults to
    today, `from` to `to - days + 1` (days defaults to 30); `all_time` starts
    at the first day with usage (or today when there is none). The span is
    clamped to MAX_WINDOW_DAYS by moving `from` forward. Raises ValueError
    when from > to — the route turns that into a 422."""
    t = _epoch_day(today) if to_day is None else int(to_day)
    if all_time:
        f = int(first_day) if first_day is not None else t
        f = min(f, t)
    elif from_day is not None:
        f = int(from_day)
    else:
        n = 30 if days is None else max(1, min(int(days), MAX_WINDOW_DAYS))
        f = t - n + 1
    if f > t:
        raise ValueError("from is after to")
    if t - f + 1 > MAX_WINDOW_DAYS:
        f = t - MAX_WINDOW_DAYS + 1
    return f, t


def _words_split() -> dict[str, int]:
    return {"all": 0, **{k: 0 for k in KINDS}}


def _add_words(split: dict[str, int], kind: str, words: int) -> None:
    split["all"] += words
    if kind in split:
        split[kind] += words


Ids = str | Sequence[str] | None


def _in_clause(col: str, ids: Ids, clauses: list[str], params: list[Any]) -> None:
    """Append `col = ?` for one id or `col IN (?, …)` for several; None or
    an empty sequence adds nothing. An empty sequence is "no filter", not
    "nothing" — the /stats filter chips send nothing when cleared."""
    if ids is None:
        return
    if isinstance(ids, str):
        clauses.append(f"{col} = ?")
        params.append(ids)
        return
    vals = list(dict.fromkeys(ids))
    if not vals:
        return
    clauses.append(f"{col} IN ({', '.join('?' * len(vals))})")
    params.extend(vals)


def _scope_where(user_id: Ids, key_id: Ids = None, *,
                 col: str = "user_id", key_col: str = "key_id",
                 kinds: Sequence[str] | None = None, kind_col: str | None = None,
                 ) -> tuple[str, list[Any]]:
    """`(" WHERE …", params)` narrowing a table to an owner / owners and
    a key / keys (str = one, a sequence = any of them, None = no filter:
    the admin's whole-server document), plus `kinds` when the table has a
    kind column (`kind_col`; the stage / dictation / app rollups do not and
    stay unfiltered by kind). Always yields a WHERE so callers can append
    " AND …" tails unconditionally."""
    clauses: list[str] = ["1=1"]
    params: list[Any] = []
    _in_clause(col, user_id, clauses, params)
    _in_clause(key_col, key_id, clauses, params)
    if kind_col and kinds:
        _in_clause(kind_col, [_norm_kind(k) for k in kinds], clauses, params)
    return " WHERE " + " AND ".join(clauses), params


def distinct_ids(col: str) -> list[str]:
    """Every distinct user_id / key_id the hourly rollups have seen; the
    /stats route maps a non-admin viewer's opaque labels back to ids with
    it. `col` is validated (it is interpolated)."""
    if col not in ("user_id", "key_id"):
        raise ValueError(col)
    conn = _require_conn()
    return [r[0] for r in conn.execute(
        f"SELECT DISTINCT {col} FROM usage_hourly ORDER BY 1")]


def _first_day(conn: sqlite3.Connection, user_id: Ids,
               tz: zoneinfo.ZoneInfo | None,
               with_stages: tuple[str, ...],
               key_id: Ids = None,
               kinds: Sequence[str] | None = None) -> int | None:
    where, params = _scope_where(user_id, key_id, kinds=kinds, kind_col="kind")
    if with_stages:
        row = conn.execute(
            "SELECT MIN(created_ts) AS ts FROM usage_jobs" + where
            + _with_clause(with_stages), (*params, *with_stages),
        ).fetchone()
        if row is None or row["ts"] is None:
            return None
        return _epoch_day(_date_of(float(row["ts"]), tz))
    row = conn.execute(
        "SELECT MIN(hour) AS h FROM usage_hourly" + where, params,
    ).fetchone()
    if row is None or row["h"] is None:
        return None
    return _epoch_day(_date_of(int(row["h"]) * 3600, tz))


def _with_clause(with_stages: tuple[str, ...]) -> str:
    """SQL tail restricting usage_jobs rows to jobs that ran ALL the given
    stages (one EXISTS per stage; the stage names are bound parameters)."""
    return "".join(
        " AND EXISTS (SELECT 1 FROM usage_job_stages s WHERE s.job_id ="
        " usage_jobs.job_id AND s.stage = ?)" for _ in with_stages)


def document(
    user_id: str | None,
    *,
    tz: zoneinfo.ZoneInfo | None,
    tz_name: str,
    days: int | None = None,
    from_day: int | None = None,
    to_day: int | None = None,
    all_time: bool = False,
    with_stages: tuple[str, ...] = (),
    jobs_retention_days: int = 365,
    now: float | None = None,
    key_id: Ids = None,
    all_stages: bool = False,
    kinds: Sequence[str] | None = None,
) -> dict[str, Any]:
    """One user's statistics document (everything /v1/usage returns except
    the username), reckoned in `tz` (None = server-local; `tz_name` is only
    echoed). The window is resolve_window()'s; `today` is the caller's
    current day whatever the window; `streak` runs over the FULL retained
    history so a short window never caps it.

    `user_id=None` is the whole server (the admin's /stats view): every
    figure sums over all users. `key_id` narrows further to one API key —
    but only the tables that carry a key (usage_hourly, usage_jobs); the
    stage, dictation-outcome and app rollups have no key column and stay
    at user scope, which `range.key_scoped` discloses as False.

    Days on the wire are days-since-epoch of the caller-local calendar date
    (×86 400 → UTC midnight of that date, the client's label math). Series,
    calendar and hours are sparse: a day/slot without usage is absent.

    `with_stages` (non-empty) recomputes the whole document from the per-job
    rows restricted to jobs that ran every listed stage — `range.source`
    says "jobs" and the client discloses `jobs_retention_days`.

    `user_id` / `key_id` may also be sequences (any of those owners /
    keys: the /stats filter chips). `kinds` keeps only those job kinds in
    the tables that carry a kind (usage_hourly, usage_jobs); the stage,
    dictation-outcome and app rollups have no kind column and stay at the
    owner scope, which `range.kind_scoped` discloses as False."""
    conn = _require_conn()
    kinds = tuple(dict.fromkeys(_norm_kind(k) for k in (kinds or ())))
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"unknown kind: {k}")
    now = time.time() if now is None else float(now)
    today = _date_of(now, tz)
    with_stages = tuple(dict.fromkeys(with_stages))
    for st in with_stages:
        if st not in WITH_STAGES:
            raise ValueError(f"unknown stage: {st}")
    first_day = _first_day(conn, user_id, tz, with_stages, key_id, kinds)
    f, t = resolve_window(today=today, days=days, from_day=from_day,
                          to_day=to_day, all_time=all_time, first_day=first_day)
    doc = empty_document(
        from_day=f, to_day=t, tz=tz_name,
        source="jobs" if with_stages else "rollups",
        jobs_retention_days=jobs_retention_days, first_day=first_day)
    if key_id is not None:
        # Partial by construction: see the docstring.
        doc["range"]["key_scoped"] = False
    if kinds:
        doc["range"]["kind_scoped"] = False
    start_hour = _midnight_hour(_from_epoch_day(f), tz)
    end_hour = _midnight_hour(_from_epoch_day(t) + datetime.timedelta(days=1), tz)
    today_start = _midnight_hour(today, tz)
    today_end = _midnight_hour(today + datetime.timedelta(days=1), tz)
    if with_stages:
        _fill_from_jobs(conn, doc, user_id, tz, today, with_stages,
                        start_hour * 3600, end_hour * 3600,
                        today_start * 3600, today_end * 3600, key_id, kinds)
    else:
        _fill_from_rollups(conn, doc, user_id, tz, today,
                           start_hour, end_hour, today_start, today_end, key_id, kinds)
    if not all_stages:
        # the desktop app's order: the optional stages as it lists them
        keep = {st["stage"]: st for st in doc["stages"] if st["stage"] in STAGE_APPLIES_TO}
        doc["stages"] = [keep[k] for k in STAGE_APPLIES_TO if k in keep]
    dictation = doc["total"]["dictation"]
    doc["time_saved_s"] = round(max(
        0.0, dictation["words"] / TYPING_WPM * 60.0 - dictation["audio_s"]), 1)
    return doc


# ---------------------------------------------------------------------------
# overview(): the /stats/usage v2 gather — document() plus a breakdown
# ---------------------------------------------------------------------------

BREAKDOWNS: tuple[str, ...] = ("kind", "user", "key", "model", "stage")
BUCKETS: tuple[str, ...] = ("auto", "day", "week", "month")
COMPARES: tuple[str, ...] = ("off", "prev", "yoy")
# Past this many buckets the auto rule steps up a size regardless of span
# (ten years of days is 3 650 points × K lines — nothing a chart can show).
MAX_BUCKETS = 1500


def bucket_mode(days: int) -> str:
    """The frontend's bucketMode(): day up to 120 days, ISO week up to two
    years, month beyond."""
    if days <= 120:
        return "day"
    if days <= 730:
        return "week"
    return "month"


def _bucket_start(day: int, mode: str) -> int:
    """Days-since-epoch of the bucket containing `day`: the day itself, its
    Monday, or the first of its month."""
    if mode == "week":
        # Day 0 (1970-01-01) was a Thursday: (day + 3) % 7 is 0 on Mondays.
        return day - ((day + 3) % 7)
    if mode == "month":
        d = _from_epoch_day(day)
        return _epoch_day(d.replace(day=1))
    return day


def _axis(from_day: int, to_day: int, mode: str) -> list[int]:
    """Every bucket start from `from_day` to `to_day`, dense and ascending —
    the shared x-axis every line and the compare series align to."""
    out: list[int] = []
    d = _bucket_start(from_day, mode)
    while d <= to_day:
        out.append(d)
        if mode == "day":
            d += 1
        elif mode == "week":
            d += 7
        else:
            first = _from_epoch_day(d)
            nxt = (first.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
            d = _epoch_day(nxt)
    return out


def _stage_metric(metric: str, r: sqlite3.Row) -> float:
    """usage_stage_hourly has runs / audio_s / secs. Sessions and requests
    read as runs, processing_s as secs; words has no stage meaning (0) and errors
    per stage only arrive with the phase-2 ledger columns (0 until then)."""
    if metric in ("sessions", "requests"):
        return float(r["runs"] or 0)
    if metric == "audio_s":
        return float(r["audio_s"] or 0.0)
    if metric == "processing_s":
        return float(r["secs"] or 0.0)
    return 0.0


def _year_back(day: int) -> int:
    d = _from_epoch_day(day)
    try:
        return _epoch_day(d.replace(year=d.year - 1))
    except ValueError:          # Feb 29 → Feb 28
        return _epoch_day(d.replace(year=d.year - 1, day=28))


def overview(
    *,
    user_id: str | None,
    key_id: str | None = None,
    tz: zoneinfo.ZoneInfo | None,
    tz_name: str,
    days: int | None = None,
    from_day: int | None = None,
    to_day: int | None = None,
    all_time: bool = False,
    with_stages: tuple[str, ...] = (),
    by: str = "kind",
    metric: str = "audio_s",
    bucket: str = "auto",
    compare: str = "off",
    top_k: int = 8,
    limit: int = 50,
    jobs_retention_days: int = 365,
    now: float | None = None,
    kinds: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The /stats/usage v2 document: document() for the window (totals,
    today, stages, hours, range) plus a dense-axis breakdown of `metric` by
    `by` (kind | user | key | model | stage), a leaderboard over the same
    entities, an optional comparison window and a per-model table.

    `user_id=None` is the whole server; `key_id` narrows the key-bearing
    tables (see document()). `bucket` auto-resolves from the span. The
    breakdown source: kind/user/key/stage read the hourly rollups, model
    reads the per-job rows (retention-limited, `breakdown.source="jobs"`);
    `with_stages` narrows the document and the model breakdown only.
    `compare` = prev (the window shifted back by its own span) or yoy (the
    same calendar dates a year earlier, Feb 29 clamped) returns the other
    window's totals and lines, index-aligned to this axis."""
    if by not in BREAKDOWNS:
        by = "kind"
    if metric not in _METRICS:
        metric = "audio_s"
    if bucket not in BUCKETS:
        bucket = "auto"
    if compare not in COMPARES:
        compare = "off"
    kinds = tuple(dict.fromkeys(_norm_kind(k) for k in (kinds or ())))
    doc = document(user_id, tz=tz, tz_name=tz_name, days=days, from_day=from_day,
                   to_day=to_day, all_time=all_time, with_stages=with_stages,
                   jobs_retention_days=jobs_retention_days, now=now, key_id=key_id,
                   all_stages=True, kinds=kinds)
    f, t = int(doc["range"]["from"]), int(doc["range"]["to"])
    span = t - f + 1
    mode = bucket_mode(span) if bucket == "auto" else bucket
    while mode != "month" and len(_axis(f, t, mode)) > MAX_BUCKETS:
        mode = "week" if mode == "day" else "month"
    axis = _axis(f, t, mode)
    index = {d: i for i, d in enumerate(axis)}
    n = len(axis)

    def slot(day: int) -> int | None:
        return index.get(_bucket_start(day, mode))

    conn = _require_conn()
    start_hour = _midnight_hour(_from_epoch_day(f), tz)
    end_hour = _midnight_hour(_from_epoch_day(t) + datetime.timedelta(days=1), tz)

    # entity id → {"label", "user_id"?, "values": [...], "totals": cell}
    ents: dict[str, dict[str, Any]] = {}

    def ent(eid: str, **meta: Any) -> dict[str, Any]:
        e = ents.get(eid)
        if e is None:
            e = ents[eid] = {"id": eid, "values": [0.0] * n,
                             "totals": _zero_cell(), **meta}
        return e

    def add(e: dict[str, Any], day: int, cell: dict[str, Any]) -> None:
        i = slot(day)
        for k, v in cell.items():
            e["totals"][k] += v
        if i is not None:
            e["values"][i] += float(cell.get(metric, 0.0) or 0.0)

    source = "rollups"
    key_scoped = True
    if by == "kind":
        for p in doc["series"]:
            day = int(p["day"])
            rest = dict(p["all"])
            for k in KINDS:
                cell = p[k]
                add(ent(k, label=k), day, cell)
                for m in rest:
                    rest[m] -= cell[m]
            # Each cell was rounded to 3 decimals on its own, so `all` minus
            # the four kinds leaves up to a few thousandths of dust; only a
            # real remainder is an "unknown" kind (rows recorded before
            # kinds existed).
            rest = {m: (v if v > 0.05 else 0) for m, v in rest.items()}
            if any(v > 0 for v in rest.values()):
                add(ent(UNKNOWN_KIND, label=UNKNOWN_KIND), day, rest)
    elif by in ("user", "key"):
        where, params = _scope_where(user_id, key_id, kinds=kinds, kind_col="kind")
        for r in conn.execute(
            "SELECT hour, user_id, key_id, SUM(requests) AS requests,"
            " SUM(errors) AS errors, SUM(words) AS words, SUM(audio_s) AS audio_s,"
            " SUM(processing_s) AS processing_s, SUM(sessions) AS sessions FROM usage_hourly"
            + where + " AND hour >= ? AND hour < ? GROUP BY hour, user_id, key_id",
            (*params, start_hour, end_hour),
        ):
            day = _epoch_day(_date_of(int(r["hour"]) * 3600, tz))
            cell = {"sessions": int(r["sessions"] or 0),
                    "requests": int(r["requests"] or 0),
                    "errors": int(r["errors"] or 0),
                    "words": int(r["words"] or 0),
                    "audio_s": float(r["audio_s"] or 0.0),
                    "processing_s": float(r["processing_s"] or 0.0)}
            if by == "user":
                add(ent(r["user_id"], label=r["user_id"], user_id=r["user_id"]),
                    day, cell)
            else:
                add(ent(r["key_id"], label=r["key_id"], user_id=r["user_id"],
                        key_id=r["key_id"]), day, cell)
    elif by == "model":
        source = "jobs"
        where, params = _scope_where(user_id, key_id, kinds=kinds, kind_col="kind")
        start_ts = start_hour * 3600
        end_ts = end_hour * 3600
        for job in conn.execute(
            "SELECT model, created_ts, status, audio_s, words, processing_s, utterances"
            " FROM usage_jobs" + where + _with_clause(with_stages)
            + " AND created_ts >= ? AND created_ts < ?",
            (*params, *with_stages, float(start_ts), float(end_ts)),
        ):
            name = job["model"] or "(unknown)"
            add(ent(name, label=name),
                _epoch_day(_date_of(float(job["created_ts"]), tz)), _job_cell(job))
    else:  # stage
        key_scoped = False
        where, params = _scope_where(user_id)
        for r in conn.execute(
            "SELECT hour, stage, SUM(runs) AS runs, SUM(audio_s) AS audio_s,"
            " SUM(secs) AS secs FROM usage_stage_hourly" + where
            + " AND hour >= ? AND hour < ? GROUP BY hour, stage",
            (*params, start_hour, end_hour),
        ):
            day = _epoch_day(_date_of(int(r["hour"]) * 3600, tz))
            runs = int(r["runs"] or 0)
            cell = {"sessions": runs, "requests": runs, "errors": 0, "words": 0,
                    "audio_s": float(r["audio_s"] or 0.0),
                    "processing_s": float(r["secs"] or 0.0)}
            add(ent(r["stage"], label=r["stage"]), day, cell)

    ranked = sorted(ents.values(),
                    key=lambda e: (-float(e["totals"].get(metric, 0) or 0), e["id"]))
    for e in ranked:
        tot = e["totals"]
        tot["audio_s"] = round(tot["audio_s"], 3)
        tot["processing_s"] = round(tot["processing_s"], 3)
        e["rtf"] = round(tot["processing_s"] / tot["audio_s"], 3) if tot["audio_s"] > 0 else None
        e["values"] = [round(v, 3) for v in e["values"]]

    lines: list[dict[str, Any]] = []
    for e in ranked[:max(1, int(top_k))]:
        line = {k: v for k, v in e.items() if k not in ("totals", "rtf")}
        lines.append(line)
    tail = ranked[max(1, int(top_k)):]
    if tail:
        others = [0.0] * n
        for e in tail:
            for i, v in enumerate(e["values"]):
                others[i] += v
        if any(v > 0 for v in others):
            lines.append({"id": "__others__", "label": f"others ({len(tail)})",
                          "values": [round(v, 3) for v in others], "others": True})

    board = [
        {k: v for k, v in e.items() if k != "values"}
        for e in ranked[:max(1, int(limit))]
    ]

    models = _models_in_window(conn, user_id, key_id, with_stages,
                               start_hour * 3600, end_hour * 3600, kinds)

    out: dict[str, Any] = {
        "v": 2,
        "days": axis,
        "bucket": mode,
        "metric": metric,
        "by": by,
        "tz": tz_name,
        "range": doc["range"],
        "filter": {"user_id": user_id, "key_id": key_id, "kinds": list(kinds),
                   "key_scoped": key_scoped if key_id is not None else True,
                   "kind_scoped": (by == "kind" or by in ("user", "key", "model"))
                                  if kinds else True},
        "totals": doc["total"],
        "today": doc["today"],
        "stages": doc["stages"],
        "hours": doc["hours"],
        "dom_hours": doc["dom_hours"],
        "series": doc["series"],
        "lines": lines,
        "leaderboard": board,
        "breakdown": {"source": source, "key_scoped": key_scoped},
        "models": models,
        "compare": None,
        "time_saved_s": doc["time_saved_s"],
    }
    if compare != "off":
        if compare == "prev":
            pf, pt = f - span, f - 1
        else:
            pf, pt = _year_back(f), _year_back(t)
        prev = overview(user_id=user_id, key_id=key_id, tz=tz, tz_name=tz_name,
                        from_day=pf, to_day=pt, with_stages=with_stages, by=by,
                        metric=metric, bucket=mode, compare="off", top_k=top_k,
                        limit=limit, jobs_retention_days=jobs_retention_days,
                        now=now, kinds=kinds)
        by_id = {ln["id"]: ln["values"] for ln in prev["lines"]}
        cmp_lines = []
        for ln in lines:
            vals = list(by_id.get(ln["id"], []))[:n]
            vals += [0.0] * (n - len(vals))
            cmp_lines.append({"id": ln["id"], "values": vals})
        out["compare"] = {"mode": compare,
                          "range": {"from": pf, "to": pt, "days": pt - pf + 1},
                          "totals": prev["totals"], "lines": cmp_lines,
                          "hours": prev["hours"], "dom_hours": prev["dom_hours"],
                          "series": prev["series"]}
    return out


def _models_in_window(conn: sqlite3.Connection, user_id: Ids,
                      key_id: Ids, with_stages: tuple[str, ...],
                      start_ts: float, end_ts: float,
                      kinds: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Per-model totals over the window from the per-job rows (the decode
    model of each job; stage models live in usage_job_stages and are not
    folded in). Feeds the loaded-models table's audio/RTF columns."""
    where, params = _scope_where(user_id, key_id, kinds=kinds, kind_col="kind")
    out = []
    for r in conn.execute(
        "SELECT COALESCE(model, '(unknown)') AS model, COUNT(*) AS sessions,"
        " SUM(MAX(1, COALESCE(utterances, 0))) AS requests,"
        " SUM(status <> 'ok') AS errors, SUM(words) AS words,"
        " SUM(audio_s) AS audio_s, SUM(processing_s) AS processing_s"
        " FROM usage_jobs" + where + _with_clause(with_stages)
        + " AND created_ts >= ? AND created_ts < ? GROUP BY model"
        " ORDER BY audio_s DESC, model",
        (*params, *with_stages, float(start_ts), float(end_ts)),
    ):
        audio = float(r["audio_s"] or 0.0)
        proc = float(r["processing_s"] or 0.0)
        out.append({
            "model": r["model"], "sessions": int(r["sessions"] or 0),
            "requests": int(r["requests"] or 0), "errors": int(r["errors"] or 0),
            "words": int(r["words"] or 0), "audio_s": round(audio, 3),
            "processing_s": round(proc, 3),
            "rtf": round(proc / audio, 3) if audio > 0 else None,
        })
    return out


# ---------------------------------------------------------------------------
# tail(): the /stats/tail document — queue wait, turnaround, failures,
# per-model, compare — from the per-job rows within retention
# ---------------------------------------------------------------------------

# Fixed log-spaced edges for the turnaround histogram (seconds): the last
# bucket is open-ended. Fixed so two windows (and the compare) share bins.
TURNAROUND_EDGES_S: tuple[int, ...] = (0, 1, 2, 5, 10, 30, 60, 120, 300, 900)


def _jobs_where(user_id: Ids, key_id: Ids, kind: Ids,
                start_ts: float, end_ts: float, alias: str = ""
                ) -> tuple[str, list[Any]]:
    """`(" WHERE …", params)` over usage_jobs for a window, optional
    owner(s) / key(s) / kind(s). `alias` prefixes the columns for joined
    queries."""
    p = alias + "." if alias else ""
    kinds = [kind] if isinstance(kind, str) else list(kind or ())
    where, params = _scope_where(user_id, key_id, col=p + "user_id",
                                 key_col=p + "key_id", kinds=kinds,
                                 kind_col=p + "kind")
    where += f" AND {p}created_ts >= ? AND {p}created_ts < ?"
    params += [float(start_ts), float(end_ts)]
    return where, params


def _nearest_rank(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def truncated_to_days(start_ts: float, retention_days: int,
                      now: float | None = None) -> int | None:
    """The per-job rows keep USAGE_JOBS_RETENTION_DAYS: a window that starts
    earlier answers with fewer jobs than the rollups know about. Returns the
    retention when that is the case, else None (the caller discloses it)."""
    days = int(retention_days or 0)
    if days <= 0:
        return None
    now = time.time() if now is None else float(now)
    return days if float(start_ts) < now - days * 86400 else None


def wait_quantiles(*, start_ts: float, end_ts: float, user_id: str | None = None,
                   key_id: str | None = None, kind: str | None = None,
                   model: str | None = None) -> dict[str, Any]:
    """`{n, p50, p95, max}` of wait_s over the window's jobs. Nearest-rank
    via ORDER BY … LIMIT 1 OFFSET k, so a year of jobs costs two indexed
    reads, not a Python sort."""
    conn = _require_conn()
    where, params = _jobs_where(user_id, key_id, kind, start_ts, end_ts)
    if model is not None:
        where += " AND model = ?"
        params.append(model)
    row = conn.execute(
        "SELECT COUNT(*) AS n, MAX(wait_s) AS mx FROM usage_jobs" + where, params,
    ).fetchone()
    n = int(row["n"] or 0)
    if n == 0:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def q(frac: float) -> float:
        k = max(0, min(n - 1, int(round(frac * (n - 1)))))
        r = conn.execute(
            "SELECT wait_s FROM usage_jobs" + where
            + " ORDER BY wait_s LIMIT 1 OFFSET ?", params + [k]).fetchone()
        return round(float(r["wait_s"] or 0.0), 3) if r else 0.0
    return {"n": n, "p50": q(0.5), "p95": q(0.95),
            "max": round(float(row["mx"] or 0.0), 3)}


def wait_series_by_day(*, start_ts: float, end_ts: float,
                       tz: zoneinfo.ZoneInfo | None, user_id: str | None = None,
                       key_id: str | None = None, kind: str | None = None
                       ) -> list[dict[str, Any]]:
    """`[{day, n, p50, p95}]` of wait_s per caller-local day, ascending."""
    conn = _require_conn()
    where, params = _jobs_where(user_id, key_id, kind, start_ts, end_ts)
    by_day: dict[int, list[float]] = {}
    for r in conn.execute("SELECT created_ts, wait_s FROM usage_jobs" + where, params):
        d = _epoch_day(_date_of(float(r["created_ts"]), tz))
        by_day.setdefault(d, []).append(float(r["wait_s"] or 0.0))
    out = []
    for d in sorted(by_day):
        vals = sorted(by_day[d])
        out.append({"day": d, "n": len(vals), "p50": round(_nearest_rank(vals, 0.5), 3),
                    "p95": round(_nearest_rank(vals, 0.95), 3)})
    return out


def turnaround_histogram(*, start_ts: float, end_ts: float,
                         user_id: str | None = None, key_id: str | None = None,
                         kind: str | None = None,
                         edges: tuple[int, ...] = TURNAROUND_EDGES_S) -> dict[str, Any]:
    """Jobs bucketed by end-to-end time (processing_s + wait_s) into fixed edges;
    `wait_share` per bucket is the fraction of that bucket's turnaround that
    was queue wait (what the chart hatches); `by_kind` splits each bucket's
    count per job kind (the chart stacks them). Also p50/p95 of turnaround."""
    conn = _require_conn()
    where, params = _jobs_where(user_id, key_id, kind, start_ts, end_ts)
    counts = [0] * len(edges)
    total = [0.0] * len(edges)
    waited = [0.0] * len(edges)
    by_kind: dict[str, list[int]] = {}
    turns: list[float] = []
    for r in conn.execute("SELECT processing_s, wait_s, kind FROM usage_jobs" + where, params):
        w = float(r["wait_s"] or 0.0)
        t = float(r["processing_s"] or 0.0) + w
        turns.append(t)
        i = 0
        for j, e in enumerate(edges):
            if t >= e:
                i = j
        counts[i] += 1
        total[i] += t
        waited[i] += w
        by_kind.setdefault(r["kind"] or "unknown", [0] * len(edges))[i] += 1
    turns.sort()
    return {
        "edges_s": list(edges),
        "counts": counts,
        "wait_share": [round(waited[i] / total[i], 3) if total[i] > 0 else 0.0
                       for i in range(len(edges))],
        "by_kind": by_kind,
        "n": len(turns),
        "p50": round(_nearest_rank(turns, 0.5), 3),
        "p95": round(_nearest_rank(turns, 0.95), 3),
        "max": round(turns[-1], 3) if turns else 0.0,
    }


def failures(*, start_ts: float, end_ts: float, user_id: str | None = None,
             key_id: str | None = None, kind: str | None = None) -> dict[str, Any]:
    """Failures grouped by stage and class: the terminal ones from the job
    rows (status ≠ ok; class/stage as classified, `other` / `(job)` when a
    pre-v2 row has none) unioned with the soft-failed stage rows (the job
    went on without the stage). `jobs` and `failed` count job rows."""
    conn = _require_conn()
    where, params = _jobs_where(user_id, key_id, kind, start_ts, end_ts)
    by_stage: dict[str, dict[str, int]] = {}
    by_class: dict[str, int] = {}

    def bump(stage: str, cls: str, n: int) -> None:
        by_stage.setdefault(stage, {})
        by_stage[stage][cls] = by_stage[stage].get(cls, 0) + n
        by_class[cls] = by_class.get(cls, 0) + n

    row = conn.execute(
        "SELECT COUNT(*) AS n, SUM(status <> 'ok') AS failed FROM usage_jobs" + where,
        params).fetchone()
    for r in conn.execute(
        "SELECT COALESCE(error_stage, '(job)') AS stage,"
        " COALESCE(error_class, 'other') AS cls, COUNT(*) AS n FROM usage_jobs"
        + where + " AND status <> 'ok' GROUP BY 1, 2", params,
    ):
        bump(r["stage"], r["cls"], int(r["n"] or 0))
    jw, jp = _jobs_where(user_id, key_id, kind, start_ts, end_ts, alias="j")
    for r in conn.execute(
        "SELECT s.stage AS stage, s.error AS cls, COUNT(*) AS n"
        " FROM usage_job_stages s JOIN usage_jobs j ON j.job_id = s.job_id"
        + jw + " AND s.error IS NOT NULL GROUP BY 1, 2", jp,
    ):
        bump(r["stage"], r["cls"], int(r["n"] or 0))
    return {
        "jobs": int(row["n"] or 0),
        "failed": int(row["failed"] or 0),
        "by_stage": by_stage,
        "by_class": dict(sorted(by_class.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def by_model(*, start_ts: float, end_ts: float, user_id: str | None = None,
             key_id: str | None = None, kind: str | None = None,
             top: int = 8) -> list[dict[str, Any]]:
    """Per decode model over the window: runs, audio, GPU seconds, RTF,
    errors, and the wait p50 for the busiest `top` models."""
    conn = _require_conn()
    where, params = _jobs_where(user_id, key_id, kind, start_ts, end_ts)
    out = []
    for r in conn.execute(
        "SELECT COALESCE(model, '(unknown)') AS model, COUNT(*) AS runs,"
        " SUM(status <> 'ok') AS errors, SUM(audio_s) AS audio_s,"
        " SUM(processing_s) AS processing_s FROM usage_jobs" + where
        + " GROUP BY model ORDER BY audio_s DESC, model", params,
    ):
        audio = float(r["audio_s"] or 0.0)
        proc = float(r["processing_s"] or 0.0)
        out.append({"model": r["model"], "runs": int(r["runs"] or 0),
                    "errors": int(r["errors"] or 0), "audio_s": round(audio, 3),
                    "processing_s": round(proc, 3),
                    "rtf": round(proc / audio, 3) if audio > 0 else None,
                    "wait_p50": None})
    for m in out[:max(0, int(top))]:
        mid = None if m["model"] == "(unknown)" else m["model"]
        if mid is None:
            continue
        m["wait_p50"] = wait_quantiles(start_ts=start_ts, end_ts=end_ts,
                                       user_id=user_id, key_id=key_id, kind=kind,
                                       model=mid)["p50"]
    return out


def tail(
    *,
    user_id: str | None,
    key_id: str | None = None,
    kind: str | None = None,
    tz: zoneinfo.ZoneInfo | None,
    tz_name: str,
    days: int | None = None,
    from_day: int | None = None,
    to_day: int | None = None,
    all_time: bool = False,
    jobs_retention_days: int = 365,
    now: float | None = None,
) -> dict[str, Any]:
    """The /stats/tail document over one window plus the same figures for
    the immediately preceding window of equal length: queue wait
    (quantiles + by day), turnaround histogram, failures by stage / class,
    per-model table, and compare deltas."""
    now = time.time() if now is None else float(now)
    today = _date_of(now, tz)
    first_day = (_first_day(_require_conn(), user_id, tz, (), key_id,
                            [kind] if isinstance(kind, str) else kind)
                 if all_time else None)
    f, t = resolve_window(today=today, days=days, from_day=from_day,
                          to_day=to_day, all_time=all_time, first_day=first_day)
    span = t - f + 1

    def bounds(fd: int, td: int) -> tuple[float, float]:
        return (_midnight_hour(_from_epoch_day(fd), tz) * 3600.0,
                _midnight_hour(_from_epoch_day(td) + datetime.timedelta(days=1), tz) * 3600.0)

    start_ts, end_ts = bounds(f, t)
    pstart, pend = bounds(f - span, f - 1)
    scope = dict(user_id=user_id, key_id=key_id, kind=kind)
    wait = wait_quantiles(start_ts=start_ts, end_ts=end_ts, **scope)
    wait["by_day"] = wait_series_by_day(start_ts=start_ts, end_ts=end_ts, tz=tz, **scope)
    turn = turnaround_histogram(start_ts=start_ts, end_ts=end_ts, **scope)
    fails = failures(start_ts=start_ts, end_ts=end_ts, **scope)
    models = by_model(start_ts=start_ts, end_ts=end_ts, **scope)
    pwait = wait_quantiles(start_ts=pstart, end_ts=pend, **scope)
    pturn = turnaround_histogram(start_ts=pstart, end_ts=pend, **scope)
    pfails = failures(start_ts=pstart, end_ts=pend, **scope)
    conn = _require_conn()

    def audio(s: float, e: float) -> float:
        w, p = _jobs_where(user_id, key_id, kind, s, e)
        r = conn.execute("SELECT SUM(audio_s) AS a FROM usage_jobs" + w, p).fetchone()
        return round(float(r["a"] or 0.0), 3)

    def cmp(cur: float, prev: float) -> dict[str, Any]:
        return {"cur": cur, "prev": prev, "delta": round(cur - prev, 3)}

    return {
        "from": f, "to": t, "days": span, "tz": tz_name,
        "range": {"from": f, "to": t, "days": span,
                  "truncated_to_days": truncated_to_days(start_ts, jobs_retention_days, now)},
        "wait": wait,
        "turnaround": turn,
        "failures": fails,
        "models": models,
        "compare": {
            "from": f - span, "to": f - 1,
            "wait_p50": cmp(wait["p50"], pwait["p50"]),
            "wait_p95": cmp(wait["p95"], pwait["p95"]),
            "turnaround_p50": cmp(turn["p50"], pturn["p50"]),
            "turnaround_p95": cmp(turn["p95"], pturn["p95"]),
            "runs": cmp(fails["jobs"], pfails["jobs"]),
            "errors": cmp(fails["failed"], pfails["failed"]),
            "audio_s": cmp(audio(start_ts, end_ts), audio(pstart, pend)),
        },
    }


def _slot_of(ts: float, tz: zoneinfo.ZoneInfo | None) -> tuple[int, int, int]:
    """(weekday 0=Mon, day of month 1..31, hour 0..23) of `ts` in `tz`.
    _finish folds these into the weekday × hour `hours` grid and the
    day-of-month × hour `dom_hours` grid."""
    dt = datetime.datetime.fromtimestamp(ts, tz)
    return dt.weekday(), dt.day, dt.hour


def _fold_slots(words_by_slot, extra_by_slot, key):
    """Sum the (dow, dom, hour) slots into coarser ones chosen by `key`
    (a function of the slot tuple), returning (words, extras) maps."""
    words_out: dict[Any, dict[str, int]] = {}
    extra_out: dict[Any, dict[str, dict[str, Any]]] = {}
    for slot, words in words_by_slot.items():
        dst = words_out.setdefault(key(slot), _words_split())
        for k, v in words.items():
            dst[k] += v
    for slot, extra in extra_by_slot.items():
        dst = extra_out.setdefault(key(slot), _slot_extra())
        for name, split in extra.items():
            for k, v in split.items():
                dst[name][k] += v
    return words_out, extra_out


# The per-slot measures beside words: every measure the stats console can
# put on its busy-hours grid, each split by kind like the words are.
SLOT_MEASURES = ("processing_s", "audio_s", "sessions", "requests", "errors")


def _slot_extra() -> dict[str, dict[str, Any]]:
    return {"processing_s": {"all": 0.0, **{k: 0.0 for k in KINDS}},
            "audio_s": {"all": 0.0, **{k: 0.0 for k in KINDS}},
            "sessions": {"all": 0, **{k: 0 for k in KINDS}},
            "requests": {"all": 0, **{k: 0 for k in KINDS}},
            "errors": {"all": 0, **{k: 0 for k in KINDS}}}


def _add_slot_extra(extra: dict[str, dict[str, Any]], kind: str,
                    cell: dict[str, Any]) -> None:
    """Add one cell (processing_s / audio_s / sessions / requests / errors) to
    the slot's splits: always to `all`, to its kind when known."""
    for name in SLOT_MEASURES:
        split, v = extra[name], cell.get(name) or 0
        split["all"] += v
        if kind in split:
            split[kind] += v


def _finish(doc: dict[str, Any], *, by_day, words_by_day, words_by_slot,
            words_all_days, today: datetime.date, extra_by_slot=None) -> None:
    """Common tail: the sparse arrays and the per-kind streaks. A slot in
    `hours` carries its words flat (the desktop app reads `h[kind]`) plus
    nested `processing_s`, `audio_s`, `sessions`, `requests` and `errors` splits
    (SLOT_MEASURES), so the backend's busy-hours grid can show whichever
    measure the console has picked. A slot is emitted when it saw any
    words, any request or any processing time."""
    extra_by_slot = extra_by_slot or {}
    doc["series"] = [{"day": _epoch_day(d), **_rounded(by_day[d])}
                     for d in sorted(by_day)]
    doc["calendar"] = [{"day": _epoch_day(d), **words_by_day[d]}
                       for d in sorted(words_by_day) if words_by_day[d]["all"] > 0]
    def grid(first: str, key) -> list[dict[str, Any]]:
        w_map, e_map = _fold_slots(words_by_slot, extra_by_slot, key)
        out = []
        for slot in sorted(set(w_map) | set(e_map)):
            words = w_map.get(slot) or _words_split()
            extra = e_map.get(slot) or _slot_extra()
            if (words["all"] <= 0 and extra["processing_s"]["all"] <= 0
                    and extra["requests"]["all"] <= 0):
                continue
            out.append({
                first: slot[0], "hour": slot[1], **words,
                "processing_s": {k: round(v, 3) for k, v in extra["processing_s"].items()},
                "audio_s": {k: round(v, 3) for k, v in extra["audio_s"].items()},
                "sessions": dict(extra["sessions"]),
                "requests": dict(extra["requests"]),
                "errors": dict(extra["errors"]),
            })
        return out
    doc["hours"] = grid("dow", lambda s: (s[0], s[2]))
    # The same slots by day of month (1..31) × hour: the stats console's
    # "busy days" rhythm. Not read by the desktop app.
    doc["dom_hours"] = grid("dom", lambda s: (s[1], s[2]))
    doc["streak"] = {
        k: _streak({d: w[k] for d, w in words_all_days.items()}, today)
        for k in ("all", *KINDS)}


def _fill_from_rollups(conn, doc, user_id, tz, today, start_hour, end_hour,
                       today_start, today_end, key_id=None, kinds=()) -> None:
    today_split = _zero_split()
    window = _zero_split()
    by_day: dict[datetime.date, dict[str, dict[str, Any]]] = {}
    words_by_day: dict[datetime.date, dict[str, int]] = {}
    words_by_slot: dict[tuple[int, int], dict[str, int]] = {}
    extra_by_slot: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    words_all_days: dict[datetime.date, dict[str, int]] = {}
    where, params = _scope_where(user_id, key_id, kinds=kinds, kind_col="kind")
    for r in conn.execute(
        "SELECT hour, kind, sessions, requests, errors, words, audio_s, processing_s"
        " FROM usage_hourly" + where, params,
    ):
        h = int(r["hour"])
        ts = h * 3600
        day = _date_of(ts, tz)
        words = int(r["words"] or 0)
        kind = r["kind"]
        if words > 0:
            _add_words(words_all_days.setdefault(day, _words_split()), kind, words)
        if today_start <= h < today_end:
            _add_row(today_split, r)
        if start_hour <= h < end_hour:
            _add_row(window, r)
            split = by_day.get(day)
            if split is None:
                split = by_day[day] = _zero_split()
            _add_row(split, r)
            _add_words(words_by_day.setdefault(day, _words_split()), kind, words)
            slot = _slot_of(ts, tz)
            _add_words(words_by_slot.setdefault(slot, _words_split()), kind, words)
            _add_slot_extra(extra_by_slot.setdefault(slot, _slot_extra()), kind,
                            {"processing_s": float(r["processing_s"] or 0.0),
                             "audio_s": float(r["audio_s"] or 0.0),
                             "sessions": int(r["sessions"] or 0),
                             "requests": int(r["requests"] or 0),
                             "errors": int(r["errors"] or 0)})
    doc["total"] = _rounded(window)
    doc["today"] = _rounded(today_split)
    doc["stages"] = _stages(conn, user_id, start_hour, end_hour, window)
    doc["dictation"] = _dictation(conn, user_id, start_hour, end_hour,
                                  window["dictation"])
    doc["dictation"]["targets"] = _dictation_targets(
        conn, user_id, start_hour * 3600, end_hour * 3600, key_id=key_id)
    app_where, app_params = _scope_where(user_id)
    doc["apps"] = [
        {"app_id": r["app_id"], "sessions": int(r["sessions"] or 0),
         "words": int(r["words"] or 0)}
        for r in conn.execute(
            "SELECT app_id, SUM(sessions) AS sessions, SUM(words) AS words"
            " FROM usage_app_hourly" + app_where + " AND hour >= ? AND hour < ?"
            " GROUP BY app_id ORDER BY sessions DESC, words DESC, app_id"
            " LIMIT 8", (*app_params, start_hour, end_hour),
        )
    ]
    _finish(doc, by_day=by_day, words_by_day=words_by_day,
            words_by_slot=words_by_slot, words_all_days=words_all_days,
            today=today, extra_by_slot=extra_by_slot)


def _job_cell(job: sqlite3.Row) -> dict[str, Any]:
    return {
        "sessions": 1,
        "requests": max(1, int(job["utterances"] or 0)),
        "errors": 0 if job["status"] == "ok" else 1,
        "words": int(job["words"] or 0),
        "audio_s": float(job["audio_s"] or 0.0),
        "processing_s": float(job["processing_s"] or 0.0),
    }


def _add_cell(split: dict[str, dict[str, Any]], kind: str,
              cell: dict[str, Any]) -> None:
    for target in (split["all"], split.get(kind)):
        if target is None:
            continue
        for k, v in cell.items():
            target[k] += v


def _fill_from_jobs(conn, doc, user_id, tz, today, with_stages,
                    start_ts, end_ts, today_start_ts, today_end_ts,
                    key_id=None, kinds=()) -> None:
    """The `with=` document: every figure from the per-job rows that ran all
    of `with_stages`. Sessions are jobs, requests are utterances, an error
    is a job whose last status was not ok. The dictation buckets come from
    the reported outcome columns (an unreported job stays unbucketed, as in
    the rollup); kept_original for translation is the count of jobs whose
    reported outcome kept the original."""
    today_split = _zero_split()
    window = _zero_split()
    by_day: dict[datetime.date, dict[str, dict[str, Any]]] = {}
    words_by_day: dict[datetime.date, dict[str, int]] = {}
    words_by_slot: dict[tuple[int, int], dict[str, int]] = {}
    extra_by_slot: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    words_all_days: dict[datetime.date, dict[str, int]] = {}
    activation = {a: 0 for a in ACTIVATIONS}
    delivery = {d: 0 for d in DELIVERIES}
    translation = {t: 0 for t in TRANSLATIONS}
    apps: dict[str, dict[str, int]] = {}
    window_jobs: list[str] = []
    where, params = _scope_where(user_id, key_id, kinds=kinds, kind_col="kind")
    for job in conn.execute(
        "SELECT job_id, kind, created_ts, status, audio_s, words, processing_s,"
        " utterances, activation, delivery, translation, app_id"
        " FROM usage_jobs" + where + _with_clause(with_stages)
        + " ORDER BY created_ts", (*params, *with_stages),
    ):
        ts = float(job["created_ts"])
        day = _date_of(ts, tz)
        kind = job["kind"]
        cell = _job_cell(job)
        words = cell["words"]
        if words > 0:
            _add_words(words_all_days.setdefault(day, _words_split()), kind, words)
        if today_start_ts <= ts < today_end_ts:
            _add_cell(today_split, kind, cell)
        if not (start_ts <= ts < end_ts):
            continue
        window_jobs.append(job["job_id"])
        _add_cell(window, kind, cell)
        _add_cell(by_day.setdefault(day, _zero_split()), kind, cell)
        _add_words(words_by_day.setdefault(day, _words_split()), kind, words)
        slot = _slot_of(ts, tz)
        _add_words(words_by_slot.setdefault(slot, _words_split()), kind, words)
        _add_slot_extra(extra_by_slot.setdefault(slot, _slot_extra()), kind, cell)
        if job["activation"] in activation:
            activation[job["activation"]] += 1
        if job["delivery"] in delivery:
            delivery[job["delivery"]] += 1
        if job["translation"] in translation:
            translation[job["translation"]] += 1
        if job["app_id"]:
            a = apps.setdefault(job["app_id"], {"sessions": 0, "words": 0})
            a["sessions"] += 1
            a["words"] += words
    doc["total"] = _rounded(window)
    doc["today"] = _rounded(today_split)
    cell = window["dictation"]
    audio_s = float(cell["audio_s"])
    doc["dictation"] = {
        "sessions": int(cell["sessions"]),
        "words": int(cell["words"]),
        "audio_s": round(audio_s, 3),
        "wpm": round(cell["words"] / (audio_s / 60.0), 1) if audio_s > 0 else 0.0,
        "activation": activation,
        "delivery": delivery,
        "translation": translation,
        "targets": _dictation_targets(conn, user_id, start_ts, end_ts, with_stages,
                                      key_id=key_id),
    }
    doc["apps"] = [
        {"app_id": a, **v} for a, v in sorted(
            apps.items(), key=lambda kv: (-kv[1]["sessions"], -kv[1]["words"], kv[0]))
    ][:8]
    doc["stages"] = _stages_from_jobs(conn, window_jobs, window, translation)
    _finish(doc, by_day=by_day, words_by_day=words_by_day,
            words_by_slot=words_by_slot, words_all_days=words_all_days,
            today=today, extra_by_slot=extra_by_slot)


def _stages_from_jobs(conn: sqlite3.Connection, job_ids: list[str],
                      window: dict[str, dict[str, Any]],
                      translation: dict[str, int]) -> list[dict[str, Any]]:
    """Stage rows over the narrowed jobs. A stage the caller filtered on
    shows runs == of_runs; the others show how often they co-occurred."""
    agg: dict[str, dict[str, Any]] = {}
    targets: dict[str, dict[str, int]] = {}
    # Chunk the IN list: SQLite's default variable limit is generous but
    # a year of jobs can exceed it.
    for i in range(0, len(job_ids), 500):
        chunk = job_ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            "SELECT s.stage, s.secs, s.targets, s.speakers, s.retained,"
            " j.audio_s FROM usage_job_stages s JOIN usage_jobs j"
            " ON j.job_id = s.job_id WHERE s.job_id IN (" + marks + ")", chunk,
        ):
            stage = r["stage"]
            if stage not in STAGE_ELIGIBLE:
                continue
            a = agg.setdefault(stage, {"runs": 0, "audio_s": 0.0, "secs": 0.0,
                                       "speakers": 0, "retained_sum": 0.0})
            a["runs"] += 1
            a["audio_s"] += float(r["audio_s"] or 0.0)
            a["secs"] += float(r["secs"] or 0.0)
            a["speakers"] += int(r["speakers"] or 0)
            a["retained_sum"] += float(r["retained"] or 0.0)
            for code in (r["targets"] or "").split(","):
                if code:
                    tg = targets.setdefault(stage, {})
                    tg[code] = tg.get(code, 0) + 1
    out: list[dict[str, Any]] = []
    for stage, a in agg.items():
        runs = a["runs"]
        row: dict[str, Any] = {
            "stage": stage,
            "runs": runs,
            "of_runs": sum(window[k]["sessions"] for k in STAGE_ELIGIBLE[stage]),
            "audio_s": round(a["audio_s"], 3),
            "secs": round(a["secs"], 3),
            "targets": [{"code": c, "runs": n} for c, n in sorted(
                targets.get(stage, {}).items(), key=lambda kv: (-kv[1], kv[0]))][:16],
        }
        if stage == "diarizing":
            row["speakers_avg"] = round(a["speakers"] / runs, 2) if runs else 0.0
        elif stage == "vad":
            row["retained_avg"] = round(a["retained_sum"] / runs, 3) if runs else 0.0
        elif stage == "translating":
            row["kept_original"] = int(translation.get("kept_original", 0))
        out.append(row)
    order = {s: i for i, s in enumerate(STAGE_ELIGIBLE)}
    out.sort(key=lambda s: order[s["stage"]])
    return out


def _stages(conn: sqlite3.Connection, user_id: str | None, start_hour: int,
            end_hour: int, window: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Stage rows for the window, every recorded stage in pipeline order
    (document() drops the decode and the download again unless asked for
    all_stages — the desktop app lists the optional ones). The per-stage extras are
    emitted only where they mean something: a speaker average for
    diarization, a retained-audio average for silence skipping, a
    kept-original count for translation."""
    targets: dict[str, list[dict[str, Any]]] = {}
    where, params = _scope_where(user_id)
    for r in conn.execute(
        "SELECT stage, target, SUM(runs) AS runs FROM usage_target_hourly"
        + where + " AND hour >= ? AND hour < ? GROUP BY stage, target"
        " ORDER BY runs DESC, target", (*params, start_hour, end_hour),
    ):
        targets.setdefault(r["stage"], []).append(
            {"code": r["target"], "runs": int(r["runs"] or 0)})
    out: list[dict[str, Any]] = []
    for r in conn.execute(
        "SELECT stage, SUM(runs) AS runs, SUM(audio_s) AS audio_s,"
        " SUM(secs) AS secs, SUM(speakers) AS speakers,"
        " SUM(retained_sum) AS retained_sum, SUM(kept_original) AS kept_original"
        " FROM usage_stage_hourly" + where + " AND hour >= ? AND hour < ?"
        " GROUP BY stage", (*params, start_hour, end_hour),
    ):
        stage = r["stage"]
        applies = STAGE_ELIGIBLE.get(stage)
        if applies is None:
            continue
        runs = int(r["runs"] or 0)
        row: dict[str, Any] = {
            "stage": stage,
            "runs": runs,
            "of_runs": sum(window[k]["sessions"] for k in applies),
            "audio_s": round(float(r["audio_s"] or 0.0), 3),
            "secs": round(float(r["secs"] or 0.0), 3),
            "targets": targets.get(stage, [])[:16],
        }
        if stage == "diarizing":
            row["speakers_avg"] = (round(int(r["speakers"] or 0) / runs, 2)
                                   if runs else 0.0)
        elif stage == "vad":
            row["retained_avg"] = (round(float(r["retained_sum"] or 0.0) / runs, 3)
                                   if runs else 0.0)
        elif stage == "translating":
            row["kept_original"] = int(r["kept_original"] or 0)
        out.append(row)
    order = {s: i for i, s in enumerate(STAGE_ELIGIBLE)}
    out.sort(key=lambda s: order[s["stage"]])
    return out


def _dictation(conn: sqlite3.Connection, user_id: str | None, start_hour: int,
               end_hour: int, cell: dict[str, Any]) -> dict[str, Any]:
    activation = {a: 0 for a in ACTIVATIONS}
    delivery = {d: 0 for d in DELIVERIES}
    translation = {t: 0 for t in TRANSLATIONS}
    where, params = _scope_where(user_id)
    for r in conn.execute(
        "SELECT activation, delivery, translation, SUM(sessions) AS sessions"
        " FROM usage_dictation_hourly" + where + " AND hour >= ? AND hour < ?"
        " GROUP BY activation, delivery, translation",
        (*params, start_hour, end_hour),
    ):
        n = int(r["sessions"] or 0)
        if r["activation"] in activation:
            activation[r["activation"]] += n
        if r["delivery"] in delivery:
            delivery[r["delivery"]] += n
        if r["translation"] in translation:
            translation[r["translation"]] += n
    audio_s = float(cell["audio_s"])
    return {
        "sessions": int(cell["sessions"]),
        "words": int(cell["words"]),
        "audio_s": round(audio_s, 3),
        "wpm": round(cell["words"] / (audio_s / 60.0), 1) if audio_s > 0 else 0.0,
        "activation": activation,
        "delivery": delivery,
        "translation": translation,
    }


def _dictation_targets(conn: sqlite3.Connection, user_id: str | None,
                       start_ts: float, end_ts: float,
                       with_stages: tuple[str, ...] = (),
                       key_id: str | None = None) -> list[dict[str, Any]]:
    """Which languages dictations were translated into, over the window:
    `[{code, runs, kept_original}]`, busiest first, at most 16. Read from
    the per-job stage rows (a dictation that translated carries a
    translating stage with its target codes) joined to the job's reported
    outcome, so `kept_original` says how many of those runs reverted to
    the original. Per-job rows keep USAGE_JOBS_RETENTION_DAYS, so a window
    beyond that shows fewer runs here than the outcome buckets do."""
    by_code: dict[str, dict[str, int]] = {}
    where, params = _scope_where(user_id, key_id, col="usage_jobs.user_id",
                                 key_col="usage_jobs.key_id")
    for r in conn.execute(
        "SELECT s.targets, usage_jobs.translation FROM usage_jobs"
        " JOIN usage_job_stages s ON s.job_id = usage_jobs.job_id"
        + where + " AND usage_jobs.kind = 'dictation'"
        " AND s.stage = 'translating' AND usage_jobs.created_ts >= ?"
        " AND usage_jobs.created_ts < ?" + _with_clause(with_stages),
        (*params, float(start_ts), float(end_ts), *with_stages),
    ):
        kept = r["translation"] == "kept_original"
        for code in (r["targets"] or "").split(","):
            code = code.strip()
            if not code:
                continue
            t = by_code.setdefault(code, {"runs": 0, "kept_original": 0})
            t["runs"] += 1
            if kept:
                t["kept_original"] += 1
    return [{"code": c, **v} for c, v in sorted(
        by_code.items(), key=lambda kv: (-kv[1]["runs"], kv[0]))][:16]


def _streak(words_by_day: dict[datetime.date, int],
            today: datetime.date) -> dict[str, int]:
    """Consecutive days with any words. `current` runs back from today —
    or from yesterday while today is still empty, so a streak is not shown
    as broken at breakfast. `best` is the longest run in the whole retained
    history."""
    active = {d for d, w in words_by_day.items() if w > 0}
    current = 0
    day = today if today in active else today - datetime.timedelta(days=1)
    while day in active:
        current += 1
        day -= datetime.timedelta(days=1)
    best = run = 0
    prev: datetime.date | None = None
    for d in sorted(active):
        run = run + 1 if prev is not None and d - prev == datetime.timedelta(days=1) else 1
        best = max(best, run)
        prev = d
    return {"current": current, "best": best}
