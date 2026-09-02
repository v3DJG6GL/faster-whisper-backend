"""Durable per-key / per-user usage rollup.

SQLite (stdlib) in WAL mode — single-file, crash-safe, indexed. Lives at
cfg.USAGE_DB (defaults to usage.local.sqlite3 alongside config.local.json).

Why a separate store: the recent-transcriptions table
(transcriptions_store.py) is a pruned rolling window (row-cap + 30-day TTL),
so it cannot back lifetime usage totals. This store keeps compact HOURLY
ROLLUPS that are never aggressively pruned, so lifetime totals are a SUM over
hours.

Tables (every rollup is keyed by UTC epoch-hour, see below):

  usage_hourly            hour × key_id × kind — requests/errors/words/audio_s/
                          proc_s and `sessions` (jobs, counted once per job_id)
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

import datetime
import logging
import sqlite3
import threading
import time
import zoneinfo
from typing import Any

import store_common

logger = logging.getLogger("whisper-api")

_EPOCH = datetime.date(1970, 1, 1)

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

# Sentinel for rows we can't attribute to a real key/user. Kept as a literal
# id string so the NOT NULL columns stay satisfied and aggregation treats it
# as its own bucket. The UI renders it as a plain label.
OPEN_MODE_ID = "(open-mode)"

# ORDER BY column names are interpolated, so they MUST come from this set.
_METRICS: frozenset[str] = frozenset(("requests", "errors", "words", "audio_s"))

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
  proc_s   REAL    NOT NULL DEFAULT 0,
  sessions INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, key_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_usage_user_hour ON usage_hourly(user_id, hour);
CREATE INDEX IF NOT EXISTS idx_usage_hour      ON usage_hourly(hour);

CREATE TABLE IF NOT EXISTS usage_jobs (
  job_id      TEXT    PRIMARY KEY,
  user_id     TEXT    NOT NULL,
  key_id      TEXT    NOT NULL,
  kind        TEXT    NOT NULL,
  created_ts  REAL    NOT NULL,
  status      TEXT    NOT NULL,
  audio_s     REAL    NOT NULL DEFAULT 0,
  words       INTEGER NOT NULL DEFAULT 0,
  proc_s      REAL    NOT NULL DEFAULT 0,
  utterances  INTEGER NOT NULL DEFAULT 0,
  model       TEXT,
  language    TEXT,
  activation  TEXT,
  delivery    TEXT,
  app_id      TEXT,
  translation TEXT,
  reported_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_usage_jobs_created ON usage_jobs(created_ts);

CREATE TABLE IF NOT EXISTS usage_job_stages (
  job_id   TEXT NOT NULL,
  stage    TEXT NOT NULL,
  secs     REAL NOT NULL DEFAULT 0,
  model    TEXT,
  targets  TEXT,
  speakers INTEGER,
  retained REAL,
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
    transcriptions_store.init_db."""
    global _conn
    _conn = store_common.open_wal_db(path)
    _conn.execute("PRAGMA temp_store=MEMORY;")
    _park_legacy_hourly(_conn)
    _conn.executescript(_SCHEMA)
    _fold_legacy_hourly(_conn)
    _reclassify_unknown_hourly(_conn)
    store_common.secure_db_file(path)


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
                "  audio_s, proc_s, sessions)"
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
                "  audio_s, proc_s, sessions)"
                " SELECT hour, key_id, user_id, 'dictation', requests, errors,"
                "  words, audio_s, proc_s, sessions"
                " FROM usage_hourly WHERE kind=?"
                " ON CONFLICT(hour, key_id, kind) DO UPDATE SET"
                "  requests = requests + excluded.requests,"
                "  errors   = errors + excluded.errors,"
                "  words    = words + excluded.words,"
                "  audio_s  = audio_s + excluded.audio_s,"
                "  proc_s   = proc_s + excluded.proc_s,"
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
        out.append({
            "stage": _STAGE_ALIASES.get(name, name),
            "secs": float(s.get("secs") or 0.0),
            "model": s.get("model") if isinstance(s.get("model"), str) else None,
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
    proc_s: float | None = None,
) -> None:
    """Record one transcription request (a batch run, a text translation, or
    ONE dictation utterance) into the rollups. Best-effort: any failure is
    logged, never raised — a usage write must not break a transcription.
    Falsy ids fall back to the open-mode sentinel so the NOT NULL columns
    stay valid.

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
        p = float(proc_s or 0.0)
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
                            "  audio_s, words, proc_s, utterances, model, language)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)"
                            " ON CONFLICT(job_id) DO UPDATE SET"
                            "  utterances = utterances + 1,"
                            "  audio_s    = audio_s + excluded.audio_s,"
                            "  words      = words + excluded.words,"
                            "  proc_s     = proc_s + excluded.proc_s,"
                            "  status     = excluded.status,"
                            "  model      = COALESCE(excluded.model, model),"
                            "  language   = COALESCE(excluded.language, language)",
                            (jid, uid, kid, k, created_ts, status, a, w, p,
                             model or None, language or None),
                        )
                conn.execute(
                    "INSERT INTO usage_hourly"
                    " (hour, key_id, user_id, kind, requests, errors, words,"
                    "  audio_s, proc_s, sessions)"
                    " VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(hour, key_id, kind) DO UPDATE SET"
                    "  requests = requests + 1,"
                    "  errors   = errors + excluded.errors,"
                    "  words    = words  + excluded.words,"
                    "  audio_s  = audio_s + excluded.audio_s,"
                    "  proc_s   = proc_s + excluded.proc_s,"
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
    if jid is not None:
        conn.execute(
            "INSERT INTO usage_job_stages"
            " (job_id, stage, secs, model, targets, speakers, retained)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(job_id, stage) DO UPDATE SET"
            "  secs     = secs + excluded.secs,"
            "  model    = COALESCE(excluded.model, model),"
            "  targets  = COALESCE(excluded.targets, targets),"
            "  speakers = COALESCE(excluded.speakers, speakers),"
            "  retained = COALESCE(excluded.retained, retained)",
            (jid, st["stage"], st["secs"], st["model"],
             ",".join(st["targets"]) or None, st["speakers"], st["retained"]),
        )
    conn.execute(
        "INSERT INTO usage_stage_hourly"
        " (hour, user_id, stage, runs, audio_s, secs, speakers, retained_sum,"
        "  kept_original)"
        " VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)"
        " ON CONFLICT(hour, user_id, stage) DO UPDATE SET"
        "  runs          = runs + 1,"
        "  audio_s       = audio_s + excluded.audio_s,"
        "  secs          = secs + excluded.secs,"
        "  speakers      = speakers + excluded.speakers,"
        "  retained_sum  = retained_sum + excluded.retained_sum,"
        "  kept_original = kept_original + excluded.kept_original",
        (hour, uid, st["stage"], audio_s, st["secs"], st["speakers"] or 0,
         st["retained"] or 0.0, st["kept_original"]),
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
        "SELECT hour, requests, errors, words, audio_s"
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
                               "words": 0, "audio_s": 0.0}
        cell["requests"] += int(r["requests"] or 0)
        cell["errors"] += int(r["errors"] or 0)
        cell["words"] += int(r["words"] or 0)
        cell["audio_s"] += float(r["audio_s"] or 0.0)
    return [agg[d] for d in sorted(agg)]


def leaderboard(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
    by: str = "user",
    metric: str = "audio_s",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Top entities for a window, ranked by `metric`. by='user' groups by
    user_id; by='key' groups by key_id (carrying user_id). `metric` is
    validated against the column whitelist (it is interpolated into the
    ORDER BY)."""
    if metric not in _METRICS:
        metric = "audio_s"
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    if by == "key":
        group, cols = "key_id", "key_id, user_id"
    else:
        group, cols = "user_id", "user_id"
    cur = conn.execute(
        f"SELECT {cols},"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
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
            "audio_s": 0.0, "proc_s": 0.0}


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
        cell["proc_s"] += float(r["proc_s"] or 0.0)


def _rounded(split: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for cell in split.values():
        cell["audio_s"] = round(cell["audio_s"], 3)
        cell["proc_s"] = round(cell["proc_s"], 3)
    return split


# The optional stages a caller may narrow the document to (`with=`). The
# decode and a URL download are the job, not a stage of it.
WITH_STAGES: tuple[str, ...] = tuple(STAGE_APPLIES_TO)

# Widest window the document will reckon: ten years of days.
MAX_WINDOW_DAYS = 3650


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
        "today": _zero_split(),
        "total": _zero_split(),
        "series": [],
        "stages": [],
        "dictation": {
            "sessions": 0, "words": 0, "audio_s": 0.0, "wpm": 0.0,
            "activation": {a: 0 for a in ACTIVATIONS},
            "delivery": {d: 0 for d in DELIVERIES},
            "translation": {t: 0 for t in TRANSLATIONS},
        },
        "apps": [],
        "calendar": [],
        "hours": [],
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


def _first_day(conn: sqlite3.Connection, user_id: str,
               tz: zoneinfo.ZoneInfo | None,
               with_stages: tuple[str, ...]) -> int | None:
    if with_stages:
        row = conn.execute(
            "SELECT MIN(created_ts) AS ts FROM usage_jobs WHERE user_id = ?"
            + _with_clause(with_stages), (user_id, *with_stages),
        ).fetchone()
        if row is None or row["ts"] is None:
            return None
        return _epoch_day(_date_of(float(row["ts"]), tz))
    row = conn.execute(
        "SELECT MIN(hour) AS h FROM usage_hourly WHERE user_id = ?", (user_id,),
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
    user_id: str,
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
) -> dict[str, Any]:
    """One user's statistics document (everything /v1/usage returns except
    the username), reckoned in `tz` (None = server-local; `tz_name` is only
    echoed). The window is resolve_window()'s; `today` is the caller's
    current day whatever the window; `streak` runs over the FULL retained
    history so a short window never caps it.

    Days on the wire are days-since-epoch of the caller-local calendar date
    (×86 400 → UTC midnight of that date, the client's label math). Series,
    calendar and hours are sparse: a day/slot without usage is absent.

    `with_stages` (non-empty) recomputes the whole document from the per-job
    rows restricted to jobs that ran every listed stage — `range.source`
    says "jobs" and the client discloses `jobs_retention_days`."""
    conn = _require_conn()
    now = time.time() if now is None else float(now)
    today = _date_of(now, tz)
    with_stages = tuple(dict.fromkeys(with_stages))
    for st in with_stages:
        if st not in WITH_STAGES:
            raise ValueError(f"unknown stage: {st}")
    first_day = _first_day(conn, user_id, tz, with_stages)
    f, t = resolve_window(today=today, days=days, from_day=from_day,
                          to_day=to_day, all_time=all_time, first_day=first_day)
    doc = empty_document(
        from_day=f, to_day=t, tz=tz_name,
        source="jobs" if with_stages else "rollups",
        jobs_retention_days=jobs_retention_days, first_day=first_day)
    start_hour = _midnight_hour(_from_epoch_day(f), tz)
    end_hour = _midnight_hour(_from_epoch_day(t) + datetime.timedelta(days=1), tz)
    today_start = _midnight_hour(today, tz)
    today_end = _midnight_hour(today + datetime.timedelta(days=1), tz)
    if with_stages:
        _fill_from_jobs(conn, doc, user_id, tz, today, with_stages,
                        start_hour * 3600, end_hour * 3600,
                        today_start * 3600, today_end * 3600)
    else:
        _fill_from_rollups(conn, doc, user_id, tz, today,
                           start_hour, end_hour, today_start, today_end)
    dictation = doc["total"]["dictation"]
    doc["time_saved_s"] = round(max(
        0.0, dictation["words"] / TYPING_WPM * 60.0 - dictation["audio_s"]), 1)
    return doc


def _slot_of(ts: float, tz: zoneinfo.ZoneInfo | None) -> tuple[int, int]:
    """(weekday 0=Mon, hour 0..23) of `ts` in `tz`."""
    dt = datetime.datetime.fromtimestamp(ts, tz)
    return dt.weekday(), dt.hour


def _finish(doc: dict[str, Any], *, by_day, words_by_day, words_by_slot,
            words_all_days, today: datetime.date) -> None:
    """Common tail: the sparse arrays and the per-kind streaks."""
    doc["series"] = [{"day": _epoch_day(d), **_rounded(by_day[d])}
                     for d in sorted(by_day)]
    doc["calendar"] = [{"day": _epoch_day(d), **words_by_day[d]}
                       for d in sorted(words_by_day) if words_by_day[d]["all"] > 0]
    doc["hours"] = [{"dow": dow, "hour": hh, **words_by_slot[(dow, hh)]}
                    for (dow, hh) in sorted(words_by_slot)
                    if words_by_slot[(dow, hh)]["all"] > 0]
    doc["streak"] = {
        k: _streak({d: w[k] for d, w in words_all_days.items()}, today)
        for k in ("all", *KINDS)}


def _fill_from_rollups(conn, doc, user_id, tz, today, start_hour, end_hour,
                       today_start, today_end) -> None:
    today_split = _zero_split()
    window = _zero_split()
    by_day: dict[datetime.date, dict[str, dict[str, Any]]] = {}
    words_by_day: dict[datetime.date, dict[str, int]] = {}
    words_by_slot: dict[tuple[int, int], dict[str, int]] = {}
    words_all_days: dict[datetime.date, dict[str, int]] = {}
    for r in conn.execute(
        "SELECT hour, kind, sessions, requests, errors, words, audio_s, proc_s"
        " FROM usage_hourly WHERE user_id = ?", (user_id,),
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
            _add_words(words_by_slot.setdefault(_slot_of(ts, tz), _words_split()),
                       kind, words)
    doc["total"] = _rounded(window)
    doc["today"] = _rounded(today_split)
    doc["stages"] = _stages(conn, user_id, start_hour, end_hour, window)
    doc["dictation"] = _dictation(conn, user_id, start_hour, end_hour,
                                  window["dictation"])
    doc["apps"] = [
        {"app_id": r["app_id"], "sessions": int(r["sessions"] or 0),
         "words": int(r["words"] or 0)}
        for r in conn.execute(
            "SELECT app_id, SUM(sessions) AS sessions, SUM(words) AS words"
            " FROM usage_app_hourly WHERE user_id = ? AND hour >= ? AND hour < ?"
            " GROUP BY app_id ORDER BY sessions DESC, words DESC, app_id"
            " LIMIT 8", (user_id, start_hour, end_hour),
        )
    ]
    _finish(doc, by_day=by_day, words_by_day=words_by_day,
            words_by_slot=words_by_slot, words_all_days=words_all_days,
            today=today)


def _job_cell(job: sqlite3.Row) -> dict[str, Any]:
    return {
        "sessions": 1,
        "requests": max(1, int(job["utterances"] or 0)),
        "errors": 0 if job["status"] == "ok" else 1,
        "words": int(job["words"] or 0),
        "audio_s": float(job["audio_s"] or 0.0),
        "proc_s": float(job["proc_s"] or 0.0),
    }


def _add_cell(split: dict[str, dict[str, Any]], kind: str,
              cell: dict[str, Any]) -> None:
    for target in (split["all"], split.get(kind)):
        if target is None:
            continue
        for k, v in cell.items():
            target[k] += v


def _fill_from_jobs(conn, doc, user_id, tz, today, with_stages,
                    start_ts, end_ts, today_start_ts, today_end_ts) -> None:
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
    words_all_days: dict[datetime.date, dict[str, int]] = {}
    activation = {a: 0 for a in ACTIVATIONS}
    delivery = {d: 0 for d in DELIVERIES}
    translation = {t: 0 for t in TRANSLATIONS}
    apps: dict[str, dict[str, int]] = {}
    window_jobs: list[str] = []
    for job in conn.execute(
        "SELECT job_id, kind, created_ts, status, audio_s, words, proc_s,"
        " utterances, activation, delivery, translation, app_id"
        " FROM usage_jobs WHERE user_id = ?" + _with_clause(with_stages)
        + " ORDER BY created_ts", (user_id, *with_stages),
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
        _add_words(words_by_slot.setdefault(_slot_of(ts, tz), _words_split()),
                   kind, words)
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
    }
    doc["apps"] = [
        {"app_id": a, **v} for a, v in sorted(
            apps.items(), key=lambda kv: (-kv[1]["sessions"], -kv[1]["words"], kv[0]))
    ][:8]
    doc["stages"] = _stages_from_jobs(conn, window_jobs, window, translation)
    _finish(doc, by_day=by_day, words_by_day=words_by_day,
            words_by_slot=words_by_slot, words_all_days=words_all_days,
            today=today)


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
            if stage not in STAGE_APPLIES_TO:
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
            "of_runs": sum(window[k]["sessions"] for k in STAGE_APPLIES_TO[stage]),
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
    order = {s: i for i, s in enumerate(STAGE_APPLIES_TO)}
    out.sort(key=lambda s: order[s["stage"]])
    return out


def _stages(conn: sqlite3.Connection, user_id: str, start_hour: int,
            end_hour: int, window: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Stage rows for the window. Only the OPTIONAL stages are listed (the
    ones a user can switch on and wonder about); the decode itself and a URL
    download are the job, not a stage of it. The per-stage extras are
    emitted only where they mean something: a speaker average for
    diarization, a retained-audio average for silence skipping, a
    kept-original count for translation."""
    targets: dict[str, list[dict[str, Any]]] = {}
    for r in conn.execute(
        "SELECT stage, target, SUM(runs) AS runs FROM usage_target_hourly"
        " WHERE user_id = ? AND hour >= ? AND hour < ? GROUP BY stage, target"
        " ORDER BY runs DESC, target", (user_id, start_hour, end_hour),
    ):
        targets.setdefault(r["stage"], []).append(
            {"code": r["target"], "runs": int(r["runs"] or 0)})
    out: list[dict[str, Any]] = []
    for r in conn.execute(
        "SELECT stage, SUM(runs) AS runs, SUM(audio_s) AS audio_s,"
        " SUM(secs) AS secs, SUM(speakers) AS speakers,"
        " SUM(retained_sum) AS retained_sum, SUM(kept_original) AS kept_original"
        " FROM usage_stage_hourly WHERE user_id = ? AND hour >= ? AND hour < ?"
        " GROUP BY stage", (user_id, start_hour, end_hour),
    ):
        stage = r["stage"]
        applies = STAGE_APPLIES_TO.get(stage)
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
    order = {s: i for i, s in enumerate(STAGE_APPLIES_TO)}
    out.sort(key=lambda s: order[s["stage"]])
    return out


def _dictation(conn: sqlite3.Connection, user_id: str, start_hour: int,
               end_hour: int, cell: dict[str, Any]) -> dict[str, Any]:
    activation = {a: 0 for a in ACTIVATIONS}
    delivery = {d: 0 for d in DELIVERIES}
    translation = {t: 0 for t in TRANSLATIONS}
    for r in conn.execute(
        "SELECT activation, delivery, translation, SUM(sessions) AS sessions"
        " FROM usage_dictation_hourly WHERE user_id = ? AND hour >= ? AND hour < ?"
        " GROUP BY activation, delivery, translation",
        (user_id, start_hour, end_hour),
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
