"""Durable browser-session store — the cookie layer on top of api_keys_store.

The WebUI exchanges a pasted API key at /auth/login for an HttpOnly session
cookie. This module persists those sessions so they survive a browser/computer
restart (unlike the old tab-scoped sessionStorage). Non-browser clients (curl,
SDKs) never touch this — they keep sending `Authorization: Bearer`.

Storage layout:

  cfg.SESSIONS_DB — SQLite (WAL) with two tables:
    sessions — { token_hash, user_id, key_id, csrf_token, created_ts,
                 expires_ts, revoked_ts }  (key_id = the API key the
               session signed in with, so per-key locks bind on cookies)
    meta     — { k, v }: one 'revocations' counter, bumped per logout so a
               sibling worker's config_version() notices it (see
               revocation_generation)

Session tokens at rest are SHA-256(raw_token) hex — same rationale as
api_keys_store: a high-entropy random token (256-bit) makes slow password
hashes pointless. The raw token lives only in the HttpOnly cookie.

`csrf_token` is the double-submit pairing value. It is delivered to JS via a
readable cookie (and /auth/whoami) and echoed back as X-CSRF-Token on
cookie-authenticated mutations; the CSRF middleware compares it against this
stored value. Stored plaintext because it is, by design, handed to the client.

Lookup is O(1) via an in-memory `_SESSION_INDEX: dict[token_hash, row]`,
maintained INCREMENTALLY on create/revoke and rebuilt in full only on init,
purge, or when PRAGMA data_version shows a sibling worker's commit.

Expiry is absolute: a row lives SESSION_TTL_S from login, whatever the
activity, and lookups never write. The cookie is set once at login with that
max_age, so a row that slid forward only ever served a token replayed outside
the browser (a copied cookie stayed valid for as long as it kept being used).
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
import threading
import time
from hashlib import sha256
from typing import Any

from faster_whisper_backend.core import store_common
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import version as settings_version

logger = logging.getLogger("whisper-api")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
# Set as the LAST statement of init_db: every earlier statement (the PRAGMAs,
# the schema script, the migration) can still raise, and main treats an
# init_db failure as non-fatal. Gating on `_conn is not None` alone left a
# partial-init window where the connection exists but the table does not, so
# every cookie-bearing request 500'd instead of degrading to bearer-only auth.
# Mirrors api_keys_store._DB_READY.
_DB_READY: bool = False

# In-memory token_hash → session row index. Maintained incrementally on
# create/revoke; rebuilt in full only on init, purge, or a detected sibling
# commit. INVARIANT: any incremental insert must carry exactly the keys
# _rebuild_index_locked() produces (user_id, key_id, csrf_token, created_ts,
# expires_ts).
_SESSION_INDEX: dict[str, dict[str, Any]] = {}

# Last `PRAGMA data_version` observed on _conn — same role as in
# api_keys_store: SQLite only moves it for commits made by ANOTHER
# connection, so it is how one uvicorn worker notices a sibling worker's
# logout / session revoke instead of honouring a dead cookie indefinitely.
_DATA_VERSION: int = -1

# Throttle for the read-path sibling-commit check: every sibling login and
# logout is its own autocommit and moves PRAGMA data_version for every OTHER
# worker, so with SERVER_WORKERS>1 unthrottled lookups would rebuild the whole
# index (O(live sessions) under _lock) once per such commit. This caps the
# cost at one rebuild per interval per worker while keeping a cross-worker
# revocation visible within ~1 s. The pre-write check in create/revoke stays
# unthrottled (correctness before a write). The throttle applies to index
# HITS only; a miss forces the check so a sibling's fresh login is never bounced.
# Measured on the monotonic clock: a backwards wall-clock step would make the
# elapsed time negative and skip every HIT's check for the size of the step.
_REFRESH_MIN_INTERVAL_S = 1.0
_LAST_REFRESH_TS: float = float("-inf")
# Last 'revocations' counter revocation_generation() returned. When it moves,
# the index is refreshed past the throttle above (see that function).
_LAST_REV_GEN: int = -1

_TOKEN_BYTES = 32   # secrets.token_urlsafe(32) → 43-char base64url, 256-bit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  token_hash  TEXT PRIMARY KEY,
  user_id     TEXT NOT NULL,
  key_id      TEXT,
  csrf_token  TEXT NOT NULL,
  created_ts  REAL NOT NULL,
  expires_ts  REAL NOT NULL,
  revoked_ts  REAL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_ts);
CREATE TABLE IF NOT EXISTS meta (
  k  TEXT PRIMARY KEY,
  v  INTEGER NOT NULL
);
"""


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Open the SQLite DB (WAL) and ensure the schema exists. Idempotent.
    Purges expired rows and builds the in-memory index from active ones."""
    global _conn, _DB_READY, _SESSION_INDEX
    _DB_READY = False
    # Drop the previous DB's sessions with it (mirrors api_keys_store's
    # _KEY_INDEX reset): if anything below raises, lookup_session must not
    # keep honouring tokens from a store that is no longer open.
    with _lock:
        _SESSION_INDEX = {}
    # Close the previous handle before rebinding (mirrors api_keys_store), or
    # every re-init leaks a connection plus its WAL/-shm handles.
    if _conn is not None:
        try:
            _conn.close()
        except sqlite3.Error:
            pass
        finally:
            _conn = None
    _conn = store_common.open_wal_db(db_path)
    _conn.executescript(_SCHEMA)
    _migrate_add_key_id(_conn)
    store_common.secure_db_file(db_path)
    with _lock:
        _purge_expired_locked()
        _rebuild_index_locked()
    _DB_READY = True


def _migrate_add_key_id(conn: sqlite3.Connection) -> None:
    """Add the `key_id` column to a sessions table created before login began
    stamping the key. Pre-migration sessions keep `key_id` NULL, so they resolve
    to the `(session)` sentinel (no key layer) — the prior behaviour — until the
    user next logs in and a fresh, key-stamped session is issued."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    if "key_id" not in cols:
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN key_id TEXT")
        except sqlite3.Error:
            pass  # best-effort; a fresh DB already has the column


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("sessions_store.init_db() was not called before use.")
    return _conn


# ---------------------------------------------------------------------
# Token + index helpers
# ---------------------------------------------------------------------

def hash_token(raw_token: str) -> str:
    """SHA-256 hex of the raw session token (UTF-8)."""
    return sha256(raw_token.encode("utf-8")).hexdigest()


def _rebuild_index_locked() -> None:
    """Rebuild _SESSION_INDEX from the DB. Caller holds _lock (or is in
    init). Only non-revoked, non-expired rows are indexed."""
    global _SESSION_INDEX, _DATA_VERSION
    conn = _require_conn()
    now = time.time()
    # Snapshot the version BEFORE the SELECT: a sibling commit landing between
    # the two would otherwise be marked as already seen without being read.
    _DATA_VERSION = _data_version_locked()
    rows = conn.execute(
        "SELECT token_hash, user_id, key_id, csrf_token, created_ts, expires_ts"
        " FROM sessions WHERE revoked_ts IS NULL AND expires_ts > ?",
        (now,),
    ).fetchall()
    _SESSION_INDEX = {
        r["token_hash"]: {
            "user_id": r["user_id"],
            "key_id": r["key_id"],
            "csrf_token": r["csrf_token"],
            "created_ts": float(r["created_ts"]),
            "expires_ts": float(r["expires_ts"]),
        }
        for r in rows
    }


def _data_version_locked() -> int:
    """Read `PRAGMA data_version` off the shared connection. Falls back to the
    cached value on error so a transient sqlite failure can't be mistaken for
    a sibling commit. Caller holds _lock (or is in init)."""
    try:
        return int(_require_conn().execute("PRAGMA data_version").fetchone()[0])
    except (sqlite3.Error, RuntimeError):
        return _DATA_VERSION


def _refresh_if_sibling_committed(force: bool = False) -> None:
    """Rebuild _SESSION_INDEX when another PROCESS committed since the last
    check — one header read on the connection we already hold. Mirrors
    api_keys_store._refresh_if_sibling_committed(); without it a logout in one
    uvicorn worker would leave the cookie valid in every other worker.

    Throttled to one check per _REFRESH_MIN_INTERVAL_S (see the constant's
    comment): sibling login/logout commits can be frequent, and each detected one
    costs a full O(live sessions) rebuild under _lock. `force=True` skips the
    interval (miss path of lookup_session): a session created by a sibling
    worker in the last second must not 401 the request that carries its
    brand-new cookie; the PRAGMA header read is cheap and a rebuild still
    happens only when data_version actually moved."""
    global _LAST_REFRESH_TS
    if _conn is None or not _DB_READY:
        return
    if not force and time.monotonic() - _LAST_REFRESH_TS < _REFRESH_MIN_INTERVAL_S:
        return
    with _lock:
        _LAST_REFRESH_TS = time.monotonic()
        if _data_version_locked() != _DATA_VERSION:
            _rebuild_index_locked()


def _purge_expired_locked() -> None:
    """Best-effort delete of revoked/expired rows. Caller holds _lock.
    `created_ts` past SESSION_TTL_S counts as expired too (see
    _effective_expiry)."""
    conn = _require_conn()
    now = time.time()
    try:
        conn.execute(
            "DELETE FROM sessions WHERE revoked_ts IS NOT NULL OR expires_ts <= ?"
            " OR created_ts <= ?",
            (now, now - float(cfg.SESSION_TTL_S)),
        )
    except sqlite3.Error:
        pass  # cleanup is non-fatal


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def create_session(user_id: str, ttl_s: float,
                   key_id: str | None = None) -> tuple[str, str]:
    """Create a session for `user_id` valid for `ttl_s` seconds. Returns
    `(raw_token, csrf_token)`. The raw token is the ONLY way the caller
    will see it — set it in the HttpOnly cookie and discard.

    `key_id` is the API key exchanged at login. Stamping it lets per-key
    overrides/locks bind on the resulting cookie-authenticated requests too
    (without it the session resolves to the `(session)` sentinel = no key
    layer, so per-key restrictions would silently stop applying)."""
    raw_token = secrets.token_urlsafe(_TOKEN_BYTES)
    csrf_token = secrets.token_urlsafe(_TOKEN_BYTES)
    th = hash_token(raw_token)
    now = time.time()
    expires = now + float(ttl_s)
    conn = _require_conn()
    with _lock:
        # Absorb any sibling worker's commit BEFORE we write. Re-stamping
        # _DATA_VERSION after our own commit (as this used to) reads a counter
        # that has already moved for the sibling's commit, marking it as seen
        # and stranding e.g. a revocation that landed in another worker.
        # NOTE: with SERVER_WORKERS>1 this check can still trigger one full
        # rebuild here when a sibling committed since the last check (bounded
        # by _REFRESH_MIN_INTERVAL_S on the read path) — it is only the
        # STEADY-STATE per-login rebuild that is gone.
        if _data_version_locked() != _DATA_VERSION:
            _rebuild_index_locked()
        conn.execute(
            "INSERT INTO sessions"
            " (token_hash, user_id, key_id, csrf_token, created_ts,"
            " expires_ts, revoked_ts)"
            " VALUES (?,?,?,?,?,?,NULL)",
            (th, user_id, key_id, csrf_token, now, expires),
        )
        # Insert the one new row rather than re-reading the whole table.
        # /auth/login takes any valid key, is throttled only on FAILURES
        # (`LOGIN_FAILURE_RATE`, so a scripted valid-key login loop is
        # unthrottled), has no per-user session cap, and rows are reaped only
        # hourly by `_sessions_purge_loop` in main.py — so a full STEADY-STATE
        # rebuild here made N logins cost O(N²)
        # and, because `login` is async and calls this inline, every one of those
        # rebuilds runs on the event loop. Our own commit does not move
        # PRAGMA data_version on this connection, so _DATA_VERSION stays valid
        # without a re-stamp — the pre-write check above is what keeps the
        # sibling-detection contract in _refresh_if_sibling_committed.
        _SESSION_INDEX[th] = {
            "user_id": user_id,
            "key_id": key_id,
            "csrf_token": csrf_token,
            "created_ts": now,
            "expires_ts": expires,
        }
    logger.info("[auth] session created user=%s ttl=%.0fs", user_id[:8], ttl_s)
    return raw_token, csrf_token


def lookup_session(raw_token: str) -> dict[str, Any] | None:
    """Resolve a raw session token to its active record. Returns
    `{user_id, key_id, csrf_token, created_ts, expires_ts}` (key_id may be
    None for pre-migration sessions) or None if missing / revoked / expired.

    No sliding window: /auth/login is the only Set-Cookie, with
    max_age=SESSION_TTL_S, so the browser re-logs in SESSION_TTL_S after
    login regardless of activity, and the row ends at the same moment (see
    _effective_expiry). Lookups never write.
    """
    if not raw_token or not _DB_READY:
        return None
    _refresh_if_sibling_committed()
    th = hash_token(raw_token)
    rec = _SESSION_INDEX.get(th)
    if rec is None:
        # A miss may be a session a sibling worker created inside the
        # throttle window (login on worker A, next request on worker B).
        _refresh_if_sibling_committed(force=True)
        rec = _SESSION_INDEX.get(th)
        if rec is None:
            return None
    now = time.time()
    if _effective_expiry(rec) <= now:
        # Lazily evict an index entry that lapsed since the last rebuild.
        with _lock:
            _SESSION_INDEX.pop(th, None)
        return None
    return dict(rec)


def _effective_expiry(rec: dict[str, Any]) -> float:
    """The stored expiry, capped at created_ts + the CURRENT SESSION_TTL_S.
    The cap ends rows that older builds slid forward (their windows grew by
    the session's age on every slide, so expires_ts can lie centuries out)
    and lets a lowered SESSION_TTL_S reach sessions that already exist."""
    return min(rec["expires_ts"], rec["created_ts"] + float(cfg.SESSION_TTL_S))


def revoke_session(raw_token: str) -> None:
    """Soft-revoke a session (used by /auth/logout). No-op if unknown.
    Bumps the config version so a live cookie-authenticated streaming socket
    re-authenticates and closes (streaming.routes._refresh_ident), and the
    'revocations' counter so a SIBLING worker's config_version() does too.
    Both only when a row was actually revoked: /auth/logout is
    unauthenticated, so a junk cookie must not move the global version (every
    live stream re-resolves auth on each move).

    No-op while the store is not ready (init_db failed, which main treats as
    non-fatal), like lookup_session: logout then still clears the cookies
    instead of failing with a 500."""
    if not raw_token or not _DB_READY:
        return
    th = hash_token(raw_token)
    now = time.time()
    conn = _require_conn()
    with _lock:
        # See create_session: pick up a sibling worker's commit BEFORE writing,
        # never by re-stamping the counter after our own commit. As there,
        # this check can still cost one full rebuild on the event loop when a
        # sibling worker has committed since the last check.
        if _data_version_locked() != _DATA_VERSION:
            _rebuild_index_locked()
        cur = conn.execute(
            "UPDATE sessions SET revoked_ts = ?"
            " WHERE token_hash = ? AND revoked_ts IS NULL",
            (now, th),
        )
        revoked = cur.rowcount > 0
        if revoked:
            conn.execute(
                "INSERT INTO meta (k, v) VALUES ('revocations', 1)"
                " ON CONFLICT(k) DO UPDATE SET v = v + 1")
        # Drop the one key rather than re-reading every live session, mirroring
        # the incremental insert create_session already does. The full rebuild
        # here was O(live sessions) on the event loop — measured ~37 ms at
        # 20 000 rows, and /auth/login is throttled only on failures
        # (`LOGIN_FAILURE_RATE`) with no per-user cap, so N is
        # caller-growable. No post-commit re-stamp: our own commit does not
        # move PRAGMA data_version on this connection, and re-stamping would
        # swallow a sibling's commit that landed since the last check.
        _SESSION_INDEX.pop(th, None)
    if revoked:
        settings_version.bump_config_version()   # signed-out identity's live streaming idents re-auth


def revocation_generation() -> int:
    """Monotonic count of revoked sessions, or -1 before init_db().

    Read by settings_version.config_version()'s cross-worker probe: a logout
    bumps the config version only in the worker that served it, and the
    sessions table lives in its own DB file, so api_keys_store.data_version()
    never moves for it. Not `PRAGMA data_version`: every sibling login moves
    that, and every live stream would re-resolve auth on each one.

    When the counter moved, the index is refreshed with force=True before
    returning: the caller bumps the config version, a live stream then
    re-authenticates through lookup_session, and a throttled HIT on the
    stale index would let the revoked cookie through and spend the bump."""
    global _LAST_REV_GEN
    if not _DB_READY:
        return -1
    try:
        with _lock:
            row = _require_conn().execute(
                "SELECT v FROM meta WHERE k = 'revocations'").fetchone()
    except (sqlite3.Error, RuntimeError):
        return -1
    gen = int(row[0]) if row else 0
    if gen != _LAST_REV_GEN:
        _LAST_REV_GEN = gen
        _refresh_if_sibling_committed(force=True)
    return gen


def purge_expired() -> None:
    """Public best-effort cleanup of revoked/expired rows + index rebuild."""
    with _lock:
        _purge_expired_locked()
        _rebuild_index_locked()


def _reset_for_tests() -> None:
    """Drop the in-memory caches so the autouse test fixture starts clean."""
    global _SESSION_INDEX, _DATA_VERSION, _DB_READY, _LAST_REFRESH_TS
    global _LAST_REV_GEN
    _SESSION_INDEX = {}
    _DATA_VERSION = -1
    _DB_READY = _conn is not None
    _LAST_REFRESH_TS = float("-inf")
    _LAST_REV_GEN = -1
