"""Filesystem hardening shared by the SQLite stores.

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

import os
import re
import sqlite3

# Cap and control-character screen for a CALLER-supplied label that goes into a
# log line. Lives here rather than in main so the stores can reach it without an
# import cycle (main imports the stores, not the other way round).
LOG_FIELD_MAX = 120
# \x00-\x1f already covers ESC, so the classic ANSI-escape angle is closed.
# What it did NOT cover: U+2028/U+2029, which the /logs viewer renders as a
# forced line break under `white-space: pre-wrap` (the same forged-record
# problem as a bare LF), and the BiDi overrides U+202A-U+202E / U+2066-U+2069,
# which reorder text WITHIN a line in both a terminal and a browser. C1/DEL
# (\x7f-\x9f) is largely theoretical on a UTF-8 terminal but is inert in every
# legitimate value, so it goes in too. Deliberately NOT all of \p{Cf}: that
# would eat ZWNJ/ZWJ (U+200C/D), which occur in real Persian and Indic
# filenames. Transcripts never pass through log_safe.
_LOG_UNSAFE_RE = re.compile("[\r\n\x00-\x1f\x7f-\x9f  ‪-‮⁦-⁩]")


def log_safe(s) -> str:
    """Collapse control characters in a caller-supplied label and cap its
    length. A bare CR/LF would otherwise split one record into what the /logs
    viewer renders as extra, attacker-written lines — indistinguishable from
    genuine records, including their severity styling."""
    return _LOG_UNSAFE_RE.sub("?", s or "")[:LOG_FIELD_MAX]

# How long a statement waits for another connection's write lock before
# raising "database is locked". Within one process each store's _lock already
# serialises its writers; the wait matters when SERVER_WORKERS > 1 (sibling
# processes on the same file) and during WAL checkpoints. 5 s is the value
# Litestream and the SQLite docs recommend; pysqlite's own default happens to
# be the same, but the contract is spelled out here rather than inherited.
BUSY_TIMEOUT_S = 5.0


def open_wal_db(path: str) -> sqlite3.Connection:
    """THE connection contract for the SQLite stores: create the parent dir,
    open `path` with a Row factory, a busy timeout, and switch it to WAL.
    Every store's init_db goes through here so a change to the contract lands
    in one place instead of eight.

    isolation_level=None puts pysqlite in autocommit mode; every statement
    commits independently (each store's _lock serialises its writers). WAL
    gives crash-safety per statement; synchronous=NORMAL is the standard WAL
    recommendation (full durability against power loss is FULL, but NORMAL is
    fine against process crash and ~10x faster on small writes)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None,
                           timeout=BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_S * 1000)};")
    return conn


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

    Used for surfaces that hold the same plaintext dictation as the stores but
    are not SQLite: the rotating server log (every request block carries RAW
    WHISPER / FINAL text — main.py), config.local.json (admin host allowlist
    and compiled pipeline rules — config_store.py), the model-sizes cache
    (model_sizes.py), downloaded URL media files (url_media_store.py) and
    the merged / VAD-trimmed dictation WAVs written through the tmp+os.replace
    swap (audio_merge.merge_wavs, audio_vad_trim.trim_wav) — os.replace
    carries the tmp inode's mode onto the destination, so the tmp is pinned
    owner-only before the swap (merge_wavs additionally creates its tmp 0600
    up front; this call is belt-and-braces there).
    """
    _chmod(path, 0o600)


def _chmod(path: str, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass  # unsupported filesystem, foreign owner, or not yet created
