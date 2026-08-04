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


def _chmod(path: str, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass  # unsupported filesystem, foreign owner, or not yet created
