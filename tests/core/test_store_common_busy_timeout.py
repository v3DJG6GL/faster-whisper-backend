"""open_wal_db: the connection contract spells out its busy timeout and
creates a store's file owner-only."""
import os
import sqlite3
import stat
import sys

import pytest

from faster_whisper_backend.core import store_common


def test_open_wal_db_sets_busy_timeout(tmp_path):
    conn = store_common.open_wal_db(str(tmp_path / "x.sqlite3"))
    try:
        ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert ms == int(store_common.BUSY_TIMEOUT_S * 1000)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX file modes")
def test_open_wal_db_creates_the_file_and_its_sidecars_owner_only(tmp_path):
    """A store holds transcript text and key hashes, so its file is born 0600
    instead of sitting at the umask until init_db's secure_db_file lands. The
    WAL sidecars take the main file's mode."""
    path = str(tmp_path / "x.sqlite3")
    old_umask = os.umask(0o022)
    conn = None
    try:
        conn = store_common.open_wal_db(path)
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('row')")
        # No secure_db_file call: these are the modes at creation.
        modes = {suffix: stat.S_IMODE(os.stat(path + suffix).st_mode)
                 for suffix in ("", "-wal", "-shm")}
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "row"
    finally:
        os.umask(old_umask)
        if conn is not None:
            conn.close()
    assert modes == {"": 0o600, "-wal": 0o600, "-shm": 0o600}, \
        {k: oct(m) for k, m in modes.items()}


def test_open_wal_db_keeps_an_existing_store(tmp_path):
    path = str(tmp_path / "old.sqlite3")
    seed = sqlite3.connect(path)
    seed.execute("CREATE TABLE t (v TEXT)")
    seed.execute("INSERT INTO t VALUES ('kept')")
    seed.commit()
    seed.close()
    size = os.path.getsize(path)
    conn = store_common.open_wal_db(path)
    try:
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "kept"
        assert os.path.getsize(path) >= size
    finally:
        conn.close()


def test_open_wal_db_reports_the_connect_error_when_the_file_cannot_be_made(
        monkeypatch, tmp_path):
    """Creating the file first is best-effort: when it fails, the store still
    opens (or fails) exactly as sqlite3.connect decides."""
    def refuse(path, flags, mode=0o777, **kw):
        raise PermissionError(13, "denied", str(path))

    monkeypatch.setattr(store_common.os, "open", refuse)
    conn = store_common.open_wal_db(str(tmp_path / "x.sqlite3"))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()
    monkeypatch.undo()
    # A path sqlite cannot open either (a directory) raises sqlite's own error.
    (tmp_path / "d.sqlite3").mkdir()
    with pytest.raises(sqlite3.OperationalError):
        store_common.open_wal_db(str(tmp_path / "d.sqlite3"))
