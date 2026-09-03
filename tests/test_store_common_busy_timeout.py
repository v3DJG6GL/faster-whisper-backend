"""open_wal_db: the connection contract spells out its busy timeout."""
from faster_whisper_backend.core import store_common


def test_open_wal_db_sets_busy_timeout(tmp_path):
    conn = store_common.open_wal_db(str(tmp_path / "x.sqlite3"))
    try:
        ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert ms == int(store_common.BUSY_TIMEOUT_S * 1000)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()
