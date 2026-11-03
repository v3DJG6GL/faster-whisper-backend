"""A failed session-index rebuild must not mark the sibling commit as seen."""

import sqlite3

import pytest

from tests.auth._store_helpers import load_sessions_store_copy, sibling_write


@pytest.fixture
def worker(tmp_path):
    """An independent copy of sessions_store on its own DB file, like one
    uvicorn worker with its own connection and in-memory index."""
    mod = load_sessions_store_copy("sessions_store_rebuild_retry", str(tmp_path / "sessions.db"))
    yield mod
    mod._conn.close()
    mod._conn = None


def test_sibling_logout_still_seen_after_a_failed_rebuild(worker, monkeypatch):
    """The version used to be recorded before the SELECT, so after one failed
    rebuild every later lookup skipped the rebuild and kept accepting the
    cookie a sibling worker had logged out."""
    w = worker
    raw, _csrf = w.create_session("victim", 3600.0)
    assert w.lookup_session(raw) is not None
    sibling_write(w, ("UPDATE sessions SET revoked_ts = 1",))

    real = w._require_conn()

    class _FailingRebuild:
        def execute(self, sql, *args):
            if sql.startswith("SELECT token_hash"):
                raise sqlite3.OperationalError("database is locked")
            return real.execute(sql, *args)

    w._REFRESH_MIN_INTERVAL_S = 0.0     # "the interval has elapsed"
    monkeypatch.setattr(w, "_require_conn", _FailingRebuild)
    with pytest.raises(sqlite3.OperationalError):
        w.lookup_session(raw)
    monkeypatch.undo()
    assert w.lookup_session(raw) is None
