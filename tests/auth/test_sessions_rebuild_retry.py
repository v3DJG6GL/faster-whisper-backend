"""A failed session-index rebuild must not mark the sibling commit as seen."""

import importlib.util
import os
import sqlite3

import pytest

from faster_whisper_backend.paths import REPO_ROOT


@pytest.fixture
def worker(tmp_path):
    """An independent copy of sessions_store on its own DB file, like one
    uvicorn worker with its own connection and in-memory index."""
    spec = importlib.util.spec_from_file_location(
        "sessions_store_rebuild_retry",
        os.path.join(REPO_ROOT, "faster_whisper_backend", "auth", "sessions_store.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init_db(str(tmp_path / "sessions.db"))
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
    db_path = w._require_conn().execute(
        "PRAGMA database_list").fetchone()["file"]
    sibling = sqlite3.connect(db_path)
    try:
        with sibling:
            sibling.execute("UPDATE sessions SET revoked_ts = 1")
    finally:
        sibling.close()

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
