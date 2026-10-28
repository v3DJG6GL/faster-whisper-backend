"""A failed forced refresh after a sibling logout must not mark the
'revocations' counter as seen."""

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
        "sessions_store_revocation_retry",
        os.path.join(REPO_ROOT, "faster_whisper_backend", "auth", "sessions_store.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init_db(str(tmp_path / "sessions.db"))
    yield mod
    mod._conn.close()
    mod._conn = None


def _sibling_logout(w):
    """Revoke every session and bump the counter from a second connection,
    as revoke_session does on a sibling worker."""
    db_path = w._require_conn().execute(
        "PRAGMA database_list").fetchone()["file"]
    sibling = sqlite3.connect(db_path)
    try:
        with sibling:
            sibling.execute("UPDATE sessions SET revoked_ts = 1")
            sibling.execute(
                "INSERT INTO meta (k, v) VALUES ('revocations', 1)"
                " ON CONFLICT(k) DO UPDATE SET v = v + 1")
    finally:
        sibling.close()


def _fail_rebuild(w, monkeypatch):
    real = w._require_conn()

    class _FailingRebuild:
        def execute(self, sql, *args):
            if sql.startswith("SELECT token_hash"):
                raise sqlite3.OperationalError("database is locked")
            return real.execute(sql, *args)

    monkeypatch.setattr(w, "_require_conn", _FailingRebuild)


def test_forced_refresh_is_retried_after_a_failed_rebuild(worker, monkeypatch):
    """The counter used to be recorded before the refresh, so after one failed
    rebuild the next probe returned the new count with no refresh, and the
    re-auth it triggered hit the stale index inside the 1 s throttle."""
    w = worker
    raw, _csrf = w.create_session("victim", 3600.0)
    assert w.revocation_generation() == 0
    assert w.lookup_session(raw) is not None
    _sibling_logout(w)
    _fail_rebuild(w, monkeypatch)
    assert w.revocation_generation() == -1
    monkeypatch.undo()
    assert w.revocation_generation() == 1
    assert w.lookup_session(raw) is None


def test_config_version_bumps_only_once_the_index_is_fresh(worker, monkeypatch):
    """A failed refresh must neither escape config_version() nor be adopted:
    the bump waits for the probe whose refresh succeeded, and the re-auth that
    bump triggers rejects the signed-out cookie."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.auth import sessions_store
    from faster_whisper_backend.settings import version as settings_version
    w = worker
    monkeypatch.setattr(sessions_store, "revocation_generation",
                        w.revocation_generation)
    monkeypatch.setattr(api_keys_store, "data_version", lambda: 5)
    monkeypatch.setattr(settings_version, "_KEYS_DATA_VERSION", -1)
    monkeypatch.setattr(settings_version, "_SESSIONS_REV_GEN", -1)
    monkeypatch.setattr(settings_version, "_KEYS_PROBE_MIN_INTERVAL_S", 0.0)
    raw, _csrf = w.create_session("victim", 3600.0)
    v0 = settings_version.config_version()          # adopts both samples
    assert w.lookup_session(raw) is not None
    _sibling_logout(w)
    require_conn = w._require_conn
    _fail_rebuild(w, monkeypatch)
    assert settings_version.config_version() == v0
    monkeypatch.setattr(w, "_require_conn", require_conn)
    assert settings_version.config_version() == v0 + 1
    assert w.lookup_session(raw) is None
