"""A failed forced refresh after a sibling logout must not mark the
'revocations' counter as seen."""

import pytest

from tests.auth._store_helpers import (
    count_rebuilds, failing_conn, failing_rebuild_conn, load_sessions_store_copy,
    sibling_write)


@pytest.fixture
def worker(tmp_path):
    """An independent copy of sessions_store on its own DB file, like one
    uvicorn worker with its own connection and in-memory index."""
    mod = load_sessions_store_copy("sessions_store_revocation_retry", str(tmp_path / "sessions.db"))
    yield mod
    mod._conn.close()
    mod._conn = None


def _sibling_logout(w):
    """Revoke every session and bump the counter from a second connection,
    as revoke_session does on a sibling worker."""
    sibling_write(w, ("UPDATE sessions SET revoked_ts = 1",),
                  ("INSERT INTO meta (k, v) VALUES ('revocations', 1)"
                   " ON CONFLICT(k) DO UPDATE SET v = v + 1",))


def _fail_rebuild(w, monkeypatch):
    monkeypatch.setattr(w, "_require_conn",
                        failing_rebuild_conn(w, "SELECT token_hash"))


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


def test_failed_data_version_read_still_rebuilds_on_a_moved_counter(
        worker, monkeypatch):
    """A failing `PRAGMA data_version` answers with the cached value, so the
    refresh gated on it saw no sibling commit, skipped the rebuild, and the
    counter was recorded as seen: the re-auth then accepted the signed-out
    cookie from the stale index and nothing bumped again."""
    w = worker
    raw, _csrf = w.create_session("victim", 3600.0)
    assert w.revocation_generation() == 0
    assert w.lookup_session(raw) is not None
    _sibling_logout(w)
    real = w._require_conn()
    monkeypatch.setattr(w, "_require_conn",
                        lambda: failing_conn(real, "PRAGMA data_version", "disk I/O error"))
    assert w.revocation_generation() == 1
    monkeypatch.undo()
    assert w.lookup_session(raw) is None


def test_own_logout_skips_the_forced_rebuild(worker, monkeypatch):
    """The worker that served a logout already popped the token, yet its next
    probe saw the counter it bumped itself and redid the full O(live
    sessions) rebuild on the event loop for every local logout. A sibling's
    revocation interleaved before ours must still force that rebuild."""
    w = worker
    mine, _ = w.create_session("me", 3600.0)
    victim, _ = w.create_session("victim", 3600.0)
    other, _ = w.create_session("other", 3600.0)
    assert w.revocation_generation() == 0
    rebuilds = count_rebuilds(w)
    w.revoke_session(mine)
    assert w.revocation_generation() == 1
    assert rebuilds == []
    assert w.lookup_session(mine) is None
    # A sibling revokes `victim` (gap of two once ours lands on top).
    sibling_write(w,
                  ("UPDATE sessions SET revoked_ts = 1 WHERE user_id = 'victim'",),
                  ("UPDATE meta SET v = v + 1 WHERE k = 'revocations'",))
    monkeypatch.setattr(w, "_data_version_locked", lambda: w._DATA_VERSION)
    w.revoke_session(other)
    assert rebuilds == []
    assert w.revocation_generation() == 3
    assert rebuilds == [1]
    assert w.lookup_session(victim) is None
