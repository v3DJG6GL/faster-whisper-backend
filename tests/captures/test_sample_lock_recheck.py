"""Ungroup and Regenerate re-check a non-admin's lock guard under the per-sid
rebuild lock: a sample locked (from another tab) while the request waited on
an in-flight VAD rebuild must stay intact."""

import contextlib

from tests.captures.test_routes_captures import _insert_sample
from tests.conftest import bearer


def _lock_while_waiting(monkeypatch, conn, sid):
    """Make acquiring the rebuild lock lock the sample first — the window in
    which the request waits out a VAD rebuild and the row changes."""
    from faster_whisper_backend.captures import samples as capture_samples

    real = capture_samples._rebuild_lock

    @contextlib.contextmanager
    def _locks_meanwhile(s):
        conn.execute("UPDATE capture_samples SET is_locked = 1 WHERE id = ?", (s,))
        with real(s):
            yield

    monkeypatch.setattr(capture_samples, "_rebuild_lock", _locks_meanwhile)


def _setup(client, make_user_key, monkeypatch, sid):
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    _lock_while_waiting(monkeypatch, conn, sid)
    return gs, bearer(raw)


def test_dissolve_refuses_a_sample_locked_while_waiting(
        client, make_user_key, monkeypatch):
    sid = "relock01sid"
    gs, h = _setup(client, make_user_key, monkeypatch, sid)
    r = client.delete(f"/captures/api/samples/{sid}", headers=h)
    assert r.status_code == 409
    assert gs.get_sample(sid) is not None


def test_regenerate_refuses_a_sample_locked_while_waiting(
        client, make_user_key, monkeypatch):
    from faster_whisper_backend.captures import samples as capture_samples

    sid = "relock02sid"
    gs, h = _setup(client, make_user_key, monkeypatch, sid)
    built = []
    monkeypatch.setattr(capture_samples, "_build_merged_wav",
                        lambda **kw: built.append(kw) or (1000, {}, {}))
    r = client.post(f"/captures/api/samples/{sid}/regenerate", headers=h)
    assert r.status_code == 409
    assert built == []
    assert gs.get_sample(sid)["merged_duration_ms"] == 5000


def test_dissolve_sample_by_default_still_deletes_a_locked_sample(
        captures_store_db, groups_store_db):
    gs = groups_store_db
    _insert_sample(captures_store_db._require_conn(), gs, "relock03sid", locked=True)
    gs.dissolve_sample("relock03sid")
    assert gs.get_sample("relock03sid") is None
