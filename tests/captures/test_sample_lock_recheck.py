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


class _LocksOnEnter:
    """Stand-in for routes._corrections_write_lock that locks the sample as
    the save acquires it — the window in which another save held the lock."""

    def __init__(self, conn, sid):
        import threading
        self._conn, self._sid, self._real = conn, sid, threading.Lock()

    def __enter__(self):
        self._conn.execute(
            "UPDATE capture_samples SET is_locked = 1 WHERE id = ?", (self._sid,))
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)


def test_group_chip_save_refuses_a_sample_locked_while_waiting(
        client, make_user_key, monkeypatch):
    from faster_whisper_backend.captures import routes
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    sid = "relock04sid"
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    monkeypatch.setattr(routes, "_corrections_write_lock", _LocksOnEnter(conn, sid))
    r = client.patch(f"/captures/api/samples/{sid}", headers=bearer(raw),
                     json={"corrections": [], "status": "reviewed"})
    assert r.status_code == 409
    assert gs.get_sample(sid)["status"] == "new"


def _member_locked_after_guard(client, make_user_key, monkeypatch, sid, cid):
    """A member of an unlocked sample whose sample gets locked right after the
    route's lock guard passed — the hop in which an admin locks it."""
    from faster_whisper_backend.captures import routes
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store
    from tests.captures.test_routes_captures import _insert_member

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    _insert_member(conn, cid, sid, user_id=uid)
    real = routes._assert_member_sample_not_locked

    def _guard_then_lock(row, user):
        real(row, user)
        conn.execute("UPDATE capture_samples SET is_locked = 1 WHERE id = ?", (sid,))

    monkeypatch.setattr(routes, "_assert_member_sample_not_locked", _guard_then_lock)
    return gs, captures_store, bearer(raw)


def test_member_delete_refuses_a_sample_locked_since_the_guard(
        client, make_user_key, monkeypatch):
    sid, cid = "relock05sid", "aa05member0001"
    gs, cs, h = _member_locked_after_guard(client, make_user_key, monkeypatch, sid, cid)
    r = client.delete(f"/captures/api/{cid}", headers=h)
    assert r.status_code == 409
    assert cs.get_capture(cid) is not None
    assert gs.get_sample(sid) is not None


def test_bulk_delete_skips_a_member_whose_sample_was_locked_since_the_guard(
        client, make_user_key, monkeypatch):
    sid, cid = "relock06sid", "aa06member0001"
    gs, cs, h = _member_locked_after_guard(client, make_user_key, monkeypatch, sid, cid)
    r = client.post("/captures/api/bulk-delete", headers=h, json={"ids": [cid]})
    assert r.status_code == 200
    assert r.json()["deleted"] == []
    assert r.json()["skipped"] == [{"id": cid, "reason": "locked"}]
    assert cs.get_capture(cid) is not None
    assert gs.get_sample(sid) is not None


def test_lock_toggle_waits_out_an_in_flight_rebuild(
        client, make_user_key, monkeypatch):
    """The is_locked write runs under the per-sid rebuild lock, so the
    rebuilders' re-check under that lock is authoritative: a sample cannot be
    locked between a VAD rebuild's check and its update_sample."""
    from faster_whisper_backend.captures import samples as capture_samples
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    sid = "relock07sid"
    _insert_sample(captures_store._require_conn(), gs, sid, locked=False, user_id=uid)
    held = []
    real_lock, real_update = capture_samples._rebuild_lock, gs.update_sample

    @contextlib.contextmanager
    def _tracking(s):
        with real_lock(s):
            held.append(s)
            try:
                yield
            finally:
                held.remove(s)

    seen = []
    monkeypatch.setattr(capture_samples, "_rebuild_lock", _tracking)
    monkeypatch.setattr(gs, "update_sample",
                        lambda s, patch: seen.append(list(held)) or real_update(s, patch))
    r = client.patch(f"/captures/api/samples/{sid}", headers=bearer(raw),
                     json={"is_locked": True})
    assert r.status_code == 200
    assert seen == [[sid]]
    assert gs.get_sample(sid)["is_locked"]
