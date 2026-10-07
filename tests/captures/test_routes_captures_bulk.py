"""/captures/api/stats, the comma-separated speaker filter on the list, and
the bulk status / delete endpoints (the summary strip + bulk bar backend)."""

import time

from tests.captures.test_routes_captures import _insert_member, _insert_sample
from tests.conftest import bearer


def _row(conn, cid, *, user_id, status="new", audio_s=2.0, created_ts=1.0,
         reviewed_ts=None):
    _insert_member(conn, cid, None, user_id=user_id)
    conn.execute(
        "UPDATE captures SET status = ?, audio_s = ?, created_ts = ?,"
        " reviewed_ts = ? WHERE id = ?",
        (status, audio_s, created_ts, reviewed_ts, cid),
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def test_stats_empty_queue(client):
    st = client.get("/captures/api/stats").json()
    assert st["total"] == {"n": 0, "s": 0.0, "median_s": None}
    assert st["by_status"]["new"] == {"n": 0, "s": 0.0}
    assert st["review"] == {"handled_n": 0, "oldest_new_ts": None}
    assert st["ready"] == {"n": 0, "s": 0.0, "week_n": 0, "week_s": 0.0}
    assert st["by_user"] == []


def test_stats_arithmetic_median_ready_week_oldest_new(client, make_user_key):
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, _ = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    conn = captures_store._require_conn()
    now = time.time()
    _row(conn, "a1a1a1a1a1a1", user_id=uid_a, audio_s=1.0, created_ts=100.0)
    _row(conn, "a2a2a2a2a2a2", user_id=uid_a, audio_s=2.0, created_ts=50.0)
    _row(conn, "b1b1b1b1b1b1", user_id=uid_b, status="ready", audio_s=10.0,
         created_ts=200.0, reviewed_ts=now - 3600)          # ready this week
    _row(conn, "b2b2b2b2b2b2", user_id=uid_b, status="ready", audio_s=4.0,
         created_ts=300.0, reviewed_ts=now - 8 * 86400)     # ready, but old
    _row(conn, "b3b3b3b3b3b3", user_id=uid_b, status="dismissed", audio_s=3.0,
         created_ts=400.0, reviewed_ts=now)

    st = client.get("/captures/api/stats", headers=bearer(raw_root)).json()
    assert st["total"]["n"] == 5 and st["total"]["s"] == 20.0
    assert st["total"]["median_s"] == 3.0                    # 1,2,3,4,10
    assert st["by_status"]["new"] == {"n": 2, "s": 3.0}
    assert st["by_status"]["ready"] == {"n": 2, "s": 14.0}
    assert st["review"]["handled_n"] == 3                     # ready×2 + dismissed
    assert st["review"]["oldest_new_ts"] == 50.0
    assert st["ready"] == {"n": 2, "s": 14.0, "week_n": 1, "week_s": 10.0}
    # by_user: ranked by seconds desc, usernames resolved
    assert [(u["username"], u["n"], u["s"]) for u in st["by_user"]] == [
        ("bob", 3, 17.0), ("alice", 2, 3.0)]
    assert st["is_admin"] is True


def test_stats_scoped_to_caller_and_admin_only_override(client, make_user_key):
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _row(conn, "a1a1a1a1a1a1", user_id=uid_a, audio_s=1.0)
    _row(conn, "b1b1b1b1b1b1", user_id=uid_b, audio_s=5.0)
    _row(conn, "b2b2b2b2b2b2", user_id=uid_b, audio_s=5.0)

    # own-scope alice: her rows only, even when she asks for bob
    st = client.get("/captures/api/stats?user_id=" + uid_b,
                    headers=bearer(raw_a)).json()
    assert st["total"]["n"] == 1
    assert [u["user_id"] for u in st["by_user"]] == [uid_a]
    # admin: everything, and the override narrows
    st = client.get("/captures/api/stats", headers=bearer(raw_root)).json()
    assert st["total"]["n"] == 3
    st = client.get("/captures/api/stats?user_id=" + uid_b,
                    headers=bearer(raw_root)).json()
    assert st["total"] == {"n": 2, "s": 10.0, "median_s": 5.0}


def test_stats_folds_speakers_past_eight_into_others(client, make_user_key):
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    conn = captures_store._require_conn()
    for i in range(10):
        uid, _ = make_user_key(f"u{i}", pages={"captures": "own"})
        _row(conn, f"c{i:011d}", user_id=uid, audio_s=float(10 - i))
    st = client.get("/captures/api/stats", headers=bearer(raw_root)).json()
    assert len(st["by_user"]) == 9
    assert st["by_user"][-1] == {"user_id": None, "username": "others",
                                 "n": 2, "s": 3.0}                # 2 + 1
    # The fold is for the strip only: the speaker picker reads by_user_all,
    # which keeps every speaker pickable (incl. the two folded above).
    assert len(st["by_user_all"]) == 10
    assert all(u["user_id"] for u in st["by_user_all"])
    assert [u["username"] for u in st["by_user_all"]] == [
        f"u{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# list: comma-separated speakers
# ---------------------------------------------------------------------------

def test_list_user_id_accepts_several_speakers(client, make_user_key):
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    uid_c, _ = make_user_key("carla", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _row(conn, "a1a1a1a1a1a1", user_id=uid_a)
    _row(conn, "b1b1b1b1b1b1", user_id=uid_b)
    _row(conn, "c1c1c1c1c1c1", user_id=uid_c, status="ready")

    body = client.get(f"/captures/api/list?user_id={uid_a},{uid_b}",
                      headers=bearer(raw_root)).json()
    assert sorted(r["id"] for r in body["captures"]) == [
        "a1a1a1a1a1a1", "b1b1b1b1b1b1"]
    assert body["counts"]["new"] == 2 and body["counts"]["ready"] == 0
    assert body["total_count"] == 2
    # a non-admin's speaker list is ignored: still pinned to herself
    body = client.get(f"/captures/api/list?user_id={uid_a},{uid_b}",
                      headers=bearer(raw_a)).json()
    assert [r["id"] for r in body["captures"]] == ["a1a1a1a1a1a1"]


def test_list_user_id_rejects_an_oversized_speaker_list(client, make_user_key):
    """Every id is one SQL bind variable in four queries; an uncapped list
    ends in sqlite's "too many SQL variables" (a 500), so it is a 422."""
    from faster_whisper_backend.captures import routes as captures_routes

    _root, raw_root = make_user_key("root", is_admin=True)
    cap = captures_routes._MAX_OWNER_FILTER_IDS
    ok = ",".join(f"u{i}" for i in range(cap))
    assert client.get("/captures/api/list?user_id=" + ok,
                      headers=bearer(raw_root)).status_code == 200
    for path in ("/captures/api/list", "/captures/api/stats",
                 "/captures/api/samples"):
        r = client.get(f"{path}?user_id={ok},one-too-many",
                       headers=bearer(raw_root))
        assert r.status_code == 422, path


def test_samples_user_id_accepts_several_speakers(client, make_user_key):
    """/captures/api/samples scopes like the list: the comma-separated form
    used to be bound as one `user_id = ?` string and matched nothing."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    uid_c, _ = make_user_key("carla", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, "sampleaaaaa1", locked=False, user_id=uid_a)
    _insert_sample(conn, gs, "samplebbbbb1", locked=False, user_id=uid_b)
    _insert_sample(conn, gs, "sampleccccc1", locked=False, user_id=uid_c)

    def ids(q, raw):
        body = client.get("/captures/api/samples" + q, headers=bearer(raw)).json()
        return sorted(g["id"] for g in body["samples"])

    assert ids(f"?user_id={uid_a},{uid_b}", raw_root) == [
        "sampleaaaaa1", "samplebbbbb1"]
    assert ids(f"?user_id={uid_c}", raw_root) == ["sampleccccc1"]   # single id
    assert len(ids("", raw_root)) == 3
    # a non-admin's speaker list is ignored: still pinned to herself
    assert ids(f"?user_id={uid_b},{uid_c}", raw_a) == ["sampleaaaaa1"]


# ---------------------------------------------------------------------------
# bulk status
# ---------------------------------------------------------------------------

def test_bulk_status_updates_and_reports_prev(client, make_user_key):
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, _ = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _row(conn, "a1a1a1a1a1a1", user_id=uid_a)
    _row(conn, "a2a2a2a2a2a2", user_id=uid_a, status="reviewed",
         reviewed_ts=123.0)

    r = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                     json={"ids": ["a2a2a2a2a2a2", "a1a1a1a1a1a1",
                                   "a2a2a2a2a2a2"], "status": "ready"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready" and body["skipped"] == []
    assert body["updated"] == [          # de-duplicated, in request order
        {"id": "a2a2a2a2a2a2", "prev_status": "reviewed"},
        {"id": "a1a1a1a1a1a1", "prev_status": "new"}]
    for cid in ("a1a1a1a1a1a1", "a2a2a2a2a2a2"):
        row = captures_store.get_capture(cid)
        assert row["status"] == "ready" and row["reviewed_ts"] > 1000.0
    # back to new NULLs reviewed_ts (the undo path)
    client.patch("/captures/api/bulk", headers=bearer(raw_root),
                 json={"ids": ["a1a1a1a1a1a1"], "status": "new"})
    assert captures_store.get_capture("a1a1a1a1a1a1")["reviewed_ts"] is None


def test_bulk_status_keeps_reviewed_ts_of_rows_already_at_the_target(
        client, make_user_key):
    """A mixed selection marked ready must not re-stamp a row that has been
    ready for months — stats() would count it as "ready this week"."""
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, _ = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    old_ts = time.time() - 90 * 86400
    _row(conn, "a1a1a1a1a1a1", user_id=uid_a)
    _row(conn, "a2a2a2a2a2a2", user_id=uid_a, status="ready",
         reviewed_ts=old_ts)

    r = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                     json={"ids": ["a1a1a1a1a1a1", "a2a2a2a2a2a2"],
                           "status": "ready"})
    assert r.status_code == 200
    assert r.json()["updated"] == [
        {"id": "a1a1a1a1a1a1", "prev_status": "new"},
        {"id": "a2a2a2a2a2a2", "prev_status": "ready"}]
    assert captures_store.get_capture("a2a2a2a2a2a2")["reviewed_ts"] == old_ts
    assert captures_store.get_capture("a1a1a1a1a1a1")["reviewed_ts"] > old_ts
    st = client.get("/captures/api/stats", headers=bearer(raw_root)).json()
    assert st["ready"]["n"] == 2 and st["ready"]["week_n"] == 1


def test_single_patch_keeps_reviewed_ts_when_status_is_unchanged(
        client, make_user_key):
    """The card's Save used to resend the current status with every chip or
    note edit; update_capture re-stamped reviewed_ts, so a note on a
    months-old ready capture moved it into "ready this week"."""
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, _ = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    old_ts = time.time() - 8 * 86400
    _row(conn, "a3a3a3a3a3a3", user_id=uid_a, status="ready",
         reviewed_ts=old_ts)

    r = client.patch("/captures/api/a3a3a3a3a3a3", headers=bearer(raw_root),
                     json={"status": "ready", "admin_notes": "checked"})
    assert r.status_code == 200
    row = captures_store.get_capture("a3a3a3a3a3a3")
    assert row["admin_notes"] == "checked" and row["reviewed_ts"] == old_ts
    st = client.get("/captures/api/stats", headers=bearer(raw_root)).json()
    assert st["ready"]["week_n"] == 0
    # A real transition still stamps.
    client.patch("/captures/api/a3a3a3a3a3a3", headers=bearer(raw_root),
                 json={"status": "reviewed"})
    assert captures_store.get_capture("a3a3a3a3a3a3")["reviewed_ts"] > old_ts


def test_single_patch_on_audio_missing_row_saves_edits_but_not_status(
        client, make_user_key):
    """audio_missing is system-set: chips/notes still save, but a status
    PATCH is refused like the bulk route's audio_missing skip."""
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid, _ = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _row(conn, "gone00000002", user_id=uid, status="audio_missing")

    r = client.patch("/captures/api/gone00000002", headers=bearer(raw_root),
                     json={"corrections": [], "admin_notes": "file lost"})
    assert r.status_code == 200
    row = captures_store.get_capture("gone00000002")
    assert row["status"] == "audio_missing" and row["admin_notes"] == "file lost"

    r = client.patch("/captures/api/gone00000002", headers=bearer(raw_root),
                     json={"status": "ready"})
    assert r.status_code == 409
    assert captures_store.get_capture("gone00000002")["status"] == "audio_missing"


def _restore_audio(captures_store, cid):
    import os
    p = captures_store.abs_audio_path(
        captures_store.get_capture(cid)["audio_relpath"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"RIFF")


def test_audio_missing_row_with_restored_file_can_be_retriaged(
        client, make_user_key):
    """audio_missing is only set at boot and never cleared: once the WAV is
    back (late mount, restore) both the single and the bulk status PATCH
    accept the row again instead of locking it out of the export."""
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid, _ = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _row(conn, "back00000001", user_id=uid, status="audio_missing")
    _row(conn, "back00000002", user_id=uid, status="audio_missing")
    _restore_audio(captures_store, "back00000001")
    _restore_audio(captures_store, "back00000002")

    r = client.patch("/captures/api/back00000001", headers=bearer(raw_root),
                     json={"status": "reviewed"})
    assert r.status_code == 200, r.text
    assert captures_store.get_capture("back00000001")["status"] == "reviewed"

    body = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                        json={"ids": ["back00000002"], "status": "ready"}).json()
    assert body["skipped"] == []
    assert body["updated"] == [{"id": "back00000002",
                                "prev_status": "audio_missing"}]
    assert captures_store.get_capture("back00000002")["status"] == "ready"


def test_bulk_status_skips_locked_member_for_nonadmin_only(client, make_user_key):
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, "locked01sid", locked=True, user_id=uid)
    _insert_member(conn, "locked01cid", "locked01sid", user_id=uid)
    _row(conn, "free0001cid0", user_id=uid)

    body = client.patch("/captures/api/bulk", headers=bearer(raw),
                        json={"ids": ["locked01cid", "free0001cid0"],
                              "status": "reviewed"}).json()
    assert body["skipped"] == [{"id": "locked01cid", "reason": "locked"}]
    assert [u["id"] for u in body["updated"]] == ["free0001cid0"]
    assert captures_store.get_capture("locked01cid")["status"] == "new"
    # admins are exempt from the lock, as on the single-id route
    body = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                        json={"ids": ["locked01cid"], "status": "reviewed"}).json()
    assert body["skipped"] == [] and captures_store.get_capture("locked01cid")["status"] == "reviewed"


def test_bulk_status_foreign_and_missing_are_both_not_found(client, make_user_key):
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _row(conn, "b1b1b1b1b1b1", user_id=uid_b)

    body = client.patch("/captures/api/bulk", headers=bearer(raw_a),
                        json={"ids": ["b1b1b1b1b1b1", "nope00000000"],
                              "status": "ready"}).json()
    assert body["updated"] == []
    assert body["skipped"] == [{"id": "b1b1b1b1b1b1", "reason": "not_found"},
                               {"id": "nope00000000", "reason": "not_found"}]
    assert captures_store.get_capture("b1b1b1b1b1b1")["status"] == "new"


def test_bulk_status_skips_audio_missing_rows(client, make_user_key):
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid, _ = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _row(conn, "gone00000001", user_id=uid, status="audio_missing")
    body = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                        json={"ids": ["gone00000001"], "status": "ready"}).json()
    assert body["skipped"] == [{"id": "gone00000001", "reason": "audio_missing"}]
    assert captures_store.get_capture("gone00000001")["status"] == "audio_missing"


def test_bulk_status_validation(client):
    bad = [
        {"ids": [], "status": "ready"},
        {"ids": ["x" * 12] * 1001, "status": "ready"},
        {"ids": ["x" * 12], "status": "audio_missing"},
        {"ids": ["x" * 12], "status": "ready", "extra": 1},
        {"status": "ready"},
    ]
    for payload in bad:
        assert client.patch("/captures/api/bulk", json=payload).status_code == 422, payload
    # the literal path must not be swallowed by /captures/api/{cid}
    assert client.patch("/captures/api/bulk", json={}).status_code == 422
    assert client.post("/captures/api/bulk-delete", json={}).status_code == 422


def test_bulk_routes_require_captures_page_when_locked(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("nobody", pages={"captures": "none"})
    h = bearer(raw)
    assert client.patch("/captures/api/bulk", headers=h,
                        json={"ids": ["x" * 12], "status": "ready"}).status_code == 403
    assert client.post("/captures/api/bulk-delete", headers=h,
                       json={"ids": ["x" * 12]}).status_code == 403
    assert client.get("/captures/api/stats", headers=h).status_code == 403


# ---------------------------------------------------------------------------
# bulk delete
# ---------------------------------------------------------------------------

def test_bulk_delete_removes_rows_and_skips_locked(client, make_user_key):
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, "locked01sid", locked=True, user_id=uid)
    _insert_member(conn, "locked01cid", "locked01sid", user_id=uid)
    _row(conn, "free0001cid0", user_id=uid)
    _row(conn, "free0002cid0", user_id=uid)

    body = client.post("/captures/api/bulk-delete", headers=bearer(raw),
                       json={"ids": ["free0001cid0", "locked01cid",
                                     "free0002cid0", "nope00000000"]}).json()
    assert body["deleted"] == ["free0001cid0", "free0002cid0"]
    assert body["skipped"] == [{"id": "locked01cid", "reason": "locked"},
                               {"id": "nope00000000", "reason": "not_found"}]
    assert captures_store.get_capture("free0001cid0") is None
    assert captures_store.get_capture("locked01cid") is not None
    assert captures_store.count(user_id=uid) == 1


def test_bulk_guard_never_loads_the_full_capture_row(client, make_user_key,
                                                     monkeypatch):
    """Admission reads id / user_id / status / sample_id only. get_capture
    json.loads the words + segments blobs — per id, up to 1000 times."""
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, _ = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    ids = [f"bulk{i:08d}" for i in range(50)]
    for cid in ids:
        _row(conn, cid, user_id=uid_a)

    def _boom(cid):
        raise AssertionError("bulk guard must not call get_capture")
    # Scoped: a bare monkeypatch.undo() would also revert every fixture patch
    # (the *_DB paths, config overrides, the model loader) mid-test.
    with monkeypatch.context() as m:
        m.setattr(captures_store, "get_capture", _boom)
        body = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                            json={"ids": ids, "status": "reviewed"}).json()
    assert [u["id"] for u in body["updated"]] == ids
    assert body["skipped"] == []
    assert captures_store.get_capture(ids[-1])["status"] == "reviewed"


def test_bulk_guard_looks_a_shared_sample_lock_up_once(client, make_user_key,
                                                       monkeypatch):
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, "locked01sid", locked=True, user_id=uid)
    for i in range(3):
        _insert_member(conn, f"locked0{i}cid", "locked01sid", user_id=uid)
    _row(conn, "free0001cid0", user_id=uid)

    calls: list[str] = []
    real = gs.get_sample
    monkeypatch.setattr(gs, "get_sample",
                        lambda sid: (calls.append(sid), real(sid))[1])
    body = client.patch(
        "/captures/api/bulk", headers=bearer(raw),
        json={"ids": ["locked00cid", "locked01cid", "free0001cid0",
                      "locked02cid"], "status": "ready"}).json()
    assert [u["id"] for u in body["updated"]] == ["free0001cid0"]
    assert body["skipped"] == [
        {"id": c, "reason": "locked"}
        for c in ("locked00cid", "locked01cid", "locked02cid")]
    assert calls == ["locked01sid"]


def test_bulk_status_reports_a_row_deleted_after_the_guard(client, make_user_key,
                                                           monkeypatch):
    """updated + skipped must account for every submitted id: a row that
    vanishes between admission and the write is `not_found`, as in
    bulk-delete — not silently absent from both lists."""
    from faster_whisper_backend.captures import store as captures_store

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, _ = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _row(conn, "a1a1a1a1a1a1", user_id=uid_a)
    _row(conn, "a2a2a2a2a2a2", user_id=uid_a)

    real = captures_store.bulk_update_status

    def _racing(ids, new_status):
        conn.execute("DELETE FROM captures WHERE id = ?", ("a2a2a2a2a2a2",))
        return real(ids, new_status)
    monkeypatch.setattr(captures_store, "bulk_update_status", _racing)

    body = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                        json={"ids": ["a1a1a1a1a1a1", "a2a2a2a2a2a2"],
                              "status": "ready"}).json()
    assert body["updated"] == [{"id": "a1a1a1a1a1a1", "prev_status": "new"}]
    assert body["skipped"] == [{"id": "a2a2a2a2a2a2", "reason": "not_found"}]
