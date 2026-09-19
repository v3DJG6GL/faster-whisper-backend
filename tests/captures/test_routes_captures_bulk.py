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
    from faster_whisper_backend.captures import store as cs

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, _ = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    conn = cs._require_conn()
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
    from faster_whisper_backend.captures import store as cs

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    conn = cs._require_conn()
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
    from faster_whisper_backend.captures import store as cs

    _root, raw_root = make_user_key("root", is_admin=True)
    conn = cs._require_conn()
    for i in range(10):
        uid, _ = make_user_key(f"u{i}", pages={"captures": "own"})
        _row(conn, f"c{i:011d}", user_id=uid, audio_s=float(10 - i))
    st = client.get("/captures/api/stats", headers=bearer(raw_root)).json()
    assert len(st["by_user"]) == 9
    assert st["by_user"][-1] == {"user_id": None, "username": "others",
                                 "n": 2, "s": 3.0}                # 2 + 1


# ---------------------------------------------------------------------------
# list: comma-separated speakers
# ---------------------------------------------------------------------------

def test_list_user_id_accepts_several_speakers(client, make_user_key):
    from faster_whisper_backend.captures import store as cs

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    uid_c, _ = make_user_key("carla", pages={"captures": "own"})
    conn = cs._require_conn()
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


# ---------------------------------------------------------------------------
# bulk status
# ---------------------------------------------------------------------------

def test_bulk_status_updates_and_reports_prev(client, make_user_key):
    from faster_whisper_backend.captures import store as cs

    _root, raw_root = make_user_key("root", is_admin=True)
    uid_a, _ = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
    _row(conn, "a1a1a1a1a1a1", user_id=uid_a)
    _row(conn, "a2a2a2a2a2a2", user_id=uid_a, status="reviewed",
         reviewed_ts=123.0)

    r = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                     json={"ids": ["a1a1a1a1a1a1", "a2a2a2a2a2a2",
                                   "a1a1a1a1a1a1"], "status": "ready"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready" and body["skipped"] == []
    assert body["updated"] == [                      # de-duplicated, in order
        {"id": "a1a1a1a1a1a1", "prev_status": "new"},
        {"id": "a2a2a2a2a2a2", "prev_status": "reviewed"}]
    for cid in ("a1a1a1a1a1a1", "a2a2a2a2a2a2"):
        row = cs.get_capture(cid)
        assert row["status"] == "ready" and row["reviewed_ts"] > 1000.0
    # back to new NULLs reviewed_ts (the undo path)
    client.patch("/captures/api/bulk", headers=bearer(raw_root),
                 json={"ids": ["a1a1a1a1a1a1"], "status": "new"})
    assert cs.get_capture("a1a1a1a1a1a1")["reviewed_ts"] is None


def test_bulk_status_skips_locked_member_for_nonadmin_only(client, make_user_key):
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as cs

    _root, raw_root = make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
    _insert_sample(conn, gs, "locked01sid", locked=True, user_id=uid)
    _insert_member(conn, "locked01cid", "locked01sid", user_id=uid)
    _row(conn, "free0001cid0", user_id=uid)

    body = client.patch("/captures/api/bulk", headers=bearer(raw),
                        json={"ids": ["locked01cid", "free0001cid0"],
                              "status": "reviewed"}).json()
    assert body["skipped"] == [{"id": "locked01cid", "reason": "locked"}]
    assert [u["id"] for u in body["updated"]] == ["free0001cid0"]
    assert cs.get_capture("locked01cid")["status"] == "new"
    # admins are exempt from the lock, as on the single-id route
    body = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                        json={"ids": ["locked01cid"], "status": "reviewed"}).json()
    assert body["skipped"] == [] and cs.get_capture("locked01cid")["status"] == "reviewed"


def test_bulk_status_foreign_and_missing_are_both_not_found(client, make_user_key):
    from faster_whisper_backend.captures import store as cs

    make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _ = make_user_key("bob", pages={"captures": "own"})
    conn = cs._require_conn()
    _row(conn, "b1b1b1b1b1b1", user_id=uid_b)

    body = client.patch("/captures/api/bulk", headers=bearer(raw_a),
                        json={"ids": ["b1b1b1b1b1b1", "nope00000000"],
                              "status": "ready"}).json()
    assert body["updated"] == []
    assert body["skipped"] == [{"id": "b1b1b1b1b1b1", "reason": "not_found"},
                               {"id": "nope00000000", "reason": "not_found"}]
    assert cs.get_capture("b1b1b1b1b1b1")["status"] == "new"


def test_bulk_status_skips_audio_missing_rows(client, make_user_key):
    from faster_whisper_backend.captures import store as cs

    _root, raw_root = make_user_key("root", is_admin=True)
    uid, _ = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
    _row(conn, "gone00000001", user_id=uid, status="audio_missing")
    body = client.patch("/captures/api/bulk", headers=bearer(raw_root),
                        json={"ids": ["gone00000001"], "status": "ready"}).json()
    assert body["skipped"] == [{"id": "gone00000001", "reason": "audio_missing"}]
    assert cs.get_capture("gone00000001")["status"] == "audio_missing"


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
    from faster_whisper_backend.captures import store as cs

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
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
    assert cs.get_capture("free0001cid0") is None
    assert cs.get_capture("locked01cid") is not None
    assert cs.count(user_id=uid) == 1
