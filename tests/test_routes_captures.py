"""Integration tests for /captures routes.

Captures rows require real audio transcode (ffmpeg) to create, so these
tests focus on the read/list/route-ordering/auth surface that works without
fabricating audio blobs.
"""

import json
import os

import pytest
from starlette.testclient import TestClient

from conftest import bearer


def test_captures_page(client):
    r = client.get("/captures")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_captures_list_open_mode(client):
    r = client.get("/captures/api/list")
    assert r.status_code == 200
    body = r.json()
    assert "captures" in body and "counts" in body
    assert "is_admin" in body


def test_reprocess_vad_job_lifecycle(client):
    # Status endpoint is registered and reports a known state.
    s0 = client.get("/captures/api/reprocess-vad/status")
    assert s0.status_code == 200
    assert s0.json()["status"] in ("idle", "running", "done", "error")
    # Start the bulk VAD re-merge on an empty store → runs and finishes clean.
    assert client.post("/captures/api/reprocess-vad").status_code == 200
    import time
    s = s0.json()
    for _ in range(30):
        s = client.get("/captures/api/reprocess-vad/status").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert s["status"] == "done"
    assert s["total"] == 0 and s["rebuilt"] == 0


def test_samples_route_not_swallowed_by_cid(client):
    # Regression: /captures/api/samples must resolve to the sample-list handler,
    # NOT the parameterized /captures/api/{cid} handler (which would 404 with
    # cid="samples"). A 200 with a "samples" key proves correct route ordering.
    r = client.get("/captures/api/samples")
    assert r.status_code == 200
    assert "samples" in r.json()


def test_export_route_not_swallowed_by_cid(client):
    # /captures/api/export is also a literal route declared before /{cid}.
    r = client.get("/captures/api/export")
    assert r.status_code == 200
    assert "application/gzip" in r.headers.get("content-type", "")


def test_unknown_cid_404(client):
    r = client.get("/captures/api/does-not-exist")
    assert r.status_code == 404


def test_propose_merges_ok(client):
    r = client.get("/captures/api/propose-merges")
    assert r.status_code == 200
    assert "proposals" in r.json()


def test_by_request_id_ok(client):
    r = client.get("/captures/api/by-request/unknown-req")
    assert r.status_code == 200
    assert r.json()["captures"] == []  # no captures for an unknown request id


def test_host_gate_rejects_non_loopback(app_module, monkeypatch):
    # /captures is user-tier (require_user_webui_host / USER_WEBUI_ALLOWED_HOSTS).
    # The list defaults OPEN, so narrow it to loopback to exercise the host gate:
    # a non-loopback host is then 403 before the page-permission check.
    import config as cfg
    monkeypatch.setattr(
        cfg, "USER_WEBUI_ALLOWED_HOSTS", ["127.0.0.1", "::1"], raising=False
    )
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        assert c.get("/captures/api/list").status_code == 403


def test_list_requires_page_when_locked(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"captures": "none"})
    r = client.get("/captures/api/list", headers=bearer(raw))
    assert r.status_code == 403


def test_clear_requires_admin_when_locked(client, make_user_key):
    # POST /captures/api/clear additionally Depends(require_admin).
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"captures": "own"})
    r = client.post("/captures/api/clear", headers=bearer(raw))
    assert r.status_code == 403


def test_merge_member_scope_guard_precedes_state_checks(
        captures_store_db, monkeypatch, tmp_path):
    """A scope=own caller probing ANOTHER user's capture id must get a uniform
    404 from the ownership guard — not a 400/410 that would leak the capture's
    existence + state. Regression guard for _validate_merge_payload: the
    per-member scope check must run BEFORE the already-in-sample / audio-missing
    checks."""
    import wave

    import audio_transcode
    import auth
    import captures_routes
    from fastapi import HTTPException

    cs = captures_store_db

    def _fake_transcode(src_path, dst_path):
        with wave.open(dst_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 100)
        return 1234

    monkeypatch.setattr(
        audio_transcode, "transcode_to_wav_16k_mono", _fake_transcode)

    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = cs.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language="de", duration_seconds=1.0, raw="r", final="f",
        words=[], segments=[], user_id="alice",
    )
    # Delete the audio so the OLD ordering would raise 410 ("audio is missing"),
    # leaking that the row exists; the fix must 404 for a non-owner first.
    os.unlink(cs.abs_audio_path(cs.get_capture(cid)["audio_relpath"]))

    # bob: scope=own captures user, NOT the owner and NOT admin → uniform 404.
    bob = {
        "user_id": "bob",
        "permissions": auth.Permissions(
            {"pages": {"captures": "own"}}, is_admin=False),
    }
    with pytest.raises(HTTPException) as ei:
        captures_routes._validate_merge_payload([cid], 0, bob)
    assert ei.value.status_code == 404

    # The OWNER still reaches the real state check (410), proving the guard
    # blocks only cross-user probes — not the owner's own legitimate errors.
    alice = {
        "user_id": "alice",
        "permissions": auth.Permissions(
            {"pages": {"captures": "own"}}, is_admin=False),
    }
    with pytest.raises(HTTPException) as ei2:
        captures_routes._validate_merge_payload([cid], 0, alice)
    assert ei2.value.status_code == 410


def _insert_sample(conn, gs, sid, *, locked, user_id="alice"):
    """Insert a capture_samples row directly (no audio merge needed)."""
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " transcript_join_strategy, member_hashes_json,"
        " inter_segment_silence_ms, is_stale, is_locked, status,"
        " admin_notes, language, member_trims_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, user_id, 1.0, gs._relpath_for(sid), 5000, "t", "space",
         "{}", 300, 0, 1 if locked else 0, "new", "", "de", "{}"),
    )


def _insert_member(conn, cid, sid, user_id="alice"):
    """Insert a captures row (optionally a member of sample `sid`) directly."""
    rel = os.path.join(cid[0:2], cid[2:4], f"{cid}.wav")
    conn.execute(
        "INSERT INTO captures (id, created_ts, request_id, model, language,"
        " duration_seconds, audio_relpath, audio_format, raw, final,"
        " words_json, segments_json, corrections_json, status, user_id,"
        " sample_id, sample_order)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, 1.0, None, "m", "de", 2.0, rel, "wav", "r", "f", "[]", "[]",
         "[]", "new", user_id, sid, 0),
    )


def test_member_delete_respects_sample_lock(captures_store_db, groups_store_db):
    """A non-admin cannot mutate/delete a capture that is a member of a LOCKED
    sample — deleting it would auto-dissolve (and destroy the merged WAV of) an
    admin-locked sample, bypassing the same guard dissolve_sample_api enforces.
    Regression guard for _assert_member_sample_not_locked."""
    import auth
    import captures_routes
    from fastapi import HTTPException

    cs = captures_store_db
    gs = groups_store_db
    conn = cs._require_conn()

    _insert_sample(conn, gs, "locked00sid", locked=True)
    _insert_member(conn, "locked00cid", "locked00sid")
    _insert_sample(conn, gs, "open000sid", locked=False)
    _insert_member(conn, "open000cid", "open000sid")
    _insert_member(conn, "free0000cid", None)  # no parent sample

    def _user(is_admin):
        return {
            "user_id": "alice",
            "is_admin": is_admin,
            "permissions": auth.Permissions(
                {"pages": {"captures": "own"}}, is_admin=is_admin),
        }

    locked_row = cs.get_capture("locked00cid")
    # Non-admin (even the owner) is refused on a locked sample's member.
    with pytest.raises(HTTPException) as ei:
        captures_routes._assert_member_sample_not_locked(locked_row, _user(False))
    assert ei.value.status_code == 409
    # Admin passes through.
    captures_routes._assert_member_sample_not_locked(locked_row, _user(True))
    # A member of an UNLOCKED sample, and a member of NO sample, pass through.
    captures_routes._assert_member_sample_not_locked(
        cs.get_capture("open000cid"), _user(False))
    captures_routes._assert_member_sample_not_locked(
        cs.get_capture("free0000cid"), _user(False))


def _fake_wav_transcode(monkeypatch):
    """Route audio_transcode to a stub that writes a tiny valid WAV."""
    import wave

    import audio_transcode

    def _fake(src_path, dst_path):
        with wave.open(dst_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 100)
        return 1234

    monkeypatch.setattr(
        audio_transcode, "transcode_to_wav_16k_mono", _fake)


def test_merge_member_404_body_uniform_missing_vs_foreign(
        captures_store_db, monkeypatch, tmp_path):
    """Missing id and another-user's id must yield the SAME 404 template on
    the merge surface. The guard's uniform 404 STATUS is pointless if the
    body still says "capture X not found" for missing ids but "not found"
    for foreign ones — the body becomes the existence oracle."""
    import auth
    import captures_routes
    from fastapi import HTTPException

    cs = captures_store_db
    _fake_wav_transcode(monkeypatch)
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = cs.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language="de", duration_seconds=1.0, raw="r", final="f",
        words=[], segments=[], user_id="alice",
    )

    bob = {
        "user_id": "bob",
        "permissions": auth.Permissions(
            {"pages": {"captures": "own"}}, is_admin=False),
    }
    with pytest.raises(HTTPException) as e_missing:
        captures_routes._validate_merge_payload(["nosuchcid000"], 0, bob)
    with pytest.raises(HTTPException) as e_foreign:
        captures_routes._validate_merge_payload([cid], 0, bob)
    assert e_missing.value.status_code == e_foreign.value.status_code == 404
    # Same template once the probed id (which the caller sent) is masked out.
    assert (e_missing.value.detail.replace("nosuchcid000", "{id}")
            == e_foreign.value.detail.replace(cid, "{id}"))


def test_capture_404_body_uniform_missing_vs_foreign(
        client, make_user_key, monkeypatch, tmp_path):
    """GET /captures/api/{cid}: a scope=own caller gets byte-identical 404
    bodies for a nonexistent id and for another user's id."""
    import captures_store as cs

    make_user_key("root", is_admin=True)
    owner_uid, _raw_owner = make_user_key("alice", pages={"captures": "own"})
    _uid, raw_bob = make_user_key("bob", pages={"captures": "own"})

    _fake_wav_transcode(monkeypatch)
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = cs.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language="de", duration_seconds=1.0, raw="r", final="f",
        words=[], segments=[], user_id=owner_uid,
    )

    r_missing = client.get(
        "/captures/api/does-not-exist", headers=bearer(raw_bob))
    r_foreign = client.get(f"/captures/api/{cid}", headers=bearer(raw_bob))
    assert r_missing.status_code == r_foreign.status_code == 404
    assert r_missing.json() == r_foreign.json()


def test_locked_member_mutations_blocked_at_endpoints(client, make_user_key):
    """The sample lock must hold on the LIVE routes, not only in the helper:
    PATCH, DELETE and reprocess on a locked sample's member all 409 for the
    non-admin owner. (Removing the _assert_member_sample_not_locked call
    sites would pass the helper test but fail this one.)"""
    import capture_samples_store as gs
    import captures_store as cs

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
    _insert_sample(conn, gs, "locked01sid", locked=True, user_id=uid)
    _insert_member(conn, "locked01cid", "locked01sid", user_id=uid)

    h = bearer(raw)
    assert client.patch(
        "/captures/api/locked01cid", json={"status": "reviewed"}, headers=h,
    ).status_code == 409
    assert client.delete(
        "/captures/api/locked01cid", headers=h,
    ).status_code == 409
    assert client.post(
        "/captures/api/locked01cid/reprocess", headers=h,
    ).status_code == 409
    # Row untouched and still present.
    row = cs.get_capture("locked01cid")
    assert row is not None and row["status"] == "new"


def test_nonadmin_can_unlock_but_not_edit_a_locked_sample(client, make_user_key):
    """`is_locked` is writable by any captures-scoped caller, so it must also
    be RELEASABLE by them — otherwise it is a one-way switch only an admin can
    undo. A non-admin may send the unlock and nothing else while locked."""
    import capture_samples_store as gs
    import captures_store as cs

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
    _insert_sample(conn, gs, "unlock01sid", locked=True, user_id=uid)
    h = bearer(raw)

    # Any other edit stays frozen...
    assert client.patch(
        "/captures/api/samples/unlock01sid",
        json={"status": "reviewed"}, headers=h,
    ).status_code == 409
    # ...including an unlock smuggled alongside one.
    assert client.patch(
        "/captures/api/samples/unlock01sid",
        json={"is_locked": False, "status": "reviewed"}, headers=h,
    ).status_code == 409
    assert gs.get_sample("unlock01sid")["is_locked"] == 1

    # The bare unlock goes through.
    assert client.patch(
        "/captures/api/samples/unlock01sid",
        json={"is_locked": False}, headers=h,
    ).status_code == 200
    assert gs.get_sample("unlock01sid")["is_locked"] == 0

    # And once unlocked, ordinary edits work again.
    assert client.patch(
        "/captures/api/samples/unlock01sid",
        json={"status": "reviewed"}, headers=h,
    ).status_code == 200


def test_locked_member_view_does_not_rewrite_text(
        client, make_user_key, app_module, monkeypatch):
    """Viewing a locked sample's member — or the sample itself — must not
    self-heal-rewrite the member's stored text: the lock freezes what was
    curated. An UNLOCKED member still self-heals on view (contrast case)."""
    import capture_samples_store as gs
    import captures_store as cs

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
    _insert_sample(conn, gs, "locked02sid", locked=True, user_id=uid)
    _insert_member(conn, "locked02cid", "locked02sid", user_id=uid)
    _insert_sample(conn, gs, "open0002sid", locked=False, user_id=uid)
    _insert_member(conn, "open0002cid", "open0002sid", user_id=uid)

    def _pp(raw_text, **kw):
        return "REWRITTEN"

    monkeypatch.setattr(app_module, "_postprocess_text", _pp)

    h = bearer(raw)
    # Locked member: the GET succeeds but the stored text stays frozen.
    assert client.get("/captures/api/locked02cid", headers=h).status_code == 200
    assert cs.get_capture("locked02cid")["final"] == "f"
    # Locked sample view (the _enrich_sample member loop): still frozen.
    assert client.get(
        "/captures/api/samples/locked02sid", headers=h).status_code == 200
    assert cs.get_capture("locked02cid")["final"] == "f"
    # Contrast: an unlocked member self-heals to the current pipeline output.
    assert client.get("/captures/api/open0002cid", headers=h).status_code == 200
    assert cs.get_capture("open0002cid")["final"] == "REWRITTEN"


def test_list_toolbar_counts_scoped_to_caller(client, make_user_key):
    """GET /captures/api/list: a scope=own caller's counts/total_count cover
    only their OWN rows (the global cross-user breakdown must not leak);
    an admin keeps the global numbers. Pins the user_id= plumbing from the
    route into captures_store.count/counts_by_status."""
    import captures_store as cs

    _uid_root, raw_root = make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _raw_b = make_user_key("bob", pages={"captures": "own"})
    conn = cs._require_conn()
    _insert_member(conn, "alicecap0001", None, user_id=uid_a)
    _insert_member(conn, "bobcap000001", None, user_id=uid_b)
    _insert_member(conn, "bobcap000002", None, user_id=uid_b)

    body = client.get("/captures/api/list", headers=bearer(raw_a)).json()
    assert body["total_count"] == 1
    assert body["counts"]["new"] == 1

    admin_body = client.get("/captures/api/list", headers=bearer(raw_root)).json()
    assert admin_body["total_count"] == 3
    assert admin_body["counts"]["new"] == 3


# ---------------------------------------------------------------------------
# /captures/api/samples pagination
# ---------------------------------------------------------------------------

def _insert_sample_at(conn, gs, sid, *, ts, user_id="alice"):
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " transcript_join_strategy, member_hashes_json,"
        " inter_segment_silence_ms, is_stale, is_locked, status,"
        " admin_notes, language, member_trims_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, user_id, ts, gs._relpath_for(sid), 5000, "t", "space",
         "{}", 300, 0, 0, "new", "", "de", "{}"),
    )


def test_samples_are_paged_and_the_cursor_walks_every_row(client, make_user_key):
    import capture_samples_store as gs
    import captures_store as cs

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
    # Deliberately give two of them an IDENTICAL created_ts: samples merged in
    # one call share a timestamp, and a timestamp-only cursor drops whichever
    # ties land on a page boundary.
    stamps = [5.0, 4.0, 3.0, 3.0, 2.0]
    for i, ts in enumerate(stamps):
        _insert_sample_at(conn, gs, f"page{i}sid00000", ts=ts, user_id=uid)

    h = bearer(raw)
    seen, cursor, pages = [], None, 0
    while True:
        q = "/captures/api/samples?limit=2"
        if cursor:
            q += f"&before_ts={cursor['before_ts']}&before_id={cursor['before_id']}"
        body = client.get(q, headers=h).json()
        assert len(body["samples"]) <= 2
        seen.extend(s["id"] for s in body["samples"])
        cursor = body["next"]
        pages += 1
        if not cursor:
            break
        assert pages < 10, "cursor is not advancing"

    assert len(seen) == len(stamps)
    assert len(set(seen)) == len(stamps)      # no row served twice
    # Newest-first across page boundaries.
    order = [stamps[int(s[4])] for s in seen]
    assert order == sorted(order, reverse=True)


def test_last_page_reports_no_cursor(client, make_user_key):
    import capture_samples_store as gs
    import captures_store as cs

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = cs._require_conn()
    for i in range(2):
        _insert_sample_at(conn, gs, f"exact{i}sid0000", ts=float(i), user_id=uid)

    # A page that is exactly `limit` long with nothing after it must NOT
    # advertise a next cursor.
    body = client.get("/captures/api/samples?limit=2", headers=bearer(raw)).json()
    assert len(body["samples"]) == 2
    assert body["next"] is None


# ---------------------------------------------------------------------------
# Cross-user read audit trail + preview-audio cache policy
# ---------------------------------------------------------------------------

def _insert_capture_with_request(conn, cid, request_id, user_id):
    rel = os.path.join(cid[0:2], cid[2:4], f"{cid}.wav")
    conn.execute(
        "INSERT INTO captures (id, created_ts, request_id, model, language,"
        " duration_seconds, audio_relpath, audio_format, raw, final,"
        " words_json, segments_json, corrections_json, status, user_id,"
        " sample_id, sample_order)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, 1.0, request_id, "m", "de", 2.0, rel, "wav", "r", "f", "[]",
         "[]", "[]", "new", user_id, None, 0),
    )


def test_by_request_id_audits_cross_user_read(client, make_user_key, caplog):
    """The by-request lookup hands a scope=all NON-admin the full capture row
    (raw/final text + the owner's username) for someone else's capture. That
    is exactly what the eleven read-by-id siblings audit, and it was the one
    cross-user read-by-key path with no log line — DSARs are answered from
    this log stream."""
    import logging
    import captures_store as cs

    make_user_key("root", is_admin=True)
    uid_owner, _raw_owner = make_user_key("alice", pages={"captures": "own"})
    _uid_v, raw_viewer = make_user_key("viewer", pages={"captures": "all"})
    conn = cs._require_conn()
    _insert_capture_with_request(conn, "byreqcap0001", "req-xyz", uid_owner)

    with caplog.at_level(logging.INFO, logger="captures_routes"):
        r = client.get(
            "/captures/api/by-request/req-xyz", headers=bearer(raw_viewer))
    assert r.status_code == 200
    assert len(r.json()["captures"]) == 1
    audit = [m for m in caplog.messages if "cross-user-read" in m]
    assert audit and "capture-by-request" in audit[0]

    # Self-reads must stay silent (same rule the siblings follow).
    caplog.clear()
    uid_self, raw_self = make_user_key("solo", pages={"captures": "all"})
    _insert_capture_with_request(conn, "byreqcap0002", "req-own", uid_self)
    with caplog.at_level(logging.INFO, logger="captures_routes"):
        client.get("/captures/api/by-request/req-own", headers=bearer(raw_self))
    assert not [m for m in caplog.messages if "cross-user-read" in m]


def test_propose_merges_audits_cross_user_read(client, make_user_key,
                                               monkeypatch, caplog):
    """Proposals carry other users' capture previews + resolved usernames to a
    scope=all non-admin; that read is audited like the read-by-id siblings."""
    import logging
    import captures_routes

    make_user_key("root", is_admin=True)
    uid_owner, _raw_owner = make_user_key("alice", pages={"captures": "own"})
    _uid_v, raw_viewer = make_user_key("viewer", pages={"captures": "all"})

    def _fake_propose(**kw):
        return ([{
            "member_ids": ["propcap00001"],
            "member_previews": [{"id": "propcap00001", "user_id": uid_owner}],
            "user_id": uid_owner,
        }], False)

    monkeypatch.setattr(
        captures_routes.captures_merge_proposer, "propose_merges",
        _fake_propose)

    with caplog.at_level(logging.INFO, logger="captures_routes"):
        r = client.get(
            "/captures/api/propose-merges", headers=bearer(raw_viewer))
    assert r.status_code == 200
    audit = [m for m in caplog.messages if "cross-user-read" in m]
    assert audit and "merge-proposal" in audit[0]


def test_preview_merge_audio_is_not_cacheable(client, make_user_key,
                                              monkeypatch, tmp_path):
    """The merged preview WAV is PHI behind a per-row owner check. FileResponse
    alone sends only ETag/Last-Modified, which makes the body heuristically
    cacheable by any shared cache in front of the app — the two sibling audio
    routes both send Cache-Control: no-store."""
    import wave
    import audio_merge
    import captures_routes

    _uid, raw = make_user_key("root", is_admin=True)

    monkeypatch.setattr(
        captures_routes, "_validate_merge_payload",
        lambda ids, silence_ms, user: ([], "root", ["/nonexistent.wav"], 0))

    def _fake_merge(paths, dst, **kw):
        with wave.open(dst, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16)
        return {"duration_ms": 1}

    monkeypatch.setattr(audio_merge, "merge_wavs", _fake_merge)

    r = client.post(
        "/captures/api/samples/preview-audio",
        json={"member_ids": ["prevcap00001"]},
        headers=bearer(raw),
    )
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"


def test_audio_rate_limit_is_hot_and_per_identity(client, app_module,
                                                  monkeypatch):
    """The cap is read from config on every call, so a test can lower it to 2
    and raise it to 0 (= unlimited) without restarting anything. A missing cid
    404s, but only AFTER the limiter runs — which is what we are measuring."""
    monkeypatch.setattr(app_module.cfg, "CAPTURES_AUDIO_RATE_PER_MIN", 2,
                        raising=False)
    assert client.get("/captures/api/nope0001/audio").status_code == 404
    assert client.get("/captures/api/nope0001/audio").status_code == 404
    r = client.get("/captures/api/nope0001/audio")
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["param"] == "CAPTURES_AUDIO_RATE_PER_MIN"
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert body["detail"] == body["error"]["message"]

    # 0 = unlimited, applied to the very next request with no reset.
    monkeypatch.setattr(app_module.cfg, "CAPTURES_AUDIO_RATE_PER_MIN", 0,
                        raising=False)
    for _ in range(20):
        assert client.get("/captures/api/nope0001/audio").status_code == 404


# ---------------------------------------------------------------------------
# Export: the English translate row
# ---------------------------------------------------------------------------

def _export_manifest(only_status="ready"):
    import io
    import tarfile

    import captures_routes

    blob = b"".join(captures_routes._build_export_stream(only_status, False))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        text = tar.extractfile("manifest.jsonl").read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _ready_capture(cs, monkeypatch, tmp_path, *, language, translations):
    _fake_wav_transcode(monkeypatch)
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = cs.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language=language, duration_seconds=1.0, raw="r", final="quelle",
        words=[], segments=[], user_id="alice", translations=translations,
        translation_model="HY-MT", translation_source="cascade-mt",
    )
    cs.update_capture(cid, {"status": "ready"})
    return cid


def test_export_matches_english_track_by_base_subtag(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """`en-US` is an accepted translate target and is stored under that key;
    the exporter used to look for the literal "en" and never emit the row."""
    _ready_capture(captures_store_db, monkeypatch, tmp_path,
                   language="de", translations={"en-US": "hi there"})
    rows = _export_manifest()
    assert [r["task"] for r in rows] == ["transcribe", "translate"]
    assert rows[1]["text"] == "hi there"
    assert rows[1]["model"] == "HY-MT"


def test_export_skips_translate_row_for_english_source(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """An English source with target `en` is the same-language short-circuit:
    the "translation" IS the transcript, and English audio labelled
    task=translate is never valid training data."""
    _ready_capture(captures_store_db, monkeypatch, tmp_path,
                   language="en-GB", translations={"en": "quelle"})
    rows = _export_manifest()
    assert [r["task"] for r in rows] == ["transcribe"]
