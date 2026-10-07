"""Integration tests for /captures routes.

Captures rows require real audio transcode (ffmpeg) to create, so these
tests focus on the read/list/route-ordering/auth surface that works without
fabricating audio blobs.
"""

import json
import os

import pytest
from starlette.testclient import TestClient

from faster_whisper_backend.pipeline import engine as pl_engine
from tests.conftest import bearer


def test_captures_page(client):
    r = client.get("/captures")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # The raw-string template must ship the literal glyph, not a JS escape
    # that innerHTML would render verbatim as "Whisper\u2019s".
    html = r.text
    assert "Whisper’s" in html and "\\u2019" not in html


def test_captures_page_403_landing_short_circuits_for_admin(client):
    """Admins are stored with permissions '{}' so whoami returns pages = {},
    which is truthy in JS and has no `captures` key. The landing gate must
    bail on is_admin before it looks at the pages map, and treat an EMPTY
    map as "permissions absent" rather than "no scope" — otherwise any 403
    (stale CSRF token) wipes <main> for the admin."""
    html = client.get("/captures").text
    gate = html[html.index("async function _renderAdminOnlyIfNonAdmin()"):]
    gate = gate[:gate.index("_renderNoAccessLanding({ page: 'captures' })")]
    assert "if (j && j.is_admin) return false;" in gate
    assert "if (pages && !Object.keys(pages).length) pages = null;" in gate
    # The admin short-circuit precedes the pages-keyed test.
    assert gate.index("j.is_admin) return false") < gate.index("pages.captures")


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
    assert s0.json()["status"] == "idle"
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
    from faster_whisper_backend.settings import config as cfg
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
    from faster_whisper_backend.auth import dependencies as auth
    from faster_whisper_backend.captures import routes as captures_routes
    from fastapi import HTTPException

    captures_store = captures_store_db
    _fake_wav_transcode(monkeypatch)

    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = captures_store.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language="de", audio_s=1.0, raw="r", final="f",
        words=[], segments=[], user_id="alice",
    )
    # Delete the audio so the OLD ordering would raise 410 ("audio is missing"),
    # leaking that the row exists; the fix must 404 for a non-owner first.
    os.unlink(captures_store.abs_audio_path(captures_store.get_capture(cid)["audio_relpath"]))

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
        " transcript_join_strategy, member_hashes,"
        " inter_segment_silence_ms, is_stale, is_locked, status,"
        " admin_notes, language, member_trims)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, user_id, 1.0, gs._relpath_for(sid), 5000, "t", "space",
         "{}", 300, 0, 1 if locked else 0, "new", "", "de", "{}"),
    )


def _insert_member(conn, cid, sid, user_id="alice"):
    """Insert a captures row (optionally a member of sample `sid`) directly."""
    rel = os.path.join(cid[0:2], cid[2:4], f"{cid}.wav")
    conn.execute(
        "INSERT INTO captures (id, created_ts, request_id, model, language,"
        " audio_s, audio_relpath, audio_format, raw_text, final_text,"
        " words, segments, corrections, status, user_id,"
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
    from faster_whisper_backend.auth import dependencies as auth
    from faster_whisper_backend.captures import routes as captures_routes
    from fastapi import HTTPException

    captures_store = captures_store_db
    gs = groups_store_db
    conn = captures_store._require_conn()

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

    locked_row = captures_store.get_capture("locked00cid")
    # Non-admin (even the owner) is refused on a locked sample's member.
    with pytest.raises(HTTPException) as ei:
        captures_routes._assert_member_sample_not_locked(locked_row, _user(False))
    assert ei.value.status_code == 409
    # Admin passes through.
    captures_routes._assert_member_sample_not_locked(locked_row, _user(True))
    # A member of an UNLOCKED sample, and a member of NO sample, pass through.
    captures_routes._assert_member_sample_not_locked(
        captures_store.get_capture("open000cid"), _user(False))
    captures_routes._assert_member_sample_not_locked(
        captures_store.get_capture("free0000cid"), _user(False))


def _fake_wav_transcode(monkeypatch):
    """Route audio_transcode to a stub that writes a tiny valid WAV."""
    import wave

    from faster_whisper_backend.audio import transcode as audio_transcode

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
    from faster_whisper_backend.auth import dependencies as auth
    from faster_whisper_backend.captures import routes as captures_routes
    from fastapi import HTTPException

    captures_store = captures_store_db
    _fake_wav_transcode(monkeypatch)
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = captures_store.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language="de", audio_s=1.0, raw="r", final="f",
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
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    owner_uid, _raw_owner = make_user_key("alice", pages={"captures": "own"})
    _uid, raw_bob = make_user_key("bob", pages={"captures": "own"})

    _fake_wav_transcode(monkeypatch)
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = captures_store.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language="de", audio_s=1.0, raw="r", final="f",
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
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
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
    row = captures_store.get_capture("locked01cid")
    assert row is not None and row["status"] == "new"


def test_nonadmin_can_unlock_but_not_edit_a_locked_sample(client, make_user_key):
    """`is_locked` is writable by any captures-scoped caller, so it must also
    be RELEASABLE by them — otherwise it is a one-way switch only an admin can
    undo. A non-admin may send the unlock and nothing else while locked."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
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
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, "locked02sid", locked=True, user_id=uid)
    _insert_member(conn, "locked02cid", "locked02sid", user_id=uid)
    _insert_sample(conn, gs, "open0002sid", locked=False, user_id=uid)
    _insert_member(conn, "open0002cid", "open0002sid", user_id=uid)

    def _pp(raw_text, **kw):
        return "REWRITTEN"

    monkeypatch.setattr(pl_engine, "_postprocess_text", _pp)

    h = bearer(raw)
    # Locked member: the GET succeeds but the stored text stays frozen.
    assert client.get("/captures/api/locked02cid", headers=h).status_code == 200
    assert captures_store.get_capture("locked02cid")["final"] == "f"
    # Locked sample view (the _enrich_sample member loop): still frozen.
    assert client.get(
        "/captures/api/samples/locked02sid", headers=h).status_code == 200
    assert captures_store.get_capture("locked02cid")["final"] == "f"
    # Contrast: an unlocked member self-heals to the current pipeline output.
    assert client.get("/captures/api/open0002cid", headers=h).status_code == 200
    assert captures_store.get_capture("open0002cid")["final"] == "REWRITTEN"


def test_list_toolbar_counts_scoped_to_caller(client, make_user_key):
    """GET /captures/api/list: a scope=own caller's counts/total_count cover
    only their OWN rows (the global cross-user breakdown must not leak);
    an admin keeps the global numbers. Pins the user_id= plumbing from the
    route into captures_store.count/counts_by_status."""
    from faster_whisper_backend.captures import store as captures_store

    _uid_root, raw_root = make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"captures": "own"})
    uid_b, _raw_b = make_user_key("bob", pages={"captures": "own"})
    conn = captures_store._require_conn()
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
        " transcript_join_strategy, member_hashes,"
        " inter_segment_silence_ms, is_stale, is_locked, status,"
        " admin_notes, language, member_trims)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, user_id, ts, gs._relpath_for(sid), 5000, "t", "space",
         "{}", 300, 0, 0, "new", "", "de", "{}"),
    )


def test_samples_are_paged_and_the_cursor_walks_every_row(client, make_user_key):
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    # Deliberately give two of them an IDENTICAL created_ts that straddles the
    # first page boundary (page 1 = [5.0, the higher-id 3.0]): samples merged
    # in one call share a timestamp, and a timestamp-only cursor drops the
    # other 3.0 row.
    stamps = [5.0, 3.0, 3.0, 2.0, 1.0]
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
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
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
        " audio_s, audio_relpath, audio_format, raw_text, final_text,"
        " words, segments, corrections, status, user_id,"
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
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid_owner, _raw_owner = make_user_key("alice", pages={"captures": "own"})
    _uid_v, raw_viewer = make_user_key("viewer", pages={"captures": "all"})
    conn = captures_store._require_conn()
    _insert_capture_with_request(conn, "byreqcap0001", "req-xyz", uid_owner)

    with caplog.at_level(logging.INFO, logger="whisper-api"):
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
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        client.get("/captures/api/by-request/req-own", headers=bearer(raw_self))
    assert not [m for m in caplog.messages if "cross-user-read" in m]


def test_propose_merges_audits_cross_user_read(client, make_user_key,
                                               monkeypatch, caplog):
    """Proposals carry other users' capture previews + resolved usernames to a
    scope=all non-admin; that read is audited like the read-by-id siblings."""
    import logging
    from faster_whisper_backend.captures import routes as captures_routes

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

    with caplog.at_level(logging.INFO, logger="whisper-api"):
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
    from faster_whisper_backend.captures import merge as audio_merge
    from faster_whisper_backend.captures import routes as captures_routes

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

    from faster_whisper_backend.captures import routes as captures_routes

    blob = b"".join(captures_routes._build_export_stream(only_status, False))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        text = tar.extractfile("manifest.jsonl").read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _ready_capture(captures_store, monkeypatch, tmp_path, *, language, translations):
    _fake_wav_transcode(monkeypatch)
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = captures_store.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language=language, audio_s=1.0, raw="r", final="quelle",
        words=[], segments=[], user_id="alice", translations=translations,
        translation_model="HY-MT", translation_source="cascade-mt",
    )
    captures_store.update_capture(cid, {"status": "ready"})
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


def test_export_skips_translate_row_for_translate_task_capture(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """A capture made through /v1/audio/translations already IS a translate
    row (English text, source language, task=translate). A cascade-MT track
    on the same row would emit a second task=translate line with other text
    and another model for the same audio file."""
    captures_store = captures_store_db
    cid = _ready_capture(captures_store, monkeypatch, tmp_path,
                         language="de", translations={"en": "other text"})
    with captures_store._lock:
        with captures_store._require_conn() as conn:
            conn.execute("UPDATE captures SET task = 'translate' WHERE id = ?",
                         (cid,))
    rows = _export_manifest()
    assert [r["task"] for r in rows] == ["translate"]
    assert rows[0]["text"] == "quelle"


def test_export_duration_describes_the_trimmed_companion(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """A legacy VAD-trimmed capture packs the trimmed file, so both manifest
    lines must report its length (audio_s minus lead + trail), not the
    untrimmed audio_s."""
    import shutil

    captures_store = captures_store_db
    cid = _ready_capture(captures_store, monkeypatch, tmp_path,
                         language="de", translations={"en": "hi there"})
    row = captures_store.get_capture(cid)
    trimmed_rel = row["audio_relpath"] + ".trim.wav"
    shutil.copyfile(captures_store.abs_audio_path(row["audio_relpath"]),
                    captures_store.abs_audio_path(trimmed_rel))
    with captures_store._lock:
        with captures_store._require_conn() as conn:
            conn.execute(
                "UPDATE captures SET audio_s = 12.4, audio_trimmed_relpath = ?,"
                " audio_trim_lead_ms = 2000, audio_trim_trail_ms = 1300"
                " WHERE id = ?", (trimmed_rel, cid))
    rows = _export_manifest()
    assert [r["task"] for r in rows] == ["transcribe", "translate"]
    assert [r["duration"] for r in rows] == [pytest.approx(9.1)] * 2


def test_rebuild_lock_survives_prune_while_in_flight():
    """A Lock handed out but not yet acquired must not be pruned: the prune in
    _get_rebuild_lock skips sids pinned in _rebuild_inflight, so a second
    caller for the same sid gets the SAME object instead of minting a new one
    (which would let two rebuilds run concurrently)."""
    from faster_whisper_backend.captures import samples as capture_samples

    saved_locks = dict(capture_samples._rebuild_locks)
    saved_inflight = dict(capture_samples._rebuild_inflight)
    capture_samples._rebuild_locks.clear()
    capture_samples._rebuild_inflight.clear()
    try:
        sid = "sid-in-flight"
        # Mirror _rebuild_lock's handout window: sid pinned, lock not yet
        # acquired (so v.locked() alone would not protect it).
        with capture_samples._rebuild_locks_guard:
            capture_samples._rebuild_inflight[sid] = 1
        first = capture_samples._get_rebuild_lock(sid)
        assert not first.locked()
        # Trigger the opportunistic prune with a flood of other sids.
        for i in range(capture_samples._REBUILD_LOCKS_MAX + 1):
            capture_samples._get_rebuild_lock(f"sid-filler-{i}")
        assert capture_samples._get_rebuild_lock(sid) is first
        # And _release_rebuild_lock must not drop a pinned sid either.
        capture_samples._release_rebuild_lock(sid)
        assert capture_samples._rebuild_locks.get(sid) is first
    finally:
        capture_samples._rebuild_locks.clear()
        capture_samples._rebuild_locks.update(saved_locks)
        capture_samples._rebuild_inflight.clear()
        capture_samples._rebuild_inflight.update(saved_inflight)


def test_rebuild_lock_contextmanager_pins_and_unpins():
    """_rebuild_lock registers the sid in _rebuild_inflight for the whole
    handout-to-release span and cleans up after itself."""
    from faster_whisper_backend.captures import samples as capture_samples

    sid = "sid-ctx-pin"
    with capture_samples._rebuild_lock(sid):
        assert capture_samples._rebuild_inflight.get(sid) == 1
        assert capture_samples._rebuild_locks[sid].locked()
    assert sid not in capture_samples._rebuild_inflight
    assert not capture_samples._rebuild_locks[sid].locked()
    capture_samples._release_rebuild_lock(sid)
    assert sid not in capture_samples._rebuild_locks


def _grouped_capture(captures_store, monkeypatch, tmp_path, sid):
    """One ready capture packed into sample `sid`; returns the capture id."""
    from faster_whisper_backend.captures import routes as cr

    cid = _ready_capture(captures_store, monkeypatch, tmp_path, language="de",
                         translations=None)
    cr._insert_sample_with_sid(
        sid=sid, user_id="alice", member_ids=[cid], transcript="quelle",
        join_strategy="space", silence_ms=300, member_hash_map={cid: "h"},
        duration_ms=1000, language="de", member_trims={},
    )
    return cid


def test_reprocess_vad_worker_uses_pinned_rebuild_lock(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """The worker holds its per-sid lock across build + DB write for many
    sids in a loop, so it must take the pinning _rebuild_lock: with the raw
    _get_rebuild_lock a route's prune could drop the sid between handout and
    acquire and mint a SECOND Lock for a concurrent regenerate."""
    import inspect

    from faster_whisper_backend.captures import samples as capture_samples
    from faster_whisper_backend.captures import vad_reprocess as vr

    assert "_get_rebuild_lock" not in inspect.getsource(vr)

    sid = "a" * 32
    _grouped_capture(captures_store_db, monkeypatch, tmp_path, sid)
    entered = []
    real = capture_samples._rebuild_lock

    def _counting(s):
        entered.append(s)
        return real(s)

    monkeypatch.setattr(capture_samples, "_rebuild_lock", _counting)
    monkeypatch.setattr(capture_samples, "_build_merged_wav",
                        lambda **kw: (1000, {}, {}))
    vr._run()
    assert vr.status()["status"] == "done"
    assert vr.status()["rebuilt"] == 1
    assert entered == [sid]


def test_reprocess_vad_failure_fallback_spares_a_sample_locked_since_snapshot(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """A failure outside the build try falls back to is_stale=1, but with the
    same fresh-row re-check as the main path: a sample locked after the
    job-start snapshot is left alone (counted skipped, not stale)."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import vad_reprocess as vr

    sid = "d" * 32
    _grouped_capture(captures_store_db, monkeypatch, tmp_path, sid)
    conn = captures_store_db._require_conn()

    def _members_then_fail(s):
        conn.execute("UPDATE capture_samples SET is_locked = 1 WHERE id = ?",
                     (s,))
        raise RuntimeError("database is locked")

    monkeypatch.setattr(gs, "get_members", _members_then_fail)
    vr._run()
    st = vr.status()
    assert st["status"] == "done" and st["stale"] == 0 and st["skipped"] == 1
    assert gs.get_sample(sid)["is_stale"] == 0


def test_reprocess_vad_failure_fallback_writes_under_the_rebuild_lock(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """The fallback stale write must hold the per-sid rebuild lock, or it can
    land after a concurrent regenerate cleared the flag."""
    import contextlib

    from faster_whisper_backend.captures import samples as capture_samples
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import vad_reprocess as vr

    sid = "e" * 32
    _grouped_capture(captures_store_db, monkeypatch, tmp_path, sid)
    held = []
    writes = []

    @contextlib.contextmanager
    def _recording_lock(s):
        held.append(s)
        try:
            yield
        finally:
            held.remove(s)

    real_update = gs.update_sample

    def _update(s, patch):
        writes.append((s, dict(patch), list(held)))
        return real_update(s, patch)

    def _fail(s):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(capture_samples, "_rebuild_lock", _recording_lock)
    monkeypatch.setattr(gs, "update_sample", _update)
    monkeypatch.setattr(gs, "get_members", _fail)
    vr._run()
    assert vr.status()["stale"] == 1
    assert writes == [(sid, {"is_stale": 1}, [sid])]
    assert gs.get_sample(sid)["is_stale"] == 1


def test_reprocess_vad_failure_fallback_counts_a_failed_flag_write_skipped(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """When the best-effort is_stale write fails too, the sample is still
    exported: counting it "stale" (flagged, excluded from export) misreported
    it."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import vad_reprocess as vr

    sid = "f" * 32
    _grouped_capture(captures_store_db, monkeypatch, tmp_path, sid)

    def _fail(*a, **kw):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(gs, "get_members", _fail)
    monkeypatch.setattr(gs, "update_sample", _fail)
    vr._run()
    st = vr.status()
    assert st["status"] == "done" and st["stale"] == 0 and st["skipped"] == 1
    assert gs.get_sample(sid)["is_stale"] == 0


def test_create_sample_rejects_member_already_grouped(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """create_sample_api now awaits between validation and insert, so a
    double-submitted Merge can validate twice. The member UPDATE's rowcount
    is the gate: the loser gets a 409 and its capture_samples row is rolled
    back instead of persisting member-less."""
    from fastapi import HTTPException

    from faster_whisper_backend.captures import routes as cr

    captures_store = captures_store_db
    cid = _grouped_capture(captures_store, monkeypatch, tmp_path, "b" * 32)
    with pytest.raises(HTTPException) as ei:
        cr._insert_sample_with_sid(
            sid="c" * 32, user_id="alice", member_ids=[cid],
            transcript="quelle", join_strategy="space", silence_ms=300,
            member_hash_map={cid: "h"}, duration_ms=1000, language="de",
            member_trims={},
        )
    assert ei.value.status_code == 409
    conn = captures_store._require_conn()
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM capture_samples ORDER BY id").fetchall()]
    assert ids == ["b" * 32]
    assert captures_store.get_capture(cid)["sample_id"] == "b" * 32


def test_insert_sample_with_sid_holds_captures_lock(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """The two stores share one autocommit connection, so the explicit
    BEGIN..COMMIT in _insert_sample_with_sid must also hold
    captures_store._lock — otherwise a bare captures_store write from
    another thread joins the transaction and is discarded with a losing
    merge's ROLLBACK."""
    from faster_whisper_backend.captures import routes as cr
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    cid = _ready_capture(captures_store, monkeypatch, tmp_path, language="de",
                         translations=None)
    real_conn = gs._require_conn()
    seen: dict[str, bool] = {}

    class _Conn:
        def execute(self, sql, *args, **kwargs):
            if str(sql).lstrip().upper().startswith("BEGIN"):
                seen["captures_lock_held"] = captures_store._lock.locked()
            return real_conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_conn, name)

    monkeypatch.setattr(gs, "_require_conn", lambda: _Conn())
    sid = "d" * 32
    cr._insert_sample_with_sid(
        sid=sid, user_id="alice", member_ids=[cid], transcript="quelle",
        join_strategy="space", silence_ms=300, member_hash_map={cid: "h"},
        duration_ms=1000, language="de", member_trims={},
    )
    assert seen == {"captures_lock_held": True}
    assert captures_store.get_capture(cid)["sample_id"] == sid


def test_insert_sample_with_sid_rolls_back_a_failed_commit(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """COMMIT sat outside the try: a commit-time SQLITE_FULL / IOERR left the
    transaction open on the shared autocommit connection, and every later
    BEGIN failed with "cannot start a transaction within a transaction"."""
    import sqlite3

    from faster_whisper_backend.captures import routes as cr
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    cid = _ready_capture(captures_store, monkeypatch, tmp_path, language="de",
                         translations=None)
    real_conn = gs._require_conn()
    failed = []

    class _Conn:
        def execute(self, sql, *args, **kwargs):
            if str(sql).strip().upper() == "COMMIT" and not failed:
                failed.append(sql)
                raise sqlite3.OperationalError("database or disk is full")
            return real_conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_conn, name)

    monkeypatch.setattr(gs, "_require_conn", lambda: _Conn())
    kwargs = dict(user_id="alice", member_ids=[cid], transcript="quelle",
                  join_strategy="space", silence_ms=300,
                  member_hash_map={cid: "h"}, duration_ms=1000, language="de",
                  member_trims={})
    with pytest.raises(sqlite3.OperationalError):
        cr._insert_sample_with_sid(sid="e" * 32, **kwargs)
    assert failed and not real_conn.in_transaction
    assert real_conn.execute("SELECT COUNT(*) FROM capture_samples").fetchone()[0] == 0
    assert captures_store.get_capture(cid)["sample_id"] is None
    # The connection is usable again: the next merge commits.
    cr._insert_sample_with_sid(sid="f" * 32, **kwargs)
    assert captures_store.get_capture(cid)["sample_id"] == "f" * 32


def test_create_sample_runs_the_insert_off_the_event_loop():
    """_insert_sample_with_sid blocks on both store locks, which clear_all
    holds across a full VACUUM; called inline it parked the event loop."""
    import inspect

    from faster_whisper_backend.captures import routes as cr

    src = inspect.getsource(cr.create_sample_api)
    assert ("await asyncio.to_thread(functools.partial(\n"
            "            _insert_sample_with_sid,") in src
    assert "        _insert_sample_with_sid(\n" not in src


def test_list_samples_projects_chip_offsets_without_hydrating_words(
        client, make_user_key, monkeypatch):
    """The list path projects member chips onto global word indices from
    get_members' word_count alone — no per-member get_capture (that was a
    full SELECT * + words JSON decode per member per page)."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    sid = "chipsid000000001"
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    _insert_member(conn, "chipmember00", sid, user_id=uid)
    _insert_member(conn, "chipmember01", sid, user_id=uid)
    conn.execute(
        "UPDATE captures SET words = ?, sample_order = 0 WHERE id = ?",
        (json.dumps([{"w": "a"}, {"w": "b"}]), "chipmember00"))
    conn.execute(
        "UPDATE captures SET words = ?, corrections = ?, sample_order = 1"
        " WHERE id = ?",
        (json.dumps([{"w": "c"}, {"w": "d"}, {"w": "e"}]),
         json.dumps([{"idx": 1, "wrong": "a", "correct": "b"}]),
         "chipmember01"))

    monkeypatch.setattr(
        captures_store, "get_capture",
        lambda cid: pytest.fail("list path must not hydrate members"))
    body = client.get("/captures/api/samples", headers=bearer(raw)).json()
    groups = [g for g in body["samples"] if g["id"] == sid]
    assert len(groups) == 1
    assert groups[0]["corrections"] == [
        {"idx": 3, "wrong": "a", "correct": "b"}]


def test_list_samples_carries_what_the_filters_match_on(client, make_user_key):
    """Merged groups used to ignore the page's model filter and search box
    (they stayed on screen whatever was typed): a group has no model or
    request of its own. The list now projects both from its members."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    conn = captures_store._require_conn()
    sid = "filtersid0000001"
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    _insert_member(conn, "filtmember00", sid, user_id=uid)
    _insert_member(conn, "filtmember01", sid, user_id=uid)
    conn.execute("UPDATE captures SET model = 'org/model-a', request_id = 'req-a',"
                 " sample_order = 0 WHERE id = 'filtmember00'")
    conn.execute("UPDATE captures SET model = 'org/model-b', request_id = NULL,"
                 " sample_order = 1 WHERE id = 'filtmember01'")

    body = client.get("/captures/api/samples", headers=bearer(raw)).json()
    g = next(g for g in body["samples"] if g["id"] == sid)
    assert g["models"] == ["org/model-a", "org/model-b"]
    assert g["member_ids"] == ["filtmember00", "filtmember01"]
    assert g["member_request_ids"] == ["req-a"]


def test_page_search_matches_ids_and_filters_groups(client):
    """Source pins: the search matches the capture id and request id (what
    the log block prints as `captured=` / `req=`) by PREFIX, kept out of the
    text haystack — a substring match on 32-hex ids let any hex-only word hit
    random rows — and group cards go through the model + search filters like
    capture cards do."""
    html = client.get("/captures").text
    assert "_idPrefix(q, [r.id, r.request_id])" in html
    assert "_idPrefix(q, [g.id].concat(g.member_ids || [], g.member_request_ids || []))" in html
    assert "String(ids[i]).toLowerCase().indexOf(q) === 0" in html
    assert "(r.id || '') + ' ' + (r.request_id || '')" not in html
    assert "(g.member_ids || []).join(' ')" not in html
    assert "function sampleMatchesFilters(g)" in html
    assert "return sampleMatchesFilters(g);" in html
    assert "_allSamples.slice()" not in html


def test_audio_original_switch_serves_the_untrimmed_file(client):
    """The default is the trimmed WAV (what the player + export use);
    `?original=1` serves the utterance the decode received. Both are reachable
    for a capture of ANY status — Export ready only covers `ready` rows."""
    from faster_whisper_backend.captures import store as captures_store

    conn = captures_store._require_conn()
    cid = "origaudio001"
    _insert_member(conn, cid, None)
    rel = os.path.join(cid[0:2], cid[2:4], f"{cid}.wav")
    trel = os.path.join(cid[0:2], cid[2:4], f"{cid}.trim.wav")
    for r, payload in ((rel, b"RIFF....WAVEoriginal"), (trel, b"RIFF....WAVEtrimmed")):
        path = captures_store.abs_audio_path(r)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(payload)
    conn.execute("UPDATE captures SET audio_trimmed_relpath = ? WHERE id = ?", (trel, cid))

    r = client.get(f"/captures/api/{cid}/audio")
    assert r.status_code == 200 and r.content.endswith(b"trimmed")
    r = client.get(f"/captures/api/{cid}/audio?original=1")
    assert r.status_code == 200 and r.content.endswith(b"original")
    assert f"{cid}.original.wav" in r.headers["content-disposition"]
    assert r.headers["cache-control"] == "no-store"


def test_capture_card_offers_both_audio_downloads(client):
    html = client.get("/captures").text
    assert "compact-player-btn compact-player-dl" in html
    assert "audioUrl + '?original=1'" in html
    assert "a.compact-player-dl {" in html


def test_page_merge_gate_cancels_a_stale_trim_estimate(client):
    """A pending merge-estimate timer / in-flight POST from an earlier valid
    selection must not re-enable Merge once the selection turns invalid (too
    many rows, mixed speakers, in-sample) or empty: both paths cancel it."""
    html = client.get("/captures").text
    bar = html[html.index("function _updateActionBar()"):]
    bar = bar[:bar.index("function _cancelTrimEstimate()")]
    assert "if (n === 0) { _cancelTrimEstimate(); return; }" in bar
    assert bar.count("_cancelTrimEstimate();") == 2
    assert "n <= MERGE_MAX_MEMBERS" in bar and "var MERGE_MAX_MEMBERS = 30;" in html
    cancel = html[html.index("function _cancelTrimEstimate()"):]
    cancel = cancel[:cancel.index("function _fetchTrimEstimate(")]
    assert "_meterEstimateToken++;" in cancel and "clearTimeout(_meterEstimateTimer)" in cancel


def test_page_translation_eligibility_mirrors_the_exporter(client):
    """The exporter skips the en track of an English-source capture and of a
    translate-task row; the card must not tag either as training data."""
    html = client.get("/captures").text
    assert ("var eligible = enKey && !srcIsEn && !isTr && trs[enKey]"
            " && String(trs[enKey]).trim();") in html
    assert "var srcIsEn = String(r.language || '')" in html
    assert "var isTr = (r.task || 'transcribe') === 'translate';" in html
    assert "(eligible && lg === enKey ? ' cc-tr-eligible' : '')" in html


def test_page_merge_gate_mirrors_the_language_and_task_partition(client):
    html = client.get("/captures").text
    bar = html[html.index("function _updateActionBar()"):]
    bar = bar[:bar.index("function _cancelTrimEstimate()")]
    assert "var mixedLangTask = langs.size > 1 || tasks.size > 1;" in bar
    assert "&& !mixedLangTask;" in bar


def test_page_load_drops_a_superseded_filter_response(client):
    """Out-of-order /list responses for two quick speaker-picker ticks must
    not leave the older filter's rows on screen."""
    html = client.get("/captures").text
    load = html[html.index("async function load()"):]
    load = load[:load.index("async function loadMoreSamples()")]
    assert "var seq = ++_loadSeq;" in load
    assert load.count("if (seq !== _loadSeq) return;") >= 3
    more = html[html.index("async function loadMoreSamples()"):]
    more = more[:more.index("async function reloadCounts()")]
    assert "var seq = _loadSeq;" in more and "if (seq !== _loadSeq) return;" in more
    # reloadCounts (after a bulk status / undo / delete) is filter-scoped too,
    # and a total_count of 0 is a real count, not a missing field.
    counts = html[html.index("async function reloadCounts()"):]
    counts = counts[:counts.index("reloadStats();")]
    assert "var seq = _loadSeq;" in counts and "if (seq !== _loadSeq) return;" in counts
    assert "if (typeof j.total_count === 'number') _totalCount = j.total_count;" in counts


def test_sample_save_does_not_resend_status(client):
    """The group's status buttons auto-save a narrow PATCH; Save resending
    the status loaded with the view reverted another tab's change."""
    html = client.get("/captures").text
    save = html[html.index("saveTBtn.onclick = function() {"):]
    save = save[:save.index("saveTBtn.disabled = true;")]
    assert "admin_notes:   sampleState.adminNotes," in save
    assert "status:" not in save


def _mixed_pair(task_b=None, lang_b="de"):
    """Two same-user captures with real WAVs on disk, differing in the second
    member's task / language."""
    import wave
    from faster_whisper_backend.captures import store as captures_store

    conn = captures_store._require_conn()
    ids = ["mixpair0000a", "mixpair0000b"]
    for cid in ids:
        _insert_member(conn, cid, None, user_id="alice")
        p = captures_store.abs_audio_path(
            os.path.join(cid[0:2], cid[2:4], f"{cid}.wav"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with wave.open(p, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 1600)
    conn.execute("UPDATE captures SET task = ?, language = ? WHERE id = ?",
                 (task_b, lang_b, ids[1]))
    return ids


@pytest.mark.parametrize("task_b,lang_b,detail", [
    ("translate", "de", "members must all share one task"),
    (None, "en", "members must all share one language"),
])
def test_merge_rejects_mixed_task_or_language(client, make_user_key,
                                              task_b, lang_b, detail):
    """Same partition as the proposer: a sample has ONE language and ONE task
    label, so a de+en (or transcribe+translate) merge would export
    wrong-language training data."""
    _uid, raw = make_user_key("root", is_admin=True)
    ids = _mixed_pair(task_b, lang_b)
    for path in ("/captures/api/samples/merge-estimate", "/captures/api/samples"):
        r = client.post(path, json={"member_ids": ids}, headers=bearer(raw))
        assert r.status_code == 400, (path, r.text)
        assert r.json()["detail"] == detail


def test_merge_accepts_region_variants_of_one_language(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    ids = _mixed_pair(None, "de-CH")
    r = client.post("/captures/api/samples/merge-estimate",
                    json={"member_ids": ids}, headers=bearer(raw))
    assert r.status_code == 200, r.text


def test_regenerate_of_a_sample_dissolved_mid_wait_is_404(client, make_user_key,
                                                          monkeypatch):
    """dissolve_sample holds the rebuild lock, so a regenerate that waited on
    it must re-read the row and 404 instead of rebuilding an orphan WAV and
    500ing in _enrich_sample."""
    from faster_whisper_backend.captures import samples as capture_samples
    from faster_whisper_backend.captures import samples_store as gs

    _uid, raw = make_user_key("root", is_admin=True)
    sid = "regendis0000001"
    from faster_whisper_backend.captures import store as captures_store
    _insert_sample(captures_store._require_conn(), gs, sid, locked=False)
    real_lock = capture_samples._rebuild_lock

    import contextlib

    @contextlib.contextmanager
    def _dissolve_then_lock(s):
        # Simulates the dissolve that won the lock while regenerate waited.
        gs._require_conn().execute("DELETE FROM capture_samples WHERE id = ?", (s,))
        with real_lock(s):
            yield
    monkeypatch.setattr(capture_samples, "_rebuild_lock", _dissolve_then_lock)
    built = []
    monkeypatch.setattr(capture_samples, "_build_merged_wav",
                        lambda **kw: built.append(kw) or (0, {}, {}))
    r = client.post(f"/captures/api/samples/{sid}/regenerate", headers=bearer(raw))
    assert r.status_code == 404, r.text
    assert built == []


def _de_scoped_pipeline(monkeypatch):
    """Stand-in for a de-only rule: lowercases when scoped to "de"."""
    langs: list = []

    def _pp(text, **kw):
        langs.append(kw.get("language"))
        return text.lower() if kw.get("language") == "de" else text

    monkeypatch.setattr(pl_engine, "_postprocess_text", _pp)
    return langs


def _translate_capture(conn, cid, user_id, *, task):
    _insert_member(conn, cid, None, user_id=user_id)
    raw = "It is over. Was it good?"
    conn.execute(
        "UPDATE captures SET raw_text = ?, final_text = ?, text_for_training = ?,"
        " task = ? WHERE id = ?", (raw, raw, raw, task, cid))
    return raw


def test_reprocess_and_self_heal_scope_a_translate_capture_by_english(
        client, make_user_key, monkeypatch):
    """language stores the SPOKEN language; a task=translate capture's text
    is English, so /reprocess and the page-view self-heal must scope the
    pipeline by "en" (as the live run did) and leave its text alone. A
    transcribe twin still gets the de scope."""
    from faster_whisper_backend.captures import store as captures_store

    uid, raw_key = make_user_key("root", is_admin=True)
    conn = captures_store._require_conn()
    text = _translate_capture(conn, "transl000001", uid, task="translate")
    _translate_capture(conn, "transc000001", uid, task="transcribe")
    langs = _de_scoped_pipeline(monkeypatch)
    h = bearer(raw_key)

    # Page view (self-heal) — no rewrite of the stored English text.
    assert client.get("/captures/api/transl000001", headers=h).status_code == 200
    row = captures_store.get_capture("transl000001")
    assert row["final"] == text and row["text_for_training"] == text
    # Explicit /reprocess — nothing changes.
    r = client.post("/captures/api/transl000001/reprocess", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["changed"] == []
    assert captures_store.get_capture("transl000001")["final"] == text
    assert set(langs) == {"en"}
    # Contrast: the transcribe twin is scoped "de".
    r = client.post("/captures/api/transc000001/reprocess", headers=h)
    assert r.json()["capture"]["final"] == text.lower()


def test_patch_capture_runs_off_the_loop_under_the_corrections_lock(
        client, make_user_key, monkeypatch):
    """update_capture takes captures_store._lock (held across clear_all's
    VACUUM), so the PATCH must not call it on the event loop; a chip save
    holds the corrections lock across its read-merge-write."""
    import asyncio

    from faster_whisper_backend.captures import routes as captures_routes
    from faster_whisper_backend.captures import store as captures_store

    uid, raw_key = make_user_key("root", is_admin=True)
    _insert_member(captures_store._require_conn(), "patchthr0001", None,
                   user_id=uid)
    seen: list = []
    real = captures_store.update_capture

    def _spy(cid, patch):
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        seen.append((on_loop, captures_routes._corrections_write_lock.locked()))
        return real(cid, patch)

    monkeypatch.setattr(captures_store, "update_capture", _spy)
    h = bearer(raw_key)
    r = client.patch("/captures/api/patchthr0001", headers=h,
                     json={"admin_notes": "n"})
    assert r.status_code == 200, r.text
    r = client.patch("/captures/api/patchthr0001", headers=h,
                     json={"corrections": [], "baseline_corrections": []})
    assert r.status_code == 200, r.text
    assert seen == [(False, False), (False, True)]
    # Guards still surface as their HTTP codes from inside the thread.
    assert client.patch("/captures/api/nosuchcid000", headers=h,
                        json={"admin_notes": "n"}).status_code == 404
    assert client.patch("/captures/api/patchthr0001", headers=h,
                        json={"status": "bogus"}).status_code in (400, 422)


def test_patch_of_a_sample_dissolved_mid_apply_is_404(client, make_user_key,
                                                     monkeypatch):
    """A dissolve committed between the route's get_sample and _apply's
    update_sample leaves update_sample returning None — a 404, not a 500
    from _enrich_sample(None)."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    _uid, raw_key = make_user_key("root", is_admin=True)
    sid = "patchdis000001"
    _insert_sample(captures_store._require_conn(), gs, sid, locked=False)
    monkeypatch.setattr(gs, "update_sample", lambda s, patch: None)
    r = client.patch(f"/captures/api/samples/{sid}", headers=bearer(raw_key),
                     json={"admin_notes": "x"})
    assert r.status_code == 404, r.text


def test_sample_chip_save_holds_the_corrections_lock(client, make_user_key,
                                                     monkeypatch):
    """The sample PATCH's member read + per-member chip writes run under the
    same lock as a member's own PATCH, so neither can land between the
    other's read and write."""
    from faster_whisper_backend.captures import routes as captures_routes
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    uid, raw_key = make_user_key("root", is_admin=True)
    sid = "patchlck000001"
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    _insert_member(conn, "patchlckmem1", sid, user_id=uid)
    held: list = []
    real = gs.get_members

    def _spy(s):
        held.append(captures_routes._corrections_write_lock.locked())
        return real(s)

    monkeypatch.setattr(gs, "get_members", _spy)
    r = client.patch(f"/captures/api/samples/{sid}", headers=bearer(raw_key),
                     json={"corrections": [], "baseline_corrections": []})
    assert r.status_code == 200, r.text
    assert held and held[0] is True


def test_preview_save_chips_writes_off_the_loop_and_skips_newly_grouped(
        client, make_user_key, monkeypatch):
    """Up to 30 update_capture calls must not run on the event loop, and a
    member that a concurrent create_sample grouped after the validation is
    left alone (its chips now belong to the sample)."""
    import asyncio

    from faster_whisper_backend.captures import routes as captures_routes
    from faster_whisper_backend.captures import store as captures_store

    _uid, raw_key = make_user_key("root", is_admin=True)
    ids = _mixed_pair(None, "de")
    real_light = captures_store.get_captures_light

    def _light(cids):
        out = real_light(cids)
        out[ids[1]] = dict(out[ids[1]], sample_id="grabbedsid01")
        return out

    seen: list = []
    real_update = captures_store.update_capture

    def _spy(cid, patch):
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        seen.append((cid, on_loop, captures_routes._corrections_write_lock.locked()))
        return real_update(cid, patch)

    monkeypatch.setattr(captures_store, "get_captures_light", _light)
    monkeypatch.setattr(captures_store, "update_capture", _spy)
    r = client.post("/captures/api/samples/preview-save-chips",
                    headers=bearer(raw_key),
                    json={"member_ids": ids, "corrections": []})
    assert r.status_code == 200, r.text
    assert seen == [(ids[0], False, True)]
    assert list(r.json()["saved"]) == [ids[0]]
