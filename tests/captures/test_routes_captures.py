"""Integration tests for /captures routes. Audio-backed rows use
`install_fake_wav_transcode` (a stub WAV writer) instead of a real ffmpeg transcode.
"""

import inspect
import io
import json
import os
import re
import tarfile

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from faster_whisper_backend.captures import routes as captures_routes
from faster_whisper_backend.captures import samples as capture_samples
from faster_whisper_backend.captures import samples_store as gs
from faster_whisper_backend.captures import store as captures_store
from faster_whisper_backend.captures import vad_reprocess as vr
from faster_whisper_backend.core import text_corrections
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config as cfg
from tests._page_helpers import js_body, js_upto
from tests.captures._store_helpers import insert_capture, insert_sample
from tests.conftest import bearer, install_fake_wav_transcode, locked_user, write_bytes, write_wav


@pytest.fixture
def html(client):
    return client.get("/captures").text


def test_captures_page(client):
    r = client.get("/captures")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # The template must ship the literal glyph, not a JS escape
    # that innerHTML would render verbatim as "Whisper\u2019s".
    html = r.text
    assert "Whisper’s" in html and "\\u2019" not in html


def test_captures_page_403_landing_short_circuits_for_admin(html):
    """Admins are stored with permissions '{}' so whoami returns pages = {},
    which is truthy in JS and has no `captures` key. The landing gate must
    bail on is_admin before it looks at the pages map, and treat an EMPTY
    map as "permissions absent" rather than "no scope" — otherwise any 403
    (stale CSRF token) wipes <main> for the admin."""
    gate = js_upto(html, "async function _renderAdminOnlyIfNonAdmin()",
                   "_renderNoAccessLanding({ page: 'captures' })")
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
    # /captures/api/samples must resolve to the sample-list handler, NOT the
    # parameterized /captures/api/{cid} one (which would 404 with cid="samples").
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
    assert r.json()["captures"] == []


def test_host_gate_rejects_non_loopback(narrowed_remote_client):
    # /captures is user-tier (require_user_webui_host / USER_WEBUI_ALLOWED_HOSTS).
    # The list defaults OPEN, so narrow it to loopback to exercise the host gate:
    # a non-loopback host is then 403 before the page-permission check.
    assert narrowed_remote_client.get("/captures/api/list").status_code == 403


def test_list_requires_page_when_locked(client, make_user_key):
    _uid, raw = locked_user(make_user_key, pages={"captures": "none"})
    r = _get(client, raw, "list")
    assert r.status_code == 403


def test_clear_requires_admin_when_locked(client, make_user_key):
    # POST /captures/api/clear additionally Depends(require_admin).
    _uid, raw = locked_user(make_user_key, pages={"captures": "own"})
    r = client.post("/captures/api/clear", headers=bearer(raw))
    assert r.status_code == 403


def test_merge_member_scope_guard_precedes_state_checks(
        captures_store_db, monkeypatch, tmp_path):
    """A scope=own caller probing ANOTHER user's capture id must get a uniform
    404 from the ownership guard — not a 400/410 that would leak the capture's
    existence + state. Regression guard for _validate_merge_payload: the
    per-member scope check must run BEFORE the already-in-sample / audio-missing
    checks."""
    captures_store = captures_store_db
    cid = _audio_capture(captures_store, monkeypatch, tmp_path)
    # Audio gone: the state check would 410 ("audio is missing") and leak that
    # the row exists, so the scope guard must 404 a non-owner first.
    os.unlink(captures_store.abs_audio_path(captures_store.get_capture(cid)["audio_relpath"]))

    # bob: scope=own captures user, NOT the owner and NOT admin → uniform 404.
    with pytest.raises(HTTPException) as ei:
        captures_routes._validate_merge_payload([cid], 0, _own_user("bob"))
    assert ei.value.status_code == 404

    # The OWNER still reaches the real state check (410), proving the guard
    # blocks only cross-user probes — not the owner's own legitimate errors.
    with pytest.raises(HTTPException) as ei2:
        captures_routes._validate_merge_payload([cid], 0, _own_user("alice"))
    assert ei2.value.status_code == 410


def _insert_sample(conn, gs, sid, *, locked, user_id="alice", ts=1.0):
    """Insert a capture_samples row directly (no audio merge needed)."""
    insert_sample(conn, sid, user_id=user_id, created_ts=ts, language="de",
                  is_locked=1 if locked else 0)


def _insert_member(conn, cid, sid, user_id="alice", request_id=None, **cols):
    """Insert a captures row (optionally a member of sample `sid`) directly;
    `cols` override column values."""
    insert_capture(conn, cid, **{
        "request_id": request_id, "language": "de", "audio_s": 2.0,
        "user_id": user_id, "sample_id": sid, "sample_order": 0, **cols})


def _root_and_alice(make_user_key):
    """Admin root + own-scope alice; returns (raw_root, alice_uid, raw_alice, conn)."""
    _uid, raw_root = make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"captures": "own"})
    return raw_root, uid, raw, captures_store._require_conn()


def _get(client, raw, path):
    """GET /captures/api/{path} as `raw`'s key."""
    return client.get(f"/captures/api/{path}", headers=bearer(raw))


def _patch(client, raw, path, **body):
    """PATCH /captures/api/{path} (a cid or samples/{sid}) as `raw`'s key."""
    return client.patch(f"/captures/api/{path}", headers=bearer(raw), json=body)


def _post_members(client, raw, path, ids, **body):
    """POST `ids` as member_ids (plus `body`) to a sample route as `raw`'s key."""
    return client.post(path, json={"member_ids": ids, **body}, headers=bearer(raw))


def test_member_delete_respects_sample_lock(captures_store_db, groups_store_db):
    """A non-admin cannot mutate/delete a capture that is a member of a LOCKED
    sample — deleting it would auto-dissolve (and destroy the merged WAV of) an
    admin-locked sample, bypassing the same guard dissolve_sample_api enforces.
    Regression guard for _assert_member_sample_not_locked."""
    from faster_whisper_backend.auth import dependencies as auth

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


def _on_loop():
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _write_silent_wav(path, frames):
    write_wav(path, b"\x00\x00" * frames)


def _own_user(uid):
    from faster_whisper_backend.auth import dependencies as auth

    return {"user_id": uid, "permissions": auth.Permissions(
        {"pages": {"captures": "own"}}, is_admin=False)}


def _audio_capture(captures_store, monkeypatch, tmp_path, **fields):
    """create_capture over a stub-transcoded source; `fields` override the defaults."""
    install_fake_wav_transcode(monkeypatch)
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    return captures_store.create_capture(audio_src_path=str(src), **{
        "request_id": "r1", "model": "small", "language": "de", "audio_s": 1.0,
        "raw": "r", "final": "f", "words": [], "segments": [],
        "user_id": "alice", **fields})


def test_merge_member_404_body_uniform_missing_vs_foreign(
        captures_store_db, monkeypatch, tmp_path):
    """Missing id and another-user's id must yield the SAME 404 template on
    the merge surface. The guard's uniform 404 STATUS is pointless if the
    body still says "capture X not found" for missing ids but "not found"
    for foreign ones — the body becomes the existence oracle."""
    captures_store = captures_store_db
    cid = _audio_capture(captures_store, monkeypatch, tmp_path)

    bob = _own_user("bob")
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
    owner_uid, _raw_owner = locked_user(make_user_key, pages={"captures": "own"})
    _uid, raw_bob = make_user_key("bob", pages={"captures": "own"})

    cid = _audio_capture(captures_store, monkeypatch, tmp_path,
                         user_id=owner_uid)

    r_missing = _get(client, raw_bob, "does-not-exist")
    r_foreign = _get(client, raw_bob, cid)
    assert r_missing.status_code == r_foreign.status_code == 404
    assert r_missing.json() == r_foreign.json()


def test_locked_member_mutations_blocked_at_endpoints(client, make_user_key):
    """The sample lock must hold on the LIVE routes, not only in the helper:
    PATCH, DELETE and reprocess on a locked sample's member all 409 for the
    non-admin owner. (Removing the _assert_member_sample_not_locked call
    sites would pass the helper test but fail this one.)"""
    _, uid, raw, conn = _root_and_alice(make_user_key)
    _insert_sample(conn, gs, "locked01sid", locked=True, user_id=uid)
    _insert_member(conn, "locked01cid", "locked01sid", user_id=uid)

    h = bearer(raw)
    assert _patch(client, raw, "locked01cid", status="reviewed").status_code == 409
    assert client.delete(
        "/captures/api/locked01cid", headers=h,
    ).status_code == 409
    assert client.post(
        "/captures/api/locked01cid/reprocess", headers=h,
    ).status_code == 409
    row = captures_store.get_capture("locked01cid")
    assert row is not None and row["status"] == "new"


def test_nonadmin_can_unlock_but_not_edit_a_locked_sample(client, make_user_key):
    """`is_locked` is writable by any captures-scoped caller, so it must also
    be RELEASABLE by them — otherwise it is a one-way switch only an admin can
    undo. A non-admin may send the unlock and nothing else while locked."""
    _, uid, raw, conn = _root_and_alice(make_user_key)
    _insert_sample(conn, gs, "unlock01sid", locked=True, user_id=uid)
    path = "samples/unlock01sid"

    # Any other edit stays frozen...
    assert _patch(client, raw, path, status="reviewed").status_code == 409
    # ...including an unlock smuggled alongside one.
    assert _patch(client, raw, path, is_locked=False,
                  status="reviewed").status_code == 409
    assert gs.get_sample("unlock01sid")["is_locked"] == 1

    # The bare unlock goes through.
    assert _patch(client, raw, path, is_locked=False).status_code == 200
    assert gs.get_sample("unlock01sid")["is_locked"] == 0

    # And once unlocked, ordinary edits work again.
    assert _patch(client, raw, path, status="reviewed").status_code == 200


def test_locked_member_view_does_not_rewrite_text(
        client, make_user_key, app_module, monkeypatch):
    """Viewing a locked sample's member — or the sample itself — must not
    self-heal-rewrite the member's stored text: the lock freezes what was
    curated. An UNLOCKED member still self-heals on view (contrast case)."""
    _, uid, raw, conn = _root_and_alice(make_user_key)
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
    raw_root, uid_a, raw_a, conn = _root_and_alice(make_user_key)
    uid_b, _raw_b = make_user_key("bob", pages={"captures": "own"})
    _insert_member(conn, "alicecap0001", None, user_id=uid_a)
    _insert_member(conn, "bobcap000001", None, user_id=uid_b)
    _insert_member(conn, "bobcap000002", None, user_id=uid_b)

    body = _get(client, raw_a, "list").json()
    assert body["total_count"] == 1
    assert body["counts"]["new"] == 1

    admin_body = _get(client, raw_root, "list").json()
    assert admin_body["total_count"] == 3
    assert admin_body["counts"]["new"] == 3


def test_samples_are_paged_and_the_cursor_walks_every_row(client, make_user_key):
    _, uid, raw, conn = _root_and_alice(make_user_key)
    # Deliberately give two of them an IDENTICAL created_ts that straddles the
    # first page boundary (page 1 = [5.0, the higher-id 3.0]): samples merged
    # in one call share a timestamp, and a timestamp-only cursor drops the
    # other 3.0 row.
    stamps = [5.0, 3.0, 3.0, 2.0, 1.0]
    for i, ts in enumerate(stamps):
        _insert_sample(conn, gs, f"page{i}sid00000", locked=False, ts=ts,
                       user_id=uid)

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
    assert len(set(seen)) == len(stamps)
    # Newest-first across page boundaries.
    order = [stamps[int(s[4])] for s in seen]
    assert order == sorted(order, reverse=True)


def test_last_page_reports_no_cursor(client, make_user_key):
    _, uid, raw, conn = _root_and_alice(make_user_key)
    for i in range(2):
        _insert_sample(conn, gs, f"exact{i}sid0000", locked=False,
                       ts=float(i), user_id=uid)

    body = _get(client, raw, "samples?limit=2").json()
    assert len(body["samples"]) == 2
    assert body["next"] is None


def test_by_request_id_audits_cross_user_read(client, make_user_key, caplog):
    """The by-request lookup hands a scope=all NON-admin the full capture row
    (raw/final text + the owner's username) for someone else's capture, so it
    is audited like the read-by-id siblings — DSARs are answered from this log
    stream."""
    import logging

    _, uid_owner, _, conn = _root_and_alice(make_user_key)
    _uid_v, raw_viewer = make_user_key("viewer", pages={"captures": "all"})
    _insert_member(conn, "byreqcap0001", None, user_id=uid_owner,
                   request_id="req-xyz")

    with caplog.at_level(logging.INFO, logger="whisper-api"):
        r = _get(client, raw_viewer, "by-request/req-xyz")
    assert r.status_code == 200
    assert len(r.json()["captures"]) == 1
    audit = [m for m in caplog.messages if "cross-user-read" in m]
    assert audit and "capture-by-request" in audit[0]

    # Self-reads must stay silent (same rule the siblings follow).
    caplog.clear()
    uid_self, raw_self = make_user_key("solo", pages={"captures": "all"})
    _insert_member(conn, "byreqcap0002", None, user_id=uid_self,
                   request_id="req-own")
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        _get(client, raw_self, "by-request/req-own")
    assert not [m for m in caplog.messages if "cross-user-read" in m]


def test_propose_merges_audits_cross_user_read(client, make_user_key,
                                               monkeypatch, caplog):
    """Proposals carry other users' capture previews + resolved usernames to a
    scope=all non-admin; that read is audited like the read-by-id siblings."""
    import logging

    uid_owner, _raw_owner = locked_user(make_user_key, pages={"captures": "own"})
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
        r = _get(client, raw_viewer, "propose-merges")
    assert r.status_code == 200
    audit = [m for m in caplog.messages if "cross-user-read" in m]
    assert audit and "merge-proposal" in audit[0]


def test_propose_merges_narrows_to_several_picked_speakers(client, make_user_key,
                                                          root_key, monkeypatch):
    """Two or more picked speakers go to the proposer as one pick list. The
    all-users sweep is already cut to the global top proposals, so filtering
    it afterwards lets a busy third speaker leave the picked ones with none."""
    _uid_o, raw_own = make_user_key("solo", pages={"captures": "own"})
    seen = []
    monkeypatch.setattr(
        captures_routes.captures_merge_proposer, "propose_merges",
        lambda **kw: (seen.append(kw["user_id_filter"]) or [], False))

    for q in ("?user_id=alice,%20carol", "?user_id=bob", "?user_id=,", ""):
        r = _get(client, root_key, "propose-merges" + q)
        assert r.status_code == 200
    _get(client, raw_own, "propose-merges?user_id=alice,carol")
    assert seen == [["alice", "carol"], "bob", None, None, None]

    too_many = ",".join(f"u{i}" for i in range(captures_routes._MAX_OWNER_FILTER_IDS + 1))
    r = _get(client, root_key, "propose-merges?user_id=" + too_many)
    assert r.status_code == 422


def test_preview_merge_audio_is_not_cacheable(client, root_key,
                                              monkeypatch, tmp_path):
    """The merged preview WAV is PHI behind a per-row owner check. FileResponse
    alone sends only ETag/Last-Modified, which makes the body heuristically
    cacheable by any shared cache in front of the app — the two sibling audio
    routes both send Cache-Control: no-store."""
    from faster_whisper_backend.captures import merge as audio_merge

    monkeypatch.setattr(
        captures_routes, "_validate_merge_payload",
        lambda ids, silence_ms, user: ([], "root", ["/nonexistent.wav"], 0))

    def _fake_merge(paths, dst, **kw):
        _write_silent_wav(dst, 16)
        return {"duration_ms": 1}

    monkeypatch.setattr(audio_merge, "merge_wavs", _fake_merge)

    r = _post_members(client, root_key, "/captures/api/samples/preview-audio",
                      ["prevcap00001"])
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"


def test_preview_merge_audio_removes_its_temp_file(
        client, root_key, monkeypatch):
    """The merged WAV is gone once the preview answer is sent."""
    from faster_whisper_backend.captures import merge as audio_merge

    monkeypatch.setattr(
        captures_routes, "_validate_merge_payload",
        lambda ids, silence_ms, user: ([], "root", ["/nonexistent.wav"], 0))
    written = []

    def _fake_merge(paths, dst, **kw):
        written.append(dst)
        _write_silent_wav(dst, 16)
        return {"duration_ms": 1}

    monkeypatch.setattr(audio_merge, "merge_wavs", _fake_merge)
    r = _post_members(client, root_key, "/captures/api/samples/preview-audio",
                      ["prevcap00001"])
    assert r.status_code == 200
    assert len(written) == 1
    assert not os.path.exists(written[0])


@pytest.mark.parametrize("range_header, code", [
    (None, 200), ("bytes=0-3", 206), ("bytes=99999-", 416), ("bytes=abc", 400)])
def test_temp_file_response_unlinks_on_every_range_answer(
        tmp_path, range_header, code):
    """Starlette runs a FileResponse's background task only after a normal
    body send; on its 400/416 Range answers the merged WAV stayed in the temp
    dir for good. Driven through a GET: Starlette >= 1.10 ignores Range on
    a POST like preview-audio, but an older one (fastapi allows it) does not."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from faster_whisper_backend.captures.routes import _TempFileResponse

    path = tmp_path / "merged.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 60)
    app = Starlette(routes=[Route(
        "/f", lambda request: _TempFileResponse(path=str(path),
                                                media_type="audio/wav"))])
    headers = {"Range": range_header} if range_header else {}
    r = TestClient(app).get("/f", headers=headers)
    assert r.status_code == code
    assert not path.exists()


def test_audio_rate_limit_is_hot_and_per_identity(client, app_module,
                                                  monkeypatch, make_user_key):
    """The cap is read from config on every call, so a test can lower it to 2
    and raise it to 0 (= unlimited) without restarting anything. A missing cid
    404s, but only AFTER the limiter runs — which is what we are measuring.
    The budget is per identity: a second caller keeps its own after the
    first one is throttled."""
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

    # A different identity (a keyed user, charged by user id, not by the
    # anonymous caller's host) is not throttled by the first one's bucket.
    _uid, raw = make_user_key("root", is_admin=True)
    assert _get(client, raw, "nope0001/audio").status_code == 404

    # 0 = unlimited, applied to the very next request with no reset.
    monkeypatch.setattr(app_module.cfg, "CAPTURES_AUDIO_RATE_PER_MIN", 0,
                        raising=False)
    for _ in range(20):
        assert _get(client, raw, "nope0001/audio").status_code == 404


def _export_tar(only_status, include_audio):
    """The export as an open TarFile (the caller uses it as a context manager)."""
    blob = b"".join(captures_routes._build_export_stream(only_status, include_audio))
    return tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")


def _manifest_rows(tar):
    text = tar.extractfile("manifest.jsonl").read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _export_manifest(only_status="ready"):
    with _export_tar(only_status, False) as tar:
        return _manifest_rows(tar)


def _mark_trimmed(captures_store, cid, trimmed_rel):
    """A legacy VAD trim of a 12.4 s capture: 2 s lead + 1.3 s trail, 9.1 s left."""
    with captures_store._lock:
        with captures_store._require_conn() as conn:
            conn.execute(
                "UPDATE captures SET audio_s = 12.4, audio_trimmed_relpath = ?,"
                " audio_trim_lead_ms = 2000, audio_trim_trail_ms = 1300"
                " WHERE id = ?", (trimmed_rel, cid))


def _ready_capture(captures_store, monkeypatch, tmp_path, *, language="de", translations=None):
    cid = _audio_capture(
        captures_store, monkeypatch, tmp_path, language=language,
        final="quelle", translations=translations, translation_model="HY-MT",
        translation_source="cascade-mt",
    )
    captures_store.update_capture(cid, {"status": "ready"})
    return cid


def test_export_matches_english_track_by_base_subtag(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """`en-US` is an accepted translate target and is stored under that key,
    so the exporter matches the English track by its base subtag, not "en"."""
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


def test_export_skips_a_capture_whose_training_text_the_pipeline_blanked(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """A removal rule that blanked a Whisper hallucination over near-silence
    leaves text_for_training "" (not NULL): the export used to fall back to
    the raw hallucination and train the model to emit it over silence."""
    cid = _audio_capture(
        captures_store_db, monkeypatch, tmp_path,
        raw="Untertitel im Auftrag des ZDF", final="", text_for_training="",
    )
    captures_store_db.update_capture(cid, {"status": "ready"})
    assert _export_manifest() == []


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
    _mark_trimmed(captures_store, cid, trimmed_rel)
    rows = _export_manifest()
    assert [r["task"] for r in rows] == ["transcribe", "translate"]
    assert [r["duration"] for r in rows] == [pytest.approx(9.1)] * 2


def test_export_falls_back_to_the_original_when_the_trimmed_file_is_gone(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """A stale audio_trimmed_relpath (partial restore, manual cleanup) must not
    drop the row: like the audio endpoint, the export packs the original at
    its full length."""
    captures_store = captures_store_db
    cid = _ready_capture(captures_store, monkeypatch, tmp_path,
                         language="de", translations={})
    row = captures_store.get_capture(cid)
    _mark_trimmed(captures_store, cid, row["audio_relpath"] + ".trim.wav")
    rows = _export_manifest()
    assert [r["duration"] for r in rows] == [pytest.approx(12.4)]
    with _export_tar("ready", True) as tar:
        assert [n for n in tar.getnames() if n.startswith("audio/")] == [
            f"audio/{cid}.wav"]


def _write_merged_wav(gs, sid, data):
    return write_bytes(gs.abs_path_for(gs.get_sample(sid)["merged_wav_relpath"]), data)


def test_export_rereads_a_sample_rebuilt_after_its_list_snapshot(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """The export walks a list_samples() snapshot, and a VAD reprocess running
    alongside a long export os.replaces the merged WAV and patches
    merged_duration_ms after it: the manifest shipped the new audio with the
    old duration, and a sample the pass flagged stale was exported anyway.
    Each group is re-read under its rebuild lock before it is packed."""
    rebuilt, staled = "b" * 32, "c" * 32
    for sid in (rebuilt, staled):
        _grouped_capture(captures_store_db, monkeypatch, tmp_path, sid)
        _write_merged_wav(gs, sid, b"old audio")
    real_list = gs.list_samples

    def _list_then_reprocess(**kw):
        snapshot = real_list(**kw)
        _write_merged_wav(gs, rebuilt, b"new, longer audio")
        gs.update_sample(rebuilt, {"merged_duration_ms": 2500})
        gs.update_sample(staled, {"is_stale": 1})
        return snapshot

    entered = []
    real_lock = capture_samples._rebuild_lock

    def _counting(s):
        entered.append(s)
        return real_lock(s)

    monkeypatch.setattr(gs, "list_samples", _list_then_reprocess)
    monkeypatch.setattr(capture_samples, "_rebuild_lock", _counting)
    with _export_tar(None, True) as tar:
        rows = _manifest_rows(tar)
        assert [n for n in tar.getnames() if n.startswith("audio/")] == [
            f"audio/{rebuilt}.wav"]
        audio = tar.extractfile(f"audio/{rebuilt}.wav").read()
    assert [(r["audio_filepath"], r["duration"]) for r in rows] == [
        (f"audio/{rebuilt}.wav", pytest.approx(2.5))]
    assert audio == b"new, longer audio"
    assert sorted(entered) == [rebuilt, staled]


def test_export_joins_the_text_with_the_strategy_of_the_reread_sample(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """The text was built from the list_samples() snapshot before the rebuild
    lock, so a VAD reprocess that restamped transcript_join_strategy after it
    shipped the new WAV with text joined by the old separator. Members and
    text come from the row re-read under the lock."""
    sid = "e" * 32
    cids = [_ready_capture(captures_store_db, monkeypatch, tmp_path) for _ in range(2)]
    captures_routes._insert_sample_with_sid(
        sid=sid, **_sample_kwargs(cids, "quelle quelle"))
    _write_merged_wav(gs, sid, b"old audio")
    real_list = gs.list_samples

    def _list_then_reprocess(**kw):
        snapshot = real_list(**kw)
        _write_merged_wav(gs, sid, b"new audio")
        gs.update_sample(sid, {"transcript_join_strategy": "period_space",
                               "transcript": "quelle. quelle"})
        return snapshot

    monkeypatch.setattr(gs, "list_samples", _list_then_reprocess)
    with _export_tar(None, True) as tar:
        rows = _manifest_rows(tar)
        assert tar.extractfile(f"audio/{sid}.wav").read() == b"new audio"
    assert [(r["text"], r["member_count"]) for r in rows] == [
        ("quelle. quelle", 2)]


def test_export_skips_a_sample_whose_members_no_longer_match_its_wav(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """delete_capture commits the member's row DELETE before its auto-dissolve
    takes the rebuild lock (and swallows a dissolve failure), so the export
    could pack the 2-member WAV with the surviving member's text and
    member_count 1. A sample with no recorded build (empty member_hashes)
    is still exported."""
    drifted, legacy = "e" * 32, "f" * 32
    for sid, hashed in ((drifted, True), (legacy, False)):
        cids = [_ready_capture(captures_store_db, monkeypatch, tmp_path)
                for _ in range(2)]
        over = {} if hashed else {"member_hash_map": {}}
        captures_routes._insert_sample_with_sid(
            sid=sid, **_sample_kwargs(cids, "quelle quelle", **over))
        _write_merged_wav(gs, sid, b"both members audio")
    gone = gs.get_members(drifted)[1]["id"]
    with captures_store_db._lock:
        with captures_store_db._require_conn() as conn:
            conn.execute("DELETE FROM captures WHERE id = ?", (gone,))
    with _export_tar(None, True) as tar:
        rows = _manifest_rows(tar)
        assert [n for n in tar.getnames() if n.startswith("audio/")] == [
            f"audio/{legacy}.wav"]
    assert [(r["audio_filepath"], r["member_count"]) for r in rows] == [
        (f"audio/{legacy}.wav", 2)]


def test_export_sizes_each_wav_from_the_file_it_opened(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """info.size came from a getsize before the open, so an os.replace landing
    in between (a rebuild, a VAD trim) made tarfile copy the old byte count
    out of the new file: a shorter one aborted the whole streamed download
    with "unexpected end of data". Both the group and the singleton site."""
    import builtins

    sid = "d" * 32
    _grouped_capture(captures_store_db, monkeypatch, tmp_path, sid)
    group_abs = _write_merged_wav(gs, sid, b"old merged audio, long")
    cid = _ready_capture(captures_store_db, monkeypatch, tmp_path)
    single_abs = captures_store_db.abs_audio_path(
        captures_store_db.get_capture(cid)["audio_relpath"])
    swaps = {group_abs: b"short", single_abs: b"tiny"}

    def _replace_then_open(path, *a, **kw):
        data = swaps.pop(path, None)
        if data is not None:
            with builtins.open(path + ".new", "wb") as f:
                f.write(data)
            os.replace(path + ".new", path)
        return builtins.open(path, *a, **kw)

    monkeypatch.setattr(captures_routes, "open", _replace_then_open,
                        raising=False)
    with _export_tar(None, True) as tar:
        assert swaps == {}
        assert tar.extractfile(f"audio/{sid}.wav").read() == b"short"
        assert tar.extractfile(f"audio/{cid}.wav").read() == b"tiny"


def test_detail_keeps_original_timeline_when_the_trimmed_file_is_gone(
        client, monkeypatch, tmp_path):
    """The audio route falls back to the original when the trimmed WAV is
    gone, but the detail routes still shifted every word back by lead_ms and
    reported the trimmed length: the karaoke ran 2 s ahead of the speech and
    every word past 9.1 s collapsed onto 9.1 s. They project onto the trimmed
    timeline only while the trimmed file the audio route serves exists."""
    cid = _audio_capture(
        captures_store, monkeypatch, tmp_path, request_id="trim-gone",
        audio_s=12.4, raw="eins zehn", final="eins zehn",
        words=[{"word": "eins", "start": 3.0, "end": 3.5},
               {"word": " zehn", "start": 10.0, "end": 10.5}],
    )
    rel = captures_store.get_capture(cid)["audio_relpath"]
    _mark_trimmed(captures_store, cid, rel + ".trim.wav")

    cap = client.get(f"/captures/api/{cid}").json()["capture"]
    assert cap["effective_audio_s"] == pytest.approx(12.4)
    assert [(w["start"], w["end"]) for w in cap["words"]] == [
        (3.0, 3.5), (10.0, 10.5)]
    [hit] = client.get("/captures/api/by-request/trim-gone").json()["captures"]
    assert hit["effective_audio_s"] == pytest.approx(12.4)

    with open(captures_store.abs_audio_path(rel + ".trim.wav"), "wb") as f:
        f.write(b"RIFF")
    cap = client.get(f"/captures/api/{cid}").json()["capture"]
    assert cap["effective_audio_s"] == pytest.approx(9.1)
    assert [(w["start"], w["end"]) for w in cap["words"]] == [
        pytest.approx((1.0, 1.5)), pytest.approx((8.0, 8.5))]


def test_rebuild_lock_survives_prune_while_in_flight():
    """A Lock handed out but not yet acquired must not be pruned: the prune in
    _get_rebuild_lock skips sids pinned in _rebuild_inflight, so a second
    caller for the same sid gets the SAME object instead of minting a new one
    (which would let two rebuilds run concurrently)."""
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
    sid = "sid-ctx-pin"
    with capture_samples._rebuild_lock(sid):
        assert capture_samples._rebuild_inflight.get(sid) == 1
        assert capture_samples._rebuild_locks[sid].locked()
    assert sid not in capture_samples._rebuild_inflight
    assert not capture_samples._rebuild_locks[sid].locked()
    capture_samples._release_rebuild_lock(sid)
    assert sid not in capture_samples._rebuild_locks


def _sample_kwargs(cids, transcript="quelle", **over):
    """_insert_sample_with_sid's arguments (all but sid) for alice's `cids`."""
    return {"user_id": "alice", "member_ids": cids, "transcript": transcript,
            "join_strategy": "space", "silence_ms": 300,
            "member_hash_map": {c: "h" for c in cids}, "duration_ms": 1000,
            "language": "de", "member_trims": {}, **over}


def _grouped_capture(captures_store, monkeypatch, tmp_path, sid):
    """One ready capture packed into sample `sid`; returns the capture id."""
    cid = _ready_capture(captures_store, monkeypatch, tmp_path)
    captures_routes._insert_sample_with_sid(sid=sid, **_sample_kwargs([cid]))
    return cid


def test_sample_wav_self_heal_keeps_the_stale_flag(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """The VAD reprocess flags a sample stale when it no longer fits the
    current globals, and the export skips it. The missing-WAV self-heal
    rebuilds with the sample's stored silence, so it proves nothing about
    that, yet it wrote back is_stale=0: opening the audio of a stale sample
    whose WAV had gone re-admitted it to the next export."""
    sid = "d" * 32
    _grouped_capture(captures_store_db, monkeypatch, tmp_path, sid)
    gs.update_sample(sid, {"is_stale": 1})

    def _build(*, sid, member_ids, silence_ms):
        _write_merged_wav(gs, sid, b"rebuilt")
        return 1500, {m: "h2" for m in member_ids}, {}

    monkeypatch.setattr(capture_samples, "_build_merged_wav", _build)
    abs_p = captures_routes._ensure_sample_wav(gs.get_sample(sid))
    with open(abs_p, "rb") as f:
        assert f.read() == b"rebuilt"
    g = gs.get_sample(sid)
    assert g["merged_duration_ms"] == 1500
    assert g["is_stale"] is True


def test_reprocess_vad_worker_uses_pinned_rebuild_lock(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """The worker holds its per-sid lock across build + DB write for many
    sids in a loop, so it must take the pinning _rebuild_lock: with the raw
    _get_rebuild_lock a route's prune could drop the sid between handout and
    acquire and mint a SECOND Lock for a concurrent regenerate."""
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
    """create_sample_api awaits between validation and insert, so a
    double-submitted Merge can validate twice. The member UPDATE's rowcount
    is the gate: the loser gets a 409 and its capture_samples row is rolled
    back instead of persisting member-less."""
    captures_store = captures_store_db
    cid = _grouped_capture(captures_store, monkeypatch, tmp_path, "b" * 32)
    with pytest.raises(HTTPException) as ei:
        captures_routes._insert_sample_with_sid(sid="c" * 32, **_sample_kwargs([cid]))
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
    cid = _ready_capture(captures_store, monkeypatch, tmp_path)
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
    captures_routes._insert_sample_with_sid(sid=sid, **_sample_kwargs([cid]))
    assert seen == {"captures_lock_held": True}
    assert captures_store.get_capture(cid)["sample_id"] == sid


def test_insert_sample_with_sid_rolls_back_a_failed_commit(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """COMMIT sat outside the try: a commit-time SQLITE_FULL / IOERR left the
    transaction open on the shared autocommit connection, and every later
    BEGIN failed with "cannot start a transaction within a transaction"."""
    import sqlite3

    cid = _ready_capture(captures_store, monkeypatch, tmp_path)
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
    kwargs = _sample_kwargs([cid])
    with pytest.raises(sqlite3.OperationalError):
        captures_routes._insert_sample_with_sid(sid="e" * 32, **kwargs)
    assert failed and not real_conn.in_transaction
    assert real_conn.execute("SELECT COUNT(*) FROM capture_samples").fetchone()[0] == 0
    assert captures_store.get_capture(cid)["sample_id"] is None
    captures_routes._insert_sample_with_sid(sid="f" * 32, **kwargs)
    assert captures_store.get_capture(cid)["sample_id"] == "f" * 32


def test_create_sample_runs_the_insert_off_the_event_loop():
    """_insert_sample_with_sid blocks on both store locks, which clear_all
    holds across a full VACUUM; called inline it parked the event loop."""
    src = inspect.getsource(captures_routes.create_sample_api)
    assert ("await asyncio.to_thread(functools.partial(\n"
            "            _insert_sample_with_sid,") in src
    assert "        _insert_sample_with_sid(\n" not in src


def test_list_samples_projects_chip_offsets_without_hydrating_words(
        client, make_user_key, monkeypatch):
    """The list path projects member chips onto global word indices from
    get_members' word_count alone — no per-member get_capture (that was a
    full SELECT * + words JSON decode per member per page)."""
    _, uid, raw, conn = _root_and_alice(make_user_key)
    sid = "chipsid000000001"
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    _insert_member(conn, "chipmember00", sid, user_id=uid,
                   words=json.dumps([{"w": "a"}, {"w": "b"}]))
    _insert_member(conn, "chipmember01", sid, user_id=uid,
                   words=json.dumps([{"w": "c"}, {"w": "d"}, {"w": "e"}]),
                   corrections=json.dumps(
                       [{"idx": 1, "wrong": "a", "correct": "b"}]),
                   sample_order=1)

    monkeypatch.setattr(
        captures_store, "get_capture",
        lambda cid: pytest.fail("list path must not hydrate members"))
    body = _get(client, raw, "samples").json()
    groups = [g for g in body["samples"] if g["id"] == sid]
    assert len(groups) == 1
    assert groups[0]["corrections"] == [
        {"idx": 3, "wrong": "a", "correct": "b"}]


def test_list_samples_carries_what_the_filters_match_on(client, make_user_key):
    """A group has no model or request of its own, so the list projects both
    from its members; without them the page's model filter and search box
    never hid a group."""
    _, uid, raw, conn = _root_and_alice(make_user_key)
    sid = "filtersid0000001"
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    _insert_member(conn, "filtmember00", sid, user_id=uid,
                   request_id="req-a", model="org/model-a")
    _insert_member(conn, "filtmember01", sid, user_id=uid,
                   model="org/model-b", sample_order=1)

    body = _get(client, raw, "samples").json()
    g = next(g for g in body["samples"] if g["id"] == sid)
    assert g["models"] == ["org/model-a", "org/model-b"]
    assert g["member_ids"] == ["filtmember00", "filtmember01"]
    assert g["member_request_ids"] == ["req-a"]


def test_page_search_matches_ids_and_filters_groups(html):
    """Source pins: the search matches the capture id and request id (what
    the log block prints as `captured=` / `req=`) by PREFIX, kept out of the
    text haystack — a substring match on 32-hex ids let any hex-only word hit
    random rows — and group cards go through the model + search filters like
    capture cards do."""
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
    conn = captures_store._require_conn()
    cid = "origaudio001"
    rel = captures_store._relpath_for(cid, "wav")
    trel = os.path.join(cid[0:2], cid[2:4], f"{cid}.trim.wav")
    _insert_member(conn, cid, None, audio_trimmed_relpath=trel)
    for r, payload in ((rel, b"RIFF....WAVEoriginal"), (trel, b"RIFF....WAVEtrimmed")):
        write_bytes(captures_store.abs_audio_path(r), payload)

    r = client.get(f"/captures/api/{cid}/audio")
    assert r.status_code == 200 and r.content.endswith(b"trimmed")
    r = client.get(f"/captures/api/{cid}/audio?original=1")
    assert r.status_code == 200 and r.content.endswith(b"original")
    assert f"{cid}.original.wav" in r.headers["content-disposition"]
    assert r.headers["cache-control"] == "no-store"


def test_capture_card_offers_both_audio_downloads(html):
    assert "compact-player-btn compact-player-dl" in html
    assert "audioUrl + '?original=1'" in html
    assert "a.compact-player-dl {" in html


def test_page_merge_gate_cancels_a_stale_trim_estimate(html):
    """A pending merge-estimate timer / in-flight POST from an earlier valid
    selection must not re-enable Merge once the selection turns invalid (too
    many rows, mixed speakers, in-sample) or empty: both paths cancel it."""
    bar = js_upto(
        html, "function _updateActionBar()", "function _cancelTrimEstimate()")
    assert "if (n === 0) { _cancelTrimEstimate(); return; }" in bar
    assert bar.count("_cancelTrimEstimate();") == 2
    assert "n <= MERGE_MAX_MEMBERS" in bar and "var MERGE_MAX_MEMBERS = 30;" in html
    cancel = js_upto(
        html, "function _cancelTrimEstimate()", "function _fetchTrimEstimate(")
    assert "_meterEstimateToken++;" in cancel and "clearTimeout(_meterEstimateTimer)" in cancel


def test_page_translation_eligibility_mirrors_the_exporter(html):
    """The exporter skips the en track of an English-source capture and of a
    translate-task row; the card must not tag either as training data."""
    assert ("var eligible = enKey && !srcIsEn && !isTr && trs[enKey]"
            " && String(trs[enKey]).trim();") in html
    assert "var srcIsEn = String(r.language || '')" in html
    assert "var isTr = (r.task || 'transcribe') === 'translate';" in html
    assert "(eligible && lg === enKey ? ' cc-tr-eligible' : '')" in html


def test_page_merge_gate_mirrors_the_language_and_task_partition(html):
    bar = js_upto(
        html, "function _updateActionBar()", "function _cancelTrimEstimate()")
    assert "var mixedLangTask = langs.size > 1 || tasks.size > 1;" in bar
    assert "&& !mixedLangTask;" in bar


def test_page_unsaved_edits_guard_covers_groups_and_bulk_actions(html):
    """Every render() rebuilds open capture cards from the server and
    collapses open group cards. The "Load more groups" guard only saw capture
    cards, and the bulk Mark / Undo / Delete paths had no guard at all, so
    unsaved chip / notes edits vanished without a confirm."""
    guard = js_upto(
        html, "function _anyOpenDirty()", "async function _okToDropOpenEdits(")
    assert "_openRows[id]" in guard and "st.dirty" in guard
    assert "_openSamples[id]" in guard and "st.state && st.state.dirty" in guard
    assert "_openSamples[g.id] = { audio: audio, state: sampleState };" in html
    for fn, nxt in (("async function loadMoreSamples()", "_samplesLoading = true;"),
                    ("async function _bulkStatus(", "await api("),
                    ("async function _undoBulk(", "await api("),
                    ("async function _bulkDelete()", "await api(")):
        body = js_upto(html, fn, nxt)
        assert "_okToDropOpenEdits(" in body, fn


def test_page_filters_and_background_reloads_ask_before_dropping_edits(html):
    """The filter controls (status, speaker, model, search, Refresh) end in
    render() / load(), which rebuild open capture cards from a fresh GET and
    collapse group cards: they ask first and revert on a cancel. A finished
    background job or modal action skips its reload while a card is dirty,
    and loadMoreSamples renders only through that dirty-card check, never in
    `finally`."""
    helper = js_upto(html, "async function _filterChange(apply, revert)",
                     "function _loadUnlessDirty(msg, err)")
    assert "await _okToDropOpenEdits(" in helper
    assert "if (ok) apply(); else revert();" in helper
    unless = js_body(html, "function _loadUnlessDirty(msg, err)")
    assert "if (!_anyOpenDirty()) {" in unless and "return load();" in unless
    assert unless.index("return load();") < unless.index("List not refreshed")
    wire = js_upto(html, "var grp = buildStatusButtonGroup(",
                   "document.getElementById('btn-export')")
    assert wire.count("_filterChange(") == 4
    assert "onChange: function(ids) { _filtSpeakers = ids;" not in wire
    assert ("addEventListener('click', async function() {\n"
            "    if (await _okToDropOpenEdits('Refresh the list?'") in wire
    assert "getElementById('btn-refresh').addEventListener('click', load)" not in wire
    poll = js_upto(
        html, "function _pollJob(statusUrl, render)", "}).catch(function(e) {")
    assert "_loadUnlessDirty(failed ?" in poll
    assert "load();" not in poll and "toast(" not in poll
    commit = js_upto(html, "getElementById('merge-commit').onclick",
                     "api('POST', '/captures/api/samples'")
    assert "await _okToDropOpenEdits(" in commit
    dissolve = js_upto(
        html, "dissolveBtn.onclick = async function() {", "api('DELETE'")
    assert "await _okToDropOpenEdits(" in dissolve
    assert "        load();   // refresh main captures list" not in html
    assert "    load();              // refresh main captures list" not in html
    more = js_upto(
        html, "async function loadMoreSamples()", "async function reloadCounts()")
    fin = more[more.index("} finally {"):more.index("if (!loaded) return;")]
    assert "render();" not in fin
    tail = more[more.index("if (!loaded) return;"):]
    assert "await _renderUnlessKeptOpenEdits(agreed, 'Show the loaded groups?'" in tail


def test_page_render_drops_hidden_rows_from_the_selection(html):
    """A filter or search change kept hidden ids selected, so a bulk Dismiss /
    Delete reached rows no longer on screen."""
    r = js_upto(html, "render = function() {", "list.innerHTML = '';")
    assert "_selection.delete(id)" in r
    assert "if (!r.sample_id) selectable[r.id] = true;" in r
    assert "_updateActionBar();" in r


def test_page_error_toasts_read_a_422_detail_list(html):
    """FastAPI's 422 `detail` is a list of {msg, ...}; assigning it to the
    toast text printed "[object Object]"."""
    helper = html[html.index("window._detailMsg=function(j,fallback){"):]
    assert "typeof d==='string'" in helper
    assert "Array.isArray(d)" in helper and "d[0].msg" in helper
    assert re.search(r"\b\w*msg = \w+\.detail;", html) is None
    assert html.count("_detailMsg(await ") == 1
    assert html.count("_respErrMsg(") == 5
    assert "function _detailMsg(" not in html


def test_page_load_drops_a_superseded_filter_response(html):
    """Out-of-order /list responses for two quick speaker-picker ticks must
    not leave the older filter's rows on screen."""
    load = js_upto(html, "async function load(", "async function loadMoreSamples()")
    assert "var seq = ++_loadSeq;" in load
    assert load.count("if (seq !== _loadSeq) return;") >= 3
    more = js_upto(
        html, "async function loadMoreSamples()", "async function reloadCounts()")
    assert "var seq = _loadSeq;" in more and "if (seq !== _loadSeq) return;" in more
    # reloadCounts (after a bulk status / undo / delete) is filter-scoped too,
    # and a total_count of 0 is a real count, not a missing field.
    counts = js_upto(html, "async function reloadCounts()", "reloadStats();")
    assert "var seq = _loadSeq;" in counts and "if (seq !== _loadSeq) return;" in counts
    assert "if (typeof j.total_count === 'number') _totalCount = j.total_count;" in counts


def test_page_group_autoload_skips_open_capture_cards(html):
    """loadMoreSamples ends in a full render(), which rebuilds every open
    capture card from the server: the scroll auto-trigger must hold off
    while one is open, and a manual load asks before dropping unsaved
    edits. Under audio_missing (no group has it) there is no footer."""
    foot = js_upto(html, "function appendMoreFooter(list) {", "io.observe(more);")
    assert "_filtStatus === 'audio_missing') return;" in foot
    assert "|| Object.keys(_openRows).length !== 0) return;" in foot
    more = js_upto(
        html, "async function loadMoreSamples()", "async function reloadCounts()")
    assert (more.index("await _okToDropOpenEdits('Load more groups?'")
            < more.index("var agreed = _dirtyOpenEdits();")
            < more.index("await _renderUnlessKeptOpenEdits(agreed"))


def test_sample_save_does_not_resend_status(html):
    """The group's status buttons auto-save a narrow PATCH; Save resending
    the status loaded with the view reverted another tab's change."""
    save = js_upto(
        html, "saveTBtn.onclick = function() {", "saveTBtn.disabled = true;")
    assert "payload.admin_notes = sampleState.adminNotes;" in save
    assert "status:" not in save


def _mixed_pair(task_b=None, lang_b="de"):
    """Two same-user captures with real WAVs on disk, differing in the second
    member's task / language."""
    conn = captures_store._require_conn()
    ids = ["mixpair0000a", "mixpair0000b"]
    for cid in ids:
        _insert_member(conn, cid, None, user_id="alice")
        p = captures_store.abs_audio_path(captures_store._relpath_for(cid, "wav"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        _write_silent_wav(p, 1600)
    conn.execute("UPDATE captures SET task = ?, language = ? WHERE id = ?",
                 (task_b, lang_b, ids[1]))
    return ids


@pytest.mark.parametrize("task_b,lang_b,detail", [
    ("translate", "de", "members must all share one task"),
    (None, "en", "members must all share one language"),
])
def test_merge_rejects_mixed_task_or_language(client, root_key,
                                              task_b, lang_b, detail):
    """Same partition as the proposer: a sample has ONE language and ONE task
    label, so a de+en (or transcribe+translate) merge would export
    wrong-language training data."""
    ids = _mixed_pair(task_b, lang_b)
    for path in ("/captures/api/samples/merge-estimate", "/captures/api/samples"):
        r = _post_members(client, root_key, path, ids)
        assert r.status_code == 400, (path, r.text)
        assert r.json()["detail"] == detail


def test_merge_accepts_region_variants_of_one_language(client, root_key):
    ids = _mixed_pair(None, "de-CH")
    r = _post_members(client, root_key, "/captures/api/samples/merge-estimate", ids)
    assert r.status_code == 200, r.text


def test_regenerate_of_a_sample_dissolved_mid_wait_is_404(client, root_key,
                                                          monkeypatch):
    """dissolve_sample holds the rebuild lock, so a regenerate that waited on
    it must re-read the row and 404 instead of rebuilding an orphan WAV and
    500ing in _enrich_sample."""
    sid = "regendis0000001"
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
    r = client.post(f"/captures/api/samples/{sid}/regenerate", headers=bearer(root_key))
    assert r.status_code == 404, r.text
    assert built == []


def test_build_merged_wav_resolves_member_audio_from_the_light_row(
        client, monkeypatch):
    """Only audio_relpath is needed, under the rebuild lock: no full-row
    get_capture (a json.loads of the words + segments blobs) per member."""
    ids = _mixed_pair(None, "de")

    def _boom(cid):
        raise AssertionError("_build_merged_wav must not call get_capture")

    monkeypatch.setattr(captures_store, "get_capture", _boom)
    monkeypatch.setattr(capture_samples.cfg,
                        "CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES", False,
                        raising=False)
    duration_ms, hashes, _trims = capture_samples._build_merged_wav(
        sid="lightwav00001", member_ids=ids, silence_ms=100)
    assert sorted(hashes) == sorted(ids) and duration_ms > 0
    with pytest.raises(HTTPException) as ei:
        capture_samples._build_merged_wav(
            sid="lightwav00002", member_ids=[ids[0], "nosuchcid000"],
            silence_ms=100)
    assert ei.value.status_code == 404


def _de_scoped_pipeline(monkeypatch):
    """Stand-in for a de-only rule: lowercases when scoped to "de"."""
    langs: list = []

    def _pp(text, **kw):
        langs.append(kw.get("language"))
        return text.lower() if kw.get("language") == "de" else text

    monkeypatch.setattr(pl_engine, "_postprocess_text", _pp)
    return langs


def _translate_capture(conn, cid, user_id, *, task):
    raw = "It is over. Was it good?"
    _insert_member(conn, cid, None, user_id=user_id, raw_text=raw,
                   final_text=raw, text_for_training=raw, task=task)
    return raw


def test_reprocess_and_self_heal_scope_a_translate_capture_by_english(
        client, make_user_key, monkeypatch):
    """language stores the SPOKEN language; a task=translate capture's text
    is English, so /reprocess and the page-view self-heal must scope the
    pipeline by "en" (as the live run did) and leave its text alone. A
    transcribe twin still gets the de scope."""
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
    uid, raw_key = make_user_key("root", is_admin=True)
    _insert_member(captures_store._require_conn(), "patchthr0001", None,
                   user_id=uid)
    seen: list = []
    real = captures_store.update_capture

    def _spy(cid, patch):
        seen.append((_on_loop(), captures_routes._corrections_write_lock.locked()))
        return real(cid, patch)

    monkeypatch.setattr(captures_store, "update_capture", _spy)
    r = _patch(client, raw_key, "patchthr0001", admin_notes="n")
    assert r.status_code == 200, r.text
    r = _patch(client, raw_key, "patchthr0001", corrections=[],
               baseline_corrections=[])
    assert r.status_code == 200, r.text
    assert seen == [(False, False), (False, True)]
    # Guards still surface as their HTTP codes from inside the thread.
    assert _patch(client, raw_key, "nosuchcid000",
                  admin_notes="n").status_code == 404
    assert _patch(client, raw_key, "patchthr0001",
                  status="bogus").status_code in (400, 422)


def test_audio_and_delete_read_the_row_off_the_loop(client, make_user_key,
                                                    monkeypatch):
    """get_capture json.loads the words + segments blobs on the connection
    clear_all VACUUMs; the audio route (reviewers fire it in bursts), the
    card-expand detail route and the delete route must not run it on the
    event loop."""
    uid, raw_key = make_user_key("root", is_admin=True)
    _insert_member(captures_store._require_conn(), "offloop00001", None,
                   user_id=uid)
    seen: list = []
    real = captures_store.get_capture

    def _spy(cid):
        seen.append(_on_loop())
        return real(cid)

    monkeypatch.setattr(captures_store, "get_capture", _spy)
    h = bearer(raw_key)
    # No WAV on disk: the row is found, then the file probe answers 410.
    assert client.get("/captures/api/offloop00001/audio",
                      headers=h).status_code == 410
    assert client.get("/captures/api/offloop00001",
                      headers=h).status_code == 200
    assert client.delete("/captures/api/offloop00001",
                         headers=h).status_code == 200
    assert seen and not any(seen)


def test_sample_and_by_request_routes_read_the_store_off_the_loop():
    """The samples store shares the captures connection that clear_all holds
    across a VACUUM, so no route body may call one of these reads inline —
    only inside a helper handed to asyncio.to_thread."""
    import ast
    import textwrap

    banned = {"get_sample", "get_members", "find_by_request_id",
              "get_capture", "_admit_capture",
              "_assert_member_sample_not_locked"}

    def _inline_calls(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        route = tree.body[0]
        out, todo = [], list(route.body)
        while todo:
            node = todo.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.Lambda)):
                continue  # a helper body runs wherever it is handed
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(
                    f, "id", None)
                if name in banned:
                    out.append(name)
            todo.extend(ast.iter_child_nodes(node))
        return out

    for fn in (captures_routes.by_request_id_api, captures_routes.get_capture_api,
               captures_routes.delete_capture_api, captures_routes.get_sample_api,
               captures_routes.patch_sample_api, captures_routes.regenerate_sample_api,
               captures_routes.dissolve_sample_api, captures_routes.get_sample_audio_api):
        assert _inline_calls(fn) == [], fn.__name__


def test_patch_of_a_sample_dissolved_mid_apply_is_404(client, root_key,
                                                     monkeypatch):
    """A dissolve committed between the route's get_sample and _apply's
    update_sample leaves update_sample returning None — a 404, not a 500
    from _enrich_sample(None)."""
    sid = "patchdis000001"
    _insert_sample(captures_store._require_conn(), gs, sid, locked=False)
    monkeypatch.setattr(gs, "update_sample", lambda s, patch: None)
    r = _patch(client, root_key, f"samples/{sid}", admin_notes="x")
    assert r.status_code == 404, r.text


def test_sample_chip_save_holds_the_corrections_lock(client, make_user_key,
                                                     monkeypatch):
    """The sample PATCH's member read + per-member chip writes run under the
    same lock as a member's own PATCH, so neither can land between the
    other's read and write."""
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
    r = _patch(client, raw_key, f"samples/{sid}", corrections=[],
               baseline_corrections=[])
    assert r.status_code == 200, r.text
    assert held and held[0] is True


def test_a_group_holding_more_chips_than_one_member_may_saves(
        client, make_user_key):
    """A group's chips are the projection of every member's chips, so the
    group schemas must not bound it by the PER-MEMBER cap: two members with
    101 chips each made every group Save a 422 before the real per-member
    check could run. That per-member check is still the limit."""
    cap = text_corrections.CAP_CORRECTIONS
    uid, raw_key = make_user_key("root", is_admin=True)
    sid = "grpcap0000001"
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    n_words = cap + 1
    words = json.dumps([{"word": f"w{i}", "start": i * 0.1,
                         "end": i * 0.1 + 0.05} for i in range(n_words)])
    half = cap // 2 + 1
    chips = json.dumps([{"wrong": "", "correct": "x", "idx": i}
                        for i in range(half)])
    for order, cid in enumerate(("grpcapmem001", "grpcapmem002")):
        _insert_member(conn, cid, sid, user_id=uid, words=words,
                       corrections=chips, sample_order=order)
    projected = ([{"wrong": "", "correct": "x", "idx": i}
                  for i in range(half)]
                 + [{"wrong": "", "correct": "x", "idx": n_words + i}
                    for i in range(half)])
    assert len(projected) > cap
    r = _patch(client, raw_key, f"samples/{sid}", corrections=projected,
               baseline_corrections=projected)
    assert r.status_code == 200, r.text
    over = ([{"wrong": "", "correct": "y", "idx": i} for i in range(cap + 1)]
            + projected[half:])
    r = _patch(client, raw_key, f"samples/{sid}", corrections=over,
               baseline_corrections=projected)
    assert r.status_code == 422, r.text
    assert f"a member capture holds at most {cap}" in r.json()["detail"]


def test_preview_save_chips_writes_off_the_loop_and_skips_newly_grouped(
        client, root_key, monkeypatch):
    """Up to 30 update_capture calls must not run on the event loop, and a
    member that a concurrent create_sample grouped after the validation is
    left alone (its chips now belong to the sample)."""
    ids = _mixed_pair(None, "de")
    real_light = captures_store.get_captures_light

    def _light(cids):
        out = real_light(cids)
        out[ids[1]] = dict(out[ids[1]], sample_id="grabbedsid01")
        return out

    seen: list = []
    real_update = captures_store.update_capture

    def _spy(cid, patch):
        seen.append((cid, _on_loop(), captures_routes._corrections_write_lock.locked()))
        return real_update(cid, patch)

    monkeypatch.setattr(captures_store, "get_captures_light", _light)
    monkeypatch.setattr(captures_store, "update_capture", _spy)
    r = _post_members(client, root_key, "/captures/api/samples/preview-save-chips",
                      ids, corrections=[])
    assert r.status_code == 200, r.text
    assert seen == [(ids[0], False, True)]
    assert list(r.json()["saved"]) == [ids[0]]


def _two_word_member(conn, cid, sid, uid, order, chips):
    words = json.dumps([{"word": w, "start": i * 0.1, "end": i * 0.1 + 0.05}
                        for i, w in enumerate(("x", "y"))])
    _insert_member(conn, cid, sid, user_id=uid, words=words, sample_order=order)
    captures_store.update_capture(cid, {"corrections": chips})


def test_group_chip_save_keeps_a_members_anchorless_chip(
        client, make_user_key):
    """An anchorless chip (no idx — a PATCH that omitted it, or an idx the
    cleaner stripped) has no place in the group projection, so the split
    cannot hand it back: the REPLACE fan-out must keep it on any chip save
    of the group, even one editing a different member."""
    uid, raw_key = make_user_key("root", is_admin=True)
    sid = "anchorless0001"
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    free = {"wrong": "foo", "correct": "bar"}
    _two_word_member(conn, "anchlessmem1", sid, uid, 0,
                     [{"wrong": "x", "correct": "X", "idx": 0}, free])
    _two_word_member(conn, "anchlessmem2", sid, uid, 1,
                     [{"wrong": "x", "correct": "P", "idx": 0}])
    h = bearer(raw_key)
    proj = client.get(f"/captures/api/samples/{sid}",
                      headers=h).json()["sample"]["corrections"]
    assert [c["idx"] for c in proj] == [0, 2]
    edited = [proj[0], dict(proj[1], correct="Q")]
    r = _patch(client, raw_key, f"samples/{sid}", corrections=edited,
               baseline_corrections=proj)
    assert r.status_code == 200, r.text
    m1 = captures_store.get_capture("anchlessmem1")["corrections"]
    assert any(c["wrong"] == "foo" and c["correct"] == "bar"
               and c.get("idx") is None for c in m1)
    m2 = captures_store.get_capture("anchlessmem2")["corrections"]
    assert [c["correct"] for c in m2] == ["Q"]


def test_group_chip_save_stores_the_transcript_with_the_new_chips(
        client, make_user_key):
    """The re-derive after the chip fan-out read the cached members, which
    still held the pre-edit chips, so the stored transcript lagged one edit
    (and the next bulk reapply saw a spurious diff)."""
    uid, raw_key = make_user_key("root", is_admin=True)
    sid = "chiptrans0001"
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, sid, locked=False, user_id=uid)
    for order, cid in enumerate(("chiptransmem1", "chiptransmem2")):
        _two_word_member(conn, cid, sid, uid, order, [])
        conn.execute("UPDATE captures SET final_text = 'x y' WHERE id = ?",
                     (cid,))
    chip = {"wrong": "x", "correct": "Z", "idx": 0}
    r = _patch(client, raw_key, f"samples/{sid}", corrections=[chip],
               baseline_corrections=[])
    assert r.status_code == 200, r.text
    assert gs.get_sample(sid)["transcript"] == "Z y x y"


def test_preview_save_chips_keeps_a_members_anchorless_chip(
        client, root_key):
    """Same REPLACE fan-out on a not-yet-merged proposal: the anchorless chip
    of an unmerged member survives a chip save."""
    ids = _mixed_pair(None, "de")
    captures_store.update_capture(ids[0], {"corrections": [
        {"wrong": "foo", "correct": "bar"}]})
    r = _post_members(client, root_key, "/captures/api/samples/preview-save-chips",
                      ids, corrections=[])
    assert r.status_code == 200, r.text
    kept = captures_store.get_capture(ids[0])["corrections"]
    assert [(c["wrong"], c["correct"]) for c in kept] == [("foo", "bar")]
    assert r.json()["saved"][ids[0]] == 1


def test_patch_keeps_every_chip_the_schema_accepts(client, root_key):
    """The store keeps every chip the schema accepts (200): a lower store
    cap would toast "Saved." and lose the extra chips on reload."""
    _insert_member(captures_store._require_conn(), "chipcap00001", None)
    chips = [{"wrong": "w", "correct": f"c{i}", "idx": i} for i in range(51)]
    r = _patch(client, root_key, "chipcap00001", corrections=chips)
    assert r.status_code == 200, r.text
    assert len(r.json()["capture"]["corrections"]) == 51
    assert len(captures_store.get_capture("chipcap00001")["corrections"]) == 51


@pytest.mark.parametrize("field", ["wrong", "correct"])
def test_patch_refuses_chip_text_past_the_stored_cap(client, root_key, field):
    """The store keeps CAP_CORRECTION_FIELD chars of a chip's `wrong` and
    `correct`, so a longer one is a 422: saving it cut behind a 200 made the
    export rewrite only part of the span."""
    _insert_member(captures_store._require_conn(), "chipcap00003", None)
    cap = text_corrections.CAP_CORRECTION_FIELD
    chip = {"wrong": "w", "correct": "c", "idx": 0}
    r = _patch(client, root_key, "chipcap00003",
               corrections=[dict(chip, **{field: "x" * (cap + 1)})])
    assert r.status_code == 422
    assert captures_store.get_capture("chipcap00003")["corrections"] == []
    r = _patch(client, root_key, "chipcap00003",
               corrections=[dict(chip, **{field: "x" * cap})])
    assert r.status_code == 200, r.text
    assert captures_store.get_capture("chipcap00003")["corrections"][0][field] == "x" * cap


def test_patch_refuses_a_merge_that_grows_past_the_chip_cap(
        client, root_key):
    """A three-way merge can add the user's chips on top of a full stored
    list; that is a 422 naming the limit, not a silently shortened save."""
    _insert_member(captures_store._require_conn(), "chipcap00002", None)
    cap = text_corrections.CAP_CORRECTIONS
    full = [{"wrong": "w", "correct": f"c{i}", "idx": i} for i in range(cap)]
    captures_store.update_capture("chipcap00002", {"corrections": full})
    r = _patch(client, root_key, "chipcap00002",
               corrections=[{"wrong": "w", "correct": "x", "idx": cap + 5}],
               baseline_corrections=[])
    assert r.status_code == 422
    assert str(cap) in r.json()["detail"]
    assert len(captures_store.get_capture("chipcap00002")["corrections"]) == cap


def _two_member_sample(gs, sid):
    """A 'space'-joined sample with two one-word members (raw 'r', final 'f':
    the read path refreshes final from raw, the stored snapshot does not)."""
    conn = captures_store._require_conn()
    _insert_sample(conn, gs, sid, locked=False)
    for i, cid in enumerate((sid[:-1] + "1", sid[:-1] + "2")):
        _insert_member(conn, cid, sid, sample_order=i)


def test_regenerate_adopts_the_global_join_strategy(client, root_key,
                                                    monkeypatch):
    """The join strategy is a global setting that "applies to every new or
    regenerated sample": regenerate must restamp it (and the stored
    transcript), or every read and the export keep the old separator."""
    sid = "regenjoin00000"
    _two_member_sample(gs, sid)
    monkeypatch.setattr(cfg, "CAPTURES_SAMPLE_JOIN_STRATEGY", "period_space",
                        raising=False)
    monkeypatch.setattr(capture_samples, "_build_merged_wav",
                        lambda **kw: (1000, {}, {}))
    r = client.post(f"/captures/api/samples/{sid}/regenerate", headers=bearer(root_key))
    assert r.status_code == 200, r.text
    assert r.json()["sample"]["transcript"] == "r. r"  # read path: refreshed
    row = gs.get_sample(sid)
    assert row["transcript_join_strategy"] == "period_space"
    assert row["transcript"] == "f. f"


def test_reprocess_vad_adopts_the_global_join_strategy(client, monkeypatch):
    """The bulk reprocess is the all-samples regenerate: it restamps the
    global join strategy too."""
    sid = "vadjoin0000000"
    _two_member_sample(gs, sid)
    monkeypatch.setattr(cfg, "CAPTURES_SAMPLE_JOIN_STRATEGY", "period_space",
                        raising=False)
    monkeypatch.setattr(capture_samples, "_build_merged_wav",
                        lambda **kw: (1000, {}, {}))
    vr._run()
    assert vr.status()["rebuilt"] == 1
    row = gs.get_sample(sid)
    assert row["transcript_join_strategy"] == "period_space"
    assert row["transcript"] == "f. f"


def test_reprocess_vad_started_mid_run_rebuilds_again_with_the_new_settings(
        client, monkeypatch):
    """A second "Reprocess all · VAD silence" while a job runs queues one
    more pass: otherwise samples rebuilt before a settings edit keep the old
    silence layout while the job still ends "done"."""
    sid = "vadrerun000000"
    _two_member_sample(gs, sid)
    monkeypatch.setattr(cfg, "CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS", 300,
                        raising=False)
    builds = []

    def _build(**kw):
        builds.append(kw["silence_ms"])
        if len(builds) == 1:
            # The admin fixes the margin and clicks Reprocess again.
            monkeypatch.setattr(cfg, "CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS",
                                500, raising=False)
            assert vr.start()["status"] == "running"
        return 1000, {}, {}

    monkeypatch.setattr(capture_samples, "_build_merged_wav", _build)
    vr._state.update(status="running")
    vr._run()
    assert builds == [300, 500]
    assert gs.get_sample(sid)["inter_segment_silence_ms"] == 500
    st = vr.status()
    assert st["status"] == "done"
    assert (st["total"], st["processed"], st["rebuilt"]) == (1, 1, 1)
    assert vr._rerun_requested is False
