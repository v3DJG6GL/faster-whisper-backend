"""The _max_body_mw ceiling, and the tighter one it applies to JSON.

FastAPI parses `await request.json()` BEFORE solve_dependencies, so a JSON body
is buffered and json.loads-expanded ahead of the host gate, get_current_user and
any in-handler rate limiter. The middleware therefore caps a declared
application/json body far below MAX_REQUEST_BYTES; multipart audio uploads must
keep the full ceiling. A non-JSON body to any other route is buffered whole
before auth just the same (and no such route takes one), so it shares the
JSON ceiling; the raw-body media upload and the package route keep their own.
"""

import inspect
import json

from faster_whisper_backend.auth import dependencies as auth
from tests._transcription_helpers import post_json_ok

_JSON = {"Content-Type": "application/json"}


def _big_json(nbytes):
    return json.dumps({"blob": {"pad": "x" * nbytes}, "base_version": 0}).encode()


def _put_json(client, payload, chunked=False):
    if chunked:
        # No Content-Length: only the streaming byte counter can catch this.
        def _gen():
            for i in range(0, len(payload), 65536):
                yield payload[i:i + 65536]
        return client.put("/v1/synced-client-settings", content=_gen(), headers=_JSON)
    return client.put("/v1/synced-client-settings", content=payload, headers=_JSON)


def test_small_json_body_is_not_rejected(client):
    r = _put_json(client, _big_json(1024))
    assert r.status_code == 200, r.text


def test_desktop_clients_largest_sync_push_is_accepted(client):
    # The desktop client refuses to push a body past SYNC_MAX_BODY (4 MiB)
    # less its 4 KiB SYNC_RESPONSE_MARGIN (faster-whisper-frontend
    # transport/sync.rs), and client_settings' _CAP_BLOB sits at the same
    # figure. A body of exactly that size, envelope included, must clear the
    # JSON body clamp AND the store cap — a lower clamp would 413 every push
    # of a large but valid config.
    from faster_whisper_backend.client_settings import store
    limit = 4 * 1024 * 1024 - 4096
    shell = {"blob": {"pad": ""}, "base_version": 0, "device": "desk"}
    pad = limit - len(json.dumps(shell, separators=(",", ":")).encode())
    shell["blob"]["pad"] = "x" * pad
    payload = json.dumps(shell, separators=(",", ":")).encode()
    assert len(payload) == limit
    assert store._CAP_BLOB >= limit
    r = _put_json(client, payload)
    assert r.status_code == 200, r.text


def test_oversize_json_body_is_413(client):
    r = _put_json(client, _big_json(8 * 1024 * 1024))
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}


def test_oversize_json_is_rejected_before_the_route_runs(client, app_module):
    # A body this size would otherwise be json.loads-expanded (~24x RSS) before
    # any dependency — including auth — got to run.
    seen = []

    def _spy():
        seen.append(1)
        return {"user_id": "u", "is_admin": True, "permissions_raw": {},
                "permissions": auth.Permissions({}, True)}

    app_module.app.dependency_overrides[auth.get_current_user] = _spy
    try:
        assert _put_json(client, _big_json(8 * 1024 * 1024)).status_code == 413
        assert seen == []
    finally:
        app_module.app.dependency_overrides.pop(auth.get_current_user, None)


def test_oversize_json_with_plus_json_subtype_is_413(client):
    # FastAPI parses application/*+json bodies as JSON too — a prefix test on
    # "application/json" would hand them the 256 MiB non-upload backstop.
    r = client.put("/v1/synced-client-settings", content=_big_json(8 * 1024 * 1024),
                   headers={"Content-Type": "application/merge-patch+json"})
    assert r.status_code == 413
    # The middleware's own 413: the route answers 413 for an 8 MiB blob too.
    assert r.json() == {"detail": "request body too large"}


def test_oversize_body_with_no_content_type_is_413(client):
    # No Content-Type at all: FastAPI still buffers the body whole (and
    # JSON-parses it if strict_content_type is ever turned off; routes that
    # call request.json() themselves parse it anyway), so an absent header
    # must count as JSON for the cap.
    r = client.put("/v1/synced-client-settings", content=_big_json(8 * 1024 * 1024))
    assert "content-type" not in r.request.headers
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}


def test_small_body_with_no_content_type_is_not_413(client):
    # Legitimate small header-less traffic keeps flowing (the route may still
    # reject it on its own terms, but never with the middleware's 413).
    r = client.put("/v1/synced-client-settings", content=_big_json(1024))
    assert r.status_code != 413


def test_chunked_oversize_json_is_cut_off(client, app_module):
    # Content-Length absent → the receive-side counter is what enforces it.
    # The counter turns the stream into a ClientDisconnect, which FastAPI
    # reports as a 400 parse error; a 413 here would mean the ROUTE
    # (client-settings' own blob cap), not the counter, rejected the body.
    seen = []

    def _spy():
        seen.append(1)
        return {"user_id": "u", "is_admin": True, "permissions_raw": {},
                "permissions": auth.Permissions({}, True)}

    app_module.app.dependency_overrides[auth.get_current_user] = _spy
    try:
        r = _put_json(client, _big_json(8 * 1024 * 1024), chunked=True)
    finally:
        app_module.app.dependency_overrides.pop(auth.get_current_user, None)
    assert r.status_code == 400, r.text
    assert "parsing the body" in r.json()["detail"]
    assert seen == []


def test_json_cap_honours_a_cfg_override(client, app_module, monkeypatch):
    # MAX_JSON_BODY_BYTES is an internal escape hatch (getattr with a 4 MiB
    # default), not an operator-facing setting — it has no config_store
    # registry entry.
    monkeypatch.setattr(app_module.cfg, "MAX_JSON_BODY_BYTES", 1024, raising=False)
    assert _put_json(client, _big_json(4096)).status_code == 413
    assert _put_json(client, _big_json(16)).status_code == 200


def test_multipart_upload_is_not_subject_to_the_json_cap(client, app_module,
                                                         monkeypatch):
    monkeypatch.setattr(app_module.cfg, "MAX_JSON_BODY_BYTES", 4096, raising=False)
    assert post_json_ok(
        client,
        files={"file": ("a.wav", b"RIFFxxxxWAVE" + b"\0" * 200_000, "audio/wav")}
    ) == {"text": "hallo welt"}


def test_non_json_body_to_a_json_route_is_held_to_the_json_cap(client):
    # FastAPI buffers a text/plain body whole (request.body()) before the
    # host gate and get_current_user, and the route could only 422 it: it
    # gets the JSON ceiling, not the upload one.
    r = client.put("/v1/synced-client-settings", content=b"x" * (8 * 1024 * 1024),
                   headers={"Content-Type": "text/plain"})
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}


def test_small_non_json_body_reaches_the_route(client):
    # Under the cap the route answers on its own terms (a 422 here).
    r = client.put("/v1/synced-client-settings", content=b"x" * 1024,
                   headers={"Content-Type": "text/plain"})
    assert r.status_code == 422


def _spy_request_body(monkeypatch):
    """Lengths of every body Starlette's Request.body() hands back."""
    from starlette.requests import Request
    seen = []
    real = Request.body

    async def _body(self):
        b = await real(self)
        seen.append(len(b))
        return b
    monkeypatch.setattr(Request, "body", _body)
    return seen


def test_non_json_body_is_refused_before_it_is_buffered(client, monkeypatch):
    # Unauthenticated, before the reports host gate: a 5 MiB text/plain body
    # is refused on its declared length, and none of it is read.
    seen = _spy_request_body(monkeypatch)
    r = client.post("/quick-config/reports/api/submit",
                    content=b"x" * (5 * 1024 * 1024),
                    headers={"Content-Type": "text/plain"})
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}
    assert seen == []


def test_chunked_non_json_body_is_cut_off_at_the_json_cap(client, monkeypatch):
    # No Content-Length: the receive-side counter stops it at the JSON cap.
    seen = _spy_request_body(monkeypatch)
    payload = b"x" * (8 * 1024 * 1024)

    def _gen():
        for i in range(0, len(payload), 65536):
            yield payload[i:i + 65536]
    client.post("/captures/api/samples/preview-audio", content=_gen(),
                headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert all(n <= 4 * 1024 * 1024 for n in seen)


def test_form_encoded_post_to_an_upload_route_keeps_the_backstop(
        client, app_module, monkeypatch):
    # The upload routes take Form() params, which FastAPI also reads from an
    # application/x-www-form-urlencoded body; their credential is checked
    # before the parse, so that body keeps the backstop, not the JSON cap.
    monkeypatch.setattr(app_module.cfg, "MAX_JSON_BODY_BYTES", 1024, raising=False)
    r = client.post("/v1/audio/transcriptions",
                    content=b"model=whisper-1&prompt=" + b"x" * 4096,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code != 413


def test_non_multipart_non_json_to_an_upload_route_keeps_a_256mib_backstop(
        client, app_module, monkeypatch):
    # MAX_REQUEST_BYTES is sized for a MEDIA_MAX_BYTES video upload (GiB);
    # only a multipart body to an upload route gets that ceiling, so a
    # text/plain POST there declaring 300 MiB is still refused up front.
    monkeypatch.setattr(app_module.cfg, "MAX_REQUEST_BYTES",
                        10 * 1024 * 1024 * 1024, raising=False)
    r = client.post("/v1/audio/transcriptions", content=b"x",
                    headers={"Content-Type": "text/plain",
                             "Content-Length": str(300 * 1024 * 1024)})
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}


def test_raw_media_upload_keeps_its_own_ceiling(client, app_module,
                                                monkeypatch):
    # The raw-body video upload (/v1/audio/media) is not held to the JSON
    # cap: its ceiling is MEDIA_MAX_BYTES.
    from faster_whisper_backend.media import media_store
    monkeypatch.setattr(app_module.cfg, "MAX_JSON_BODY_BYTES", 1024, raising=False)
    monkeypatch.setattr(media_store, "media_max_bytes", lambda: 10_000)
    small = client.post("/v1/audio/media", content=b"x" * 4096,
                        headers={"Content-Type": "video/mp4"})
    assert small.status_code != 413
    big = client.post("/v1/audio/media", content=b"x" * 20_000,
                      headers={"Content-Type": "video/mp4"})
    assert big.status_code == 413


def test_self_parsing_route_holds_a_non_json_type_to_the_json_cap(client):
    # /auth/login calls request.json() itself, and Starlette parses the body
    # whatever its Content-Type: a text/plain body there must not ride the
    # 256 MiB backstop into json.loads (unauthenticated, CSRF-exempt, before
    # the login failure limiter).
    r = client.post("/auth/login", content=b"x" * (8 * 1024 * 1024),
                    headers={"Content-Type": "text/plain"})
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}


def test_chunked_non_json_body_to_a_self_parsing_route_is_cut_off(
        client, monkeypatch):
    # No Content-Length: the receive-side counter must stop the body at the
    # JSON cap, so login's own parse fails and it sees no key at all —
    # never the 8 MiB one.
    from faster_whisper_backend.auth import api_keys_store
    monkeypatch.setattr(api_keys_store, "is_locked_down", lambda: True)
    keys = []
    monkeypatch.setattr(api_keys_store, "lookup_by_raw_key",
                        lambda key: keys.append(key))
    payload = json.dumps({"key": "x" * (8 * 1024 * 1024)}).encode()

    def _gen():
        for i in range(0, len(payload), 65536):
            yield payload[i:i + 65536]
    client.post("/auth/login", content=_gen(),
                headers={"Content-Type": "text/plain"})
    assert keys == [""]


def test_every_self_parsing_post_route_is_held_to_the_json_cap(app_module):
    # A new route that parses its own JSON must join the path set, which
    # holds it to the JSON cap whatever ceiling its Content-Type gets.

    def _walk(routes):
        # FastAPI 0.141 wraps each include_router in an _IncludedRouter.
        for r in routes:
            if hasattr(r, "original_router"):
                yield from _walk(r.original_router.routes)
            else:
                yield r
    found = set()
    for route in _walk(app_module.app.routes):
        if "POST" not in (getattr(route, "methods", None) or ()):
            continue
        src = inspect.getsource(route.endpoint)
        if "request.json()" in src or "_url_request(" in src:
            found.add(route.path)
    # The package route parses its own JSON under its own, larger ceiling.
    found.discard("/v1/audio/media/{media_id}/package")
    assert found == app_module._SELF_PARSED_JSON_PATHS


def test_multipart_label_on_a_non_upload_route_gets_the_json_cap(
        client, app_module, monkeypatch):
    # FastAPI buffers a non-JSON body whole before auth on every route with
    # a body field: a multipart label must not hand a JSON route the
    # gigabyte upload ceiling.
    monkeypatch.setattr(app_module.cfg, "MAX_JSON_BODY_BYTES", 1024, raising=False)
    r = client.put("/v1/synced-client-settings", content=b"x" * 4096,
                   headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}


def test_multipart_upload_route_is_not_held_to_the_backstop(
        client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "_NON_UPLOAD_BODY_BACKSTOP", 1024)
    post_json_ok(
        client,
        files={"file": ("a.wav", b"RIFFxxxxWAVE" + b"\0" * 200_000, "audio/wav")})


def test_every_multipart_route_is_an_upload_path(app_module):
    # A new route taking File()/Form() must join the path set, or its
    # uploads are clamped to the non-upload backstop.
    import re

    def _walk(routes):
        for r in routes:
            if hasattr(r, "original_router"):
                yield from _walk(r.original_router.routes)
            else:
                yield r
    found = set()
    for route in _walk(app_module.app.routes):
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        src = inspect.getsource(endpoint)
        # \b: NamedTemporaryFile( is not a File() param.
        if re.search(r"\b(?:File|Form)\(|request\.form\(\)", src):
            found.add(route.path)
    assert found == app_module._MULTIPART_UPLOAD_PATHS
