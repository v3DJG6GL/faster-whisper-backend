"""The _max_body_mw ceiling, and the tighter one it applies to JSON.

FastAPI parses `await request.json()` BEFORE solve_dependencies, so a JSON body
is buffered and json.loads-expanded ahead of the host gate, get_current_user and
any in-handler rate limiter. The middleware therefore caps a declared
application/json body far below MAX_REQUEST_BYTES; multipart audio uploads must
keep the full ceiling.
"""

import json

_JSON = {"Content-Type": "application/json"}


def _big_json(nbytes):
    return json.dumps({"blob": {"pad": "x" * nbytes}, "base_version": 0}).encode()


def _put_json(client, payload, chunked=False):
    if chunked:
        # No Content-Length: only the streaming byte counter can catch this.
        def _gen():
            for i in range(0, len(payload), 65536):
                yield payload[i:i + 65536]
        return client.put("/v1/client-settings", content=_gen(), headers=_JSON)
    return client.put("/v1/client-settings", content=payload, headers=_JSON)


def test_small_json_body_is_not_rejected(client):
    r = _put_json(client, _big_json(1024))
    assert r.status_code == 200, r.text


def test_oversize_json_body_is_413(client):
    r = _put_json(client, _big_json(8 * 1024 * 1024))
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}


def test_oversize_json_is_rejected_before_the_route_runs(client, app_module):
    # A body this size would otherwise be json.loads-expanded (~24x RSS) before
    # any dependency — including auth — got to run. The spy proves the
    # ordering the name claims: the 413 lands with auth never invoked.
    from faster_whisper_backend.auth import dependencies as auth
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
    # "application/json" would hand them the 256 MB service-wide ceiling.
    r = client.put("/v1/client-settings", content=_big_json(8 * 1024 * 1024),
                   headers={"Content-Type": "application/merge-patch+json"})
    assert r.status_code == 413


def test_oversize_body_with_no_content_type_is_413(client):
    # No Content-Type at all: FastAPI still buffers and json.loads-expands the
    # body, so an absent header must count as JSON for the cap.
    r = client.put("/v1/client-settings", content=_big_json(8 * 1024 * 1024))
    assert "content-type" not in r.request.headers
    assert r.status_code == 413


def test_small_body_with_no_content_type_is_not_413(client):
    # Legitimate small header-less traffic keeps flowing (the route may still
    # reject it on its own terms, but never with the middleware's 413).
    r = client.put("/v1/client-settings", content=_big_json(1024))
    assert r.status_code != 413


def test_chunked_oversize_json_is_cut_off(client):
    # Content-Length absent → the receive-side counter is what enforces it.
    r = _put_json(client, _big_json(8 * 1024 * 1024), chunked=True)
    assert r.status_code >= 400


def test_json_cap_honours_a_cfg_override(client, app_module, monkeypatch):
    # MAX_JSON_BODY_BYTES is an internal escape hatch (getattr with a 4 MiB
    # default), not an operator-facing setting — it has no config_store
    # registry entry. This pins the escape hatch, not a published knob.
    monkeypatch.setattr(app_module.cfg, "MAX_JSON_BODY_BYTES", 1024, raising=False)
    assert _put_json(client, _big_json(4096)).status_code == 413
    assert _put_json(client, _big_json(16)).status_code == 200


def test_multipart_upload_is_not_subject_to_the_json_cap(client, app_module,
                                                         monkeypatch):
    # A multipart body well past the JSON ceiling still reaches the route.
    monkeypatch.setattr(app_module.cfg, "MAX_JSON_BODY_BYTES", 4096, raising=False)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"RIFFxxxxWAVE" + b"\0" * 200_000, "audio/wav")},
        data={"model": "whisper-1", "response_format": "json"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "hallo welt"}


def test_non_json_content_type_keeps_the_service_wide_ceiling(client):
    # text/plain is neither JSON nor an upload: it keeps MAX_REQUEST_BYTES, so a
    # body that would trip the JSON cap is not rejected by the middleware (the
    # route rejects it on its own terms instead).
    r = client.put("/v1/client-settings", content=b"x" * (8 * 1024 * 1024),
                   headers={"Content-Type": "text/plain"})
    assert r.status_code != 413
