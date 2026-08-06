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


def test_oversize_json_is_rejected_before_the_route_runs(client):
    # A body this size would otherwise be json.loads-expanded (~24x RSS) before
    # any dependency — including auth — got to run.
    assert _put_json(client, _big_json(8 * 1024 * 1024)).status_code == 413


def test_chunked_oversize_json_is_cut_off(client):
    # Content-Length absent → the receive-side counter is what enforces it.
    r = _put_json(client, _big_json(8 * 1024 * 1024), chunked=True)
    assert r.status_code >= 400
    assert r.status_code != 200


def test_json_cap_is_configurable(client, app_module, monkeypatch):
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
