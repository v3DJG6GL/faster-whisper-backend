"""POST /v1/audio/transcriptions and /v1/audio/translations refuse a caller
with no valid credential BEFORE the multipart body is parsed. FastAPI runs
`await request.form()` ahead of solve_dependencies, so the route's own
Depends(get_current_user) only fires after Starlette has spooled every file
part to TMPDIR (up to MAX_REQUEST_BYTES) and held every text field in RAM."""

import pytest
from starlette.formparsers import MultiPartParser

from tests.conftest import bearer

_UPLOAD_PATHS = ("/v1/audio/transcriptions", "/v1/audio/translations")
_FILES = {"file": ("a.wav", b"RIFF" + b"\x00" * 4096, "audio/wav")}


@pytest.fixture
def form_parses(monkeypatch):
    """Records each multipart parse, then lets it run."""
    seen = []
    real = MultiPartParser.parse

    async def _parse(self):
        seen.append(1)
        return await real(self)
    monkeypatch.setattr(MultiPartParser, "parse", _parse)
    return seen


def _unauth_shape(r):
    return (r.status_code, r.json(), r.headers.get("www-authenticate"))


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
@pytest.mark.parametrize("cred", ["none", "bad-bearer", "dead-cookie"])
def test_unauthenticated_upload_is_refused_before_the_body_is_parsed(
        client, app_module, locked_down, form_parses, path, cred):
    headers = {
        "none": {},
        "bad-bearer": bearer("not-a-real-key"),
        # A dead cookie on an unsafe method is the CSRF guard's 403, as
        # before: the pre-auth check runs inside it.
        "dead-cookie": {"Cookie": f"{app_module.cfg.SESSION_COOKIE_NAME}=junk"},
    }[cred]
    r = client.post(path, files=_FILES, data={"response_format": "json"},
                    headers=headers)
    assert r.status_code in (401, 403), r.text
    if r.status_code == 401:
        assert _unauth_shape(r) == (
            401, {"detail": "invalid or missing API key"}, "Bearer")
    assert form_parses == []


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
def test_preauth_401_matches_the_dependency_401(
        client, app_module, locked_down, path):
    """Same status, body and WWW-Authenticate as get_current_user's 401 on a
    route whose body is not a form, plus the headers the outer middlewares
    stamp on every answer."""
    r = client.post(path, files=_FILES)
    ref = client.get("/v1/models")
    assert ref.status_code == 401
    assert _unauth_shape(r) == _unauth_shape(ref)
    for h in ("cache-control", "x-frame-options", "x-content-type-options",
              "referrer-policy", "content-security-policy", "content-type"):
        assert r.headers.get(h) == ref.headers.get(h), h


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
def test_valid_bearer_upload_still_parses_and_transcribes(
        client, make_user_key, form_parses, path):
    raw = make_user_key("root", is_admin=True)[1]
    r = client.post(path, files=_FILES, data={"response_format": "json"},
                    headers=bearer(raw))
    assert r.status_code == 200, r.text
    assert form_parses == [1]


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
def test_open_mode_upload_still_parses_and_transcribes(
        client, form_parses, path):
    r = client.post(path, files=_FILES, data={"response_format": "json"})
    assert r.status_code == 200, r.text
    assert form_parses == [1]


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
def test_cookie_session_upload_still_parses_with_one_session_lookup(
        client, make_user_key, form_parses, monkeypatch, path):
    from faster_whisper_backend.auth import sessions_store
    raw = make_user_key("root", is_admin=True)[1]
    csrf = client.post("/auth/login", json={"key": raw}).json()["csrf_token"]
    lookups = []
    real = sessions_store.lookup_session

    def _lookup(token):
        lookups.append(token)
        return real(token)
    monkeypatch.setattr(sessions_store, "lookup_session", _lookup)
    r = client.post(path, files=_FILES, data={"response_format": "json"},
                    headers={"X-CSRF-Token": csrf})
    assert r.status_code == 200, r.text
    assert form_parses == [1]
    assert len(lookups) == 1
