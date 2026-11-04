"""CORS: opt-in cross-origin support for the JSON API (off by default).

Covers the AdminConfig origin validator, the default no-CORS behavior, and that
listing an origin in CORS_ALLOW_ORIGINS makes the CORSMiddleware emit the
Access-Control-Allow-Origin header (so the /dictate batch fetch works cross-origin).

Also covers TRUSTED_ORIGINS — the CORS-free sibling read only by the
unsafe-method origin guard (for a reverse proxy that rewrites Host).
"""

import importlib

import pytest
from starlette.testclient import TestClient

from tests.conftest import close_app_stores, isolate_app_env


def test_cors_origin_validator_accepts_origins_and_star():
    from faster_whisper_backend.settings.schema import AdminConfig
    m = AdminConfig.model_validate({
        "CORS_ALLOW_ORIGINS": ["https://app.example.com", "http://192.168.1.50:8000", "*"]})
    assert m.CORS_ALLOW_ORIGINS == ["https://app.example.com", "http://192.168.1.50:8000", "*"]


@pytest.mark.parametrize("bad", [
    "app.example.com",                 # no scheme
    "https://app.example.com/",        # trailing slash / path
    "https://app.example.com/dictate", # path
    "ftp://x",                         # wrong scheme
    "https://a b",                     # space
])
def test_cors_origin_validator_rejects_bad(bad):
    from pydantic import ValidationError
    from faster_whisper_backend.settings.schema import AdminConfig
    with pytest.raises(ValidationError):
        AdminConfig.model_validate({"CORS_ALLOW_ORIGINS": [bad]})


def test_trusted_origins_validator_accepts_origins():
    from faster_whisper_backend.settings.schema import AdminConfig
    m = AdminConfig.model_validate({
        "TRUSTED_ORIGINS": ["https://whisper.example.com", "http://192.168.1.50:8000"]})
    assert m.TRUSTED_ORIGINS == ["https://whisper.example.com",
                                 "http://192.168.1.50:8000"]


@pytest.mark.parametrize("bad", [
    "*",                               # would disable the origin guard outright
    "app.example.com",                 # no scheme
    "https://app.example.com/",        # trailing slash / path
    "ftp://x",                         # wrong scheme
    "https://a b",                     # space
])
def test_trusted_origins_validator_rejects_bad(bad):
    from pydantic import ValidationError
    from faster_whisper_backend.settings.schema import AdminConfig
    with pytest.raises(ValidationError):
        AdminConfig.model_validate({"TRUSTED_ORIGINS": [bad]})


def test_no_cors_headers_by_default(app_module):
    """Default (empty allowlist) → no CORSMiddleware, no Access-Control-* headers."""
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        r = client.get("/v1/models", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


@pytest.fixture
def reload_main(tmp_path, monkeypatch):
    """Re-import config+main with a temp store layout and a CORS allowlist set,
    through the same isolate_app_env the conftest app_module fixture uses
    (which can't parameterize env). Teardown closes the stores, drops both env
    vars and re-imports config + main."""
    def _reload(cors_env, trusted_env=""):
        monkeypatch.setenv("WHISPER_CORS_ALLOW_ORIGINS", cors_env)
        monkeypatch.setenv("WHISPER_TRUSTED_ORIGINS", trusted_env)
        isolate_app_env(tmp_path, monkeypatch)
        from faster_whisper_backend.settings import config as cfg
        from faster_whisper_backend import main
        importlib.reload(main)
        return main, cfg
    yield _reload
    close_app_stores()
    monkeypatch.delenv("WHISPER_CORS_ALLOW_ORIGINS", raising=False)
    monkeypatch.delenv("WHISPER_TRUSTED_ORIGINS", raising=False)
    importlib.reload(importlib.import_module("faster_whisper_backend.settings.config"))
    importlib.reload(importlib.import_module("faster_whisper_backend.main"))


def _preflight(client, origin):
    return client.options("/v1/audio/transcriptions", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization",
    })


def test_cors_preflight_allows_configured_origin(reload_main):
    origin = "http://localhost:9999"
    main, cfg = reload_main(origin)
    assert cfg.CORS_ALLOW_ORIGINS == [origin]   # env CSV → list
    with TestClient(main.app, client=("127.0.0.1", 12345)) as client:
        r = _preflight(client, origin)
    assert r.headers.get("access-control-allow-origin") == origin
    assert "POST" in (r.headers.get("access-control-allow-methods") or "")


def test_trusted_origin_passes_guard_without_enabling_cors(reload_main):
    """The reverse-proxy escape hatch: a POST whose Origin is the PUBLIC origin
    (Host rewritten to the upstream by the proxy) is accepted, while CORS stays
    off — no middleware, no Access-Control-* headers."""
    origin = "https://whisper.example.com"
    main, cfg = reload_main("", trusted_env=origin)
    assert cfg.CORS_ALLOW_ORIGINS == []
    assert cfg.TRUSTED_ORIGINS == [origin]
    with TestClient(main.app, client=("127.0.0.1", 12345)) as client:
        r = client.post("/auth/logout", headers={"Origin": origin})
        assert r.status_code == 200
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
        assert _preflight(client, origin).headers.get(
            "access-control-allow-origin") is None


def test_cross_site_origin_still_rejected_with_trusted_origins_set(reload_main):
    """A listed trusted origin doesn't widen the guard: any other Origin is
    still 403, with the distinct message (no CSRF token is involved)."""
    main, _cfg = reload_main("", trusted_env="https://whisper.example.com")
    with TestClient(main.app, client=("127.0.0.1", 12345)) as client:
        r = client.post("/auth/logout", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert r.json()["detail"] == "Origin not allowed for this host"


def test_cors_star_allows_any_origin(reload_main):
    """'*' is the simplest setting for a remote/file:// demo page — it echoes
    Access-Control-Allow-Origin: * for any origin (credentials disabled)."""
    main, cfg = reload_main("*")
    assert cfg.CORS_ALLOW_ORIGINS == ["*"]
    with TestClient(main.app, client=("127.0.0.1", 12345)) as client:
        r = _preflight(client, "http://anything.example:1234")
    assert r.headers.get("access-control-allow-origin") == "*"


def test_an_early_413_still_carries_the_cors_header(reload_main, monkeypatch):
    """CORSMiddleware wraps the body cap and the CSRF guard: their early
    413/403 must still name the allowed origin, or the browser hides the
    refusal behind an opaque CORS error."""
    origin = "http://localhost:9999"
    main, cfg = reload_main(origin)
    monkeypatch.setattr(cfg, "MAX_JSON_BODY_BYTES", 1024, raising=False)
    with TestClient(main.app, client=("127.0.0.1", 12345)) as client:
        r = client.put("/v1/synced-client-settings",
                       content=b'{"pad": "' + b"x" * 4096 + b'"}',
                       headers={"Origin": origin,
                                "Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == origin
