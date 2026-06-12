"""CORS: opt-in cross-origin support for the JSON API (off by default).

Covers the AdminConfig origin validator, the default no-CORS behavior, and that
listing an origin in CORS_ALLOW_ORIGINS makes the CORSMiddleware emit the
Access-Control-Allow-Origin header (so the /dictate batch fetch works cross-origin).
"""

import importlib

import pytest
from starlette.testclient import TestClient


# ---- origin validation ----------------------------------------------------

def test_cors_origin_validator_accepts_origins_and_star():
    from config_store import AdminConfig
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
    from config_store import AdminConfig
    with pytest.raises(ValidationError):
        AdminConfig.model_validate({"CORS_ALLOW_ORIGINS": [bad]})


# ---- runtime behavior -----------------------------------------------------

def test_no_cors_headers_by_default(app_module):
    """Default (empty allowlist) → no CORSMiddleware, no Access-Control-* headers."""
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        r = client.get("/v1/models", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def _reload_main(tmp_path, monkeypatch, cors_env):
    """Re-import config+main with a temp store layout and a CORS allowlist set,
    mirroring the conftest app_module fixture (which can't parameterize env)."""
    for var, fn in (
        ("WHISPER_API_KEYS_DB", "api_keys.sqlite3"),
        ("WHISPER_SESSIONS_DB", "sessions.sqlite3"),
        ("WHISPER_REPORTS_DB", "reports.sqlite3"),
        ("WHISPER_RECENT_TRANSCRIPTIONS_DB", "recent.sqlite3"),
        ("WHISPER_USAGE_DB", "usage.sqlite3"),
        ("WHISPER_CAPTURES_DB", "captures.sqlite3"),
        ("WHISPER_CAPTURES_DIR", "captures_audio"),
        ("WHISPER_LOG_FILE", "whisper.log"),
    ):
        monkeypatch.setenv(var, str(tmp_path / fn))
    monkeypatch.setenv("WHISPER_CORS_ALLOW_ORIGINS", cors_env)

    import config as cfg
    importlib.reload(cfg)
    monkeypatch.setattr(cfg, "PRELOAD_MODELS", [], raising=False)
    monkeypatch.setattr(cfg, "DEFAULT_MODEL", "", raising=False)
    import config_store
    monkeypatch.setattr(config_store, "OVERRIDES_PATH", str(tmp_path / "config.local.json"),
                        raising=False)
    for _fn in (config_store.load_overrides, config_store.save_overrides):
        d = list(_fn.__defaults__ or ())
        if d:
            d[-1] = str(tmp_path / "config.local.json")
            monkeypatch.setattr(_fn, "__defaults__", tuple(d), raising=False)
    import main
    importlib.reload(main)
    return main, cfg


def _preflight(client, origin):
    return client.options("/v1/audio/transcriptions", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization",
    })


def test_cors_preflight_allows_configured_origin(tmp_path, monkeypatch):
    origin = "http://localhost:9999"
    main, cfg = _reload_main(tmp_path, monkeypatch, origin)
    assert cfg.CORS_ALLOW_ORIGINS == [origin]   # env CSV → list
    try:
        with TestClient(main.app, client=("127.0.0.1", 12345)) as client:
            r = _preflight(client, origin)
        assert r.headers.get("access-control-allow-origin") == origin
        assert "POST" in (r.headers.get("access-control-allow-methods") or "")
    finally:
        monkeypatch.delenv("WHISPER_CORS_ALLOW_ORIGINS", raising=False)
        importlib.reload(__import__("config"))
        importlib.reload(main)


def test_cors_star_allows_any_origin(tmp_path, monkeypatch):
    """'*' is the simplest setting for a remote/file:// demo page — it echoes
    Access-Control-Allow-Origin: * for any origin (credentials disabled)."""
    main, cfg = _reload_main(tmp_path, monkeypatch, "*")
    assert cfg.CORS_ALLOW_ORIGINS == ["*"]
    try:
        with TestClient(main.app, client=("127.0.0.1", 12345)) as client:
            r = _preflight(client, "http://anything.example:1234")
        assert r.headers.get("access-control-allow-origin") == "*"
    finally:
        monkeypatch.delenv("WHISPER_CORS_ALLOW_ORIGINS", raising=False)
        importlib.reload(__import__("config"))
        importlib.reload(main)
