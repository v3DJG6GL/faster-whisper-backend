"""Misc app-level routes: /v1/models, /logs, /sev, /auth/whoami."""

from conftest import bearer


def test_v1_models_requires_a_user(client, make_user_key):
    # User-tier auth like its /v1 siblings: the payload carries the build
    # version, the per-process boot_id and the whole ALLOWED_MODELS list.
    make_user_key("root", is_admin=True)   # locks the server down
    assert client.get("/v1/models").status_code == 401
    _uid, raw = make_user_key("alice")
    assert client.get("/v1/models", headers=bearer(raw)).status_code == 200


def test_v1_models_shape(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert "boot_id" in body and isinstance(body["boot_id"], str)
    # Build identity travels with the model list so clients can display
    # "faster-whisper-backend · <version>" (exact version varies per build).
    assert body["server_name"] == "faster-whisper-backend"
    assert isinstance(body["server_version"], str) and body["server_version"]
    assert isinstance(body["data"], list)
    for entry in body["data"]:
        assert entry["object"] == "model"
        assert "id" in entry
        assert "loaded" in entry  # bool flag
        assert isinstance(entry["loaded"], bool)
    # No model is loaded in the harness (preload neutralised), so every
    # listed model reports loaded=False.
    assert all(e["loaded"] is False for e in body["data"])


def test_logs_page_open_no_auth(client):
    r = client.get("/logs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_sev_shape(client):
    r = client.get("/sev")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"warn", "err", "crit"}
    assert all(isinstance(v, int) for v in body.values())


def test_whoami_open_mode_admin(client):
    r = client.get("/auth/whoami")
    assert r.status_code == 200
    body = r.json()
    assert body["open_mode"] is True
    assert body["is_admin"] is True
    assert "permissions" in body and "pages" in body["permissions"]


def test_logs_older_open(client):
    # /logs/older needs the 'logs' scope='all'. In open mode the synthetic
    # admin bypasses the page gate, so it returns the pagination envelope.
    r = client.get("/logs/older")
    assert r.status_code == 200
    body = r.json()
    assert "lines" in body and "next_skip" in body


# ---------------------------------------------------------------------------
# _security_headers_mw — the outermost response-header layer
# ---------------------------------------------------------------------------

def test_security_headers_on_every_response(client):
    r = client.get("/")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    # No script-src / default-src: every page relies on inline <script>, an
    # inline onclick= in the shared header, blob: AudioWorklets and data: SVGs.
    csp = r.headers["Content-Security-Policy"]
    assert "script-src" not in csp and "default-src" not in csp


def test_data_responses_default_to_no_store(client):
    assert client.get("/").headers["Cache-Control"] == "no-store"
    # ...including the early returns from the inner middlewares.
    assert client.get("/logs/older?skip=0&limit=5").headers["Cache-Control"]


def test_static_assets_stay_cacheable(client):
    r = client.get("/static/favicon.svg")
    assert r.status_code == 200
    assert r.headers.get("Cache-Control") is None


def test_an_explicit_cache_control_is_not_overridden(client):
    # /settings sends a stronger value of its own; the middleware defaults,
    # it does not stamp over a handler's deliberate choice.
    cc = client.get("/settings").headers["Cache-Control"]
    assert "must-revalidate" in cc
