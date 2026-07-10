"""Route tests for the / landing hub (home_routes.py).

The hub is a user-tier HTML shell: host-gated only, with tile visibility
resolved client-side from /auth/whoami. Server-side the tests can still pin
down what the shell CONTAINS — which tiles are rendered (ADMIN_UI_ENABLED
gating happens at request time), the cache header, and the host gate.
Tiles are asserted via their `data-hub="<key>"` markers because several page
paths also appear inside the shared chrome JS (e.g. the open-mode banner
links /settings/api-keys), so raw href substring checks would false-positive.
"""

from starlette.testclient import TestClient


def _tile_marker(key: str) -> str:
    return f'data-hub="{key}"'


def test_root_serves_hub_shell(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-store"
    # launcher scaffolding + brand wordmark + login-gate chrome all present
    assert 'class="hub-grid"' in r.text
    assert '<span class="w-b">whisper</span>' in r.text
    assert "_refreshAuthChrome" in r.text


def test_root_renders_every_tile_when_admin_ui_enabled(client):
    r = client.get("/")
    for key in ("quick_config", "captures", "reports", "stats", "logs",
                "dictate", "settings", "pipeline", "keys", "overrides"):
        assert _tile_marker(key) in r.text
    # the admin zone exists and user-tier tiles carry their permission key
    assert 'aria-label="Admin pages"' in r.text
    assert 'data-page="quick_config"' in r.text


def test_root_drops_gated_tiles_when_admin_ui_disabled(app_module, monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg, "ADMIN_UI_ENABLED", False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as c:
        r = c.get("/")
    assert r.status_code == 200
    # pages that ride the ADMIN_UI_ENABLED switch lose their tiles ...
    for key in ("quick_config", "captures", "reports",
                "settings", "pipeline", "keys", "overrides"):
        assert _tile_marker(key) not in r.text
    assert 'aria-label="Admin pages"' not in r.text
    # ... while the always-registered pages keep theirs
    for key in ("stats", "logs", "dictate"):
        assert _tile_marker(key) in r.text


def test_root_host_gate(app_module, monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg, "USER_WEBUI_ALLOWED_HOSTS", [])
    with TestClient(app_module.app, client=("203.0.113.9", 1234)) as c:
        assert c.get("/").status_code == 403


def test_header_brand_lockup_links_home(client):
    r = client.get("/logs")
    assert '<a class="brand-link" href="/"' in r.text
