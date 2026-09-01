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


# ---------------------------------------------------------------------------
# render_page memoization (web_common)
# ---------------------------------------------------------------------------
# The substitution chain rebuilds a ~270 KB shell per request and these pages
# render before any credential is examined, so it is memoized. The contract
# worth pinning is the CACHE KEY: nothing per-user may be substituted, and the
# three hot-mutable cfg reads must invalidate.

_TPL = (
    "<title>{{HEADER_TITLE}}</title>{{NAV}}{{PAGE_META}}"
    "{{LOG_VIEWER_INITIAL_LINES}}/{{LOG_VIEWER_DOM_MAX}}"
)


def test_render_page_is_memoized_per_key():
    import web_common
    a = web_common.render_page(_TPL, "logs")
    b = web_common.render_page(_TPL, "logs")
    # Same key -> the identical (immutable) str object, i.e. no rebuild.
    assert a is b
    # A different `current` is a different key and must not be served the
    # cached body.
    assert web_common.render_page(_TPL, "stats") != a


def test_render_page_key_tracks_hot_mutable_cfg(monkeypatch):
    """ADMIN_UI_ENABLED and the two LOG_VIEWER_* values are mutated at runtime
    by the settings save path, so they are part of the key rather than read at
    import."""
    import config as cfg
    import web_common

    monkeypatch.setattr(cfg, "ADMIN_UI_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000, raising=False)
    monkeypatch.setattr(cfg, "LOG_VIEWER_DOM_MAX", 0, raising=False)
    on = web_common.render_page(_TPL, "logs")
    assert "2000/8000" in on  # DOM_MAX=0 resolves to initial x 4

    monkeypatch.setattr(cfg, "ADMIN_UI_ENABLED", False, raising=False)
    off = web_common.render_page(_TPL, "logs")
    assert off != on, "ADMIN_UI_ENABLED must invalidate the memo"

    monkeypatch.setattr(cfg, "LOG_VIEWER_INITIAL_LINES", 55, raising=False)
    assert "55/220" in web_common.render_page(_TPL, "logs")


def test_render_page_substitutes_no_per_user_value():
    """Guard against a future per-user substitution silently entering a shared
    cache. Every placeholder must be resolved, and none from a request.

    The template is DERIVED from _render_page_cached's own substitution list
    rather than hand-written, so a placeholder added there is covered
    automatically instead of leaving a 4-placeholder stub green."""
    import inspect
    import re
    import web_common
    names = sorted(set(re.findall(
        r"\{\{[A-Z_]+\}\}", inspect.getsource(web_common._render_page_cached))))
    assert len(names) >= 20, names
    tpl = "".join(names)
    out = web_common.render_page(tpl, "logs")
    assert "{{" not in out
