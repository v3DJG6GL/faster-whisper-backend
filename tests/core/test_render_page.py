"""web_common.render_page: the shared page-chrome substitution (memoised per
key, tracking the hot-mutable cfg values, never baking in per-user values)
and the header brand lockup it renders on every page."""


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
    from faster_whisper_backend.core import web_common
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
    from faster_whisper_backend.settings import config as cfg
    from faster_whisper_backend.core import web_common

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
    from faster_whisper_backend.core import web_common
    names = sorted(set(re.findall(
        r"\{\{[A-Z_]+\}\}", inspect.getsource(web_common._render_page_cached))))
    assert len(names) >= 20, names
    tpl = "".join(names)
    out = web_common.render_page(tpl, "logs")
    assert "{{" not in out
