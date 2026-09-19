"""Page canvas: header, toolbar and content share ONE width column per page.

Pins the contract behind the wide-screen layout (see NAV_CSS `:root --col-*`):
every page render_page() knows about maps to a width class, that class is
stamped on <body>, and the header / toolbar rules read the canvas tokens
instead of the old hard-coded 150rem / 68.75rem caps.
"""

import re

from faster_whisper_backend.core import web_common


_CLASSES = {"col-read", "col-form", "col-data", "col-fluid"}


def test_every_nav_page_has_an_explicit_canvas_class():
    keys = {current for (_label, _href, current, _gated) in web_common._NAV_SPEC}
    keys |= {"home"}
    for current in keys:
        assert current in web_common._COL_CLASS_BY_CURRENT, current
        assert web_common._COL_CLASS_BY_CURRENT[current] in _CLASSES


def test_unknown_page_falls_back_to_data_canvas():
    assert web_common._col_class_for("nonexistent-page") == "col-data"


def test_nav_css_declares_canvas_tokens_and_uses_them():
    css = web_common.NAV_CSS
    for tok in ("--col-read:", "--col-form:", "--col-data:", "--col-fluid:", "--gutter:"):
        assert tok in css, tok
    assert "body.col-form  { --col: var(--col-form); }" in css
    assert "html.pref-fluid { --col-data: var(--col-fluid); }" in css
    # header floored at the data canvas; toolbar exactly on the page column
    assert "max-width: max(var(--col), var(--col-data))" in css
    assert re.search(r"header \.subbar \{[^}]*max-width: var\(--col\)", css)
    # the old fixed caps are gone from the header rules
    assert "max-width: 150rem" not in css
    assert "max-width: 68.75rem" not in css
    # long <option> labels no longer dictate the toolbar row width
    assert "header .subbar select { max-width: 14rem;" in css


def test_width_preference_control_and_bootstrap():
    assert 'id="width-toggle"' in web_common.SCALE_PICKER_HTML
    assert "whisper-ui-width" in web_common.SCALE_BOOTSTRAP_HEAD
    assert "pref-fluid" in web_common.SCALE_BOOTSTRAP_HEAD
    assert "whisper-ui-width" in web_common.SCALE_PICKER_JS
    # hidden while the window is narrower than the fixed data canvas
    assert "@container hdr (max-width: 100rem)" in web_common.NAV_CSS


def test_rendered_pages_carry_their_canvas_class(client):
    expected = {
        "/captures": "col-data",
        "/stats": "col-data",
        "/settings": "col-form",
        "/logs": "col-fluid",
        "/": "col-read",
    }
    for path, cls in expected.items():
        r = client.get(path)
        assert r.status_code == 200, path
        m = re.search(r'<body class="([^"]*)"', r.text)
        assert m and cls in m.group(1).split(), (path, m and m.group(1))
