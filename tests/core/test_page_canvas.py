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
    # The fluid preference lifts data pages only: an unscoped rule widened
    # the header rail on form / read pages, where its toggle is hidden.
    assert ("html.pref-fluid body.col-data { --col-data: var(--col-fluid);"
            " --col: var(--col-fluid); }") in css
    assert "html.pref-fluid {" not in css
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
    # ... and on every page the preference cannot widen (it lifts --col-data
    # only). A :not() hide, so it cannot resurrect the button below 100rem.
    assert "body:not(.col-data) header .width-toggle { display: none; }" in web_common.NAV_CSS


def test_scale_picker_survives_blocked_storage_and_a_stale_value():
    """Blocked storage (getItem throws) or a persisted value matching no
    option (selectedIndex -1) used to stop the IIFE before the cycle button
    and the width toggle were wired."""
    js = web_common.SCALE_PICKER_JS
    # every storage call sits inside its own try{ ... }
    assert js.count("try{") >= js.count("localStorage.") > 0
    assert "var saved=null;try{saved=localStorage.getItem(KEY);}catch(e){}" in js
    assert "[].some.call(sel.options" in js
    sync = js[js.index("function sync()"):]
    sync = sync[:sync.index("}")]
    assert "var o=sel.options[sel.selectedIndex];if(cyc&&o)" in sync
    head = web_common.SCALE_BOOTSTRAP_HEAD
    assert "(function(){try{" in head and "}catch(e){}})();" in head


def test_scale_picker_resyncs_the_page_to_a_stale_value_fallback():
    """The head bootstrap applies any persisted value pre-paint; a stale one
    rendered the page at that size while the select said "100%"."""
    js = web_common.SCALE_PICKER_JS
    assert ("else if(saved){document.documentElement.style.setProperty("
            "'--fs-base',sel.value+'px');\n"
            "    try{localStorage.removeItem(KEY);}catch(e){}}") in js


def test_header_utility_cluster_self_aligns():
    css = web_common.NAV_CSS
    # drawer mode takes #navrow out of flow, so nothing else in the row grows
    assert re.search(r"header \.hdr-right \{[^}]*margin-left: auto", css)
    spacer = re.search(r"\nheader \.spacer \{[^}]*\}", css).group(0)
    assert "flex: 0 0 0.25rem" in spacer


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


# ---- floating layers stay inside the canvas -------------------------------

def test_shared_popover_placer_is_loaded_on_pages_with_pick_lists(client):
    for path in ("/captures", "/stats"):
        t = client.get(path).text
        assert "window._anchorPopover = _anchorPopover" in t, path
        assert "_renderPickList" in t, path
        # the pick-list opens a top-layer popover placed by the shared ladder
        assert "pop.setAttribute('popover', 'manual')" in t, path
        assert "boundary: function() { return btn.closest('.subbar')" in t, path


def test_pickers_re_place_their_layer_after_a_redraw():
    """The layer is placed at show() time; the pick list then grows from its
    one-line "loading…" note and the language dropdown changes height on
    every search keystroke, so each redraw must re-run the placement."""
    pick = web_common.PICK_LIST_JS
    assert "function replace() { if (ctl && _open === inst) ctl.place(); }" in pick
    draw = pick[pick.index("function draw(needle)"):pick.index("function open()")]
    assert "footer(); replace();" in draw
    # Text typed while the rows loaded is honoured by the first draw.
    assert "draw(q.value.trim().toLowerCase()); label();" in pick
    assert "draw(''); label();" not in pick
    lang = web_common.LANG_PICKER_JS
    body = lang[lang.index("function _renderDropdown(query)"):]
    assert "if (popCtl && popCtl.isOpen()) popCtl.place();" in body


def test_pick_pop_and_activity_pop_are_no_longer_absolute():
    css = web_common.NAV_CSS
    pick = re.search(r"\.pick-pop \{[^}]*\}", css).group(0)
    assert "position: absolute" not in pick
    assert "max-width: min(90vw, var(--col))" in pick
    hact = re.search(r"\.hact-pop \{[^}]*\}", css).group(0)
    assert "position: absolute" not in hact


def test_popover_fallback_portals_even_when_the_layer_is_already_unhidden():
    """pick_list and the activity cluster set hidden=false before show(): the
    fallback isOpen() was then already true and the body portal, nested in
    the `if (!isOpen())` guard, was skipped on the first open."""
    js = web_common.POPOVER_JS
    show = js[js.index("function show() {"):js.index("function hide() {")]
    portal = ("if (!nativePop && popEl.parentNode !== document.body) "
              "document.body.appendChild(popEl);")
    assert show.count("document.body.appendChild(popEl)") == 1
    assert show.index(portal) < show.index("if (!isOpen()) {")


def test_activity_popover_treats_zero_vram_as_no_reading():
    """Sibling of the /stats models-table fix: a cuda NVML delta of 0 is
    stored as vram_mb 0.0, which the header popover rendered as "0.0G"."""
    js = web_common.ACTIVITY_CLUSTER_JS
    assert "m.vram_mb ? gb(m.vram_mb) + 'G' : null" in js
    assert "vram_mb != null" not in js


def test_activity_popover_markup_is_a_native_popover(client):
    t = client.get("/settings").text
    assert 'id="hact-pop" class="hact-pop" popover="manual" hidden' in t


# ---- header height must not depend on the page's body line-height ---------

def test_header_pins_its_own_line_height():
    css = web_common.NAV_CSS
    hdr = re.search(r"\nheader \{[^}]*\}", css).group(0)
    assert re.search(r"line-height: 1\.5;", hdr), hdr
