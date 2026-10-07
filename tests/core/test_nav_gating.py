"""Header chrome gating — the whoami gates (`.admin-only`, `.page-link`) must
hold wherever a link ends up, and chrome whose whole content is gated must not
leave an empty box or a dangling separator behind.

These are CSS/JS-in-Python strings, so the pins are on the generated text.
"""
from __future__ import annotations

import re

from faster_whisper_backend.core import home_routes
from faster_whisper_backend.core import web_common


def _rule(css: str, selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r" \{[^}]*\}", css)
    assert m, selector
    return m.group(0)


# ---- "more" overflow menu must not reveal gated links ----------------------

def test_more_list_never_sets_display_unconditionally():
    css = web_common.NAV_CSS
    # (0,2,1) would out-rank `header .admin-only` / `header .page-link` (0,1,1)
    assert "display" not in _rule(css, "header .nav-more-list .navlink")
    assert ("header .nav-more-list .navlink:not(.admin-only):not(.page-link),\n"
            "body.role-admin header .nav-more-list .admin-only,\n"
            "header .nav-more-list .page-link.allowed { display: flex; }") in css


def test_tuck_skips_unrendered_links():
    js = web_common.NAV_OVERFLOW_JS
    tuck = js[js.index("function tuck()"):js.index("function layout()")]
    guard = tuck.index("if(el.offsetParent===null)continue;")
    # the guard runs before the node is moved into the list
    assert guard < tuck.index("li.appendChild(el)")
    # separators are still hidden in place, before the rendered-check
    assert tuck.index("el.hidden=true;continue;") < guard


def test_more_list_has_no_separator_rule():
    # tuck() hides a separator in place and never moves one into the list
    assert ".nav-more-list .nav-gsep" not in web_common.NAV_CSS


# ---- all-admin chrome is gated as a whole ----------------------------------

def test_sevpills_wrapper_and_group_hairline_are_hidden_for_non_admins():
    css = web_common.NAV_CSS
    assert "body:not(.role-admin) header .sevpills { display: none; }" in css
    assert "body:not(.role-admin) header .nav-gsep { display: none; }" in css
    # hide-only gates: a `body.role-admin ... { display: ... }` reveal would
    # out-rank the ladder (`header.c9 .sevpills`) and `.nav-gsep[hidden]`
    assert not re.search(r"body\.role-admin header \.(sevpills|nav-gsep) \{", css)


def test_hot_pills_carry_no_dead_border_colour():
    css = web_common.NAV_CSS
    assert "border: 0;" in _rule(css, "header .sevpill")
    for level in ("warn", "err ", "crit"):
        sel = "header .sevpill." + level.strip() + ".hot"
        assert "border-color" not in re.search(
            re.escape(sel) + r"\s*\{[^}]*\}", css).group(0)


# ---- popovers degrade when the shared positioner is missing ----------------

def test_popovers_fall_back_when_the_positioner_is_missing():
    for js in (web_common.PICK_LIST_JS, web_common.ACTIVITY_CLUSTER_JS):
        placer = js[js.index("function placer()"):]
        placer = placer[:placer.index("return ctl;")]
        assert "window._anchorPopover" in placer
        assert "pop.removeAttribute('popover')" in placer



def test_activity_popover_holds_its_markup_while_a_pointer_is_down():
    """A 1 Hz re-render between mousedown and mouseup detached the pressed
    cancel button and dropped the click; logout must close the popover
    through closePop so the placement controller hides too."""
    js = web_common.ACTIVITY_CLUSTER_JS
    render = js[js.index("function renderPop()"):]
    render = render[:render.index("pop.innerHTML = h;")]
    assert "if (pop._held) return;" in render
    assert "pop.addEventListener('pointerdown'" in js
    sync = js[js.index("function syncAllowed()"):]
    sync = sync[:sync.index("var header =")]
    assert "closePop();" in sync and "pop.hidden = true" not in sync


# ---- hub strip: pills are two wrappers below .hub-sev -----------------------

def test_hub_carries_the_pill_gap_into_the_wrappers():
    frag = web_common.sev_pills_html()
    assert 'class="hdr-status"' in frag and 'class="sevpills"' in frag
    assert re.search(
        r"\.hub-sev \.hdr-status, \.hub-sev \.sevpills \{ display: inline-flex;"
        r"\s*align-items: center; gap: 0\.25rem; \}", home_routes._HUB_HTML)


def test_hub_css_never_spells_the_pills_placeholder():
    """render_page substitutes {{SEV_PILLS}} everywhere, so the literal in a
    CSS comment injected the whole pills + activity fragment into the hub's
    <style> block (inert only while the fragment has no `*/`)."""
    css = home_routes._HUB_HTML[:home_routes._HUB_HTML.index("</style>")]
    assert "{{SEV_PILLS}}" not in css


# ---- c8: the lone pills chip gets its border back ---------------------------

def test_activity_fuse_rules_skip_the_c8_step():
    """At c8 the activity half is display:none but keeps .allowed; the fuse
    rule (0,4,1) out-ranked `header.c8 .sevpills` (0,2,1), so the lone pills
    chip kept border-left:0 and square left corners."""
    css = web_common.NAV_CSS
    assert ("header:not(.c8) .hact-wrap:has(.hdr-activity.allowed)"
            " + .sevpills {") in css
    assert ("body.role-admin header:not(.c8) .hdr-status"
            " .hdr-activity.allowed {") in css
    assert "\nheader .hact-wrap:has(" not in css


# ---- activity cluster: no phantom GPU, no endless 401 retry -----------------

def test_activity_cluster_reads_no_gpu_on_a_cpu_only_server():
    """The own-scope lite payload keeps a coarse gpu dict on a GPU-less box;
    only server.gpu.present says so. Both the header readout and the popover
    must go through the present-aware accessor."""
    js = web_common.ACTIVITY_CLUSTER_JS
    acc = js[js.index("function gpuOf(snap)"):]
    acc = acc[:acc.index("\n  }\n")]
    assert "sg.present === false" in acc and "return null" in acc
    feed = js[js.index("function feed(snap)"):js.index("function renderPop()")]
    assert "var gpu = gpuOf(snap);" in feed
    pop = js[js.index("function renderPop()"):js.index("function releaseHold()")]
    assert "var gpu = gpuOf(s)" in pop
    assert "snap.gpu || null;\n    setBar" not in js


def test_activity_cluster_stops_on_a_401_or_403():
    """After the stream ends on an expired session or a revoked stats
    permission, the reopen got 401/403 every 30 s forever in every tab. A
    fatal close probes once; 401/403 hides the button and stops until the
    next auth change."""
    js = web_common.ACTIVITY_CLUSTER_JS
    onerr = js[js.index("es.onerror = function()"):js.index("function closeStream()")]
    assert "fetch('/stats/snapshot?lite=1'" in onerr
    assert "st === 401 || st === 403" in onerr
    deny = onerr[onerr.index("st === 401"):onerr.index("return;")]
    assert "denied = true;" in deny and "btn.hidden = true;" in deny
    # the retry is only scheduled after the probe, never straight away
    assert onerr.index("fetch(") < onerr.index("setTimeout(")
    opener = js[js.index("function openStream()"):js.index("es = new EventSource")]
    assert "denied) return;" in opener
    auth = js[js.index("'whisper:auth-changed'"):]
    assert "denied = false;" in auth[:auth.index("syncAllowed();")]
