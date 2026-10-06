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
