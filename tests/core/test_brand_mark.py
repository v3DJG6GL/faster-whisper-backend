"""The brand mark (skewed five-bar waveform tile) exists as several copies: the
page header, the login gate and the hub hero (rendered by
web_common.brand_mark_svg), and the brand sources in docs/brand/. They differ
on purpose in gradient id and bar class, but the geometry and colours must stay
identical — a hand edit to one copy would otherwise drift unnoticed."""
import re

from faster_whisper_backend.core import home_routes
from faster_whisper_backend.core import web_common
from tests._page_helpers import js_upto
from tests.conftest import read_repo_text as _read

_RECT = re.compile(r'<rect\b([^>]*)/?>')
_ATTR = re.compile(r'\b(x|y|width|height|rx|fill|stroke|stroke-width)="([^"]*)"')
_STOPS = re.compile(r'stop-color="(#[0-9a-fA-F]{6})"')


def _geometry(svg: str):
    rects = []
    for m in _RECT.finditer(svg):
        attrs = dict(_ATTR.findall(m.group(1)))
        # The bars' fill is url(#<per-copy gradient id>); compare only the shape.
        if attrs.get("fill", "").startswith("url("):
            attrs.pop("fill")
        rects.append(tuple(sorted(attrs.items())))
    skew = re.findall(r'transform="(translate\(13 2\) skewX\(-9\))"', svg)
    return rects, _STOPS.findall(svg), skew


def test_every_inline_brand_mark_copy_has_the_canonical_geometry():
    canonical = _geometry(_read("docs", "brand", "icon.svg"))
    rects, stops, skew = canonical
    assert len(rects) == 6 and stops == ["#79c0ff", "#7ee787"] and len(skew) == 1

    login_gate_js = web_common.OPEN_MODE_BANNER_JS.replace("'\n  +", "").replace("'", "")
    copies = {
        "header (_BRAND_MARK_SVG)": web_common._BRAND_MARK_SVG,
        "login gate": js_upto(login_gate_js, 'class="lg-mark"', "</svg>"),
        "hub hero": js_upto(home_routes._HUB_HTML, 'class="mark"', "</svg>"),
        "docs/brand/logo.html": js_upto(
            _read("docs", "brand", "logo.html"), '<svg viewBox="0 0 120 120"', "</svg>"),
        # The generator of the committed logo-{dark,light}.svg.
        "docs/brand/gen-logo-svg.py": _read("docs", "brand", "gen-logo-svg.py")
        .split('BE_MARK = """', 1)[1].split('"""', 1)[0],
    }
    for name, svg in copies.items():
        assert _geometry(svg) == canonical, f"{name}: brand mark drifted from docs/brand/icon.svg"
