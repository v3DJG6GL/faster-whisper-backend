"""The brand mark (skewed five-bar waveform tile) exists as several inline
copies: the page header, the login gate, the hub hero, and the brand sources
in docs/brand/. They differ on purpose in gradient id and bar class, but the
geometry and colours must stay identical — a hand edit to one copy would
otherwise drift unnoticed."""
import os
import re

from faster_whisper_backend.admin import home_routes
from faster_whisper_backend.core import web_common
from faster_whisper_backend.paths import REPO_ROOT

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


def _read(*rel):
    with open(os.path.join(REPO_ROOT, *rel), encoding="utf-8") as fh:
        return fh.read()


def _mark(text: str, start_marker: str) -> str:
    i = text.index(start_marker)
    return text[i:text.index("</svg>", i)]


def test_every_inline_brand_mark_copy_has_the_canonical_geometry():
    canonical = _geometry(_read("docs", "brand", "icon.svg"))
    rects, stops, skew = canonical
    assert len(rects) == 6 and stops == ["#79c0ff", "#7ee787"] and len(skew) == 1

    login_gate_js = web_common.OPEN_MODE_BANNER_JS.replace("'\n  +", "").replace("'", "")
    copies = {
        "header (_BRAND_MARK_SVG)": web_common._BRAND_MARK_SVG,
        "login gate": _mark(login_gate_js, 'class="lg-mark"'),
        "hub hero": _mark(home_routes._HUB_HTML, 'class="mark"'),
        "docs/brand/logo.html": _mark(_read("docs", "brand", "logo.html"), '<svg viewBox="0 0 120 120"'),
    }
    for name, svg in copies.items():
        assert _geometry(svg) == canonical, f"{name}: brand mark drifted from docs/brand/icon.svg"
