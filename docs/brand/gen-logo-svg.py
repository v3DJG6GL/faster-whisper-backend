#!/usr/bin/env python3
"""Regenerate docs/brand/logo-{dark,light}.svg — the logo with the wordmark
text converted to paths (fontTools), so the SVGs render identically everywhere
with zero font dependencies. Run from the repo root:

    python3 docs/brand/gen-logo-svg.py

Needs: fontTools + brotli (pip install fonttools brotli).
"""

import sys
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools.pens.svgPathPen import SVGPathPen

FONT_DIR = "static"
HUBOT = f"{FONT_DIR}/hubot-sans-latin-wght-normal.woff2"
GEIST = f"{FONT_DIR}/geist-mono-latin-wght-normal.woff2"

_cache = {}
def face(path, weight):
    key = (path, weight)
    if key not in _cache:
        f = TTFont(path)
        _cache[key] = instancer.instantiateVariableFont(f, {"wght": weight})
    return _cache[key]

def run_paths(text, font, size, tracking_em, fill, x, baseline_y):
    """Return (svg_paths, end_x) for a text run starting at x."""
    upm = font["head"].unitsPerEm
    scale = size / upm
    cmap = font.getBestCmap()
    glyphs = font.getGlyphSet()
    hmtx = font["hmtx"]
    track = tracking_em * size
    out = []
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:
            x += size * 0.5  # crude fallback advance (unused glyphs)
            continue
        pen = SVGPathPen(glyphs)
        glyphs[gname].draw(pen)
        d = pen.getCommands()
        if d:  # spaces have no outline
            out.append(
                f'<path transform="translate({x:.2f} {baseline_y:.2f}) '
                f'scale({scale:.6f} {-scale:.6f})" fill="{fill}" d="{d}"/>'
            )
        x += hmtx[gname][0] * scale + track
    return "".join(out), x

# ---- the two marks, verbatim from each repo's canonical artwork -------------
FE_MARK = """<g transform="translate(0 10) scale(0.1015625)">
  <defs>
    <linearGradient id="tile" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#262019"/><stop offset="1" stop-color="#0e0d0b"/>
    </linearGradient>
    <linearGradient id="bar" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffb95e"/><stop offset="1" stop-color="#ff9e2c"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" rx="224" fill="url(#tile)"/>
  <g fill="url(#bar)" transform="translate(512 512) skewX(-9) translate(-512 -512)">
    <rect x="152" y="392" width="104" height="240" rx="52"/>
    <rect x="308" y="292" width="104" height="440" rx="52"/>
    <rect x="464" y="192" width="104" height="640" rx="52"/>
    <rect x="620" y="332" width="104" height="360" rx="52"/>
    <rect x="776" y="432" width="104" height="160" rx="52"/>
  </g>
</g>"""

BE_MARK = """<g transform="translate(0 10) scale(0.8667)">
  <defs>
    <linearGradient id="fw" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#79c0ff"/><stop offset="1" stop-color="#7ee787"/>
    </linearGradient>
  </defs>
  <rect x="6" y="6" width="108" height="108" rx="26" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <g transform="translate(13 2) skewX(-9)" fill="url(#fw)">
    <rect x="16" y="74" width="11" height="20" rx="5.5"/>
    <rect x="35" y="52" width="11" height="42" rx="5.5"/>
    <rect x="54" y="22" width="11" height="72" rx="5.5"/>
    <rect x="73" y="44" width="11" height="50" rx="5.5"/>
    <rect x="92" y="66" width="11" height="28" rx="5.5"/>
  </g>
</g>"""

VARIANTS = {
    ("frontend", "dark"):  dict(ink="#f3eee6", accent="#ff9e2c", faint="#6f675c"),
    ("frontend", "light"): dict(ink="#221b13", accent="#bf6f12", faint="#9a9082"),
    ("backend", "dark"):   dict(ink="#f0f6fc", accent="#7ee787", faint="#8b949e"),
    ("backend", "light"):  dict(ink="#1f2328", accent="#1a7f37", faint="#59636e"),
}

def build(product, theme, out_path):
    c = VARIANTS[(product, theme)]
    label = product.upper()
    mark = FE_MARK if product == "frontend" else BE_MARK
    # Geometry mirroring logo.html (values measured against the PNG renders):
    # mark 104px tall at y=10; title baseline ~54, sub baseline ~114.
    text_x = 104 + 40
    title_y, sub_y = 56.0, 114.0
    parts = [mark]
    p, x = run_paths("faster", face(HUBOT, 430), 60, -0.025, c["ink"], text_x, title_y)
    parts.append(p)
    p, x = run_paths("whisper", face(HUBOT, 730), 60, -0.025, c["accent"], x, title_y)
    parts.append(p)
    title_end = x
    p, x = run_paths(">", face(GEIST, 700), 40, 0.14, c["accent"], text_x, sub_y)
    parts.append(p)
    p, x = run_paths(" " + label, face(GEIST, 500), 40, 0.14, c["faint"], x, sub_y)
    parts.append(p)
    width = round(max(title_end, x) + 2)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="124" '
        f'viewBox="0 0 {width} 124" role="img" '
        f'aria-label="fasterwhisper — {product}">\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )
    open(out_path, "w").write(svg)
    print(out_path, f"{width}x124")

if __name__ == "__main__":
    for theme in ("dark", "light"):
        build("backend", theme, f"docs/brand/logo-{theme}.svg")
