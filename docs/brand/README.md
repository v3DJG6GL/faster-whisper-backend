# Brand assets

The icon is an audio waveform skewed 9° forward — *whisper* (the equalizer
bars) meets *faster* (the lean reads as motion) — on a rounded terminal-dark
tile, cyan `#79c0ff` → green `#7ee787`. The wordmark follows the brand-family
grammar shared with
[faster-whisper-frontend](https://github.com/v3DJG6GL/faster-whisper-frontend):
light `faster` (Hubot Sans 430) in ink + bold `whisper` (730) in terminal
green, then a green `>` prompt before the tracked-caps `BACKEND` label
(Geist Mono). See the Brand section of the repo README for the full spec.

| File | What it is |
|---|---|
| `icon.svg` | Icon only, vector. Copy of the canonical `static/logo.svg` (that one is served by the WebUI — keep this copy in sync). |
| `icon.png` | Icon only, 512 px raster. |
| `logo-dark.svg` / `logo-light.svg` | Full logo (icon + wordmark), vector, wordmark converted to paths — renders everywhere with zero font dependencies. Regenerate with `python3 docs/brand/gen-logo-svg.py` (needs fontTools + brotli). |
| `logo-dark.png` / `logo-light.png` | Full logo, raster (@2×, ~1060 px wide), transparent background. The repo README serves them via a `prefers-color-scheme` `<picture>`. |
| `logo.html` | Raster source — renders the logo with the real vendored webfonts (`static/*.woff2`). Regen commands are documented inside the file. |
| `gen-logo-svg.py` | Vector source — draws the wordmark glyph outlines via fontTools and emits the two logo SVGs. |

The WebUI's sticky header renders the compact one-line form of the same logo
(`web_common.py` — `_header_brand_for` + the `.bw-*` rules in `NAV_CSS`).
