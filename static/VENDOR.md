# Vendored static assets

These files are committed to the repo so the `/stats` dashboard works
fully offline — no CDN fetch at page-load.

## uPlot

- **Version**: 1.6.32
- **Source**: https://github.com/leeoniya/uPlot
- **License**: MIT
- **Files**:
  - `uplot.iife.min.js` (~50 KB) — IIFE build, exposes the global `uPlot`.
  - `uplot.min.css`     (~2 KB)  — default theme (we override colors via CSS vars).

## GridStack

- **Version**: UNKNOWN — recorded as the floating `10.x`, and the bundle
  embeds no version string, so the exact release cannot be recovered from the
  file. The SHA-256 below still detects any change to what is committed, but
  a re-download cannot be verified against it until someone pins the exact
  version here. Do that at the next GridStack update.
- **Source**: https://github.com/gridstack/gridstack.js
- **License**: MIT
- **Files**:
  - `gridstack.min.js`  (~80 KB) — UMD build (gridstack-all), exposes the global `GridStack`.
  - `gridstack.min.css` (~4 KB)  — default theme (we override colors via CSS vars).
- **Used by**: `/stats` dashboard for drag-to-reorder + click-to-resize tiles.

## Brand fonts (Hubot Sans, Geist Mono)

- **Source**: the `@fontsource-variable/hubot-sans` and `@fontsource-variable/geist-mono`
  packages (same pinned files the faster-whisper-frontend app bundles).
- **License**: SIL OFL-1.1 (both).
- **Files**:
  - `hubot-sans-latin-wght-normal.woff2` (~48 KB) — variable weight 200–900.
  - `geist-mono-latin-wght-normal.woff2` (~30 KB) — variable weight 100–900.
- **Used by**: the header brand logo on every WebUI page (family wordmark
  grammar shared with faster-whisper-frontend), and `docs/brand/logo.html`
  for rendering the README logo PNGs.

## Swagger UI / ReDoc (API docs pages)

- **Versions**: `swagger-ui-dist` 5.30.2, `redoc` 2.5.0
- **Source**: https://github.com/swagger-api/swagger-ui,
  https://github.com/Redocly/redoc
- **License**: Apache-2.0 (both)
- **Files**:
  - `swagger-ui-bundle.js`  (~1.5 MB) — exposes the global `SwaggerUIBundle`.
  - `swagger-ui.css`        (~152 KB) — self-contained (all images are data: URIs).
  - `redoc.standalone.js`   (~890 KB) — exposes the global `Redoc`.
- **Used by**: `/docs` and `/redoc` in `main.py`.
- **Why vendored**: FastAPI's `get_swagger_ui_html` / `get_redoc_html` default
  to `cdn.jsdelivr.net` (and `fastapi.tiangolo.com` for the favicon). Both
  pages render in the app's own origin and are opened by an admin carrying a
  session cookie, so whoever controls that CDN response executes code with
  admin rights against this backend. Same reasoning as uPlot/GridStack above.
  `main.py` passes explicit `/static/...` URLs for the JS, the CSS and the
  favicon — if you ever drop those arguments, the CDN defaults come back.
- **Known residue**: `redoc.standalone.js` renders a Redocly logo from
  `https://cdn.redoc.ly/redoc/logo-mini.svg` behind an `onError` fallback. It
  is an image, not executable code, and the page degrades cleanly without it —
  but it does mean opening `/redoc` makes one outbound image request.
  `swagger-ui-bundle.js` and `swagger-ui.css` make no load-time external
  requests at all.
- **Known residue (2)**: `/docs` used to emit swagger-ui's OnlineValidatorBadge,
  an `<img>`/`<a>` pointing at `https://validator.swagger.io/validator?url=<this
  server's openapi.json>` — suppressed only while the admin browses over
  loopback, because its guard is a "localhost"/"127.0.0.1" substring test on
  the definition URL. `main._swagger_ui` now passes
  `swagger_ui_parameters={"validatorUrl": None}`, which drops the badge
  entirely. Keep that parameter on any future edit to that handler: vendoring
  the bundle does not prevent this, it is a runtime config default.

## SHA-256 digests

As vendored — verify after any re-download. Every executable asset here loads
into the admin origin, so all of them are pinned, not just the docs bundles.

  - `swagger-ui-bundle.js` `002503ad9e92c33a9c9e2f7d4910a6fba4dd9dd8c57cfdb53b090629df0f5787`
  - `swagger-ui.css`       `bc5e8d5c013477cf1f35e2fb8ba1dff66be0f72f24e669a509635657145e1acb`
  - `redoc.standalone.js`  `0ec05be285ac885a330289b02f470e1bdbd2b6b3223a9fa213f24bf805a851d1`
  - `uplot.iife.min.js`    `19c8d4c6ad88929a79f4ae49d6f7161566dfd0ba3d15cc495e974f787eb78f1f`
  - `uplot.min.css`        `df630c6a8d6f8eeaff264b50f73ce5b114f646ffd9a0bb74f049b0a00135fa04`
  - `gridstack.min.js`     `52c37cdf838aa8ead6156f0c778cde9af24e87e38dc6a0c3ffef7c6cb7d879cb`
  - `gridstack.min.css`    `1c232b47b98089dd61dd55d24ebd6e89f5be347d746a5e3d714d3aac385aca1f`
  - `hubot-sans-latin-wght-normal.woff2` `bf14f3d03f7e62cf5039f85fc2f0bf4a8022b679ba5ca6f876fec0d73b175f66`
  - `geist-mono-latin-wght-normal.woff2` `af61b969e7f999969f6af576e584ee85dca301a008a76be1251d172d56b9904c`

## How to update

```bash
curl -sL -o uplot.iife.min.js \
  "https://cdn.jsdelivr.net/npm/uplot@<NEW_VERSION>/dist/uPlot.iife.min.js"
curl -sL -o uplot.min.css \
  "https://cdn.jsdelivr.net/npm/uplot@<NEW_VERSION>/dist/uPlot.min.css"
curl -sL -o gridstack.min.js \
  "https://cdn.jsdelivr.net/npm/gridstack@<NEW_VERSION>/dist/gridstack-all.js"
curl -sL -o gridstack.min.css \
  "https://cdn.jsdelivr.net/npm/gridstack@<NEW_VERSION>/dist/gridstack.min.css"
curl -sL -o swagger-ui-bundle.js \
  "https://cdn.jsdelivr.net/npm/swagger-ui-dist@<NEW_VERSION>/swagger-ui-bundle.js"
curl -sL -o swagger-ui.css \
  "https://cdn.jsdelivr.net/npm/swagger-ui-dist@<NEW_VERSION>/swagger-ui.css"
curl -sL -o redoc.standalone.js \
  "https://cdn.jsdelivr.net/npm/redoc@<NEW_VERSION>/bundles/redoc.standalone.js"

# Record the new digests in this file (ALL of them, not just the changed one):
sha256sum swagger-ui-bundle.js swagger-ui.css redoc.standalone.js \
          uplot.iife.min.js uplot.min.css gridstack.min.js gridstack.min.css \
          hubot-sans-latin-wght-normal.woff2 geist-mono-latin-wght-normal.woff2
```

Pin an EXACT version for every URL above (never a floating `@5` major) so a
re-download is reproducible and the recorded digests stay meaningful.

Then bump the version in this file. Do not hand-edit the JS or CSS — keep
them byte-identical to the upstream release so `git blame` stays meaningful.
