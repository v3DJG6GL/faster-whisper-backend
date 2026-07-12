"""Root landing hub — GET / serves the WebUI's front door.

Signed out, the page body stays hidden and the shared login gate (injected by
OPEN_MODE_BANNER_JS on the whoami-401) covers the viewport — the visitor sees
exactly the familiar auth screen instead of FastAPI's default 404 JSON.
Signed in, the page reveals a launcher: one "channel strip" tile per WebUI
page, filtered client-side to what the caller's key can actually reach (same
/auth/whoami contract the shared nav uses), plus a slim status strip (model
state, admin-only severity pills, identity, sign-out).

Auth shape matches the other user-tier page shells: the HTML is gated only by
USER_WEBUI_ALLOWED_HOSTS (loopback always allowed); nothing sensitive is
rendered server-side — tile visibility, identity and model state are all
resolved after a successful whoami. The tile hrefs themselves are the same
public knowledge as the shared nav links every page already embeds.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

import build_info
import config as cfg
import system_stats
from html import escape as _esc
from web_common import render_page, require_user_webui_host

router = APIRouter()


# (tier, permission-key, label, href, description, wave, admin_ui_gated)
#   tier "user"  — data page gated per-key via permissions.pages[<key>]
#   tier "any"   — visible to every signed-in identity (no per-page scope)
#   tier "admin" — visible to admins only
# `wave` is the tile's 5-bar micro-waveform (heights out of a 16-unit viewBox)
# — a per-page signature riff on the brand mark (keys = key teeth, dictate =
# speech burst, stats = rising chart). `admin_ui_gated` mirrors _NAV_SPEC:
# those pages are only registered when cfg.ADMIN_UI_ENABLED, so their tiles
# drop server-side alongside them.
_TILE_SPEC: list[tuple[str, str, str, str, str, list[int], bool]] = [
    ("user", "quick_config", "quick config", "/quick-config",
     "curated pipeline rules", [7, 11, 6, 12, 8], True),
    ("user", "captures", "captures", "/captures",
     "fine-tuning audio review", [5, 9, 13, 9, 5], True),
    ("user", "reports", "reports", "/reports",
     "transcription error reports", [10, 6, 11, 7, 12], True),
    ("user", "stats", "stats", "/stats",
     "system dashboard", [4, 7, 10, 13, 9], False),
    ("user", "logs", "logs", "/logs",
     "live log stream", [6, 7, 12, 7, 6], False),
    ("any", "dictate", "dictate", "/dictate",
     "live dictation demo", [8, 11, 13, 11, 8], False),
    ("admin", "settings", "settings", "/settings",
     "full configuration", [8, 9, 10, 9, 8], True),
    ("admin", "pipeline", "pipeline", "/settings/pipeline",
     "post-processing rules", [6, 9, 7, 10, 8], True),
    ("admin", "keys", "keys", "/settings/api-keys",
     "users &amp; API keys", [12, 5, 12, 5, 12], True),
    ("admin", "overrides", "overrides", "/settings/overrides",
     "per-identity overrides", [7, 13, 7, 13, 7], True),
]


def _wave_svg(heights: list[int]) -> str:
    """Render a tile's 5-bar micro-waveform. Bars sit on the baseline of a
    33×16 viewBox; per-bar `--i` drives the staggered rise + hover EQ delays."""
    bars = "".join(
        f'<rect class="wb" style="--i:{i}" x="{1 + i * 6.5:g}" y="{16 - h}" '
        f'width="4" height="{h}" rx="2"/>'
        for i, h in enumerate(heights)
    )
    return f'<svg class="wave" viewBox="0 0 33 16" aria-hidden="true">{bars}</svg>'


def _tile_html(tier: str, key: str, label: str, href: str, desc: str,
               wave: list[int], idx: int) -> str:
    page_attr = f' data-page="{key}"' if tier == "user" else ""
    return (
        f'<a class="tile" data-tier="{tier}" data-hub="{key}"{page_attr} '
        f'href="{href}" style="--td:{idx * 55}ms">'
        f"{_wave_svg(wave)}"
        f'<span class="t-label"><span class="t-prompt" aria-hidden="true">&#9656;</span>{label}</span>'
        f'<span class="t-desc">{desc}</span>'
        f'<kbd class="t-key" aria-hidden="true"></kbd>'
        f"</a>"
    )


def _tiles_html() -> tuple[str, str]:
    """Build the (workspace, admin-zone) tile fragments, honouring
    cfg.ADMIN_UI_ENABLED at request time exactly like web_common._nav_items:
    pages that aren't registered don't get tiles."""
    admin_ui = bool(getattr(cfg, "ADMIN_UI_ENABLED", False))
    user_parts: list[str] = []
    admin_parts: list[str] = []
    idx = 0
    for tier, key, label, href, desc, wave, gated in _TILE_SPEC:
        if gated and not admin_ui:
            continue
        html = _tile_html(tier, key, label, href, desc, wave, idx)
        idx += 1
        (admin_parts if tier == "admin" else user_parts).append(html)
    admin_zone = ""
    if admin_parts:
        admin_zone = (
            '<div class="admin-zone">'
            '<div class="hub-rule">admin</div>'
            f'<nav class="hub-grid" aria-label="Admin pages">{"".join(admin_parts)}</nav>'
            "</div>"
        )
    return "".join(user_parts), admin_zone


_HUB_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
{{SCALE_BOOTSTRAP_HEAD}}
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d; --input-bg: #0d1117;
  }
  /* Same atmosphere as the login gate (#login-gate in NAV_CSS): deep #0b0e14
     plus the two radial brand glows. The gate covers this page for signed-out
     visitors, so keeping the backgrounds identical makes the post-login
     reveal read as the card dissolving into the hub, not a page swap. */
  html { height: 100%; }
  body.hub { margin: 0; min-height: 100%; color: var(--fg);
    font: 1rem/1.5 var(--font-mono);
    background-color: #0b0e14;
    background-image:
      radial-gradient(120% 80% at 50% -10%, rgba(121,192,255,0.12), transparent 60%),
      radial-gradient(90% 60% at 50% 112%, rgba(126,231,135,0.09), transparent 60%);
    background-attachment: fixed; }
  main { max-width: 56rem; margin: 0 auto; padding: 3.2rem 1.25rem 3rem;
    box-sizing: border-box; }
  #hub-body[hidden] { display: none; }

  /* hero — brand mark + family wordmark, bars rise like the login card */
  .hero { display: flex; flex-direction: column; align-items: center; gap: 1rem; }
  .hero .mark { width: 4.75rem; height: 4.75rem;
    filter: drop-shadow(0 0 1.2rem rgba(121,192,255,0.22)); }
  .hero .hb { transform-box: fill-box; transform-origin: center bottom;
    animation: hub-bar-rise .55s ease-out backwards; }
  .hero .hb:nth-child(1) { animation-delay: .06s }
  .hero .hb:nth-child(2) { animation-delay: .13s }
  .hero .hb:nth-child(3) { animation-delay: .20s }
  .hero .hb:nth-child(4) { animation-delay: .27s }
  .hero .hb:nth-child(5) { animation-delay: .34s }
  @keyframes hub-bar-rise { from { transform: scaleY(0.12); opacity: .35 } }
  .word { font-family: "Hubot Sans", var(--font-sans); font-weight: 430;
    font-size: 2.1rem; letter-spacing: -0.02em; line-height: 1; white-space: nowrap; }
  .word .w-a { color: var(--bold); }
  .word .w-b { color: var(--green); font-weight: 730; }
  .word .w-sep { color: var(--green); font-weight: 700; margin: 0 0.3em;
    font-family: "Geist Mono", var(--font-mono); }
  .word .w-c { color: var(--dim); font-weight: 500; font-size: 0.62em;
    font-family: "Geist Mono", var(--font-mono);
    letter-spacing: 0.16em; text-transform: uppercase; }
  /* build line — quiet mono caption between the wordmark and the status
     strip: full version · docker/bare-metal (cpu|gpu) · uptime */
  .buildline { margin: -0.45rem 0 0; font-family: "Geist Mono", var(--font-mono);
    font-size: var(--fs-sm); color: var(--dim); }
  .buildline .v { color: var(--help); }
  .buildline .bl-sep { color: var(--border); margin: 0 0.45em; }

  /* status strip */
  .strip { display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap;
    margin: 2rem 0 1.9rem; padding: 0.55rem 0.9rem;
    background: rgba(22,27,34,0.74); border: 1px solid var(--border);
    border-radius: 0.7rem; font-size: var(--fs-sm);
    -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); }
  .s-model { display: inline-flex; align-items: center; gap: 0.5em;
    color: var(--fg); min-width: 0; }
  .s-model[hidden] { display: none; }
  .s-model .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .dot { width: 0.5em; height: 0.5em; border-radius: 50%; background: var(--dim);
    flex-shrink: 0; }
  .dot.live { background: var(--green); box-shadow: 0 0 0.4rem rgba(126,231,135,0.65); }
  .s-dim { color: var(--dim); }
  /* Severity pills — same markup/ids as the header cluster (SEV_POLLER_JS
     updates by id), restyled locally because NAV_CSS scopes its pill rules
     to `header`. Admin-only, driven by body.hub-admin from the hub's own
     whoami pass (the pills' .admin-only class only has effect in <header>). */
  .hub-sev { display: none; align-items: center; gap: 0.25rem; }
  body.hub-admin .hub-sev { display: inline-flex; }
  .hub-sev .sevpill { font-size: var(--fs-xs); padding: 0.1rem 0.5rem;
    border-radius: 4px; border: 1px solid var(--border); color: var(--dim);
    text-decoration: none; display: inline-flex; gap: 0.3em;
    align-items: baseline; white-space: nowrap; }
  .hub-sev .sevpill .n { font-variant-numeric: tabular-nums; }
  .hub-sev .sevpill.warn.hot { color: var(--yellow); border-color: #4d3e1f; }
  .hub-sev .sevpill.err.hot  { color: var(--red); border-color: #5a2424; }
  .hub-sev .sevpill.crit.hot { color: var(--red); border-color: #5a2424;
    background: #2d1414; }
  .hub-sev .sevpill.zero { opacity: 0.45; }
  .hub-sev .sevpill.flash { animation: sev-flash .6s ease-out; }
  .sp { flex: 1 1 0; min-width: 0.5rem; }
  .s-ident { color: var(--dim); white-space: nowrap; }
  .s-ident .who { color: var(--bold); }
  .icon-btn { display: inline-flex; align-items: center; justify-content: center;
    padding: 0.3rem; line-height: 0; background: transparent; color: var(--dim);
    border: 1px solid transparent; border-radius: 6px; cursor: pointer; }
  .icon-btn svg { width: 1.15em; height: 1.15em; display: block; }
  .icon-btn:hover { color: var(--cyan); border-color: var(--border); }
  .icon-btn:focus-visible { outline: 2px solid var(--cyan); outline-offset: 1px; }
  .icon-btn[hidden], .auth-action[hidden] { display: none; }

  /* launcher grid — "channel strip" tiles. Hidden until the hub's whoami
     pass marks each reachable tile `.on`; the stagger token --td is set
     server-side per tile so the reveal sweeps left-to-right. */
  .hub-grid { display: grid; gap: 0.9rem;
    grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr)); }
  .tile { display: none; }
  .tile.on { display: flex; flex-direction: column; gap: 0.5rem; position: relative;
    padding: 1rem 1.05rem 1.05rem; text-decoration: none; color: var(--fg);
    background: rgba(22,27,34,0.74); border: 1px solid var(--border);
    border-radius: 0.9rem;
    -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px);
    transition: border-color .15s ease, background .15s ease, transform .15s ease;
    animation: hub-tile-in .45s cubic-bezier(.2,.7,.3,1) backwards;
    animation-delay: var(--td, 0ms); }
  @keyframes hub-tile-in { from { opacity: 0; transform: translateY(0.6rem); } }
  .tile.on:hover { border-color: #3d444d; background: rgba(28,34,43,0.85);
    transform: translateY(-2px); }
  .tile.on:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }
  .tile .wave { height: 1.05rem; width: auto; align-self: flex-start; }
  .tile .wb { transform-box: fill-box; transform-origin: center bottom;
    animation: hub-bar-rise .5s ease-out backwards;
    animation-delay: calc(var(--td, 0ms) + var(--i) * 45ms); }
  /* five-step blue→green ramp between the brand gradient's endpoints */
  .tile .wb:nth-of-type(1) { fill: #79c0ff; }
  .tile .wb:nth-of-type(2) { fill: #7acae1; }
  .tile .wb:nth-of-type(3) { fill: #7cd4c3; }
  .tile .wb:nth-of-type(4) { fill: #7ddda5; }
  .tile .wb:nth-of-type(5) { fill: #7ee787; }
  .tile.on:hover .wb, .tile.on:focus-visible .wb {
    animation: hub-bar-eq 0.9s ease-in-out infinite alternate;
    animation-delay: calc(var(--i) * -160ms); }
  @keyframes hub-bar-eq { from { transform: scaleY(0.45); } to { transform: scaleY(1.06); } }
  .t-label { color: var(--bold); font-size: var(--fs-lg); font-weight: 600; }
  .t-prompt { color: var(--green); margin-right: 0.45em; }
  .t-desc { color: var(--help); font-size: var(--fs-sm); line-height: 1.35; }
  .t-key { position: absolute; top: 0.7rem; right: 0.75rem; color: var(--dim);
    font-size: var(--fs-xs); border: 1px solid var(--border); border-radius: 4px;
    padding: 0 0.4em; background: var(--input-bg); font-family: var(--font-mono); }
  .t-key:empty { display: none; }

  /* admin zone + odds and ends */
  .admin-zone { display: none; }
  body.hub-admin .admin-zone { display: block; }
  .hub-rule { display: flex; align-items: center; gap: 0.75rem; color: var(--dim);
    font-size: var(--fs-xs); letter-spacing: 0.16em; text-transform: uppercase;
    margin: 1.7rem 0 0.9rem; }
  .hub-rule::before, .hub-rule::after { content: ""; height: 1px;
    background: var(--border); flex: 1; }
  .hub-note { color: var(--help); font-size: var(--fs-sm); margin: 0 0 1rem;
    padding: 0.6rem 0.9rem; border: 1px dashed var(--border); border-radius: 0.5rem; }
  .hub-note[hidden] { display: none; }
  .kbd-hint { text-align: center; color: var(--dim); font-size: var(--fs-xs);
    margin-top: 2.4rem; }
  .kbd-hint kbd { border: 1px solid var(--border); border-radius: 4px;
    padding: 0 0.35em; background: var(--input-bg); font-family: var(--font-mono); }

  @media (prefers-reduced-motion: reduce) {
    .hero .hb, .tile.on, .tile .wb { animation: none !important; }
  }
  {{NAV_CSS}}
</style></head>
<body class="hub">
<main>
  <section class="hero">
    <svg class="mark" viewBox="0 0 120 120" aria-hidden="true">
      <defs><linearGradient id="fw-hub" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#79c0ff"/><stop offset="1" stop-color="#7ee787"/>
      </linearGradient></defs>
      <rect x="6" y="6" width="108" height="108" rx="26" fill="#161b22" stroke="#30363d" stroke-width="2"/>
      <g transform="translate(13 2) skewX(-9)" fill="url(#fw-hub)">
      <rect class="hb" x="16" y="74" width="11" height="20" rx="5.5"/>
      <rect class="hb" x="35" y="52" width="11" height="42" rx="5.5"/>
      <rect class="hb" x="54" y="22" width="11" height="72" rx="5.5"/>
      <rect class="hb" x="73" y="44" width="11" height="50" rx="5.5"/>
      <rect class="hb" x="92" y="66" width="11" height="28" rx="5.5"/>
      </g>
    </svg>
    <h1 class="word"><span class="w-a">faster</span><span class="w-b">whisper</span><span class="w-sep">&rsaquo;</span><span class="w-c">backend</span></h1>
    <p class="buildline">{{BUILD_LINE}}</p>
  </section>
  <div id="hub-body" hidden>
    <div class="strip">
      <span class="s-model" id="hub-model" hidden><span class="dot" id="hub-dot"></span><span class="name" id="hub-model-name"></span><span class="s-dim" id="hub-model-state"></span></span>
      <span class="hub-sev">{{SEV_PILLS}}</span>
      <span class="sp"></span>
      <span class="s-ident"><span class="who" id="hub-who"></span><span id="hub-role"></span></span>
      {{LOGOUT}}
    </div>
    <p class="hub-note" id="hub-note" hidden>Your API key does not grant access to any workspace pages. Ask an admin to grant access.</p>
    <nav class="hub-grid" aria-label="Pages">{{HUB_TILES}}</nav>
    {{HUB_ADMIN_ZONE}}
    <p class="kbd-hint"><kbd>1</kbd>&#8211;<kbd>9</kbd> jump &#183; <kbd>&#8592;</kbd><kbd>&#8594;</kbd> move &#183; <kbd>&#9166;</kbd> open</p>
  </div>
</main>
{{SEV_POLLER_JS}}
<script>
(function () {
  'use strict';

  function setText(id, s) {
    var el = document.getElementById(id);
    if (el) el.textContent = s;
  }

  function visibleTiles() {
    return Array.prototype.slice.call(document.querySelectorAll('.tile.on'));
  }

  // Fetch the model listing AFTER a successful whoami (the endpoint itself
  // is public, but the strip is auth-only chrome) and fill in the strip's
  // model segment: loaded models when any are warm, else the configured
  // default with an "idle" dot.
  function loadModels() {
    fetch('/v1/models', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.data || !j.data.length) return;
        var loaded = j.data.filter(function (m) { return m.loaded; });
        var names = (loaded.length ? loaded : [j.data[0]])
          .map(function (m) { return m.id; });
        setText('hub-model-name', names.join(' · '));
        setText('hub-model-state', loaded.length ? '· loaded' : '· idle');
        var dot = document.getElementById('hub-dot');
        if (dot) dot.className = 'dot' + (loaded.length ? ' live' : '');
        var wrap = document.getElementById('hub-model');
        if (wrap) {
          wrap.title = loaded.length
            ? 'Model loaded in memory'
            : 'No model loaded — the first request loads it';
          wrap.hidden = false;
        }
      })
      .catch(function () {});
  }

  // The shared chrome's _refreshAuthChrome only applies `.allowed` inside
  // <header>, so the hub runs its own whoami pass for the tiles. On a 401
  // this leaves #hub-body hidden — the shared login gate is already covering
  // the page — and login/logout reload the page anyway (see _signOut and the
  // gate's submit handler), so this pass is effectively the page-load render.
  function apply(j) {
    var hubBody = document.getElementById('hub-body');
    if (!j) {
      if (hubBody) hubBody.hidden = true;
      document.body.classList.remove('hub-admin');
      visibleTiles().forEach(function (el) { el.classList.remove('on'); });
      return;
    }
    var isAdmin = !!j.is_admin;
    var perms = (j.permissions && j.permissions.pages) || {};
    document.body.classList.toggle('hub-admin', isAdmin);
    setText('hub-who', j.open_mode ? 'open mode' : (j.username || 'user'));
    setText('hub-role', j.open_mode ? '' : ' · ' + (isAdmin ? 'admin' : 'user'));
    var anyUserTile = false;
    document.querySelectorAll('.tile').forEach(function (el) {
      var tier = el.getAttribute('data-tier');
      var scope = perms[el.getAttribute('data-page')];
      var on = tier === 'any' || isAdmin
        || (tier === 'user' && scope && scope !== 'none');
      el.classList.toggle('on', !!on);
      if (on && tier === 'user') anyUserTile = true;
    });
    var note = document.getElementById('hub-note');
    if (note) note.hidden = isAdmin || anyUserTile;
    visibleTiles().forEach(function (el, i) {
      var k = el.querySelector('.t-key');
      if (k) k.textContent = i < 9 ? String(i + 1) : '';
    });
    if (hubBody) hubBody.hidden = false;
    loadModels();
  }

  function refresh() {
    fetch('/auth/whoami', {
      headers: { Accept: 'application/json' }, cache: 'no-store',
    })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(apply)
      .catch(function () {});
  }

  // Boot-menu keys: 1-9 open the n-th visible tile, arrows move focus,
  // Enter is the anchors' native activation. Inert while the login gate is
  // up or while typing in a field (the gate's key input, notably).
  document.addEventListener('keydown', function (e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    var gate = document.getElementById('login-gate');
    if (gate && !gate.hidden) return;
    var t = e.target;
    if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
    var tiles = visibleTiles();
    if (!tiles.length) return;
    if (e.key >= '1' && e.key <= '9') {
      var n = +e.key - 1;
      if (tiles[n]) { e.preventDefault(); tiles[n].click(); }
      return;
    }
    var d = (e.key === 'ArrowRight' || e.key === 'ArrowDown') ? 1
      : (e.key === 'ArrowLeft' || e.key === 'ArrowUp') ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    var idx = tiles.indexOf(document.activeElement);
    var next = idx < 0 ? (d > 0 ? 0 : tiles.length - 1)
      : Math.min(tiles.length - 1, Math.max(0, idx + d));
    tiles[next].focus();
  });

  window.addEventListener('whisper:auth-changed', refresh);
  refresh();
})();
</script>
</body></html>"""


def _build_line_html() -> str:
    """The hero's build caption: full version · docker/bare-metal (cpu|gpu) ·
    uptime. Uptime is rendered per page load (the hub reloads often enough
    that a static value stays honest)."""
    variant = f"{build_info.runs_as()} · {'gpu' if system_stats.NVML_OK else 'cpu'}"
    sep = '<span class="bl-sep">·</span>'
    return (
        f'<span class="v">{_esc(build_info.APP_VERSION)}</span>'
        f"{sep}{variant}{sep}up {build_info.uptime_str()}"
    )


@router.get(
    "/",
    response_class=HTMLResponse,
    dependencies=[Depends(require_user_webui_host)],
)
async def home_page():
    # User-tier shell, same contract as /logs and /stats: host-gated HTML, a
    # keyless browser navigation loads the shell + the shared login gate; all
    # data the page shows is fetched with the caller's own credentials.
    tiles, admin_zone = _tiles_html()
    html = (
        _HUB_HTML
        .replace("{{HUB_TILES}}", tiles)
        .replace("{{HUB_ADMIN_ZONE}}", admin_zone)
        .replace("{{BUILD_LINE}}", _build_line_html())
    )
    return HTMLResponse(
        render_page(html, current="home"),
        headers={"Cache-Control": "no-store"},
    )
