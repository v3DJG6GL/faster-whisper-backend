"""Shared page chrome for every WebUI page: render_page, the nav row and
header fragments, the shared JS/CSS snippets, SSE response headers and the
in-memory severity ring. The host-allowlist gates live in auth/hosts.py and
are re-exported here."""

from __future__ import annotations

import functools
import logging
from collections import deque

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.core import templates
# The client-host allowlist gate lives in auth/hosts.py (beside the
# same-origin guard); re-exported here because the page routers reference it
# as web_common.require_user_webui_host / require_admin_webui_host. Plain
# function objects that nothing patches, so each alias IS auth.hosts' object.
from faster_whisper_backend.auth.hosts import host_in_allowlist as host_in_allowlist
from faster_whisper_backend.auth.hosts import require_admin_webui_host as require_admin_webui_host
from faster_whisper_backend.auth.hosts import require_allowed_host as require_allowed_host
from faster_whisper_backend.auth.hosts import require_user_webui_host as require_user_webui_host


# (created, levelno) of every WARNING+ record since process start; bounded
# so burst logging keeps memory flat. The nav pills ship as zeroes and are
# filled from the authenticated GET /sev (no error-rate oracle for an
# unauthenticated caller; see sev_pills_html).
_SEVERITY_LOG: deque[tuple[float, int]] = deque(maxlen=2000)


class SeverityCounter(logging.Handler):
    """Append (time, levelno) to the in-memory severity ring on every record.

    Attached alongside the console+file handlers by core/log_setup.py. WARNING-
    and-up only, so the ring stays small under chatty INFO-level traffic."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _SEVERITY_LOG.append((record.created, record.levelno))
        except Exception:
            # Never let a logging failure kill the request that triggered it.
            pass


def severity_counts() -> dict[str, int]:
    """Return {warn, err, crit} counts since process start (restart resets
    them); capped by the 2000-entry ring under sustained WARNING+ traffic."""
    warn = err = crit = 0
    # Iterate a snapshot: emit() appends from any logging thread, and a
    # Python-level walk of the live deque raises "deque mutated during
    # iteration" (a 500 on /sev, a dead /stats/stream) during exactly the
    # warning bursts these counts exist for. tuple() copies at C level, in
    # one step under the GIL.
    for _ts, lvl in tuple(_SEVERITY_LOG):
        if lvl >= logging.CRITICAL:
            crit += 1
        elif lvl >= logging.ERROR:
            err += 1
        elif lvl >= logging.WARNING:
            warn += 1
    return {"warn": warn, "err": err, "crit": crit}


# Headers every SSE endpoint must send so a buffering reverse proxy streams the
# response instead of accumulating it. Without these, nginx (proxy_buffering on
# by default) holds an infinite text/event-stream body until its buffers/timeout
# fire, then severs the HTTP/2 stream mid-body — Firefox reports
# NS_ERROR_NET_PARTIAL_TRANSFER and the page reconnects every few seconds.
#   X-Accel-Buffering: no  — nginx disables buffering for THIS response even when
#                            proxy_buffering is on globally (the decisive header).
#   Cache-Control: no-store, no-cache, no-transform — no proxy cache; no gzip
#                            rewrite (gzip buffers to compress, which also
#                            breaks SSE).
# `no-store` must stay: the middleware only setdefaults Cache-Control, so this
# value wins, and bare `no-cache` still lets a shared cache STORE the body —
# these streams carry transcript text (/logs/stream, /quick-config/recent/stream).
# Connection is intentionally omitted: it's hop-by-hop, managed by uvicorn/nginx;
# setting it from ASGI is ignored.
SSE_HEADERS = {
    "Cache-Control": "no-store, no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


def sse_response(generator):
    """StreamingResponse preconfigured for proxy-safe Server-Sent Events."""
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        generator, media_type="text/event-stream", headers=dict(SSE_HEADERS),
    )


def validation_error_response(err):
    """422 {"errors": [...]} for a settings ValidationError (the /settings and
    /overrides save contract)."""
    from fastapi import status
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"errors": settings_schema.format_validation_errors(err)},
    )


# Pipeline-stage and job-kind hues — ONE definition for every backend page (/stats
# cards and rings, /logs receipts, /quick-config traces) and the same values
# the desktop app's app.css declares (--c-download, --c-separate, --c-ok for
# transcribing, --c-diarize, --c-translate), so a stage is the same colour
# wherever it appears. `vad` (silence skipping) has no desktop counterpart;
# its lavender is unclaimed by any other stage or state hue. Consumers use
# `var(--stage-<name>)` in CSS; canvas/SVG code resolves the token through
# getComputedStyle (static/stats.js STAGE_COLOR) rather than copying hexes.
STAGE_COLORS: dict[str, str] = {
    "downloading": "#d9a45b",
    "separating": "#6faed9",
    "vad": "#a493e8",
    "transcribing": "#93b76f",
    "diarizing": "#c68fb4",
    "translating": "#4dd0c4",
}
# Job-kind hues (the usage charts' categorical palette): the desktop app's
# --c-chart-dict / -file / -link / -text, validated there for colour-vision
# separation against each other and the surface. Text imports are the faint
# neutral on purpose — they are rare and never a volume story.
KIND_COLORS: dict[str, str] = {
    "dictation": "#cf7b00",
    "file": "#3e96ea",
    "url": "#d76797",
    "text": "#6f675c",
}
_STAGE_TOKENS = "\n".join(
    [f"  --stage-{k}: {v};" for k, v in STAGE_COLORS.items()]
    + [f"  --kind-{k}: {v};" for k, v in KIND_COLORS.items()])

NAV_CSS = templates.load(__file__, "nav.css").replace("{{STAGE_TOKENS}}", _STAGE_TOKENS)


# Injected at the top of every page's <head>:
#   1. the viewport meta — MANDATORY for any responsive CSS to take effect.
#      Without it mobile browsers use a ~980px layout viewport and shrink to
#      fit, so no @media query ever matches. Centralised here (rather than
#      per-page) so a page can never silently ship without it. Standard form
#      only — never add maximum-scale / user-scalable=no (WCAG 1.4.4 failure).
#   2. the favicon links — scalable SVG first (modern browsers), then PNG +
#      ICO fallbacks (Safari/legacy don't render SVG favicons) and an
#      apple-touch-icon. All brand-mark assets live in static/.
#   3. a bootstrap script that applies the persisted UI scale BEFORE the
#      page's CSS parses, avoiding a flash-of-default-size on navigation.
#   4. window._escHtml, the null-safe & < > " HTML escape (' left alone) the
#      shared widgets and page scripts alias as esc / _uEsc: defined here so
#      it exists before every body script, including the inlined pick list.
#   5. window._detailMsg(j, fallback), an error body's toast text: FastAPI's
#      `detail` string, or the first `msg` of a 422 detail list (a bare
#      assignment toasted "[object Object]"), else `fallback`.
SCALE_BOOTSTRAP_HEAD = (
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    '<link rel="icon" type="image/svg+xml" href="/static/favicon.svg">'
    '<link rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32.png">'
    '<link rel="icon" type="image/png" sizes="16x16" href="/static/favicon-16.png">'
    '<link rel="icon" href="/static/favicon.ico" sizes="any">'
    '<link rel="apple-touch-icon" href="/static/apple-touch-icon.png">'
    "<script>window._escHtml=function(s){return String(s==null?'':s)"
    ".replace(/&/g,'&amp;').replace(/</g,'&lt;')"
    ".replace(/>/g,'&gt;').replace(/\"/g,'&quot;');};"
    "window._detailMsg=function(j,fallback){var d=j&&j.detail;"
    "if(typeof d==='string'&&d)return d;"
    "if(Array.isArray(d)&&d[0]&&d[0].msg)return String(d[0].msg);"
    "return fallback;};"
    "(function(){try{var v=localStorage.getItem('whisper-ui-fs-base');"
    "if(v)document.documentElement.style.setProperty('--fs-base',v+'px');"
    "if(localStorage.getItem('whisper-ui-width')==='fluid')"
    "document.documentElement.classList.add('pref-fluid');}catch(e){}})();</script>"
)


# Header scale/width controls — placed just before the action cluster (reload, logout).
SCALE_PICKER_HTML = (
    '<select id="scale-picker" class="scale-picker" title="UI scale">'
    '<option value="13">90%</option>'
    '<option value="15" selected>100%</option>'
    '<option value="17">110%</option>'
    '<option value="18">120%</option>'
    '<option value="20">130%</option>'
    '</select>'
    # Compact stand-in shown by the ladder's c7 step: one click = next step.
    '<button id="scale-cycle" class="icon-btn scale-cycle" type="button" '
    'title="UI scale 100% — click for next" aria-label="UI scale">Aa</button>'
    # Page width preference (fixed 100rem canvas / fluid 150rem). Hidden by
    # NAV_CSS on non-data-canvas pages and while the window is narrower than
    # the fixed canvas.
    '<button id="width-toggle" class="icon-btn width-toggle" type="button" '
    'aria-pressed="false" title="Page width: fixed — click for fluid" '
    'aria-label="Page width">\u27f7</button>'
)


# The {{LOGOUT}} fragment. One HttpOnly session cookie covers every page, so
# one logout works everywhere; OPEN_MODE_BANNER_JS wires the click and reveals
# the (initially hidden) button only while logged in.
LOGOUT_BTN_HTML = (
    '<button id="logout-btn" class="icon-btn auth-action" type="button" '
    'title="Sign out" aria-label="Sign out" hidden>'
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>'
    '<polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>'
    '</svg></button>'
)


# Global reload button — the {{RELOAD}} fragment, in every page's right-hand
# header utility cluster (left of logout). Wired in OPEN_MODE_BANNER_JS: pages
# that expose a soft refresh set `window._pageReload` (settings/keys re-fetch
# their data); everywhere else it falls back to a full location.reload().
# Always visible (a refresh is meaningful regardless of auth state).
RELOAD_BTN_HTML = (
    '<button id="reload-btn" class="icon-btn" type="button" '
    'title="Reload" aria-label="Reload">'
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
    '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'
    '</svg></button>'
)


# Wire-up JS (end of <body>) for the scale picker, its cycle button and the
# width toggle: restores and persists the saved choices. The <head> bootstrap
# only applies them pre-paint; this binds the handlers.
SCALE_PICKER_JS = templates.load(__file__, "scale_picker.js")


# Mobile nav drawer — appended to SCALE_PICKER_JS so every page ships it
# without a new placeholder. Self-contained IIFE: it must not rely on helpers
# defined by later-injected scripts.
NAV_DRAWER_JS = templates.load(__file__, "nav_drawer.js")


# Priority+ nav overflow + compaction ladder. Three things, in order:
#   1. On a single row, shed only the cheap label steps (c1a, c1) until the
#      bar fits.
#   2. If it still doesn't, switch to header.nav-row2 (nav + utility buttons
#      on their own row) and re-run the FULL ladder on the brand row alone.
#   3. If the nav still overflows its row, tuck links — right end first,
#      never the active page, never a lone separator — into the .nav-more
#      disclosure that follows #navrow.
# Everything is measured (scrollWidth vs clientWidth; in two-row mode, "the
# brand and the status rail share a line"), never thresholded, so it is right
# at every --fs-base. Nodes are MOVED, not cloned, so the per-link gating
# classes and the drawer's click-to-close keep working. Re-runs on resize
# (ResizeObserver on the row, the nav, the brand and the status rail — the
# pollers change the rail's width) and when a link's class list changes (the
# whoami gate adds .allowed after first paint). In drawer mode (≤40em,
# .nav-toggle shown) the nav is off-canvas: links are restored, no nav-row2,
# and the ladder runs on the brand row only.
NAV_OVERFLOW_JS = templates.load(__file__, "nav_overflow.js")


POPOVER_JS = templates.load(__file__, "popover.js")

# The central whoami script on every WebUI page (end of <body>, via
# {{SEV_POLLER_JS}}). Fetches /auth/whoami on the HttpOnly session cookie and
# from it sets window.__whoami, the nav gating (.allowed, body.role-admin),
# the logout/reload wiring and the no-access landing. With open_mode=true it
# also prepends a red banner telling the operator to bootstrap an admin key.
OPEN_MODE_BANNER_JS = templates.load(__file__, "open_mode_banner.js")


# Shared timestamp helpers (own <script> block; pages opt in with a
# `{{TIME_HELPERS_JS}}` line). Format contract: YYYY.MM.DD | HH:MM:SS, 24-hour;
# fmtWhen appends " | 5m ago"-style suffixes within 24 h. timeTick() (default
# `[data-ts]`, 30 s) re-renders those elements so the suffixes age in place.
TIME_HELPERS_JS = templates.load(__file__, "time_helpers.js")


# No-access landing card for a valid bearer that lacks access to the current
# page: reads `window.__whoami` (set by OPEN_MODE_BANNER_JS) and renders one
# button per page the caller CAN reach plus Sign out, or a "no pages
# available" message. Raw (no <script> wrapper) for pages that inject it into
# their own IIFE via {{NOT_ADMIN_LANDING_JS}}. `_renderNotAdminLanding` is a
# legacy alias with no remaining in-tree caller.
NOT_ADMIN_LANDING_JS = templates.load(__file__, "not_admin_landing.js")


# Global copy, injected by render_page beside OPEN_MODE_BANNER_JS so its
# auto-landing path can call `_renderNoAccessLanding` on pages that never
# include the {{NOT_ADMIN_LANDING_JS}} placeholder.
NOT_ADMIN_LANDING_GLOBAL_JS = "<script>" + NOT_ADMIN_LANDING_JS + "</script>"


# Shared tag-picker widget used on /settings (per-rule tag editor) and
# /settings/api-keys (per-user tag editor in the permissions matrix).
# DOM-pure factory: `_renderTagPicker(opts) -> { el, getTags, setTags,
# setAvailable }`. Caller mounts the returned element wherever and
# subscribes to `opts.onChange(newTags)`.
#
# The JS mirrors `settings_schema.TAG_RE` for a red border only; the server
# (set_user_permissions / Pydantic) re-validates, so a JS bypass can't store a
# malformed tag. `available` = tags in use, offered as autocomplete.
TAG_PICKER_JS = templates.load(__file__, "tag_picker.js")


# Shared pick-list widget — a button that opens a searchable checklist.
# `window._renderPickList(opts)` is DOM-pure (no page globals), like
# _renderTagPicker. Used by /stats for the users / keys filters (rows ranked
# by the measure from /stats/pick) and by /captures for the speaker filter
# (rows ranked by hours from /captures/api/stats).
#
#   opts = {
#     mount:       existing element to become the root (keeps page ids), or
#                  omit and append the returned `el` yourself;
#     wordPlural:  button text ("users"), placeholder defaults to "search …";
#     title, ariaLabel, anyLabel ("any"), errorNote;
#     fetchRows:   () -> Promise<[{id, label, value?, sub?, me?, stale?}]>,
#                  called on every open (rows may change with the window);
#     fmt:         value -> string for the right-hand column (optional);
#     picked:      initial ids;  multi: true (checkboxes) | false (radios);
#     onChange:    (ids) -> void, fired per tick / clear.
#   }
#   returns { el, getPicked, setPicked(ids), setLabels({id: label}), close }
#
# One document-level outside-click / Escape handler serves every instance.
PICK_LIST_JS = templates.load(__file__, "pick_list.js")


LANG_PICKER_JS = templates.load(__file__, "lang_picker.js")


# Severity pill poller: GET /sev every 5 s. It is the sole pill updater —
# /stats SSE deliberately defers to it.
SEV_POLLER_JS = templates.load(__file__, "sev_poller.js")


# Header activity cluster: fills the activity_cluster_html shell from
# /stats/stream?lite=1 (SSE) on every page EXCEPT /stats, where the page's
# own full-payload renderer feeds it through window._fwFeedActivity to avoid
# a second stream. Emitted with SEV_POLLER_JS on every template.
ACTIVITY_CLUSTER_JS = templates.load(__file__, "activity_cluster.js")


# Per-rule body editors shared by /settings and /quick-config; each page
# passes its own `commitData` callback (fired on every input/change) and does
# its own dirty-tracking. Keep in lockstep with settings/schema.py: a new rule
# type needs a Pydantic class there plus, in templates/rule_editor.js, a
# renderTypeEditor branch and a _PIPELINE_TYPES entry.

# cb:map entry cap read off settings_schema.MapRule and baked into
# RULE_EDITOR_JS, so the editor's "n / cap" readout can never drift from the
# save-path bound on any page that embeds it.
_MAP_MAX_ENTRIES: int = next(
    (m.max_length
     for m in settings_schema.MapRule.model_fields["map"].metadata
     if getattr(m, "max_length", None) is not None),
    10_000,  # fallback mirrors MapRule.map's max_length
)

# The repeat switch's help text names the pass cap the engine applies.
RULE_EDITOR_JS = (templates.load(__file__, "rule_editor.js")
                  .replace("__MAP_CAP__", str(_MAP_MAX_ENTRIES))
                  .replace("__REPEAT_CAP__", str(settings_schema.RULE_REPEAT_MAX_PASSES)))


def _nav_items(current: str) -> list[tuple[str, str, bool]]:
    """Return [(label, href, active), ...] in left-to-right display order.
    `_NAV_SPEC` is the single source of truth for order and gating: links
    whose admin_gated flag is set only exist when cfg.ADMIN_UI_ENABLED
    registered the admin UI."""
    admin = getattr(cfg, "ADMIN_UI_ENABLED", False)
    return [
        (label, href, current == key)
        for label, href, key, admin_gated in _NAV_SPEC
        if admin or not admin_gated
    ]


# (label, href, current-key, admin_gated) — left-to-right nav order.
_NAV_SPEC: list[tuple[str, str, str, bool]] = [
    ("quick",    "/quick-config",      "quick-config", True),
    ("stats",    "/stats",             "stats",        False),
    ("reports",  "/reports",           "reports",      True),
    ("captures", "/captures",          "captures",     True),
    ("logs",     "/logs",              "logs",         False),
    # No per-page permission key and no admin gate: /dictate is the streaming
    # endpoint's own front end, reachable by every signed-in identity (the
    # WebSocket enforces the same auth as the API). It therefore renders as a
    # plain navlink — neither `admin-only` nor `page-link` — which is exactly
    # what nav_html does for a label absent from both maps.
    ("dictate",  "/dictate",           "dictate",      False),
    ("settings", "/settings",          "settings",     True),
    ("keys",     "/settings/api-keys", "api-keys",     True),
    ("pipeline", "/settings/pipeline", "pipeline",     True),
    ("overrides", "/settings/overrides", "overrides",  True),
]


def nav_html(current: str) -> str:
    """Render the primary nav links as the {{NAV}} fragment. The severity
    pills are rendered separately by sev_pills_html() ({{SEV_PILLS}}) so they
    can live in the header's right-hand utility cluster."""
    # Two visibility tracks, both default-hidden by CSS:
    # - "admin-only": all-or-nothing on `body.role-admin`, which
    #   OPEN_MODE_BANNER_JS's /auth/whoami pass sets for admins (quick-config
    #   and captures also set it from their own load); the sev pills use it too.
    # - "page-link" + data-page="<X>": OPEN_MODE_BANNER_JS adds `.allowed`
    #   when /auth/whoami's permissions.pages[X] grants it (admins always do).
    # Hamburger toggle (shown only ≤40em via NAV_CSS) + the nav links. The
    # links keep their admin-only / page-link gating classes; on narrow
    # screens NAV_CSS turns the same #navrow into an off-canvas drawer and
    # NAV_DRAWER_JS wires open/close + focus handling.
    parts: list[str] = [
        '<button class="nav-toggle" type="button" aria-label="Menu" '
        'aria-expanded="false" aria-controls="navrow">☰</button>',
        '<span class="navrow" id="navrow">',
    ]
    for label, href, active in _nav_items(current):
        if label == "settings" and len(parts) > 2:
            # hairline between the work group and the admin group
            parts.append('<span class="nav-gsep" aria-hidden="true"></span>')
        classes = ["navlink"]
        extra_attr = ""
        if active:
            classes.append("active")
            extra_attr += ' aria-current="page"'
        if label in _ADMIN_ONLY_LABELS:
            classes.append("admin-only")
        elif label in _PAGE_LINK_LABELS:
            classes.append("page-link")
            extra_attr += f' data-page="{_PAGE_LINK_LABELS[label]}"'
        parts.append(
            f'<a class="{" ".join(classes)}" href="{href}"{extra_attr}>'
            f'{label}</a>'
        )
    parts.append("</span>")
    # Priority+ overflow shell (outside #navrow so its list escapes the
    # clip). NAV_OVERFLOW_JS moves links in/out; disclosure semantics only.
    parts.append(
        '<span class="nav-more" hidden>'
        '<button type="button" class="nav-more-btn" aria-haspopup="true" '
        'aria-expanded="false" aria-controls="nav-more-list">more '
        '<span class="cnt"></span></button>'
        '<ul id="nav-more-list" class="nav-more-list" hidden></ul></span>'
    )
    # forced line break for header.nav-row2 (display:none otherwise)
    parts.append('<span class="nav-brk" aria-hidden="true"></span>')
    parts.append('<div class="nav-backdrop"></div>')
    return "".join(parts)


def activity_cluster_html() -> str:
    """Header activity cluster — an empty shell (jobs count + GPU/VRAM
    micro-bars + spinner) plus its popover container, emitted as part of
    {{SEV_PILLS}} so every template gets it with zero template edits.

    Static shell, zeroes only (same caching stance as the sev pills: page
    shells are host-gated and memoized, so no live value may render here).
    ACTIVITY_CLUSTER_JS fills it from /stats/stream?lite=1 — visibility is
    driven by the stats nav-link's `.allowed` class (the same whoami gate),
    mirrored onto the button; default-hidden via [hidden] + no .allowed."""
    return (
        '<span class="hact-wrap">'
        '<button id="hact" class="hdr-activity page-link" data-page="stats" '
        'type="button" hidden aria-haspopup="true" aria-expanded="false" '
        'title="server activity — click for running jobs">'
        '<span class="hact-jobs"><span class="hact-ring"></span>'
        '<span id="hact-jobs" class="v">0</span></span>'
        '<span class="hact-m gpum" title="GPU util"><span class="lbl">GPU</span>'
        '<span class="hact-bar"><i id="hact-gpu"></i></span>'
        '<span id="hact-gpuv" class="v">&ndash;</span></span>'
        '<span class="hact-m vramm" title="VRAM"><span class="lbl">VRAM</span>'
        '<span class="hact-bar vram"><i id="hact-vram"></i></span>'
        '<span id="hact-vramv" class="v">&ndash;</span></span>'
        '</button>'
        '<div id="hact-pop" class="hact-pop" popover="manual" hidden></div>'
        '</span>'
    )


def sev_pills_html() -> str:
    """Render the warn/err/crit pills (plus the activity cluster) as the
    {{SEV_PILLS}} fragment for the header's utility cluster. Stable IDs let
    SEV_POLLER_JS rewrite just the `.n` span on each /sev tick (preserving
    focus/click state).

    Rendered as ZEROES, deliberately: the page shell is gated only by a host
    allowlist (USER_WEBUI_ALLOWED_HOSTS defaults to 0.0.0.0/0) while GET /sev
    requires auth, so baking live counts in would hand an unauthenticated
    caller an error-rate oracle. Authenticated users get the numbers on the
    first poll."""
    # Activity cluster first: it sits left of the pills in the utility
    # cluster and rides the same placeholder so all templates inherit it.
    parts: list[str] = ['<span class="hdr-status">', activity_cluster_html(),
                        '<span class="sevpills">']
    for level, key in (("warn", "WARNING"), ("err", "ERROR"), ("crit", "CRITICAL")):
        cls = f"sevpill admin-only {level} zero"
        title = f"{key}+ since process start — click to filter logs"
        parts.append(
            f'<a id="sev-{level}" class="{cls}" '
            f'href="/logs?filter={key}" title="{title}">'
            f'<span class="lbl">{level}</span> '
            f'<span class="n">0</span></a>'
        )
    parts.append("</span></span>")
    return "".join(parts)


@functools.lru_cache(maxsize=64)
def _render_page_cached(
    template: str,
    current: str,
    admin_ui: bool,
    log_initial: int,
    log_dom: int,
    seg_shown: int,
    stage_colors: bool = True,
) -> str:
    """The actual placeholder substitution, memoized on everything it can
    vary on.

    Rebuilding the shell per request measured 2.55 ms of event-loop CPU for a
    269,666-byte body on the real /captures template — and these pages render
    BEFORE any credential is examined (the router dependency is
    require_user_webui_host only, USER_WEBUI_ALLOWED_HOSTS defaults to
    0.0.0.0/0, and the login gate is a client-side overlay shipped inside this
    very response), so an unauthenticated caller can drive both the CPU and the
    ~270 KB of egress at will.

    The key is exhaustive by construction. All 24 placeholders were traced:
    13 substitute module-level constants; {{SEV_PILLS}} hardcodes n = 0 by
    design (see sev_pills_html); {{HEADER_VTAG}} is computed once at import;
    {{NAV}} varies only on `current` plus cfg.ADMIN_UI_ENABLED; the rest vary
    on `current`, the two LOG_VIEWER_* values, LOG_SEGMENT_ROWS_SHOWN or
    LOG_STAGE_COLORS. No nonce, CSRF token,
    username or per-request version is substituted server-side — if one is
    ever added it MUST enter this key, or the cache becomes a cross-user
    poisoning bug. The cfg reads are hot-mutable via the settings save
    path, hence their presence in the key rather than a read at import.

    Returns an immutable str, so sharing one object across callers is safe.
    """
    return (
        template
        .replace("{{NAV}}", nav_html(current))
        .replace("{{SEV_PILLS}}", sev_pills_html())
        .replace("{{NAV_CSS}}", NAV_CSS)
        .replace("{{LOG_VIEWER_INITIAL_LINES}}", str(log_initial))
        .replace("{{LOG_VIEWER_DOM_MAX}}", str(log_dom))
        .replace("{{LOG_SEGMENT_ROWS_SHOWN}}", str(seg_shown))
        .replace("{{LOG_STAGE_COLORS}}", "true" if stage_colors else "false")
        .replace("{{SCALE_PICKER}}", SCALE_PICKER_HTML)
        .replace("{{RELOAD}}", RELOAD_BTN_HTML)
        .replace("{{LOGOUT}}", LOGOUT_BTN_HTML)
        .replace("{{SCALE_PICKER_JS}}",
                 POPOVER_JS + SCALE_PICKER_JS + NAV_DRAWER_JS + NAV_OVERFLOW_JS)
        .replace(
            "{{SEV_POLLER_JS}}",
            # Order matters: the global landing helpers must be defined
            # BEFORE OPEN_MODE_BANNER_JS runs, because the central script
            # calls `_renderNoAccessLanding` once whoami resolves. The
            # activity cluster comes last — it observes the .allowed class
            # OPEN_MODE_BANNER_JS's chrome refresh sets.
            SEV_POLLER_JS + NOT_ADMIN_LANDING_GLOBAL_JS + OPEN_MODE_BANNER_JS
            + ACTIVITY_CLUSTER_JS,
        )
        .replace("{{SCALE_BOOTSTRAP_HEAD}}", SCALE_BOOTSTRAP_HEAD)
        .replace("{{RULE_EDITOR_JS}}", RULE_EDITOR_JS)
        .replace("{{TIME_HELPERS_JS}}", TIME_HELPERS_JS)
        .replace("{{NOT_ADMIN_LANDING_JS}}", NOT_ADMIN_LANDING_JS)
        .replace("{{PAGE_META}}", _page_meta_tag(current))
        .replace("{{PAGE_CLASS}}", _col_class_for(current))
        .replace("{{TAG_PICKER_JS}}", TAG_PICKER_JS)
        .replace("{{LANG_PICKER_JS}}", LANG_PICKER_JS)
        .replace("{{PICK_LIST_JS}}", PICK_LIST_JS)
        .replace("{{HEADER_TITLE}}", _header_title_for(current))
        .replace("{{HEADER_BRAND}}", _header_brand_for(current))
        .replace("{{HEADER_VTAG}}", _HEADER_VTAG_HTML)
    )


def render_page(template: str, current: str) -> str:
    """Substitute placeholders in a page template:
      - {{NAV}}                  → primary nav links (left of global bar)
      - {{SEV_PILLS}}            → severity pills (right utility cluster)
      - {{NAV_CSS}}              → shared header/scale-token CSS
      - {{SCALE_PICKER}}         → scale dropdown + scale-cycle and width-toggle buttons (header)
      - {{SCALE_PICKER_JS}}      → wire-up script (end of body); also carries
                                   POPOVER_JS (window._anchorPopover, the
                                   shared dropdown placement ladder),
                                   NAV_DRAWER_JS and NAV_OVERFLOW_JS
      - {{SEV_POLLER_JS}}        → 5-s pill re-sync + the global no-access
                                   landing helpers + open-mode admin-key
                                   warning banner + ACTIVITY_CLUSTER_JS
                                   (end of body)
      - {{RELOAD}} / {{LOGOUT}}  → header reload / logout buttons
      - {{LOG_VIEWER_INITIAL_LINES}} / {{LOG_VIEWER_DOM_MAX}} /
        {{LOG_SEGMENT_ROWS_SHOWN}} / {{LOG_STAGE_COLORS}}
                                 → /logs viewer knobs from cfg (DOM_MAX 0
                                   resolves to initial × 4)
      - {{SCALE_BOOTSTRAP_HEAD}} → tiny pre-paint script (top of <head>)
      - {{RULE_EDITOR_JS}}       → shared per-rule body editors
      - {{TIME_HELPERS_JS}}      → absTime / relTime / fmtWhen / stampTs / timeTick
      - {{NOT_ADMIN_LANDING_JS}} → _renderNoAccessLanding() helper
                                   (+ _renderNotAdminLanding alias)
      - {{PAGE_META}}            → <meta name="page-key" ...> carrier so
                                   shared JS knows which page it's on
      - {{PAGE_CLASS}}           → body class naming the page canvas width
                                  (col-read / col-form / col-data / col-fluid)
      - {{TAG_PICKER_JS}}        → window._renderTagPicker(opts) widget
                                   shared by /settings rule editor +
                                   /settings/api-keys permissions matrix
      - {{LANG_PICKER_JS}}       → window._renderLanguagePicker(opts) pill
                                   picker for language-code lists
      - {{PICK_LIST_JS}}         → window._renderPickList(opts) searchable
                                   checklist shared by /stats (users, keys)
                                   and /captures (speakers)
      - {{HEADER_TITLE}}         → uniform page-title string —
                                   "faster-whisper-backend · <slug>"
                                   plain text, used inside <title>
      - {{HEADER_BRAND}}         → branded header lockup (waveform mark +
                                   wordmark, links to /) for <span class="title">
      - {{HEADER_VTAG}}          → build-version chip (hover = full build
                                   tooltip, click = copy) placed right after
                                   the .title span in the global bar

    Pages that don't include a given placeholder are returned unchanged."""
    _log_initial = int(getattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000))
    _log_dom = int(getattr(cfg, "LOG_VIEWER_DOM_MAX", 0)) or (_log_initial * 4)
    _admin_ui = bool(getattr(cfg, "ADMIN_UI_ENABLED", False))
    _seg_shown = int(getattr(cfg, "LOG_SEGMENT_ROWS_SHOWN", 15) or 15)
    _stage_colors = bool(getattr(cfg, "LOG_STAGE_COLORS", True))
    return _render_page_cached(
        template, current, _admin_ui, _log_initial, _log_dom, _seg_shown,
        _stage_colors,
    )


# Maps the `current=` argument passed by each page route into the
# permission-key used by api_keys_store.PAGES. The admin-only pages
# (settings, api-keys, ...) map to the __admin_only__ sentinel, so the JS can
# tell "no per-page perm to check" from a data page's permissions key. Pages
# absent from this map (dictate, home) get no page-key meta tag at all, so
# they don't participate in per-page scope gating; the central JS
# short-circuits for them.
_PAGE_KEY_BY_CURRENT: dict[str, str] = {
    "logs":         "logs",
    "stats":        "stats",
    "quick-config": "quick_config",
    "reports":      "reports",
    "captures":     "captures",
    "settings":     "__admin_only__",
    "pipeline":     "__admin_only__",
    "api-keys":     "__admin_only__",
    "overrides":    "__admin_only__",
}

# Address-bar URL per page, the no-access landing's heading slug: the
# permission key alone would say "No access to /api-keys" for
# /settings/api-keys.
_PAGE_PATH_BY_CURRENT: dict[str, str] = {
    key: href for _label, href, key, _gated in _NAV_SPEC
    if key in _PAGE_KEY_BY_CURRENT
}

# nav_html's two visibility tracks, derived from the same page map: a
# data page's nav label -> its permission key, and the admin-only labels.
_PAGE_LINK_LABELS: dict[str, str] = {
    label: _PAGE_KEY_BY_CURRENT[key] for label, _href, key, _gated in _NAV_SPEC
    if _PAGE_KEY_BY_CURRENT.get(key, "__admin_only__") != "__admin_only__"
}
_ADMIN_ONLY_LABELS = frozenset(
    label for label, _href, key, _gated in _NAV_SPEC
    if _PAGE_KEY_BY_CURRENT.get(key) == "__admin_only__"
)


# Human-readable slug per page for the {{HEADER_TITLE}} <title> string
# `faster-whisper-backend · <slug>`.
_HEADER_SLUG_BY_CURRENT: dict[str, str] = {
    "logs":         "logs",
    "stats":        "stats",
    "settings":     "settings",
    "api-keys":     "API keys",
    "quick-config": "quick config",
    "reports":      "reports",
    "captures":     "captures",
    "overrides":    "overrides",
    "pipeline":     "pipeline",
    "dictate":      "dictate",
}


def _header_title_for(current: str) -> str:
    """{{HEADER_TITLE}}: plain-text (no markup) page title for <title>."""
    slug = _HEADER_SLUG_BY_CURRENT.get(current, "")
    if not slug:
        return "faster-whisper-backend"
    return f"faster-whisper-backend · {slug}"


# Inline brand mark: the forward-skewed audio-waveform tile (same geometry
# as docs/brand/icon.svg; tests/core/test_brand_mark.py keeps the inline
# copies in step). Sized in em so it tracks the --fs-base UI scale.
_BRAND_MARK_SVG = (
    '<svg class="brand-mark" viewBox="0 0 120 120" aria-hidden="true" focusable="false">'
    '<defs><linearGradient id="fw-hdr" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#79c0ff"/><stop offset="1" stop-color="#7ee787"/>'
    '</linearGradient></defs>'
    '<rect x="6" y="6" width="108" height="108" rx="26" fill="#161b22" '
    'stroke="#30363d" stroke-width="2"/>'
    '<g transform="translate(13 2) skewX(-9)" fill="url(#fw-hdr)">'
    '<rect x="16" y="74" width="11" height="20" rx="5.5"/>'
    '<rect x="35" y="52" width="11" height="42" rx="5.5"/>'
    '<rect x="54" y="22" width="11" height="72" rx="5.5"/>'
    '<rect x="73" y="44" width="11" height="50" rx="5.5"/>'
    '<rect x="92" y="66" width="11" height="28" rx="5.5"/>'
    '</g></svg>'
)


def _header_brand_for(current: str) -> str:
    """{{HEADER_BRAND}} HTML for <span class="title">: waveform mark +
    wordmark `fasterwhisper > backend`, linking to the / hub. No page slug —
    the active nav link (aria-current) names the page. `current` is unused,
    kept for a uniform {{...}}-builder signature."""
    return (
        '<a class="brand-link" href="/" title="Home">'
        f"{_BRAND_MARK_SVG}"
        '<span class="brand-word">'
        '<span class="bw-a">faster</span><span class="bw-b">whisper</span>'
        '<span class="bw-sep">&gt;</span><span class="bw-c">backend</span>'
        "</span></a>"
    )


# {{HEADER_VTAG}}: an empty shell only — the shared header rides on host-gated
# (keyless) pages, so the version/boot/start facts must never be baked into
# it. _fillBuildChip() in OPEN_MODE_BANNER_JS writes them in from the
# authenticated /auth/whoami payload; until then the chip is :empty and the
# CSS hides it. _fwCopyBuild is defined here (once per page) and reused by any
# other copy-report button (e.g. the /settings server-identity card).
def _header_vtag_html() -> str:
    return (
        '<button id="hdr-vtag" class="vtag" type="button" data-tip="" '
        'data-build="" aria-label="Copy server build info" '
        'onclick="_fwCopyBuild(this)"></button>'
        # navigator.clipboard needs a secure context (https / localhost); over
        # plain-http LAN fall back to textarea + execCommand (same approach as
        # the api-keys page's copy button).
        "<script>\n"
        "function _fwCopyBuild (btn) {\n"
        "  var txt = btn.getAttribute('data-build') || '';\n"
        "  var done = function (ok) { if (!ok) return;\n"
        "    var old = btn.classList.contains('copied')\n"
        "      ? btn.getAttribute('data-lbl') : btn.textContent;\n"
        "    btn.setAttribute('data-lbl', old);\n"
        "    btn.classList.add('copied'); btn.textContent = 'copied';\n"
        "    setTimeout(function () { btn.textContent = old;\n"
        "      btn.classList.remove('copied'); }, 900);\n"
        "  };\n"
        "  if (navigator.clipboard && window.isSecureContext) {\n"
        "    navigator.clipboard.writeText(txt)\n"
        "      .then(function () { done(true); }, function () { done(false); });\n"
        "  } else {\n"
        "    var ta = document.createElement('textarea'); ta.value = txt;\n"
        "    ta.style.position = 'fixed'; ta.style.opacity = '0';\n"
        "    document.body.appendChild(ta); ta.select();\n"
        "    var ok = false; try { ok = document.execCommand('copy'); } catch (_) {}\n"
        "    document.body.removeChild(ta); done(ok);\n"
        "  }\n"
        "}\n"
        "</script>"
    )


_HEADER_VTAG_HTML = _header_vtag_html()


# Page canvas width class stamped on <body> (see NAV_CSS :root --col-*).
# Every `current` value render_page() is called with must be listed: an
# unknown page falls back to the data canvas, and the test suite pins the
# table against the nav spec so a new page cannot silently land unmapped.
_COL_CLASS_BY_CURRENT: dict[str, str] = {
    "home":         "col-read",
    "quick-config": "col-form",
    "dictate":      "col-form",
    "settings":     "col-form",
    "pipeline":     "col-form",
    "stats":        "col-data",
    "reports":      "col-data",
    "captures":     "col-data",
    "api-keys":     "col-data",
    "overrides":    "col-data",
    "logs":         "col-fluid",
}


def _col_class_for(current: str) -> str:
    return _COL_CLASS_BY_CURRENT.get(current, "col-data")


def _page_meta_tag(current: str) -> str:
    """Render a `<meta name="page-key" ...>` tag describing the current
    page. Carries both the permission-key (which the central script uses
    to decide whether to auto-render the landing) and the full URL path
    (which the landing renders as the heading slug — matches the address
    bar instead of just the permission slug)."""
    key = _PAGE_KEY_BY_CURRENT.get(current, "")
    if not key:
        return ""
    path = _PAGE_PATH_BY_CURRENT.get(current, "")
    extra = f' data-page-path="{path}"' if path else ""
    return f'<meta name="page-key" content="{key}"{extra}>'
