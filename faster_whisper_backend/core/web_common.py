"""
Shared helpers used by the /logs, /settings, /stats, and /quick-config pages.

  - require_allowed_host(allowlist) — FastAPI dependency that 403s callers
    not in the allowlist. Allowlist accepts bare IPs or CIDRs. Defined in
    auth/hosts.py, re-exported here with the two tier gates.
  - nav_html(current)               — server-rendered nav row HTML.
  - severity_counts()               — WARNING+ counts since process start
                                      (bounded by the 2000-entry ring).
"""

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


# --- Severity ring (in-memory log-level counts, since process start) ---------
# A logging.Handler appends (timestamp, levelno) on every record. The /stats
# page reads severity_counts() at request time; the nav pills ship as ZEROES
# and are filled client-side by SEV_POLLER_JS from the authenticated GET /sev
# (deliberately -- see sev_pills_html's docstring: the host-allowlisted page
# shell must not hand an unauthenticated caller an error-rate oracle).
# Bounded ring
# (maxlen=2000) keeps memory predictable under burst logging — once the ring
# fills, oldest entries fall off and the per-level counters cap accordingly.
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
    """Return {warn, err, crit} counts since process start.

    Reads the entire in-memory _SEVERITY_LOG ring. The ring is bounded at
    2000 entries — under sustained WARNING+ traffic, oldest entries fall off
    and the counter caps. In practice this matches a per-run "session
    counter" the user investigates via the /logs?filter=<level> link on each
    pill. Restart resets to zero."""
    warn = err = crit = 0
    for _ts, lvl in _SEVERITY_LOG:
        if lvl >= logging.CRITICAL:
            crit += 1
        elif lvl >= logging.ERROR:
            err += 1
        elif lvl >= logging.WARNING:
            warn += 1
    return {"warn": warn, "err": err, "crit": crit}


# --- Server-Sent Events ------------------------------------------------------

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
# `no-store` is load-bearing and must stay: the middleware only *defaults*
# Cache-Control to no-store (setdefault), so an explicit value here wins, and
# bare `no-cache` still lets a shared cache STORE the body — these streams carry
# transcript text (/logs/stream, /quick-config/recent/stream). It changes no
# proxy behaviour: X-Accel-Buffering and no-transform are what protect
# streaming, and stats_page already ships the same combination.
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


# --- Nav row + severity pills ------------------------------------------------

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

# Inline CSS so each page can drop the nav into its existing <header> without
# duplicating styles. Color tokens reuse the page-level CSS vars.
#
# `header .spacer { flex: 0 0 0.25rem }` — single canonical spacer rule: a
# fixed gap, NOT a grower. Pages place `<span class="spacer"></span>` between
# the nav block and the action cluster; `header .navrow` absorbs the slack in
# the single row and `header .hdr-right { margin-left: auto }` keeps the
# cluster right-aligned when the nav leaves the flow (drawer mode). The spacer
# is hidden entirely in nav-row2.
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
SCALE_BOOTSTRAP_HEAD = (
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    '<link rel="icon" type="image/svg+xml" href="/static/favicon.svg">'
    '<link rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32.png">'
    '<link rel="icon" type="image/png" sizes="16x16" href="/static/favicon-16.png">'
    '<link rel="icon" href="/static/favicon.ico" sizes="any">'
    '<link rel="apple-touch-icon" href="/static/apple-touch-icon.png">'
    "<script>(function(){try{var v=localStorage.getItem('whisper-ui-fs-base');"
    "if(v)document.documentElement.style.setProperty('--fs-base',v+'px');"
    "if(localStorage.getItem('whisper-ui-width')==='fluid')"
    "document.documentElement.classList.add('pref-fluid');}catch(e){}})();</script>"
)


# Header dropdown HTML — placed just before the action cluster (logout etc.).
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
    # NAV_CSS while the window is narrower than the fixed canvas.
    '<button id="width-toggle" class="icon-btn width-toggle" type="button" '
    'aria-pressed="false" title="Page width: fixed — click for fluid" '
    'aria-label="Page width">\u27f7</button>'
)


# Global sign-out button — the {{LOGOUT}} fragment, rendered into every page's
# right-hand header utility cluster. Auth is a single HttpOnly session cookie
# shared across all pages, so one logout works everywhere; the click handler +
# visibility toggle live in OPEN_MODE_BANNER_JS. Starts `hidden`
# (class .auth-action) and is revealed only while logged in. Icon-only to
# save space, but with an authoritative title + aria-label (outward door-arrow).
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


# Wire-up JS — placed at the end of <body>. Restores the saved value into
# the dropdown and persists future selections. Independent of the <head>
# bootstrap (which only sets the inline style); this binds the change handler.
SCALE_PICKER_JS = templates.load(__file__, "scale_picker.js")


# Mobile nav drawer wire-up — appended to SCALE_PICKER_JS at the end of
# <body> on every page (so it ships without a new per-page placeholder).
# Self-contained IIFE (no reliance on later-defined helpers — see the
# injected-JS-parse-order pitfall). Toggles `header.nav-open`, traps focus
# by marking the rest of the page `inert`, closes on Esc / backdrop / link,
# and restores focus to the toggle on close.
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

# Open-mode warning banner — JS-injected at the top of <body> on every
# WebUI page. Fetches /auth/whoami; if open_mode=true, prepends a red
# banner reminding the operator to bootstrap an admin key. Auth rides the
# HttpOnly session cookie, sent automatically (no manual header).
OPEN_MODE_BANNER_JS = templates.load(__file__, "open_mode_banner.js")


# Shared timestamp helpers — one source of truth across every admin page.
# Pages opt in by inserting `{{TIME_HELPERS_JS}}` at the top of their inline
# <script> block; render_page substitutes the constant below.
#
# Format contract: HH:MM:SS | YYYY.MM.DD — 24-hour clock, dot-separated date,
# space-pipe-space separator. fmtWhen appends a relative suffix (e.g.
# " | 5m ago") while the event is within 24 h, then drops it.
#
# `timeTick(rootSelector, intervalMs=30000)` walks every element matching
# rootSelector (defaults to `[data-ts]`) and re-renders its textContent from
# fmtWhen(parseFloat(el.dataset.ts)). Cheap; pages that want their cards'
# relative suffixes to age in place can call timeTick() once at boot.
TIME_HELPERS_JS = templates.load(__file__, "time_helpers.js")


# No-access landing card used when the bearer is valid but the caller
# lacks access to the current page. Replaces the old `Admin only`
# hard-coded card. The new version reads `window.__whoami` (set by
# OPEN_MODE_BANNER_JS) to list every other page the caller CAN reach,
# rendering one button per accessible page plus a Sign-out button.
# Falls back to a "no pages available" message when the caller has
# scope=none everywhere.
#
# `_renderNotAdminLanding` is kept as a thin alias so existing callers
# (`reports_routes._renderAdminOnlyIfNonAdmin`, the similar inline helper
# in captures_routes, and api_keys_routes._check403) keep working.
#
# Two flavours:
#   - NOT_ADMIN_LANDING_JS         — raw (no <script> wrapper). Injected
#                                    inside a page's existing <script>
#                                    IIFE via {{NOT_ADMIN_LANDING_JS}}.
#                                    Used by pages that already host a
#                                    big inline IIFE (reports, captures,
#                                    api_keys).
#   - NOT_ADMIN_LANDING_GLOBAL_JS  — full <script> wrapper that puts the
#                                    helpers on the global window. Pulled
#                                    in alongside OPEN_MODE_BANNER_JS so
#                                    every page has access regardless of
#                                    whether it includes the placeholder.
NOT_ADMIN_LANDING_JS = templates.load(__file__, "not_admin_landing.js")


# Global wrapper around NOT_ADMIN_LANDING_JS — puts the helpers on the
# window so every page has access regardless of whether it includes the
# {{NOT_ADMIN_LANDING_JS}} placeholder. Injected via render_page next to
# OPEN_MODE_BANNER_JS so the centralised auto-landing path can call
# `_renderNoAccessLanding` for /logs, /stats, /quick-config without
# requiring their templates to opt in to the older placeholder.
NOT_ADMIN_LANDING_GLOBAL_JS = "<script>" + NOT_ADMIN_LANDING_JS + "</script>"


# Shared tag-picker widget used on /settings (per-rule tag editor) and
# /settings/api-keys (per-user tag editor in the permissions matrix).
# DOM-pure factory: `_renderTagPicker(opts) -> { el, getTags, setTags,
# setAvailable }`. Caller mounts the returned element wherever and
# subscribes to `opts.onChange(newTags)`.
#
# Tag format matches the server-side `settings_schema.TAG_RE`: lowercase
# letters/digits/hyphens, 1-32 chars, no leading/trailing hyphen.
# Validation happens BOTH client-side (visual red border on bad input)
# AND server-side (set_user_permissions / Pydantic validator) so a
# JS-side bypass can't smuggle malformed tags into the DB.
#
# Autocomplete: caller passes `available` = the union of every tag
# currently in use (rule tags for the matrix UI, or all rule tags for
# the rule editor). The widget suggests matches as the user types;
# clicking a suggestion adds the tag.
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


# Severity pill poller — placed at the end of <body> on every page that shows
# the nav. Polls /sev every 5 s and writes the counts into the three pills.
# Server-side severity_counts() is the authoritative source (WARNING+ records
# since process start, ring-bounded). The poller is the sole pill updater:
# /stats SSE explicitly defers to it, and the /logs per-line bumps were
# dropped — so every page just trusts the 5 s tick.
#
# Skips the work if no pills exist on the page (e.g. tests, future pages).
SEV_POLLER_JS = templates.load(__file__, "sev_poller.js")


# Header activity cluster: fills the activity_cluster_html shell from
# /stats/stream?lite=1 (SSE) on every page EXCEPT /stats, where the page's
# own full-payload renderer feeds it through window._fwFeedActivity to avoid
# a second stream. Emitted with SEV_POLLER_JS on every template.
ACTIVITY_CLUSTER_JS = templates.load(__file__, "activity_cluster.js")


# Per-rule body editors shared by /settings (full editor with drag-reorder etc.)
# and /quick-config (read-only header + body editor only). Defined as
# top-level functions so both pages can call them with their own
# `commitData` callback. The `commitData` argument is invoked by every
# input/change event inside an editor; each page implements its own dirty-
# tracking on top of that callback.
#
# Keep the per-type rendering here in lockstep with settings/schema.py rule
# schemas. Adding a new rule type requires:
#   1. New Pydantic class in settings/schema.py
#   2. New `if (rule.type === '<type>')` branch in renderTypeEditor below
#   3. New entry in _PIPELINE_TYPES for the pill label
# Schema-derived cap on cb:map entries, read off settings_schema.MapRule so the
# editor's "n / cap" readout can never drift from the save-path bound. Baked
# into RULE_EDITOR_JS below so EVERY page embedding the shared editor ships
# it -- previously only /quick-config's load() assigned window.__mme, leaving
# /settings with a bare count and a never-disabled add button.
_MAP_MAX_ENTRIES: int = next(
    (m.max_length
     for m in settings_schema.MapRule.model_fields["map"].metadata
     if getattr(m, "max_length", None) is not None),
    10_000,  # fallback mirrors MapRule.map's max_length
)

RULE_EDITOR_JS = templates.load(__file__, "rule_editor.js").replace("__MAP_CAP__", str(_MAP_MAX_ENTRIES))


def _nav_items(current: str) -> list[tuple[str, str, bool]]:
    """Return [(label, href, active), ...] in left-to-right display order,
    honoring cfg.ADMIN_UI_ENABLED. `_NAV_SPEC` is the single source of truth
    for nav order — edit it to reorder the bar. The trailing flag marks links
    that only exist when the admin UI is registered (quick-config / reports /
    captures / settings / api-keys); logs + stats are always served and so
    render unconditionally."""
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
    # Two visibility tracks:
    #
    #   - "admin-only" — settings + api-keys + sev pills. These stay all-or-
    #     nothing on the existing `body.role-admin` class (CSS hides by
    #     default; pages add the class after a successful admin-API ping).
    #
    #   - "page-link" + data-page="<X>" — logs / stats / quick-config /
    #     reports / captures. Per-user gated by OPEN_MODE_BANNER_JS via
    #     /auth/whoami → permissions.pages[X]. CSS default-hides; the JS
    #     adds `.allowed` per link the caller can reach. Admins always
    #     pass via the is_admin short-circuit on the server side.
    page_link_labels: dict[str, str] = {
        "logs":     "logs",
        "stats":    "stats",
        "quick":    "quick_config",
        "reports":  "reports",
        "captures": "captures",
    }
    admin_only_labels = {"settings", "pipeline", "keys", "overrides"}
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
        if label in admin_only_labels:
            classes.append("admin-only")
        elif label in page_link_labels:
            classes.append("page-link")
            extra_attr += f' data-page="{page_link_labels[label]}"'
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
    """Render the three severity-count pills (warn/err/crit) as the
    {{SEV_PILLS}} fragment, grouped in a `.sevpills` wrapper for the header's
    right-hand utility cluster. Split out of nav_html so the pills sit with
    the other status/utility chrome (scale picker) rather than beside the
    nav links.

    Stable IDs let SEV_POLLER_JS update just the `.n` inner span on each /sev
    tick without rebuilding the link (preserves focus/click state).

    Rendered as ZEROES, deliberately. Every page shell that carries this
    fragment is gated by a host allowlist ONLY — USER_WEBUI_ALLOWED_HOSTS
    defaults to 0.0.0.0/0 — while GET /sev, which serves these same three
    integers, requires authentication and calls itself user-tier. Baking the
    live counts into the shell handed an unauthenticated caller a running
    error-rate oracle for the server, one HTTP hop inside the gate the product
    puts on exactly that data. SEV_POLLER_JS already rewrites these `.n` spans
    from /sev, so for an authenticated user the only change is that the numbers
    arrive on the first poll instead of in the initial HTML."""
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
      - {{SCALE_PICKER}}         → scale dropdown (header)
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
      - {{TIME_HELPERS_JS}}      → absTime / relTime / fmtWhen / timeTick
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
                                   wordmark + slug) for <span class="title">
      - {{HEADER_VTAG}}          → build-version chip (hover = full build
                                   tooltip, click = copy) placed right after
                                   the .title span in the global bar

    Pages that don't include a given placeholder are returned unchanged."""
    # Resolve the /logs DOM cap: 0 in config means "auto = initial × 4".
    # Computed here so the JS gets a final integer and doesn't need its
    # own resolver. These reads stay per-call (they're two attribute reads)
    # and are passed into the memo key, because the settings save path
    # mutates them at runtime.
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
# (settings, api-keys, ...) map to the __admin_only__ sentinel below. Pages
# absent from this map (dictate, home) get no page-key meta tag at all, so
# they don't participate in per-page scope gating; the central JS
# short-circuits for them.
_PAGE_KEY_BY_CURRENT: dict[str, str] = {
    "logs":         "logs",
    "stats":        "stats",
    "quick-config": "quick_config",
    "reports":      "reports",
    "captures":     "captures",
    # Admin-only pages: emit a sentinel so the JS can distinguish
    # "no per-page perm to check" from "we're on a data page that maps
    # to a key in the permissions dict".
    "settings":     "__admin_only__",
    "pipeline":     "__admin_only__",
    "api-keys":     "__admin_only__",
    "overrides":    "__admin_only__",
}

# Display-URL for each page, used as the heading slug in the no-access
# landing. Keeps the rendered slug in lockstep with what the user typed
# in the address bar (the permission-key alone is misleading for nested
# routes like /settings/api-keys — without this, the landing would say
# "No access to /api-keys").
_PAGE_PATH_BY_CURRENT: dict[str, str] = {
    "logs":         "/logs",
    "stats":        "/stats",
    "quick-config": "/quick-config",
    "reports":      "/reports",
    "captures":     "/captures",
    "settings":     "/settings",
    "pipeline":     "/settings/pipeline",
    "api-keys":     "/settings/api-keys",
    "overrides":    "/settings/overrides",
}


# Human-readable slug per page, used in the uniform header string
# `faster-whisper-backend · <slug>`. Centralising this here means
# adding a page or renaming one is a single-line change instead of
# touching every template's <title> + <span class="title">.
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
    """Build the uniform page-title string substituted into every page's
    <title> via {{HEADER_TITLE}} (plain text — no markup). Format:
    `faster-whisper-backend · <slug>`. Unknown `current` values fall
    back to the app name alone."""
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
    """Build the branded header lockup substituted into every page's
    <span class="title"> via {{HEADER_BRAND}} (HTML — kept separate from
    the plain-text {{HEADER_TITLE}} used inside <title>). Renders the
    waveform mark + compact wordmark `fasterwhisper › backend`. The current
    page is conveyed by the active nav link (aria-current), so the brand no
    longer repeats it as a slug. The whole lockup links to the / landing hub
    so every page has a one-click way back to the launcher. `current` is
    unused but kept for a uniform {{...}}-builder signature."""
    return (
        '<a class="brand-link" href="/" title="Home">'
        f"{_BRAND_MARK_SVG}"
        '<span class="brand-word">'
        '<span class="bw-a">faster</span><span class="bw-b">whisper</span>'
        '<span class="bw-sep">&gt;</span><span class="bw-c">backend</span>'
        "</span></a>"
    )


# ── header build-version tag ({{HEADER_VTAG}}) ──────────────────────────────
# Empty shell only — the shared header rides on host-gated (keyless) pages, so
# the version/boot/start facts must never be baked into it. _fillBuildChip()
# in OPEN_MODE_BANNER_JS writes them in from the authenticated /auth/whoami
# payload; until then the chip is :empty and the CSS hides it. _fwCopyBuild is
# defined here (once per page) and reused by any other copy-report button
# (e.g. the /settings server-identity card).
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
