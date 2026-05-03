"""
Shared helpers used by the /logs, /config, and /stats web pages.

  - require_allowed_host(allowlist) — FastAPI dependency that 403s callers
    not in the allowlist. Allowlist accepts bare IPs or CIDRs.
  - nav_html(current, request)      — server-rendered nav row HTML.
  - severity_counts()               — log-level counters (last 60 s).
"""

from __future__ import annotations

import ipaddress
import logging
import time
from collections import deque
from typing import Callable

from fastapi import HTTPException, Request, status

import config as cfg


# IPv4-mapped-in-IPv6 prefix surfaces on Windows dual-stack `::` binds when a
# v4 client connects, e.g. "::ffff:127.0.0.1". `ipaddress.ip_address` already
# parses these, but the .ipv4_mapped attribute is what we actually compare on.
def _to_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _build_networks(allowlist: list[str]) -> list[ipaddress._BaseNetwork]:
    nets: list[ipaddress._BaseNetwork] = []
    for entry in allowlist or []:
        entry = entry.strip()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            # Bad entry — skip silently. The /config endpoint validates inputs;
            # this is a runtime defense for handwritten config edits.
            continue
    return nets


def require_allowed_host(allowlist_ref: Callable[[], list[str]]) -> Callable[[Request], None]:
    """Returns a FastAPI dependency that rejects callers outside the allowlist.

    `allowlist_ref` is a zero-arg callable that returns the current allowlist —
    NOT the list itself. This indirection matters: the admin WebUI can edit
    cfg.ADMIN_ALLOWED_HOSTS at runtime, and we want the next request to pick
    up the new value without re-creating the dependency.

    Loopback (`127.0.0.1`, `::1`) is ALWAYS allowed in addition to the
    configured list, so a misconfigured CIDR can never lock the local
    operator out of /config — they can still fix the entry from the box.
    """

    def _dep(request: Request) -> None:
        client = request.client
        if client is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "no client info")
        ip = _to_ip(client.host)
        if ip is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "unparseable client host")
        if ip.is_loopback:
            return
        for net in _build_networks(allowlist_ref()):
            if ip in net:
                return
        raise HTTPException(status.HTTP_403_FORBIDDEN, "host not in allowlist")

    return _dep


# --- Severity ring (in-memory log-level counts, last 60s) ---------------------
# A logging.Handler appends (timestamp, levelno) on every record. The /stats
# page and the nav row read severity_counts() at request time. Bounded ring
# (maxlen=2000) keeps memory predictable under burst logging.
_SEVERITY_LOG: deque[tuple[float, int]] = deque(maxlen=2000)


class SeverityCounter(logging.Handler):
    """Append (time, levelno) to the in-memory severity ring on every record.

    Attached alongside the existing console+file handlers in main.py. WARNING-
    and-up only, so the ring stays small under chatty INFO-level traffic."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _SEVERITY_LOG.append((record.created, record.levelno))
        except Exception:
            # Never let a logging failure kill the request that triggered it.
            pass


def severity_counts(window_sec: float = 60.0) -> dict[str, int]:
    """Return {warn, err, crit} counts from the last `window_sec` seconds."""
    cutoff = time.time() - window_sec
    warn = err = crit = 0
    # Iterate from the right (newest first) and break once we cross the
    # cutoff — the deque is append-ordered, so older entries follow.
    for ts, lvl in reversed(_SEVERITY_LOG):
        if ts < cutoff:
            break
        if lvl >= logging.CRITICAL:
            crit += 1
        elif lvl >= logging.ERROR:
            err += 1
        elif lvl >= logging.WARNING:
            warn += 1
    return {"warn": warn, "err": err, "crit": crit}


# --- Nav row + severity pills ------------------------------------------------

# Inline CSS so each page can drop the nav into its existing <header> without
# duplicating styles. Color tokens reuse the page-level CSS vars.
#
# `header .spacer { flex: 1 }` — single canonical spacer rule. Pages place
# `<span class="spacer"></span>` between the nav block and the action cluster
# so the right side stays right-aligned regardless of how many actions a page
# has.
NAV_CSS = """
header .spacer { flex: 1; }
header .navrow { display: flex; gap: 4px; }
header .navlink { padding: 3px 10px; border-radius: 4px; color: var(--dim);
  text-decoration: none; font-size: 12px; border: 1px solid transparent; }
header .navlink:hover { background: #21262d; color: var(--fg); }
header .navlink.active { color: var(--bold); background: #21262d;
  border-color: var(--border); }
header .sevpill { font-size: 11px; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--dim); text-decoration: none;
  display: inline-flex; gap: 4px; align-items: baseline; }
header .sevpill .n { font-variant-numeric: tabular-nums; }
header .sevpill.warn.hot { color: var(--yellow); border-color: #4d3e1f; }
header .sevpill.err.hot  { color: var(--red);    border-color: #5a2424; }
header .sevpill.crit.hot { color: var(--red);    border-color: #5a2424;
  background: #2d1414; }
header .sevpill.zero { opacity: 0.45; }
@keyframes sev-flash { 0% { background: #5a2424 } 100% { background: transparent } }
header .sevpill.flash { animation: sev-flash .6s ease-out; }
"""


def _nav_items(current: str) -> list[tuple[str, str, bool]]:
    """Return [(label, href, active), ...] honoring cfg.ADMIN_UI_ENABLED."""
    items: list[tuple[str, str, bool]] = [
        ("logs",  "/logs",  current == "logs"),
        ("stats", "/stats", current == "stats"),
    ]
    if getattr(cfg, "ADMIN_UI_ENABLED", False):
        items.append(("config", "/config", current == "config"))
    return items


def nav_html(current: str) -> str:
    """Render the nav row + severity pills as an HTML fragment.

    Pills link to /logs?filter=<level> so a click jumps to the relevant log
    rows. Counts of zero render dimmed; non-zero render colored ("hot")."""
    counts = severity_counts()
    parts: list[str] = ['<span class="navrow">']
    for label, href, active in _nav_items(current):
        cls = "navlink active" if active else "navlink"
        parts.append(f'<a class="{cls}" href="{href}">{label}</a>')
    parts.append("</span>")

    # Stable IDs let JS update just the .n inner span on each SSE tick without
    # rebuilding the link (preserves focus/click state). The initial counts
    # rendered here are a "best effort at page load" — the client takes over
    # immediately, so they're correct for the first render and live thereafter.
    for level, key in (("warn", "WARNING"), ("err", "ERROR"), ("crit", "CRITICAL")):
        n = counts[level]
        cls = f"sevpill {level} {'hot' if n else 'zero'}"
        title = f"{key}+ in the last 60 s — click to filter logs"
        parts.append(
            f'<a id="sev-{level}" class="{cls}" '
            f'href="/logs?filter={key}" title="{title}">'
            f'<span class="lbl">{level}</span> '
            f'<span class="n">{n}</span></a>'
        )
    return "".join(parts)


def render_page(template: str, current: str) -> str:
    """Substitute {{NAV}} and {{NAV_CSS}} placeholders in a page template.

    Pages that don't include the placeholders are returned unchanged."""
    return template.replace("{{NAV}}", nav_html(current)).replace("{{NAV_CSS}}", NAV_CSS)
