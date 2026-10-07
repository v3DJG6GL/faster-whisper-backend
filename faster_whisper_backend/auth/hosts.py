"""Request-origin gates: the client-host allowlist (require_allowed_host and
the two tier gates, require_user_webui_host / require_admin_webui_host, plus
the non-raising host_in_allowlist) and the same-origin guard for unsafe
methods (_origin_is_allowed, with its throttled rejection log) that main's
CSRF middleware and the streaming WebSocket handshake share. Imports only
config and store_common.
"""
from __future__ import annotations

import ipaddress
import logging
import time
from typing import Callable

from fastapi import HTTPException, Request, status
from starlette.requests import HTTPConnection

from faster_whisper_backend.core import store_common
from faster_whisper_backend.settings import config as cfg

logger = logging.getLogger("whisper-api")
_log_safe = store_common.log_safe


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
            # Bad entry — skip silently. The /settings endpoint validates inputs;
            # this is a runtime defense for handwritten config edits.
            continue
    return nets


def require_allowed_host(allowlist_ref: Callable[[], list[str]]) -> Callable[[Request], None]:
    """Returns a FastAPI dependency that rejects callers outside the allowlist.

    `allowlist_ref` is a zero-arg callable that returns the current allowlist —
    NOT the list itself. This indirection matters: the admin WebUI can edit
    cfg.ADMIN_WEBUI_ALLOWED_HOSTS at runtime, and we want the next request to pick
    up the new value without re-creating the dependency.

    Loopback (`127.0.0.1`, `::1`) is ALWAYS allowed in addition to the
    configured list, so a misconfigured CIDR can never lock the local
    operator out of /settings — they can still fix the entry from the box.
    """

    def _dep(request: Request) -> None:
        client = request.client
        if client is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "no client info")
        if _to_ip(client.host) is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "unparseable client host")
        if not host_in_allowlist(request, allowlist_ref()):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "host not in allowlist")

    return _dep


def host_in_allowlist(request: HTTPConnection, allowlist: list[str]) -> bool:
    """True if the client IP is loopback or inside `allowlist`. Non-raising —
    the boolean core behind `require_allowed_host` (which turns False into a
    403). Loopback is always allowed, matching require_allowed_host's contract.

    Typed on HTTPConnection, not Request: auth.open_mode_host_ok also passes
    the streaming WebSocket, and `.client` is all this reads.
    """
    client = request.client
    if client is None:
        return False
    ip = _to_ip(client.host)
    if ip is None:
        return False
    if ip.is_loopback:
        return True
    return any(ip in net for net in _build_networks(allowlist))


# Concrete tier gates, bucketed by privilege (loopback always allowed). The
# lambdas re-read cfg per request so the admin WebUI can broaden/narrow access
# without a restart. Both are the OUTER host layer; the INNER key layer
# (require_page(...) / require_admin) is stacked on the data endpoints.
#   require_admin_webui_host — /settings, /settings/api-keys, /docs.
#   require_user_webui_host  — /quick-config, /captures, /reports, /stats,
#                              /logs, /dictate, /sev (default-open allowlist).
require_admin_webui_host = require_allowed_host(lambda: cfg.ADMIN_WEBUI_ALLOWED_HOSTS)
require_user_webui_host = require_allowed_host(lambda: cfg.USER_WEBUI_ALLOWED_HOSTS)


# The origin lists the same-origin guard accepts besides the request's own
# Host: CORS_ALLOW_ORIGINS (compatibility) and TRUSTED_ORIGINS. main computes
# them once while it builds the app (the CORS middleware needs the first one)
# and hands them over through configure_origins(), so a reload of main — the
# test suite's app_module fixture — re-arms the guard with the reloaded config.
_cors_origins: list[str] = []
_trusted_origins: list[str] = []


def configure_origins(cors_origins: list[str], trusted_origins: list[str]) -> None:
    """Install the origin allowlists _origin_is_allowed compares against."""
    global _cors_origins, _trusted_origins
    _cors_origins = cors_origins
    _trusted_origins = trusted_origins


# A rejected Origin is logged at most this often: the check runs before any
# credential, so an unauthenticated caller could otherwise drive the log at
# request rate (same reasoning as auth/dependencies.py's open-mode nag interval).
_ORIGIN_REJECT_LOG_INTERVAL_S = 60.0
# -inf, not 0.0: time.monotonic() counts from host boot, so 0.0 swallowed the
# first rejection of a server started within a minute of boot.
_origin_reject_logged_at = float("-inf")


def _origin_is_allowed(request) -> bool:
    """True when an unsafe-method request may proceed based on its Origin.

    Takes any Starlette HTTPConnection — only `.headers` is read — so the
    WebSocket handshake can reuse it. `@app.middleware("http")` wraps in
    BaseHTTPMiddleware, which passes non-http scopes straight through, so a
    websocket scope never reaches main's _csrf_mw and has to call this itself.

    Browsers attach Origin to every cross-site unsafe-method request (a plain
    auto-submitting <form> included), so this catches the case the token check
    cannot: OPEN mode, where a request carries no cookie and no bearer yet
    still resolves to the synthetic admin. Non-browser clients (curl, SDKs)
    send no Origin at all — absent means allow.

    Only host:port is compared against Host: a TLS-terminating proxy rewrites
    neither header, but the scheme in front of it is unknowable from here. A
    proxy that DOES rewrite Host to the upstream never matches, which is what
    TRUSTED_ORIGINS is for (CORS_ALLOW_ORIGINS also counts, for compatibility
    — but it additionally switches CORS on, so it is the wrong knob here).

    CORS_ALLOW_ORIGINS="*" does NOT satisfy this check. The two answer
    different questions: CORS decides whether a cross-origin page may READ a
    response, this decides whether it may PERFORM a state change — and the
    wildcard used to short-circuit here, silently disabling the guard for every
    unsafe method app-wide (and for the WebSocket handshake, which calls this
    directly). A deployment that genuinely needs cross-origin writes lists its
    real origins in TRUSTED_ORIGINS, which exists for exactly that.
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    if origin in _cors_origins or origin in _trusted_origins:
        return True
    host = request.headers.get("host", "")
    from urllib.parse import urlsplit
    return bool(host) and urlsplit(origin).netloc == host


def _log_origin_rejected(request) -> None:
    """Throttled WARNING naming the two headers that decided it — the only way
    an operator can tell a genuine cross-site POST from a reverse proxy that
    rewrote Host. Both values are attacker-controlled, hence _log_safe."""
    global _origin_reject_logged_at
    now = time.monotonic()
    if now - _origin_reject_logged_at < _ORIGIN_REJECT_LOG_INTERVAL_S:
        return
    _origin_reject_logged_at = now
    logger.warning(
        "Rejected %s %s: Origin %r does not match Host %r. If this is your own "
        "reverse proxy rewriting Host, add the public origin to "
        "TRUSTED_ORIGINS (WHISPER_TRUSTED_ORIGINS); otherwise it was a "
        "cross-site request. Further rejections are logged at most every %.0f s.",
        getattr(request, "method", "WEBSOCKET"), _log_safe(request.url.path),
        _log_safe(request.headers.get("origin")),
        _log_safe(request.headers.get("host")),
        _ORIGIN_REJECT_LOG_INTERVAL_S,
    )


def _reset_for_tests() -> None:
    """Re-open the origin-rejection log throttle (tests/conftest.py
    _RESET_HOOKS). The origin lists are configuration main installs, not
    per-test state."""
    global _origin_reject_logged_at
    _origin_reject_logged_at = float("-inf")
