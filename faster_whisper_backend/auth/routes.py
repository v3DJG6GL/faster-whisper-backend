"""Browser sign-in: POST /auth/login (API key → HttpOnly session cookie +
CSRF cookie, with the per-host failure lockout), POST /auth/logout and
GET /auth/whoami. main includes this router WITHOUT the fail-soft helper, so
an import error fails startup instead of silently removing login.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse as _JSONResponse

from faster_whisper_backend.auth import rate_limit as _rl
from faster_whisper_backend.auth.dependencies import Permissions, get_current_user as _get_current_user_dep
from faster_whisper_backend.core import store_common
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend import build_info
from faster_whisper_backend.auth import api_keys_store as _ak
from faster_whisper_backend.auth import dependencies as _auth
from faster_whisper_backend.auth import sessions_store

logger = logging.getLogger("whisper-api")
_log_safe = store_common.log_safe

router = APIRouter()


@router.get("/auth/whoami")
async def whoami(
    request: Request,
    user: dict = Depends(_get_current_user_dep),
):
    """Resolve the caller to a user payload the WebUI uses to render the
    login modal + user-aware chrome.

    Returns `{open_mode, user_id, username, is_admin, permissions, build,
    csrf_token?}`. `build` carries the header chip's facts (version, boot,
    start) — the shared header ships as an empty shell because its pages are
    only host-gated, so the facts ride this authenticated route instead. The
    `permissions` object is `{pages: {logs:
    'own'|'all'|'none', ...}}` — used by each page's JS to hide nav links
    the user can't reach and to render scope hints. `csrf_token` is
    present only for cookie-authenticated callers (set by
    user_from_session_cookie on request.state) so the client can attach
    X-CSRF-Token without parsing the cookie. A 401 means no valid
    credential AND the server is locked down — the WebUI re-prompts."""
    perms = user.get("permissions")
    out = {
        # Only the synthetic admin is "open mode": a host outside the admin
        # allowlist signs in with a real key while no admin key exists, and
        # needs the sign-out button and its own name like any session.
        "open_mode": user.get("user_id") == _ak.OPEN_MODE_USER["user_id"],
        "user_id": user.get("user_id"),
        "username": user.get("username"),
        "is_admin": bool(user.get("is_admin")),
        "permissions": perms.to_dict() if perms is not None else {"pages": {}},
        "build": {
            "server": build_info.SERVER_NAME,
            "version": build_info.APP_VERSION,
            "version_short": build_info.VERSION_SHORT,
            "boot": build_info.BOOT_ID[:8],
            "started": build_info.STARTED_UTC,
        },
    }
    csrf = getattr(request.state, "session_csrf", None)
    if csrf:
        out["csrf_token"] = csrf
    # Per-identity, and it carries the session's CSRF token. A 200 GET with no
    # Cache-Control and no Vary is heuristically cacheable under RFC 9111, and
    # this deployment expects a reverse proxy in front (TRUSTED_ORIGINS exists
    # for exactly that) — a shared cache could otherwise hand one user's
    # identity payload, admin flag and CSRF token to the next caller. The page
    # JS already sends cache:'no-store', but that binds only the browser's own
    # cache, not an intermediary's.
    return _JSONResponse(out, headers={"Cache-Control": "no-store"})


# Keyed by client HOST, not identity: a login attempt has no identity yet, and
# the key it presents is exactly what must not be trusted. Only FAILURES are
# counted and a success clears the window, so an operator fat-fingering one
# paste never walks toward a lockout.
#
# Operational caveat: the key is the SOCKET PEER (request.client.host), which
# is the real client only when uvicorn's proxy_headers trusts the proxy
# (default: forwarded_allow_ips=127.0.0.1, i.e. a same-host reverse proxy).
# A reverse proxy on another host that is not in FORWARDED_ALLOW_IPS puts
# every user into ONE bucket, where a single attacker's failures lock out
# all sign-ins for the window. Add the proxy to FORWARDED_ALLOW_IPS.
_login_failures = _rl.FixedWindow(
    config_field="LOGIN_FAILURE_RATE",
    window_s=60.0,
    default_max=10,
    message="too many failed sign-ins ({limit}/min) — "
            "retry in {retry_after}s",
)


@router.post("/auth/login")
async def login(request: Request, response: Response):
    """Exchange a pasted API key for an HttpOnly session cookie.

    Open mode → no-op, but only for callers on the admin host allowlist
    (auth.open_mode_host_ok), who are already the synthetic admin. Any other
    host in open mode — and everyone once locked down — validates the key
    via api_keys_store, creates a server-side session, and sets two
    cookies: the HttpOnly session token and a JS-readable CSRF token
    (double-submit). Open mode only means "no admin key exists yet"; ordinary
    user keys can, and a LAN browser whose /auth/whoami 401s needs this
    route to turn one into a cookie, or its login gate loops forever.
    Returns the identity part of /auth/whoami's shape (no `build` object)
    plus the CSRF token. CSRF-exempt (no session exists yet)."""
    if not _ak.is_locked_down() and _auth.open_mode_host_ok(request):
        return {"open_mode": True}
    # Below the open-mode short-circuit on purpose: open mode checks no
    # credential, so there is nothing to throttle, and locking an operator out
    # of an already-unlocked box would be absurd.
    host = request.client.host if request.client else ""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed/empty body → treat as no key
        body = {}
    key = body.get("key") if isinstance(body, dict) else None
    if not isinstance(key, str):  # non-string JSON value → same as no key
        key = ""
    # guard → lookup → penalize must have NO await between them: guard only
    # reads the counter, so with the body read in between, N concurrent
    # attempts all passed guard before any of them penalized, and one window
    # admitted ~N guesses instead of LOGIN_FAILURE_RATE. All three are sync,
    # so after the body read they run atomically on the event loop.
    _login_failures.guard(host)
    rec = _ak.lookup_by_raw_key(key)
    if rec is None:
        # NEVER log the attempted key — it is a credential, right or wrong.
        if _login_failures.penalize(host):
            logger.info(
                "[auth] login failure limit (LOGIN_FAILURE_RATE=%d) reached "
                "for host %s", _login_failures.limit(), _log_safe(host))
        raise HTTPException(
            401, "invalid API key", headers={"WWW-Authenticate": "Bearer"},
        )
    # Clear BEFORE create_session: a session-store failure below is the
    # server's problem, and must not leave the host carrying penalties for a
    # credential that was in fact correct.
    _login_failures.reset(host)
    raw_token, csrf_token = sessions_store.create_session(
        rec["user_id"], cfg.SESSION_TTL_S, key_id=rec.get("key_id"),
    )
    ttl = int(cfg.SESSION_TTL_S)
    secure = bool(cfg.SESSION_COOKIE_SECURE)
    response.set_cookie(
        cfg.SESSION_COOKIE_NAME, raw_token, max_age=ttl,
        httponly=True, samesite="lax", secure=secure, path="/",
    )
    response.set_cookie(
        cfg.SESSION_CSRF_COOKIE_NAME, csrf_token, max_age=ttl,
        httponly=False, samesite="lax", secure=secure, path="/",
    )
    perms = Permissions(rec.get("permissions_raw") or {}, bool(rec.get("is_admin")))
    return {
        "open_mode": False,
        "csrf_token": csrf_token,
        "user_id": rec.get("user_id"),
        "username": rec.get("username"),
        "is_admin": bool(rec.get("is_admin")),
        "permissions": perms.to_dict(),
    }


@router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Revoke the current session and clear its cookies. CSRF-protected
    like any other cookie-authenticated mutation (the WebUI sends the
    X-CSRF-Token header)."""
    raw = request.cookies.get(cfg.SESSION_COOKIE_NAME, "")
    if raw:
        sessions_store.revoke_session(raw)
    # Same attributes as login set them: a browser refuses the clearing
    # Set-Cookie of a __Host- / __Secure- name that lacks Secure.
    secure = bool(cfg.SESSION_COOKIE_SECURE)
    response.delete_cookie(cfg.SESSION_COOKIE_NAME, path="/", secure=secure,
                           httponly=True, samesite="lax")
    response.delete_cookie(cfg.SESSION_CSRF_COOKIE_NAME, path="/",
                           secure=secure, samesite="lax")
    return {"ok": True}
