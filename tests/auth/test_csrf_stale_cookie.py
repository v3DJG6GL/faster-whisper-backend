"""_csrf_mw and a session cookie that no longer resolves.

The cookie's max_age is SESSION_TTL_S (30 d) while the server-side row
can vanish sooner (sessions DB wiped/moved, row expired). auth._resolve_user
gives the bearer header priority over the cookie, so a request authenticating
purely by `Authorization: Bearer` must not be 403'd just because a dead cookie
rides along — a bearer request cannot be CSRF'd. The skip is deliberately
NARROW: a dead cookie with no bearer stays fail-closed, because in open mode
the request would otherwise resolve to the synthetic admin behind nothing but
the Origin check."""

from tests.conftest import bearer


def test_bearer_with_stale_cookie_passes_the_csrf_guard(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    # A cookie value that resolves to no live session.
    client.cookies.set("whisper_session", "stale-token-with-no-row")
    r = client.post("/auth/logout", headers=bearer(raw))
    assert r.status_code == 200


def test_stale_cookie_without_bearer_is_still_403(client, make_user_key):
    make_user_key("root", is_admin=True)
    client.cookies.set("whisper_session", "stale-token-with-no-row")
    r = client.post("/v1/audio/transcriptions")
    assert r.status_code == 403
    assert r.json() == {"detail": "CSRF token missing or invalid"}


def test_stale_cookie_can_still_sign_out(client, make_user_key):
    """/auth/logout with a dead cookie and no bearer used to get the same
    403, so the browser could never clear the cookie: every Sign-out click
    alerted "Sign-out failed" over a session that was already gone."""
    from faster_whisper_backend.settings import config as cfg
    make_user_key("root", is_admin=True)
    client.cookies.set("whisper_session", "stale-token-with-no-row")
    r = client.post("/auth/logout", headers={"X-CSRF-Token": "whatever"})
    assert r.status_code == 200
    lines = r.headers.get_list("set-cookie")
    for name in (cfg.SESSION_COOKIE_NAME, cfg.SESSION_CSRF_COOKIE_NAME):
        assert any(l.startswith(name + "=") for l in lines), (name, lines)


def test_live_cookie_with_bearer_still_requires_the_token(client, make_user_key):
    # Pre-existing contract pinned: a LIVE cookie plus a bearer header still
    # runs the token check (the skip applies only when the session is gone).
    _uid, raw = make_user_key("root", is_admin=True)
    client.post("/auth/login", json={"key": raw})
    r = client.post("/auth/logout", headers=bearer(raw))
    assert r.status_code == 403
