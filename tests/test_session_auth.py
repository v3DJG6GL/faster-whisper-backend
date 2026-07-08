"""Cookie-session auth: /auth/login + /auth/logout, cookie-authenticated
access to protected routes, sliding TTL, user revocation, the Secure flag,
and the CSRF guard (enforced for cookie auth, exempt for bearer clients).

TestClient keeps an httpx cookie jar across requests on the same instance,
so a login() call leaves the session + CSRF cookies in place for the
follow-up requests — exactly like a browser."""

from starlette.testclient import TestClient

from conftest import bearer


def _set_cookie_lines(resp):
    """All Set-Cookie header values on a response (httpx collapses dup keys)."""
    return [v for k, v in resp.headers.multi_items() if k.lower() == "set-cookie"]


# --- login / logout basics --------------------------------------------------

def test_login_open_mode_is_noop(client):
    # No admin key => open mode => login is a no-op, no cookie issued.
    r = client.post("/auth/login", json={"key": "anything"})
    assert r.status_code == 200
    assert r.json() == {"open_mode": True}
    assert not _set_cookie_lines(r)


def test_login_good_key_sets_cookies(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.post("/auth/login", json={"key": raw})
    assert r.status_code == 200
    body = r.json()
    assert body["open_mode"] is False
    assert body["is_admin"] is True
    assert body["csrf_token"]
    lines = " ; ".join(_set_cookie_lines(r)).lower()
    assert "whisper_session=" in lines
    assert "whisper_csrf=" in lines
    assert "httponly" in lines  # the session cookie is HttpOnly


def test_login_bad_key_is_401_no_cookie(client, make_user_key):
    make_user_key("root", is_admin=True)
    r = client.post("/auth/login", json={"key": "wk_not_real"})
    assert r.status_code == 401
    assert not _set_cookie_lines(r)


# --- cookie-authenticated access to protected routes ------------------------

def test_cookie_auth_reaches_admin_route(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    # Locked down: no bearer, no cookie yet -> 401.
    assert client.get("/settings/state").status_code == 401
    # After login the session cookie alone admits the admin route.
    client.post("/auth/login", json={"key": raw})
    assert client.get("/settings/state").status_code == 200


def test_cookie_auth_respects_page_permissions(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    client.post("/auth/login", json={"key": raw})
    # Permitted page via cookie -> 200; admin-only route -> 403.
    assert client.get("/quick-config/state").status_code == 200
    assert client.get("/settings/state").status_code == 403


def test_logout_clears_session(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    tok = client.post("/auth/login", json={"key": raw}).json()["csrf_token"]
    assert client.get("/settings/state").status_code == 200
    r = client.post("/auth/logout", headers={"X-CSRF-Token": tok})
    assert r.status_code == 200
    # Session revoked: subsequent cookie-only request is rejected.
    assert client.get("/settings/state").status_code == 401


# --- sliding TTL / expiry ---------------------------------------------------

def test_expired_session_is_rejected(client, make_user_key):
    import sessions_store
    _uid, raw = make_user_key("root", is_admin=True)
    client.post("/auth/login", json={"key": raw})
    assert client.get("/settings/state").status_code == 200
    # Force every session into the past, then purge+rebuild the index.
    sessions_store._require_conn().execute(
        "UPDATE sessions SET expires_ts = 1"
    )
    sessions_store.purge_expired()
    assert client.get("/settings/state").status_code == 401


def test_active_session_slides_expiry(client, make_user_key):
    import time
    import sessions_store
    _uid, raw = make_user_key("root", is_admin=True)
    client.post("/auth/login", json={"key": raw})
    # Read the stored expiry, then force a slide by clearing the debounce.
    before = sessions_store._require_conn().execute(
        "SELECT expires_ts FROM sessions"
    ).fetchone()[0]
    sessions_store._SLIDE_CACHE.clear()
    time.sleep(0.01)
    assert client.get("/settings/state").status_code == 200
    after = sessions_store._require_conn().execute(
        "SELECT expires_ts FROM sessions"
    ).fetchone()[0]
    assert after > before


# --- user revocation kills live sessions ------------------------------------

def test_revoked_user_session_dies(client, make_user_key):
    import api_keys_store
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    client.post("/auth/login", json={"key": raw})
    assert client.get("/quick-config/state").status_code == 200
    # Revoking the user means get_user_record() returns None -> 401, even
    # though the session row itself hasn't expired.
    api_keys_store.revoke_user(uid)
    assert client.get("/quick-config/state").status_code == 401


def test_revoked_key_session_dies(client, make_user_key):
    """Revoking the LOGIN KEY must cut its sessions, not widen them: a revoked
    key_id resolves to the empty key binding (default-allow gates), so a
    surviving session would shed the key's per-key restrictions."""
    import api_keys_store
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    client.post("/auth/login", json={"key": raw})
    assert client.get("/quick-config/state").status_code == 200
    kid = api_keys_store.list_keys(uid)[0]["id"]
    api_keys_store.revoke_key(kid)
    # Bearer and cookie now agree: both are rejected immediately.
    assert client.get("/quick-config/state").status_code == 401
    assert client.get(
        "/quick-config/state", headers=bearer(raw),
    ).status_code == 401


def test_pre_migration_session_without_key_still_works(client, make_user_key):
    """A session created before login stamped key_id (key_id NULL in the DB)
    keeps authenticating with the old no-key-layer behaviour."""
    import config
    import sessions_store
    make_user_key("root", is_admin=True)
    uid, _raw = make_user_key("alice", pages={"quick_config": "own"})
    raw_token, _csrf = sessions_store.create_session(uid, 3600.0)  # no key_id
    client.cookies.set(config.SESSION_COOKIE_NAME, raw_token)
    assert client.get("/quick-config/state").status_code == 200


def test_session_use_touches_key_last_used(client, make_user_key):
    """Cookie-authenticated requests count as key activity: usage rollups
    already attribute to the stamped key, so last_used_ts must move too —
    otherwise the key looks dormant on /settings/api-keys while its usage
    numbers grow."""
    import api_keys_store
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    kid = api_keys_store.list_keys(uid)[0]["id"]
    client.post("/auth/login", json={"key": raw})
    # The login itself touches (bearer-style lookup) — reset AFTER it so the
    # touch under test can only come from the cookie-authenticated request.
    api_keys_store._LAST_USED_CACHE.clear()
    with api_keys_store._lock:
        api_keys_store._require_conn().execute(
            "UPDATE api_keys SET last_used_ts = NULL WHERE id = ?", (kid,),
        )
    assert api_keys_store.get_key(kid)["last_used_ts"] is None
    assert client.get("/quick-config/state").status_code == 200
    assert api_keys_store.get_key(kid)["last_used_ts"] is not None


# --- Secure flag ------------------------------------------------------------

def test_secure_flag_marks_cookies(client, make_user_key, monkeypatch):
    import config
    monkeypatch.setattr(config, "SESSION_COOKIE_SECURE", True)
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.post("/auth/login", json={"key": raw})
    lines = " ; ".join(_set_cookie_lines(r)).lower()
    assert "secure" in lines


def test_secure_flag_off_by_default(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.post("/auth/login", json={"key": raw})
    lines = " ; ".join(_set_cookie_lines(r)).lower()
    assert "secure" not in lines


# --- CSRF: enforced for cookie auth, exempt for bearer ----------------------

def test_csrf_missing_token_blocks_cookie_mutation(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    client.post("/auth/login", json={"key": raw})
    # Cookie present, no X-CSRF-Token header -> 403 from the CSRF middleware.
    r = client.post("/auth/logout")
    assert r.status_code == 403


def test_csrf_valid_token_allows_cookie_mutation(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    tok = client.post("/auth/login", json={"key": raw}).json()["csrf_token"]
    r = client.post("/auth/logout", headers={"X-CSRF-Token": tok})
    assert r.status_code == 200


def test_csrf_wrong_token_blocks_cookie_mutation(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    client.post("/auth/login", json={"key": raw})
    r = client.post("/auth/logout", headers={"X-CSRF-Token": "bogus"})
    assert r.status_code == 403


def test_csrf_exempt_for_bearer_clients(client, make_user_key):
    # A bearer/API client (no session cookie) is never subject to CSRF, so a
    # mutation without any X-CSRF-Token still passes the middleware. Uses a
    # fresh client that never logged in (no session cookie in the jar).
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.post("/auth/logout", headers=bearer(raw))
    assert r.status_code == 200


def test_csrf_covers_router_mounted_mutation(client, make_user_key):
    # The app-level CSRF middleware must also guard router-mounted routes
    # (e.g. /quick-config/*), not just app-level endpoints like /auth/logout.
    _uid, raw = make_user_key("root", is_admin=True)
    tok = client.post("/auth/login", json={"key": raw}).json()["csrf_token"]
    # Cookie present, no token -> blocked before the route runs.
    assert client.post("/quick-config/state", json={"rules_patch": {}}).status_code == 403
    # Valid token -> passes the middleware (route then handles it: 200).
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {}},
        headers={"X-CSRF-Token": tok},
    )
    assert r.status_code != 403


def test_bearer_still_works_when_cookie_login_available(client, make_user_key):
    # Regression guard: adding cookie auth must not break header-bearer auth
    # on the transcription/admin surface (Vowen / curl path).
    _uid, raw = make_user_key("root", is_admin=True)
    assert client.get("/settings/state", headers=bearer(raw)).status_code == 200
