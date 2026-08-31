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


# --- Origin: checked on every unsafe method, whatever the credential --------

def test_cross_site_origin_blocked_in_open_mode(client):
    # Open mode resolves every request to the synthetic admin, so a cookie-less
    # cross-site POST would otherwise run the handler. The Origin header the
    # browser attaches is the only signal available here.
    r = client.post("/auth/logout", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_cross_site_origin_blocked_for_bearer_client(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.post("/auth/logout",
                    headers={**bearer(raw), "Origin": "http://evil.example"})
    assert r.status_code == 403


def test_cross_site_origin_blocked_on_login(client, make_user_key):
    # /auth/login is exempt from the TOKEN check (no session exists yet) but
    # NOT from the origin check: it hands out a session cookie.
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.post("/auth/login", json={"key": raw},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_same_origin_mutation_allowed(client, make_user_key):
    # TestClient's base_url is http://testserver, so Origin matches Host.
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.post("/auth/logout",
                    headers={**bearer(raw), "Origin": "http://testserver"})
    assert r.status_code == 200


def test_bearer_still_works_when_cookie_login_available(client, make_user_key):
    # Regression guard: adding cookie auth must not break header-bearer auth
    # on the transcription/admin surface (Vowen / curl path).
    _uid, raw = make_user_key("root", is_admin=True)
    assert client.get("/settings/state", headers=bearer(raw)).status_code == 200


# --- multi-worker (SERVER_WORKERS > 1) index coherence ----------------------

def _worker(name, db_path):
    """Load a second, independent copy of sessions_store bound to the same DB
    file — faithful to two uvicorn workers, each with its own connection and
    its own in-memory _SESSION_INDEX."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "sessions_store.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init_db(db_path)
    return mod


def test_revoke_in_sibling_worker_is_seen_after_local_write(tmp_path):
    """A revocation committed by worker B must not survive worker A's own
    next write. create_session/revoke_session used to re-stamp _DATA_VERSION
    AFTER their own commit; since a connection's own commit does not move its
    own PRAGMA data_version, that re-stamp absorbed B's commit and marked it
    seen — leaving the revoked cookie authenticating in A forever (the sliding
    TTL renews it on every lookup)."""
    db = str(tmp_path / "sessions.db")
    a = _worker("sessions_store_a", db)
    b = _worker("sessions_store_b", db)

    raw, _csrf = a.create_session("victim", 3600.0)
    assert a.lookup_session(raw) is not None
    assert b.lookup_session(raw) is not None      # B picks it up from disk

    b.revoke_session(raw)                          # sibling worker logs out
    assert b.lookup_session(raw) is None

    # A now does an unrelated local write. The old code re-stamped the version
    # here and never noticed B's revocation.
    a.create_session("someone-else", 3600.0)
    assert a.lookup_session(raw) is None

    # And the same holds when A's own write is a revoke.
    raw2, _ = a.create_session("victim2", 3600.0)
    raw3, _ = b.create_session("victim3", 3600.0)
    a.revoke_session(raw2)
    assert a.lookup_session(raw3) is not None      # B's new session is visible


def test_slide_expiry_touches_slide_cache_only_under_the_lock(tmp_path):
    """_slide_expiry_debounced read and wrote _SLIDE_CACHE OUTSIDE _lock while
    _rebuild_index_locked() iterates that same dict under the lock — a
    threadpool auth lookup racing the hourly purge_expired() task raised
    'dictionary changed size during iteration'. Assert the invariant directly
    (a timing race makes a flaky test): every access happens with _lock held."""
    db = str(tmp_path / "sessions.db")
    w = _worker("sessions_store_slide", db)
    unlocked = []

    class Probe(dict):
        def get(self, key, *a):
            if not w._lock.locked():
                unlocked.append(("get", key))
            return super().get(key, *a)

        def __setitem__(self, key, value):
            if not w._lock.locked():
                unlocked.append(("set", key))
            super().__setitem__(key, value)

    w._SLIDE_CACHE = Probe()
    raw, _csrf = w.create_session("u", 3600.0)
    assert w.lookup_session(raw) is not None      # drives the slide path
    assert unlocked == []


# --- failed-login throttle --------------------------------------------------

def test_login_failures_are_throttled(client, app_module, make_user_key):
    """Keyed by client host (a login has no identity yet). The attempt past
    LOGIN_FAILURE_RATE is refused before the key is even looked up."""
    make_user_key("root", is_admin=True)
    limit = int(app_module.cfg.LOGIN_FAILURE_RATE)
    for _ in range(limit):
        assert client.post("/auth/login",
                           json={"key": "wk_nope"}).status_code == 401
    r = client.post("/auth/login", json={"key": "wk_nope"})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    body = r.json()
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert body["error"]["param"] == "LOGIN_FAILURE_RATE"
    assert body["detail"] == body["error"]["message"]


def test_login_success_resets_the_window(client, app_module, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    limit = int(app_module.cfg.LOGIN_FAILURE_RATE)
    for _ in range(limit - 1):
        assert client.post("/auth/login",
                           json={"key": "wk_nope"}).status_code == 401
    assert client.post("/auth/login", json={"key": raw}).status_code == 200
    # The window is cleared, so a fresh run of failures is admitted again.
    for _ in range(limit):
        assert client.post("/auth/login",
                           json={"key": "wk_nope"}).status_code == 401


def test_login_open_mode_is_never_throttled(client, app_module, monkeypatch):
    # No admin key => open mode => no credential is checked, so there is
    # nothing to throttle and nobody to lock out.
    monkeypatch.setattr(app_module.cfg, "LOGIN_FAILURE_RATE", 1,
                        raising=False)
    for _ in range(5):
        r = client.post("/auth/login", json={"key": "anything"})
        assert r.status_code == 200 and r.json() == {"open_mode": True}


def test_login_failure_rate_zero_is_unlimited(client, app_module,
                                              make_user_key, monkeypatch):
    make_user_key("root", is_admin=True)
    monkeypatch.setattr(app_module.cfg, "LOGIN_FAILURE_RATE", 0,
                        raising=False)
    for _ in range(30):
        assert client.post("/auth/login",
                           json={"key": "wk_nope"}).status_code == 401
