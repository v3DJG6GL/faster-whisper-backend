"""Auth-model integration tests: open mode vs locked-down, admin vs
non-admin keys, host gate, cross-user 404, and the SSE ?key= fallback."""

from starlette.testclient import TestClient

from conftest import bearer


def test_open_mode_admin_everywhere(client):
    # No admin key => open mode => synthetic admin can reach admin routes.
    assert client.get("/auth/whoami").json()["is_admin"] is True
    assert client.get("/settings/state").status_code == 200


def test_locked_down_no_bearer_is_401(client, make_user_key):
    # Creating the first admin key flips the server to locked-down.
    make_user_key("root", is_admin=True)
    r = client.get("/settings/state")
    assert r.status_code == 401


def test_locked_down_admin_bearer_ok(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.get("/settings/state", headers=bearer(raw))
    assert r.status_code == 200


def test_locked_down_bad_bearer_is_401(client, make_user_key):
    make_user_key("root", is_admin=True)
    r = client.get("/settings/state", headers=bearer("wk_not_a_real_key"))
    assert r.status_code == 401


def test_non_admin_403_on_admin_route(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", is_admin=False)
    # /settings/state requires admin -> 403 for a non-admin key.
    r = client.get("/settings/state", headers=bearer(raw))
    assert r.status_code == 403


def test_non_admin_200_on_permitted_page(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key(
        "alice", is_admin=False, pages={"quick_config": "own"}
    )
    # quick_config page granted -> the /state JSON API is reachable.
    r = client.get("/quick-config/state", headers=bearer(raw))
    assert r.status_code == 200


def test_non_admin_403_on_unpermitted_page(client, make_user_key):
    # New non-admins default to quick_config/reports/captures="own"; explicitly
    # revoke quick_config to "none" to exercise the require_page 403 path.
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key(
        "alice", is_admin=False, pages={"quick_config": "none"}
    )
    r = client.get("/quick-config/state", headers=bearer(raw))
    assert r.status_code == 403


def test_host_gate_rejects_non_loopback(app_module):
    # Build a SEPARATE client from a non-loopback source IP. The host gate
    # 403s before auth on host-gated routes.
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        r = c.get("/settings/state")
        assert r.status_code == 403


def test_host_gate_allows_loopback(client):
    # The default loopback client is admitted by the host gate.
    assert client.get("/settings/state").status_code == 200


def test_cross_user_report_read_is_404(client, make_user_key):
    # Two non-admin users with reports scope=own; one cannot read the other's
    # report by id -> 404 (anti-IDOR), not 403.
    import reports_store

    make_user_key("root", is_admin=True)
    uid_a, _raw_a = make_user_key("alice", pages={"reports": "own"})
    uid_b, raw_b = make_user_key("bob", pages={"reports": "own"})

    rid_a, _ = reports_store.upsert_report(
        user_id=uid_a, request_id="req-a", trace_ts=0.0, model="whisper-1",
        raw="r", final="f", steps=[], corrections=[],
        intended_text="x", user_comment="c", reporter_role="user",
        reporter_host="127.0.0.1",
    )
    # Bob (scope=own) tries to PATCH Alice's report -> 404.
    r = client.patch(
        f"/reports/api/{rid_a}",
        headers=bearer(raw_b),
        json={"status": "resolved"},
    )
    assert r.status_code == 404


# The /quick-config/stream SSE body is an infinite generator (keepalive loop),
# so driving it over HTTP with TestClient hangs on close. The auth/?key=
# fallback logic lives entirely in require_user_or_admin_sse and runs BEFORE
# any streaming — so we exercise that dependency directly with a constructed
# ASGI scope. This is the documented "assert without consuming the stream"
# approach.

def _fake_request(headers=None, query=b""):
    from starlette.requests import Request

    raw_headers = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/quick-config/stream",
        "headers": raw_headers,
        "query_string": query,
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def test_sse_key_query_fallback_admits(client, make_user_key):
    # SSE endpoints accept ?key=<raw> since EventSource cannot set headers.
    import quick_config_routes

    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    req = _fake_request(query=f"key={raw}".encode())
    rec = quick_config_routes.require_user_or_admin_sse(req)
    assert rec["username"] == "alice"


def test_sse_missing_key_rejected_when_locked(client, make_user_key):
    import quick_config_routes
    from fastapi import HTTPException

    make_user_key("root", is_admin=True)
    req = _fake_request()  # no bearer, no ?key=
    try:
        quick_config_routes.require_user_or_admin_sse(req)
    except HTTPException as e:
        assert e.status_code == 401
    else:
        raise AssertionError("expected 401 HTTPException")


# --- Tiered host gates -------------------------------------------------------
# Admin tier (ADMIN_WEBUI_ALLOWED_HOSTS, loopback default): /docs, /redoc are
# host-only shells; /openapi.json adds an admin key; /settings* same shape.
# User tier (USER_WEBUI_ALLOWED_HOSTS, OPEN default): /stats, /logs, … shells
# are host-only on the open list; their data endpoints add a per-page key; /sev
# adds "any authenticated user". Loopback always passes either list.

_REMOTE = ("203.0.113.9", 1234)  # TEST-NET-3, not in the admin allowlist
_DOCS_SHELLS = ("/docs", "/redoc")  # host-only admin shells


def test_docs_shells_loopback_ok(client, make_user_key):
    # Admin-host shells: loopback always admitted (host-only, no key needed).
    make_user_key("root", is_admin=True)  # locked down
    for path in _DOCS_SHELLS:
        assert client.get(path).status_code == 200, path


def test_docs_shells_remote_403_even_with_admin_key(client, make_user_key):
    # Host-only admin gate: a remote host outside ADMIN_WEBUI_ALLOWED_HOSTS is
    # 403 regardless of key — the key cannot open the host gate.
    _uid, raw = make_user_key("root", is_admin=True)
    with TestClient(client.app, client=_REMOTE) as c:
        for path in _DOCS_SHELLS:
            assert c.get(path).status_code == 403, path
            assert c.get(path, headers=bearer(raw)).status_code == 403, path


def test_docs_shells_remote_403_open_mode(app_module):
    # Even in OPEN mode the pure host gate rejects a non-admin-host remote
    # (no synthetic-admin bypass on a host-only check).
    with TestClient(app_module.app, client=_REMOTE) as c:
        for path in _DOCS_SHELLS:
            assert c.get(path).status_code == 403, path


def test_openapi_loopback_open_mode_ok(client):
    # /openapi.json = admin host AND admin key; open mode → synthetic admin.
    assert client.get("/openapi.json").status_code == 200


def test_openapi_loopback_locked_needs_admin(client, make_user_key):
    _ruid, araw = make_user_key("root", is_admin=True)  # lock down + admin key
    assert client.get("/openapi.json").status_code == 401  # no credential
    _uid, nraw = make_user_key("alice", is_admin=False)
    assert client.get("/openapi.json", headers=bearer(nraw)).status_code == 403
    assert client.get("/openapi.json", headers=bearer(araw)).status_code == 200


def test_openapi_remote_403(client, make_user_key):
    # Host-only admin gate fires before the key — remote is 403 even for admin.
    _uid, araw = make_user_key("root", is_admin=True)
    with TestClient(client.app, client=_REMOTE) as c:
        assert c.get("/openapi.json").status_code == 403
        assert c.get("/openapi.json", headers=bearer(araw)).status_code == 403


def test_sev_open_mode_remote_ok(app_module):
    # User host is open by default; open mode → synthetic admin satisfies auth.
    with TestClient(app_module.app, client=_REMOTE) as c:
        assert c.get("/sev").status_code == 200


def test_sev_locked_remote_no_key_401(client, make_user_key):
    make_user_key("root", is_admin=True)
    with TestClient(client.app, client=_REMOTE) as c:
        assert c.get("/sev").status_code == 401


def test_sev_locked_remote_any_key_ok(client, make_user_key):
    # /sev needs any authenticated user (no specific page perm) — a non-admin
    # key is enough; the user host list is open so the host gate passes.
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", is_admin=False)
    with TestClient(client.app, client=_REMOTE) as c:
        assert c.get("/sev", headers=bearer(raw)).status_code == 200


def test_logs_shell_remote_open_host_ok(client, make_user_key):
    # The /logs SHELL is host-only on the OPEN user allowlist → reachable from
    # any host even locked-down with no key (the login popup runs in-page).
    make_user_key("root", is_admin=True)
    with TestClient(client.app, client=_REMOTE) as c:
        assert c.get("/logs").status_code == 200


def test_logs_data_locked_no_key_401(client, make_user_key):
    # The data endpoint stacks require_page("logs") → no key is 401.
    make_user_key("root", is_admin=True)
    with TestClient(client.app, client=_REMOTE) as c:
        assert c.get("/logs/older").status_code == 401


def test_user_host_narrowing_blocks_shell(client, make_user_key, monkeypatch):
    # Narrowing USER_WEBUI_ALLOWED_HOSTS to loopback blocks a remote host from
    # even the user-page shell (the host gate fires before the in-page login).
    import config as cfg
    make_user_key("root", is_admin=True)
    monkeypatch.setattr(
        cfg, "USER_WEBUI_ALLOWED_HOSTS", ["127.0.0.1", "::1"], raising=False
    )
    with TestClient(client.app, client=_REMOTE) as c:
        assert c.get("/stats").status_code == 403
        assert c.get("/logs").status_code == 403
