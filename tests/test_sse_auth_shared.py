"""auth.resolve_user_for_page_sse — the shared core behind the SSE auth
variants (stats /stream, quick-config /stream), plus the open-mode host
confinement on the existing per-module gates.

The /stream bodies are infinite generators, so like tests/test_routes_auth.py
we exercise the dependencies directly with a constructed ASGI scope instead of
driving them over HTTP."""

from fastapi import HTTPException
from starlette.requests import Request

from conftest import bearer


def _fake_request(headers=None, client=("127.0.0.1", 12345)):
    raw_headers = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/stats/stream",
        "headers": raw_headers,
        "query_string": b"",
        "client": client,
    })


_REMOTE = ("203.0.113.9", 1234)  # TEST-NET-3, outside the admin allowlist


def _status_of(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except HTTPException as e:
        return e.status_code
    return 200


def test_bearer_header_admits_with_page_access(client, make_user_key):
    import auth
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"stats": "all"})
    rec = auth.resolve_user_for_page_sse(_fake_request(headers=bearer(raw)),
                                         "stats")
    assert rec["username"] == "alice"
    assert rec["permissions"].can("stats")


def test_no_credential_is_401(client, make_user_key):
    import auth
    make_user_key("root", is_admin=True)
    assert _status_of(auth.resolve_user_for_page_sse,
                      _fake_request(), "stats") == 401


def test_no_page_access_is_403_with_the_page_named(client, make_user_key):
    import auth
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"stats": "all"})
    try:
        # "logs" defaults to scope "none" for a fresh non-admin user.
        auth.resolve_user_for_page_sse(_fake_request(headers=bearer(raw)),
                                       "logs")
    except HTTPException as e:
        assert e.status_code == 403
        assert e.detail == "no access to /logs"
    else:
        raise AssertionError("expected 403 HTTPException")


def test_open_mode_loopback_gets_the_synthetic_admin(client):
    import auth
    rec = auth.resolve_user_for_page_sse(_fake_request(), "stats")
    assert rec["is_admin"] is True


def test_open_mode_off_allowlist_is_401(client):
    import auth
    assert _status_of(auth.resolve_user_for_page_sse,
                      _fake_request(client=_REMOTE), "stats") == 401


def test_existing_sse_gates_confine_open_mode_to_the_admin_hosts(client):
    # The per-module SSE gates must agree with the shared core: an
    # off-allowlist open-mode caller gets 401 from both stream endpoints.
    import quick_config_routes
    import stats_routes
    for dep in (stats_routes._require_stats_page_sse,
                quick_config_routes.require_user_or_admin_sse):
        assert _status_of(dep, _fake_request(client=_REMOTE)) == 401, dep
