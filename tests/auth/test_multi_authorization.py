"""The API key behind a basic-auth reverse proxy: several Authorization values.

A client whose Server URL is `https://user:pass@host` (nginx/Caddy basic auth
in front of the backend) sends TWO Authorization headers — `Basic …` from the
URL first, `Bearer <api_key>` second. Every auth path reads the key through
`auth.bearer_credentials`, which scans all values (and a proxy-merged
"Basic x, Bearer y" value) for the first Bearer credential; a single header
behaves exactly as before."""

from starlette.requests import Request

from tests.conftest import bearer
from faster_whisper_backend.auth import dependencies as auth


_BASIC = "Basic dXNlcjpwYXNz"   # user:pass


def _request(*values):
    """A bare GET Request carrying each value as its own Authorization header."""
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/stats/stream",
        "headers": [(b"authorization", v.encode()) for v in values],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    })


def _key(*values):
    creds = auth.bearer_credentials(_request(*values))
    return None if creds is None else creds.credentials


# --- the parser ---------------------------------------------------------------

def test_single_bearer_is_unchanged():
    assert _key("Bearer wk_abc") == "wk_abc"
    assert _key("bearer   wk_abc  ") == "wk_abc"


def test_basic_then_bearer_finds_the_key():
    assert _key(_BASIC, "Bearer wk_abc") == "wk_abc"


def test_first_bearer_wins():
    assert _key(_BASIC, "Bearer wk_one", "Bearer wk_two") == "wk_one"


def test_merged_value_finds_the_key_in_either_order():
    assert _key(f"{_BASIC}, Bearer wk_abc") == "wk_abc"
    assert _key(f"Bearer wk_abc, {_BASIC}") == "wk_abc"


def test_only_basic_is_no_key():
    assert _key(_BASIC) is None
    assert _key() is None
    assert _key("Bearer ") is None
    assert _key("Bearer") is None


def test_auth_param_commas_do_not_split():
    assert _key('Digest realm="a", nonce="b"') is None


# --- HTTP route (the Depends path) -------------------------------------------

def test_http_route_basic_then_bearer_admits(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.get("/settings/state", headers=[
        ("Authorization", _BASIC), ("Authorization", f"Bearer {raw}"),
    ])
    assert r.status_code == 200


def test_http_route_single_bearer_admits(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    assert client.get("/settings/state", headers=bearer(raw)).status_code == 200


def test_http_route_only_basic_is_401(client, make_user_key):
    make_user_key("root", is_admin=True)
    r = client.get("/settings/state", headers={"Authorization": _BASIC})
    assert r.status_code == 401


# --- SSE path (resolve_user_for_page_sse) ------------------------------------

def test_sse_basic_then_bearer_admits(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"stats": "all"})
    rec = auth.resolve_user_for_page_sse(
        _request(_BASIC, f"Bearer {raw}"), "stats")
    assert rec["username"] == "alice"


# --- CSRF middleware: a dead cookie riding along with the second header ------

def test_csrf_stale_cookie_with_basic_then_bearer_passes(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    client.cookies.set("whisper_session", "stale-token-with-no-row")
    r = client.post("/auth/logout", headers=[
        ("Authorization", _BASIC), ("Authorization", f"Bearer {raw}"),
    ])
    assert r.status_code == 200


def test_csrf_stale_cookie_with_only_basic_is_403(client, make_user_key):
    make_user_key("root", is_admin=True)
    client.cookies.set("whisper_session", "stale-token-with-no-row")
    r = client.post("/auth/logout", headers={"Authorization": _BASIC})
    assert r.status_code == 403
