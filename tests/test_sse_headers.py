"""Every Server-Sent-Events endpoint must ship the proxy-safe headers from
web_common.sse_response, or a buffering reverse proxy (nginx, proxy_buffering on
by default) accumulates the infinite text/event-stream body and severs the
HTTP/2 stream mid-flight — the browser reports NS_ERROR_NET_PARTIAL_TRANSFER and
the page "reconnects" every few seconds.

Asserting through a live TestClient would hang: the SSE generators never end and
starlette's streaming client blocks at teardown. Instead we unit-test the shared
helper directly (headers are set at construction, no iteration needed) and verify
each route is wired through it.
"""

import pathlib

import pytest


async def _agen():  # finite async generator — never actually iterated here
    yield "data: x\n\n"


def test_sse_response_sets_proxy_safe_headers():
    import web_common
    resp = web_common.sse_response(_agen())
    assert resp.media_type == "text/event-stream"
    assert resp.headers.get("x-accel-buffering") == "no"
    cache = resp.headers.get("cache-control", "")
    assert "no-cache" in cache and "no-transform" in cache, cache
    # `no-store` too: SSE bodies here carry transcript text, and an explicit
    # Cache-Control on the response suppresses the middleware's no-store
    # default (it uses setdefault). Bare `no-cache` still permits a shared
    # cache to STORE the body.
    assert "no-store" in cache, cache


# (source file, snippet proving the stream endpoint is wired to the helper)
_ROUTE_FILES = ["stats_routes.py", "quick_config_routes.py", "main.py"]
_REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("fname", _ROUTE_FILES)
def test_sse_routes_use_helper_not_bare_streamingresponse(fname):
    src = pathlib.Path(_REPO, fname).read_text(encoding="utf-8")
    # Routes the SSE stream through the proxy-safe helper...
    assert "web_common.sse_response(" in src, f"{fname} should use sse_response()"
    # ...and no longer hands back a bare event-stream StreamingResponse (which
    # would skip the X-Accel-Buffering header).
    assert 'media_type="text/event-stream"' not in src, (
        f"{fname} still constructs a bare event-stream response"
    )


def test_no_module_builds_a_bare_event_stream_response():
    """Repo-wide half of the invariant: a NEW SSE route in a module nobody
    listed in _ROUTE_FILES must still go through web_common.sse_response.
    Top level only (where the route modules live) — tests/ holds the literal
    itself. web_common.py is the one legitimate site."""
    offenders = [
        p.name for p in sorted(_REPO.glob("*.py"))
        if p.name != "web_common.py"
        and 'media_type="text/event-stream"' in p.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{offenders} construct a bare event-stream response; route SSE "
        "through web_common.sse_response() so the proxy-safe headers ship"
    )
