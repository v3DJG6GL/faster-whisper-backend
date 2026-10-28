"""hosts._origin_is_allowed — the CSRF Origin-vs-Host check that runs on every
unsafe-method request and every WebSocket handshake, before any credential."""

import types

from faster_whisper_backend.auth import hosts


def test_malformed_origin_is_rejected_not_raised():
    # urlsplit raises on an unbalanced IPv6 bracket; that escaped as a 500
    # with a traceback per request instead of the throttled 403.
    req = types.SimpleNamespace(
        headers={"origin": "http://[x", "host": "localhost"})
    assert hosts._origin_is_allowed(req) is False
