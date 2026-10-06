"""The unified login gate lives in the shared chrome (web_common) and ships on
every WebUI page; the old per-page #token-modal / #login-wrap login UIs are gone.

In OPEN mode (the default `client` fixture, loopback) every page renders, so we
can assert on the served HTML directly."""

import pytest

# (path, current-page label) for every WebUI page that renders shared chrome.
_PAGES = [
    "/stats",
    "/logs",
    "/quick-config",
    "/captures",
    "/reports",
    "/settings",
    "/settings/api-keys",
]


@pytest.mark.parametrize("path", _PAGES)
def test_page_ships_shared_login_gate(client, path):
    r = client.get(path)
    assert r.status_code == 200, (path, r.status_code)
    html = r.text
    # The shared gate markup + API ride the OPEN_MODE_BANNER_JS chrome.
    assert 'id="login-gate"' in html, f"{path} missing the shared login gate"
    assert "_showLoginGate" in html, f"{path} missing the gate API"
    # The markup, not the bare class: NAV_CSS's `#login-gate .lg-mark {`
    # selector ships on every page and would satisfy a class-only check.
    assert '<svg class="lg-mark"' in html, f"{path} missing the waveform brand mark"


@pytest.mark.parametrize("path", _PAGES)
def test_page_has_no_legacy_login_ui(client, path):
    html = client.get(path).text
    # The per-page login UIs were removed in favour of the shared gate.
    assert 'id="token-modal"' not in html, f"{path} still has a #token-modal"
    assert 'id="login-wrap"' not in html, f"{path} still has the #login-wrap card"
    assert 'id="login-token"' not in html, f"{path} still has the login-card input"


def test_gate_reports_rejected_only_for_a_401():
    """A 429 (failed sign-in window exhausted) refuses even a correct key and
    a 5xx is the server failing to open a session; the gate used to call
    every non-2xx "That key was rejected."."""
    import re

    from faster_whisper_backend.core import web_common

    js = web_common.OPEN_MODE_BANNER_JS
    assert re.search(r"r\.status === 401\) throw 'That key was rejected\.'", js)
    assert "j.detail" in js                       # the RateLimited reason
    assert "could not create a session" in js
    assert "throw 0" not in js


def test_correct_key_in_a_tripped_window_gets_429_with_detail(
        client, app_module, make_user_key):
    """Server half the gate now surfaces: the guard runs before the lookup,
    so the RIGHT key is refused too, with a `detail` naming the retry."""
    _uid, raw = make_user_key("root", is_admin=True)
    for _ in range(int(app_module.cfg.LOGIN_FAILURE_RATE)):
        assert client.post("/auth/login",
                           json={"key": "wk_nope"}).status_code == 401
    r = client.post("/auth/login", json={"key": raw})
    assert r.status_code == 429
    assert "retry in" in r.json()["detail"]
