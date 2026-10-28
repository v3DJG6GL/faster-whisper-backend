"""window._signOut reloaded the page whatever /auth/logout answered: a 403
(the cached whoami CSRF token went stale after another tab signed in again),
a 5xx or a failed fetch reloaded into the same signed-in page with no word.
It now reloads only on a 2xx (or a 401: no session to end), retries a 403
once with the live whisper_csrf cookie's token, and says so when sign-out
failed.

Reuses the chrome DOM stub of test_whoami_unavailable_retry; the fetch stub
answers whoami with a cached token S1 while the cookie carries S2."""

import json
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common
from tests.core.test_whoami_unavailable_retry import _CHROME_STUB


_FETCH = r"""
document.cookie = 'other=x; ' + CSRF_COOKIE + '=S2';
const tokens = [];
let reloads = 0, alerts = 0;
global.location = { reload() { reloads++; } };
window.alert = () => { alerts++; };
window.dispatchEvent = () => {};
global.fetch = (url, opts) => {
  if (url === '/auth/whoami') return Promise.resolve({ ok: true, status: 200,
    json: () => Promise.resolve(Object.assign({ open_mode: false,
      csrf_token: 'S1' }, WHOAMI_EXTRA)) });
  if (url !== '/auth/logout') return Promise.resolve({ ok: false, status: 404 });
  tokens.push(opts.headers['X-CSRF-Token']);
  const s = LOGOUT[Math.min(tokens.length - 1, LOGOUT.length - 1)];
  if (s === 'reject') return Promise.reject(new TypeError('network'));
  return Promise.resolve({ ok: s >= 200 && s < 300, status: s });
};
"""

_DRIVE = r"""
window._showLoginGate = () => {};
window._hideLoginGate = () => {};
setTimeout(() => {
  for (let i = 0; i < CLICKS; i++) setTimeout(() => window._signOut(), i * GAP_MS);
  setTimeout(() => {
    console.log(JSON.stringify({ tokens, reloads, alerts }));
    process.exit(0);
  }, 50);
}, 50);
"""


def _sign_out(logout: list, clicks: int = 1, gap_ms: int = 0,
              csrf_cookie: str = "whisper_csrf",
              whoami_extra: dict | None = None) -> dict:
    js = web_common.OPEN_MODE_BANNER_JS.strip()
    js = js[len("<script>"):-len("</script>")]
    given = (f"const LOGOUT = {json.dumps(logout)};\n"
             f"const CLICKS = {clicks}, GAP_MS = {gap_ms};\n"
             f"const CSRF_COOKIE = {json.dumps(csrf_cookie)};\n"
             f"const WHOAMI_EXTRA = {json.dumps(whoami_extra or {})};\n")
    out = subprocess.run(
        ["node", "-"], input=given + _CHROME_STUB + _FETCH + js + _DRIVE,
        capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("logout, expected", [
    ([200], {"tokens": ["S1"], "reloads": 1, "alerts": 0}),
    ([401], {"tokens": ["S1"], "reloads": 1, "alerts": 0}),
    ([403, 200], {"tokens": ["S1", "S2"], "reloads": 1, "alerts": 0}),
    ([403, 403], {"tokens": ["S1", "S2"], "reloads": 0, "alerts": 1}),
    ([500], {"tokens": ["S1"], "reloads": 0, "alerts": 1}),
    (["reject"], {"tokens": ["S1"], "reloads": 0, "alerts": 1}),
])
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_sign_out_reloads_only_once_the_session_ended(logout, expected):
    assert _sign_out(logout) == expected


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_double_click_sends_one_sign_out():
    # The second click's POST used to land after the first had revoked the
    # session; its 403 (no live cookie left) alerted "Sign-out failed" over
    # a sign-out that succeeded.
    assert _sign_out([200, 403], clicks=2) == {
        "tokens": ["S1"], "reloads": 1, "alerts": 0}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_a_failed_sign_out_can_be_retried():
    assert _sign_out([500, 200], clicks=2, gap_ms=20) == {
        "tokens": ["S1", "S1"], "reloads": 1, "alerts": 1}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_retry_reads_a_renamed_csrf_cookie():
    # SESSION_CSRF_COOKIE_NAME renamed the cookie (e.g. __Host- behind
    # HTTPS): the live-cookie lookup matched only "whisper_csrf", found
    # nothing and skipped the retry. whoami now names the cookie.
    name = "__Host-whisper_csrf"
    assert _sign_out([403, 200], csrf_cookie=name,
                     whoami_extra={"csrf_cookie": name}) == {
        "tokens": ["S1", "S2"], "reloads": 1, "alerts": 0}
