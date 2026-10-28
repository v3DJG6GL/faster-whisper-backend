"""Shared chrome whoami pass (OPEN_MODE_BANNER_JS): every non-ok
/auth/whoami used to read as "not signed in", so a 500 (session store briefly
locked), a proxy 502/503 or a 429 asked a signed-in user for their API key.
Only a 401/403 is a verdict on the session now; any other status re-applies
the last known whoami.

The script runs under node against a minimal DOM stub and a fetch stub that
answers whoami with a chosen status, no browser needed."""

import json
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common


_STUB = r"""
const roles = new Set();
global.window = globalThis;
window.addEventListener = () => {};
global.document = {
  cookie: '',
  body: { classList: { add(c) { roles.add(c); }, remove(c) { roles.delete(c); } } },
  querySelector() { return null; },
  getElementById() { return null; },
  querySelectorAll() { return []; },
};
if (CACHED) window.__whoami = CACHED;
global.fetch = () => Promise.resolve({
  ok: STATUS === 200, status: STATUS,
  json: () => Promise.resolve({ open_mode: false, is_admin: true }),
});
"""

_DRIVE = r"""
let gate = 0;
window._showLoginGate = () => { gate++; };
window._hideLoginGate = () => {};
// exit explicitly: an uncached 5xx leaves a backoff retry pending
setTimeout(() => {
  console.log(JSON.stringify({
    gate, admin: roles.has('role-admin'), whoami: window.__whoami || null,
  }));
  process.exit(0);
}, 50);
"""


def _run(status: int, cached) -> dict:
    js = web_common.OPEN_MODE_BANNER_JS.strip()
    js = js[len("<script>"):-len("</script>")]
    given = f"const STATUS = {status}, CACHED = {json.dumps(cached)};\n"
    out = subprocess.run(
        ["node", "-"], input=given + _STUB + js + _DRIVE,
        capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


_ADMIN = {"open_mode": False, "is_admin": True}


@pytest.mark.parametrize("status", [500, 502, 503, 429])
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_server_error_keeps_the_signed_in_chrome(status):
    res = _run(status, _ADMIN)
    assert res == {"gate": 0, "admin": True, "whoami": _ADMIN}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_server_error_without_a_cached_whoami_shows_no_gate():
    assert _run(500, None)["gate"] == 0


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_rejected_session_shows_the_gate_and_drops_whoami(status):
    res = _run(status, _ADMIN)
    assert res == {"gate": 1, "admin": False, "whoami": None}
