"""The hub's own whoami pass (core/templates/hub.html) read every non-ok
/auth/whoami as "not signed in" and hid the tiles, so a 500 (session store
briefly locked), a proxy 502/503 or a 429 blanked the hub for a signed-in
user. Only a 401/403 hides it now; any other status keeps the current tiles.

The hub script runs under node against a minimal DOM stub and a fetch stub
whose whoami status changes between the page-load pass and a later
whisper:auth-changed refresh, no browser needed."""

import json
import pathlib
import re
import shutil
import subprocess

import pytest

from faster_whisper_backend.paths import REPO_ROOT


_STUB = r"""
function el() {
  const cls = new Set();
  return {
    hidden: true, textContent: '', className: '', title: '',
    getAttribute(a) { return a === 'data-tier' ? 'any' : 'home'; },
    querySelector() { return null; },
    classList: {
      add(c) { cls.add(c); }, remove(c) { cls.delete(c); },
      toggle(c, on) { if (on) cls.add(c); else cls.delete(c); },
      has(c) { return cls.has(c); },
    },
  };
}
const els = {};
const tile = el();
const handlers = {};
global.window = globalThis;
window.addEventListener = (ev, fn) => { handlers[ev] = fn; };
global.document = {
  body: el(),
  addEventListener() {},
  getElementById(id) { return els[id] || (els[id] = el()); },
  querySelector() { return null; },
  querySelectorAll(sel) {
    if (sel === '.tile') return [tile];
    if (sel === '.tile.on') return tile.classList.has('on') ? [tile] : [];
    return [];
  },
};
let status = STATUSES[0];
global.fetch = (url) => Promise.resolve(url === '/auth/whoami'
  ? { ok: status === 200, status,
      json: () => Promise.resolve({ open_mode: false, is_admin: true }) }
  : { ok: false, status: 404 });
"""

_DRIVE = r"""
setTimeout(() => {
  status = STATUSES[1];
  handlers['whisper:auth-changed']();
  setTimeout(() => console.log(JSON.stringify({
    body_hidden: els['hub-body'].hidden, tile_on: tile.classList.has('on'),
  })), 50);
}, 50);
"""


def _hub_script() -> str:
    src = (pathlib.Path(REPO_ROOT) / "faster_whisper_backend" / "core"
           / "templates" / "hub.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", src, re.S)
    return next(s for s in scripts if "'/auth/whoami'" in s)


def _run(first: int, then: int) -> dict:
    given = f"const STATUSES = {json.dumps([first, then])};\n"
    out = subprocess.run(
        ["node", "-"], input=given + _STUB + _hub_script() + _DRIVE,
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("status", [500, 502, 503, 429])
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_server_error_keeps_the_hub_visible(status):
    assert _run(200, status) == {"body_hidden": False, "tile_on": True}


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_rejected_session_hides_the_hub(status):
    assert _run(200, status) == {"body_hidden": True, "tile_on": False}
