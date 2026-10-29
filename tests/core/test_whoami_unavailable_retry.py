"""A 5xx/429 (or a failed fetch) on the very first /auth/whoami left the page
blank: the shared chrome showed no gate and no nav, the hub kept #hub-body
hidden, and nothing retried until a manual reload. Both whoami passes now
retry with a capped backoff while nothing is painted, and the shared chrome
shows a "server unavailable" notice until an answer arrives.

The scripts run under node against a minimal DOM stub and a fetch stub that
answers whoami from a status sequence. setTimeout is wrapped so the backoff
delays are recorded but fire at once, no browser needed."""

import json
import pathlib
import re
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common
from faster_whisper_backend.paths import REPO_ROOT


_CLOCK = r"""
const realSetTimeout = setTimeout;
const delays = [];
global.setTimeout = (fn, ms) => {
  if (ms >= 1000) { delays.push(ms); return realSetTimeout(fn, 0); }
  return realSetTimeout(fn, ms);
};
let calls = 0;
global.fetch = (url) => {
  if (url !== '/auth/whoami') return Promise.resolve({ ok: false, status: 404 });
  const s = STATUSES[Math.min(calls++, STATUSES.length - 1)];
  if (s === 'reject') return Promise.reject(new TypeError('network'));
  return Promise.resolve({ ok: s === 200, status: s,
    json: () => Promise.resolve({ open_mode: false, is_admin: true }) });
};
"""

_CHROME_STUB = r"""
const roles = new Set();
const byId = {};
global.window = globalThis;
window.addEventListener = () => {};
const body = {
  firstChild: null,
  classList: { add(c) { roles.add(c); }, remove(c) { roles.delete(c); } },
  insertBefore(n) { n.parentNode = body; byId[n.id] = n; },
  removeChild(n) { delete byId[n.id]; },
};
global.document = {
  cookie: '', body,
  createElement() { return { style: {}, setAttribute() {} }; },
  querySelector() { return null; },
  getElementById(id) { return byId[id] || null; },
  querySelectorAll() { return []; },
};
let gate = 0;
let notes = [];
"""

_CHROME_DRIVE = r"""
window._showLoginGate = () => { gate++; };
window._hideLoginGate = () => {};
const seen = setInterval(() => {
  const n = byId['whoami-retry-note'];
  if (n && !notes.includes(n.textContent)) notes.push(n.textContent);
}, 0);
realSetTimeout(() => {
  clearInterval(seen);
  console.log(JSON.stringify({
    gate, admin: roles.has('role-admin'), delays, calls, notes,
    note_left: !!byId['whoami-retry-note'],
  }));
  process.exit(0);
}, 150);
"""


def _run_chrome(statuses: list) -> dict:
    js = web_common.OPEN_MODE_BANNER_JS.strip()
    js = js[len("<script>"):-len("</script>")]
    given = f"const STATUSES = {json.dumps(statuses)};\n"
    out = subprocess.run(
        ["node", "-"], input=given + _CHROME_STUB + _CLOCK + js + _CHROME_DRIVE,
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("first", [503, 429, "reject"])
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_chrome_retries_and_notes_an_unavailable_server_on_load(first):
    res = _run_chrome([first, first, 200])
    assert res == {
        "gate": 0, "admin": True, "delays": [2000, 4000], "calls": 3,
        "notes": ["Server unavailable — retrying…"], "note_left": False,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_chrome_retry_that_gets_a_401_shows_the_gate():
    res = _run_chrome([502, 401])
    assert (res["gate"], res["calls"], res["note_left"]) == (1, 2, False)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_chrome_backoff_is_capped():
    res = _run_chrome([500] * 8 + [200])
    assert res["delays"] == [2000, 4000, 8000, 16000, 30000, 30000, 30000, 30000]


_HUB_STUB = r"""
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
global.window = globalThis;
window.addEventListener = () => {};
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
"""

_HUB_DRIVE = r"""
realSetTimeout(() => {
  console.log(JSON.stringify({
    body_hidden: els['hub-body'].hidden, tile_on: tile.classList.has('on'),
    delays, calls,
  }));
  process.exit(0);
}, 150);
"""


def _hub_script() -> str:
    src = (pathlib.Path(REPO_ROOT) / "faster_whisper_backend" / "core"
           / "templates" / "hub.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", src, re.S)
    return next(s for s in scripts if "'/auth/whoami'" in s)


@pytest.mark.parametrize("first", [500, 502, 429, "reject"])
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_hub_retries_until_whoami_answers_on_load(first):
    given = f"const STATUSES = {json.dumps([first, 200])};\n"
    out = subprocess.run(
        ["node", "-"], input=given + _HUB_STUB + _CLOCK + _hub_script() + _HUB_DRIVE,
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    assert json.loads(out.stdout) == {
        "body_hidden": False, "tile_on": True, "delays": [2000], "calls": 2,
    }


def _run_hub(statuses: list, extra: str, drive: str) -> dict:
    given = f"const STATUSES = {json.dumps(statuses)};\n"
    stub = _HUB_STUB.replace(
        "window.addEventListener = () => {};",
        "const handlers = {};\n"
        "window.addEventListener = (ev, fn) => { handlers[ev] = fn; };")
    # getElementById so a hub that never painted reads as hidden, not a crash.
    drive = drive.replace(
        "els['hub-body'].hidden", "document.getElementById('hub-body').hidden")
    out = subprocess.run(
        ["node", "-"],
        input=given + stub + _CLOCK + extra + _hub_script() + drive,
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_hub_rejected_fetch_paints_the_cached_whoami():
    # The chrome's whoami filled the cache before the hub's fetch was
    # rejected: the hub paints from it instead of backing off again.
    res = _run_hub(
        ["reject"], "window.__whoami = { open_mode: false, is_admin: true };\n",
        _HUB_DRIVE)
    assert (res["body_hidden"], res["tile_on"], res["calls"]) == (
        False, True, 1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_hub_paints_when_the_chrome_answers_during_its_backoff():
    # The hub's retry was rejected with an empty cache; the chrome's whoami
    # answers right after. The hub paints at once and stops retrying.
    hold_retries = r"""
const fastSetTimeout = global.setTimeout;
let held = false;
global.setTimeout = (fn, ms) => {
  if (ms >= 1000) { delays.push(ms); held = true; return 0; }
  return fastSetTimeout(fn, ms);
};
"""
    drive = r"""
realSetTimeout(() => {
  window.__whoami = { open_mode: false, is_admin: true };
  if (handlers['whisper:whoami']) handlers['whisper:whoami']();
  realSetTimeout(() => {
    console.log(JSON.stringify({
      body_hidden: els['hub-body'].hidden, tile_on: tile.classList.has('on'),
      held, calls,
    }));
    process.exit(0);
  }, 50);
}, 50);
"""
    res = _run_hub(["reject"], hold_retries, drive)
    assert res == {
        "body_hidden": False, "tile_on": True, "held": True, "calls": 1}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_chrome_announces_a_fresh_whoami():
    js = web_common.OPEN_MODE_BANNER_JS.strip()
    js = js[len("<script>"):-len("</script>")]
    events = (
        "const events = [];\n"
        "window.dispatchEvent = (e) => { events.push(e.type); };\n")
    drive = _CHROME_DRIVE.replace("gate, admin:", "events, gate, admin:")
    out = subprocess.run(
        ["node", "-"],
        input="const STATUSES = [200];\n" + _CHROME_STUB + _CLOCK + events
        + js + drive,
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    assert json.loads(out.stdout)["events"] == ["whisper:whoami"]
