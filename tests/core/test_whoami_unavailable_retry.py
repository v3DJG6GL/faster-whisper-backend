"""On a 5xx/429 (or a failed fetch) from the very first /auth/whoami, both
whoami passes retry with a capped backoff while nothing is painted, and the
shared chrome shows a "server unavailable" notice until an answer arrives.
Without the retry the page would stay blank (no gate, no nav, #hub-body
hidden) until a manual reload.

The scripts run under node against a minimal DOM stub and a fetch stub that
answers whoami from a status sequence. setTimeout is wrapped so the backoff
delays are recorded but fire at once, no browser needed."""

import json
import re

import pytest

from faster_whisper_backend.core import web_common
from tests._page_helpers import needs_node, node_json, script_src
from tests.conftest import read_repo_text


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
    js = script_src(web_common.OPEN_MODE_BANNER_JS)
    given = f"const STATUSES = {json.dumps(statuses)};\n"
    return node_json(given + _CHROME_STUB + _CLOCK + js + _CHROME_DRIVE)


@pytest.mark.parametrize("first", [503, 429, "reject"])
@needs_node
def test_chrome_retries_and_notes_an_unavailable_server_on_load(first):
    res = _run_chrome([first, first, 200])
    assert res == {
        "gate": 0, "admin": True, "delays": [2000, 4000], "calls": 3,
        "notes": ["Server unavailable — retrying…"], "note_left": False,
    }


@needs_node
def test_chrome_retry_that_gets_a_401_shows_the_gate():
    res = _run_chrome([502, 401])
    assert (res["gate"], res["calls"], res["note_left"]) == (1, 2, False)


@needs_node
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
    src = read_repo_text("faster_whisper_backend", "core", "templates", "hub.html")
    scripts = re.findall(r"<script>(.*?)</script>", src, re.S)
    return next(s for s in scripts if "'/auth/whoami'" in s)


@pytest.mark.parametrize("first", [500, 502, 429, "reject"])
@needs_node
def test_hub_retries_until_whoami_answers_on_load(first):
    given = f"const STATUSES = {json.dumps([first, 200])};\n"
    assert node_json(given + _HUB_STUB + _CLOCK + _hub_script() + _HUB_DRIVE) == {
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
    return node_json(given + stub + _CLOCK + extra + _hub_script() + drive)


@needs_node
def test_hub_rejected_fetch_paints_the_cached_whoami():
    # The chrome's whoami filled the cache before the hub's fetch was
    # rejected: the hub paints from it instead of backing off again.
    res = _run_hub(
        ["reject"], "window.__whoami = { open_mode: false, is_admin: true };\n",
        _HUB_DRIVE)
    assert (res["body_hidden"], res["tile_on"], res["calls"]) == (
        False, True, 1)


@needs_node
def test_hub_paints_when_the_chrome_answers_during_its_backoff():
    # The hub's first fetch is rejected with an empty cache, so it backs off;
    # the chrome's whoami answers during that backoff. The hub paints at once
    # and stops retrying.
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


@needs_node
def test_chrome_announces_a_fresh_whoami():
    js = script_src(web_common.OPEN_MODE_BANNER_JS)
    events = (
        "const events = [];\n"
        "window.dispatchEvent = (e) => { events.push(e.type); };\n")
    drive = _CHROME_DRIVE.replace("gate, admin:", "events, gate, admin:")
    res = node_json("const STATUSES = [200];\n" + _CHROME_STUB + _CLOCK + events
                    + js + drive)
    assert res["events"] == ["whisper:whoami"]
