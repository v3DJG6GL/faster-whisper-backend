"""Behaviour of the /settings/api-keys page scripts, executed under node.

The page is client-rendered, so each test cuts one function out of the
served HTML and runs it against a tiny fake DOM (just the members that
function touches). Skipped when node is not installed."""

import json
import shutil
import subprocess

import pytest

_NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(_NODE is None, reason="node unavailable")


def _fn(html, name, indent="  "):
    """Source of `function <name>(` up to its closing brace at `indent`."""
    start = html.index(indent + "function " + name + "(")
    end = html.index("\n" + indent + "}\n", start)
    return html[start:end + len(indent) + 2]


def _run(js, tmp_path):
    f = tmp_path / "probe.js"
    f.write_text(js, encoding="utf-8")
    p = subprocess.run([_NODE, str(f)], capture_output=True, text=True,
                       encoding="utf-8", timeout=30)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


# Minimal DOM: elements are created on first getElementById and keep a real
# classList so `.show` toggling is observable.
_FAKE_DOM = r"""
function mkEl() {
  var s = new Set();
  return { textContent: '', value: '', style: {}, onclick: null,
    classList: { add: function (c) { s.add(c); }, remove: function (c) { s.delete(c); },
                 contains: function (c) { return s.has(c); } },
    setAttribute: function () {}, select: function () {}, setSelectionRange: function () {} };
}
var els = {}, toasts = [], loads = 0, execOk = false, clip = null;
var document = {
  body: { appendChild: function () {}, removeChild: function () {} },
  getElementById: function (id) { return els[id] || (els[id] = mkEl()); },
  createElement: function () { return mkEl(); },
  createRange: function () { return { selectNodeContents: function () {} }; },
  execCommand: function () { return execOk; },
  addEventListener: function () {}, removeEventListener: function () {},
};
var window = { isSecureContext: false,
  getSelection: function () { return { removeAllRanges: function () {}, addRange: function () {} }; } };
var navigator = {};
function showToast(msg, kind) { toasts.push([msg, kind]); }
function load() { loads++; }
"""


def test_copy_and_close_keeps_modal_open_when_copy_fails(client, tmp_path):
    """'Copy & close' closed the show-once key modal straight after starting
    the copy, so a failed copy (LAN HTTP with execCommand blocked, or a
    rejected clipboard write) threw the raw key away. It must close only after
    the copy succeeded."""
    html = client.get("/settings/api-keys").text
    js = _FAKE_DOM + _fn(html, "showKeyModal") + r"""
async function scenario(secure, clipOk, fallbackOk) {
  els = {}; toasts = []; loads = 0; execOk = fallbackOk;
  window.isSecureContext = secure;
  navigator = secure ? { clipboard: { writeText: function () {
    return clipOk ? Promise.resolve() : Promise.reject(new Error('denied')); } } } : {};
  showKeyModal('fwb_raw', 'alice', 'laptop');
  els['key-modal-copyclose'].onclick();
  await new Promise(function (r) { setTimeout(r, 0); });
  return { open: els['key-modal'].classList.contains('show'), loads: loads };
}
(async function () {
  var out = {
    http_fail: await scenario(false, false, false),
    http_ok: await scenario(false, false, true),
    secure_fail: await scenario(true, false, false),
    secure_ok: await scenario(true, true, false),
  };
  console.log(JSON.stringify(out));
})();
"""
    out = _run(js, tmp_path)
    assert out["http_fail"] == {"open": True, "loads": 0}
    assert out["secure_fail"] == {"open": True, "loads": 0}
    assert out["http_ok"] == {"open": False, "loads": 1}
    assert out["secure_ok"] == {"open": False, "loads": 1}


def test_save_perms_baseline_is_the_sent_state(client, tmp_path):
    """Only Save is disabled while the permissions PATCH runs, but the new
    baseline was read from the row when the request resolved: a scope click
    made in flight became "saved", the dirty tint cleared, and the edit was
    lost on the next load()."""
    html = client.get("/settings/api-keys").text
    js = (_fn(html, "permRowState") + "\n" + _fn(html, "recomputePermDirty")
          + "\n" + _fn(html, "savePerms") + r"""
function group(page, val) {
  var g = { dataset: { page: page }, val: val };
  g.querySelector = function () { return { dataset: { val: g.val } }; };
  return g;
}
var cls = new Set();
var groups = [group('logs', 'none'), group('stats', 'own')];
var tr = {
  querySelectorAll: function () { return groups; },
  _tagPicker: { getTags: function () { return ['a']; } },
  classList: { toggle: function (c, on) { if (on) cls.add(c); else cls.delete(c); } },
};
tr._permBaseline = permRowState(tr);
var resolve, sentBody;
function api(method, url, body) {
  sentBody = body;
  return new Promise(function (r) { resolve = r; });
}
function showToast() {}
(async function () {
  groups[0].val = 'all';          // the edit being saved
  savePerms('u1', tr, { disabled: false });
  groups[1].val = 'all';          // a further click while the PATCH runs
  resolve({ ok: true, json: function () { return Promise.resolve({}); } });
  await new Promise(function (r) { setTimeout(r, 0); });
  console.log(JSON.stringify({ sent: sentBody.pages, dirty: cls.has('dirty') }));
})();
""")
    out = _run(js, tmp_path)
    assert out["sent"] == {"logs": "all", "stats": "own"}
    assert out["dirty"] is True
