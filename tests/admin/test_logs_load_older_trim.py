"""The /logs "Load older" cursor against the live-tail DOM trim, run under
node: append() and the Load older click handler are cut out of the served
page and driven against a list-backed fake #log. Skipped when node is not
installed."""

import json
import shutil
import subprocess

import pytest

_NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(_NODE is None, reason="node unavailable")


def _block(html, start, end="\n  }\n"):
    i = html.index(start)
    return html[i:html.index(end, i) + len(end)]


_HARNESS = r"""
function kidsOf(o) {
  o.kids = [];
  Object.defineProperty(o, 'childElementCount', { get: function () { return o.kids.length; } });
  Object.defineProperty(o, 'firstChild', { get: function () { return o.kids[0] || null; } });
  return o;
}
var log = kidsOf({ insertBefore: function (frag) {
  frag.kids.forEach(function (el) { el.parentNode = log; });
  log.kids = frag.kids.concat(log.kids); frag.kids = []; } });
function appendLine(target, line) {
  var el = { text: line, parentNode: target,
    classList: { contains: function (c) { return c === 'line'; } },
    remove: function () { var p = this.parentNode; p.kids.splice(p.kids.indexOf(this), 1); } };
  target.kids.push(el);
}
var btn = { style: {}, disabled: false, textContent: 'Load older',
  addEventListener: function (ev, fn) { btn.click = fn; } };
var document = { getElementById: function () { return btn; },
  createDocumentFragment: function () { return kidsOf({}); } };
var window = { scrollTo: function () {} };
const _LOG_DOM_MAX = 5;
let _olderInDom = 0;
const _liveDim = { dimLeft: 0 };
let filterText = '', paused = true;
function _unfold() {}
// Server side: the log chain, oldest first; /logs/older counts `skip` back
// from its newest line at the time the request arrives.
var chain = [];
for (var i = 0; i < 40; i++) chain.push('L' + i);
var pending = null;
function fetch(url) {
  var skip = +/skip=(\d+)/.exec(url)[1];
  var nf = chain.slice().reverse(), win = nf.slice(skip, skip + 4);
  var body = { lines: win.reverse(), next_skip: skip + win.length };
  return new Promise(function (res) {
    pending = function () { res({ ok: true, json: function () { return Promise.resolve(body); } }); };
  });
}
function live(n) {
  for (var k = 0; k < n; k++) { var l = 'L' + chain.length; chain.push(l); append(l); }
}
var tick = function () { return new Promise(function (r) { setTimeout(r, 0); }); };
"""


def test_load_older_stays_contiguous_when_live_lines_trim_during_the_fetch(client, tmp_path):
    """The cursor was read when Load older was sent, but live lines arriving
    during the fetch kept trimming the oldest DOM rows (and rewinding the
    cursor). The batch then landed with a silent gap above the live rows,
    and the next click refetched lines the batch already showed."""
    html = client.get("/logs").text
    js = (_HARNESS + _block(html, "  function append(line) {")
          + _block(html, "  let _logsSkip = 0;") + r"""
(async function () {
  var out = {};
  chain.slice(-5).forEach(append);         // backlog fills the DOM to the cap
  var c1 = btn.click(); live(2); pending(); await c1;
  var c2 = btn.click(); pending(); await c2;
  out.dom = log.kids.map(function (e) { return e.text; });
  out.chain_tail = chain.slice(-out.dom.length);
  console.log(JSON.stringify(out));
})();
""")
    f = tmp_path / "probe.js"
    f.write_text(js, encoding="utf-8")
    p = subprocess.run([_NODE, str(f)], capture_output=True, text=True,
                       encoding="utf-8", timeout=30)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    # Every row shown is the chain's newest lines in order: no gap, no repeat.
    assert out["dom"] == out["chain_tail"]
    assert len(out["dom"]) == 5 + 2 + 4 + 4
