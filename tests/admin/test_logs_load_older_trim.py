"""The /logs "Load older" cursor against the live-tail DOM trim, run under
node: append() and the Load older click handler are cut out of the served
page and driven against a list-backed fake #log. Skipped when node is not
installed."""

import shutil

import pytest

from tests._page_helpers import js_through as _block, node_file_json

_NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(_NODE is None, reason="node unavailable")


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
Object.defineProperty(log, 'innerHTML', { set: function () { log.kids = []; } });
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
let filterText = '', paused = true, matchCurEl = null;
function _unfold() {}
function _resetLiveDim() {}
function updateNav() {}
var clearBtn = { addEventListener: function (ev, fn) { clearBtn.click = fn; } };
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
    out = node_file_json(tmp_path, js)
    # Every row shown is the chain's newest lines in order: no gap, no repeat.
    assert out["dom"] == out["chain_tail"]
    assert len(out["dom"]) == 5 + 2 + 4 + 4


def test_load_older_after_clear_resumes_right_behind_the_screen(client, tmp_path):
    """Clear emptied #log but kept the cursor at the cleared line count.
    Live lines then trimmed the DOM and stepped the cursor back as if the
    DOM still bordered the cleared block, so "Load older" skipped lines and
    showed others twice. A batch in flight at Clear also landed in the
    emptied pane at its old offset."""
    html = client.get("/logs").text
    js = (_HARNESS + _block(html, "  function append(line) {")
          + _block(html, "  clearBtn.addEventListener('click', () => {", "\n  });\n")
          + _block(html, "  let _logsSkip = 0;") + r"""
(async function () {
  var out = {};
  chain.slice(-5).forEach(append);
  clearBtn.click();
  live(7);                                 // trims 2 of the post-Clear rows
  var c1 = btn.click(); pending(); await c1;
  var c2 = btn.click(); pending(); await c2;
  out.after_trim = { dom: log.kids.map(function (e) { return e.text; }) };
  out.after_trim.chain_tail = chain.slice(-out.after_trim.dom.length);
  var c3 = btn.click(); clearBtn.click();  // Clear while a batch is in flight
  live(2); pending(); await c3;
  var c4 = btn.click(); pending(); await c4;
  out.in_flight = { dom: log.kids.map(function (e) { return e.text; }) };
  out.in_flight.chain_tail = chain.slice(-out.in_flight.dom.length);
  console.log(JSON.stringify(out));
})();
""")
    out = node_file_json(tmp_path, js)
    assert out["after_trim"]["dom"] == out["after_trim"]["chain_tail"]
    assert len(out["after_trim"]["dom"]) == 5 + 4 + 4
    assert out["in_flight"]["dom"] == out["in_flight"]["chain_tail"]
    assert len(out["in_flight"]["dom"]) == 2 + 4
