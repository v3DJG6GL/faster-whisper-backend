"""Behaviour of the /captures page's inline JS.

The page is client-rendered, so these tests pull a function's source out of
the served HTML and run it under `node` against small DOM stand-ins (house
style: tests/quick_config/test_pipeline_popover_color.py). Where the logic is
too DOM-bound to isolate, the test pins the served source instead."""

import json
import shutil
import subprocess

import pytest

_NODE = shutil.which("node")
needs_node = pytest.mark.skipif(_NODE is None, reason="node unavailable")


def _js_block(src, start):
    """Source from `start` through the brace that closes the first `{` after
    it. Skips string literals and comments, so braces inside them don't
    count."""
    i = src.index("{", start)
    depth = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c in "'\"`":
            i += 1
            while src[i] != c:
                i += 2 if src[i] == "\\" else 1
        elif src.startswith("//", i):
            i = src.index("\n", i)
        elif src.startswith("/*", i):
            i = src.index("*/", i) + 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    raise AssertionError("unbalanced block")


def _js_function(html, name):
    """The full `function name(...) {...}` declaration (async or not)."""
    sig = "function " + name + "("
    start = html.index(sig)
    if html[start - 6:start] == "async ":
        start -= 6
    return _js_block(html, start)


def _run_node(tmp_path, code):
    f = tmp_path / "probe.js"
    # UTF-8 explicitly: the template has non-Latin-1 chars and Windows CI
    # defaults to cp1252.
    f.write_text(code, encoding="utf-8")
    p = subprocess.run([_NODE, str(f)], capture_output=True, text=True,
                       encoding="utf-8", timeout=30)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture
def page(client):
    return client.get("/captures").text


# ---- _confirm ---------------------------------------------------------------

@needs_node
def test_confirm_enter_on_focused_cancel_cancels(page, tmp_path):
    """The dialog's document-level Enter handler confirmed regardless of
    focus, and its preventDefault swallowed the button's own activation, so
    Tab to Cancel + Enter ran the destructive action."""
    fn = _js_function(page, "_confirm")
    out = _run_node(tmp_path, fn + r"""
var els = {};
function el(id) {
  return els[id] || (els[id] = { id: id, focus: function() {},
    classList: { add: function() {}, remove: function() {} } });
}
var handlers = [];
global.document = {
  activeElement: null,
  getElementById: el,
  addEventListener: function(t, h) { handlers.push(h); },
  removeEventListener: function(t, h) {
    handlers = handlers.filter(function(x) { return x !== h; });
  },
};
function enter() { handlers[0]({ key: 'Enter', preventDefault: function() {} }); }
(async function() {
  var res = [];
  var p = _confirm({ confirmLabel: 'Delete', danger: true });
  document.activeElement = el('dialog-cancel');
  enter();
  res.push(await p);
  p = _confirm({});
  document.activeElement = el('dialog-ok');
  enter();
  res.push(await p);
  console.log(JSON.stringify(res));
})();
""")
    assert out == [False, True]


# ---- bulk status / undo -------------------------------------------------------

@needs_node
def test_bulk_status_ignores_a_second_click_while_in_flight(page, tmp_path):
    """A second bulk PATCH sent while the first was in flight reported the
    first's freshly written status as prev_status, and its toast replaced
    the first's Undo: the original status was unrecoverable."""
    fns = _js_function(page, "_bulkStatus") + "\n" + _js_function(page, "_undoBulk")
    out = _run_node(tmp_path, "var _bulkBusy = false;\n" + fns + r"""
var calls = [], pending = [], toasts = [];
function api(m, u, body) {
  calls.push(body.status);
  return new Promise(function(res) { pending.push(res); });
}
function _selectedRows() { return [{ id: 'a' }]; }
async function _okToDropOpenEdits() { return true; }
var _allCaptures = [{ id: 'a', status: 'new' }];
function _clearSelection() {} function render() {} function reloadCounts() {}
function toast(msg, err, action) { toasts.push({ msg: msg, action: !!action }); }
function tick() { return new Promise(function(r) { setTimeout(r, 0); }); }
(async function() {
  _bulkStatus('reviewed');
  await tick();
  _bulkStatus('dismissed');                 // double-click / second button
  await tick();
  var during = calls.slice();
  _undoBulk([{ id: 'a', prev_status: 'new' }]);   // Undo pressed mid-flight
  await tick();
  var afterUndo = calls.slice();
  pending[0]({ updated: [{ id: 'a', prev_status: 'new' }], skipped: [] });
  await tick();
  _bulkStatus('ready');                     // guard released afterwards
  await tick();
  console.log(JSON.stringify({ during: during, afterUndo: afterUndo,
    after: calls, undoReoffered: toasts[0].action }));
})();
""")
    assert out["during"] == ["reviewed"]
    assert out["afterUndo"] == ["reviewed"]
    assert out["undoReoffered"] is True
    assert out["after"] == ["reviewed", "ready"]


# ---- single-capture Save ------------------------------------------------------

@needs_node
def test_capture_resave_keeps_a_concurrent_chip(page, tmp_path):
    """After a Save the baseline was set to the server's merged chips while
    the on-screen chips stayed as they were, so a chip another tab added
    (kept by the first merge) read as "user removed it" on the next Save and
    was deleted. Runs two Saves of one open card through the real merge."""
    from faster_whisper_backend.core.text_corrections import three_way_merge_corrections

    x = {"wrong": "a", "correct": "A", "idx": 5}       # tab B's chip
    y = {"wrong": "b", "correct": "B", "idx": 2}
    fn = _js_function(page, "onSave")
    bodies = _run_node(tmp_path, fn + r"""
var bodies = [], replies = [];
async function api(m, u, body) {
  bodies.push(JSON.parse(JSON.stringify(body)));
  return replies.shift();
}
function clearDirty() {} function toast() {} function reloadCounts() {}
var X = """ + json.dumps(x) + ", Y = " + json.dumps(y) + r""";
var state = { cid: 'c1', corrections: [Y], baselineCorrections: [], adminNotes: '' };
(async function() {
  replies.push({ capture: { corrections: [X, Y] } });   // merge kept X
  await onSave(state, {});
  state.corrections[0] = { wrong: 'b', correct: 'Bee', idx: 2 };
  replies.push({ capture: { corrections: [] } });
  await onSave(state, {});
  console.log(JSON.stringify(bodies));
})();
""")
    db = [x]
    for body in bodies:
        db = three_way_merge_corrections(body["baseline_corrections"], body["corrections"], db)
    assert x in db
    assert {"wrong": "b", "correct": "Bee", "idx": 2} in db


@needs_node
def test_capture_status_click_repaints_the_header_pill(page, tmp_path):
    """The status buttons auto-save, but the collapsed header's pill
    (painted once by renderCard) kept saying the old status."""
    build = _js_function(page, "buildBody")
    handler = _js_block(build, build.index("async function(v) {\n        // Auto-save on click."))
    out = _run_node(tmp_path, r"""
var pill = { className: 'pill status-new', textContent: 'new' }, asked = [];
var body = { parentNode: { querySelector: function(s) { asked.push(s); return pill; } } };
var r = { id: 'c1', status: 'new' };
var state = { cid: 'c1', newStatus: 'new' };
var statusGrp = { setValue: function() {} };
async function api() { return { capture: { status: 'ready' } }; }
function reloadCounts() {} function toast() {}
var onStatus = """ + handler + r""";
(async function() {
  await onStatus('ready');
  console.log(JSON.stringify({ pill: pill, asked: asked }));
})();
""")
    assert out["pill"] == {"className": "pill status-ready", "textContent": "ready"}
    assert out["asked"] == ['.cc-head .pill[class*="status-"]']


# ---- chip strip -----------------------------------------------------------------

@needs_node
def test_shift_extend_clears_stale_inline_replacements(page, tmp_path):
    """Shift-click extending the last chip left the green `.word-replacement`
    at its old end word (and at any absorbed chip's end) in the strip, so the
    correction showed twice until the card was rebuilt."""
    fns = (_js_function(page, "extendLastChip") + "\n"
           + _js_function(page, "clearReplacementInline"))
    out = _run_node(tmp_path, fns + r"""
var removed = [];
var strip = { removeChild: function(n) { removed.push(n.at); } };
function repl(at) { return { at: at, parentNode: strip,
  classList: { contains: function(c) { return c === 'word-replacement'; } } }; }
var wordEls = [0, 1, 2, 3, 4, 5].map(function(i) { return { nextSibling: null }; });
wordEls[2].nextSibling = repl(2);   // last chip's sibling (word 2)
wordEls[4].nextSibling = repl(4);   // earlier chip at word 4, absorbed
var state = { wordEls: wordEls, corrections: [
  { wrong: 'e', correct: 'E', idx: 4, idx_end: 4 },
  { wrong: 'c', correct: 'foo', idx: 2, idx_end: 2 },
] };
function selectWord() {} function renderChips() {} function focusLastInput() {}
function markDirty() {} function applyCorrectionsToGround() {}
function spanText() { return 'c d e f'; }
extendLastChip(state, 5);
console.log(JSON.stringify({ removed: removed.sort(), chips: state.corrections }));
""")
    assert out["removed"] == [2, 4]
    assert out["chips"] == [{"wrong": "c d e f", "correct": "foo", "idx": 2, "idx_end": 5}]
