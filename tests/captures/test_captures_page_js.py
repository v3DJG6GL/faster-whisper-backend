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


@needs_node
def test_strip_keyboard_follows_the_latest_preview_activation(page, tmp_path):
    """A merge-preview panel reuses its strip element, but each ▶ builds a
    fresh state. The keyboard handlers bound once kept the first state, so
    from the second activation on a typed chip saved activation 1's stale
    list over the members' chips."""
    fn = _js_function(page, "_bindStripKeyboard")
    out = _run_node(tmp_path, fn + r"""
var handlers = {};
var strip = { parentNode: null, setAttribute: function() {},
  classList: { add: function() {}, remove: function() {} },
  addEventListener: function(t, h) { handlers[t] = h; } };
global.document = { createElement: function() {
  return { classList: { add: function() {}, remove: function() {} } }; } };
function _moveCursor(s, i) { s.cursorIdx = i; }
var first = { stripEl: strip, words: ['a', 'b', 'c'], cursorIdx: 0 };
var second = { stripEl: strip, words: ['a', 'b', 'c'], cursorIdx: 0 };
_bindStripKeyboard(first);
_bindStripKeyboard(second);       // same panel, next activation
handlers.keydown({ key: 'ArrowRight', preventDefault: function() {} });
console.log(JSON.stringify([first.cursorIdx, second.cursorIdx]));
""")
    assert out == [0, 1]


# ---- merge preview ▶ ------------------------------------------------------------

def _preview_click_handler(page):
    anchor = "toggleBtn.addEventListener('click', async function() {"
    return _js_block(page, page.index(anchor) + len("toggleBtn.addEventListener('click', "))


@needs_node
def test_preview_load_superseded_while_in_flight_never_binds(page, tmp_path):
    """A ▶ load still in flight when the preview was stopped (another card's
    ▶, batch advance or close) bound and played anyway: its blob URL and
    shared-<audio> listeners leaked and it played under the wrong card, or
    with the batch modal already closed."""
    out = _run_node(tmp_path, r"""
var _previewAudio = null, _previewBlobUrl = null, _previewActivePanel = null,
    _previewBound = null, _previewGen = 0;
""" + _js_function(page, "_stopAnyPreview") + r"""
var played = 0, bound = 0, blobs = 0, pending = [];
var audio = { paused: true, play: async function() { played++; }, pause: function() {},
  removeAttribute: function() {}, load: function() {} };
function _getPreviewAudio() { return (_previewAudio = audio); }
global.fetch = function() { return new Promise(function(r) { pending.push(r); }); };
global.window = {};
global.URL = { createObjectURL: function() { blobs++; return 'blob:x'; },
  revokeObjectURL: function() {} };
function _bindPanelToAudio() { bound++; return function() {}; }
function _renderWordStrip() { return []; }
function _bindStripKeyboard() {} function renderChips() {}
function applyCorrectionsToGround() {} function toast() {}
var toggleBtn = { _setIcon: function(i) { toggleBtn.icon = i; } };
var panel = {}, stripEl = {}, chipBox = {}, gtArea = {};
function memberIdsFn() { return ['m1', 'm2']; }
var onToggle = """ + _preview_click_handler(page) + r""";
function tick() { return new Promise(function(r) { setTimeout(r, 0); }); }
(async function() {
  onToggle();
  await tick();
  _stopAnyPreview();               // Esc / next batch card / other card
  pending.forEach(function(r) {
    r({ ok: true, blob: async function() { return {}; },
        json: async function() { return { words: [] }; } });
  });
  await tick(); await tick();
  console.log(JSON.stringify({ bound: bound, played: played, blobs: blobs,
    active: _previewActivePanel === panel, icon: toggleBtn.icon,
    disabled: toggleBtn.disabled }));
})();
""")
    assert out == {"bound": 0, "played": 0, "blobs": 0, "active": False,
                   "icon": "play", "disabled": False}


@needs_node
def test_preview_panel_hidden_really_hides_and_keeps_the_inline_play(page, tmp_path):
    """`.merge-preview-panel { display: flex }` beat the UA [hidden] rule, so
    the manual merge modal showed an empty Corrections box before ▶. With
    hiding effective, stopping an inline-player card must fold only its
    editing sections: its ▶ lives inside the panel."""
    assert ".merge-preview-panel[hidden] { display: none; }" in page
    out = _run_node(tmp_path, r"""
var _previewAudio = null, _previewBlobUrl = null, _previewBound = null,
    _previewGen = 0, _previewActivePanel = null;
""" + _js_function(page, "_stopAnyPreview") + r"""
function mk(inline) {
  return { hidden: false, _inline: inline, _corrSec: { hidden: false },
    _gtSec: { hidden: false },
    _toggleBtn: { _setIcon: function() {}, disabled: true } };
}
var inline = mk(true), modal = mk(false);
_previewActivePanel = inline; _stopAnyPreview();
_previewActivePanel = modal; _stopAnyPreview();
console.log(JSON.stringify({
  inline: [inline.hidden, inline._corrSec.hidden, inline._gtSec.hidden],
  modal: modal.hidden }));
""")
    assert out == {"inline": [False, True, True], "modal": True}


# ---- merge proposals --------------------------------------------------------------

@needs_node
@pytest.mark.parametrize("picked, url, shown", [
    ([], "/captures/api/propose-merges", ["pa", "pb", "pc"]),
    (["bob"], "/captures/api/propose-merges?user_id=bob", ["pb"]),
    (["alice", "carol"], "/captures/api/propose-merges", ["pa", "pc"]),
])
def test_proposals_follow_the_speaker_filter(page, tmp_path, picked, url, shown):
    """The proposals fetch ignored the speaker filter, so an admin saw every
    user's proposals while the list (and Accept's eligibility check) held
    only the picked speakers': Accept failed with a "try Refresh" toast
    that Refresh could never fix."""
    out = _run_node(tmp_path, _js_function(page, "_loadProposals") + r"""
var _filtSpeakers = """ + json.dumps(picked) + r""";
var urls = [], rendered = null;
global.document = { getElementById: function() { return {}; } };
async function api(m, u) {
  urls.push(u);
  return { proposals: [{ id: 'pa', user_id: 'alice' }, { id: 'pb', user_id: 'bob' },
                       { id: 'pc', user_id: 'carol' }] };
}
function _renderProposals(ps) { rendered = ps.map(function(p) { return p.id; }); }
_loadProposals().then(function() {
  console.log(JSON.stringify({ urls: urls, shown: rendered }));
});
""")
    assert out == {"urls": [url], "shown": shown}


# ---- batch review -------------------------------------------------------------------

_BATCH_STUBS = r"""
var _batchActive = true, _batchAccepted = 0, _batchDismissed = 0,
    _batchCurrentCard = null, _batchUndo = null, _batchBusy = false;
var built = 0, posts = [], pendingPost = [];
function fakeEl() {
  return { classList: { add: function() {}, remove: function() {} }, style: {},
    appendChild: function() {}, addEventListener: function() {},
    setAttribute: function() {}, querySelector: function() { return null; } };
}
global.document = { createElement: fakeEl, getElementById: function() { return fakeEl(); } };
function _buildProposalCard() { built++; return fakeEl(); }
function _svgIcon() { return fakeEl(); }
function _renderBatchDone() {}
function _stopAnyPreview() {} function toast() {} function _loadUnlessDirty() {}
function _onBatchTouchStart() {} function _onBatchTouchMove() {} function _onBatchTouchEnd() {}
var _allCaptures = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
async function api(m, u, body) {
  posts.push(body.member_ids[0]);
  return new Promise(function(r) { pendingPost.push(r); });
}
var _proposals = ['a', 'b', 'c'].map(function(id) { return { member_ids: [id] }; });
function wait(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }
global.setInterval = function() { return 0; };   // the strip auto-focus poll
global.clearInterval = function() {};
"""


def _batch_fns(page):
    return (_js_function(page, "_renderBatchCard") + "\n"
            + _js_function(page, "_advanceBatch") + "\n")


@needs_node
def test_batch_second_trigger_during_the_transition_is_ignored(page, tmp_path):
    """Dismiss / Accept splice the queue before the 260 ms re-render, so a
    double-click or a held Ctrl+arrow acted on the next, unseen proposal:
    dismissed it, or POSTed a real sample for it."""
    out = _run_node(tmp_path, _batch_fns(page) + _BATCH_STUBS + r"""
(async function() {
  _batchCurrentCard = fakeEl();
  _advanceBatch('dismiss');
  _advanceBatch('dismiss');            // double-click / key auto-repeat
  _advanceBatch('accept');
  var queued = _proposals.length;
  await wait(300);                     // next card rendered
  _advanceBatch('dismiss');
  console.log(JSON.stringify({ queued: queued, after: _proposals.length,
    dismissed: _batchDismissed, posts: posts }));
})();
""")
    assert out == {"queued": 2, "after": 1, "dismissed": 2, "posts": []}


@needs_node
def test_batch_closed_during_accept_does_not_render_or_autoplay(page, tmp_path):
    """Closing the batch while an Accept POST was in flight still rendered
    the next card into the hidden modal and auto-clicked its ▶, playing
    audio nobody could stop."""
    out = _run_node(tmp_path, _batch_fns(page) + _BATCH_STUBS + r"""
(async function() {
  _batchCurrentCard = fakeEl();
  _advanceBatch('accept');
  await wait(0);
  _batchActive = false;                // Esc / Close (_exitBatchMode)
  pendingPost[0]({ sample_id: 's1' });
  await wait(300);
  console.log(JSON.stringify({ built: built, posts: posts }));
})();
""")
    assert out == {"built": 0, "posts": ["a"]}


# ---- group card ---------------------------------------------------------------------

@needs_node
def test_group_regenerate_asks_before_dropping_unsaved_edits(page, tmp_path):
    """Regenerate (and the missing-audio banner's button) ends in
    applyServerSample, which replaced unsaved chips and notes with the
    server copy and cleared the dirty marker, with no prompt."""
    anchor = "regenBtn.onclick = "
    handler = _js_block(page, page.index(anchor) + len(anchor))
    out = _run_node(tmp_path, r"""
var sampleState = { dirty: true }, answers = [], asked = 0, posts = 0, applied = 0;
var g = { id: 'g1' }, regenBtn = {};
async function _confirm() { asked++; return answers.shift(); }
async function api() { posts++; return { sample: {} }; }
function applyServerSample() { applied++; }
function toast() {} function reloadCounts() {}
""" + _js_function(page, "okToRegenerate") + "\nvar onRegen = " + handler + r""";
(async function() {
  answers.push(false);
  await onRegen();                      // dirty + Cancel → nothing happens
  var cancelled = [posts, applied];
  answers.push(true);
  await onRegen();                      // dirty + confirm → regenerates
  sampleState.dirty = false;
  await onRegen();                      // clean → no prompt
  console.log(JSON.stringify({ cancelled: cancelled, asked: asked,
    posts: posts, applied: applied }));
})();
""")
    assert out == {"cancelled": [0, 0], "asked": 2, "posts": 2, "applied": 2}
    banner = page[page.index("fixBtn.onclick = "):]
    banner = banner[:banner.index("fixBtn.disabled = true;")]
    assert "if (!(await okToRegenerate())) return;" in banner


@needs_node
def test_group_save_keeps_edits_made_while_it_ran(page, tmp_path):
    """Only the Save button is disabled during the (slow, alignment-heavy)
    group PATCH; its response went through applyServerSample, which
    overwrote chips / notes typed meanwhile and cleared the dirty flag."""
    anchor = "saveTBtn.onclick = "
    handler = _js_block(page, page.index(anchor) + len(anchor))
    out = _run_node(tmp_path, _js_function(page, "markDirty") + r"""
var chip = { wrong: 'a', correct: 'A', idx: 1, idx_end: 1 };
var sampleState = { dirty: true, adminNotes: 'n', corrections: [chip],
  baselineCorrections: [] };
var g = { id: 'g1' }, saveTBtn = {}, applied = 0, pending = [];
function api() { return new Promise(function(r) { pending.push(r); }); }
function applyServerSample() { applied++; sampleState.dirty = false; }
function toast() {} function reloadCounts() {}
var onSave = """ + handler + r""";
function tick() { return new Promise(function(r) { setTimeout(r, 0); }); }
(async function() {
  onSave();
  sampleState.adminNotes = 'n + typed meanwhile';
  markDirty(sampleState);
  pending[0]({ sample: {} });
  await tick();
  var raced = { applied: applied, dirty: sampleState.dirty,
    notes: sampleState.adminNotes, baseline: sampleState.baselineCorrections };
  onSave();                             // nothing edited during this one
  pending[1]({ sample: {} });
  await tick();
  console.log(JSON.stringify({ raced: raced, applied: applied }));
})();
""")
    assert out["raced"] == {"applied": 0, "dirty": True, "notes": "n + typed meanwhile",
                            "baseline": [{"wrong": "a", "correct": "A", "idx": 1, "idx_end": 1}]}
    assert out["applied"] == 1
