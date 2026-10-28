"""Behaviour of the /captures page's inline JS.

The page is client-rendered, so these tests pull a function's source out of
the served HTML and run it under `node` against small DOM stand-ins (house
style: tests/quick_config/test_pipeline_popover_color.py). Where the logic is
too DOM-bound to isolate, the test pins the served source instead."""

import json
import re
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
    fns = "\n".join(_js_function(page, f) for f in (
        "_bulkStatus", "_undoBulk", "_renderUnlessKeptOpenEdits"))
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
def test_capture_save_keeps_edits_made_while_it_ran(page, tmp_path):
    """The capture Save cleared the dirty flag whatever happened during its
    PATCH, so a chip or note typed meanwhile (never sent) read as saved and
    the next filter change or refresh dropped it without a prompt."""
    fns = "\n".join(_js_function(page, f) for f in ("onSave", "markDirty", "clearDirty"))
    out = _run_node(tmp_path, fns + r"""
var pending = [], toasts = [];
function api() { return new Promise(function(r) { pending.push(r); }); }
function toast(m) { toasts.push(m); } function reloadCounts() {}
var state = { cid: 'c1', corrections: [{ wrong: 'a', correct: 'A', idx: 1 }],
  baselineCorrections: [], adminNotes: '', dirty: false };
function tick() { return new Promise(function(r) { setTimeout(r, 0); }); }
(async function() {
  markDirty(state);
  var p = onSave(state, {});
  state.adminNotes = 'typed meanwhile';
  markDirty(state);
  pending[0]({ capture: {} });
  await p;
  var raced = state.dirty;
  p = onSave(state, {});                // nothing edited during this one
  pending[1]({ capture: {} });
  await p;
  console.log(JSON.stringify({ raced: raced, dirty: state.dirty, toasts: toasts }));
})();
""")
    assert out == {"raced": True, "dirty": False,
                   "toasts": ["Saved · newer edits not saved yet", "Saved."]}


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
    (["alice", "carol"], "/captures/api/propose-merges?user_id=alice%2Ccarol", ["pa", "pc"]),
])
def test_proposals_follow_the_speaker_filter(page, tmp_path, picked, url, shown):
    """The proposals fetch ignored the speaker filter, so an admin saw every
    user's proposals while the list (and Accept's eligibility check) held
    only the picked speakers': Accept failed with a "try Refresh" toast
    that Refresh could never fix."""
    out = _run_node(tmp_path, _js_function(page, "_loadProposals") + r"""
var _filtSpeakers = """ + json.dumps(picked) + r""";
var urls = [], rendered = null, _batchActive = false;
function _stopAnyPreview() {}
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

@needs_node
def test_batch_open_closed_while_loading_never_enters_batch_mode(page, tmp_path):
    """Closing "Batch review merges" before its proposals GET returned still
    entered batch mode in the hidden modal, where Ctrl+ArrowRight anywhere
    on the page accepted (POSTed) a proposal the user never saw."""
    fns = _js_function(page, "_openProposeModal") + "\n" + _js_function(page, "_closePropose")
    out = _run_node(tmp_path, fns + r"""
var shown = false, entered = 0, pending = [];
var modal = { classList: {
  add: function() { shown = true; }, remove: function() { shown = false; },
  contains: function(c) { return c === 'show' && shown; } } };
global.document = { getElementById: function() { return modal; } };
var _batchActive = false, _proposals = [];
function _loadProposals() { return new Promise(function(r) { pending.push(r); }); }
function _enterBatchMode() { entered++; _batchActive = true; }
function _exitBatchMode() { _batchActive = false; }
function _stopAnyPreview() {}
function tick() { return new Promise(function(r) { setTimeout(r, 0); }); }
(async function() {
  _openProposeModal({ startInBatch: true });
  _closePropose();                       // Close before the GET returned
  _proposals = [{ member_ids: ['a'] }];
  pending[0]();
  await tick();
  var closed = entered;
  _openProposeModal({ startInBatch: true });
  pending[1]();
  await tick();
  console.log(JSON.stringify({ closed: closed, open: entered }));
})();
""")
    assert out == {"closed": 0, "open": 1}


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


@needs_node
def test_group_status_lock_and_save_repaint_the_header_pills(page, tmp_path):
    """The group card's header pills were painted once: Ready left "new" in
    the header, Lock added no "locked" pill, and a Regenerate that cleared
    is_stale kept the "stale" pill, until the next full render()."""
    anchor = "async function(v) {\n            var prev = sampleState.newStatus;"
    on_status = _js_block(page, page.index(anchor))
    out = _run_node(tmp_path, _js_function(page, "_syncSampleHead") + r"""
function mk(cls, text) { return { className: cls, textContent: text }; }
var kids = [mk('expand-arrow'), mk('pill group-pill', 'sample'),
  mk('pill status-new', 'new'), mk('pill stale-pill', 'stale'), mk('spacer')];
function has(el, c) { return (' ' + el.className + ' ').indexOf(' ' + c + ' ') !== -1; }
var head = {
  querySelector: function(sel) {
    return kids.find(function(k) {
      return sel === '.pill[class*="status-"]' ? /status-/.test(k.className)
        : has(k, sel.slice(1));
    }) || null;
  },
  insertBefore: function(el, ref) { kids.splice(kids.indexOf(ref), 0, el); },
  removeChild: function(el) { kids.splice(kids.indexOf(el), 1); },
};
var card = { querySelector: function(sel) { return sel === '.cc-head' ? head : null; } };
global.document = { createElement: function() { return {}; } };
var g = { id: 'g1' }, _allSamples = [{ id: 'g1', status: 'new' }];
var sampleState = { newStatus: 'new' }, statusGrp = { setValue: function() {} };
async function api() { return {}; }
function reloadCounts() {} function toast() {}
var onStatus = """ + on_status + r""";
function pills() { return kids.map(function(k) { return k.textContent || k.className; }); }
(async function() {
  await onStatus('ready');
  var afterStatus = pills();
  _syncSampleHead(card, { is_locked: true });          // Lock
  var afterLock = pills();
  _syncSampleHead(card, { status: 'ready', is_stale: false, is_locked: true });
  var afterRegen = pills();
  _syncSampleHead(card, { is_stale: true, is_locked: false });
  console.log(JSON.stringify({ afterStatus: afterStatus, afterLock: afterLock,
    afterRegen: afterRegen, end: pills() }));
})();
""")
    assert out == {
        "afterStatus": ["expand-arrow", "sample", "ready", "stale", "spacer"],
        "afterLock": ["expand-arrow", "sample", "ready", "stale", "locked", "spacer"],
        "afterRegen": ["expand-arrow", "sample", "ready", "locked", "spacer"],
        "end": ["expand-arrow", "sample", "ready", "stale", "spacer"],
    }
    apply_fn = _js_function(page, "applyServerSample")
    assert "_syncSampleHead(card, d);" in apply_fn
    lock = page[page.index("lockBtn.onclick = "):]
    assert "_syncSampleHead(card, { is_locked: d.is_locked });" in lock[:lock.index("toast('Updated');")]


@needs_node
def test_group_audio_fetch_superseded_or_after_collapse_is_dropped(page, tmp_path):
    """refreshAudio revoked only the src present when it started: a fetch
    overtaken by a newer refresh, or landing after the card collapsed, set
    a blob URL nothing would revoke, and an older WAV could win the race
    against the regenerated one."""
    out = _run_node(tmp_path, "var audioSeq = 0;\n" + _js_function(page, "refreshAudio") + r"""
var made = [], pending = [];
var audioPlayerWrap = {}, audioSlot = { firstChild: audioPlayerWrap };
var audio = { src: '', removeAttribute: function() { this.src = ''; } };
var g = { id: 'g1' }, _openSamples = { g1: { audio: audio } };
global.fetch = function() { return new Promise(function(r) { pending.push(r); }); };
global.URL = { createObjectURL: function(b) { made.push(b.n); return 'blob:' + b.n; },
  revokeObjectURL: function() {} };
function reply(i, n) { pending[i]({ ok: true, blob: function() { return { n: n }; } }); }
function tick() { return new Promise(function(r) { setTimeout(r, 0); }); }
(async function() {
  refreshAudio();                 // expand
  refreshAudio();                 // Regenerate before the first landed
  reply(1, 'new'); await tick();
  reply(0, 'old'); await tick();
  var raced = { made: made.slice(), src: audio.src };
  refreshAudio();
  delete _openSamples.g1;         // collapsed while fetching
  reply(2, 'late'); await tick();
  console.log(JSON.stringify({ raced: raced, made: made }));
})();
""")
    assert out == {"raced": {"made": ["new"], "src": "blob:new"}, "made": ["new"]}


# ---- stacking -------------------------------------------------------------------------

def _z_index(page, selector):
    """The z-index set by the page's `  <selector> {` rules."""
    found = re.findall(r"\n  " + re.escape(selector) + r" \{[^}]*?z-index: (\d+);", page)
    assert len(found) == 1, (selector, found)
    return int(found[0])


def test_confirm_dialog_and_toast_stack_above_the_merge_modals(page):
    """The merge modal's Create asks _confirm while the modal is open, but
    the dialog (and the toast) sat under its full-screen overlay: invisible,
    yet a blind Enter still confirmed and dropped the open card's edits."""
    overlay = max(_z_index(page, "#merge-modal"), _z_index(page, "#propose-modal"))
    assert _z_index(page, ".modal") > overlay
    assert _z_index(page, "#toast") > _z_index(page, ".modal")


# ---- chip placement by aligned words ------------------------------------------------

# "die Katze die Maus" after four fillers the pipeline removed: the chip on the
# FIRST "die" sits at karaoke idx 4, which is text token 2 (the second "die")
# when the removed words are not mapped out.
_KATZE_TEXT = "die Katze die Maus"
_KATZE_CHIP = {"wrong": "die", "correct": "Die", "idx": 4}
_KATZE_RAW = [{"word": w} for w in ("äh", " äh", " äh", " äh", " die", " Katze", " die", " Maus")]
_KATZE_ALIGNED = ([{"word": "äh", "removed": True}] * 4
                  + [{"word": w} for w in (" die", " Katze", " die", " Maus")])


@needs_node
def test_merge_modal_preview_places_chips_by_aligned_words(page, tmp_path):
    """The merge modal's transcript and member lines, and the collapsed
    capture card's preview, applied chips without the aligned words, so a
    chip on a repeated word after removed fillers landed on the wrong copy:
    the reviewer approved a text the export would not emit."""
    fns = "\n".join(_js_function(page, f) for f in (
        "_applyChipsToText", "_buildDefaultTranscript", "_renderMergePreview",
        "_capturePreviewText"))
    out = _run_node(tmp_path, "var JOIN_STR = { space: ' ', period_space: '. ' };\n" + fns + r"""
function escapeHtml(s) { return String(s); }
var lines = [];
var host = { set innerHTML(v) {}, appendChild: function(l) { lines.push(l.innerHTML); } };
global.document = { getElementById: function() { return host; },
  createElement: function() { return {}; } };
var row = { text_for_training: """ + json.dumps(_KATZE_TEXT) + r""",
  corrections: [""" + json.dumps(_KATZE_CHIP) + r"""],
  words: """ + json.dumps(_KATZE_ALIGNED) + r""", audio_s: 1 };
var bare = Object.assign({}, row, { words: undefined });
_renderMergePreview([row]);
console.log(JSON.stringify({
  transcript: _buildDefaultTranscript([row, row], 'space'),
  member: lines[0].replace(/^<span[^>]*>[^<]*<\/span>/, ''),
  card: _capturePreviewText(row, row.words),
  unaligned: _capturePreviewText(bare, bare.words) }));
""")
    assert out == {"transcript": "Die Katze die Maus Die Katze die Maus",
                   "member": "Die Katze die Maus",
                   "card": "Die Katze die Maus",
                   "unaligned": "die Katze Die Maus"}


# The open group card's applyServerSample under node: every DOM handle it
# touches is a catch-all stand-in except the members list and status buttons.
_SAMPLE_CARD_STUBS = r"""
function fake() {
  var store = {};
  return new Proxy(function() {}, {
    get: function(t, k) {
      if (k === Symbol.toPrimitive) return function() { return ''; };
      if (k === 'then') return undefined;
      if (!(k in store)) store[k] = fake();
      return store[k];
    },
    set: function(t, k, v) { store[k] = v; return true; },
    apply: function() { return fake(); },
  });
}
global.document = { createElement: function() { return fake(); } };
var memberLines = [], statusShown = [], headSynced = [];
var membersDiv = { set innerHTML(v) {}, appendChild: function(l) { memberLines.push(l.innerHTML); } };
var statusGrp = { setValue: function(v) { statusShown.push(v); }, root: fake() };
var refsWrap = fake(), notesArea = fake(), staleHint = fake(), regenBtn = fake(),
    lockBtn = fake(), dissolveBtn = fake(), strip = fake(), card = fake(),
    saveTBtn = fake(), detail = {};
function _textLine() { return fake(); }
function applyCorrectionsToGround() {} function clearDirty() {}
function refreshSettingsHint() {} function renderChips() {} function selectWord() {}
function refreshAudio() {} function onWordClick() {}
function _dispWord(w) { return w.word; } function _dispRemoved(w) { return !!w.removed; }
function _syncSampleHead(c, d) { headSynced.push(d.status); }
function reloadCounts() {} function toast() {}
function escapeHtml(s) { return String(s); }   // the page's /"/ regex trips _js_block
"""


@needs_node
def test_group_members_list_places_chips_by_aligned_words(page, tmp_path):
    """The open group card's members list applied each member's chips
    without its aligned words (only raw words come with the members), so a
    chip after removed fillers moved to another copy of its word."""
    fns = "\n".join(_js_function(page, f) for f in (
        "applyServerSample", "_applyChipsToText"))
    member = {"id": "m1", "text_for_training": _KATZE_TEXT, "corrections": [_KATZE_CHIP],
              "words": _KATZE_RAW, "sample_order": 0, "audio_s": 1.0}
    merged = [dict(w, start=i * 0.1, end=i * 0.1 + 0.05, member_idx=0)
              for i, w in enumerate(_KATZE_ALIGNED)]
    out = _run_node(tmp_path, _SAMPLE_CARD_STUBS + fns + r"""
var g = { id: 'g1' }, _allSamples = [], sampleState = {};
var d = { id: 'g1', members: [""" + json.dumps(member) + r"""],
  merged_words: """ + json.dumps(merged) + r""" };
applyServerSample(d);
var full = memberLines[0];
d.merged_words = d.merged_words.slice(1);        // a word lost to the trim
memberLines = [];
applyServerSample(d);
console.log(JSON.stringify({ aligned: full, trimmed: memberLines[0] }));
""")
    assert out == {"aligned": "[1] 1.0s · Die Katze die Maus",
                   "trimmed": "[1] 1.0s · die Katze Die Maus"}


# ---- proposals Refresh ------------------------------------------------------------------

@needs_node
@pytest.mark.parametrize("batch", [True, False])
def test_proposals_refresh_stops_the_preview_and_redraws_the_batch_card(page, tmp_path, batch):
    """Refresh replaced _proposals but left the batch card on the proposal
    shown before, so Accept (which takes _proposals[0]) merged one nobody
    saw, often one just dismissed; and the ▶ audio of a card the redraw
    detached played on with no control left to stop it."""
    out = _run_node(tmp_path, _js_function(page, "_loadProposals") + r"""
var _filtSpeakers = [], _batchActive = """ + json.dumps(batch) + r""";
var _batchUndo = { action: 'dismiss', proposals: [] };
var events = [], _proposals = [];
global.document = { getElementById: function() { return {}; } };
async function api() { events.push('get'); return { proposals: [{ id: 'pa' }, { id: 'pb' }] }; }
function _stopAnyPreview() { events.push('stop'); }
function _renderProposals(ps) { _proposals = ps; }
function _renderBatchCard() { events.push('card:' + _proposals[0].id); }
_loadProposals().then(function() {
  console.log(JSON.stringify({ events: events, undo: _batchUndo }));
});
""")
    if batch:
        assert out == {"events": ["stop", "get", "card:pa"], "undo": None}
    else:
        assert out == {"events": ["stop", "get"], "undo": {"action": "dismiss", "proposals": []}}


# ---- bulk actions vs edits made during the request ---------------------------------------

_BULK_DIRTY_STUBS = r"""
var _openRows = {}, _openSamples = {}, _bulkBusy = false, renders = 0, asked = [];
var answers = [], pending = [];
async function _confirm(o) { asked.push(o.title); return answers.shift(); }
function api() { return new Promise(function(r) { pending.push(r); }); }
var _allCaptures = [{ id: 'a', status: 'new', audio_s: 1 }];
function _selectedRows() { return _allCaptures.slice(); }
function _clearSelection() {} function render() { renders++; } function reloadCounts() {}
function toast() {} function fmtHours() { return '1s'; } function collapse() {}
global.document = { querySelector: function() { return null; } };
function tick() { return new Promise(function(r) { setTimeout(r, 0); }); }
"""


@needs_node
@pytest.mark.parametrize("action, reply", [
    ("_bulkStatus('reviewed')", {"updated": [{"id": "a", "prev_status": "new"}], "skipped": []}),
    ("_undoBulk([{ id: 'a', prev_status: 'reviewed' }])", {"updated": [{"id": "a"}], "skipped": []}),
    ("_bulkDelete()", {"deleted": [], "skipped": []}),
])
def test_bulk_action_asks_again_before_rendering_over_edits_made_meanwhile(
        page, tmp_path, action, reply):
    """The bulk status change, its Undo and bulk delete asked about open
    edits only before their request. A card opened and edited while it ran
    lost its chips / notes to the render() after it, with no prompt."""
    fns = "\n".join(_js_function(page, f) for f in (
        "_bulkStatus", "_undoBulk", "_bulkDelete", "_renderUnlessKeptOpenEdits",
        "_okToDropOpenEdits", "_anyOpenDirty"))
    out = _run_node(tmp_path, fns + _BULK_DIRTY_STUBS + r"""
var reply = """ + json.dumps(reply) + r""";
async function run(keep) {
  _openRows = {};
  answers = [true];                     // bulk delete's own confirm
  var p = """ + action + r""";
  await tick();
  _openRows.c1 = { dirty: true };       // opened + edited mid-request
  answers = [!keep];
  pending.shift()(reply);
  await p; await tick();
}
(async function() {
  await run(true);
  var kept = { renders: renders, asked: asked.slice() };
  asked = [];
  await run(false);
  console.log(JSON.stringify({ kept: kept, dropped: { renders: renders, asked: asked } }));
})();
""")
    assert out["kept"]["renders"] == 0
    assert out["kept"]["asked"][-1] == "Show the updated list?"
    assert out["dropped"]["renders"] == 1
