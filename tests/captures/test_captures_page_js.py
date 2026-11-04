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


def _js_after(src, anchor):
    """The block that starts right after `anchor` (e.g. an `x.onclick = `
    handler)."""
    return _js_block(src, src.index(anchor) + len(anchor))


def _js_function(html, name):
    """The full `function name(...) {...}` declaration (async or not)."""
    sig = "function " + name + "("
    start = html.index(sig)
    if html[start - 6:start] == "async ":
        start -= 6
    return _js_block(html, start)


def _js_functions(html, *names):
    """The named function declarations, newline-joined."""
    return "\n".join(_js_function(html, n) for n in names)


_TICK_JS = "function tick() { return new Promise(function(r) { setTimeout(r, 0); }); }\n"


def _run_node(tmp_path, code):
    """Every probe gets `tick()`: await it for one macrotask turn."""
    f = tmp_path / "probe.js"
    # UTF-8 explicitly: the template has non-Latin-1 chars and Windows CI
    # defaults to cp1252.
    f.write_text(_TICK_JS + code, encoding="utf-8")
    p = subprocess.run([_NODE, str(f)], capture_output=True, text=True,
                       encoding="utf-8", timeout=30)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture
def page(client):
    return client.get("/captures").text


def _confirm_harness(page):
    """_confirm with a stand-in dialog; `handlers` holds the keydown ones.
    `enter(target)` presses Enter on `target` (default: the focused
    element); only the two buttons are inside the dialog."""
    return "var _confirmClose = null;\n" + _js_function(page, "_confirm") + r"""
var els = {};
function el(id) {
  return els[id] || (els[id] = { id: id,
    focus: function() { document.activeElement = this; },
    contains: function(x) {
      return id === 'dialog-modal'
        && (x === els['dialog-ok'] || x === els['dialog-cancel']);
    },
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
function enter(target) {
  if (handlers[0]) handlers[0]({ key: 'Enter', target: target || document.activeElement,
    preventDefault: function() {} });
}
"""


@needs_node
def test_confirm_enter_on_focused_cancel_cancels(page, tmp_path):
    """The dialog's document-level Enter handler confirmed regardless of
    focus, and its preventDefault swallowed the button's own activation, so
    Tab to Cancel + Enter ran the destructive action."""
    out = _run_node(tmp_path, _confirm_harness(page) + r"""
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


@needs_node
def test_confirm_opened_over_another_cancels_that_one(page, tmp_path):
    """A load() that lands opens "Show the refreshed list?" on its own, and
    a filter change may be asking right then: the new dialog took over the
    buttons, so the filter's prompt never settled (every later filter change
    was ignored) and its Enter listener stayed, confirming it on the next
    Enter anywhere."""
    out = _run_node(tmp_path, _confirm_harness(page) + r"""
(async function() {
  var first = _confirm({ title: 'Change the filter?' });
  var second = _confirm({ title: 'Show the refreshed list?' });
  var res = [await first, handlers.length];
  document.activeElement = el('dialog-ok');
  enter();
  res.push(await second, handlers.length);
  console.log(JSON.stringify(res));
})();
""")
    assert out == [False, 1, True, 0]


@needs_node
def test_confirm_ignores_an_enter_from_outside_the_dialog(page, tmp_path):
    """load() opens "Show the refreshed list?" on its own while a card is
    being edited: the Enter that commits a chip input bubbled up to the
    dialog's document listener and confirmed it, dropping the edits."""
    out = _run_node(tmp_path, _confirm_harness(page) + r"""
(async function() {
  var res = [];
  var settled = false;
  var p = _confirm({ confirmLabel: 'Show', danger: true });
  p.then(function() { settled = true; });
  var chipInput = { id: 'chip-input' };
  document.activeElement = chipInput;
  enter(chipInput);
  await null;
  res.push(settled, handlers.length);
  document.activeElement = el('dialog-ok');
  enter();
  res.push(await p);
  console.log(JSON.stringify(res));
})();
""")
    assert out == [False, 1, True]


@needs_node
def test_unasked_refresh_prompt_focuses_cancel(page, tmp_path):
    """A keystroke typed as the unasked prompt appears lands on its focused
    button: OK dropped the open card's edits. The user's own prompts keep
    focusing OK."""
    out = _run_node(tmp_path, _confirm_harness(page) + r"""
var focused = [];
function opened(o) {
  _confirm(o);
  return new Promise(function(r) {
    setTimeout(function() {
      focused.push(document.activeElement && document.activeElement.id);
      _confirmClose(false);
      r();
    }, 80);
  });
}
(async function() {
  await opened({ title: 'Refresh the list?' });
  await opened({ title: 'Show the refreshed list?', focusCancel: true });
  console.log(JSON.stringify(focused));
})();
""")
    assert out == ["dialog-ok", "dialog-cancel"]


def test_load_refresh_prompt_is_marked_unasked(page):
    """Only load()'s own "Show the refreshed list?" pops up unasked; the
    prompt helper passes that on as focusCancel."""
    load = _js_function(page, "load")
    assert "_okToDropOpenEdits('Show the refreshed list?', 'Show', true)" in load
    helper = _js_function(page, "_okToDropOpenEdits")
    assert "focusCancel: !!unasked" in helper


@needs_node
def test_bulk_status_ignores_a_second_click_while_in_flight(page, tmp_path):
    """A second bulk PATCH sent while the first was in flight reported the
    first's freshly written status as prev_status, and its toast replaced
    the first's Undo: the original status was unrecoverable."""
    fns = _js_functions(page, "_bulkStatus", "_undoBulk", "_renderUnlessKeptOpenEdits",
                        "_freshOpenEdits", "_dirtyOpenEdits")
    out = _run_node(tmp_path, "var _bulkBusy = false, _openRows = {}, _openSamples = {};\n"
                    + fns + r"""
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


@needs_node
def test_undo_cancelled_at_the_open_edits_prompt_stays_on_offer(page, tmp_path):
    """Clicking Undo hides its toast, and the previous statuses live only in
    that closure: cancelling the "Undo the bulk change?" prompt (to save the
    open edits first) lost the bulk Undo for good."""
    fns = _js_functions(page, "_undoBulk", "_renderUnlessKeptOpenEdits", "_freshOpenEdits",
                        "_dirtyOpenEdits")
    out = _run_node(tmp_path, "var _bulkBusy = false, _openRows = {}, _openSamples = {};\n"
                    + fns + r"""
var calls = [], toasts = [], answers = [false, true];
function api(m, u, body) { calls.push(body.status); return Promise.resolve({ updated: [] }); }
async function _okToDropOpenEdits() { return answers.shift(); }
var _allCaptures = [];
function render() {} function reloadCounts() {}
function toast(msg, err, action) { toasts.push(action || null); }
(async function() {
  await _undoBulk([{ id: 'a', prev_status: 'new' }]);
  var sent = calls.slice();
  await toasts[0].fn();
  console.log(JSON.stringify({ sent: sent, label: toasts[0].label, after: calls }));
})();
""")
    assert out == {"sent": [], "label": "Undo", "after": ["new"]}


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
function _syncCapturePreview() {}
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
    fns = _js_functions(page, "onSave", "markDirty", "clearDirty")
    out = _run_node(tmp_path, fns + r"""
var pending = [], toasts = [];
function api() { return new Promise(function(r) { pending.push(r); }); }
function toast(m) { toasts.push(m); } function reloadCounts() {}
function _syncCapturePreview() {}
var state = { cid: 'c1', corrections: [{ wrong: 'a', correct: 'A', idx: 1 }],
  baselineCorrections: [], adminNotes: '', dirty: false };
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
def test_capture_save_ignores_a_second_save_while_in_flight(page, tmp_path):
    """The capture Save had no in-flight guard: a second Save before the
    first reply sent the stale baseline, so the server's three-way merge
    kept a chip removed in between and the DB resurrected it."""
    out = _run_node(tmp_path, _js_function(page, "onSave") + r"""
var pending = [], bodies = [];
function api(m, u, body) {
  bodies.push(JSON.parse(JSON.stringify(body)));
  return new Promise(function(r) { pending.push(r); });
}
function clearDirty() {} function toast() {} function reloadCounts() {}
function _syncCapturePreview() {}
var btn = { disabled: false };
var state = { cid: 'c1', corrections: [{ wrong: 'foo', correct: 'bar', idx: 3 }],
  baselineCorrections: [], adminNotes: '', notesBaseline: '' };
(async function() {
  var p = onSave(state, {}, btn);
  var during = btn.disabled;
  state.corrections = [];               // chip removed, Save clicked again
  onSave(state, {}, btn);
  pending[0]({ capture: {} });
  await p;
  var after = btn.disabled;
  var p2 = onSave(state, {}, btn);      // the retry after the reply goes out
  pending[pending.length - 1]({ capture: {} });
  await p2;
  console.log(JSON.stringify({ during: during, after: after, bodies: bodies }));
})();
""")
    chip = {"wrong": "foo", "correct": "bar", "idx": 3}
    assert out == {"during": True, "after": False, "bodies": [
        {"corrections": [chip], "baseline_corrections": []},
        {"corrections": [], "baseline_corrections": [chip]}]}


@needs_node
def test_capture_save_sends_the_notes_only_when_changed(page, tmp_path):
    """The capture Save resent the notes from the card's GET on every Save:
    a chip-only Save erased a note another tab had saved since."""
    out = _run_node(tmp_path, _js_function(page, "onSave") + r"""
var bodies = [];
async function api(m, u, body) { bodies.push(JSON.parse(JSON.stringify(body))); return { capture: {} }; }
function clearDirty() {} function toast() {} function reloadCounts() {}
function _syncCapturePreview() {}
var state = { cid: 'c1', corrections: [], baselineCorrections: [],
  adminNotes: 'old', notesBaseline: 'old' };
(async function() {
  await onSave(state, {});              // chips only: the notes stay out
  state.adminNotes = 'new';
  await onSave(state, {});
  await onSave(state, {});              // sent once, unchanged since
  console.log(JSON.stringify(bodies.map(function(b) { return b.admin_notes === undefined ? null : b.admin_notes; })));
})();
""")
    assert out == [None, "new", None]


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
function reloadCounts() {} function toast() {} function _syncCapturePreview() {}
var onStatus = """ + handler + r""";
(async function() {
  await onStatus('ready');
  console.log(JSON.stringify({ pill: pill, asked: asked }));
})();
""")
    assert out["pill"] == {"className": "pill status-ready", "textContent": "ready"}
    assert out["asked"] == ['.cc-head .pill[class*="status-"]']


def _chip_max(page):
    """The page's CHIP_TEXT_MAX declaration, for functions that read it."""
    return re.search(r"var CHIP_TEXT_MAX = \d+;", page).group(0) + "\n"


@needs_node
def test_shift_extend_clears_stale_inline_replacements(page, tmp_path):
    """Shift-click extending the last chip left the green `.word-replacement`
    at its old end word (and at any absorbed chip's end) in the strip, so the
    correction showed twice until the card was rebuilt."""
    fns = _chip_max(page) + _js_functions(page, "extendLastChip", "clearReplacementInline")
    out = _run_node(tmp_path, fns + r"""
var removed = [];
var strip = { removeChild: function(n) { removed.push(n.at); } };
function repl(at) { return { at: at, parentNode: strip,
  classList: { contains: function(c) { return c === 'word-replacement'; } } }; }
var wordEls = [0, 1, 2, 3, 4, 5].map(function(i) { return { nextSibling: null }; });
wordEls[2].nextSibling = repl(2);   // last chip's sibling (word 2)
wordEls[4].nextSibling = repl(4);   // earlier chip at word 4, absorbed
var state = { wordEls: wordEls, words: [], corrections: [
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
def test_shift_extend_stops_at_a_member_boundary(page, tmp_path):
    """On a group strip, shift-click extended a chip into the next member.
    The server files a chip under the member owning its first word and
    clips its end, keeping both members' text as `wrong`, so the save
    succeeded but the correction never applied. Within a member (and on a
    single capture's strip, which has no member_idx) it still extends."""
    fn = _chip_max(page) + _js_function(page, "extendLastChip")
    out = _run_node(tmp_path, fn + r"""
var toasts = [];
function toast(m) { toasts.push(m); }
function selectWord() {} function renderChips() {} function focusLastInput() {}
function markDirty() {} function applyCorrectionsToGround() {} function clearReplacementInline() {}
function spanText() { return 'span'; }
function run(words, to) {
  var state = { words: words, corrections: [{ wrong: 'foo', correct: 'x', idx: 1, idx_end: 1 }] };
  extendLastChip(state, to);
  return state.corrections[0].idx_end;
}
var grp = [{ member_idx: 0 }, { member_idx: 0 }, { member_idx: 1 }, { member_idx: 1 }];
console.log(JSON.stringify({ across: run(grp, 2), within: run(grp, 0),
  single: run([{}, {}, {}], 2), toasts: toasts.length }));
""")
    assert out == {"across": 1, "within": 1, "single": 2, "toasts": 1}


@needs_node
def test_range_chip_over_a_removed_word_keeps_it_out_of_wrong(page, tmp_path):
    """A shift-click range across a rule-removed filler put the raw filler
    into the chip's `wrong` ("gehe äh heim."). The exported text has no
    filler, so the chip never matched and export silently dropped it."""
    fns = _js_functions(page, "_dispWord", "_dispRemoved", "spanText", "recomputeWrong",
                        "_applyChipsToText")
    out = _run_node(tmp_path, fns + r"""
var words = [{ word: ' Ich' }, { word: ' gehe' }, { word: ' äh', removed: true },
             { word: ' heim.' }];
var trained = words.map(function(w) {
  return { word: w.word, train_word: w.word, train_removed: !!w.removed };
});
var chip = { wrong: 'stale', correct: 'fahre nach Hause.', idx: 1, idx_end: 3 };
var span = spanText({ words: words }, 1, 3);
chip.wrong = span;
console.log(JSON.stringify({ span: span,
  heal: recomputeWrong({ words: trained }, chip),
  text: _applyChipsToText('Ich gehe heim.', [chip], words) }));
""")
    assert out == {"span": "gehe heim.", "heal": "gehe heim.",
                   "text": "Ich fahre nach Hause."}


@needs_node
def test_range_chip_on_served_words_keeps_the_spaces(page, tmp_path, monkeypatch):
    """When the training text equals `final`, the served words carry no
    train_word, and their `word` lost Whisper's leading space. A shift-click
    range over "Hallo Welt" then got wrong="HalloWelt", which the Final
    result preview and the export never found, so the correction vanished."""
    from faster_whisper_backend.captures import routes as cr
    from faster_whisper_backend.pipeline import engine as pl_engine
    monkeypatch.setattr(pl_engine, "_postprocess_text",
                        lambda text, **kw: text)
    words = cr._align_member_words({
        "words": [{"word": w, "start": i, "end": i + 1}
                  for i, w in enumerate((" Hallo", " Welt", " heute"))],
        "final": "Hallo Welt heute", "text_for_training": "Hallo Welt heute",
        "model": None,
    })
    fns = _js_functions(page, "_dispWord", "_dispRemoved", "spanText", "recomputeWrong",
                        "_applyChipsToText")
    out = _run_node(tmp_path, fns + r"""
var words = """ + json.dumps(words) + r""";
var chip = { wrong: 'HalloWelt', correct: 'Servus', idx: 0, idx_end: 1 };
var span = spanText({ words: words }, 0, 1);
chip.wrong = recomputeWrong({ words: words }, chip);
console.log(JSON.stringify({ span: span, heal: chip.wrong,
  text: _applyChipsToText('Hallo Welt heute', [chip], words) }));
""")
    assert out == {"span": "Hallo Welt", "heal": "Hallo Welt",
                   "text": "Servus heute"}


def _heal_fns(page):
    """renderChips and its heal helpers, with a do-nothing DOM."""
    return _chip_max(page) + _js_functions(page, "_dispWord", "_dispRemoved", "recomputeWrong",
                                           "renderChips", "markDirty", "_el") + r"""
function el() { return { classList: { remove: function() {} },
  appendChild: function() {}, addEventListener: function() {} }; }
global.document = { createElement: el };
function refreshAllReplacementsInline() {}
"""


def test_chip_text_cap_matches_the_server():
    """The page's chip-text cap mirrors what clean_corrections keeps."""
    from faster_whisper_backend.captures.routes import _CAPTURES_HTML
    from faster_whisper_backend.core import text_corrections
    assert (f"var CHIP_TEXT_MAX = {text_corrections.CAP_CORRECTION_FIELD};"
            in _CAPTURES_HTML)


@needs_node
def test_chip_healed_on_render_marks_the_card_unsaved(page, tmp_path):
    """A chip stored with a filler inside `wrong` (made before the range fix)
    got its `wrong` rewritten on expand, so the Final result showed the
    correction, but the card stayed clean: no "unsaved" marker, no Save
    prompt, and the export still dropped the stored chip. A chip already in
    its healed form leaves the card clean."""
    out = _run_node(tmp_path, _heal_fns(page) + r"""
var words = [{ word: ' foo' }, { word: ' um', removed: true }, { word: ' bar' },
             { word: ' baz' }];
function run(wrong) {
  var state = { words: words, chipBox: el(), dirty: false,
    corrections: [{ wrong: wrong, correct: 'X', idx: 0, idx_end: 2 }] };
  renderChips(state);
  return [state.corrections[0].wrong, state.dirty];
}
console.log(JSON.stringify({ stale: run('foo um bar'), healed: run('foo bar') }));
""")
    assert out == {"stale": ["foo bar", True], "healed": ["foo bar", False]}
    # The capture card renders its chips before it builds the marker, so the
    # marker has to start out shown when that render already marked it.
    assert "dirty.className = state.dirty ? 'dirty' : 'dirty hidden';" in page


@needs_node
def test_heal_keeps_a_stored_wrong_that_covers_the_words(page, tmp_path):
    """The heal cut a stored `wrong` that covers the words at its idx down
    to those words and marked the card unsaved: a chip saved on a whole
    unspaced sentence then replaced only its first char in the preview
    ("我喜欢狗。喜欢猫。"), a legacy "New York" chip only "New", and a Save
    stored the cut form, so the export did the same for good."""
    out = _run_node(tmp_path, _heal_fns(page) + _js_function(
        page, "_applyChipsToText") + r"""
function run(text, words, chip) {
  var state = { words: words, chipBox: el(), dirty: false, corrections: [chip] };
  renderChips(state);
  return [state.corrections[0].wrong, state.dirty,
          _applyChipsToText(text, state.corrections, words)];
}
console.log(JSON.stringify([
  run('我喜欢猫。', [{ word: ' 我' }, { word: '喜欢' }, { word: '猫' }, { word: '。' }],
      { wrong: '我喜欢猫。', correct: '我喜欢狗。', idx: 0, idx_end: 0 }),
  run('New York City', [{ word: ' New' }, { word: ' York' }, { word: ' City' }],
      { wrong: 'New York', correct: 'Newark', idx: 0 })]));
""")
    assert out == [["我喜欢猫。", False, "我喜欢狗。"],
                   ["New York", False, "Newark City"]]


@needs_node
def test_heal_the_server_stores_the_same_leaves_the_card_clean(page, tmp_path):
    """The server keeps a chip's `wrong` trimmed and cut at 200 chars, so a
    long range chip never matched its full recomputed text: the card read
    "unsaved" on every open, and again right after each Save."""
    out = _run_node(tmp_path, _heal_fns(page) + r"""
var words = [];
for (var i = 0; i < 40; i++) words.push({ word: ' wort' + (i < 10 ? '0' : '') + i });
var full = words.map(function(w) { return w.word; }).join('').trim();
var state = { words: words, chipBox: el(), dirty: false,
  corrections: [{ wrong: full.slice(0, 200), correct: 'X', idx: 0, idx_end: 39 }] };
renderChips(state);
console.log(JSON.stringify({ long: full.length > 200, dirty: state.dirty,
  kept: state.corrections[0].wrong === full.slice(0, 200) }));
""")
    # The full text is never put back either: Save would refuse it.
    assert out == {"long": True, "dirty": False, "kept": True}


@needs_node
def test_chip_text_past_the_server_cap_is_refused_on_the_page(page, tmp_path):
    """The store keeps 200 chars of a chip's `wrong` and `correct`, and Save
    toasted "Saved." anyway: the export then rewrote only part of a long span
    or emitted a cut correction. Extending or clicking past the cap now
    toasts instead, and the correction input stops at the cap."""
    fns = _heal_fns(page) + _js_functions(page, "extendLastChip", "spanText", "onWordClick", "_chipIndexAt")
    out = _run_node(tmp_path, fns + r"""
var toasts = [], inputs = [];
function toast(m, err) { toasts.push(m); }
function selectWord() {} function focusLastInput() {} function chipCovers() { return false; }
function clearReplacementInline() {} function applyCorrectionsToGround() {}
global.document = { createElement: function(tag) { var e = el(); if (tag === 'input') inputs.push(e); return e; } };
var words = [];
for (var i = 0; i < 40; i++) words.push({ word: ' wort' + (i < 10 ? '0' : '') + i });
words.push({ word: ' ' + 'x'.repeat(201) });
var state = { words: words, chipBox: el(), dirty: false,
  corrections: [{ wrong: 'wort00', correct: '', idx: 0, idx_end: 0 }] };
extendLastChip(state, 20);
var within = state.corrections[0].idx_end;
extendLastChip(state, 39);
var past = state.corrections[0].idx_end;
onWordClick(state, 40, false);
console.log(JSON.stringify({ within: within, past: past, chips: state.corrections.length,
  toasts: toasts.length, maxLength: inputs[0].maxLength }));
""")
    assert out == {"within": 20, "past": 20, "chips": 1, "toasts": 2,
                   "maxLength": 200}


@needs_node
def test_heal_on_a_proposal_preview_does_not_auto_save(page, tmp_path):
    """On a proposal preview markDirty is the debounced preview-save-chips
    hook, so pressing ▶ on a proposal with a stale chip rewrote the
    members' chips without any user edit. The heal stays in memory."""
    out = _run_node(tmp_path, _heal_fns(page) + r"""
var saves = 0;
var state = { words: [{ word: ' foo' }, { word: ' um', removed: true }, { word: ' bar' }],
  chipBox: el(), dirty: false, onMarkDirty: function() { saves++; },
  corrections: [{ wrong: 'foo um bar', correct: 'X', idx: 0, idx_end: 2 }] };
renderChips(state);
console.log(JSON.stringify({ wrong: state.corrections[0].wrong, saves: saves }));
""")
    assert out == {"wrong": "foo bar", "saves": 0}


@needs_node
@pytest.mark.parametrize("admin, dirty", [(False, False), (True, True)])
def test_heal_on_a_locked_group_marks_it_unsaved_only_for_an_admin(
        page, tmp_path, admin, dirty):
    """The server 409s a non-admin's Save on a locked sample, so a heal
    left such a group card "unsaved" for good, and every filter change
    asked to drop edits nobody made. An admin can save the heal."""
    fns = _js_functions(page, "applyServerSample", "_applyChipsToText", "_chipFields")
    out = _run_node(tmp_path, _SAMPLE_CARD_STUBS + fns + _heal_fns(page) + r"""
global.document = { createElement: function() { return fake(); } };
function clearDirty(state) { state.dirty = false; }
var _isAdmin = """ + json.dumps(admin) + r""";
var g = { id: 'g1' }, _allSamples = [], sampleState = { chipBox: el() };
applyServerSample({ id: 'g1', is_locked: true, members: [],
  merged_words: [{ word: ' foo', member_idx: 0 }, { word: ' um', removed: true, member_idx: 0 },
                 { word: ' bar', member_idx: 0 }],
  corrections: [{ wrong: 'foo um bar', correct: 'X', idx: 0, idx_end: 2 }] });
console.log(JSON.stringify(sampleState.dirty));
""")
    assert out is dirty


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


def _stop_any_preview_js(page):
    """_stopAnyPreview with its module globals (and the proposal save chain
    the ▶ handler reads)."""
    return r"""
var _previewAudio = null, _previewBlobUrl = null, _previewActivePanel = null,
    _previewBound = null, _previewGen = 0;
var _proposalSaveChain = Promise.resolve(), _proposalSaveFlush = null;
""" + _js_function(page, "_stopAnyPreview")


def _preview_click_handler(page):
    anchor = "toggleBtn.addEventListener('click', async function() {"
    return _js_block(page, page.index(anchor) + len("toggleBtn.addEventListener('click', "))


@needs_node
def test_preview_load_superseded_while_in_flight_never_binds(page, tmp_path):
    """A ▶ load still in flight when the preview was stopped (another card's
    ▶, batch advance or close) bound and played anyway: its blob URL and
    shared-<audio> listeners leaked and it played under the wrong card, or
    with the batch modal already closed."""
    out = _run_node(tmp_path, _stop_any_preview_js(page) + r"""
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
def test_preview_load_waits_for_another_cards_pending_chip_save(page, tmp_path):
    """Chip saves were chained per ▶ activation only: a ▶ on another card
    (or the batch advance's auto-play) within the 250 ms debounce or while
    the save ran read the members' chips first. An overlapping member then
    showed the old chips, and the next edit wrote them back over the save."""
    out = _run_node(tmp_path, _stop_any_preview_js(page) + r"""
var log = [], saves = [];
var audio = { paused: true, play: async function() {}, pause: function() {},
  removeAttribute: function() {}, load: function() {} };
function _getPreviewAudio() { return (_previewAudio = audio); }
global.fetch = async function(u) {
  log.push(u.split('/').pop());
  return { ok: true, blob: async function() { return {}; },
           json: async function() { return { words: [] }; } };
};
global.window = {};
global.URL = { createObjectURL: function() { return 'blob:x'; }, revokeObjectURL: function() {} };
function _saveProposalChips() {
  log.push('save');
  return new Promise(function(r) { saves.push(r); });
}
function _bindPanelToAudio() { return function() {}; }
function _renderWordStrip() { return []; }
function _bindStripKeyboard() {} function renderChips() {}
function applyCorrectionsToGround() {} function toast() {}
function mk(ids) {
  var toggleBtn = { _setIcon: function() {} }, panel = { _toggleBtn: toggleBtn };
  var stripEl = {}, chipBox = {}, gtArea = {};
  function memberIdsFn() { return ids; }
  return { panel: panel, click: """ + _preview_click_handler(page) + r""" };
}
(async function() {
  var a = mk(['x', 'y']), b = mk(['x', 'z']);
  await a.click();
  log = [];
  a.panel._state.onMarkDirty();            // chip typed on A ...
  var pb = b.click();                      // ... and B's ▶ at once
  await tick();
  var during = log.slice();
  saves.shift()();
  await pb;
  console.log(JSON.stringify({ during: during, after: log }));
})();
""")
    assert out == {"during": ["save"],
                   "after": ["save", "preview-audio", "preview-words"]}


@needs_node
def test_preview_panel_hidden_really_hides_and_keeps_the_inline_play(page, tmp_path):
    """`.merge-preview-panel { display: flex }` beat the UA [hidden] rule, so
    the manual merge modal showed an empty Corrections box before ▶. With
    hiding effective, stopping an inline-player card must fold only its
    editing sections: its ▶ lives inside the panel."""
    assert ".merge-preview-panel[hidden] { display: none; }" in page
    out = _run_node(tmp_path, _stop_any_preview_js(page) + r"""
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


@needs_node
@pytest.mark.parametrize("picked, url, shown", [
    ([], "/captures/api/propose-merges", ["pa", "pb", "pc"]),
    (["bob"], "/captures/api/propose-merges?user_id=bob", ["pb"]),
    (["alice", "carol"], "/captures/api/propose-merges?user_id=alice%2Ccarol", ["pa", "pc"]),
])
def test_proposals_follow_the_speaker_filter(page, tmp_path, picked, url, shown):
    """The proposals fetch ignored the speaker filter, so an admin saw every
    user's proposals while the list held only the picked speakers'."""
    out = _run_node(tmp_path, _js_function(page, "_loadProposals") + r"""
var _filtSpeakers = """ + json.dumps(picked) + r""";
var urls = [], rendered = null, _batchActive = false, _proposalsSeq = 0;
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


@needs_node
def test_batch_open_closed_while_loading_never_enters_batch_mode(page, tmp_path):
    """Closing "Batch review merges" before its proposals GET returned still
    entered batch mode in the hidden modal, where Ctrl+ArrowRight anywhere
    on the page accepted (POSTed) a proposal the user never saw."""
    fns = _js_functions(page, "_openProposeModal", "_closePropose")
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
(async function() {
  _openProposeModal({ startInBatch: true });
  _closePropose();                       // Close before the GET returned
  _proposals = [{ member_ids: ['a'] }];
  pending[0](true);
  await tick();
  var closed = entered;
  _openProposeModal({ startInBatch: true });
  pending[1](true);
  await tick();
  console.log(JSON.stringify({ closed: closed, open: entered }));
})();
""")
    assert out == {"closed": 0, "open": 1}


_BATCH_STUBS = r"""
var _batchActive = true, _batchAccepted = 0, _batchDismissed = 0,
    _batchCurrentCard = null, _batchUndo = null, _batchBusy = false,
    _batchLoadHold = false, _proposalsSeq = 0;
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
    return _js_functions(page, "_renderBatchCard", "_advanceBatch", "_el") + "\n"


def _batch_reload_harness(page):
    """The batch card plus _loadProposals; `gets` is for the test's GET stub,
    and a reload replaces the queue."""
    return (_batch_fns(page) + _js_function(page, "_loadProposals") + _BATCH_STUBS
            + "\nvar _filtSpeakers = [], gets = [];\n"
            + "function _renderProposals(ps) { _proposals = ps; }")


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


@needs_node
def test_batch_refresh_blocks_the_card_until_the_queue_reloads(page, tmp_path):
    """A batch-mode Refresh left the current card live during its GET: an
    Accept meanwhile POSTed a sample, then the reloaded queue (swept before
    that create) showed the merged proposal again with its Revert gone."""
    out = _run_node(tmp_path, _batch_reload_harness(page) + r"""
api = function(m, u, body) {
  if (m === 'GET') return new Promise(function(r, j) { gets.push({ r: r, j: j }); });
  posts.push(body.member_ids[0]);
  return new Promise(function(r) { pendingPost.push(r); });
};
(async function() {
  var p = _loadProposals();               // Refresh
  _advanceBatch('accept');                // Ctrl+Right before the GET returns
  gets[0].r({ proposals: [{ member_ids: ['a'] }] });
  await p;
  var afterOk = _batchBusy;
  p = _loadProposals();
  gets[1].j(new Error('HTTP 502'));       // a failed Refresh keeps the old card usable
  await p;
  console.log(JSON.stringify({ posts: posts, afterOk: afterOk, afterFail: _batchBusy }));
})();
""")
    assert out == {"posts": [], "afterOk": False, "afterFail": False}


@needs_node
def test_batch_refresh_during_an_accept_leaves_the_card_held(page, tmp_path):
    """A Refresh while an Accept POST was in flight released the batch hold
    when its GET landed: the card showed the proposal still being POSTed,
    and the next Accept sent the same members a second time."""
    out = _run_node(tmp_path, _batch_reload_harness(page) + r"""
api = function(m, u, body) {
  if (m === 'GET') return new Promise(function(r) { gets.push(r); });
  posts.push(body.member_ids[0]);
  return new Promise(function(r) { pendingPost.push(r); });
};
(async function() {
  _batchCurrentCard = fakeEl();
  _advanceBatch('accept');                // POST 'a' in flight
  _loadProposals();                       // Refresh
  if (gets.length) gets[0]({ proposals: [{ member_ids: ['b'] }, { member_ids: ['c'] }] });
  await wait(0);
  _advanceBatch('accept');
  var busy = _batchBusy;
  pendingPost[0]({ sample_id: 's1' });
  await wait(300);
  console.log(JSON.stringify({ gets: gets.length, busy: busy, posts: posts,
    next: _proposals[0].member_ids[0] }));
})();
""")
    assert out == {"gets": 0, "busy": True, "posts": ["a"], "next": "b"}


@needs_node
@pytest.mark.parametrize("first", ["ok", "fail"])
def test_batch_double_refresh_holds_the_card_until_the_latest_reload(page, tmp_path, first):
    """The batch hold had no sequence token: with two Refreshes in flight the
    first reply (or failure) released it, an Accept then went through, and
    the second reply put back the queue swept before that Accept."""
    out = _run_node(tmp_path, _batch_reload_harness(page) + r"""
api = function(m, u, body) {
  if (m === 'GET') return new Promise(function(r, j) { gets.push({ r: r, j: j }); });
  posts.push(body.member_ids[0]);
  return Promise.resolve({ sample_id: 's1' });
};
var FIRST = """ + json.dumps(first) + r""";
(async function() {
  var p1 = _loadProposals(), p2 = _loadProposals();   // double-click Refresh
  if (FIRST === 'ok') gets[0].r({ proposals: [{ member_ids: ['old'] }] });
  else gets[0].j(new Error('HTTP 502'));
  await p1;
  var held = _batchBusy;
  _advanceBatch('accept');
  gets[1].r({ proposals: [{ member_ids: ['a'] }, { member_ids: ['b'] }] });
  await p2;
  console.log(JSON.stringify({ held: held, posts: posts, busy: _batchBusy,
    queue: _proposals.map(function(p) { return p.member_ids[0]; }) }));
})();
""")
    assert out == {"held": True, "posts": [], "busy": False, "queue": ["a", "b"]}


@needs_node
@pytest.mark.parametrize("mode", ["list", "batch"])
def test_accept_lets_the_server_judge_members_the_list_is_stale_on(page, tmp_path, mode):
    """Accept refused a member the loaded captures list did not know (a clip
    recorded since the page loaded) or still showed as grouped (its sample
    was dissolved in another tab since), and the modal's Refresh re-ran only
    the proposer: the same proposal came back and was refused on every try.
    The server judges the members, and its refusal is what the user sees."""
    out = _run_node(tmp_path, _batch_fns(page) + _js_function(page, "_acceptProposal")
                    + _BATCH_STUBS + r"""
var MODE = """ + json.dumps(mode) + r""", toasts = [];
toast = function(m) { toasts.push(m); };
function _redrawProposalList() {}
api = function(m, u, body) {
  posts.push(body.member_ids.join('+'));
  return body.member_ids[0] === 'x'
    ? Promise.reject(new Error('capture x is already in a sample'))
    : Promise.resolve({ sample_id: 's1' });
};
_allCaptures = [{ id: 'a' }, { id: 'g', sample_id: 's0' }];
_proposals = [{ member_ids: ['g', 'a'] }, { member_ids: ['new', 'a2'] },
              { member_ids: ['x', 'b'] }];
(async function() {
  _batchCurrentCard = fakeEl();
  if (MODE === 'list') {
    _proposals.slice().forEach(function(p) { _acceptProposal(p, null); });
  } else {
    for (var n = 0; n < 3; n++) {
      _advanceBatch('accept');
      await wait(300);                   // next card rendered
    }
  }
  await wait(0);
  console.log(JSON.stringify({ posts: posts, toasts: toasts }));
})();
""")
    assert out == {"posts": ["g+a", "new+a2", "x+b"],
                   "toasts": ["capture x is already in a sample"]}


@needs_node
@pytest.mark.parametrize("second, active, cards", [
    ("refresh", False, []),
    ("open", True, ["b"]),
])
def test_batch_open_superseded_while_loading_waits_for_the_latest_reload(
        page, tmp_path, second, active, cards):
    """A Batch open whose GET a Refresh or second click superseded still
    entered batch mode when it landed, on the queue left from an earlier
    session: the card showed (and auto-played) a stale proposal, live for an
    Accept that the newer reload then released mid-POST."""
    fns = _js_functions(page, "_loadProposals", "_openProposeModal", "_enterBatchMode",
                        "_renderBatchCard", "_advanceBatch", "_el")
    out = _run_node(tmp_path, fns + _BATCH_STUBS + r"""
var _filtSpeakers = [], gets = [], cards = [], shown = false, _batchKeyHandler = null;
_batchActive = false;
_proposals = [{ member_ids: ['old'] }];   // left from an earlier batch session
var modal = fakeEl();
modal.classList = { add: function() { shown = true; }, remove: function() { shown = false; },
  contains: function(c) { return c === 'show' && shown; } };
document.getElementById = function(id) { return id === 'propose-modal' ? modal : fakeEl(); };
document.addEventListener = function() {};
function _onBatchKey() {}
_buildProposalCard = function(p) { cards.push(p.member_ids[0]); return fakeEl(); };
function _renderProposals(ps) { _proposals = ps; }
api = function(m) { return new Promise(function(r) { gets.push(r); }); };
var SECOND = """ + json.dumps(second) + r""";
(async function() {
  _openProposeModal({ startInBatch: true });
  if (SECOND === 'open') _openProposeModal({ startInBatch: true });
  else _loadProposals();                 // the modal's Refresh while loading
  gets[0]({ proposals: [{ member_ids: ['a'] }] });
  await wait(0);
  gets[1]({ proposals: [{ member_ids: ['b'] }] });
  await wait(0);
  console.log(JSON.stringify({ active: _batchActive, cards: cards }));
})();
""")
    assert out == {"active": active, "cards": cards}


@needs_node
def test_group_regenerate_asks_before_dropping_unsaved_edits(page, tmp_path):
    """Regenerate (and the missing-audio banner's button) ends in
    applyServerSample, which replaced unsaved chips and notes with the
    server copy and cleared the dirty marker, with no prompt."""
    handler = _js_after(page, "regenBtn.onclick = ")
    out = _run_node(tmp_path, r"""
var sampleState = { dirty: true }, answers = [], asked = 0, posts = 0, applied = 0;
var g = { id: 'g1' }, regenBtn = {};
async function _confirm() { asked++; return answers.shift(); }
async function api() { posts++; return { sample: {} }; }
function applyServerSample() { applied++; }
function toast() {} function reloadCounts() {}
""" + _js_functions(page, "okToRegenerate", "applyRegenerated")
        + "\nvar onRegen = " + handler + r""";
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
    handler = _js_after(page, "saveTBtn.onclick = ")
    out = _run_node(tmp_path, _js_functions(page, "markDirty", "_chipFields") + r"""
var chip = { wrong: 'a', correct: 'A', idx: 1, idx_end: 1 };
var sampleState = { dirty: true, adminNotes: 'n', corrections: [chip],
  baselineCorrections: [] };
var g = { id: 'g1' }, saveTBtn = {}, applied = 0, pending = [];
function api() { return new Promise(function(r) { pending.push(r); }); }
function applyServerSample() { applied++; sampleState.dirty = false; }
function toast() {} function reloadCounts() {}
var onSave = """ + handler + r""";
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
def test_group_save_sends_the_notes_only_when_changed(page, tmp_path):
    """The group Save resent the notes from the card's GET on every Save:
    a chip-only Save erased a note another tab had saved since."""
    handler = _js_after(page, "saveTBtn.onclick = ")
    out = _run_node(tmp_path, _js_function(page, "_chipFields") + r"""
var sampleState = { adminNotes: 'old', notesBaseline: 'old', corrections: [],
  baselineCorrections: [], editGen: 0 };
var g = { id: 'g1' }, saveTBtn = {}, bodies = [];
function api(m, u, body) { bodies.push(body); return Promise.resolve({ sample: {} }); }
function applyServerSample() { sampleState.notesBaseline = sampleState.adminNotes; }
function toast() {} function reloadCounts() {}
var onSave = """ + handler + r""";
(async function() {
  onSave(); await tick();               // chips only: the notes stay out
  sampleState.adminNotes = 'new';
  onSave(); await tick();
  console.log(JSON.stringify(bodies.map(function(b) { return 'admin_notes' in b ? b.admin_notes : null; })));
})();
""")
    assert out == [None, "new"]


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


# "die Katze die Maus" after four fillers the pipeline removed: the chip on the
# FIRST "die" sits at karaoke idx 4, which is text token 2 (the second "die")
# when the removed words are not mapped out.
_KATZE_TEXT = "die Katze die Maus"
_KATZE_CHIP = {"wrong": "die", "correct": "Die", "idx": 4}
_KATZE_RAW = [{"word": w} for w in ("äh", " äh", " äh", " äh", " die", " Katze", " die", " Maus")]
_KATZE_ALIGNED = ([{"word": "äh", "removed": True}] * 4
                  + [{"word": w} for w in (" die", " Katze", " die", " Maus")])


# _buildDefaultTranscript and the helpers it calls, sliced together.
_TRANSCRIPT_FNS = ("_endsSentence", "_applyChipsToText", "_cjkEdge", "_buildDefaultTranscript")


@needs_node
def test_merge_modal_preview_places_chips_by_aligned_words(page, tmp_path):
    """The merge modal's transcript and member lines, and the collapsed
    capture card's preview, applied chips without the aligned words, so a
    chip on a repeated word after removed fillers landed on the wrong copy:
    the reviewer approved a text the export would not emit."""
    fns = _js_functions(page, *_TRANSCRIPT_FNS, "_renderMergePreview", "_capturePreviewText")
    out = _run_node(tmp_path, fns + r"""
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


@needs_node
def test_merge_preview_skips_a_member_the_pipeline_blanked(page, tmp_path):
    """A removal rule that blanked a Whisper hallucination leaves
    text_for_training "" (not NULL): the merge preview used to fall back to
    the raw hallucination, as the server's transcript builder did. A legacy
    row (NULL column) still falls back to `final`."""
    fns = _js_functions(page, *_TRANSCRIPT_FNS)
    out = _run_node(tmp_path, fns + r"""
console.log(JSON.stringify(_buildDefaultTranscript([
  { text_for_training: '', final: '', raw: 'Untertitel im Auftrag des ZDF' },
  { text_for_training: null, final: 'Wie geht es?', raw: 'wie geht es' }],
  'space')));
""")
    assert out == "Wie geht es?"


@needs_node
def test_card_preview_keeps_a_pipeline_blanked_capture_empty(page, tmp_path):
    """The collapsed card's one-line preview fell back to the raw text when
    the chips-applied result was empty, so a hallucination a removal rule
    blanked showed as the card's label while the merge preview and the
    export left it out. A legacy row (NULL columns) still previews raw."""
    fns = _js_functions(page, "_applyChipsToText", "_cjkEdge", "_capturePreviewText")
    out = _run_node(tmp_path, fns + r"""
console.log(JSON.stringify([
  _capturePreviewText({ text_for_training: '', final: '',
    raw: 'Untertitel im Auftrag des ZDF', corrections: [] }),
  _capturePreviewText({ text_for_training: null, final: null,
    raw: 'wie geht es', corrections: [] })]));
""")
    assert out == ["(empty)", "wie geht es"]


@needs_node
@pytest.mark.parametrize("finals, expected", [
    (["Hallo Welt.", "Wie geht es?", "gut"], "Hallo Welt. Wie geht es? gut"),
    (["Stopp!", "«Ja.»", "nein"], "Stopp! «Ja.» nein"),
    (["eins", "zwei…", "drei"], "eins. zwei… drei"),
    # German „…“ / ‚…‘ closers, CJK and other scripts' sentence enders.
    (["Er sagte: „Hallo Welt.“", "Wie geht es?"],
     "Er sagte: „Hallo Welt.“ Wie geht es?"),
    (["‚Stopp!‘", "gut"], "‚Stopp!‘ gut"),
    (["你好。", "「好！」", "（是？）", "gut"], "你好。「好！」（是？） gut"),
    (["مرحبا؟", "नमस्ते।", "Wie⁉", "Was‼", "gut"],
     "مرحبا؟ नमस्ते। Wie⁉ Was‼ gut"),
    (["{eins.}", "《二。》", "〈三！〉", "›vier?‹", "fünf"],
     "{eins.} 《二。》〈三！〉 ›vier?‹ fünf"),
    (["یہ ٹھیک ہے۔", "Լավ է։", "ደህና ነው።", "ကောင်းတယ်။", "ល្អ។", "ॐ॥", "gut"],
     "یہ ٹھیک ہے۔ Լավ է։ ደህና ነው። ကောင်းတယ်။ ល្អ។ ॐ॥ gut"),
    (["Il a dit : « Oui. »", "non"], "Il a dit : « Oui. » non"),
])
def test_merge_preview_period_space_join_matches_the_server(
        page, tmp_path, finals, expected):
    """The JS twin of samples._build_default_transcript still put a literal
    ". " between members after the server stopped doubling a member's own
    sentence punctuation, so the preview showed "Hallo Welt.. Wie …"."""
    fns = _js_functions(page, *_TRANSCRIPT_FNS)
    rows = [{"final": f} for f in finals]
    out = _run_node(tmp_path, fns + "\nconsole.log(JSON.stringify("
                    "_buildDefaultTranscript(" + json.dumps(rows)
                    + ", 'period_space')));")
    assert out == expected


@needs_node
@pytest.mark.parametrize("strategy", ["space", "period_space"])
@pytest.mark.parametrize("finals, expected", [
    (["今天天气很好", "我们去公园吧。"],
     {"space": "今天天气很好我们去公园吧。", "period_space": "今天天气很好。我们去公园吧。"}),
    (["𠀀", "𠀁"], {"space": "𠀀𠀁", "period_space": "𠀀。𠀁"}),
    (["こんにちは", "元気"], {"space": "こんにちは元気", "period_space": "こんにちは。元気"}),
    (["你好。", "我们"], {"space": "你好。我们", "period_space": "你好。我们"}),
    # Past its CJK punctuation / closers / openers a member is still zh / ja.
    (["今日はいい天気です。", "散歩に行きましょう。"],
     {"space": "今日はいい天気です。散歩に行きましょう。",
      "period_space": "今日はいい天気です。散歩に行きましょう。"}),
    (["你好！", "「好」"], {"space": "你好！「好」", "period_space": "你好！「好」"}),
    (["「はい」", "元気"], {"space": "「はい」元気", "period_space": "「はい」。元気"}),
    (["今天天气很好，", "我们去公园"],
     {"space": "今天天气很好，我们去公园", "period_space": "今天天气很好，。我们去公园"}),
    (["你好。", "Hallo"], {"space": "你好。 Hallo", "period_space": "你好。 Hallo"}),
    (["Hallo.", "「你好」"], {"space": "Hallo. 「你好」", "period_space": "Hallo. 「你好」"}),
    (["他说“你好”", "我们走吧"],
     {"space": "他说“你好”我们走吧", "period_space": "他说“你好”。我们走吧"}),
    (["你好。", "“我们走吧。”"],
     {"space": "你好。“我们走吧。”", "period_space": "你好。“我们走吧。”"}),
    (["我想想……", "‘好’"], {"space": "我想想……‘好’", "period_space": "我想想……‘好’"}),
    (["好——", "我们"], {"space": "好——我们", "period_space": "好——。我们"}),
    (["Hallo“", "“你好"], {"space": "Hallo“ “你好", "period_space": "Hallo“. “你好"}),
    (["我用", "Python"], {"space": "我用 Python", "period_space": "我用。 Python"}),
    (["Python", "很好"], {"space": "Python 很好", "period_space": "Python. 很好"}),
    (["今天", "Hallo"], {"space": "今天 Hallo", "period_space": "今天。 Hallo"}),
    (["Hallo Welt", "今天"], {"space": "Hallo Welt 今天", "period_space": "Hallo Welt. 今天"}),
    (["สวัสดี", "ครับ"], {"space": "สวัสดี ครับ", "period_space": "สวัสดี. ครับ"}),
    (["Hallo Welt", "Wie geht's"],
     {"space": "Hallo Welt Wie geht's", "period_space": "Hallo Welt. Wie geht's"}),
])
def test_merge_preview_join_of_unspaced_members_matches_the_server(
        page, tmp_path, finals, expected, strategy):
    """The JS twin joined every member with " " and closed it with ".", so
    the merge preview of zh / ja members showed "今天天气很好. 我们…" while
    the server joins Han / kana with no space and closes them with "。"."""
    from faster_whisper_backend.captures.samples import _build_default_transcript

    fns = _js_functions(page, *_TRANSCRIPT_FNS)
    rows = [{"final": f} for f in finals]
    out = _run_node(tmp_path, fns + "\nconsole.log(JSON.stringify("
                    "_buildDefaultTranscript(" + json.dumps(rows)
                    + ", " + json.dumps(strategy) + ")));")
    assert out == expected[strategy]
    assert out == _build_default_transcript(rows, strategy)


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
    fns = _js_functions(page, "applyServerSample", "_applyChipsToText", "_chipFields")
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


@needs_node
def test_capture_save_and_reprocess_redraw_the_card_preview(page, tmp_path):
    """The collapsed card's one-line preview was painted once by renderCard:
    after a chip Save or a Reprocess it kept the old text until the next
    full render(). The PATCH reply's raw words also replaced the row's
    aligned ones, which mis-placed the chip in the next preview."""
    fns = _js_functions(page, "onSave", "onReprocess", "_syncCapturePreview", "_capturePreviewText",
                        "_applyChipsToText")
    out = _run_node(tmp_path, fns + r"""
var preview = { textContent: 'die Katze die Maus' }, asked = [];
global.document = { querySelector: function(s) { asked.push(s); return preview; } };
var replies = [];
async function api() { return replies.shift(); }
function clearDirty() {} function toast() {} function reloadCounts() {}
function applyCorrectionsToGround() {}
var chip = """ + json.dumps(_KATZE_CHIP) + r""";
var aligned = """ + json.dumps(_KATZE_ALIGNED) + r""";
var r = { id: 'c1', text_for_training: """ + json.dumps(_KATZE_TEXT) + r""",
  corrections: [], words: aligned };
var state = { cid: 'c1', corrections: [chip], baselineCorrections: [],
  adminNotes: '', words: aligned };
(async function() {
  replies.push({ capture: { corrections: [chip], words: """ + json.dumps(_KATZE_RAW) + r""" } });
  await onSave(state, r);
  var saved = preview.textContent, kept = r.words === aligned;
  replies.push({ capture: { text_for_training: 'die Hund die Maus' }, changed: [] });
  await onReprocess(state, r, { textContent: 'Reprocess' });
  console.log(JSON.stringify({ saved: saved, kept: kept,
    reprocessed: preview.textContent, asked: asked[0] }));
})();
""")
    assert out == {"saved": "Die Katze die Maus", "kept": True,
                   "reprocessed": "Die Hund die Maus",
                   "asked": '.capture-card[data-id="c1"] .cc-preview'}
    assert "preview.className = 'cc-preview';" in _js_function(page, "renderCard")


@needs_node
@pytest.mark.parametrize("batch", [True, False])
def test_proposals_refresh_stops_the_preview_and_redraws_the_batch_card(page, tmp_path, batch):
    """Refresh replaced _proposals but left the batch card on the proposal
    shown before, so Accept (which takes _proposals[0]) merged one nobody
    saw, often one just dismissed; and the ▶ audio of a card the redraw
    detached played on with no control left to stop it."""
    out = _run_node(tmp_path, _js_function(page, "_loadProposals") + r"""
var _filtSpeakers = [], _batchActive = """ + json.dumps(batch) + r""";
var _batchUndo = { action: 'dismiss', proposals: [] }, _batchBusy = false;
var _batchLoadHold = false, _proposalsSeq = 0;
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


@needs_node
def test_list_accept_stops_a_preview_started_during_the_merge(page, tmp_path):
    """Accept stopped the preview before its merge POST but not before the
    redraw after it: a ▶ clicked on another card meanwhile played on from a
    detached card with no control left to pause it."""
    out = _run_node(tmp_path, _js_function(page, "_acceptProposal") + r"""
var events = [], pending = [];
var _allCaptures = [{ id: 'a' }, { id: 'b' }];
var p = { member_ids: ['a'] }, _proposals = [p, { member_ids: ['b'] }];
function api() { events.push('post'); return new Promise(function(r) { pending.push(r); }); }
function _stopAnyPreview() { events.push('stop'); }
function _redrawProposalList() { events.push('redraw'); }
function _loadUnlessDirty() {} function toast() {}
(async function() {
  var done = _acceptProposal(p, null);
  events.push('play');                 // ▶ on the other card mid-merge
  pending.shift()({});
  await done;
  console.log(JSON.stringify(events));
})();
""")
    assert out == ["stop", "post", "play", "stop", "redraw"]


@needs_node
def test_manual_merge_create_stops_a_preview_started_during_the_post(page, tmp_path):
    """The merge modal's Create stopped the preview only before its POST:
    a ▶ clicked while the sample was built played on after the modal hid,
    with no control left to pause it."""
    start = page.index("async function()", page.index("getElementById('merge-commit').onclick"))
    out = _run_node(tmp_path, "var commit = " + _js_block(page, start) + r""";
var events = [], pending = [];
var rows = [{ id: 'a' }, { id: 'b' }];
var modal = { classList: { remove: function() { events.push('hide'); } } };
async function _okToDropOpenEdits() { return true; }
function _dirtyOpenEdits() { return []; }
function api() { events.push('post'); return new Promise(function(r) { pending.push(r); }); }
function _stopAnyPreview() { events.push('stop'); }
function load() { events.push('load'); } function toast() {}
(async function() {
  await commit();
  events.push('play');                 // ▶ while "Creating" is in flight
  pending.shift()({});
  await tick();
  console.log(JSON.stringify(events));
})();
""")
    assert out == ["stop", "post", "play", "stop", "hide", "load"]


# The "ask before dropping open edits" helpers, sliced together.
_OPEN_EDITS_FNS = ("_freshOpenEdits", "_okToDropOpenEdits", "_anyOpenDirty", "_dirtyOpenEdits")

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
    fns = _js_functions(page, "_bulkStatus", "_undoBulk", "_bulkDelete",
                        "_renderUnlessKeptOpenEdits", *_OPEN_EDITS_FNS, "_plural")
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


@needs_node
def test_group_status_click_during_save_survives_the_save_reply(page, tmp_path):
    """The group Save reply carried the status from before a status click
    made while it ran, and applyServerSample painted it back over the
    buttons, the header pill and the list entry while the server held the
    clicked status."""
    anchor = "async function(v) {\n            var prev = sampleState.newStatus;"
    on_status = _js_block(page, page.index(anchor))
    on_save = _js_after(page, "saveTBtn.onclick = ")
    fns = _js_functions(page, "applyServerSample", "_applyChipsToText", "_chipFields")
    out = _run_node(tmp_path, _SAMPLE_CARD_STUBS + fns + r"""
var g = { id: 'g1' }, _allSamples = [{ id: 'g1', status: 'new' }];
var sampleState = { newStatus: 'new', corrections: [], adminNotes: '' };
var saves = [];
function api(m, u, body) {
  if (body && body.status) return Promise.resolve({});
  return new Promise(function(r) { saves.push(r); });
}
var onStatus = """ + on_status + r""";
var onSave = """ + on_save + r""";
(async function() {
  onSave();
  await onStatus('ready');              // clicked while the Save runs
  saves[0]({ sample: { id: 'g1', status: 'new' } });
  await tick();
  var raced = { shown: statusShown[statusShown.length - 1], state: sampleState.newStatus,
    list: _allSamples[0].status, head: headSynced[headSynced.length - 1] };
  onSave();                             // no click this time: server wins
  saves[1]({ sample: { id: 'g1', status: 'dismissed' } });
  await tick();
  console.log(JSON.stringify({ raced: raced, plain: sampleState.newStatus }));
})();
""")
    assert out == {"raced": {"shown": "ready", "state": "ready", "list": "ready",
                             "head": "ready"},
                   "plain": "dismissed"}


def test_action_bar_sticks_below_the_sticky_header(page):
    """The selection bar stuck at top: 0 like the shared header (z-index
    10), so once scrolled it slid under the header and hid the bulk actions
    and the duration meter."""
    bar = re.search(r"\n  #action-bar \{[^}]*\}", page).group(0)
    assert "top: var(--header-h, 0px);" in bar
    assert _z_index(page, "#action-bar") < 10
    assert ("document.documentElement.style.setProperty('--header-h', "
            "_hdr.offsetHeight + 'px');") in page
    assert "new ResizeObserver(_setHeaderH).observe(_hdr);" in page


@needs_node
def test_capture_status_click_keeps_the_aligned_words(page, tmp_path):
    """The status auto-save copied the PATCH reply's raw words onto the row,
    so the next collapsed preview and the merge modal put a chip after
    removed fillers on another copy of its word."""
    build = _js_function(page, "buildBody")
    handler = _js_block(build, build.index("async function(v) {\n        // Auto-save on click."))
    fns = _js_functions(page, "_syncCapturePreview", "_capturePreviewText", "_applyChipsToText")
    out = _run_node(tmp_path, fns + r"""
var preview = { textContent: '' };
global.document = { querySelector: function() { return preview; } };
var body = { parentNode: null };
var chip = """ + json.dumps(_KATZE_CHIP) + r""";
var aligned = """ + json.dumps(_KATZE_ALIGNED) + r""";
var r = { id: 'c1', status: 'new', text_for_training: """ + json.dumps(_KATZE_TEXT) + r""",
  corrections: [chip], words: aligned };
var state = { cid: 'c1', newStatus: 'new', words: aligned };
var statusGrp = { setValue: function() {} };
async function api() {
  return { capture: { status: 'reviewed', corrections: [chip],
                      words: """ + json.dumps(_KATZE_RAW) + r""" } };
}
function reloadCounts() {} function toast() {}
var onStatus = """ + handler + r""";
(async function() {
  await onStatus('reviewed');
  console.log(JSON.stringify({ kept: r.words === aligned, status: r.status,
    preview: preview.textContent,
    merge: _applyChipsToText(r.text_for_training, r.corrections, r.words) }));
})();
""")
    assert out == {"kept": True, "status": "reviewed",
                   "preview": "Die Katze die Maus", "merge": "Die Katze die Maus"}


@needs_node
@pytest.mark.parametrize("action, reply, confirms", [
    ("_bulkStatus('reviewed')", {"updated": [{"id": "a", "prev_status": "new"}], "skipped": []},
     ["Mark 1 reviewed?"]),
    ("_undoBulk([{ id: 'a', prev_status: 'reviewed' }])", {"updated": [{"id": "a"}], "skipped": []},
     ["Undo the bulk change?"]),
    ("_bulkDelete()", {"deleted": [], "skipped": []}, ["Delete 1 selected?", "Delete 1 capture?"]),
])
def test_bulk_action_asks_once_about_edits_already_agreed_to_drop(
        page, tmp_path, action, reply, confirms):
    """The re-ask after the request fired for any dirty card, including the
    one the user had just agreed to drop: a second (for delete a third)
    prompt about the same edits, and a Cancel there left a stale list."""
    fns = _js_functions(page, "_bulkStatus", "_undoBulk", "_bulkDelete",
                        "_renderUnlessKeptOpenEdits", *_OPEN_EDITS_FNS, "markDirty", "_plural")
    out = _run_node(tmp_path, fns + _BULK_DIRTY_STUBS + r"""
var reply = """ + json.dumps(reply) + r""";
async function run(editMeanwhile) {
  var st = { dirty: false };
  markDirty(st);
  _openRows = { c1: st };               // dirty before the action
  answers = [true, true, true];
  var p = """ + action + r""";
  await tick();
  if (editMeanwhile) markDirty(st);     // edited again mid-request
  pending.shift()(reply);
  await p; await tick();
}
(async function() {
  await run(false);
  var same = { renders: renders, asked: asked.slice() };
  asked = []; renders = 0;
  await run(true);
  console.log(JSON.stringify({ same: same, newer: { renders: renders, asked: asked } }));
})();
""")
    assert out["same"] == {"renders": 1, "asked": confirms}
    assert out["newer"] == {"renders": 1, "asked": confirms + ["Show the updated list?"]}


@needs_node
def test_load_more_groups_asks_once_about_edits_already_agreed_to_drop(page, tmp_path):
    """loadMoreSamples re-asked "Show the loaded groups?" about the very
    edits "Load more groups?" had covered; a Cancel there hid the groups the
    user asked for. Edits made while the page loaded still get the re-ask,
    and a page superseded by a filter change still never renders."""
    fns = _js_functions(page, "loadMoreSamples", "_renderUnlessKeptOpenEdits", *_OPEN_EDITS_FNS,
                        "markDirty")
    out = _run_node(tmp_path, fns + _BULK_DIRTY_STUBS + r"""
var _samplesNext, _samplesLoading = false, _loadSeq = 0, _allSamples = [];
function _speakerQs() { return ''; }
global.document.getElementById = function() { return null; };
async function run(editMeanwhile, newFilter) {
  _samplesNext = { before_ts: 1, before_id: 'x' };
  var st = { dirty: false };
  markDirty(st);
  _openRows = { c1: st };
  answers = [true, true];
  var p = loadMoreSamples();
  await tick();
  if (editMeanwhile) markDirty(st);
  if (newFilter) _loadSeq++;
  pending.shift()({ samples: [], next: null });
  await p; await tick();
  var res = { renders: renders, asked: asked };
  renders = 0; asked = [];
  return res;
}
(async function() {
  console.log(JSON.stringify([await run(false, false), await run(true, false),
                              await run(true, true)]));
})();
""")
    first = "Load more groups?"
    assert out == [{"renders": 1, "asked": [first]},
                   {"renders": 1, "asked": [first, "Show the loaded groups?"]},
                   {"renders": 0, "asked": [first]}]


@needs_node
def test_load_asks_before_rendering_over_edits_made_meanwhile(page, tmp_path):
    """Refresh, a filter change or a finished job asked about open edits
    only before load() started: a card opened and edited while its two GETs
    ran lost its chips / notes to the render() after them, with no prompt.
    On a Cancel the list on screen (and its group cursor) stays as it was;
    edits the caller already agreed to drop are not asked about again."""
    fns = _js_functions(page, "load", *_OPEN_EDITS_FNS, "markDirty")
    out = _run_node(tmp_path, fns + r"""
var cursor = { before_ts: 100, before_id: 'a1' };
var _samplesNext, _samplesNextQs = '', _loadSeq = 0, _allSamples, _allCaptures;
var _counts = {}, _totalCount = 0, _me = null, _isAdmin = true, _filtSpeakers = [];
var _pendingModel = null, _speakerPicker = null, _openRows = {}, _openSamples = {};
var renders = 0, asked = [], answers = [], toasts = [], pending = [];
async function _confirm(o) { asked.push(o.title); return answers.shift(); }
function api() { return new Promise(function(r) { pending.push(r); }); }
function _listUrl() { return '/captures/api/list'; }
function _speakerQs() { return ''; }
function updateCaptureBadge() {} function rebuildModelFilter() {} function updateCounts() {}
function _clearSelection() {} function render() { renders++; } function _syncUrl() {}
function reloadStats() {} function toast(m) { toasts.push(m); }
var footer = { hidden: true };
global.document = { getElementById: function() { return footer; },
                    body: { classList: { add: function() {} } } };
async function run(dirtyBefore, editMeanwhile, answer) {
  _samplesNext = cursor; _allSamples = [{ id: 'g0' }]; _allCaptures = [{ id: 'c0' }];
  var st = { dirty: false };
  if (dirtyBefore) markDirty(st);
  _openRows = { c1: st };
  answers = [answer];
  var p = load();
  await tick();
  if (editMeanwhile) markDirty(st);
  pending.shift()({ captures: [{ id: 'c9' }], is_admin: true });
  await tick();
  pending.shift()({ samples: [{ id: 'g9' }], next: null });
  await p;
  var res = { renders: renders, asked: asked, toasts: toasts,
    captures: _allCaptures.map(function(r) { return r.id; }),
    samples: _allSamples.map(function(g) { return g.id; }),
    next: _samplesNext, footerHidden: footer.hidden };
  renders = 0; asked = []; toasts = []; footer.hidden = true;
  return res;
}
(async function() {
  console.log(JSON.stringify([await run(false, false), await run(false, true, true),
                              await run(false, true, false), await run(true, false)]));
})();
""")
    new = {"renders": 1, "toasts": [], "captures": ["c9"], "samples": ["g9"],
           "next": None, "footerHidden": True}
    ask = "Show the refreshed list?"
    assert out == [
        dict(new, asked=[]),
        dict(new, asked=[ask]),
        {"renders": 0, "asked": [ask],
         "toasts": ["List not refreshed: an open card has unsaved edits. "
                    "Refresh after saving."],
         "captures": ["c0"], "samples": ["g0"],
         "next": {"before_ts": 100, "before_id": "a1"}, "footerHidden": False},
        dict(new, asked=[])]


@needs_node
def test_cancelled_refresh_after_a_speaker_switch_keeps_the_shown_speakers(
        page, tmp_path):
    """The picker sets the new speakers and the URL before load() runs: a
    Cancel on "Show the refreshed list?" kept Alice's list under Bob's
    picker and URL, so the next render hid Alice's groups and the counts
    and Load more went by Bob."""
    fns = _js_functions(page, "load", "_speakerQs", *_OPEN_EDITS_FNS, "markDirty")
    out = _run_node(tmp_path, fns + r"""
var cursor = { before_ts: 100, before_id: 'a1' };
var _samplesNext = cursor, _samplesNextQs = '&user_id=alice', _loadSeq = 0;
var _allSamples = [{ id: 'alice0' }], _allCaptures = [{ id: 'c0' }];
var _counts = {}, _totalCount = 0, _me = null, _isAdmin = true;
var _filtSpeakers = ['bob'], _shownSpeakers = ['alice'];   // picker: Alice -> Bob
var _pendingModel = null, _openRows = {}, _openSamples = {};
var picked = [], urls = [], pending = [];
var _speakerPicker = { setPicked: function(ids) { picked.push(ids.slice()); } };
async function _confirm() { return false; }
function api() { return new Promise(function(r) { pending.push(r); }); }
function _listUrl() { return '/captures/api/list'; }
function _syncUrl() { urls.push(_filtSpeakers.join(',')); }
function render() {} function toast() {}
var footer = { hidden: true };
global.document = { getElementById: function() { return footer; } };
(async function() {
  var st = { dirty: false };
  _openRows = { c1: st };
  var p = load();
  await tick();
  markDirty(st);                          // a chip typed while the GETs run
  pending.shift()({ captures: [{ id: 'c9' }], is_admin: true });
  await tick();
  pending.shift()({ samples: [{ id: 'bob9' }], next: null });
  await p;
  console.log(JSON.stringify({ speakers: _filtSpeakers, picked: picked, urls: urls,
    samples: _allSamples.map(function(g) { return g.id; }),
    next: _samplesNext, footerHidden: footer.hidden }));
})();
""")
    assert out == {"speakers": ["alice"], "picked": [["alice"]], "urls": ["alice"],
                   "samples": ["alice0"],
                   "next": {"before_ts": 100, "before_id": "a1"}, "footerHidden": False}


@needs_node
@pytest.mark.parametrize("anchor", [
    "dissolveBtn.onclick = ",
    "document.getElementById('merge-commit').onclick = ",
])
def test_load_after_a_request_asks_about_edits_made_during_it(
        page, tmp_path, anchor):
    """Dissolve and the merge modal's Create await their DELETE / POST and
    only then call load(), which took the edits it may drop when it
    started: a chip typed during the request (a dissolve can wait seconds
    on a running rebuild) counted as agreed and the refreshed list dropped
    it with no prompt."""
    handler = _js_after(page, anchor)
    fns = _js_functions(page, "load", *_OPEN_EDITS_FNS, "markDirty")
    out = _run_node(tmp_path, fns + r"""
var _samplesNext = null, _samplesNextQs = '', _loadSeq = 0, _allSamples = [], _allCaptures = [];
var _counts = {}, _totalCount = 0, _me = null, _isAdmin = true, _filtSpeakers = [];
var _shownSpeakers = [], _pendingModel = null, _speakerPicker = null;
var _openRows = {}, _openSamples = {};
var renders = 0, asked = [], pending = [];
async function _confirm(o) { asked.push(o.title); return o.title !== 'Show the refreshed list?'; }
function api(m, u) {
  if (m !== 'GET') return new Promise(function(r) { pending.push(r); });
  return Promise.resolve({ captures: [], samples: [], is_admin: true });
}
function _listUrl() { return '/captures/api/list'; }
function _speakerQs() { return ''; }
function updateCaptureBadge() {} function rebuildModelFilter() {} function updateCounts() {}
function _clearSelection() {} function render() { renders++; } function _syncUrl() {}
function reloadStats() {} function toast() {} function _stopAnyPreview() {}
var g = { id: 'g1' }, rows = [{ id: 'c1' }], modal = { classList: { remove: function() {} } };
global.document = { getElementById: function() { return {}; },
                    body: { classList: { add: function() {} } } };
(async function() {
  var st = { dirty: false };
  _openRows = { c9: st };
  await (""" + handler + r""")();
  markDirty(st);                          // a chip typed during the request
  pending.shift()({});
  for (var i = 0; i < 5; i++) await tick();
  console.log(JSON.stringify({ asked: asked.slice(-1), renders: renders }));
})();
""")
    assert out == {"asked": ["Show the refreshed list?"], "renders": 0}


def _load_paging_harness(page):
    """load() + loadMoreSamples with the open-edits prompts answered yes and
    no speaker filter."""
    return _js_functions(page, "load", "loadMoreSamples", "_freshOpenEdits") + r"""
var _totalCount = 0, _me = null, _isAdmin = true, _filtSpeakers = [];
var _pendingModel = null, _speakerPicker = null;
function _listUrl() { return '/captures/api/list'; }
function _speakerQs() { return ''; }
async function _okToDropOpenEdits() { return true; }
function _dirtyOpenEdits() { return []; }
async function _renderUnlessKeptOpenEdits() {}
"""


@needs_node
def test_load_more_groups_during_a_load_never_pages_the_old_cursor(page, tmp_path):
    """load() bumps _loadSeq at once but replaced _samplesNext only after
    both its GETs: a Load more meanwhile paged the OLD filter's cursor under
    the NEW seq, spliced those groups into the new list and overwrote its
    cursor, so every group between the two cursors went missing."""
    out = _run_node(tmp_path, _load_paging_harness(page) + r"""
var _samplesNext = { before_ts: 100, before_id: 'a1' }, _samplesLoading = false;
var _loadSeq = 0, _allSamples = [{ id: 'old0' }], _allCaptures = [], _counts = {};
var urls = [], pending = {};
function api(m, u) { urls.push(u); return new Promise(function(res) { pending[u] = res; }); }
function updateCaptureBadge() {} function rebuildModelFilter() {} function updateCounts() {}
function _clearSelection() {} function render() {} function _syncUrl() {} function reloadStats() {}
function toast() {}
global.document = { getElementById: function() { return null; },
                    body: { classList: { add: function() {} } } };
(async function() {
  var pl = load();
  await tick();
  var pm = loadMoreSamples();             // footer click while load() runs
  await tick();
  pending['/captures/api/list']({ captures: [], is_admin: true });
  await tick();
  pending['/captures/api/samples?limit=200']({ samples: [{ id: 'g1' }],
                                              next: { before_ts: 500, before_id: 'g1' } });
  await pl;
  var stale = pending['/captures/api/samples?limit=200&before_ts=100&before_id=a1'];
  if (stale) stale({ samples: [{ id: 'old' }], next: { before_ts: 50 } });
  await pm;
  console.log(JSON.stringify({ urls: urls, samples: _allSamples.map(function(g) { return g.id; }),
                               next: _samplesNext }));
})();
""")
    assert out == {"urls": ["/captures/api/list", "/captures/api/samples?limit=200"],
                   "samples": ["g1"], "next": {"before_ts": 500, "before_id": "g1"}}


@needs_node
def test_failed_load_keeps_the_old_lists_load_more_working(page, tmp_path):
    """load() dropped the group cursor before its GETs and only set it
    again on success: after a failed refresh the old list's "Load more
    groups" button silently did nothing (and a Load more that finished
    meanwhile left it hidden)."""
    out = _run_node(tmp_path, _load_paging_harness(page) + r"""
var cursor = { before_ts: 100, before_id: 'a1' };
var _samplesNext = cursor, _samplesNextQs = '', _samplesLoading = false;
var _loadSeq = 0, _allSamples = [{ id: 'old0' }], _allCaptures = [], _counts = {};
var urls = [], toasts = [];
function api(m, u) {
  urls.push(u);
  return u === '/captures/api/list' ? Promise.reject(new Error('HTTP 502'))
                                    : Promise.resolve({ samples: [{ id: 'g2' }], next: null });
}
function toast(m) { toasts.push(m); }
var footer = { hidden: true };
global.document = { getElementById: function() { return footer; } };
(async function() {
  await load();
  var shown = !footer.hidden;
  await loadMoreSamples();
  console.log(JSON.stringify({ shown: shown, urls: urls, toasts: toasts,
                               samples: _allSamples.map(function(g) { return g.id; }) }));
})();
""")
    assert out == {"shown": True,
                   "urls": ["/captures/api/list",
                            "/captures/api/samples?limit=200&before_ts=100&before_id=a1"],
                   "toasts": ["Failed to load captures: HTTP 502"],
                   "samples": ["old0", "g2"]}


@needs_node
def test_failed_load_after_a_speaker_switch_drops_the_old_lists_cursor(page, tmp_path):
    """The picker sets the new speakers before load() runs: a failed load()
    gave the old list its cursor back, and Load more then paged Bob's
    cursor under the new filter, splicing everyone's older groups onto
    Bob's list."""
    out = _run_node(tmp_path, _load_paging_harness(page) + r"""
var _samplesNext = { before_ts: 100, before_id: 'a1' }, _samplesLoading = false;
var _samplesNextQs = '&user_id=bob';   // the picker then switched Bob -> anyone
var _loadSeq = 0, _allSamples = [{ id: 'bob0' }], _allCaptures = [], _counts = {};
var urls = [];
function api(m, u) {
  urls.push(u);
  return u === '/captures/api/list' ? Promise.reject(new Error('HTTP 502'))
                                    : Promise.resolve({ samples: [{ id: 'x' }], next: null });
}
function toast() {}
var footer = { hidden: false };
global.document = { getElementById: function() { return footer; } };
(async function() {
  await load();
  await loadMoreSamples();
  console.log(JSON.stringify({ hidden: footer.hidden, urls: urls,
                               samples: _allSamples.map(function(g) { return g.id; }) }));
})();
""")
    assert out == {"hidden": True, "urls": ["/captures/api/list"], "samples": ["bob0"]}


@needs_node
def test_header_collapse_asks_before_dropping_a_cards_unsaved_edits(page, tmp_path):
    """A header click collapsed an open capture or group card holding
    unsaved chips / notes with no prompt; the edits were gone."""
    fns = _js_functions(page, "_okToCollapse", "toggleExpand", "collapse", "_revokeRowBlob",
                        "_toggleSampleExpand")
    out = _run_node(tmp_path, fns + r"""
var asked = 0, answers = [];
async function _confirm() { asked++; return answers.shift(); }
function mkCard() {
  var open = true, body = { dataset: { built: '1' }, innerHTML: 'x' };
  return { classList: { contains: function(c) { return c === 'open' && open; },
                        remove: function() { open = false; }, add: function() { open = true; } },
           querySelector: function() { return body; }, isOpen: function() { return open; } };
}
var _openRows = {}, _openSamples = {};
(async function() {
  var res = {};
  var c = mkCard();
  _openRows.c1 = { dirty: true };
  answers = [false];
  await toggleExpand(c, { id: 'c1' });
  res.capKept = [c.isOpen(), !!_openRows.c1];
  answers = [true];
  await toggleExpand(c, { id: 'c1' });
  res.capDropped = [c.isOpen(), !!_openRows.c1];
  var c2 = mkCard();
  _openRows.c2 = { dirty: false };
  await toggleExpand(c2, { id: 'c2' });
  res.cleanClosed = c2.isOpen();
  var s = mkCard();
  _openSamples.g1 = { audio: null, state: { dirty: true } };
  answers = [false];
  await _toggleSampleExpand(s, { id: 'g1' });
  res.grpKept = [s.isOpen(), !!_openSamples.g1];
  answers = [true];
  await _toggleSampleExpand(s, { id: 'g1' });
  res.grpDropped = [s.isOpen(), !!_openSamples.g1];
  res.asked = asked;
  console.log(JSON.stringify(res));
})();
""")
    assert out == {"capKept": [True, True], "capDropped": [False, False],
                   "cleanClosed": False, "grpKept": [True, True],
                   "grpDropped": [False, False], "asked": 4}


@needs_node
def test_failed_reopen_after_render_drops_the_old_cards_state(page, tmp_path):
    """render() re-opens a rebuilt card while the old card's state is still
    in _openRows. When that GET failed the stale (dirty) state stayed:
    every refresh was skipped for "unsaved edits" no card showed, and its
    blob URL was never revoked. A first open that fails has no state."""
    fns = _js_functions(page, "toggleExpand", "collapse", "_revokeRowBlob")
    out = _run_node(tmp_path, fns + r"""
var revoked = [], paused = 0;
global.URL = { revokeObjectURL: function(u) { revoked.push(u); } };
function api() { return Promise.reject(new Error('HTTP 503')); }
function toast() {}
function mkCard() {
  var open = false, body = { dataset: { built: '0' }, innerHTML: '' };
  return { classList: { contains: function(c) { return c === 'open' && open; },
                        remove: function() { open = false; }, add: function() { open = true; } },
           querySelector: function() { return body; }, isOpen: function() { return open; } };
}
var _openRows = { c1: { dirty: true, blobUrl: 'blob:1',
                        audio: { pause: function() { paused++; } } } };
(async function() {
  var c = mkCard();
  await toggleExpand(c, { id: 'c1' });
  var c2 = mkCard();
  await toggleExpand(c2, { id: 'c2' });
  console.log(JSON.stringify({ open: [c.isOpen(), c2.isOpen()], rows: Object.keys(_openRows),
    revoked: revoked, paused: paused }));
})();
""")
    assert out == {"open": [False, False], "rows": [], "revoked": ["blob:1"], "paused": 1}


@needs_node
def test_superseded_reopen_leaves_the_live_cards_state(page, tmp_path):
    """Two quick renders re-open the same card twice. When the first
    (now detached) card's GET landed last, its buildBody replaced the live
    card's dirty state in _openRows with a clean one, so the unsaved-edits
    prompt no longer fired. A card collapsed while its GET ran stays shut."""
    fns = _js_functions(page, "toggleExpand", "collapse", "_revokeRowBlob")
    out = _run_node(tmp_path, fns + r"""
var pending = [];
function api() { return new Promise(function(res) { pending.push(res); }); }
function toast() {}
function buildBody(body, r) { _openRows[r.id] = { card: body.card, dirty: false }; }
function mkCard(name) {
  var open = false, body = { dataset: { built: '0' }, innerHTML: '', card: name };
  return { isConnected: true,
           classList: { contains: function(c) { return c === 'open' && open; },
                        remove: function() { open = false; }, add: function() { open = true; } },
           querySelector: function() { return body; } };
}
var _openRows = {};
(async function() {
  var a1 = mkCard('a1'), a2 = mkCard('a2');
  var p1 = toggleExpand(a1, { id: 'c1' });
  a1.isConnected = false;                 // render 2 rebuilt the list
  var p2 = toggleExpand(a2, { id: 'c1' });
  pending[1]({ capture: { id: 'c1' } });
  await p2;
  _openRows.c1.dirty = true;              // user edits the live card
  pending[0]({ capture: { id: 'c1' } });
  await p1;
  var b = mkCard('b');
  var p3 = toggleExpand(b, { id: 'c2' });
  b.classList.remove('open');             // collapsed while loading
  pending[2]({ capture: { id: 'c2' } });
  await p3;
  console.log(JSON.stringify({ c1: _openRows.c1, c2: !!_openRows.c2 }));
})();
""")
    assert out == {"c1": {"card": "a2", "dirty": True}, "c2": False}


@needs_node
def test_capture_status_click_during_save_or_reprocess_survives_the_reply(page, tmp_path):
    """The capture Save and Reprocess replies carry the row read before a
    status click (or chip Save) made while they ran, and copying it whole
    put the older status / chips back on the list entry, so the next
    render filed the card under the wrong status."""
    build = _js_function(page, "buildBody")
    handler = _js_block(build, build.index("async function(v) {\n        // Auto-save on click."))
    fns = _js_functions(page, "onSave", "onReprocess")
    out = _run_node(tmp_path, fns + r"""
var body = { parentNode: null }, slow = [];
var statusGrp = { setValue: function() {} };
function api(m, u, b) {
  if (b && b.status) return Promise.resolve({ capture: { status: b.status } });
  return new Promise(function(res) { slow.push(res); });
}
function clearDirty() {} function toast() {} function reloadCounts() {}
function _syncCapturePreview() {} function applyCorrectionsToGround() {}
var chip = { wrong: 'a', correct: 'A', idx: 1 };
var r = { id: 'c1', status: 'new', corrections: [] };
var state = { cid: 'c1', newStatus: 'new', corrections: [chip],
  baselineCorrections: [], adminNotes: '' };
var onStatus = """ + handler + r""";
(async function() {
  var p = onSave(state, r);
  await onStatus('ready');               // clicked while the Save runs
  slow.shift()({ capture: { status: 'new', corrections: [chip] } });
  await p;
  var saved = { status: r.status, corrections: r.corrections.length };
  p = onReprocess(state, r, { textContent: 'Reprocess' });
  await onStatus('reviewed');            // clicked while Reprocess runs
  slow.shift()({ capture: { status: 'new', corrections: [], final: 'f',
                            text_for_training: 't' }, changed: [] });
  await p;
  var reproc = { status: r.status, corrections: r.corrections.length,
    text: r.text_for_training };
  p = onSave(state, r);                  // no click this time: server wins
  slow.shift()({ capture: { status: 'dismissed', corrections: [chip] } });
  await p;
  console.log(JSON.stringify({ saved: saved, reproc: reproc, plain: r.status }));
})();
""")
    assert out == {"saved": {"status": "ready", "corrections": 1},
                   "reproc": {"status": "reviewed", "corrections": 1, "text": "t"},
                   "plain": "dismissed"}


@needs_node
def test_proposal_chip_save_updates_the_list_rows(page, tmp_path):
    """A chip typed in a proposal / merge preview auto-saved to the member
    captures, but their list rows kept the old chips: the collapsed cards
    and a reopened merge modal showed the uncorrected text until Refresh."""
    fns = _js_functions(page, "_saveProposalChips", "_capturePreviewText", "_applyChipsToText")
    chip = {"wrong": "die", "correct": "Die", "idx": 4}
    out = _run_node(tmp_path, fns + r"""
var previews = {};
global.window = {};
global.document = { querySelector: function(s) {
  var id = /data-id="([^"]+)"/.exec(s)[1];
  return previews[id] || (previews[id] = { textContent: '' });
} };
global.fetch = async function() {
  return { ok: true, json: async function() {
    return { saved: { m1: 1 }, members_corrections: { m1: [""" + json.dumps(chip) + r"""] } };
  } };
};
var aligned = """ + json.dumps(_KATZE_ALIGNED) + r""";
var _allCaptures = [
  { id: 'm1', text_for_training: """ + json.dumps(_KATZE_TEXT) + r""", corrections: [],
    words: aligned },
  { id: 'x', text_for_training: 'other', corrections: [] }];
var state = { corrections: [""" + json.dumps(chip) + r"""] };
_saveProposalChips(state, ['m1']).then(function() {
  console.log(JSON.stringify({ rows: _allCaptures.map(function(r) { return r.corrections; }),
    previews: Object.keys(previews).map(function(k) { return [k, previews[k].textContent]; }) }));
});
""")
    assert out == {"rows": [[chip], []], "previews": [["m1", "Die Katze die Maus"]]}


@needs_node
def test_proposal_chip_saves_run_one_at_a_time_with_the_latest_chips(page, tmp_path):
    """Each debounced save sends the full chip set and the server replaces
    the members' chips with it. A save fired while the previous one was in
    flight raced it, and the older set could land last: the latest edit
    was lost."""
    start = page.index("var saveTimer = null")
    block = _js_block(page, page.index("state.onMarkDirty = ", start))
    block = page[start:page.index(block, start) + len(block)]
    out = _run_node(tmp_path, r"""
var _proposalSaveChain = Promise.resolve(), _proposalSaveFlush = null;
var state = { corrections: ['foo'] }, ids = ['m1'], sent = [], pending = [];
function _saveProposalChips(st) {
  sent.push(st.corrections.slice());
  return new Promise(function(r) { pending.push(r); });
}
function toast() {}
function wait(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }
""" + block + r""";
(async function() {
  state.onMarkDirty();
  await wait(300);
  state.corrections = ['food'];
  state.onMarkDirty();
  await wait(300);
  var during = sent.length;
  pending.shift()();
  await wait(0);
  console.log(JSON.stringify({ during: during, sent: sent }));
})();
""")
    assert out == {"during": 1, "sent": [["foo"], ["food"]]}


@needs_node
def test_group_regenerate_asks_before_dropping_edits_made_while_it_ran(page, tmp_path):
    """Regenerate checked for unsaved edits only before its POST; a chip or
    note typed during the slow rebuild was replaced by the server copy and
    marked clean, with no prompt."""
    handler = _js_after(page, "regenBtn.onclick = ")
    out = _run_node(tmp_path, _js_function(page, "markDirty") + r"""
var sampleState = { dirty: false }, answers = [], asked = [], applied = 0, pending = [];
var g = { id: 'g1' }, regenBtn = {}, toasts = [];
async function _confirm(o) { asked.push(o.title); return answers.shift(); }
function api() { return new Promise(function(r) { pending.push(r); }); }
function applyServerSample() { applied++; sampleState.dirty = false; }
function toast(m) { toasts.push(m); } function reloadCounts() {}
""" + _js_functions(page, "okToRegenerate", "applyRegenerated")
        + "\nvar onRegen = " + handler + r""";
async function run(answer) {
  sampleState.dirty = false;
  await onRegen();
  markDirty(sampleState);               // chip typed during the rebuild
  answers = [answer];
  pending.shift()({ sample: {} });
  await tick(); await tick();
}
(async function() {
  await run(false);
  var kept = { applied: applied, dirty: sampleState.dirty };
  await run(true);
  console.log(JSON.stringify({ kept: kept, applied: applied, asked: asked,
    toasts: toasts }));
})();
""")
    assert out["kept"] == {"applied": 0, "dirty": True}
    assert out["applied"] == 1
    assert out["asked"] == ["Show the regenerated sample?"] * 2
    assert out["toasts"][-1] == "Regenerated"


@needs_node
def test_chip_on_a_word_after_a_bracket_rewrites_that_word(page, tmp_path):
    """A word glued to a "(" or „ counted as the next token, so a chip on
    it was placed on an earlier copy of the word (the server twin in
    captures/samples.py had the same miscount)."""
    out = _run_node(tmp_path, _js_function(page, "_applyChipsToText") + r"""
console.log(JSON.stringify([
  _applyChipsToText('ja die (die Sache', [{ wrong: 'die', correct: 'DER', idx: 2 }]),
  _applyChipsToText('ja die \u201edie Sache', [{ wrong: 'die', correct: 'DER', idx: 2 }]),
  _applyChipsToText('ja die (die Sache', [{ wrong: 'die', correct: 'DER', idx: 1 }])]));
""")
    assert out == ["ja die (DER Sache", "ja die \u201eDER Sache", "ja DER (die Sache"]


@needs_node
def test_blank_chip_correction_leaves_the_final_result_preview_alone(page, tmp_path):
    """Save drops a chip whose correction is only spaces (as does the
    server), but the Final-result preview applied it: "um" vanished from
    the preview while the export kept it."""
    fns = _js_functions(page, "_applyChipsToText", "_renderGroundSpans")
    out = _run_node(tmp_path, fns + r"""
global.document = { createElement: function() { return {}; } };
var spans = [];
var area = { appendChild: function(e) { spans.push(e.textContent); } };
var words = [{ word: ' so' }, { word: ' um' }, { word: ' yes' }];
_renderGroundSpans({ gtArea: area, words: words,
  corrections: [{ wrong: 'um', correct: '  ', idx: 1 },
                { wrong: 'yes', correct: ' yeah ', idx: 2 }] });
console.log(JSON.stringify([
  _applyChipsToText('so um yes', [{ wrong: 'um', correct: '  ', idx: 1 }]),
  _applyChipsToText('so um yes', [{ wrong: 'yes', correct: ' yeah ', idx: 2 }]),
  spans.join('')]));
""")
    assert out == ["so um yes", "so um yeah", "so um yeah"]


def _ground_spans_harness(page):
    """_renderGroundSpans on a stand-in area: `render(words, corrections)`
    returns the joined span text."""
    return _js_functions(page, "_cjkEdge", "_renderGroundSpans") + r"""
global.document = { createElement: function() { return {}; } };
function render(words, corrections) {
  var spans = [];
  var area = { appendChild: function(e) { spans.push(e.textContent); } };
  _renderGroundSpans({ gtArea: area, words: words, corrections: corrections || [] });
  return spans.join('');
}
"""


@needs_node
def test_final_result_preview_joins_unspaced_words_without_a_space(page, tmp_path):
    """Once a Chinese / Thai sentence is split among its words, the
    Final-result preview put a space between every two of them; the space
    between spaced words stays."""
    out = _run_node(tmp_path, _ground_spans_harness(page) + r"""
console.log(JSON.stringify([
  render([{ word: '我们', member_idx: 0 }, { word: '去', member_idx: 0 },
          { word: '北京。', member_idx: 0 }, { word: '你好', member_idx: 1 }]),
  render([{ word: 'สวัสดี' }, { word: 'ครับ' }],
         [{ wrong: 'ครับ', correct: 'ค่ะ', idx: 1 }]),
  render([{ word: ' so' }, { word: ' 好' }, { word: ' yes' }])]));
""")
    assert out == ["我们去北京。你好", "สวัสดีค่ะ", "so 好 yes"]


@needs_node
def test_final_result_preview_joins_zh_ja_members_like_the_server(page, tmp_path):
    """The server joins two zh / ja members with no space (also past the
    member's "。" / "」"), but the group card's Final result still put " "
    at every member boundary: it showed "我们去北京 你好" for a sample whose
    transcript and export read "我们去北京你好"."""
    from faster_whisper_backend.captures.samples import _build_default_transcript

    out = _run_node(tmp_path, _ground_spans_harness(page) + r"""
console.log(JSON.stringify([
  render([{ word: '我们', member_idx: 0 }, { word: '去', member_idx: 0 },
          { word: '北京', member_idx: 0 }, { word: '你好', member_idx: 1 }]),
  render([{ word: '他说', member_idx: 0 }, { word: '「好', member_idx: 0 },
          { word: '」', member_idx: 0 }, { word: '「是」', member_idx: 1 }]),
  render([{ word: '我们', member_idx: 0 }, { word: '北京', member_idx: 0 },
          { word: '你好', member_idx: 1 }],
         [{ wrong: '北京', correct: '上海', idx: 1 }]),
  render([{ word: '你好。', member_idx: 0 }, { word: ' Hallo', member_idx: 1 }]),
  render([{ word: 'สวัสดี', member_idx: 0 }, { word: 'ครับ', member_idx: 1 }])]));
""")
    assert out == ["我们去北京你好", "他说「好」「是」", "我们上海你好",
                   "你好。 Hallo", "สวัสดี ครับ"]
    # The same joins as the server's transcript (out[2] has a chip, which a
    # bare join of finals lacks).
    assert out[:2] + out[3:] == [
        _build_default_transcript([{"final": a}, {"final": b}], "space")
        for a, b in (("我们去北京", "你好"), ("他说「好」", "「是」"),
                     ("你好。", "Hallo"), ("สวัสดี", "ครับ"))]


@needs_node
def test_final_result_preview_reads_a_member_past_an_opener_word_of_its_own(page, tmp_path):
    """faster-whisper keeps "「" / "（" as a word of their own, and the Final
    result tested the next member only on that first word, so it showed
    "我们去北京。 「你好」" where the transcript and export read
    "我们去北京。「你好」"."""
    from faster_whisper_backend.captures.samples import _build_default_transcript

    out = _run_node(tmp_path, _ground_spans_harness(page) + r"""
function m(member, list) {
  return list.map(function(w) {
    return typeof w === 'string' ? { word: w, member_idx: member }
      : Object.assign({ member_idx: member }, w);
  });
}
console.log(JSON.stringify([
  render(m(0, ['我们', '去', '北京。']).concat(m(1, ['「', '你好', '」']))),
  render(m(0, ['你好。']).concat(m(1, ['（', '笑', '）']))),
  render(m(0, ['你好']).concat(m(1, [{ word: '嗯', removed: true }, '「', '是', '」'])))]));
""")
    assert out == ["我们去北京。「你好」", "你好。（笑）", "你好「是」"]
    assert out == [
        _build_default_transcript([{"final": a}, {"final": b}], "space")
        for a, b in (("我们去北京。", "「你好」"), ("你好。", "（笑）"),
                     ("你好", "「是」"))]


@needs_node
def test_karaoke_highlight_scrolls_only_the_word_strip(page, tmp_path):
    """The active word was brought on screen with scrollIntoView, which also
    scrolls the page and the merge modal: a playing card pulled the view
    back to itself on every word."""
    fns = _js_functions(page, "_scrollStripTo", "_setActiveWord")
    out = _run_node(tmp_path, fns + r"""
var calls = 0;
function word(top) {
  return { classList: { add: function() {}, remove: function() {} },
    scrollIntoView: function() { calls++; },
    getBoundingClientRect: function() {
      return { top: top - strip.scrollTop, bottom: top + 20 - strip.scrollTop };
    } };
}
// The strip shows content rows 0..100 when scrollTop is 0.
var strip = { scrollTop: 0, clientTop: 1, clientHeight: 100,
  getBoundingClientRect: function() { return { top: 500, bottom: 602 }; } };
var state = { stripEl: strip,
  wordEls: [word(510), word(560), word(650), word(505)] };
var tops = [];
[1, 2, 3, 0].forEach(function(i) {
  _setActiveWord(state, i);
  tops.push(strip.scrollTop);
});
console.log(JSON.stringify([calls, tops]));
""")
    # Visible: no scroll; below the strip: down to its bottom edge; above
    # the strip's top: back up to it; visible again: left alone.
    assert out == [0, [0, 69, 4, 4]]


@needs_node
def test_keyboard_cursor_scrolls_only_the_word_strip(page, tmp_path):
    """The cursor word was brought on screen with scrollIntoView, and a
    Save / Regenerate / Lock reply repaints the cursor: a user who had
    scrolled away from the card was pulled back to it."""
    fns = _js_functions(page, "_scrollStripTo", "_redrawCursor")
    out = _run_node(tmp_path, fns + r"""
var calls = 0;
function word(top) {
  return { classList: { add: function() {}, remove: function() {} },
    scrollIntoView: function() { calls++; },
    getBoundingClientRect: function() {
      return { top: top - strip.scrollTop, bottom: top + 20 - strip.scrollTop };
    } };
}
var strip = { scrollTop: 0, clientTop: 1, clientHeight: 100,
  getBoundingClientRect: function() { return { top: 500, bottom: 602 }; } };
var state = { stripEl: strip, cursorIdx: 1, wordEls: [word(510), word(650)] };
_redrawCursor(state);
console.log(JSON.stringify([calls, strip.scrollTop]));
""")
    assert out == [0, 69]


@needs_node
def test_keyboard_cursor_move_scrolls_the_page_to_the_cursor(page, tmp_path):
    """Only a repaint must leave the page alone: an arrow key moves the
    cursor (the strip's key handlers preventDefault it), so with the strip
    partly below the window the cursor moved off screen and the page did
    not follow."""
    fns = _js_functions(page, "_scrollStripTo", "_redrawCursor", "_moveCursor")
    out = _run_node(tmp_path, fns + r"""
var calls = [];
function word(i, top) {
  return { classList: { add: function() {}, remove: function() {} },
    scrollIntoView: function(o) { calls.push([i, o.block, o.inline]); },
    getBoundingClientRect: function() {
      return { top: top - strip.scrollTop, bottom: top + 20 - strip.scrollTop };
    } };
}
var strip = { scrollTop: 0, clientTop: 1, clientHeight: 100,
  getBoundingClientRect: function() { return { top: 500, bottom: 602 }; } };
var state = { stripEl: strip, cursorIdx: 0,
  wordEls: [word(0, 510), word(1, 560), word(2, 650)] };
_moveCursor(state, 1);
_moveCursor(state, 2);
console.log(JSON.stringify([calls, strip.scrollTop]));
""")
    assert out == [[[1, "nearest", "nearest"], [2, "nearest", "nearest"]], 69]


@needs_node
def test_first_focus_of_the_strip_leaves_the_page_alone(page, tmp_path):
    """A mouse click focuses the strip on mousedown, and the first focus put
    the cursor on word 0 with a page scroll: with the strip's top above the
    window the page jumped before mouseup, and the click on a word further
    down was lost."""
    fns = _js_functions(page, "_scrollStripTo", "_redrawCursor", "_moveCursor",
                        "_bindStripKeyboard")
    out = _run_node(tmp_path, fns + r"""
var calls = 0, handlers = {};
function word() {
  return { classList: { add: function() {}, remove: function() {} },
    scrollIntoView: function() { calls++; },
    getBoundingClientRect: function() { return { top: 510, bottom: 530 }; } };
}
var strip = { scrollTop: 0, clientTop: 1, clientHeight: 100, parentNode: null,
  setAttribute: function() {},
  classList: { add: function() {}, remove: function() {} },
  addEventListener: function(t, h) { handlers[t] = h; },
  getBoundingClientRect: function() { return { top: 500, bottom: 602 }; } };
global.document = { createElement: function() {
  return { classList: { add: function() {}, remove: function() {} } }; } };
var state = { stripEl: strip, words: ['a', 'b'], cursorIdx: -1,
  wordEls: [word(), word()] };
_bindStripKeyboard(state);
handlers.focus();
console.log(JSON.stringify([state.cursorIdx, calls]));
""")
    assert out == [0, 0]


def _strip_focus_harness(page):
    """_bindStripKeyboard on a strip scrolled down by 200px (playback or the
    wheel), so word 0 sits above its visible top; `handlers` holds the
    strip's listeners."""
    fns = _js_functions(page, "_scrollStripTo", "_redrawCursor", "_moveCursor",
                        "_bindStripKeyboard")
    return fns + r"""
var handlers = {};
function word(top) {
  return { classList: { add: function() {}, remove: function() {} },
    scrollIntoView: function() {},
    getBoundingClientRect: function() {
      return { top: top - strip.scrollTop, bottom: top + 20 - strip.scrollTop };
    } };
}
var strip = { scrollTop: 200, clientTop: 1, clientHeight: 100, parentNode: null,
  setAttribute: function() {},
  classList: { add: function() {}, remove: function() {} },
  addEventListener: function(t, h) { handlers[t] = h; },
  getBoundingClientRect: function() { return { top: 500, bottom: 602 }; } };
global.document = { createElement: function() {
  return { classList: { add: function() {}, remove: function() {} } }; } };
var state = { stripEl: strip, words: ['a', 'b'], cursorIdx: -1,
  wordEls: [word(510), word(720)] };
_bindStripKeyboard(state);
"""


@needs_node
def test_mouse_first_focus_leaves_a_scrolled_strip_alone(page, tmp_path):
    """The first focus scrolled the strip itself to word 0: with the strip
    scrolled down (playback follows the playing word), a click on a visible
    word pulled the strip to its top on mousedown, mouseup landed on
    another word and the click went to the strip, so no chip opened."""
    out = _run_node(tmp_path, _strip_focus_harness(page) + r"""
if (handlers.mousedown) handlers.mousedown();
handlers.focus();
console.log(JSON.stringify([state.cursorIdx, strip.scrollTop]));
""")
    assert out == [0, 200]


@needs_node
def test_keyboard_first_focus_scrolls_the_strip_to_the_cursor(page, tmp_path):
    """Tab into the strip (no mousedown, or one whose focus came long
    after): the cursor's word 0 is brought into the strip's view."""
    out = _run_node(tmp_path, _strip_focus_harness(page) + r"""
var res = [];
handlers.focus();
res.push(strip.scrollTop);
state.cursorIdx = -1;
strip.scrollTop = 200;
handlers.mousedown();   // a click on an already focused strip: no focus event
setTimeout(function() {
  handlers.focus();
  res.push(strip.scrollTop);
  console.log(JSON.stringify(res));
}, 5);
""")
    # Word 0 (content row 510) back at the strip's inner top (501).
    assert out == [9, 9]
