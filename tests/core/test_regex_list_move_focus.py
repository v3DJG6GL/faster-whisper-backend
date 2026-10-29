"""regex-list ↑/↓ stepper: _rlMove reinserted the row that holds the pressed
button, and moving a focused node blurs it, so a keyboard user's focus fell
to <body> after one step and a second Space did nothing. The neighbour is
moved instead; the order is the same and the focused row stays put.
A move onto an end-stop disables the pressed button, which blurs it too, so
focus is handed to the row's other arrow.

The real _makeEntryRow / _readEntries / _renumberEntries run under node
against a minimal DOM stub whose node moves and disabled setter blur a
focused node like a browser does."""

import json
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common


_DOM_STUB = r"""
function detach(c) {
  const p = c.parentNode;
  if (!p) return;
  if (c.contains(document.activeElement)) document.activeElement = document.body;
  p.children.splice(p.children.indexOf(c), 1);
  c.parentNode = null;
}
function matches(e, sel) {
  return (' ' + (e.className || '') + ' ').indexOf(' ' + sel.slice(1) + ' ') >= 0;
}
function all(e, sel, out) {
  e.children.forEach((c) => { if (matches(c, sel)) out.push(c); all(c, sel, out); });
  return out;
}
function el(tag) {
  const e = {
    tag, children: [], parentNode: null, listeners: {}, attrs: {}, style: {},
    className: '', value: '', textContent: '', innerHTML: '', _disabled: false,
    get disabled() { return this._disabled; },
    set disabled(v) {   // focus fixup: a focused control that turns disabled blurs
      this._disabled = !!v;
      if (v && document.activeElement === this) document.activeElement = document.body;
    },
    classList: {
      s: new Set(),
      add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
      toggle(c, on) { on ? this.s.add(c) : this.s.delete(c); },
      contains(c) { return this.s.has(c); },
    },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    fire(t) { (this.listeners[t] || []).forEach(f => f({})); },
    focus() { document.activeElement = this; },
    contains(n) { for (; n; n = n.parentNode) if (n === this) return true; return false; },
    appendChild(c) { return this.insertBefore(c, null); },
    insertBefore(c, ref) {
      detach(c);
      const i = ref ? this.children.indexOf(ref) : this.children.length;
      this.children.splice(i, 0, c); c.parentNode = this; return c;
    },
    remove() { detach(this); },
    get previousElementSibling() {
      const p = this.parentNode; return p ? p.children[p.children.indexOf(this) - 1] || null : null;
    },
    get nextElementSibling() {
      const p = this.parentNode; return p ? p.children[p.children.indexOf(this) + 1] || null : null;
    },
    querySelectorAll(sel) { return all(this, sel, []); },
    querySelector(sel) { return all(this, sel, [])[0] || null; },
  };
  return e;
}
global.document = { createElement: el, body: el('body'), activeElement: null };
document.activeElement = document.body;
"""


def _editor_fns() -> str:
    js = web_common.RULE_EDITOR_JS
    esc = js[js.index("function _esc(s) {"):js.index("const _PIPELINE_TYPES")]
    enter = js[js.index("function _bindEnterCommit("):
               js.index("function _makeMonoLabeledInput(")]
    rows = js[js.index("const MOVE_CHEV_UP"):js.index("function renderTypeEditor(")]
    return esc + enter + rows


def _run(drive: str) -> dict:
    out = subprocess.run(["node", "-"], input=_DOM_STUB + _editor_fns() + drive,
                         capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


_SETUP = r"""
const rule = { entries: [] };
let commits = 0;
const box = document.body.appendChild(el('div'));
const rows = (PATTERNS).map((p) =>
  box.appendChild(_makeEntryRow(rule, { pattern: p }, () => commits++, null)));
_renumberEntries(box);
const btn = (r, dir) => r.querySelector('.rl-move-' + dir);
const order = () => rule.entries.map((e) => e.pattern).join('');
const focusOn = (r) => document.activeElement === btn(r, 'up') ? 'up'
  : document.activeElement === btn(r, 'down') ? 'down' : 'lost';
const press = (r, dir) => { btn(r, dir).fire('click');
  return { order: order(), focus: focusOn(r) }; };
"""


def _setup(patterns: str) -> str:
    return _SETUP.replace("PATTERNS", json.dumps(list(patterns)))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_move_up_keeps_focus_on_the_pressed_button():
    res = _run(_setup("abc") + r"""
btn(rows[2], 'up').focus();
const first = press(rows[2], 'up');
const second = press(rows[2], 'up');   // reaches the top: ↑ is now disabled
console.log(JSON.stringify({ first, second,
  top: btn(rows[2], 'up').disabled, commits }));
""")
    assert res == {"first": {"order": "acb", "focus": "up"},
                   "second": {"order": "cab", "focus": "down"},
                   "top": True, "commits": 2}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_move_down_keeps_focus_on_the_pressed_button():
    res = _run(_setup("abc") + r"""
btn(rows[0], 'down').focus();
const first = press(rows[0], 'down');
const second = press(rows[0], 'down');
console.log(JSON.stringify({ first, second,
  last: btn(rows[0], 'down').disabled, ord: rows[0].querySelector('.rl-ord').textContent }));
""")
    assert res == {"first": {"order": "bac", "focus": "down"},
                   "second": {"order": "bca", "focus": "up"},
                   "last": True, "ord": "3"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_end_stop_move_in_a_two_entry_list_hands_focus_to_the_other_arrow():
    # every move in a 2-entry list lands on an end-stop
    res = _run(_setup("ab") + r"""
btn(rows[1], 'up').focus();
const up = press(rows[1], 'up');
const down = press(rows[1], 'down');
console.log(JSON.stringify({ up, down }));
""")
    assert res == {"up": {"order": "ba", "focus": "down"},
                   "down": {"order": "ab", "focus": "up"}}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_end_stop_move_without_focus_leaves_focus_alone():
    # a mouse click that never focused the button must not pull focus to the row
    res = _run(_setup("ab") + r"""
const up = press(rows[1], 'up');
console.log(JSON.stringify({ up, body: document.activeElement === document.body }));
""")
    assert res == {"up": {"order": "ba", "focus": "lost"}, "body": True}


# Drag-reorder: dragover moves the row live, so dragend decides whether the
# new order sticks. A cancelled drag (Esc, or a release outside the list:
# dropEffect 'none') used to keep the moved order and mark the card dirty.
_DRAG = r"""
let _rlDragSrc = null;
const DRAG_AUTOSCROLL = { begin() {}, move() {}, stop() {} };
const grip = (r) => r.querySelector('.rl-grip');
const dragTo = (r, idx, dropEffect) => {
  const dt = { setData() {}, dropEffect: 'none' };
  grip(r).listeners.dragstart.forEach((f) => f({ dataTransfer: dt }));
  box.insertBefore(r, box.children.filter((c) => c !== r)[idx] || null);  // dragover
  dt.dropEffect = dropEffect;
  grip(r).listeners.dragend.forEach((f) => f({ dataTransfer: dt }));
  return { dom: box.children.map((c) => rows.indexOf(c)).join(''),
    ords: box.children.map((c) => c.querySelector('.rl-ord').textContent).join(''),
    order: order(), commits };
};
rule.entries = rows.map((_, i) => ({ pattern: 'abcd'[i] }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("start", [0, 3])
def test_cancelled_drag_puts_the_row_back_and_commits_nothing(start):
    res = _run(_setup("abcd") + _DRAG + f"""
console.log(JSON.stringify(dragTo(rows[{start}], {3 - start}, 'none')));
""")
    assert res == {"dom": "0123", "ords": "1234", "order": "abcd", "commits": 0}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_real_drag_drop_keeps_the_new_order():
    res = _run(_setup("abcd") + _DRAG + r"""
console.log(JSON.stringify(dragTo(rows[0], 3, 'move')));
""")
    assert res == {"dom": "1230", "ords": "1234", "order": "bcda", "commits": 1}
