"""regex-list ↑/↓ stepper: _rlMove reinserted the row that holds the pressed
button, and moving a focused node blurs it, so a keyboard user's focus fell
to <body> after one step and a second Space did nothing. The neighbour is
moved instead; the order is the same and the focused row stays put.

The real _makeEntryRow / _readEntries / _renumberEntries run under node
against a minimal DOM stub whose node moves blur a focused node like a
browser does."""

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
    className: '', value: '', textContent: '', innerHTML: '', disabled: false,
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
const rows = ['a', 'b', 'c'].map((p) =>
  box.appendChild(_makeEntryRow(rule, { pattern: p }, () => commits++, null)));
_renumberEntries(box);
const btn = (r, dir) => r.querySelector('.rl-move-' + dir);
const order = () => rule.entries.map((e) => e.pattern).join('');
const press = (r, dir) => { const b = btn(r, dir); b.focus(); b.fire('click');
  return { order: order(), focused: document.activeElement === b }; };
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_move_up_keeps_focus_on_the_pressed_button():
    res = _run(_SETUP + r"""
const first = press(rows[2], 'up');
const b = btn(rows[2], 'up'); b.fire('click');   // second press, focus kept
console.log(JSON.stringify({ first, second: order(),
  focused: document.activeElement === b, commits }));
""")
    assert res == {"first": {"order": "acb", "focused": True},
                   "second": "cab", "focused": True, "commits": 2}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_move_down_keeps_focus_on_the_pressed_button():
    res = _run(_SETUP + r"""
const first = press(rows[0], 'down');
const second = press(rows[0], 'down');
console.log(JSON.stringify({ first, second,
  last: btn(rows[0], 'down').disabled, ord: rows[0].querySelector('.rl-ord').textContent }));
""")
    assert res == {"first": {"order": "bac", "focused": True},
                   "second": {"order": "bca", "focused": True},
                   "last": True, "ord": "3"}
