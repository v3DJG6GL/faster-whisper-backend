"""cb:map editor: two rows with the same spoken word collapse into one dict
key on read (last row wins), so Save dropped the earlier replacement with no
warning while the "n / cap" readout still counted both rows. _readMap now
flags every row sharing a key.

The real _makeMapRow / _readMap run under node against a minimal DOM stub
(table > tr > td > input), no browser needed."""

import json
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common


_DOM_STUB = r"""
function el(tag) {
  const e = {
    tag, children: [], parentNode: null, listeners: {}, attrs: {},
    value: '', title: '', type: '', style: {},
    classList: {
      s: new Set(),
      add(c) { this.s.add(c); },
      toggle(c, on) { on ? this.s.add(c) : this.s.delete(c); },
      contains(c) { return this.s.has(c); },
    },
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'title') this.title = String(v); },
    removeAttribute(k) { delete this.attrs[k]; if (k === 'title') this.title = ''; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    fire(t) { (this.listeners[t] || []).forEach(f => f({})); },
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    remove() {
      const p = this.parentNode;
      if (p) p.children.splice(p.children.indexOf(this), 1);
      this.parentNode = null;
    },
    querySelectorAll(sel) { return sel === 'tr' ? this.children.slice() : []; },
    querySelector(sel) {
      const cell = { 'td:first-child input': 0, 'td:nth-child(2) input': 1 }[sel];
      return cell === undefined ? null : this.children[cell].children[0];
    },
  };
  return e;
}
global.document = { createElement: el };
"""


def _editor_fns() -> str:
    js = web_common.RULE_EDITOR_JS
    esc = js[js.index("function _esc(s) {"):js.index("const _PIPELINE_TYPES")]
    rows = js[js.index("function _bindEnterCommit("):
              js.index("// ---- regex-list entry editor")]
    return esc + rows


def _run(drive: str) -> dict:
    out = subprocess.run(["node", "-"], input=_DOM_STUB + _editor_fns() + drive,
                         capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


_SETUP = r"""
const rule = { map: { foo: 'A', bar: 'B' } };
let commits = 0;
const tbl = el('table');
const r1 = tbl.appendChild(_makeMapRow(rule, 'foo', 'A', () => commits++, null, null, 0, false));
const r2 = tbl.appendChild(_makeMapRow(rule, 'bar', 'B', () => commits++, null, null, 0, false));
const key = (r) => r.children[0].children[0];
const dup = (r) => key(r).classList.contains('map-dup');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_duplicate_spoken_word_flags_both_rows_and_clears_when_fixed():
    res = _run(_SETUP + r"""
key(r2).value = 'foo'; key(r2).fire('input');
const clash = { map: { ...rule.map }, d1: dup(r1), d2: dup(r2), title: key(r1).title };
key(r2).value = 'baz'; key(r2).fire('input');
console.log(JSON.stringify({ clash, d1: dup(r1), d2: dup(r2), title: key(r1).title }));
""")
    assert res["clash"]["map"] == {"foo": "B"}
    assert res["clash"]["d1"] is True and res["clash"]["d2"] is True
    assert "only the last row is saved" in res["clash"]["title"]
    assert (res["d1"], res["d2"], res["title"]) == (False, False, "")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_deleting_the_clashing_row_clears_the_flag():
    res = _run(_SETUP + r"""
key(r2).value = 'foo'; key(r2).fire('input');
const before = dup(r1);
r2.children[2].children[0].fire('click');   // the row's × button
console.log(JSON.stringify({ before, map: rule.map, d1: dup(r1) }));
""")
    assert res == {"before": True, "map": {"foo": "A"}, "d1": False}


def test_duplicate_flag_is_styled():
    assert ".rule-editor .map-table input.map-dup {" in web_common.NAV_CSS
