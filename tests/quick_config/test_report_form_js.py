"""Behaviour of the /quick-config report form's chip editor (client JS).

The functions are pulled out of the served page and run under node against a
tiny fake DOM — just enough element API (classList, children/nextSibling,
querySelector on `.class` / `[data-idx="N"]` / descendant selectors) for the
chip helpers. Anything the page's code touches beyond that is a bug in this
harness, not in the page."""

import json
import re
import shutil
import subprocess

import pytest

_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(_NODE is None, reason="node unavailable")

_FAKE_DOM = r"""
class El {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.parentNode = null;
    this.dataset = {}; this.style = {}; this._cls = new Set();
    this.textContent = ''; this.value = ''; this.disabled = false;
    this.listeners = {};
  }
  set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get className() { return [...this._cls].join(' '); }
  get classList() {
    const s = this._cls;
    return {
      contains: (c) => s.has(c),
      add: (c) => s.add(c),
      remove: (c) => s.delete(c),
      toggle: (c, on) => {
        if (on === undefined) on = !s.has(c);
        if (on) s.add(c); else s.delete(c);
        return on;
      },
    };
  }
  set innerHTML(html) {
    // Flat parse: one child per element carrying a class attribute. The
    // page's static form markup only needs to be findable by class.
    this.children = [];
    const re = /<(\w+)[^>]*?class="([^"]+)"/g;
    let m;
    while ((m = re.exec(html)) !== null) {
      const e = new El(m[1]); e.className = m[2]; this.appendChild(e);
    }
  }
  setAttribute() {}
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  insertBefore(c, ref) {
    c.parentNode = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
    return c;
  }
  removeChild(c) {
    const i = this.children.indexOf(c);
    if (i >= 0) this.children.splice(i, 1);
    c.parentNode = null;
    return c;
  }
  get nextSibling() {
    if (!this.parentNode) return null;
    const sib = this.parentNode.children;
    return sib[sib.indexOf(this) + 1] || null;
  }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  focus() {}
  closest(sel) {
    for (let e = this.parentNode; e; e = e.parentNode) if (_matchOne(e, sel)) return e;
    return null;
  }
  _all() {
    const out = [];
    const walk = (e) => {
      for (const c of e.children) if (c instanceof El) { out.push(c); walk(c); }
    };
    walk(this);
    return out;
  }
  querySelectorAll(sel) { return this._all().filter((e) => _matches(e, sel, this)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}
function _matchOne(e, part) {
  const m = part.match(/^(\w*)((?:\.[\w-]+)*)(?:\[data-(\w+)="([^"]*)"\])?$/);
  if (!m) throw new Error('fake DOM: unsupported selector ' + part);
  if (m[1] && e.tagName !== m[1]) return false;
  for (const c of m[2].split('.').filter(Boolean)) if (!e._cls.has(c)) return false;
  if (m[3] !== undefined && e.dataset[m[3]] !== m[4]) return false;
  return true;
}
function _matches(e, sel, root) {
  const parts = sel.trim().split(/\s+/);
  if (!_matchOne(e, parts.pop())) return false;
  let cur = e.parentNode;
  while (parts.length) {
    while (cur && cur !== root && !_matchOne(cur, parts[parts.length - 1])) cur = cur.parentNode;
    if (!cur || cur === root) return false;
    parts.pop();
    cur = cur.parentNode;
  }
  return true;
}
const document = {
  createElement: (t) => new El(t),
  createTextNode: (t) => ({ textContent: t, parentNode: null }),
};
const _ls = {};
const localStorage = {
  getItem: (k) => (k in _ls ? _ls[k] : null),
  setItem: (k, v) => { _ls[k] = String(v); },
};
const _toasts = [];
function showToast(msg, kind) { _toasts.push([kind, msg]); }
function _words(form) {
  return form.querySelectorAll('.rep-final-words .word').map((w) => ({
    text: w.textContent, selected: w._cls.has('selected'),
  }));
}
function _replacements(form) {
  return form.querySelectorAll('.rep-final-words .word-replacement')
    .map((r) => r.textContent);
}
let _serverReportedSet = new Set();
let _serverReportedChips = {};
"""

_REPORT_FUNCS = (
    "_wordifyFinal", "_buildReportForm", "_chipCovers", "_selectWord",
    "_setReplacementInline", "_clearReplacementInline", "_spanText",
    "_focusLastInput", "_removeChip", "_extendLastChip", "toggleCorrection",
    "_renderCorrections", "getReportedIds", "isReported",
)


def _extract_function(page: str, name: str) -> str:
    m = re.search(r"^(?:async )?function " + re.escape(name) + r"\(", page, re.M)
    assert m, f"function {name} not found in /quick-config"
    depth = 0
    for i in range(page.index("{", m.end()), len(page)):
        if page[i] == "{":
            depth += 1
        elif page[i] == "}":
            depth -= 1
            if depth == 0:
                return page[m.start():i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _run_js(client, tmp_path, body, funcs=_REPORT_FUNCS):
    """Run `body` after the extracted page functions; it must print one JSON
    value via `out(...)`. Returns that value."""
    page = client.get("/quick-config").text
    word_re = re.search(r"^const _WORD_RE = .*$", page, re.M)
    assert word_re
    src = "\n".join(
        [_FAKE_DOM, word_re.group(0)]
        + [_extract_function(page, n) for n in funcs]
        + ["const out = (v) => console.log(JSON.stringify(v));",
           "(async () => {", body, "})().catch((e) => { console.error(e); process.exit(1); });"]
    )
    f = tmp_path / "report_form.js"
    f.write_text(src, encoding="utf-8")
    p = subprocess.run([_NODE, str(f)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_seeded_single_word_chip_is_normalised_and_struck_through(client, tmp_path):
    # The server drops idx_end on single-word chips; the reopened form used to
    # keep it undefined, so clicking the chip's word added a duplicate chip
    # and shift-click turned the range into NaN with an empty `wrong`.
    res = _run_js(client, tmp_path, r"""
      const seed = { corrections: [{ wrong: 'Beta', correct: 'Bet', idx: 1 }] };
      const entry = { request_id: 'rid1', final: 'Alpha Beta Gamma Delta' };

      _serverReportedChips = { rid1: JSON.parse(JSON.stringify(seed)) };
      const f1 = _buildReportForm(entry);
      const seeded = { chips: JSON.parse(JSON.stringify(f1._corrections)), words: _words(f1) };
      toggleCorrection(f1, 1, 'Beta', false);
      const afterClick = f1._corrections.length;

      _serverReportedChips = { rid1: JSON.parse(JSON.stringify(seed)) };
      const f2 = _buildReportForm(entry);
      toggleCorrection(f2, 2, 'Gamma', true);
      out({ seeded, afterClick, extended: f2._corrections });
    """)
    assert res["seeded"]["chips"] == [
        {"wrong": "Beta", "correct": "Bet", "idx": 1, "idx_end": 1},
    ]
    assert [w["selected"] for w in res["seeded"]["words"]] == [False, True, False, False]
    assert res["afterClick"] == 0
    assert res["extended"] == [
        {"wrong": "Beta Gamma", "correct": "Bet", "idx": 1, "idx_end": 2},
    ]
