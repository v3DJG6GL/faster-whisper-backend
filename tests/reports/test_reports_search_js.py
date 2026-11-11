"""reports.html search box driven under node: the per-row search haystack memo
and the debounced input listener.

Each row's raw + final + intended + comment + notes + corrections text is
lowercased once and kept on the row, not rebuilt for every row on every
keystroke; a notes save drops the memo on both the row it was saved from and
the live row a Refresh put in its place, so the saved notes are searchable."""

import json
import re

from tests._page_helpers import js_fn, needs_node, node_json
from tests.conftest import read_repo_text


def _html() -> str:
    return read_repo_text("faster_whisper_backend", "reports", "templates", "reports.html")


def _harness(drive: str) -> str:
    html = _html()
    filters_src = js_fn(html, "  function applyFilters(rows) {")
    save_src = js_fn(html, "  async function onSaveNotes(r, ta, btn, dirty) {")
    return r"""
const _vals = { 'filt-status': 'all', 'filt-model': 'all', 'filt-search': '' };
const document = { getElementById(id) { return { value: _vals[id] }; } };
var _allReports = [];
var _drafts = {};
function _hasDraft(id) { return Object.prototype.hasOwnProperty.call(_drafts, id); }
var _loadsOpen = 0;
function load() {}
function toast() {}
var _savedNotes = '';
async function api() { return { report: { admin_notes: _savedNotes } }; }
function search(q, rows) { _vals['filt-search'] = q; return applyFilters(rows).map(r => r.id); }
function controls() {
  return { btn: { isConnected: true, disabled: false },
           dirty: { classList: { toggle() {} } } };
}
""" + filters_src + "\n" + save_src + r"""
(async () => {
""" + drive + r"""
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
"""


def _row(i, **kw):
    base = {"id": i, "raw": "", "final": "", "intended_text": "", "user_comment": "",
            "admin_notes": "", "corrections": []}
    base.update(kw)
    return base


@needs_node
def test_search_matches_every_field_and_spans_field_boundaries():
    rows = [
        _row(1, raw="Alpha RAW"),
        _row(2, final="beta final"),
        _row(3, intended_text="Gamma"),
        _row(4, user_comment="delta comment", admin_notes="epsilon"),
        _row(5, corrections=[{"wrong": "Zeta", "correct": "eta"}]),
    ]
    out = node_json(_harness(r"""
  const rows = """ + json.dumps(rows) + r""";
  console.log(JSON.stringify({
    raw: search('alpha raw', rows),
    final: search('BETA', rows),
    intended: search('gamma', rows),
    boundary: search('comment epsilon', rows),
    corr: search('zeta eta', rows),
    none: search('omega', rows),
    all: search('', rows),
  }));
"""))
    assert out == {"raw": [1], "final": [2], "intended": [3], "boundary": [4],
                   "corr": [5], "none": [], "all": [1, 2, 3, 4, 5]}


@needs_node
def test_haystack_is_built_once_per_row():
    out = node_json(_harness(r"""
  const rows = [{ id: 1, raw: 'foo' }];
  const first = search('foo', rows);
  // Nothing on the page rewrites raw client-side; a changed raw that still
  // matches the old text proves the second keystroke reused the memo.
  rows[0].raw = 'bar';
  console.log(JSON.stringify({ first, again: search('foo', rows), bar: search('bar', rows) }));
"""))
    assert out == {"first": [1], "again": [1], "bar": []}


@needs_node
def test_notes_save_invalidates_the_memo_on_both_row_objects():
    out = node_json(_harness(r"""
  const stale = { id: 7, raw: 'foo', admin_notes: 'old' };
  const live = { id: 7, raw: 'foo', admin_notes: 'old' };
  _allReports = [live];
  search('old', [stale]); search('old', [live]);
  _savedNotes = 'zzz new';
  const c = controls();
  await onSaveNotes(stale, { value: 'zzz new' }, c.btn, c.dirty);
  console.log(JSON.stringify({
    stale: search('zzz', [stale]), live: search('zzz', [live]),
    oldGone: search('old', [stale, live]),
  }));
"""))
    assert out == {"stale": [7], "live": [7], "oldGone": []}


def test_search_input_is_debounced_and_selects_stay_immediate():
    html = _html()
    assert "getElementById('filt-search').addEventListener('input', render)" not in html
    m = re.search(
        r"getElementById\('filt-search'\)\.addEventListener\('input', function\(\) \{"
        r"\s*clearTimeout\((\w+)\);\s*\1 = setTimeout\(render, (\d+)\);\s*\}\);",
        html,
    )
    assert m, "filt-search input listener is not a clearTimeout/setTimeout(render) debounce"
    assert 50 <= int(m.group(2)) <= 300
    assert "getElementById('filt-status').addEventListener('change', render)" in html
    assert "getElementById('filt-model').addEventListener('change', render)" in html
