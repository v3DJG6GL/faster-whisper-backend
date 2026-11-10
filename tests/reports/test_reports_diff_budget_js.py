"""reports.html diff rendering driven under node: the page-wide LCS budget and
the diff memo.

render() rebuilds every card on each load, filter change and search
keystroke, and each card with an intended text runs an O(n*m) LCS diff. The
per-card budget alone still lets many at-cap reports freeze the tab, so one
render() may only spend a page-wide budget of cells (later cards fall back to
the plain del/ins view) and an unchanged pair is diffed once, not on every
re-render."""

from tests._page_helpers import js_fn, js_upto, needs_node, node_json
from tests.conftest import read_repo_text


def _html() -> str:
    return read_repo_text("faster_whisper_backend", "reports", "templates", "reports.html")


def _harness(drive: str) -> str:
    html = _html()
    diff_src = js_upto(html, "  var WORDISH = ", "  var _allReports = [];")
    render_src = js_fn(html, "  function render() {")
    load_src = js_fn(html, "  async function load() {")
    return r"""
function _el(tag, cls, text) {
  return { tag, className: cls, textContent: text === undefined ? '' : text, kids: [],
           appendChild(k) { this.kids.push(k); return k; } };
}
const listEl = _el('div', 'list');
Object.defineProperty(listEl, 'innerHTML', { set(v) { listEl.kids = []; } });
const document = { getElementById(id) { return listEl; } };
var _allReports = [];
var _counts = {}, _truncated = false;
var _loadSeq = 0, _loadLanded = 0, _loadsOpen = 0;
var _apiReports = [];
async function api() { return { reports: _apiReports, counts: {} }; }
function rebuildModelFilter() {}
function updateCounts() {}
function toast() {}
function applyFilters(rows) { return rows; }
function renderCard(r) { return renderDiff(r.final || '', r.intended_text); }
""" + diff_src + render_src + "\n" + load_src + r"""
let lcsCalls = 0;
const _realLcs = lcsDiff;
lcsDiff = function(a, b) { lcsCalls++; return _realLcs(a, b); };
function spans(div) { return div.kids.map(k => [k.className, k.textContent]); }
function plainFallback(div, before) {
  return div.kids.length === 3 && div.kids[0].className === 'diff-del'
    && div.kids[0].textContent === before;
}
// ~1000 x 2000 punctuation tokens: right at the per-card budget of 2e6 cells.
function atCap(i) {
  return { final: 'f' + i + '.'.repeat(999), intended_text: 'i' + i + ','.repeat(1999) };
}
(async () => {
""" + drive + r"""
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
"""


@needs_node
def test_one_render_spends_at_most_the_page_budget():
    out = node_json(_harness(r"""
  _allReports = [];
  for (let i = 0; i < 15; i++) _allReports.push(atCap(i));
  render();
  const first = lcsCalls;
  const plain = listEl.kids.filter((d, i) => plainFallback(d, _allReports[i].final)).length;
  console.log(JSON.stringify({ first, plain, cards: listEl.kids.length }));
"""))
    assert out["cards"] == 15
    # 2e7 page budget / 2e6 cells per at-cap pair.
    assert out["first"] <= 10
    assert out["plain"] == 15 - out["first"]


@needs_node
def test_rerender_reuses_cached_diffs():
    out = node_json(_harness(r"""
  _allReports = [atCap(0), atCap(1)];
  render();
  const first = lcsCalls;
  const before = listEl.kids.map(spans);
  render(); render();
  const after = listEl.kids.map(spans);
  console.log(JSON.stringify({ first, total: lcsCalls, same: JSON.stringify(before) === JSON.stringify(after) }));
"""))
    assert out["first"] == 2
    assert out["total"] == 2
    assert out["same"]


@needs_node
def test_cached_cards_do_not_consume_the_budget():
    out = node_json(_harness(r"""
  _allReports = [];
  for (let i = 0; i < 10; i++) _allReports.push(atCap(i));
  render();
  for (let i = 10; i < 20; i++) _allReports.push(atCap(i));
  render();
  const diffed = listEl.kids.filter((d, i) => !plainFallback(d, _allReports[i].final)).length;
  console.log(JSON.stringify({ calls: lcsCalls, diffed }));
"""))
    # The ten cached diffs are free, so the second render still has the
    # whole budget for the ten new pairs.
    assert out["calls"] == 20
    assert out["diffed"] == 20


@needs_node
def test_a_new_snapshot_drops_the_old_diffs():
    out = node_json(_harness(r"""
  _allReports = [atCap(0)];
  render();
  _apiReports = [atCap(1)];
  await load();
  console.log(JSON.stringify({ size: _diffCache.size, calls: lcsCalls }));
"""))
    assert out == {"size": 1, "calls": 2}


@needs_node
def test_small_pair_renders_the_same_word_diff():
    out = node_json(_harness(r"""
  _allReports = [{ final: 'hello world', intended_text: 'hello there world' }];
  render();
  console.log(JSON.stringify(spans(listEl.kids[0])));
"""))
    assert out == [
        ["diff-eq", "hello"], ["diff-eq", " "], ["diff-ins", "there"],
        ["diff-ins", " "], ["diff-eq", "world"],
    ]


@needs_node
def test_over_the_per_card_budget_still_falls_back():
    out = node_json(_harness(r"""
  const big = { final: '.'.repeat(3000), intended_text: ','.repeat(2000) };
  _allReports = [big];
  render();
  console.log(JSON.stringify({ calls: lcsCalls, plain: plainFallback(listEl.kids[0], big.final) }));
"""))
    assert out == {"calls": 0, "plain": True}
