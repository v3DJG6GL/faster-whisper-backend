"""Behaviour of /quick-config client JS: the report form's chip editor and
the rule-save flow.

The functions are pulled out of the served page and run under node against a
tiny fake DOM — just enough element API (classList, children/nextSibling,
querySelector on `.class` / `[data-idx="N"]` / descendant selectors) for the
chip helpers; anything else a test needs it stubs on globalThis. Anything the page's code touches beyond that is a bug in this
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
const _body = new El('body');
document.querySelectorAll = (sel) => _body.querySelectorAll(sel);
function _traceItem(entry, form) {
  // A rendered trace row as syncReportedBadges / submitReport see it.
  const item = new El('div'); item.className = 'trace-item';
  item.dataset.entry = JSON.stringify(entry);
  const badge = new El('span'); badge.className = 'trace-reported-badge';
  item.appendChild(badge); item.appendChild(form); _body.appendChild(item);
  return { item, badge };
}
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
    consts = []
    for name in ("_WORD_RE", "_REPORTED_KEY", "_REPORTED_MAX"):
        m = re.search(r"^const " + name + r" = .*$", page, re.M)
        assert m, f"const {name} not found in /quick-config"
        consts.append(m.group(0))
    src = "\n".join(
        [_FAKE_DOM, *consts]
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


def test_remove_report_404_clears_stale_local_hint(client, tmp_path):
    # The localStorage hint outlives a report deleted elsewhere; the DELETE
    # then 404s, and treating that as a failure left the badge (and the
    # Remove button) up with no way to clear them.
    res = _run_js(client, tmp_path, r"""
      globalThis.confirm = () => true;
      globalThis._detailText = (j, d) => (j && j.detail) || d;
      globalThis.api = async () => ({
        ok: false, status: 404, json: async () => ({ detail: 'no report to delete' }),
      });
      markReported('rid1');
      const entry = { request_id: 'rid1', final: 'Alpha Beta' };
      const form = _buildReportForm(entry);
      const { badge } = _traceItem(entry, form);
      badge.style.display = 'inline-flex';
      await removeReport(form);
      out({
        reported: isReported('rid1'),
        badge: badge.style.display,
        removeBtn: form.querySelector('.rep-remove').style.display,
        toasts: _toasts,
      });
    """, funcs=_REPORT_FUNCS + (
        "markReported", "unmarkReported", "syncReportedBadges", "removeReport",
    ))
    assert res["reported"] is False
    assert res["badge"] == "none"
    assert res["removeBtn"] == "none"
    assert not [t for t in res["toasts"] if t[0] == "err"]


def test_first_submit_reveals_remove_button(client, tmp_path):
    # The button was only ever sized at form build or by a /state sync, so a
    # trace reported for the first time could not be withdrawn on reopen.
    res = _run_js(client, tmp_path, r"""
      globalThis._detailText = (j, d) => d;
      globalThis.api = async () => ({
        ok: true, status: 200, json: async () => ({ was_updated: false }),
      });
      const entry = { request_id: 'rid1', final: 'Alpha Beta' };
      const form = _buildReportForm(entry);
      _traceItem(entry, form);
      const removeBtn = form.querySelector('.rep-remove');
      const before = removeBtn.style.display;
      form.querySelector('.rep-comment').value = 'wrong name';
      await submitReport(form);
      out({ before, after: removeBtn.style.display, toasts: _toasts });
    """, funcs=_REPORT_FUNCS + ("markReported", "submitReport"))
    assert res["before"] == "none"
    assert res["after"] == "inline-flex"
    assert res["toasts"] == [["ok", "Report sent — thank you."]]


def test_extending_a_chip_moves_its_inline_replacement(client, tmp_path):
    # Shift-extend re-anchored the last chip at its new idx_end but left the
    # old green replacement (and any absorbed chip's) in the words strip.
    res = _run_js(client, tmp_path, r"""
      const entry = { request_id: 'rid1', final: 'Alpha Beta Gamma Delta Echo' };
      const form = _buildReportForm(entry);
      toggleCorrection(form, 3, 'Delta', false);
      form._corrections[0].correct = 'D';
      toggleCorrection(form, 1, 'Beta', false);
      form._corrections[1].correct = 'B';
      _renderCorrections(form);
      const before = _replacements(form);
      toggleCorrection(form, 3, 'Delta', true);  // Beta..Delta absorbs Delta's chip
      out({
        before,
        chips: form._corrections.map((c) => [c.idx, c.idx_end, c.correct]),
        after: _replacements(form),
        nextToDelta: form.querySelector('.rep-final-words .word[data-idx="3"]')
          .nextSibling.textContent,
      });
    """)
    assert res["before"] == ["B", "D"]
    assert res["chips"] == [[1, 3, "B"]]
    assert res["after"] == ["B"]
    assert res["nextToDelta"] == "B"


def test_partial_conflict_save_still_reapplies_saved_rules(client, tmp_path):
    # A save with some rules saved and others conflicting returned before the
    # only startReapplyJobSilent() call, so the saved rules never reached the
    # historical captures unless the admin saved again.
    res = _run_js(client, tmp_path, r"""
      const calls = [];
      globalThis.setStatus = () => {};
      globalThis.liveRules = [];
      globalThis.initialRules = [];
      globalThis.api = async () => ({
        ok: true, status: 200,
        json: async () => ({
          saved: ['dict'], conflicts: [{ slug: 'other' }], captures_count: 7,
        }),
      });
      globalThis.fetchState = async () => ({ rules: [] });
      globalThis._diffsSince = () => new Map();
      globalThis.applyState = () => {};
      globalThis._reapplyDiffs = () => false;
      globalThis.renderCards = () => {};
      globalThis.updateButtons = () => {};
      globalThis.startReapplyJobSilent = (n) => { calls.push(n); };
      await _doSave({ dict: {} }, {});
      out({ calls, toasts: _toasts });
    """, funcs=("_doSave",))
    assert res["calls"] == [7]
    assert res["toasts"][0][0] == "err" and "Saved 1." in res["toasts"][0][1]


def test_resubmit_sends_seeded_chips_as_baseline(client, tmp_path):
    # The form posts its whole edited chip list; without the seed it came from
    # the server unioned it with the stored chips, so a chip removed with x
    # (or re-ranged by shift-click) came back on the next /state load.
    res = _run_js(client, tmp_path, r"""
      globalThis._detailText = (j, d) => d;
      const sent = [];
      globalThis.api = async (_m, _u, body) => {
        sent.push(JSON.parse(JSON.stringify(body)));
        return { ok: true, status: 200, json: async () => ({ was_updated: true }) };
      };
      _serverReportedChips = { rid1: { corrections: [
        { wrong: 'Beta', correct: 'Bet', idx: 1 },
        { wrong: 'Delta', correct: 'Del', idx: 3 },
      ] } };
      const entry = { request_id: 'rid1', final: 'Alpha Beta Gamma Delta' };
      const form = _buildReportForm(entry);
      _traceItem(entry, form);
      _removeChip(form, 1);
      await submitReport(form);
      await submitReport(form);
      out(sent.map((b) => [b.corrections, b.baseline_corrections]));
    """, funcs=_REPORT_FUNCS + ("markReported", "submitReport"))
    bet = {"wrong": "Beta", "correct": "Bet", "idx": 1}
    assert res[0] == [[bet], [bet, {"wrong": "Delta", "correct": "Del", "idx": 3}]]
    # The still-open form's second submit diffs against what it just sent.
    assert res[1] == [[bet], [bet]]


def test_reopened_form_keeps_the_stored_comment(client, tmp_path):
    # A resubmit overwrites the stored comment with the textarea, which was
    # always built empty, so adding one chip to a reported trace erased it.
    res = _run_js(client, tmp_path, r"""
      globalThis._detailText = (j, d) => d;
      const sent = [];
      globalThis.api = async (_m, _u, body) => {
        sent.push(body.user_comment);
        return { ok: true, status: 200, json: async () => ({ was_updated: true }) };
      };
      _serverReportedChips = { rid1: { corrections: [], user_comment: 'speaker said X' } };
      const entry = { request_id: 'rid1', final: 'Alpha Beta' };
      const form = _buildReportForm(entry);
      _traceItem(entry, form);
      const seeded = form.querySelector('.rep-comment').value;
      form.querySelector('.rep-comment').value = 'speaker said Y';
      await submitReport(form);
      out({ seeded, sent, snapshot: _serverReportedChips.rid1.user_comment });
    """, funcs=_REPORT_FUNCS + ("markReported", "submitReport"))
    assert res["seeded"] == "speaker said X"
    assert res["sent"] == ["speaker said Y"]
    assert res["snapshot"] == "speaker said Y"


def test_edit_made_while_the_report_posts_keeps_the_form_open(client, tmp_path):
    # Only Submit was disabled while the POST ran. A chip or comment edit made
    # meanwhile was never sent, yet the form collapsed under "Report sent" and
    # the client snapshot (later rebuild seed + merge baseline) took the
    # unsent live chips, so a chip removed in flight came back on resubmit.
    res = _run_js(client, tmp_path, r"""
      globalThis._detailText = (j, d) => d;
      let release;
      globalThis.api = (_m, _u, body) => new Promise((res) => {
        release = () => res({ ok: true, status: 200, json: async () => ({ was_updated: false }) });
      });
      const entry = { request_id: 'rid1', final: 'Alpha Beta Gamma' };
      const form = _buildReportForm(entry);
      _traceItem(entry, form);
      form.classList.add('open');
      toggleCorrection(form, 1, 'Beta', false);
      form._corrections[0].correct = 'Bet';
      const p = submitReport(form);
      form._corrections[0].correct = 'Bed';          // typo fix while in flight
      release();
      await p;
      out({
        open: form.classList.contains('open'),
        snapshot: _serverReportedChips.rid1.corrections,
        baseline: form._baseline,
        toasts: _toasts,
      });
    """, funcs=_REPORT_FUNCS + ("markReported", "submitReport"))
    sent = [{"wrong": "Beta", "correct": "Bet", "idx": 1}]
    assert res["open"] is True
    assert res["snapshot"] == sent
    assert res["baseline"] == sent
    assert [t[0] for t in res["toasts"]] == ["err"]
    assert "not" in res["toasts"][0][1] and "submit again" in res["toasts"][0][1]


def test_trace_row_stashes_only_the_fields_its_readers_need(client, tmp_path):
    # The whole entry (raw/final twice plus steps, up to ~400k chars) was kept
    # as a row attribute and re-parsed on every save to read request_id.
    res = _run_js(client, tmp_path, r"""
      const made = [];
      const orig = document.createElement;
      // Stop renderTrace right after the row element is stashed: the rest of
      // the row's markup is not under test here.
      document.createElement = (t) => {
        if (made.length) throw new Error('stop');
        const e = orig(t); made.push(e); return e;
      };
      try {
        renderTrace({
          request_id: 'r1', ts: 5, tokens: ['a'], bigrams: ['a b'],
          raw: 'x'.repeat(1000), raw_text: 'x'.repeat(1000), final: 'y',
          final_text: 'y', steps: [{ name: 's' }], status: 'ok',
        });
      } catch (e) { if (e.message !== 'stop') throw e; }
      document.createElement = orig;
      out(JSON.parse(made[0].dataset.entry));
    """, funcs=("renderTrace",))
    assert res == {"request_id": "r1", "ts": 5, "tokens": ["a"], "bigrams": ["a b"]}


def test_conflict_merge_keeps_the_other_writers_regex_list_entries(client, tmp_path):
    # A regex-list rule's `entries` diffed as a scalar, so a conflict re-merge
    # put back the user's whole load-time list plus their edit and the other
    # writer's new entry was dropped by the next save, under an "Auto-merged"
    # toast.
    page = client.get("/quick-config").text
    field_sets = "\n".join(
        "globalThis." + m.group(1) + " = " + m.group(2)
        for m in re.finditer(r"^const (_\w+_FIELDS) = (.*)$", page, re.M))
    res = _run_js(client, tmp_path, field_sets + r"""
      const e = (p) => ({ pattern: p, replacement: p.toUpperCase() });
      const base = [e('a'), e('b'), e('c')];
      // The user moved c to the top, dropped b and appended y; meanwhile the
      // other writer inserted x after a.
      const mine = [e('c'), e('a'), e('y')];
      const theirs = [e('a'), e('x'), e('b'), e('c')];
      const d = _fieldDiff(base, mine, _diffKindFor('entries'));
      const merged = _applyFieldDiff(theirs, d);
      // The other writer deleted a and appended z after c: a goes, z follows c.
      const merged2 = _applyFieldDiff([e('b'), e('c'), e('z')], d);
      out({ merged: merged.map(x => x.pattern), merged2: merged2.map(x => x.pattern),
            noop: _fieldDiff(base, base.map(x => ({ ...x })), _diffKindFor('entries')) });
    """, funcs=("_stringify", "_olistKeyed", "_olistAlign", "_fieldDiff", "_applyFieldDiff",
                "_diffKindFor"))
    assert res["merged"] == ["c", "a", "x", "y"]
    assert res["merged2"] == ["c", "z", "y"]
    assert res["noop"] is None


_OLIST_FUNCS = ("_stringify", "_olistKeyed", "_olistAlign", "_fieldDiff", "_applyFieldDiff",
                "_diffKindFor", "_reapplyDiffs")


def _olist_consts(page):
    return "\n".join(
        "globalThis." + m.group(1) + " = " + m.group(2)
        for m in re.finditer(r"^const (_\w+_FIELDS) = (.*)$", page, re.M))


def test_conflict_merge_keeps_an_entry_edited_in_place_as_one_entry(client, tmp_path):
    # The regex-list merge keyed entries by their whole content, so an edit in
    # place looked like delete + add: an entry the user edited could not
    # anchor the other writer's new entries (they were spliced in front of it,
    # changing the apply order), and an entry both writers edited came back
    # twice, the other writer's copy first, shadowing the user's.
    page = client.get("/quick-config").text
    res = _run_js(client, tmp_path, _olist_consts(page) + r"""
      const e = (p, r, extra) => ({ pattern: p, replacement: r || p.toUpperCase(), ...(extra || {}) });
      const merge = (base, mine, theirs) => {
        const clashes = [];
        const m = _applyFieldDiff(theirs, _fieldDiff(base, mine, _diffKindFor('entries')), clashes);
        const v = m.map(x => x.pattern + '>' + x.replacement);
        return clashes.length ? ['CONFLICT', clashes, v] : v;
      };
      out({
        // user edits x, other writer edits y (the next entry): order holds.
        order: merge([e('a'), e('x'), e('y')], [e('a'), e('x', 'X2'), e('y')],
                     [e('a'), e('x'), e('y', 'Y2')]),
        // user edits x, other writer inserts n right after x.
        anchor: merge([e('a'), e('x'), e('y')], [e('a'), e('x', 'X2'), e('y')],
                      [e('a'), e('x'), e('n'), e('y')]),
        // both edit the same entry differently: a clash, the user's version kept.
        both: merge([e('foo', 'bar'), e('x')], [e('foo', 'baz'), e('x')],
                    [e('foo', 'qux'), e('x')]),
        bothNote: merge([e('foo', 'X')], [e('foo', 'X', { note: 'n' })], [e('foo', 'Y')]),
        // both made the same edit: one entry.
        same: merge([e('foo', 'bar')], [e('foo', 'baz')], [e('foo', 'baz')]),
        // the user deleted b, the other writer edited it: the deletion wins.
        deleted: merge([e('a'), e('b'), e('c')], [e('a'), e('c')],
                       [e('a'), e('b', 'THEIRS'), e('c')]),
        // the other writer deleted b, the user edited it: the edit survives.
        kept: merge([e('a'), e('b'), e('c')], [e('a'), e('b', 'MINE'), e('c')],
                    [e('a'), e('c')]),
      });
    """, funcs=_OLIST_FUNCS)
    assert res["order"] == ["a>A", "x>X2", "y>Y2"]
    assert res["anchor"] == ["a>A", "x>X2", "n>N", "y>Y"]
    assert res["both"] == ["CONFLICT", ["foo"], ["foo>baz", "x>X"]]
    assert res["bothNote"] == ["CONFLICT", ["foo"], ["foo>X"]]
    assert res["same"] == ["foo>baz"]
    assert res["deleted"] == ["a>A", "c>C"]
    assert res["kept"] == ["a>A", "b>MINE", "c>C"]


def test_conflict_merge_pairs_a_pattern_edited_in_place_with_its_entry(client, tmp_path):
    # Entries were keyed by pattern alone, so editing an entry's pattern
    # looked like delete + add: the entry came back twice, the other writer's
    # new entry was spliced in front of it, or the other writer's edit to the
    # same entry was dropped, all under an "Auto-merged" toast.
    page = client.get("/quick-config").text
    res = _run_js(client, tmp_path, _olist_consts(page) + r"""
      const e = (p, r) => ({ pattern: p, replacement: r || p.toUpperCase() });
      const merge = (base, mine, theirs) => {
        const clashes = [];
        const m = _applyFieldDiff(theirs, _fieldDiff(base, mine, _diffKindFor('entries')), clashes);
        const v = m.map(x => x.pattern + '>' + x.replacement);
        return clashes.length ? ['CONFLICT', v] : v;
      };
      out({
        // the other writer fixed the pattern, the user changed the replacement.
        theirsPattern: merge([e('a'), e('colour', 'color')], [e('a'), e('colour', 'COLOR')],
                             [e('a'), e('colou?r', 'color')]),
        // the user fixed the pattern, the other writer changed the replacement.
        minePattern: merge([e('a'), e('teh', 'the'), e('y')], [e('a'), e('teh+', 'the'), e('y')],
                           [e('a'), e('teh', 'THE'), e('y')]),
        // both made the same pattern fix: one entry.
        samePattern: merge([e('x')], [e('x2', 'X')], [e('x2', 'X')]),
        // the user fixed x's pattern, the other writer inserted n after x.
        anchor: merge([e('a'), e('x'), e('y')], [e('a'), e('x2', 'X'), e('y')],
                      [e('a'), e('x'), e('n'), e('y')]),
        // two adjacent pattern fixes, the other writer edits the second entry.
        two: merge([e('a'), e('x'), e('y')], [e('a'), e('x2', 'X'), e('y2', 'Y')],
                   [e('a'), e('x'), e('y', 'Y9')]),
        // the user's pattern fix while the other writer edits a different entry.
        clean: merge([e('a'), e('x'), e('y')], [e('a'), e('x2', 'X'), e('y')],
                     [e('a'), e('x'), e('y', 'Y2')]),
      });
    """, funcs=_OLIST_FUNCS)
    assert res["theirsPattern"] == ["CONFLICT", ["a>A", "colour>COLOR"]]
    assert res["minePattern"] == ["CONFLICT", ["a>A", "teh+>the", "y>Y"]]
    assert res["samePattern"] == ["x2>X"]
    assert res["anchor"] == ["a>A", "x2>X", "n>N", "y>Y"]
    assert res["clean"] == ["a>A", "x2>X", "y>Y2"]
    assert res["two"] == ["CONFLICT", ["a>A", "x2>X", "y2>Y"]]


def test_reapply_keeps_the_users_edits_on_a_rule_whose_entry_both_writers_edited(client, tmp_path):
    # One entry both writers edited made _reapplyDiffs skip the whole rule:
    # every other unsaved edit on it (added entries, the enabled toggle, other
    # entries) vanished with no undo. The rule now takes all the user's diffs,
    # keeps the user's version of the clashing entry, stays dirty, and its slug
    # is reported so the toast can name it.
    page = client.get("/quick-config").text
    res = _run_js(client, tmp_path, _olist_consts(page) + r"""
      const e = (p, r) => ({ pattern: p, replacement: r });
      globalThis.dirty = new Set();
      globalThis.liveRules = [
        { name: 'rx', enabled: true, entries: [e('foo', 'qux'), e('a', 'A')] },
        { name: 'ok', enabled: true, entries: [e('a', 'A')] },
      ];
      const diffs = new Map([
        ['rx', { enabled: _fieldDiff(true, false, 'scalar'),
                 entries: _fieldDiff([e('foo', 'bar'), e('a', 'A')],
                                     [e('foo', 'baz'), e('a', 'A'), e('n', 'N')], 'olist') }],
        ['ok', { entries: _fieldDiff([e('a', 'A')], [e('a', 'A2')], 'olist') }],
      ]);
      const clashed = [];
      const merged = _reapplyDiffs(diffs, clashed);
      out({ merged, clashed, dirty: [...dirty].sort(), rules: liveRules });
    """, funcs=_OLIST_FUNCS)
    assert res["merged"] == 2
    assert res["clashed"] == ["rx"]
    assert res["dirty"] == ["ok", "rx"]
    assert res["rules"][0] == {"name": "rx", "enabled": False, "entries": [
        {"pattern": "foo", "replacement": "baz"}, {"pattern": "a", "replacement": "A"},
        {"pattern": "n", "replacement": "N"}]}
    assert res["rules"][1]["entries"] == [{"pattern": "a", "replacement": "A2"}]


def test_conflict_toast_names_the_rule_that_could_not_be_merged(client, tmp_path):
    res = _run_js(client, tmp_path, r"""
      globalThis.setStatus = () => {};
      globalThis.liveRules = [];
      globalThis.initialRules = [];
      globalThis.api = async () => ({
        ok: true, status: 200,
        json: async () => ({ saved: [], conflicts: [{ slug: 'rx' }, { slug: 'ok' }] }),
      });
      globalThis.fetchState = async () => ({ rules: [] });
      globalThis._diffsSince = () => new Map();
      globalThis.applyState = () => {};
      let calls = 0;  // the conflicted rules' re-apply, then the in-flight one
      globalThis._reapplyDiffs = (d, clashed) => { if (!calls++) clashed.push('rx'); return 1; };
      globalThis.renderCards = () => {};
      globalThis.updateButtons = () => {};
      await _doSave({ rx: {}, ok: {} }, {});
      out(_toasts);
    """, funcs=("_doSave",))
    msg = res[0][1]
    assert "Auto-merged your changes for ok on top" in msg
    assert "Conflict on rx: another writer also edited an entry you edited; your version is kept." in msg
    assert "Review and Save again." in msg


def test_save_toasts_a_clash_on_an_edit_typed_while_the_post_was_in_flight(client, tmp_path):
    # The in-flight re-apply passed no `clashed` array, so an edit typed
    # during the POST that another writer had also made went unreported.
    res = _run_js(client, tmp_path, r"""
      globalThis.setStatus = () => {};
      globalThis.liveRules = [];
      globalThis.api = async () => ({
        ok: true, status: 200, json: async () => ({ saved: ['r1'], conflicts: [] }),
      });
      globalThis.fetchState = async () => ({ rules: [] });
      globalThis._diffsSince = () => new Map();
      globalThis.applyState = () => {};
      globalThis._reapplyDiffs = (d, clashed) => { clashed.push('rx'); return 1; };
      globalThis.renderCards = () => {};
      globalThis.updateButtons = () => {};
      await _doSave({ r1: {} }, {});
      out(_toasts);
    """, funcs=("_doSave",))
    assert res[-1][0] == "err"
    assert "Conflict on rx: another writer also edited an entry you edited" in res[-1][1]
