"""Behaviour of /quick-config client JS: the report form (chip editor,
submit / remove, the "reported" hint and badges) and the rule-save flow.

The functions are pulled out of the served page and run under node against a
tiny fake DOM — just enough element API (classList, children/nextSibling,
querySelector on `.class` / `[data-idx="N"]` / descendant selectors) for the
chip helpers; anything else a test needs it stubs on globalThis. Anything the
page's code touches beyond that is a bug in this harness, not in the page."""

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
function _traceItem(final) {
  // A rendered rid1 trace row with its report form, as syncReportedBadges / submitReport see it.
  const entry = { request_id: 'rid1', final };
  const form = _buildReportForm(entry);
  const item = new El('div'); item.className = 'trace-item';
  item.dataset.entry = JSON.stringify(entry);
  const badge = new El('span'); badge.className = 'trace-reported-badge';
  item.appendChild(badge); item.appendChild(form); _body.appendChild(item);
  return { form, item, badge };
}
const _ls = {};
const localStorage = {
  getItem: (k) => (k in _ls ? _ls[k] : null),
  setItem: (k, v) => { _ls[k] = String(v); },
  removeItem: (k) => { delete _ls[k]; },
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
let _reportedOwner = 'u1';
"""

_REPORT_FUNCS = (
    "_el", "_wordifyFinal", "_buildReportForm", "_chipCovers", "_wordSpan", "_selectWord",
    "_setReplacementInline", "_clearReplacementInline", "_spanText",
    "_focusLastInput", "_removeChip", "_extendLastChip", "toggleCorrection",
    "_renderCorrections", "_reportedKey", "getReportedIds", "isReported", "_rowEntry",
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
      const { form, badge } = _traceItem('Alpha Beta');
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


def test_remove_report_404_does_not_claim_it_removed_a_triaged_report(client, tmp_path):
    # The server also 404s a withdraw of a report an admin already resolved
    # or dismissed (the row and its notes stay); "Report removed." would be
    # false there.
    res = _run_js(client, tmp_path, r"""
      globalThis.confirm = () => true;
      globalThis._detailText = (j, d) => (j && j.detail) || d;
      globalThis.api = async () => ({
        ok: false, status: 404, json: async () => ({ detail: 'no report to delete' }),
      });
      markReported('rid1');
      const { form } = _traceItem('Alpha Beta');
      await removeReport(form);
      out({ reported: isReported('rid1'), toasts: _toasts });
    """, funcs=_REPORT_FUNCS + (
        "markReported", "unmarkReported", "syncReportedBadges", "removeReport",
    ))
    assert res["reported"] is False
    assert len(res["toasts"]) == 1
    kind, msg = res["toasts"][0]
    assert kind != "err"
    assert msg != "Report removed."
    assert "already" in msg


_APPLY_STATE_STUBS = r"""
  globalThis.window = {};
  document.body = { classList: { add() {}, remove() {} } };
  globalThis.renderCards = () => {};
  globalThis.updateButtons = () => {};
  globalThis.loadMyUsage = () => {};
"""

_SAVE_STUBS = r"""
  globalThis.setStatus = () => {};
  globalThis.liveRules = [];
  globalThis.initialRules = [];
  globalThis.fetchState = async () => ({ rules: [] });
  globalThis._diffsSince = () => new Map();
  globalThis.applyState = () => {};
  globalThis.renderCards = () => {};
  globalThis.updateButtons = () => {};
"""


def _run_save_js(client, tmp_path, body):
    """_run_js on _doSave alone, with the page state it touches stubbed."""
    return _run_js(client, tmp_path, _SAVE_STUBS + body, funcs=("_doSave", "_redrawRules"))


@pytest.mark.parametrize("server_chips", [
    # The server set holds only the 100 newest open reports (and a /state
    # built before a submit committed lacks that id) ...
    "{ rid2: { corrections: [] } }",
    # ... and is empty when the reports store read fails.
    "{}",
])
def test_state_load_keeps_hint_ids_the_server_set_misses(client, tmp_path, server_chips):
    # Every /state load pruned the hint to the server set, so a report beyond
    # it lost "✓ reported" and its Remove button for good.
    res = _run_js(client, tmp_path, _APPLY_STATE_STUBS + r"""
      markReported('rid1');
      const { form, badge } = _traceItem('Alpha Beta');
      applyState({ rules: [], user_id: 'u1', reported_chips: %s });
      out({
        rid1: isReported('rid1'),
        badge: badge.style.display,
        removeBtn: form.querySelector('.rep-remove').style.display,
      });
    """ % server_chips, funcs=_REPORT_FUNCS + (
        "markReported", "syncReportedBadges", "applyState", "_redrawRules",
    ))
    assert res == {"rid1": True, "badge": "inline-flex", "removeBtn": "inline-flex"}


def test_state_load_never_reads_another_accounts_reported_hint(client, tmp_path):
    # On a shared browser the next account saw "✓ reported" and a Remove
    # button on the previous account's trace, and Remove 404'd into "Report
    # removed.". The old browser-wide key cannot be attributed: it is dropped.
    res = _run_js(client, tmp_path, _APPLY_STATE_STUBS + r"""
      _reportedOwner = 'uA';
      markReported('rid1');
      _ls[_REPORTED_KEY] = JSON.stringify(['rid9']);
      const { form, badge } = _traceItem('Alpha Beta');
      badge.style.display = 'inline-flex';
      applyState({ rules: [], user_id: 'uB', reported_chips: {} });
      const asB = {
        rid1: isReported('rid1'), rid9: isReported('rid9'),
        badge: badge.style.display,
        removeBtn: form.querySelector('.rep-remove').style.display,
      };
      markReported('rid3');
      applyState({ rules: [], user_id: 'uA', reported_chips: {} });
      out({
        asB, legacyKey: _REPORTED_KEY in _ls,
        asA: { rid1: isReported('rid1'), rid3: isReported('rid3') },
      });
    """, funcs=_REPORT_FUNCS + (
        "markReported", "syncReportedBadges", "applyState", "_redrawRules",
    ))
    assert res == {
        "asB": {"rid1": False, "rid9": False, "badge": "none", "removeBtn": "none"},
        "legacyKey": False,
        "asA": {"rid1": True, "rid3": False},
    }


def test_state_load_drops_hint_ids_of_reports_an_admin_closed(client, tmp_path):
    # The hint outlived a report resolved or dismissed on /reports: its trace
    # kept "✓ reported", and Remove deleted the triaged row.
    res = _run_js(client, tmp_path, _APPLY_STATE_STUBS + r"""
      markReported('rid1'); markReported('rid2'); markReported('rid3');
      const { form, badge } = _traceItem('Alpha Beta');
      badge.style.display = 'inline-flex';
      // rid3 was closed, then reported again: its open row wins.
      applyState({ rules: [], user_id: 'u1', closed_report_ids: ['rid1', 'rid3'],
                   reported_chips: { rid3: { corrections: [] } } });
      applyState({ rules: [], user_id: 'u1', reported_chips: {} });
      out({
        rid1: isReported('rid1'), rid2: isReported('rid2'), rid3: isReported('rid3'),
        badge: badge.style.display,
        removeBtn: form.querySelector('.rep-remove').style.display,
      });
    """, funcs=_REPORT_FUNCS + (
        "markReported", "syncReportedBadges", "applyState", "_redrawRules",
    ))
    assert res == {
        "rid1": False, "rid2": True, "rid3": True,
        "badge": "none", "removeBtn": "none",
    }


def test_first_submit_reveals_remove_button(client, tmp_path):
    # The button was only ever sized at form build or by a /state sync, so a
    # trace reported for the first time could not be withdrawn on reopen.
    res = _run_js(client, tmp_path, r"""
      globalThis._detailText = (j, d) => d;
      globalThis.api = async () => ({
        ok: true, status: 200, json: async () => ({ was_updated: false }),
      });
      const { form } = _traceItem('Alpha Beta');
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
    res = _run_save_js(client, tmp_path, r"""
      const calls = [];
      globalThis.api = async () => ({
        ok: true, status: 200,
        json: async () => ({
          saved: ['dict'], conflicts: [{ slug: 'other' }], captures_count: 7,
        }),
      });
      globalThis._reapplyDiffs = () => false;
      globalThis.startReapplyJobSilent = (n) => { calls.push(n); };
      await _doSave({ dict: {} }, {});
      out({ calls, toasts: _toasts });
    """)
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
      const { form } = _traceItem('Alpha Beta Gamma Delta');
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
      const { form } = _traceItem('Alpha Beta');
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
      const { form } = _traceItem('Alpha Beta Gamma');
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
    """, funcs=("_el", "renderTrace"))
    assert res == {"request_id": "r1", "ts": 5, "tokens": ["a"], "bigrams": ["a b"]}


_OLIST_FUNCS = ("_stringify", "_olistKeyed", "_olistAlign", "_fieldDiff", "_applyFieldDiff",
                "_diffKindFor", "_reapplyDiffs")


def _olist_consts(page):
    return "\n".join(
        "globalThis." + m.group(1) + " = " + m.group(2)
        for m in re.finditer(r"^const (_\w+_FIELDS) = (.*)$", page, re.M))


def _run_olist_js(client, tmp_path, body):
    """_run_js with the page's _*_FIELDS sets as globals and the olist merge functions."""
    page = client.get("/quick-config").text
    return _run_js(client, tmp_path, _olist_consts(page) + body, funcs=_OLIST_FUNCS)


_ENTRIES_MERGE_JS = r"""
      // [clash keys, "pattern>replacement" list] of mine's edits merged onto theirs.
      const merge = (base, mine, theirs) => {
        const clashes = [];
        const m = _applyFieldDiff(theirs, _fieldDiff(base, mine, _diffKindFor('entries')), clashes);
        return [clashes, m.map(x => x.pattern + '>' + x.replacement)];
      };
"""


def test_conflict_merge_keeps_the_other_writers_regex_list_entries(client, tmp_path):
    # A regex-list rule's `entries` diffed as a scalar, so a conflict re-merge
    # put back the user's whole load-time list plus their edit and the other
    # writer's new entry was dropped by the next save, under an "Auto-merged"
    # toast.
    res = _run_olist_js(client, tmp_path, r"""
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
    """)
    assert res["merged"] == ["c", "a", "x", "y"]
    assert res["merged2"] == ["c", "z", "y"]
    assert res["noop"] is None


def test_conflict_merge_keeps_an_entry_edited_in_place_as_one_entry(client, tmp_path):
    # The regex-list merge keyed entries by their whole content, so an edit in
    # place looked like delete + add: an entry the user edited could not
    # anchor the other writer's new entries (they were spliced in front of it,
    # changing the apply order), and an entry both writers edited came back
    # twice, the other writer's copy first, shadowing the user's.
    res = _run_olist_js(client, tmp_path, r"""
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
    """)
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
    res = _run_olist_js(client, tmp_path, r"""
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
    """)
    # The other writer's pattern fix may as well be a delete + add, so it is
    # kept next to the user's version for review rather than dropped.
    assert res["theirsPattern"] == ["CONFLICT", ["a>A", "colou?r>color", "colour>COLOR"]]
    assert res["minePattern"] == ["CONFLICT", ["a>A", "teh+>the", "y>Y"]]
    assert res["samePattern"] == ["x2>X"]
    assert res["anchor"] == ["a>A", "x2>X", "n>N", "y>Y"]
    assert res["clean"] == ["a>A", "x2>X", "y>Y2"]
    assert res["two"] == ["CONFLICT", ["a>A", "x2>X", "y2>Y"]]


def test_conflict_merge_keeps_the_other_writers_entry_added_where_they_deleted_one(client, tmp_path):
    # The pattern-edit pairing ran on the other writer's list too, so their
    # new entry added where they deleted one took the deleted entry's id: when
    # the user had deleted that entry as well it vanished with no clash, and
    # when the user had edited it only the user's version came back.
    res = _run_olist_js(client, tmp_path, r"""
      const e = (p, r) => ({ pattern: p, replacement: r || p.toUpperCase() });
      const merge = (base, mine, theirs) => {
        const clashes = [];
        const m = _applyFieldDiff(theirs, _fieldDiff(base, mine, _diffKindFor('entries')), clashes);
        const v = m.map(x => x.pattern + '>' + x.replacement);
        return clashes.length ? ['CONFLICT', v] : v;
      };
      out({
        // both deleted b; the other writer appended c.
        tail: merge([e('a'), e('b')], [e('a')], [e('a'), e('c')]),
        // both deleted b; the other writer added c in its place.
        mid: merge([e('x'), e('b'), e('y')], [e('x'), e('y')], [e('x'), e('c'), e('y')]),
        // the user edited b; the other writer replaced it with c.
        edited: merge([e('x'), e('b')], [e('x'), e('b', 'B2')], [e('x'), e('c')]),
        // the user left b alone; the other writer fixed its pattern: in place.
        inPlace: merge([e('x'), e('b'), e('y')], [e('x'), e('b'), e('y', 'Y2')],
                       [e('x'), e('b2', 'B'), e('y')]),
      });
    """)
    assert res["tail"] == ["a>A", "c>C"]
    assert res["mid"] == ["x>X", "c>C", "y>Y"]
    assert res["edited"] == ["CONFLICT", ["x>X", "c>C", "b>B2"]]
    assert res["inPlace"] == ["x>X", "b2>B", "y>Y2"]


def test_conflict_merge_keeps_both_new_entries_around_a_moved_one(client, tmp_path):
    # After a side moved an entry up, a later gap overlapped an earlier one and
    # the same deleted entry's id went to two new entries: one of the other
    # writer's new entries was dropped under a clean "Auto-merged" toast.
    res = _run_olist_js(client, tmp_path, r"""
      const e = (p, r) => ({ pattern: p, replacement: r || p.toUpperCase() });
      const base = [e('a'), e('b'), e('c'), e('d')];
      // The other writer added X on top, deleted b, moved a after c and added
      // Y after a; the user only changed d.
      const theirs = [e('X'), e('c'), e('a'), e('Y'), e('d')];
      const clashes = [];
      const m = _applyFieldDiff(theirs, _fieldDiff(base, [e('a'), e('b'), e('c'), e('d', 'D9')],
                                                   _diffKindFor('entries')), clashes);
      const keys = _olistAlign(_olistKeyed(base), _olistKeyed(theirs)).map(([k]) => k);
      out({ merged: m.map(x => x.pattern + '>' + x.replacement).sort(), clashes,
            uniqueKeys: new Set(keys).size === keys.length });
    """)
    assert res["merged"] == ["X>X", "Y>Y", "a>A", "c>C", "d>D9"]
    assert res["clashes"] == []
    assert res["uniqueKeys"] is True


def test_conflict_merge_keeps_one_entry_when_both_writers_made_the_same_pattern_fix(client, tmp_path):
    # Both writers fixed x's pattern to y and one also changed the
    # replacement: the other writer's entry went back in under its own id as
    # well, so the list held y twice and theirs, first, shadowed the user's.
    res = _run_olist_js(client, tmp_path, _ENTRIES_MERGE_JS + r"""
      const e = (p, r) => ({ pattern: p, replacement: r });
      out({
        mineRepl: merge([e('x', '1'), e('a', '2')], [e('y', 'U'), e('a', '2')],
                        [e('y', '1'), e('a', '2')]),
        theirsRepl: merge([e('x', '1'), e('a', '2')], [e('y', '1'), e('a', '2')],
                          [e('y', 'S'), e('a', '2')]),
      });
    """)
    assert res["mineRepl"] == [["y"], ["y>U", "a>2"]]
    assert res["theirsRepl"] == [["y"], ["y>1", "a>2"]]


def test_conflict_merge_reports_the_other_writers_pattern_fix_the_user_also_added(client, tmp_path):
    # The other writer's pattern fix x -> y went back in under its own id,
    # but the user had added a y entry of their own: theirs was skipped with
    # no clash, so the next save overwrote it under an "Auto-merged" toast.
    res = _run_olist_js(client, tmp_path, _ENTRIES_MERGE_JS + r"""
      const e = (p, r) => ({ pattern: p, replacement: r });
      out({
        // the user edited x's replacement and appended y.
        edited: merge([e('a', '1'), e('x', '1')], [e('a', '1'), e('x', 'U'), e('y', '5')],
                      [e('a', '1'), e('y', '1')]),
        // the user appended the very same y entry: nothing to review there.
        same: merge([e('a', '1'), e('x', '1')], [e('a', '1'), e('x', 'U'), e('y', '1')],
                    [e('a', '1'), e('y', '1')]),
      });
    """)
    assert res["edited"] == [["x", "y"], ["a>1", "x>U", "y>5"]]
    assert res["same"] == [["x"], ["a>1", "x>U", "y>1"]]


def test_conflict_merge_reports_the_users_new_entry_shadowed_by_the_other_writers_pattern_fix(
        client, tmp_path):
    # The user left x alone and appended a y entry while the other writer
    # fixed x's pattern to y: the merge put their fix in x's place, ahead of
    # the user's y, so the user's entry never matched (or sat there twice)
    # under a clean "Auto-merged" toast.
    res = _run_olist_js(client, tmp_path, _ENTRIES_MERGE_JS + r"""
      const e = (p, r) => ({ pattern: p, replacement: r });
      const base = [e('a', 'A'), e('x', 'X'), e('z', 'Z')];
      const theirs = [e('a', 'A'), e('y', 'X'), e('z', 'Z')];
      out({
        other: merge(base, [...base, e('y', 'W')], theirs),
        same: merge(base, [...base, e('y', 'X')], theirs),
      });
    """)
    assert res["other"] == [["y"], ["a>A", "y>X", "z>Z", "y>W"]]
    assert res["same"] == [[], ["a>A", "y>X", "z>Z"]]


def test_conflict_merge_reports_the_other_writers_pattern_fix_to_an_entry_the_user_deleted(
        client, tmp_path):
    # The user deleted b while the other writer only fixed b's pattern: the
    # entry came back under a clean "Auto-merged" toast, though a deleted
    # entry the other writer changed the replacement of stays deleted.
    res = _run_olist_js(client, tmp_path, _ENTRIES_MERGE_JS + r"""
      const e = (p, r) => ({ pattern: p, replacement: r || p.toUpperCase() });
      out({
        fixed: merge([e('x'), e('b'), e('y')], [e('x'), e('y')], [e('x'), e('b2', 'B'), e('y')]),
        // ... and the user appended an entry with the fixed pattern.
        added: merge([e('a', '1'), e('x', '1'), e('b', '1')], [e('a', '1'), e('b', '1'), e('y', '5')],
                     [e('a', '1'), e('y', '1'), e('b', '1')]),
        // the other writer put a new entry in b's place: kept, nothing to review.
        replaced: merge([e('x'), e('b'), e('y')], [e('x'), e('y')], [e('x'), e('c'), e('y')]),
      });
    """)
    assert res["fixed"] == [["b2"], ["x>X", "b2>B", "y>Y"]]
    assert res["added"] == [["y"], ["a>1", "b>1", "y>5"]]
    assert res["replaced"] == [[], ["x>X", "c>C", "y>Y"]]


def test_reapply_tells_an_entry_the_user_deleted_coming_back_from_a_both_edited_one(
        client, tmp_path):
    # The deleted entry the other writer fixed the pattern of comes back in
    # their version, but its rule was reported like a both-edited clash, so
    # the toast said "your version is kept" of a deletion that was undone.
    res = _run_olist_js(client, tmp_path, r"""
      const e = (p, r) => ({ pattern: p, replacement: r || p.toUpperCase() });
      globalThis.dirty = new Set();
      globalThis.liveRules = [
        { name: 'del', entries: [e('x'), e('b2', 'B'), e('y')] },
        { name: 'both', entries: [e('foo', 'qux')] },
      ];
      const diffs = new Map([
        ['del', { entries: _fieldDiff([e('x'), e('b'), e('y')], [e('x'), e('y')], 'olist') }],
        ['both', { entries: _fieldDiff([e('foo', 'bar')], [e('foo', 'baz')], 'olist') }],
      ]);
      const clashed = [], restored = [];
      _reapplyDiffs(diffs, clashed, restored);
      const clashes = [], back = [];
      _applyFieldDiff([e('x'), e('b2', 'B'), e('y')],
                      _fieldDiff([e('x'), e('b'), e('y')], [e('x'), e('y')], 'olist'), clashes, back);
      out({ clashed, restored, clashes, back });
    """)
    assert res["clashed"] == ["both"]
    assert res["restored"] == ["del"]
    assert res["clashes"] == ["b2"] and res["back"] == ["b2"]


def test_reapply_keeps_a_both_edited_clash_when_the_users_entry_keeps_the_other_writers_fix_out(
        client, tmp_path):
    # The user deleted b and appended their own b2, so the other writer's
    # pattern fix b2 never went in, yet the toast said "their entry is back,
    # delete it again" (pointing at the user's own entry) instead of "your
    # version is kept"; the same b2 on both sides was a clash too.
    res = _run_olist_js(client, tmp_path, r"""
      const e = (p, r) => ({ pattern: p, replacement: r || p.toUpperCase() });
      const base = [e('x'), e('b'), e('y')];
      const theirs = [e('x'), e('b2', 'B'), e('y')];
      const merge = (mine) => {
        const clashes = [], back = [];
        const m = _applyFieldDiff(theirs, _fieldDiff(base, mine, 'olist'), clashes, back);
        return [clashes, back, m.map(x => x.pattern + '>' + x.replacement)];
      };
      globalThis.dirty = new Set();
      globalThis.liveRules = [{ name: 'kept', entries: theirs }];
      const clashed = [], restored = [];
      _reapplyDiffs(new Map([['kept', { entries: _fieldDiff(base, [e('x'), e('y'), e('b2', 'W')],
                                                            'olist') }]]), clashed, restored);
      out({
        other: merge([e('x'), e('y'), e('b2', 'W')]),
        same: merge([e('x'), e('y'), e('b2', 'B')]),
        clashed, restored,
      });
    """)
    assert res["other"] == [["b2"], [], ["x>X", "y>Y", "b2>W"]]
    assert res["same"] == [[], [], ["x>X", "y>Y", "b2>B"]]
    assert res["clashed"] == ["kept"] and res["restored"] == []


def test_reapply_reports_a_scalar_both_writers_changed_differently_as_a_clash(
        client, tmp_path):
    # A callback rule's pattern is a scalar: the merge kept the user's value
    # over the other writer's different one and the toast called the rule
    # "Auto-merged", so their edit was dropped by the next save unseen.
    res = _run_olist_js(client, tmp_path, r"""
      globalThis.dirty = new Set();
      const reapply = (theirs) => {
        globalThis.liveRules = [{ name: 'lw', type: 'callback:lowercase-wordlist',
                                  enabled: true, pattern: theirs }];
        const clashed = [], restored = [];
        _reapplyDiffs(new Map([['lw', { pattern: _fieldDiff('a', 'b', _diffKindFor('pattern')) }]]),
                      clashed, restored);
        return [liveRules[0].pattern, clashed, restored];
      };
      out({ other: reapply('c'), same: reapply('b'), untouched: reapply('a') });
    """)
    assert res["other"] == ["b", ["lw"], []]
    assert res["same"] == ["b", [], []]
    assert res["untouched"] == ["b", [], []]


_RESTORED_TOAST = ("another writer changed an entry you deleted; their entry is back,"
                   " delete it again if you still want it gone")


def test_save_toasts_say_an_entry_the_user_deleted_is_back(client, tmp_path):
    # Every toast naming a clash said "your version is kept", also for a
    # deleted entry the merge brought back in the other writer's version.
    res = _run_save_js(client, tmp_path, r"""
      const toasts = {};
      globalThis._isAdmin = () => true;
      globalThis._detailText = (j, msg) => j.detail;
      let calls = 0;
      globalThis._reapplyDiffs = (d, clashed, restored) => {
        if (!calls++) restored.push('rx'); return 1;
      };
      const run = async (name, resp) => {
        calls = 0; _toasts.length = 0;
        globalThis.api = async () => resp;
        await _doSave({ rx: {} }, {});
        toasts[name] = _toasts[_toasts.length - 1][1];
      };
      await run('merged', { ok: true, status: 200,
                            json: async () => ({ saved: [], conflicts: [{ slug: 'rx' }] }) });
      await run('inflight', { ok: true, status: 200,
                              json: async () => ({ saved: ['rx'], conflicts: [] }) });
      await run('refused', { ok: false, status: 403, json: async () => ({ detail: 'locked' }) });
      calls = 1;  // the conflicted rule took no diff, the in-flight one did
      globalThis._reapplyDiffs = (d, clashed, restored) => {
        if (calls++ === 2) { restored.push('rx'); return 1; } return 0;
      };
      globalThis.api = async () => ({ ok: true, status: 200,
                                      json: async () => ({ saved: [], conflicts: [{ slug: 'rx' }] }) });
      _toasts.length = 0;
      await _doSave({ rx: {} }, {});
      toasts.unmerged = _toasts[_toasts.length - 1][1];
      out(toasts);
    """)
    for name in ("merged", "inflight", "refused", "unmerged"):
        msg = res[name]
        assert _RESTORED_TOAST in msg, name
        assert "your version is kept" not in msg, name
        assert "Auto-merged" not in msg, name


def test_reapply_keeps_the_users_edits_on_a_rule_whose_entry_both_writers_edited(client, tmp_path):
    # One entry both writers edited made _reapplyDiffs skip the whole rule:
    # every other unsaved edit on it (added entries, the enabled toggle, other
    # entries) vanished with no undo. Its slug is reported so the toast can
    # name the rule.
    res = _run_olist_js(client, tmp_path, r"""
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
    """)
    assert res["merged"] == 2
    assert res["clashed"] == ["rx"]
    assert res["dirty"] == ["ok", "rx"]
    assert res["rules"][0] == {"name": "rx", "enabled": False, "entries": [
        {"pattern": "foo", "replacement": "baz"}, {"pattern": "a", "replacement": "A"},
        {"pattern": "n", "replacement": "N"}]}
    assert res["rules"][1]["entries"] == [{"pattern": "a", "replacement": "A2"}]


def test_reapply_reports_a_dictionary_key_both_writers_changed(client, tmp_path):
    # The dictionary merge laid the user's keys over the fresh map without
    # looking at it: a key the other writer had also changed (or also added)
    # to another value was overwritten under a clean "Auto-merged" toast, and
    # the next save lost their edit. It keeps the user's value and reports
    # the rule, as a regex-list entry both writers edited does.
    res = _run_olist_js(client, tmp_path, r"""
      globalThis.dirty = new Set();
      const base = { komma: ',', punkt: '.', fragezeichen: '?' };
      globalThis.liveRules = [
        { name: 'both', map: { komma: ', ', punkt: '.', fragezeichen: '?', neu: 'A' } },
        { name: 'same', map: { komma: ';', punkt: '.', fragezeichen: '?' } },
        { name: 'theirs', map: { komma: ',', punkt: '. ', fragezeichen: '?' } },
        { name: 'gone', map: { punkt: '.', fragezeichen: '?' } },
      ];
      const mine = { komma: ';', punkt: '.', fragezeichen: '?', neu: 'B' };
      const diffs = new Map(['both', 'same', 'theirs', 'gone'].map(
        s => [s, { map: _fieldDiff(base, s === 'both' ? mine : { ...base, komma: ';' },
                                   _diffKindFor('map')) }]));
      const clashed = [];
      const merged = _reapplyDiffs(diffs, clashed, []);
      const keys = [];
      _applyFieldDiff({ komma: ', ', neu: 'A' }, _fieldDiff(base, mine, 'dict'), keys);
      out({ merged, clashed, keys, maps: liveRules.map(r => r.map) });
    """)
    assert res["merged"] == 4
    assert res["clashed"] == ["both"]
    assert sorted(res["keys"]) == ["komma", "neu"]
    assert res["maps"][0] == {"komma": ";", "punkt": ".", "fragezeichen": "?", "neu": "B"}
    assert res["maps"][1]["komma"] == ";"
    assert res["maps"][2] == {"komma": ";", "punkt": ". ", "fragezeichen": "?"}
    assert res["maps"][3] == {"komma": ";", "punkt": ".", "fragezeichen": "?"}


def test_reapply_brings_back_a_dictionary_key_the_user_deleted_and_the_other_writer_changed(
        client, tmp_path):
    # The dictionary merge deleted every key the user removed from the fresh
    # map: a key the other writer had changed the value of was lost under a
    # clean "Auto-merged" toast, and so was their value under the user's
    # rename of that key (a remove plus an add). Their key is back and the
    # rule reported, as for a regex-list entry the user deleted.
    res = _run_olist_js(client, tmp_path, r"""
      globalThis.dirty = new Set();
      const base = { komma: ',', punkt: '.' };
      globalThis.liveRules = [
        { name: 'changed', map: { komma: ', ', punkt: '.' } },
        { name: 'renamed', map: { komma: ', ', punkt: '.' } },
        { name: 'clean', map: { komma: ',', punkt: '. ' } },
      ];
      const diffs = new Map([
        ['changed', { map: _fieldDiff(base, { punkt: '.' }, _diffKindFor('map')) }],
        ['renamed', { map: _fieldDiff(base, { Komma: ',', punkt: '.' }, 'dict') }],
        ['clean', { map: _fieldDiff(base, { punkt: '.' }, 'dict') }],
      ]);
      const clashed = [], restored = [];
      const merged = _reapplyDiffs(diffs, clashed, restored);
      const clashes = [], back = [];
      _applyFieldDiff({ komma: ', ', punkt: '.' }, _fieldDiff(base, { punkt: '.' }, 'dict'),
                      clashes, back);
      out({ merged, clashed, restored, clashes, back, maps: liveRules.map(r => r.map) });
    """)
    assert res["merged"] == 3
    # A rename that only recases the key keeps the user's key alone (the
    # server refuses komma next to Komma): a clash, not a restore.
    assert res["clashed"] == ["renamed"]
    assert res["restored"] == ["changed"]
    assert res["clashes"] == ["komma"] and res["back"] == ["komma"]
    assert res["maps"][0] == {"komma": ", ", "punkt": "."}
    assert res["maps"][1] == {"Komma": ",", "punkt": "."}
    assert res["maps"][2] == {"punkt": ". "}


def test_conflict_merge_deletes_a_dictionary_key_the_other_writer_only_recased(client, tmp_path):
    # The removed-keys loop looked the user's deletion up by exact key: komma
    # deleted while the other writer recased it to Komma kept their Komma
    # under a clean "Auto-merged" toast, and the next save stored the word
    # the user deleted. A pure recase takes the deletion; a recase with a
    # changed value is back and reported, as for the same spelling.
    res = _run_olist_js(client, tmp_path, r"""
      const base = { komma: ',', punkt: '.' };
      const d = _fieldDiff(base, { punkt: '.' }, 'dict');
      const res = {};
      for (const [name, theirs] of [['recased', { Komma: ',', punkt: '.' }],
                                    ['changed', { Komma: ';', punkt: '.' }]]) {
        const clashes = [], back = [];
        res[name] = { map: _applyFieldDiff(theirs, d, clashes, back), clashes, back };
      }
      out(res);
    """)
    assert res["recased"] == {"map": {"punkt": "."}, "clashes": [], "back": []}
    assert res["changed"] == {"map": {"Komma": ";", "punkt": "."},
                              "clashes": ["Komma"], "back": ["Komma"]}


def test_reapply_keeps_one_spelling_of_a_dictionary_key_both_writers_added_in_another_case(
        client, tmp_path):
    # The dictionary merge matched keys case-sensitively: the user adding
    # Komma while the other writer added komma merged to a map holding both,
    # under a clean "Auto-merged" toast, and the next save got a 422 (map
    # keys collide when lowercased). The user's key is kept and the rule
    # reported, as for a key both writers added with different values.
    res = _run_olist_js(client, tmp_path, r"""
      globalThis.dirty = new Set();
      const base = { punkt: '.' };
      globalThis.liveRules = [
        { name: 'added', map: { komma: ', ', punkt: '.' } },
        { name: 'recased', map: { KOMMA: ',', punkt: '.' } },
      ];
      const diffs = new Map([
        ['added', { map: _fieldDiff(base, { Komma: ',', punkt: '.' }, _diffKindFor('map')) }],
        ['recased', { map: _fieldDiff({ komma: ',', punkt: '.' }, { Komma: ',', punkt: '.' },
                                      'dict') }],
      ]);
      const clashed = [], restored = [];
      const merged = _reapplyDiffs(diffs, clashed, restored);
      const plain = _applyFieldDiff({ komma: ', ' }, _fieldDiff(base, { Komma: ',' }, 'dict'));
      out({ merged, clashed, restored, plain, maps: liveRules.map(r => r.map) });
    """)
    assert res["merged"] == 2
    assert res["clashed"] == ["added", "recased"]
    assert res["restored"] == []
    assert res["maps"][0] == {"Komma": ",", "punkt": "."}
    assert res["maps"][1] == {"Komma": ",", "punkt": "."}
    assert res["plain"] == {"Komma": ","}


def test_conflict_toast_names_the_rule_that_could_not_be_merged(client, tmp_path):
    res = _run_save_js(client, tmp_path, r"""
      globalThis.api = async () => ({
        ok: true, status: 200,
        json: async () => ({ saved: [], conflicts: [{ slug: 'rx' }, { slug: 'ok' }] }),
      });
      let calls = 0;  // the conflicted rules' re-apply, then the in-flight one
      globalThis._reapplyDiffs = (d, clashed) => { if (!calls++) clashed.push('rx'); return 1; };
      await _doSave({ rx: {}, ok: {} }, {});
      out(_toasts);
    """)
    msg = res[0][1]
    assert "Auto-merged your changes for ok on top" in msg
    assert "Conflict on rx: another writer also edited an entry you edited; your version is kept." in msg
    assert "Review and Save again." in msg


def test_save_toasts_a_clash_on_an_edit_typed_while_the_post_was_in_flight(client, tmp_path):
    # The in-flight re-apply passed no `clashed` array, so an edit typed
    # during the POST that another writer had also made went unreported.
    res = _run_save_js(client, tmp_path, r"""
      globalThis.api = async () => ({
        ok: true, status: 200, json: async () => ({ saved: ['r1'], conflicts: [] }),
      });
      globalThis._reapplyDiffs = (d, clashed) => { clashed.push('rx'); return 1; };
      await _doSave({ r1: {} }, {});
      out(_toasts);
    """)
    assert res[-1][0] == "err"
    assert "Conflict on rx: another writer also edited an entry you edited" in res[-1][1]
