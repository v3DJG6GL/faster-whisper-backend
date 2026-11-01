"""Behaviour of the /settings + /settings/pipeline inline script
(admin/templates/settings.html).

Where a piece of logic can be cut out of the served page it runs under node
against small stubs (same approach as
tests/quick_config/test_pipeline_popover_color.py); the rest pins the served
source like test_routes_settings.py does."""

import json
import shutil
import subprocess

import pytest

_NODE = shutil.which("node")
needs_node = pytest.mark.skipif(_NODE is None, reason="node unavailable")


def _html(client, url="/settings"):
    r = client.get(url)
    assert r.status_code == 200
    return r.text


def _block(html, start, end="\n  }\n"):
    """Source from `start` up to and including the first `end` after it."""
    i = html.index(start)
    j = html.index(end, i)
    return html[i:j + len(end)]


def _node(script):
    """Run `script` under node; its last stdout line is JSON."""
    p = subprocess.run([_NODE, "-"], input=script, capture_output=True,
                       text=True, encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


# Just enough DOM for the builders below: elements record children, classes,
# dataset and listeners; fire(el, type) runs the listeners; find(el, pred)
# walks the tree.
_FAKE_DOM = r"""
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = []; this.dataset = {}; this.style = {};
    this.listeners = {}; this.attrs = {}; this.textContent = '';
    this.disabled = false; this.parentElement = null;
    const cls = new Set();
    this._cls = cls;
    this.classList = {
      add: (...c) => c.forEach(x => cls.add(x)),
      remove: (...c) => c.forEach(x => cls.delete(x)),
      toggle: (c, f) => { if (f === undefined) f = !cls.has(c);
                          if (f) cls.add(c); else cls.delete(c); return f; },
      contains: (c) => cls.has(c),
    };
  }
  set className(v) { this._cls.clear(); String(v).split(/\s+/).filter(Boolean).forEach(c => this._cls.add(c)); }
  get className() { return [...this._cls].join(' '); }
  set innerHTML(v) { this.children = []; this._html = v; }
  get innerHTML() { return this._html || ''; }
  appendChild(c) { this.children.push(c); c.parentElement = this; return c; }
  append(...cs) { cs.forEach(c => this.appendChild(c)); }
  replaceChildren(...cs) { this.children = []; this.append(...cs); }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
  focus() {}
}
const document = {
  createElement: (t) => new El(t),
  createTextNode: (t) => { const n = new El('#text'); n.textContent = t; return n; },
  getElementById: () => null,
  querySelector: () => null,
};
async function fire(el, type, ev) {
  for (const f of (el.listeners[type] || [])) await f(ev || { target: el });
}
function find(el, pred) {
  if (pred(el)) return el;
  for (const c of el.children) { const r = find(c, pred); if (r) return r; }
  return null;
}
"""


def _captures_harness(html, body, real_list=False):
    """makeEditor()'s captures branch. By default makeRuleListEditor is
    stubbed out and `editor` holds the opts the checklist was built with;
    `real_list` runs the page's own makeRuleListEditor on the fake DOM."""
    branch = _block(html, "  if (name === 'CAPTURES_PIPELINE_RULES_EXCLUDE') {")
    if real_list:
        rle = _block(html, "function makeRuleListEditor(", "\n}\n")
        rle += """
function fieldDef(n) { return { value: state[n], default_value: [] }; }
function _lsGet() { return null; }
function _typePill(t) { return t; }
"""
    else:
        rle = """
function makeRuleListEditor(name, rules, mode, opts) {
  editor = { name, rules, mode, opts };
  return editor;
}
"""
    return _FAKE_DOM + f"""
const state = {{}};
const listeners = {{}};
let editor = null;
function currentValue(n) {{ return state[n]; }}
function setDirty(n, v) {{ state[n] = v; }}
{rle}
function jumpToGlobalRule(slug) {{}}
function _registerAdminListener(ev, key, fn) {{ listeners[ev + '|' + key] = fn; }}
function makeEditor(name) {{
  const v = currentValue(name);
{branch}
  return null;
}}
{body}
"""


@needs_node
def test_captures_exclude_toggle_back_keeps_server_order(client):
    """The server sends the set sorted; re-excluding a default-excluded rule
    appended it at the end, so the field stayed dirty with the same members."""
    out = _node(_captures_harness(_html(client), """
state.PIPELINE_RULES = [{name: 'a'}, {name: 'b'}, {name: 'c'}];
state.CAPTURES_PIPELINE_RULES_EXCLUDE = ['a', 'c'];
makeEditor('CAPTURES_PIPELINE_RULES_EXCLUDE');
editor.opts.onToggle('a', true, false);
editor.opts.onToggle('a', false, false);
console.log(JSON.stringify(state.CAPTURES_PIPELINE_RULES_EXCLUDE));
"""))
    assert out == ["a", "c"]


@needs_node
def test_captures_checklist_offers_no_force_include(client):
    """Captures have no include list: ticking a globally-disabled rule showed
    'force-included' while nothing was saved and the rule stayed off."""
    out = _node(_captures_harness(_html(client), """
state.PIPELINE_RULES = [
  { name: 'on', type: 'regex', enabled: true },
  { name: 'off', type: 'regex', enabled: false },
];
state.CAPTURES_PIPELINE_RULES_EXCLUDE = [];
const wrap = makeEditor('CAPTURES_PIPELINE_RULES_EXCLUDE');
const cb = slug => find(find(wrap, e => e.dataset.slug === slug),
                        e => e.tagName === 'INPUT');
console.log(JSON.stringify([cb('on').disabled, cb('off').disabled]));
""", real_list=True))
    assert out == [False, True]


@needs_node
def test_captures_checklist_footer_follows_a_toggle(client):
    """The footer counted from a Set that the captures onToggle never
    touched, so 'N active' stayed at its first-paint value."""
    out = _node(_captures_harness(_html(client), """
state.PIPELINE_RULES = [
  { name: 'a', type: 'regex', enabled: true },
  { name: 'b', type: 'regex', enabled: true },
];
state.CAPTURES_PIPELINE_RULES_EXCLUDE = [];
const wrap = makeEditor('CAPTURES_PIPELINE_RULES_EXCLUDE');
const footer = find(wrap, e => e.classList.contains('rule-checklist-footer'));
const cb = find(find(wrap, e => e.dataset.slug === 'a'), e => e.tagName === 'INPUT');
const before = footer.textContent;
cb.checked = false;
fire(cb, 'change').then(() => {
  console.log(JSON.stringify([before, footer.textContent,
                              state.CAPTURES_PIPELINE_RULES_EXCLUDE]));
});
""", real_list=True))
    assert out == ["2 rules total · 2 active for this model",
                   "2 rules total · 1 active for this model", ["a"]]


def test_model_overrides_rerender_keeps_env_pinned_editor_disabled(client):
    """fieldRow() disabled an env-pinned MODEL_OVERRIDES editor once; a sidebar
    click, a model-list change or a PIPELINE_RULES edit rebuilt enabled
    controls whose edits setDirty() then dropped without a word."""
    html = _html(client)
    mo = _block(html, "function modelOverridesEditor(name, v) {", "\n}\n")
    assert "if (isEnvPinned(name)) disableEnvPinnedEditor(wrap);" in mo
    for fn in ("function renderSidebar() {", "function renderMain() {",
               "function refreshPipelineSection() {"):
        assert "keepPinned();" in _block(mo, fn), fn


@needs_node
def test_model_override_seeds_from_the_current_global_value(client):
    """'+ override' seeded the global value captured when the row was built;
    an unsaved global edit later only refreshed the 'inherits' hint."""
    row_fn = _block(_html(client), "  function renderFieldRow(field) {")
    out = _node(_FAKE_DOM + """
let selectedId = 'm';
const FIELD_META = { BEAM_SIZE: { kind: 'int' } };
const LOAD_TIME_FIELDS = new Set();
const state = { BEAM_SIZE: 5 };
function globalValue(f) { return state[f]; }
function getOverrideValue(id, f) { return undefined; }
let seeded = null;
function setOverrideValue(id, f, v) { seeded = v; }
function clearOverride() {}
function fillInherits() {}
function fmtValue(v) { return String(v); }
function defaultForKind() { return 0; }
function renderMain() {}
function renderSidebar() {}
""" + row_fn + """
const row = renderFieldRow('BEAM_SIZE');
state.BEAM_SIZE = 3;   // unsaved global edit after the row was painted
fire(find(row, e => e.textContent === '+ override'), 'click').then(() => {
  console.log(JSON.stringify(seeded));
});
""")
    assert out == 3


def _model_pipeline_harness(html, body):
    """The MODEL_OVERRIDES per-model pipeline checklist with
    makeRuleListEditor stubbed out: `editor.opts` is what the rows call."""
    section = _block(html, "  function renderPipelineSection() {")
    return _FAKE_DOM + """
const state = {};
function currentValue(n) { return state[n]; }
let selectedId = 'm';
let overrides = {};
let editor = null;
function makeRuleListEditor(name, rules, mode, opts) {
  editor = { name, rules, mode, opts };
  return document.createElement('div');
}
function persist() {}
function refreshMarkers() {}
function jumpToGlobalRule() {}
""" + section + body


@needs_node
@pytest.mark.parametrize("enabled, start, want_active, expect", [
    # Force-included while off, then switched on globally: unticking
    # force-disables it and must drop the stale INCLUDE entry.
    (True, {"PIPELINE_RULES_INCLUDE": ["x"]}, False,
     {"PIPELINE_RULES_EXCLUDE": ["x"]}),
    # Force-disabled while on, then switched off globally: ticking
    # force-includes it and must drop the stale EXCLUDE entry.
    (False, {"PIPELINE_RULES_EXCLUDE": ["x"]}, True,
     {"PIPELINE_RULES_INCLUDE": ["x"]}),
])
def test_model_pipeline_toggle_never_puts_a_slug_in_both_lists(
        client, enabled, start, want_active, expect):
    """ModelOverride rejects a slug in EXCLUDE and INCLUDE (a 422 for the whole
    save); a rule whose global state flipped after the override ended up in
    both."""
    out = _node(_model_pipeline_harness(_html(client), f"""
state.PIPELINE_RULES = [{{ name: 'x', type: 'regex', enabled: {json.dumps(enabled)} }}];
overrides = {{ m: {json.dumps(start)} }};
renderPipelineSection();
editor.opts.onToggle('x', {json.dumps(want_active)}, {json.dumps(enabled)});
console.log(JSON.stringify(overrides.m));
"""))
    assert out == expect


@needs_node
def test_rule_status_badge_ignores_a_stale_reply(client):
    """The 250 ms debounce only cancels calls not yet sent: a slow reply for
    the old pattern landed after the fresh one and repainted the badge."""
    block = _block(_html(client, "/settings/pipeline"),
                   "      let timer = null;\n", "        }, 250);\n      };\n")
    out = _node(_FAKE_DOM + """
const tick = () => new Promise(r => setTimeout(r, 0));
(async function () {
  // The debounce fires at once here; the replies resolve by hand.
  function setTimeout(f) { f(); return 1; }
  function clearTimeout() {}
  const status = document.createElement('div');
  const rule = { name: 'r', type: 'regex', enabled: true };
  const pending = [];
  function _testOneRule() { return new Promise(res => pending.push(res)); }
  let refresh;
""" + block + """
  refresh();   // backtracking pattern: its reply takes the full 2 s
  refresh();   // the fixed pattern
  pending[1]({ type: 'regex', matches: 1 });
  await tick();
  pending[0]({ type: 'regex', slow: true });
  await tick();
  console.log(JSON.stringify(status.textContent));
})();
""")
    assert out == "✓ valid · 1 match in sample"


@needs_node
def test_rule_status_names_the_passes_of_a_repeating_rule(client):
    """A step of a rule with `repeat` on names the passes that changed the
    text (test-pipeline `passes`); one that changed it once or not at all
    says nothing about passes, like a non-repeating rule's."""
    block = _block(_html(client, "/settings/pipeline"),
                   "      let timer = null;\n", "        }, 250);\n      };\n")
    out = _node(_FAKE_DOM + """
const tick = () => new Promise(r => setTimeout(r, 0));
(async function () {
  function setTimeout(f) { f(); return 1; }
  function clearTimeout() {}
  const status = document.createElement('div');
  const rule = { name: 'r', type: 'callback:upper', enabled: true, repeat: true };
  const replies = [{ type: 'callback:upper', matches: 3, passes: 3 },
                   { type: 'callback:upper', matches: 1, passes: 1 },
                   { type: 'callback:upper', matches: 0, passes: 0 }];
  function _testOneRule() { return Promise.resolve(replies.shift()); }
  let refresh;
""" + block + """
  const seen = [];
  for (let i = 0; i < 3; i++) { refresh(); await tick(); seen.push(status.textContent); }
  console.log(JSON.stringify(seen));
})();
""")
    assert out == ["✓ valid · 3 matches in sample · 3 passes", "✓ valid · 1 match in sample",
                   "✓ valid · 0 matches in sample"]


@needs_node
def test_rule_status_badge_reports_a_network_error(client):
    """api() lets fetch's rejection through; _testOneRule promised null on a
    transport error, and the badge kept its old verdict instead."""
    fn = _block(_html(client, "/settings/pipeline"),
                "async function _testOneRule(rule) {", "\n}\n")
    out = _node("""
const document = { getElementById: () => null };
const TEST_PRESETS = { 'default': 'x' };
async function api() { throw new TypeError('Failed to fetch'); }
""" + fn + """
_testOneRule({ name: 'r' }).then(
  v => console.log(JSON.stringify({ value: v })),
  e => console.log(JSON.stringify({ threw: String(e) })));
""")
    assert out == {"value": None}


@needs_node
def test_rule_drop_clears_the_drag_state(client):
    """The drop repaints the list, which detaches the source row; dragend
    (the only cleanup) does not reach a detached source, so a later
    external drag over the list still moved rules[staleIdx]."""
    drop = _block(_html(client, "/settings/pipeline"),
                  "  list.addEventListener('drop', (e) => {", "\n  });\n")
    out = _node(_FAKE_DOM + """
const DRAG_AUTOSCROLL = { stop() {} };
const list = document.createElement('div');
const placeholder = document.createElement('div');
let rules = [{ name: 'a' }, { name: 'b' }, { name: 'c' }];
const rows = rules.map(() => {
  const r = document.createElement('div'); r.className = 'rule-row'; return r;
});
list.append(rows[0], rows[1], rows[2], placeholder);   // drag 'a' to the end
list.classList.add('dnd-active');
let dragSrcEl = rows[0];
let dragSrcIdx = 0;
let committed = 0;
function commitFull() { committed++; }
""" + drop + """
fire(list, 'drop', { preventDefault() {} }).then(() => {
  console.log(JSON.stringify([rules.map(r => r.name), committed, dragSrcEl === null,
                              dragSrcIdx, list.classList.contains('dnd-active')]));
});
""")
    assert out == [["b", "c", "a"], 1, True, None, False]


def test_rule_tag_commit_keeps_the_tag_picker(client):
    """A committed tag ran commitFull(), whose repaint threw away the picker
    and its focused input mid-typing: 'foo,bar' kept only 'foo'."""
    picker = _block(_html(client, "/settings/pipeline"),
                    "const picker = window._renderTagPicker({", "\n      });\n")
    assert "commitData();" in picker
    assert "commitFull();" not in picker


def _label_editor_harness(html, body):
    """A rule card's label editor (renderRow) with the real _slugify /
    _ensureUniqueSlug, spliced into `body` at {editor}."""
    editor = _block(html, "    // Membership in config.json — not the seeded flag",
                    "      body.appendChild(labelWrap);\n    }\n")
    helpers = (_block(html, "function _slugify(s) {", "\n}\n")
               + _block(html, "function _ensureUniqueSlug(slug, existing) {", "\n}\n"))
    return _FAKE_DOM + helpers + """
const name = 'PIPELINE_RULES';
const saved = [{ name: 'my-rule', label: 'My rule', type: 'regex' }];
function fieldDef() { return { value: saved }; }
const factoryRules = [];
function _factoryHas() { return false; }
const expandedNames = new Set();
function _saveExpanded() {}
function commitData() {}
const lbl = document.createElement('span');
const slug = document.createElement('span');
const row = document.createElement('div');
""" + body.replace("{editor}", editor)


@needs_node
def test_label_edit_keeps_a_saved_rules_slug(client):
    """A saved server-only rule re-slugged on every keystroke, so a per-model,
    profile or captures reference to it 422'd the next save."""
    out = _node(_label_editor_harness(_html(client, "/settings/pipeline"), """
function relabel(rule, rules, text) {
  const body = document.createElement('div');
{editor}
  const labelInp = find(body, e => e.tagName === 'INPUT');
  labelInp.value = text;
  fire(labelInp, 'input');
  return rule.name;
}
const savedRule = { name: 'my-rule', label: 'My rule', type: 'regex' };
const fresh = { name: 'rule', label: '', type: 'regex' };
console.log(JSON.stringify([
  relabel(savedRule, [savedRule], 'Renamed rule'),
  relabel(fresh, [savedRule, fresh], 'Brand new'),
]));
"""))
    assert out == ["my-rule", "brand-new"]


@needs_node
def test_reset_order_rereads_config_json(client):
    """Reset order sorted by the order cached at page load, so after another
    admin's promote (or a git pull) it restored the old order."""
    handler = _block(_html(client, "/settings/pipeline"),
                     "  resetOrderBtn.addEventListener('click', ", "\n  });\n")
    out = _node(_FAKE_DOM + """
const resetOrderBtn = document.createElement('button');
let factoryRules = [{ name: 'a' }, { name: 'b' }];        // cached at load
async function _refreshFactory() {
  factoryRules = [{ name: 'b' }, { name: 'a' }];          // config.json now
  return true;
}
function _factoryHas(n) { return factoryRules.some(b => b.name === n); }
let rules = [{ name: 'a' }, { name: 'b' }, { name: 'trim', type: 'terminal' }];
function commitFull() {}
""" + handler + """
fire(resetOrderBtn, 'click').then(() => {
  console.log(JSON.stringify(rules.map(r => r.name)));
});
""")
    assert out == ["b", "a", "trim"]


@needs_node
def test_unique_slug_suffix_stays_within_64_chars(client):
    """_slugify caps at 64 chars, then the '-2' suffix pushed a second rule
    with the same long label past RuleSlug's max_length (422 on save)."""
    fn = _block(_html(client, "/settings/pipeline"),
                "function _ensureUniqueSlug(slug, existing) {", "\n}\n")
    out = _node(fn + """
const base = 'a'.repeat(64);
const two = _ensureUniqueSlug(base, new Set([base]));
const three = _ensureUniqueSlug(base, new Set([base, two]));
console.log(JSON.stringify([two, three, _ensureUniqueSlug('x', new Set(['x']))]));
""")
    assert out == ["a" * 62 + "-2", "a" * 62 + "-3", "x-2"]


@needs_node
def test_label_editor_respects_rule_label_bounds(client):
    """RuleLabel is 1-80 chars: the card's label input took 81+ and stored a
    cleared label as '', and either one failed the whole save."""
    out = _node(_label_editor_harness(_html(client, "/settings/pipeline"), """
const body = document.createElement('div');
const rule = { name: 'de-tidy', label: 'Tidy', type: 'regex', seeded: true };
const rules = [rule];
{editor}
const labelInp = find(body, e => e.tagName === 'INPUT');
labelInp.value = '';
fire(labelInp, 'input');
console.log(JSON.stringify([labelInp.maxLength, rule.label]));
"""))
    assert out == [80, "de-tidy"]


def test_add_rule_label_input_is_capped(client):
    html = _html(client, "/settings/pipeline")
    i = html.index("labelInp.placeholder = 'Friendly label (e.g.")
    assert "labelInp.maxLength = 80;" in html[i:html.index("body.appendChild(", i)]


def _test_panel_harness(html, body):
    """The pipeline test panel's run() (plus the helpers declared beside it)
    against a hand-driven api()."""
    panel = _block(html, "  function testBadge(kind, text) {",
                   "  runBtn.addEventListener('click', run);\n")
    return _FAKE_DOM + """
const out = document.createElement('div');
const runBtn = document.createElement('button');
const sample = { value: 'Hallo' };
const langSel = { value: '' };
function currentValue() { return []; }
function _typePill(t) { return t; }
let api;
""" + panel + body


@needs_node
def test_pipeline_test_panel_keeps_the_newest_run(client):
    """A slow earlier run that finished last replaced the newer table."""
    out = _node(_test_panel_harness(_html(client, "/settings/pipeline"), """
const pending = [];
api = () => new Promise(res => pending.push(res));
const reply = final => ({ ok: true, json: async () => ({ steps: [], final }) });
const tick = () => new Promise(r => setTimeout(r, 0));
(async function () {
  const a = run();   // slow run for the old sample
  const b = run();   // re-run after editing the sample
  pending[1](reply('NEW'));
  await b;
  pending[0](reply('OLD'));
  await a;
  await tick();
  const cells = [];
  (function walk(e) { if (e.className === 'out') cells.push(e); e.children.forEach(walk); })(out);
  console.log(JSON.stringify(cells.map(c => c.textContent)));
})();
"""))
    assert out == ["NEW"]


@needs_node
def test_pipeline_test_panel_reports_a_network_error(client):
    """fetch's rejection escaped run(), leaving 'running…' up for good."""
    out = _node(_test_panel_harness(_html(client, "/settings/pipeline"), """
api = async () => { throw new TypeError('Failed to fetch'); };
run().catch(() => {}).then(() => {
  const em = out.children[0];
  console.log(JSON.stringify(em ? [em.className, em.textContent] : out.innerHTML));
});
"""))
    assert out == ["err", "test endpoint unreachable: TypeError: Failed to fetch"]


@needs_node
def test_pipeline_test_panel_sends_the_sample_language(client):
    """The engine skips a rule whose `languages` leave out the detected
    language, but the panel sent no language, so "Run all enabled rules" and
    each card's status ran German-only rules on every sample. The panel now
    starts at DEFAULT_LANGUAGE and says which steps the language skipped."""
    html = _html(client, "/settings/pipeline")
    src = "\n".join([
        _block(html, "function pipelineTestPanel() {", "\n}\n"),
        _block(html, "async function _testOneRule(rule) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const TEST_PRESETS = { 'default': 'Hallo' };
const WHISPER_LANGS = [{ code: 'de', name: 'German' }, { code: 'en', name: 'English' }];
const state = { fields: { DEFAULT_LANGUAGE: { value: 'de' } } };
function currentValue(n) { return n === 'PIPELINE_RULES' ? [] : state.fields[n].value; }
function _typePill(t) { return t; }
const sent = [];
async function api(m, url, body) {
  sent.push(body.language);
  return { ok: true, json: async () => ({ final: 'Hallo', steps: [
    { ordinal: 1, label: 'de-only', type: 'regex', before: 'Hallo', after: 'Hallo',
      skipped: true, skipped_language: true }] }) };
}
""" + src + """
(async () => {
  const wrap = pipelineTestPanel();
  const langSel = find(wrap, e => e.id === 'pipeline-test-language');
  const runBtn = find(wrap, e => e.textContent === 'Run all enabled rules');
  const first = langSel.value;
  await fire(runBtn, 'click');
  const badge = find(wrap, e => e.classList.contains('tag'));
  langSel.value = '';
  await fire(runBtn, 'click');
  // A card's own status check reads the panel's language too.
  document.getElementById = id => id === 'pipeline-test-language' ? { value: 'en' } : null;
  await _testOneRule({ name: 'r' });
  console.log(JSON.stringify([first, sent, badge && badge.textContent]));
})();
""")
    assert out == ["de", ["de", None, "en"], "skipped · language"]


@needs_node
def test_rule_status_says_the_language_skipped_it(client):
    """A step the sample's language skipped read "∅ empty pattern — rule
    skipped" on an enabled rule whose pattern is anything but empty."""
    block = _block(_html(client, "/settings/pipeline"),
                   "      let timer = null;\n", "        }, 250);\n      };\n")
    out = _node(_FAKE_DOM + """
document.getElementById = id => id === 'pipeline-test-language' ? { value: 'en' } : null;
(async function () {
  function setTimeout(f) { f(); return 1; }
  function clearTimeout() {}
  const status = document.createElement('div');
  const rule = { name: 'r', type: 'regex', enabled: true, languages: ['de', 'es'] };
  async function _testOneRule() { return { type: 'regex', skipped: true, skipped_language: true }; }
  let refresh;
""" + block + """
  refresh();
  await new Promise(r => globalThis.setTimeout(r, 0));
  console.log(JSON.stringify(status.textContent));
})();
""")
    assert out == ("∅ skipped — language: the rule runs only for de, es, "
                   "the test sample is en")


@needs_node
def test_nullable_number_survives_a_partial_number(client):
    """A number input reads '' while it holds '-' (LOG_PROB_THRESHOLD is
    always negative) or is cleared to retype; that nulled the field and
    disabled the input under the cursor."""
    fn = _block(_html(client), "function nullableNumberEditor(name, v, fallback = 0) {", "\n}\n")
    out = _node(_FAKE_DOM + """
const state = { LOG_PROB_THRESHOLD: -1.0 };
function currentValue(n) { return state[n]; }
function setDirty(n, v) { state[n] = v; }
""" + fn + """
const wrap = nullableNumberEditor('LOG_PROB_THRESHOLD', -1.0);
const i = find(wrap, e => e.tagName === 'INPUT');
(async function () {
  i.value = '';                 // the admin typed '-' over the selection
  await fire(i, 'input');
  const typing = [state.LOG_PROB_THRESHOLD, i.disabled];
  i.value = '-0.8';
  await fire(i, 'input');
  const typed = state.LOG_PROB_THRESHOLD;
  i.value = '';                 // cleared, then left
  await fire(i, 'input');
  await fire(i, 'blur');
  console.log(JSON.stringify([typing, typed, state.LOG_PROB_THRESHOLD, i.value]));
})();
""")
    assert out == [[-1.0, False], -0.8, -0.8, -0.8]


@needs_node
def test_null_default_number_field_never_saves_an_empty_string(client):
    """DIARIZATION_NUM_SPEAKERS (int | None, default null) has no runtime
    type while null: it got a text box, and typing '3' then clearing it
    saved '' (AdminConfig 422s, taking every other dirty field with it).
    Once saved, the next render gave it a plain number box that ignores a
    clear, so 'auto' (null) was unreachable."""
    html = _html(client)
    src = "\n".join([
        _block(html, "const NULLABLE_NUMBER_FIELDS = {", "\n};\n"),
        _block(html, "function makeEditor(name) {", "\n}\n"),
        _block(html, "function nullableNumberEditor(name, v, fallback = 0) {", "\n}\n"),
        _block(html, "function stringEditor(name, v) {", "\n}\n"),
        _block(html, "function numberEditor(name, v) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const state = { DIARIZATION_NUM_SPEAKERS: null };
const seen = [];
function currentValue(n) { return state[n]; }
function setDirty(n, v) { state[n] = v; seen.push(v); }
function fieldDef(n) { return { value: null, default_value: null, choices: null }; }
""" + src + """
const input = (ed) => find(ed, e => e.tagName === 'INPUT');
const button = (ed) => find(ed, e => e.tagName === 'BUTTON');
(async function () {
  let ed = makeEditor('DIARIZATION_NUM_SPEAKERS');
  const fresh = input(ed).disabled;
  if (button(ed)) await fire(button(ed), 'click');   // enable
  const enabled = state.DIARIZATION_NUM_SPEAKERS;
  input(ed).value = '3';
  await fire(input(ed), 'input');
  input(ed).value = '';
  await fire(input(ed), 'input');
  await fire(input(ed), 'blur');
  ed = makeEditor('DIARIZATION_NUM_SPEAKERS');      // re-render with 3 saved
  if (button(ed)) await fire(button(ed), 'click');   // disable → auto
  console.log(JSON.stringify([fresh, enabled, seen, state.DIARIZATION_NUM_SPEAKERS]));
})();
""")
    assert out == [True, 2, [2, 3, None], None]


@needs_node
def test_threshold_with_a_number_default_gets_no_disable_button(client):
    """A save drops a null NO_SPEECH_THRESHOLD (exclude_none), so its
    config.json default 0.6 comes back: a "disable" button there looked like
    it switched the check off while the filter kept running. Only a field
    whose null can stick (null default, or null now) keeps the toggle."""
    html = _html(client)
    src = "\n".join([
        _block(html, "const NULLABLE_NUMBER_FIELDS = {", "\n};\n"),
        _block(html, "function makeEditor(name) {", "\n}\n"),
        _block(html, "function nullableNumberEditor(name, v, fallback = 0) {", "\n}\n"),
        _block(html, "function numberEditor(name, v) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const state = { NO_SPEECH_THRESHOLD: 0.6, LOG_PROB_THRESHOLD: null,
                HALLUCINATION_SILENCE_THRESHOLD: 2 };
const defaults = { NO_SPEECH_THRESHOLD: 0.6, LOG_PROB_THRESHOLD: -1.0,
                   HALLUCINATION_SILENCE_THRESHOLD: null };
function currentValue(n) { return state[n]; }
function setDirty(n, v) { state[n] = v; }
function fieldDef(n) { return { value: state[n], default_value: defaults[n], choices: null }; }
""" + src + """
const hasButton = (n) => !!find(makeEditor(n), e => e.tagName === 'BUTTON');
console.log(JSON.stringify(Object.keys(state).map(hasButton)));
""")
    assert out == [False, True, True]


@needs_node
def test_multiline_text_fields_get_a_textarea_that_keeps_newlines(client):
    """Only DEFAULT_PROMPT got a <textarea>: the glossary (one pair per line)
    and hotwords got a text <input>, which strips the newlines, so a pasted
    multi-line glossary saved as one line and every pair after the first
    was gone."""
    html = _html(client)
    src = "\n".join([
        _block(html, "const NULLABLE_NUMBER_FIELDS = {", "\n};\n"),
        _block(html, "const MULTILINE_TEXT_FIELDS = ", ";\n"),
        _block(html, "function makeEditor(name) {", "\n}\n"),
        _block(html, "function stringEditor(name, v) {", "\n}\n"),
        _block(html, "function textareaEditor(name, v) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const state = { TRANSLATION_GLOSSARY: 'Niere = kidney\\nLeber = liver',
                DEFAULT_HOTWORDS: 'a\\nb', DEFAULT_PROMPT: 'p\\nq',
                DEFAULT_LANGUAGE: 'de' };
function currentValue(n) { return state[n]; }
function setDirty(n, v) { state[n] = v; }
function fieldDef(n) { return { value: state[n], default_value: '', choices: null }; }
""" + src + """
(async function () {
  const out = {};
  for (const n of Object.keys(state)) {
    const ed = makeEditor(n);
    out[n] = [ed.tagName, ed.value];
  }
  const g = makeEditor('TRANSLATION_GLOSSARY');
  g.value += '\\nMilz = spleen';
  await fire(g, 'input');
  out.saved = state.TRANSLATION_GLOSSARY;
  console.log(JSON.stringify(out));
})();
""")
    assert out == {
        "TRANSLATION_GLOSSARY": ["TEXTAREA", "Niere = kidney\nLeber = liver"],
        "DEFAULT_HOTWORDS": ["TEXTAREA", "a\nb"],
        "DEFAULT_PROMPT": ["TEXTAREA", "p\nq"],
        "DEFAULT_LANGUAGE": ["INPUT", "de"],
        "saved": "Niere = kidney\nLeber = liver\nMilz = spleen",
    }


@needs_node
def test_env_pinned_plain_field_cannot_be_typed_into(client):
    """disableEnvPinnedEditor() only disabled controls INSIDE the editor, but a
    plain text/textarea field's editor is the bare control: an env-pinned
    DEFAULT_PROMPT stayed editable and setDirty() dropped the edit unseen."""
    html = _html(client)
    src = "\n".join([
        _block(html, "const NULLABLE_NUMBER_FIELDS = {", "\n};\n"),
        _block(html, "const MULTILINE_TEXT_FIELDS = ", ";\n"),
        _block(html, "function makeEditor(name) {", "\n}\n"),
        _block(html, "function stringEditor(name, v) {", "\n}\n"),
        _block(html, "function textareaEditor(name, v) {", "\n}\n"),
        _block(html, "function disableEnvPinnedEditor(el) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const CONTROL = ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'];
El.prototype.matches = function () { return CONTROL.includes(this.tagName); };
El.prototype.querySelectorAll = function () {
  const out = [];
  (function walk(n) { n.children.forEach(c => { if (c.matches()) out.push(c); walk(c); }); })(this);
  return out;
};
const state = { DEFAULT_PROMPT: 'p', DEFAULT_LANGUAGE: 'de' };
function currentValue(n) { return state[n]; }
function setDirty(n, v) { state[n] = v; }
function fieldDef(n) { return { value: state[n], default_value: '', choices: null }; }
""" + src + """
const eds = ['DEFAULT_PROMPT', 'DEFAULT_LANGUAGE'].map(makeEditor);
const wrap = document.createElement('div');
const inner = wrap.appendChild(document.createElement('input'));
const diag = wrap.appendChild(document.createElement('button'));
diag.dataset.diagnostic = '1';
[...eds, wrap].forEach(disableEnvPinnedEditor);
console.log(JSON.stringify([...eds.map(e => [e.tagName, e.disabled]),
                            inner.disabled, diag.disabled]));
""")
    assert out == [["TEXTAREA", True], ["INPUT", True], True, False]


@needs_node
def test_prompt_lab_glossary_box_keeps_newlines(client):
    """The prompt lab's glossary box was a text <input>, which strips the
    newlines of a pasted multi-line glossary, so the preview and the test
    run got one garbled pair instead of the glossary that would be saved."""
    box = _block(_html(client), "  const gloss = document.createElement(",
                 "left.appendChild(gloss);\n")
    out = _node(_FAKE_DOM + "const left = document.createElement('div');\n"
                + box + "console.log(JSON.stringify(gloss.tagName));")
    assert out == "TEXTAREA"


@needs_node
def test_prompt_lab_empty_glossary_box_uses_the_glossary_setting(client):
    """The lab's glossary box starts empty and labBody sent null, so the
    preview and the test run left out the TRANSLATION_GLOSSARY every real
    request inherits; a glossary edit on the page did not re-render either."""
    ed = _block(_html(client), "function translationTemplateEditor(name, v) {",
                "\n}\n")
    out = _node(_FAKE_DOM + """
const state = { TRANSLATION_PROMPT_FAMILY: 'hunyuan',
                TRANSLATION_GLOSSARY: 'Niere = kidney',
                TRANSLATION_DEFAULT_MODEL: 'm', TRANSLATION_ALLOWED_MODELS: [] };
function fieldDef(n) { return { value: state[n] }; }
function currentValue(n) { return state[n]; }
function setDirty(n, val) { state[n] = val; }
const listeners = {};
function _registerAdminListener(ev, key, fn) { listeners[ev + '|' + key] = fn; }
let timers = [];
function setTimeout(f) { timers.push(f); return timers.length; }
function clearTimeout() {}
const sent = [];
async function api(m, url, body) {
  sent.push(body.glossary);
  return { ok: false, status: 503, json: async () => ({}) };
}
async function flush() { const t = timers; timers = []; for (const f of t) await f(); }
""" + ed + """
(async () => {
  const wrap = translationTemplateEditor('TRANSLATION_PROMPT_TEMPLATE', '');
  const gloss = find(wrap, c => c.tagName === 'TEXTAREA'
                     && c.attrs['aria-label'] === 'glossary line');
  await flush();
  // An unsaved glossary edit on the page re-renders with the new value …
  state.TRANSLATION_GLOSSARY = 'Leber = liver';
  listeners['admin:dirty|translationLab']({ detail: { name: 'TRANSLATION_GLOSSARY' } });
  await flush();
  // … and a glossary typed into the lab box wins over the setting.
  gloss.value = 'Milz = spleen';
  await fire(gloss, 'input');
  await flush();
  console.log(JSON.stringify([sent, gloss.placeholder]));
})();
""")
    sent, placeholder = out
    assert sent == ["Niere = kidney", "Leber = liver", "Milz = spleen"]
    assert "empty uses the Translation glossary setting" in placeholder


@needs_node
def test_view_link_on_settings_opens_the_pipeline_page(client):
    """The per-model and captures checklists live on /settings, which has no
    PIPELINE_RULES editor since it moved to /settings/pipeline: '↑ view'
    found no row and did nothing."""
    html = _html(client)
    fn = _block(html, "function jumpToGlobalRule(ruleName) {", "\n}\n")
    out = _node("""
const document = { querySelector: () => null };   // /settings: no editor row
const opened = [];
const window = { open: (...a) => opened.push(a) };
function expandSectionContaining() {}
""" + fn + """
jumpToGlobalRule('de-tidy');
console.log(JSON.stringify(opened));
""")
    assert out == [["/settings/pipeline#rule=de-tidy", "_blank", "noopener"]]
    # … and the pipeline page jumps to that rule once it has rendered.
    boot = _block(html, "\n  await loadState();\n", "\n});\n")
    assert "/^#rule=(.+)$/.exec(location.hash)" in boot
    assert "jumpToGlobalRule(decodeURIComponent(jump[1]))" in boot


@needs_node
def test_restart_network_error_takes_the_progress_modal_down(client):
    """A rejected POST /settings/restart escaped doRestart and left the
    close-button-less 'Restarting…' modal up for good, with no toast."""
    fn = _block(_html(client), "async function doRestart() {", "\n}\n")
    out = _node(_FAKE_DOM + """
const els = {};
function $(id) { return els[id] || (els[id] = document.createElement('div')); }
$('restart-progress').classList.add('show');
async function fetch() { return { ok: false }; }
async function api() { throw new TypeError('Failed to fetch'); }
const toasts = [];
function toast(msg, err) { toasts.push([msg, !!err]); }
""" + fn + """
doRestart().catch(e => toasts.push(['escaped: ' + e, true])).then(() => {
  console.log(JSON.stringify([$('restart-progress').classList.contains('show'), toasts]));
});
""")
    assert out == [False, [["restart request failed: TypeError: Failed to fetch", True]]]


def test_restart_dialog_fits_a_small_screen(client):
    """.modal-box set no max-height or overflow, so many restart-required
    fields on a small viewport pushed Cancel / "Restart now" off screen."""
    html = _html(client)
    box = _block(html, "  .modal-box { ", "}")
    assert "max-height: calc(100vh - 2rem)" in box
    assert "overflow: auto" in box
    assert "box-sizing: border-box" in box
    # The field list scrolls on its own, keeping the buttons in view.
    lst = _block(html, "  #restart-fields { ", "}")
    assert "max-height: 40vh" in lst and "overflow: auto" in lst
    for dlg in ('<div id="restart-modal" class="modal">',
                '<div id="restart-progress" class="modal">'):
        assert dlg + '\n  <div class="modal-box">' in html
    modal = _block(html, '<div id="restart-modal" class="modal">', "\n</div>\n")
    assert modal.index('<ul id="restart-fields">') < modal.index('id="restart-now"')


def test_plain_checkbox_in_input_col_keeps_its_themed_width(client):
    """`.input-col input[type=checkbox] { width: auto }` out-ranks the themed
    `input[type="checkbox"] { appearance: none; width: 1rem }`, and an
    appearance:none box has no intrinsic width: every plain checkbox in a
    field (the model allowlist checklist) drew as a 2px sliver."""
    html = _html(client)
    rule = _block(html, ".input-col input[type=checkbox] {", "}")
    assert "width: 1rem" in rule
    assert "auto" not in rule


# Minimal querySelector for the `:scope > a > b .c` selectors jumpToGlobalRule
# uses: compound class selectors joined by '>' (child) or ' ' (descendant).
_SELECTOR_DOM = r"""
class Node {
  constructor(cls, kids) {
    this.cls = new Set(cls.split('.').filter(Boolean)); this.kids = kids || [];
    this.clicked = 0;
    this.classList = { contains: (c) => this.cls.has(c) };
  }
  click() { this.clicked++; }
  querySelector(sel) {
    const toks = sel.replace(/\s*>\s*/g, ' > ').trim().split(/\s+/);
    if (toks[0] !== ':scope') throw new Error('unsupported: ' + sel);
    let cur = [this];
    for (let i = 1; i < toks.length; i++) {
      const child = toks[i] === '>';
      if (child) i++;
      const want = toks[i].split('.').filter(Boolean);
      const next = [];
      const walk = (n) => n.kids.forEach(k => {
        if (want.every(c => k.cls.has(c))) next.push(k);
        if (!child) walk(k);
      });
      cur.forEach(walk);
      cur = next;
    }
    return cur[0] || null;
  }
}
"""


@needs_node
def test_view_link_expands_the_rule_card_on_the_pipeline_page(client):
    """renderRow nests the header in the card's .rule-main column (beside the
    rail), so `:scope > .row-header .expand-btn` matched nothing and the
    jumped-to rule stayed collapsed."""
    html = _html(client, "/settings/pipeline")
    # The fake card below mirrors renderRow's nesting.
    assert "cardMain.className = 'rule-main';" in html
    assert "cardMain.appendChild(head);" in html
    assert "row.appendChild(cardMain);" in html
    assert "head.appendChild(headLine1);" in html
    assert "headLine1.appendChild(headActions);" in html
    assert "headActions.appendChild(expandBtn);" in html
    fn = _block(html, "function jumpToGlobalRule(ruleName) {", "\n}\n")
    out = _node(_SELECTOR_DOM + """
const btn = new Node('.expand-btn');
const row = new Node('.rule-row.rule-card', [
  new Node('.rule-rail'),
  new Node('.rule-main', [new Node('.row-header', [
    new Node('.row-header-line1', [new Node('.head-actions', [btn])])])]),
]);
row.scrollIntoView = () => {};
const fieldRow = { querySelector: () => row };
const document = { querySelector: () => fieldRow };
const CSS = { escape: (s) => s };
function expandSectionContaining() {}
""" + fn + """
jumpToGlobalRule('de-tidy');
console.log(JSON.stringify(btn.clicked));
""")
    assert out == 1


@needs_node
def test_rule_drag_payload_is_not_a_text_type(client):
    """A text/plain payload is a native drop for every text field: releasing
    the grip over a rule's text input or textarea inserted the row index
    there. Firefox still needs some setData to start the drag."""
    start = _block(_html(client, "/settings/pipeline"),
                   "      drag.addEventListener('dragstart', (e) => {",
                   "\n      });\n")
    out = _node(_FAKE_DOM + """
const DRAG_AUTOSCROLL = { begin() {} };
const list = document.createElement('div');
const placeholder = document.createElement('div');
const row = document.createElement('div');
const drag = document.createElement('span');
const idx = 3;
let dragSrcEl = null;
let dragSrcIdx = null;
function setTimeout() {}
""" + start + """
const types = [];
const dt = { setData(t) { types.push(t); }, setDragImage() {} };
fire(drag, 'dragstart', { dataTransfer: dt }).then(() => {
  console.log(JSON.stringify([types, dragSrcIdx]));
});
""")
    types, src_idx = out
    assert types and not [t for t in types if t.lower().startswith("text")]
    assert src_idx == 3


@needs_node
def test_locked_rule_freezes_its_regex_entry_grips(client):
    """A locked rule disabled the entries' ↑/↓ buttons but left the shared
    editor's .rl-grip draggable, so a drag still reordered the entries of a
    rule shown read-only and the next Save wrote the new order."""
    lock = _block(_html(client, "/settings/pipeline"),
                  "    function _applyLockState() {", "\n    }\n")
    out = _node(_FAKE_DOM + """
El.prototype.querySelectorAll = function (sel) {
  const want = sel.split(',').map(s => s.trim());
  const hit = (n) => want.some(s => s.startsWith('.')
    ? n.classList.contains(s.slice(1)) : n.tagName === s.toUpperCase());
  const out = [];
  (function walk(n) { n.children.forEach(c => { if (hit(c)) out.push(c); walk(c); }); })(this);
  return out;
};
const rule = { type: 'regex-list', locked: true };
const body = document.createElement('div');
const headLine1 = document.createElement('div');
const headLine2 = document.createElement('div');
const drag = document.createElement('span');
const upBtn = null, downBtn = null, _langPicker = null;
const grips = [0, 1].map(() => {
  const entry = document.createElement('div');
  const grip = document.createElement('span');
  grip.className = 'rl-grip'; grip.draggable = true;
  entry.appendChild(grip); body.appendChild(entry);
  return grip;
});
""" + lock + """
_applyLockState();
console.log(JSON.stringify(grips.map(g => g.draggable)));
""")
    assert out == [False, False]


@needs_node
def test_env_pinned_rule_list_repaint_stays_read_only(client):
    """fieldRow() disabled an env-pinned PIPELINE_RULES editor once, but its
    rule grips stayed draggable and paintAll() rebuilt every row enabled: one
    drag made the list editable while setDirty() dropped each edit unseen."""
    html = _html(client, "/settings/pipeline")
    src = "\n".join([
        _block(html, "  function paintAll() {"),
        _block(html, "function disableEnvPinnedEditor(el) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const CONTROL = ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'];
El.prototype.matches = function () { return CONTROL.includes(this.tagName); };
El.prototype.querySelectorAll = function (sel) {
  const want = sel.split(',').map(s => s.trim());
  const hit = (n) => want.some(s => s.startsWith('.')
    ? n.classList.contains(s.slice(1)) : n.tagName === s.toUpperCase());
  const out = [];
  (function walk(n) { n.children.forEach(c => { if (hit(c)) out.push(c); walk(c); }); })(this);
  return out;
};
const name = 'PIPELINE_RULES';
const rules = [{ name: 'a' }, { name: 'b' }];
function isEnvPinned(n) { return n === 'PIPELINE_RULES'; }
function refreshControlsVisibility() {}
function renderRow(rule, idx) {
  const row = document.createElement('div');
  for (const cls of ['drag-handle', 'rl-grip']) {
    const g = row.appendChild(document.createElement('span'));
    g.className = cls; g.draggable = true;
  }
  row.appendChild(document.createElement('input'));
  return row;
}
function paint(isChecklist) {
  const wrap = document.createElement('div');
  const list = wrap.appendChild(document.createElement('div'));
""" + src + """
  paintAll();
  // Per row: is each grip still draggable, is the input still editable?
  return list.children.map(r => r.children.map(c => c.tagName === 'INPUT'
    ? !c.disabled : c.draggable));
}
console.log(JSON.stringify([paint(false), paint(true)]));
""")
    assert out == [[[False] * 3] * 2, [[True] * 3] * 2]


@needs_node
def test_env_pinned_rule_list_cards_still_open_for_reading(client):
    """disableEnvPinnedEditor() disabled each card's "edit ▾" toggle too, so on
    an env-pinned PIPELINE_RULES no rule body could be opened to read it, and
    the "↑ view" jump link's click on that toggle did nothing. Once live, the
    toggle still offered to "edit" a card whose controls were all dead."""
    html = _html(client, "/settings/pipeline")
    src = "\n".join([
        _block(html, "    const expandBtn = document.createElement('button');",
               "\n    });\n"),
        _block(html, "function disableEnvPinnedEditor(el) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const CONTROL = ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'];
El.prototype.matches = function () { return CONTROL.includes(this.tagName); };
El.prototype.querySelectorAll = function () {
  const out = [];
  (function walk(n) { n.children.forEach(c => { if (c.matches()) out.push(c); walk(c); }); })(this);
  return out;
};
const rule = { name: 'a', type: 'regex' };
const name = 'PIPELINE_RULES';
const isChecklist = false;
let pinned = true;
function isEnvPinned(n) { return pinned && n === 'PIPELINE_RULES'; }
const expandedNames = new Set();
function _saveExpanded() {}
let refresh = null;
function card() {
  const row = document.createElement('div');
""" + src + """
  const body = row.appendChild(document.createElement('div'));
  const inp = body.appendChild(document.createElement('input'));
  row.appendChild(expandBtn);
  if (pinned) disableEnvPinnedEditor(row);
  return { row, expandBtn, inp };
}
(async () => {
  const { row, expandBtn, inp } = card();
  const label = expandBtn.textContent;
  // A disabled button never sees the click, so only fire it when live.
  if (!expandBtn.disabled) await fire(expandBtn, 'click');
  const pin = [expandBtn.disabled, row.classList.contains('expanded'), inp.disabled,
               label, expandBtn.textContent];
  pinned = false;
  console.log(JSON.stringify([pin, card().expandBtn.textContent]));
})();
""")
    pin, free_label = out
    assert pin == [False, True, True, "view ▾", "view ▴"]
    assert free_label == "edit ▾"


@needs_node
def test_env_pinned_full_row_field_says_why_it_is_read_only(client):
    """fieldRow()'s full-row branch disabled an env-pinned MODEL_OVERRIDES but
    skipped the env badge and the "Set by …" note every other pinned field
    gets, so the whole editor just went dead with no reason given."""
    html = _html(client)
    src = "\n".join([
        _block(html, "const FULLROW_FIELDS = ", ";\n"),
        _block(html, "function makeBadges(name) {", "\n}\n"),
        _block(html, "function envPinnedNote(name) {", "\n}\n"),
        _block(html, "function fieldRow(name) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const pinned = new Set();
function isEnvPinned(n) { return pinned.has(n); }
function fieldDef(n) {
  return { provenance: pinned.has(n) ? 'env' : 'default', env_var: 'WHISPER_' + n };
}
function makeEditor(n) { return document.createElement('div'); }
function disableEnvPinnedEditor(el) { el.disabled = true; }
""" + src + """
function paint() {
  const row = fieldRow('MODEL_OVERRIDES');
  const note = find(row, c => c.classList.contains('help-env-pinned'));
  const badges = find(row, c => c.classList.contains('badges'));
  return [row.classList.contains('env-pinned'), note && note.textContent,
          badges && badges.innerHTML.includes('env: WHISPER_MODEL_OVERRIDES')];
}
const free = paint();
pinned.add('MODEL_OVERRIDES');
console.log(JSON.stringify([free, paint()]));
""")
    free, pin = out
    assert free == [False, None, None]
    assert pin[0] is True and pin[2] is True
    assert pin[1].startswith("Set by WHISPER_MODEL_OVERRIDES — read-only here")


@needs_node
def test_env_pinned_field_offers_no_reset_to_default(client):
    """The "↺ Reset to default" link sits outside the editor that
    disableEnvPinnedEditor() locks, so on an env-pinned field it stayed
    clickable and only repainted the row (setDirty() drops the reset)."""
    html = _html(client)
    src = "\n".join([
        _block(html, "const FULLROW_FIELDS = ", ";\n"),
        _block(html, "function makeBadges(name) {", "\n}\n"),
        _block(html, "function envPinnedNote(name) {", "\n}\n"),
        _block(html, "function fieldRow(name) {", "\n}\n"),
    ])
    out = _node(_FAKE_DOM + """
const pinned = new Set();
function isEnvPinned(n) { return pinned.has(n); }
function fieldDef(n) {
  return { provenance: pinned.has(n) ? 'env' : 'default', env_var: 'WHISPER_' + n,
           value: 'env value', default_value: '' };
}
function currentValue(n) { return fieldDef(n).value; }
function makeEditor(n) { return document.createElement('input'); }
function disableEnvPinnedEditor(el) { el.disabled = true; }
""" + src + """
function reset() {
  const btn = find(fieldRow('DEFAULT_PROMPT'), c => c.textContent === '↺ Reset to default');
  return btn && btn.parentElement.style.display !== 'none' && !btn.disabled;
}
const free = reset();
pinned.add('DEFAULT_PROMPT');
console.log(JSON.stringify([free, reset()]));
""")
    assert out == [True, None]


@needs_node
def test_env_pinned_editor_keeps_its_view_only_controls_live(client):
    """The checklist "↑ view" jump link and the MODEL_OVERRIDES "Global — dim
    what matches" switch change no value, but disableEnvPinnedEditor() disabled
    them too, so a pinned field could not even be inspected."""
    html = _html(client)
    out = _node(_captures_harness(html, """
const CONTROL = ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'];
El.prototype.matches = function () { return CONTROL.includes(this.tagName); };
El.prototype.querySelectorAll = function () {
  const out = [];
  (function walk(n) { n.children.forEach(c => { if (c.matches()) out.push(c); walk(c); }); })(this);
  return out;
};
""" + _block(html, "function disableEnvPinnedEditor(el) {", "\n}\n") + """
state.PIPELINE_RULES = [{ name: 'a', type: 'regex', enabled: true }];
state.CAPTURES_PIPELINE_RULES_EXCLUDE = [];
const wrap = makeEditor('CAPTURES_PIPELINE_RULES_EXCLUDE');
disableEnvPinnedEditor(wrap);
const view = find(wrap, e => e.textContent === '↑ view');
const cb = find(find(wrap, e => e.dataset.slug === 'a'), e => e.tagName === 'INPUT');
console.log(JSON.stringify([view.disabled, cb.disabled]));
""", real_list=True))
    assert out == [False, True]
    # The dim switch lives in the big MODEL_OVERRIDES builder: pin its marker.
    side = _block(html, "    const toggleCb = document.createElement('input');",
                  "toggleWrap.appendChild(toggleCb);\n")
    assert "toggleCb.dataset.diagnostic = '1';" in side
    assert "setDirty" not in side
