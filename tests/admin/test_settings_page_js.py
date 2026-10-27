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


# ---- CAPTURES_PIPELINE_RULES_EXCLUDE checklist --------------------------------

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


# ---- MODEL_OVERRIDES editor -----------------------------------------------------

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


# ---- PIPELINE_RULES editor (/settings/pipeline) ---------------------------------

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
