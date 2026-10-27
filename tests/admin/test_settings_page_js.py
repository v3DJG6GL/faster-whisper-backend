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

def _captures_harness(html, body):
    """makeEditor()'s captures branch with makeRuleListEditor stubbed out:
    `editor` holds the opts the checklist was built with."""
    branch = _block(html, "  if (name === 'CAPTURES_PIPELINE_RULES_EXCLUDE') {")
    return f"""
const state = {{}};
const listeners = {{}};
let editor = null;
function currentValue(n) {{ return state[n]; }}
function setDirty(n, v) {{ state[n] = v; }}
function makeRuleListEditor(name, rules, mode, opts) {{
  editor = {{ name, rules, mode, opts }};
  return editor;
}}
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
