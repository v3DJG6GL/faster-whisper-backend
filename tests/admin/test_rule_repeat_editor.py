"""The "Repeat until nothing changes" switch in the /settings pipeline editor:
a checkbox in the rule body (rule_editor.js renderTypeEditor, `showRepeat`,
which only /settings passes — /quick-config cannot patch `repeat`). Off is
deleted, never stored as false, and the editor's fingerprint (_ruleHash /
_ruleStatus in settings.html) reads a false like an absent field: the server
dumps off as absent (schema _repeat_off_is_absent), and every rule saved
before the switch existed must keep its config_rev.

The real editor functions run under node against a small DOM stub."""

import json
import os
import shutil
import subprocess

import pytest

from faster_whisper_backend import paths
from faster_whisper_backend.core import web_common
from faster_whisper_backend.settings import config_store as cs
from faster_whisper_backend.settings import schema as settings_schema
from tests._page_helpers import js_fn

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

_SETTINGS_HTML = os.path.join(paths.REPO_ROOT, "faster_whisper_backend", "admin",
                              "templates", "settings.html")

_DOM = r"""
function el(tag) {
  const e = {
    tag, children: [], listeners: {}, style: {}, dataset: {}, attrs: {},
    className: '', textContent: '', value: '', checked: false,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    fire(t) { (this.listeners[t] || []).forEach(f => f({ target: this })); },
  };
  return e;
}
function find(e, cls) {
  if (e.className === cls) return e;
  for (const c of e.children) { const f = find(c, cls); if (f) return f; }
  return null;
}
global.document = { createElement: el };
global.window = {};
"""


def _editor_js() -> str:
    js = web_common.RULE_EDITOR_JS
    return js[js.index("<script>") + len("<script>"):js.rindex("</script>")]


def _node(script: str) -> dict:
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True,
                         encoding="utf-8", timeout=30, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@needs_node
def test_the_checkbox_round_trips_the_field():
    res = _node(_DOM + _editor_js() + r"""
const rule = { name: 'h', type: 'callback:upper', pattern: 'x' };
let commits = 0;
const box = renderTypeEditor(rule, () => commits++, { showRepeat: true });
const cb = find(box, 'rule-repeat-cb');
const start = cb.checked;
cb.checked = true; cb.fire('change');
const on = { repeat: rule.repeat, commits };
cb.checked = false; cb.fire('change');
const off = { has: 'repeat' in rule, commits };
const again = find(renderTypeEditor({ type: 'callback:dedup', repeat: true, pattern: 'x' },
                                    () => {}, { showRepeat: true }), 'rule-repeat-cb').checked;
console.log(JSON.stringify({ start, on, off, again }));
""")
    assert res == {"start": False, "on": {"repeat": True, "commits": 1},
                   "off": {"has": False, "commits": 2}, "again": True}


@needs_node
def test_only_the_admin_editor_shows_it_and_never_on_the_terminal_rule():
    res = _node(_DOM + _editor_js() + r"""
const has = (rule, opts) => !!find(renderTypeEditor(rule, () => {}, opts), 'rule-repeat-cb');
console.log(JSON.stringify({
  quick: has({ type: 'callback:upper', pattern: 'x' }, {}),
  terminal: has({ type: 'terminal' }, { showRepeat: true }),
  admin: has({ type: 'callback:lowercase-wordlist', pattern: 'x', wordlist: [] },
             { showRepeat: true }),
}));
""")
    assert res == {"quick": False, "terminal": False, "admin": True}


def test_the_help_text_names_the_engine_pass_cap(client):
    js = web_common.RULE_EDITOR_JS
    assert "__REPEAT_CAP__" not in js
    assert f"at most {settings_schema.RULE_REPEAT_MAX_PASSES} times" in js
    html = client.get("/settings/pipeline").text
    assert "renderTypeEditor(rule, commitData, { showNote: true, showRepeat: true })" in html


@needs_node
def test_a_false_repeat_fingerprints_like_none_and_reads_unedited():
    with open(_SETTINGS_HTML, encoding="utf-8") as fh:
        src = fh.read()
    js = "".join(js_fn(src, f"  function {n}(") + "\n" for n in (
        "_sortDeep", "_withDefaults", "_ruleStrip", "_ruleContentEqual", "_ruleHash",
        "_factoryRule", "_ruleStatus"))
    base = {"name": "h", "label": "H", "type": "callback:upper", "pattern": "x"}
    res = _node(js + "\nconst base = " + json.dumps(base) + r""";
let factoryRules = [base];
const rev = _ruleHash(base);
const on = Object.assign({}, base, { repeat: true, config_rev: rev });
console.log(JSON.stringify({
  same: _ruleHash(Object.assign({}, base, { repeat: false })) === rev,
  differs: _ruleHash(on) !== rev,
  on: _ruleStatus(on),
  off: _ruleStatus(Object.assign({}, base, { repeat: false, config_rev: rev })),
  hash_on: _ruleHash(on),
}));
""")
    assert res["same"] and res["differs"]
    assert (res["on"], res["off"]) == ("edited", "factory")
    # config_store's copy of _ruleHash agrees, on and off.
    assert res["hash_on"] == cs._editor_rule_hash(cs._editor_view(dict(base, repeat=True)))
    assert (cs._editor_rule_hash(cs._editor_view(dict(base, repeat=False)))
            == cs._editor_rule_hash(cs._editor_view(base)))
