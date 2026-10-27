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
