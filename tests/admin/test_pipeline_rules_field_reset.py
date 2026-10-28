"""The generic "↺ Reset to default" link on a /settings field row is not
offered for PIPELINE_RULES (admin/templates/settings.html fieldRow()).

It stored the rev-less config.json list as the pending value; Save then wrote
a pinned local copy (PIPELINE_RULES is exempt from pruning) with no
config_rev on any rule, so a later config.json update read as a local edit
instead of "config.json is newer". The rule editor has its own reset flows."""

import pytest

from tests.admin.test_settings_page_js import _FAKE_DOM, _block, _html, _node, needs_node


@needs_node
@pytest.mark.parametrize("name, want_link", [
    ("PIPELINE_RULES", False),
    ("BEAM_SIZE", True),
])
def test_field_row_reset_link_skips_pipeline_rules(client, name, want_link):
    row_fn = _block(_html(client), "function fieldRow(name) {", "\n}\n")
    out = _node(_FAKE_DOM + """
const FULLROW_FIELDS = new Set(['MODEL_OVERRIDES']);
const DEFAULTS = { PIPELINE_RULES: [{ name: 'a' }], BEAM_SIZE: 5 };
const CURRENT = { PIPELINE_RULES: [{ name: 'a', config_rev: 'r1' }], BEAM_SIZE: 3 };
function fieldDef(n) { return { default_value: DEFAULTS[n], description: '' }; }
function currentValue(n) { return CURRENT[n]; }
function makeBadges() { return document.createElement('span'); }
function makeEditor() { return document.createElement('div'); }
function isEnvPinned() { return false; }
function setHelpText() {}
function setDirty() {}
""" + row_fn + """
const row = fieldRow(%r);
const link = find(row, e => e.classList.contains('reset-link'));
console.log(JSON.stringify(!!link));
""" % name)
    assert out is want_link
