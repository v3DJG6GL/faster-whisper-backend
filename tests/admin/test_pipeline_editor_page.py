"""Source pins for the /settings/pipeline editor script (inline JS). The
behaviour itself was exercised in headless Chromium when it was written:
all five sync dialogs against a local list holding an edited rule, a
not-edited-but-outdated rule and a missing config.json rule, no JS errors.
"""

from faster_whisper_backend.settings import config_store as cs
from faster_whisper_backend.settings import schema as settings_schema


def _html(client):
    r = client.get("/settings/pipeline")
    assert r.status_code == 200
    return r.text


def test_every_sync_action_goes_through_the_review_dialog(client):
    """Promote showed a diff without naming its sides; reset and reset-all
    showed none (reset-all was a browser confirm). One labelled dialog now
    serves promote, reset / update, add, promote all and reset all."""
    html = _html(client)
    for fn in ("_promoteOne", "_promoteAll", "_resetOne", "_resetAll", "_addFromConfig"):
        body = html[html.index(f"function {fn}("):]
        body = body[:body.index("\n  }\n")]
        assert "_reviewDialog({" in body, fn
    # Both sides are named, and the left column is what gets overwritten.
    assert "const _RV_SERVER = { t: 'This server'" in html
    assert "const _RV_CONFIG = { t: 'config.json'" in html
    assert "'gets changed'" in html
    # Colour is never the only signal.
    assert "cell('o', '−', r.o)" in html and "cell('n', '+', r.n)" in html
    # No native confirm() left on the reset path.
    assert "Discard every local edit and restore all rules" not in html


def test_diff_is_per_map_key_and_per_list_entry(client):
    html = _html(client)
    assert "add(k + ' · ' + JSON.stringify(key), js(o[key]), js(n[key]));" in html
    assert "'same entries, different order'" in html


def test_sync_actions_are_buttons_above_the_list(client):
    html = _html(client)
    assert "syncBar.appendChild(promoteAllBtn);" in html
    assert "syncBar.appendChild(resetAllBtn);" in html
    assert "resetAllBtn.className = 'sync-btn down';" in html
    assert "promoteAllBtn.className = 'sync-btn up';" in html
    assert "reset.className = 'sync-btn sm down row-reset';" in html
    assert "'↺ reset to default'" not in html
    assert ".sync-notice[hidden] { display: none; }" in html


def test_new_config_json_rules_get_a_notice(client):
    """A saved local rule list replaces the factory list, so a rule that an
    update adds to config.json stayed invisible (2026-09-19: es-punctuation)."""
    html = _html(client)
    assert "function _missingFactoryRules()" in html
    assert "'↓ Review and add'" in html


def test_promote_all_never_removes_a_config_json_rule_silently(client):
    """A config.json rule missing here used to be dropped by "Promote all" as
    "deleted locally" — it may just be new. Removal is an explicit tick."""
    html = _html(client)
    assert "tick to REMOVE it from config.json" in html
    assert "_buildFactoryPayload(promote, true).filter(r => !remove.has(r.name))" in html


def test_outdated_rules_are_told_apart_from_edited_ones(client):
    html = _html(client)
    assert "return _ruleHash(rule) === rev ? 'behind' : 'diverged';" in html
    assert "'◇ config.json is newer'" in html
    # 'behind' has no edits of its own: promoting it would write OLD content.
    assert "(st === 'edited' || st === 'diverged' || st === 'local-only')" in html


def _rule(**kw):
    return {"name": "r", "label": "R", "type": "regex-list",
            "entries": [{"pattern": "a", "replacement": "b"}], **kw}


def test_config_rev_is_stored_and_forgiving():
    ok = settings_schema.AdminConfig.model_validate({"PIPELINE_RULES": [_rule(config_rev="09efe7b7acfad2")]})
    assert ok.model_dump(exclude_none=True)["PIPELINE_RULES"][0]["config_rev"] == "09efe7b7acfad2"
    # A malformed value must never fail validation (that drops ALL overrides).
    bad = settings_schema.AdminConfig.model_validate({"PIPELINE_RULES": [_rule(config_rev="not hex!")]})
    assert "config_rev" not in bad.model_dump(exclude_none=True)["PIPELINE_RULES"][0]


def test_config_rev_never_reaches_config_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"BEAM_SIZE": 5}', encoding="utf-8")
    term = {"name": "trim-edges", "label": "Trim", "type": "terminal"}
    out = cs.save_factory_rules([_rule(config_rev="09efe7b7acfad2"), term], str(path))
    assert all("config_rev" not in r for r in out)
    assert "config_rev" not in path.read_text(encoding="utf-8")


def test_language_picker_change_never_rebuilds_the_rows(client):
    """commitFull() -> paintAll() wipes every row, which destroyed the
    language picker and its open multi-select dropdown on the first pick
    (and leaked the popover's scroll/resize listeners)."""
    html = _html(client)
    body = html[html.index("_renderLanguagePicker({"):]
    body = body[:body.index("});")]
    assert "commitData();" in body
    assert "commitFull();" not in body


def test_config_revs_are_stamped_before_the_first_edit(client):
    """Every commit runs AFTER the rule was mutated, so without a stamp at
    construction a rule's first edit never gets a config_rev and the
    'diverged' state (plus its overwrite warning) is unreachable for it."""
    html = _html(client)
    tail = html[html.rindex("function _afterPromoteAll("):]
    assert "  _stampConfigRevs();\n  paintAll();\n  return wrap;" in tail


def test_promote_keeps_an_absent_config_json_rule_in_place(client):
    """A config.json rule missing here (a new es-punctuation in the middle of
    the list) was re-appended at the END of the payload: a silent pipeline
    reorder for every other deployment. It is spliced in before the next
    config.json rule that exists here instead."""
    html = _html(client)
    body = html[html.index("function _buildFactoryPayload("):]
    body = body[:body.index("\n  }\n")]
    assert "if (!out.some(o => o.name === b.name)) out.push(" not in body
    assert "let at = out.findIndex(o => after.indexOf(o.name) !== -1);" in body
    assert "out.splice(at, 0, JSON.parse(JSON.stringify(b)));" in body


def test_entry_diff_pairs_duplicate_labels_separately(client):
    """Entry labels are free text: two entries sharing one collapsed onto a
    single Map key, so the review dialog hid the earlier one's change."""
    html = _html(client)
    assert "const idOf = (e, i) => (e.label ? 'label:' + e.label : 'pos:' + i);" not in html
    assert "return 'label:' + e.label + (c === 1 ? '' : '#' + c);" in html
    # A fresh counter per side: the Nth duplicate pairs with the Nth.
    assert "const oid = mkId(), nid = mkId();" in html


def test_single_rule_dialog_with_no_rows_cannot_be_confirmed(client):
    html = _html(client)
    assert "const nothing = single && !o.groups[0].rows.length;" in html
    assert "ok.disabled = sel.length === 0 || nothing;" in html
    assert "No differences — both sides are already identical." in html


def test_down_actions_reread_config_json_first(client):
    """config.json can move under an open page; reset / update / add acted on
    the copy captured at page load while every promote re-fetched it."""
    html = _html(client)
    for fn in ("_resetOne", "_resetAll", "_addFromConfig"):
        body = html[html.index(f"async function {fn}("):]
        assert body.index("await _refreshFactory()") < body.index("_reviewDialog({"), fn


def test_sync_dialog_leftovers_are_gone(client):
    html = _html(client)
    for dead in ("_changeListEl", "_baselineList", "const selected = ", "allowEmpty",
                 "⇪ Promote order", "wrap.appendChild(ctrls);\n  wrap.appendChild(ctrls);"):
        assert dead not in html, dead


def test_language_badge_mentions_the_unknown_language_case(client):
    """main._postprocess_text skips a language-scoped rule only when a
    language IS known; with none detected the rule runs."""
    html = _html(client)
    assert "Only runs when the detected language is" not in html
    assert "(or when the language is unknown)" in html
    ov = client.get("/settings/overrides")
    assert ov.status_code == 200
    assert "Only runs when the detected language is" not in ov.text
    assert "(or when the language is unknown)" in ov.text
