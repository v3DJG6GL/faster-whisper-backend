"""Source pins for the /settings/pipeline editor script (inline JS). The
behaviour itself was exercised in headless Chromium when it was written:
all five sync dialogs against a local list holding an edited rule, a
not-edited-but-outdated rule and a missing config.json rule, no JS errors.
"""
import pytest

from faster_whisper_backend import config_store as cs


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
    ok = cs.AdminConfig.model_validate({"PIPELINE_RULES": [_rule(config_rev="09efe7b7acfad2")]})
    assert ok.model_dump(exclude_none=True)["PIPELINE_RULES"][0]["config_rev"] == "09efe7b7acfad2"
    # A malformed value must never fail validation (that drops ALL overrides).
    bad = cs.AdminConfig.model_validate({"PIPELINE_RULES": [_rule(config_rev="not hex!")]})
    assert "config_rev" not in bad.model_dump(exclude_none=True)["PIPELINE_RULES"][0]


def test_config_rev_never_reaches_config_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"BEAM_SIZE": 5}', encoding="utf-8")
    term = {"name": "trim-edges", "label": "Trim", "type": "terminal"}
    out = cs.save_factory_rules([_rule(config_rev="09efe7b7acfad2"), term], str(path))
    assert all("config_rev" not in r for r in out)
    assert "config_rev" not in path.read_text(encoding="utf-8")
