"""config_renames.UPGRADED_RULE_LANGUAGES: a factory rule that gained a
rule-level language scope in config.json carries it into a saved
PIPELINE_RULES copy (config.local.json / WHISPER_PIPELINE_RULES), which
otherwise replaces the factory list and keeps the rule on every language.
"""
import copy
import json
import os

import pytest

from faster_whisper_backend import paths
from faster_whisper_backend.settings import config_renames as renames


def _factory_rules():
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["PIPELINE_RULES"]


def _rule(rules, name):
    return next(r for r in rules if r.get("name") == name)


def test_every_language_upgrade_matches_its_factory_rule():
    for name, ((typ, pattern), langs) in renames.UPGRADED_RULE_LANGUAGES.items():
        r = _rule(_factory_rules(), name)
        assert (r["type"], r.get("pattern"), r["languages"]) == (typ, pattern, list(langs))


@pytest.mark.parametrize("stored_languages", [[], None, "absent"])
def test_stored_hyphen_rule_without_languages_becomes_german_only(stored_languages):
    rules = copy.deepcopy(_factory_rules())
    r = _rule(rules, "capitalize-letter-after-hyphen")
    if stored_languages == "absent":
        del r["languages"]
    else:
        r["languages"] = stored_languages
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == ["capitalize-letter-after-hyphen/languages"]
    assert raw["PIPELINE_RULES"] == _factory_rules()
    assert renames.upgrade_rule_entries(raw) == []


def test_edited_hyphen_rule_keeps_its_own_scope_and_pattern():
    rules = copy.deepcopy(_factory_rules())
    scoped = _rule(rules, "capitalize-letter-after-hyphen")
    scoped["languages"] = ["de", "en"]
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == []
    assert scoped["languages"] == ["de", "en"]
    rules = copy.deepcopy(_factory_rules())
    edited = _rule(rules, "capitalize-letter-after-hyphen")
    edited["languages"] = []
    edited["pattern"] = "(-)([a-z])"
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == []
    assert edited["languages"] == []


def test_load_overrides_scopes_a_stored_hyphen_rule_to_german(tmp_path):
    from faster_whisper_backend.settings import config_store as cs
    rules = copy.deepcopy(_factory_rules())
    _rule(rules, "capitalize-letter-after-hyphen")["languages"] = []
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"PIPELINE_RULES": rules}), encoding="utf-8")
    out = cs.load_overrides(str(p))
    assert _rule(out["PIPELINE_RULES"], "capitalize-letter-after-hyphen")["languages"] == ["de"]


_DE_SCOPED_IN_C4FEFE2 = ("de-strip-and-lowercase-non-noun",
                         "de-split-punctuation-keywords", "ch-symbol-cleanup",
                         "de-dictation-map")


def test_rules_scoped_to_german_in_c4fefe2_upgrade_a_stored_copy(tmp_path):
    from faster_whisper_backend.settings import config_store as cs
    rules = copy.deepcopy(_factory_rules())
    for name in _DE_SCOPED_IN_C4FEFE2:
        r = _rule(rules, name)
        del r["languages"]
        if name == "de-dictation-map":
            r["name"] = "dictation-map"  # its slug before c4fefe2
            del r["map"]["Neuenzeile"]  # the factory map of 518ee09..923d171
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"PIPELINE_RULES": rules}), encoding="utf-8")
    out = cs.load_overrides(str(p))
    for name in _DE_SCOPED_IN_C4FEFE2:
        assert _rule(out["PIPELINE_RULES"], name)["languages"] == ["de"], name


def test_edited_wordlist_rule_pattern_keeps_every_language():
    rules = copy.deepcopy(_factory_rules())
    r = _rule(rules, "de-strip-and-lowercase-non-noun")
    r["languages"] = []
    r["pattern"] = "[.?!](\\s*)([A-Z])(\\w*)"
    assert "de-strip-and-lowercase-non-noun/languages" not in renames.upgrade_rule_entries(
        {"PIPELINE_RULES": rules})
    assert r["languages"] == []


@pytest.mark.parametrize("name", ["de-split-punctuation-keywords",
                                  "ch-symbol-cleanup", "de-dictation-map",
                                  "de-strip-and-lowercase-non-noun"])
def test_edited_entries_or_map_keep_the_admins_language_scope(name):
    """Only their content tells a pre-c4fefe2 factory copy from one the admin
    rewrote (an English-extended dictation map, own symbol entries, own
    function words under the factory pattern) and set to every language."""
    rules = copy.deepcopy(_factory_rules())
    r = _rule(rules, name)
    r["languages"] = []
    if name == "de-dictation-map":
        r["map"][" new line "] = "\n"
    elif name == "de-strip-and-lowercase-non-noun":
        r["wordlist"].append("Sowieso")
    else:
        r["entries"].append({"pattern": "mine", "replacement": "", "label": "mine"})
    assert f"{name}/languages" not in renames.upgrade_rule_entries(
        {"PIPELINE_RULES": rules})
    assert r["languages"] == []


def test_relabelled_factory_entries_still_count_as_the_factory_copy():
    """Labels and notes changed across releases; only what the rule does is
    compared."""
    rules = copy.deepcopy(_factory_rules())
    r = _rule(rules, "ch-symbol-cleanup")
    r["languages"] = []
    for e in r["entries"]:
        e["label"], e["note"] = "renamed", ""
    assert renames.upgrade_rule_entries({"PIPELINE_RULES": rules}) == [
        "ch-symbol-cleanup/languages"]
