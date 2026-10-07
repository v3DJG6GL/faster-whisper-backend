"""Factory PIPELINE_RULES (config.json): the digit guard on
delete-punct-before-hyphen, the language tags on language-specific rules, the
Spanish opening-mark rule, and the dictation-map → de-dictation-map rename.

2026-09-19: "vom 3.-10 Oktober" left the pipeline as "vom 310 Oktober" —
final-cleanup deleted ".-" between two numbers. A silently changed date or
dose is the worst kind of error in a medical dictation.
"""
import copy
import json
import os
import re

import pytest

from faster_whisper_backend.settings import config_renames as renames
from faster_whisper_backend import paths
from faster_whisper_backend.pipeline import engine as pl_engine


def _factory_rules():
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["PIPELINE_RULES"]


def _by_name():
    return {r["name"]: r for r in _factory_rules()}


def _hyphen_pattern():
    entry = next(e for e in _by_name()["final-cleanup"]["entries"]
                 if e["label"] == "delete-punct-before-hyphen")
    return entry["pattern"], entry["replacement"]


@pytest.mark.parametrize("text", [
    "Die Patientin ist vom 3.-10 Oktober abwesend.",
    "Termin am 12.-14.03.2026",
    "Dosis 2,-5 mg",
    "vom 3.- 10. Oktober",
])
def test_punct_before_hyphen_is_kept_after_a_digit(text):
    pat, repl = _hyphen_pattern()
    assert re.sub(pat, repl, text) == text


def test_punct_before_hyphen_still_removed_after_a_word():
    """The rule's documented purpose (a,-b → ab) is unchanged."""
    pat, repl = _hyphen_pattern()
    assert re.sub(pat, repl, "Belastung.- Neue Zeile") == "BelastungNeue Zeile"
    assert re.sub(pat, repl, "a,-b") == "ab"


def test_date_range_survives_the_whole_factory_pipeline(app_module):
    app_module.cfg.PIPELINE_RULES = copy.deepcopy(_factory_rules())
    pl_engine.rebuild_caches()
    out = pl_engine._postprocess_text(
        " Die Patientin ist vom 3.-10 Oktober abwesend Punkt",
        model_name="", language="de")
    assert out == "Die Patientin ist vom 3.-10 Oktober abwesend."


def test_language_specific_rules_are_tagged():
    rules = _by_name()
    for name in ("de-strip-and-lowercase-non-noun", "de-split-punctuation-keywords",
                 "ch-symbol-cleanup", "de-dictation-map"):
        assert rules[name].get("languages") == ["de"], name
    assert rules["es-punctuation"]["languages"] == ["es"]
    # A language-prefixed slug without a language tag would run everywhere.
    for name, r in rules.items():
        if re.match(r"^(de|ch|es|fr|it|en)-", name):
            assert r.get("languages"), f"{name} has a language prefix but no tag"


def test_spanish_opening_marks(app_module):
    app_module.cfg.PIPELINE_RULES = [
        copy.deepcopy(_by_name()["es-punctuation"]),
        {"name": "trim-edges", "label": "Trim", "type": "terminal"}]
    pl_engine.rebuild_caches()
    run = lambda t, lang: pl_engine._postprocess_text(t, model_name="", language=lang)
    assert run("Qué hora es? No lo sé. Qué bien!", "es") == "¿Qué hora es? No lo sé. ¡Qué bien!"
    assert run("¿Ya está? ¡Sí!", "es") == "¿Ya está? ¡Sí!", "existing marks are not doubled"
    assert run("Wie spät ist es?", "de") == "Wie spät ist es?", "German is untouched"


def test_spanish_opening_marks_on_consecutive_sentences(app_module):
    """The sentence-start anchor is zero-width: a consumed terminator made the
    scan resume past it, so only every other question got its opening mark."""
    app_module.cfg.PIPELINE_RULES = [
        copy.deepcopy(_by_name()["es-punctuation"]),
        {"name": "trim-edges", "label": "Trim", "type": "terminal"}]
    pl_engine.rebuild_caches()
    run = lambda t: pl_engine._postprocess_text(t, model_name="", language="es")
    assert run("Qué? Cómo?") == "¿Qué? ¿Cómo?"
    assert run("Hola! Adiós!") == "¡Hola! ¡Adiós!"
    assert run("Qué hora es? Y luego? Vale.") == "¿Qué hora es? ¿Y luego? Vale."
    assert run("Uno\nDos? Tres!") == "Uno\n¿Dos? ¡Tres!"
    # digit guard: a decimal point is not a sentence end
    assert run("Vale. Cuesta 3.5 euros?") == "Vale. ¿Cuesta 3.5 euros?"
    assert run("Son las 10.23? Vale!") == "¿Son las 10.23? ¡Vale!"


def test_spanish_patterns_pass_the_regex_guard():
    from faster_whisper_backend.pipeline import regex_guard
    entries = _by_name()["es-punctuation"]["entries"]
    regex_guard.validate([[e["label"], e["pattern"], e["replacement"]] for e in entries])


def test_spanish_marks_survive_the_whole_factory_pipeline(app_module):
    """es-punctuation runs ahead of strip-auto-punctuation, which takes an
    opening mark along with its closing one: no ¿/¡ lands on the wrong
    sentence or is left orphaned."""
    app_module.cfg.PIPELINE_RULES = copy.deepcopy(_factory_rules())
    pl_engine.rebuild_caches()
    run = lambda t: pl_engine._postprocess_text(t, model_name="", language="es")
    assert run("Hola. Tienes 3?") == "Hola ¿Tienes 3?"
    assert run("Cuántos son? 25!") == "Cuántos son ¡25!"
    assert run("¿Qué hora es?") == "Qué hora es"
    off = copy.deepcopy(_factory_rules())
    next(r for r in off if r["name"] == "strip-auto-punctuation")["enabled"] = False
    app_module.cfg.PIPELINE_RULES = off
    pl_engine.rebuild_caches()
    assert run("Qué hora es? Qué bien!") == "¿Qué hora es? ¡Qué bien!"


def test_german_map_does_not_touch_spanish(app_module):
    app_module.cfg.PIPELINE_RULES = copy.deepcopy(_factory_rules())
    pl_engine.rebuild_caches()
    assert "Punkt" in pl_engine._postprocess_text("El Punkt rojo", model_name="", language="es")


def test_rule_rename_migrates_every_stored_reference():
    raw = {
        "PIPELINE_RULES": [{"name": "dictation-map", "label": "x"},
                           {"name": "other", "label": "y"}],
        "CAPTURES_PIPELINE_RULES_EXCLUDE": ["capitalize-after-terminator", "dictation-map"],
        "MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["dictation-map"]}},
        "OVERRIDE_PROFILES": {"p": {"PIPELINE_RULES_INCLUDE": ["dictation-map", "de-dictation-map"]}},
    }
    # Both run on every stored / env path (config_store._migrate_legacy_keys,
    # config.py's JSON env loop): the bundles' slug lists are migrate_bundle's.
    renames.migrate_rule_slugs(raw)
    renames.migrate_bundle_keys(raw)
    assert [r["name"] for r in raw["PIPELINE_RULES"]] == ["de-dictation-map", "other"]
    assert raw["CAPTURES_PIPELINE_RULES_EXCLUDE"] == ["capitalize-after-terminator", "de-dictation-map"]
    assert raw["MODEL_OVERRIDES"]["m"]["PIPELINE_RULES_EXCLUDE"] == ["de-dictation-map"]
    assert raw["OVERRIDE_PROFILES"]["p"]["PIPELINE_RULES_INCLUDE"] == ["de-dictation-map"]


def test_rule_rename_keeps_an_already_migrated_rule():
    raw = {"PIPELINE_RULES": [{"name": "de-dictation-map", "label": "new"},
                              {"name": "dictation-map", "label": "old"}]}
    renames.migrate_rule_slugs(raw)
    assert [r["name"] for r in raw["PIPELINE_RULES"]] == ["de-dictation-map", "dictation-map"]


def test_rename_slugs_keeps_container_type():
    assert renames.rename_slugs({"dictation-map"}) == {"de-dictation-map"}
    assert renames.rename_slugs(["a", "dictation-map"]) == ["a", "de-dictation-map"]
    assert renames.rename_slugs(None) is None


def test_no_factory_default_names_a_renamed_rule():
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        text = fh.read()
    for old in renames.RENAMED_RULES:
        assert f'"{old}"' not in text, f"config.json still names {old}"


# ---- factory entries whose text changed: stored copies are upgraded -------

def _quote_entry(rules):
    return next(e for r in rules for e in (r.get("entries") or [])
                if e.get("label") == "tighten-quote-spacing")


def test_upgraded_entries_match_config_json():
    """The NEW side of every upgrade is what config.json ships — an upgrade
    that wrote anything else would fork stored copies from the factory."""
    for label, (_old, new) in renames.UPGRADED_RULE_ENTRIES.items():
        e = next(e for r in _factory_rules() for e in (r.get("entries") or [])
                 if e.get("label") == label)
        assert (e["pattern"], e["replacement"]) == new


def test_stored_old_factory_entry_is_upgraded_once():
    rules = copy.deepcopy(_factory_rules())
    old_pat, old_rep = renames.UPGRADED_RULE_ENTRIES["tighten-quote-spacing"][0]
    e = _quote_entry(rules)
    e["pattern"], e["replacement"] = old_pat, old_rep
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == ["digit-and-whitespace-tidy/tighten-quote-spacing"]
    assert raw["PIPELINE_RULES"] == _factory_rules()
    # idempotent: a second pass finds nothing to do
    assert renames.upgrade_rule_entries(raw) == []


def test_stored_old_hyphen_entry_is_upgraded_once():
    """A stored copy from before c4fefe2 still merged "3.-10" into "310"."""
    rules = copy.deepcopy(_factory_rules())
    (old_pat, old_rep), (new_pat, new_rep) = \
        renames.UPGRADED_RULE_ENTRIES["delete-punct-before-hyphen"]
    e = next(e for r in rules for e in (r.get("entries") or [])
             if e.get("label") == "delete-punct-before-hyphen")
    e["pattern"], e["replacement"] = old_pat, old_rep
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == ["final-cleanup/delete-punct-before-hyphen"]
    assert raw["PIPELINE_RULES"] == _factory_rules()
    assert renames.upgrade_rule_entries(raw) == []
    assert re.sub(old_pat, old_rep, "vom 3.-10 Oktober") == "vom 310 Oktober"
    assert re.sub(new_pat, new_rep, "vom 3.-10 Oktober") == "vom 3.-10 Oktober"
    assert re.sub(new_pat, new_rep, "a,-b") == "ab"


def test_stored_old_strip_entry_is_upgraded_once():
    """A stored copy from before es-punctuation moved ahead of the strip left
    a Whisper-written "¿Qué hora es?" as an orphaned "¿Qué hora es"."""
    rules = copy.deepcopy(_factory_rules())
    (old_pat, old_rep), (new_pat, new_rep) = \
        renames.UPGRADED_RULE_ENTRIES["Strip terminators & commas"]
    e = next(e for r in rules for e in (r.get("entries") or [])
             if e.get("label") == "Strip terminators & commas")
    e["pattern"], e["replacement"] = old_pat, old_rep
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == [
        "strip-auto-punctuation/Strip terminators & commas"]
    assert raw["PIPELINE_RULES"] == _factory_rules()
    assert renames.upgrade_rule_entries(raw) == []
    assert re.sub(old_pat, old_rep, "¿Qué hora es?") == "¿Qué hora es"
    assert re.sub(new_pat, new_rep, "¿Qué hora es?") == "Qué hora es"
    assert re.sub(new_pat, new_rep, "¿Vienes a las 10?") == "¿Vienes a las 10?"


def test_edited_factory_entry_is_left_alone():
    rules = copy.deepcopy(_factory_rules())
    old_pat, _old_rep = renames.UPGRADED_RULE_ENTRIES["tighten-quote-spacing"][0]
    e = _quote_entry(rules)
    e["pattern"], e["replacement"] = old_pat, '"\\1" '       # admin's own replacement
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == []
    assert _quote_entry(raw["PIPELINE_RULES"])["pattern"] == old_pat
    assert renames.upgrade_rule_entries({}) == []
    assert renames.upgrade_rule_entries({"PIPELINE_RULES": "junk"}) == []


def test_load_overrides_upgrades_a_stored_old_quote_entry(tmp_path):
    from faster_whisper_backend.settings import config_store as cs
    rules = copy.deepcopy(_factory_rules())
    old_pat, old_rep = renames.UPGRADED_RULE_ENTRIES["tighten-quote-spacing"][0]
    e = _quote_entry(rules)
    e["pattern"], e["replacement"] = old_pat, old_rep
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"PIPELINE_RULES": rules}), encoding="utf-8")
    out = cs.load_overrides(str(p))
    new = renames.UPGRADED_RULE_ENTRIES["tighten-quote-spacing"][1]
    got = _quote_entry(out["PIPELINE_RULES"])
    assert (got["pattern"], got["replacement"]) == new
