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

from faster_whisper_backend import config_renames as renames
from faster_whisper_backend import paths


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
    app_module.rebuild_caches()
    out = app_module._postprocess_text(
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
    app_module.rebuild_caches()
    run = lambda t, lang: app_module._postprocess_text(t, model_name="", language=lang)
    assert run("Qué hora es? No lo sé. Qué bien!", "es") == "¿Qué hora es? No lo sé. ¡Qué bien!"
    assert run("¿Ya está? ¡Sí!", "es") == "¿Ya está? ¡Sí!", "existing marks are not doubled"
    assert run("Wie spät ist es?", "de") == "Wie spät ist es?", "German is untouched"


def test_german_map_does_not_touch_spanish(app_module):
    app_module.cfg.PIPELINE_RULES = copy.deepcopy(_factory_rules())
    app_module.rebuild_caches()
    assert "Punkt" in app_module._postprocess_text("El Punkt rojo", model_name="", language="es")


def test_rule_rename_migrates_every_stored_reference():
    raw = {
        "PIPELINE_RULES": [{"name": "dictation-map", "label": "x"},
                           {"name": "other", "label": "y"}],
        "CAPTURES_PIPELINE_RULES_EXCLUDE": ["capitalize-after-terminator", "dictation-map"],
        "MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["dictation-map"]}},
        "OVERRIDE_PROFILES": {"p": {"PIPELINE_RULES_INCLUDE": ["dictation-map", "de-dictation-map"]}},
    }
    renames.migrate_rule_slugs(raw)
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
