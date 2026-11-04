"""capitalize-letter-after-hyphen fires only after a word that starts with a
capital, so a German adjective compound Whisper wrote right is left alone;
a saved copy of the old factory rule is upgraded to it.

The old pattern `(?<!\\d)(-)([a-zäöüß])` capitalised the letter after any
word: "schwarz-weiß" → "schwarz-Weiß", "deutsch-französische" →
"deutsch-Französische". The new one matches the word before the hyphen, so
one re.sub pass reaches only the first lowercase part of a chain
("Hals-Nasen-ohren-arzt"): the rule repeats until nothing changes.
"""

import copy

import pytest

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.pipeline import regex_guard
from faster_whisper_backend.settings import config_renames as renames
from tests.settings._rules_helpers import (
    assert_upgrades_to_factory_once as _assert_upgrades_to_factory_once,
    factory_rules as _factory_rules,
)

_NAME = "capitalize-letter-after-hyphen"
# As shipped from 0a9403f8 until the capital-first pattern, and before 0a9403f8.
_OLD_PATTERNS = ("(?<!\\d)(-)([a-zäöüß])", "(-)([a-zäöüß])")
_OLD_NOTE = "Capitalizes a lowercase letter right after a hyphen: Eisen-infusion → Eisen-Infusion."


def _rule(rules):
    return next(r for r in rules if r["name"] == _NAME)


@pytest.mark.parametrize("text, want", [
    (" Ein schwarz-weißes Foto.", "Ein schwarz-weisses Foto"),
    (" Das deutsch-französische Treffen.", "Das deutsch-französische Treffen"),
    (" Im anterior-posterior Strahlengang.", "Im anterior-posterior Strahlengang"),
    (" Ein rot-grün blinder Mann.", "Ein rot-grün blinder Mann"),
    (" Die Eisen-infusion.", "Die Eisen-Infusion"),
    (" Eisen Bindestrich infusion.", "Eisen-Infusion"),
    (" Nach der Covid-impfung.", "Nach der Covid-Impfung"),
    (" Ein A4-blatt.", "Ein A4-Blatt"),
    (" Der Vitamin-D-spiegel.", "Der Vitamin-D-Spiegel"),
    (" Das ist 3-fach gesichert.", "Das ist 3-fach gesichert"),
    (" Die 45-jährige Patientin.", "Die 45-jährige Patientin"),
    (" Das Herz-kreislauf-system.", "Das Herz-Kreislauf-System"),
    (" Beim Hals-nasen-ohren-arzt.", "Beim Hals-Nasen-Ohren-Arzt"),
    (" Ein schwarz-weiß-rotes Band.", "Ein schwarz-weiss-rotes Band"),
])
def test_only_a_compound_after_a_capitalised_word_is_capitalised(app_module, text, want):
    app_module.cfg.PIPELINE_RULES = _factory_rules()
    pl_engine.rebuild_caches()
    assert pl_engine._postprocess_text(text, model_name="", language="de") == want


def test_the_factory_rule_repeats():
    assert _rule(_factory_rules())["repeat"] is True


def test_the_dry_run_of_the_factory_list_is_the_engine(client, app_module):
    rules = _factory_rules()
    sample = " Beim Hals-nasen-ohren-arzt, im Herz-kreislauf-system."
    r = client.post("/settings/test-pipeline",
                    json={"sample": sample, "rules": rules, "language": "de"})
    assert r.status_code == 200, r.text
    app_module.cfg.PIPELINE_RULES = rules
    pl_engine.rebuild_caches()
    want = pl_engine._postprocess_text(sample, language="de")
    # (The factory list drops Whisper's soft-pause commas.)
    assert r.json()["final"] == want == "Beim Hals-Nasen-Ohren-Arzt im Herz-Kreislauf-System"
    step = next(st for st in r.json()["steps"] if st["label"] == _rule(rules)["label"])
    assert (step["passes"], step["matches"]) == (3, 5)


def test_the_pattern_passes_the_regex_guard():
    regex_guard.validate([[_NAME, _rule(_factory_rules())["pattern"], ""]])


@pytest.mark.parametrize("old", _OLD_PATTERNS)
def test_a_saved_old_factory_copy_becomes_the_factory_rule_once(old):
    rules = _factory_rules()
    r = _rule(rules)
    r["pattern"], r["note"] = old, _OLD_NOTE
    del r["repeat"]  # saved before the rule repeated
    _assert_upgrades_to_factory_once(rules, [f"{_NAME}/pattern"])


def test_an_edited_note_stays_while_the_old_pattern_is_upgraded():
    rules = _factory_rules()
    r = _rule(rules)
    r["pattern"], r["note"] = _OLD_PATTERNS[0], "mine"
    del r["repeat"]
    renames.upgrade_rule_entries({"PIPELINE_RULES": rules})
    assert (r["pattern"], r["note"], r["repeat"]) == (
        _rule(_factory_rules())["pattern"], "mine", True)


def test_an_edited_pattern_stays_and_gets_no_repeat():
    rules = _factory_rules()
    r = _rule(rules)
    r["pattern"], r["note"] = "(\\w-)([a-zäöüß])", _OLD_NOTE
    del r["repeat"]
    mine = copy.deepcopy(r)
    assert renames.upgrade_rule_entries({"PIPELINE_RULES": rules}) == []
    assert r == mine
