"""dedup-punct runs ahead of the dictation map, so dictated punctuation comes
out as spoken; a saved rule list still in the old factory order moves it too.

Behind the map (and behind strip-auto-punctuation, which has deleted Whisper's
own marks by then) it only ever collapsed DICTATED runs: "Fragezeichen
Ausrufezeichen" came out "!", "Punkt Punkt Punkt" ".".
"""

import pytest

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config_renames as renames
from faster_whisper_backend.settings import config_store as cs
from tests.settings._rules_helpers import factory_rules as _factory_rules
from tests.settings._rules_helpers import seed_local as _seed_local


def _names(rules):
    return [r["name"] for r in rules]


def _old_order(rules, es_after_hyphen=False):
    """The factory list as shipped before the move (optionally the v0.1.137-era
    order with es-punctuation behind the hyphen rule)."""
    by = {r["name"]: r for r in rules}
    order = [n for n in _names(rules) if n != "dedup-punct"]
    order.insert(order.index("digit-and-whitespace-tidy") + 1, "dedup-punct")
    if es_after_hyphen:
        order.remove("es-punctuation")
        order.insert(order.index("capitalize-letter-after-hyphen") + 1, "es-punctuation")
    return [by[n] for n in order]


@pytest.mark.parametrize("text, want", [
    (" Wirklich Fragezeichen Ausrufezeichen", "Wirklich?!"),
    (" Und dann Punkt Punkt Punkt", "Und dann..."),
    (" Wie bitte Fragezeichen Fragezeichen", "Wie bitte??"),
    (" Hallo Komma Punkt", "Hallo,."),
])
def test_dictated_punctuation_comes_out_as_spoken(app_module, text, want):
    app_module.cfg.PIPELINE_RULES = _factory_rules()
    pl_engine.rebuild_caches()
    assert pl_engine._postprocess_text(text, model_name="", language="de") == want


def test_dedup_runs_right_before_the_dictation_map():
    names = _names(_factory_rules())
    assert names.index("dedup-punct") + 1 == names.index("de-dictation-map")
    assert names.index("strip-auto-punctuation") < names.index("dedup-punct")


@pytest.mark.parametrize("es_after_hyphen", [False, True])
def test_a_saved_list_in_the_old_factory_order_is_moved_once(es_after_hyphen):
    rules = _old_order(_factory_rules(), es_after_hyphen)
    want = [n for n in _names(rules) if n != "dedup-punct"]
    want.insert(want.index("de-dictation-map"), "dedup-punct")
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == ["dedup-punct/position"]
    assert _names(raw["PIPELINE_RULES"]) == want
    assert renames.upgrade_rule_entries(raw) == []


def test_load_overrides_moves_it_in_a_stored_file(tmp_path):
    path = _seed_local(tmp_path, {"PIPELINE_RULES": _old_order(_factory_rules())})
    names = _names(cs.load_overrides(str(path))["PIPELINE_RULES"])
    assert names.index("dedup-punct") + 1 == names.index("de-dictation-map")


def test_a_reordered_list_keeps_its_order():
    rules = _old_order(_factory_rules())
    own = {"name": "mine", "label": "Mine", "type": "regex-list", "entries": []}
    rules.insert(_names(rules).index("dedup-punct"), own)
    moved = _factory_rules()
    moved.insert(0, moved.pop(_names(moved).index("dedup-punct")))
    for stored in (rules, moved):
        before = _names(stored)
        renames.upgrade_rule_entries({"PIPELINE_RULES": stored})
        assert _names(stored) == before
