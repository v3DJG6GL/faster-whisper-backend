"""The per-rule "Repeat until nothing changes" switch (`repeat`, schema
_RepeatableRule): the engine applies the rule to its own output again (a
regex-list card: all its entries) until a pass changes nothing, at most
_REPEAT_MAX_PASSES passes.

A re.sub scans the text it was handed once, so a match that overlaps the one
before it is never seen in the same pass: the capital-first hyphen rule
capitalised only the first lowercase part of a German compound chain.
"""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.pipeline.apply import canon_rules
from faster_whisper_backend.settings import schema as settings_schema

_HYPHEN = "(\\b[A-ZÄÖÜ]\\w*-)([a-zäöüß])"


def _upper(**extra):
    return {"name": "hyphen", "label": "Hyphen", "type": "callback:upper",
            "pattern": _HYPHEN, **extra}


def _run(app_module, rules, text, **kw):
    app_module.cfg.PIPELINE_RULES = rules
    pl_engine.rebuild_caches()
    return pl_engine._postprocess_text(text, **kw)


@pytest.mark.parametrize("text, want", [
    ("Herz-kreislauf-system", "Herz-Kreislauf-System"),
    ("Hals-nasen-ohren-arzt", "Hals-Nasen-Ohren-Arzt"),
    ("schwarz-weiß", "schwarz-weiß"),
    ("deutsch-französische", "deutsch-französische"),
])
def test_a_repeating_rule_reaches_every_part_of_a_chain(app_module, text, want):
    assert _run(app_module, [_upper(repeat=True)], text) == want


@pytest.mark.parametrize("extra", [{}, {"repeat": False}])
def test_without_repeat_a_rule_runs_once(app_module, extra):
    assert _run(app_module, [_upper(**extra)], "Hals-nasen-ohren-arzt") == "Hals-Nasen-ohren-arzt"


def test_a_rule_that_never_settles_stops_at_the_pass_cap(app_module):
    grow = {"name": "grow", "label": "Grow", "type": "regex-list", "repeat": True,
            "entries": [{"pattern": "$", "replacement": "x"}]}
    out = _run(app_module, [grow], "a")
    assert out == "a" + "x" * pl_engine._REPEAT_MAX_PASSES
    assert pl_engine._REPEAT_MAX_PASSES == settings_schema.RULE_REPEAT_MAX_PASSES == 8


def test_a_regex_list_repeats_its_whole_entry_batch(app_module):
    """Entry 2's output is entry 1's input on the next pass. One pass leaves
    "aa"; repeating each entry on its own before the next one does too."""
    rl = {"name": "rl", "label": "RL", "type": "regex-list", "repeat": True,
          "entries": [{"pattern": "a", "replacement": "b"},
                      {"pattern": "bb", "replacement": "a"}]}
    # "aaaa" -> "bbbb" -> "aa"; "aa" -> "bb" -> "a"; "a" -> "b"; "b" settled.
    assert _run(app_module, [rl], "aaaa") == "b"
    del rl["repeat"]
    assert _run(app_module, [rl], "aaaa") == "aa"


def test_the_trace_names_each_pass(app_module):
    trace: list = []
    _run(app_module, [_upper(repeat=True)], "Hals-nasen-ohren-arzt", trace=trace)
    assert [t[0] for t in trace] == ["#1 Hyphen", "#1 Hyphen (pass 2)", "#1 Hyphen (pass 3)"]


def test_per_model_scoping_is_unchanged(app_module):
    rule = _upper(repeat=True, enabled=False)
    app_module.cfg.MODEL_OVERRIDES = {
        "on": {"PIPELINE_RULES_INCLUDE": ["hyphen"]},
        "off": {"PIPELINE_RULES_EXCLUDE": ["hyphen"]},
    }
    text = "Hals-nasen-ohren-arzt"
    assert _run(app_module, [rule], text) == text
    assert _run(app_module, [rule], text, model_name="on") == "Hals-Nasen-Ohren-Arzt"
    rule["enabled"] = True
    assert _run(app_module, [rule], text, model_name="off") == text
    assert _run(app_module, [dict(rule, languages=["en"])], text, language="de") == text


def test_the_seam_walk_repeats_like_the_engine(app_module):
    """seam_culprit gives each text the engine's passes: "Qcd" leaves pass 1
    as "Qdd" (still extends "Q"), and only pass 2 joins "Qd" into "R"."""
    rl = {"name": "rl", "label": "RL", "type": "regex-list", "repeat": True,
          "entries": [{"pattern": "Qd", "replacement": "R", "label": "join"},
                      {"pattern": "cd", "replacement": "dd", "label": "spread"}]}
    assert _run(app_module, [rl], "Qcd") == "Rd"
    assert pl_engine.seam_culprit("Q", "Qcd") == "#1.1 RL · join"
    del rl["repeat"]
    assert _run(app_module, [rl], "Qcd") == "Qdd"
    assert pl_engine.seam_culprit("Q", "Qcd") == "none found"


# ---- schema -----------------------------------------------------------------

_ADAPTER = TypeAdapter(settings_schema.PipelineRule)


@pytest.mark.parametrize("rule", [
    _upper(),
    {"name": "rl", "label": "RL", "type": "regex-list", "entries": []},
    {"name": "m", "label": "M", "type": "callback:map", "map": {"a": "b"}},
    {"name": "d", "label": "D", "type": "callback:dedup", "pattern": "[,.]{2,}"},
    {"name": "w", "label": "W", "type": "callback:lowercase-wordlist",
     "pattern": "x", "wordlist": ["Und"]},
])
def test_every_rule_type_but_terminal_takes_repeat(rule):
    out = _ADAPTER.validate_python(dict(rule, repeat=True)).model_dump(exclude_none=True)
    assert out["repeat"] is True


def test_repeat_off_is_stored_absent():
    """Absent and false dump alike, so a rule saved before the switch existed
    keeps its dump (and every fingerprint over it)."""
    assert canon_rules([_upper(repeat=False)]) == canon_rules([_upper()])
    assert "repeat" not in canon_rules([_upper(repeat=False)])[0]


@pytest.mark.parametrize("bad", ["yes", "true", 1, 0, [], {}])
def test_repeat_must_be_a_bool(bad):
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(_upper(repeat=bad))


def test_the_terminal_rule_refuses_repeat():
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(
            {"name": "trim-edges", "label": "Trim", "type": "terminal", "repeat": True})


def test_a_save_keeps_repeat_on_and_stores_off_as_absent(tmp_path):
    """The editor's Save goes through save_overrides: on is written, off
    (unchecked: the editor deletes the key; a false from a hand-rolled body
    alike) leaves no key behind."""
    from faster_whisper_backend.settings import config_store as cs

    p = str(tmp_path / "config.local.json")
    term = {"name": "trim-edges", "label": "Trim", "type": "terminal"}
    for sent, stored in ((True, True), (False, None)):
        cs.save_overrides({"PIPELINE_RULES": [_upper(repeat=sent), term]}, p)
        on_disk = json.loads(open(p, encoding="utf-8").read())["PIPELINE_RULES"][0]
        assert on_disk.get("repeat") is stored
        assert cs.load_overrides(p)["PIPELINE_RULES"][0].get("repeat") is stored

