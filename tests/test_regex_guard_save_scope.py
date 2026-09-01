"""A legacy rule that fails today's structural screen must not brick saves
that do not touch it.

`regex_guard._nested_repetition` is deterministic: `(\\w+ ?)+` passed the old
timed probe in microseconds (which is how such a rule got onto disk) but is
now refused on shape. `config_store.save_overrides` always validated the
WHOLE merged document with `guard_regex`, so once one stored rule trips the
screen, NO admin save of ANY setting could complete until config.local.json
was hand-edited. The guard must only re-screen rules the save submits.
"""

import json

import pytest

import config_store as cs


_LEGACY_RULES = [
    {"name": "legacy", "label": "legacy", "type": "regex-list",
     "entries": [{"pattern": r"(\w+ ?)+", "replacement": "x"}]},
    {"name": "trim-edges", "label": "Trim", "type": "terminal"},
]


def _write_legacy(path):
    path.write_text(json.dumps({"PIPELINE_RULES": _LEGACY_RULES}), encoding="utf-8")


def test_settings_only_save_ignores_a_stored_rule_that_fails_the_screen(tmp_path):
    p = tmp_path / "config.local.json"
    _write_legacy(p)
    changed = cs.save_overrides({"BEST_OF": 3}, str(p))
    assert changed == {"BEST_OF": 3}
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["BEST_OF"] == 3
    # The untouched rule survives byte-for-byte; the save did not rewrite it.
    assert on_disk["PIPELINE_RULES"][0]["entries"][0]["pattern"] == r"(\w+ ?)+"


def test_submitting_the_rules_themselves_still_trips_the_screen(tmp_path):
    p = tmp_path / "config.local.json"
    _write_legacy(p)
    with pytest.raises(cs.ValidationError) as ei:
        cs.save_overrides({"PIPELINE_RULES": _LEGACY_RULES}, str(p))
    assert "nested repetition" in str(ei.value)
