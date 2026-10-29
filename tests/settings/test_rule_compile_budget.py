"""A save's in-process regex compiles are charged against the guard's budget.

`_validate_pipeline_rules` compiles every pattern in-process BEFORE the
killable out-of-process guard. Compiling is not cheap for every pattern: with
IGNORECASE, CPython walks each code point of a character-class range, so a
512-char run of `[\\x00-\\U0010ffff]` takes ~0.4 s. A quick-config user could
submit 200 of them in one regex-list rule and pin a save thread for minutes
— the guard's 2 s timeout never applied, because the child ran only after.
"""

import re
import subprocess

import pytest
from pydantic import ValidationError

from faster_whisper_backend.pipeline import regex_guard
from faster_whisper_backend.settings import schema as settings_schema

# ~4 ms to compile (walks all 1.1M code points); far above timer resolution.
_SLOW_TO_COMPILE = "(?i)[\x00-\U0010ffff]"


def _rules(name, pattern):
    return [
        {"name": name, "label": name, "type": "regex-list",
         "entries": [{"pattern": pattern, "replacement": "x"}]},
        {"name": "trim-edges", "label": "Trim", "type": "terminal"},
    ]


@pytest.fixture
def no_compile_budget(monkeypatch):
    # Every guarded compile overspends a zero budget; the spy proves the save
    # is refused before the out-of-process guard would have been reached.
    monkeypatch.setattr(regex_guard, "_GUARD_TIMEOUT", 0.0)
    monkeypatch.setattr(regex_guard, "_PER_CHECK_BUDGET", 0.0)
    monkeypatch.setattr(regex_guard, "_GUARD_TIMEOUT_MAX", 0.0)
    calls = []
    monkeypatch.setattr(regex_guard, "validate", calls.append)
    re.purge()  # a cached compile would cost ~nothing
    return calls


def test_save_refuses_once_the_compile_budget_is_spent(no_compile_budget):
    with pytest.raises(ValidationError) as ei:
        settings_schema.AdminConfig.model_validate(
            {"PIPELINE_RULES": _rules("slow", _SLOW_TO_COMPILE)},
            context={"guard_regex": True})
    assert "rule 0 ('slow') entry 0: compiling the submitted patterns took" in str(ei.value)
    assert no_compile_budget == []


def test_load_and_untouched_rules_are_not_charged(no_compile_budget):
    # Load (no guard_regex context): the admin-owned file validates as before.
    settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": _rules("slow", _SLOW_TO_COMPILE)})
    re.purge()
    # A save scoped to other rules does not charge the untouched one.
    settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": _rules("slow", _SLOW_TO_COMPILE)},
        context={"guard_regex": True, "guard_slugs": frozenset({"other"})})


def test_structural_screen_is_charged_against_the_same_budget(monkeypatch):
    """The in-process screen compiles each branch atom again, IGNORECASE:
    here ~40 ms against a top-level compile the re cache serves for free, so
    a resent save used to pass the compile budget and then spend unbounded
    time screening before the killable child ever started."""
    monkeypatch.setattr(regex_guard, "_GUARD_TIMEOUT", 5e-3)
    monkeypatch.setattr(regex_guard, "_PER_CHECK_BUDGET", 0.0)
    monkeypatch.setattr(regex_guard, "_GUARD_TIMEOUT_MAX", 5e-3)
    child = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: child.append(a)
                        or subprocess.CompletedProcess(a, 0, '{"ok": true}', ""))
    cls = r"[\x00-+\--\U0010fff%d]"
    pat = "(?:(?:" + cls % 1 + "|" + cls % 2 + cls % 3 + "),)+"
    re.compile(pat)  # cached: the charged top-level compile costs ~nothing
    with pytest.raises(ValidationError) as ei:
        settings_schema.AdminConfig.model_validate(
            {"PIPELINE_RULES": _rules("wide", pat)}, context={"guard_regex": True})
    assert "rule 0 ('wide') entry 0: checking the submitted patterns took" in str(ei.value)
    assert child == []
