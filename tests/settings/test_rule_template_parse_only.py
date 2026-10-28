"""The in-process replacement-template check must parse, never match.

`_validate_pipeline_rules` used to smoke-test each template with
`re.compile(pat).sub(repl, "")`, which RUNS the pattern. `(?:|){N}(?!)`
passes the fixed-count screen yet backtracks 2**N times even on "", so a
save (before the killable out-of-process guard) or a startup load of such a
rule hung the validator for hours while holding the settings save lock.
"""

import time

import pytest
from pydantic import ValidationError

from faster_whisper_backend.settings import schema as settings_schema

# ~20 s per call before the fix on a fast machine (doubles with each extra
# alternation), well under a millisecond after it.
_EMPTY_INPUT_BOMB = "(?:|){30}(?!)"


def _rules(pattern, replacement):
    return [
        {"name": "b", "label": "b", "type": "regex-list",
         "entries": [{"pattern": pattern, "replacement": replacement}]},
        {"name": "trim-edges", "label": "Trim", "type": "terminal"},
    ]


def test_load_validation_does_not_run_the_pattern():
    t0 = time.monotonic()
    settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": _rules(_EMPTY_INPUT_BOMB, "x")})
    assert time.monotonic() - t0 < 2.0


def test_save_reaches_the_out_of_process_guard_without_matching_first(monkeypatch):
    from faster_whisper_backend.pipeline import regex_guard

    seen = []
    monkeypatch.setattr(regex_guard, "validate", seen.extend)
    t0 = time.monotonic()
    settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": _rules(_EMPTY_INPUT_BOMB, "x")},
        context={"guard_regex": True})
    assert time.monotonic() - t0 < 2.0
    assert [c[1] for c in seen] == [_EMPTY_INPUT_BOMB]


@pytest.mark.parametrize("pattern,replacement,needle", [
    ("(a)", r"\3", "invalid group reference"),
    ("(?P<n>a)", r"\g<x>", "unknown group name"),
    ("a", "\\", "bad escape"),
])
def test_bad_template_still_rejected_on_load(pattern, replacement, needle):
    with pytest.raises(ValidationError) as ei:
        settings_schema.AdminConfig.model_validate(
            {"PIPELINE_RULES": _rules(pattern, replacement)})
    assert "regex test failed" in str(ei.value)
    assert needle in str(ei.value)
