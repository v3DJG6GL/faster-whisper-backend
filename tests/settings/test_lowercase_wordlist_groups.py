"""A save refuses a callback:lowercase-wordlist pattern with fewer than 3
groups. The replacer reads group 1 as the space after the terminator and
groups 2 + 3 as the next word; with fewer groups it keeps the match as is, so
a quick-config user's edited pattern would silently do nothing."""

import pytest
from pydantic import ValidationError

from faster_whisper_backend.settings import schema as settings_schema


def _rules(pattern):
    return [
        {"name": "lower", "label": "Lower", "type": "callback:lowercase-wordlist",
         "pattern": pattern, "wordlist": ["und"]},
        {"name": "trim-edges", "label": "Trim", "type": "terminal"},
    ]


@pytest.mark.parametrize("pattern", [r"[.?!](\s*)([A-ZÄÖÜ]\w*)", r"[.?!]\s*"])
def test_save_refuses_fewer_than_three_groups(pattern):
    with pytest.raises(ValidationError) as ei:
        settings_schema.AdminConfig.model_validate(
            {"PIPELINE_RULES": _rules(pattern)}, context={"guard_regex": True})
    assert "rule 0 ('lower'): a lowercase-wordlist pattern needs 3 groups" in str(ei.value)


def test_load_and_three_group_save_still_pass():
    # Load (no guard_regex context): an existing rule keeps loading.
    settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": _rules(r"[.?!](\s*)([A-ZÄÖÜ]\w*)")})
    settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": _rules(r"(?<!\d)[.?!](\s*)([A-ZÄÖÜ])(\w*)")},
        context={"guard_regex": True})
