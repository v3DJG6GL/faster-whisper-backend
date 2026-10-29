"""Engine tests for the callback:upper rule: it uppercases group(2) in place
and keeps every other part of the match. The pattern is editable from
/settings/pipeline and /quick-config, so it is not limited to the two seeded
shapes whose match happens to be exactly group(1) + group(2)."""

import re

from faster_whisper_backend.pipeline import engine as pl_engine


def _sub(pattern, text):
    return re.sub(pattern, pl_engine._upper_callback, text)


def test_seeded_patterns_unchanged():
    assert _sub(r"(-)([a-zäöüß])", "E-mail an x-ray") == "E-Mail an x-Ray"
    assert _sub(r"([.?!]\s+|\n+\s*)([a-zäöüß])", "Gut. danke\n\n  ja") == "Gut. Danke\n\n  Ja"


def test_text_outside_the_two_groups_is_kept():
    # literal between the groups, a third group, and an ungrouped prefix
    assert _sub(r"([.?!]) ([a-zäöüß])", "Gut. danke") == "Gut. Danke"
    assert _sub(r"([.?!]\s+)([a-zäöüß])(\w*)", "Gut. danke sehr") == "Gut. Danke sehr"
    assert _sub(r"x(-)([a-z])", "ax-b") == "ax-B"


def test_missing_group_or_fewer_groups():
    assert _sub(r"(\.)?( )([a-z])", "a b") == "a b"  # group 1 absent: match kept as is
    assert _sub(r"(a)(b)?", "ac") == "ac"  # group 2 absent
    assert _sub(r"(b)", "abc") == "aBc"  # one group: whole match uppercased


def test_group_2_outside_the_match_keeps_the_text():
    # a lookahead/lookbehind capture lies outside the match: no letter is duplicated
    assert _sub(r"([.?!]\s+)(?=([a-zäöüß]))", "Gut. danke") == "Gut. danke"
    assert _sub(r"(?=([.?!]\s+([a-z])))\.", "Gut. danke") == "Gut. danke"
    assert _sub(r"(?<=([.?!])(\s))([a-z])", "end. foo") == "end. foo"
