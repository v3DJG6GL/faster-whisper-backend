"""POST /settings/test-pipeline with a pattern whose inline flags conflict.
re.compile raises a plain ValueError (not re.error) for `(?a)(?u)`, so the dry
run must catch it itself: a step error on a callback rule, a skipped entry in
a regex list whose valid siblings still apply."""

from tests.admin._pipeline_helpers import dry_run, list_step

_BAD_FLAGS = "(?a)(?u)bar"


def test_test_pipeline_incompatible_flags_is_a_step_error(client):
    step = dry_run(client, [{"name": "up", "type": "callback:upper", "enabled": True,
                             "pattern": _BAD_FLAGS}], "foo bar")["steps"][0]
    assert step["after"] == "foo bar"
    assert "incompatible" in (step["error"] or "")


def test_test_pipeline_regex_list_skips_incompatible_flags_entry(client):
    step = list_step(client, [{"pattern": "foo", "replacement": "FOO"},
                              {"pattern": _BAD_FLAGS, "replacement": "x"}], sample="foo bar")
    assert step["after"] == "FOO bar"
    assert step["matches"] == 1
    assert "incompatible" in (step["error"] or "")
