"""POST /settings/test-pipeline with a regex whose range repeat count is past
sre's MAXREPEAT. re.compile raises OverflowError (not re.error) for it, and
the fixed-count screen only checks fixed counts, so the dry run must catch it
itself, like the save path and the engine do."""

from tests.admin._pipeline_helpers import dry_run, list_step, upper_rule

_HUGE_RANGE = "a{1,4294967296}"


def test_test_pipeline_huge_range_repeat_is_a_step_error(client):
    step = dry_run(client, [upper_rule(_HUGE_RANGE)], "aaa")["steps"][0]
    assert step["after"] == "aaa"
    assert "repetition number is too large" in (step["error"] or "")


def test_test_pipeline_regex_list_skips_huge_range_repeat_entry(client):
    step = list_step(client, [{"pattern": "a", "replacement": "b"},
                              {"pattern": _HUGE_RANGE, "replacement": "c"}], sample="aaa")
    assert step["after"] == "bbb"
    assert step["matches"] == 3
    assert "repetition number is too large" in (step["error"] or "")
