"""POST /settings/test-pipeline with a regex whose range repeat count is past
sre's MAXREPEAT. re.compile raises OverflowError (not re.error) for it, and
the fixed-count screen only checks fixed counts, so the dry run must catch it
itself, like the save path and the engine do."""

_HUGE_RANGE = "a{1,4294967296}"


def test_test_pipeline_huge_range_repeat_is_a_step_error(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "aaa",
        "rules": [{"name": "up", "type": "callback:upper", "enabled": True,
                   "pattern": _HUGE_RANGE}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["after"] == "aaa"
    assert "repetition number is too large" in (step["error"] or "")


def test_test_pipeline_regex_list_skips_huge_range_repeat_entry(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "aaa",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "a", "replacement": "b"},
            {"pattern": _HUGE_RANGE, "replacement": "c"},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    # Only the bad entry is skipped; the valid sibling still applies.
    assert step["after"] == "bbb"
    assert step["matches"] == 3
    assert "repetition number is too large" in (step["error"] or "")
