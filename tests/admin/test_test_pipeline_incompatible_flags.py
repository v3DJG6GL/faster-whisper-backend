"""POST /settings/test-pipeline with a pattern whose inline flags conflict.
re.compile raises a plain ValueError (not re.error) for `(?a)(?u)`, so the dry
run must catch it itself: a step error on a callback rule, a skipped entry in
a regex list whose valid siblings still apply."""

_BAD_FLAGS = "(?a)(?u)bar"


def test_test_pipeline_incompatible_flags_is_a_step_error(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "foo bar",
        "rules": [{"name": "up", "type": "callback:upper", "enabled": True,
                   "pattern": _BAD_FLAGS}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["after"] == "foo bar"
    assert "incompatible" in (step["error"] or "")


def test_test_pipeline_regex_list_skips_incompatible_flags_entry(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "foo bar",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "foo", "replacement": "FOO"},
            {"pattern": _BAD_FLAGS, "replacement": "x"},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    # Only the bad entry is skipped; the valid sibling still applies.
    assert step["after"] == "FOO bar"
    assert step["matches"] == 1
    assert "incompatible" in (step["error"] or "")
