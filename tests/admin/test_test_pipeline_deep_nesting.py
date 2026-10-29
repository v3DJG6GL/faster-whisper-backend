"""POST /settings/test-pipeline with a pattern whose groups nest too deep for
sre's parser. re.compile raises RecursionError (not re.error) for it, so the
dry run must catch it itself, as the save path reports it as an invalid regex:
a step error on a callback rule, a skipped entry in a regex list."""

_DEEP = "(" * 512


def test_test_pipeline_deeply_nested_pattern_is_a_step_error(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "foo a",
        "rules": [{"name": "up", "type": "callback:upper", "enabled": True,
                   "pattern": _DEEP}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["after"] == "foo a"
    assert "recursion" in (step["error"] or "")


def test_test_pipeline_regex_list_skips_deeply_nested_entry(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "foo a",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "foo", "replacement": "bar"},
            {"pattern": _DEEP, "replacement": "x"},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    # Only the bad entry is skipped; the valid sibling still applies.
    assert step["after"] == "bar a"
    assert step["matches"] == 1
    assert "recursion" in (step["error"] or "")
