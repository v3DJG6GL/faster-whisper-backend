"""POST /settings/test-pipeline with a pattern whose groups nest too deep for
sre's parser. re.compile raises RecursionError (not re.error) for it, so the
dry run must catch it itself, as the save path reports it as an invalid regex:
a step error on a callback rule, a skipped entry in a regex list."""

from tests.admin._pipeline_helpers import dry_run, list_step, upper_rule

_DEEP = "(" * 512


def test_test_pipeline_deeply_nested_pattern_is_a_step_error(client):
    step = dry_run(client, [upper_rule(_DEEP)], "foo a")["steps"][0]
    assert step["after"] == "foo a"
    assert "recursion" in (step["error"] or "")


def test_test_pipeline_regex_list_skips_deeply_nested_entry(client):
    step = list_step(client, [{"pattern": "foo", "replacement": "bar"},
                              {"pattern": _DEEP, "replacement": "x"}], sample="foo a")
    assert step["after"] == "bar a"
    assert step["matches"] == 1
    assert "recursion" in (step["error"] or "")


def _post_raw(client, body):
    return client.post("/settings/test-pipeline", content=body,
                       headers={"content-type": "application/json"})


def test_test_pipeline_non_string_sample_is_refused(client):
    """The sample was run as its str(): json.loads accepts a list nested far
    deeper than str() can walk, so that body was an unhandled RecursionError."""
    deep = "[" * 50_000 + "]" * 50_000
    for sample in (deep, "5", '{"a": 1}', '["a"]', "true"):
        r = _post_raw(client, '{"sample": ' + sample + ', "rules": []}')
        assert r.status_code == 400, r.text
        assert r.json() == {"error": "sample must be a string"}


def test_test_pipeline_null_or_missing_sample_is_the_empty_sample(client):
    for body in ('{"sample": null, "rules": []}', '{"rules": []}'):
        r = _post_raw(client, body)
        assert r.status_code == 200, r.text
        assert r.json() == {"steps": [], "final": ""}
