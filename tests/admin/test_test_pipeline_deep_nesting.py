"""POST /settings/test-pipeline with a pattern whose groups nest too deep for
sre's parser. re.compile raises RecursionError (not re.error) for it, so the
dry run must catch it itself, as the save path reports it as an invalid regex:
a step error on a callback rule, a skipped entry in a regex list."""

import sys

import pytest

from faster_whisper_backend.admin import routes as admin_routes
from tests.admin._pipeline_helpers import dry_run, list_rule, list_step, upper_rule

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
        if sample is deep and r.json() == {"detail": "There was an error parsing the body"}:
            # A smaller C stack (Windows) gives json.loads its RecursionError
            # first, inside FastAPI's body parse: refused there.
            continue
        assert r.json() == {"error": "sample must be a string"}


def test_test_pipeline_null_or_missing_sample_is_the_empty_sample(client):
    for body in ('{"sample": null, "rules": []}', '{"rules": []}'):
        r = _post_raw(client, body)
        assert r.status_code == 200, r.text
        assert r.json() == {"steps": [], "final": ""}


def test_test_pipeline_render_recursion_error_is_refused(client, monkeypatch):
    def _too_deep(steps, final):
        raise RecursionError("Stack overflow")
    monkeypatch.setattr(admin_routes, "_render_dry_run", _too_deep)
    r = client.post("/settings/test-pipeline", json={
        "sample": "ab", "rules": [list_rule([{"pattern": "a", "replacement": "b"}])]})
    assert r.status_code == 400, r.text
    assert "JSON cannot carry" in r.json()["error"]


# The depth json.loads takes and json.dumps refuses depends on the thread's
# stack; this one is measured for Linux, and the handling is the same everywhere.
@pytest.mark.skipif(sys.platform == "win32",
                    reason="nesting depth measured for the Linux thread stack")
@pytest.mark.parametrize("key", ["label", "type", "name"])
def test_test_pipeline_deeply_nested_label_type_or_name_is_refused(client, key):
    """A rule's label, type and name are echoed untyped in its step, and
    json.loads builds an object nested deeper than the render can walk: that
    body was an unhandled RecursionError."""
    rule = {"type": '"regex-list"', "name": '"n"',
            "entries": '[{"pattern": "a", "replacement": "b"}]',
            key: '{"a":' * 45_000 + "1" + "}" * 45_000}
    body = ('{"sample": "ab", "rules": [{'
            + ", ".join(f'"{k}": {v}' for k, v in rule.items()) + "}]}")
    r = _post_raw(client, body)
    assert r.status_code == 400, r.text
    assert "JSON cannot carry" in r.json()["error"]
