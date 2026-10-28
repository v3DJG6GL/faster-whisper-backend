"""POST /settings/test-pipeline with a regex-list entry whose replacement
fails at substitution time with something other than re.error: an unknown
group name raises IndexError, a non-string replacement raises TypeError. The
engine skips only that entry, so the dry run must keep the earlier entries'
output and show the failure as an advisory."""

import pytest


@pytest.mark.parametrize(("replacement", "err_part"), [
    (r"\g<nope>", "unknown group name"),
    (5, ""),
])
def test_regex_list_bad_replacement_skips_only_that_entry(client, replacement, err_part):
    r = client.post("/settings/test-pipeline", json={
        "sample": "foo a",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "foo", "replacement": "bar"},
            {"pattern": "(a)", "replacement": replacement},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["after"] == "bar a"
    assert step["matches"] == 1
    assert step["error"]
    assert err_part in step["error"]
