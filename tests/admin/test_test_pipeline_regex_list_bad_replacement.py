"""POST /settings/test-pipeline with a regex-list entry whose replacement
fails at substitution time with something other than re.error: an unknown
group name raises IndexError, a non-string replacement raises TypeError. A
save refuses such an entry, but the dry run runs the unsaved card, so it must
skip only that entry, keep the earlier entries' output and show the failure
as an advisory."""

import pytest

from tests.admin._pipeline_helpers import list_step


@pytest.mark.parametrize(("replacement", "err_part"), [
    (r"\g<nope>", "unknown group name"),
    (5, ""),
])
def test_regex_list_bad_replacement_skips_only_that_entry(client, replacement, err_part):
    step = list_step(client, [{"pattern": "foo", "replacement": "bar"},
                              {"pattern": "(a)", "replacement": replacement}],
                     sample="foo a")
    assert step["after"] == "bar a"
    assert step["matches"] == 1
    assert step["error"]
    assert err_part in step["error"]
