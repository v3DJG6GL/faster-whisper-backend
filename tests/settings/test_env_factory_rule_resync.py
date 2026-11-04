"""WHISPER_PIPELINE_RULES gets the untouched-factory-copy resync a
config.local.json copy gets (config_store._resync_untouched_factory_copies),
in memory on every start: the env var is never written.

Without it an env copy pasted from the editor before a factory change kept
that change out (a new entry, a new note) while the same copy saved in
config.local.json picked it up.
"""

import copy
import json

from faster_whisper_backend.settings import config
from faster_whisper_backend.settings import config_store as cs
from tests.settings.test_config_env import _assert_warned, _reload_with_env
from tests.settings.test_factory_rule_resync import (
    _QUOTE_LABELS, _factory_rules, _older_tidy, _rev,
)

_TIDY = "digit-and-whitespace-tidy"


def _env_tidy(monkeypatch, rules):
    _reload_with_env(monkeypatch, WHISPER_PIPELINE_RULES=json.dumps(rules))
    return next(r for r in config.PIPELINE_RULES if r["name"] == _TIDY)


def test_an_untouched_older_factory_copy_in_the_env_becomes_the_current_rule(
        monkeypatch, restore_config):
    rules = copy.deepcopy(_factory_rules())
    _older_tidy(rules)
    got = _env_tidy(monkeypatch, rules)
    want = next(r for r in cs.load_factory_rules() if r["name"] == _TIDY)
    assert got == {**want, "config_rev": _rev(want)}
    _assert_warned("WHISPER_PIPELINE_RULES: updated rules", _TIDY)


def test_edited_and_unrecorded_env_copies_stay_as_written(monkeypatch, restore_config):
    rules = copy.deepcopy(_factory_rules())
    edited = _older_tidy(rules)
    edited["entries"][-1]["note"] = "mine"           # rev no longer matches
    unrecorded = copy.deepcopy(rules)
    del _older_tidy(unrecorded)["config_rev"]        # hand-written, no rev
    for env_rules in (rules, unrecorded):
        want = next(r for r in env_rules if r["name"] == _TIDY)
        got = _env_tidy(monkeypatch, env_rules)
        assert got["entries"] == want["entries"]
        assert got.get("config_rev") == want.get("config_rev")
        assert not set(_QUOTE_LABELS) & {e["label"] for e in got["entries"]}
