"""POST /settings/test-pipeline runs every pattern over the sample in a killable
child before matching it in-process. sre holds the GIL for the whole match, so
the 2 s guard thread's join cannot return while `(?:|){N}(?!)` (no nested
repetition, a tiny fixed count) backtracks 2**N times at every position: the
whole server would freeze, not one core."""

import subprocess

from faster_whisper_backend.admin import routes as admin_routes

# ~0.1-0.2 s per position on "" alone; well past the lowered budget on 4 chars.
_EMPTY_BACKTRACK = "(?:|){22}(?!)"


def test_regex_list_entry_too_slow_on_the_sample_is_not_run(client, monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    r = client.post("/settings/test-pipeline", json={
        "sample": "aaaa",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "a", "replacement": "b"},
            {"pattern": _EMPTY_BACKTRACK, "replacement": "x"},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["not_run"] is True
    assert "killable probe" in step["error"]
    # The valid sibling still applies.
    assert step["after"] == "bbbb"


def test_callback_pattern_too_slow_on_the_sample_is_not_run(client, monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    r = client.post("/settings/test-pipeline", json={
        "sample": "aaaa",
        "rules": [{"name": "up", "type": "callback:upper", "enabled": True,
                   "pattern": _EMPTY_BACKTRACK}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["not_run"] is True
    assert step["after"] == "aaaa"
    assert "killable probe" in step["error"]


def test_probe_names_each_slow_pattern_and_keeps_probing_the_rest(monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    slow2 = "(?:|){23}(?!)"
    assert admin_routes._probe_slow_patterns(
        "aaaa", ["a", _EMPTY_BACKTRACK, "b", slow2, "c"]) == {_EMPTY_BACKTRACK, slow2}


def test_probe_passes_ordinary_patterns():
    assert admin_routes._probe_slow_patterns("foo bar", ["foo", r"\bbar\b", "("]) == set()


def test_probe_fails_open_when_the_child_cannot_start(monkeypatch):
    def _no_spawn(*a, **k):
        raise OSError("no fork")
    monkeypatch.setattr(subprocess, "run", _no_spawn)
    assert admin_routes._probe_slow_patterns("aaaa", [_EMPTY_BACKTRACK]) == set()
