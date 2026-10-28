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


# ~5 ms per position: harmless on a 1-char sample, ~0.5 s per pass over the
# 100 chars an earlier rule makes of it (in-process: finditer AND sub, ~1 s
# with the GIL held — under the 2 s join, so the old code ran it to the end).
_GROWN_BACKTRACK = "(?:|){18}(?!)"


def _grow(n=100):
    return {"name": "grow", "type": "regex-list", "enabled": True,
            "entries": [{"pattern": "x", "replacement": "a" * n}]}


def test_callback_pattern_slow_only_on_an_earlier_rules_output_is_not_run(client, monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    r = client.post("/settings/test-pipeline", json={
        "sample": "x",
        "rules": [_grow(), {"name": "up", "type": "callback:upper", "enabled": True,
                            "pattern": _GROWN_BACKTRACK}],
    })
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps[0]["after"] == "a" * 100
    assert steps[1].get("not_run") is True
    assert "killable probe" in steps[1]["error"]
    assert steps[1]["after"] == "a" * 100


def test_regex_list_entry_slow_only_on_an_earlier_entrys_output_is_not_run(client, monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    r = client.post("/settings/test-pipeline", json={
        "sample": "x",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "x", "replacement": "a" * 100},
            {"pattern": _GROWN_BACKTRACK, "replacement": "z"},
            {"pattern": "a{100}", "replacement": "done"},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step.get("not_run") is True
    assert "killable probe" in step["error"]
    # The entries around the culprit still apply, in order.
    assert step["after"] == "done"


def test_text_a_callback_changed_is_probed_again_before_the_next_rule(client, monkeypatch):
    # The child cannot replay a map, so the next rule's input digest differs
    # from the walk's and the probe walks again from there on the real text.
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    r = client.post("/settings/test-pipeline", json={
        "sample": "xy",
        "rules": [{"name": "m", "type": "callback:map", "enabled": True,
                   "map": {"xy": "a" * 60}},
                  {"name": "up", "type": "callback:upper", "enabled": True,
                   "pattern": _GROWN_BACKTRACK}],
    })
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps[0]["after"] == "a" * 60
    assert steps[1].get("not_run") is True


def test_ordinary_chain_runs_with_one_probe(client, monkeypatch):
    calls = []
    real = admin_routes._probe_chain
    monkeypatch.setattr(admin_routes, "_probe_chain",
                        lambda text, units: calls.append(text) or real(text, units))
    r = client.post("/settings/test-pipeline", json={
        "sample": " x foo ",
        "rules": [_grow(3),
                  {"name": "up", "type": "callback:upper", "enabled": True,
                   "pattern": "foo"},
                  {"name": "t", "type": "terminal"}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["final"] == "aaa FOO"
    # The upper callback changed the text, but no pattern rule follows it.
    assert calls == [" x foo "]


def _units(sample_pats):
    return [["mark", 0], *(["match", 0, p] for p in sample_pats)]


def test_probe_names_each_slow_unit_and_keeps_probing_the_rest(monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    slow2 = "(?:|){23}(?!)"
    marks, slow, stop = admin_routes._probe_chain(
        "aaaa", _units(["a", _EMPTY_BACKTRACK, "b", slow2, "c"]))
    assert slow == {2, 4}
    assert stop is None
    assert marks == {0: admin_routes._text_sha("aaaa")}


def test_probe_reports_where_it_gave_up(monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_ROUNDS", 1)
    _marks, slow, stop = admin_routes._probe_chain(
        "aaaa", _units([_EMPTY_BACKTRACK, "a"]))
    assert slow == {1}
    assert stop == 2


def test_probe_passes_ordinary_patterns():
    marks, slow, stop = admin_routes._probe_chain(
        "foo bar", [["mark", 0], ["sub", 0, "foo", "baz"], ["sub", 0, "(", ""],
                    ["mark", 1], ["match", 1, r"\bbar\b"], ["trim", 2], ["mark", 3]])
    assert (slow, stop) == (set(), None)
    assert marks == {0: admin_routes._text_sha("foo bar"),
                     1: admin_routes._text_sha("baz bar"),
                     3: admin_routes._text_sha("baz bar")}


def test_probe_fails_open_when_the_child_cannot_start(monkeypatch):
    def _no_spawn(*a, **k):
        raise OSError("no fork")
    monkeypatch.setattr(subprocess, "run", _no_spawn)
    assert admin_routes._probe_chain("aaaa", _units([_EMPTY_BACKTRACK])) is None
