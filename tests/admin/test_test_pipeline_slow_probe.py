"""POST /settings/test-pipeline runs every pattern over the sample in a killable
child before matching it in-process. sre holds the GIL for the whole match, so
the 2 s guard thread's join cannot return while `(?:|){N}(?!)` (no nested
repetition, a tiny fixed count) backtracks 2**N times at every position: the
whole server would freeze, not one core."""

import subprocess
import time

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
                        lambda text, *a: calls.append(text) or real(text, *a))
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


def _count_probe_timeouts(monkeypatch):
    killed = []
    real = subprocess.run

    def _run(*a, **k):
        try:
            return real(*a, **k)
        except subprocess.TimeoutExpired:
            killed.append(k.get("timeout"))
            raise
    monkeypatch.setattr(subprocess, "run", _run)
    return killed


def test_a_pattern_found_slow_is_not_timed_again_by_each_later_walk(client, monkeypatch):
    # Every map changes the text, so the next pattern rule walks again; the
    # list after it writes the text back, so the slow tail is handed the
    # same text in every walk and must cost one kill, not one per walk.
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    killed = _count_probe_timeouts(monkeypatch)
    rules = []
    for n in "123":
        rules += [{"name": "up" + n, "type": "callback:upper", "enabled": True,
                   "pattern": "q"},
                  {"name": "m" + n, "type": "callback:map", "enabled": True,
                   "map": {"1": "a"}},
                  {"name": "rl" + n, "type": "regex-list", "enabled": True,
                   "entries": [{"pattern": "a", "replacement": "1"}]}]
    rules.append({"name": "slow", "type": "callback:upper", "enabled": True,
                  "pattern": _EMPTY_BACKTRACK})
    r = client.post("/settings/test-pipeline", json={"sample": "1 1 1 1", "rules": rules})
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert [s["after"] for s in steps[1:3]] == ["a a a a", "1 1 1 1"]
    assert steps[-1].get("not_run") is True
    assert len(killed) == 1


def test_probe_time_across_walks_is_capped(client, monkeypatch):
    # Each `(?:|c){18}(?!)` is slow only on the c-runs its own map writes, so
    # every walk kills a NEW unit: the request's probe time still stops at
    # rounds x timeout, and what was not probed is not run.
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    killed = _count_probe_timeouts(monkeypatch)
    rules = []
    for n, c in zip("123456", "bcdefg"):
        rules += [{"name": "m" + n, "type": "callback:map", "enabled": True,
                   "map": {n: c * 60}},
                  {"name": "up" + n, "type": "callback:upper", "enabled": True,
                   "pattern": "(?:|%s){18}(?!)" % c}]
    r = client.post("/settings/test-pipeline",
                    json={"sample": "1 1 2 2 3 3 4 4 5 5 6 6", "rules": rules})
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert all(s.get("not_run") is True for s in steps[1::2])
    assert sum(killed) <= 0.2 * admin_routes._TEST_PIPELINE_PROBE_ROUNDS + 1e-6


def test_a_pattern_slow_on_other_text_is_timed_again_on_the_text_it_is_handed(client, monkeypatch):
    # Walk 1 times the entry on the c-run (the child cannot replay the map);
    # the map turns the real text into "ok", where the entry is instant, so
    # the walk from there must time it again instead of skipping it as slow.
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    r = client.post("/settings/test-pipeline", json={
        "sample": "c" * 100,
        "rules": [{"name": "up", "type": "callback:upper", "enabled": True,
                   "pattern": "q"},
                  {"name": "m", "type": "callback:map", "enabled": True,
                   "map": {"c" * 100: "ok"}},
                  {"name": "rl", "type": "regex-list", "enabled": True, "entries": [
                      {"pattern": _GROWN_BACKTRACK.replace("|)", "|c)") + "|ok",
                       "replacement": "DONE"}]}],
    })
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps[1]["after"] == "ok"
    assert steps[2].get("not_run") is not True
    assert steps[2]["after"] == "DONE"


def test_healthy_walks_do_not_spend_the_probe_time(client, monkeypatch):
    # Every upper rule changes the text, so each pattern rule walks again;
    # walks that time nothing out must not use up the request's probe time
    # and leave the tail reported as too slow.
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    words = ["w%d" % n for n in range(60)]
    rules = [{"name": w, "type": "callback:upper", "enabled": True,
              "pattern": r"\b%s\b" % w} for w in words]
    rules.append({"name": "rl", "type": "regex-list", "enabled": True,
                  "entries": [{"pattern": "W59", "replacement": "end"}]})
    r = client.post("/settings/test-pipeline",
                    json={"sample": " ".join(words), "rules": rules})
    assert r.status_code == 200, r.text
    body = r.json()
    assert not [s["ordinal"] for s in body["steps"] if s.get("not_run")]
    assert body["final"] == " ".join(w.upper() for w in words[:-1]) + " end"


def test_probe_budget_is_charged_only_by_killed_rounds(monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    budget = {"left": 0.6}
    assert admin_routes._probe_chain("aaaa", _units(["a"]), None, budget)[1:] == (set(), None)
    assert budget["left"] == 0.6
    _marks, slow, stop = admin_routes._probe_chain(
        "aaaa", _units([_EMPTY_BACKTRACK, "a"]), None, budget)
    assert (slow, stop) == ({1}, None)
    assert abs(budget["left"] - 0.4) < 1e-9


def test_a_round_cut_short_by_the_budget_blames_no_unit(monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 2.0)
    budget = {"left": 0.2}
    _marks, slow, stop = admin_routes._probe_chain(
        "aaaa", _units(["a", "(?:|){24}(?!)"]), None, budget)
    assert slow == set()
    assert stop == 2


def test_probe_walks_that_each_finish_still_stop_at_the_request_wall_time(client, monkeypatch):
    # Every upper rule changes the text, so each pattern rule walks again and
    # each walk finishes: they must still stop at the request's probe wall
    # time, and what they never reached is reported as unprobed, not as slow.
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.2)
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_WALL", 0.3)
    spent = []
    real = subprocess.run

    def _run(*a, **k):
        t0 = time.monotonic()
        try:
            return real(*a, **k)
        finally:
            spent.append(time.monotonic() - t0)
    monkeypatch.setattr(subprocess, "run", _run)
    words = ["w%d" % n for n in range(60)]
    rules = [{"name": w, "type": "callback:upper", "enabled": True,
              "pattern": r"\b%s\b" % w} for w in words]
    r = client.post("/settings/test-pipeline",
                    json={"sample": " ".join(words), "rules": rules})
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    held = [s for s in steps if s.get("not_run")]
    assert held and held[-1]["ordinal"] == 60
    assert all("ran out" in s["error"] and "did not finish" not in s["error"]
               for s in held)
    assert steps[0]["after"].startswith("W0 ")
    # One walk past the wall at most, and every walk under the round timeout.
    assert sum(spent) <= 0.3 + 0.2 + 0.05 * len(spent)
    assert len(spent) < 60


def test_fast_patterns_adding_up_past_the_timeout_blame_none_of_them(monkeypatch):
    # Each `(?:|){18}(?!)` takes ~0.04 s on "aaaa"; twelve of them pass the
    # round's timeout together, so the one running at the kill must get a
    # round of its own instead of being reported as too slow.
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_PROBE_TIMEOUT", 0.3)
    _marks, slow, _stop = admin_routes._probe_chain(
        "aaaa", _units(["(?:|){18}(?!)"] * 12))
    assert slow == set()
