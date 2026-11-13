"""POST /settings/test-pipeline refuses a regex-list replacement or a map value
longer than a saved rule may carry: editor rules skip the schema adapter, so
the dry run applies the save's bounds itself."""

import threading
import time
from typing import get_args

from faster_whisper_backend.admin import routes as admin_routes
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import schema as settings_schema
from tests.admin._pipeline_helpers import dry_run, list_rule, list_step, refused_dry_run, upper_rule
from tests.conftest import record_calls


def test_caps_are_read_off_the_schema():
    # The fallbacks equal today's bounds, so pin the walks themselves.
    assert admin_routes._field_max_length(
        settings_schema.RegexListEntry.model_fields["replacement"]) == 512
    assert admin_routes._str_max_length(
        get_args(settings_schema.MapRule.model_fields["map"].annotation)[1]) == 64
    assert admin_routes._str_max_length(
        get_args(settings_schema.MapRule.model_fields["map"].annotation)[0]) == 64
    assert admin_routes._field_max_length(
        settings_schema.MapRule.model_fields["map"]) == 10_000
    assert admin_routes._TEST_PIPELINE_MAX_MAP_KEY == 64
    assert admin_routes._TEST_PIPELINE_MAX_MAP_KEYS == 10_000


def test_oversized_regex_list_replacement_is_rejected(client):
    error = refused_dry_run(
        client, [list_rule([{"pattern": "a", "replacement": "x" * 20000}])], "a" * 8192)
    assert "replacement too long" in error


def test_oversized_map_value_is_rejected(client):
    error = refused_dry_run(client, [{"name": "m", "type": "callback:map", "enabled": True,
                                      "map": {"komma": "," * 5000}}], "komma " * 100)
    assert "map value too long" in error


def _map_rule(mapping):
    return {"name": "m", "type": "callback:map", "enabled": True, "map": mapping}


def test_oversized_map_key_is_rejected(client):
    # A map runs in-process (it is not timed by the killable probe), and its
    # scan grows with the keys' length: a saved rule's key cap applies.
    error = refused_dry_run(client, [_map_rule({"k" * 65: "x"})], "k" * 65)
    assert error == "rule 1: map key too long (max 64 chars)"


def test_too_many_map_entries_is_rejected(client):
    error = refused_dry_run(
        client, [upper_rule("x"), _map_rule({f"w{i}": "x" for i in range(10_001)})], "w1")
    assert error == "rule 2: too many map entries (max 10000)"


def test_map_key_at_the_cap_still_runs(client):
    step = dry_run(client, [_map_rule({"k" * 64: "x"})], "k" * 64)["steps"][0]
    assert step["after"] == "x" and step["matches"] == 1


def test_map_of_another_rule_type_is_not_counted(client):
    # Only a callback:map rule reads `map`; a stray one elsewhere is ignored.
    rule = {**upper_rule("(a)"), "map": {"k" * 65: "x"}}
    assert dry_run(client, [rule], "a")["steps"][0]["after"] == "A"


def test_replacement_at_the_cap_still_runs(client):
    step = list_step(client, [{"pattern": "foo", "replacement": "y" * 512}], sample="foo")
    assert step["after"] == "y" * 512


def _bound_at(monkeypatch, n):
    """Make every walk's output bound `n`, whatever the sample's length."""
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", n)
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_FLOOR", n)


def test_chained_entries_stop_before_passing_the_output_bound(client, monkeypatch):
    # Entry 1 stays under the bound; entry 2 would multiply that text 512x in
    # one sub. The dry run must stop BEFORE running it, not after.
    _bound_at(monkeypatch, 40_000)
    j = dry_run(client, [list_rule([
        {"pattern": "x", "replacement": "x" * 512},
        {"pattern": "x", "replacement": "x" * 512},
    ])], "x" * 78)
    step = j["steps"][0]
    assert step["capped"] is True
    assert step["after"] == "x" * (78 * 512)
    assert len(j["final"]) <= 40_000


def test_backref_template_counts_its_expansion_against_the_bound(client, monkeypatch):
    # A 510-char template of `\g<0>` copies the whole match 102 times.
    _bound_at(monkeypatch, 40_000)
    step = list_step(client, [{"pattern": "(?s).+", "replacement": "\\g<0>" * 4},
                              {"pattern": "(?s).+", "replacement": "\\g<0>" * 102}],
                     sample="a" * 8000)
    assert step["capped"] is True
    assert step["after"] == "a" * 32000


def test_map_rule_stops_before_passing_the_output_bound(client, monkeypatch):
    _bound_at(monkeypatch, 40_000)
    j = dry_run(client, [
        list_rule([{"pattern": "a", "replacement": "komma " * 60}]),
        {"name": "m", "type": "callback:map", "enabled": True,
         "map": {"komma": "," * 64}},
        upper_rule("(a)"),
    ], "a" * 100)
    steps = j["steps"]
    assert len(steps[0]["after"]) == 36_000
    assert steps[1]["capped"] is True
    assert steps[1]["after"] == steps[0]["after"]
    # Engine parity: every later rule is skipped once the bound is hit.
    assert steps[2]["skipped"] is True and steps[2]["capped"] is True
    assert len(j["final"]) <= 40_000


def test_pattern_cap_is_read_off_the_schema():
    assert admin_routes._field_max_length(
        settings_schema.RegexListEntry.model_fields["pattern"]) == 512


def test_oversized_callback_pattern_is_rejected_before_the_screen(client, monkeypatch):
    def _screen(pat):
        raise AssertionError("structural screen ran on an over-long pattern")
    monkeypatch.setattr(admin_routes.regex_guard, "_nested_repetition", _screen)
    alternation = "(?:" + "|".join(f"w{i:06d}" for i in range(700)) + ")+"
    error = refused_dry_run(client, [{"name": "d", "type": "callback:dedup", "enabled": True,
                                      "pattern": alternation}], "w000001")
    assert error == "rule 1: pattern too long (max 512 chars)"


def test_oversized_regex_list_pattern_is_rejected(client):
    error = refused_dry_run(client, [list_rule([
        {"pattern": "a", "replacement": "b"},
        {"pattern": "a" * 513, "replacement": "b"},
    ])], "a")
    assert "pattern too long" in error


def test_pattern_at_the_cap_still_runs(client):
    step = list_step(client, [{"pattern": "a" * 512, "replacement": "b"}], sample="a" * 512)
    assert step["after"] == "b"


def _no_screen(monkeypatch):
    def _screen(*args):
        raise AssertionError("structural screen ran on a non-string pattern")
    monkeypatch.setattr(admin_routes.regex_guard, "_nested_repetition", _screen)
    monkeypatch.setattr(admin_routes.regex_guard, "check_fixed_counts", _screen)


# A list skips the str-only length cap, and its repr is a valid alternation.
_LIST_PATTERN = ["](?:" + "|".join(f"w{i:06d}" for i in range(800)) + ")+["]


def test_non_string_callback_pattern_is_refused_before_the_screens(client, monkeypatch):
    _no_screen(monkeypatch)
    step = dry_run(client, [{"name": "d", "type": "callback:dedup", "enabled": True,
                             "pattern": _LIST_PATTERN}], "w000001")["steps"][0]
    assert step["error"] == "pattern must be a string"
    assert step["after"] == "w000001"


def test_non_string_regex_list_pattern_is_skipped_before_the_screens(client, monkeypatch):
    _no_screen(monkeypatch)
    step = list_step(client, [{"pattern": _LIST_PATTERN, "replacement": "x"}], sample="w000001")
    assert step["error"] == "pattern must be a string"
    assert step["after"] == "w000001"


def test_entries_cap_is_read_off_the_schema():
    assert admin_routes._field_max_length(
        settings_schema.RegexListRule.model_fields["entries"]) == 200
    assert admin_routes._TEST_PIPELINE_MAX_ENTRIES == 200


def test_too_many_regex_list_entries_are_rejected(client):
    # Each distinct pattern costs a structural screen; a saved rule cannot
    # carry more than 200 entries, so the dry run refuses more too.
    entries = [{"pattern": f"w{i:04d}", "replacement": "x"} for i in range(201)]
    error = refused_dry_run(client, [list_rule(entries)], "w0001")
    assert error == "rule 1: too many entries (max 200)"


def test_regex_list_entries_at_the_cap_still_run(client):
    entries = [{"pattern": f"w{i:04d}", "replacement": "x"} for i in range(200)]
    step = list_step(client, entries, sample="w0199")
    assert step["after"] == "x"


def test_patterns_are_screened_off_the_event_loop(client, monkeypatch):
    # The screen is a regex compile plus pairwise overlap matching per
    # distinct pattern; run on the loop, a list of them stalls every client.
    from tests.conftest import on_event_loop

    seen = []

    def _screen(pat):
        seen.append(on_event_loop())
        return False
    monkeypatch.setattr(admin_routes.regex_guard, "_nested_repetition", _screen)
    step = list_step(client, [{"pattern": "foo", "replacement": "bar"},
                              {"pattern": "baz", "replacement": "qux"}], sample="foo baz")
    assert step["after"] == "bar qux"
    assert seen and not any(seen)


def _timed_screens(monkeypatch, wall, secs=0.0, refuse=()):
    """Every structural screen takes `secs` and refuses the patterns in
    `refuse`, against `wall` seconds of screen time for one dry run. Returns
    the patterns screened, one per call."""
    asked = []

    def _screen(pat):
        asked.append(pat)
        time.sleep(secs)
        return pat in refuse
    monkeypatch.setattr(admin_routes.regex_guard, "_nested_repetition", _screen)
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_SCREEN_WALL", wall)
    return asked


def test_each_pattern_is_screened_once(client, monkeypatch):
    # The probe walk, the list's run and a later rule all ask about the same
    # patterns; each screen is a compile plus pairwise overlap matching.
    asked = _timed_screens(monkeypatch, wall=15.0)
    steps = dry_run(client, [list_rule([{"pattern": "foo", "replacement": "bar"},
                                        {"pattern": "baz", "replacement": "qux"}]),
                             upper_rule("baz")], "foo baz")["steps"]
    assert steps[0]["after"] == "bar qux"
    assert sorted(asked) == ["baz", "foo"]


def test_screens_stop_at_their_time_budget(client, monkeypatch):
    # 41 distinct patterns at 0.03 s a screen against 0.05 s: the first two
    # are screened and run, the rest are reported, neither matched here nor
    # handed to the probe child.
    asked = _timed_screens(monkeypatch, wall=0.05, secs=0.03)
    walked = record_calls(monkeypatch, admin_routes, "_probe_chain",
                          lambda text, units, *a, **kw: units)
    rules = [list_rule([{"pattern": f"w{r}{i:02d}", "replacement": "x"} for i in range(20)],
                       name=f"r{r}") for r in range(2)]
    sample = "w019 w119"
    steps = dry_run(client, [*rules, upper_rule(r"w\d+")], sample)["steps"]
    assert asked == ["w000", "w001"]
    for step in steps[:3]:
        assert step["not_run"] is True and "screen time ran out" in step["error"]
        assert step["after"] == sample
    assert {u[2] for units in walked for u in units if u[0] in ("sub", "match")} == set(asked)


def test_an_unscreened_entry_does_not_hide_a_screen_verdict(client, monkeypatch):
    # The first rule's screen spends the time; the second rule's `a` is then
    # unscreened ("may be harmless") beside the verdict already known.
    _timed_screens(monkeypatch, wall=0.01, secs=0.02, refuse={"(a+)+$"})
    nested = {"pattern": "(a+)+$", "replacement": "x"}
    step = dry_run(client, [list_rule([nested]),
                            list_rule([{"pattern": "a", "replacement": "b"}, nested])],
                   "aaaa")["steps"][1]
    assert (step["not_run"], step["after"]) == (True, "aaaa")
    assert step["error"].startswith("nested repetition")
    assert "may be harmless" not in step["error"]


def test_a_rank_one_error_beside_an_unscreened_entry_says_why_it_did_not_run(client, monkeypatch):
    _timed_screens(monkeypatch, wall=0.0)
    step = list_step(client, [{"pattern": "a", "replacement": "b"},
                              {"pattern": ["("], "replacement": "x"}])
    assert (step["not_run"], step["after"]) == (True, "aaaa")
    assert step["error"].startswith("pattern must be a string")
    assert "another entry is not executed here" in step["error"]
    assert "screen time ran out" in step["error"]


def test_a_pattern_left_unscreened_stays_unrun_when_a_stranded_thread_clears_it(client, monkeypatch):
    # Rule 1's guard thread is left behind inside the screen of `q` (its
    # join gives up). `p` then spends the screen time, so rule 2's `q` is
    # left out of the probe walk. The stranded thread clearing `q` after
    # that walk must not let it run: nothing timed it on this text.
    release, inside = threading.Event(), threading.Event()
    stranded = []

    def _screen(pat):
        if pat == "q" and not inside.is_set():
            inside.set()
            release.wait(5)
        elif pat == "p":
            inside.wait(5)
            time.sleep(0.06)
        return False
    monkeypatch.setattr(admin_routes.regex_guard, "_nested_repetition", _screen)
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_SCREEN_WALL", 0.05)
    real_join = threading.Thread.join

    def _join(self, timeout=None):
        if self.name.endswith("(_do_list)") and not stranded:
            stranded.append(self)
            timeout = 0.05
        return real_join(self, timeout)
    monkeypatch.setattr(threading.Thread, "join", _join)

    def _walk(text, units, *a, **kw):
        release.set()
        real_join(stranded[0], 5)
        return {1: admin_routes._text_sha(text)}, set(), None
    monkeypatch.setattr(admin_routes, "_probe_chain", _walk)
    try:
        steps = dry_run(client, [list_rule([{"pattern": "q", "replacement": 5}]),
                                 list_rule([{"pattern": "p", "replacement": "P"},
                                            {"pattern": "q", "replacement": "Q"}])],
                        "p q")["steps"]
    finally:
        release.set()
    assert steps[0]["slow"] is True
    assert (steps[1]["after"], steps[1]["not_run"]) == ("P q", True)
    assert "screen time ran out" in steps[1]["error"]
