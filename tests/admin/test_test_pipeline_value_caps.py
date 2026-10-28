"""POST /settings/test-pipeline refuses a regex-list replacement or a map value
longer than a saved rule may carry. Editor rules skip the schema adapter, and
one re.sub of the 8192-char sample with an oversized replacement allocates
sample x replacement before the output bound is checked."""

from typing import get_args

from faster_whisper_backend.admin import routes as admin_routes
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import schema as settings_schema


def test_caps_are_read_off_the_schema():
    # The fallbacks equal today's bounds, so pin the walks themselves.
    assert admin_routes._field_max_length(
        settings_schema.RegexListEntry.model_fields["replacement"]) == 512
    assert admin_routes._str_max_length(
        get_args(settings_schema.MapRule.model_fields["map"].annotation)[1]) == 64


def test_oversized_regex_list_replacement_is_rejected(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "a" * 8192,
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "a", "replacement": "x" * 20000},
        ]}],
    })
    assert r.status_code == 400, r.status_code
    assert "replacement too long" in r.json()["error"]


def test_oversized_map_value_is_rejected(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "komma " * 100,
        "rules": [{"name": "m", "type": "callback:map", "enabled": True,
                   "map": {"komma": "," * 5000}}],
    })
    assert r.status_code == 400, r.status_code
    assert "map value too long" in r.json()["error"]


def test_replacement_at_the_cap_still_runs(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "foo",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "foo", "replacement": "y" * 512},
        ]}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["steps"][0]["after"] == "y" * 512


def test_chained_entries_stop_before_passing_the_output_bound(client, monkeypatch):
    # Entry 1 stays under the bound; entry 2 would multiply that text 512x in
    # one sub. The dry run must stop BEFORE running it, not after.
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", 40_000)
    r = client.post("/settings/test-pipeline", json={
        "sample": "x" * 78,
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "x", "replacement": "x" * 512},
            {"pattern": "x", "replacement": "x" * 512},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["capped"] is True
    assert step["after"] == "x" * (78 * 512)
    assert len(r.json()["final"]) <= 40_000


def test_backref_template_counts_its_expansion_against_the_bound(client, monkeypatch):
    # A 510-char template of `\g<0>` copies the whole match 102 times.
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", 40_000)
    r = client.post("/settings/test-pipeline", json={
        "sample": "a" * 8000,
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "(?s).+", "replacement": "\\g<0>" * 4},
            {"pattern": "(?s).+", "replacement": "\\g<0>" * 102},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["capped"] is True
    assert step["after"] == "a" * 32000


def test_map_rule_stops_before_passing_the_output_bound(client, monkeypatch):
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", 40_000)
    r = client.post("/settings/test-pipeline", json={
        "sample": "a" * 100,
        "rules": [
            {"name": "rl", "type": "regex-list", "enabled": True, "entries": [
                {"pattern": "a", "replacement": "komma " * 60},
            ]},
            {"name": "m", "type": "callback:map", "enabled": True,
             "map": {"komma": "," * 64}},
            {"name": "up", "type": "callback:upper", "enabled": True,
             "pattern": "(a)"},
        ],
    })
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert len(steps[0]["after"]) == 36_000
    assert steps[1]["capped"] is True
    assert steps[1]["after"] == steps[0]["after"]
    # Engine parity: every later rule is skipped once the bound is hit.
    assert steps[2]["skipped"] is True and steps[2]["capped"] is True
    assert len(r.json()["final"]) <= 40_000


def test_pattern_cap_is_read_off_the_schema():
    assert admin_routes._field_max_length(
        settings_schema.RegexListEntry.model_fields["pattern"]) == 512


def test_oversized_callback_pattern_is_rejected_before_the_screen(client, monkeypatch):
    def _screen(pat):
        raise AssertionError("structural screen ran on an over-long pattern")
    monkeypatch.setattr(admin_routes.regex_guard, "_nested_repetition", _screen)
    alternation = "(?:" + "|".join(f"w{i:06d}" for i in range(700)) + ")+"
    r = client.post("/settings/test-pipeline", json={
        "sample": "w000001",
        "rules": [{"name": "d", "type": "callback:dedup", "enabled": True,
                   "pattern": alternation}],
    })
    assert r.status_code == 400, r.status_code
    assert r.json()["error"] == "rule 1: pattern too long (max 512 chars)"


def test_oversized_regex_list_pattern_is_rejected(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "a",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "a", "replacement": "b"},
            {"pattern": "a" * 513, "replacement": "b"},
        ]}],
    })
    assert r.status_code == 400, r.status_code
    assert "pattern too long" in r.json()["error"]


def test_pattern_at_the_cap_still_runs(client):
    r = client.post("/settings/test-pipeline", json={
        "sample": "a" * 512,
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": "a" * 512, "replacement": "b"},
        ]}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["steps"][0]["after"] == "b"


def _no_screen(monkeypatch):
    def _screen(*args):
        raise AssertionError("structural screen ran on a non-string pattern")
    monkeypatch.setattr(admin_routes.regex_guard, "_nested_repetition", _screen)
    monkeypatch.setattr(admin_routes.regex_guard, "check_fixed_counts", _screen)


# A list skips the str-only length cap, and its repr is a valid alternation.
_LIST_PATTERN = ["](?:" + "|".join(f"w{i:06d}" for i in range(800)) + ")+["]


def test_non_string_callback_pattern_is_refused_before_the_screens(client, monkeypatch):
    _no_screen(monkeypatch)
    r = client.post("/settings/test-pipeline", json={
        "sample": "w000001",
        "rules": [{"name": "d", "type": "callback:dedup", "enabled": True,
                   "pattern": _LIST_PATTERN}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["error"] == "pattern must be a string"
    assert step["after"] == "w000001"


def test_non_string_regex_list_pattern_is_skipped_before_the_screens(client, monkeypatch):
    _no_screen(monkeypatch)
    r = client.post("/settings/test-pipeline", json={
        "sample": "w000001",
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True, "entries": [
            {"pattern": _LIST_PATTERN, "replacement": "x"},
        ]}],
    })
    assert r.status_code == 200, r.text
    step = r.json()["steps"][0]
    assert step["error"] == "pattern must be a string"
    assert step["after"] == "w000001"
