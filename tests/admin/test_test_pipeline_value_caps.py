"""POST /settings/test-pipeline refuses a regex-list replacement or a map value
longer than a saved rule may carry. Editor rules skip the schema adapter, and
one re.sub of the 8192-char sample with an oversized replacement allocates
sample x replacement before the output bound is checked."""

from typing import get_args

from faster_whisper_backend.admin import routes as admin_routes
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
