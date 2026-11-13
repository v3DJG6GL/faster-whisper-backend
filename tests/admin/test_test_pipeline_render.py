"""POST /settings/test-pipeline response rendering.

Up to 200 steps each carry before/after text, several MB in all (the steps
share one size budget). It is serialised step by step off the event loop (json's C
encoder holds the GIL for one whole dump, so dumping it all in one call even
on a worker thread still stalls the loop), and the bytes stay exactly what
JSONResponse would send."""

import asyncio
import json

import pytest
from fastapi.responses import JSONResponse

from faster_whisper_backend.admin import routes as admin_routes
from tests.admin._pipeline_helpers import list_rule, upper_rule


def _post(client, rules, sample):
    return client.post("/settings/test-pipeline", json={"sample": sample, "rules": rules})


def test_response_body_matches_jsonresponse_bytes(client):
    rules = [
        list_rule([{"pattern": "a", "replacement": "ä"}]),
        {**upper_rule("x"), "enabled": False},
        upper_rule("b", name="up2"),
    ]
    r = _post(client, rules, "  abc ✓ 𝄞 ab  ")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/json"
    assert r.content == JSONResponse(r.json()).body


@pytest.mark.parametrize("ch", ["a", "ä", "✓", "𝄞"])
def test_render_matches_jsonresponse_for_every_utf8_width(ch):
    text = ch * 300 + '"\\\n'
    steps = [{"ordinal": i + 1, "label": f"r{i}", "before": text, "after": text + ch,
              "matches": i, "skipped": False, "error": None, "slow": False}
             for i in range(5)]
    for st, fin in ((steps, text), ([], "")):
        assert admin_routes._render_dry_run(st, fin) == \
            JSONResponse({"steps": st, "final": fin}).body


@pytest.mark.parametrize("body", [
    '{"sample":"abc","rules":[{"label":NaN,"type":"regex-list","enabled":true,"entries":[]}]}',
    '{"sample":"abc","rules":[{"label":Infinity,"type":"regex-list","enabled":true,"entries":[]}]}',
    '{"sample":"a\\ud800","rules":[]}',
    '{"sample":"abc","rules":[{"label":"x\\ud800","type":"regex-list","enabled":true,"entries":[]}]}',
], ids=["nan-label", "infinity-label", "surrogate-sample", "surrogate-label"])
def test_a_value_the_body_cannot_carry_is_a_400(client, body):
    """The render refuses NaN / Infinity (allow_nan=False) and cannot encode
    an unpaired surrogate; both arrive in a body json.loads accepts and were
    an unhandled error past the dry run."""
    r = client.post("/settings/test-pipeline", content=body,
                    headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    assert "JSON cannot carry" in r.json()["error"]



def test_dry_run_body_is_serialised_off_the_event_loop(client, monkeypatch):
    real = json.dumps
    on_loop: list[bool] = []

    def spy(obj, *a, **kw):
        if isinstance(obj, dict) and ("steps" in obj or "before" in obj):
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
        return real(obj, *a, **kw)

    monkeypatch.setattr(json, "dumps", spy)
    r = _post(client, [upper_rule("b")], "abc")
    assert r.status_code == 200, r.text
    assert on_loop and not any(on_loop), on_loop


def _disabled(n):
    return [{**upper_rule("x", name=f"off{i}"), "enabled": False} for i in range(n)]


def test_echoed_text_is_bounded_across_steps(client, monkeypatch):
    """A disabled step echoes the text it is handed twice, so one rule that
    grows the sample to just under the output bound was repeated by every
    step after it. Past the budget a step carries no text; `final` does."""
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_MAX_ECHO", 2000)
    rules = [list_rule([{"pattern": "a", "replacement": "b" * 10}]), *_disabled(20)]
    r = _post(client, rules, "a" * 50)
    assert r.status_code == 200, r.text
    j = r.json()
    grown = "b" * 500
    assert j["final"] == grown
    steps = j["steps"]
    assert len(steps) == 21
    # 550 chars for step 1, 1000 for step 2: step 3 would pass 2000.
    assert [s["after"] for s in steps[:2]] == [grown, grown]
    assert not any("text_omitted" in s for s in steps[:2])
    for s in steps[2:]:
        assert s["text_omitted"] is True and s["before"] == s["after"] == ""
        assert s["skipped"] is True     # the rest of the step is unchanged
    assert sum(len(s["before"]) + len(s["after"]) for s in steps) <= 2000


def test_echo_budget_covers_steps_that_change_the_text_and_the_final_trim(
        client, monkeypatch):
    monkeypatch.setattr(admin_routes, "_TEST_PIPELINE_MAX_ECHO", 100)
    rules = [list_rule([{"pattern": "a", "replacement": "bb"}], name="grow"),
             list_rule([{"pattern": "b", "replacement": "c"}], name="swap")]
    j = _post(client, rules, " " + "a" * 30).json()
    assert [s.get("text_omitted", False) for s in j["steps"]] == [False, True, True]
    assert j["steps"][1]["matches"] == 60 and j["steps"][1]["after"] == ""
    assert j["steps"][2]["type"] == "terminal"      # the implicit final trim
    assert j["final"] == "c" * 60


def test_ordinary_request_omits_no_text(client):
    # The most a real panel request echoes: a full sample through a full list.
    sample = "a" * admin_routes._TEST_PIPELINE_MAX_SAMPLE
    rules = [upper_rule("x"), *_disabled(admin_routes._TEST_PIPELINE_MAX_RULES - 1)]
    j = _post(client, rules, sample).json()
    assert len(j["steps"]) == admin_routes._TEST_PIPELINE_MAX_RULES
    assert not any("text_omitted" in s for s in j["steps"])
    assert all(s["before"] == s["after"] == sample for s in j["steps"])
    assert 2 * len(sample) * len(rules) < admin_routes._TEST_PIPELINE_MAX_ECHO
