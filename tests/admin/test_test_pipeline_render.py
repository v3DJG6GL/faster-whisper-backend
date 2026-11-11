"""POST /settings/test-pipeline response rendering.

Up to 200 steps each carry the whole before/after text, so the body can run
to hundreds of MB. It is serialised step by step off the event loop (json's C
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
