"""DECODE_SKIP_RESIDUAL_WINDOWS on the batch route.

The stop rule ends faster-whisper's segment generator with a
ResidualWindowSkipped raised from inside the window loop. The batch
collector (`_collect` in the transcriptions route) must treat that as the end
of the stream: every segment yielded before it is the response, the receipt
carries the guard row, and the knob resolves through cfg_for like its
siblings so a per-model / per-identity override reaches the decode.
"""

import json
import logging

from faster_whisper_backend.core import decode_trace as dt
from tests.conftest import FakeInfo, FakeSegment

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def _stopping_generator():
    yield FakeSegment("erster satz", 0.0, 3.9)
    raise dt.ResidualWindowSkipped("window 2 starts after the end")


def test_batch_keeps_segments_yielded_before_the_stop(client, app_module, fake_model, caplog):
    fake_model.transcribe = lambda path, **kw: (_stopping_generator(), FakeInfo(duration=4.8))
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        r = _post(client, response_format="verbose_json")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"].strip() == "erster satz"
    assert len(body["segments"]) == 1
    block = "\n".join(rec.getMessage() for rec in caplog.records
                      if "Post-decode guards" in rec.getMessage())
    guard = next(l for l in block.splitlines() if "skip_residual_windows" in l)
    assert guard.rstrip().endswith("true")


def test_batch_off_switch_reaches_the_receipt(client, app_module, fake_model, caplog):
    app_module.cfg.DECODE_SKIP_RESIDUAL_WINDOWS = False
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        r = _post(client, response_format="json")
    assert r.status_code == 200, r.text
    block = "\n".join(rec.getMessage() for rec in caplog.records
                      if "Post-decode guards" in rec.getMessage())
    guard = next(l for l in block.splitlines() if "skip_residual_windows" in l)
    assert guard.rstrip().endswith("false *")


def test_batch_capture_is_armed_from_config(client, app_module, fake_model, monkeypatch):
    """The executor thread's capture carries the resolved bool."""
    seen = {}
    real = dt.capture

    def spy(kwargs=None, *, skip_residual=False, **kw):
        seen["skip"] = skip_residual
        return real(kwargs, skip_residual=skip_residual, **kw)
    monkeypatch.setattr(app_module._decode_trace, "capture", spy)
    assert _post(client).status_code == 200
    assert seen == {"skip": True}
    app_module.cfg.DECODE_SKIP_RESIDUAL_WINDOWS = False
    assert _post(client).status_code == 200
    assert seen == {"skip": False}


def test_batch_gets_the_token_cap_and_keeps_best_of(client, app_module, fake_model,
                                                    monkeypatch, caplog):
    """DECODE_TOKEN_CAP_PER_SECOND reaches the batch capture and the receipt;
    STREAMING_FINAL_BEST_OF is a streaming knob and must not leak here."""
    seen = []
    real = dt.capture

    def spy(kwargs=None, **kw):
        seen.append(kw.get("token_cap_per_s"))
        return real(kwargs, **kw)
    monkeypatch.setattr(app_module._decode_trace, "capture", spy)
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        assert _post(client).status_code == 200
    assert seen == [10.0]
    assert fake_model.last_kwargs["best_of"] == app_module.cfg.BEST_OF == 5
    block = "\n".join(rec.getMessage() for rec in caplog.records
                      if "Post-decode guards" in rec.getMessage())
    guard = next(l for l in block.splitlines() if "token_cap_per_second" in l)
    assert guard.rstrip().endswith("10.0")

    app_module.cfg.DECODE_TOKEN_CAP_PER_SECOND = 0.0
    seen.clear(); caplog.clear()
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        assert _post(client).status_code == 200
    assert seen == [0.0]
    block = "\n".join(rec.getMessage() for rec in caplog.records
                      if "Post-decode guards" in rec.getMessage())
    guard = next(l for l in block.splitlines() if "token_cap_per_second" in l)
    assert guard.rstrip().endswith("*")
