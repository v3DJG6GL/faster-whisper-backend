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

    def spy(kwargs=None, *, skip_residual=False):
        seen["skip"] = skip_residual
        return real(kwargs, skip_residual=skip_residual)
    monkeypatch.setattr(app_module._decode_trace, "capture", spy)
    assert _post(client).status_code == 200
    assert seen == {"skip": True}
    app_module.cfg.DECODE_SKIP_RESIDUAL_WINDOWS = False
    assert _post(client).status_code == 200
    assert seen == {"skip": False}
