"""LEADING_SILENCE_PAD_MS — batch-endpoint leading-silence pad.

The pad defuses a decoder failure mode: a recording that starts mid-speech at
t=0 combined with DEFAULT_HOTWORDS (injected as fake previous-transcript
context after <|startofprev|>) makes Whisper drop the opening clause as
"already transcribed". The endpoint pre-decodes the upload to 16 kHz mono,
prepends the pad, and shifts every reported time back so the response,
/captures rows, and word timestamps stay on the ORIGINAL audio's timeline.
"""

import io
import wave

import numpy as np
import pytest

from tests.conftest import FakeInfo, FakeModel, FakeSegment, FakeWord
from faster_whisper_backend.transcription import models as tx_models


def _wav_bytes(seconds=1.0, sr=16000):
    """A decodable 16 kHz mono PCM WAV with constant nonzero amplitude, so the
    injected zero-pad is distinguishable from the payload."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.full(int(seconds * sr), 8000, dtype="<i2").tobytes())
    return buf.getvalue()


def _post(client, wav, **data):
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions",
                       files={"file": ("a.wav", wav, "audio/wav")}, data=data)


def test_model_receives_padded_array(client, app_module, fake_model):
    app_module.cfg.LEADING_SILENCE_PAD_MS = 500
    r = _post(client, _wav_bytes(seconds=1.0))
    assert r.status_code == 200
    audio = fake_model.last_audio
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 24000                    # 8000 pad + 16000 payload
    assert np.all(audio[:8000] == 0.0)
    assert np.abs(audio[8000:]).min() > 0.0


def test_timestamps_shifted_back_to_original_timeline(client, app_module, monkeypatch):
    # Simulate the decoder's view of a 1.0 s upload behind a 0.5 s pad: it
    # reports duration 1.5 and places speech at 0.6-1.4 on the PADDED timeline.
    words = [FakeWord("hallo", 0.6, 0.9), FakeWord("welt", 0.9, 1.4)]
    seg = FakeSegment("hallo welt", 0.6, 1.4, words)
    info = FakeInfo(duration=1.5)
    info.duration_after_vad = 1.2
    model = FakeModel(segments=[seg], info=info)

    async def _loader(name, *, lease=False):
        return model
    monkeypatch.setattr(tx_models, "_get_or_load_model", _loader)

    app_module.cfg.LEADING_SILENCE_PAD_MS = 500
    r = _post(client, _wav_bytes(seconds=1.0), response_format="verbose_json")
    assert r.status_code == 200
    body = r.json()
    assert body["duration"] == pytest.approx(1.0)
    assert body["segments"][0]["start"] == pytest.approx(0.1)
    assert body["segments"][0]["end"] == pytest.approx(0.9)
    assert [(w["start"], w["end"]) for w in body["words"]] == [
        (pytest.approx(0.1), pytest.approx(0.4)),
        (pytest.approx(0.4), pytest.approx(0.9)),
    ]


def test_live_progress_and_plan_use_the_original_timeline(
        client, app_module, monkeypatch):
    # _collect runs BEFORE the shift: the decoder's padded duration must not
    # reach the run plan (the "decoder" rank wins, so the receipt, the ETA
    # and every learned rate would carry the pad) nor the progress entry.
    from faster_whisper_backend.transcription import progress as tx_progress
    from faster_whisper_backend.transcription import run_plan
    seg = FakeSegment("hallo welt", 0.6, 1.4, [])
    info = FakeInfo(duration=1.5)
    info.duration_after_vad = 1.2
    model = FakeModel(segments=[seg], info=info)

    async def _loader(name, *, lease=False):
        return model
    monkeypatch.setattr(tx_models, "_get_or_load_model", _loader)
    secs = []
    real_set = run_plan.RunPlan.set_audio_seconds

    def _spy_set(self, s, *, src):
        if src == "decoder":
            secs.append(s)
        return real_set(self, s, src=src)
    monkeypatch.setattr(run_plan.RunPlan, "set_audio_seconds", _spy_set)
    ticks = []
    real_ps = tx_progress._progress_set

    def _spy_ps(pid, **fields):
        ticks.append(fields)
        return real_ps(pid, **fields)
    monkeypatch.setattr(tx_progress, "_progress_set", _spy_ps)

    app_module.cfg.LEADING_SILENCE_PAD_MS = 500
    r = _post(client, _wav_bytes(seconds=1.0), progress_id="0adf1e0e0001")
    assert r.status_code == 200
    assert secs == [pytest.approx(1.0)]
    assert [t["duration"] for t in ticks if "duration" in t] == [
        pytest.approx(1.0)]
    assert [t["position"] for t in ticks
            if t.get("position") is not None] == [pytest.approx(0.9)]


def test_pad_zero_passes_path_through(client, app_module, fake_model):
    app_module.cfg.LEADING_SILENCE_PAD_MS = 0
    r = _post(client, _wav_bytes(seconds=1.0))
    assert r.status_code == 200
    assert isinstance(fake_model.last_audio, str)


def test_undecodable_upload_falls_back_to_path(client, app_module, fake_model):
    # Pre-decode of junk bytes fails -> the handler must fall back to the
    # tmp-file path (unpadded), not 500. Every pre-existing dummy-WAV test
    # rides this same fallback.
    app_module.cfg.LEADING_SILENCE_PAD_MS = 500
    r = _post(client, b"RIFFxxxxWAVE")
    assert r.status_code == 200
    assert isinstance(fake_model.last_audio, str)


def test_shift_clamps_into_pad_and_dav(app_module):
    # An onset the VAD speech-padding placed INSIDE the injected silence
    # clamps to 0; duration_after_vad never exceeds the original duration.
    words = [FakeWord("a", 0.2, 0.7)]
    seg = FakeSegment("a", 0.2, 0.7, words)
    info = FakeInfo(duration=1.5)
    info.duration_after_vad = 1.5
    segs, out = app_module._shift_to_original_timeline([seg], info, 0.5)
    assert segs[0].start == 0.0
    assert segs[0].end == pytest.approx(0.2)
    assert segs[0].words[0].start == 0.0
    assert out.duration == pytest.approx(1.0)
    assert out.duration_after_vad == pytest.approx(1.0)
