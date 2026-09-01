"""Per-stage rows and receipt blocks of POST /v1/audio/transcriptions: what a
stage row bills and claims, and which receipt sections appear, must follow
what the stage actually did — not the shape of its result."""

import io
import os
import tempfile
import time
import wave

import numpy as np

import bgm_separation
import diarization
import translation
from conftest import FakeModel

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, files=_FILE, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=files, data=data)


def _wav_bytes(seconds=1.0, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.full(int(seconds * sr), 8000, dtype="<i2").tobytes())
    return buf.getvalue()


def _capture_receipt(app_module, monkeypatch, seen):
    real_block = app_module._format_request_block

    def _capture(**kw):
        seen.update(kw)
        seen["rendered"] = real_block(**kw)
        return seen["rendered"]
    monkeypatch.setattr(app_module, "_format_request_block", _capture)


def _stub_separate(monkeypatch):
    async def _fake(path, *, model_filename=None, progress_cb=None,
                    cancel_check=None):
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        return out
    monkeypatch.setattr(bgm_separation, "separate", _fake)


# --- "vad" row bills the lead-pad pre-decode ----------------------------------

def test_vad_row_bills_the_lead_pad_pre_decode(client, app_module,
                                               monkeypatch):
    """With LEADING_SILENCE_PAD_MS the handler decodes the upload itself and
    hands transcribe() an ndarray, so the decode cost used to slide into the
    "transcribing" row while the vad row's detail still claimed
    "audio decode + Silero"."""
    import faster_whisper.audio as _fwa
    real_decode = _fwa.decode_audio

    def _slow_decode(*a, **kw):
        time.sleep(0.05)
        return real_decode(*a, **kw)
    monkeypatch.setattr(_fwa, "decode_audio", _slow_decode)
    app_module.cfg.LEADING_SILENCE_PAD_MS = 500
    seen = {}
    _capture_receipt(app_module, monkeypatch, seen)
    r = _post(client, files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
              vad_filter="true")
    assert r.status_code == 200, r.text
    vad = next(s for s in seen["stages"] if s["name"] == "vad")
    assert vad["secs"] >= 0.05
    assert "decode" in (vad.get("detail") or "")


# --- Diarization receipt block follows the stage, not the result -------------

def test_diarization_block_present_when_stage_labels_nobody(
        client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True

    async def _no_turns(path, **kw):
        return []
    monkeypatch.setattr(diarization, "diarize", _no_turns)
    seen = {}
    _capture_receipt(app_module, monkeypatch, seen)
    try:
        r = _post(client, diarize="true")
        assert r.status_code == 200, r.text
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False
    assert any(s["name"] == "diarizing" for s in seen["stages"])
    assert seen["diarization"] is not None
    assert seen["diarization"]["result"].startswith("0 speakers")
    assert "─── Diarization" in seen["rendered"]


# --- Separation row's "incl. transcode" detail --------------------------------

def test_separation_row_has_no_transcode_detail_for_wav_input(
        client, app_module, monkeypatch):
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    _stub_separate(monkeypatch)
    seen = {}
    _capture_receipt(app_module, monkeypatch, seen)
    try:
        r = _post(client, separate_bgm="true")
        assert r.status_code == 200, r.text
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False
    row = next(s for s in seen["stages"] if s["name"] == "separating")
    assert not row.get("detail")


def test_separation_row_keeps_transcode_detail_when_transcode_ran(
        client, app_module, monkeypatch):
    import audio_transcode
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    _stub_separate(monkeypatch)

    def _fake_transcode(src, dst, *, rate=44100, layout="stereo"):
        with open(dst, "wb") as f:
            f.write(b"RIFFtcWAVE")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _fake_transcode)
    seen = {}
    _capture_receipt(app_module, monkeypatch, seen)
    try:
        r = _post(client, files={"file": ("a.mp3", b"ID3xxxx", "audio/mpeg")},
                  separate_bgm="true")
        assert r.status_code == 200, r.text
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False
    row = next(s for s in seen["stages"] if s["name"] == "separating")
    assert row.get("detail") == "incl. transcode"


def test_separation_row_drops_transcode_detail_when_transcode_failed(
        client, app_module, monkeypatch):
    import audio_transcode
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    _stub_separate(monkeypatch)

    def _boom(src, dst, *, rate=44100, layout="stereo"):
        raise RuntimeError("no decoder")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _boom)
    seen = {}
    _capture_receipt(app_module, monkeypatch, seen)
    try:
        r = _post(client, files={"file": ("a.mp3", b"ID3xxxx", "audio/mpeg")},
                  separate_bgm="true")
        assert r.status_code == 200, r.text
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False
    row = next(s for s in seen["stages"] if s["name"] == "separating")
    assert not row.get("detail")


# --- inverted speaker range is reconciled, not forwarded ----------------------

def test_min_speakers_above_max_is_clamped_down(client, app_module,
                                                monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    calls = []

    async def _fake(path, *, num_speakers=None, min_speakers=None,
                    max_speakers=None, model_id=None, progress_cb=None,
                    cancel_check=None):
        calls.append({"num": num_speakers, "min": min_speakers,
                      "max": max_speakers})
        return [(0.0, 1.0, "SPEAKER_00")]
    monkeypatch.setattr(diarization, "diarize", _fake)
    try:
        r = _post(client, diarize="true", min_speakers="10", max_speakers="2")
        assert r.status_code == 200, r.text
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False
    assert calls == [{"num": None, "min": 2, "max": 2}]
    assert "diarization failed" not in r.text


# --- translation reports why it did not run -----------------------------------

def test_translation_skipped_warning_without_speech_segments(
        client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)

    async def _never(*args, **kw):
        raise AssertionError("translate_segments() must not run")
    monkeypatch.setattr(translation, "translate_segments", _never)

    async def _loader(name, *, lease=False):
        return FakeModel(segments=[])
    monkeypatch.setattr(app_module, "_get_or_load_model", _loader)
    r = _post(client, translate_to="en")
    assert r.status_code == 200, r.text
    assert "translation skipped: no speech segments" in r.json()["warnings"]
