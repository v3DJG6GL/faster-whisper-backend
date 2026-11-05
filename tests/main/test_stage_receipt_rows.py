"""Per-stage rows and receipt blocks of POST /v1/audio/transcriptions: what a
stage row bills and claims, and which receipt sections appear, must follow
what the stage actually did — not the shape of its result."""

import os
import tempfile
import time

import pytest

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import diarization
from faster_whisper_backend.audio import transcode as audio_transcode
from faster_whisper_backend.translation import engine as translation
from tests.conftest import FakeModel
from tests._translation_helpers import enable_translation
from tests._transcription_helpers import (
    post_verbose as _post, post_verbose_ok as _post_ok, serve_model, wav_bytes as _wav_bytes)
from tests.main._stage_helpers import capture_receipt, must_not_run, transcode_raises
from tests.main.test_transcription_overrides import _record_diarize


def _separating_row(client, monkeypatch, upload):
    seen = capture_receipt(monkeypatch)
    _post_ok(client, files={"file": upload}, separate_bgm="true")
    return next(s for s in seen["stages"] if s["name"] == "separating")


_STUB_CREATED: list[str] = []


def _stub_separate(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True, raising=False)

    async def _fake(path, *, model_filename=None, progress_cb=None,
                    cancel_check=None):
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        _STUB_CREATED.append(out)
        return out
    monkeypatch.setattr(bgm_separation, "separate", _fake)


@pytest.fixture(autouse=True)
def _cleanup_stub_files():
    yield
    while _STUB_CREATED:
        p = _STUB_CREATED.pop()
        try:
            os.unlink(p)
        except OSError:
            pass


def test_vad_row_bills_the_lead_pad_pre_decode(client, app_module,
                                               monkeypatch):
    """With LEADING_SILENCE_PAD_MS the handler decodes the upload itself and
    hands transcribe() an ndarray; that decode is billed to the vad row
    ("audio decode + Silero"), not to "transcribing"."""
    import faster_whisper.audio as _fwa
    real_decode = _fwa.decode_audio

    def _slow_decode(*a, **kw):
        time.sleep(0.05)
        return real_decode(*a, **kw)
    monkeypatch.setattr(_fwa, "decode_audio", _slow_decode)
    app_module.cfg.LEADING_SILENCE_PAD_MS = 500
    seen = capture_receipt(monkeypatch)
    _post_ok(client, files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
             vad_filter="true")
    vad = next(s for s in seen["stages"] if s["name"] == "vad")
    assert vad["secs"] >= 0.05
    assert "decode" in (vad.get("detail") or "")


def test_diarization_block_present_when_stage_labels_nobody(
        client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True, raising=False)

    async def _no_turns(path, **kw):
        return []
    monkeypatch.setattr(diarization, "diarize", _no_turns)
    seen = capture_receipt(monkeypatch)
    _post_ok(client, diarize="true")
    assert any(s["name"] == "diarizing" for s in seen["stages"])
    assert seen["diarization"] is not None
    assert seen["diarization"]["result"].startswith("0 speakers")
    assert "─── Diarization" in seen["rendered"]


def test_separation_row_has_no_transcode_detail_for_wav_input(
        client, app_module, monkeypatch):
    _stub_separate(app_module, monkeypatch)
    # The detail alone cannot tell a skipped transcode from a failed one
    # (the real transcode raises on these fake bytes and the fallback drops
    # the detail too): the transcode must not even be attempted. Recorded,
    # not raised — the fallback arm would swallow an AssertionError.
    calls = []
    monkeypatch.setattr(audio_transcode, "transcode_to_wav",
                        lambda *a, **k: calls.append(a))
    for name in ("a.wav", "a.flac"):
        row = _separating_row(client, monkeypatch, (name, b"RIFFxxxxWAVE", "audio/wav"))
        assert not row.get("detail")
    assert calls == []


def test_separation_row_keeps_transcode_detail_when_transcode_ran(
        client, app_module, monkeypatch):
    _stub_separate(app_module, monkeypatch)

    def _fake_transcode(src, dst, *, rate=44100, layout="stereo"):
        with open(dst, "wb") as f:
            f.write(b"RIFFtcWAVE")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _fake_transcode)
    row = _separating_row(client, monkeypatch, ("a.mp3", b"ID3xxxx", "audio/mpeg"))
    assert row.get("detail") == "incl. transcode"


def test_separation_row_drops_transcode_detail_when_transcode_failed(
        client, app_module, monkeypatch):
    _stub_separate(app_module, monkeypatch)
    transcode_raises(monkeypatch, RuntimeError("no decoder"))
    row = _separating_row(client, monkeypatch, ("a.mp3", b"ID3xxxx", "audio/mpeg"))
    assert not row.get("detail")


def test_min_speakers_above_max_is_clamped_down(client, app_module,
                                                monkeypatch):
    calls = _record_diarize(monkeypatch, app_module)
    r = _post(client, diarize="true", min_speakers="10", max_speakers="2")
    assert r.status_code == 200, r.text
    assert calls == [{"num": None, "min": 2, "max": 2}]
    assert "diarization failed" not in r.text


def test_zero_speaker_hint_means_auto_not_one_speaker(client, app_module,
                                                      monkeypatch):
    calls = _record_diarize(monkeypatch, app_module)
    _post_ok(client, diarize="true", num_speakers="0",
             min_speakers="0", max_speakers="0")
    assert calls == [{"num": None, "min": None, "max": None}]


def test_translation_skipped_warning_without_speech_segments(
        client, app_module, monkeypatch):
    enable_translation(app_module, monkeypatch)
    must_not_run(monkeypatch, translation, "translate_segments",
                 "translate_segments() must not run")
    serve_model(monkeypatch, FakeModel(segments=[]))
    body = _post_ok(client, translate_to="en")
    assert "translation skipped: no speech segments" in body["warnings"]
