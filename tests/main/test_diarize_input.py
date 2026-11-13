"""What the diarization stage on POST /v1/audio/transcriptions hands pyannote.

pyannote decodes its whole input as float32 at the file's native rate and
channel count before it downmixes and resamples to 16 kHz mono, so the route
gives it a private 16 kHz mono WAV instead: its input then costs what the
transcription's own decode costs, whatever the upload's shape. pyannote is
never imported — diarize() is stubbed at the boundary the handler uses."""

import asyncio
import os
import shutil
import tempfile
import wave
from collections import namedtuple

import numpy as np
import pytest

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import diarization
from faster_whisper_backend.audio import transcode as audio_transcode
from tests._transcription_helpers import post_verbose as _post
from tests._transcription_helpers import post_verbose_ok as _post_ok
from tests._transcription_helpers import wav_bytes
from tests.audio.test_transcode import mp3_whisper_stops_early
from tests.conftest import write_wav

_Usage = namedtuple("_Usage", "total used free")


def _stereo_wav(path, *, rate=48000, seconds=1.0) -> str:
    n = int(rate * seconds)
    tone = (np.sin(2 * np.pi * 440 * np.arange(n) / rate) * 8000).astype("<i2")
    return write_wav(str(path), np.repeat(tone, 2).tobytes(), nchannels=2,
                     rate=rate)


@pytest.fixture
def diarize_seen(app_module, monkeypatch):
    """diarize() stubbed: records the path it got and that file's WAV shape
    (rate, channels, frames), or None when it is not a WAV."""
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True,
                        raising=False)
    seen = []

    async def _fake(path, *, num_speakers=None, min_speakers=None,
                    max_speakers=None, model_id=None, progress_cb=None,
                    cancel_check=None):
        try:
            with wave.open(path, "rb") as w:
                shape = (w.getframerate(), w.getnchannels(), w.getnframes())
        except (wave.Error, EOFError):
            shape = None
        seen.append((path, shape))
        return [(0.0, 1.0, "SPEAKER_00")]
    monkeypatch.setattr(diarization, "diarize", _fake)
    return seen


def test_pyannote_gets_16k_mono_for_a_48k_stereo_upload(
        client, fake_model, diarize_seen, tmp_path):
    src = _stereo_wav(tmp_path / "src.wav", rate=48000, seconds=1.0)
    with open(src, "rb") as f:
        body = _post_ok(client, diarize="true",
                        files={"file": ("a.wav", f.read(), "audio/wav")})
    assert body["speakers"] == ["SPEAKER_00"]
    assert "warnings" not in body
    [(path, shape)] = diarize_seen
    assert shape == (16000, 1, 16000)
    # The 16 kHz copy is the request's own: gone once the response is out.
    assert not os.path.exists(path)


def test_the_separation_stem_is_downmixed_too(
        client, app_module, fake_model, diarize_seen, monkeypatch, tmp_path):
    """With separation on, the vocals stem (44.1 kHz stereo) is what
    diarization reads: it gets the same 16 kHz mono copy."""
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module, "_sep_ram_cap", lambda: None)

    async def _separate(path, **_kw):
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        os.close(fd)
        return _stereo_wav(out, rate=44100, seconds=1.0)
    monkeypatch.setattr(bgm_separation, "separate", _separate)
    _post_ok(client, diarize="true", separate_bgm="true")
    [(_path, shape)] = diarize_seen
    assert shape == (16000, 1, 16000)


_NOT_PREPARED = "diarization skipped: the file could not be prepared"


def test_a_copy_the_disk_has_no_room_for_skips_diarization(
        client, fake_model, diarize_seen, monkeypatch, tmp_path):
    """pyannote reads the private 16 kHz copy or nothing: when the copy
    cannot be written (here, a disk with no room for it) the upload itself
    is not handed to it at its native rate and channel count. The
    transcript is answered without speaker labels, with the reason."""
    real = shutil.disk_usage

    def _usage(path):
        if os.path.realpath(path) != os.path.realpath(tempfile.gettempdir()):
            return real(path)
        return _Usage(10_000, 0, 10_000)
    monkeypatch.setattr(shutil, "disk_usage", _usage)
    src = _stereo_wav(tmp_path / "src.wav", rate=48000, seconds=1.0)
    with open(src, "rb") as f:
        body = _post_ok(client, diarize="true",
                        files={"file": ("a.wav", f.read(), "audio/wav")})
    assert diarize_seen == []
    assert _NOT_PREPARED in body["warnings"]
    assert body["text"] and "speakers" not in body


def test_a_copy_that_fails_for_any_reason_skips_diarization(
        client, fake_model, diarize_seen, monkeypatch):
    made = []

    def _boom(src, dst, *, rate, layout, **_kw):
        made.append(dst)
        raise RuntimeError("no decoder")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _boom)
    body = _post_ok(client, diarize="true")
    assert diarize_seen == []
    assert _NOT_PREPARED in body["warnings"]
    assert body["text"]
    assert made and not os.path.exists(made[0])


def test_a_refused_input_is_not_handed_to_pyannote(
        client, fake_model, diarize_seen, monkeypatch):
    """A refusal by the transcode itself (a plain ValueError, not an av
    error: here the 60 s frame cap) skips diarization like the separation
    path does: pyannote's own decoder has no such cap, so the upload is not
    handed to it as the fallback."""
    made = []

    def _refuse(src, dst, *, rate, layout, **_kw):
        made.append(dst)
        raise ValueError("unsupported audio: a decoded frame runs past 60 s")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _refuse)
    body = _post_ok(client, diarize="true")
    assert diarize_seen == []
    assert _NOT_PREPARED in body["warnings"]
    assert made and not os.path.exists(made[0])


@pytest.mark.parametrize("error", ["InvalidDataError", "PermissionError"])
def test_an_av_error_in_the_copy_skips_diarization_too(
        client, fake_model, diarize_seen, monkeypatch, error):
    """The copy reads on past the packet Whisper's decode stopped at, so a
    damaged tail the transcription never met can fail it with a decoder
    error (a truncated AAC element answers "operation not permitted"). The
    client chooses those bytes: they do not choose the raw upload as
    pyannote's input."""
    import av

    def _bad(src, dst, *, rate, layout, **_kw):
        raise getattr(av.error, error)(1, "avcodec_send_packet()")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _bad)
    body = _post_ok(client, diarize="true")
    assert diarize_seen == []
    assert _NOT_PREPARED in body["warnings"]
    assert body["text"]


def test_a_cancel_during_the_copy_leaves_no_copy(
        client, app_module, fake_model, diarize_seen, monkeypatch):
    """A dropped connection cancels the handler at the copy's await while
    the worker thread writes on; the copy's path exists before that await,
    so the stage's finally removes what the thread wrote. The stub writes
    and then raises the CancelledError the await would."""
    made = []

    def _write_then_cancel(src, dst, *, rate, layout, **_kw):
        with open(dst, "wb") as f:
            f.write(b"\0" * 4096)
        made.append(dst)
        raise asyncio.CancelledError()
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _write_then_cancel)
    with pytest.raises(BaseException):
        _post(client, diarize="true")
    assert diarize_seen == []
    assert made and os.path.basename(made[0]).startswith("whisperup-diar-")
    assert not os.path.exists(made[0])


def test_the_copy_stops_a_few_seconds_past_what_whisper_decoded(app_module, tmp_path):
    """faster-whisper's decode ends at an mp3's first invalid packet; the
    transcode skips it and reads on. pyannote gets what Whisper heard (plus
    a margin), not the hour of audio behind the damaged frame — speaker
    turns past the transcript have no words to label anyway."""
    src = mp3_whisper_stops_early(str(tmp_path / "d.mp3"))
    decoded_s = audio_transcode.decode_16k_bounded(src, max_seconds=None).size / 16000
    assert decoded_s < 20
    dst = app_module._diar_input_wav(src, decoded_s, app_module._new_diar_dst())
    try:
        assert dst is not None
        assert os.path.getsize(dst) <= (decoded_s + 5) * 32000 + 44
    finally:
        os.unlink(dst)


def test_the_copy_of_intact_audio_is_unchanged(app_module, tmp_path):
    src = _stereo_wav(tmp_path / "src.wav", rate=48000, seconds=3.0)
    plain = str(tmp_path / "plain.wav")
    audio_transcode.transcode_to_wav(src, plain, rate=16000, layout="mono")
    dst = app_module._diar_input_wav(src, 3.0, app_module._new_diar_dst())
    try:
        with open(dst, "rb") as a, open(plain, "rb") as b:
            assert a.read() == b.read()
    finally:
        os.unlink(dst)


def test_the_route_bounds_the_copy_by_the_transcribed_duration(
        client, app_module, fake_model, diarize_seen):
    """The handler passes the duration Whisper reported: a model that heard
    1 s of a 10 s upload hands pyannote at most 6 s."""
    assert fake_model._info.duration == 1.0
    app_module.cfg.LEADING_SILENCE_PAD_MS = 0
    _post_ok(client, diarize="true",
             files={"file": ("a.wav", wav_bytes(seconds=10.0), "audio/wav")})
    [(_path, shape)] = diarize_seen
    assert shape[:2] == (16000, 1)
    assert shape[2] <= 6 * 16000


def test_the_diarization_device_is_resolved_off_the_event_loop(
        client, app_module, fake_model, monkeypatch):
    """_resolve_device can import torch and probe CUDA (seconds, once per
    process): the plan's device lookup runs in a worker thread, even when
    the stage then soft-skips because diarization is off."""
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", False,
                        raising=False)
    on_loop = []

    def _resolve():
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return "cpu"
    monkeypatch.setattr(diarization, "_resolve_device", _resolve)
    _post_ok(client, diarize="true")
    assert on_loop and not any(on_loop)
