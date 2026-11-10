"""What the diarization stage on POST /v1/audio/transcriptions hands pyannote.

pyannote decodes its whole input as float32 at the file's native rate and
channel count before it downmixes and resamples to 16 kHz mono, so the route
gives it a private 16 kHz mono WAV instead: its input then costs what the
transcription's own decode costs, whatever the upload's shape. pyannote is
never imported — diarize() is stubbed at the boundary the handler uses."""

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


def test_a_failed_downmix_keeps_the_original_path(
        client, fake_model, diarize_seen, monkeypatch, tmp_path):
    """Nothing new is refused: when the 16 kHz copy cannot be written
    (here, a disk with no room for it), pyannote reads the upload as
    before."""
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
    assert "warnings" not in body
    [(path, shape)] = diarize_seen
    assert shape == (48000, 2, 48000)
    assert os.path.basename(path).startswith("whisperup-")
    assert not os.path.basename(path).startswith("whisperup-diar-")


def test_an_undecodable_upload_keeps_the_original_path(
        client, fake_model, diarize_seen, monkeypatch):
    def _boom(src, dst, *, rate, layout, **_kw):
        raise RuntimeError("no decoder")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _boom)
    _post_ok(client, diarize="true")
    [(path, _shape)] = diarize_seen
    assert not os.path.basename(path).startswith("whisperup-diar-")


def test_the_copy_stops_a_few_seconds_past_what_whisper_decoded(app_module, tmp_path):
    """faster-whisper's decode ends at an mp3's first invalid packet; the
    transcode skips it and reads on. pyannote gets what Whisper heard (plus
    a margin), not the hour of audio behind the damaged frame — speaker
    turns past the transcript have no words to label anyway."""
    src = mp3_whisper_stops_early(str(tmp_path / "d.mp3"))
    decoded_s = audio_transcode.decode_16k_bounded(src, max_seconds=None).size / 16000
    assert decoded_s < 20
    dst = app_module._diar_input_wav(src, decoded_s)
    try:
        assert dst is not None
        assert os.path.getsize(dst) <= (decoded_s + 5) * 32000 + 44
    finally:
        os.unlink(dst)


def test_the_copy_of_intact_audio_is_unchanged(app_module, tmp_path):
    src = _stereo_wav(tmp_path / "src.wav", rate=48000, seconds=3.0)
    plain = str(tmp_path / "plain.wav")
    audio_transcode.transcode_to_wav(src, plain, rate=16000, layout="mono")
    dst = app_module._diar_input_wav(src, 3.0)
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
