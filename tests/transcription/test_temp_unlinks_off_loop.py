"""POST /v1/audio/transcriptions deletes its temp files — the upload's copy,
the pre-separation original and the 44.1 kHz separation WAV (each can be
gigabytes) — in a worker thread, never on the event loop, and on every path."""

import os
import tempfile

import pytest

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import transcode as audio_transcode
from faster_whisper_backend.core import store_common
from tests._transcription_helpers import post_verbose_ok, wav_bytes
from tests.conftest import on_event_loop

_M4A_UPLOAD = {"file": ("a.m4a", b"\x00\x00\x00 ftypM4A ", "audio/mp4")}


@pytest.fixture
def unlinks(monkeypatch):
    """{basename prefix: [ran on the event loop?, ...]} per unlink_quietly
    call, plus each path in "paths"."""
    seen: dict = {"paths": []}
    real = store_common.unlink_quietly

    def _spy(path):
        if path:
            name = os.path.basename(path)
            prefix = name.split("-", 1)[0] if "-" in name else name
            seen.setdefault(prefix, []).append(on_event_loop())
            seen["paths"].append(path)
        return real(path)
    monkeypatch.setattr(store_common, "unlink_quietly", _spy)
    return seen


@pytest.fixture
def separating(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)

    async def _fake(path, *, model_filename=None, progress_cb=None,
                    cancel_check=None):
        fd, out = tempfile.mkstemp(prefix="vocals-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        return out
    monkeypatch.setattr(bgm_separation, "separate", _fake)


def _gone(seen):
    return not any(os.path.exists(p) for p in seen["paths"])


def test_upload_copy_is_removed_off_the_loop(client, unlinks):
    post_verbose_ok(client)
    assert unlinks.get("whisperup") == [False]
    assert _gone(unlinks)


def test_separation_temp_files_are_removed_off_the_loop(
        client, monkeypatch, separating, unlinks):
    def _transcode(src, dst, *, rate, layout, max_bytes=None):
        with open(dst, "wb") as f:
            f.write(b"RIFF44kWAVE")
        return 11
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _transcode)
    post_verbose_ok(client, files=_M4A_UPLOAD, separate_bgm="true")
    # The separation WAV, the pre-separation original, then the vocals
    # the run transcribed in its place.
    assert unlinks.get("sepsrc") == [False]
    assert unlinks.get("whisperup") == [False]
    assert unlinks.get("vocals") == [False]
    assert _gone(unlinks)


def test_a_failed_separation_transcode_drops_its_wav_off_the_loop(
        client, monkeypatch, separating, unlinks):
    import av

    def _transcode(src, dst, *, rate, layout, max_bytes=None):
        with open(dst, "wb") as f:
            f.write(b"partial")
        raise av.error.InvalidDataError(1094995529, "Invalid data", "x")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _transcode)
    post_verbose_ok(client, separate_bgm="true",
                    files={"file": ("a.mp3", wav_bytes(), "audio/mpeg")})
    assert unlinks.get("sepsrc") == [False]
    assert _gone(unlinks)


def test_a_failed_separation_still_removes_every_temp_file(
        client, app_module, monkeypatch, unlinks):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)

    async def _fails(path, **kw):
        raise bgm_separation.BgmSeparationError("stub separator")
    monkeypatch.setattr(bgm_separation, "separate", _fails)

    def _transcode(src, dst, *, rate, layout, max_bytes=None):
        with open(dst, "wb") as f:
            f.write(b"RIFF44kWAVE")
        return 11
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _transcode)
    post_verbose_ok(client, files=_M4A_UPLOAD, separate_bgm="true")
    assert unlinks.get("sepsrc") == [False]
    assert unlinks.get("whisperup") == [False]
    assert _gone(unlinks)


def test_the_diarization_input_copy_is_removed_off_the_loop(
        client, app_module, fake_model, monkeypatch):
    from faster_whisper_backend.audio import diarization

    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True,
                        raising=False)
    handed = []

    async def _fake(path, **kw):
        handed.append(path)
        return [(0.0, 1.0, "SPEAKER_00")]
    monkeypatch.setattr(diarization, "diarize", _fake)

    def _transcode(src, dst, *, rate, layout, **_kw):
        with open(dst, "wb") as f:
            f.write(b"RIFF16kWAVE")
        return 11
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _transcode)
    on_loop = {}
    real = store_common.unlink_quietly

    def _spy(path):
        if path:
            on_loop.setdefault(path, []).append(on_event_loop())
        return real(path)
    monkeypatch.setattr(store_common, "unlink_quietly", _spy)
    post_verbose_ok(client, diarize="true")
    [diar_wav] = handed
    assert os.path.basename(diar_wav).startswith("whisperup-diar-")
    assert on_loop[diar_wav] == [False]
    assert not os.path.exists(diar_wav)
