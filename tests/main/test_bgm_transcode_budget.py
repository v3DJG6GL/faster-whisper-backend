"""The pre-separation transcode on POST /v1/audio/transcriptions is bounded
by the free space of its temp dir: a low-bitrate upload expands several
hundred times into 44.1 kHz stereo WAV, so the route hands transcode_to_wav
a byte budget (free space minus a margin) and an over-budget transcode takes
the existing "separator decodes the original" fallback."""

import os
import shutil
import tempfile
from collections import namedtuple

import pytest

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import transcode as audio_transcode
from tests._transcription_helpers import post_verbose_ok as _post_ok

_M4A_UPLOAD = {"file": ("a.m4a", b"\x00\x00\x00 ftypM4A ", "audio/mp4")}
_Usage = namedtuple("_Usage", "total used free")


@pytest.fixture
def separate_calls(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)
    calls = []

    async def _fake(path, *, model_filename=None, progress_cb=None,
                    cancel_check=None):
        calls.append(path)
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        return out

    monkeypatch.setattr(bgm_separation, "separate", _fake)
    return calls


def _disk_usage_reporting(monkeypatch, free=None, exc=None):
    real = shutil.disk_usage

    def _usage(path):
        if os.path.realpath(path) != os.path.realpath(tempfile.gettempdir()):
            return real(path)
        if exc is not None:
            raise exc
        return _Usage(free * 2, free, free)
    monkeypatch.setattr(shutil, "disk_usage", _usage)


def _record_budget(monkeypatch, *, over_budget=False):
    budgets = []

    def _fake_transcode(src, dst, *, rate, layout, max_bytes=None):
        budgets.append(max_bytes)
        if over_budget:
            raise RuntimeError("transcode output exceeds the free-space budget")
        with open(dst, "wb") as f:
            f.write(b"RIFF44kWAVE")
        return 11
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _fake_transcode)
    return budgets


def test_transcode_budget_is_free_space_minus_margin(
        client, monkeypatch, fake_model, separate_calls):
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    budgets = _record_budget(monkeypatch)
    assert "warnings" not in _post_ok(client, files=_M4A_UPLOAD,
                                      separate_bgm="true")
    assert budgets == [4_000_000_000]
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".wav")


def test_transcode_budget_floors_at_zero_on_a_nearly_full_disk(
        client, monkeypatch, fake_model, separate_calls):
    _disk_usage_reporting(monkeypatch, free=10_000)
    budgets = _record_budget(monkeypatch, over_budget=True)
    assert "warnings" not in _post_ok(client, files=_M4A_UPLOAD,
                                      separate_bgm="true")
    assert budgets == [0]
    # Over budget is not a refusal: the separator decodes the original.
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".m4a")


def test_transcode_budget_fails_open_when_free_space_is_unknown(
        client, monkeypatch, fake_model, separate_calls):
    _disk_usage_reporting(monkeypatch, exc=OSError("statvfs failed"))
    budgets = _record_budget(monkeypatch)
    _post_ok(client, files=_M4A_UPLOAD, separate_bgm="true")
    assert budgets == [None]
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".wav")
