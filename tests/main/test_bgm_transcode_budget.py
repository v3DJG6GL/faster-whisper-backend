"""The pre-separation transcode on POST /v1/audio/transcriptions is bounded
by the free space of its temp dir: a low-bitrate upload expands several
hundred times into 44.1 kHz stereo WAV, so the route hands transcode_to_wav
a byte budget (free space minus a margin) and an over-budget transcode takes
the existing "separator decodes the original" fallback.

The separator itself loads its whole input as float32 44.1 kHz stereo, so
the input is also bounded by available RAM: a transcode that trips the RAM
cap, or a wav/flac (passed to the separator as is) whose duration is past
it, skips separation instead of handing the separator what it cannot hold."""

import os
import shutil
import tempfile
from collections import namedtuple

import av
import numpy as np
import psutil
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


def _available_ram(monkeypatch, available=None, exc=None):
    """psutil.virtual_memory() reporting `available` bytes (or raising
    `exc`); every other field stays real for the stats sampler."""
    real = psutil.virtual_memory

    def _vm():
        if exc is not None:
            raise exc
        return real()._replace(available=available)
    monkeypatch.setattr(psutil, "virtual_memory", _vm)


def _record_budget(monkeypatch, *, over_budget=False, exc=RuntimeError):
    budgets = []

    def _fake_transcode(src, dst, *, rate, layout, max_bytes=None):
        budgets.append(max_bytes)
        if over_budget:
            raise exc("transcode output exceeds the free-space budget")
        with open(dst, "wb") as f:
            f.write(b"RIFF44kWAVE")
        return 11
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _fake_transcode)
    return budgets


def test_transcode_budget_is_free_space_minus_margin(
        client, monkeypatch, fake_model, separate_calls):
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    budgets = _record_budget(monkeypatch)
    assert "warnings" not in _post_ok(client, files=_M4A_UPLOAD,
                                      separate_bgm="true")
    assert budgets == [4_000_000_000]
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".wav")


def test_transcode_budget_floors_at_zero_on_a_nearly_full_disk(
        client, monkeypatch, fake_model, separate_calls):
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=10_000)
    budgets = _record_budget(monkeypatch, over_budget=True)
    assert "warnings" not in _post_ok(client, files=_M4A_UPLOAD,
                                      separate_bgm="true")
    assert budgets == [0]
    # Over budget is not a refusal: the separator decodes the original.
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".m4a")


def test_transcode_budget_fails_open_when_free_space_is_unknown(
        client, monkeypatch, fake_model, separate_calls):
    _available_ram(monkeypatch, exc=OSError("no /proc/meminfo"))
    _disk_usage_reporting(monkeypatch, exc=OSError("statvfs failed"))
    budgets = _record_budget(monkeypatch)
    _post_ok(client, files=_M4A_UPLOAD, separate_bgm="true")
    assert budgets == [None]
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".wav")


# 44.1 kHz stereo s16: what the RAM cap is counted in.
_S16_BYTES_PER_S = 44100 * 2 * 2


def _flac(path, seconds=5.0, rate=8000):
    """A real `seconds`-long mono FLAC of silence."""
    with av.open(str(path), "w", format="flac") as out:
        st = out.add_stream("flac", rate=rate)
        st.layout = "mono"
        n, step = 0, 4000
        while n < seconds * rate:
            fr = av.AudioFrame.from_ndarray(
                np.zeros((1, step), dtype=np.int16), format="s16",
                layout="mono")
            fr.sample_rate, fr.pts = rate, n
            for pkt in st.encode(fr):
                out.mux(pkt)
            n += step
        for pkt in st.encode(None):
            out.mux(pkt)
    return path.read_bytes()


def _without_total_samples(flac: bytes) -> bytes:
    """`flac` with STREAMINFO's total-samples field zeroed ("unknown"), so
    the container reports no duration."""
    b = bytearray(flac)
    v = int.from_bytes(b[18:26], "big") & ~((1 << 36) - 1)
    b[18:26] = v.to_bytes(8, "big")
    return bytes(b)


def _skipped(body):
    return any("music separation skipped" in w
               for w in body.get("warnings") or [])


def test_ram_cap_overrun_skips_separation(
        client, monkeypatch, fake_model, separate_calls):
    """The transcode gets the RAM cap (available // 4, in s16 bytes) when it
    is tighter than the disk budget; tripping it skips separation."""
    _available_ram(monkeypatch, available=4_000_000)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    budgets = _record_budget(
        monkeypatch, over_budget=True,
        exc=getattr(audio_transcode, "TranscodeBudgetExceeded", RuntimeError))
    body = _post_ok(client, files=_M4A_UPLOAD, separate_bgm="true")
    assert budgets == [1_000_000]
    assert separate_calls == []
    assert _skipped(body)


def test_disk_only_overrun_still_falls_back_when_ram_is_roomy(
        client, monkeypatch, fake_model, separate_calls):
    """A budget overrun where the disk budget, not the RAM cap, was the bound
    keeps the fallback even when the error is the budget subclass."""
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=2_000_000_000)
    budgets = _record_budget(
        monkeypatch, over_budget=True,
        exc=getattr(audio_transcode, "TranscodeBudgetExceeded", RuntimeError))
    assert "warnings" not in _post_ok(client, files=_M4A_UPLOAD,
                                      separate_bgm="true")
    assert budgets == [1_000_000_000]
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".m4a")


def test_disk_only_overrun_skips_a_file_whose_duration_is_past_the_ram_cap(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """The disk fallback hands the separator the original: a file the
    container says is longer than the RAM cap is skipped there too."""
    flac = _flac(tmp_path / "src.flac", seconds=5.0)
    _available_ram(monkeypatch, available=4 * 2 * _S16_BYTES_PER_S)  # 2 s
    _disk_usage_reporting(monkeypatch, free=10_000)
    _record_budget(monkeypatch, over_budget=True)
    body = _post_ok(client, separate_bgm="true",
                    files={"file": ("a.m4a", flac, "audio/mp4")})
    assert separate_calls == []
    assert _skipped(body)


def test_long_flac_skips_separation_by_duration(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """wav/flac skip the transcode (and its bound): their duration decides."""
    flac = _flac(tmp_path / "src.flac", seconds=5.0)
    _available_ram(monkeypatch, available=400_000)   # cap 100 kB < 882 kB
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    calls = _record_budget(monkeypatch)
    body = _post_ok(client, separate_bgm="true",
                    files={"file": ("a.flac", flac, "audio/flac")})
    assert separate_calls == [] and calls == []
    assert _skipped(body)


def test_short_flac_keeps_the_direct_separator_path(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    flac = _flac(tmp_path / "src.flac", seconds=5.0)
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    calls = _record_budget(monkeypatch)
    assert "warnings" not in _post_ok(
        client, separate_bgm="true",
        files={"file": ("a.flac", flac, "audio/flac")})
    assert calls == []
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".flac")


def test_flac_without_a_duration_takes_the_bounded_transcode(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """A wav/flac whose container reports no duration cannot be measured up
    front, so it goes through the bounded transcode instead of reaching the
    separator unmeasured."""
    flac = _without_total_samples(_flac(tmp_path / "src.flac", seconds=5.0))
    _available_ram(monkeypatch, available=400_000)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    budgets = _record_budget(
        monkeypatch, over_budget=True,
        exc=getattr(audio_transcode, "TranscodeBudgetExceeded", RuntimeError))
    body = _post_ok(client, separate_bgm="true",
                    files={"file": ("a.flac", flac, "audio/flac")})
    assert budgets == [100_000]
    assert separate_calls == []
    assert _skipped(body)


def test_ram_cap_unknown_keeps_the_direct_flac_path(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """No RAM reading fails open: a duration-less flac still goes straight
    to the separator, as before the cap existed."""
    flac = _without_total_samples(_flac(tmp_path / "src.flac", seconds=5.0))
    _available_ram(monkeypatch, exc=RuntimeError("psutil broke"))
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    calls = _record_budget(monkeypatch)
    _post_ok(client, separate_bgm="true",
             files={"file": ("a.flac", flac, "audio/flac")})
    assert calls == []
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".flac")
