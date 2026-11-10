"""The pre-separation transcode on POST /v1/audio/transcriptions is bounded
by the free space of its temp dir: a low-bitrate upload expands several
hundred times into 44.1 kHz stereo WAV, so the route hands transcode_to_wav
a byte budget (free space minus a margin) and an over-budget transcode takes
the existing "separator decodes the original" fallback.

The separator itself loads its whole input as float32 44.1 kHz stereo, so
the input is also bounded by available RAM: a transcode that trips the RAM
cap, or a wav/flac (passed to the separator as is) whose duration is past
it, skips separation instead of handing the separator what it cannot hold."""

import io
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
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=10_000)
    budgets = _record_budget(monkeypatch, over_budget=True)
    # A real body: the fallback measures the original, and one PyAV cannot
    # open is skipped instead.
    m4a = _flac(tmp_path / "s.mkv", seconds=1.0, fmt="matroska")
    assert "warnings" not in _post_ok(
        client, separate_bgm="true",
        files={"file": ("a.m4a", m4a, "audio/mp4")})
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


def _flac(path, seconds=5.0, rate=8000, layout="mono", fmt="flac"):
    """A real `seconds`-long FLAC of silence (in a `fmt` container)."""
    channels = av.AudioLayout(layout).nb_channels
    with av.open(str(path), "w", format=fmt) as out:
        st = out.add_stream("flac", rate=rate)
        st.layout = layout
        n, step = 0, 4000
        while n < seconds * rate:
            fr = av.AudioFrame.from_ndarray(
                np.zeros((1, step * channels), dtype=np.int16), format="s16",
                layout=layout)
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
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """A budget overrun where the disk budget, not the RAM cap, was the bound
    keeps the fallback even when the error is the budget subclass."""
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=2_000_000_000)
    budgets = _record_budget(
        monkeypatch, over_budget=True,
        exc=getattr(audio_transcode, "TranscodeBudgetExceeded", RuntimeError))
    m4a = _flac(tmp_path / "s.mkv", seconds=1.0, fmt="matroska")
    assert "warnings" not in _post_ok(
        client, separate_bgm="true",
        files={"file": ("a.m4a", m4a, "audio/mp4")})
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


def test_multichannel_hirate_flac_is_charged_by_its_real_shape(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """The separator loads a wav/flac at its native rate and channel count
    before resampling it: 2 s of 7.1 at 384 kHz costs it what 64 s of
    stereo at 44.1 kHz would, past a cap of 8 s."""
    flac = _flac(tmp_path / "src.flac", seconds=2.0, rate=384000,
                 layout="7.1")
    _available_ram(monkeypatch, available=4 * 8 * _S16_BYTES_PER_S)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    calls = _record_budget(monkeypatch)
    body = _post_ok(client, separate_bgm="true",
                    files={"file": ("a.flac", flac, "audio/flac")})
    assert separate_calls == [] and calls == []
    assert _skipped(body)


def test_a_header_without_a_channel_count_skips_separation(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    flac = _flac(tmp_path / "src.flac", seconds=2.0)
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    monkeypatch.setattr(audio_transcode, "declared_shape",
                        lambda p: (2.0, 8000, None))
    calls = _record_budget(monkeypatch)
    body = _post_ok(client, separate_bgm="true",
                    files={"file": ("a.flac", flac, "audio/flac")})
    assert separate_calls == [] and calls == []
    assert _skipped(body)


def test_wav_with_unreadable_channel_count_skips_separation(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """PyAV reports 0 channels for a 1024-channel WAV and cannot transcode
    it (an FFmpegError: the fallback hands over the original), while
    libsndfile reads it whole: its size is unknown, so it is skipped."""
    import wave
    with wave.open(str(tmp_path / "src.wav"), "wb") as w:
        w.setnchannels(1024)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(b"\x80" * (1024 * 800))
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    body = _post_ok(client, separate_bgm="true",
                    files={"file": ("a.wav", (tmp_path / "src.wav").read_bytes(),
                                    "audio/wav")})
    assert separate_calls == []
    assert _skipped(body)


def _with_declared_duration(mkv: bytes, seconds: float) -> bytes:
    """`mkv` with its Segment Info Duration (an 8-byte float in the
    segment's ms timescale) overwritten to declare `seconds`."""
    import struct
    b = bytearray(mkv)
    i = b.index(b"\x44\x89\x88")
    b[i + 3:i + 11] = struct.pack(">d", seconds * 1000.0)
    return bytes(b)


def test_disk_fallback_measures_the_original_by_decoding_it(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """The disk fallback hands the separator the original, whose header the
    uploader wrote: one declaring 0.5 s while holding 6 s, past a 2 s cap,
    is measured by decoding it and skipped."""
    mkv = _with_declared_duration(
        _flac(tmp_path / "src.mkv", seconds=6.0, fmt="matroska"), 0.5)
    (tmp_path / "lie.mkv").write_bytes(mkv)
    with av.open(str(tmp_path / "lie.mkv")) as c:
        assert c.duration / 1_000_000 < 1
    _available_ram(monkeypatch, available=4 * 2 * _S16_BYTES_PER_S)
    _disk_usage_reporting(monkeypatch, free=10_000)
    body = _post_ok(client, separate_bgm="true",
                    files={"file": ("a.mkv", mkv, "video/x-matroska")})
    assert separate_calls == []
    assert _skipped(body)


def test_disk_fallback_keeps_a_short_original(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    mkv = _flac(tmp_path / "src.mkv", seconds=1.0, fmt="matroska")
    _available_ram(monkeypatch, available=4 * 2 * _S16_BYTES_PER_S)
    _disk_usage_reporting(monkeypatch, free=10_000)
    assert "warnings" not in _post_ok(
        client, separate_bgm="true",
        files={"file": ("a.mkv", mkv, "video/x-matroska")})
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".mkv")


def _mat5(samples: int) -> bytes:
    """A MATLAB 5 file as libsndfile writes it: `samples` u8 frames, mono,
    at 1 Hz. libsndfile (the separator's decoder) reads it whole, but PyAV
    cannot open it, so nothing on the route has measured it."""
    import struct

    def matrix(name, dims, data, pad=0):
        nm = name.encode()
        body = (struct.pack("<4I", 6, 8, 6, 0)
                + struct.pack("<2I2i", 5, 8, *dims)
                + struct.pack("<2I", 1, len(nm))
                + nm.ljust(16 if len(nm) > 8 else 8, b"\0")
                + data)
        return struct.pack("<2I", 14, len(body) + pad) + body
    rate = matrix("samplerate", (1, 1), struct.pack("<HHI", 4, 2, 1))
    # libsndfile counts the data matrix's tag 8 bytes long and expects that.
    wave = matrix("wavedata", (1, samples),
                  struct.pack("<2I", 2, samples)
                  + (b"\x80" * samples).ljust(-(-samples // 8) * 8, b"\0"),
                  pad=8)
    return b"MATLAB 5.0 MAT-file\0".ljust(124, b" ") + b"\x00\x01IM" + rate + wave


@pytest.mark.parametrize("name", ["a.wav", "a.mat"])
def test_a_body_pyav_cannot_open_never_reaches_the_separator(
        client, monkeypatch, fake_model, separate_calls, name):
    """2 KB of MAT5 at 1 Hz costs the separator's 44.1 kHz load about a
    gigabyte: a file PyAV cannot open is not measured, so it is skipped
    whatever its name says."""
    body_bytes = _mat5(2000)
    with pytest.raises(av.FFmpegError):
        av.open(io.BytesIO(body_bytes)).close()
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    body = _post_ok(client, separate_bgm="true",
                    files={"file": (name, body_bytes, "audio/wav")})
    assert separate_calls == []
    assert any("could not be prepared" in w for w in body["warnings"])


def _mp3_in_wav(path, seconds=10.0, rate=8000) -> bytes:
    """A `seconds`-long MP3 stream in a WAV container whose fmt byte rate
    and fact sample count are forged so the header declares about 1 ms."""
    import struct
    with av.open(str(path), "w", format="wav") as out:
        st = out.add_stream("libmp3lame", rate=rate)
        st.layout = "mono"
        t = np.arange(int(seconds * rate)) / rate
        pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        for n in range(0, len(pcm), 1152):
            fr = av.AudioFrame.from_ndarray(pcm[None, n:n + 1152],
                                            format="s16", layout="mono")
            fr.sample_rate, fr.pts = rate, n
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    b = bytearray(path.read_bytes())
    i = b.index(b"fmt ")
    b[i + 16:i + 20] = struct.pack("<I", 10_000_000)
    j = b.index(b"fact")
    b[j + 8:j + 12] = struct.pack("<I", 100)
    return bytes(b)


def test_a_compressed_wav_is_measured_by_the_transcode_not_its_header(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    """Only a PCM WAV's header is a true size: an MP3 inside a WAV whose
    byte rate lies declares 1 ms for 10 s of audio, past a 2 s cap."""
    wav = _mp3_in_wav(tmp_path / "src.wav")
    (tmp_path / "lie.wav").write_bytes(wav)
    with av.open(str(tmp_path / "lie.wav")) as c:
        assert c.duration / 1_000_000 < 1
    _available_ram(monkeypatch, available=4 * 2 * _S16_BYTES_PER_S)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    body = _post_ok(client, separate_bgm="true",
                    files={"file": ("a.wav", wav, "audio/wav")})
    assert separate_calls == []
    assert _skipped(body)


def test_a_pcm_wav_keeps_the_direct_separator_path(
        client, monkeypatch, fake_model, separate_calls, tmp_path):
    import wave
    with wave.open(str(tmp_path / "src.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\0\0" * 2 * 44100)
    _available_ram(monkeypatch, available=64_000_000_000)
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    calls = _record_budget(monkeypatch)
    assert "warnings" not in _post_ok(
        client, separate_bgm="true",
        files={"file": ("a.wav", (tmp_path / "src.wav").read_bytes(),
                        "audio/wav")})
    assert calls == []
    assert len(separate_calls) == 1 and separate_calls[0].endswith(".wav")
