"""Tests for the streaming audio transports (raw PCM passthrough + ffmpeg decode).

The WebM tests generate a real WebM/Opus clip with the system ffmpeg and decode
it back to PCM, so they exercise the actual MediaRecorder-style path; they skip
if ffmpeg (or its libopus encoder) is unavailable.
"""

import asyncio
import logging
import struct
import subprocess
import sys

import pytest

from faster_whisper_backend.audio.ffmpeg import ffmpeg_exe
from faster_whisper_backend.streaming.transport import (
    FfmpegTransport,
    RawPcmTransport,
    make_transport,
)
from tests._streaming_helpers import (
    discard as _nosink, drain_frames, handshake, ready_ws, stop_and_drain, stream_ws)


def test_ffmpeg_exe_resolves_to_runnable_binary():
    exe = ffmpeg_exe()
    assert exe
    r = subprocess.run([exe, "-version"], capture_output=True, timeout=15)
    assert r.returncode == 0
    assert b"ffmpeg version" in r.stdout


def _gen_webm(seconds: float) -> bytes:
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
             "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-f", "webm", "pipe:1"],
            capture_output=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg unavailable")
    if p.returncode != 0 or not p.stdout:
        pytest.skip("ffmpeg webm/opus encode unavailable")
    return p.stdout


def test_raw_transport_is_passthrough():
    got = bytearray()

    async def sink(b):
        got.extend(b)

    async def run():
        t = make_transport("pcm_s16le", sink)
        assert isinstance(t, RawPcmTransport)
        await t.start()
        await t.feed(b"\x01\x02\x03\x04")
        assert not t.dead
        await t.aclose()

    asyncio.run(run())
    assert bytes(got) == b"\x01\x02\x03\x04"


def test_ffmpeg_transport_decodes_webm_to_pcm():
    webm = _gen_webm(1.0)
    got = bytearray()

    async def sink(b):
        got.extend(b)

    async def run():
        t = make_transport("webm", sink)
        assert isinstance(t, FfmpegTransport)
        await t.start()
        await t.feed(webm)
        await t.aclose()

    asyncio.run(run())
    # ~1 s of 16 kHz mono s16le ≈ 32000 bytes; allow generous tolerance.
    assert len(got) > 16000


class _StuckProc:
    """An ffmpeg stand-in whose stdout never yields and that never exits."""
    returncode = None
    killed = False

    class stdin:
        @staticmethod
        def is_closing():
            return False

        @staticmethod
        def close():
            pass

    class stdout:
        @staticmethod
        async def read(_n):
            await asyncio.Event().wait()

    async def wait(self):
        await asyncio.Event().wait()

    def kill(self):
        self.killed = True


def test_ffmpeg_transport_dead_reader_warns_once(caplog):
    """Once ffmpeg has exited, feed() is still called once per inbound frame at
    the client's pace — it must log the dead decoder once, not per frame."""
    caplog.set_level(logging.WARNING, logger="faster_whisper_backend.streaming.transport")

    async def run():
        t = FfmpegTransport(_nosink)
        await t.start()
        t._proc.kill()
        for _ in range(100):
            if t._reader_dead:
                break
            await asyncio.sleep(0.05)
        assert t._reader_dead
        assert t.dead                  # the route ends the session on this
        for _ in range(50):
            await t.feed(b"\x00")
        await t.aclose()
        assert not t.dead              # a closed transport is finished, not dead

    asyncio.run(run())
    assert sum("ffmpeg exited" in r.getMessage() for r in caplog.records) == 1


def test_ffmpeg_transport_cancelled_aclose_still_kills_the_process():
    """A cancel while aclose() waits (server shutdown) must still kill ffmpeg:
    _closed is already True, so the routes backstop aclose() returns at once
    without killing it. The cancel itself must still propagate."""
    async def run():
        t = FfmpegTransport(_nosink)
        t._proc = _StuckProc()
        task = asyncio.create_task(t.aclose())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return t._proc.killed

    assert asyncio.run(run())


def test_ffmpeg_transport_cancel_during_reader_join_propagates():
    """A cancel while aclose() joins the stdout reader lands on the reader
    task; aclose() must still raise it (not return normally inside a handler
    meant to be cancelled), and the process must still be killed."""
    async def run():
        t = FfmpegTransport(_nosink)
        t._proc = _StuckProc()
        t._reader = asyncio.create_task(t._drain_stdout())
        task = asyncio.create_task(t.aclose())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return t._proc.killed, t._reader_dead

    assert asyncio.run(run()) == (True, True)


def test_ffmpeg_transport_reader_join_timeout_still_returns(monkeypatch):
    """The other side: a reader that outlives the join timeout is cancelled by
    wait_for, which reports a TimeoutError — aclose() returns normally."""
    real_wait_for = asyncio.wait_for

    def _short(aw, timeout):
        return real_wait_for(aw, timeout=0.05)

    monkeypatch.setattr(asyncio, "wait_for", _short)

    async def run():
        t = FfmpegTransport(_nosink)
        t._proc = _StuckProc()
        t._reader = asyncio.create_task(t._drain_stdout())
        await t.aclose()
        return t._proc.killed, t._reader.cancelled()

    assert asyncio.run(run()) == (True, True)


def _tiny_rate_wav(data_bytes: int, sample_rate: int = 1) -> bytes:
    """A pcm_u8 mono WAV at a near-zero sample rate: each byte of it resamples
    to 32 000 bytes of 16 kHz PCM at 1 Hz."""
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate, 1, 8)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", data_bytes) + b"\x80" * data_bytes)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_ffmpeg_transport_stops_a_decode_amplification_bomb():
    """A 128 KiB WAV at 1 Hz decodes to 4 GB of 16 kHz PCM. Output far beyond
    what any real codec yields per input byte must kill the decoder (dead →
    the route's decoder_failed close) instead of feeding the session. The
    input is above the ~64 KiB ffmpeg buffers before it decodes anything
    while stdin stays open."""
    wav = _tiny_rate_wav(128 * 1024)
    total = 0

    async def sink(b):
        nonlocal total
        total += len(b)

    async def run():
        t = FfmpegTransport(sink)
        await t.start()
        await t.feed(wav)
        for _ in range(100):
            if t.dead:
                break
            await asyncio.sleep(0.05)
        dead = t.dead
        await t.aclose()
        return dead

    assert asyncio.run(run())
    assert total <= len(wav) * 1024 + 120 * 16000 * 2


def _tiny_rate_flac() -> bytes:
    """A FLAC of 1 Hz noise in maximal 65535-sample blocks: one block
    resamples to ~1e9 samples at 16 kHz in a single frame. Noise keeps the
    blocks verbatim (~128 KiB each), past ffmpeg's ~64 KiB input probe, so it
    decodes while stdin stays open as it does in a live session."""
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", "1", "-ac", "1", "-i", "pipe:0",
             "-c:a", "flac", "-frame_size", "65535", "-f", "flac", "pipe:1"],
            input=bytes(range(7, 256, 7)) * 7282, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg unavailable")
    if r.returncode != 0 or not r.stdout:
        pytest.skip("ffmpeg flac encode unavailable")
    return r.stdout


def _peak_rss_kib(pid: int) -> int:
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_ffmpeg_transport_bounds_a_one_frame_allocation_bomb():
    """The output ratio never fires on a 1 Hz FLAC: ffmpeg allocates the whole
    resampled frame (~2 GB) before writing a byte — at the latest when the
    session ends and aclose() hands it EOF to flush. The per-allocation
    ceiling refuses that frame, so the child stays small and decodes nothing."""
    flac = _tiny_rate_flac()
    total = 0

    async def sink(b):
        nonlocal total
        total += len(b)

    async def run():
        t = FfmpegTransport(sink)
        await t.start()
        await t.feed(flac)
        pid = t._proc.pid
        peak = 0

        async def sample():
            nonlocal peak
            while True:
                try:
                    peak = max(peak, _peak_rss_kib(pid))
                except OSError:
                    return    # ffmpeg exited
                await asyncio.sleep(0.02)

        sampler = asyncio.create_task(sample())
        await t.aclose()      # EOF: ffmpeg flushes the buffered block
        sampler.cancel()
        return peak

    assert asyncio.run(run()) < 512 * 1024
    assert total == 0


def test_stream_route_accepts_webm_via_ffmpeg(client, energy_vad):
    webm = _gen_webm(2.0)
    with stream_ws(client) as ws:
        ready = handshake(ws, audio={"format": "webm"})
        assert ready["type"] == "ready"
        assert ready["audio_format"] == "webm"
        ws.send_bytes(webm)
        msgs = stop_and_drain(ws)[0]
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals, "expected a final from the decoded WebM"
    assert "welt" in "".join(m["committed"] + m.get("tail", "") for m in finals)


def test_stream_route_ends_the_session_when_the_decoder_dies(client, monkeypatch, energy_vad):
    """A dead ffmpeg discards every later frame while the audio keeps re-arming
    the idle deadline — the producer must say so and end the session instead
    of holding a STREAMING_MAX_SESSIONS slot until the client gives up."""
    from faster_whisper_backend.streaming import routes as stream_routes
    feeds = []

    class _DyingTransport:
        dead = False

        def __init__(self, sink, **_kw):
            pass

        async def start(self):
            pass

        async def feed(self, data):
            feeds.append(data)
            self.dead = True

        async def aclose(self):
            pass

    monkeypatch.setattr(stream_routes, "make_transport",
                        lambda fmt, sink, **kw: _DyingTransport(sink, **kw))
    with ready_ws(client, audio={"format": "webm"}) as ws:
        ws.send_bytes(b"\x1a\x45\xdf\xa3 corrupt")
        msgs = drain_frames(ws, 50)
    errors = [m for m in msgs if m.get("type") == "error"]
    assert [m["code"] for m in errors] == ["decoder_failed"], msgs
    assert len(feeds) == 1
