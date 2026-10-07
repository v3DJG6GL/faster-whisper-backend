"""Tests for the streaming audio transports (raw PCM passthrough + ffmpeg decode).

The ffmpeg tests generate a real WebM/Opus clip with the system ffmpeg and decode
it back to PCM, so they exercise the actual MediaRecorder-style path. Skipped if
ffmpeg (or libopus) is unavailable.
"""

import asyncio
import logging
import subprocess

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from faster_whisper_backend.audio.ffmpeg import ffmpeg_exe
from faster_whisper_backend.streaming.transport import (
    FfmpegTransport,
    RawPcmTransport,
    make_transport,
)


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


def test_ffmpeg_transport_dead_reader_warns_once(caplog):
    """Once ffmpeg has exited, feed() is still called once per inbound frame at
    the client's pace — it must log the dead decoder once, not per frame."""
    caplog.set_level(logging.WARNING, logger="faster_whisper_backend.streaming.transport")

    async def sink(b):
        pass

    async def run():
        t = FfmpegTransport(sink)
        await t.start()
        t._proc.kill()
        for _ in range(100):
            if t._reader_dead:
                break
            await asyncio.sleep(0.05)
        assert t._reader_dead
        for _ in range(50):
            await t.feed(b"\x00")
        await t.aclose()

    asyncio.run(run())
    assert sum("ffmpeg exited" in r.getMessage() for r in caplog.records) == 1


def test_ffmpeg_transport_cancelled_aclose_still_kills_the_process():
    """A cancel while aclose() waits (server shutdown) used to skip kill(), and
    _closed was already True, so the routes backstop aclose() returned at once
    and ffmpeg was never killed. The cancel itself must still propagate."""
    class _Stdin:
        def is_closing(self):
            return False

        def close(self):
            pass

    class _Proc:
        returncode = None
        stdin = _Stdin()
        killed = False

        async def wait(self):
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True

    async def sink(b):
        pass

    async def run():
        t = FfmpegTransport(sink)
        t._proc = _Proc()
        task = asyncio.create_task(t.aclose())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return t._proc.killed

    assert asyncio.run(run())


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


async def _nosink(b):
    pass


def test_ffmpeg_transport_cancel_during_reader_join_propagates():
    """A cancel while aclose() joins the stdout reader went to the reader task,
    which swallowed it and returned — so aclose() carried on and returned
    normally inside a handler that was meant to be cancelled. It must raise,
    and the process must still be killed."""
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


def test_stream_route_accepts_webm_via_ffmpeg(app_module, monkeypatch):
    # Force the energy gate — the synthetic sine tone is not real speech, so the
    # Silero VAD would reject it; this test only checks the ffmpeg decode path.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    webm = _gen_webm(2.0)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({"type": "config", "model": "whisper-1",
                          "audio": {"format": "webm"}})
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["audio_format"] == "webm"
            ws.send_bytes(webm)
            ws.send_json({"type": "stop"})
            msgs = []
            try:
                for _ in range(200):
                    msgs.append(ws.receive_json())
            except WebSocketDisconnect:
                pass
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals, "expected a final from the decoded WebM"
    assert "welt" in "".join(m["committed"] + m.get("tail", "") for m in finals)
