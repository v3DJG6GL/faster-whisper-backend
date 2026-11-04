"""Helpers shared by the streaming test modules (PCM frames, the config
handshake, WebSocket drain, an inert-decoder StreamSession).

Kept out of conftest.py so the fixture file stays about fixtures; import as
``from tests._streaming_helpers import ...`` (tests/ is a package and
pytest.ini puts the repo root on sys.path).
"""

import numpy as np
from starlette.websockets import WebSocketDisconnect

from faster_whisper_backend.streaming.session import StreamConfig, StreamSession
from faster_whisper_backend.streaming.vad import EnergyEndpointer
from tests.conftest import bearer


def const_pcm(level: int, ms: int, sample_rate: int = 16000) -> bytes:
    """One constant-level int16 PCM frame of ``ms`` milliseconds."""
    return np.full(sample_rate * ms // 1000, level, dtype="<i2").tobytes()


def config_frame(**extra) -> dict:
    """The minimal raw-PCM handshake config frame, plus any extra keys."""
    return {"type": "config", "model": "whisper-1",
            "audio": {"format": "pcm_s16le", "sample_rate": 16000}, **extra}


def handshake(ws, **extra) -> dict:
    """Send ``config_frame(**extra)`` and return the server's first reply."""
    ws.send_json(config_frame(**extra))
    return ws.receive_json()


def ws_drain(ws, limit: int = 200) -> tuple[list[dict], int | None]:
    """Receive JSON frames until the server closes (or ``limit`` frames).

    Returns ``(msgs, close_code)``; ``close_code`` is None when the limit was
    reached before the socket closed."""
    msgs: list[dict] = []
    try:
        for _ in range(limit):
            msgs.append(ws.receive_json())
    except WebSocketDisconnect as exc:
        return msgs, exc.code
    return msgs, None


def drain_frames(ws, limit: int = 200) -> list[dict]:
    """``ws_drain`` without the close code."""
    return ws_drain(ws, limit)[0]


def stop_and_drain(ws, limit: int = 200) -> tuple[list[dict], int | None]:
    """Send the stop frame and return ``ws_drain``'s ``(msgs, close_code)``."""
    ws.send_json({"type": "stop"})
    return ws_drain(ws, limit)


def stream_ws(client, raw_key):
    """Open the streaming socket with ``raw_key``'s bearer header (use as ``with``)."""
    return client.websocket_connect(
        "/v1/audio/transcriptions/stream", headers=bearer(raw_key))


def config_drain(client, raw_key, limit: int = 200) -> list[dict]:
    """Connect with ``raw_key``'s bearer header, send ``config_frame()`` and
    return every frame up to the server's close (``ws_drain``)."""
    with stream_ws(client, raw_key) as ws:
        ws.send_json(config_frame())
        return drain_frames(ws, limit)


def speak_and_stop(ws, limit: int = 200) -> tuple[list[dict], int | None]:
    """One ~2.5 s loud utterance, ~1.5 s silence (finalizes it), then stop;
    returns ``ws_drain``'s ``(msgs, close_code)``."""
    ws.send_bytes(const_pcm(8000, 2500))
    ws.send_bytes(const_pcm(0, 1500))
    return stop_and_drain(ws, limit)


class AlwaysSpeech:
    """VAD stuck on: background noise (room mic, HVAC, music) whose flicker keeps
    resetting commit_silence_ms, so the utterance never ends on its own."""

    def is_speech(self, frame):
        return True

    def reset(self) -> None:
        pass


def collector():
    """``(items, sink)``: an async one-argument callback (emit / on_final)
    that appends each call's argument to ``items``."""
    items: list = []

    async def sink(x):
        items.append(x)
    return items, sink


def const_final(text: str = "", dropped_all: bool = False):
    """Fake decode_final that always returns ``(text, [], dropped_all)``."""
    async def decode_final(audio, prompt):
        return (text, [], dropped_all)
    return decode_final


def const_partial(*words):
    """Fake decode_partial that always returns ``list(words)``
    (``(start, end, text)`` triples)."""
    async def decode_partial(audio, prompt):
        return list(words)
    return decode_partial


def raising_decoder(exc_type: type[BaseException], message: str):
    """Fake decode_partial / decode_final that raises ``exc_type(message)``
    on every call."""
    async def decode(audio, prompt):
        raise exc_type(message)
    return decode


async def _no_partial(audio, prompt):
    return []


async def _empty_final(audio, prompt):
    return ("", [], False)


def inert_session(cfg=None, *, decode_partial=None, decode_final=None,
                  endpointer=None, postprocess=lambda raw: raw, **kw):
    """StreamSession with inert fake decoders (no partial words, an empty
    final) unless given; ``kw`` are the optional seams (holdback, format_key,
    diagnose, on_final). Returns ``(session, msgs)``, every emitted frame."""
    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    s = StreamSession(
        config=cfg or StreamConfig(), endpointer=endpointer or EnergyEndpointer(),
        decode_partial=decode_partial or _no_partial,
        decode_final=decode_final or _empty_final,
        postprocess=postprocess, emit=emit, **kw)
    return s, msgs


def fast_config(**over) -> StreamConfig:
    """Test-speed StreamConfig: 96 ms cadence, 192 ms commit silence, no trim."""
    base = dict(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    base.update(over)
    return StreamConfig(**base)
