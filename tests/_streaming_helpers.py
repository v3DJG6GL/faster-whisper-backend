"""Helpers shared by the streaming test modules (PCM frames, the config
handshake, WebSocket drain).

Kept out of conftest.py so the fixture file stays about fixtures; import as
``from tests._streaming_helpers import ...`` (tests/ is a package and
pytest.ini puts the repo root on sys.path).
"""

import numpy as np
from starlette.websockets import WebSocketDisconnect

from faster_whisper_backend.streaming.session import StreamConfig


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


def speak_and_stop(ws, limit: int = 200) -> tuple[list[dict], int | None]:
    """One ~2.5 s loud utterance, ~1.5 s silence (finalizes it), then stop;
    returns ``ws_drain``'s ``(msgs, close_code)``."""
    ws.send_bytes(const_pcm(8000, 2500))
    ws.send_bytes(const_pcm(0, 1500))
    ws.send_json({"type": "stop"})
    return ws_drain(ws, limit)


class AlwaysSpeech:
    """VAD stuck on: background noise (room mic, HVAC, music) whose flicker keeps
    resetting commit_silence_ms, so the utterance never ends on its own."""

    def is_speech(self, frame):
        return True

    def reset(self) -> None:
        pass


def fast_config(**over) -> StreamConfig:
    """Test-speed StreamConfig: 96 ms cadence, 192 ms commit silence, no trim."""
    base = dict(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    base.update(over)
    return StreamConfig(**base)
