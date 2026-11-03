"""Helpers shared by the streaming test modules (PCM frames + WebSocket drain).

Kept out of conftest.py so the fixture file stays about fixtures; import as
``from tests._streaming_helpers import ...`` (tests/ is a package and
pytest.ini puts the repo root on sys.path).
"""

import numpy as np
from starlette.websockets import WebSocketDisconnect


def const_pcm(level: int, ms: int, sample_rate: int = 16000) -> bytes:
    """One constant-level int16 PCM frame of ``ms`` milliseconds."""
    return np.full(sample_rate * ms // 1000, level, dtype="<i2").tobytes()


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
