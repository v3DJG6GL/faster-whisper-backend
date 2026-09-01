"""Helpers shared by the streaming test modules (PCM frames + WebSocket drain).

Kept out of conftest.py so the fixture file stays about fixtures; import with
the same style the module already uses for conftest (``from _streaming_helpers
import ...`` or ``from tests._streaming_helpers import ...``).
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
