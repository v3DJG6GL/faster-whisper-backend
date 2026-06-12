"""Audio transport decoders for the streaming endpoint.

The session always consumes 16 kHz mono s16le PCM. Clients may send that raw
(lowest latency, recommended) or a browser-native encoded container
(WebM/Opus/Ogg via MediaRecorder), which we decode with a per-session ffmpeg
stdin→stdout pipe.

Both transports push decoded PCM to an async ``sink(bytes)`` callback (the
session's feed, serialized under a lock by the caller).
"""

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

Sink = Callable[[bytes], Awaitable[None]]

# Raw-PCM format tags (fed straight through); anything else goes via ffmpeg.
RAW_FORMATS = {"pcm_s16le", "pcm", "s16le", "raw"}
ENCODED_FORMATS = {
    "webm", "ogg", "ogg_opus", "opus", "wav", "mp4", "m4a", "mp3", "flac", "aac",
}


class RawPcmTransport:
    """Pass-through: the client already sends 16 kHz mono s16le PCM."""

    def __init__(self, sink: Sink):
        self._sink = sink

    async def start(self) -> None:
        pass

    async def feed(self, data: bytes) -> None:
        await self._sink(data)

    async def aclose(self) -> None:
        pass


class FfmpegTransport:
    """Decode an arbitrary container/codec to 16 kHz mono s16le PCM via ffmpeg.

    Container bytes are written to ffmpeg's stdin as they arrive; a background
    reader task drains stdout and pushes PCM to the sink as ffmpeg produces it.
    """

    def __init__(self, sink: Sink, *, sample_rate: int = 16000, read_size: int = 8192):
        self._sink = sink
        self._sr = sample_rate
        self._read_size = read_size
        self._proc: "asyncio.subprocess.Process | None" = None
        self._reader: "asyncio.Task | None" = None
        self._closed = False

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(self._sr),
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._reader = asyncio.create_task(self._drain_stdout())

    async def _drain_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                chunk = await self._proc.stdout.read(self._read_size)
                if not chunk:
                    break
                await self._sink(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ffmpeg-transport] reader error: %s", exc)

    async def feed(self, data: bytes) -> None:
        if self._proc is None or self._proc.stdin is None or self._closed:
            return
        try:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            pass

    async def aclose(self) -> None:
        if self._closed or self._proc is None:
            return
        self._closed = True
        # Close stdin → ffmpeg sees EOF and flushes any remaining PCM.
        try:
            if self._proc.stdin is not None and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        if self._reader is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._reader), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._reader.cancel()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass


def make_transport(audio_format: str, sink: Sink, *, sample_rate: int = 16000):
    """Pick the transport for a client-declared audio format."""
    if audio_format in RAW_FORMATS:
        return RawPcmTransport(sink)
    return FfmpegTransport(sink, sample_rate=sample_rate)
