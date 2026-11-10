"""Audio transport decoders for the streaming endpoint.

The session always consumes 16 kHz mono s16le PCM. Clients may send that raw
(lowest latency, recommended) or a browser-native encoded container
(WebM/Opus/Ogg via MediaRecorder), which we decode with a per-session ffmpeg
stdin→stdout pipe.

Both transports push decoded PCM to an async ``sink(bytes)`` callback (the
route's sink, which queues it for the session's single consumer task).
"""

import asyncio
import logging
from typing import Awaitable, Callable

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg

logger = logging.getLogger(__name__)

Sink = Callable[[bytes], Awaitable[None]]

# Raw-PCM format tags (fed straight through); anything else goes via ffmpeg.
RAW_FORMATS = {"pcm_s16le", "pcm", "s16le", "raw"}
ENCODED_FORMATS = {
    "webm", "ogg", "ogg_opus", "opus", "wav", "mp4", "m4a", "mp3", "flac", "aac",
}

# Ceiling on decoded PCM bytes per container byte. Real codecs at 16 kHz mono
# stay far below it (Opus 6 kbps ≈ 43x, MP3 ≈ 8x, FLAC digital silence ≈ 800x),
# while a container declaring a near-zero sample rate (a 1 Hz WAV resamples at
# 32 000x) would otherwise turn a single frame into gigabytes of PCM.
_MAX_DECODE_RATIO = 1024
# Output allowed on top of the ratio, in seconds of PCM: container headers
# arrive before any audio, and an encoder may open with long near-silence, so
# a stream's first stretch must never trip the ratio on its own.
_DECODE_HEADROOM_S = 120.0
# Ceiling on any single allocation inside ffmpeg. The output ratio above only
# sees PCM once ffmpeg writes it, but a FLAC block declaring a 1 Hz rate is
# resampled to 16 kHz as ONE frame: an 8 KB stream allocates gigabytes before
# a byte reaches stdout. Real streams (65535-sample FLAC blocks at 48 kHz
# stereo, Opus, MP3, AAC) never come near it; a bomb frame fails to decode
# and the child stays at its baseline size.
_MAX_FFMPEG_ALLOC = 64 * 1024 * 1024


class RawPcmTransport:
    """Pass-through: the client already sends 16 kHz mono s16le PCM."""

    def __init__(self, sink: Sink):
        self._sink = sink

    @property
    def dead(self) -> bool:
        """Never: there is no decoder to die (see FfmpegTransport.dead)."""
        return False

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
        self._reader_dead = False
        self._dead_logged = False
        self._in_bytes = 0
        self._out_bytes = 0

    @property
    def dead(self) -> bool:
        """True once ffmpeg stopped producing PCM before aclose(): every later
        feed() is discarded, so the route ends the session instead of letting
        it hold a slot while decoding nothing."""
        return self._reader_dead and not self._closed

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            audio_ffmpeg.ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-max_alloc", str(_MAX_FFMPEG_ALLOC),
            # Restrict the input demuxer to the stdin pipe only. The client's
            # container bytes are self-contained (WebM/Opus/Ogg/mp4/…), so a
            # crafted playlist/concat/ffconcat input must not be able to coax
            # libavformat into following file:// or http:// external references
            # (the classic ffmpeg local-file-read / SSRF surface).
            "-protocol_whitelist", "pipe",
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
                self._out_bytes += len(chunk)
                if self._out_bytes > self._max_out_bytes():
                    # Treated as a dead decoder: the kill also breaks a feed()
                    # parked in stdin.drain(), and the route ends the session.
                    logger.warning("[ffmpeg-transport] decoded output exceeds %dx the "
                                   "input (%d -> %d bytes); stopping the decoder",
                                   _MAX_DECODE_RATIO, self._in_bytes, self._out_bytes)
                    try:
                        self._proc.kill()
                    except ProcessLookupError:
                        pass
                    break
                await self._sink(chunk)
        # A cancel is NOT swallowed: aclose() awaits this task through
        # wait_for, which hands an outer cancel (server shutdown) on to it —
        # returning normally here would lose that cancel.
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ffmpeg-transport] reader error: %s", exc)
        finally:
            self._reader_dead = True

    def _max_out_bytes(self) -> int:
        return (self._in_bytes * _MAX_DECODE_RATIO
                + int(_DECODE_HEADROOM_S * self._sr * 2))

    async def feed(self, data: bytes) -> None:
        if self._proc is None or self._proc.stdin is None or self._closed:
            return
        if self._reader_dead:
            # Client-paced (one call per inbound frame) — log once, not per frame,
            # or a dead decoder lets the client roll the rotating log (same
            # reasoning as routes._SHED_LOG_INTERVAL_S).
            if not self._dead_logged:
                self._dead_logged = True
                logger.warning("[ffmpeg-transport] ffmpeg exited; discarding audio "
                               "for the rest of this session")
            return
        self._in_bytes += len(data)
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
        try:
            if self._reader is not None:
                try:
                    await asyncio.wait_for(self._reader, timeout=5.0)
                except asyncio.TimeoutError:
                    # wait_for has already cancelled AND awaited the reader —
                    # nothing to join.
                    pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        finally:
            # Also on a cancel of this task (server shutdown): the
            # CancelledError still propagates, but _closed is already True,
            # so no later aclose() would come back to kill the process.
            if self._proc.returncode is None:
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass


def make_transport(audio_format: str, sink: Sink, *, sample_rate: int = 16000):
    """Pick the transport for a client-declared audio format."""
    if audio_format in RAW_FORMATS:
        return RawPcmTransport(sink)
    return FfmpegTransport(sink, sample_rate=sample_rate)
