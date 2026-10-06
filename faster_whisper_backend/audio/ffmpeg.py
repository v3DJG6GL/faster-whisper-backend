"""Locate the ffmpeg executable, shared by the streaming transport and the
media package (download, captions, subtitle muxing)."""

import functools
import logging
import shutil

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Resolve the ffmpeg executable. Prefers a system ffmpeg on PATH (usually
    newer/faster), else the bundled imageio-ffmpeg binary (cross-platform, pulled
    by requirements.txt), else the bare name ``"ffmpeg"`` as a last resort (which
    will surface a clear FileNotFoundError if truly absent)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # noqa: PLC0415 — optional, bundled binary fallback
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ffmpeg] no system ffmpeg and imageio-ffmpeg "
                       "unavailable (%s); encoded transports will fail.", exc)
        return "ffmpeg"
