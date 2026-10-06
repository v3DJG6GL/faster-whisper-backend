"""Locate the ffmpeg executable, shared by the streaming transport and the
media package (download, captions, subtitle muxing), and name the demuxers a
client-supplied file must never open as."""

import functools
import logging
import shutil

logger = logging.getLogger(__name__)

# Demuxers that pull in OTHER inputs: playlists, session descriptions, image
# sequences, composition manifests. `protocol_whitelist=file` does not close
# them for local reads — their nested opens use the whitelisted file protocol,
# and concat's default safe=1 still admits a bare same-directory name (the
# demuxer is chosen by CONTENT, so an extensionless upload reading
# "ffconcat version 1.0 / file 'x.wav'" decodes x.wav). A real clip is
# self-contained, so refusing these by the opened format's name rejects
# nothing legitimate.
MULTI_INPUT_DEMUXERS = frozenset({
    "concat", "hls", "applehttp", "dash", "imf", "sdp", "rtp", "rtsp",
    "image2", "avisynth", "vapoursynth",
})


def is_multi_input_format(name: "str | None") -> bool:
    """True when a demuxer name (PyAV's ``container.format.name``, which can
    be a comma list such as "mov,mp4,m4a") is one of MULTI_INPUT_DEMUXERS."""
    return not MULTI_INPUT_DEMUXERS.isdisjoint((name or "").split(","))


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
