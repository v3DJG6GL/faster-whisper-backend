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
def single_input_demuxers() -> str:
    """PyAV's `format_whitelist` for client-supplied bytes: every input
    demuxer its libavformat has, minus MULTI_INPUT_DEMUXERS. The refusal by
    the opened format's name comes too late on its own: a playlist demuxer
    opens its first entry inside av.open, so a missing and a present sibling
    already fail differently. libavformat checks this list right after
    probing, before the demuxer reads its header, and answers EINVAL."""
    import av  # noqa: PLC0415 — optional dependency; callers import it first
    names = set()
    for n in av.formats_available:
        try:
            name = av.ContainerFormat(n, "r").name   # muxer-only names raise
        except ValueError:
            continue
        if not is_multi_input_format(name):
            names.update(name.split(","))
    return ",".join(sorted(names))


def format_whitelist_args(format_name: "str | None") -> "list[str]":
    """The ffmpeg CLI input options ahead of a client file's `-i`: file
    protocol only and, given the demuxer PyAV probed the file with (its full
    comma name, e.g. "matroska,webm"), only that demuxer — the CLI binary
    can carry demuxers PyAV lacks (dash, imf) whose content probe could win
    for the same bytes. -format_whitelist keeps the CLI's own detection,
    unlike -f, and refuses whatever else it would pick."""
    args = ["-protocol_whitelist", "file"]
    if format_name:
        args += ["-format_whitelist", format_name]
    return args


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Prefer a system ffmpeg on PATH (usually newer/faster), else the
    bundled imageio-ffmpeg binary (cross-platform, pulled by
    requirements-heavy.txt), else the bare name ``"ffmpeg"``, which surfaces
    a clear FileNotFoundError if ffmpeg is truly absent."""
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
