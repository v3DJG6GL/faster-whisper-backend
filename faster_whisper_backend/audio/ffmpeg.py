"""Locate the ffmpeg executable, shared by the streaming transport and the
media package (download, captions, subtitle muxing), name the demuxers a
client-supplied file must never open as, and list the audio decoders of
the binary and of PyAV."""

import functools
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Demuxers that pull in OTHER inputs: playlists, session descriptions, image
# sequences, composition manifests. `protocol_whitelist=file` does not close
# them for local reads — their nested opens use the whitelisted file protocol,
# and concat's default safe=1 still admits a bare same-directory name (the
# demuxer is chosen by CONTENT, so an extensionless upload reading
# "ffconcat version 1.0 / file 'x.wav'" decodes x.wav). A real clip is
# self-contained, so refusing these by the opened format's name rejects
# nothing legitimate. vobsub belongs here too: its header read opens the
# sibling <name>.sub, and a VobSub index never carries audio.
MULTI_INPUT_DEMUXERS = frozenset({
    "concat", "hls", "applehttp", "dash", "imf", "sdp", "rtp", "rtsp",
    "image2", "avisynth", "vapoursynth", "vobsub",
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



@functools.lru_cache(maxsize=1)
def pyav_audio_decoders() -> str:
    """PyAV's `codec_whitelist` for client-supplied bytes: every audio
    decoder its own libavcodec has (it can differ from the CLI binary's,
    see audio_decoder_whitelist), comma-joined. libavformat opens a decoder
    for every stream while probing, so without it a video track in an
    mkv/mp4 runs a video decoder in-process that the audio-only readers
    never need. "" when none is found: the caller then omits the option,
    as an empty whitelist would refuse every decoder."""
    import av  # noqa: PLC0415 — optional dependency; callers import it first
    names = []
    for n in sorted(av.codecs_available):
        try:
            if av.Codec(n, "r").type == "audio":   # encoder-only names raise
                names.append(n)
        except Exception:  # noqa: BLE001 — a name PyAV cannot open as a decoder
            continue
    return ",".join(names)

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


def audio_decoder_names(listing: str) -> "list[str]":
    """The decoder names in `ffmpeg -decoders` output whose type flag is
    audio ("A....." rows). The legend above the ------ rule has an
    " A..... = Audio" row of its own, so only rows below the rule count."""
    names = []
    in_table = False
    for line in listing.splitlines():
        parts = line.split()
        if not in_table:
            in_table = bool(parts) and set(parts[0]) == {"-"}
            continue
        if len(parts) >= 2 and parts[0].startswith("A"):
            names.append(parts[1])
    return names


# Binaries whose listing failure is already logged: the listing runs once per
# streaming session until it succeeds, and sessions are client-paced.
_decoder_listing_warned: "set[str]" = set()


@functools.lru_cache(maxsize=4)
def _listed_audio_decoders(exe: str) -> str:
    out = subprocess.run([exe, "-hide_banner", "-decoders"], capture_output=True,
                         encoding="utf-8", errors="replace", check=True,
                         timeout=15).stdout
    names = audio_decoder_names(out)
    if not names:
        raise ValueError("no audio decoder in the listing")
    return ",".join(names)


def audio_decoder_whitelist(exe: str) -> str:
    """`-codec_whitelist` value for client-supplied audio: every audio
    decoder of the binary `exe`, comma-joined. ffmpeg opens and decodes a
    frame of every stream while probing, so without it a video track in an
    mkv/mp4 runs a video decoder the audio-only output never needs. "" when
    the listing fails: the caller then omits the flag, as an empty whitelist
    would refuse every decoder. A failure is not cached (lru_cache keeps no
    exception), so the next caller asks the binary again. Blocking: call it
    off the event loop."""
    try:
        return _listed_audio_decoders(exe)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        if exe in _decoder_listing_warned:
            return ""
        _decoder_listing_warned.add(exe)
        logger.warning("[ffmpeg] could not list the audio decoders of %s (%s); "
                       "decoding without a decoder whitelist.", exe, exc)
        return ""
