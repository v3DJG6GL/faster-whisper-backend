"""Locate the ffmpeg executable, shared by the streaming transport and the
media package (download, captions, subtitle muxing), name the demuxers a
client-supplied file must never open as, list the audio decoders of the
binary and of PyAV, and cap what PyAV's libav allocates in one piece and
how many streams it takes of one file."""

import concurrent.futures
import ctypes
import errno
import functools
import glob
import logging
import os
import shutil
import subprocess
import threading

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


# The largest single allocation PyAV's libav makes in this process. A
# container's header is trusted for sizes inside av.open, ahead of every
# decode budget: the mov demuxer builds its index for as many samples as the
# sample table (or a fragment's track run) DECLARES, ~36 bytes of RAM each,
# so a file of under 5 KB costs 3 GB per track at libav's own limit
# (INT_MAX). A real file needs little: four hours of 48 kHz AAC in an m4a
# decode in full from 32 MiB up — and SHORT, without an error, below that, so
# the cap stays far above it (~32 h of such audio). One track under the cap
# still costs up to ~1.5x the cap, and tracks add up: PYAV_MAX_STREAMS bounds
# how many.
_LIBAV_MAX_ALLOC = 256 * 1024 * 1024

# The streams (video, audio, subtitle, attachment: every kind counts) an
# in-process open of a client-supplied file takes, libav's `max_streams`
# (its own default is 1000). Each track of an MP4 / MOV can cost ~1.5x
# _LIBAV_MAX_ALLOC at open, so forty tracks in 20 KB were 14 GB under that
# cap alone; with this the worst file costs twenty tracks. What a file with
# more streams gets is the demuxer's choice: mov fails the open (ENOMEM,
# answered as TOO_MUCH_TO_OPEN), Matroska opens with its first twenty and
# leaves the rest out. A recording has one to three streams; a film rip
# lists its video and audio first, ahead of the subtitles and fonts that
# make it long.
PYAV_MAX_STREAMS = 20
TOO_MUCH_TO_OPEN = (f"unsupported container: more than {PYAV_MAX_STREAMS} "
                    "tracks, or more than the server opens of one file")


def pyav_stream_limit() -> "dict[str, str]":
    """The av.open option that holds a client-supplied file to
    PYAV_MAX_STREAMS streams."""
    return {"max_streams": str(PYAV_MAX_STREAMS)}


def refused_as_too_much(exc: BaseException) -> bool:
    """True when an av.open failed for lack of memory: a stream past
    PYAV_MAX_STREAMS (mov), or an allocation past _LIBAV_MAX_ALLOC. A refusal
    of the file, not a fault of the server."""
    return getattr(exc, "errno", None) == errno.ENOMEM

_libav_alloc_lock = threading.Lock()
_libav_alloc_capped: "bool | None" = None      # None: not tried yet


def _libavutil_paths() -> "list[str]":
    """The libavutil files the installed PyAV wheel bundles — the library
    its extension modules loaded: next to the package in av.libs (Linux,
    Windows) or inside it in .dylibs (macOS)."""
    import av  # noqa: PLC0415 — optional dependency; callers import it first
    package = os.path.dirname(os.path.abspath(av.__file__))
    libs = os.path.join(os.path.dirname(package), "av.libs")
    found = set()
    for pattern in (os.path.join(libs, "libavutil*.so*"),
                    os.path.join(libs, "avutil*.dll"),
                    os.path.join(package, ".dylibs", "libavutil*.dylib")):
        found.update(glob.glob(pattern))
    return sorted(found)


def _set_libav_max_alloc() -> bool:
    try:
        paths = _libavutil_paths()
        if len(paths) != 1:
            # None: a PyAV built against system libraries, or a new wheel
            # layout. Several: capping one of them may cap the wrong one.
            raise LookupError(f"{len(paths)} libavutil libraries found")
        # Loading by the path PyAV loaded it from yields that same library,
        # not a second copy with a limit of its own.
        av_max_alloc = ctypes.CDLL(paths[0]).av_max_alloc
        av_max_alloc.argtypes = [ctypes.c_size_t]
        av_max_alloc.restype = None
        av_max_alloc(_LIBAV_MAX_ALLOC)
    except Exception as exc:  # noqa: BLE001 — fail open: decoding goes on uncapped
        logger.warning("[ffmpeg] could not set PyAV's allocation cap (%s: %s); "
                       "a crafted container can make an open allocate "
                       "gigabytes.", type(exc).__name__, exc)
        return False
    return True


def cap_libav_alloc() -> bool:
    """Limit a single allocation of PyAV's libav to _LIBAV_MAX_ALLOC, for
    the whole process (av_max_alloc is a libavutil global, shared by PyAV's
    own opens and faster-whisper's). Call it ahead of every in-process open
    of client-supplied bytes; the work is done once, later calls only read
    the result. False when the library was not found or did not take the
    call: logged once, and decoding goes on without the cap."""
    global _libav_alloc_capped
    if _libav_alloc_capped is None:
        with _libav_alloc_lock:
            if _libav_alloc_capped is None:
                _libav_alloc_capped = _set_libav_max_alloc()
    return _libav_alloc_capped


# Header opens of client bytes (the export probe, the upload's input check)
# run one at a time, process-wide: one crafted open may hold PYAV_MAX_STREAMS
# x ~1.5 x _LIBAV_MAX_ALLOC while it lasts, and the default executor would run
# as many of them at once as it has workers. A real file's header open takes
# milliseconds, so the queue is invisible to an ordinary upload.
_CLIENT_OPEN_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="client-open")


def client_open_pool() -> concurrent.futures.ThreadPoolExecutor:
    """The one-worker executor every header open of a client file runs on."""
    return _CLIENT_OPEN_POOL


def max_alloc_args() -> "list[str]":
    """The ffmpeg CLI global option that holds a child reading a client file
    to the allocation cap the in-process opens have. A file the capped
    probe let through (the index it refused is dropped there) would
    otherwise have that index built in full by the child, a gigabyte and
    more per track.

    The cap is the child's for everything, its muxer included. The MP4
    muxer keeps 64-72 bytes per packet of a track in one array, so an MP4
    export ends with an error past ~3.7 million packets in one track (about
    17 h of 60 fps video, 22 h of 48 kHz AAC); Matroska keeps no such
    table. A stream copy needs nothing else near the cap."""
    return ["-max_alloc", str(_LIBAV_MAX_ALLOC)]


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
