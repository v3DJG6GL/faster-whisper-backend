"""Subtitle packaging: mux the client's SRT tracks into a retained video as
soft subtitle streams (MKV: SubRip tracks; MP4: 3GPP timed text), so an
exported video carries the languages its transcript was translated into.

Contract (same stance as url/download.py):
  - str(PackageError) is CLIENT-SAFE; ffmpeg's stderr never reaches a
    caller — it is logged (bounded, log_safe) and classified.
  - No client-supplied string ever becomes an ffmpeg input path or an option:
    the SRT texts are written into a private mkdtemp under fixed names, the
    source is a media-store path, and every input runs with
    `-protocol_whitelist file` (the local-file-read / SSRF surface a crafted
    playlist-shaped input would otherwise open — see audio/transcode.py).
  - Stream copy only: the picture and the sound are never re-encoded.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time

from faster_whisper_backend.core.store_common import log_safe

logger = logging.getLogger("whisper-api")

MAX_TRACKS = 12
MAX_SRT_BYTES = 2 * 1024 * 1024
CONTAINERS = ("mkv", "mp4")
_STDERR_TAIL_MAX = 4096


class PackageError(RuntimeError):
    """Packaging failed; str() is client-safe by contract."""


class SubtitleParseError(PackageError):
    """One of the SRT inputs could not be parsed."""


class PackageTimeout(PackageError):
    """The mux ran past MEDIA_PACKAGE_TIMEOUT_S."""


@dataclasses.dataclass(frozen=True)
class MediaStreams:
    """The facts about a media file the export UI decides on."""
    video_codec: "str | None"
    audio_codec: "str | None"
    width: "int | None"
    height: "int | None"
    duration: "float | None"
    mp4_ok: bool
    mp4_reason: "str | None"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SubtitleTrack:
    lang: str
    label: str
    srt: str


@dataclasses.dataclass(frozen=True)
class FfmpegCaps:
    available: bool
    mkv: bool
    mp4: bool
    reason: "str | None"
    version: "str | None"


# Stream families an MP4 carries with universal player support. ffmpeg CAN
# mux VP9/AV1/Opus into MP4, but playback support is uneven; Matroska holds
# them all losslessly. The reason text tells the client which codec stops
# MP4 so the panel can say why the choice is greyed out.
_MP4_VIDEO = frozenset({"h264", "hevc", "mpeg4"})
_MP4_AUDIO = frozenset({"aac", "mp3", "ac3", "eac3", "alac"})

# ISO 639-1 → ISO 639-2/T (terminology codes — `deu` not `ger`, `fra` not
# `fre`: what ffmpeg writes and what Matroska/MP4 players expect) for the
# languages Whisper outputs, plus the few 3-letter codes it uses directly.
_ISO639_1_TO_2T: "dict[str, str]" = {
    "af": "afr", "am": "amh", "ar": "ara", "as": "asm", "az": "aze", "ba": "bak",
    "be": "bel", "bg": "bul", "bn": "ben", "bo": "bod", "br": "bre", "bs": "bos",
    "ca": "cat", "cs": "ces", "cy": "cym", "da": "dan", "de": "deu", "el": "ell",
    "en": "eng", "es": "spa", "et": "est", "eu": "eus", "fa": "fas", "fi": "fin",
    "fo": "fao", "fr": "fra", "gl": "glg", "gu": "guj", "ha": "hau", "haw": "haw",
    "he": "heb", "hi": "hin", "hr": "hrv", "ht": "hat", "hu": "hun", "hy": "hye",
    "id": "ind", "is": "isl", "it": "ita", "ja": "jpn", "jv": "jav", "jw": "jav",
    "ka": "kat", "kk": "kaz", "km": "khm", "kn": "kan", "ko": "kor", "la": "lat",
    "lb": "ltz", "ln": "lin", "lo": "lao", "lt": "lit", "lv": "lav", "mg": "mlg",
    "mi": "mri", "mk": "mkd", "ml": "mal", "mn": "mon", "mr": "mar", "ms": "msa",
    "mt": "mlt", "my": "mya", "ne": "nep", "nl": "nld", "nn": "nno", "no": "nor",
    "oc": "oci", "pa": "pan", "pl": "pol", "ps": "pus", "pt": "por", "ro": "ron",
    "ru": "rus", "sa": "san", "sd": "snd", "si": "sin", "sk": "slk", "sl": "slv",
    "sn": "sna", "so": "som", "sq": "sqi", "sr": "srp", "su": "sun", "sv": "swe",
    "sw": "swa", "ta": "tam", "te": "tel", "tg": "tgk", "th": "tha", "tk": "tuk",
    "tl": "tgl", "tr": "tur", "tt": "tat", "uk": "ukr", "ur": "urd", "uz": "uzb",
    "vi": "vie", "yi": "yid", "yo": "yor", "yue": "yue", "zh": "zho",
}


def iso639_2t(code: "str | None") -> str:
    """The 639-2/T tag for a client language code ("pt-BR" → "por"); the
    region is dropped (containers store 639-2 only) and an unknown code
    becomes "und" rather than an invalid tag."""
    base = (code or "").strip().lower().split("-")[0]
    if len(base) == 3 and base.isalpha() and base not in _ISO639_1_TO_2T:
        return base   # already a 639-2 code
    return _ISO639_1_TO_2T.get(base, "und")


def lang_name(code: "str | None") -> str:
    """English name for the track title ("German", "Portuguese (BR)")."""
    from faster_whisper_backend.audio import translation as _tr
    raw = (code or "").strip()
    if not raw:
        return "Unknown"
    base, _, region = raw.partition("-")
    name = _tr._lang_name(base) or base
    return f"{name} ({region.upper()})" if region else name


def mp4_compatibility(video_codec: "str | None",
                      audio_codec: "str | None") -> "tuple[bool, str | None]":
    v = (video_codec or "").lower()
    a = (audio_codec or "").lower()
    if v and v not in _MP4_VIDEO:
        return False, f"MP4 can't carry {v.upper()} video without re-encoding — choose MKV"
    if a and a not in _MP4_AUDIO:
        return False, f"MP4 can't carry {a.upper()} audio without re-encoding — choose MKV"
    return True, None


def probe_streams(path: str) -> MediaStreams:
    """Codec facts via PyAV (the lean image has no ffprobe). `-protocol_whitelist
    file` here too: the file came from a client upload or a site download."""
    import av  # optional dependency; the route maps ImportError to 503

    with av.open(path, options={"protocol_whitelist": "file"}) as c:
        videos = [s for s in c.streams if s.type == "video"]
        # A cover-art stream (MJPEG/PNG in an m4a) is not the picture.
        real = [s for s in videos
                if (s.codec_context.name or "") not in ("mjpeg", "png", "bmp", "gif")]
        v = (real or videos or [None])[0]
        a = next((s for s in c.streams if s.type == "audio"), None)
        vc = (v.codec_context.name if v is not None else None)
        ac = (a.codec_context.name if a is not None else None)
        width = int(v.codec_context.width) if v is not None and v.codec_context.width else None
        height = int(v.codec_context.height) if v is not None and v.codec_context.height else None
        duration = (float(c.duration) / 1_000_000.0) if c.duration else None
    ok, reason = mp4_compatibility(vc, ac)
    return MediaStreams(video_codec=vc, audio_codec=ac, width=width, height=height,
                        duration=duration, mp4_ok=ok, mp4_reason=reason)


def build_package_argv(src: str, srt_paths: "list[str]", tracks: "list[SubtitleTrack]",
                       *, container: str, out_path: str,
                       default_track: "int | None",
                       original_track: "int | None" = None,
                       audio_lang: "str | None" = None,
                       audio_label: "str | None" = None) -> "list[str]":
    """The exact ffmpeg invocation (separate so tests can pin and swap it).
    `-map 0:v:0` takes the FIRST video stream only — a cover-art stream in an
    m4a must never become the picture; `-map 0:a?` keeps every audio track.
    `original_track` gets the Matroska original-language flag (ffmpeg's
    `original` disposition, FlagOriginal since 4.4; MP4 has no such flag and
    drops it) — the track NAME stays the plain language name. `audio_lang`
    tags every audio stream with the spoken language (the source file usually
    carries the uploader's default, "en" for a German video)."""
    from faster_whisper_backend.streaming.transport import ffmpeg_exe

    if container not in CONTAINERS:
        container = "mkv"
    argv = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-protocol_whitelist", "file", "-i", src]
    for p in srt_paths:
        argv += ["-protocol_whitelist", "file", "-f", "srt", "-i", p]
    argv += ["-map", "0:v:0", "-map", "0:a?"]
    for i in range(len(srt_paths)):
        argv += ["-map", f"{i + 1}:0"]
    argv += ["-c:v", "copy", "-c:a", "copy"]
    if srt_paths:
        argv += ["-c:s", "mov_text" if container == "mp4" else "srt"]
    for i, t in enumerate(tracks):
        flags = [f for f, on in (("default", default_track == i),
                                 ("original", original_track == i)) if on]
        argv += [f"-metadata:s:s:{i}", f"language={iso639_2t(t.lang)}",
                 f"-metadata:s:s:{i}", f"title={t.label}",
                 f"-disposition:s:{i}", "+".join(flags) if flags else "0"]
    if audio_lang:
        argv += ["-metadata:s:a", f"language={iso639_2t(audio_lang)}",
                 "-metadata:s:a", f"title={audio_label or lang_name(audio_lang)}"]
    argv += ["-max_muxing_queue_size", "4096"]
    if container == "mp4":
        argv += ["-movflags", "+faststart", "-f", "mp4", out_path]
    else:
        argv += ["-f", "matroska", out_path]
    return argv


_SRT_HINT_RE = re.compile(r"sub_(\d+)\.srt")


async def package(src: str, tracks: "list[SubtitleTrack]", *, container: str,
                  default_track: "int | None", timeout: float,
                  original_track: "int | None" = None,
                  audio_lang: "str | None" = None,
                  audio_label: "str | None" = None) -> str:
    """Mux `tracks` into `src` as soft subtitles; returns the output path
    inside a fresh `pkg-` workdir the CALLER removes after streaming it.
    Every failure removes the workdir here."""
    if container not in CONTAINERS:
        container = "mkv"
    workdir = tempfile.mkdtemp(prefix="pkg-")
    try:
        srt_paths: "list[str]" = []
        for i, t in enumerate(tracks):
            p = os.path.join(workdir, f"sub_{i}.srt")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(t.srt.replace("\r\n", "\n").replace("\r", "\n"))
            srt_paths.append(p)
        out = os.path.join(workdir, f"out.{container}")
        argv = build_package_argv(src, srt_paths, tracks, container=container,
                                  original_track=original_track,
                                  audio_lang=audio_lang, audio_label=audio_label,
                                  out_path=out, default_track=default_track)
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            await _kill(proc)
            raise PackageTimeout("packaging timed out") from None
        except asyncio.CancelledError:
            await _kill(proc)
            raise
        tail = (err or b"")[-_STDERR_TAIL_MAX:].decode("utf-8", "replace")
        if proc.returncode != 0:
            logger.warning("[package] ffmpeg exited %s: %s", proc.returncode,
                           log_safe(tail[-300:]))
            m = _SRT_HINT_RE.search(tail)
            if m or "Invalid data found" in tail:
                n = (int(m.group(1)) + 1) if m else 1
                raise SubtitleParseError(f"subtitle track {n} could not be parsed")
            raise PackageError("packaging failed")
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            raise PackageError("packaging produced no output")
        logger.info("[package] %s + %d track(s) → %s in %.1fs",
                    container, len(tracks), os.path.basename(out),
                    time.monotonic() - t0)
        return out
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


async def _kill(proc, grace: float = 5.0) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()


def _has(listing: str, name: str) -> bool:
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", listing, re.M) is not None


@functools.lru_cache(maxsize=1)
def ffmpeg_capabilities() -> FfmpegCaps:
    """Whether the server's ffmpeg can package at all, and into which
    containers. Cached: the binary cannot change without a restart. Two
    subprocesses on first call — the lifespan warms it off the loop."""
    from faster_whisper_backend.streaming.transport import ffmpeg_exe

    exe = ffmpeg_exe()
    try:
        mux = subprocess.run([exe, "-hide_banner", "-muxers"], capture_output=True,
                             text=True, timeout=15).stdout
        enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=15).stdout
        ver_out = subprocess.run([exe, "-version"], capture_output=True, text=True,
                                 timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return FfmpegCaps(False, False, False,
                          "no ffmpeg binary on the server (install ffmpeg or the "
                          "imageio-ffmpeg wheel)", None)
    ver = None
    m = re.search(r"ffmpeg version (\S+)", ver_out or "")
    if m:
        ver = m.group(1)[:32]
    mkv = _has(mux, "matroska") and _has(enc, "srt")
    mp4 = _has(mux, "mp4") and _has(enc, "mov_text")
    if not mkv:
        return FfmpegCaps(False, False, mp4,
                          "the server's ffmpeg lacks the matroska muxer or the srt encoder",
                          ver)
    return FfmpegCaps(True, True, mp4, None, ver)


def _reset_for_tests() -> None:
    ffmpeg_capabilities.cache_clear()
