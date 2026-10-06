"""Subtitle packaging: mux the client's SRT tracks into a retained video as
soft subtitle streams (MKV: SubRip tracks; MP4: 3GPP timed text), so an
exported video carries the languages its transcript was translated into.

Contract (same stance as media/download.py):
  - str(PackageError) is CLIENT-SAFE; ffmpeg's stderr never reaches a
    caller — it is logged (bounded, log_safe) and classified.
  - No client-supplied string ever becomes an ffmpeg input path or an option:
    the SRT texts are written into a private mkdtemp under fixed names, the
    source is a media-store path, and every input runs with
    `-protocol_whitelist file` (no network protocol). The whitelist does not
    stop a playlist-shaped source (ffconcat naming a sibling retained file)
    reading through the file protocol itself: probe_streams refuses those
    demuxers (audio/ffmpeg.MULTI_INPUT_DEMUXERS), and the route packages
    nothing the probe called unreadable.
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

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from faster_whisper_backend.core import proc as core_proc
from faster_whisper_backend.core.languages import iso639_2t, language_label
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
    # Why `video_codec` is None, so the route can answer with the right
    # error instead of one "no video stream" for all three cases.
    unreadable: bool = False        # the container could not be opened
    cover_art_only: bool = False    # the only "video" is embedded artwork
    # Position of the picture among the file's VIDEO streams (`0:v:N`) —
    # non-zero when cover art sits in front of the real video.
    video_index: int = 0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SubtitleTrack:
    lang: str
    label: str
    srt: str
    # Per-track dispositions (several tracks may carry each one).
    default: bool = False
    original: bool = False
    hearing_impaired: bool = False


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

def mp4_compatibility(video_codec: "str | None",
                      audio_codec: "str | None") -> "tuple[bool, str | None]":
    v = (video_codec or "").lower()
    a = (audio_codec or "").lower()
    if v and v not in _MP4_VIDEO:
        return False, f"MP4 can't carry {v.upper()} video without re-encoding — choose MKV"
    if a and a not in _MP4_AUDIO:
        return False, f"MP4 can't carry {a.upper()} audio without re-encoding — choose MKV"
    return True, None


UNREADABLE_REASON = "the file could not be read"

_AV_DISPOSITION_ATTACHED_PIC = 0x0400
_COVER_ART_CODECS = ("mjpeg", "png", "bmp", "gif")


def unreadable_streams() -> MediaStreams:
    """The facts for a file the probe could not open — what the route caches
    when probe_streams() raises, flagged so it never reads as "no video"."""
    return MediaStreams(None, None, None, None, None, False, UNREADABLE_REASON,
                        unreadable=True)


def _is_cover_art(stream) -> bool:
    """ffmpeg's attached-picture disposition (a real MJPEG video — a camera
    .avi/.mov — is NOT cover art); the codec name is only the fallback for a
    PyAV build that does not expose the disposition."""
    disp = getattr(stream, "disposition", None)
    if disp is not None:
        try:
            return bool(int(disp) & _AV_DISPOSITION_ATTACHED_PIC)
        except (TypeError, ValueError):
            pass
    return (stream.codec_context.name or "") in _COVER_ART_CODECS


def probe_streams(path: str) -> MediaStreams:
    """Codec facts via PyAV (the lean image has no ffprobe). `-protocol_whitelist
    file` here too: the file came from a client upload or a site download.
    ValueError for a demuxer that pulls in other files (an ffconcat upload
    listing another retained video would otherwise probe as that video and
    stream-copy it N times past MEDIA_MAX_BYTES) — the route caches that as
    unreadable."""
    import av  # optional dependency; the route maps ImportError to 503

    with av.open(path, options={"protocol_whitelist": "file"}) as c:
        if audio_ffmpeg.is_multi_input_format(getattr(c.format, "name", None)):
            raise ValueError("unsupported container")
        videos = [s for s in c.streams if s.type == "video"]
        # A cover-art stream (MJPEG/PNG in an m4a) is not the picture — a
        # file with nothing else has NO video (never fall back to it).
        real = [s for s in videos if not _is_cover_art(s)]
        v = real[0] if real else None
        # Every audio stream is muxed (`-map 0:a?`), so every one counts
        # for the MP4 verdict; the first is the one reported.
        acs = [(s.codec_context.name or "") for s in c.streams if s.type == "audio"]
        vc = (v.codec_context.name if v is not None else None)
        ac = (acs[0] or None) if acs else None
        width = int(v.codec_context.width) if v is not None and v.codec_context.width else None
        height = int(v.codec_context.height) if v is not None and v.codec_context.height else None
        duration = (float(c.duration) / 1_000_000.0) if c.duration else None
        video_index = videos.index(v) if v is not None else 0
        cover_only = bool(videos) and v is None
    ok, reason = mp4_compatibility(vc, ac)
    for name in acs[1:]:
        if not ok:
            break
        ok, reason = mp4_compatibility(vc, name)   # first offender wins
    return MediaStreams(video_codec=vc, audio_codec=ac, width=width, height=height,
                        duration=duration, mp4_ok=ok, mp4_reason=reason,
                        cover_art_only=cover_only, video_index=video_index)


def build_package_argv(src: str, srt_paths: "list[str]", tracks: "list[SubtitleTrack]",
                       *, container: str, out_path: str,
                       default_track: "int | None",
                       original_track: "int | None" = None,
                       audio_lang: "str | None" = None,
                       audio_label: "str | None" = None,
                       video_index: int = 0,
                       strip_empty_captions: bool = False) -> "list[str]":
    """The exact ffmpeg invocation (separate so tests can pin and swap it).
    `-map 0:v:N` takes ONE video stream only — `video_index` is
    MediaStreams.video_index, so a cover-art stream in front of the real
    video never becomes the picture; `-map 0:a?` keeps every audio track.
    A track is `default` / `original` when its own flag is set OR its index
    is `default_track` / `original_track` (the single-index form older
    clients send). `original` is the Matroska original-language flag
    (ffmpeg's `original` disposition, FlagOriginal since 4.4) — the track
    NAME stays the plain language name. MP4 has no such flag (movenc drops it
    silently), so the argv leaves it out there; `default` and
    `hearing_impaired` survive in MP4. MP4 tracks also get `handler_name` =
    the title: players such as Jellyfin read the hdlr name as the track
    title, and movenc writes no `title` for mov_text. `audio_lang`
    tags every audio stream with the spoken language (the source file usually
    carries the uploader's default, "en" for a German video).
    `strip_empty_captions` drops the H.264 SEI units, which takes broadcast
    captions that never carry text (media/captions.py) out of the picture —
    players otherwise list them as empty "Closed captions 1–4" tracks."""
    if container not in CONTAINERS:
        container = "mkv"
    argv = [audio_ffmpeg.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-protocol_whitelist", "file", "-i", src]
    for p in srt_paths:
        argv += ["-protocol_whitelist", "file", "-f", "srt", "-i", p]
    argv += ["-map", f"0:v:{max(0, int(video_index))}", "-map", "0:a?"]
    for i in range(len(srt_paths)):
        argv += ["-map", f"{i + 1}:0"]
    argv += ["-c:v", "copy", "-c:a", "copy"]
    if strip_empty_captions:
        argv += ["-bsf:v", "filter_units=remove_types=6"]
    if srt_paths:
        argv += ["-c:s", "mov_text" if container == "mp4" else "srt"]
    for i, t in enumerate(tracks):
        flags = [f for f, on in (
            ("default", t.default or default_track == i),
            ("original", (t.original or original_track == i) and container != "mp4"),
            ("hearing_impaired", t.hearing_impaired)) if on]
        argv += [f"-metadata:s:s:{i}", f"language={iso639_2t(t.lang)}",
                 f"-metadata:s:s:{i}", f"title={t.label}"]
        if container == "mp4":
            argv += [f"-metadata:s:s:{i}", f"handler_name={t.label}"]
        argv += [f"-disposition:s:{i}", "+".join(flags) if flags else "0"]
    if audio_lang:
        argv += ["-metadata:s:a", f"language={iso639_2t(audio_lang)}",
                 "-metadata:s:a", f"title={audio_label or language_label(audio_lang)}"]
    argv += ["-max_muxing_queue_size", "4096"]
    if container == "mp4":
        argv += ["-movflags", "+faststart", "-f", "mp4", out_path]
    else:
        argv += ["-f", "matroska", out_path]
    return argv


_SRT_HINT_RE = re.compile(r"sub_(\d+)\.srt")


def _write_srts(workdir: str, tracks: "list[SubtitleTrack]") -> "list[str]":
    """The tracks as `sub_N.srt` under fixed names, line endings normalised.
    Up to MAX_TRACKS × MAX_SRT_BYTES of rewriting + I/O — run off the loop."""
    srt_paths: "list[str]" = []
    for i, t in enumerate(tracks):
        p = os.path.join(workdir, f"sub_{i}.srt")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(t.srt.replace("\r\n", "\n").replace("\r", "\n"))
        srt_paths.append(p)
    return srt_paths


async def package(src: str, tracks: "list[SubtitleTrack]", *, container: str,
                  default_track: "int | None", timeout: float,
                  original_track: "int | None" = None,
                  audio_lang: "str | None" = None,
                  audio_label: "str | None" = None,
                  video_index: int = 0,
                  video_codec: "str | None" = None) -> str:
    """Mux `tracks` into `src` as soft subtitles; returns the output path
    inside a fresh `fwb-pkg-` workdir the CALLER removes after streaming it.
    Every failure removes the workdir here."""
    if container not in CONTAINERS:
        container = "mkv"
    workdir = tempfile.mkdtemp(prefix="fwb-pkg-")
    try:
        srt_paths = await asyncio.to_thread(_write_srts, workdir, tracks)
        strip = await _empty_captions_to_strip(src, video_codec, video_index, timeout)
        out = os.path.join(workdir, f"out.{container}")
        argv = build_package_argv(src, srt_paths, tracks, container=container,
                                  original_track=original_track,
                                  audio_lang=audio_lang, audio_label=audio_label,
                                  out_path=out, default_track=default_track,
                                  # Only when the picture is not 0:v:0 —
                                  # swapped-in argv builders keep their shape.
                                  **({"video_index": video_index}
                                     if video_index else {}),
                                  **({"strip_empty_captions": True} if strip else {}))
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            await core_proc.terminate_then_kill(proc)
            raise PackageTimeout("packaging timed out") from None
        except asyncio.CancelledError:
            await core_proc.terminate_then_kill(proc)
            raise
        tail = (err or b"")[-_STDERR_TAIL_MAX:].decode("utf-8", "replace")
        if proc.returncode != 0:
            logger.warning("[package] ffmpeg exited %s: %s", proc.returncode,
                           log_safe(tail[-300:]))
            m = _SRT_HINT_RE.search(tail)
            # The bare marker only points at a subtitle when there IS one and
            # the line does not name the source (input #0, or its file name)
            # — otherwise it is the source that ffmpeg could not demux.
            if m or (tracks and "Invalid data found" in tail
                     and os.path.basename(src) not in tail and "in#0" not in tail):
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


async def _empty_captions_to_strip(src: str, video_codec: "str | None",
                                   video_index: int, timeout: float) -> bool:
    """Whether this H.264 source carries only empty broadcast captions (and
    the server's ffmpeg can drop them). Never fails the package: any doubt
    keeps the captions."""
    if (video_codec or "").lower() != "h264":
        return False
    # Off the loop: the first call per process spawns `ffmpeg -bsfs`.
    if not await asyncio.to_thread(ffmpeg_has_bsf, "filter_units"):
        return False
    from faster_whisper_backend.media import captions as _cc
    try:
        return await asyncio.to_thread(_cc.embedded_captions_empty, src,
                                       video_index=video_index,
                                       timeout=min(300.0, max(10.0, timeout / 3)))
    except Exception as e:  # noqa: BLE001 — a scan failure must not fail the export
        logger.warning("[package] caption scan failed: %s", log_safe(str(e)[:200]))
        return False


@functools.lru_cache(maxsize=8)
def ffmpeg_has_bsf(name: str) -> bool:
    """Whether the server's ffmpeg has the bitstream filter `name`."""
    try:
        out = subprocess.run([audio_ffmpeg.ffmpeg_exe(), "-hide_banner", "-bsfs"], capture_output=True,
                             text=True, check=False, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(rf"^\s*{re.escape(name)}\s*$", out or "", re.M) is not None




def _has(listing: str, name: str) -> bool:
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", listing, re.M) is not None


@functools.lru_cache(maxsize=1)
def ffmpeg_capabilities() -> FfmpegCaps:
    """Whether the server's ffmpeg can package at all, and into which
    containers. Cached: the binary cannot change without a restart. Three
    subprocesses on first call — the lifespan warms it off the loop."""
    exe = audio_ffmpeg.ffmpeg_exe()
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
    ffmpeg_has_bsf.cache_clear()
