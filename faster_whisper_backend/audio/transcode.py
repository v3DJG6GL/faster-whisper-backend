"""In-process audio transcoder using PyAV (already a faster-whisper dep,
so no extra requirement and no ffmpeg-on-PATH needed on Windows).

Output is signed 16-bit little-endian PCM in a RIFF/WAVE container; rate
and layout are per-caller:

- captures: 16 kHz mono — Whisper's native input rate AND the only format
  every browser plays without a system codec (Firefox on Linux ships no
  AAC decoder, so storing the dictation client's raw .m4a would mean a
  dead Play button on the /captures page).
- BGM separation: 44.1 kHz stereo — what the UVR/MDX separator natively
  consumes; anything else it would re-decode/resample itself, slowly.

decode_pieces_16k / decode_span_16k read short 16 kHz float pieces straight
into memory for a link's language check; all open the source through
_open_audio. refuse_multi_input applies the same demuxer refusal to a path
headed for a decoder that does not (faster-whisper, pyannote).
"""
from __future__ import annotations

import os

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from faster_whisper_backend.core import store_common

_OUT_FORMAT = "s16"
_OUT_CODEC = "pcm_s16le"     # WAV's native uncompressed codec

# Bits per sample of each PCM codec a WAV carries, for its byte-rate check.
_PCM_BITS = {"pcm_u8": 8, "pcm_s8": 8, "pcm_alaw": 8, "pcm_mulaw": 8,
             "pcm_s16le": 16, "pcm_s16be": 16, "pcm_s24le": 24,
             "pcm_s24be": 24, "pcm_s32le": 32, "pcm_s32be": 32,
             "pcm_f32le": 32, "pcm_f32be": 32, "pcm_s64le": 64,
             "pcm_f64le": 64, "pcm_f64be": 64}


class TranscodeBudgetExceeded(RuntimeError):
    """transcode_to_wav's output passed its `max_bytes` budget. A
    RuntimeError, so a caller that only knows the base class still reads it
    as a failed transcode; one that passed a budget for a reason of its own
    (main.py's BGM RAM cap) can tell it apart."""


def _av():
    try:
        import av
    except ImportError as e:
        raise RuntimeError("PyAV (av) not installed; cannot transcode") from e
    return av


def _open_audio(src_path: str):
    """(container, first audio stream) of `src_path`; the caller closes the
    container. ValueError when there is no audio stream."""
    # The source is an uploaded clip (or a downloaded link) whose bytes AND
    # filename extension a client chose, so a crafted concat/ffconcat, HLS
    # playlist or SDP input can coax the demuxer into following external
    # references — the classic ffmpeg local-file-read / SSRF surface. Two
    # locks, both needed: the file-protocol whitelist stops http:// (and
    # every other network protocol) — streaming/transport.py pins
    # "-protocol_whitelist pipe" on the realtime path for the same reason —
    # and the demuxer refusal below stops a playlist reading a sibling
    # local file through the whitelisted file protocol itself.
    # metadata_errors="ignore" as in faster-whisper's decode_audio: PyAV's
    # default "strict" raises UnicodeDecodeError (a ValueError — read as a
    # refusal upstream) on a non-UTF-8 tag, e.g. a cp1252 RIFF INFO title.
    container = _av().open(src_path, options={"protocol_whitelist": "file"},
                           metadata_errors="ignore")
    if audio_ffmpeg.is_multi_input_format(getattr(container.format, "name", None)):
        container.close()
        raise ValueError("unsupported container")
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        raise ValueError("source has no audio stream")
    return container, stream


def declared_shape(src_path: str) -> "tuple[float | None, int | None, int | None]":
    """(seconds, sample rate, channels) the container (or else its audio
    stream) declares for `src_path`; None for any it does not declare — PyAV
    reports 0 channels for a WAV of more than 64. Header data only: nothing
    is decoded. Raises what _open_audio raises."""
    container, stream = _open_audio(src_path)
    try:
        if container.duration and container.duration > 0:
            secs = container.duration / 1_000_000    # AV_TIME_BASE units
        elif stream.duration and stream.duration > 0 and stream.time_base:
            secs = float(stream.duration * stream.time_base)
        else:
            secs = None
        rate = stream.rate if (stream.rate or 0) > 0 else None
        channels = stream.channels if (stream.channels or 0) > 0 else None
        return secs, rate, channels
    finally:
        container.close()


def is_pcm_wav_or_flac(src_path: str) -> bool:
    """Whether `src_path` IS a PCM WAV or a FLAC — container and codec as the
    demuxer reads the bytes, whatever the file is named. Only for those does
    declared_shape's duration bound what a decoder reads: an MP3 or ADPCM
    stream inside a WAV is timed from a byte rate the uploader wrote. A PCM
    WAV counts only when its byte rate is its sample shape (rate x channels
    x bits): FFmpeg times a WAV whose data size overruns the file by that
    byte rate, while libsndfile reads every sample there is. Its fmt chunk's
    container width and block align must also be that codec's (see
    _wav_fmt_shape); an RF64 or an fmt the RIFF walk cannot read goes the
    transcode way. An unknown pcm_* codec fails closed. Raises what
    _open_audio raises."""
    container, stream = _open_audio(src_path)
    try:
        fmt = getattr(container.format, "name", None) or ""
        codec = getattr(stream.codec_context, "name", None) or ""
        if fmt == "wav":
            bits = _PCM_BITS.get(codec)
            rate, ch = stream.rate or 0, stream.channels or 0
            if not (bits is not None and rate > 0 and ch > 0
                    and stream.codec_context.bit_rate == rate * ch * bits):
                return False
            # FFmpeg picks the codec from an extensible header's
            # wValidBitsPerSample, libsndfile frames the samples by
            # wBitsPerSample: a container narrower than the valid bits is
            # timed short by FFmpeg and read whole by the separator. Both
            # round the width up to whole bytes (20-in-24, a 20-bit PCM).
            shape = _wav_fmt_shape(src_path)
            return (shape is not None
                    and (shape[0] + 7) // 8 * 8 == bits
                    and shape[1] == ch * bits // 8)
        return fmt == "flac" and codec == "flac"
    finally:
        container.close()


_WAV_MAX_CHUNKS = 64


def _wav_fmt_shape(src_path: str) -> "tuple[int, int] | None":
    """(wBitsPerSample, nBlockAlign) of the one 'fmt ' chunk ahead of 'data'
    in a RIFF (or big-endian RIFX) WAVE file; None for anything else — RF64,
    a second fmt, an fmt shorter than 16 bytes, no data within
    _WAV_MAX_CHUNKS chunks, a truncated header. Raises OSError when the
    file cannot be read."""
    import struct
    with open(src_path, "rb") as f:
        hdr = f.read(12)
        if len(hdr) < 12 or hdr[8:12] != b"WAVE":
            return None
        endian = {b"RIFF": "<", b"RIFX": ">"}.get(hdr[:4])
        if endian is None:
            return None
        found = None
        for _ in range(_WAV_MAX_CHUNKS):
            head = f.read(8)
            if len(head) < 8:
                return None
            cid, size = head[:4], struct.unpack(endian + "I", head[4:])[0]
            if cid == b"data":
                return found
            if cid == b"fmt ":
                body = f.read(16)
                if found is not None or size < 16 or len(body) < 16:
                    return None
                block_align, bits = struct.unpack(endian + "HH", body[12:16])
                found = (bits, block_align)
                f.seek(size - 16 + (size & 1), os.SEEK_CUR)
            else:
                f.seek(size + (size & 1), os.SEEK_CUR)
        return None


def decoded_cost_within(src_path: str, *, cost_per_s, max_cost: float) -> bool:
    """Whether `src_path`, decoded to its end, costs at most `max_cost`: each
    frame is charged its duration times cost_per_s(rate, channels) at its own
    shape. Nothing is written; False as soon as the sum passes `max_cost`, or
    on a frame without a channel count. Unlike declared_shape this does not
    trust the header, which the uploader wrote: a Matroska Segment Duration
    can declare a second for hours of audio. Raises what the decode raises."""
    container, stream = _open_audio(src_path)
    try:
        cost = 0.0
        for frame in _decoded_frames(container, stream):
            channels = len(frame.layout.channels)
            if channels <= 0 or not frame.sample_rate:
                return False
            cost += (frame.samples / frame.sample_rate
                     * cost_per_s(frame.sample_rate, channels))
            if cost > max_cost:
                return False
        return True
    finally:
        container.close()


def refuse_multi_input(path: str) -> None:
    """ValueError("unsupported container") when `path` opens as a demuxer
    that pulls in OTHER inputs (see audio_ffmpeg.MULTI_INPUT_DEMUXERS).

    For paths handed to a decoder that does not go through _open_audio
    (faster-whisper's decode_audio, pyannote): run it on the raw upload
    first. No audio stream is required, and an av error is NOT refused —
    the same libavformat open in the real decoder raises it anyway, with
    that caller's own error handling."""
    av = _av()
    try:
        container = av.open(path, options={"protocol_whitelist": "file"},
                            metadata_errors="ignore")
    except av.FFmpegError:
        return
    try:
        if audio_ffmpeg.is_multi_input_format(
                getattr(container.format, "name", None)):
            raise ValueError("unsupported container")
    finally:
        container.close()


def _decoded_frames(container, stream):
    """The audio frames of `stream` from where the container stands, skipping
    a packet the decoder rejects as invalid data. One damaged frame in an
    otherwise playable file (a bad mp3 frame, a flac block with a broken
    CRC) must not abort the whole decode: faster-whisper's own decode_audio
    tolerates it, so the file transcribes. Any other decode error stays a
    PyAV FFmpegError (a decode failure: the caller falls back), NOT the
    plain ValueError main.py treats as a refused input."""
    invalid = _av().error.InvalidDataError
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except invalid:
            continue
        yield from frames


def _resampled(frames, *, format: str, layout: str, rate: int):
    """`frames` resampled to `format`/`layout`/`rate`, flush included. A
    resampler is built for the first frame's input format and raises a plain
    ValueError on a frame that differs (a concatenated mp3, a TS recording
    switching 5.1 and stereo) — read as a refused input upstream — so a new
    one replaces it, after flushing, whenever the input changes. Output pts
    count samples across resamplers: each new one would restart at 0 and the
    muxer refuses timestamps that go backwards."""
    from fractions import Fraction

    av = _av()
    resampler, setup, done = None, None, 0

    def _out(resampled):
        nonlocal done
        for out in resampled:
            out.pts, out.time_base = done, Fraction(1, rate)
            done += out.samples
            yield out

    for frame in frames:
        frame_setup = (frame.format.name, frame.layout.name, frame.sample_rate)
        if frame_setup != setup:
            if resampler is not None:
                yield from _out(resampler.resample(None))
            resampler = av.AudioResampler(format=format, layout=layout,
                                          rate=rate)
            setup = frame_setup
        # PyAV recomputes pts when None; the input frame's pts is on the
        # input timebase and would corrupt the output otherwise.
        frame.pts = None
        yield from _out(resampler.resample(frame))
    if resampler is not None:
        yield from _out(resampler.resample(None))


def _read_16k(container, stream, want: int, *, skip: int = 0,
              after: "float | None" = None):
    """Decode `stream` from where the container stands into 16 kHz mono
    float32: drop frames that end before `after` (seconds, STREAM time —
    media time plus the stream's start_time) and then the first `skip`
    samples, return the next `want` (fewer at EOF)."""
    import numpy as np

    # A seek lands on the keyframe at or before `after`.
    frames = (f for f in _decoded_frames(container, stream)
              if after is None or f.time is None
              or f.time + f.samples / f.sample_rate >= after)
    # The skipped samples are dropped as they decode, never collected: the
    # skip of a link's language-check piece comes from a remote playlist, so
    # holding it would let a tiny body ask for gigabytes.
    chunks, have, drop = [], 0, skip
    # Fresh resampler per call: it keeps samples buffered across a seek.
    for out in _resampled(frames, format="flt", layout="mono", rate=16000):
        chunk = out.to_ndarray().reshape(-1)
        if drop:
            cut = min(drop, chunk.size)
            drop -= cut
            chunk = chunk[cut:]
            if not chunk.size:
                continue
        chunks.append(chunk)
        have += chunk.size
        if have >= want:
            break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[:want]


def decode_pieces_16k(src_path: str, starts: "list[float]",
                      seconds: float) -> list:
    """`seconds` of audio from each of `starts` (seconds into the media) as
    16 kHz mono float32 arrays — what Whisper's language detection takes.
    Seeks instead of decoding the whole file; a piece past the end comes
    back empty."""
    want = int(seconds * 16000)
    pieces = []
    container, stream = _open_audio(src_path)
    try:
        # `starts` are media time; seek and frame.time are STREAM time, which
        # an MPEG-TS/HLS download (or a remux keeping the offset) starts at
        # start_time, not 0.
        st0 = (float(stream.start_time * stream.time_base)
               if stream.start_time is not None else 0.0)
        for start in starts:
            t = start + st0
            container.seek(int(t / stream.time_base), stream=stream)
            pieces.append(_read_16k(container, stream, want, after=t))
    finally:
        container.close()
    return pieces


def decode_span_16k(src_path: str, skip: float, seconds: float):
    """`seconds` of audio after the first `skip` seconds of a short file, as
    one 16 kHz mono float32 array. Counts decoded samples instead of
    seeking: a run of concatenated stream segments (a link's language-check
    piece) carries the stream's own timestamps, which neither start at 0
    nor need be continuous."""
    container, stream = _open_audio(src_path)
    try:
        return _read_16k(container, stream, int(seconds * 16000),
                         skip=int(skip * 16000))
    finally:
        container.close()


def _close_quietly(container):
    """Best-effort close of a PyAV container (None = nothing to close).
    Returns None once it is closed, else the still-open handle."""
    try:
        if container is not None:
            container.close()
            return None
    except Exception:
        pass
    return container


def transcode_to_wav_16k_mono(src_path: str, dst_path: str) -> int:
    return transcode_to_wav(src_path, dst_path, rate=16000, layout="mono")


def transcode_to_wav(src_path: str, dst_path: str, *,
                     rate: int, layout: str,
                     max_bytes: "int | None" = None) -> int:
    """Decode anything PyAV understands, resample to `rate`/`layout`, write
    a RIFF/WAVE file at dst_path. Returns bytes written. On any failure
    the destination is best-effort unlinked.

    `max_bytes` bounds the PCM written: a few hundred KB of low-bitrate
    Opus/MP3 expands several hundred times into 44.1 kHz stereo, so a
    caller writing to a shared temp dir passes its free space and the
    transcode aborts (TranscodeBudgetExceeded) before it fills the disk."""
    av = _av()
    in_container = None
    out_container = None
    try:
        in_container, in_stream = _open_audio(src_path)

        out_container = av.open(dst_path, mode="w", format="wav")
        out_stream = out_container.add_stream(_OUT_CODEC, rate=rate)
        out_stream.layout = layout
        out_stream.format = _OUT_FORMAT

        written = 0
        for resampled in _resampled(
                _decoded_frames(in_container, in_stream),
                format=_OUT_FORMAT, layout=layout, rate=rate):
            # s16: two bytes per sample per channel.
            written += resampled.samples * len(resampled.layout.channels) * 2
            if max_bytes is not None and written > max_bytes:
                raise TranscodeBudgetExceeded(
                    "transcode output exceeds the free-space budget")
            for packet in out_stream.encode(resampled):
                out_container.mux(packet)

        for packet in out_stream.encode(None):
            out_container.mux(packet)
        # Close inside the try: close writes the trailer and flushes the
        # buffered data (all of it for a short clip), so a write error there
        # (ENOSPC) must fail the transcode, not return a truncated WAV.
        out_container.close()
        out_container = None

    except Exception:
        # Release the output handle before unlink (Windows holds the
        # lock otherwise).
        out_container = _close_quietly(out_container)
        store_common.unlink_quietly(dst_path)
        raise
    finally:
        _close_quietly(out_container)
        _close_quietly(in_container)

    return os.path.getsize(dst_path)
