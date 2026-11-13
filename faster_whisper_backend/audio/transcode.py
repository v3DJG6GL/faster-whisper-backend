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
into memory for a link's language check, and decode_16k_bounded reads the
whole upload for Whisper under a duration ceiling; all open the source
through _open_audio. refuse_multi_input applies the same demuxer refusal to a path
headed for a decoder that does not (faster-whisper, pyannote).

Once the app has started its decode helpers (audio/decode_helper.py) every
function here that opens or decodes a file runs in one of them, under a
memory limit: the public name sends the request and reads the answer, the
underscore twin below it is the work itself (run in the helper, or right
here while no helpers run — the test suite, a platform that starts none).
"""
from __future__ import annotations

import errno
import os
import time

from faster_whisper_backend.audio import decode_helper as _decode_helper
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


# Longest decoded frame any reader here resamples. A frame goes through the
# resampler in one call, ahead of every output-side budget, and its rate is
# whatever the container declares: one 8192-sample block at 1 Hz is
# ~1.3 * 10^8 output samples (2+ GB). Real encoders stop at 8.2 s (65535
# samples of 8 kHz FLAC).
_MAX_FRAME_S = 60.0

# Most samples x channels of one decoded frame any reader here takes.
# _MAX_FRAME_S reads a frame's length at the rate the file declares, while
# buffering and resampling it cost its sample count: a codec whose header
# sets the frame length (ALAC: up to 2^24 samples, 26 bits per 65 536 of
# silence) gives a frame of a second or less at a declared rate in the MHz,
# and several hundred MB per channel on its way through the fifo and the
# resampler. Real frames stay far below: 65 535 samples x 8 channels of FLAC,
# 131 072 x channels of WavPack, ~1 s x channels of TTA (1.6 M at 192 kHz
# 7.1); the packets of a PCM WAV are 16 384 samples at that rate.
_MAX_FRAME_SAMPLES = 1 << 22

# Changes of input format, layout or rate _resampled follows in one stream:
# _MAX_SETUP_CHANGES up front plus one per _FRAMES_PER_SETUP_CHANGE frames
# decoded so far, a change counting rate // 48000 (at least 1). Each one
# flushes and builds a resampler, ~2 ms up to 192 kHz and ~18 ms at 655 kHz,
# while a 13-byte FLAC frame that names a new rate yields no output sample:
# unbounded, 280 KB of them hold a thread for ~30 s and no output-side budget
# ends it. The refill is in frames, not seconds of audio — a frame costs the
# sender bytes, while a 16-sample frame at 1 Hz is 16 s for the same 13. A
# real file changes 0-2 times; a day of MPEG-TS is allowed several hundred.
_MAX_SETUP_CHANGES = 128
_FRAMES_PER_SETUP_CHANGE = 4096

# What a decode demuxes of packets that give it no frame — rejected by the
# decoder, or accepted and turned into nothing — on top of the file's own
# size. The packets of a real file are disjoint parts of it, so those never
# add up to more than the file; the margin is for a demuxer that hands out a
# little more than it read.
_SKIPPED_PACKETS_MARGIN = 1 << 20


class DecodeTooLong(Exception):
    """decode_16k_bounded's input ran past its `max_seconds` ceiling, or one
    of its frames past _MAX_FRAME_S or _MAX_FRAME_SAMPLES."""


class ResampleFailed(Exception):
    """decode_16k_bounded's s16 mono resampler refused a group (e.g. more
    channels than it downmixes); __cause__ is the resampler's own error."""


class DecodeFailed(Exception):
    """decode_16k_bounded's decode failed after _open_audio had opened the
    file; __cause__ is the error itself, the decoder's av error or a plain
    one (the fifo's ValueError on a frame of another rate or sample format,
    _fw_bounded_groups' own on a frame with fewer channels than the first).
    Told apart from a failed open: the decode already ran, so decoding the
    path again meets the same error."""


class TranscodeBudgetExceeded(RuntimeError):
    """transcode_to_wav's output passed its `max_bytes` budget. A
    RuntimeError, so a caller that only knows the base class still reads it
    as a failed transcode; one that passed a budget for a reason of its own
    (main.py's BGM RAM cap) can tell it apart."""


# The helper process an operation ran in ended without an answer while it
# read the file: the decoder crashed on it, or the process was killed. Never
# raised while the operations run in-process — there the same file ends the
# server.
DecoderEnded = _decode_helper.HelperDied

# This module's own exceptions, which a helper's error comes back as.
_HELPER_ERRORS = {c.__name__: c for c in (
    DecodeTooLong, ResampleFailed, DecodeFailed, TranscodeBudgetExceeded)}

# How long a helper gets for an operation that only opens the file: the
# opens' own budget (granted to each of an open's two phases) and a margin.
# Past it the helper is killed, which also ends an index build no read
# timeout interrupts.
_OPEN_OP_TIMEOUT_S = 2 * audio_ffmpeg.HEADER_OPEN_BUDGET_S + 30.0
# The same for a link's language-check pieces: each is held to the discard
# budget (see _DISCARD_MAX_FRAMES), seconds of one thread at its worst.
_PIECE_OP_TIMEOUT_S = 60.0
_PIECE_OP_TIMEOUT_EACH_S = 30.0
TOO_SLOW_TO_DECODE = "unsupported audio: the piece takes too long to decode"

# What _helped answers when the caller is to do the work in this process.
_LOCAL = object()


def _helped(op: str, args: dict, *, opens: bool = False,
            timeout: "float | None" = None, long: bool = False, on_data=None):
    """The answer of operation `op` run in a decode helper, or _LOCAL when
    it is to run in this process (no helpers started, or this is one).
    Raises what the operation raised there. A helper that passed its memory
    budget is a MemoryError, one that ended on the file DecoderEnded — for
    an operation that only `opens` the file both are the open's own plain
    ValueError (a refusal of the file), and so is the deadline."""
    if not _decode_helper.routed():
        return _LOCAL
    try:
        return _decode_helper.run(op, args, timeout=timeout, long=long,
                                  on_data=on_data, classes=_HELPER_ERRORS)
    except _decode_helper.NotStarted:
        return _LOCAL
    except _decode_helper.HelperTimeout:
        raise ValueError(audio_ffmpeg.TOO_SLOW_TO_OPEN if opens
                         else TOO_SLOW_TO_DECODE) from None
    except _decode_helper.HelperOutOfMemory:
        if opens:
            raise ValueError(audio_ffmpeg.TOO_MUCH_TO_OPEN) from None
        raise
    except DecoderEnded:
        if opens:
            raise ValueError(audio_ffmpeg.STOPS_THE_DECODER) from None
        raise


def _unreadable_answer() -> DecoderEnded:
    return DecoderEnded(None, "the decode helper sent an unreadable answer")


def _frame_too_big(frame) -> bool:
    """Whether `frame` holds more than _MAX_FRAME_SAMPLES samples x
    channels."""
    return (frame.samples * max(1, len(frame.layout.channels))
            > _MAX_FRAME_SAMPLES)


def _av():
    try:
        import av
    except ImportError as e:
        raise RuntimeError("PyAV (av) not installed; cannot transcode") from e
    # Ahead of every open below: a container can declare an index of
    # gigabytes that av.open builds before any decode budget runs.
    audio_ffmpeg.cap_libav_alloc()
    return av


def _open_client_file(av, path: str, *, timeout: "float | None" = None):
    """av.open of client-chosen bytes: file protocol only, no demuxer that
    pulls in other inputs or reads a whole text file at the open
    (audio_ffmpeg.single_input_demuxers), and audio
    decoders only (audio_ffmpeg.pyav_audio_decoders). A
    playlist fails that allowlist with EINVAL before it opens a single
    entry, a subtitle text file before its demuxer reads a line; either is
    answered ValueError("unsupported container") — a plain
    ValueError, the refusal callers tell apart from an av decode error. So
    is a file with more streams than audio_ffmpeg.PYAV_MAX_STREAMS whose
    demuxer fails on the one past it, or one whose open asks for more than
    the allocation cap (audio_ffmpeg.TOO_MUCH_TO_OPEN).

    `timeout` (seconds; None: none) stops the open's file reads once it has
    passed. The open then fails with its usual av error, or returns a
    container holding what was read up to then — the caller tells by the
    clock. Reads of the returned container are never timed."""
    # metadata_errors="ignore" as in faster-whisper's decode_audio: PyAV's
    # default "strict" raises UnicodeDecodeError (a ValueError — read as a
    # refusal upstream) on a non-UTF-8 tag, e.g. a cp1252 RIFF INFO title.
    options = {"protocol_whitelist": "file",
               "format_whitelist": audio_ffmpeg.single_input_demuxers(),
               **audio_ffmpeg.pyav_stream_limit()}
    # Audio decoders only: stream probing would otherwise open the decoder
    # of a (crafted) video track that nothing here reads. The video stream
    # is still listed, just never decoded.
    if decoders := audio_ffmpeg.pyav_audio_decoders():
        options["codec_whitelist"] = decoders
    # PyAV's (open, read) pair: a limit on the open alone, so a decode of
    # the container handed back is never cut short by it.
    timed = {} if timeout is None else {"timeout": (timeout, None)}
    try:
        return av.open(path, options=options, metadata_errors="ignore", **timed)
    except av.FFmpegError as e:
        if getattr(e, "errno", None) == errno.EINVAL:
            raise ValueError("unsupported container") from e
        if audio_ffmpeg.refused_as_too_much(e):
            raise ValueError(audio_ffmpeg.TOO_MUCH_TO_OPEN) from e
        raise


def _open_audio(src_path: str, *, timeout: "float | None" = None):
    """(container, first audio stream) of `src_path`; the caller closes the
    container. ValueError when there is no audio stream, or when that stream
    is in a channel order this PyAV cannot release safely
    (audio_ffmpeg.CUSTOM_CHANNEL_ORDER). `timeout` is _open_client_file's:
    for a caller that only inspects the open, not one that decodes."""
    # The source is an uploaded clip (or a downloaded link) whose bytes AND
    # filename extension a client chose, so a crafted concat/ffconcat, HLS
    # playlist or SDP input can coax the demuxer into following external
    # references — the classic ffmpeg local-file-read / SSRF surface. Two
    # locks, both needed: the file-protocol whitelist stops http:// (and
    # every other network protocol) — streaming/transport.py pins
    # "-protocol_whitelist pipe" on the realtime path for the same reason —
    # and the demuxer allowlist (_open_client_file) stops a playlist reading
    # a sibling local file through the whitelisted file protocol itself. The
    # refusal by name below backs the allowlist up.
    container = _open_client_file(_av(), src_path, timeout=timeout)
    if audio_ffmpeg.is_multi_input_format(getattr(container.format, "name", None)):
        container.close()
        raise ValueError("unsupported container")
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        raise ValueError("source has no audio stream")
    # Before anything reads the stream's channel count or decodes a frame:
    # either would release a layout wrapper, and on a custom channel order
    # that ends the process (audio_ffmpeg.holds_custom_channel_map).
    context = stream.codec_context
    if context is not None and audio_ffmpeg.holds_custom_channel_map(context.layout):
        container.close()
        raise ValueError(audio_ffmpeg.CUSTOM_CHANNEL_ORDER)
    return container, stream


def declared_shape(src_path: str) -> "tuple[float | None, int | None, int | None]":
    """(seconds, sample rate, channels) the container (or else its audio
    stream) declares for `src_path`; None for any it does not declare — PyAV
    reports 0 channels for a WAV of more than 64. Header data only: nothing
    is decoded. Raises what _open_audio raises."""
    said = _helped("declared_shape", {"src": src_path}, opens=True,
                   timeout=_OPEN_OP_TIMEOUT_S)
    if said is _LOCAL:
        return _declared_shape(src_path)
    if not (isinstance(said, list) and len(said) == 3):
        raise _unreadable_answer()
    secs, rate, channels = said
    return (float(secs) if _is_measure(secs) else None,
            int(rate) if _is_measure(rate) else None,
            int(channels) if _is_measure(channels) else None)


def _is_measure(value) -> bool:
    """Whether a helper's answer is a positive number of ordinary size (a
    duration, a rate, a channel count)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and 0 < value < 1e15)


def _declared_shape(src_path: str):
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
    said = _helped("is_pcm_wav_or_flac", {"src": src_path}, opens=True,
                   timeout=_OPEN_OP_TIMEOUT_S)
    return _is_pcm_wav_or_flac(src_path) if said is _LOCAL else said is True


def _is_pcm_wav_or_flac(src_path: str) -> bool:
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


# The BGM separator loads its whole input as float32 44.1 kHz stereo; its RAM
# cap is counted in the bytes of that audio as s16 (main._sep_ram_cap).
SEP_S16_BYTES_PER_S = 44100 * 2 * 2


def separator_bytes_per_s(rate: int, channels: int) -> float:
    """What a second of `rate`/`channels` audio handed over as is costs the
    separator, in the cap's 44.1 kHz stereo s16 bytes. It reads the file
    whole at its native rate and channel count before resampling, so more
    channels or a higher rate scale the charge; mono or stereo at 48 kHz or
    below costs what the transcoded WAV would."""
    return (SEP_S16_BYTES_PER_S * max(1.0, channels / 2)
            * max(1.0, rate / 48000))


# The cost functions decoded_cost_within takes by name when it runs in a
# helper: a function of the caller's does not exist there.
COST_FUNCTIONS = {"separator": separator_bytes_per_s}


def decoded_cost_within(src_path: str, *, cost_per_s, max_cost: float) -> bool:
    """Whether `src_path`, decoded to its end, costs at most `max_cost`: each
    frame is charged its duration times cost_per_s(rate, channels) at its own
    shape. Nothing is written; False as soon as the sum passes `max_cost`, or
    on a frame without a channel count, or when the decode stops short of the
    end (see _decoded_frames): what lies behind was not measured, and another
    decoder may read it. Unlike declared_shape this does not trust the
    header, which the uploader wrote: a Matroska Segment Duration can declare
    a second for hours of audio. Raises what the decode raises. While decode
    helpers run `cost_per_s` must be one of COST_FUNCTIONS (TypeError)."""
    if _decode_helper.routed():
        name = next((n for n, f in COST_FUNCTIONS.items() if f is cost_per_s), None)
        if name is None:
            raise TypeError("decoded_cost_within: cost_per_s is not one of "
                            "COST_FUNCTIONS, which a decode helper needs")
        said = _helped("decoded_cost_within",
                       {"src": src_path, "cost": name, "max_cost": max_cost},
                       long=True)
        if said is not _LOCAL:
            return said is True
    return _decoded_cost_within(src_path, cost_per_s=cost_per_s,
                                max_cost=max_cost)


def _decoded_cost_within(src_path: str, *, cost_per_s, max_cost: float) -> bool:
    container, stream = _open_audio(src_path)
    try:
        cost, state = 0.0, {}
        for frame in _decoded_frames(container, stream, state):
            channels = len(frame.layout.channels)
            if (channels <= 0 or not frame.sample_rate
                    or _frame_too_big(frame)):
                return False
            cost += (frame.samples / frame.sample_rate
                     * cost_per_s(frame.sample_rate, channels))
            if cost > max_cost:
                return False
        return not state.get("cut")
    finally:
        container.close()


def _frameless_limit(container) -> int:
    """The packet bytes a decode of `container` may demux without getting a
    frame for them: the file's size plus _SKIPPED_PACKETS_MARGIN."""
    size = container.size
    if size <= 0:
        try:
            size = os.path.getsize(container.name)
        except OSError:
            size = 0
    return size + _SKIPPED_PACKETS_MARGIN


def _fw_decoded_frames(container, stream):
    """container.decode(stream), packet by packet as PyAV runs it, ending as
    at the end of the file once the packets that decoded to no frame add up
    to more than _frameless_limit (see _decoded_frames): no duration ceiling
    counts them. An InvalidDataError still ends the generator, as it ends
    PyAV's, which is what _fw_valid_frames relies on."""
    limit, frameless = _frameless_limit(container), 0
    for packet in container.demux(stream):
        frames = packet.decode()
        if not frames:
            frameless += packet.size
            if frameless > limit:
                return
            continue
        yield from frames


def _fw_valid_frames(frames):
    """faster_whisper.audio._ignore_invalid_frames, verbatim: an
    InvalidDataError ends the decode generator (PyAV's, and
    _fw_decoded_frames alike), so the decode stops at the first invalid
    packet (unlike _decoded_frames, which skips it)."""
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except _av().error.InvalidDataError:
            continue


def _fw_bounded_groups(frames, num_samples: int, max_seconds: "float | None"):
    """faster_whisper.audio._group_frames plus the duration ceiling. Each
    frame is charged its own length as it leaves the decoder, BEFORE the
    group reaches the resampler: one 500 000-sample group of a stream that
    declares a 1 Hz rate would resample to ~10^10 samples in a single call.
    For the same reason a group is also handed on once what it resamples
    to (16 kHz output) reaches `num_samples`, or once it holds 2 x
    `num_samples` sample-channels (the stereo-equivalent), so a 254-channel
    stream meets the resampler on a ~4 MB group instead of ~0.5 GB; and a
    single frame past _MAX_FRAME_S, or past _MAX_FRAME_SAMPLES before the
    fifo copies it, is refused, and so is a frame with fewer channels than
    the first (ValueError, the fifo's own for a changed rate or format).
    The resampler streams, so where a group ends changes no output
    sample."""
    fifo = _av().audio.fifo.AudioFifo()
    channels = None
    seconds = 0.0
    out = 0.0
    for frame in frames:
        if _frame_too_big(frame):
            raise DecodeTooLong(
                "a decoded frame is larger than the server resamples")
        if frame.sample_rate:
            dur = frame.samples / frame.sample_rate
            if dur > _MAX_FRAME_S:
                raise DecodeTooLong(
                    f"a decoded frame runs past {_MAX_FRAME_S:.0f} s")
            if max_seconds is not None:
                seconds += dur
                if seconds > max_seconds:
                    raise DecodeTooLong(
                        f"audio runs past {max_seconds:.0f} s of decoded input")
            out += dur * 16000
        frame.pts = None  # Ignore timestamp check.
        # PyAV's fifo compares the format, rate and time base of a later
        # frame with its first, not the channel count: it copies as many
        # planes (or packed channels) as it was allocated with, so a frame
        # with fewer is read past its end. The allocation outlives every
        # fifo.read(); the fifo ignores an empty frame.
        if frame.samples:
            count = frame.layout.nb_channels
            if channels is None:
                channels = count
            elif count < channels:
                raise ValueError("Frame does not match AudioFifo parameters.")
        fifo.write(frame)
        if (fifo.samples >= num_samples or out >= num_samples
                or fifo.samples * len(frame.layout.channels)
                >= 2 * num_samples):
            yield fifo.read()
            out = 0.0
    if fifo.samples > 0:
        yield fifo.read()


def decode_16k_bounded(src_path: str, *, max_seconds: "float | None"):
    """faster-whisper's decode_audio(src_path, sampling_rate=16000), sample
    for sample, with a ceiling: DecodeTooLong once the input decoded so far
    runs past `max_seconds` (None: no ceiling), so a small, highly compressed
    upload is refused before it materialises as gigabytes of float32. Opens
    through _open_audio (file protocol only, playlists refused) and decodes
    its first audio stream, the one decode_audio's audio=0 picks.

    The steps mirror faster_whisper/audio.py 1.2.1 (s16 mono resampler,
    500 000-sample groups, flush, s16 -> float32 / 32768);
    tests/audio/test_decode_bounded.py compares the two on every format the
    server takes, so a faster-whisper upgrade that changes them shows up.

    In a decode helper the samples come back as they are resampled, so the
    helper holds one group whatever the file's length; MemoryError when it
    passed its budget all the same, DecoderEnded when it ended on the
    file."""
    import io

    import numpy as np

    raw_buffer = io.BytesIO()
    # What the helper may send: the ceiling's own samples and a margin for
    # the resampler's rounding. A helper is no more trusted with the
    # server's memory than the file it reads.
    most = (None if max_seconds is None
            else int(max_seconds * 16000 * 2 * 1.05) + (1 << 20))

    def _take(chunk) -> None:
        if most is not None and raw_buffer.tell() + len(chunk) > most:
            raise DecodeTooLong(
                f"audio runs past {max_seconds:.0f} s of decoded input")
        raw_buffer.write(chunk)

    said = _helped("decode_16k", {"src": src_path, "max_seconds": max_seconds},
                   long=True, on_data=_take)
    if said is not _LOCAL:
        got = raw_buffer.getbuffer()
        audio = np.frombuffer(got[:len(got) - len(got) % 2], dtype=np.int16)
        return audio.astype(np.float32) / 32768.0
    dtype = None

    def _keep(array) -> None:
        nonlocal dtype
        dtype = array.dtype
        raw_buffer.write(array)

    _decode_16k_frames(src_path, max_seconds, _keep)
    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    return audio.astype(np.float32) / 32768.0


def _decode_16k_frames(src_path: str, max_seconds: "float | None", sink) -> None:
    """decode_16k_bounded's decode: every resampled frame (s16 mono 16 kHz,
    as an ndarray) goes to `sink` in order."""
    import gc
    import itertools

    av = _av()
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=16000)
    container, stream = _open_audio(src_path)
    try:
        try:
            frames = _fw_valid_frames(_fw_decoded_frames(container, stream))
            frames = _fw_bounded_groups(frames, 500000, max_seconds)
            for group in itertools.chain(frames, [None]):
                try:
                    resampled = resampler.resample(group)
                except MemoryError:
                    raise  # the caller's own arm: no property of the group
                except Exception as e:
                    raise ResampleFailed(str(e)) from e
                for frame in resampled:
                    sink(frame.to_ndarray())
        except (DecodeTooLong, ResampleFailed, MemoryError):
            # MemoryError ahead of the wrap: PyAV's error for a libav ENOMEM
            # is an av error and a MemoryError, and the caller answers the
            # latter as "does not fit", not as a failed decode.
            raise
        except Exception as e:
            # Whatever its type: an error that is no av error (the fifo's)
            # left unwrapped reads to the caller as a failed open, which it
            # answers by handing the path to a decoder with no whitelist.
            raise DecodeFailed(str(e)) from e
    finally:
        container.close()
    # As in decode_audio: resampler objects are only freed by a collection.
    del resampler
    gc.collect()


def refuse_input(path: str) -> None:
    """ValueError when `path` opens but is no input for faster-whisper: a
    playlist container, or one with no audio stream (a video-only clip, a
    text file the tty demuxer opens) — decode_audio raises a bare IndexError
    on that, a 500. A file that does not open at all is not refused here:
    the real decoder raises the av error, answered as undecodable — unless
    FFmpeg's default open reads it as a playlist, reads it only without the
    stream limit, or finds audio in a channel order this PyAV cannot release
    safely (see below; _open_audio refuses that one itself). The opens share
    one time budget (audio_ffmpeg.HEADER_OPEN_BUDGET_S): a file that uses it
    up is refused as well. In a decode helper, so is a file whose opens pass
    the helper's memory budget (audio_ffmpeg.TOO_MUCH_TO_OPEN) or end the
    helper (audio_ffmpeg.STOPS_THE_DECODER)."""
    if _helped("refuse_input", {"path": path}, opens=True,
               timeout=_OPEN_OP_TIMEOUT_S) is _LOCAL:
        _refuse_input(path)


def _refuse_input(path: str) -> None:
    av = _av()   # caps libav's allocations: also for the reopens below
    # libavformat opens a decoder for every stream while it probes: the two
    # reopens below take audio decoders only, like the strict open. Omitted
    # when the list is empty, as an empty whitelist refuses every decoder.
    decoders = audio_ffmpeg.pyav_audio_decoders()
    audio_only = {"codec_whitelist": decoders} if decoders else {}
    # This runs on the one worker every upload's check waits for, and an
    # open that finds no header reads the whole file looking for one. Each
    # open gets what is left of the budget; PyAV stops its reads then, and
    # the open comes back with its ordinary error or — the stop read as the
    # end of the file — with a container, so only the clock tells. Either
    # way the check is unfinished: a refusal, never "left to the decoder",
    # which would read the same bytes again without a limit.
    deadline = time.monotonic() + audio_ffmpeg.HEADER_OPEN_BUDGET_S

    def _left() -> float:
        return max(0.1, deadline - time.monotonic())

    def _refuse_when_out_of_time(cause: "BaseException | None" = None) -> None:
        if time.monotonic() >= deadline:
            raise ValueError(audio_ffmpeg.TOO_SLOW_TO_OPEN) from cause

    try:
        container, _stream = _open_audio(path, timeout=_left())
    except av.FFmpegError as e:  # some (InvalidDataError) are ValueErrors: still "does not open", not a refusal
        _refuse_when_out_of_time(e)
        # The fallbacks — transcribe(path), pyannote, the separator — open
        # with FFmpeg's default nested whitelist (file,crypto,data), not the
        # file-only one: an HLS playlist decrypting a sibling segment fails
        # only the latter, and must not reach them.
        try:
            container = av.open(
                path, options={**audio_ffmpeg.pyav_stream_limit(), **audio_only},
                metadata_errors="ignore", timeout=(_left(), None))
        except av.FFmpegError as e:
            _refuse_when_out_of_time(e)
            # Those fallbacks open without the stream limit: a file that
            # runs into it here is refused, not left to them.
            if audio_ffmpeg.refused_as_too_much(e):
                raise ValueError(audio_ffmpeg.TOO_MUCH_TO_OPEN) from e
            # Not every demuxer answers the limit that way (wtv: invalid
            # data), so ask the fallbacks' question itself: does the file
            # open without it? No stream limit, but still no video decoder
            # for any of the streams it then lists.
            try:
                unlimited = av.open(path, options=dict(audio_only),
                                    metadata_errors="ignore",
                                    timeout=(_left(), None))
            except av.FFmpegError as unopened:
                _refuse_when_out_of_time(unopened)
                return      # opens for nobody: left to the decoder
            unlimited.close()
            _refuse_when_out_of_time(e)
            raise ValueError(audio_ffmpeg.TOO_MUCH_TO_OPEN) from e
        try:
            _refuse_when_out_of_time()
            if audio_ffmpeg.is_multi_input_format(
                    getattr(container.format, "name", None)):
                raise ValueError("unsupported container")
            # The fallback decoders release PyAV's layout wrappers like any
            # other reader: a custom channel order must not reach them.
            for stream in container.streams:
                context = stream.codec_context if stream.type == "audio" else None
                if context is not None and audio_ffmpeg.holds_custom_channel_map(
                        context.layout):
                    raise ValueError(audio_ffmpeg.CUSTOM_CHANNEL_ORDER)
        finally:
            container.close()
        return
    container.close()
    _refuse_when_out_of_time()


def refuse_multi_input(path: str) -> None:
    """ValueError("unsupported container") when `path` opens as a demuxer
    that pulls in OTHER inputs (see audio_ffmpeg.MULTI_INPUT_DEMUXERS), or
    as one off the allowlist for reading a whole text file at the open
    (audio_ffmpeg.WHOLE_FILE_TEXT_DEMUXERS).

    For paths handed to a decoder that does not go through _open_audio
    (faster-whisper's decode_audio, pyannote): run it on the raw upload
    first. No audio stream is required, and an av error is NOT refused —
    the same libavformat open in the real decoder raises it anyway, with
    that caller's own error handling — except the EINVAL of a demuxer
    outside the allowlist, which is the playlist (or the text file)
    itself."""
    av = _av()
    try:
        container = _open_client_file(av, path)
    except av.FFmpegError:
        return
    try:
        if audio_ffmpeg.is_multi_input_format(
                getattr(container.format, "name", None)):
            raise ValueError("unsupported container")
    finally:
        container.close()


def _decoded_frames(container, stream, state: "dict | None" = None):
    """The audio frames of `stream` from where the container stands, skipping
    a packet the decoder rejects as invalid data. One damaged frame in an
    otherwise playable file (a bad mp3 frame, a flac block with a broken
    CRC) must not abort the whole decode: faster-whisper's own decode_audio
    tolerates it, so the file transcribes. Any other decode error stays a
    PyAV FFmpegError (a decode failure: the caller falls back), NOT the
    plain ValueError main.py treats as a refused input.

    The frames end, as at the end of the file, once the packets that gave
    no frame add up to more than the file's size plus
    _SKIPPED_PACKETS_MARGIN, and state["cut"] is set: a sample table can
    name one byte range any number of times at four bytes each, and a
    rejected packet — or one the decoder accepts and turns into no frame
    (FFmpeg's flac decoder takes any packet that starts like a metadata
    block that way) — is demuxed in full but charged to none of the
    budgets, which all count decoded frames."""
    invalid = _av().error.InvalidDataError
    limit, skipped = _frameless_limit(container), 0
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except invalid:
            frames = None
        if not frames:
            skipped += packet.size
            if skipped > limit:
                if state is not None:
                    state["cut"] = True
                return
            continue
        yield from frames


def _resampled(frames, *, format: str, layout: str, rate: int):
    """`frames` resampled to `format`/`layout`/`rate`, flush included. A
    resampler is built for the first frame's input format and raises a plain
    ValueError on a frame that differs (a concatenated mp3, a TS recording
    switching 5.1 and stereo) — read as a refused input upstream — so a new
    one replaces it, after flushing, whenever the input changes. Output pts
    count samples across resamplers: each new one would restart at 0 and the
    muxer refuses timestamps that go backwards. A stream that changes more
    often than its allowance (see _MAX_SETUP_CHANGES) is refused with the
    same plain ValueError."""
    from fractions import Fraction

    av = _av()
    resampler, setup, done = None, None, 0
    changes, seen = 0, 0

    def _out(resampled):
        nonlocal done
        for out in resampled:
            out.pts, out.time_base = done, Fraction(1, rate)
            done += out.samples
            yield out

    for frame in frames:
        seen += 1
        # Checked before the resampler sees the frame: it resamples the
        # whole frame in one call (see _MAX_FRAME_S, _MAX_FRAME_SAMPLES).
        if frame.sample_rate and frame.samples / frame.sample_rate > _MAX_FRAME_S:
            raise ValueError(
                f"unsupported audio: a decoded frame runs past "
                f"{_MAX_FRAME_S:.0f} s")
        if _frame_too_big(frame):
            raise ValueError("unsupported audio: a decoded frame is larger "
                             "than the server resamples")
        frame_setup = (frame.format.name, frame.layout.name, frame.sample_rate)
        if frame_setup != setup:
            if resampler is not None:
                # The first build is no change. The rate weighs the change:
                # building a resampler costs more the higher its input rate.
                changes += max(1, (frame.sample_rate or 0) // 48000)
                if changes > (_MAX_SETUP_CHANGES
                              + seen // _FRAMES_PER_SETUP_CHANGE):
                    raise ValueError("unsupported audio: the stream changes "
                                     "format too often")
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


# _drop_before keeps every frame that ends within _SKIP_GUARD_S of the cut,
# so the resampler's filter has settled on real input again by the first
# kept sample; it holds at most _SKIP_HOLD_MAX_S (or _SKIP_HOLD_MAX_BYTES,
# or _SKIP_HOLD_MAX_FRAMES: a decoded frame object costs ~1 KB whatever its
# sample count) of frames while looking for a point on the 16 kHz grid.
# Where dropping stops short of the cut (a format change), the rest of the
# skip is resampled up to _SKIP_RESAMPLE_MAX of work (input samples x
# channels x max(1, 16000 / rate), plus _SKIP_SETUP_CHANGE_COST per change of
# format, layout or rate): ~380 s of 44.1 kHz stereo. A change makes
# _resampled flush and build a new resampler, ~1 ms or ~58k units of steady
# resampling; uncharged, a stream that alternates its rate every frame would
# cost ~70x the budget in rebuilds.
_SKIP_GUARD_S = 1.0
_SKIP_HOLD_MAX_S = 60.0
_SKIP_HOLD_MAX_BYTES = 64 << 20
_SKIP_HOLD_MAX_FRAMES = 16384
_SKIP_RESAMPLE_MAX = 1 << 25
_SKIP_SETUP_CHANGE_COST = 1 << 16

# What one _read_16k call decodes only to throw away ahead of its piece: the
# frames between where a seek lands and `after` (_drop_ahead) plus the ones
# _drop_before drops of a skip — _DISCARD_MAX_FRAMES frames or
# _DISCARD_MAX_SAMPLES samples x channels, whichever comes first, then the
# piece is refused. Nothing else charges them, and the file decides how many
# there are: a sample table that marks only its first sample as a sync
# sample sends every seek back to the start, a playlist can declare any
# skip, and a table naming one frame a million times costs four bytes each.
# Counted in decode work, not seconds: a second is whatever rate the file
# declares, and a long skip is no fault (three hours of 1.1 kHz FLAC is 181
# frames). A real seek lands within a packet or one Matroska cluster of its
# piece (4.6 s / 191 frames at worst over 32 formats, 33 s by the format's
# own limit), and a real skip ends inside its first segment, whose 2 MiB
# hold under 140 000 frames and 2^27 samples even as 6 kbit/s Opus. The
# budget is 93 min of 48 kHz mono AAC, and a few seconds of one thread:
# 1-3 s for 2^18 of the cheapest frames, 2-9 s for 2^28 samples of the
# slowest lossless decoders.
_DISCARD_MAX_FRAMES = 1 << 18
_DISCARD_MAX_SAMPLES = 1 << 28


def _charge_discard(state: dict, frames: int, samples: int) -> None:
    """Add `frames` frames of `samples` samples x channels to what one
    _read_16k call (`state`) threw away ahead of its piece; ValueError — a
    refused piece, as in _charged_until_cut — past the budget."""
    state["discard_frames"] = state.get("discard_frames", 0) + frames
    state["discard_samples"] = state.get("discard_samples", 0) + samples
    if (state["discard_frames"] > _DISCARD_MAX_FRAMES
            or state["discard_samples"] > _DISCARD_MAX_SAMPLES):
        raise ValueError("unsupported audio: too much audio ahead of the "
                         "piece")


def _drop_ahead(frames, after: float, state: dict):
    """`frames` without those that end before `after` (seconds, stream
    time); a frame without a timestamp is kept. Each dropped frame is
    charged to the call's discard budget (see _DISCARD_MAX_FRAMES)."""
    for f in frames:
        if f.time is None or f.time + f.samples / f.sample_rate >= after:
            yield f
        else:
            _charge_discard(state, 1, f.samples * len(f.layout.channels))


def _drop_before(frames, skip_out: int, state: dict):
    """`frames` without whole frames that end more than _SKIP_GUARD_S ahead
    of output sample `skip_out` (16 kHz), so the samples a skip discards are
    decoded but never resampled: the skip of a link's language-check piece
    comes from a remote playlist, and a few KB of FLAC at an odd rate
    resamples ~50x slower than it decodes. state["n"] / state["rate"] count
    the input samples dropped.

    The output stays sample-for-sample what resampling everything gives:
    frames are dropped only in runs that keep the 16 kHz output on the same
    grid (n * 16000 % rate == 0). The first frame always passes (an MP3's
    is trimmed to the encoder delay, off every grid point), and dropping
    stops at the first frame whose format, layout or rate differs (the
    resampler restarts there) or that runs past _MAX_FRAME_S or
    _MAX_FRAME_SAMPLES (_resampled refuses it). After a format change the
    rest of the skip is resampled up to _SKIP_RESAMPLE_MAX of work, past
    which the piece is refused (ValueError; the language check then
    downloads the whole file). A run that meets no grid point within
    _SKIP_HOLD_MAX_S, _SKIP_HOLD_MAX_BYTES or _SKIP_HOLD_MAX_FRAMES (only a
    crafted rate or block size) is dropped off the grid anyway. What is
    dropped is charged to the discard budget `state` carries (see
    _DISCARD_MAX_FRAMES), past which the piece is refused the same way."""
    skip_s = skip_out / 16000
    it = iter(frames)
    first = next(it, None)
    if first is None:
        return
    yield first
    setup = (first.format.name, first.layout.name, first.sample_rate)
    channels = len(first.layout.channels)   # of every frame dropped below
    seen, held, held_n, held_bytes = first.samples, [], 0, 0
    for f in it:
        r = f.sample_rate
        changed = (f.format.name, f.layout.name, r) != setup
        if (changed or not r or f.samples / r > _MAX_FRAME_S
                or _frame_too_big(f)
                or (seen + held_n + f.samples) / r >= skip_s - _SKIP_GUARD_S):
            yield from held
            if changed:
                at = (seen + held_n) / setup[2] if setup[2] else 0.0
                yield from _charged_until_cut(f, it, at, skip_s)
                return
            yield f
            yield from it
            return
        held.append(f)
        held_n += f.samples
        held_bytes += f.samples * len(f.layout.channels) * f.format.bytes
        if ((state["n"] + held_n) * 16000 % r == 0
                or held_n / r > _SKIP_HOLD_MAX_S
                or held_bytes > _SKIP_HOLD_MAX_BYTES
                or len(held) >= _SKIP_HOLD_MAX_FRAMES):
            _charge_discard(state, len(held), held_n * channels)
            state["n"] += held_n
            state["rate"] = r
            seen += held_n
            held, held_n, held_bytes = [], 0, 0
    yield from held


def _charged_until_cut(f, it, at: float, skip_s: float):
    """`f` and the rest of `it`, for a skip _drop_before stopped dropping
    `at` seconds in, short of the cut: the frames before _SKIP_GUARD_S
    ahead of it are resampled only to be discarded, so their resample work
    is charged and past _SKIP_RESAMPLE_MAX the piece is refused. Each
    change of format, layout or rate, `f`'s own included, adds
    _SKIP_SETUP_CHANGE_COST: _resampled rebuilds its resampler there. A
    frame without a rate, or over _MAX_FRAME_S or _MAX_FRAME_SAMPLES, ends
    the charge: _resampled handles it as before."""
    work, setup = 0.0, None
    while f is not None:
        r = f.sample_rate
        if (not r or f.samples / r > _MAX_FRAME_S or _frame_too_big(f)
                or at >= skip_s - _SKIP_GUARD_S):
            break
        frame_setup = (f.format.name, f.layout.name, r)
        if frame_setup != setup:
            work += _SKIP_SETUP_CHANGE_COST
            setup = frame_setup
        work += f.samples * len(f.layout.channels) * max(1.0, 16000 / r)
        if work > _SKIP_RESAMPLE_MAX:
            raise ValueError("unsupported audio: the skipped part changes "
                             "format")
        at += f.samples / r
        yield f
        f = next(it, None)
    if f is not None:
        yield f
        yield from it


def _read_16k(container, stream, want: int, *, skip: int = 0,
              after: "float | None" = None):
    """Decode `stream` from where the container stands into 16 kHz mono
    float32: drop frames that end before `after` (seconds, STREAM time —
    media time plus the stream's start_time) and then the first `skip`
    samples, return the next `want` (fewer at EOF). ValueError once the
    frames decoded only to be dropped pass the discard budget (see
    _DISCARD_MAX_FRAMES), which the two kinds of drop share."""
    import numpy as np

    state = {"n": 0, "rate": None}
    frames = _decoded_frames(container, stream)
    if after is not None:
        # A seek lands on the keyframe at or before `after`.
        frames = _drop_ahead(frames, after, state)
    if skip:
        frames = _drop_before(frames, skip, state)
    # The skipped samples are dropped as they decode, never collected: the
    # skip of a link's language-check piece comes from a remote playlist, so
    # holding it would let a tiny body ask for gigabytes.
    chunks, have, consumed = [], 0, 0
    # Fresh resampler per call: it keeps samples buffered across a seek.
    for out in _resampled(frames, format="flt", layout="mono", rate=16000):
        chunk = out.to_ndarray().reshape(-1)
        # What _drop_before took out never reached the resampler.
        dropped = (state["n"] * 16000 // state["rate"]) if state["rate"] else 0
        remaining = skip - dropped - consumed
        if remaining > 0:
            cut = min(remaining, chunk.size)
            consumed += cut
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
    back empty. ValueError when a seek lands so far ahead of its piece that
    reaching it passes the discard budget (see _DISCARD_MAX_FRAMES)."""
    starts = [float(start) for start in starts]
    pieces, take = _piece_taker(len(starts), seconds)
    said = _helped(
        "decode_pieces_16k",
        {"src": src_path, "starts": starts, "seconds": float(seconds)},
        timeout=_PIECE_OP_TIMEOUT_S + _PIECE_OP_TIMEOUT_EACH_S * len(starts),
        on_data=take)
    if said is _LOCAL:
        return _decode_pieces_16k(src_path, starts, seconds)
    if len(pieces) != len(starts):
        raise _unreadable_answer()
    return pieces


def _piece_taker(count: int, seconds: float):
    """([], take) — `take` appends each piece a helper sends (float32
    sample bytes) to the list as an array of its own, and refuses more
    pieces than `count` or one longer than `seconds`."""
    import numpy as np

    pieces: list = []
    most = int(seconds * 16000) * 4

    def take(chunk) -> None:
        if len(pieces) >= count or len(chunk) > most or len(chunk) % 4:
            raise _unreadable_answer()
        pieces.append(np.frombuffer(chunk, dtype=np.float32).copy())

    return pieces, take


def _decode_pieces_16k(src_path: str, starts: "list[float]",
                       seconds: float) -> list:
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
    nor need be continuous. ValueError when the skip takes more decoding
    than the discard budget (see _DISCARD_MAX_FRAMES)."""
    pieces, take = _piece_taker(1, seconds)
    said = _helped(
        "decode_span_16k",
        {"src": src_path, "skip": float(skip), "seconds": float(seconds)},
        timeout=_PIECE_OP_TIMEOUT_S + _PIECE_OP_TIMEOUT_EACH_S, on_data=take)
    if said is _LOCAL:
        return _decode_span_16k(src_path, skip, seconds)
    if len(pieces) != 1:
        raise _unreadable_answer()
    return pieces[0]


def _decode_span_16k(src_path: str, skip: float, seconds: float):
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


def transcode_to_wav_16k_mono(src_path: str, dst_path: str, *,
                              max_bytes: "int | None" = None,
                              stop_at_bytes: "int | None" = None) -> int:
    return transcode_to_wav(src_path, dst_path, rate=16000, layout="mono",
                            max_bytes=max_bytes, stop_at_bytes=stop_at_bytes)


def transcode_to_wav(src_path: str, dst_path: str, *,
                     rate: int, layout: str,
                     max_bytes: "int | None" = None,
                     stop_at_bytes: "int | None" = None) -> int:
    """Decode anything PyAV understands, resample to `rate`/`layout`, write
    a RIFF/WAVE file at dst_path. Returns bytes written. On any failure
    the destination is best-effort unlinked.

    `max_bytes` bounds the PCM written: a few hundred KB of low-bitrate
    Opus/MP3 expands several hundred times into 44.1 kHz stereo, so a
    caller writing to a shared temp dir passes its free space and the
    transcode aborts (TranscodeBudgetExceeded) before it fills the disk.

    `stop_at_bytes` ends the WAV instead, a valid file of at most that much
    PCM: for a copy of what another decode already read. That decode
    (faster-whisper's) stops at an mp3's first invalid packet, while this
    one skips the packet and reads on, possibly for hours of audio."""
    try:
        said = _helped(
            "transcode_to_wav",
            {"src": src_path, "dst": dst_path, "rate": rate, "layout": layout,
             "max_bytes": max_bytes, "stop_at_bytes": stop_at_bytes},
            long=True)
    except BaseException:
        # The helper unlinks what it wrote when the transcode fails; one
        # that was killed, or ended on the file, did not get to.
        store_common.unlink_quietly(dst_path)
        raise
    if said is _LOCAL:
        return _transcode_to_wav(src_path, dst_path, rate=rate, layout=layout,
                                 max_bytes=max_bytes,
                                 stop_at_bytes=stop_at_bytes)
    return os.path.getsize(dst_path)


def _transcode_to_wav(src_path: str, dst_path: str, *,
                      rate: int, layout: str,
                      max_bytes: "int | None" = None,
                      stop_at_bytes: "int | None" = None) -> int:
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
            if stop_at_bytes is not None and written > stop_at_bytes:
                break
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


# --------------------------------------------------------------------------
# The decode helper's side (audio/decode_helper.py)
# --------------------------------------------------------------------------

def helper_warm() -> None:
    """Load PyAV and numpy, cap libav's allocations and fill the demuxer and
    decoder lists: a helper does it once at its start, not inside its first
    request's memory budget."""
    import numpy  # noqa: F401, PLC0415 - loaded for the operations below

    _av()
    audio_ffmpeg.single_input_demuxers()
    audio_ffmpeg.pyav_audio_decoders()


def _op_decode_16k(send, *, src, max_seconds):
    _decode_16k_frames(src, max_seconds, lambda array: send(array.tobytes()))


def _op_decode_pieces_16k(send, *, src, starts, seconds):
    for piece in _decode_pieces_16k(src, starts, seconds):
        send(piece.astype("float32", copy=False).tobytes())


def _op_decode_span_16k(send, *, src, skip, seconds):
    send(_decode_span_16k(src, skip, seconds).astype(
        "float32", copy=False).tobytes())


# Operation name -> handler(send, **args): the work of the public function
# of that name, run in a helper. `send` takes sample bytes for the parent;
# the return value is the answer (JSON scalars).
HELPER_OPS = {
    "refuse_input": lambda send, *, path: _refuse_input(path),
    "declared_shape": lambda send, *, src: list(_declared_shape(src)),
    "is_pcm_wav_or_flac": lambda send, *, src: bool(_is_pcm_wav_or_flac(src)),
    "decoded_cost_within": lambda send, *, src, cost, max_cost: bool(
        _decoded_cost_within(src, cost_per_s=COST_FUNCTIONS[cost],
                             max_cost=max_cost)),
    "decode_16k": _op_decode_16k,
    "decode_pieces_16k": _op_decode_pieces_16k,
    "decode_span_16k": _op_decode_span_16k,
    "transcode_to_wav": lambda send, *, src, dst, rate, layout, max_bytes,
    stop_at_bytes: _transcode_to_wav(
        src, dst, rate=rate, layout=layout, max_bytes=max_bytes,
        stop_at_bytes=stop_at_bytes),
}
