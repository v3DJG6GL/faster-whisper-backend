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
_open_audio.
"""
from __future__ import annotations

import os

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg

_OUT_FORMAT = "s16"          # signed 16-bit
_OUT_CODEC = "pcm_s16le"     # WAV's native uncompressed codec


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
    container = _av().open(src_path, options={"protocol_whitelist": "file"})
    if audio_ffmpeg.is_multi_input_format(getattr(container.format, "name", None)):
        container.close()
        raise ValueError("unsupported container")
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        raise ValueError("source has no audio stream")
    return container, stream


def _read_16k(container, stream, want: int, *, skip: int = 0,
              after: "float | None" = None):
    """Decode `stream` from where the container stands into 16 kHz mono
    float32: drop frames that end before `after` (seconds, stream time) and
    then the first `skip` samples, return the next `want` (fewer at EOF)."""
    import numpy as np

    # Fresh per call: a resampler keeps samples buffered across a seek.
    resampler = _av().AudioResampler(format="flt", layout="mono", rate=16000)
    chunks, have = [], 0
    for frame in container.decode(stream):
        # A seek lands on the keyframe at or before `after`.
        if (after is not None and frame.time is not None
                and frame.time + frame.samples / frame.sample_rate < after):
            continue
        frame.pts = None
        for out in resampler.resample(frame):
            chunk = out.to_ndarray().reshape(-1)
            chunks.append(chunk)
            have += chunk.size
        if have >= skip + want:
            break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[skip:skip + want]


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
        for start in starts:
            container.seek(int(start / stream.time_base), stream=stream)
            pieces.append(_read_16k(container, stream, want, after=start))
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


def transcode_to_wav_16k_mono(src_path: str, dst_path: str) -> int:
    return transcode_to_wav(src_path, dst_path, rate=16000, layout="mono")


def transcode_to_wav(src_path: str, dst_path: str, *,
                     rate: int, layout: str) -> int:
    """Decode anything PyAV understands, resample to `rate`/`layout`, write
    a RIFF/WAVE file at dst_path. Returns bytes written. On any failure
    the destination is best-effort unlinked."""
    av = _av()
    in_container = None
    out_container = None
    try:
        in_container, in_stream = _open_audio(src_path)

        out_container = av.open(dst_path, mode="w", format="wav")
        out_stream = out_container.add_stream(_OUT_CODEC, rate=rate)
        out_stream.layout = layout
        out_stream.format = _OUT_FORMAT

        resampler = av.AudioResampler(
            format=_OUT_FORMAT, layout=layout, rate=rate,
        )

        for frame in in_container.decode(in_stream):
            # PyAV recomputes pts when None; the input frame's pts is on
            # the input timebase and would corrupt the output otherwise.
            frame.pts = None
            for resampled in resampler.resample(frame):
                for packet in out_stream.encode(resampled):
                    out_container.mux(packet)

        # Flush resampler and encoder.
        for resampled in resampler.resample(None):
            for packet in out_stream.encode(resampled):
                out_container.mux(packet)
        for packet in out_stream.encode(None):
            out_container.mux(packet)

    except Exception:
        # Release the output handle before unlink (Windows holds the
        # lock otherwise).
        try:
            if out_container is not None:
                out_container.close()
                out_container = None
        except Exception:
            pass
        try:
            if os.path.exists(dst_path):
                os.unlink(dst_path)
        except OSError:
            pass
        raise
    finally:
        try:
            if out_container is not None:
                out_container.close()
        except Exception:
            pass
        try:
            if in_container is not None:
                in_container.close()
        except Exception:
            pass

    return os.path.getsize(dst_path)
