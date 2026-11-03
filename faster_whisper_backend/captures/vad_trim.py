"""Silence-trim capture audio using the Silero VAD that ships with
faster-whisper.

trim_pcm_for_merge is the sample path: with
cfg.CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES, merge_wavs runs it on every member
before concatenation. It drops the outer silence and caps internal gaps, so
all silence in a sample is uniform. trim_wav (whole-file leading/trailing
trim) has no production caller; only the tests exercise it.

Output keeps the captures/merge.py format (16 kHz mono signed-16-bit PCM
RIFF/WAVE), so a Whisper fine-tune loader sees the same audio shape for
trimmed and untrimmed samples. Calm-Whisper (arXiv:2505.12969) documents
long silence as a hallucination trigger during fine-tune; this is the
captures-side mitigation.
"""
from __future__ import annotations

import logging
import os
import tempfile
import wave

from faster_whisper_backend.captures import merge as audio_merge
from faster_whisper_backend.core import store_common

logger = logging.getLogger("whisper-api")

_REQ_RATE = 16000


def trim_wav(
    src_path: str,
    dst_path: str,
    *,
    margin_ms: int = 300,
    threshold: float = 0.5,
) -> dict:
    """Trim leading/trailing silence from src_path, write to dst_path.

    Returns a dict with the trim outcome — callers need both the success
    flag AND the offsets to keep stored word-level timestamps in sync
    with the trimmed audio:

      {
        "trimmed":           bool,   # True iff dst was written
        "lead_ms":           int,    # ms cut from front (0 if not trimmed)
        "trail_ms":          int,    # ms cut from back  (0 if not trimmed)
        "orig_duration_ms":  int,    # original WAV duration in ms
        "new_duration_ms":   int,    # post-trim duration (equals orig when
                                     # trimmed=False)
      }

    `margin_ms` is the silence preserved on each side of detected speech
    (default 300 ms — matches the merge inter-segment gap so a trimmed
    sample sounds natural). `threshold` is Silero VAD's speech-probability
    cut-off; 0.5 is the library default.

    Format requirements: src must be 16 kHz mono signed-16-bit PCM
    (audio_merge.read_pcm raises otherwise); dst is written in the same format.

    Failure modes:
      - import of faster_whisper.vad fails → log warning, return a dict
        with trimmed=False (caller should treat this as "trim
        unavailable" and use src as-is).
      - VAD finds no speech (silent clip) → trimmed=False, no write.
      - Source/dest IO error → propagated.

    Atomic write: writes to a mkstemp "<dst>.*.tmp" beside dst_path and
    os.replace.
    """
    def _no_trim(orig_ms: int) -> dict:
        return {
            "trimmed": False, "lead_ms": 0, "trail_ms": 0,
            "orig_duration_ms": orig_ms,
            "new_duration_ms": orig_ms,
        }

    # Lazy import — VAD pulls numpy + the Silero model on first call;
    # we don't want the module to be unimportable on hosts without
    # faster-whisper installed (e.g. CI lint).
    try:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as e:
        logger.warning(
            "[vad-trim] faster_whisper.vad unavailable (%s); skipping trim", e,
        )
        return _no_trim(0)

    pcm, n_samples = audio_merge.read_pcm(src_path)
    orig_duration_ms = _samples_to_ms(n_samples)
    if n_samples == 0:
        return _no_trim(0)

    # Silero VAD expects normalised float audio in [-1, 1].
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # Use a relatively tight min_silence_duration to detect genuine
    # leading/trailing silence; speech_pad_ms here is the VAD's internal
    # pad (not the same as our margin) — keep it small so VAD reports
    # tight boundaries and we apply our margin ourselves.
    opts = VadOptions(
        threshold=float(threshold),
        min_silence_duration_ms=200,
        speech_pad_ms=0,
        min_speech_duration_ms=0,
    )
    speeches = get_speech_timestamps(audio, opts, sampling_rate=_REQ_RATE)
    if not speeches:
        logger.info(
            "[vad-trim] no speech detected in %s; skipping",
            os.path.basename(src_path),
        )
        return _no_trim(orig_duration_ms)

    first_start = int(speeches[0]["start"])
    last_end = int(speeches[-1]["end"])
    margin_samples = int(margin_ms * _REQ_RATE / 1000)
    start_sample = max(0, first_start - margin_samples)
    end_sample = min(n_samples, last_end + margin_samples)
    if start_sample >= end_sample:
        return _no_trim(orig_duration_ms)

    # Bail out if the trim wouldn't actually save anything meaningful
    # (under ~50 ms in total) AND we're overwriting in place.
    # When dst != src we still write so the caller can rely on dst
    # existing.
    leading_trimmed = start_sample
    trailing_trimmed = n_samples - end_sample
    if (leading_trimmed + trailing_trimmed) < int(_REQ_RATE * 0.05):
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            return _no_trim(orig_duration_ms)

    out_bytes = pcm[start_sample * audio_merge.BYTES_PER_SAMPLE:
                    end_sample * audio_merge.BYTES_PER_SAMPLE]

    dst_dir = os.path.dirname(dst_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    # Same as audio_merge.merge_wavs: a fresh 0600 mkstemp tmp (O_EXCL, random
    # name) so the PCM is never world-readable while wave.open writes it and
    # a pre-existing "<dst>.tmp" inode is never reused.
    fd, tmp_path = tempfile.mkstemp(
        dir=dst_dir, prefix=os.path.basename(dst_path) + ".", suffix=".tmp")
    os.close(fd)
    try:
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(audio_merge._REQ_SAMPWIDTH_BYTES)
            w.setframerate(_REQ_RATE)
            w.writeframes(out_bytes)
        # Belt-and-braces, as in audio_merge.merge_wavs (mkstemp already made
        # the tmp 0600; this also covers Windows / foreign-umask oddities):
        # os.replace carries the tmp inode's mode onto dst_path. Trimmed
        # dictation audio is PHI; keep it owner-only across the swap.
        store_common.secure_file(tmp_path)
        try:
            with open(tmp_path, "rb") as fp:
                os.fsync(fp.fileno())
        except OSError:
            pass
        audio_merge.replace_with_retry(tmp_path, dst_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise

    lead_ms = int(leading_trimmed * 1000 / _REQ_RATE)
    trail_ms = int(trailing_trimmed * 1000 / _REQ_RATE)
    new_samples = end_sample - start_sample
    new_duration_ms = _samples_to_ms(new_samples)
    logger.info(
        "[vad-trim] %s: trimmed %d ms leading, %d ms trailing (margin=%d ms)",
        os.path.basename(src_path), lead_ms, trail_ms, margin_ms,
    )
    return {
        "trimmed": True,
        "lead_ms": lead_ms,
        "trail_ms": trail_ms,
        "orig_duration_ms": orig_duration_ms,
        "new_duration_ms": new_duration_ms,
    }


_samples_to_ms = audio_merge.samples_to_ms


def trim_pcm_for_merge(
    pcm: bytes,
    n_samples: int,
    *,
    edge_pad_ms: int = 50,
    max_internal_gap_ms: int = 300,
    threshold: float = 0.5,
) -> dict:
    """Plan + apply a per-member silence trim for the group merge path.

    Unlike `trim_wav` (outer leading/trailing only, keeps a margin), this
    removes a member's outer leading/trailing silence ENTIRELY (no outer pad —
    the merge layer owns the uniform outer margin), pads only the inner sides
    of internal speech spans by `edge_pad_ms`, AND collapses any internal
    silence longer than `max_internal_gap_ms` down to that cap. The result
    feeds `audio_merge.merge_wavs`, which then adds the uniform outer margin
    and joins trimmed members with the inter-segment gap — so all silence in
    the final clip is uniform and bounded (the multi-second dead air at member
    joins is removed).

    Operates on in-memory PCM only; the caller never touches the original
    capture file on disk (so dissolving the sample still restores raw audio).

    Returns:
      {
        "trimmed":          bool,        # True iff any silence was removed
        "pcm":              bytes,       # trimmed PCM (orig pcm when not trimmed)
        "new_n_samples":    int,
        "lead_ms":          int,         # leading silence removed (info)
        "new_duration_ms":  int,
        "segments": [[orig_start_ms, orig_end_ms, new_start_ms], ...],
      }

    `segments` is the piecewise time-map from original member time to trimmed
    member-local time — one entry per kept speech span. The route layer uses
    it to re-place per-word karaoke timestamps. When nothing is trimmed (VAD
    unavailable, no speech, or a clip already tight) the map is the identity
    span `[[0, dur_ms, 0]]` so callers can treat every member uniformly.
    """
    orig_duration_ms = _samples_to_ms(n_samples)

    def _identity() -> dict:
        return {
            "trimmed": False,
            "pcm": pcm,
            "new_n_samples": n_samples,
            "lead_ms": 0,
            "new_duration_ms": orig_duration_ms,
            "segments": [[0, orig_duration_ms, 0]],
        }

    if n_samples == 0:
        return _identity()

    try:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as e:
        logger.warning(
            "[vad-trim] faster_whisper.vad unavailable (%s); skipping "
            "per-member trim", e,
        )
        return _identity()

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    opts = VadOptions(
        threshold=float(threshold),
        min_silence_duration_ms=200,
        speech_pad_ms=0,
        min_speech_duration_ms=0,
    )
    speeches = get_speech_timestamps(audio, opts, sampling_rate=_REQ_RATE)
    if not speeches:
        return _identity()

    pad = max(0, int(edge_pad_ms)) * _REQ_RATE // 1000
    max_gap = max(0, int(max_internal_gap_ms)) * _REQ_RATE // 1000
    n_spans = len(speeches)

    bsps = audio_merge.BYTES_PER_SAMPLE
    out_chunks: list[bytes] = []
    segments: list[list[int]] = []
    new_pos = 0          # cursor in trimmed-output samples
    prev_kept_end = 0    # original-sample end of the previous kept span

    for idx, seg in enumerate(speeches):
        left_pad = 0 if idx == 0 else pad
        right_pad = 0 if idx == n_spans - 1 else pad
        s = max(0, int(seg["start"]) - left_pad)
        e = min(n_samples, int(seg["end"]) + right_pad)
        # Don't let this span's padded start swallow audio already emitted by
        # the previous span (overlapping pads on close segments).
        if s < prev_kept_end:
            s = prev_kept_end
        if e <= s:
            continue

        if not segments:
            gap_keep = 0
        else:
            orig_gap = s - prev_kept_end
            gap_keep = min(orig_gap, max_gap)
            if gap_keep > 0:
                out_chunks.append(pcm[prev_kept_end * bsps:
                                      (prev_kept_end + gap_keep) * bsps])
                new_pos += gap_keep

        seg_start_new = new_pos
        out_chunks.append(pcm[s * bsps:e * bsps])
        span_len = e - s
        new_pos += span_len
        segments.append([_samples_to_ms(s), _samples_to_ms(e),
                         _samples_to_ms(seg_start_new)])
        prev_kept_end = e

    if not segments:
        return _identity()

    out_pcm = b"".join(out_chunks)
    new_n = new_pos
    new_duration_ms = _samples_to_ms(new_n)
    lead_ms = _samples_to_ms(int(speeches[0]["start"]))

    # Nothing meaningfully removed (under ~50 ms total) → identity, so the
    # merge keeps the original PCM and a clean identity map.
    if (n_samples - new_n) < int(_REQ_RATE * 0.05):
        return _identity()

    logger.info(
        "[vad-trim] member: %d→%d ms over %d span(s) (edge_pad=%d, max_gap=%d)",
        orig_duration_ms, new_duration_ms, len(segments),
        edge_pad_ms, max_internal_gap_ms,
    )
    return {
        "trimmed": True,
        "pcm": out_pcm,
        "new_n_samples": new_n,
        "lead_ms": lead_ms,
        "new_duration_ms": new_duration_ms,
        "segments": segments,
    }
