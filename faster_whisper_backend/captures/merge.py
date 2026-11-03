"""In-process PCM-WAV concatenation utility for Whisper training-data
packing.

We pack one or more short same-speaker captures into a single WAV under
the configurable duration cap (cfg.CAPTURES_SAMPLE_MAX_DURATION_S,
default 29.9 s; Whisper's encoder hard-caps at 30 s, so the cap leaves a
safety margin). A single capture yields a trimmed one-member sample; 2+
get the silence-separated layout below. Per the
Low-Resource Whisper paper (arXiv 2412.15726) we insert a short silence
between segments to preserve the encoder's noise-floor model at the
joins; butt-splicing degrades transition quality.

Because every input is already 16 kHz mono signed-16-bit PCM (produced by
`audio/transcode.py` at capture time), we don't need PyAV here — pure
stdlib `wave` byte splicing is faster, smaller, and lossless. The
output is a valid RIFF/WAVE file every browser plays and `datasets.Audio()`
loads natively.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import wave
from hashlib import sha256

from faster_whisper_backend.core import store_common

logger = logging.getLogger("whisper-api")

# Required source format. Anything else gets rejected (defense-in-depth
# against an upstream pipeline change). The capture-time transcode is the
# only writer; if it ever produces a different format we want to know.
_REQ_CHANNELS = 1
_REQ_SAMPWIDTH_BYTES = 2     # signed 16-bit
_REQ_RATE = 16000

# Merged-WAV sample cap used only when config can't be read. The live cap
# is cfg.CAPTURES_SAMPLE_MAX_DURATION_S, re-read per merge by
# _max_merged_samples() so an admin change applies immediately.
MAX_MERGED_SAMPLES = 448_000  # 28 s × 16 kHz


def _max_merged_samples() -> int:
    """Live merged-WAV sample cap from config (samples = seconds × 16 kHz)."""
    try:
        from faster_whisper_backend.settings import config as cfg
        cap_s = float(getattr(cfg, "CAPTURES_SAMPLE_MAX_DURATION_S", 29.9))
        return int(cap_s * _REQ_RATE)
    except Exception:
        return MAX_MERGED_SAMPLES

BYTES_PER_SAMPLE = _REQ_CHANNELS * _REQ_SAMPWIDTH_BYTES


class WavFormatError(ValueError):
    """Raised when a source WAV doesn't match the required 16 kHz mono s16
    format. The transcode pipeline normalises all internal files; a
    mismatch indicates upstream drift."""


def read_pcm(src_path: str) -> tuple[bytes, int]:
    """Read the raw PCM frames + sample count from a WAV file. Validates
    format. Returns `(pcm_bytes, n_samples)`."""
    with wave.open(src_path, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (
            _REQ_CHANNELS, _REQ_SAMPWIDTH_BYTES, _REQ_RATE,
        ):
            raise WavFormatError(
                f"{os.path.basename(src_path)}: expected "
                f"({_REQ_CHANNELS}, {_REQ_SAMPWIDTH_BYTES}, {_REQ_RATE}), "
                f"got ({w.getnchannels()}, {w.getsampwidth()}, "
                f"{w.getframerate()})"
            )
        n = w.getnframes()
        return w.readframes(n), n


def silence_bytes(ms: int) -> bytes:
    """Generate `ms` milliseconds of digital silence (all-zero PCM bytes)
    in our fixed format. Silence and feature-extractor zero-padding are
    byte-identical, so we don't need anything fancier than this."""
    n_samples = int(round(ms / 1000.0 * _REQ_RATE))
    return bytes(n_samples * BYTES_PER_SAMPLE)


def samples_to_ms(samples: int) -> int:
    """Sample count -> milliseconds (rounded) in our fixed format."""
    return int(round(samples * 1000 / _REQ_RATE))


def replace_with_retry(src_path: str, dst_path: str) -> None:
    """os.replace with the Windows-AV-lock retry: 3 attempts, 0.1 s apart.
    Raises the last OSError; the caller owns removing src_path."""
    last_err: OSError | None = None
    for _attempt in range(3):
        try:
            os.replace(src_path, dst_path)
            return
        except OSError as e:
            last_err = e
            time.sleep(0.1)
    raise last_err


def merge_wavs(
    src_paths: list[str],
    dst_path: str,
    *,
    gap_ms: int = 300,
    trim: bool = False,
    edge_pad_ms: int = 50,
    max_internal_gap_ms: int = 300,
    threshold: float = 0.5,
) -> dict:
    """Concatenate the given WAVs into a single PCM WAV at dst_path.

    When `trim` is set (audio_vad_trim.trim_pcm_for_merge available), the
    output uses a UNIFORM-silence layout: each member is trimmed to its
    speech body (no edge padding, internal gaps capped at
    `max_internal_gap_ms`), then concatenated as
    `[edge_pad_ms] body0 [gap_ms] body1 [gap_ms] … bodyN [edge_pad_ms]`.
    So every member join carries exactly `gap_ms` of silence and both outer
    ends carry `edge_pad_ms`; a member's own trailing + leading silence never
    stacks onto the gap into multi-second dead air. Each member dict carries
    `offset_ms`, its absolute start in the merged WAV, so the route layer can
    re-place per-word karaoke timestamps.

    When `trim` is False (VAD unavailable / disabled), falls back to the
    legacy layout: full members joined by `gap_ms`, no outer margin.

    Returns a dict:
      {
        "bytes":        int,           # size of the written WAV on disk
        "n_samples":    int,           # total samples in the merged WAV
        "duration_ms":  int,
        "members": [                   # one entry per src_paths, in order
          {"lead_ms": int, "new_duration_ms": int,
           "segments": [[orig_start_ms, orig_end_ms, new_start_ms], ...],
           "offset_ms": int},          # absolute body start in merged WAV
          ...                          #   (trim path only; omitted when trim=False)
        ],
      }

    The per-member `segments` map is what the route layer persists
    (member_trims) so group word-level karaoke timestamps stay in sync
    with the trimmed audio. When `trim` is False (or VAD is unavailable) each
    member carries an identity map over its full duration.

    Raises:
      WavFormatError: any source doesn't match (1ch, 16-bit, 16 kHz).
      ValueError:     no sources, total duration over the configured cap, or empty source.
      OSError:        disk write failure (atomic .tmp + os.replace).
    """
    if len(src_paths) < 1:
        raise ValueError("need at least 1 source")
    if gap_ms < 0:
        raise ValueError("gap_ms must be ≥ 0")
    _max_samples = _max_merged_samples()

    trimmer = None
    if trim:
        try:
            from faster_whisper_backend.captures import vad_trim as audio_vad_trim
            trimmer = audio_vad_trim.trim_pcm_for_merge
        except Exception as _e:  # pragma: no cover - import guard
            logger.warning("[merge] per-member trim unavailable: %s", _e)

    pieces: list[bytes] = []
    total_samples = 0
    members: list[dict] = []

    def _over_cap(reserve: int, i: int) -> None:
        if total_samples + reserve > _max_samples:
            raise ValueError(
                f"merged duration would exceed the sample cap "
                f"({_max_samples / _REQ_RATE:.2f} s) — got "
                f"{(total_samples + reserve) / _REQ_RATE:.2f} s after "
                f"{i+1} of {len(src_paths)} sources"
            )

    if trimmer is not None:
        edge = silence_bytes(edge_pad_ms)
        edge_samples = len(edge) // BYTES_PER_SAMPLE
        join = silence_bytes(gap_ms)
        join_samples = len(join) // BYTES_PER_SAMPLE
        pieces.append(edge)                 # leading outer margin
        total_samples += edge_samples
        for i, sp in enumerate(src_paths):
            pcm, n = read_pcm(sp)
            if n == 0:
                raise ValueError(f"source {os.path.basename(sp)} is empty")
            res = trimmer(
                pcm, n,
                edge_pad_ms=edge_pad_ms,
                max_internal_gap_ms=max_internal_gap_ms,
                threshold=threshold,
            )
            body = res["pcm"]
            bn = res["new_n_samples"]
            if i > 0:
                pieces.append(join)
                total_samples += join_samples
            offset_ms = samples_to_ms(total_samples)
            members.append({
                "lead_ms": int(res["lead_ms"]),
                "new_duration_ms": int(res["new_duration_ms"]),
                "segments": res["segments"],
                "offset_ms": offset_ms,
            })
            pieces.append(body)
            total_samples += bn
            _over_cap(edge_samples, i)       # reserve the trailing margin
        pieces.append(edge)                  # trailing outer margin
        total_samples += edge_samples
    else:
        # No-trim path: full members joined by gap_ms, no outer margin.
        gap = silence_bytes(gap_ms)
        gap_samples = len(gap) // BYTES_PER_SAMPLE
        for i, sp in enumerate(src_paths):
            pcm, n = read_pcm(sp)
            if n == 0:
                raise ValueError(f"source {os.path.basename(sp)} is empty")
            dur_ms = samples_to_ms(n)
            members.append({
                "lead_ms": 0,
                "new_duration_ms": dur_ms,
                "segments": [[0, dur_ms, 0]],
            })
            if i > 0:
                pieces.append(gap)
                total_samples += gap_samples
            pieces.append(pcm)
            total_samples += n
            _over_cap(0, i)

    out_pcm = b"".join(pieces)

    # Atomic write: tmp + fsync + replace_with_retry (the Windows-AV-lock
    # loop captures_store and audio_vad_trim share).
    # `or "."` so a bare-filename dst_path (no dirname) doesn't pass "" to
    # os.makedirs, which raises FileNotFoundError on some platforms.
    dst_dir = os.path.dirname(dst_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    # Create the tmp owner-only and fresh. `wave.open` uses a plain builtin
    # open(), so left to itself the process umask (typically 022 -> 0644)
    # would decide the mode and the whole merged PCM — PHI — would sit
    # world-readable in e.g. the shared system temp dir for the duration of
    # the write. mkstemp is O_EXCL + 0600 under a random name, so a file
    # someone pre-created at a predictable "<dst>.tmp" (which would keep its
    # owner and mode through open('wb')) is never reused. The ".tmp" suffix
    # keeps reconcile's stale-tmp sweep matching a crashed write's leftover.
    fd, tmp_path = tempfile.mkstemp(
        dir=dst_dir, prefix=os.path.basename(dst_path) + ".", suffix=".tmp")
    os.close(fd)
    try:
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(_REQ_CHANNELS)
            w.setsampwidth(_REQ_SAMPWIDTH_BYTES)
            w.setframerate(_REQ_RATE)
            w.writeframes(out_pcm)
        # Belt-and-braces over mkstemp's 0600 (Windows / foreign umask):
        # os.replace carries THIS inode's mode onto dst_path, which would
        # widen a 0600 destination such as the mkstemp preview file in the
        # world-writable system temp dir.
        store_common.secure_file(tmp_path)
        try:
            with open(tmp_path, "rb") as fp:
                os.fsync(fp.fileno())
        except OSError:
            pass
        replace_with_retry(tmp_path, dst_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # No outer trim of the merged WAV: with a trimmer each member was trimmed
    # before the join, so its edges and internal gaps are already bounded.
    return {
        "bytes": os.path.getsize(dst_path),
        "n_samples": total_samples,
        "duration_ms": samples_to_ms(total_samples),
        "members": members,
    }


def hash_wav_pcm(src_path: str) -> str:
    """SHA-256 hex of a source file's PCM frame bytes, WAV header excluded.
    samples.py snapshots one per member into the sample's `member_hashes`
    at merge time."""
    pcm, _ = read_pcm(src_path)
    return sha256(pcm).hexdigest()
