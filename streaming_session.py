"""Per-connection streaming dictation state machine.

One :class:`StreamSession` per WebSocket. It consumes 16 kHz mono PCM, runs the
partial/final decode loop, stabilizes live text with LocalAgreement-2, and emits:

  * ``partial`` messages — raw Whisper text (committed prefix + provisional tail),
    updated ~1×/s while speaking. **No post-processing.**
  * ``final`` messages — post-processed, **append-only** text, emitted per utterance
    once end-of-speech silence (or a forced commit) stabilizes it.

The class is **dependency-injected**: the model decode calls, the post-processing
function, and the emit sink are passed in, so this module imports nothing from
``main.py`` (no circular import) and is unit-testable without faster-whisper.

Post-processing is run on the session's *rolling whole-document raw transcript*
(``raw_confirmed``) — identical semantics to the batch route — and only the
provably-stable prefix is emitted. This dissolves every cross-utterance "seam"
hazard in the 17-rule pipeline (split ``"neue Zeile"``, capitalize-after-terminator,
punctuation dedup, …) instead of patching each one.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from streaming_localagreement import LocalAgreementProcessor
from streaming_vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, iter_frames, rms_dbfs

logger = logging.getLogger(__name__)

# A fresh decode hypothesis: buffer-relative word triples (start_s, end_s, text).
Hypothesis = list[tuple[float, float, str]]
# Final decode: raw verbatim text + optional word list for verbose_json.
FinalResult = tuple[str, list[dict]]

DecodePartial = Callable[[np.ndarray, str], Awaitable[Hypothesis]]
DecodeFinal = Callable[[np.ndarray, str], Awaitable[FinalResult]]
Postprocess = Callable[[str], str]
Emit = Callable[[dict], Awaitable[None]]

_TERMINATOR_RE = re.compile(r"[.?!\n]")


@dataclass
class StreamConfig:
    """Streaming parameters. Defaults are the validated German-dictation /
    12–16 GB-GPU set; every field is overridden from ``WHISPER_STREAMING_*`` config."""

    sample_rate: int = SAMPLE_RATE
    min_chunk_ms: int = 1000          # partial cadence: new audio before re-decoding
    min_speech_ms: int = 500          # skip inference below this much speech (anti-hallucination)
    vad_min_silence_ms: int = 700     # inner gate: silence that triggers a boundary partial
    commit_silence_ms: int = 1200     # outer gate: silence that finalizes the utterance
    forced_commit_sec: float = 25.0   # hard cap on speech before a forced finalize (< 30 s mel field)
    buffer_trim_sec: float = 15.0     # trim the audio buffer when it grows past this
    buffer_trim_keep_sec: float = 10.0  # audio kept (anchored at a committed word) after a trim
    rms_gate_dbfs: float = -42.0      # skip inference if the buffer is quieter than this
    preroll_keep_ms: int = 500        # leading silence retained before speech starts
    prompt_words: int = 200           # cross-utterance context carried as initial_prompt
    max_hold_chars: int = 400         # safety: flush a held tail that grows past this
    tail_margin_chars: int = 24       # chars kept unflushed by the safety flush (≥ longest dictation phrase)


class StreamSession:
    def __init__(
        self,
        *,
        config: StreamConfig,
        endpointer,
        decode_partial: DecodePartial,
        decode_final: DecodeFinal,
        postprocess: Postprocess,
        emit: Emit,
        base_prompt: str = "",
        on_final: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        self.cfg = config
        self.endpointer = endpointer
        self.decode_partial = decode_partial
        self.decode_final = decode_final
        self.postprocess = postprocess
        self.emit = emit
        self.base_prompt = base_prompt
        self.on_final = on_final

        self._min_chunk_samples = int(config.min_chunk_ms * config.sample_rate / 1000)
        self._preroll_keep_samples = int(config.preroll_keep_ms * config.sample_rate / 1000)

        self.la = LocalAgreementProcessor()
        self.audio = np.zeros(0, dtype=np.float32)
        self._buffer_offset = 0.0          # wall start (s) of audio[0] within this utterance
        self._frame_tail = np.zeros(0, dtype=np.float32)  # < 512 samples awaiting a full frame

        self._in_utterance = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._new_since_partial = 0

        self.raw_confirmed = ""            # cross-utterance verbatim accumulator
        self._emitted_len = 0              # chars of processed text already emitted (append-only)
        self._utterance_index = 0
        self._prompt = base_prompt.strip()
        self._closed = False

    # ---- public API -------------------------------------------------------

    async def feed_pcm(self, pcm_int16_le: bytes) -> None:
        """Feed a chunk of raw 16 kHz mono signed-16-bit little-endian PCM."""
        if self._closed or not pcm_int16_le:
            return
        samples = np.frombuffer(pcm_int16_le, dtype="<i2").astype(np.float32) / 32768.0
        if self._frame_tail.size:
            samples = np.concatenate([self._frame_tail, samples])
        frames = list(iter_frames(samples))
        used = len(frames) * FRAME_SAMPLES
        self._frame_tail = samples[used:].copy()
        for frame in frames:
            await self._consume_frame(frame)

    async def flush_utterance(self) -> None:
        """Force-finalize the current utterance (client 'flush' control message)."""
        if self._in_utterance:
            await self._finalize(forced=True)

    async def close(self) -> None:
        """Drain: finalize any in-flight utterance and flush the held tail as final."""
        if self._closed:
            return
        self._closed = True
        if self._in_utterance:
            await self._finalize(forced=True)
        processed = self.postprocess(self.raw_confirmed)
        await self._emit_final_delta(processed, flush_all=True, last=True)

    # ---- frame pump -------------------------------------------------------

    async def _consume_frame(self, frame: np.ndarray) -> None:
        speech = self.endpointer.is_speech(frame)
        self.audio = np.concatenate([self.audio, frame])
        self._new_since_partial += FRAME_SAMPLES

        if speech:
            self._in_utterance = True
            self._speech_ms += FRAME_MS
            self._silence_ms = 0
        elif self._in_utterance:
            self._silence_ms += FRAME_MS

        if not self._in_utterance:
            self._trim_preroll()
            return

        if self._silence_ms >= self.cfg.commit_silence_ms:
            await self._finalize()
            return
        if self._speech_ms >= self.cfg.forced_commit_sec * 1000:
            await self._finalize(forced=True)
            return

        # Fire a partial roughly every min_chunk of new audio — but ONLY while
        # actively speaking (silence below the inner gate). Re-decoding during
        # trailing silence is wasteful AND pathological: each partial decode is
        # awaited synchronously (~1 s+), so triggering one per silent frame makes
        # the silence timer advance ~1 frame (32 ms) per decode, inflating the
        # commit wait from ~1.2 s to ~20 s. Once the speaker pauses we let silence
        # accumulate in real time so _finalize() fires at commit_silence_ms.
        if (self._silence_ms < self.cfg.vad_min_silence_ms
                and self._new_since_partial >= self._min_chunk_samples):
            await self._run_partial()

    def _trim_preroll(self) -> None:
        """Keep only a short lead-in of pre-speech silence so the buffer doesn't
        grow without bound during quiet periods."""
        if self.audio.shape[0] > self._preroll_keep_samples:
            self.audio = self.audio[-self._preroll_keep_samples:]
            self._buffer_offset = 0.0

    # ---- decode steps -----------------------------------------------------

    async def _run_partial(self) -> None:
        self._new_since_partial = 0
        if self._speech_ms < self.cfg.min_speech_ms:
            return
        if rms_dbfs(self.audio) < self.cfg.rms_gate_dbfs:
            return
        words = await self.decode_partial(self.audio.copy(), self._prompt)
        self.la.insert_hypothesis(words or [], self._buffer_offset)
        self.la.commit()
        await self.emit({
            "type": "partial",
            "utterance": self._utterance_index,
            "committed": self.la.committed_text,
            "pending": self.la.text_of(self.la.provisional()),
        })
        self._maybe_trim()

    def _maybe_trim(self) -> None:
        dur = self.audio.shape[0] / self.cfg.sample_rate
        if dur <= self.cfg.buffer_trim_sec:
            return
        target = self._buffer_offset + (dur - self.cfg.buffer_trim_keep_sec)
        cut = None
        for w in self.la.committed:        # committed words carry absolute timestamps
            if w.end <= target:
                cut = w.end
            else:
                break
        if cut is not None and cut > self._buffer_offset:
            cut_samples = int((cut - self._buffer_offset) * self.cfg.sample_rate)
            self.audio = self.audio[cut_samples:]
            self._buffer_offset = cut
            self.la.pop_committed(cut)

    async def _finalize(self, forced: bool = False) -> None:
        audio = self.audio
        # Anti-hallucination: never run the final decode on near-silence.
        if self._speech_ms < self.cfg.min_speech_ms or rms_dbfs(audio) < self.cfg.rms_gate_dbfs:
            self._reset_utterance()
            return
        audio_dur = audio.shape[0] / self.cfg.sample_rate
        t0 = time.perf_counter()
        raw, words = await self.decode_final(audio.copy(), self._prompt)
        proc_dur = time.perf_counter() - t0
        if not (raw and raw.strip()):
            # fall back to the LocalAgreement transcript if the final decode is empty
            raw = self.la.committed_text + self.la.text_of(self.la.finish())
        self.raw_confirmed += raw
        self._prompt = self._make_prompt()
        processed = self.postprocess(self.raw_confirmed)
        await self._emit_final_delta(processed, forced=forced, words=words)
        if self.on_final is not None:
            await self.on_final({
                "utterance": self._utterance_index,
                "audio_dur": audio_dur,
                "proc_dur": proc_dur,
                "raw_text": raw,
                "words": len((raw or "").split()),
                "forced": forced,
            })
        self._utterance_index += 1
        self._reset_utterance()

    # ---- emission ---------------------------------------------------------

    async def _emit_final_delta(
        self, processed: str, *, forced: bool = False, flush_all: bool = False,
        last: bool = False, words: Optional[list[dict]] = None,
    ) -> None:
        boundary = len(processed) if flush_all else self._stable_boundary(processed)
        delta = processed[self._emitted_len:boundary]
        if not delta:
            return
        msg = {
            "type": "final",
            "utterance": self._utterance_index,
            "text": delta,
            "append": True,
        }
        if forced:
            msg["forced"] = True
        if last:
            msg["last"] = True
        if words:
            msg["words"] = words
        await self.emit(msg)
        self._emitted_len = boundary

    def _stable_boundary(self, processed: str) -> int:
        """Index up to which ``processed`` is safe to emit append-only: through the
        last sentence terminator char itself, but NOT the whitespace after it.

        That trailing whitespace and the following sentence are held back because
        appending more raw text can still rewrite them — e.g. a later 'Zeile'
        turns '. neue ' into '.\\n' (the post-terminator space becomes part of the
        newline). The held whitespace is emitted later as the next sentence's
        leading edge. A safety valve flushes an over-long held tail."""
        last = None
        for m in _TERMINATOR_RE.finditer(processed):
            last = m
        boundary = 0 if last is None else last.end()
        held = len(processed) - boundary
        if held > self.cfg.max_hold_chars:
            safe = len(processed) - self.cfg.tail_margin_chars
            boundary = max(boundary, safe)
        return max(boundary, self._emitted_len)

    # ---- utterance lifecycle ---------------------------------------------

    def _make_prompt(self) -> str:
        tail = " ".join(self.raw_confirmed.split()[-self.cfg.prompt_words:])
        return (self.base_prompt + " " + tail).strip()

    def _reset_utterance(self) -> None:
        self.la.reset()
        self.audio = np.zeros(0, dtype=np.float32)
        self._buffer_offset = 0.0
        self._in_utterance = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._new_since_partial = 0
        self.endpointer.reset()
