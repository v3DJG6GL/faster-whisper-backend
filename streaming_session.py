"""Per-connection streaming dictation state machine.

One :class:`StreamSession` per WebSocket. It consumes 16 kHz mono PCM, runs the
partial/final decode loop, stabilizes live text with LocalAgreement-2, and emits:

  * ``partial`` messages — raw Whisper text (committed prefix + provisional tail),
    updated ~1×/s while speaking. **No post-processing.**
  * ``final`` messages — the post-processed document, split into a stable
    ``committed`` prefix (append-only, never rewritten on screen) and a provisional
    ``tail`` (shown immediately but still revisable). Emitted per utterance once
    end-of-speech silence (or a forced commit) produces a fresh decode.

The class is **dependency-injected**: the model decode calls, the post-processing
function, and the emit sink are passed in, so this module imports nothing from
``main.py`` (no circular import) and is unit-testable without faster-whisper.

Post-processing is run on the session's *rolling whole-document raw transcript*
(``raw_confirmed``) — identical semantics to the batch route — and only the
provably-stable prefix is emitted. This dissolves every cross-utterance "seam"
hazard in the 17-rule pipeline (split ``"neue Zeile"``, capitalize-after-terminator,
punctuation dedup, …) instead of patching each one.
"""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from streaming_localagreement import LocalAgreementProcessor
from streaming_vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, iter_frames, rms_dbfs

logger = logging.getLogger(__name__)

# A fresh decode hypothesis: buffer-relative word triples (start_s, end_s, text).
Hypothesis = list[tuple[float, float, str]]
# Final decode: raw verbatim text, optional word list for verbose_json, and a
# flag — True when the decode produced segments but dropped EVERY one as a
# hallucination, so an empty ``raw`` is authoritative and must NOT be replaced by
# the partial-built LocalAgreement transcript (see _finalize).
FinalResult = tuple[str, list[dict], bool]

DecodePartial = Callable[[np.ndarray, str], Awaitable[Hypothesis]]
DecodeFinal = Callable[[np.ndarray, str], Awaitable[FinalResult]]
Postprocess = Callable[[str], str]
Emit = Callable[[dict], Awaitable[None]]

_TERMINATOR_RE = re.compile(r"[.?!\n]")


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common leading substring of ``a`` and ``b``."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class StreamConfig:
    """Streaming parameters. Defaults are the validated German-dictation /
    12–16 GB-GPU set; every field is overridden from ``WHISPER_STREAMING_*`` config."""

    sample_rate: int = SAMPLE_RATE
    min_chunk_ms: int = 1000          # partial cadence: new audio before re-decoding
    min_speech_ms: int = 500          # skip inference below this much speech (anti-hallucination)
    vad_min_silence_ms: int = 700     # inner gate: silence that triggers a boundary partial
    commit_silence_ms: int = 1200     # outer gate: silence that finalizes the utterance
    hard_break_silence_ms: int = 5000  # silence that ends the whole grouping → fresh document (0 = off)
    hard_break_separator: str = ""    # client-typed separator between documents ("\n" = newline, " " = space)
    forced_commit_sec: float = 25.0   # hard cap on speech before a forced finalize (< 30 s mel field)
    buffer_trim_sec: float = 15.0     # trim the audio buffer when it grows past this
    buffer_trim_keep_sec: float = 10.0  # audio kept (anchored at a committed word) after a trim
    max_buffer_sec: float = 600.0     # decode-independent buffer ceiling → forced finalize
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
        session_id: str = "",
    ) -> None:
        self.cfg = config
        self.session_id = session_id      # for log lines only (route's connection id)
        self.endpointer = endpointer
        self.decode_partial = decode_partial
        self.decode_final = decode_final
        self.postprocess = postprocess
        self.emit = emit
        self.base_prompt = base_prompt
        self.on_final = on_final

        # Silero VAD is a synchronous ONNX call and it runs once per 32 ms
        # frame, so doing it inline pinned the event loop for as long as a
        # client kept uploading — the decodes were already offloaded, this was
        # the one hot path left on the loop.
        #
        # ONE worker, and one pool per session, for two reasons: the endpointer
        # keeps a rolling buffer that must see frames in order (a single worker
        # serialises them), and a dedicated pool keeps these ~31 calls/second
        # out of the default executor the Whisper decode shares — offloading
        # onto that pool would just move the contention.
        self._vad_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"vad-{session_id[:8] or 'stream'}",
        )

        self._min_chunk_samples = int(config.min_chunk_ms * config.sample_rate / 1000)
        self._preroll_keep_samples = int(config.preroll_keep_ms * config.sample_rate / 1000)

        self.la = LocalAgreementProcessor()
        # Utterance audio, held as pending chunks and joined lazily — see the
        # ``audio`` property. Never touch these three directly: _append_audio /
        # _set_audio are the only writers, so the sample counter and the cache
        # can never drift apart.
        self._audio_chunks: "list[np.ndarray]" = []
        self._audio_samples = 0            # len(self.audio) WITHOUT materializing it
        self._audio_cache: "np.ndarray | None" = None
        self._set_audio(np.zeros(0, dtype=np.float32))
        self._buffer_offset = 0.0          # wall start (s) of audio[0] within this utterance
        self._frame_tail = np.zeros(0, dtype=np.float32)  # < 512 samples awaiting a full frame
        self._byte_tail = b""              # odd trailing PCM byte awaiting its pair

        self._in_utterance = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._idle_silence_ms = 0          # continuous silence since speech stopped (survives finalize)
        self._new_since_partial = 0

        self.raw_confirmed = ""            # cross-utterance verbatim accumulator
        self._committed_len = 0            # chars of processed text locked as append-only committed
        self._prev_processed = ""          # last whole-doc post-process (document-level LocalAgreement)
        self._trimmed_text = ""            # committed text whose audio _maybe_trim cut away
        self._trimmed_sec = 0.0            # seconds of utterance audio _maybe_trim cut away
        # Audio + word dicts banked by _maybe_trim so on_final can hand captures
        # the WHOLE utterance, freed on reset. The old claim here — that
        # forced_commit_sec (25 s) caps this at ~2 MB — was wrong twice over:
        # forced_commit_sec counts SPEECH (_speech_ms), not buffered wall
        # clock, and max_buffer_sec tested only the LIVE buffer, which
        # _maybe_trim keeps pinned near buffer_trim_keep_sec. A caller emitting
        # one speech frame per ~1.1 s of silence banked 860 s / 56 MB against a
        # documented 600 s / 38 MB ceiling that never fired. _trimmed_samples
        # tracks the bank so the ceiling can measure the whole utterance.
        self._trimmed_audio: "list[np.ndarray]" = []
        self._trimmed_samples = 0
        self._trimmed_words: "list[dict]" = []
        self._utterance_index = 0
        self._prompt = base_prompt.strip()
        self._closed = False
        # Set by the streaming route's consumer when it has fallen behind realtime:
        # skip the (expensive) partial decode so we can catch up. Audio is still fed
        # (VAD/endpointing stays intact) and finals still run.
        self._skip_partials = False

    # ---- audio buffer -----------------------------------------------------
    #
    # Frames arrive every 32 ms but the buffer is only *read* per partial decode
    # (min_chunk_ms, ~1 s) — so appending must not copy. Frames are queued as
    # chunks and joined on the first read, which caches the result and collapses
    # the list back to one chunk; further reads with no append in between hand
    # back the cache. A decode still needs one contiguous array, so a join must
    # happen — it just happens ~30× less often, and with it the transient in
    # which both the old and the joined array are resident.
    #
    # The corollary: anything that only wants a LENGTH must read
    # ``_audio_samples``, never ``len(self.audio)``. The max_buffer_sec ceiling
    # runs after every single frame; going through the property there would join
    # on every frame and defeat the whole arrangement.

    @property
    def audio(self) -> np.ndarray:
        """The utterance buffer as one contiguous float32 array."""
        if self._audio_cache is None:
            # _append_audio is the only writer that clears the cache, and it
            # always leaves at least one chunk behind.
            self._audio_cache = (self._audio_chunks[0] if len(self._audio_chunks) == 1
                                 else np.concatenate(self._audio_chunks))
            self._audio_chunks = [self._audio_cache]
        return self._audio_cache

    def _append_audio(self, chunk: np.ndarray) -> None:
        """Queue a chunk — O(1), no copy. Invalidates the joined cache."""
        self._audio_chunks.append(chunk)
        self._audio_samples += chunk.shape[0]
        self._audio_cache = None

    def _set_audio(self, buf: np.ndarray) -> None:
        """Replace the whole buffer (reset / trim). ``buf`` may be a slice of the
        current buffer — kept as-is, so a view stays a view."""
        self._audio_chunks = [buf]
        self._audio_samples = buf.shape[0]
        self._audio_cache = buf

    # ---- public API -------------------------------------------------------

    async def feed_pcm(self, pcm_int16_le: bytes) -> None:
        """Feed a chunk of raw 16 kHz mono signed-16-bit little-endian PCM."""
        if self._closed or not pcm_int16_le:
            return
        # Chunk boundaries (notably ffmpeg's stdout pipe reads, but also a
        # misframed raw-PCM client) need not fall on 2-byte sample boundaries;
        # carry an odd trailing byte to the next feed so np.frombuffer never
        # sees a non-even buffer (which would raise ValueError and kill the
        # session).
        if self._byte_tail:
            pcm_int16_le = self._byte_tail + pcm_int16_le
        if len(pcm_int16_le) & 1:
            self._byte_tail = pcm_int16_le[-1:]
            pcm_int16_le = pcm_int16_le[:-1]
        else:
            self._byte_tail = b""
        if not pcm_int16_le:
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
        """Drain: finalize any in-flight utterance and commit the whole document."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._in_utterance:
                await self._finalize(forced=True)
            processed = self.postprocess(self.raw_confirmed)
            await self._emit_document(processed, flush_all=True, last=True)
        finally:
            # One thread per session; releasing it here (rather than leaving it
            # to GC) keeps the count tied to live sessions. wait=False so a
            # teardown never blocks the loop on an in-flight VAD frame.
            self._vad_pool.shutdown(wait=False)

    # ---- frame pump -------------------------------------------------------

    async def _consume_frame(self, frame: np.ndarray) -> None:
        # Off the loop (see _vad_pool). Per frame rather than per chunk: a
        # finalize inside this loop calls endpointer.reset(), so pre-computing
        # a whole chunk's decisions would answer with a buffer that no longer
        # exists. _pump is the sole caller and the sole session mutator, so the
        # extra await introduces no new interleaving on session state.
        speech = await asyncio.get_running_loop().run_in_executor(
            self._vad_pool, self.endpointer.is_speech, frame,
        )
        self._append_audio(frame)
        self._new_since_partial += FRAME_SAMPLES

        # Decode-independent ceiling on the utterance buffer. _maybe_trim is the
        # normal bound, but it only runs from _run_partial — behind the skip-
        # partials/speech/RMS gates, and it needs a committed word to anchor its
        # cut. None of those hold while VAD flicker (a room mic, HVAC, music)
        # keeps an utterance open and silence piles up here frame by frame, and
        # forced_commit_sec can't help: it counts speech, not buffered audio.
        # Checked unconditionally so no gate can shadow it. Finalizing (not
        # discarding) keeps whatever was actually said — the session just
        # continues in a fresh utterance.
        # Counter, not the property: this runs on every frame (see the buffer
        # note above).
        # Live buffer PLUS what _maybe_trim banked: the bank is still held for
        # the finalize, so it is the utterance's real memory cost.
        _held = self._audio_samples + self._trimmed_samples
        if _held >= self.cfg.max_buffer_sec * self.cfg.sample_rate:
            logger.warning("[stream %s] audio buffer hit the %.0f s ceiling (%.1f s "
                           "held: %.1f s live + %.1f s trimmed) — forcing a finalize",
                           self.session_id[:8], self.cfg.max_buffer_sec,
                           _held / self.cfg.sample_rate,
                           self._audio_samples / self.cfg.sample_rate,
                           self._trimmed_samples / self.cfg.sample_rate)
            await self._finalize(forced=True)
            return

        if speech:
            self._in_utterance = True
            self._speech_ms += FRAME_MS
            self._silence_ms = 0
            self._idle_silence_ms = 0
        else:
            self._idle_silence_ms += FRAME_MS
            if self._in_utterance:
                self._silence_ms += FRAME_MS

        # Hard break: once the current utterance has been finalized (we're idle again)
        # and the accumulated document has been silent long enough, reset to a fresh
        # document — pauses become paragraph boundaries and a multi-minute latch
        # session can't grow without bound. Fires once per quiet gap (the
        # raw_confirmed guard) and never closes the socket.
        if (self.cfg.hard_break_silence_ms > 0
                and not self._in_utterance
                and self.raw_confirmed
                and self._idle_silence_ms >= self.cfg.hard_break_silence_ms):
            await self._hard_break()

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
        if self._audio_samples > self._preroll_keep_samples:
            self._set_audio(self.audio[-self._preroll_keep_samples:])
            self._buffer_offset = 0.0

    # ---- decode steps -----------------------------------------------------

    async def _run_partial(self) -> None:
        self._new_since_partial = 0
        if self._skip_partials:
            return  # behind realtime — skip the partial decode so we can catch up
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
        dur = self._audio_samples / self.cfg.sample_rate
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
            # The DECODE buffer loses the cut span — later decodes can't re-hear
            # it, so bank its committed words' text for _finalize (prepended to
            # the final decode's result) and fold it into the rolling prompt so
            # the partial/final decodes of the now-mid-sentence buffer get the
            # preceding words as context (an uncontexted seam mishears its
            # opening, e.g. "on"→"and"). Bank the audio + word dicts too, so
            # on_final can hand captures the WHOLE utterance, not a fragment.
            # la.committed spans the whole utterance, so bound below by the
            # previous cut (== the current _buffer_offset) or a second trim
            # would re-bank the first trim's words. Bank BEFORE the buffer is
            # re-sliced; .copy() detaches from the old array (a bare view would
            # pin the whole pre-trim buffer in memory).
            cut_words = [w for w in self.la.committed
                         if self._buffer_offset < w.end <= cut]
            cut_text = self.la.text_of(cut_words)
            self._trimmed_text += cut_text
            self._trimmed_sec += cut - self._buffer_offset
            buf = self.audio
            self._trimmed_audio.append(buf[:cut_samples].copy())
            self._trimmed_samples += cut_samples
            self._trimmed_words.extend(
                {"word": w.text, "start": w.start, "end": w.end}
                for w in cut_words)
            self._set_audio(buf[cut_samples:])
            self._prompt = " ".join(
                (self._prompt + " " + cut_text).split()[-self.cfg.prompt_words:])
            self._buffer_offset = cut
            self.la.pop_committed(cut)

    async def _finalize(self, forced: bool = False) -> None:
        audio = self.audio
        # Anti-hallucination: never run the final decode on near-silence.
        if self._speech_ms < self.cfg.min_speech_ms or rms_dbfs(audio) < self.cfg.rms_gate_dbfs:
            # The gate judges the LIVE (post-trim) buffer only — text already
            # committed by LocalAgreement from earlier LOUD audio must not be
            # discarded with the near-silent tail (reachable via the
            # max_buffer_sec ceiling: a trim banks words, then piled-up quiet
            # frames sink the whole-buffer RMS under the gate). Skip the DECODE
            # — that is all the gate promises — but still emit what was already
            # agreed. la.committed spans the WHOLE utterance (pop_committed
            # only prunes the agreement buffer), so this already includes any
            # trim-banked words — do NOT prepend _trimmed_text here.
            tail = self.la.finish()
            raw = self.la.committed_text + self.la.text_of(tail)
            if not raw.strip():
                self._reset_utterance()
                return
            proc_dur = 0.0
            decoded = False
            # No decode ran, so LocalAgreement is the only word source. Its
            # committed list + tail span the WHOLE utterance in absolute
            # (utterance) time and already hold any trim-banked words, so the
            # list is complete as it stands — the trim merge below must NOT
            # re-base it or prepend _trimmed_words (every banked word would
            # appear twice).
            words = [{"word": w.text, "start": w.start, "end": w.end}
                     for w in (self.la.committed + tail)]
        else:
            decoded = True
            t0 = time.perf_counter()
            raw, words, dropped_all = await self.decode_final(audio.copy(), self._prompt)
            proc_dur = time.perf_counter() - t0
            if not (raw and raw.strip()) and not dropped_all:
                # The final decode produced nothing at all (e.g. its VAD filter trimmed
                # the whole buffer) — fall back to the partial-built LocalAgreement
                # transcript. But when the decode DID produce segments and dropped them
                # all as hallucinations (dropped_all), the empty result is authoritative:
                # the partials run at a fixed temperature and so never trip the drop —
                # i.e. they still hold the hallucination — so keep the empty result.
                # la.committed spans the WHOLE utterance (pop_committed only prunes the
                # agreement buffer), so this path already includes any trim-banked words.
                raw = self.la.committed_text + self.la.text_of(self.la.finish())
            else:
                # The decode only heard the (possibly trim-shortened) buffer: text whose
                # audio _maybe_trim cut from it survives in the banked committed words —
                # without this prefix an over-15s continuous utterance loses its opening
                # (the decode result would replace the already-committed-and-shown text).
                # Applies to the dropped_all case too: the drop verdict judged the
                # remaining buffer, not the banked (multi-partial-agreed) prefix.
                raw = self._trimmed_text + raw
        # Reassemble the WHOLE utterance for on_final: the banked audio slices +
        # the remaining buffer, and the banked word dicts (absolute times) + the
        # final decode's words shifted from buffer-relative to utterance time.
        # Captures therefore store the full audio↔text pair, not a fragment.
        # The gate path's words are already utterance-absolute and already
        # include the banked prefix (see above), so only a real decode's
        # buffer-relative words are re-based here.
        if self._trimmed_audio:
            full_audio = np.concatenate([*self._trimmed_audio, audio])
            if decoded:
                off = self._buffer_offset
                words = self._trimmed_words + [
                    {**w, "start": w["start"] + off, "end": w["end"] + off}
                    for w in words]
        else:
            full_audio = audio
        self.raw_confirmed += raw
        self._prompt = self._make_prompt()
        processed = self.postprocess(self.raw_confirmed)
        await self._emit_document(processed, forced=forced, words=words)
        if self.on_final is not None:
            await self.on_final({
                "utterance": self._utterance_index,
                "audio_dur": full_audio.shape[0] / self.cfg.sample_rate,
                "trimmed_sec": self._trimmed_sec,  # >0 → a mid-utterance trim fired
                # False → the near-silence gate skipped the final decode: the
                # text is the partial-committed transcript, no decoder
                # diagnostics belong to this utterance, and proc_dur is 0.
                "decoded": decoded,
                "proc_dur": proc_dur,
                "raw_text": raw,       # full utterance text (incl. trim-banked prefix)
                "words": words,        # word-timestamp dicts (for captures / verbose)
                "audio": full_audio,   # float32 PCM of the WHOLE utterance (for captures)
                "forced": forced,
            })
        self._utterance_index += 1
        self._reset_utterance()

    async def _hard_break(self) -> None:
        """End the whole grouping after a long silence and start a fresh document,
        without closing the WebSocket.

        Emits a ``boundary`` marker so the client resets its injection baseline (and
        optionally types ``hard_break_separator`` between documents), then clears the
        cross-utterance accumulators. The rolling prompt is reset too — a long pause
        is treated as a new context; to instead keep terminology across breaks, drop
        the ``self._prompt`` reset below."""
        await self.emit({
            "type": "boundary",
            "utterance": self._utterance_index,
            "separator": self.cfg.hard_break_separator,
        })
        self.raw_confirmed = ""
        self._committed_len = 0
        self._prev_processed = ""
        self._prompt = self.base_prompt.strip()
        self._idle_silence_ms = 0

    # ---- emission ---------------------------------------------------------

    async def _emit_document(
        self, processed: str, *, forced: bool = False, flush_all: bool = False,
        last: bool = False, words: Optional[list[dict]] = None,
    ) -> None:
        """Emit the post-processed document split into a stable ``committed`` prefix
        and a provisional ``tail``.

        ``committed`` is append-only — it only ever grows and is never rewritten on
        screen. ``tail`` is the still-unstable remainder: shown live (so the most
        recent sentence is visible immediately) but explicitly provisional, since
        appending the next utterance can still reshape it. ``flush_all`` (session
        close) commits the whole document. Both are full authoritative strings, not
        byte deltas — the client replaces each region, so a seam rewrite in the
        post-processing can never desync the display."""
        commit_len = len(processed) if flush_all else self._stable_commit_len(processed)
        committed = processed[:commit_len]
        tail = processed[commit_len:]
        self._committed_len = commit_len
        self._prev_processed = processed
        if not committed and not tail:
            return
        msg = {
            "type": "final",
            "utterance": self._utterance_index,
            "committed": committed,
            "tail": tail,
        }
        if forced:
            msg["forced"] = True
        if last:
            msg["last"] = True
        if words:
            msg["words"] = words
        await self.emit(msg)

    def _stable_commit_len(self, processed: str) -> int:
        """Index up to which ``processed`` is safe to commit append-only.

        Document-level LocalAgreement: commit only through the last sentence
        terminator (``. ? ! \\n``) that lies within the prefix the last *two*
        whole-document post-processes agree on. Appending a later utterance can
        still rewrite earlier text — a 'neue Zeile' split across the seam, the
        capitalization after a terminator, a number spanning the boundary, even a
        repeated sentence whose punctuation collapses — so requiring two passes to
        agree before locking keeps the committed region flicker-free. The cost is
        that the newest sentence stays provisional for one extra finalize, but it is
        still shown (as the tail). A safety valve commits an over-long un-agreed
        tail so the held region can't grow without bound."""
        agree = _common_prefix_len(processed, self._prev_processed)
        boundary = 0
        for m in _TERMINATOR_RE.finditer(processed):
            if m.end() <= agree:
                boundary = m.end()
            else:
                break
        held = len(processed) - boundary
        if held > self.cfg.max_hold_chars:
            boundary = max(boundary, len(processed) - self.cfg.tail_margin_chars)
        return max(boundary, self._committed_len)

    # ---- utterance lifecycle ---------------------------------------------

    def _make_prompt(self) -> str:
        tail = " ".join(self.raw_confirmed.split()[-self.cfg.prompt_words:])
        return (self.base_prompt + " " + tail).strip()

    def _reset_utterance(self) -> None:
        self.la.reset()
        self._set_audio(np.zeros(0, dtype=np.float32))
        self._buffer_offset = 0.0
        self._trimmed_text = ""
        self._trimmed_sec = 0.0
        self._trimmed_audio = []
        self._trimmed_samples = 0
        self._trimmed_words = []
        self._in_utterance = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._new_since_partial = 0
        self.endpointer.reset()
