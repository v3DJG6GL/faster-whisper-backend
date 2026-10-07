"""Per-connection streaming dictation state machine.

One :class:`StreamSession` per WebSocket. It consumes 16 kHz mono PCM, runs the
partial/final decode loop, stabilizes live text with LocalAgreement-2, and emits:

  * ``partial`` messages — raw Whisper text (committed prefix + provisional tail),
    updated ~1×/s while speaking. **No post-processing.**
  * ``final`` messages — the post-processed document. Within one document it
    only ever GROWS: every final's ``committed + tail`` starts with the previous
    final's, so a client that types the difference never has to take text back.
    ``committed`` is the part that is also final on screen; ``tail`` (the newest
    sentence) is a display hint only — it will not change either, it just has
    not been confirmed by a second pass yet. Emitted per utterance once
    end-of-speech silence (or a forced commit) produces a fresh decode.
  * ``utterance`` messages — the lifecycle of the utterance the server is
    holding, so a client can show "the backend is working" instead of guessing
    from silence: ``{"type":"utterance","utterance":N,"state":...}`` with
    ``open`` (once, when the utterance holds ``min_speech_ms`` of speech — the
    same gate the partials use, so a noise blip never announces), ``decoding``
    (at most once, right before the final decode) and ``dropped`` (+ ``reason``:
    ``no_speech`` | ``empty`` | ``error``) when it ends without a ``final``.
    **Every announced utterance ends in exactly one terminal frame: a ``final``
    carrying the same ordinal, or ``dropped``.** Ordinals are unique and only
    ever grow; ``close()``'s closing document is not an utterance and never
    emits ``dropped``.
  * a **release** ``final`` (``"flush": true``, no ``utterance``) — the trailing
    words the session withheld (see below), sent when nothing follows them: right
    before a ``boundary``, on a client ``flush`` with no utterance in flight,
    after 5 s of silence when hard breaks are off, and folded into the closing
    ``last`` final.

The class is **dependency-injected**: the model decode calls, the post-processing
function, and the emit sink are passed in, so this module imports nothing from
``main.py`` (no circular import) and is unit-testable without faster-whisper.

Post-processing is run on the session's rolling raw transcript
(``raw_confirmed``) — the batch route's semantics, whole document at a time —
but the pipeline is not prefix-stable: a few rules decide by looking at the
words that FOLLOW (a ``"neue Zeile"`` split across the pause, a comma the
newline tidy eats, ``120 Schrägstrich | 80``, an opening quote). Two things keep
sent text fixed anyway:

  * **hold-back** — the trailing raw words such a rule could still join with the
    next utterance (``holdback``, built from the active dictation maps; see
    pipeline/seam_holdback.py) are neither formatted nor sent yet. The raw text is
    held, never formatted text: the pipeline is not idempotent, so formatted
    output is never run through it again.
  * **re-anchor** — if a new document still does not extend what was sent
    (a self-correction like "Punkt | Strichpunkt"), a WARNING names the rule
    (``diagnose``) and the rest of the document is formatted on its own from
    the sent text on, joined by :meth:`StreamSession._seam_join`. The same
    happens, at INFO, when the formatting language changes mid-document
    (``format_key``). Both last until the next boundary.
"""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from faster_whisper_backend.streaming.localagreement import LocalAgreementProcessor
from faster_whisper_backend.streaming.vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, iter_frames, rms_dbfs

logger = logging.getLogger(__name__)


class CloseAbort(Exception):
    """Raised out of a decode to abort ``StreamSession.close()`` WITHOUT emitting
    the closing document (the route's credential-revocation guard subclasses
    it). Every other decode error on close is logged and the already-confirmed
    text is still committed, matching the pump's per-utterance tolerance."""


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
# Index into a raw text from which its trailing words are held back.
Holdback = Callable[[str], int]
# (sent part's raw, whole raw) → the rule that made them diverge, for the log.
Diagnose = Callable[[str, str], str]

_TERMINATOR_RE = re.compile(r"[.?!\n]")
# With hard breaks off nothing else ever releases held-back words while the
# speaker stays silent, so they go out after this much silence.
_IDLE_RELEASE_MS = 5000
# _emit_document's default: tag the frame with the current utterance ordinal.
_CURRENT_UTTERANCE = object()
# _seam_join: no space after an opening bracket / low quote, none before
# closing punctuation. “ and ‘ are left out: in German they CLOSE („…“).
_OPENING_BRACKETS = "([{„‚«"
_CLOSING_PUNCT = ".,:;!?%)]}…»“‘"


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common leading substring of ``a`` and ``b``."""
    # The document only grows (see _compose), so one is nearly always a prefix
    # of the other: answer that in C instead of a per-character Python loop
    # over the whole, unbounded document on every final.
    if a.startswith(b):
        return len(b)
    if b.startswith(a):
        return len(a)
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
        holdback: Optional[Holdback] = None,
        format_key: Optional[Callable[[], object]] = None,
        diagnose: Optional[Diagnose] = None,
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
        # Seam handling (see the module docstring). All three are optional: a
        # session without `holdback` formats everything at once (sent text can
        # then only be kept fixed by re-anchoring); without `format_key` a
        # language change is not noticed; without `diagnose` the seam warning
        # names no rule.
        self.holdback = holdback
        self.format_key = format_key
        self.diagnose = diagnose

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
        self._prev_processed = ""          # last emitted document (document-level LocalAgreement)
        self._reset_document()
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
        # Lifecycle of the current utterance as the CLIENT has been told it:
        # None (nothing announced) → "open" → "decoding". Announced utterances
        # always end in exactly one terminal frame (``final`` or
        # ``utterance``/``dropped``) — see _emit_utterance / _end_utterance.
        self._utt_state: Optional[str] = None
        self._final_emitted = False        # per-_finalize: a ``final`` went out
        self._terminal_sent = False        # per-_finalize: ``final`` or ``dropped`` went out
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
        """Force-finalize the current utterance (client 'flush' control message).
        A flush means "give me everything": the finalize releases held-back
        words too, and with no utterance in flight a held tail goes out as a
        release final."""
        if self._in_utterance:
            await self._finalize(forced=True, flush_hold=True)
        # Also after a finalize that dropped the utterance (near-silence,
        # failed decode): those paths never reach the release.
        if self._has_held():
            await self._release_held()

    async def close(self) -> None:
        """Drain: finalize any in-flight utterance and commit the whole document."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._in_utterance:
                try:
                    await self._finalize(forced=True)
                except CloseAbort:
                    raise
                except Exception as exc:  # noqa: BLE001
                    # Same tolerance as the pump: a failed finalize of the
                    # in-flight utterance must not lose the closing document
                    # (the frame that locks everything confirmed so far as
                    # committed). A failed final DECODE never gets here —
                    # _finalize_inner falls back to the partial transcript —
                    # so this is the emit or on_final raising. A postprocess
                    # that raised gets here too but is NOT recovered: the
                    # closing commit below re-runs postprocess over the same
                    # text and raises again, to the route's error handling.
                    # Type only — the message can carry a client-chosen
                    # handshake string.
                    logger.warning("[stream %s] finalize failed on close (%s); "
                                   "committing confirmed text",
                                   self.session_id[:8], type(exc).__name__)
            await self._emit_update(flush_hold=True, flush_all=True, last=True)
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
            # Announce the utterance once it holds enough speech to be decoded
            # at all. Here and not in _run_partial: that one returns early while
            # the consumer is behind realtime, which is exactly when a client
            # most needs to know the server is holding its words.
            if self._utt_state is None and self._speech_ms >= self.cfg.min_speech_ms:
                await self._emit_utterance("open")
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
        # Hard breaks off: nothing else would release held-back words while the
        # speaker stays quiet. Fires once per gap — the release leaves nothing held.
        elif (self.cfg.hard_break_silence_ms == 0
                and not self._in_utterance
                and self._idle_silence_ms >= _IDLE_RELEASE_MS
                and self._has_held()):
            await self._release_held()

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
            if self._preroll_keep_samples == 0:
                self._set_audio(np.zeros(0, dtype=np.float32))
            else:
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

    async def _finalize(self, forced: bool = False, flush_hold: bool = False) -> None:
        # Did this utterance reach the wire as a ``final``? / has the client been
        # given its terminal frame (``final`` or ``dropped``)? Both feed the
        # lifecycle contract: _end_utterance advances the ordinal off the first,
        # the error path below closes the cycle off the second.
        # Instance state rather than return values, because the interesting case
        # is a raise AFTER the final went out (on_final failing): the handlers
        # below still need to know it did.
        self._final_emitted = False
        self._terminal_sent = False
        try:
            await self._finalize_inner(forced, flush_hold)
        except CloseAbort:
            # The route is tearing the session down (revoked credential): say
            # nothing more, exactly as close() does.
            raise
        except Exception:
            # postprocess / emit / on_final failed. Whatever the caller does with
            # the error, a client that was told "open"/"decoding" must not be left
            # waiting on this utterance forever.
            if self._utt_state is not None and not self._terminal_sent:
                try:
                    await self._emit_utterance("dropped", "error")
                except Exception:  # noqa: BLE001 — the sink itself may be what failed
                    pass
            raise
        finally:
            # Never emit from here: teardown cancels the pump mid-await, and an
            # await inside ``finally`` would swallow that cancellation.
            self._end_utterance(self._final_emitted)

    async def _finalize_inner(self, forced: bool, flush_hold: bool = False) -> None:
        """The finalize body. _finalize owns the reset + ordinal advance, so
        every exit from here — return or raise — closes the utterance exactly
        once."""
        audio = self.audio
        decode_failed = False
        # A decode ran but its words were replaced by LocalAgreement's, which
        # are already utterance-absolute (see the trim merge below).
        words_absolute = False
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
                # Nothing to say for it. An utterance the client was told about
                # still gets its terminal frame — a noise burst long enough to
                # pass min_speech_ms but too quiet for the RMS gate lands here.
                if self._utt_state is not None:
                    await self._emit_utterance("dropped", "no_speech")
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
            await self._emit_utterance("decoding")
            t0 = time.perf_counter()
            try:
                raw, words, dropped_all = await self.decode_final(audio.copy(), self._prompt)
            except CloseAbort:
                raise
            except Exception as exc:  # noqa: BLE001
                # A failed final decode used to propagate with the utterance
                # still open: _silence_ms stayed over the commit threshold, so
                # the very next frame re-entered here and decoded the same
                # buffer again — for as long as the fault lasted — while the
                # raise also aborted feed_pcm's frame loop and lost the rest of
                # that chunk. Treat it like the decode that returned nothing:
                # what LocalAgreement already agreed (and the client already
                # showed as partials) is the transcript. Same tolerance, and the
                # same type-only log, as close(): the message can carry a
                # client-chosen handshake string.
                logger.warning("[stream %s] final decode failed (%s) — using the "
                               "partial transcript", self.session_id[:8], type(exc).__name__)
                decode_failed = True
                decoded = False
                tail = self.la.finish()
                raw = self.la.committed_text + self.la.text_of(tail)
                # LocalAgreement words are already utterance-absolute and hold
                # any trim-banked prefix — see the gate path above.
                words = [{"word": w.text, "start": w.start, "end": w.end}
                         for w in (self.la.committed + tail)]
                dropped_all = False
            proc_dur = time.perf_counter() - t0
            if decode_failed:
                if not raw.strip():
                    await self._emit_utterance("dropped", "error")
                    return
            elif not (raw and raw.strip()) and not dropped_all:
                # The final decode produced nothing at all (e.g. its VAD filter trimmed
                # the whole buffer) — fall back to the partial-built LocalAgreement
                # transcript. But when the decode DID produce segments and dropped them
                # all as hallucinations (dropped_all), the empty result is authoritative:
                # the partials run at a fixed temperature and so never trip the drop —
                # i.e. they still hold the hallucination — so keep the empty result.
                # la.committed spans the WHOLE utterance (pop_committed only prunes the
                # agreement buffer), so this path already includes any trim-banked words.
                # The words come from the same source as the text — the decode's
                # own (empty) list would leave the final with no word timestamps,
                # or with only the banked prefix after a trim.
                tail = self.la.finish()
                raw = self.la.committed_text + self.la.text_of(tail)
                words = [{"word": w.text, "start": w.start, "end": w.end}
                         for w in (self.la.committed + tail)]
                words_absolute = True
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
        # The LocalAgreement words (gate path, failed or empty decode) are
        # already utterance-absolute and already include the banked prefix
        # (see above), so only a real decode's buffer-relative words are
        # re-based here.
        if self._trimmed_audio:
            full_audio = np.concatenate([*self._trimmed_audio, audio])
            if decoded and not words_absolute:
                off = self._buffer_offset
                words = self._trimmed_words + [
                    {**w, "start": w["start"] + off, "end": w["end"] + off}
                    for w in words]
        else:
            full_audio = audio
        self.raw_confirmed += raw
        self._prompt = self._make_prompt()
        if await self._emit_update(flush_hold=flush_hold, forced=forced, words=words):
            self._final_emitted = True
            self._terminal_sent = True
        elif self._utt_state is not None:
            # The whole document post-processed to nothing (a lone hallucination
            # the decode dropped, a filler the pipeline strips): no ``final``
            # goes out, so close the cycle explicitly. Only here — close() calls
            # _emit_document too, and its closing document is not an utterance.
            await self._emit_utterance("dropped", "empty")
        if self.on_final is not None:
            await self.on_final({
                "utterance": self._utterance_index,
                "audio_dur": full_audio.shape[0] / self.cfg.sample_rate,
                "trimmed_sec": self._trimmed_sec,
                "decoded": decoded,
                "decode_failed": decode_failed,
                "proc_dur": proc_dur,
                "raw_text": raw,
                "words": words,
                "audio": full_audio,
                "forced": forced,
            })

    async def _hard_break(self) -> None:
        """End the whole grouping after a long silence and start a fresh document,
        without closing the WebSocket.

        Words still held back go out first, as a release final — BEFORE the
        ``boundary`` marker, so they land in the document they belong to. Then
        emits a ``boundary`` marker so the client resets its injection baseline (and
        optionally types ``hard_break_separator`` between documents), then clears the
        cross-utterance accumulators. The rolling prompt is reset too — a long pause
        is treated as a new context; to instead keep terminology across breaks, drop
        the ``self._prompt`` reset below."""
        if self._has_held():
            await self._release_held()
        await self.emit({
            "type": "boundary",
            "utterance": self._utterance_index,
            "separator": self.cfg.hard_break_separator,
        })
        self.raw_confirmed = ""
        self._committed_len = 0
        self._prev_processed = ""
        self._reset_document()
        self._prompt = self.base_prompt.strip()
        self._idle_silence_ms = 0

    # ---- document composition ---------------------------------------------

    def _reset_document(self) -> None:
        """Start a new document: nothing sent, formatting anchored at the start."""
        self._sent = ""                    # the last emitted document (what the client typed)
        self._sent_raw_end = 0             # raw_confirmed[:this] is covered by _sent
        self._anchor_raw = 0               # formatting runs on raw_confirmed[_anchor_raw:] ...
        self._anchor_text = ""             # ... and is joined onto this (the sent text at the re-anchor)
        self._anchor_key = None            # format_key() the anchored text was formatted under

    def _has_held(self) -> bool:
        """Raw words exist that no emitted document covers yet."""
        return bool(self.raw_confirmed[self._sent_raw_end:].strip())

    def _reanchor(self) -> None:
        """Freeze everything sent so far; format the rest of the document on its
        own from here (sticky until the boundary)."""
        self._anchor_raw = self._sent_raw_end
        self._anchor_text = self._sent

    def _format_from_anchor(self, end: int) -> str:
        piece = self.postprocess(self.raw_confirmed[self._anchor_raw:end])
        if not self._anchor_text:
            return piece
        return self._anchor_text + self._seam_join(self._anchor_text, piece)

    def _compose(self, flush_hold: bool = False) -> tuple[str, int]:
        """The next document, and the raw index it covers.

        Formats ``raw_confirmed[_anchor_raw:end]`` — ``end`` excludes the
        held-back words unless ``flush_hold`` — and joins it onto the anchor
        text. The result must extend what was sent; if it does not, the seam
        is logged and the document re-anchored at the sent text."""
        raw = self.raw_confirmed
        key = self.format_key() if self.format_key is not None else None
        if self._sent and self._anchor_key is not None and key != self._anchor_key:
            logger.info("[stream %s] formatting language changed (%s → %s); new text "
                        "formatted from here", self.session_id[:8], self._anchor_key, key)
            self._reanchor()
        self._anchor_key = key
        end = len(raw)
        if not flush_hold and self.holdback is not None:
            try:
                held = self.holdback(raw[self._anchor_raw:])
            except Exception as exc:  # noqa: BLE001 — a hold-back must not break dictation
                logger.warning("[stream %s] hold-back failed (%s); formatting everything",
                               self.session_id[:8], type(exc).__name__)
            else:
                # Never hold back what a release already sent.
                end = max(self._sent_raw_end, self._anchor_raw + held)
        doc = self._format_from_anchor(end)
        if not doc.startswith(self._sent):
            at = _common_prefix_len(doc, self._sent)
            culprit = "?"
            if self.diagnose is not None:
                try:
                    culprit = self.diagnose(raw[self._anchor_raw:self._sent_raw_end],
                                            raw[self._anchor_raw:end])
                except Exception as exc:  # noqa: BLE001 — diagnostics only
                    culprit = f"? ({type(exc).__name__})"
            logger.warning("[stream %s] seam: document diverges from sent text at char "
                           "%d/%d (culprit %s; sent %d chars stay as typed) — formatting "
                           "new text on its own from here", self.session_id[:8], at,
                           len(self._sent), culprit, len(self._sent))
            self._reanchor()
            doc = self._format_from_anchor(end)
        return doc, end

    @staticmethod
    def _seam_join(left: str, piece: str) -> str:
        """What to append to ``left`` (sent text) for ``piece`` (formatted on its
        own): the space the pipeline would have put there, and a capital first
        letter after a sentence end. Casing is only ever raised — the piece
        was formatted without its context and a lower-case start there may be
        the pipeline's deliberate choice, an upper-case one never needs undoing."""
        if not piece:
            return ""
        first = piece[0]
        tail = left.rstrip(" \t")
        if (not left or left[-1] in " \t\n" or piece[0] in " \t\n"
                or left[-1] in _OPENING_BRACKETS or left[-1] in "-/"
                or first in _CLOSING_PUNCT or first in "-/"):
            sep = ""
        elif left[-1] == '"':
            # An odd count means the last quote opened a quotation.
            sep = "" if left.count('"') % 2 == 1 else " "
        elif first == '"':
            # A quote that closes an open quotation hugs the text before it.
            sep = "" if left.count('"') % 2 == 1 else " "
        else:
            sep = " "
        if first.islower() and (not tail or tail[-1] in ".?!\n"):
            piece = first.upper() + piece[1:]
        return sep + piece

    async def _emit_update(self, *, flush_hold: bool = False, **kw) -> bool:
        """Compose the next document and emit it (``kw`` → _emit_document).
        Returns whether a ``final`` went out (see _emit_document)."""
        doc, end = self._compose(flush_hold)
        return await self._emit_document(doc, raw_end=end, **kw)

    async def _release_held(self) -> None:
        """Send the held-back words: a release final (flush, no utterance)."""
        await self._emit_update(flush_hold=True, flush_all=True,
                                utterance=None, flush=True)

    # ---- emission ---------------------------------------------------------

    async def _emit_document(
        self, processed: str, *, forced: bool = False, flush_all: bool = False,
        last: bool = False, words: Optional[list[dict]] = None,
        utterance: object = _CURRENT_UTTERANCE, flush: bool = False,
        raw_end: Optional[int] = None,
    ) -> bool:
        """Emit the post-processed document split into a stable ``committed`` prefix
        and a ``tail``.

        The document itself only grows (``_compose`` guarantees it extends the
        last one), so ``committed`` is truly append-only too. ``tail`` is the
        newest sentence, not yet confirmed by a second pass — a display hint;
        it does not change either. ``flush_all`` (close, release) commits the
        whole document. Both are full strings, not byte deltas — the client
        replaces each region. ``utterance=None`` omits the ordinal (a release
        final belongs to no utterance); ``flush`` marks a release. ``raw_end``
        is the raw index the document covers (from _compose)."""
        commit_len = len(processed) if flush_all else self._stable_commit_len(processed)
        committed = processed[:commit_len]
        tail = processed[commit_len:]
        self._committed_len = commit_len
        self._prev_processed = processed
        self._sent = processed
        if raw_end is not None:
            self._sent_raw_end = raw_end
        if not committed and not tail:
            return False
        msg: dict = {"type": "final"}
        if utterance is _CURRENT_UTTERANCE:
            msg["utterance"] = self._utterance_index
        elif utterance is not None:
            msg["utterance"] = utterance
        msg["committed"] = committed
        msg["tail"] = tail
        if forced:
            msg["forced"] = True
        if last:
            msg["last"] = True
        if flush:
            msg["flush"] = True
        if words:
            msg["words"] = words
        await self.emit(msg)
        return True

    async def _emit_utterance(self, state: str, reason: Optional[str] = None) -> None:
        """Tell the client where the current utterance stands (see the module
        docstring for the contract). ``open``/``decoding`` are sent at most once
        each — a caller that reaches the same state twice is a no-op, so nothing
        upstream can turn this into a per-frame flood."""
        if state in ("open", "decoding"):
            if self._utt_state == state or (state == "open" and self._utt_state is not None):
                return
            self._utt_state = state
        else:
            self._terminal_sent = True
        msg = {"type": "utterance", "utterance": self._utterance_index, "state": state}
        if reason:
            msg["reason"] = reason
        await self.emit(msg)

    def _end_utterance(self, final_emitted: bool) -> None:
        """Close the current utterance: advance the ordinal, then reset.

        The ONE place the ordinal moves. It advances whenever the ordinal may
        have reached the wire — the utterance was announced, or a ``final``
        carried it — and never otherwise. It used to be bumped at the end of the
        finalize body, which a raising postprocess/on_final skipped AFTER
        ``final(N)`` had gone out: the next utterance then reused N, and a
        client pairing ``captured`` receipts by ordinal lost that phrase's."""
        if self._utt_state is not None or final_emitted:
            self._utterance_index += 1
        self._reset_utterance()

    def _stable_commit_len(self, processed: str) -> int:
        """Index up to which ``processed`` is committed.

        Document-level LocalAgreement: commit only through the last sentence
        terminator (``. ? ! \\n``) that lies within the prefix the last *two*
        emitted documents agree on. Since the document only grows (hold-back +
        re-anchor, see _compose) the previous one is always a prefix, so this
        now commits through the last terminator of the PREVIOUS document: the
        newest sentence is reported as ``tail`` for one extra finalize. That
        split is presentation only — nothing in the tail is rewritten later. A
        safety valve commits an over-long tail so it can't grow without bound."""
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
        self._utt_state = None
        self._speech_ms = 0
        self._silence_ms = 0
        self._new_since_partial = 0
        self.endpointer.reset()
