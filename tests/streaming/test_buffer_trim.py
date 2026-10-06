"""Trim-safety: a continuous utterance longer than buffer_trim_sec must not
lose its opening.

_maybe_trim cuts the front of the audio buffer (anchored at a committed word)
to bound partial-decode latency. The cut words' audio is gone, so the final
decode can only re-hear the remaining buffer — its result used to REPLACE the
whole utterance, deleting the already-committed-and-shown opening (observed
live: a 16 s dictation losing its first clause). The fix banks the cut words'
text, folds it into the rolling prompt (seam context), and prepends it to the
final decode's result; captures pair the buffer audio with the buffer-aligned
text only.
"""

import asyncio

import numpy as np

from tests._streaming_helpers import const_pcm
from faster_whisper_backend.streaming.session import StreamConfig, StreamSession
from faster_whisper_backend.streaming.vad import FRAME_MS, FRAME_SAMPLES, EnergyEndpointer

SR = 16000
WORD_SEC = 0.2  # deterministic word grid: word i spans [i*0.2, (i+1)*0.2)


_pcm = const_pcm


def _grid_words(start_s: float, end_s: float):
    """Absolute word grid over the utterance timeline."""
    first = int(round(start_s / WORD_SEC))
    last = int(end_s / WORD_SEC)
    return [(i * WORD_SEC, (i + 1) * WORD_SEC, f" w{i}")
            for i in range(first, last)]


def _run_long_utterance(speech_ms: int):
    """Drive a session through one continuous utterance of speech_ms, with a
    buffer trim configured to fire mid-utterance. The fake decoders transcribe
    exactly the buffer they receive (like the real model: they cannot re-hear
    trimmed audio)."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    finals_meta: list[dict] = []

    async def on_final(info):
        finals_meta.append(info)

    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    session_box: list[StreamSession] = []

    async def decode_partial(audio, prompt):
        off = session_box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]

    async def decode_final(audio, prompt):
        off = session_box[0]._buffer_offset
        dur = audio.shape[0] / SR
        gw = _grid_words(off, off + dur)
        raw = "".join(t for _, _, t in gw)
        # buffer-relative word dicts, like the real decoder
        words = [{"word": t, "start": a - off, "end": b - off} for a, b, t in gw]
        return (raw, words, False)

    s = StreamSession(
        config=cfg, endpointer=EnergyEndpointer(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit, on_final=on_final,
    )
    session_box.append(s)

    # _trimmed_text is cleared by the finalize, so watch it live while feeding.
    trimmed: list[str] = []

    async def run():
        for level, ms in ((8000, speech_ms), (0, 400)):  # speech, then silence → finalize
            for _ in range(ms // FRAME_MS):
                await s.feed_pcm(_pcm(level, FRAME_MS))
                if s._trimmed_text:
                    trimmed.append(s._trimmed_text)

    asyncio.run(run())
    return s, msgs, finals_meta, trimmed


def test_trim_preserves_committed_opening_in_final_document():
    s, msgs, finals_meta, trimmed = _run_long_utterance(speech_ms=3500)
    assert finals_meta, "no final emitted"
    assert trimmed, "trim never banked text — the helper's live watch is broken"
    info = finals_meta[-1]
    # Sanity: the trim actually fired (otherwise this test regressed to the
    # short-utterance case and asserts nothing).
    assert info["trimmed_sec"] > 0.0
    # raw_text (document) must contain the utterance's FIRST word — the audio
    # for it was trimmed away mid-utterance, so only the banked committed text
    # can supply it.
    assert " w0" in info["raw_text"], (
        f"opening lost after buffer trim: {info['raw_text']!r}")
    # No duplication at the seam: every grid word appears exactly once.
    words = info["raw_text"].split()
    assert len(words) == len(set(words)), f"duplicated words: {info['raw_text']!r}"
    # The words are in order and contiguous (w0, w1, ..., wN).
    assert words == [f"w{i}" for i in range(len(words))]
    # The emitted document (committed + tail of the last final) matches.
    finals = [m for m in msgs if m["type"] == "final"]
    doc = finals[-1]["committed"] + finals[-1]["tail"]
    assert "w0" in doc

    # Capture alignment: on_final reassembles the WHOLE utterance — the banked
    # audio slices plus the remaining buffer — so audio, audio_dur, raw_text
    # and words all span the full ~3.5 s of speech fed (a trimmed buffer alone
    # would be shorter than the trim threshold's keep window).
    assert info["audio_dur"] >= 3.2
    assert abs(info["audio"].shape[0] / SR - info["audio_dur"]) < 1e-6
    # Words merge banked (absolute) + decode (shifted) entries: the utterance's
    # first word is present, times are monotonic, and there's no duplication.
    wtexts = [w["word"].strip() for w in info["words"]]
    assert wtexts and wtexts[0] == "w0"
    assert len(wtexts) == len(set(wtexts))
    starts = [w["start"] for w in info["words"]]
    assert starts == sorted(starts)


def test_trim_folds_cut_words_into_rolling_prompt():
    """After a trim the decodes see a mid-sentence buffer; the cut words must
    ride the prompt so the seam decodes with context."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    final_prompts: list[str] = []
    session_box: list[StreamSession] = []

    async def decode_partial(audio, prompt):
        off = session_box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]

    async def decode_final(audio, prompt):
        final_prompts.append(prompt)
        return (" tail.", [], False)

    async def emit(m):
        pass

    s = StreamSession(
        config=cfg, endpointer=EnergyEndpointer(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit,
    )
    session_box.append(s)

    async def run():
        await s.feed_pcm(_pcm(8000, 3500))
        await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    assert final_prompts, "final decode never ran"
    assert "w0" in final_prompts[-1], (
        f"trimmed words missing from decode prompt: {final_prompts[-1]!r}")


def test_short_utterance_unchanged_no_trim():
    """Below buffer_trim_sec nothing is banked: the final decode's text stands
    alone, no trim is reported, and the payload audio is the plain buffer
    (pre-fix behaviour preserved)."""
    s, msgs, finals_meta, trimmed = _run_long_utterance(speech_ms=800)
    info = finals_meta[-1]
    assert info["trimmed_sec"] == 0.0
    assert abs(info["audio"].shape[0] / SR - info["audio_dur"]) < 1e-6
    assert trimmed == [], "nothing should ever have been banked"


# ---- hard ceiling ---------------------------------------------------------
#
# _maybe_trim runs from _run_partial only, behind the skip-partials / speech /
# RMS gates, and needs a committed word to anchor its cut — so it is not a bound
# on the buffer, just an optimisation of the usual case. max_buffer_sec is the
# bound: checked on every frame, before every gate.


class _AlwaysSpeech:
    """VAD stuck on: background noise (room mic, HVAC, music) whose flicker keeps
    resetting commit_silence_ms, so the utterance never ends on its own."""

    def is_speech(self, frame):
        return True

    def reset(self) -> None:
        pass


def _drive_past_ceiling(cfg, level: int, feed_ms: int, *, skip_partials: bool = False):
    """Feed one frame at a time so the buffer can be sampled after EVERY frame,
    with the VAD stuck on so nothing else can finalize the utterance.

    Returns (session, msgs, info) where info holds the peak buffer duration seen,
    the ``forced`` flag of every _finalize entered, and the duration of each
    buffer that reached the final decode."""
    msgs: list[dict] = []
    decoded: list[float] = []

    async def emit(m):
        msgs.append(m)

    async def decode_partial(audio, prompt):
        return []

    async def decode_final(audio, prompt):
        decoded.append(audio.shape[0] / SR)
        return (" chunk.", [], False)

    s = StreamSession(
        config=cfg, endpointer=_AlwaysSpeech(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit,
    )
    s._skip_partials = skip_partials
    seen: list[bool] = []
    _finalize = s._finalize

    async def _spy(forced: bool = False):
        seen.append(forced)
        await _finalize(forced=forced)

    s._finalize = _spy
    peak = 0.0

    async def run():
        nonlocal peak
        for _ in range(feed_ms // FRAME_MS):
            await s.feed_pcm(_pcm(level, FRAME_MS))
            peak = max(peak, s.audio.shape[0] / SR)
        await s.close()

    asyncio.run(run())
    return s, msgs, {"peak_sec": peak, "forced": seen, "decoded": decoded}


def test_ceiling_bounds_the_buffer_when_the_rms_gate_blocks_the_trim():
    """Near-silence keeps rms_dbfs(self.audio) under the gate, so _run_partial —
    and with it _maybe_trim — never runs; the more silence accumulates the
    further the whole-buffer RMS sinks, so the trim can never recover. The
    ceiling still bounds the buffer, at the cost of a forced finalize."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-42.0, preroll_keep_ms=100, max_buffer_sec=2.0,
    )
    s, msgs, info = _drive_past_ceiling(cfg, level=1, feed_ms=8000)
    # Sanity: we fed far more than the ceiling, and nothing else could have
    # ended the utterance (VAD stuck on, forced_commit_sec 100 s away).
    assert info["forced"], "no finalize fired — the buffer grew unbounded"
    assert all(info["forced"]), "ceiling finalize must be flagged forced"
    assert info["peak_sec"] <= cfg.max_buffer_sec + FRAME_MS / 1000 + 1e-9, (
        f"buffer exceeded the ceiling: {info['peak_sec']:.3f} s")
    assert s.audio.shape[0] / SR <= cfg.max_buffer_sec
    # A buffer this quiet is dropped by the existing anti-hallucination gate
    # inside _finalize, so it never reaches the decoder — the ceiling reuses
    # that path rather than adding a second discard rule.
    assert info["decoded"] == []


def test_ceiling_force_finalizes_through_the_decoder_when_partials_are_skipped():
    """Consumer behind realtime (_skip_partials) is the other way _maybe_trim
    stops running. Here the audio is loud, so the forced finalize goes through
    the real decode: the buffer is bounded AND every fed sample is transcribed."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=100, buffer_trim_keep_sec=10,
        rms_gate_dbfs=-60, preroll_keep_ms=100, max_buffer_sec=2.0,
    )
    feed_ms = 7000
    s, msgs, info = _drive_past_ceiling(cfg, level=8000, feed_ms=feed_ms,
                                        skip_partials=True)
    assert info["peak_sec"] <= cfg.max_buffer_sec + FRAME_MS / 1000 + 1e-9, (
        f"buffer exceeded the ceiling: {info['peak_sec']:.3f} s")
    assert len(info["decoded"]) >= 3, "ceiling never force-finalized"
    # Nothing the user said is dropped: the successive forced finalizes decode
    # the whole fed stream (minus the sub-frame remainder feed_pcm still holds).
    fed_sec = (feed_ms // FRAME_MS) * FRAME_MS / 1000
    assert abs(sum(info["decoded"]) - fed_sec) < 1e-6
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals and any(m.get("forced") for m in finals)
    assert s.raw_confirmed.count("chunk") == len(info["decoded"])


def test_ceiling_finalize_keeps_trim_banked_text_when_the_rms_gate_trips():
    """Loud speech long enough for a trim to bank committed words, then quiet
    frames (VAD stuck on) pile up until the max_buffer_sec ceiling forces a
    finalize whose LIVE buffer is near-silence. The RMS gate must skip the
    decode only — the already-committed (and already-displayed) text has to be
    emitted, not silently discarded with the quiet tail."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-42.0, preroll_keep_ms=100, max_buffer_sec=6.0,
    )
    msgs: list[dict] = []
    finals_meta: list[dict] = []
    final_decodes: list[float] = []
    session_box: list[StreamSession] = []

    async def emit(m):
        msgs.append(m)

    async def on_final(info):
        finals_meta.append(info)

    async def decode_partial(audio, prompt):
        off = session_box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]

    async def decode_final(audio, prompt):
        final_decodes.append(audio.shape[0] / SR)
        return (" should-not-run.", [], False)

    s = StreamSession(
        config=cfg, endpointer=_AlwaysSpeech(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit, on_final=on_final,
    )
    session_box.append(s)

    async def run():
        # 3.5 s loud: partials commit grid words, a trim banks the opening.
        for _ in range(3500 // FRAME_MS):
            await s.feed_pcm(_pcm(8000, FRAME_MS))
        assert s._trimmed_text, "setup failed: no trim banked any text"
        # Quiet frames until the ceiling fires (VAD stuck on, so nothing else
        # can finalize; once the live buffer is quiet the RMS gate also stops
        # the partials, and with them the trim).
        for _ in range(5000 // FRAME_MS):
            await s.feed_pcm(_pcm(1, FRAME_MS))
            if finals_meta:
                break

    asyncio.run(run())
    assert finals_meta, "the ceiling never forced a finalize"
    info = finals_meta[0]
    assert info["forced"] is True
    assert info["trimmed_sec"] > 0.0  # sanity: the trim really banked audio
    # The gate skipped the decode…
    assert final_decodes == []
    # …but the committed text (incl. the trim-banked opening word) survives.
    assert " w0" in info["raw_text"], (
        f"banked opening lost to the near-silence gate: {info['raw_text']!r}")
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals, "no final document emitted"
    doc = finals[-1]["committed"] + finals[-1]["tail"]
    assert "w0" in doc


# ---- buffer accounting ----------------------------------------------------
#
# The buffer takes a chunk per 32 ms frame but is joined lazily, on read. The
# sample counter — what the ceiling above and both trims consult, so that asking
# for a length never forces a join — must therefore stay exactly in step with
# the array the property hands out.


def _quiet_session(**over):
    """A session with inert decoders: for exercising the buffer directly."""
    async def decode_partial(audio, prompt):
        return []

    async def decode_final(audio, prompt):
        return ("", [], False)

    async def emit(m):
        pass

    return StreamSession(
        config=StreamConfig(**over), endpointer=EnergyEndpointer(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit,
    )


def test_lazy_buffer_reads_back_exactly_what_was_appended():
    s = _quiet_session()
    rng = np.random.default_rng(0)
    chunks = [rng.standard_normal(FRAME_SAMPLES).astype(np.float32)
              for _ in range(500)]
    for i, c in enumerate(chunks, 1):
        s._append_audio(c)
        assert s._audio_samples == i * FRAME_SAMPLES  # length, no join
        if i % 31 == 0:  # ~the partial cadence: the join folds in the pending chunks
            assert np.array_equal(s.audio, np.concatenate(chunks[:i]))
    assert np.array_equal(s.audio, np.concatenate(chunks))
    assert s.audio.shape[0] == s._audio_samples
    # Re-reading with no append in between returns the cache, not a rebuild.
    assert s.audio is s.audio


def test_buffer_counter_survives_preroll_trim_and_reset():
    s = _quiet_session(preroll_keep_ms=100)
    for _ in range(20):
        s._append_audio(np.ones(FRAME_SAMPLES, dtype=np.float32))
    s._trim_preroll()
    assert s._audio_samples == s._preroll_keep_samples == s.audio.shape[0]
    s._reset_utterance()
    assert s._audio_samples == 0
    assert s.audio.shape[0] == 0
    s._append_audio(np.full(FRAME_SAMPLES, 0.5, dtype=np.float32))
    assert s._audio_samples == FRAME_SAMPLES
    assert np.array_equal(s.audio, np.full(FRAME_SAMPLES, 0.5, dtype=np.float32))


def test_buffer_counter_matches_the_array_after_every_frame():
    """Drive the real pump through speech, a mid-utterance trim, a finalize and
    trailing silence — every writer must leave counter and array agreeing."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    session_box: list[StreamSession] = []

    async def decode_partial(audio, prompt):
        off = session_box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]

    async def decode_final(audio, prompt):
        return (" tail.", [], False)

    async def emit(m):
        pass

    s = StreamSession(
        config=cfg, endpointer=EnergyEndpointer(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit,
    )
    session_box.append(s)

    trimmed = []  # _trimmed_text is cleared by the finalize, so watch it live

    async def run():
        for level, ms in ((8000, 3500), (0, 400), (8000, 500), (0, 400)):
            for _ in range(ms // FRAME_MS):
                await s.feed_pcm(_pcm(level, FRAME_MS))
                assert s._audio_samples == s.audio.shape[0]
                if s._trimmed_text:
                    trimmed.append(s._trimmed_text)

    asyncio.run(run())
    assert trimmed, "trim never fired — the trim writer went untested"
