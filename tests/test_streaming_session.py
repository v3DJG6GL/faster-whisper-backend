"""Tests for StreamSession: the append-only final-emission mechanics (the core
correctness claim) and the full PCM → partial → final loop with fake decoders.

Async coroutines are driven directly with asyncio.run() so no pytest-asyncio
plugin is required.
"""

import asyncio

import numpy as np

from streaming_session import StreamConfig, StreamSession
from streaming_vad import EnergyEndpointer


def _make_session(*, postprocess, decode_partial=None, decode_final=None, cfg=None):
    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    async def _dp(audio, prompt):
        return []

    async def _df(audio, prompt):
        return ("", [])

    s = StreamSession(
        config=cfg or StreamConfig(),
        endpointer=EnergyEndpointer(),
        decode_partial=decode_partial or _dp,
        decode_final=decode_final or _df,
        postprocess=postprocess,
        emit=emit,
    )
    return s, msgs


# ---- append-only emission mechanics ---------------------------------------

def test_finals_are_append_only_and_concatenate_to_full_document():
    """Across successive finalizes the emitted deltas never rewrite earlier text
    and concatenate exactly to the post-processed whole document."""
    s, msgs = _make_session(postprocess=lambda raw: raw)  # identity pipeline

    async def run():
        s.raw_confirmed = "der patient hat fieber."
        await s._emit_final_delta(s.postprocess(s.raw_confirmed))
        s.raw_confirmed += " blutdruck normal"          # no terminator yet → held
        await s._emit_final_delta(s.postprocess(s.raw_confirmed))
        s.raw_confirmed += "."                            # terminator → tail releases
        await s._emit_final_delta(s.postprocess(s.raw_confirmed))

    asyncio.run(run())
    deltas = [m["text"] for m in msgs if m["type"] == "final"]
    assert "".join(deltas) == "der patient hat fieber. blutdruck normal."
    # First delta committed the first sentence; "blutdruck" was held until its period.
    assert deltas[0] == "der patient hat fieber."
    assert "blutdruck" not in deltas[0]


def test_held_tail_prevents_premature_dictation_phrase_emission():
    """A multi-word dictation phrase split across utterances ('neue' then 'zeile')
    is never emitted half-resolved — it stays held until the phrase completes."""
    def pp(raw):                       # toy dictation map: phrase → newline
        return raw.replace("neue zeile", "\n")

    s, msgs = _make_session(postprocess=pp)

    async def run():
        s.raw_confirmed = "bla bla neue"              # incomplete phrase, no terminator
        await s._emit_final_delta(s.postprocess(s.raw_confirmed))
        s.raw_confirmed = "bla bla neue zeile text."  # phrase completes + terminator
        await s._emit_final_delta(s.postprocess(s.raw_confirmed))

    asyncio.run(run())
    deltas = [m["text"] for m in msgs if m["type"] == "final"]
    assert "neue" not in "".join(deltas)              # literal "neue" never leaked
    assert "".join(deltas) == "bla bla \n text."


def test_close_flushes_held_unterminated_tail():
    """An utterance with no sentence terminator is held during the session but
    flushed on close()."""
    s, msgs = _make_session(postprocess=lambda raw: raw)

    async def run():
        s.raw_confirmed = "hallo welt"                # no terminator
        await s._emit_final_delta(s.postprocess(s.raw_confirmed))
        assert [m for m in msgs if m["type"] == "final"] == []  # held
        await s.close()

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"] == "hallo welt"
    assert finals[0].get("last") is True


# ---- full PCM loop --------------------------------------------------------

def _pcm(level: int, ms: int, sample_rate: int = 16000) -> bytes:
    n = sample_rate * ms // 1000
    return (np.full(n, level, dtype="<i2")).tobytes()


def test_pcm_loop_emits_partials_then_a_final_after_silence():
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )

    async def decode_partial(audio, prompt):
        return [(0.0, 0.3, " hallo"), (0.3, 0.6, " welt")]

    async def decode_final(audio, prompt):
        return ("hallo welt.", [])

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )

    async def run():
        await s.feed_pcm(_pcm(8000, 500))   # ~0.5 s speech (loud)
        await s.feed_pcm(_pcm(0, 400))      # ~0.4 s silence → finalize

    asyncio.run(run())
    partials = [m for m in msgs if m["type"] == "partial"]
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(partials) >= 1
    # LocalAgreement commits the repeated hypothesis → committed text appears.
    assert any("welt" in m["committed"] for m in partials)
    assert len(finals) == 1
    assert finals[0]["text"] == "hallo welt."
    assert finals[0]["append"] is True


def test_no_partial_decode_storm_during_trailing_silence():
    """Regression: trailing silence must NOT trigger a partial decode per frame.
    The old inner-pause trigger fired one (synchronous) decode per 32 ms silent
    frame, advancing the silence timer ~1 frame per decode and inflating the
    commit wait to ~20 s. Here ~1 s of silence (≈31 frames) must cost only a
    couple of decodes, not dozens."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=2000,
        min_speech_ms=64, forced_commit_sec=100, rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    calls = {"partial": 0}

    async def decode_partial(audio, prompt):
        calls["partial"] += 1
        return [(0.0, 0.2, " x")]

    async def decode_final(audio, prompt):
        return ("x.", [])

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )

    async def run():
        await s.feed_pcm(_pcm(8000, 300))   # 0.3 s speech
        await s.feed_pcm(_pcm(0, 1000))     # 1.0 s silence (≈31 frames) → no finalize yet

    asyncio.run(run())
    # A handful of speech-phase partials only; the silence must add ~none.
    assert calls["partial"] <= 6, f"partial decode storm: {calls['partial']} decodes"
    assert [m for m in msgs if m["type"] == "final"] == []  # held (silence < commit)


def test_silence_only_input_never_finalizes_or_hallucinates():
    cfg = StreamConfig(commit_silence_ms=192, min_speech_ms=64, rms_gate_dbfs=-50)
    called = {"partial": 0, "final": 0}

    async def decode_partial(audio, prompt):
        called["partial"] += 1
        return [(0.0, 0.2, " x")]

    async def decode_final(audio, prompt):
        called["final"] += 1
        return ("x", [])

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )
    asyncio.run(s.feed_pcm(_pcm(0, 1000)))   # 1 s of pure silence
    assert called["partial"] == 0
    assert called["final"] == 0
    assert msgs == []
