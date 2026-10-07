"""Tests for StreamSession: the append-only final-emission mechanics (the core
correctness claim) and the full PCM → partial → final loop with fake decoders.

Async coroutines are driven directly with asyncio.run() so no pytest-asyncio
plugin is required.
"""

import asyncio

import pytest

from tests._streaming_helpers import const_pcm
from faster_whisper_backend.streaming.session import StreamConfig, StreamSession
from faster_whisper_backend.streaming.vad import EnergyEndpointer


def _make_session(*, postprocess, decode_partial=None, decode_final=None, cfg=None,
                  **seams):
    """``seams``: the optional holdback / format_key / diagnose hooks."""
    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    async def _dp(audio, prompt):
        return []

    async def _df(audio, prompt):
        return ("", [], False)

    s = StreamSession(
        config=cfg or StreamConfig(),
        endpointer=EnergyEndpointer(),
        decode_partial=decode_partial or _dp,
        decode_final=decode_final or _df,
        postprocess=postprocess,
        emit=emit,
        **seams,
    )
    return s, msgs


def _hold_trailing(word):
    """Toy hold-back: hold a trailing ``word`` (like the real spec holds the
    first word of a multi-word dictation key)."""
    def holdback(raw):
        stripped = raw.rstrip()
        if stripped.split()[-1:] == [word]:
            return len(stripped) - len(word)
        return len(raw)
    return holdback


# ---- committed/tail emission mechanics ------------------------------------

def test_committed_is_append_only_and_document_equals_postprocess():
    """Across successive finalizes the committed prefix only ever grows (never
    rewrites earlier text), and committed+tail always equals the post-processed
    whole document."""
    s, msgs = _make_session(postprocess=lambda raw: raw)  # identity pipeline

    async def run():
        s.raw_confirmed = "der patient hat fieber."
        await s._emit_update()
        s.raw_confirmed += " blutdruck normal"          # no terminator yet → tail
        await s._emit_update()
        s.raw_confirmed += "."                            # terminator (commits on close)
        await s._emit_update()
        await s.close()

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    committeds = [m["committed"] for m in finals]
    # append-only: each committed extends the previous, never rewrites it.
    for a, b in zip(committeds, committeds[1:]):
        assert b.startswith(a), f"committed rewritten: {a!r} -> {b!r}"
    # committed+tail reconstructs the post-processed document at every step.
    for m in finals:
        assert m["committed"] + m["tail"] in ("der patient hat fieber.",
                                              "der patient hat fieber. blutdruck normal",
                                              "der patient hat fieber. blutdruck normal.")
    # the final committed (after close) equals the whole post-processed transcript.
    assert committeds[-1] == "der patient hat fieber. blutdruck normal."
    # "blutdruck" is shown live (in some tail) but never committed mid-sentence
    # until it stabilises on close.
    assert all("blutdruck" not in c for c in committeds[:-1])
    assert any("blutdruck" in m["tail"] for m in finals)


def test_committed_never_holds_half_resolved_dictation_phrase():
    """A multi-word dictation phrase split across utterances ('neue' then 'zeile')
    is never sent half-resolved — the hold-back keeps the literal 'neue' out of
    every document until the phrase completes."""
    def pp(raw):                       # toy dictation map: phrase → newline
        return raw.replace("neue zeile", "\n")

    s, msgs = _make_session(postprocess=pp, holdback=_hold_trailing("neue"))

    async def run():
        s.raw_confirmed = "bla bla neue"              # incomplete phrase, no terminator
        await s._emit_update()
        s.raw_confirmed = "bla bla neue zeile text."  # phrase completes + terminator
        await s._emit_update()
        await s.close()

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    committeds = [m["committed"] for m in finals]
    assert all("neue" not in c for c in committeds)   # literal "neue" never committed
    assert all("neue" not in m["tail"] for m in finals)  # ... nor shown
    assert committeds[-1] == "bla bla \n text."


def test_close_commits_unterminated_tail():
    """An utterance with no sentence terminator is shown as a provisional tail
    during the session, then committed on close()."""
    s, msgs = _make_session(postprocess=lambda raw: raw)

    async def run():
        s.raw_confirmed = "hallo welt"                # no terminator
        await s._emit_update()
        pre = [m for m in msgs if m["type"] == "final"]
        # shown immediately, but provisional (not committed) — fixes the "text only
        # appears after the next utterance" bug.
        assert pre and pre[-1]["committed"] == "" and pre[-1]["tail"] == "hallo welt"
        await s.close()

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals[-1]["committed"] == "hallo welt"
    assert finals[-1]["tail"] == ""
    assert finals[-1].get("last") is True


def test_close_falls_back_on_a_failed_final_decode_and_still_commits(caplog):
    """A decode error in the in-flight utterance's final decode on close() is
    absorbed by _finalize_inner's decode_failed fallback (the partial
    transcript), so close() never sees it: the closing document still commits
    the confirmed text. close()'s own tolerance is pinned by the next test.
    The frames alone are identical either way, so the logs tell them apart."""
    import logging

    async def _df(audio, prompt):
        raise RuntimeError("CUDA out of memory")

    s, msgs = _make_session(postprocess=lambda raw: raw, decode_final=_df,
                            cfg=StreamConfig(min_speech_ms=0, rms_gate_dbfs=-200.0))

    async def run():
        s.raw_confirmed = "erster satz."
        await s._emit_update()
        await s.feed_pcm(const_pcm(8000, 1000))
        assert s._in_utterance
        with caplog.at_level(logging.WARNING, logger="faster_whisper_backend.streaming.session"):
            await s.close()

    asyncio.run(run())
    assert msgs[-1]["type"] == "final"
    assert msgs[-1]["committed"] == "erster satz." and msgs[-1]["tail"] == ""
    assert msgs[-1].get("last") is True
    assert "final decode failed (RuntimeError)" in caplog.text
    assert "finalize failed on close" not in caplog.text


def test_close_survives_failing_on_final_and_still_commits(caplog):
    """The finalize failures close() tolerates: the emit or on_final raising
    during the drain finalize. Logged, not raised — the closing document
    still commits the confirmed text. (A raising postprocess is not among
    them: the closing commit re-runs it over the same text.)"""
    import logging

    s, msgs = _make_session(postprocess=lambda raw: raw,
                            cfg=StreamConfig(min_speech_ms=0, rms_gate_dbfs=-200.0))

    async def _boom(info):
        raise RuntimeError("capture store down")
    s.on_final = _boom

    async def run():
        s.raw_confirmed = "erster satz."
        await s._emit_update()
        await s.feed_pcm(const_pcm(8000, 1000))
        assert s._in_utterance
        with caplog.at_level(logging.WARNING, logger="faster_whisper_backend.streaming.session"):
            await s.close()

    asyncio.run(run())
    assert msgs[-1]["type"] == "final"
    assert "erster satz." in msgs[-1]["committed"]
    assert msgs[-1].get("last") is True
    assert any("finalize failed on close (RuntimeError)" in r.getMessage()
               for r in caplog.records)


def test_close_abort_still_propagates_without_closing_document():
    """CloseAbort (the route's credential-revocation guard) keeps its contract:
    close() re-raises and emits no closing document."""
    from faster_whisper_backend.streaming.session import CloseAbort

    async def _df(audio, prompt):
        raise CloseAbort("credential revoked mid-session")

    s, msgs = _make_session(postprocess=lambda raw: raw, decode_final=_df,
                            cfg=StreamConfig(min_speech_ms=0, rms_gate_dbfs=-200.0))

    async def run():
        s.raw_confirmed = "erster satz."
        await s.feed_pcm(const_pcm(8000, 1000))
        with pytest.raises(CloseAbort):
            await s.close()

    asyncio.run(run())
    assert not [m for m in msgs if m.get("last")]


# ---- full PCM loop --------------------------------------------------------

_pcm = const_pcm


def test_pcm_loop_emits_partials_then_a_final_after_silence():
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )

    async def decode_partial(audio, prompt):
        return [(0.0, 0.3, " hallo"), (0.3, 0.6, " welt")]

    async def decode_final(audio, prompt):
        return ("hallo welt.", [], False)

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
    # one finalize, no terminator-agreement yet → the text is shown as the
    # provisional tail (committed + tail reconstructs the post-processed utterance).
    assert finals[0]["committed"] + finals[0]["tail"] == "hallo welt."


def test_hard_break_resets_document_after_long_silence():
    """A silence longer than hard_break_silence_ms ends the whole grouping: emit a
    `boundary` marker (carrying the separator) and reset the accumulated document,
    without closing the connection. Fires once per quiet gap."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
        hard_break_silence_ms=500, hard_break_separator="\n",
    )

    async def decode_final(audio, prompt):
        return ("hallo welt.", [], False)

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_final=decode_final, cfg=cfg,
    )

    async def run():
        await s.feed_pcm(_pcm(8000, 300))   # speech → one utterance
        await s.feed_pcm(_pcm(0, 700))      # silence: finalize (192 ms) then hard break (500 ms)

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    boundaries = [m for m in msgs if m["type"] == "boundary"]
    assert len(finals) == 1                       # one finalize before the break
    assert len(boundaries) == 1                   # exactly one break (raw_confirmed guard)
    assert boundaries[0]["separator"] == "\n"
    assert s.raw_confirmed == ""                  # document reset → fresh grouping next
    assert s._committed_len == 0
    assert s._prev_processed == ""


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
        return ("x.", [], False)

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
        return ("x", [], False)

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )
    asyncio.run(s.feed_pcm(_pcm(0, 1000)))   # 1 s of pure silence
    assert called["partial"] == 0
    assert called["final"] == 0
    assert msgs == []


def test_skip_partials_suppresses_partial_decode_but_keeps_finals():
    """B1 backpressure: when the streaming route signals it's behind realtime
    (_skip_partials), the session skips the EXPENSIVE partial decode but still
    runs finals on silence — audio/VAD stay intact so we catch up without losing
    the utterance. Clearing the flag resumes partials."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    calls = {"partial": 0, "final": 0}

    async def decode_partial(audio, prompt):
        calls["partial"] += 1
        return [(0.0, 0.2, " x")]

    async def decode_final(audio, prompt):
        calls["final"] += 1
        return ("x.", [], False)

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )

    async def run():
        s._skip_partials = True               # behind realtime
        await s.feed_pcm(_pcm(8000, 500))     # speech — partials would normally fire
        await s.feed_pcm(_pcm(0, 400))        # silence → finalize (NOT skipped)
        assert calls["partial"] == 0, "partials must be skipped when behind"
        assert calls["final"] == 1, "finals must still run when behind"
        s._skip_partials = False              # caught up → partials resume
        await s.feed_pcm(_pcm(8000, 500))
        await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    assert calls["partial"] >= 1, "partials must resume once caught up"
    assert calls["final"] == 2


# ---- anti-hallucination: all-dropped final vs empty-decode fallback --------

def test_all_dropped_final_keeps_empty_and_does_not_resurrect_partials():
    """When the FINAL decode drops every segment as a hallucination (empty raw,
    dropped_all=True), the utterance must stay empty. The partial-built
    LocalAgreement buffer ran at a fixed temperature and so never tripped the
    drop — it still holds that hallucination — so the empty-decode fallback must
    NOT resurrect it."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )

    async def decode_partial(audio, prompt):
        return [(0.0, 0.3, " thank"), (0.3, 0.6, " you")]   # the hallucination

    async def decode_final(audio, prompt):
        return ("", [], True)   # decode produced segments, dropped them ALL

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )

    async def run():
        await s.feed_pcm(_pcm(8000, 500))   # speech → partials commit "thank you"
        await s.feed_pcm(_pcm(0, 400))      # silence → finalize (all-dropped)
        await s.close()

    asyncio.run(run())
    # the partials DID commit the hallucination into LocalAgreement...
    assert any("thank" in m["committed"] for m in msgs if m["type"] == "partial")
    # ...but the all-dropped final must not let it back into the document.
    assert s.raw_confirmed == ""
    finals = [m for m in msgs if m["type"] == "final"]
    assert all("thank" not in (m["committed"] + m["tail"]) for m in finals)


def test_empty_final_still_falls_back_to_localagreement():
    """The all-dropped fix must PRESERVE the legitimate fallback: when the FINAL
    decode produces nothing at all (no segments, dropped_all=False — e.g. its VAD
    filter trimmed the buffer), the partial-built transcript is still used so a
    momentary final-decode miss doesn't silently drop spoken text."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )

    async def decode_partial(audio, prompt):
        return [(0.0, 0.3, " hallo"), (0.3, 0.6, " welt")]

    async def decode_final(audio, prompt):
        return ("", [], False)   # genuinely empty decode (no segments)

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        await s.feed_pcm(_pcm(0, 400))      # silence → finalize (empty → fall back)

    asyncio.run(run())
    assert "welt" in s.raw_confirmed        # recovered from the partials' transcript


# ---- hold-back, release finals, re-anchor ----------------------------------

_FAST = dict(min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
             min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
             rms_gate_dbfs=-60, preroll_keep_ms=100)


def _neue_zeile(raw):
    """Toy pipeline: 'neue zeile' → newline, edges trimmed like trim-edges."""
    return raw.replace(" neue zeile", "\n").strip()


def _queued_final(texts):
    """A decode_final returning each item in turn; an item is the text, or a
    ``(text, dropped_all)`` pair for a decode that dropped its segments."""
    queue = list(texts)

    async def decode_final(audio, prompt):
        item = queue.pop(0)
        text, dropped_all = item if isinstance(item, tuple) else (item, False)
        return (text, [], dropped_all)
    return decode_final


def _finals(msgs):
    return [m for m in msgs if m["type"] == "final"]


def test_held_words_are_released_before_the_boundary():
    """A held trailing word goes out as a release final (flush, no utterance)
    BEFORE the boundary marker, so it lands in the document it belongs to."""
    s, msgs = _make_session(
        postprocess=_neue_zeile, decode_final=_queued_final(["bla neue"]),
        cfg=StreamConfig(**_FAST, hard_break_silence_ms=500),
        holdback=_hold_trailing("neue"))

    async def run():
        await s.feed_pcm(_pcm(8000, 300))
        await s.feed_pcm(_pcm(0, 700))      # finalize, then the hard break

    asyncio.run(run())
    # Lifecycle ``utterance`` frames (open/decoding) are not what this test is about.
    kinds = [(m["type"], m.get("flush", False)) for m in msgs
             if m["type"] not in ("partial", "utterance")]
    assert kinds == [("final", False), ("final", True), ("boundary", False)]
    held, release = _finals(msgs)
    assert held["committed"] + held["tail"] == "bla"
    assert release["committed"] == "bla neue" and release["tail"] == ""
    assert "utterance" not in release and "utterance" in held
    assert s._sent == "" and s._sent_raw_end == 0    # new document


def test_close_releases_held_words_in_the_last_final():
    s, msgs = _make_session(
        postprocess=_neue_zeile, decode_final=_queued_final(["bla neue"]),
        cfg=StreamConfig(**_FAST, hard_break_silence_ms=0),
        holdback=_hold_trailing("neue"))

    async def run():
        await s.feed_pcm(_pcm(8000, 300))
        await s.feed_pcm(_pcm(0, 400))
        await s.close()

    asyncio.run(run())
    finals = _finals(msgs)
    assert finals[0]["committed"] + finals[0]["tail"] == "bla"
    assert finals[-1]["committed"] == "bla neue" and finals[-1].get("last") is True
    assert "utterance" in finals[-1]


def test_flush_releases_held_words_between_utterances():
    """A client flush with no utterance in flight releases the held tail as a
    release final; a second flush has nothing left to send."""
    s, msgs = _make_session(
        postprocess=_neue_zeile, decode_final=_queued_final(["bla neue"]),
        cfg=StreamConfig(**_FAST, hard_break_silence_ms=0),
        holdback=_hold_trailing("neue"))

    async def run():
        await s.feed_pcm(_pcm(8000, 300))
        await s.feed_pcm(_pcm(0, 400))
        await s.flush_utterance()
        await s.flush_utterance()

    asyncio.run(run())
    finals = _finals(msgs)
    assert len(finals) == 2
    assert finals[1] == {"type": "final", "committed": "bla neue", "tail": "", "flush": True}


def test_flush_in_utterance_finalizes_with_the_hold_released():
    s, msgs = _make_session(
        postprocess=_neue_zeile, decode_final=_queued_final(["bla neue"]),
        cfg=StreamConfig(**_FAST, hard_break_silence_ms=0),
        holdback=_hold_trailing("neue"))

    async def run():
        await s.feed_pcm(_pcm(8000, 300))
        assert s._in_utterance
        await s.flush_utterance()

    asyncio.run(run())
    (final,) = _finals(msgs)
    assert final["committed"] + final["tail"] == "bla neue"
    assert final.get("forced") is True and "utterance" in final


def test_flush_releases_the_hold_when_the_utterance_is_dropped():
    """A flush whose in-flight utterance is dropped (here: its final decode
    fails) still gives the client everything — the held tail of the one
    before goes out as a release final."""
    texts = ["bla neue"]

    async def decode_final(audio, prompt):
        if not texts:
            raise RuntimeError("decode failed")
        return (texts.pop(0), [], False)
    s, msgs = _make_session(
        postprocess=_neue_zeile, decode_final=decode_final,
        cfg=StreamConfig(**_FAST, hard_break_silence_ms=0),
        holdback=_hold_trailing("neue"))

    async def run():
        await s.feed_pcm(_pcm(8000, 300))
        await s.feed_pcm(_pcm(0, 400))
        await s.feed_pcm(_pcm(8000, 300))
        assert s._in_utterance
        await s.flush_utterance()

    asyncio.run(run())
    finals = _finals(msgs)
    assert finals[0]["committed"] + finals[0]["tail"] == "bla"
    assert finals[-1]["committed"] == "bla neue" and finals[-1].get("flush") is True


def test_idle_release_when_hard_breaks_are_off():
    """With hard breaks off nothing else would release a held word while the
    speaker stays quiet: 5 s of silence release it, once."""
    s, msgs = _make_session(
        postprocess=_neue_zeile, decode_final=_queued_final(["bla neue"]),
        cfg=StreamConfig(**_FAST, hard_break_silence_ms=0),
        holdback=_hold_trailing("neue"))

    async def run():
        await s.feed_pcm(_pcm(8000, 300))
        await s.feed_pcm(_pcm(0, 3000))
        assert len(_finals(msgs)) == 1           # not yet
        await s.feed_pcm(_pcm(0, 5000))

    asyncio.run(run())
    finals = _finals(msgs)
    assert len(finals) == 2
    assert finals[1]["committed"] == "bla neue" and finals[1].get("flush") is True
    assert not [m for m in msgs if m["type"] == "boundary"]


def test_dropped_final_keeps_the_hold():
    """An utterance whose final decode drops everything adds no text: the held
    word stays held and joins the utterance after it."""
    s, msgs = _make_session(
        postprocess=_neue_zeile, decode_final=_queued_final(["bla neue", ("", True), " zeile text."]),
        cfg=StreamConfig(**_FAST, hard_break_silence_ms=0),
        holdback=_hold_trailing("neue"))

    async def run():
        for _ in range(3):
            await s.feed_pcm(_pcm(8000, 300))
            await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    docs = [m["committed"] + m["tail"] for m in _finals(msgs)]
    assert all("neue" not in d for d in docs)
    assert docs[-1] == "bla\n text."
    for a, b in zip(docs, docs[1:]):
        assert b.startswith(a)


def test_seam_divergence_reanchors_logs_and_sticks(caplog):
    """When a new document would rewrite sent text anyway, the sent text stays
    as typed: a WARNING names the culprit, and the rest of the document is
    formatted on its own — also for later utterances (sticky)."""
    import logging

    def pp(raw):                                    # 'a b' → 'X' (looks ahead)
        return raw.replace("a b", "X").strip()

    s, msgs = _make_session(postprocess=pp, diagnose=lambda a, b: "#9 toy")

    async def run():
        s.raw_confirmed = "a"
        await s._emit_update()
        s.raw_confirmed += " b c"
        await s._emit_update()
        s.raw_confirmed += " a b"
        await s._emit_update()

    with caplog.at_level(logging.WARNING, logger="faster_whisper_backend.streaming.session"):
        asyncio.run(run())
    docs = [m["committed"] + m["tail"] for m in _finals(msgs)]
    assert docs == ["a", "a b c", "a b c X"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "seam" in warnings[0] and "#9 toy" in warnings[0]


def test_language_change_reanchors_without_rewriting_sent_text(caplog):
    import logging

    lang = ["de"]

    def pp(raw):
        return raw.strip().upper() if lang[0] == "en" else raw.strip()

    s, msgs = _make_session(postprocess=pp, format_key=lambda: lang[0])

    async def run():
        s.raw_confirmed = "hallo"
        await s._emit_update()
        lang[0] = "en"
        s.raw_confirmed += " welt"
        await s._emit_update()

    with caplog.at_level(logging.INFO, logger="faster_whisper_backend.streaming.session"):
        asyncio.run(run())
    docs = [m["committed"] + m["tail"] for m in _finals(msgs)]
    assert docs == ["hallo", "hallo WELT"]
    msgs_logged = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert any(lvl == logging.INFO and "formatting language changed" in m
               for lvl, m in msgs_logged)
    assert not any(lvl >= logging.WARNING for lvl, _ in msgs_logged)


def test_committed_region_only_grows():
    """The document-level agreement now runs on emitted documents, which only
    grow — so committed never shrinks or changes either."""
    s, msgs = _make_session(postprocess=lambda raw: raw.strip())

    async def run():
        s.raw_confirmed = "eins."
        await s._emit_update()
        s.raw_confirmed += " zwei."
        await s._emit_update()
        s.raw_confirmed += " drei"
        await s._emit_update()

    asyncio.run(run())
    committeds = [m["committed"] for m in _finals(msgs)]
    assert committeds == ["", "eins.", "eins. zwei."]


@pytest.mark.parametrize("left,piece,joined", [
    ("Er sagt", "hallo", " hallo"),
    ("Ende.", "weiter", " Weiter"),
    ("Ende.", "Weiter", " Weiter"),
    ("Zeile\n", "weiter", "Weiter"),
    ("Wert", ", 5", ", 5"),
    ("Wert", ".", "."),
    ("Befund (", "rechts", "rechts"),
    ('Sie sagt "', "mir", "mir"),
    ('Sie sagt "mir ist', '"', '"'),
    ('Er sagt "ja".', '"Nein"', ' "Nein"'),
    ("120/", "80", "80"),
    ("Eisen-", "Infusion", "Infusion"),
    ("Wert", "-5", "-5"),
    ("Wert", "", ""),
    ("frage?", "nein", " Nein"),
    ("klein", "Gross", " Gross"),     # never lower-cased
])
def test_seam_join(left, piece, joined):
    assert StreamSession._seam_join(left, piece) == joined
