"""The ``utterance`` lifecycle frames (open / decoding / dropped).

The contract under test, from the session module's docstring: an utterance is
announced once it holds ``min_speech_ms`` of speech, ``decoding`` precedes the
final decode, and EVERY announced utterance ends in exactly one terminal frame
— a ``final`` carrying the same ordinal, or ``dropped``. A client paints "the
server is working" off these frames, so an utterance that is announced and
never closed is a status indicator stuck on "working".

Also pinned here, because it is the same code path: a final decode that raises
ends the utterance (it used to be retried on every following frame).
"""

import asyncio

import pytest

from tests._streaming_helpers import const_pcm as _pcm
from faster_whisper_backend.streaming.session import CloseAbort, StreamConfig, StreamSession
from faster_whisper_backend.streaming.vad import EnergyEndpointer


class _AlwaysSpeech:
    def is_speech(self, frame):
        return True

    def reset(self) -> None:
        pass


def _cfg(**over):
    base = dict(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    base.update(over)
    return StreamConfig(**base)


def _session(*, decode_partial=None, decode_final=None, on_final=None, cfg=None,
             endpointer=None, postprocess=lambda raw: raw):
    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    async def _dp(audio, prompt):
        return []

    async def _df(audio, prompt):
        return ("hallo welt.", [], False)

    s = StreamSession(
        config=cfg or _cfg(),
        endpointer=endpointer or EnergyEndpointer(),
        decode_partial=decode_partial or _dp,
        decode_final=decode_final or _df,
        postprocess=postprocess,
        emit=emit,
        on_final=on_final,
    )
    return s, msgs


def _utt(msgs):
    return [(m["state"], m["utterance"]) for m in msgs if m["type"] == "utterance"]


def _lifecycle(msgs):
    """The frames a client's state machine sees, as (kind, ordinal) — finals
    included, the closing ``last`` document excluded (it is not an utterance)."""
    out = []
    for m in msgs:
        if m["type"] == "utterance":
            out.append((m["state"], m["utterance"]))
        elif m["type"] == "final" and not m.get("last"):
            out.append(("final", m["utterance"]))
    return out


def _assert_every_announced_utterance_closed_once(msgs):
    opened: dict[int, int] = {}
    closed: dict[int, int] = {}
    for kind, n in _lifecycle(msgs):
        if kind in ("open", "decoding"):
            opened[n] = opened.get(n, 0) + 1
        else:  # final / dropped
            closed[n] = closed.get(n, 0) + 1
    for n in opened:
        assert closed.get(n) == 1, f"utterance {n}: {closed.get(n)} terminal frames in {_lifecycle(msgs)}"


# ---- open -------------------------------------------------------------------


def test_open_is_emitted_once_at_the_min_speech_threshold():
    s, msgs = _session(cfg=_cfg(min_speech_ms=128))

    async def run():
        await s.feed_pcm(_pcm(8000, 96))     # 3 frames = 96 ms of speech: below the gate
        assert _utt(msgs) == []
        await s.feed_pcm(_pcm(8000, 320))    # crosses 128 ms, then keeps speaking
        assert _utt(msgs) == [("open", 0)]

    asyncio.run(run())


def test_open_is_emitted_even_while_partials_are_skipped():
    """Behind realtime the partial decode is skipped — which is exactly when a
    client most needs to know the server is holding its words."""
    s, msgs = _session()
    s._skip_partials = True

    async def run():
        await s.feed_pcm(_pcm(8000, 320))

    asyncio.run(run())
    assert _utt(msgs) == [("open", 0)]
    assert [m for m in msgs if m["type"] == "partial"] == []


def test_a_blip_below_min_speech_is_never_announced():
    s, msgs = _session(cfg=_cfg(min_speech_ms=640))

    async def run():
        await s.feed_pcm(_pcm(8000, 96))
        await s.feed_pcm(_pcm(0, 400))       # commit silence → finalize → near-silence gate

    asyncio.run(run())
    assert msgs == []
    assert s._utterance_index == 0           # nothing reached the wire → the ordinal is not spent


# ---- the normal cycle ---------------------------------------------------------


def test_open_then_decoding_then_final_share_one_ordinal():
    s, msgs = _session()

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    assert _lifecycle(msgs) == [("open", 0), ("decoding", 0), ("final", 0)]
    assert s._utt_state is None and not s._in_utterance
    assert s._utterance_index == 1


def test_second_utterance_gets_the_next_ordinal():
    s, msgs = _session()

    async def run():
        for _ in range(2):
            await s.feed_pcm(_pcm(8000, 500))
            await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    assert _lifecycle(msgs) == [
        ("open", 0), ("decoding", 0), ("final", 0),
        ("open", 1), ("decoding", 1), ("final", 1),
    ]


def test_forced_commit_closes_the_utterance_and_the_next_one_reopens():
    """Still speaking at forced_commit_sec: the cycle closes with a forced final
    and the continuing speech is announced as the next utterance."""
    s, msgs = _session(endpointer=_AlwaysSpeech(), cfg=_cfg(forced_commit_sec=0.512))

    async def run():
        await s.feed_pcm(_pcm(8000, 800))

    asyncio.run(run())
    assert _lifecycle(msgs) == [("open", 0), ("decoding", 0), ("final", 0), ("open", 1)]
    assert [m for m in msgs if m["type"] == "final"][0].get("forced") is True


# ---- dropped ------------------------------------------------------------------


def test_gate_with_no_text_emits_dropped_no_speech_and_spends_the_ordinal():
    """Enough 'speech' to be announced, but the whole buffer sits under the RMS
    gate and no partial ever committed a word: no decode, no final — the client
    still gets its terminal frame."""
    s, msgs = _session(cfg=_cfg(rms_gate_dbfs=-1.0))

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        await s.feed_pcm(_pcm(0, 400))
        await s.feed_pcm(_pcm(8000, 500))    # the next utterance

    asyncio.run(run())
    assert _utt(msgs) == [("open", 0), ("dropped", 0), ("open", 1)]
    dropped = [m for m in msgs if m["type"] == "utterance" and m["state"] == "dropped"]
    assert dropped[0]["reason"] == "no_speech"
    assert [m for m in msgs if m["type"] == "final"] == []


def test_empty_document_emits_dropped_empty():
    """The decode ran and threw everything away as a hallucination: nothing to
    put in a ``final``, so the cycle is closed explicitly."""
    async def _df(audio, prompt):
        return ("", [], True)

    s, msgs = _session(decode_final=_df)

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    assert _utt(msgs) == [("open", 0), ("decoding", 0), ("dropped", 0)]
    assert [m for m in msgs if m["type"] == "utterance"][-1]["reason"] == "empty"


def test_close_on_an_empty_document_never_emits_dropped():
    """close() reuses _emit_document for the closing document, which is not an
    utterance: an empty one must stay silent."""
    s, msgs = _session()
    asyncio.run(s.close())
    assert msgs == []


# ---- a failing final decode ------------------------------------------------------


def test_failed_final_decode_falls_back_to_the_partial_transcript():
    calls = []
    infos = []

    async def _dp(audio, prompt):
        return [(0.0, 0.3, " hallo"), (0.3, 0.6, " welt")]

    async def _df(audio, prompt):
        calls.append(1)
        raise RuntimeError("CUDA out of memory")

    async def _on_final(info):
        infos.append(info)

    s, msgs = _session(decode_partial=_dp, decode_final=_df, on_final=_on_final)

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())                        # does not raise
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    assert "welt" in finals[0]["committed"] + finals[0]["tail"]
    assert _lifecycle(msgs) == [("open", 0), ("decoding", 0), ("final", 0)]
    assert infos and infos[0]["decode_failed"] is True and infos[0]["decoded"] is False
    assert len(calls) == 1


def test_failed_final_decode_without_text_drops_and_is_not_retried():
    """The retry storm: the raise used to leave the utterance open with the
    silence timer past the commit threshold, so every following frame ran the
    same doomed decode again."""
    calls = []

    async def _df(audio, prompt):
        calls.append(1)
        raise RuntimeError("CUDA out of memory")

    s, msgs = _session(decode_final=_df)

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        await s.feed_pcm(_pcm(0, 400))
        await s.feed_pcm(_pcm(0, 400))        # more silence: must not decode again

    asyncio.run(run())
    assert len(calls) == 1
    assert not s._in_utterance and s._utt_state is None
    assert _utt(msgs) == [("open", 0), ("decoding", 0), ("dropped", 0)]
    assert [m for m in msgs if m["type"] == "utterance"][-1]["reason"] == "error"


def test_failed_final_decode_does_not_lose_the_rest_of_the_chunk():
    """The raise used to abort feed_pcm's frame loop, so whatever followed the
    failing frame in the same chunk — here the start of the next utterance —
    was dropped on the floor."""
    async def _df(audio, prompt):
        raise RuntimeError("CUDA out of memory")

    s, msgs = _session(decode_final=_df)

    async def run():
        await s.feed_pcm(_pcm(8000, 512) + _pcm(0, 384) + _pcm(8000, 512))

    asyncio.run(run())
    assert _utt(msgs) == [("open", 0), ("decoding", 0), ("dropped", 0), ("open", 1)]
    assert s._in_utterance


def test_on_final_failure_still_spends_the_ordinal_and_sends_no_second_terminal():
    """final(0) is already on the wire when on_final raises. The ordinal used
    to stay at 0, so the next utterance reused it; and the error path must not
    add a ``dropped`` after a ``final``."""
    async def _on_final(info):
        raise RuntimeError("capture store is down")

    s, msgs = _session(on_final=_on_final)

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        with pytest.raises(RuntimeError):
            await s.feed_pcm(_pcm(0, 400))
        await s.feed_pcm(_pcm(8000, 500))

    asyncio.run(run())
    assert _lifecycle(msgs) == [("open", 0), ("decoding", 0), ("final", 0), ("open", 1)]


def test_postprocess_failure_closes_the_cycle_with_dropped_error():
    def _boom(raw):
        raise ValueError("bad rule")

    s, msgs = _session(postprocess=_boom)

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        with pytest.raises(ValueError):
            await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    assert _utt(msgs) == [("open", 0), ("decoding", 0), ("dropped", 0)]
    assert not s._in_utterance


def test_close_abort_emits_no_terminal_frame():
    async def _df(audio, prompt):
        raise CloseAbort("credential revoked mid-session")

    s, msgs = _session(decode_final=_df)

    async def run():
        await s.feed_pcm(_pcm(8000, 500))
        with pytest.raises(CloseAbort):
            await s.close()

    asyncio.run(run())
    assert _utt(msgs) == [("open", 0), ("decoding", 0)]
    assert [m for m in msgs if m["type"] == "final"] == []


# ---- the invariant, over a mixed script ------------------------------------------


def test_every_announced_utterance_has_exactly_one_terminal_frame():
    script = iter([
        ("eins.", [], False),                 # a normal final
        RuntimeError("boom"),                 # failed decode, no partial text → dropped/error
        ("", [], True),                       # dropped_all, but the document is non-empty → final
        ("zwei.", [], False),
    ])

    async def _df(audio, prompt):
        step = next(script)
        if isinstance(step, Exception):
            raise step
        return step

    s, msgs = _session(decode_final=_df)

    async def run():
        for _ in range(4):
            await s.feed_pcm(_pcm(8000, 500))
            await s.feed_pcm(_pcm(0, 400))
        await s.close()

    asyncio.run(run())
    _assert_every_announced_utterance_closed_once(msgs)
    ordinals = [n for kind, n in _lifecycle(msgs) if kind == "open"]
    assert ordinals == [0, 1, 2, 3]           # unique and only ever growing
    # Each step closes the way the script above says it does.
    terminals = [(kind, n) for kind, n in _lifecycle(msgs) if kind in ("final", "dropped")]
    assert terminals == [("final", 0), ("dropped", 1), ("final", 2), ("final", 3)]
    assert [m.get("reason") for m in msgs
            if m["type"] == "utterance" and m["state"] == "dropped"] == ["error"]
    assert msgs[-1]["type"] == "final" and msgs[-1].get("last") is True
