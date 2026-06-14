"""The core correctness guarantee: the streaming committed document (after close)
equals the batch route's whole-document post-processing — even when a multi-word
dictation phrase ('neue Zeile') is split across the utterance seam — and the
committed prefix is append-only at every step (never rewrites earlier text). This
is what justifies running _postprocess_text on the rolling accumulator instead of
patching every cross-utterance pipeline hazard.

Uses the REAL pipeline (main._postprocess_text) via the app_module fixture.
"""

import asyncio

from streaming_session import StreamConfig, StreamSession
from streaming_vad import EnergyEndpointer


def _run_stream(pp, utterances):
    """Drive a session through a sequence of finalized raw utterances (bypassing
    audio/VAD) and return the per-final committed strings (last == full document)."""
    committeds = []

    async def emit(m):
        if m["type"] == "final":
            committeds.append(m["committed"])

    async def _noop_dp(a, p):
        return []

    async def _noop_df(a, p):
        return ("", [], False)

    s = StreamSession(
        config=StreamConfig(), endpointer=EnergyEndpointer(),
        decode_partial=_noop_dp, decode_final=_noop_df, postprocess=pp, emit=emit,
    )

    async def run():
        for u in utterances:
            s.raw_confirmed += u
            await s._emit_document(pp(s.raw_confirmed))
        await s.close()  # commit the whole document

    asyncio.run(run())
    return committeds


def _cases():
    return [
        # 'neue Zeile' split across the seam — the headline hazard.
        ["der Patient hat Fieber Punkt neue ", "Zeile Blutdruck normal Punkt"],
        # Several sentences, terminators, a comma command.
        ["Diagnose Komma Pneumonie Punkt ", "Therapie Punkt neue Zeile Antibiotika Punkt"],
        # A bracket command split across the seam.
        ["Befund Klammer ", "auf unauffaellig Klammer zu Punkt"],
        # No terminators at all (held entirely, flushed on close).
        ["hallo welt", " wie geht es"],
    ]


def test_streaming_finals_reconstruct_batch_output(app_module):
    main = app_module

    def pp(raw):
        return main._postprocess_text(raw, model_name="")

    for utterances in _cases():
        full = pp("".join(utterances))
        committeds = _run_stream(pp, utterances)
        # the committed document, once the session closes, equals batch output.
        assert committeds[-1] == full, (
            f"streaming != batch for {utterances!r}\n"
            f" batch:    {full!r}\n stream:   {committeds[-1]!r}")
        # append-only: each committed extends the previous — earlier committed text
        # is never rewritten, even across a 'neue Zeile' seam split.
        for a, b in zip(committeds, committeds[1:]):
            assert b.startswith(a), (
                f"committed text rewritten (not append-only) for {utterances!r}\n"
                f" was: {a!r}\n now: {b!r}")
