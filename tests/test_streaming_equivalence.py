"""The core correctness guarantee: the append-only streaming finals, reconstructed
by concatenation, equal the batch route's whole-document post-processing — even
when a multi-word dictation phrase ('neue Zeile') is split across the utterance
seam. This is what justifies running _postprocess_text on the rolling accumulator
instead of patching every cross-utterance pipeline hazard.

Uses the REAL pipeline (main._postprocess_text) via the app_module fixture.
"""

import asyncio

from streaming_session import StreamConfig, StreamSession
from streaming_vad import EnergyEndpointer


def _run_stream(pp, utterances):
    """Drive a session through a sequence of finalized raw utterances (bypassing
    audio/VAD) and return the concatenated emitted final text."""
    deltas = []

    async def emit(m):
        if m["type"] == "final":
            deltas.append(m["text"])

    async def _noop_dp(a, p):
        return []

    async def _noop_df(a, p):
        return ("", [])

    s = StreamSession(
        config=StreamConfig(), endpointer=EnergyEndpointer(),
        decode_partial=_noop_dp, decode_final=_noop_df, postprocess=pp, emit=emit,
    )

    async def run():
        for u in utterances:
            s.raw_confirmed += u
            await s._emit_final_delta(pp(s.raw_confirmed))
        await s.close()  # flush the held tail

    asyncio.run(run())
    return "".join(deltas)


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
        streamed = _run_stream(pp, utterances)
        assert streamed == full, (
            f"streaming != batch for {utterances!r}\n"
            f" batch:    {full!r}\n stream:   {streamed!r}")
