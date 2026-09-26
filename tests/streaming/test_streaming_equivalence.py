"""The core correctness guarantees of streaming finals, on the REAL pipeline:

* within one document every final's ``committed + tail`` EXTENDS the previous
  one — the client types only the growth and never takes text back, even when
  a multi-word dictation phrase ('neue Zeile'), a joining symbol ('Schrägstrich',
  'Bindestrich', 'Komma' before a newline) or a quotation is split across the
  utterance seam;
* ``committed`` only ever grows;
* once the session closes, the document equals the batch route's
  whole-document post-processing.

The first two used to be claimed for ``committed`` only, and the session
re-formatted the whole document per utterance: text already typed from an
earlier final could be rewritten by a later one (quotes typed twice, a stray
'neue' left before a newline). The session now holds back the raw words a
rule could still join with the next utterance (main.holdback_start) and
never re-formats sent text.

Uses main._postprocess_text / main.holdback_start via the app_module fixture.
"""

import asyncio

import pytest

from faster_whisper_backend.streaming.session import StreamConfig, StreamSession
from faster_whisper_backend.streaming.vad import EnergyEndpointer


def _run_stream(main, utterances, language):
    """Drive a session through a sequence of finalized raw utterances (bypassing
    audio/VAD) and return the final frames (the last one is the closing
    document)."""
    finals = []

    def pp(raw):
        return main._postprocess_text(raw, model_name="", language=language)

    async def emit(m):
        if m["type"] == "final":
            finals.append(m)

    async def _noop_dp(a, p):
        return []

    async def _noop_df(a, p):
        return ("", [], False)

    s = StreamSession(
        config=StreamConfig(), endpointer=EnergyEndpointer(),
        decode_partial=_noop_dp, decode_final=_noop_df, postprocess=pp, emit=emit,
        holdback=lambda raw: main.holdback_start(raw, model_name="", language=language),
        format_key=lambda: language,
        diagnose=lambda a, b: main.seam_culprit(a, b, model_name="", language=language),
    )

    async def run():
        for u in utterances:
            s.raw_confirmed += u
            await s._emit_update()
        await s.close()  # commit the whole document

    asyncio.run(run())
    return finals


def _cases():
    return [
        # 'neue Zeile' split across the seam — the headline hazard.
        ["der Patient hat Fieber Punkt neue ", "Zeile Blutdruck normal Punkt"],
        # Several sentences, terminators, a comma command.
        ["Diagnose Komma Pneumonie Punkt ", "Therapie Punkt neue Zeile Antibiotika Punkt"],
        # A bracket command split across the seam.
        ["Befund Klammer ", "auf unauffaellig Klammer zu Punkt"],
        # No terminators at all (flushed on close).
        ["hallo welt", " wie geht es"],
        # A quotation with a pause inside it: the quote opened in the first
        # utterance is closed in the next — must not be typed twice.
        [" Sie sagt Gänsefüsschen mir ist", " schwindlig Gänsefüsschen Punkt", " Weiter Punkt"],
        # A comma the newline tidy eats once 'neue Zeile' follows.
        [" Sehr geehrter Herr Kollege Komma", " neue Zeile vielen Dank Punkt"],
        # A slash between two numbers is tightened across the seam.
        [" Blutdruck 120 Schrägstrich", " 80 mmHg Punkt"],
        # A hyphen compound: the hyphen joins and capitalizes the next word.
        [" Eisen Bindestrich", " Infusion am Montag Punkt"],
        # 'neuer | Absatz' after Whisper's own soft commas.
        [" Sehr geehrte Frau Kollegin, Komma, neuer", " Absatz. Wir berichten Punkt"],
    ]


@pytest.mark.parametrize("language", [None, "de"])
def test_streaming_finals_extend_and_reconstruct_batch_output(app_module, language):
    main = app_module

    for utterances in _cases():
        full = main._postprocess_text("".join(utterances), model_name="", language=language)
        finals = _run_stream(main, utterances, language)
        docs = [m["committed"] + m["tail"] for m in finals]
        committeds = [m["committed"] for m in finals]
        # the committed document, once the session closes, equals batch output.
        assert finals[-1].get("last") is True
        assert committeds[-1] == full, (
            f"streaming != batch for {utterances!r} (language {language})\n"
            f" batch:    {full!r}\n stream:   {committeds[-1]!r}")
        # the document only grows: what a client typed is never taken back.
        for a, b in zip(docs, docs[1:]):
            assert b.startswith(a), (
                f"document rewritten for {utterances!r} (language {language})\n"
                f" was: {a!r}\n now: {b!r}")
        # append-only committed region.
        for a, b in zip(committeds, committeds[1:]):
            assert b.startswith(a), (
                f"committed text rewritten (not append-only) for {utterances!r}\n"
                f" was: {a!r}\n now: {b!r}")


def test_split_dictation_phrase_is_never_sent_half_resolved(app_module):
    """'neue' ending one utterance is held back, not sent: no document ever
    contains the literal word before the newline it becomes."""
    finals = _run_stream(app_module, [" Befund unauffällig Punkt neue", " Zeile Therapie Punkt"], "de")
    assert all("neue" not in (m["committed"] + m["tail"]) for m in finals)
    assert finals[-1]["committed"] == "Befund unauffällig.\nTherapie."
