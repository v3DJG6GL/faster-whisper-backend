"""Property test: a live-dictation document never rewrites what it already sent.

A compact port of the seam fuzzer that found the "dictation written twice"
family: Whisper-style raw passages (leading spaces, Whisper's own commas and
periods around spoken dictation words), re-split into utterances at EVERY word
boundary (as-is, and with Whisper closing the first utterance with a period)
plus seeded three-way splits. Each split is driven through a real
StreamSession with the real pipeline and hold-back, and must:

* emit documents that each extend the previous one (the client types only
  the growth and never takes text back) with ``committed`` only growing;
* close with the batch output, unless the seam safety net re-anchored;
* re-anchor (WARNING) only for the self-correction "Punkt | Strichpunkt",
  where the dedup rule legitimately lets the later mark replace the earlier
  one — sent text is kept there, the rest formatted on its own.

Before the hold-back, 96 of 1194 German pause positions rewrote sent text.
"""

import asyncio
import logging
import random
import re


from faster_whisper_backend.streaming.session import StreamConfig, StreamSession
from faster_whisper_backend.streaming.vad import EnergyEndpointer
from faster_whisper_backend.pipeline import engine as pl_engine

DE = [
    [" Sehr geehrte Frau Kollegin, Komma, neuer Absatz.",
     " Wir berichten Ihnen über die Patientin Frau Müller, Komma, geboren am 12.03.1956, Punkt.",
     " Neue Zeile. Diagnosen, Doppelpunkt, neue Zeile."],
    [" Sehr geehrter Herr Kollege Komma", " neue Zeile", " vielen Dank für die Zuweisung Punkt",
     " Neuer Absatz", " Anamnese Doppelpunkt", " Der Patient berichtet über Kopfschmerzen Punkt"],
    [" Therapie, Doppelpunkt, Bisoprolol 2, Komma, 5 mg 1-0-0, Punkt.",
     " Neue Zeile. Metformin 1000 mg 2 mal täglich."],
    [" Blutdruck 120 Schrägstrich 80 mmHg Komma Puls 72 pro Minute Punkt",
     " Temperatur 38 Komma 5 Grad Punkt"],
    [" Die Beschwerden bestehen seit 10 - 12 Tagen.",
     " Der Patient wohnt in der Hauptstraße 10 Bindestrich 12, Punkt.",
     " Kontrolle in 4 Minus 6 Wochen Punkt"],
    [" Befund Doppelpunkt neue Zeile", " Herz Doppelpunkt rhythmisch Komma rein Punkt neue Zeile",
     " Abdomen Doppelpunkt weich Komma kein Druckschmerz Punkt"],
    [" Beurteilung, Doppelpunkt, neue Säule.", " Neuer Apfel.", " Noch Ihr Absatz.",
     " Keine fokalen Defizite, Punkt."],
    [" Diagnose Doppelpunkt Pneumonie Klammer auf rechts basal Klammer zu Punkt",
     " Eckige Klammer auf Nachtrag eckige Klammer zu Punkt"],
    [" Die Patientin sagt, Doppelpunkt, Gänsefüsschen, mir ist seit gestern schwindlig, Gänsefüsschen, Punkt.",
     " Er sagte wörtlich Gänsefüsschen ich habe keine Luft mehr Gänsefüsschen Punkt"],
    [" Der Wert ist größer als 5 Punkt", " Das Kalium ist kleiner als 3 Komma 5 Punkt",
     " Tumorstadium Römisch 3 nach Ann Arbor Punkt"],
    [" Eisen Bindestrich Infusion am Montag Punkt", " Vitamin Bindestrich D Bindestrich Mangel Punkt",
     " Der Patient Binde Strich Abstand Nachsorge ist geplant Punkt"],
    [" Frau Müllerkomma bitte zur Kontrolle kommen Punkt", " Der Fuß ist geschwollen Punkt"],
    [" Hat der Patient Fieber? Fragezeichen.", " Wie lange bestehen die Beschwerden Fragezeichen",
     " Wichtig Ausrufezeichen", " Bitte dringend Rückruf Ausrufezeichen"],
    [" Der Patient klagt über... Kopfschmerzen.", " HB 12... Komma 5, Punkt.",
     " Also, die Schmerzen sind, ähm, seit gestern da."],
    [" Ergebnis Semikolon negativ Punkt", " Strichpunkt als Trennzeichen Punkt"],
    [" Medikation Doppelpunkt neue Zeile", " Pantoprazol 40 mg 1 Bindestrich 0 Bindestrich 0 neue Zeile",
     " Metamizol 500 mg 1 Minus 1 Minus 1 Punkt"],
]

EN = [
    [" Dear colleague, comma, new line.", " Thank you for referring Mr. Smith.",
     " He was seen on March 3rd, 2024."],
    [" She says, \"I feel dizzy.\"", " What are the symptoms?", " Plan: antibiotics for 7-10 days."],
]

_PUNCT_END = re.compile(r"[.,;:!?]+$")


def _clean(w):
    return re.sub(r"[^\wäöüÄÖÜß]", "", w).lower()


def _self_correction(a_words, b_words):
    """The allowed re-anchor: a dictated period corrected to a semicolon."""
    return _clean(a_words[-1]) == "punkt" and _clean(b_words[0]) == "strichpunkt"


def _run(utts, language):
    """One session over `utts`, formatted in `language` throughout."""
    finals = []

    async def emit(m):
        if m["type"] == "final":
            finals.append(m)

    async def _dp(a, p):
        return []

    async def _df(a, p):
        return ("", [], False)

    s = StreamSession(
        config=StreamConfig(), endpointer=EnergyEndpointer(),
        decode_partial=_dp, decode_final=_df,
        postprocess=lambda raw: pl_engine._postprocess_text(raw, model_name="", language=language),
        emit=emit,
        holdback=lambda raw: pl_engine.holdback_start(raw, model_name="", language=language),
        format_key=lambda: language,
        diagnose=lambda a, b: pl_engine.seam_culprit(a, b, model_name="", language=language),
    )

    async def go():
        for u in utts:
            s.raw_confirmed += u
            await s._emit_update()
        await s.close()

    asyncio.run(go())
    return finals


def _splits(words, rng):
    """(utterances, [(last word, first word) per seam]) for every 2-way split
    as-is and with a period closing the first part, then 20 seeded 3-way
    splits."""
    n = len(words)
    out = []
    for i in range(1, n):
        a, b = words[:i], words[i:]
        seam = [(a, b)]
        out.append(([" " + " ".join(a), " " + " ".join(b)], seam))
        per = " " + _PUNCT_END.sub("", " ".join(a)) + "."
        if per.strip() != ".":
            out.append(([per, " " + " ".join(b)], seam))
    pairs = [(i, j) for i in range(1, n) for j in range(i + 1, n)]
    for i, j in rng.sample(pairs, min(20, len(pairs))):
        a, b, c = words[:i], words[i:j], words[j:]
        out.append(([" " + " ".join(a), " " + " ".join(b), " " + " ".join(c)],
                    [(a, b), (b, c)]))
    return out


def _check(corpus, language, caplog):
    rng = random.Random(12345)
    checked = warned = 0
    for passage in corpus:
        words = "".join(passage).strip().split(" ")
        for utts, seams in _splits(words, rng):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="faster_whisper_backend.streaming.session"):
                finals = _run(utts, language)
            checked += 1
            docs = [m["committed"] + m["tail"] for m in finals]
            committeds = [m["committed"] for m in finals]
            for x, y in zip(docs, docs[1:]):
                assert y.startswith(x), (utts, x, y)
            for x, y in zip(committeds, committeds[1:]):
                assert y.startswith(x), (utts, x, y)
            assert all("¿" not in d and "¡" not in d for d in docs), (utts, docs)
            seam_warnings = [r for r in caplog.records if "seam:" in r.getMessage()]
            if seam_warnings:
                warned += 1
                assert any(_self_correction(a, b) for a, b in seams), (
                    utts, [r.getMessage() for r in seam_warnings])
            else:
                full = pl_engine._postprocess_text("".join(utts), model_name="",
                                                   language=language)
                assert committeds[-1] == full, (utts, committeds[-1], full)
    return checked, warned


def test_german_documents_never_rewrite_sent_text(app_module, caplog):
    # Auto language is not a separate case here: _run sets the language
    # before every compose, exactly as the route does. That route ordering
    # (no '¿' on German questions under auto) is pinned by
    # test_routes_streaming.py::test_stream_formats_the_first_final_in_the_detected_language.
    # app_module only sets up the config the pipeline formats with.
    checked, warned = _check(DE, "de", caplog)
    assert checked > 700
    # Only the Punkt | Strichpunkt splits may hit the safety net.
    assert warned <= 8


def test_english_documents_never_rewrite_sent_text(app_module, caplog):
    checked, warned = _check(EN, "en", caplog)
    assert checked > 50 and warned == 0


def test_a_prefix_in_front_of_a_held_tail_is_not_held_for_nothing():
    """Once a tail is held, the word after the earlier words is known: a bare
    key prefix it does not continue ("größer" + "neue" is no "größer als")
    goes out now instead of waiting for the next utterance."""
    from faster_whisper_backend.pipeline import seam_holdback as sh
    spec = sh.build_spec([{"größer als": ">", "neue Zeile": "\n", "Klammer auf": "(",
                           "Klammer zu": ")", "Komma": ","}])

    def held(raw):
        return raw[sh.held_start(raw, spec):].strip()

    assert held("Der Wert ist größer neue") == "neue"
    assert held("Text Klammer neue") == "neue"
    # A joiner key in front of a held tail is still held, and so is a prefix
    # the held tail continues.
    assert held("Frau Kollegin, Komma, neue") == "Komma, neue"
    assert held("Der Wert ist größer") == "größer"


def test_a_prefix_completed_into_a_whole_key_by_the_held_tail_stays_held():
    """`p q` + the held `r` is the whole key `p q r`: formatting `p q` now
    would rewrite it once `r` is sent, so the re-check keeps it."""
    from faster_whisper_backend.pipeline import seam_holdback as sh
    spec = sh.build_spec([{"p q r": "X", "r s": "Y"}])
    assert "a p q r"[sh.held_start("a p q r", spec):].strip() == "p q r"
    assert "a p r"[sh.held_start("a p r", spec):].strip() == "r"
