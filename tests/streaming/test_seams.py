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


def _run(main, utts, language_of):
    """One session over `utts`. `language_of(k)` is the formatting language
    once k utterances are in — the route sets it from each final's decode
    BEFORE formatting (auto language: the decode's detection)."""
    finals = []
    k = [0]

    def lang():
        return language_of(k[0])

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
        postprocess=lambda raw: pl_engine._postprocess_text(raw, model_name="", language=lang()),
        emit=emit,
        holdback=lambda raw: pl_engine.holdback_start(raw, model_name="", language=lang()),
        format_key=lang,
        diagnose=lambda a, b: pl_engine.seam_culprit(a, b, model_name="", language=lang()),
    )

    async def go():
        for u in utts:
            k[0] += 1
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


def _check(main, corpus, modes, caplog):
    rng = random.Random(12345)
    checked = warned = 0
    for passage in corpus:
        words = "".join(passage).strip().split(" ")
        for utts, seams in _splits(words, rng):
            for mode, language_of in modes:
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger="faster_whisper_backend.streaming.session"):
                    finals = _run(main, utts, language_of)
                checked += 1
                docs = [m["committed"] + m["tail"] for m in finals]
                committeds = [m["committed"] for m in finals]
                for x, y in zip(docs, docs[1:]):
                    assert y.startswith(x), (mode, utts, x, y)
                for x, y in zip(committeds, committeds[1:]):
                    assert y.startswith(x), (mode, utts, x, y)
                assert all("¿" not in d and "¡" not in d for d in docs), (mode, utts, docs)
                seam_warnings = [r for r in caplog.records if "seam:" in r.getMessage()]
                if seam_warnings:
                    warned += 1
                    assert any(_self_correction(a, b) for a, b in seams), (
                        mode, utts, [r.getMessage() for r in seam_warnings])
                else:
                    full = pl_engine._postprocess_text("".join(utts), model_name="",
                                                  language=language_of(len(utts)))
                    assert committeds[-1] == full, (mode, utts, committeds[-1], full)
    return checked, warned


def test_german_documents_never_rewrite_sent_text(app_module, caplog):
    # Auto language is not a separate case here: _run sets the language
    # before every compose, exactly as the route does. That route ordering
    # (no '¿' on German questions under auto) is pinned by
    # test_routes_streaming.py::test_stream_formats_the_first_final_in_the_detected_language.
    checked, warned = _check(app_module, DE, [("de", lambda k: "de")], caplog)
    assert checked > 700
    # Only the Punkt | Strichpunkt splits may hit the safety net.
    assert warned <= 8


def test_english_documents_never_rewrite_sent_text(app_module, caplog):
    checked, warned = _check(app_module, EN, [("en", lambda k: "en")], caplog)
    assert checked > 50 and warned == 0
