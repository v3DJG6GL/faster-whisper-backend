"""dictation_map.compile_map: the "dictated punctuation wins" prefix must stay
linear on a long hallucinated punctuation run.

A plain greedy `[.,:;!?…]+` prefix was retried from every start position
inside the run, each attempt backtracking char by char into the failed
lookahead — 8000 × "…" took over a second, quadratic in the run length. The
factory strip rule keeps `…` and `:` runs, so such a run reaches the map on
/transcribe, every streaming final and every captures reapply.
"""

import json
import os
import time

import pytest

from faster_whisper_backend import paths
from faster_whisper_backend.pipeline import dictation_map


@pytest.mark.parametrize("mark", ["…", ":", "."])
def test_long_punctuation_run_is_linear(mark):
    pattern, replace, _ = dictation_map.compile_map(
        {"Komma": ",", "Doppelpunkt": ":", "Hallo": "hi"})
    text = "a " + mark * 20000 + " b"
    t0 = time.perf_counter()
    assert pattern.sub(replace, text) == text
    assert time.perf_counter() - t0 < 0.2


def test_dictated_punctuation_still_replaces_whispers_marks():
    pattern, replace, _ = dictation_map.compile_map({"Komma": ",", "Punkt": "."})
    assert pattern.sub(replace, " HB 12... Komma 5") == " HB 12, 5"
    assert pattern.sub(replace, " Dosis 1, Komma, 5 mg") == " Dosis 1,, 5 mg"
    assert pattern.sub(replace, "12.5 und 12,5") == "12.5 und 12,5"
    assert pattern.sub(replace, "Ende…… Punkt") == "Ende."


@pytest.mark.parametrize("text, expected", [
    ("das z.B. hier", "das zB hier"),
    ("das z.B.", "das zB"),
    ("Z.B. so", "zB so"),
])
def test_key_ending_in_punctuation_matches(text, expected):
    """`\\b` after a key's trailing `.` needs a word character next, so a
    "z.B." key was saved and shown but never matched."""
    pattern, replace, _ = dictation_map.compile_map({"z.B.": "zB"})
    assert pattern.sub(replace, text) == expected


def test_word_bounded_key_still_needs_whole_word():
    pattern, replace, _ = dictation_map.compile_map({"Komma": ",", "Punkt": "."})
    assert pattern.sub(replace, "Kommando Punkt Punktzahl xKomma") == "Kommando . Punktzahl xKomma"
    assert pattern.sub(replace, "12... Komma 5") == "12, 5"


def _factory_de_map():
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        rules = json.load(fh)["PIPELINE_RULES"]
    return next(r["map"] for r in rules if r["name"] == "de-dictation-map")


@pytest.mark.parametrize("text, expected", [
    ("Eisen Bindestrich Infusion", "Eisen-Infusion"),
    ("Eisen Binde Strich Infusion", "Eisen-Infusion"),
    ("Das sind 50 Prozent mehr.", "Das sind 50% mehr."),
    ("5 Prozentzeichen", "5 %"),
    ("das Prozentual", "das Prozentual"),
])
def test_factory_keys_with_a_space_edge_match_in_running_text(text, expected):
    """The factory " Prozent" / " Bindestrich " keys sit right against the
    neighbouring word; a blanket `(?<!\\w)` / `(?!\\w)` around every key
    refused them, so every German transcript lost the dictated % and -."""
    pattern, replace, _ = dictation_map.compile_map(_factory_de_map())
    assert pattern.sub(replace, text) == expected


@pytest.mark.parametrize("key, text, expected", [
    ("-", "E-Mail und 5 - 6", "E-Mail und 5 – 6"),
    ("...", "Ende... und x...y", "Ende... und x...y"),
    ("...", "Ende ... und", "Ende – und"),
])
def test_punctuation_edge_keeps_its_word_bound(key, text, expected):
    """Only a space edge may sit against the neighbouring word: a "-" key
    must not rewrite the hyphen inside a compound."""
    pattern, replace, _ = dictation_map.compile_map({key: "–"})
    assert pattern.sub(replace, text) == expected
