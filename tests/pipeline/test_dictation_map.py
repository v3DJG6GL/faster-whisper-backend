"""dictation_map.compile_map: the "dictated punctuation wins" prefix must stay
linear on a long hallucinated punctuation run.

A plain greedy `[.,:;!?…]+` prefix was retried from every start position
inside the run, each attempt backtracking char by char into the failed
lookahead — 8000 × "…" took over a second, quadratic in the run length. The
factory strip rule keeps `…` and `:` runs, so such a run reaches the map on
/transcribe, every streaming final and every captures reapply.
"""

import copy
import json
import os
import time

import pytest

from faster_whisper_backend import paths
from faster_whisper_backend.pipeline import dictation_map
from faster_whisper_backend.pipeline import engine as pl_engine


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


def test_key_does_not_match_between_combining_marks():
    """`\\w` has no combining marks, so the bounds saw a word edge at every
    vowel sign: a "लत" / "คร" key (both offered as autocomplete fragments)
    rewrote the inside of "बोलते" / "ครับ" to "बोXे" / "Yับ"."""
    pattern, replace, _ = dictation_map.compile_map(
        {"लत": "X", "คร": "Y", "बोलते": "Z"})
    assert pattern.sub(replace, "हम बोलते हैं ครับ") == "हम Z हैं ครับ"
    assert pattern.sub(replace, "लत คร") == "X Y"


def test_key_does_not_match_next_to_a_zero_width_joiner():
    """A ZWNJ / ZWJ is neither `\\w` nor a mark, so the bounds saw a word
    edge at it: a Persian "می" / "ها" key rewrote the verb prefix of
    می + ZWNJ + خواهم and the plural suffix of کتاب + ZWNJ + ها."""
    pattern, replace, _ = dictation_map.compile_map({"می": "X", "ها": "Y", "र्": "Z"})
    text = "من می\u200cخواهم کتاب\u200cها र्\u200dयाम"
    assert pattern.sub(replace, text) == text
    assert pattern.sub(replace, "می ها र्") == "X Y Z"


def test_large_non_ascii_map_stays_fast():
    """One mark class per key edge made compiling a 500-key Cyrillic map
    ~0.6 s and each sub of a 20k-char transcript ~1 s, 5-9x the plain `\\w`
    bounds; the class was walked for every key at every text position."""
    letters = "абвгдежзиклмнопрстуфхцчшщэюя"
    words = [letters[i % 28] + letters[i // 28 % 28] + letters[i // 784 % 28] + "ок"
             for i in range(1000)]
    keys, others = words[:500], words[500:]
    text = " ".join(f"{k} {o}," for k, o in zip(keys, others)) * 3
    t0 = time.perf_counter()
    pattern, replace, _ = dictation_map.compile_map({k: "x" for k in keys})
    out = pattern.sub(replace, text)
    assert time.perf_counter() - t0 < 0.6
    assert out == " ".join(f"x {o}," for o in others) * 3


def _factory_rules():
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["PIPELINE_RULES"]


def _factory_de_map():
    return next(r["map"] for r in _factory_rules() if r["name"] == "de-dictation-map")


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


@pytest.mark.parametrize("key, value, text, expected", [
    (" Komma", ",", "12... Komma 5", "12, 5"),
    (" Punkt", ".", "Hallo, Punkt", "Hallo."),
    (" Punkt", ".", "Hallo. Punkt", "Hallo."),
])
def test_dictated_punctuation_wins_for_a_space_led_key(key, value, text, expected):
    """The possessive spaces after Whisper's marks ate the key's own leading
    space, so the prefix never applied and Whisper's mark stayed in front."""
    pattern, replace, _ = dictation_map.compile_map({key: value})
    assert pattern.sub(replace, text) == expected


@pytest.mark.parametrize("key", [" ", "  "])
def test_whitespace_only_key_is_ignored(key):
    """A stored config from before the schema refused it must not turn every
    space of every transcript into the key's value."""
    pattern, replace, lookup = dictation_map.compile_map({key: "x", "Komma": ","})
    assert pattern.sub(replace, "hallo Welt  Komma zwei") == "hallo Welt  , zwei"
    assert key not in lookup


@pytest.mark.parametrize("text, expected", [
    ("İstanbul ve", "IST ve"),
    ("izmir yolu", "IZM yolu"),
    ("ς x", "S x"),
])
def test_key_matched_by_sre_case_folding_is_applied(text, expected):
    """The pattern matches with sre's case folding, the value was looked up by
    `str.lower()`; the two disagree on "İ" (lowers to "i" + U+0307) and "ς",
    so the key matched and the text was silently left unchanged."""
    pattern, replace, _ = dictation_map.compile_map({"istanbul": "IST", "İzmir": "IZM", "σ": "S"})
    assert pattern.sub(replace, text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Blutdruck 120. Schrägstrich 80.", "Blutdruck 120/80"),
    ("Blutdruck 120... Schrägstrich 80", "Blutdruck 120/80"),
    ("Blutdruck 120, Schrägstrich 80.", "Blutdruck 120/80"),
    # "-" stays out: "12.-15. Mai" is a German ordinal range.
    ("Seite 12. Bindestrich 15.", "Seite 12.-15"),
    ("Patient. Schrägstrich Angehörige.", "Patient / Angehörige"),
])
def test_dictated_slash_eats_whispers_pause_mark(app_module, text, expected):
    """The prefix took only `. , : ; ! ?` values, so Whisper's period at the
    pause after a number stayed in front of a dictated Schrägstrich:
    "Blutdruck 120. Schrägstrich 80." became "Blutdruck 120. / 80"."""
    app_module.cfg.PIPELINE_RULES = copy.deepcopy(_factory_rules())
    pl_engine.rebuild_caches()
    assert pl_engine._postprocess_text(text, model_name="", language="de") == expected
