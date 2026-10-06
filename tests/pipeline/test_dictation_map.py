"""dictation_map.compile_map: the "dictated punctuation wins" prefix must stay
linear on a long hallucinated punctuation run.

A plain greedy `[.,:;!?…]+` prefix was retried from every start position
inside the run, each attempt backtracking char by char into the failed
lookahead — 8000 × "…" took over a second, quadratic in the run length. The
factory strip rule keeps `…` and `:` runs, so such a run reaches the map on
/transcribe, every streaming final and every captures reapply.
"""

import time

import pytest

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
