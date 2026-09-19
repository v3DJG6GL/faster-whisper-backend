"""core.segment_guards — the three tail-cut rules (pure, no app fixture).

The loop fixture carries the REAL word timings the production server returned on
2026-09-19 for capture 7a8db5d1 (large-v3, hotwords + previous sentence as
prompt): real words end at 5.54 s, then a made-up tail stacked at the end.
"""

from types import SimpleNamespace as NS

from faster_whisper_backend.core import segment_guards as sg


def _w(word, start, end, p=0.9):
    return NS(word=word, start=start, end=end, probability=p)


def _seg(words, text=None):
    text = "".join(w.word for w in words) if text is None else text
    return NS(start=words[0].start if words else 0.0,
              end=words[-1].end if words else 0.0, text=text, words=list(words))


_REAL = [(" Besser", 0.32, 0.74), (" wäre", 0.74, 1.0), (" wahrscheinlich", 1.0, 1.42),
         (" einmal", 1.42, 2.1), (" pro", 2.1, 2.5), (" Tag", 2.5, 2.86),
         (" zwei", 2.86, 3.52), (" Tabletten", 4.6, 5.54)]
_TAIL = [(" zu", 5.54, 5.58), (" nehmen", 5.58, 5.58), (" und", 5.58, 6.02),
         (" die", 6.02, 6.02), (" Doppelpunkte", 6.02, 6.02),
         (" abzuschliessen", 6.02, 6.02), (" und", 6.02, 6.02), (" die", 6.02, 6.02),
         (" Doppelpunkte", 6.02, 6.02), (" abzuschliessen", 6.02, 6.02),
         (" und", 6.02, 6.02), (" die", 6.02, 6.02)]
_CLEAN_TEXT = " Besser wäre wahrscheinlich einmal pro Tag zwei Tabletten"


def _loop_seg():
    return _seg([_w(*x) for x in _REAL + _TAIL])


def test_real_loop_is_cut_at_the_first_made_up_word():
    seg = _loop_seg()
    info = sg.apply_tail_guards(seg, burst=8, zero_tail=2, repeats=3)
    assert seg.text == _CLEAN_TEXT
    assert [w.word for w in seg.words][-1] == " Tabletten"
    assert seg.end == 5.54
    assert info["n"] == 12 and info["from"] == 5.54
    assert "burst" in info["rules"] and "zero_tail" in info["rules"]
    assert info["text"].startswith(" zu nehmen und die")


def test_burst_alone_cuts_at_zu_and_zero_tail_alone_at_nehmen():
    words = [_w(*x) for x in _REAL + _TAIL]
    assert sg.find_tail_cut(words, "", burst=8) == (8, ["burst"])
    # zero-length run + the sandwiched absorber " und" + " nehmen"(0)
    assert sg.find_tail_cut(words, "", zero_tail=2) == (9, ["zero_tail"])


def test_real_words_inside_the_look_back_second_are_kept():
    # three normally spaced real words start within 1 s of the pile
    words = [_w(" a", 4.0, 4.3), _w(" b", 5.04, 5.3), _w(" c", 5.30, 5.54),
             _w(" d", 5.54, 5.9)] + [_w(f" x{i}", 6.02, 6.02) for i in range(10)]
    idx, rules = sg.find_tail_cut(words, "", burst=8)
    assert rules == ["burst"]
    assert words[idx].word == " x0"


def test_clean_speech_is_untouched():
    seg = _seg([_w(*x) for x in _REAL])
    assert sg.apply_tail_guards(seg, burst=8, zero_tail=2, repeats=3) is None
    assert seg.text == _CLEAN_TEXT and len(seg.words) == 8


def test_single_zero_length_last_word_is_kept():
    words = [_w(" eins", 0.0, 0.5), _w(" zwei", 0.5, 1.0), _w(" Punkt", 1.0, 1.0)]
    assert sg.find_tail_cut(words, "", zero_tail=2) is None


def test_zero_run_reaches_minimum():
    words = [_w(" eins", 0.0, 0.5), _w(" zwei", 0.5, 1.0),
             _w(" zu", 1.0, 1.0), _w(" nehmen", 1.0, 1.0)]
    assert sg.find_tail_cut(words, "", zero_tail=2) == (2, ["zero_tail"])
    assert sg.find_tail_cut(words, "", zero_tail=3) is None


def test_real_last_word_that_absorbed_leftover_time_is_never_cut():
    # spoken word, then the zero run: the non-zero word before the run has a
    # spoken (non-zero) word before it → it stays.
    words = [_w(" eins", 0.0, 0.5), _w(" zwei", 0.5, 2.0),
             _w(" x", 2.0, 2.0), _w(" y", 2.0, 2.0)]
    assert sg.find_tail_cut(words, "", zero_tail=2) == (2, ["zero_tail"])


def test_one_frame_word_is_not_zero_length():
    words = [_w(" eins", 0.0, 0.5), _w(" a", 0.5, 0.52), _w(" b", 0.52, 0.54)]
    assert sg.find_tail_cut(words, "", zero_tail=2) is None


def _spaced(tokens, step=0.5):
    return [_w(" " + t, i * step, i * step + 0.4) for i, t in enumerate(tokens)]


def test_repeated_commands_are_never_touched():
    assert sg.find_tail_cut(_spaced(["Neue", "Zeile"] * 6), "", repeats=3) is None
    assert sg.find_tail_cut(_spaced(["ja"] * 9), "", repeats=3) is None


def test_phrase_loop_with_incomplete_last_copy_keeps_first_copy():
    toks = ["Das", "ist", "gut"] + ["und", "die", "Behandlung", "verhindern"] * 3 + ["und", "die"]
    words = _spaced(toks)
    idx, rules = sg.find_tail_cut(words, "", repeats=3)
    assert rules == ["repeat"] and idx == 7
    seg = _seg(words)
    sg.apply_tail_guards(seg, repeats=3)
    assert seg.text == " Das ist gut und die Behandlung verhindern"


def test_two_copies_are_below_three_repeats():
    toks = ["a", "b", "c", "d"] * 2
    assert sg.find_tail_cut(_spaced(toks), "", repeats=3) is None


def test_refrain_in_the_middle_is_left_alone():
    toks = ["la", "li", "lu", "le"] * 3 + ["und", "dann", "war", "Schluss"]
    assert sg.find_tail_cut(_spaced(toks), "", repeats=3) is None


def test_repeat_compares_without_case_and_punctuation():
    toks = ["und", "die", "Pause,", "Und", "die", "pause", "und", "die", "Pause."]
    assert sg.find_tail_cut(_spaced(toks), "", repeats=3) == (3, ["repeat"])


def test_no_words_runs_the_repeat_rule_on_text():
    seg = NS(start=0.0, end=9.0, words=None,
             text=" gut und dann weiter und dann weiter und dann weiter")
    info = sg.apply_tail_guards(seg, burst=8, zero_tail=2, repeats=3)
    assert seg.text == " gut und dann weiter"
    assert info["rules"] == ["repeat"] and info["from"] is None and info["n"] == 6


def test_text_walk_survives_text_that_is_not_the_joined_words():
    words = [_w("eins", 0.0, 0.5), _w("zwei", 0.5, 1.0), _w("x", 1.0, 1.0), _w("y", 1.0, 1.0)]
    seg = _seg(words, text="eins zwei x y")       # test-fake style: no leading spaces
    sg.apply_tail_guards(seg, zero_tail=2)
    assert seg.text == "eins zwei"


def test_text_walk_falls_back_to_joined_words():
    words = [_w(" eins", 0.0, 0.5), _w(" zwei", 0.5, 1.0), _w(" x", 1.0, 1.0), _w(" y", 1.0, 1.0)]
    seg = _seg(words, text="ganz anderer Text")
    info = sg.apply_tail_guards(seg, zero_tail=2)
    assert seg.text == " eins zwei" and info["text_rebuilt"] is True


def test_everything_cut_leaves_empty_text():
    seg = _seg([_w(f" x{i}", 1.0, 1.0) for i in range(12)])
    info = sg.apply_tail_guards(seg, burst=8, zero_tail=2)
    assert seg.text == "" and seg.words == [] and info["n"] == 12


def test_all_off_is_a_no_op():
    seg = _loop_seg()
    assert sg.apply_tail_guards(seg) is None
    assert len(seg.words) == 20


def test_guard_never_raises():
    assert sg.apply_tail_guards(NS(words=[object()], text=None), burst=8, zero_tail=2) is None


def test_tail_words_diag_only_when_a_zero_length_word_is_present():
    assert sg.tail_words_diag(_seg([_w(*x) for x in _REAL])) is None
    line = sg.tail_words_diag(_seg([_w(" a", 0.0, 0.5), _w(" b", 0.5, 0.5, 0.31)]))
    assert "'b'@0.50-0.50/p0.31" in line


def test_describe_cut():
    s = sg.describe_cut({"rules": ["burst", "zero_tail"], "n": 12, "from": 5.54,
                         "text": " zu nehmen"})
    assert s == "burst+zero_tail · 12 words from 5.54s: ' zu nehmen'"
