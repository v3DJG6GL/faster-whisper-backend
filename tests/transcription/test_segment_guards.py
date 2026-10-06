"""transcription.segment_guards — the three tail-cut rules (pure, no app fixture).

The loop fixture carries the REAL word timings the production server returned on
2026-09-19 for capture 7a8db5d1 (large-v3, hotwords + previous sentence as
prompt): real words end at 5.54 s, then a made-up tail stacked at the end.
"""

from types import SimpleNamespace as NS

from faster_whisper_backend.transcription import segment_guards as sg


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


def test_real_short_tail_from_the_server_is_cut():
    """2026-09-19 20:47, prompt "…Tag?, Fragezeichen": the 3-word tail the first
    version let through — real timings and probabilities from the server log."""
    words = [_w(" zwei", 2.91, 3.55, 0.44), _w(" Tabletten", 4.62, 5.52, 0.99),
             _w(" zu", 5.52, 5.58, 0.01), _w(" nehmen?,", 5.58, 6.12, 0.10),
             _w(" Fragezeichen", 6.12, 6.12, 0.33)]
    assert sg.find_tail_cut(words, "", burst=8, zero_tail=2, repeats=3) == (2, ["zero_tail"])
    seg = _seg(words)
    sg.apply_tail_guards(seg, zero_tail=2)
    assert seg.text == " zwei Tabletten" and seg.end == 5.52


def test_confident_word_before_a_zero_length_last_word_stays():
    # a dictated command that got no duration: one zero-length word, the word
    # before it is confident → nothing is cut
    words = [_w(" pro", 2.1, 2.5, 0.97), _w(" Tag", 2.5, 3.4, 0.95),
             _w(" Fragezeichen", 3.4, 3.4, 0.6)]
    assert sg.find_tail_cut(words, "", zero_tail=2) is None


def test_confident_word_between_two_single_zero_length_words_stays():
    # DTW gave a mid-segment word and the last word the same frame: the long
    # confident word between them is speech, not an absorber before a pile
    words = [_w(" Das", 0.0, 0.4), _w(" ist", 0.4, 0.4),
             _w(" wirklich", 0.4, 1.9, 0.99), _w(" gut", 1.9, 1.9)]
    assert sg.find_tail_cut(words, "", zero_tail=2) is None
    # alternating collapsed / confident words never chain into a tail
    chain = [_w(" a", 0.0, 0.4), _w(" b", 0.4, 0.4), _w(" c", 0.4, 0.9, 0.95),
             _w(" d", 0.9, 0.9), _w(" e", 0.9, 1.5, 0.95), _w(" f", 1.5, 1.5)]
    assert sg.find_tail_cut(chain, "", zero_tail=2) is None


def test_sandwiched_absorber_is_taken_once_and_only_when_short_or_unsure():
    pile = [_w(f" x{i}", 3.0, 3.0) for i in range(3)]
    # second sandwich further back: the walk stops at the first one's far side
    words = [_w(" a", 0.0, 0.5), _w(" b", 0.5, 0.5), _w(" c", 0.5, 0.5),
             _w(" d", 0.5, 1.0), _w(" e", 1.0, 1.0), _w(" f", 1.0, 1.0),
             _w(" g", 1.0, 1.5)] + pile
    assert sg.find_tail_cut(words, "", zero_tail=2) == (4, ["zero_tail"])
    long_sure = [_w(" a", 0.0, 0.5), _w(" b", 0.5, 0.5), _w(" lang", 0.5, 3.0, 0.99)] + pile
    assert sg.find_tail_cut(long_sure, "", zero_tail=2) == (3, ["zero_tail"])
    long_unsure = long_sure[:2] + [_w(" lang", 0.5, 3.0, 0.05)] + pile
    assert sg.find_tail_cut(long_unsure, "", zero_tail=2) == (1, ["zero_tail"])


def test_zero_tail_min_words_one_is_off():
    # the help text promises a single zero-length last word is always kept
    words = [_w(" pro", 2.1, 2.5), _w(" Tag", 2.5, 3.4), _w(" Fragezeichen", 3.4, 3.4)]
    assert sg.find_tail_cut(words, "", zero_tail=1) is None


def test_fast_real_words_before_the_pile_survive_the_burst_rule():
    # short function words start closer than 1/8 s but have a spoken length
    words = [_w(" a", 4.0, 5.05), _w(" und", 5.05, 5.30), _w(" die", 5.30, 5.40),
             _w(" ist", 5.40, 5.54), _w(" da", 5.54, 6.02)] \
        + [_w(f" x{i}", 6.02, 6.02) for i in range(9)]
    idx, rules = sg.find_tail_cut(words, "", burst=8, zero_tail=2)
    assert words[idx].word == " x0"
    assert rules == ["burst", "zero_tail"]


def test_burst_keeps_spoken_words_after_an_early_squeezed_word():
    # A real one-frame word close to its successor inside the last second,
    # then TWO spoken words (normal length, normal gap) before the pile: the
    # cut is tail-anchored, so it lands on the pile, not on " a".
    words = [_w(f" w{i}", 0.3 * i, 0.3 * i + 0.3) for i in range(5)] + [
        _w(" a", 4.40, 4.44), _w(" real", 4.50, 4.75),
        _w(" real2", 4.75, 4.95)] + [_w(f" x{i}", 4.95, 4.95) for i in range(8)]
    idx, rules = sg.find_tail_cut(words, "", burst=8)
    assert rules == ["burst"]
    assert words[idx].word == " x0"


def test_burst_fallback_never_starts_at_a_real_one_frame_word():
    # The same squeezed " a" with two spoken words after it, but a pile of
    # fast words that are not short (0.07 s long, 0.07 s apart): no close
    # word looks made up, so the fallback decides. It must start at the
    # pile, not at " a" — that would drop "real real2" with it.
    words = [_w(f" w{i}", 0.3 * i, 0.3 * i + 0.3) for i in range(5)] + [
        _w(" a", 4.40, 4.44), _w(" real", 4.50, 4.625),
        _w(" real2", 4.625, 4.70)] + [
        _w(f" x{i}", 4.75 + 0.07 * i, 4.82 + 0.07 * i) for i in range(9)]
    idx, rules = sg.find_tail_cut(words, "", burst=8)
    assert rules == ["burst"]
    assert words[idx].word == " x0"


def test_burst_falls_back_to_the_first_close_word_when_none_looks_made_up():
    words = [_w(" a", 0.0, 1.0)] + [_w(f" x{i}", 1.0 + 0.1 * i, 1.1 + 0.1 * i)
                                     for i in range(10)]
    assert sg.find_tail_cut(words, "", burst=8) == (1, ["burst"])


def test_unplaceable_word_starts_do_not_raise():
    nan = float("nan")
    words = [_w(" a", nan, nan), _w(" b", nan, nan)]
    assert sg.find_tail_cut(words, "", burst=8) is None


def test_only_one_unsure_absorber_is_taken():
    # two unsure normal-length words in a row before the zero-length end: only
    # the last one can be the absorber, and the walk stops at the other
    words = [_w(" eins", 0.0, 0.5, 0.9), _w(" zwei", 0.5, 1.0, 0.1),
             _w(" drei", 1.0, 1.6, 0.1), _w(" x", 1.6, 1.6, 0.3)]
    assert sg.find_tail_cut(words, "", zero_tail=2) == (2, ["zero_tail"])


def test_low_probability_alone_never_cuts():
    # no zero-length last word → the rule has no anchor
    words = [_w(" eins", 0.0, 0.5, 0.05), _w(" zwei", 0.5, 0.6, 0.05), _w(" drei", 0.6, 1.2, 0.05)]
    assert sg.find_tail_cut(words, "", zero_tail=2) is None


def test_missing_probability_is_not_low():
    words = [_w(" eins", 0.0, 0.5), NS(word=" zwei", start=0.5, end=1.2),
             _w(" x", 1.2, 1.2)]
    assert sg.find_tail_cut(words, "", zero_tail=2) is None


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
    # The 4-word prefix makes the input long enough that period 4 is really
    # searched: only the repeat count rejects two copies.
    toks = ["x", "y", "z", "w"] + ["a", "b", "c", "d"] * 2
    assert sg.find_tail_cut(_spaced(toks), "", repeats=3) is None
    toks = ["x", "y", "z", "w"] + ["a", "b", "c", "d"] * 3
    assert sg.find_tail_cut(_spaced(toks), "", repeats=3) == (8, ["repeat"])


def test_loop_ending_in_a_doubled_word_is_still_cut():
    # The doubled "gut" is a short run INSIDE the looped phrase, not the
    # loop's primitive period: the longer 4-word run wins.
    toks = ["das", "ist", "gut", "gut"] * 3
    assert sg.find_tail_cut(_spaced(toks), "", repeats=2) == (4, ["repeat"])


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
    assert seg.end == seg.start == 1.0, "the made-up end must not survive"


def test_all_off_is_a_no_op():
    seg = _loop_seg()
    assert sg.apply_tail_guards(seg) is None
    assert len(seg.words) == 20


class _Boom:
    text = "x"

    @property
    def words(self):
        raise RuntimeError("boom")


def test_guard_never_raises():
    assert sg.apply_tail_guards(NS(words=[object()], text=None), burst=8, zero_tail=2) is None
    # a raising property is not swallowed by getattr's default: only the
    # guard's own except can turn this into None
    assert sg.apply_tail_guards(_Boom(), burst=8, zero_tail=2) is None


def test_tail_words_diag_only_when_a_zero_length_word_is_present():
    assert sg.tail_words_diag(_seg([_w(*x) for x in _REAL])) is None
    line = sg.tail_words_diag(_seg([_w(" a", 0.0, 0.5), _w(" b", 0.5, 0.5, 0.31)]))
    assert "'b'@0.50-0.50/p0.31" in line


def test_describe_cut():
    s = sg.describe_cut({"rules": ["burst", "zero_tail"], "n": 12, "from": 5.54,
                         "text": " zu nehmen"})
    assert s == "burst+zero_tail · 12 words from 5.54s: ' zu nehmen'"


# ---- head echo of the decode prompt (SEGMENT_HEAD_ECHO_MIN_WORDS) -------------
#
# Live dictation sends the previous utterances as prompt; the decode sometimes
# writes their last words again before the new speech. Those words have no audio
# behind them — stacked on the first frames, zero-length or squeezed.

_PROMPT = "Der Patient nimmt zwei Tabletten am Abend."
_NEW = [(" Morgen", 0.52, 0.9, 0.97), (" kommt", 0.9, 1.2, 0.95), (" er", 1.2, 1.3, 0.9),
        (" wieder.", 1.3, 1.7, 0.93)]


def _echo(words):
    return _seg([_w(*x) for x in words + _NEW])


def test_head_echo_is_cut_to_the_new_speech():
    # 'zwei' zero-length, 'Tabletten' squeezed + unsure, 'am' under 0.07 s,
    # 'Abend.' the one unsure absorber of the time before the real speech
    seg = _echo([(" zwei", 0.0, 0.0), (" Tabletten", 0.0, 0.1, 0.05),
                 (" am", 0.1, 0.14), (" Abend.", 0.14, 0.5, 0.08)])
    info = sg.apply_head_echo_guard(seg, _PROMPT, 3)
    assert info["rules"] == ["head_echo"] and info["n"] == 4
    assert info["text"] == " zwei Tabletten am Abend."
    assert (info["from"], info["to"]) == (0.0, 0.5)
    assert seg.text == " Morgen kommt er wieder."
    assert [w.word for w in seg.words][0] == " Morgen"
    assert seg.start == 0.52


def test_spoken_repeat_of_the_prompt_stays():
    words = [(" am", 0.0, 0.3, 0.9), (" Abend", 0.3, 0.8, 0.95)]
    seg = _echo([(" zwei", 0.0, 0.0)] + [(" Tabletten", 0.0, 0.0)] + words)
    before = seg.text
    assert sg.apply_head_echo_guard(seg, _PROMPT, 3) is None
    assert seg.text == before and len(seg.words) == 8
    # …and that near miss is what the INFO diagnostic line is for
    diag = sg.head_words_diag(seg, _PROMPT, 3)
    assert diag.startswith("4 words match the prompt: ")
    assert "'Abend'@0.30-0.80/p0.95" in diag


def test_only_one_unsure_absorber():
    # two unsure words of speaking length: only one may be the absorber
    seg = _echo([(" am", 0.0, 0.4, 0.1), (" Abend", 0.4, 0.8, 0.1)])
    assert sg.find_head_echo(seg.words, _PROMPT, 2) is None
    seg = _echo([(" am", 0.0, 0.4, 0.1), (" Abend", 0.4, 0.45, 0.1)])    # squeezed
    assert sg.find_head_echo(seg.words, _PROMPT, 2) == 2


def test_head_echo_min_words_and_off():
    words = [(" am", 0.0, 0.0), (" Abend", 0.0, 0.0)]
    assert sg.find_head_echo(_echo(words).words, _PROMPT, 3) is None     # 2 < 3
    assert sg.find_head_echo(_echo(words).words, _PROMPT, 2) == 2
    for off in (0, 1, None):
        assert sg.find_head_echo(_echo(words).words, _PROMPT, off) is None


def test_head_echo_needs_the_prompt_end_not_just_any_prompt_words():
    # "zwei Tabletten" is in the prompt, but not at its end
    seg = _echo([(" zwei", 0.0, 0.0), (" Tabletten", 0.0, 0.0), (" nehmen", 0.0, 0.0)])
    assert sg.find_head_echo(seg.words, _PROMPT, 2) is None
    assert sg.find_head_echo(seg.words, "", 2) is None


def test_head_echo_normalizes_case_and_punctuation():
    seg = _echo([(" Am", 0.0, 0.0), (" abend,", 0.0, 0.0), (" …", 0.0, 0.0)])
    assert sg.find_head_echo(seg.words, "Er kommt am Abend. …", 2) == 3


def test_head_echo_falls_back_to_a_shorter_made_up_match():
    # The longest match (4 words) ends in two spoken words; the 2-word match in
    # front of them is made up and is cut on its own.
    prompt = "Befund Doppelpunkt neue Zeile neue Zeile"
    seg = _seg([_w(" neue", 0.0, 0.0), _w(" Zeile", 0.0, 0.0),
                _w(" neue", 0.2, 0.6, 0.9), _w(" Zeile", 0.6, 1.0, 0.9),
                _w(" Lunge", 1.1, 1.5, 0.9)])
    assert sg.find_head_echo(seg.words, prompt, 2) == 2


def test_whole_segment_echo_empties_it():
    seg = _seg([_w(" am", 1.0, 1.0), _w(" Abend.", 1.0, 1.02)])
    info = sg.apply_head_echo_guard(seg, _PROMPT, 2)
    assert info["n"] == 2 and seg.words == [] and not seg.text.strip()
    assert seg.start == seg.end == 1.02


def test_head_echo_needs_word_timestamps():
    seg = NS(start=0.0, end=2.0, text=" am Abend Morgen kommt er", words=[])
    assert sg.apply_head_echo_guard(seg, _PROMPT, 2) is None
    assert sg.head_words_diag(seg, _PROMPT, 2) is None


def test_head_echo_guard_never_raises():
    assert sg.apply_head_echo_guard(_Boom(), _PROMPT, 2) is None
    assert sg.head_words_diag(_Boom(), _PROMPT, 2) is None


def test_head_text_walk_falls_back_to_the_kept_words():
    seg = _echo([(" am", 0.0, 0.0), (" Abend", 0.0, 0.0)])
    seg.text = " something else entirely"
    info = sg.apply_head_echo_guard(seg, _PROMPT, 2)
    assert info["text_rebuilt"] is True
    assert seg.text == " Morgen kommt er wieder."


def test_prompt_tail_text_prefers_the_initial_prompt():
    assert sg.prompt_tail_text({"initial_prompt": "a b", "hotwords": "c"}) == "a b"
    assert sg.prompt_tail_text({"initial_prompt": None, "hotwords": "c d"}) == "c d"
    assert sg.prompt_tail_text({"initial_prompt": "  ", "hotwords": ""}) == ""
    assert sg.prompt_tail_text(None) == ""


def test_describe_head_cut():
    s = sg.describe_head_cut({"rules": ["head_echo"], "n": 2, "from": 0.0, "to": 0.5,
                              "text": " am Abend"})
    assert s == "prompt echo · 2 words 0.00-0.50s: ' am Abend'"
