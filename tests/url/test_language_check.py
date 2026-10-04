"""url/language_check.py — piece placement and the vote (pure)."""

import pytest

from faster_whisper_backend.url import language_check as lc


@pytest.mark.parametrize("duration,starts", [
    (None, [0.0]), (0, [0.0]), (30.0, [0.0]), (44.9, [0.0]),
    (45.0, [9.0, 22.5, 25.0]),          # 80 % would run past the end
    (600.0, [120.0, 300.0, 480.0]),
])
def test_piece_starts(duration, starts):
    assert lc.piece_starts(duration) == starts


@pytest.mark.parametrize("heard,language,verdict,also", [
    ([("de", 0.9), ("de", 0.8), ("de", 0.95)], "de", "detected", []),
    ([("de", 0.9), ("de", 0.8), ("en", 0.88)], "de", "mixed", ["en"]),
    ([("de", 0.9), ("de", 0.8), ("en", 0.4)], "de", "detected", []),   # en not confident
    ([("de", 0.9), ("en", 0.9), ("fr", 0.9)], "de", "unknown", []),     # no two agree
    ([("de", 0.6), ("de", 0.65), (None, 0.0)], "de", "unknown", []),    # agree, not confident
    ([(None, 0.0)] * 3, None, "unknown", []),
    ([("it", 0.75)], "it", "detected", []),                             # 1 of 1
    ([("it", 0.5)], "it", "unknown", []),
])
def test_vote(heard, language, verdict, also):
    got = lc.vote(heard)
    assert (got["language"], got["verdict"], got["also"]) == (language, verdict, also)


def test_vote_probability_is_the_agreeing_mean():
    assert lc.vote([("de", 0.9), ("de", 0.8), ("en", 0.95)])["probability"] == 0.85
