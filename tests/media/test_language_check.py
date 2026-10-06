"""media/language_check.py — piece placement and the vote (pure)."""

import pytest

from faster_whisper_backend.media import language_check as lc


@pytest.mark.parametrize("duration,starts", [
    (None, [0.0]), (0, [0.0]), (30.0, [0.0]), (44.9, [0.0]),
    (45.0, [9.0, 22.5, 25.0]),          # 80 % would run past the end
    (600.0, [120.0, 300.0, 480.0]),
    (float("nan"), [0.0]), (float("inf"), [0.0]),
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


@pytest.mark.parametrize("durations,starts,seconds,margin,runs", [
    # 4 s segments; a piece at 5 s for 6 s ± 1 s spans [4, 12] → segments 1–2.
    ([4.0] * 12, [5.0], 6.0, 1.0, [(1, 3, 1.0)]),
    # Edges: a window ending exactly on a boundary takes no extra segment.
    ([4.0] * 12, [4.0], 8.0, 0.0, [(1, 3, 0.0)]),
    # The margin reaches into the neighbours on both sides: [3.5, 8.5].
    ([4.0] * 12, [4.5], 3.0, 1.0, [(0, 3, 4.5)]),
    # From the start, and clipped at the end of the stream.
    ([4.0] * 3, [0.0, 9.0], 20.0, 1.0, [(0, 3, 0.0), (2, 3, 1.0)]),
    # Past the end: nothing.
    ([4.0] * 3, [30.0], 5.0, 1.0, [(3, 3, 0.0)]),
    # Uneven durations (EXTINF / DASH fragment lengths): edges 0 2 8 8.5 12 22.
    ([2.0, 6.0, 0.5, 3.5, 10.0], [7.0, 8.4], 4.0, 0.5, [(1, 4, 5.0), (1, 5, 6.4)]),
    ([], [0.0], 20.0, 1.0, [(0, 0, 0.0)]),
])
def test_select_segments(durations, starts, seconds, margin, runs):
    assert lc.select_segments(durations, starts, seconds, margin) == runs
