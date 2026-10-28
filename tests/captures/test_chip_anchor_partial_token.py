"""An anchored chip whose word starts after an opening bracket or quote is
placed on that word: the "(" before the occurrence is part of the same token,
not one more word ahead of it."""

import pytest

from faster_whisper_backend.captures.samples import _apply_chips_to_text


@pytest.mark.parametrize("text, expected", [
    ("ja die (die Sache", "ja die (DER Sache"),
    ("ja die „die Sache", "ja die „DER Sache"),
])
def test_chip_after_an_opening_bracket_rewrites_the_marked_word(text, expected):
    words = [{"word": w} for w in text.split()]
    chip = {"idx": 2, "wrong": "die", "correct": "DER"}
    assert _apply_chips_to_text(text, [chip], words=words) == expected


def test_chip_on_the_plain_word_still_rewrites_it():
    text = "ja die (die Sache"
    words = [{"word": w} for w in text.split()]
    chip = {"idx": 1, "wrong": "die", "correct": "DER"}
    assert _apply_chips_to_text(text, [chip], words=words) == "ja DER (die Sache"
