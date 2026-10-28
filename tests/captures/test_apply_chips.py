"""Chip application (`_apply_chips_to_text` and its JS twin `_applyChipsToText`).

This text is what the export manifest carries and what every sample
transcript is built from, so a chip must rewrite the occurrence it marks and
nothing else: never a match inside a longer word, never an earlier repeat of
the same word, never its own previous replacement. The JS twin is executed
with node so the /captures preview is pinned to the same output.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

from faster_whisper_backend.captures import samples as capture_samples
from faster_whisper_backend.paths import REPO_ROOT

_CASES = [
    # A chip on the second "the" leaves the first alone.
    ("the cat sat on the mat",
     [{"idx": 4, "wrong": "the", "correct": "a"}],
     "the cat sat on a mat"),
    # Two chips on the same word: the second must not re-hit the first's
    # replacement ("therere").
    ("the cat and the dog",
     [{"idx": 0, "wrong": "the", "correct": "there"},
      {"idx": 3, "wrong": "the", "correct": "there"}],
     "there cat and there dog"),
    # Whole words only: "cat" inside "scat" is not the chip's word.
    ("scat cat",
     [{"idx": 1, "wrong": "cat", "correct": "dog"}],
     "scat dog"),
    # Short German word that is a substring of earlier words.
    ("Verderber der Ball",
     [{"idx": 1, "wrong": "der", "correct": "den"}],
     "Verderber den Ball"),
    # A word-count-changing span earlier in the text keeps a later chip on
    # its own occurrence.
    ("a b a b",
     [{"idx": 0, "wrong": "a b", "correct": "x"},
      {"idx": 2, "wrong": "a", "correct": "y"}],
     "x y b"),
    # Punctuation at the edge of `wrong` needs no word boundary there.
    ("Tag und Nacht 134/92",
     [{"idx": 3, "wrong": "134", "correct": "134 mmHg"}],
     "Tag und Nacht 134 mmHg/92"),
    # An anchorless chip takes the first whole-word occurrence.
    ("foobar foo foo",
     [{"wrong": "foo", "correct": "bar"}],
     "foobar bar foo"),
]



def _w(word, removed=False):
    return {"word": word, "removed": True} if removed else {"word": word}


# A chip's idx counts aligned karaoke words, and a removed raw word (a filler,
# a dictated "Komma" or "neue Zeile") owns no text token. With the aligned
# words passed in, the chip still rewrites the word it marks.
_WORD_CASES = [
    # Two removed fillers before the chip must not drag it onto the later
    # "der".
    ("der Patient der Arzt",
     [{"idx": 2, "wrong": "der", "correct": "den"}],
     [_w("äh", True), _w("äh", True), _w("der"), _w("Patient"), _w("der"),
      _w("Arzt")],
     "den Patient der Arzt"),
    # Dictated punctuation and line break: the first chip stays on the first
    # "die" and the second chip still finds its own.
    ("Hallo,\ndie Katze und die Maus",
     [{"idx": 4, "wrong": "die", "correct": "der"},
      {"idx": 7, "wrong": "die", "correct": "das"}],
     [_w("Hallo,"), _w("", True), _w("", True), _w("", True), _w("die"),
      _w("Katze"), _w("und"), _w("die"), _w("Maus")],
     "Hallo,\nder Katze und das Maus"),
    # The training token wins over the runtime one when it is attached.
    ("der Patient der Arzt",
     [{"idx": 2, "wrong": "der", "correct": "den"}],
     [{"word": "äh", "train_word": "", "train_removed": True},
      {"word": "äh", "train_word": "", "train_removed": True},
      _w("der"), _w("Patient"), _w("der"), _w("Arzt")],
     "den Patient der Arzt"),
]


@pytest.mark.parametrize("text, chips, expected", _CASES)
def test_apply_chips_rewrites_only_the_marked_word(text, chips, expected):
    assert capture_samples._apply_chips_to_text(text, chips) == expected


@pytest.mark.parametrize("text, chips, words, expected", _WORD_CASES)
def test_apply_chips_maps_idx_past_removed_words(text, chips, words, expected):
    assert capture_samples._apply_chips_to_text(text, chips, words) == expected


def test_default_transcript_aligns_chips_with_the_members_words():
    text, chips, words, expected = _WORD_CASES[0]
    member = {"text_for_training": text, "corrections": chips}
    assert capture_samples._build_default_transcript(
        [member], "space", words_for=lambda m: words) == expected


def _js_function(src, name):
    start = src.index("function " + name + "(")
    depth = 0
    for i in range(src.index("{", start), len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise AssertionError(name + " has no closing brace")


_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node unavailable to run the JS twin")
def test_js_twin_matches_server_chip_application():
    src = (pathlib.Path(REPO_ROOT) / "faster_whisper_backend" / "captures"
           / "templates" / "captures.html").read_text(encoding="utf-8")
    fn = _js_function(src, "_applyChipsToText")
    cases = ([[t, c, None] for t, c, _ in _CASES]
             + [[t, c, w] for t, c, w, _ in _WORD_CASES])
    script = (fn + "\nprocess.stdout.write(JSON.stringify("
              + json.dumps(cases, ensure_ascii=False)
              + ".map(function(c) { return _applyChipsToText(c[0], c[1], c[2]); })));")
    p = subprocess.run([_NODE, "-e", script], capture_output=True,
                       text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout) == ([e for _, _, e in _CASES]
                                    + [e for *_, e in _WORD_CASES])


def test_single_capture_ground_preview_uses_the_shared_helper(client):
    # The single-capture card's ground-truth preview used its own copy of the
    # first-occurrence substitution; it must go through the mirrored helper.
    t = client.get("/captures").text
    assert ("_applyChipsToText(state.finalText, state.corrections, state.words)"
            in t)
