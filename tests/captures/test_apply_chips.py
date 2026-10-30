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
import time

import pytest

from faster_whisper_backend.captures import samples as capture_samples
from faster_whisper_backend.paths import REPO_ROOT
from faster_whisper_backend.pipeline import engine as pl_engine

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
    # A combining mark is part of its word: "लत" ends before the vowel sign
    # inside "बोलते", "คร" before the one inside "ครับ", "E" before the NFD
    # accent of "Éric", and ZWNJ joins "روم" to "می‌".
    ("वे बोलते हैं लत",
     [{"wrong": "लत", "correct": "X"}],
     "वे बोलते हैं X"),
    ("ครับ คร",
     [{"wrong": "คร", "correct": "X"}],
     "ครับ X"),
    ("Éric E",
     [{"wrong": "E", "correct": "X"}],
     "Éric X"),
    ("می‌روم روم",
     [{"wrong": "روم", "correct": "X"}],
     "می‌روم X"),
    # Chinese, Japanese and Thai put no space between words: a chip on a
    # word between two letters applies, so long as it does not part a
    # letter from its combining mark (ั in "ครับ"), a following vowel (า in
    # "ปลา") or a small kana (ょ in "ちょっと").
    ("我喜欢猫和狗",
     [{"wrong": "猫", "correct": "鱼"}],
     "我喜欢鱼和狗"),
    ("今日は良い天気",
     [{"wrong": "良い", "correct": "悪い"}],
     "今日は悪い天気"),
    ("สวัสดีครับ",
     [{"wrong": "ครับ", "correct": "ค่ะ"}],
     "สวัสดีค่ะ"),
    ("ครับ",
     [{"wrong": "คร", "correct": "X"}],
     "ครับ"),
    ("ปลา",
     [{"wrong": "ปล", "correct": "X"}],
     "ปลา"),
    ("ちょっと待って",
     [{"wrong": "ち", "correct": "X"}],
     "ちょっと待って"),
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
    # A multi-word chip across a dictated line break: its `wrong` joins the
    # words with a space, the text has "\n" there.
    ("Satz eins.\nSatz zwei",
     [{"idx": 1, "idx_end": 4, "wrong": "eins. Satz", "correct": "drei. Satz"}],
     [_w(" Satz"), _w(" eins."), _w("", True), _w("", True), _w(" Satz"),
      _w(" zwei")],
     "Satz drei. Satz zwei"),
    # A verbatim match far from idx must not beat the flexible match at it:
    # the chip marks the first pair, the line break is dictated.
    ("Hallo\nWelt und dann Hallo Welt",
     [{"idx": 0, "idx_end": 1, "wrong": "Hallo Welt", "correct": "Servus"}],
     [_w(" Hallo"), _w(" Welt"), _w(" und"), _w(" dann"), _w(" Hallo"),
      _w(" Welt")],
     "Servus und dann Hallo Welt"),
    # A chip saved on a form the pipeline has since changed ("die" is now
    # "Die" at sentence start) applies on the words at its idx, as the page
    # shows it, not on a later "die" the stale `wrong` still matches.
    ("Die Katze und die Maus",
     [{"idx": 0, "wrong": "die", "correct": "Der"}],
     [_w(" Die"), _w(" Katze"), _w(" und"), _w(" die"), _w(" Maus")],
     "Der Katze und die Maus"),
    # Same for a training token: the chip was saved on the runtime "zwei",
    # the word at its idx now reads "2".
    ("Er sagt 2 mal zwei",
     [{"idx": 2, "wrong": "zwei", "correct": "drei"}],
     [_w(" Er"), _w(" sagt"), {"word": " zwei", "train_word": " 2"},
      _w(" mal"), _w(" zwei")],
     "Er sagt drei mal zwei"),
    # A run-together stale `wrong` that also occurs later as a word.
    ("Ich sage Hallo Welt und HalloWelt",
     [{"idx": 2, "idx_end": 3, "wrong": "HalloWelt", "correct": "Hallo, Welt"}],
     [_w(" Ich"), _w(" sage"), _w(" Hallo"), _w(" Welt"), _w(" und"),
      _w(" HalloWelt")],
     "Ich sage Hallo, Welt und HalloWelt"),
    # A stale `wrong` that is a prefix of the words at its idx ties with
    # them: the page strikes and replaces the whole "Hallo,", so does the
    # export, not just "Hallo" with the comma left behind.
    ("Ja Hallo, Welt",
     [{"idx": 1, "wrong": "Hallo", "correct": "Servus"}],
     [_w(" Ja"), {"word": " Hallo", "train_word": " Hallo,"}, _w(" Welt")],
     "Ja Servus Welt"),
    # Same for a range chip whose stored `wrong` covers only its first word.
    ("Hallo Welt",
     [{"idx": 0, "idx_end": 1, "wrong": "Hallo", "correct": "Hi"}],
     [_w(" Hallo"), _w(" Welt")],
     "Hi"),
    # And a stale `wrong` that is a suffix of them: the karaoke word is
    # "(die", the card strikes and replaces all of it.
    ("ja die (die Sache",
     [{"idx": 2, "wrong": "die", "correct": "DER"}],
     [_w(" ja"), _w(" die"), _w(" (die"), _w(" Sache")],
     "ja die DER Sache"),
    # A stale `wrong` that covers the words at its idx: a chip saved when the
    # first word of an unspaced sentence owned all of it. The corrected
    # sentence replaces the sentence, not only the first char ("我喜欢狗。喜欢猫。").
    ("我喜欢猫。",
     [{"idx": 0, "idx_end": 0, "wrong": "我喜欢猫。", "correct": "我喜欢狗。"}],
     [_w(" 我"), _w("喜欢"), _w("猫"), _w("。")],
     "我喜欢狗。"),
    # A dictated line break right before the chip's word: the token it sits
    # in starts after the "\n", not at the line before (Python's `$` matched
    # before a trailing "\n" and the server rewrote the last "的").
    ("我喜欢狗\n的猫的",
     [{"idx": 1, "wrong": "的", "correct": "地"}],
     [_w("我喜欢狗"), _w("\n的"), _w("猫"), _w("的")],
     "我喜欢狗\n地猫的"),
    # A chip on a word inside an unspaced Chinese run, and on the third of
    # three "的" in one run (also after a chip that lengthened the run, and
    # in the second of two sentences): the char offset inside the run picks
    # the occurrence, not the first one.
    ("我喜欢猫和狗",
     [{"idx": 2, "wrong": "猫", "correct": "鱼"}],
     [_w("我"), _w("喜欢"), _w("猫"), _w("和"), _w("狗")],
     "我喜欢鱼和狗"),
    ("我的猫的狗的",
     [{"idx": 5, "wrong": "的", "correct": "地"}],
     [_w("我"), _w("的"), _w("猫"), _w("的"), _w("狗"), _w("的")],
     "我的猫的狗地"),
    ("我的猫的狗的",
     [{"idx": 2, "wrong": "猫", "correct": "小猫"},
      {"idx": 5, "wrong": "的", "correct": "地"}],
     [_w("我"), _w("的"), _w("猫"), _w("的"), _w("狗"), _w("的")],
     "我的小猫的狗地"),
    ("你好。 我的猫的狗",
     [{"idx": 4, "wrong": "的", "correct": "地"}],
     [_w("你好。"), _w(" 我"), _w("的"), _w("猫"), _w("的"), _w("狗")],
     "你好。 我的猫地狗"),
    # A Latin word repeated inside an unspaced run: the earlier occurrence
    # starts the token, yet the char offset still picks the marked one.
    ("Python很好，Python很强",
     [{"idx": 3, "wrong": "Python", "correct": "Java"}],
     [_w("Python"), _w("很"), _w("好，"), _w("Python"), _w("很"), _w("强")],
     "Python很好，Java很强"),
    ("Hallo世界Hallo",
     [{"idx": 2, "wrong": "Hallo", "correct": "X"}],
     [_w("Hallo"), _w("世界"), _w("Hallo")],
     "Hallo世界X"),
    # A chip saved before the words kept their spaces ("HalloWelt"): the
    # page shows it on the words at its idx, so it applies there.
    ("Hallo Welt wie geht's",
     [{"idx": 0, "idx_end": 1, "wrong": "HalloWelt", "correct": "Hallo Leute"}],
     [_w(" Hallo"), _w(" Welt"), _w(" wie"), _w(" geht's")],
     "Hallo Leute wie geht's"),
]


@pytest.mark.parametrize("text, chips, expected", _CASES)
def test_apply_chips_rewrites_only_the_marked_word(text, chips, expected):
    assert capture_samples._apply_chips_to_text(text, chips) == expected


@pytest.mark.parametrize("text, chips, words, expected", _WORD_CASES)
def test_apply_chips_maps_idx_past_removed_words(text, chips, words, expected):
    assert capture_samples._apply_chips_to_text(text, chips, words) == expected


def test_apply_chips_in_long_unspaced_runs_stays_fast():
    # Each occurrence looked up its token start with a regex over all the
    # text before it, backtracking through every earlier run: a 7k-char
    # Chinese text took seconds per chip (a 30k-char one over a minute).
    text = " ".join(["我喜欢猫和狗" * 50] * 24)
    t0 = time.monotonic()
    out = capture_samples._apply_chips_to_text(
        text, [{"idx": 0, "wrong": "猫", "correct": "鱼"}])
    assert time.monotonic() - t0 < 2.0
    assert out == "我喜欢鱼和狗" + text[6:]


def test_default_transcript_aligns_chips_with_the_members_words():
    text, chips, words, expected = _WORD_CASES[0]
    member = {"text_for_training": text, "corrections": chips}
    assert capture_samples._build_default_transcript(
        [member], "space", words_for=lambda m: words) == expected


def test_default_transcript_applies_a_chip_with_a_stale_wrong():
    # Accepting a merge proposal posts only the member ids: the sample is
    # built from the STORED chips, and a "HalloWelt" chip the preview showed
    # as applied must not be dropped from the transcript.
    text, chips, words, expected = _WORD_CASES[-1]
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


def _run_js_twin(cases):
    """_applyChipsToText(text, chips, words) for each [text, chips, words]."""
    src = (pathlib.Path(REPO_ROOT) / "faster_whisper_backend" / "captures"
           / "templates" / "captures.html").read_text(encoding="utf-8")
    fn = _js_function(src, "_applyChipsToText")
    script = (fn + "\nprocess.stdout.write(JSON.stringify("
              + json.dumps(cases, ensure_ascii=False)
              + ".map(function(c) { return _applyChipsToText(c[0], c[1], c[2]); })));")
    p = subprocess.run([_NODE, "-e", script], capture_output=True,
                       text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


@pytest.mark.skipif(_NODE is None, reason="node unavailable to run the JS twin")
def test_js_twin_matches_server_chip_application():
    cases = ([[t, c, None] for t, c, _ in _CASES]
             + [[t, c, w] for t, c, w, _ in _WORD_CASES])
    assert _run_js_twin(cases) == ([e for _, _, e in _CASES]
                                   + [e for *_, e in _WORD_CASES])


# Words as production passes them: the output of _align_words_to_final, whose
# tokens carry no leading space. A rule that puts a line break or a ", "
# between two words, or a real space in the final text, starts a new token:
# a chip on a repeated word there must not rewrite the earlier occurrence.
# (raw words, per-word rule map, final, chip, expected)
_ALIGNED_CASES = [
    (["世界", "你好", "世界"], {}, "世界 你好 世界",
     {"idx": 2, "wrong": "世界", "correct": "地球"}, "世界 你好 地球"),
    (["我", "的", "猫", "换行", "我", "的", "狗"], {"换行": "\n"},
     "我的猫\n我的狗",
     {"idx": 5, "wrong": "的", "correct": "地"}, "我的猫\n我地狗"),
    (["我", "的", "猫", "逗号", "你", "的", "狗"], {"逗号": ","},
     "我的猫, 你的狗",
     {"idx": 5, "wrong": "的", "correct": "得"}, "我的猫, 你得狗"),
    (["ปลา", "ดี", " ปลา", "ใหญ่"], {}, "ปลาดี ปลาใหญ่",
     {"idx": 2, "wrong": "ปลา", "correct": "หมา"}, "ปลาดี หมาใหญ่"),
    (["写", "字", " Python", " 写", "代码"], {}, "写字 Python 写代码",
     {"idx": 3, "wrong": "写", "correct": "画"}, "写字 Python 画代码"),
    # Still right inside one run: the third "的", not the first.
    (["我", "的", "猫", "的", "狗", "的"], {}, "我的猫的狗的",
     {"idx": 5, "wrong": "的", "correct": "地"}, "我的猫的狗地"),
]


def _aligned_words(monkeypatch, raw, mapping, final):
    monkeypatch.setattr(
        pl_engine, "_postprocess_text",
        lambda text, **kw: mapping.get(text.strip(), text))
    return capture_samples._align_words_to_final(
        [{"word": w, "start": float(i), "end": float(i)}
         for i, w in enumerate(raw)], final)


@pytest.mark.parametrize("raw, mapping, final, chip, expected", _ALIGNED_CASES)
def test_apply_chips_places_a_chip_with_aligned_words(
        monkeypatch, raw, mapping, final, chip, expected):
    words = _aligned_words(monkeypatch, raw, mapping, final)
    assert capture_samples._apply_chips_to_text(final, [chip], words) == expected


@pytest.mark.skipif(_NODE is None, reason="node unavailable to run the JS twin")
def test_js_twin_places_a_chip_with_aligned_words(monkeypatch):
    cases = [[final, [chip], _aligned_words(monkeypatch, raw, mapping, final)]
             for raw, mapping, final, chip, _ in _ALIGNED_CASES]
    assert _run_js_twin(cases) == [e for *_, e in _ALIGNED_CASES]


def test_single_capture_ground_preview_uses_the_shared_helper(client):
    # The single-capture card's ground-truth preview used its own copy of the
    # first-occurrence substitution; it must go through the mirrored helper.
    t = client.get("/captures").text
    assert ("_applyChipsToText(state.finalText, state.corrections, state.words)"
            in t)


@pytest.mark.parametrize("finals, expected", [
    (["Hallo Welt.", "Wie geht es?", "gut"], "Hallo Welt. Wie geht es? gut"),
    (["Stopp!", "«Ja.»", "nein"], "Stopp! «Ja.» nein"),
    (["eins", "zwei…", "drei"], "eins. zwei… drei"),
    (["Er sagte „Ja.“", "weiter"], "Er sagte „Ja.“ weiter"),
    (["‚Stopp!‘", "gut"], "‚Stopp!‘ gut"),
    (["›Oui?‹", "non"], "›Oui?‹ non"),
    (["你好。", "好吗？", "好"], "你好。 好吗？ 好"),
    (["「好！」", "（是？）", "مرحبا؟", "नमस्ते।", "Wie⁉", "gut"],
     "「好！」 （是？） مرحبا؟ नमस्ते। Wie⁉ gut"),
    # Urdu ۔, Armenian ։, Amharic ።, Burmese ။, Khmer ។, double danda ॥.
    (["یہ ٹھیک ہے۔", "Լավ է։", "ደህና ነው።", "ကောင်းတယ်။", "ល្អ។", "ॐ॥", "gut"],
     "یہ ٹھیک ہے۔ Լավ է։ ደህና ነው። ကောင်းတယ်။ ល្អ។ ॐ॥ gut"),
    # French typography: a space before the closing ».
    (["Il a dit : « Oui. »", "non"], "Il a dit : « Oui. » non"),
])
def test_period_space_join_adds_no_period_after_sentence_punctuation(
        finals, expected):
    """period_space put a literal ". " between members even when a member
    already ended its sentence: "Hallo Welt.. Wie geht es?. gut"."""
    members = [{"final": f} for f in finals]
    assert capture_samples._build_default_transcript(
        members, "period_space") == expected
