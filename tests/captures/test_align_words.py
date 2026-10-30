"""Regression tests for captures/samples.py _align_words_to_final.

The merge / proposal FINAL RESULT is rebuilt word-by-word from this alignment
(via _renderGroundSpans), so the alignment must assign EVERY post-processed
token to some raw word — otherwise tokens that a rule *inserts* (e.g. the raw
word "Nurtax" expanding to "nur tags") silently disappear from the merged final
while still being present in the exported text. See the bug write-up: a 1->many
word-expanding rule dropped the expansion from the group preview.

These tests pin the invariant `" ".join(non-removed word) == final` across the
word-count-changing shapes: 1->1, 1->many, many->1, 1->0.

The alignment computes each raw word's post-processed form via
pl_engine._postprocess_text(); we monkeypatch it to a deterministic per-word map so
the cases are hermetic (no real pipeline / config needed).
"""

from faster_whisper_backend.captures import routes as cr
from faster_whisper_backend.captures import samples as capture_samples
from faster_whisper_backend.pipeline import engine as pl_engine


def _words(*pairs):
    """Build a words-style list with synthetic but monotonic timings."""
    out = []
    for i, w in enumerate(pairs):
        out.append({"word": w, "start": float(i), "end": float(i) + 0.5})
    return out


def _join(out):
    """Reconstruct the displayed final from non-removed entries."""
    return " ".join(w["word"] for w in out if not w.get("removed"))


def _install_word_map(monkeypatch, mapping):
    """Fake pl_engine._postprocess_text: per-word lookup, identity by default."""
    def fake(text, model_name=None, ident=None, language=None, **kw):
        return mapping.get(text.strip(), text)
    monkeypatch.setattr(pl_engine, "_postprocess_text", fake)


# ---------------------------------------------------------------------------
# 1 -> many expansion (the reported bug: "Nurtax" -> "nur tags")
# ---------------------------------------------------------------------------

def test_one_to_many_expansion_keeps_all_tokens(monkeypatch):
    _install_word_map(monkeypatch, {"Nurtax": "nur tags"})
    words = _words("Nurtax", "137", "Schrägstrich", "94")
    final = "nur tags 137 Schrägstrich 94"
    out = capture_samples._align_words_to_final(words, final)

    assert len(out) == len(words)              # one entry per raw word
    assert _join(out) == final                 # no token dropped
    # The expanded word carries the joined run and flags the raw form.
    assert out[0]["word"] == "nur tags"
    assert not out[0].get("removed")
    assert out[0].get("raw_word") == "Nurtax"
    # Untouched words stay put.
    assert out[1]["word"] == "137"
    assert out[3]["word"] == "94"


def test_one_to_three_expansion(monkeypatch):
    # Compound-splitter style: one raw word becomes three tokens.
    _install_word_map(monkeypatch, {"Eisenbindestrichinfusion":
                                    "Eisen Bindestrich infusion"})
    words = _words("Eisenbindestrichinfusion", "heute")
    final = "Eisen Bindestrich infusion heute"
    out = capture_samples._align_words_to_final(words, final)

    assert len(out) == 2
    assert _join(out) == final
    assert out[0]["word"] == "Eisen Bindestrich infusion"
    assert not out[0].get("removed")
    assert out[1]["word"] == "heute"


def test_expansion_at_end(monkeypatch):
    # Trailing raw word expands — exercises the after-last-anchor segment.
    _install_word_map(monkeypatch, {"Nurtax": "nur tags"})
    words = _words("Hallo", "Nurtax")
    final = "Hallo nur tags"
    out = capture_samples._align_words_to_final(words, final)
    assert len(out) == 2
    assert _join(out) == final
    assert out[1]["word"] == "nur tags"
    assert not out[1].get("removed")


# ---------------------------------------------------------------------------
# many -> 1 contraction
# ---------------------------------------------------------------------------

def test_many_to_one_contraction(monkeypatch):
    # Two raw words merge into one final token: the survivor attaches to the
    # first raw word, the second is removed — no token lost or duplicated.
    _install_word_map(monkeypatch, {})  # identity; neither matches "Bindestrich"
    words = _words("Bind", "strich")
    final = "Bindestrich"
    out = capture_samples._align_words_to_final(words, final)

    assert len(out) == 2
    assert _join(out) == final
    assert out[0]["word"] == "Bindestrich"
    assert out[1].get("removed") is True


def test_symbol_substitution_keeps_token(monkeypatch):
    # Dictation-map style "Schrägstrich" -> "/": a 1->1 symbol swap whose key
    # normalises to empty must still attach the symbol, not drop it.
    _install_word_map(monkeypatch, {"Schrägstrich": "/"})
    words = _words("137", "Schrägstrich", "94")
    final = "137 / 94"
    out = capture_samples._align_words_to_final(words, final)

    assert len(out) == 3
    assert _join(out) == final
    assert out[1]["word"] == "/"
    assert not out[1].get("removed")


# ---------------------------------------------------------------------------
# 1 -> 0 deletion
# ---------------------------------------------------------------------------

def test_deletion_marks_removed(monkeypatch):
    _install_word_map(monkeypatch, {"äh": ""})
    words = _words("äh", "hallo", "welt")
    final = "hallo welt"
    out = capture_samples._align_words_to_final(words, final)

    assert len(out) == 3
    assert _join(out) == final
    assert out[0].get("removed") is True
    assert out[1]["word"] == "hallo"
    assert out[2]["word"] == "welt"


def test_deletion_keeps_a_words_final_vowel_sign(monkeypatch):
    # The edge-punctuation strip cut the vowel sign off "का" and "की" (both
    # "क"), so the kept "की" was given to the deleted "का" and the real
    # "की" was struck as removed.
    _install_word_map(monkeypatch, {})
    words = _words("राम", "का", "की", "घर")
    final = "राम की घर"
    out = capture_samples._align_words_to_final(words, final)

    assert _join(out) == final
    assert [bool(w.get("removed")) for w in out] == [False, True, False, False]
    assert out[2]["word"] == "की"
    assert "raw_word" not in out[2]


# ---------------------------------------------------------------------------
# plain 1 -> 1 (no count change)
# ---------------------------------------------------------------------------

def test_plain_substitution(monkeypatch):
    _install_word_map(monkeypatch, {"weiss": "weiß"})
    words = _words("das", "weiss", "ich")
    final = "das weiß ich"
    out = capture_samples._align_words_to_final(words, final)

    assert len(out) == 3
    assert _join(out) == final
    assert out[1]["word"] == "weiß"
    assert out[1].get("raw_word") == "weiss"
    assert not out[1].get("removed")


def test_identity_no_flags(monkeypatch):
    _install_word_map(monkeypatch, {})
    words = _words("eins", "zwei", "drei")
    final = "eins zwei drei"
    out = capture_samples._align_words_to_final(words, final)

    assert _join(out) == final
    for w in out:
        assert not w.get("removed")
        assert "raw_word" not in w  # display == raw → no diff marker


# ---------------------------------------------------------------------------
# scripts written without spaces (zh / ja / th / lo / my / km)
# ---------------------------------------------------------------------------

def test_unspaced_chinese_gives_each_word_its_own_text(monkeypatch):
    # The whole sentence is one final token: it went to the first word and
    # every later word was struck as removed by a rule that never fired.
    _install_word_map(monkeypatch, {})
    words = _words("我们", "去", "北京", "。")
    out = capture_samples._align_words_to_final(words, "我们去北京。")

    assert [w["word"] for w in out] == ["我们", "去", "北京", "。"]
    assert not any(w.get("removed") or "raw_word" in w for w in out)


def test_unspaced_thai_gives_each_word_its_own_text(monkeypatch):
    _install_word_map(monkeypatch, {})
    out = capture_samples._align_words_to_final(
        _words("สวัสดี", "ครับ"), "สวัสดีครับ")

    assert [w["word"] for w in out] == ["สวัสดี", "ครับ"]
    assert not any(w.get("removed") for w in out)


def test_unspaced_rule_change_marks_only_that_word(monkeypatch):
    _install_word_map(monkeypatch, {"北京": "上海", "去": ""})
    words = _words("我们", "去", "北京", "。")
    out = capture_samples._align_words_to_final(words, "我们上海。")

    assert "".join(w["word"] for w in out if not w.get("removed")) == "我们上海。"
    assert [bool(w.get("removed")) for w in out] == [False, True, False, False]
    assert out[2]["word"] == "上海"
    assert out[2]["raw_word"] == "北京"
    assert "raw_word" not in out[0] and "raw_word" not in out[3]


def test_unspaced_sentences_split_per_token(monkeypatch):
    # Two sentences joined by a space (faster-whisper glues the "。" onto
    # the word before it): each word stays inside its own token.
    _install_word_map(monkeypatch, {})
    words = _words("你好。", "我", "很", "好。")
    out = capture_samples._align_words_to_final(words, "你好。 我很好。")

    assert [w["word"] for w in out] == ["你好。", "我", "很", "好。"]
    assert not any(w.get("removed") for w in out)


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

def test_output_length_always_matches_raw_count(monkeypatch):
    _install_word_map(monkeypatch, {"Nurtax": "nur tags", "äh": ""})
    words = _words("äh", "Nurtax", "137")
    final = "nur tags 137"
    out = capture_samples._align_words_to_final(words, final)
    assert len(out) == len(words)   # chip word-indices depend on this
    assert _join(out) == final


def test_empty_words_returns_empty(monkeypatch):
    _install_word_map(monkeypatch, {})
    assert capture_samples._align_words_to_final([], "anything") == []


# ---------------------------------------------------------------------------
# _align_member_words: dual alignment (runtime `final` + EXCLUDE-aware
# `text_for_training`). The editable Corrections strip displays the training
# token so it matches the Final result + export; see the "134/92" write-up.
# ---------------------------------------------------------------------------

def test_align_member_words_attaches_training_tokens(monkeypatch):
    # Identity per-word map: the "134 Schrägstrich 92" -> "134/92" merge is a
    # cross-word rule, modeled here purely by the differing final/training
    # STRINGS (CAPTURES_PIPELINE_RULES_EXCLUDE drops it from training).
    _install_word_map(monkeypatch, {})
    m = {
        "words": _words(" Tag", " und", " Nacht", " 134", " Schrägstrich", " 92"),
        "final": "Tag und Nacht 134/92",
        "text_for_training": "Tag und Nacht 134 Schrägstrich 92",
        "model": None,
    }
    out = cr._align_member_words(m)
    # Runtime `word`: the de-dictation-map form; "Schrägstrich"/"92" merged away.
    assert out[3]["word"] == " 134/92"
    assert out[4].get("removed") is True
    assert out[5].get("removed") is True
    # Training tokens are index-parallel, carry the raw leading space, and are
    # NOT removed (the merge rule is excluded for captures) — so the strip
    # shows them as normal, clickable, correctable words.
    assert [w.get("train_word") for w in out] == [
        " Tag", " und", " Nacht", " 134", " Schrägstrich", " 92"]
    assert not out[4].get("train_removed")
    assert not out[5].get("train_removed")
    # A multi-word chip `wrong` (join of train tokens, leading space stripped)
    # reconstructs the training/export text verbatim, so it actually applies.
    rng = "".join(w["train_word"] for w in out[3:6]).lstrip()
    assert rng == "134 Schrägstrich 92"
    assert rng in m["text_for_training"]


def test_align_member_words_no_train_when_equal(monkeypatch):
    # When training == final there is no excluded-rule difference, so no
    # train_word is attached and the strip falls back to the runtime token.
    _install_word_map(monkeypatch, {})
    m = {
        "words": _words(" Hallo", " Welt"),
        "final": "Hallo Welt",
        "text_for_training": "Hallo Welt",
        "model": None,
    }
    out = cr._align_member_words(m)
    assert all("train_word" not in w for w in out)


def test_align_member_words_keeps_the_lead_on_word_when_equal(monkeypatch):
    # With no train_word the page builds a range chip's `wrong` from `word`
    # with join(''). Matched tokens came back without the raw lead, so a
    # chip over "Hallo Welt" got wrong="HalloWelt" and the export dropped it.
    _install_word_map(monkeypatch, {})
    m = {
        "words": _words(" Hallo", " Welt", " heute"),
        "final": "Hallo Welt heute",
        "text_for_training": "Hallo Welt heute",
        "model": None,
    }
    out = cr._align_member_words(m)
    assert [w["word"] for w in out] == [" Hallo", " Welt", " heute"]
    wrong = "".join(w["word"] for w in out[0:2]).lstrip()
    assert capture_samples._apply_chips_to_text(
        m["text_for_training"],
        [{"wrong": wrong, "correct": "Servus", "idx": 0, "idx_end": 1}], out,
    ) == "Servus heute"


def test_apply_chips_consistency_with_training_tokens():
    # The server export substitutes the chip `wrong` verbatim in the training
    # text. A chip built from the TRAINING token applies; one built from the
    # runtime-only "134/92" token is silently dropped (it isn't in the text).
    training = "Tag und Nacht 134 Schrägstrich 92"
    assert capture_samples._apply_chips_to_text(
        training, [{"wrong": "134", "correct": "134/92", "idx": 3}],
    ) == "Tag und Nacht 134/92 Schrägstrich 92"
    assert capture_samples._apply_chips_to_text(
        training, [{"wrong": "134/92", "correct": "X", "idx": 3}],
    ) == training
    # A range correction over the whole "134 Schrägstrich 92" span applies.
    assert capture_samples._apply_chips_to_text(
        training,
        [{"wrong": "134 Schrägstrich 92", "correct": "134/92",
          "idx": 3, "idx_end": 5}],
    ) == "Tag und Nacht 134/92"
