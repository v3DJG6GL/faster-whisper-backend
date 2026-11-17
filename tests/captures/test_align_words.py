"""Regression tests for captures/samples.py _align_words_to_final.

The merge / proposal FINAL RESULT is rebuilt word-by-word from this alignment
(via _renderGroundSpans), so the alignment must assign EVERY post-processed
token to some raw word — otherwise tokens that a rule *inserts* (e.g. the raw
word "Nurtax" expanding to "nur tags") silently disappear from the merged final
while still being present in the exported text.

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


def _align(monkeypatch, mapping, final, *raw):
    _install_word_map(monkeypatch, mapping)
    return capture_samples._align_words_to_final(_words(*raw), final)


def _member(monkeypatch, final, training, *raw):
    """A member row for _align_member_words, under the identity word map."""
    _install_word_map(monkeypatch, {})
    return {"words": _words(*raw), "final": final,
            "text_for_training": training, "model": None}


def test_one_to_many_expansion_keeps_all_tokens(monkeypatch):
    final = "nur tags 137 Schrägstrich 94"
    out = _align(monkeypatch, {"Nurtax": "nur tags"}, final, "Nurtax", "137", "Schrägstrich", "94")

    assert len(out) == 4                       # one entry per raw word
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
    final = "Eisen Bindestrich infusion heute"
    out = _align(monkeypatch, {"Eisenbindestrichinfusion": "Eisen Bindestrich infusion"},
                 final, "Eisenbindestrichinfusion", "heute")

    assert len(out) == 2
    assert _join(out) == final
    assert out[0]["word"] == "Eisen Bindestrich infusion"
    assert not out[0].get("removed")
    assert out[1]["word"] == "heute"


def test_expansion_at_end(monkeypatch):
    # Trailing raw word expands — exercises the after-last-anchor segment.
    final = "Hallo nur tags"
    out = _align(monkeypatch, {"Nurtax": "nur tags"}, final, "Hallo", "Nurtax")
    assert len(out) == 2
    assert _join(out) == final
    assert out[1]["word"] == "nur tags"
    assert not out[1].get("removed")


def test_many_to_one_contraction(monkeypatch):
    # Two raw words merge into one final token: the survivor attaches to the
    # first raw word, the second is removed — no token lost or duplicated.
    final = "Bindestrich"
    out = _align(monkeypatch, {}, final, "Bind", "strich")  # identity; neither matches final

    assert len(out) == 2
    assert _join(out) == final
    assert out[0]["word"] == "Bindestrich"
    assert out[1].get("removed") is True


def test_symbol_substitution_keeps_token(monkeypatch):
    # Dictation-map style "Schrägstrich" -> "/": a 1->1 symbol swap whose key
    # normalises to empty must still attach the symbol, not drop it.
    final = "137 / 94"
    out = _align(monkeypatch, {"Schrägstrich": "/"}, final, "137", "Schrägstrich", "94")

    assert len(out) == 3
    assert _join(out) == final
    assert out[1]["word"] == "/"
    assert not out[1].get("removed")


def test_deletion_marks_removed(monkeypatch):
    final = "hallo welt"
    out = _align(monkeypatch, {"äh": ""}, final, "äh", "hallo", "welt")

    assert len(out) == 3
    assert _join(out) == final
    assert out[0].get("removed") is True
    assert out[1]["word"] == "hallo"
    assert out[2]["word"] == "welt"


def test_deletion_keeps_a_words_final_vowel_sign(monkeypatch):
    # The edge-punctuation strip cut the vowel sign off "का" and "की" (both
    # "क"), so the kept "की" was given to the deleted "का" and the real
    # "की" was struck as removed.
    final = "राम की घर"
    out = _align(monkeypatch, {}, final, "राम", "का", "की", "घर")

    assert _join(out) == final
    assert [bool(w.get("removed")) for w in out] == [False, True, False, False]
    assert out[2]["word"] == "की"
    assert "raw_word" not in out[2]


def test_plain_substitution(monkeypatch):
    final = "das weiß ich"
    out = _align(monkeypatch, {"weiss": "weiß"}, final, "das", "weiss", "ich")

    assert len(out) == 3
    assert _join(out) == final
    assert out[1]["word"] == "weiß"
    assert out[1].get("raw_word") == "weiss"
    assert not out[1].get("removed")


def test_identity_no_flags(monkeypatch):
    final = "eins zwei drei"
    out = _align(monkeypatch, {}, final, "eins", "zwei", "drei")

    assert _join(out) == final
    for w in out:
        assert not w.get("removed")
        assert "raw_word" not in w  # display == raw → no diff marker


def test_unspaced_chinese_gives_each_word_its_own_text(monkeypatch):
    # The whole sentence is one final token: it went to the first word and
    # every later word was struck as removed by a rule that never fired.
    out = _align(monkeypatch, {}, "我们去北京。", "我们", "去", "北京", "。")

    assert [w["word"] for w in out] == ["我们", "去", "北京", "。"]
    assert not any(w.get("removed") or "raw_word" in w for w in out)


def test_unspaced_thai_gives_each_word_its_own_text(monkeypatch):
    out = _align(monkeypatch, {}, "สวัสดีครับ", "สวัสดี", "ครับ")

    assert [w["word"] for w in out] == ["สวัสดี", "ครับ"]
    assert not any(w.get("removed") for w in out)


def test_unspaced_rule_change_marks_only_that_word(monkeypatch):
    out = _align(monkeypatch, {"北京": "上海", "去": ""}, "我们上海。", "我们", "去", "北京", "。")

    assert "".join(w["word"] for w in out if not w.get("removed")) == "我们上海。"
    assert [bool(w.get("removed")) for w in out] == [False, True, False, False]
    assert out[2]["word"] == "上海"
    assert out[2]["raw_word"] == "北京"
    assert "raw_word" not in out[0] and "raw_word" not in out[3]


def test_unspaced_substitution_stays_on_the_substituted_word(monkeypatch):
    # A rule over the whole sentence turned 他 into 她 (no per-word form
    # explains it): the unmatched 她 went to the first MATCHED word, so 他
    # was struck as removed and 我 showed "我她".
    out = _align(monkeypatch, {}, "我她好", "我", "他", "好")

    assert [w["word"] for w in out] == ["我", "她", "好"]
    assert not any(w.get("removed") for w in out)
    assert out[1]["raw_word"] == "他"
    assert "raw_word" not in out[0] and "raw_word" not in out[2]


def test_split_unspaced_spreads_replaced_and_inserted_chars():
    split = capture_samples._split_unspaced
    # A substituted first char stays on word 0 instead of the next word.
    assert split(["他", "好"], ["她好"]) == [["她"], ["好"]]
    assert split(["我", "喜欢", "他", "的", "猫"], ["我喜欢她的猫"]) == [
        ["我"], ["喜欢"], ["她"], ["的"], ["猫"]]
    # A substitution inside the run stays on its own word (去 → 到), and a
    # trailing insertion (市) joins the word before it.
    assert split(["我们", "去", "北京"], ["我们到北京市"]) == [
        ["我们"], ["到"], ["北京市"]]
    # An insertion before the first match joins the first word, one after a
    # word joins that word.
    assert split(["我", "喜欢", "猫"], ["嗯我喜欢猫。"]) == [
        ["嗯我"], ["喜欢"], ["猫。"]]
    # No char matches at all: the caller's token-count split takes over.
    assert split(["他"], ["她"]) is None


def test_unspaced_sentences_split_per_token(monkeypatch):
    # Two sentences joined by a space (faster-whisper glues the "。" onto
    # the word before it): each word stays inside its own token.
    out = _align(monkeypatch, {}, "你好。 我很好。", "你好。", "我", "很", "好。")

    assert [w["word"] for w in out] == ["你好。", "我", "很", "好。"]
    assert not any(w.get("removed") for w in out)


def test_repeated_word_anchors_the_spaced_raw_word(monkeypatch):
    # Mixed zh / en dictation where Whisper glues the second "Python" onto
    # "，": the spaced " Python" is the whole first token, but the LCS tie
    # skipped it for the later "Python", which took the first token while
    # " Python" 写 代码 ， were struck as removed and 很 owned
    # "写代码，Python 很". A chip on idx 6 then rewrote the FIRST Python.
    final = "我用 Python 写代码，Python 很好"
    out = _align(monkeypatch, {}, final,
                 "我", "用", " Python", " 写", "代码", "，", "Python", " 很", "好")

    assert [w["word"].strip() for w in out] == [
        "我", "用", "Python", "写", "代码", "，", "Python", "很", "好"]
    assert not any(w.get("removed") for w in out)
    chip = [{"idx": 6, "wrong": "Python", "correct": "Java"}]
    assert capture_samples._apply_chips_to_text(final, chip, out) == (
        "我用 Python 写代码，Java 很好")


def test_repeated_unspaced_char_anchors_the_spaced_raw_word(monkeypatch):
    # The opposite tie: the later " 好" starts the second token, so the
    # unspaced 好 before it stays inside "你好".
    out = _align(monkeypatch, {}, "你好 好", "你", "好", " 好")

    assert [w["word"] for w in out] == ["你", "好", "好"]
    assert not any(w.get("removed") for w in out)


def test_spaced_raw_word_glued_into_a_token_does_not_take_a_later_one(monkeypatch):
    # The spaced first word sits inside the glued token "好的" / "Python很好";
    # a later repeat after a dictated line break is the standalone token. The
    # lead tie-break gave the token to the earlier word, so every word in
    # between was struck as removed by a rule.
    line_break = {"换行": "\n"}
    out = _align(monkeypatch, line_break, "好的\n好 我知道",
                 " 好", "的", "换行", "好", " 我", "知", "道")
    assert [w["word"] for w in out] == ["好", "的", "换行", "好", "我", "知", "道"]
    assert [w.get("removed", False) for w in out] == [
        False, False, True, False, False, False, False]

    out = _align(monkeypatch, line_break, "Python很好\nPython 很强",
                 " Python", "很", "好", "换行", "Python", " 很", "强")
    assert [w["word"] for w in out] == [
        "Python", "很", "好", "换行", "Python", "很", "强"]
    assert [w.get("removed", False) for w in out] == [
        False, False, False, True, False, False, False]


def test_output_length_always_matches_raw_count(monkeypatch):
    final = "nur tags 137"
    out = _align(monkeypatch, {"Nurtax": "nur tags", "äh": ""}, final, "äh", "Nurtax", "137")
    assert len(out) == 3   # chip word-indices depend on this
    assert _join(out) == final


def test_empty_words_returns_empty(monkeypatch):
    assert _align(monkeypatch, {}, "anything") == []


# _align_member_words aligns to the runtime `final` and to the EXCLUDE-aware
# `text_for_training`: the Corrections strip shows the training token so it
# matches the Final result and the export.

def test_align_member_words_attaches_training_tokens(monkeypatch):
    # Identity per-word map: the "134 Schrägstrich 92" -> "134/92" merge is a
    # cross-word rule, modeled here purely by the differing final/training
    # STRINGS (CAPTURES_PIPELINE_RULES_EXCLUDE drops it from training).
    m = _member(monkeypatch, "Tag und Nacht 134/92", "Tag und Nacht 134 Schrägstrich 92",
                " Tag", " und", " Nacht", " 134", " Schrägstrich", " 92")
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
    m = _member(monkeypatch, "Hallo Welt", "Hallo Welt", " Hallo", " Welt")
    out = cr._align_member_words(m)
    assert all("train_word" not in w for w in out)


def test_align_member_words_blanked_training_marks_every_word_removed(monkeypatch):
    # A training-only removal rule blanked the member ("" is the verdict, not
    # a missing value) while runtime `final` kept the text: every word is
    # train_removed, as the merged transcript and the export leave them out.
    m = _member(monkeypatch, "Untertitel im Auftrag des ZDF", "",
                " Untertitel", " im", " Auftrag", " des", " ZDF")
    out = cr._align_member_words(m)
    assert all("train_word" in w and w.get("train_removed") is True for w in out)
    # A missing training text still falls back to `final`.
    m.pop("text_for_training")
    assert all("train_word" not in w for w in cr._align_member_words(m))


def test_align_member_words_keeps_the_lead_on_word_when_equal(monkeypatch):
    # With no train_word the page builds a range chip's `wrong` from `word`
    # with join(''). Matched tokens came back without the raw lead, so a
    # chip over "Hallo Welt" got wrong="HalloWelt" and the export dropped it.
    m = _member(monkeypatch, "Hallo Welt heute", "Hallo Welt heute",
                " Hallo", " Welt", " heute")
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


# A card a delegated editor may save (regex_guard accepts it): eleven
# doublings, padded with identity entries. One short word grows 2048-fold,
# under the floor of the output bound of its own walk.
_DOUBLING_CARD = {"name": "dbl", "label": "Dbl", "type": "regex-list", "enabled": True,
                  "entries": [{"pattern": "(.)", "replacement": "\\1\\1"}] * 11
                  + [{"pattern": "(.)", "replacement": "\\1"}] * 189}


def _install_card(app_module, *cards):
    from tests.settings._rules_helpers import terminal_rule

    app_module.cfg.PIPELINE_RULES = [*cards, terminal_rule()]
    pl_engine.rebuild_caches()


def test_unspaced_split_is_bounded_on_an_expanding_rule(app_module):
    # Each word expands alone far past what the whole text may grow to; the
    # character split of the one final token then compares MBs of chars.
    import time

    _install_card(app_module, _DOUBLING_CARD)
    raw = ["的"] * 20
    final = pl_engine._postprocess_text("".join(raw))
    t0 = time.monotonic()
    out = capture_samples._align_words_to_final(_words(*raw), final)
    assert time.monotonic() - t0 < 2
    assert len(out) == len(raw)
    assert _join(out).replace(" ", "") == final


def test_split_unspaced_gives_up_past_its_cell_budget(monkeypatch):
    split = capture_samples._split_unspaced
    monkeypatch.setattr(capture_samples, "_SPLIT_UNSPACED_MAX_CELLS", 3)
    assert split(["他", "好"], ["他好"]) is None
    monkeypatch.setattr(capture_samples, "_SPLIT_UNSPACED_MAX_CELLS", 4)
    assert split(["他", "好"], ["他好"]) == [["他"], ["好"]]


def test_split_unspaced_gives_up_past_its_side_length(monkeypatch):
    # The cell product alone admits a very long side against a one-char one.
    split = capture_samples._split_unspaced
    monkeypatch.setattr(capture_samples, "_SPLIT_UNSPACED_MAX_CHARS", 1)
    assert split(["他", "好"], ["他好"]) is None
    assert split(["他"], ["他", "好"]) is None
    monkeypatch.setattr(capture_samples, "_SPLIT_UNSPACED_MAX_CHARS", 2)
    assert split(["他", "好"], ["他好"]) == [["他"], ["好"]]


def test_unspaced_gap_past_the_side_length_takes_the_token_count_split(monkeypatch):
    # A word a rule expands, repeated in one gap: the gap's cells fit the
    # budget, its raw side does not, so its texts are never joined.
    _install_word_map(monkeypatch, {"中": "中" * 100})
    monkeypatch.setattr(capture_samples, "_SPLIT_UNSPACED_MAX_CHARS", 250)
    calls: list[int] = []
    real = capture_samples._split_unspaced

    def spy(texts, toks, *a):
        calls.append(sum(map(len, texts)))
        return real(texts, toks, *a)
    monkeypatch.setattr(capture_samples, "_split_unspaced", spy)
    raw = [" 中"] * 3
    out = capture_samples._align_words_to_final(_words(*raw), "中")
    assert calls == []
    assert len(out) == len(raw)
    assert _join(out) == "中"
    # Two of them fit, and split as before.
    out = capture_samples._align_words_to_final(_words(*raw[:2]), "中")
    assert calls == [200]
    assert _join(out) == "中"


def test_word_walks_share_one_output_budget(app_module, monkeypatch):
    # One walk per distinct raw word, each under its own floor, must not build
    # more chars together than one walk over all of them may.
    _install_card(app_module, _DOUBLING_CARD)
    letters = "abcdefghijklmnopqrstuvwxyz"
    raw = [" w" + letters[i // 26] + letters[i % 26] + "xy" for i in range(100)]
    assert len(set(raw)) == 100
    real = pl_engine._postprocess_text
    sizes: list[int] = []

    def spy(text, *a, **kw):
        out = real(text, *a, **kw)
        sizes.append(len(out))
        return out

    monkeypatch.setattr(pl_engine, "_postprocess_text", spy)
    out = capture_samples._align_words_to_final(_words(*raw), " ".join(raw))
    assert len(out) == len(raw)
    assert sum(sizes) <= pl_engine.output_bound(sum(len(w) for w in raw))


def test_alignment_keys_are_computed_once_per_distinct_word(monkeypatch):
    # The walks are bounded per distinct word; the key work after them must be
    # too, or a word a rule expands costs its expansion once per occurrence.
    def fake(text, *a, budget=None, **kw):
        out = "x" + "." * 16000
        if budget is not None:
            budget.left -= len(out)
        return out
    monkeypatch.setattr(pl_engine, "_postprocess_text", fake)
    # A post form's key is cut from its _word_span; _align_key keys the rest.
    long_calls: list[int] = []

    def counting(real):
        def wrapped(s):
            if len(s or "") > 1000:
                long_calls.append(len(s))
            return real(s)
        return wrapped
    for name in ("_align_key", "_word_span"):
        monkeypatch.setattr(capture_samples, name, counting(getattr(capture_samples, name)))
    out = capture_samples._align_words_to_final(_words(*[" ja"] * 200), "ja")
    assert len(out) == 200
    assert len(long_calls) <= 3


def test_glued_runs_do_not_rebuild_an_expanded_first_word(monkeypatch):
    # One spaced word a rule expands, each occurrence followed by a different
    # glued word: every run is distinct, and its key must not walk or copy
    # the expansion again.
    import time
    import tracemalloc

    def fake(text, *a, budget=None, **kw):
        if text != " ja":
            return text
        out = "." * max(0, budget.left - 2) + "x"
        budget.left -= len(out)
        return out
    monkeypatch.setattr(pl_engine, "_postprocess_text", fake)
    raw = [w for i in range(1000) for w in (" ja", chr(0x4e00 + i))]
    tracemalloc.start()
    try:
        t0 = time.process_time()
        out = capture_samples._align_words_to_final(_words(*raw), "y")
        took = time.process_time() - t0
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(out) == len(raw)
    assert took < 1.5
    assert peak < 64 * 1024 * 1024


def test_unspaced_split_cells_are_one_budget_for_all_gaps(monkeypatch):
    # Many small unspaced gaps around a word a rule expands: past the call's
    # cell budget the gaps take the token-count split instead.
    _install_word_map(monkeypatch, {"中": "中" * 100})
    monkeypatch.setattr(capture_samples, "_SPLIT_UNSPACED_MAX_CELLS", 250)
    calls: list[int] = []
    real = capture_samples._split_unspaced

    def spy(texts, toks, *a):
        calls.append(1)
        return real(texts, toks, *a)
    monkeypatch.setattr(capture_samples, "_split_unspaced", spy)
    raw = [w for i in range(10) for w in (" 中", f" a{i}")]
    final = " ".join(t for i in range(10) for t in ("中", f"a{i}"))
    out = capture_samples._align_words_to_final(_words(*raw), final)
    assert len(calls) == 2
    assert _join(out) == final


def test_split_unspaced_gives_up_past_its_work_budget(monkeypatch):
    # A repetitive text that differs at every other char: the matcher searches
    # the rest again per matching block, far past the cells the two sides
    # span (160 k here, 12 M steps).
    import time

    _install_word_map(monkeypatch, {})
    monkeypatch.setattr(capture_samples, "_SPLIT_UNSPACED_MAX_WORK", 1_000_000)
    split = capture_samples._split_unspaced
    assert split(["哈"] * 400, ["哈呵" * 200]) is None
    # The same cells on an identical text are one search.
    assert split(["哈"] * 400, ["哈" * 400]) == [["哈"]] * 400
    raw = ["哈"] * 800
    final = "哈呵" * 400
    t0 = time.monotonic()
    out = capture_samples._align_words_to_final(_words(*raw), final)
    assert time.monotonic() - t0 < 2
    assert len(out) == len(raw)
    assert _join(out) == final


def test_unspaced_split_work_is_one_budget_for_all_gaps(monkeypatch):
    # Many unspaced gaps that each fit the work budget alone: together they
    # spend it once, and the gaps past it take the token-count split.
    _install_word_map(monkeypatch, {})
    texts, tok = ["哈"] * 40, "哈呵" * 20
    left = [10 ** 9]
    assert capture_samples._split_unspaced(texts, [tok], left) is not None
    one_gap = 10 ** 9 - left[0]
    assert one_gap > len(texts) * len(tok)
    monkeypatch.setattr(capture_samples, "_SPLIT_UNSPACED_MAX_WORK", 2 * one_gap + one_gap // 2)
    results: list[bool] = []
    real = capture_samples._split_unspaced

    def spy(texts, toks, *a):
        split = real(texts, toks, *a)
        results.append(split is not None)
        return split
    monkeypatch.setattr(capture_samples, "_split_unspaced", spy)
    raw = [w for i in range(6) for w in (f" a{i}", " 哈", *texts[1:])]
    final = " ".join(t for i in range(6) for t in (f"a{i}", tok))
    out = capture_samples._align_words_to_final(_words(*raw), final)
    assert results == [True, True, False, False, False, False]
    assert len(out) == len(raw)
    # A split gap shows one piece per word, a gap past the budget its token
    # on the first word.
    assert _join(out).replace(" ", "") == final.replace(" ", "")
    assert [w["word"] for w in out[1:3]] == ["哈呵", "哈呵"]
    assert out[-len(texts)]["word"] == tok
    assert all(w.get("removed") for w in out[-len(texts) + 1:])


def test_word_walks_keep_every_word_under_a_real_rule(app_module):
    # A real rule set is far inside the shared budget: every word still gets
    # its own post-processed form.
    _install_card(app_module, {"name": "sub", "label": "Sub", "type": "regex-list",
                               "enabled": True,
                               "entries": [{"pattern": "Komma", "replacement": ","}]})
    raw = [" Hallo", " Komma", " Welt"] * 200
    final = pl_engine._postprocess_text("".join(raw))
    out = capture_samples._align_words_to_final(_words(*raw), final)
    assert _join(out) == final
    assert [w["word"] for w in out] == ["Hallo", ",", "Welt"] * 200
    assert all(w["raw_word"] == " Komma" for w in out[1::3])


# Collapses every run of one char back to one: after _DOUBLING_CARD each word
# walk builds thousands of chars and returns about the word, so a budget
# charged only each walk's result never runs down.
_COLLAPSE_CARD = {"name": "col", "label": "Col", "type": "regex-list", "enabled": True,
                  "entries": [{"pattern": "(.)\\1+", "replacement": "\\1"}]}


def test_word_walks_are_charged_what_they_built(app_module, monkeypatch):
    # Expand, then shrink back: each walk's result is about its word, but it
    # built thousands of chars on the way. The shared budget must count those.
    import time

    _install_card(app_module, _DOUBLING_CARD, _COLLAPSE_CARD)
    letters = "abcdefghijklmnopqrstuvwxyz"
    raw = [" w" + letters[i // 26] + letters[i % 26] + "xy" for i in range(100)]
    peaks: list[int] = []
    real_walk, real_apply = pl_engine._postprocess_text, pl_engine._apply_bounded

    def walk(text, *a, **kw):
        peaks.append(len(text))
        return real_walk(text, *a, **kw)

    def apply(rule, text, bound):
        out = real_apply(rule, text, bound)
        if out is not None:
            peaks[-1] = max(peaks[-1], len(out))
        return out

    monkeypatch.setattr(pl_engine, "_postprocess_text", walk)
    monkeypatch.setattr(pl_engine, "_apply_bounded", apply)
    t0 = time.monotonic()
    out = capture_samples._align_words_to_final(_words(*raw), " ".join(raw))
    elapsed = time.monotonic() - t0
    assert len(out) == len(raw)
    assert sum(peaks) <= pl_engine.output_bound(sum(len(w) for w in raw))
    # A backstop only (the bound above is the check): ~1.2 s alone, and a
    # busy CI host took 2.3-3.0 s.
    assert elapsed < 10
