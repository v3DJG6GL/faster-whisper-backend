"""store_common.log_safe: the control-character screen for caller-supplied
labels that end up in a log line the /logs viewer renders."""

from faster_whisper_backend.core import store_common


def test_cr_lf_and_c0_are_collapsed():
    assert store_common.log_safe("a\r\nb") == "a??b"
    assert store_common.log_safe("a\x00b") == "a?b"
    assert store_common.log_safe("a\x1bb") == "a?b"  # ESC (ANSI) sits in C0


def test_unicode_line_separators_are_collapsed():
    # U+2028/U+2029 are a forced line break under the viewer's pre-wrap, i.e.
    # the same forged-record problem as a bare LF.
    assert store_common.log_safe("a b") == "a?b"
    assert store_common.log_safe("a b") == "a?b"


def test_bidi_overrides_are_collapsed():
    # These reorder text WITHIN a line in both a terminal and a browser.
    for ch in "‪‫‬‭‮⁦⁧⁨⁩":
        assert store_common.log_safe("a" + ch + "b") == "a?b", repr(ch)


def test_c1_and_del_are_collapsed():
    assert store_common.log_safe("a\x7fb") == "a?b"
    assert store_common.log_safe("a\x9bb") == "a?b"


def test_zero_width_joiners_survive():
    # ZWNJ/ZWJ appear in legitimate Persian and Indic filenames — widening the
    # screen to all of \p{Cf} would eat them.
    assert store_common.log_safe("mo‌ji‍.wav") == "mo‌ji‍.wav"


def test_plain_values_and_none_are_unchanged():
    assert store_common.log_safe("normal-file.wav") == "normal-file.wav"
    assert store_common.log_safe(None) == ""


def test_length_is_capped():
    assert len(store_common.log_safe("x" * 5000)) == store_common.LOG_FIELD_MAX


def test_keep_end_keeps_the_last_chars():
    # A child's stderr ends with its fatal line; the default head cap would
    # log the middle of a 300-char tail and drop it.
    tail = "x" * 300 + "\nERROR: Unsupported URL"
    out = store_common.log_safe(tail, keep_end=True)
    assert len(out) == store_common.LOG_FIELD_MAX
    assert out.endswith("?ERROR: Unsupported URL")
    assert store_common.log_safe(tail) == "x" * store_common.LOG_FIELD_MAX
    assert store_common.log_safe(None, keep_end=True) == ""


def test_lone_surrogates_are_collapsed():
    # json.loads keeps a lone surrogate; the utf-8 log file handler cannot
    # encode one, so the whole record would be dropped from the file.
    assert store_common.log_safe("a\ud800b") == "a?b"
    assert store_common.log_safe("a\udfffb") == "a?b"
    store_common.log_safe("x\ud800").encode("utf-8")
    assert store_common.log_safe("a\U0001F600b") == "a\U0001F600b"
    assert store_common.log_safe("a‌b‍b") == "a‌b‍b"


def _reference(s, *, keep_end=False):
    # The screen-then-cap formula: the cap must not change any output.
    safe = store_common._LOG_UNSAFE_RE.sub("?", s or "")
    return (safe[-store_common.LOG_FIELD_MAX:] if keep_end
            else safe[:store_common.LOG_FIELD_MAX])


def test_cap_before_screen_gives_the_same_output():
    mixed = ("a\r\nb‪c\ud800d\x7f" + "é😀" * 40) * 30 + "\nERROR: x\ud800"
    for s in (mixed, mixed[:150], "\x7f" * 4_000, "x\n" * 61, "", None):
        for keep_end in (False, True):
            assert store_common.log_safe(s, keep_end=keep_end) == \
                _reference(s, keep_end=keep_end), (repr(s)[:40], keep_end)


def test_a_huge_label_is_capped_before_it_is_screened():
    # A JSON url field can be ~4 M chars; screening it whole costs a
    # quarter second of event-loop CPU per call before any length check.
    import time
    s = "\x7f" * 4_000_000
    for keep_end in (False, True):
        t0 = time.perf_counter()
        out = store_common.log_safe(s, keep_end=keep_end)
        assert time.perf_counter() - t0 < 0.05
        assert out == "?" * store_common.LOG_FIELD_MAX
