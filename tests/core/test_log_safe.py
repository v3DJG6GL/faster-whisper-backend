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


def test_secure_log_dir_only_tightens_a_directory_we_created(monkeypatch):
    # LOG_FILE is operator-chosen: chmod-ing a pre-existing /var/log to 0700
    # would lock every other daemon out of it, so only a freshly created
    # directory is ours to secure.
    from faster_whisper_backend import main
    seen = []
    monkeypatch.setattr(store_common, "secure_dir", seen.append)
    main._secure_log_dir("/some/dir", created=False)
    assert seen == []
    main._secure_log_dir("/some/dir", created=True)
    assert seen == ["/some/dir"]
