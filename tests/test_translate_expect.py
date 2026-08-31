"""The client's `translate_expect` handshake declaration.

It decides whether each utterance's log receipt is HELD open for a translation
that arrives on a separate request. Getting `per_utterance` wrong is silent and
expensive in both directions: hold when nothing will claim, and every receipt
waits out the idle sweep before it is logged; don't hold when a claim is coming,
and the receipt and its translation are logged as unrelated blocks.
"""

from streaming_routes import _parse_translate_expect as parse


def test_absent_or_malformed_is_no_declaration():
    assert parse({}) is None
    assert parse({"translate_expect": None}) is None
    assert parse({"translate_expect": "en,fr"}) is None
    assert parse({"translate_expect": {"targets": "en"}}) is None
    assert parse({"translate_expect": {"targets": []}}) is None
    # A list of non-strings, or of blanks, carries no usable target.
    assert parse({"translate_expect": {"targets": [1, 2]}}) is None
    assert parse({"translate_expect": {"targets": ["  ", ""]}}) is None


def test_targets_are_cleaned_and_bounded():
    got = parse({"translate_expect": {"targets": [" en ", "fr", 7, "", "x" * 40]}})
    assert got["targets"] == ["en", "fr", "x" * 16]
    # At most eight, so a hostile handshake can't widen the log block.
    many = parse({"translate_expect": {"targets": [f"l{i}" for i in range(20)]}})
    assert len(many["targets"]) == 8


def test_per_utterance_defaults_to_true_for_older_clients():
    # A client from before the field held a receipt per utterance; keep that,
    # or shipping this backend first would silently drop the merged receipts.
    got = parse({"translate_expect": {"targets": ["en"]}})
    assert got["per_utterance"] is True


def test_stop_timing_declares_per_utterance_false():
    got = parse({"translate_expect": {"targets": ["en"], "per_utterance": False}})
    assert got["per_utterance"] is False
    assert got["include_original"] is False


def test_include_original_is_coerced_not_trusted():
    got = parse({"translate_expect": {"targets": ["en"], "include_original": "yes"}})
    assert got["include_original"] is True
