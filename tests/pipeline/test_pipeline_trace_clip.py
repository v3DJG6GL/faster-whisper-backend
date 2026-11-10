"""The pipeline trace keeps each step's text only up to the cap every consumer
already cuts it at (receipt, /quick-config live entry + durable row, /reports).

A card whose entries expand the text holds one intermediate string per changing
step; uncut, a card of a few hundred steps retained hundreds of MB per job. The
cut copies carry the uncut length and whether the step changed the text, so
every consumer renders exactly what it rendered from the uncut trace."""

import json

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.reports import store as reports_store
from faster_whisper_backend.stats import recent_transcriptions_store as rts
from faster_whisper_backend.transcription import receipt as tx_receipt
from tests.conftest import FakeInfo
from tests.settings._rules_helpers import terminal_rule as _term

_CAP = 50_000


def _entry(pattern, replacement):
    return {"pattern": pattern, "replacement": replacement}


def _set(app_module, *cards):
    app_module.cfg.PIPELINE_RULES = [*cards, _term()]
    pl_engine.rebuild_caches()


# Grows "ab" to 64 000 "x" + "ab", then changes the text past the cap, at its
# start, and cuts it back to exactly the cap (the cut before-prefix then equals
# the whole after-text). A disabled card adds skipped rows on the long text.
_LONG_CARD = {"name": "grow", "label": "Grow", "type": "regex-list", "enabled": True,
              "entries": [_entry(r"\A", "x" * 64),
                          *[_entry("x", "x" * 10)] * 3,
                          _entry(r"b\Z", "c"),
                          _entry(r"\Ax", "y"),
                          _entry(r"(?s)\A(.{50000}).*", r"\1"),
                          _entry(r"\Ay", "x")]}
_SKIPPED_CARD = {"name": "off", "label": "Off", "type": "regex-list", "enabled": False,
                 "entries": [_entry("q", "Q"), _entry("r", "R")]}


def _traces(app_module, monkeypatch, text, *cards):
    """(output, uncut trace, cut trace) of one walk over `cards`."""
    _set(app_module, *cards)
    with monkeypatch.context() as m:
        m.setattr(pl_engine, "_TRACE_TEXT_MAX", 10 ** 12, raising=False)
        full: list = []
        out_full = pl_engine._postprocess_text(text, model_name="", trace=full)
    cut: list = []
    out = pl_engine._postprocess_text(text, model_name="", trace=cut)
    assert out == out_full == pl_engine._postprocess_text(text, model_name="")
    return out, full, cut


def test_trace_cap_matches_every_consumer_cap():
    assert pl_engine._TRACE_TEXT_MAX == _CAP == rts._CAP_RAW == reports_store._CAP_RAW


def test_trace_keeps_at_most_the_store_cap_per_step(app_module, monkeypatch):
    _out, full, cut = _traces(app_module, monkeypatch, "ab", _LONG_CARD, _SKIPPED_CARD)
    assert max(len(t) for row in full for t in row[1:]) > _CAP
    assert [row[0] for row in cut] == [row[0] for row in full]
    for (_, fb, fa), (_, cb, ca) in zip(full, cut):
        for f, c in ((fb, cb), (fa, ca)):
            assert len(c) <= _CAP
            assert c == f[:_CAP]
            assert getattr(c, "full_len", len(c)) == len(f)


def test_cut_trace_renders_identically_in_every_consumer(app_module, monkeypatch):
    raw = "ab"
    final, full, cut = _traces(app_module, monkeypatch, raw, _LONG_CARD, _SKIPPED_CARD)

    def block(steps):
        return tx_receipt._format_request_block(
            file_label="x.wav", model_name="m", info=FakeInfo(duration=1.0),
            kwargs={}, seg_diag=[], raw=raw, final=final, steps=steps)

    assert block(cut) == block(full)
    assert "…[+14002 chars]" in block(cut)
    # The /quick-config live entry and the durable row (cap_trace_fields), and
    # the /reports row: same lists, same JSON.
    assert rts.cap_trace_fields(raw, final, cut) == rts.cap_trace_fields(raw, final, full)
    assert (json.dumps(rts._truncate_steps(cut), ensure_ascii=False)
            == json.dumps(rts._truncate_steps(full), ensure_ascii=False))
    assert reports_store._truncate_steps(cut) == reports_store._truncate_steps(full)
    assert json.dumps(cut, ensure_ascii=False) == json.dumps(
        [(n, b[:_CAP], a[:_CAP]) for n, b, a in full], ensure_ascii=False)

    from faster_whisper_backend.quick_config import recent_feed

    q = recent_feed.subscribe()
    try:
        entries = []
        for i, steps in enumerate((full, cut)):
            recent_feed.record_trace(request_id=f"r{i}", model="m", raw=raw,
                                     steps=steps, final=final, language="de")
            entries.append(q.get_nowait()["data"])
    finally:
        recent_feed.unsubscribe(q)
    for e in entries:
        for k in ("ts", "created_ts", "request_id"):
            e.pop(k)
    assert entries[0] == entries[1]
    assert json.dumps(entries[0]) == json.dumps(entries[1])


def test_trace_retained_text_is_bounded_by_steps_times_cap(app_module, monkeypatch):
    # Two 10x entries, then 198 one-char prefix steps on a 5 000-char text:
    # every step's text is ~500 000 chars, ~200 M chars retained uncut.
    card = {"name": "many", "label": "Many", "type": "regex-list", "enabled": True,
            "entries": [_entry("(.)", "\\1" * 10)] * 2 + [_entry("^", "Z")] * 198}
    _set(app_module, card)
    text = "a" * 5_000
    trace: list = []
    out = pl_engine._postprocess_text(text, model_name="", trace=trace)
    assert out == pl_engine._postprocess_text(text, model_name="")
    assert len(out) > 500_000 and len(trace) >= 200
    retained = sum(len(t) for _n, b, a in trace for t in (b, a))
    assert retained <= len(trace) * 2 * _CAP


def test_rows_on_one_long_text_share_one_cut_copy(app_module, monkeypatch):
    # 200 skipped rows on a 60 000-char text hold one cut copy, not 400.
    off = {"name": "off", "label": "Off", "type": "regex-list", "enabled": False,
           "entries": [_entry(f"q{i}", "Q") for i in range(200)]}
    _set(app_module, off)
    text = "\U0001f600" + "a" * 59_999
    trace: list = []
    pl_engine._postprocess_text(text, model_name="", trace=trace)
    assert len(trace) == 200
    texts = [t for _n, b, a in trace for t in (b, a)]
    assert len({id(t) for t in texts}) == 1
    assert all(isinstance(t, pl_engine._ClippedText) and t.full_len == 60_000
               and t.changed is False for t in texts)


def test_changing_rows_share_the_cut_copy_they_hand_on(app_module, monkeypatch):
    # Each changing step's cut after-text is the next step's cut before-text.
    card = {"name": "pre", "label": "Pre", "type": "regex-list", "enabled": True,
            "entries": [_entry(r"\A", "Z")] * 5}
    _set(app_module, card)
    trace: list = []
    pl_engine._postprocess_text("a" * 60_000, model_name="", trace=trace)
    assert len(trace) == 5
    for (_n1, _b1, a1), (_n2, b2, _a2) in zip(trace, trace[1:]):
        assert a1 is b2
        assert a1.changed is True
