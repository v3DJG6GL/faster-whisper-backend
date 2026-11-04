"""A live utterance whose final decode raised: the session falls back to the
partial-committed text, and the ledger row must say the decode failed (an
error row, classified like the identical batch failure), not a success."""

from tests._streaming_helpers import dictate_session

_STREAM_URL = "/v1/audio/transcriptions/stream"


def _oom(*a, **kw):
    # Patched over _trim_trailing_nonspeech: only decode_final calls it, so the previews still decode.
    raise RuntimeError("CUDA failed with error out of memory")


def _stream_rows():
    from faster_whisper_backend.stats import recent_transcriptions_store
    return [r for r in recent_transcriptions_store.list_recent(limit=10)
            if r.get("source") == "stream"]


def test_failed_final_decode_is_recorded_as_a_classified_error(
        client, app_module, monkeypatch, energy_vad):
    from faster_whisper_backend.streaming import routes as s_routes

    monkeypatch.setattr(s_routes, "_trim_trailing_nonspeech", _oom)

    msgs = dictate_session(client)
    assert any(m["type"] == "final" for m in msgs), msgs
    rows = _stream_rows()
    assert rows, "the fallback utterance wrote no ledger row"
    row = rows[0]
    assert row["status"] == "error"
    assert row["error_class"] == "cuda_oom"
    assert row["error_stage"] == "transcribing"


def test_failed_final_decode_receipt_says_the_decode_failed(
        client, app_module, monkeypatch, caplog, energy_vad):
    """The decode never ran, so the receipt must not call the fallback
    utterance "[!] empty output" or advise loosening the thresholds."""
    import logging

    from faster_whisper_backend.streaming import routes as s_routes

    monkeypatch.setattr(s_routes, "_trim_trailing_nonspeech", _oom)
    with caplog.at_level(logging.INFO, logger="faster_whisper_backend.streaming.routes"):
        dictate_session(client)
    receipts = [r.getMessage() for r in caplog.records
                if _STREAM_URL in r.getMessage() and "═" in r.getMessage()]
    assert receipts, "no receipt was logged"
    assert "[!] no final decode" in receipts[0]
    assert "final decode failed" in receipts[0]
    assert "empty output" not in receipts[0]
    assert "thresholds suppressed" not in receipts[0]


def test_decoded_utterance_stays_an_ok_row(client, app_module, monkeypatch, energy_vad):
    dictate_session(client)
    rows = _stream_rows()
    assert rows
    assert rows[0]["status"] == "ok"
    assert rows[0]["error_class"] is None


def test_failed_final_decode_with_no_fallback_text_still_records_the_error(
        client, app_module, monkeypatch, energy_vad):
    """The common shape of a real fault: the previews fail too, so nothing is
    committed and the session drops the utterance. No `final` goes out, but
    the error row must still land."""
    from faster_whisper_backend.streaming import routes as s_routes

    real_kwargs = s_routes._build_transcribe_kwargs

    def _kwargs(*a, final, **kw):
        if not final:
            _oom()                            # every preview fails too
        return real_kwargs(*a, final=final, **kw)

    monkeypatch.setattr(s_routes, "_build_transcribe_kwargs", _kwargs)
    monkeypatch.setattr(s_routes, "_trim_trailing_nonspeech", _oom)

    msgs = dictate_session(client)
    assert not any(m["type"] == "final" and (m.get("committed") or m.get("tail"))
                   for m in msgs), msgs
    assert any(m["type"] == "utterance" and m.get("reason") == "error" for m in msgs), msgs
    # No text, so no trace half (and no source tag): the error-path row
    # alone, like a failed batch job's — found by its kind.
    from faster_whisper_backend.stats import recent_transcriptions_store
    rows = [r for r in recent_transcriptions_store.list_recent(limit=10)
            if r.get("kind") == "dictate"]
    assert rows, "the dropped utterance wrote no ledger row"
    row = rows[0]
    assert row["status"] == "error"
    assert row["error_class"] == "cuda_oom"
    assert row["error_stage"] == "transcribing"


def test_failed_final_decode_broadcasts_an_error_entry(
        client, app_module, monkeypatch, energy_vad):
    """The live /quick-config entry must carry the same status as the row a
    reload reads, not a hard-coded "ok"."""
    from faster_whisper_backend.quick_config import recent_feed
    from faster_whisper_backend.streaming import routes as s_routes

    monkeypatch.setattr(s_routes, "_trim_trailing_nonspeech", _oom)
    q = recent_feed.subscribe()
    try:
        dictate_session(client)
        entries = []
        while not q.empty():
            item = q.get_nowait()
            if item.get("event") == "trace" and item["data"].get("source") == "stream":
                entries.append(item["data"])
    finally:
        recent_feed.unsubscribe(q)
    assert entries, "no live entry was broadcast"
    assert entries[0]["status"] == "error"
    assert _stream_rows()[0]["status"] == "error"
