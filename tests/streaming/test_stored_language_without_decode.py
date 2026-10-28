"""An utterance finalized WITHOUT a decode of its own (the near-silence gate
skipped it) still stores the language the session decoded in. A blank row
language makes a captures re-apply run every language's rules on it."""

import time

from tests._streaming_helpers import const_pcm as _pcm, ws_drain as _drain
from tests.conftest import FakeInfo

_STREAM_URL = "/v1/audio/transcriptions/stream"


def _gate_skipped_utterance(client, app_module, monkeypatch):
    """One utterance whose text the previews commit; the RMS gate is then
    forced shut so its finalize skips the decode. Returns the capture kwargs
    and the stream ledger rows."""
    from faster_whisper_backend.captures import store as captures_store
    from faster_whisper_backend.stats import recent_transcriptions_store
    from faster_whisper_backend.streaming import session as streaming_session

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True, raising=False)
    monkeypatch.setattr(captures_store, "ready_fills_cap", lambda: False)
    created = []
    monkeypatch.setattr(captures_store, "create_capture",
                        lambda **kw: created.append(kw) or "cap-test-id")

    with client.websocket_connect(_STREAM_URL) as ws:
        # No `language` key: the session inherits DEFAULT_LANGUAGE.
        ws.send_json({"type": "config", "model": "whisper-1",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))
        time.sleep(0.5)
        monkeypatch.setattr(streaming_session, "rms_dbfs", lambda a: -999.0)
        ws.send_bytes(_pcm(0, 1500))
        time.sleep(0.5)
        ws.send_json({"type": "stop"})
        msgs, _code = _drain(ws)
    assert any(m["type"] == "final" for m in msgs), msgs
    rows = [r for r in recent_transcriptions_store.list_recent(limit=10)
            if r.get("source") == "stream"]
    return created, rows


def test_gate_skipped_utterance_stores_the_previews_language(
        client, app_module, monkeypatch):
    created, rows = _gate_skipped_utterance(client, app_module, monkeypatch)
    assert created, "setup failed: no capture was handed to the store"
    assert created[0]["language"] == "de"
    assert rows and rows[0]["language"] == "de"


def test_gate_skipped_utterance_falls_back_to_default_language(
        client, app_module, monkeypatch, fake_model):
    # Previews that report no language: only the inherited default is known.
    fake_model._info = FakeInfo(language=None)
    monkeypatch.setattr(app_module.cfg, "DEFAULT_LANGUAGE", "fr", raising=False)
    created, rows = _gate_skipped_utterance(client, app_module, monkeypatch)
    assert created, "setup failed: no capture was handed to the store"
    assert created[0]["language"] == "fr"
    assert rows and rows[0]["language"] == "fr"
