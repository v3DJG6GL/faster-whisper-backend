"""The streaming handshake's model-load error frame: a 4xx refusal (model not
on ALLOWED_MODELS, malformed id) carries the same client-safe detail the batch
route returns, while a genuine load failure keeps the generic text."""

from fastapi import HTTPException

from tests._streaming_helpers import ws_drain
from tests.conftest import bearer
from faster_whisper_backend.transcription import models as tx_models


def _handshake_errors(client, raw_key):
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_key)) as ws:
        ws.send_json({"type": "config", "model": "whisper-1",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
        msgs = ws_drain(ws, 200)[0]
    return [m for m in msgs if m.get("type") == "error"]


def test_refused_model_reports_the_refusal_detail(client, make_user_key, monkeypatch):
    _, raw_alice = make_user_key("alice")
    detail = "Model 'large-v3' is not in the allowed list. Allowed: ['large-v3-turbo']"

    async def _refuse(name, *, lease=False):
        raise HTTPException(status_code=400, detail=detail)

    monkeypatch.setattr(tx_models, "_get_or_load_model", _refuse)

    errors = _handshake_errors(client, raw_alice)
    assert errors, "expected a model_load_failed frame"
    assert errors[0]["code"] == "model_load_failed"
    assert errors[0]["message"] == detail


def test_server_side_http_failure_keeps_the_generic_message(
        client, make_user_key, monkeypatch):
    _, raw_alice = make_user_key("alice")

    async def _timeout(name, *, lease=False):
        raise HTTPException(status_code=503,
                            detail="Auto-convert of '/srv/models/x' timed out")

    monkeypatch.setattr(tx_models, "_get_or_load_model", _timeout)

    errors = _handshake_errors(client, raw_alice)
    assert errors, "expected a model_load_failed frame"
    assert errors[0]["message"] == "model could not be loaded"


def test_long_refusal_detail_reaches_the_client_whole(client, make_user_key, monkeypatch):
    """A long allowlist used to be cut at the 120-char log-field cap, leaving
    the client a broken model name and a missing entry."""
    _, raw_alice = make_user_key("alice")
    detail = ("Model 'large-v3' is not in the allowed list. Allowed: "
              "['Systran/faster-whisper-large-v3', "
              "'deepdml/faster-whisper-large-v3-turbo-ct2', 'small']")
    assert len(detail) > 120

    async def _refuse(name, *, lease=False):
        raise HTTPException(status_code=400, detail=detail)

    monkeypatch.setattr(tx_models, "_get_or_load_model", _refuse)

    errors = _handshake_errors(client, raw_alice)
    assert errors, "expected a model_load_failed frame"
    assert errors[0]["message"] == detail
