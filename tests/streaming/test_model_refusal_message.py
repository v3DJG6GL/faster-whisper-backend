"""The streaming handshake's model-load error frame: a 4xx refusal (model not
on ALLOWED_MODELS, malformed id) carries the same client-safe detail the batch
route returns, while a genuine load failure keeps the generic text."""

from fastapi import HTTPException

from tests._streaming_helpers import config_drain
from tests._transcription_helpers import fail_model_load


def _handshake_errors(client, raw_key):
    return [m for m in config_drain(client, raw_key) if m.get("type") == "error"]


def test_refused_model_reports_the_refusal_detail(client, make_user_key, monkeypatch):
    _, raw_alice = make_user_key("alice")
    detail = "Model 'large-v3' is not in the allowed list. Allowed: ['large-v3-turbo']"
    fail_model_load(monkeypatch, HTTPException(status_code=400, detail=detail))

    errors = _handshake_errors(client, raw_alice)
    assert errors, "expected a model_load_failed frame"
    assert errors[0]["code"] == "model_load_failed"
    assert errors[0]["message"] == detail


def test_server_side_http_failure_keeps_the_generic_message(
        client, make_user_key, monkeypatch):
    _, raw_alice = make_user_key("alice")
    fail_model_load(monkeypatch, HTTPException(
        status_code=503, detail="Auto-convert of '/srv/models/x' timed out"))

    errors = _handshake_errors(client, raw_alice)
    assert errors, "expected a model_load_failed frame"
    assert errors[0]["message"] == "model could not be loaded"


def test_long_refusal_detail_reaches_the_client_whole(client, make_user_key, monkeypatch):
    """The 120-char log-field cap must not cut the client's detail: a cut
    allowlist leaves a broken model name and a missing entry."""
    _, raw_alice = make_user_key("alice")
    detail = ("Model 'large-v3' is not in the allowed list. Allowed: "
              "['Systran/faster-whisper-large-v3', "
              "'deepdml/faster-whisper-large-v3-turbo-ct2', 'small']")
    assert len(detail) > 120
    fail_model_load(monkeypatch, HTTPException(status_code=400, detail=detail))

    errors = _handshake_errors(client, raw_alice)
    assert errors, "expected a model_load_failed frame"
    assert errors[0]["message"] == detail
