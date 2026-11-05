"""The streaming handshake's model-load error frame: a 4xx refusal (model not
on ALLOWED_MODELS, malformed id) carries the same client-safe detail the batch
route returns, while a genuine load failure keeps the generic text."""

from fastapi import HTTPException

from tests._streaming_helpers import config_drain
from tests._transcription_helpers import fail_model_load


def _load_error_frame(client, make_user_key, monkeypatch, exc):
    """Alice's handshake when every model load raises `exc`: its first error frame."""
    _, raw_alice = make_user_key("alice")
    fail_model_load(monkeypatch, exc)
    errors = [m for m in config_drain(client, raw_alice) if m.get("type") == "error"]
    assert errors, "expected a model_load_failed frame"
    return errors[0]


def test_refused_model_reports_the_refusal_detail(client, make_user_key, monkeypatch):
    detail = "Model 'large-v3' is not in the allowed list. Allowed: ['large-v3-turbo']"
    error = _load_error_frame(client, make_user_key, monkeypatch,
                              HTTPException(status_code=400, detail=detail))
    assert error["code"] == "model_load_failed"
    assert error["message"] == detail


def test_server_side_http_failure_keeps_the_generic_message(
        client, make_user_key, monkeypatch):
    error = _load_error_frame(client, make_user_key, monkeypatch, HTTPException(
        status_code=503, detail="Auto-convert of '/srv/models/x' timed out"))
    assert error["message"] == "model could not be loaded"


def test_long_refusal_detail_reaches_the_client_whole(client, make_user_key, monkeypatch):
    """The 120-char log-field cap must not cut the client's detail: a cut
    allowlist leaves a broken model name and a missing entry."""
    detail = ("Model 'large-v3' is not in the allowed list. Allowed: "
              "['Systran/faster-whisper-large-v3', "
              "'deepdml/faster-whisper-large-v3-turbo-ct2', 'small']")
    assert len(detail) > 120
    error = _load_error_frame(client, make_user_key, monkeypatch,
                              HTTPException(status_code=400, detail=detail))
    assert error["message"] == detail
