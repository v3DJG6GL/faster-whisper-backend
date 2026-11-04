"""The streaming WebSocket path honours per-identity config — locked
overrides surfaced in the `ready` frame, and profile decode params reach the
final decode. Driven in-process; no faster-whisper needed."""

import json
import logging
import time

import pytest

from faster_whisper_backend.settings import effective_config
from tests._streaming_helpers import (
    config_drain, config_frame, const_pcm, handshake, speak_and_stop, ws_drain)
from tests._transcription_helpers import bind_profile, fail_model_load
from tests.conftest import bearer
from faster_whisper_backend.transcription import models as tx_models

OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"

_pcm = const_pcm


def _drain(ws, limit=200):
    return ws_drain(ws, limit)[0]


def _profile(client, h, name, **fields):
    r = client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text


def _bind(client, h, uid, **binding):
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h,
                     json={"pages": {}, "config": {"overrides": {}, "profiles": [], **binding}})
    assert r.status_code == 200, r.text


def test_ready_frame_reports_locked_handshake_override(client, make_user_key):
    _, raw_alice = bind_profile(client, make_user_key, "p",
                                {"TEMPERATURE": "0.0", "DEFAULT_LANGUAGE": "de",
                                 "locks": ["TEMPERATURE", "DEFAULT_LANGUAGE"]})

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        ws.send_json({"type": "config", "model": "whisper-1", "language": "fr",
                      "decode_overrides": {"temperature": 0.7},
                      "audio": {"format": "pcm_s16le"}})
        ready = ws.receive_json()
    assert ready["type"] == "ready"
    assert set(ready["overrides_ignored"]) == {"temperature", "language"}


def test_final_decode_uses_profile_beam(client, make_user_key, fake_model,
                                        app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_alice = bind_profile(client, make_user_key, "p", {"BEAM_SIZE": 7})

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        speak_and_stop(ws)

    # The final decode is the last transcribe call; it uses the assembler with
    # the profile's BEAM_SIZE (partials force STREAMING_PARTIAL_BEAM instead).
    assert fake_model.last_kwargs["beam_size"] == 7


def test_stream_request_override_profile_applies(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """A per-request `override_profile` named in the WS handshake is
    honoured — echoed as `profile_applied` in the ready frame and its decode
    params reach the final decode (least-specific layer, no admin lock)."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profile(client, h, "fast", BEAM_SIZE=9)
    _, raw_alice = make_user_key("alice")     # no binding → inherits global gate (on)

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        ready = handshake(ws, override_profile="fast")
        assert ready["type"] == "ready"
        assert ready["profile_applied"] == "fast"
        speak_and_stop(ws)

    assert fake_model.last_kwargs["beam_size"] == 9


def test_stream_picks_up_binding_change_without_reconnect(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """A binding change made WHILE a dictation WebSocket is open applies from
    the next decode: ident is re-resolved when the config version changes, not
    frozen at the handshake until reconnect."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    # Both profiles in ONE save — a second POST would replace OVERRIDE_PROFILES.
    r = client.post(f"{OV}/state", headers=h,
                    json={"OVERRIDE_PROFILES": {"p7": {"BEAM_SIZE": 7},
                                                "p3": {"BEAM_SIZE": 3}}})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice")
    _bind(client, h, uid, profiles=["p7"])

    from faster_whisper_backend.auth import api_keys_store
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        # utterance 1 under profile p7
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))
        # admin re-binds alice p7 -> p3 mid-session (bumps the config version)
        api_keys_store.set_user_permissions(
            uid, {"pages": {}, "config": {"overrides": {}, "profiles": ["p3"], "locks": []}})
        # utterance 2 must pick up p3 (beam 3) on the same connection
        speak_and_stop(ws)

    assert fake_model.last_kwargs["beam_size"] == 3


def test_final_drops_low_confidence_hallucination_segment(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """A final-decode segment that is BOTH very low confidence AND fell through
    the temperature ladder (a hallucination on near-silence) is dropped from the
    committed text; a normal segment in the same decode is kept."""
    from tests.conftest import FakeSegment
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)               # lock down
    _, raw_alice = make_user_key("alice")

    bad = FakeSegment(" thank you for watching", 0.0, 1.0)
    bad.avg_logprob, bad.temperature = -2.32, 1.0        # failed decode → drop
    good = FakeSegment(" der patient hat fieber.", 1.0, 2.0)
    good.avg_logprob, good.temperature = -0.2, 0.0       # healthy → keep
    fake_model._segments = [bad, good]

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        msgs = speak_and_stop(ws)[0]

    finals = [m for m in msgs if m["type"] == "final"]
    doc = (finals[-1]["committed"] + finals[-1]["tail"]) if finals else ""
    assert "fieber" in doc
    assert "watching" not in doc


def test_stream_prompt_sentinel_inherit_clear_value(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """The WS handshake prompt is a present-vs-absent sentinel. Absent → inherit
    DEFAULT_PROMPT; explicit "" → CLEAR (no initial_prompt); a value → used verbatim.
    The first utterance's final decode is seeded straight from base_prompt."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "DEFAULT_PROMPT", "SERVER PROMPT", raising=False)
    make_user_key("admin", is_admin=True)
    _, raw_alice = make_user_key("alice")

    def _run(conf_extra):
        with client.websocket_connect(
                "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
            assert handshake(ws, **conf_extra)["type"] == "ready"
            speak_and_stop(ws)
        return fake_model.last_kwargs.get("initial_prompt")

    assert _run({}) == "SERVER PROMPT"
    assert _run({"prompt": ""}) is None
    assert _run({"prompt": "my terms"}) == "my terms"


def test_stream_handshake_drops_lone_surrogates(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """A JSON handshake can carry a lone UTF-16 surrogate (half an emoji a
    client cut with a UTF-16 slice). Kept in the prompt it would make
    faster-whisper's tokenizer raise on every decode of the connection, so
    nothing gets typed, and in an output wrapper it would break the UTF-8
    frame, so it is dropped."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_alice = make_user_key("alice")
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        # json.dumps escapes the surrogate as "\\ud83d", as a browser sends it.
        ws.send_text(json.dumps(config_frame(
            prompt="Kardiologie \ud83d", decode_overrides={"output_prefix": ">> \ud83d"})))
        assert ws.receive_json()["type"] == "ready"
        msgs = speak_and_stop(ws)[0]
    assert fake_model.last_kwargs.get("initial_prompt") == "Kardiologie"
    finals = [m for m in msgs if m.get("type") == "final"
              and (m.get("committed") or m.get("tail"))]
    assert finals
    assert all((f.get("committed") or f.get("tail")).startswith(">> ")
               and "\ud83d" not in (f.get("committed") or f.get("tail"))
               for f in finals)


def test_setup_window_error_delivers_internal_error_and_closes(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """A failure in the handshake→loop setup window (after the audio consumer task
    has started, before the receive loop's own cleanup is armed) must still deliver
    an `internal` error frame and close without hanging."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_alice = make_user_key("alice")

    real_cfg_for = effective_config.cfg_for

    def boom(model_id, field, ident=None):
        # The per-identity idle-timeout read happens AFTER the consumer task is
        # created — i.e. squarely inside the setup window this path guards.
        if field == "STREAMING_IDLE_TIMEOUT_S":
            raise RuntimeError("kaboom in setup window")
        return real_cfg_for(model_id, field, ident)

    monkeypatch.setattr(effective_config, "cfg_for", boom)

    msgs = config_drain(client, raw_alice)

    errors = [m for m in msgs if m.get("type") == "error"]
    assert errors, f"expected an internal error frame, got {msgs!r}"
    assert errors[0]["code"] == "internal"
    # Security: the client frame must NOT leak the raw exception text (it can
    # carry filesystem/internal detail) — it is logged server-side instead.
    assert "kaboom" not in errors[0]["message"]
    assert errors[0]["message"] == "internal error"


def test_model_load_failure_delivers_generic_error_and_closes(
        client, make_user_key, app_module, monkeypatch):
    """A model-load failure during the handshake must deliver a
    `model_load_failed` frame and close — without leaking the raw exception
    text (it can carry the model dir / filesystem path); the detail is logged
    server-side instead."""
    _, raw_alice = make_user_key("alice")
    fail_model_load(monkeypatch, RuntimeError("/srv/models/secret-path missing"))

    msgs = config_drain(client, raw_alice)

    errors = [m for m in msgs if m.get("type") == "error"]
    assert errors, f"expected a model_load_failed frame, got {msgs!r}"
    assert errors[0]["code"] == "model_load_failed"
    assert "secret-path" not in errors[0]["message"]
    assert errors[0]["message"] == "model could not be loaded"


def _refused_load_with_peer_gone(client, make_user_key, monkeypatch, caplog, gone_send):
    """Drive a failed model load whose refusal send hits a gone peer
    (``gone_send`` replaces Starlette's WebSocket.send); returns the ledger
    rows and the ERROR records the handler wrote."""
    _, raw_alice = make_user_key("alice")
    fail_model_load(monkeypatch, RuntimeError("/srv/models/secret-path missing"))

    from starlette.websockets import WebSocket as _WS
    monkeypatch.setattr(_WS, "send", gone_send(_WS.send))

    from faster_whisper_backend.stats import metrics
    rows = []
    monkeypatch.setattr(metrics, "record_transcription", lambda **kw: rows.append(kw))
    caplog.set_level(logging.ERROR, logger="faster_whisper_backend.streaming.routes")

    from faster_whisper_backend.streaming import routes as streaming_routes

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        ws.send_json(config_frame())
        # No close frame ever reaches the test session (the refusal send failed),
        # so wait for the handler to reach its teardown — it drops the session
        # id AFTER the error branch would have logged / recorded the row —
        # instead of draining. Leaving the block earlier would cancel the app.
        for _ in range(400):
            if not streaming_routes._active_sessions:
                break
            time.sleep(0.025)
        assert not streaming_routes._active_sessions, "handler did not finish"

    errors = [r for r in caplog.records
              if r.levelno >= logging.ERROR and "error:" in r.getMessage()]
    return rows, errors


def _assert_one_load_failure_row(rows, errors):
    # The model-load branch writes exactly ONE classified error row; a refusal
    # error escaping to the blanket handler would add a second (and log).
    err_rows = [r for r in rows if r.get("status") == "error"]
    assert len(err_rows) == 1, rows
    assert err_rows[0]["kind"] == "dictate" and err_rows[0]["error_class"], rows
    assert not errors, [r.getMessage() for r in errors]


def test_model_load_failure_with_peer_gone_is_not_logged_as_server_error(
        client, make_user_key, app_module, monkeypatch, caplog):
    """A peer that hangs up while the handshake is being refused (here: during
    a failed model load) is a plain disconnect — not a traceback and not a
    second, unclassified error row. Here the ASGI send raises OSError, which
    Starlette turns into WebSocketDisconnect on the refusal send (the close is
    never reached). The handler's outer `except WebSocketDisconnect` would
    swallow that too, so the RuntimeError variant below is the one that pins
    the guard."""
    def _gone(real_send):
        async def _gone_send(self, message):
            # Fail at the ASGI layer (below Starlette's own send()), so its state
            # machine also flips the socket to DISCONNECTED — the exact uvicorn
            # peer-gone sequence.
            if message["type"] != "websocket.accept":
                async def _asgi_gone(_msg):
                    raise OSError("peer gone")
                self._send = _asgi_gone
            return await real_send(self, message)
        return _gone_send

    _assert_one_load_failure_row(
        *_refused_load_with_peer_gone(client, make_user_key, monkeypatch, caplog, _gone))


def test_model_load_failure_on_a_disconnected_socket_is_not_logged_as_server_error(
        client, make_user_key, app_module, monkeypatch, caplog):
    """The socket is already DISCONNECTED when the refusal is sent, so
    Starlette raises RuntimeError — which only _safe_ws_send's guard keeps
    from reaching the blanket `except Exception` (an "error:" log and a
    second error row)."""
    from starlette.websockets import WebSocketState

    def _gone(real_send):
        async def _gone_send(self, message):
            if message["type"] != "websocket.accept":
                self.application_state = WebSocketState.DISCONNECTED
            return await real_send(self, message)
        return _gone_send

    _assert_one_load_failure_row(
        *_refused_load_with_peer_gone(client, make_user_key, monkeypatch, caplog, _gone))


def test_model_load_failure_records_a_classified_error_row(
        client, make_user_key, app_module, monkeypatch):
    """The model-load refusal returns before the blanket handler's ledger row,
    so it writes its own: a CUDA OOM on a live dictation shows on /stats as
    cuda_oom, like the identical batch failure."""
    _, raw_alice = make_user_key("alice")
    fail_model_load(monkeypatch,
                    RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    from faster_whisper_backend.stats import metrics
    rows = []
    monkeypatch.setattr(metrics, "record_transcription", lambda **kw: rows.append(kw))

    msgs = config_drain(client, raw_alice)

    assert any(m.get("code") == "model_load_failed" for m in msgs), msgs
    err_rows = [r for r in rows if r.get("status") == "error"]
    assert len(err_rows) == 1, rows
    assert err_rows[0]["error_class"] == "cuda_oom"
    assert err_rows[0]["error_stage"] == "transcribing"
    assert err_rows[0]["kind"] == "dictate"


def test_partial_model_load_failure_row_names_the_partial_model(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """The final model loads fine and the STREAMING_PARTIAL_MODEL one fails:
    the error row is charged to the model that failed, not the final one."""
    _, raw_alice = make_user_key("alice")
    monkeypatch.setattr(app_module.cfg, "STREAMING_PARTIAL_MODEL", "tiny-preview",
                        raising=False)
    real = tx_models._get_or_load_model

    async def _load(name, *, lease=False):
        if name == "tiny-preview":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return await real(name, lease=lease)

    monkeypatch.setattr(tx_models, "_get_or_load_model", _load)
    from faster_whisper_backend.stats import metrics
    rows = []
    monkeypatch.setattr(metrics, "record_transcription", lambda **kw: rows.append(kw))

    msgs = config_drain(client, raw_alice)

    assert any(m.get("code") == "model_load_failed" for m in msgs), msgs
    err_rows = [r for r in rows if r.get("status") == "error"]
    assert [r["model"] for r in err_rows] == ["tiny-preview"], rows


def test_handshake_drops_unknown_decode_override_keys(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """Unknown `decode_overrides` keys are discarded at the handshake: the
    narrowed req_overrides the decode closures hold for the connection's life
    (and re-walk per partial; spied via _client_stream_values) keeps only
    known keys, the honoured key still reaches the final decode, and the junk
    keys never reach the transcribe kwargs."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_alice = make_user_key("alice")

    from faster_whisper_backend.streaming import routes as streaming_routes
    seen_keys: list = []
    real_values = streaming_routes._client_stream_values

    def _spy(overrides, ident):
        seen_keys.append(sorted(overrides))
        return real_values(overrides, ident)

    monkeypatch.setattr(streaming_routes, "_client_stream_values", _spy)

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        ready = handshake(ws, decode_overrides={"beam_size": 5,
                                                "not_a_real_key": "x" * 1024,
                                                "__proto__": 1})
        assert ready["type"] == "ready"
        assert "overrides_ignored" not in ready   # nothing LOCKED, so nothing surfaced
        speak_and_stop(ws)

    assert fake_model.last_kwargs["beam_size"] == 5
    assert "not_a_real_key" not in fake_model.last_kwargs
    assert "__proto__" not in fake_model.last_kwargs
    assert seen_keys and all(k == ["beam_size"] for k in seen_keys), seen_keys


@pytest.mark.parametrize("language", ["en\nFAKE log record", "x" * 5000])
def test_handshake_language_that_is_not_a_code_is_treated_as_absent(
        client, make_user_key, fake_model, app_module, monkeypatch, language):
    """The handshake `language` reaches the usage row and faster-whisper, which
    logs it raw (CR/LF included) on every partial of an English-only model.
    A value not shaped like a language code is dropped at the handshake."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "DEFAULT_LANGUAGE", None, raising=False)
    _, raw_alice = make_user_key("alice")

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws, language=language)["type"] == "ready"
        speak_and_stop(ws)

    assert "language" in fake_model.last_kwargs   # a decode ran
    assert fake_model.last_kwargs["language"] is None


@pytest.mark.parametrize("raw,expected", [
    ("de", "de"), ("DE", "de"), ("de-DE", "de"), ("pt-BR", "pt"),
    ("en-US", "en"), ("haw", "haw"), ("xx", None), ("xx-YY", None),
    ("English", None), ("", ""), ("  ", ""), (None, None), (7, None),
    # An underscore locale (POSIX LANG, Java's Locale.toString()) folds as
    # batch folds it: /dictate sends one language box to both modes.
    ("de_DE", "de"), ("pt_BR", "pt"), ("zh_Hant_TW", "zh"), ("xx_YY", None),
    # A withdrawn ISO 639-1 code (old Java locales, YouTube) is Whisper's code.
    ("iw_IL", "he"), ("in-ID", "id"),
    # A language Whisper spells otherwise: a Norwegian browser reports nb-NO.
    ("nb", "no"), ("nb-NO", "no"), ("nb_NO", "no"), ("nob", "no"),
    ("fil", "tl"), ("fil_PH", "tl"),
])
def test_handshake_language_reduces_to_a_whisper_code(raw, expected):
    """faster-whisper's tokenizer raises on anything outside its own code
    list, which in streaming fails every decode with no error frame — so a
    case variant or locale tag is reduced to the Whisper code, and anything
    that does not reduce to one is treated as absent. Batch reduces the same
    value the same way."""
    from faster_whisper_backend.transcription.models import whisper_language_code
    from faster_whisper_backend.streaming.routes import _handshake_language
    assert _handshake_language(raw) == expected
    if isinstance(raw, str) and raw.strip():
        assert whisper_language_code(raw) == expected


@pytest.mark.parametrize("language,expected", [("EN", "en"), ("en-US", "en"), ("xx", None)])
def test_handshake_language_reaches_the_model_as_a_whisper_code(
        client, make_user_key, fake_model, app_module, monkeypatch, language, expected):
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "DEFAULT_LANGUAGE", None, raising=False)
    _, raw_alice = make_user_key("alice")

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        ready = handshake(ws, language=language)
        assert ready["type"] == "ready"
        # An unusable value is reported, not silently swapped.
        assert ("language" in ready.get("overrides_ignored", [])) == (expected is None)
        speak_and_stop(ws)

    assert "language" in fake_model.last_kwargs   # a decode ran
    assert fake_model.last_kwargs["language"] == expected


# A WebSocket has no request boundary, so the credential is re-checked
# mid-connection: with only the handshake's authenticate_ws, a revoked key would
# keep decoding and writing captures/trace/usage rows.

_drain_with_code = ws_drain   # _drain, but also returns the server's close code


def test_stream_closes_when_key_is_revoked_mid_session(
        client, make_user_key, app_module, monkeypatch):
    """Revoking the user mid-stream cuts the session: no further transcripts,
    an `unauthorized` notice, and a 4401 close. revoke_user bumps the config
    version, which is the signal _refresh_ident already consumes."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)      # lock down, so open mode is off
    uid, raw_alice = make_user_key("alice")

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))        # speech → partial decodes run
        time.sleep(0.5)
        api_keys_store.revoke_user(uid)        # bumps the config version
        ws.send_bytes(_pcm(8000, 1500))        # next decode re-auths → revoked
        time.sleep(0.5)
        ws.send_bytes(_pcm(8000, 200))         # producer notices → closes
        try:
            # Safety net only: the socket is normally already gone, so this is a
            # no-op; if the close regresses, the stop makes the drain below fail
            # the assertions instead of blocking forever.
            ws.send_json({"type": "stop"})
        except Exception:  # noqa: BLE001 — already closed, as expected
            pass
        msgs, code = _drain_with_code(ws)

    assert code == _WS_UNAUTH, f"expected a 4401 close, got {code!r} ({msgs!r})"
    assert any(m.get("code") == "unauthorized" for m in msgs), msgs
    # The in-flight utterance is deliberately dropped: revocation takes effect
    # immediately, so no closing document is handed to the revoked identity.
    assert not any(m.get("type") == "final" and m.get("last") for m in msgs), msgs


def test_revoked_session_closes_without_a_further_client_frame(
        client, make_user_key, app_module, monkeypatch):
    """Once a decode latches the revocation, the PUMP closes the socket itself —
    a client that then goes silent (sends no further frame) must not keep the
    revoked session open until the idle timeout; the producer's own
    _auth_revoked branch is only the backstop."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)      # lock down, so open mode is off
    uid, raw_alice = make_user_key("alice")

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))        # speech → partial decodes run
        time.sleep(0.5)
        api_keys_store.revoke_user(uid)        # bumps the config version
        ws.send_bytes(_pcm(8000, 1500))        # next decode re-auths → revoked
        # NO further client frame: the pump-side close must cut the socket on
        # its own, or the drain below blocks forever.
        msgs, code = _drain_with_code(ws)

    assert code == _WS_UNAUTH, (
        f"expected a 4401 close without further client frames, got {code!r} ({msgs!r})")
    assert any(m.get("code") == "unauthorized" for m in msgs), msgs


def test_revocation_during_the_handshake_still_closes_with_4401(
        client, make_user_key, app_module, monkeypatch):
    """A revocation landing between authentication and the end of the handshake
    (the config message, a cold model load) still closes the connection: the
    version snapshot predates authentication, so _refresh_ident sees the
    revocation and the first decode re-auths and closes."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.streaming import routes as streaming_routes
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)      # lock down, so open mode is off
    uid, raw_alice = make_user_key("alice")

    real = streaming_routes.authenticate_ws
    calls = []

    def _auth_then_revoke(ws):
        user = real(ws)
        if not calls:
            api_keys_store.revoke_user(uid)    # bumps the config version
        calls.append(1)
        return user
    monkeypatch.setattr(streaming_routes, "authenticate_ws", _auth_then_revoke)

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))        # first decode re-auths → revoked
        time.sleep(0.5)
        try:
            # Safety net only: if the session stays live, the stop lets the
            # drain finish and fail the assertions instead of hanging.
            ws.send_bytes(_pcm(0, 1500))
            ws.send_json({"type": "stop"})
        except Exception:  # noqa: BLE001 — already closed, as expected
            pass
        msgs, code = _drain_with_code(ws)

    assert code == _WS_UNAUTH, f"expected a 4401 close, got {code!r} ({msgs!r})"
    assert any(m.get("code") == "unauthorized" for m in msgs), msgs


def test_stream_survives_an_unrelated_config_bump(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """The other direction: a settings save bumps the same config version, and
    the still-valid credential must NOT be treated as revoked."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    h, raw_alice = bind_profile(client, make_user_key, "p", {"BEAM_SIZE": 3})

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))
        time.sleep(0.3)
        _profile(client, h, "p", BEAM_SIZE=7)   # unrelated bump, alice untouched
        ws.send_bytes(_pcm(8000, 1500))
        ws.send_bytes(_pcm(0, 1500))            # silence → finalize
        ws.send_json({"type": "stop"})
        msgs, code = _drain_with_code(ws)

    assert not any(m.get("code") == "unauthorized" for m in msgs), msgs
    assert any(m.get("type") == "final" for m in msgs), msgs
    # The refresh still did its real job: the new profile reached the decode.
    assert fake_model.last_kwargs["beam_size"] == 7


def test_stream_does_not_reauthenticate_without_a_version_bump(
        client, make_user_key, app_module, monkeypatch):
    """The re-auth hangs off the SAME version-bump branch as the ident refresh,
    so the steady state costs exactly nothing: one authenticate_ws call, at the
    handshake, no matter how many partial/final decodes run."""
    from faster_whisper_backend.streaming import routes as streaming_routes

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)
    _, raw_alice = make_user_key("alice")

    calls = []
    real = streaming_routes.authenticate_ws
    monkeypatch.setattr(streaming_routes, "authenticate_ws",
                        lambda ws: (calls.append(1), real(ws))[1])

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        msgs, _code = speak_and_stop(ws)

    assert any(m.get("type") == "final" for m in msgs), msgs
    assert len(calls) == 1, f"re-authenticated {len(calls)}x with no version bump"


def test_stream_closes_when_session_cookie_is_revoked_mid_session(
        client, make_user_key, app_module, monkeypatch):
    """The cookie-authenticated dictation page (no bearer) must lose its
    stream on sign-out: sessions_store.revoke_session (/auth/logout) bumps the
    config version, which is the only signal _refresh_ident consumes."""
    from faster_whisper_backend.auth import sessions_store
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)      # lock down, so open mode is off
    _uid, raw_alice = make_user_key("alice")
    assert client.post("/auth/login", json={"key": raw_alice}).status_code == 200
    cookie_name = app_module.cfg.SESSION_COOKIE_NAME
    raw_cookie = client.cookies.get(cookie_name)
    assert raw_cookie

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream",
            headers={"cookie": f"{cookie_name}={raw_cookie}"}) as ws:
        assert handshake(ws)["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))        # speech → partial decodes run
        time.sleep(0.5)
        sessions_store.revoke_session(raw_cookie)   # what /auth/logout does
        ws.send_bytes(_pcm(8000, 1500))        # next decode re-auths → signed out
        time.sleep(0.5)
        ws.send_bytes(_pcm(8000, 200))         # producer notices → closes
        try:
            ws.send_json({"type": "stop"})     # no-op once the socket is gone
        except Exception:  # noqa: BLE001 — already closed, as expected
            pass
        msgs, code = _drain_with_code(ws)

    assert code == _WS_UNAUTH, f"expected a 4401 close, got {code!r} ({msgs!r})"
    assert any(m.get("code") == "unauthorized" for m in msgs), msgs


def test_revocation_during_the_closing_drain_still_closes_with_4401(
        client, make_user_key, app_module, monkeypatch):
    """Revoke AFTER the pump's last decode but BEFORE the client's stop, with
    an utterance still in flight: the drain in session.close() runs
    decode_final → _refresh_ident → _CredentialRevoked outside the pump. It
    must still close with 4401 like every other revocation path, with no
    "internal" error frame, no last final and no error ledger row."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.stats import metrics
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)      # lock down, so open mode is off
    uid, raw_alice = make_user_key("alice")
    rows = []
    monkeypatch.setattr(metrics, "record_transcription",
                        lambda **kw: rows.append(kw.get("status")))

    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))        # speech, utterance stays open
        time.sleep(0.5)                        # pump has drained the queue
        api_keys_store.revoke_user(uid)        # bumps the config version
        ws.send_json({"type": "stop"})         # no further decode until the drain
        msgs, code = _drain_with_code(ws)

    assert code == _WS_UNAUTH, f"expected a 4401 close, got {code!r} ({msgs!r})"
    assert not any(m.get("code") == "internal" for m in msgs), msgs
    assert not any(m.get("type") == "final" and m.get("last") for m in msgs), msgs
    assert "error" not in rows, rows


def _knobs_handshake(client, monkeypatch, overrides, headers=None):
    """Handshake with `overrides` and return (ready frame, seen): seen holds
    the client values _stream_config received ("client"), the config it built
    ("config") and the endpointer threshold ("threshold")."""
    from faster_whisper_backend.streaming import routes
    seen = {}
    real_cfg, real_ep = routes._stream_config, routes.make_endpointer

    def _cfg(cfg_for, ident=None, client=None):
        seen["client"] = dict(client or {})
        seen["config"] = real_cfg(cfg_for, ident, client=client)
        return seen["config"]

    def _ep(*a, **kw):
        seen["threshold"] = kw.get("threshold")
        return real_ep(*a, **kw)
    monkeypatch.setattr(routes, "_stream_config", _cfg)
    monkeypatch.setattr(routes, "make_endpointer", _ep)
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=headers or {}) as ws:
        ready = handshake(ws, decode_overrides=overrides)
        ws.send_json({"type": "stop"})
        _drain(ws)
    return ready, seen


def test_stream_knobs_applied(client, app_module, monkeypatch, caplog):
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_INNER_SILENCE_MS", 700)
    with caplog.at_level(logging.INFO, logger="faster_whisper_backend.streaming.routes"):
        ready, seen = _knobs_handshake(client, monkeypatch, {
            "streaming_vad_threshold": 0.35,
            "streaming_vad_outer_silence_ms": 2500,
            "streaming_hard_break_silence_ms": 0,
            "streaming_hard_break_separator": "\n"})
    assert ready["type"] == "ready" and "overrides_ignored" not in ready
    cfg = seen["config"]
    assert cfg.commit_silence_ms == 2500
    assert cfg.vad_min_silence_ms == 700          # server value, already below outer
    assert cfg.hard_break_silence_ms == 0         # 0 = never
    assert cfg.hard_break_separator == "\n"
    assert seen["threshold"] == 0.35
    assert any("client knobs" in r.getMessage()
               and "vad_outer_silence_ms=2500" in r.getMessage()
               for r in caplog.records)


def test_stream_knobs_clamped_to_field_bounds(client, monkeypatch):
    _, seen = _knobs_handshake(client, monkeypatch, {
        "streaming_vad_threshold": 7,
        "streaming_vad_outer_silence_ms": 99999,
        "streaming_vad_inner_silence_ms": "nope",
        "streaming_hard_break_separator": "x" * 50})
    assert seen["threshold"] == 1.0
    assert seen["config"].commit_silence_ms == 10000
    assert "STREAMING_VAD_INNER_SILENCE_MS" not in seen["client"]  # unparseable → dropped
    assert seen["config"].hard_break_separator == "x" * 8


def test_stream_knobs_inner_kept_below_outer(client, app_module, monkeypatch, caplog):
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_OUTER_SILENCE_MS", 1200)
    with caplog.at_level(logging.INFO, logger="faster_whisper_backend.streaming.routes"):
        _, seen = _knobs_handshake(client, monkeypatch,
                                   {"streaming_vad_inner_silence_ms": 3000})
    assert seen["config"].vad_min_silence_ms == 1150
    assert any("inner 3000→1150" in r.getMessage() for r in caplog.records)
    # A client outer below the server inner moves the inner too.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_INNER_SILENCE_MS", 700)
    _, seen = _knobs_handshake(client, monkeypatch,
                               {"streaming_vad_outer_silence_ms": 400})
    assert seen["config"].vad_min_silence_ms == 350


def test_stream_knobs_hard_break_kept_above_outer(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_HARD_BREAK_SILENCE_MS", 5000)
    _, seen = _knobs_handshake(client, monkeypatch,
                               {"streaming_vad_outer_silence_ms": 6000})
    assert seen["config"].hard_break_silence_ms == 7000
    _, seen = _knobs_handshake(client, monkeypatch,
                               {"streaming_hard_break_silence_ms": 800})
    assert seen["config"].hard_break_silence_ms == (
        app_module.cfg.STREAMING_VAD_OUTER_SILENCE_MS + 1000)
    # A server-only pair is no client value: fitted in _stream_config instead.
    monkeypatch.setattr(app_module.cfg, "STREAMING_HARD_BREAK_SILENCE_MS", 500)
    _, seen = _knobs_handshake(client, monkeypatch, {"beam_size": 3})
    assert seen["client"] == {} and seen["config"].hard_break_silence_ms == (
        app_module.cfg.STREAMING_VAD_OUTER_SILENCE_MS + 1000)


def test_profile_silence_gates_are_fitted_to_the_resolved_outer(monkeypatch):
    """AdminConfig refuses only a GLOBAL silence pair that does not fit: a
    profile outer above the global hard break made every dictated sentence
    its own document, and a profile inner >= outer stretched every commit."""
    from faster_whisper_backend.streaming import routes
    c = effective_config.cfg
    monkeypatch.setattr(c, "STREAMING_VAD_INNER_SILENCE_MS", 700)
    monkeypatch.setattr(c, "STREAMING_VAD_OUTER_SILENCE_MS", 1200)
    monkeypatch.setattr(c, "STREAMING_HARD_BREAK_SILENCE_MS", 5000)

    def sc_for(**values):
        ident = effective_config.Resolved(values=values)
        return routes._stream_config(effective_config.cfg_for, ident)
    sc = sc_for(STREAMING_VAD_OUTER_SILENCE_MS=6000)
    assert (sc.commit_silence_ms, sc.hard_break_silence_ms) == (6000, 7000)
    sc = sc_for(STREAMING_VAD_INNER_SILENCE_MS=1500)
    assert (sc.vad_min_silence_ms, sc.commit_silence_ms) == (1150, 1200)
    # Hard break 0 (off) stays off; a fitting pair passes through untouched.
    sc = sc_for(STREAMING_VAD_OUTER_SILENCE_MS=6000,
                STREAMING_HARD_BREAK_SILENCE_MS=0)
    assert sc.hard_break_silence_ms == 0
    sc = sc_for(STREAMING_VAD_INNER_SILENCE_MS=900,
                STREAMING_HARD_BREAK_SILENCE_MS=3000)
    assert (sc.vad_min_silence_ms, sc.commit_silence_ms,
            sc.hard_break_silence_ms) == (900, 1200, 3000)


def test_profile_trim_keep_at_or_above_trim_is_clamped_below_it(monkeypatch):
    """AdminConfig only validates the GLOBAL keep < trim pair: a profile
    keep above the global trim would stop the buffer trim from ever firing."""
    from faster_whisper_backend.streaming import routes
    monkeypatch.setattr(effective_config.cfg, "STREAMING_BUFFER_TRIM_S", 15.0)
    ident = effective_config.Resolved(
        values={"STREAMING_BUFFER_TRIM_KEEP_S": 20.0})
    sc = routes._stream_config(effective_config.cfg_for, ident)
    assert sc.buffer_trim_sec == 15.0
    assert sc.buffer_trim_keep_sec < sc.buffer_trim_sec
    # A valid pair passes through untouched.
    ident = effective_config.Resolved(
        values={"STREAMING_BUFFER_TRIM_KEEP_S": 8.0})
    sc = routes._stream_config(effective_config.cfg_for, ident)
    assert sc.buffer_trim_keep_sec == 8.0


@pytest.mark.parametrize("raw, want", [("\\n", "\n"), ("\n", "\n"), (" ", " ")])
def test_spelled_newline_separator_is_decoded_from_every_source(monkeypatch, raw, want):
    """The /settings input for the separator is single-line and its help says
    '\\n' = newline, so config.local.json, a profile and a client override all
    hold backslash + n; passed on as is, live dictation typed a literal "\\n"
    at every hard break. A real newline passes unchanged."""
    from faster_whisper_backend.streaming import routes
    field = "STREAMING_HARD_BREAK_SEPARATOR"
    monkeypatch.setattr(effective_config.cfg, field, raw)
    assert routes._stream_config(effective_config.cfg_for).hard_break_separator == want
    monkeypatch.setattr(effective_config.cfg, field, "")
    ident = effective_config.Resolved(values={field: raw})
    sc = routes._stream_config(effective_config.cfg_for, ident)
    assert sc.hard_break_separator == want
    sc = routes._stream_config(effective_config.cfg_for, client={field: raw})
    assert sc.hard_break_separator == want


def test_stream_knobs_locked_are_ignored(client, make_user_key, monkeypatch):
    _, raw_alice = bind_profile(client, make_user_key, "p",
                                {"STREAMING_VAD_OUTER_SILENCE_MS": 1500,
                                 "locks": ["STREAMING_VAD_OUTER_SILENCE_MS"]})
    ready, seen = _knobs_handshake(
        client, monkeypatch,
        {"streaming_vad_outer_silence_ms": 3000, "streaming_vad_threshold": 0.3},
        headers=bearer(raw_alice))
    assert ready["overrides_ignored"] == ["streaming_vad_outer_silence_ms"]
    assert seen["config"].commit_silence_ms == 1500
    assert seen["threshold"] == 0.3


def test_stream_knobs_never_move_a_locked_partner(client, make_user_key, monkeypatch):
    """A client outer that would push a LOCKED inner or hard break is dropped;
    the admin's locked pair stays exactly as set."""
    _, raw_alice = bind_profile(client, make_user_key, "p",
                                {"STREAMING_HARD_BREAK_SILENCE_MS": 5000,
                                 "STREAMING_VAD_INNER_SILENCE_MS": 800,
                                 "locks": ["STREAMING_HARD_BREAK_SILENCE_MS",
                                           "STREAMING_VAD_INNER_SILENCE_MS"]})
    for outer in (6000, 300):
        _, seen = _knobs_handshake(
            client, monkeypatch, {"streaming_vad_outer_silence_ms": outer},
            headers=bearer(raw_alice))
        assert seen["client"] == {}
        assert seen["config"].hard_break_silence_ms == 5000
        assert seen["config"].vad_min_silence_ms == 800
    # A compatible client outer still applies.
    _, seen = _knobs_handshake(
        client, monkeypatch, {"streaming_vad_outer_silence_ms": 2000},
        headers=bearer(raw_alice))
    assert seen["config"].commit_silence_ms == 2000
    assert seen["config"].hard_break_silence_ms == 5000


def test_stream_knobs_master_gate_off(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "ALLOW_REQUEST_DECODE_OVERRIDES", False)
    ready, seen = _knobs_handshake(client, monkeypatch,
                                   {"streaming_vad_outer_silence_ms": 3000})
    assert ready["overrides_ignored"] == ["streaming_vad_outer_silence_ms"]
    assert seen["client"] == {}


def test_stream_client_output_wrappers_survive_a_refresh(
        client, make_user_key, fake_model, app_module, monkeypatch):
    # _refresh_ident recomputes the wrappers; it must keep the client's.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "OUTPUT_PREFIX", "[cfg] ")
    _, raw_admin = make_user_key("admin", is_admin=True)
    uid, raw_alice = make_user_key("alice")
    from faster_whisper_backend.auth import api_keys_store
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws, decode_overrides={"output_prefix": ">> ",
                                               "output_suffix": " <<"})["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))
        # Any config bump makes the next utterance re-resolve ident.
        api_keys_store.set_user_permissions(
            uid, {"pages": {}, "config": {"overrides": {}, "profiles": [], "locks": []}})
        msgs = speak_and_stop(ws)[0]
    finals = [m for m in msgs if m.get("type") == "final"]
    assert len(finals) >= 2
    assert all((f.get("committed") or f.get("tail") or "").startswith(">> ")
               for f in finals if f.get("committed") or f.get("tail"))
    assert finals[-1]["last"] and finals[-1]["committed"].endswith(" <<")


def test_stream_output_wrappers_change_only_at_a_document_boundary(
        client, make_user_key, fake_model, app_module, monkeypatch):
    # Within one document every final extends the previous one (clients may
    # type just the difference), so a mid-document config bump must not swap
    # the prefix of the open document: the new one applies after the boundary.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_HARD_BREAK_SILENCE_MS", 2000)
    monkeypatch.setattr(app_module.cfg, "OUTPUT_PREFIX", "[old] ")
    uid, raw_alice = make_user_key("alice")
    from faster_whisper_backend.auth import api_keys_store
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws)["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))        # final, document stays open
        early = []
        while True:
            m = ws.receive_json()
            early.append(m)
            if m.get("type") == "final":
                break
        monkeypatch.setattr(app_module.cfg, "OUTPUT_PREFIX", "[new] ")
        api_keys_store.set_user_permissions(
            uid, {"pages": {}, "config": {"overrides": {}, "profiles": [], "locks": []}})
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 3000))        # final, then the hard break
        msgs = early + speak_and_stop(ws)[0]
    boundaries = [i for i, m in enumerate(msgs) if m.get("type") == "boundary"]
    assert len(boundaries) == 1
    texts = [(m.get("committed") or "") + (m.get("tail") or "")
             for m in msgs[:boundaries[0]] if m.get("type") == "final"]
    texts = [t for t in texts if t]
    assert len(texts) >= 2
    assert texts[0].startswith("[old] ")
    for prev, cur in zip(texts, texts[1:]):
        assert cur.startswith(prev), (prev, cur)
    after = [(m.get("committed") or "") + (m.get("tail") or "")
             for m in msgs[boundaries[0] + 1:] if m.get("type") == "final"]
    after = [t for t in after if t]
    assert after and all(t.startswith("[new] ") for t in after)


def test_stream_output_suffix_closes_every_document_at_a_hard_break(
        client, make_user_key, fake_model, app_module, monkeypatch):
    # A hard break starts a fresh document that gets the prefix again, so the
    # document before it must be closed too: the suffix goes in front of the
    # boundary's separator (typed verbatim), not only on the closing flush.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_HARD_BREAK_SILENCE_MS", 2000)
    monkeypatch.setattr(app_module.cfg, "STREAMING_HARD_BREAK_SEPARATOR", "\n")
    _, raw_alice = make_user_key("alice")
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw_alice)) as ws:
        assert handshake(ws, decode_overrides={"output_prefix": ">> ",
                                               "output_suffix": " <<"})["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 3000))        # final, then the hard break
        msgs = speak_and_stop(ws)[0]      # final, no second break
    boundaries = [i for i, m in enumerate(msgs) if m.get("type") == "boundary"]
    assert len(boundaries) == 1
    assert msgs[boundaries[0]]["separator"] == " <<\n"
    # The suffix also rides on its own, so a display client can tell it from
    # the separator proper (/dictate shows an empty one as a line break).
    assert msgs[boundaries[0]]["suffix"] == " <<"
    finals = [m for m in msgs if m.get("type") == "final"]
    after = [m for m in msgs[boundaries[0] + 1:] if m.get("type") == "final"]
    assert after and len(after) < len(finals)
    # Both documents open with the prefix; only the closing flush ends with
    # the suffix, the first document's is on the boundary.
    assert (after[0].get("committed") or after[0].get("tail")).startswith(">> ")
    assert (finals[0].get("committed") or finals[0].get("tail")).startswith(">> ")
    assert [f for f in finals if (f.get("committed") or "").endswith(" <<")] == [finals[-1]]
    assert finals[-1]["last"]


def test_stream_client_output_wrappers_master_gate_off(
        client, fake_model, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "OUTPUT_PREFIX", "[cfg] ")
    monkeypatch.setattr(app_module.cfg, "ALLOW_REQUEST_DECODE_OVERRIDES", False)
    with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
        ready = handshake(ws, decode_overrides={"output_prefix": ">> "})
        assert ready["overrides_ignored"] == ["output_prefix"]
        msgs = speak_and_stop(ws)[0]
    final = next(m for m in msgs if m.get("type") == "final" and m.get("committed"))
    assert final["committed"].startswith("[cfg] ")
