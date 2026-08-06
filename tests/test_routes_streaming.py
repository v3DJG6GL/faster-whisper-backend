"""In-process TestClient tests for the streaming WebSocket endpoint.

Runs against the real FastAPI app with the conftest fake model (so no
faster-whisper / GPU needed), exercising the config handshake, the partial/final
message contract, the stop/close drain, and the handshake's credential carriers.
Open mode (no API key) → the synthetic admin passes auth, from the admin host
allowlist only (loopback here).
"""

import time

import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from conftest import bearer


def _pcm(level, ms, sr=16000):
    return np.full(sr * ms // 1000, level, dtype="<i2").tobytes()


def _drain(ws, limit=200):
    msgs = []
    try:
        for _ in range(limit):
            msgs.append(ws.receive_json())
    except WebSocketDisconnect:
        pass
    return msgs


def test_stream_happy_path_partials_then_final(app_module, monkeypatch):
    # Force the energy gate: the synthetic constant-amplitude PCM below is "loud"
    # but not speech, so the real Silero VAD would (correctly) reject it. This test
    # exercises the routing/protocol/session flow, not the VAD model.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["sample_rate"] == 16000

            ws.send_bytes(_pcm(8000, 2500))   # ~2.5 s speech → ≥2 partials
            ws.send_bytes(_pcm(0, 1500))      # ~1.5 s silence → finalize (held, no terminator)
            ws.send_json({"type": "stop"})    # → close flushes the held final
            msgs = _drain(ws)

    partials = [m for m in msgs if m["type"] == "partial"]
    finals = [m for m in msgs if m["type"] == "final"]
    assert partials, "expected at least one partial"
    # LocalAgreement-2 commits the repeated 'hallo welt' hypothesis.
    assert any("welt" in m["committed"] for m in partials)
    assert finals, "expected a final after stop/close"
    assert "welt" in "".join(m["committed"] + m.get("tail", "") for m in finals)
    assert finals[-1].get("last") is True


def test_stream_rejects_unsupported_audio_format(app_module):
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1",
                "audio": {"format": "g729-telephony"},  # not raw, not an ffmpeg format we allow
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["code"] == "unsupported_format"


def test_dictate_page_served(app_module):
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        r = client.get("/dictate")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "/v1/audio/transcriptions/stream" in body
    assert "AudioWorkletNode" in body
    # batch mode: a mode selector + MediaRecorder POST to the file endpoint.
    assert 'id="mode"' in body
    assert "MediaRecorder" in body
    assert "startBatch" in body


def test_dictate_page_uses_the_shared_shell(app_module):
    """It renders through render_page like every other WebUI page: no
    unsubstituted placeholders, and the shared chrome is present."""
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        body = client.get("/dictate").text
    assert "{{" not in body                       # every placeholder resolved
    assert "faster-whisper-backend · dictate" in body   # {{HEADER_TITLE}}
    assert 'class="navrow"' in body                     # {{NAV}}
    assert 'href="/dictate"' in body                    # own nav entry
    assert "login-gate" in body                         # shared auth screen


def test_dictate_page_carries_no_credential_field(app_module):
    """The page authenticates with the session cookie the rest of the WebUI
    uses. It must not reintroduce an API-key input, a key-bearing subprotocol,
    or the /auth/login exchange that used to mint a 30-day origin-wide session
    from a chrome-less page with no way to sign out."""
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        body = client.get("/dictate").text
    # NB: the shared login gate legitimately carries its own key input, so only
    # the page's OWN field and credential plumbing are asserted absent.
    assert 'id="key"' not in body
    assert "bearer." not in body
    assert "wsAuth" not in body


# --- handshake credential carriers -------------------------------------------
# Three: the Authorization header (native clients), the `bearer.<key>`
# subprotocol (the one request header a browser can set on a WebSocket), and
# the session cookie. A ?key= query param is NOT one — it would land in every
# access log that records the request line.

def _handshake(ws):
    ws.send_json({"type": "config", "model": "whisper-1",
                  "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
    return ws.receive_json()


def test_ws_authorization_header_admits(client, make_user_key):
    make_user_key("root", is_admin=True)   # locks the server down
    _uid, raw = make_user_key("alice")
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=bearer(raw)) as ws:
        assert _handshake(ws)["type"] == "ready"


def test_ws_bearer_subprotocol_admits(client, make_user_key):
    from streaming_routes import _WS_BEARER_SUBPROTOCOL
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice")
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream",
            subprotocols=[_WS_BEARER_SUBPROTOCOL + raw]) as ws:
        assert _handshake(ws)["type"] == "ready"


def test_ws_key_query_param_rejected(client, make_user_key):
    from streaming_routes import _WS_UNAUTH
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice")
    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect(
                f"/v1/audio/transcriptions/stream?key={raw}") as ws:
            ws.receive_json()
    assert ei.value.code == _WS_UNAUTH


def test_ws_open_mode_remote_rejected(app_module):
    # /v1 has no host allowlist, but the open-mode synthetic admin is confined
    # to ADMIN_WEBUI_ALLOWED_HOSTS — so a remote peer on an unbootstrapped
    # server gets the unauth close, not a free session.
    from streaming_routes import _WS_UNAUTH
    with TestClient(app_module.app, client=("203.0.113.9", 1234)) as c:
        with pytest.raises(WebSocketDisconnect) as ei:
            with c.websocket_connect("/v1/audio/transcriptions/stream") as ws:
                ws.receive_json()
    assert ei.value.code == _WS_UNAUTH


def test_stream_disabled_closes_connection(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_ENABLED", False, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        try:
            with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
                # If it doesn't reject pre-accept, it must close immediately.
                with __import__("pytest").raises(WebSocketDisconnect):
                    ws.receive_json()
        except WebSocketDisconnect:
            pass  # rejected during handshake — also acceptable


def test_stream_records_trace_text_per_utterance(app_module, monkeypatch):
    """Each finalized utterance writes a recent-transcriptions row with non-empty
    raw/final text (drives /quick-config + /reports), not just numeric metrics."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    import transcriptions_store
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        before = transcriptions_store.count()
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({"type": "config", "model": "whisper-1",
                          "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_pcm(8000, 2500))   # speech
            ws.send_bytes(_pcm(0, 1500))      # silence → finalize
            ws.send_json({"type": "stop"})
            _drain(ws)
        rows = transcriptions_store.list_recent(limit=10)
        after = transcriptions_store.count()
    assert after > before, "no recent-transcription row recorded for the utterance"
    assert any((r.get("raw") or "").strip() and (r.get("final") or "").strip() for r in rows), \
        "recorded trace rows have empty raw/final (the /quick-config bug)"
    # streamed utterances are tagged source='stream' so /quick-config can chip them.
    assert any(r.get("source") == "stream" for r in rows), \
        "streamed trace not tagged source='stream'"


def test_safe_ws_send_swallows_dead_socket():
    """A page reload mid-dictation closes the socket; the session-close drain then
    sends a final to a dead socket. uvicorn raises RuntimeError('Unexpected ASGI
    message ... after ... close'); the send must be swallowed, not surface as an
    error traceback."""
    import asyncio
    import streaming_routes

    class DeadWS:
        async def send_json(self, _m):
            raise RuntimeError(
                "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'")

    class LiveWS:
        def __init__(self):
            self.sent = []

        async def send_json(self, m):
            self.sent.append(m)

    assert asyncio.run(streaming_routes._safe_ws_send(DeadWS(), {"type": "final"})) is False
    live = LiveWS()
    assert asyncio.run(streaming_routes._safe_ws_send(live, {"type": "final"})) is True
    assert live.sent == [{"type": "final"}]


def test_stream_handshake_idle_timeout_frees_slot(app_module, monkeypatch):
    # A client that connects + passes auth but never sends its config handshake
    # must not hold a session slot forever: the server abandons the wait after
    # STREAMING_IDLE_TIMEOUT_SEC and closes with the idle close code (4408).
    import pytest
    from streaming_routes import _WS_IDLE_TIMEOUT
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_SEC", 0.3, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            with pytest.raises(WebSocketDisconnect) as ei:
                ws.receive_json()   # send nothing → idle close
    assert ei.value.code == _WS_IDLE_TIMEOUT


def test_stream_session_idle_timeout_closes_and_notifies(app_module, monkeypatch):
    # After a successful handshake, a connection that goes silent mid-session is
    # closed once the idle timeout elapses, with an idle_timeout notice first.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_SEC", 0.3, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            assert ws.receive_json()["type"] == "ready"
            msgs = _drain(ws)   # send nothing further → idle close
    assert any(m.get("code") == "idle_timeout" for m in msgs)


def test_stream_idle_timeout_zero_disables(app_module, monkeypatch):
    # 0 disables the idle timeout: the normal stop/close flow still works and no
    # idle_timeout notice is emitted (the receive falls through to a plain await).
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_SEC", 0.0, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "stop"})
            msgs = _drain(ws)
    assert not any(m.get("code") == "idle_timeout" for m in msgs)


def test_stream_idle_timeout_not_re_armed_by_non_audio_frames(app_module, monkeypatch):
    """The idle deadline is anchored to the last AUDIO byte, not the last frame.

    Regression: `_receive_idle` used to be re-armed with the FULL timeout on every
    inbound frame, so a client alternating an unrecognised control JSON with an
    empty binary frame (a no-op sink write) held its session slot open forever
    while decoding nothing — enough sockets to pin STREAMING_MAX_SESSIONS.

    The assertion is on the DRAIN duration, not on the notice alone: the old code
    also emits an idle_timeout notice, just one full timeout AFTER the last frame.
    Here the socket must already be closed by the time we stop sending.
    """
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_SEC", 0.5, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            assert ws.receive_json()["type"] == "ready"
            for _ in range(3):            # 0.9 s of keepalive-ish traffic, no audio
                time.sleep(0.3)
                try:
                    ws.send_json({"type": "noop"})   # parses, matches no branch
                    ws.send_bytes(b"")               # empty audio payload
                except Exception:                    # noqa: BLE001 — already closed
                    break
            t0 = time.monotonic()
            msgs = _drain(ws)
            drained_in = time.monotonic() - t0
    assert any(m.get("code") == "idle_timeout" for m in msgs)
    # Closed at ~0.5 s (during the send loop), not 0.5 s after the last frame.
    assert drained_in < 0.25, f"socket was still armed when we stopped sending ({drained_in:.2f}s)"


def test_stream_idle_timeout_not_tripped_while_audio_flows(app_module, monkeypatch):
    """The other side of the same fix: real audio DOES re-arm the deadline, so a
    live mic (which streams PCM continuously, even through silence) is never cut
    short even though the session runs far longer than one idle timeout."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_SEC", 0.5, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            assert ws.receive_json()["type"] == "ready"
            for _ in range(5):            # 1.25 s of audio: 2.5x the idle timeout
                time.sleep(0.25)
                ws.send_bytes(_pcm(0, 100))   # silent PCM is still AUDIO
            ws.send_json({"type": "stop"})
            msgs = _drain(ws)
    assert not any(m.get("code") == "idle_timeout" for m in msgs)
