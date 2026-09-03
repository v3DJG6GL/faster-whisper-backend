"""In-process TestClient tests for the streaming WebSocket endpoint.

Runs against the real FastAPI app with the conftest fake model (so no
faster-whisper / GPU needed), exercising the config handshake, the partial/final
message contract, the stop/close drain, and the handshake's credential carriers.
Open mode (no API key) → the synthetic admin passes auth, from the admin host
allowlist only (loopback here).
"""

import time

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests._streaming_helpers import const_pcm, ws_drain
from tests.conftest import bearer

_pcm = const_pcm


def _drain(ws, limit=200):
    return ws_drain(ws, limit)[0]


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


def test_dictate_page_explains_streaming_disabled_close(app_module):
    """The WS handler refuses with close code 4503 when STREAMING_ENABLED is
    off; the page's onclose must name that refusal (next to its 4401/4403
    branches) instead of falling through to a bare "closed"."""
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        body = client.get("/dictate").text
    for code in ("4401", "4403", "4503"):
        assert code in body, code
    assert "live dictation is disabled on this server" in body


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
    from faster_whisper_backend.streaming.routes import _WS_BEARER_SUBPROTOCOL
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice")
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream",
            subprotocols=[_WS_BEARER_SUBPROTOCOL + raw]) as ws:
        assert _handshake(ws)["type"] == "ready"


def test_ws_key_query_param_rejected(client, make_user_key):
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH
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
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH
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
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()
        except WebSocketDisconnect:
            pass  # rejected during handshake — also acceptable


def test_stream_records_trace_text_per_utterance(app_module, monkeypatch):
    """Each finalized utterance writes a recent-transcriptions row with non-empty
    raw/final text (drives /quick-config + /reports), not just numeric metrics."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    from faster_whisper_backend.stats import recent_transcriptions_store
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        before = recent_transcriptions_store.count()
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({"type": "config", "model": "whisper-1",
                          "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_pcm(8000, 2500))   # speech
            ws.send_bytes(_pcm(0, 1500))      # silence → finalize
            ws.send_json({"type": "stop"})
            _drain(ws)
        rows = recent_transcriptions_store.list_recent(limit=10)
        after = recent_transcriptions_store.count()
    assert after > before, "no recent-transcription row recorded for the utterance"
    assert any((r.get("raw") or "").strip() and (r.get("final") or "").strip() for r in rows), \
        "recorded trace rows have empty raw/final (the /quick-config bug)"
    # streamed utterances are tagged source='stream' so /quick-config can chip them.
    assert any(r.get("source") == "stream" for r in rows), \
        "streamed trace not tagged source='stream'"


def test_stream_records_the_key_label_on_the_trace_row(
        client, app_module, make_user_key, monkeypatch):
    """The recent-transcriptions row snapshots the API key's label straight
    from the auth record (user["key_label"]), the same way /transcribe does."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.stats import metrics
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("root", is_admin=True)   # locks the server down
    uid = api_keys_store.create_user("alice", is_admin=False)
    raw, _rec = api_keys_store.create_key(uid, label="alice-laptop")
    with client.websocket_connect(_STREAM_URL, headers=bearer(raw)) as ws:
        assert _handshake(ws)["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))   # speech
        ws.send_bytes(_pcm(0, 1500))      # silence → finalize
        ws.send_json({"type": "stop"})
        _drain(ws)
    rows = metrics.metrics_snapshot(include_identity=True)["recent_transcriptions"]
    assert rows, "no recent-transcription row recorded for the utterance"
    assert rows[0]["key_label"] == "alice-laptop"


def test_flush_after_new_audio_is_not_coalesced(app_module, monkeypatch):
    """Flush coalescing must only collapse flushes with NO audio in between:
    `flush A → audio → flush B` has to deliver BOTH flushes to the pump (new
    audio makes the queued flush no longer authoritative), instead of silently
    degrading to flush A and leaving utterance B waiting for the silence gate."""
    import asyncio

    from faster_whisper_backend.streaming import session as streaming_session

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    flushes = []
    orig_flush = streaming_session.StreamSession.flush_utterance

    async def slow_flush(self):
        # Hold the pump on flush A so audio + flush B arrive while it is still
        # in flight — the exact window where the old coalescing dropped B.
        flushes.append(time.monotonic())
        await asyncio.sleep(0.4)
        await orig_flush(self)

    monkeypatch.setattr(streaming_session.StreamSession, "flush_utterance", slow_flush)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({"type": "config", "model": "whisper-1",
                          "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_pcm(8000, 1000))   # utterance A
            ws.send_json({"type": "flush"})   # flush A (pump held for 0.4 s)
            ws.send_bytes(_pcm(8000, 1000))   # utterance B — new audio after A's flush
            ws.send_json({"type": "flush"})   # flush B: must NOT coalesce into A
            ws.send_json({"type": "stop"})
            _drain(ws)
    assert len(flushes) == 2, (
        f"flush after new audio was coalesced away ({len(flushes)} reached the pump)")


def test_safe_ws_send_swallows_dead_socket():
    """A page reload mid-dictation closes the socket; the session-close drain then
    sends a final to a dead socket. uvicorn raises RuntimeError('Unexpected ASGI
    message ... after ... close'); the send must be swallowed, not surface as an
    error traceback."""
    import asyncio
    from faster_whisper_backend.streaming import routes as streaming_routes

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
    # STREAMING_IDLE_TIMEOUT_S and closes with the idle close code (4408).
    from faster_whisper_backend.streaming.routes import _WS_IDLE_TIMEOUT
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 0.3, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            with pytest.raises(WebSocketDisconnect) as ei:
                ws.receive_json()   # send nothing → idle close
    assert ei.value.code == _WS_IDLE_TIMEOUT


def test_stream_session_idle_timeout_closes_and_notifies(app_module, monkeypatch):
    # After a successful handshake, a connection that goes silent mid-session is
    # closed once the idle timeout elapses, with an idle_timeout notice first.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 0.3, raising=False)
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
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 0.0, raising=False)
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
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 0.5, raising=False)
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
    # 2 s timeout with a frame every 0.25 s: 1.75 s of slack per frame, so a
    # loaded CI runner (three interpreters share one box) cannot turn a stalled
    # sleep into a spurious idle close — 0.5 s / 0.25 s flaked on py3.13 in CI.
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 2.0, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            assert ws.receive_json()["type"] == "ready"
            for _ in range(12):           # 3 s of audio: 1.5x the idle timeout
                time.sleep(0.25)
                ws.send_bytes(_pcm(0, 100))   # silent PCM is still AUDIO
            ws.send_json({"type": "stop"})
            msgs = _drain(ws)
    assert not any(m.get("code") == "idle_timeout" for m in msgs)


# --- per-user session cap ----------------------------------------------------

_STREAM_URL = "/v1/audio/transcriptions/stream"


def test_per_user_cap_refuses_while_the_global_cap_has_headroom(
        client, app_module, monkeypatch, caplog):
    """One client must not be able to fill the pool. The global cap is left
    wide open, so a refusal here can only come from the per-user one."""
    import logging
    from faster_whisper_backend.streaming import routes as streaming_routes

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS", 10,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 1,
                        raising=False)
    with caplog.at_level(logging.INFO, logger=streaming_routes.logger.name):
        with client.websocket_connect(_STREAM_URL) as ws:
            assert _handshake(ws)["type"] == "ready"
            assert len(streaming_routes._active_sessions) == 1
            with pytest.raises(WebSocketDisconnect) as ei:
                with client.websocket_connect(_STREAM_URL) as ws2:
                    ws2.receive_json()
            assert ei.value.code == streaming_routes._WS_TOO_MANY
    assert any("STREAMING_MAX_SESSIONS_PER_USER" in r.getMessage()
               for r in caplog.records)


def test_per_user_cap_zero_is_unlimited(client, app_module, monkeypatch):
    from faster_whisper_backend.streaming import routes as streaming_routes

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 0,
                        raising=False)
    with client.websocket_connect(_STREAM_URL) as a, \
            client.websocket_connect(_STREAM_URL) as b, \
            client.websocket_connect(_STREAM_URL) as c:
        for ws in (a, b, c):
            assert _handshake(ws)["type"] == "ready"
        assert len(streaming_routes._active_sessions) == 3
        # limit<=0 short-circuits in InFlight.acquire, so no key is ever booked
        assert streaming_routes._stream_sessions._counts == {}


def test_serial_sessions_never_leak_a_slot(client, app_module, monkeypatch):
    """The leak guard: open and close cap+2 sessions one at a time. Each must
    succeed, and the gauge must be empty afterwards — a release skipped on any
    teardown path would strand a slot and refuse the next connection."""
    from faster_whisper_backend.streaming import routes as streaming_routes

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 2,
                        raising=False)
    for _ in range(4):
        with client.websocket_connect(_STREAM_URL) as ws:
            assert _handshake(ws)["type"] == "ready"
    assert streaming_routes._stream_sessions._counts == {}


def test_per_user_cap_is_per_user(client, make_user_key, app_module,
                                  monkeypatch):
    """Open mode resolves every caller to one synthetic admin, so real keys
    are what prove the cap is keyed per identity."""
    from faster_whisper_backend.streaming import routes as streaming_routes

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 1,
                        raising=False)
    make_user_key("root", is_admin=True)          # locks the server down
    _uid_a, key_a = make_user_key("alice")
    _uid_b, key_b = make_user_key("bob")

    with client.websocket_connect(_STREAM_URL, headers=bearer(key_a)) as a:
        assert _handshake(a)["type"] == "ready"
        with pytest.raises(WebSocketDisconnect) as ei:
            with client.websocket_connect(_STREAM_URL,
                                          headers=bearer(key_a)) as dup:
                dup.receive_json()
        assert ei.value.code == streaming_routes._WS_TOO_MANY
        # bob's own slot is untouched by alice holding hers.
        with client.websocket_connect(_STREAM_URL, headers=bearer(key_b)) as b:
            assert _handshake(b)["type"] == "ready"


def test_aborted_handshake_releases_the_per_user_slot(client, app_module,
                                                       monkeypatch):
    """ws.accept() raises when the client is already gone. The per-user slot
    is taken BEFORE accept and the releasing finally only starts after it, so
    every aborted handshake used to burn one slot for the life of the process."""
    import starlette.websockets
    from faster_whisper_backend.streaming import routes as streaming_routes

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 1,
                        raising=False)

    real_accept = starlette.websockets.WebSocket.accept

    async def _gone(self, *args, **kwargs):
        raise RuntimeError("client went away during the handshake")
    monkeypatch.setattr(starlette.websockets.WebSocket, "accept", _gone)
    with pytest.raises(Exception):
        with client.websocket_connect(_STREAM_URL) as ws:
            ws.receive_json()
    assert streaming_routes._stream_sessions._counts == {}

    # The slot is free again: the very next handshake from the same identity
    # goes through instead of being refused with 4429.
    monkeypatch.setattr(starlette.websockets.WebSocket, "accept", real_accept)
    with client.websocket_connect(_STREAM_URL) as ws:
        assert _handshake(ws)["type"] == "ready"


def test_cancel_during_a_completed_load_still_releases_the_lease(
        client, app_module, monkeypatch):
    """A cancellation delivered while the keepalive loop waits on a load that
    has ALREADY returned: the lease was taken, the record of it never made.
    The teardown must still find and release it."""
    import asyncio
    from faster_whisper_backend.streaming import routes as streaming_routes

    _real = app_module._get_or_load_model

    async def _loader(name, *, lease=False):
        model = await _real(name, lease=False)
        if lease:
            app_module._model_leases[name] = \
                app_module._model_leases.get(name, 0) + 1
        return model
    monkeypatch.setattr(app_module, "_get_or_load_model", _loader)

    real_wait = asyncio.wait

    async def _wait_then_cancel(fs, *args, **kwargs):
        done, pending = await real_wait(fs, *args, **kwargs)
        if done and all(t.done() for t in fs):
            raise asyncio.CancelledError()
        return done, pending
    monkeypatch.setattr(streaming_routes.asyncio, "wait", _wait_then_cancel)

    with pytest.raises(BaseException):
        with client.websocket_connect(_STREAM_URL) as ws:
            _handshake(ws)
            ws.receive_json()
    assert app_module._model_leases == {}
    assert streaming_routes._active_sessions == set()


# --- usage job id ------------------------------------------------------------

def _dictate_one_utterance(client, conf_extra):
    with client.websocket_connect(_STREAM_URL) as ws:
        ws.send_json({"type": "config", "model": "whisper-1",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000},
                      **conf_extra})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))   # speech
        ws.send_bytes(_pcm(0, 1500))      # silence → finalize
        ws.send_json({"type": "stop"})
        _drain(ws)


def test_stream_client_job_names_the_usage_job(app_module, monkeypatch):
    """The handshake's `client_job` becomes the usage job id, so the outcome
    the client posts afterwards lands on the session's own utterances."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    from faster_whisper_backend.stats import usage_store
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        _dictate_one_utterance(client, {"client_job": "ab" * 16})
        rows = usage_store._require_conn().execute(
            "SELECT job_id, kind, utterances FROM usage_jobs").fetchall()
    assert [tuple(r) for r in rows] == [("ab" * 16, "dictation", 1)]


def test_stream_malformed_client_job_falls_back_to_session_id(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    from faster_whisper_backend.stats import usage_store
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        _dictate_one_utterance(client, {"client_job": "../not-hex"})
        rows = usage_store._require_conn().execute(
            "SELECT job_id, kind FROM usage_jobs").fetchall()
    assert len(rows) == 1 and rows[0]["kind"] == "dictation"
    assert len(rows[0]["job_id"]) == 32 and rows[0]["job_id"] != "../not-hex"
