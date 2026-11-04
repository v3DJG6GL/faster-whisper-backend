"""In-process TestClient tests for the streaming WebSocket endpoint.

Runs against the real FastAPI app with the conftest fake model (so no
faster-whisper / GPU needed), exercising the config handshake, the partial/final
message contract, the stop/close drain, and the handshake's credential carriers.
Open mode (no API key) → the synthetic admin passes auth, from the admin host
allowlist only (loopback here).
"""

import json
import shutil
import subprocess
import time

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests._streaming_helpers import (
    const_pcm, dictate_session, drain_frames as _drain, handshake as _handshake,
    locked_down_user, ready_ws, speak_and_stop, stop_and_drain, stream_ws)
from tests._transcription_helpers import guard_line
from faster_whisper_backend.streaming import routes as streaming_routes
from faster_whisper_backend.transcription import models as tx_models

_pcm = const_pcm


def test_stream_happy_path_partials_then_final(client, energy_vad):
    with stream_ws(client) as ws:
        ready = _handshake(ws, response_format="json")
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == 16000

        # ~2.5 s speech → ≥2 partials; the silence finalizes (held, no
        # terminator) and the stop's close flushes the held final.
        msgs = speak_and_stop(ws)[0]

    partials = [m for m in msgs if m["type"] == "partial"]
    finals = [m for m in msgs if m["type"] == "final"]
    assert partials, "expected at least one partial"
    # LocalAgreement-2 commits the repeated 'hallo welt' hypothesis.
    assert any("welt" in m["committed"] for m in partials)
    assert finals, "expected a final after stop/close"
    assert "welt" in "".join(m["committed"] + m.get("tail", "") for m in finals)
    assert finals[-1].get("last") is True


def test_stream_announces_the_utterance_lifecycle(client, energy_vad):
    """Over the real socket: ``ready`` advertises the frames, and the one
    utterance is announced, marked as decoding, and closed by its final — in
    that order, all on one ordinal. A client paints "the server is working"
    between ``open`` and the terminal frame, so the order IS the contract."""
    with stream_ws(client) as ws:
        ready = _handshake(ws, response_format="json")
        assert ready["type"] == "ready"
        assert ready["utterance_frames"] is True

        msgs = speak_and_stop(ws)[0]

    seen = []
    for m in msgs:
        if m["type"] == "utterance":
            seen.append((m["state"], m["utterance"]))
        elif m["type"] == "final" and not m.get("last"):
            seen.append(("final", m["utterance"]))
    assert seen == [("open", 0), ("decoding", 0), ("final", 0)], msgs


def test_dictate_page_shows_the_decode_as_transcribing(client):
    """The status follows the utterance frames ("decoding" shows
    "transcribing…"), but only while running, so the post-Stop drain's frames
    can't overwrite "finishing…". A dropped utterance gets no final, so the
    frame itself clears the faded partial (before the guard: harmless after
    Stop)."""
    body = client.get("/dictate").text
    block = body.split('m.type === "utterance"', 1)[1].split("else if (m.type", 1)[0]
    guard = block.index('if (!running || statusEl.className === "error") return;')
    decoding = block.index('m.state === "decoding") setStatus("transcribing…", "live")')
    clear = block.index('if (m.state === "dropped") renderPartial("", "");')
    assert clear < guard < decoding


def test_stream_rejects_unsupported_audio_format(client):
    with stream_ws(client) as ws:
        ws.send_json({
            "type": "config", "model": "whisper-1",
            "audio": {"format": "g729-telephony"},  # not raw, not an ffmpeg format we allow
        })
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "unsupported_format"


@pytest.mark.parametrize("fmt", [["x"], {"a": 1}])
def test_stream_rejects_a_non_string_audio_format(client, monkeypatch, caplog, fmt):
    """An unhashable audio.format used to raise TypeError at the set-membership
    test and reach the handler's blanket except: a full traceback, an error
    usage row, and a generic error frame instead of unsupported_format."""
    import logging

    from faster_whisper_backend.stats import metrics
    rows = []
    monkeypatch.setattr(metrics, "record_transcription", lambda **kw: rows.append(kw))
    caplog.set_level(logging.ERROR, logger="faster_whisper_backend.streaming.routes")
    with stream_ws(client) as ws:
        ws.send_json({"type": "config", "model": ["whisper-1"],
                      "audio": {"format": fmt}})
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["code"] == "unsupported_format"
    assert not [r for r in caplog.records if r.getMessage().startswith("[stream")]
    assert not [r for r in rows if r.get("status") == "error"]


def test_stream_treats_a_non_string_model_as_absent(client):
    """A non-string handshake `model` falls back to whisper-1 instead of riding
    into the job row and the model load."""
    with stream_ws(client) as ws:
        ws.send_json({"type": "config", "model": ["x"],
                      "audio": {"format": "pcm_s16le"}})
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["model"] == tx_models._resolve_model_name("whisper-1")
        stop_and_drain(ws)


def test_dictate_page_served(client):
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


def test_dictate_page_explains_streaming_disabled_close(client):
    """Each refusal close the WS handler sends (4401/4403/4503/4429/4408 —
    4503 is STREAMING_ENABLED off) gets its own message in the page's
    onclose instead of falling through to a bare "closed"."""
    body = client.get("/dictate").text
    for code in ("4401", "4403", "4503", "4429", "4408"):
        assert code in body, code
    assert "live dictation is disabled on this server" in body
    assert "too many live sessions" in body
    assert "no audio for a while" in body


def test_dictate_page_keeps_the_socket_open_until_closing(client):
    """The server emits the held final and the last document after
    {"type":"stop"} and closes the socket itself after {"type":"closing"}, so
    stop() must not close the socket / detach onmessage synchronously: it
    waits for "closing" with a fallback timer."""
    body = client.get("/dictate").text
    stop_body = body.split("function stop() {", 1)[1].split("\n  }\n", 1)[0]
    assert 'ws.send(JSON.stringify({ type: "stop" }))' in stop_body
    assert 'setStatus("finishing…")' in stop_body
    assert "STOP_WAIT_MS)" in stop_body and "armStopTimer(" in stop_body
    # finishStop() runs synchronously only in the two fallbacks: the send
    # failed, or there is no open socket to wait on.
    assert stop_body.count("finishStop()") == 2
    assert "catch { finishStop(); return; }" in stop_body
    assert "} else {\n      finishStop();" in stop_body
    # A server-initiated close (idle timeout) also releases the microphone.
    closing = body.split('m.type === "closing"', 1)[1].split("}", 1)[0]
    assert "releaseMic()" in closing


def test_dictate_page_boundary_strips_the_suffix_before_the_line_break(client):
    """The server puts a configured output suffix in front of the boundary's
    separator (and repeats it as m.suffix). With the default empty separator
    the preview must still show a line break between documents, so the
    fallback applies to the separator minus the suffix — not to the raw
    separator, which the suffix alone made truthy (">> A <<>> B")."""
    body = client.get("/dictate").text
    block = body.split('m.type === "boundary"', 1)[1].split("else if (m.type", 1)[0]
    assert 'const suf = m.suffix || "";' in block
    assert 'const sep = (m.separator || "").slice(suf.length);' in block
    assert 'if (docPrefix) docPrefix += suf + (sep || "\\n");' in block
    assert 'docPrefix += m.separator || "\\n"' not in block


def test_dictate_page_stop_flow_guards(client):
    """Pins the client half of the stop flow, which has no JS test harness:
    a Stop during the mic prompt / worklet load cancels the pending start
    (generation guard), Stop flushes the partly filled PCM frame first, the
    fallback timer is re-armed per frame (long window for the forced final
    decode) and says when it fired, and a server error frame stays on
    screen through the closing drain's final."""
    body = client.get("/dictate").text
    assert "const gen = ++startGen;" in body
    for fn, use in (("async function start(gen) {", "new WebSocket("),
                    ("async function startBatch(gen) {", "new MediaRecorder(")):
        fbody = body.split(fn, 1)[1].split("\n  }\n", 1)[0]
        after_mic = fbody.split("getUserMedia(", 1)[1]
        assert after_mic.index("if (cancelled(gen))") < after_mic.index(use), fn
    start_body = body.split("async function start(gen) {", 1)[1].split("\n  }\n", 1)[0]
    after_worklet = start_body.split("addModule(blobUrl);", 1)[1]
    assert after_worklet.index("cancelled(gen)") < after_worklet.index("createMediaStreamSource")
    stop_body = body.split("function stop() {", 1)[1].split("\n  }\n", 1)[0]
    assert stop_body.index("pcmBuf.buffer.slice(0, pcmLen * 2)") < stop_body.index('type: "stop"')
    on_msg = body.split("function onMessage(ev) {", 1)[1].split("\n  }\n", 1)[0]
    assert "if (stopTimer) armStopTimer(STOP_WAIT_MS);" in on_msg
    assert 'if (stopTimer && m.state === "decoding") armStopTimer(STOP_DECODE_WAIT_MS);' in on_msg
    assert "finished without the server's last answer" in body
    assert 'if (running && statusEl.className !== "error") setStatus("listening", "live");' in on_msg
    assert 'if (!running || statusEl.className === "error") return;' in on_msg
    # A decode announced BEFORE Stop sends nothing until it finishes, so
    # stop() itself picks the long window from the remembered state, which a
    # final or a dropped frame clears. Only the known states set it: an
    # unknown state after "decoding" must not cut Stop to the short window.
    assert ('armStopTimer(uttState === "decoding" ? STOP_DECODE_WAIT_MS : STOP_WAIT_MS);'
            in stop_body)
    assert 'if (m.state === "dropped") uttState = null;' in on_msg
    assert ('else if (m.state === "open" || m.state === "decoding") uttState = m.state;'
            in on_msg)
    final_branch = on_msg.split('m.type === "final"', 1)[1].split("else if (m.type", 1)[0]
    assert "uttState = null;" in final_branch
    # Stop during the worklet load: dropStart detaches the socket's handlers,
    # so the cancelled branches finish the stop that owns this socket (or the
    # button stays disabled until the fallback blames the server).
    assert after_worklet.split("createMediaStreamSource", 1)[0].count(
        "if (ws === sock) finishStop();") == 1
    catch = start_body.split("} catch (err) {", 1)[1]
    assert "if (cancelled(gen)) { if (ws === sock) finishStop(); return; }" in catch
    # Batch mode: a recorder that cannot be built or started must release the
    # mic it was handed — the click handler's catch only resets the page.
    batch_body = body.split("async function startBatch(gen) {", 1)[1].split("\n  }\n", 1)[0]
    rec_catch = batch_body.split("recorder.start();", 1)[1].split("throw err;", 1)[0]
    assert "stream.getTracks().forEach((t) => t.stop());" in rec_catch
    assert "recStream = null;" in rec_catch


@pytest.mark.parametrize("exc", [OSError("gone"), ValueError("CR line endings")])
def test_dictate_page_unreadable_template_is_a_named_500(client, monkeypatch, exc):
    """The template is read per request: a missing file, or one an editor
    saved with CRLF, answers the page's own 500, not a bare traceback."""
    from faster_whisper_backend.core import templates

    def _boom(*a, **kw):
        raise exc
    monkeypatch.setattr(templates, "load", _boom)
    r = client.get("/dictate")
    assert r.status_code == 500 and "dictate unavailable" in r.text


def test_dictate_page_uses_the_shared_shell(client):
    """It renders through render_page like every other WebUI page: no
    unsubstituted placeholders, and the shared chrome is present."""
    body = client.get("/dictate").text
    assert "{{" not in body                       # every placeholder resolved
    assert "faster-whisper-backend · dictate" in body   # {{HEADER_TITLE}}
    assert 'class="navrow"' in body                     # {{NAV}}
    assert 'href="/dictate"' in body                    # own nav entry
    assert "login-gate" in body                         # shared auth screen


def test_dictate_page_carries_no_credential_field(client):
    """The page authenticates with the session cookie the rest of the WebUI
    uses. It must not reintroduce an API-key input, a key-bearing subprotocol,
    or the /auth/login exchange that used to mint a 30-day origin-wide session
    from a chrome-less page with no way to sign out."""
    body = client.get("/dictate").text
    # NB: the shared login gate legitimately carries its own key input, so only
    # the page's OWN field and credential plumbing are asserted absent.
    assert 'id="key"' not in body
    assert "bearer." not in body
    assert "wsAuth" not in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_dictate_page_sends_a_cleared_language_as_auto_detect(client):
    """The server reads `language` tri-state: absent / null inherits
    DEFAULT_LANGUAGE, "" auto-detects. The page sent a cleared box as null
    (streaming) or left the field out (batch), so on a server with a default
    language set, clearing the box to get auto-detect still decoded in the
    default. Runs the page's own handshake and form-building code."""
    body = client.get("/dictate").text
    onopen = body.split("sock.onopen = () => {", 1)[1]
    handshake = onopen.split("sock.send(JSON.stringify(", 1)[1].split("));\n", 1)[0]
    batch = body.split("const fd = new FormData();", 1)[1].split(
        'fd.append("response_format"', 1)[0]
    script = (
        "const box = process.argv[1];\n"
        "const $ = (id) => ({ value: id === 'lang' ? box : '' });\n"
        "const TARGET_SR = 16000, blob = null, ext = 'webm';\n"
        "const sent = {};\n"
        "const fd = { append: (k, v) => { sent[k] = v; } };\n"
        + batch
        + "\nconst cfg = " + handshake + ";\n"
        "console.log(JSON.stringify({ ws: cfg.language,"
        " form: 'language' in sent ? sent.language : 'ABSENT' }));\n"
    )
    for box, want in (("  ", ""), (" de ", "de")):
        out = subprocess.run(["node", "-e", script, box], capture_output=True,
                             text=True, check=True).stdout
        assert json.loads(out) == {"ws": want, "form": want}, box


# The handshake has three credential carriers: the Authorization header
# (native clients), the `bearer.<key>` subprotocol (the one request header a
# browser can set on a WebSocket), and the session cookie. A ?key= query
# param is NOT one — it would land in every access log that records the
# request line.

def test_ws_authorization_header_admits(client, make_user_key):
    _uid, raw = locked_down_user(make_user_key)
    with stream_ws(client, raw) as ws:
        assert _handshake(ws)["type"] == "ready"


def test_ws_basic_then_bearer_authorization_headers_admit(client, make_user_key):
    # A basic-auth reverse proxy in front (Server URL user:pass@host): the
    # client sends `Basic …` first and the API key's `Bearer` second.
    # httpx.Headers, not a list: websocket_connect setdefault()s on it.
    import httpx2 as httpx   # the declared dev dep; httpx is no longer transitive
    _uid, raw = locked_down_user(make_user_key)
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream", headers=httpx.Headers([
                ("Authorization", "Basic dXNlcjpwYXNz"),
                ("Authorization", f"Bearer {raw}"),
            ])) as ws:
        assert _handshake(ws)["type"] == "ready"


def test_ws_bearer_subprotocol_admits(client, make_user_key):
    from faster_whisper_backend.streaming.routes import _WS_BEARER_SUBPROTOCOL
    _uid, raw = locked_down_user(make_user_key)
    with client.websocket_connect(
            "/v1/audio/transcriptions/stream",
            subprotocols=[_WS_BEARER_SUBPROTOCOL + raw]) as ws:
        assert _handshake(ws)["type"] == "ready"


def test_ws_key_query_param_rejected(client, make_user_key):
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH
    _uid, raw = locked_down_user(make_user_key)
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


def test_stream_disabled_closes_connection(client, app_module, monkeypatch):
    from faster_whisper_backend.streaming.routes import _WS_DISABLED
    monkeypatch.setattr(app_module.cfg, "STREAMING_ENABLED", False, raising=False)
    # _refuse always accepts first, so the refusal is a post-accept close
    # carrying the application code — not a 4401 or a crash.
    with stream_ws(client) as ws:
        with pytest.raises(WebSocketDisconnect) as ei:
            ws.receive_json()
    assert ei.value.code == _WS_DISABLED


def test_stream_records_trace_text_per_utterance(client, energy_vad):
    """Each finalized utterance writes a recent-transcriptions row with non-empty
    raw/final text (drives /quick-config + /reports), not just numeric metrics."""
    from faster_whisper_backend.stats import recent_transcriptions_store
    before = recent_transcriptions_store.count()
    dictate_session(client)
    rows = recent_transcriptions_store.list_recent(limit=10)
    after = recent_transcriptions_store.count()
    assert after > before, "no recent-transcription row recorded for the utterance"
    assert any((r.get("raw") or "").strip() and (r.get("final") or "").strip() for r in rows), \
        "recorded trace rows have empty raw/final (the /quick-config bug)"
    # streamed utterances are tagged source='stream' so /quick-config can chip them.
    assert any(r.get("source") == "stream" for r in rows), \
        "streamed trace not tagged source='stream'"


def test_stream_records_the_key_label_on_the_trace_row(
        client, app_module, make_user_key, monkeypatch, energy_vad):
    """The recent-transcriptions row snapshots the API key's label straight
    from the auth record (user["key_label"]), the same way /transcribe does."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.stats import metrics
    make_user_key("root", is_admin=True)   # locks the server down
    uid = api_keys_store.create_user("alice", is_admin=False)
    raw, _rec = api_keys_store.create_key(uid, label="alice-laptop")
    dictate_session(client, raw)
    rows = metrics.metrics_snapshot(include_identity=True)["recent_transcriptions"]
    assert rows, "no recent-transcription row recorded for the utterance"
    assert rows[0]["key_label"] == "alice-laptop"


def test_flush_after_new_audio_is_not_coalesced(client, monkeypatch, energy_vad):
    """Flush coalescing must only collapse flushes with NO audio in between:
    `flush A → audio → flush B` has to deliver BOTH flushes to the pump (new
    audio makes the queued flush no longer authoritative), instead of silently
    degrading to flush A and leaving utterance B waiting for the silence gate."""
    import asyncio

    from faster_whisper_backend.streaming import session as streaming_session

    flushes = []
    orig_flush = streaming_session.StreamSession.flush_utterance

    async def slow_flush(self):
        # Hold the pump on flush A so audio + flush B arrive while it is still
        # in flight.
        flushes.append(time.monotonic())
        await asyncio.sleep(0.4)
        await orig_flush(self)

    monkeypatch.setattr(streaming_session.StreamSession, "flush_utterance", slow_flush)
    with ready_ws(client) as ws:
        ws.send_bytes(_pcm(8000, 1000))   # utterance A
        ws.send_json({"type": "flush"})   # flush A (pump held for 0.4 s)
        ws.send_bytes(_pcm(8000, 1000))   # utterance B — new audio after A's flush
        ws.send_json({"type": "flush"})   # flush B: must NOT coalesce into A
        stop_and_drain(ws)
    assert len(flushes) == 2, (
        f"flush after new audio was coalesced away ({len(flushes)} reached the pump)")


def test_safe_ws_send_swallows_dead_socket():
    """A page reload mid-dictation closes the socket; the session-close drain then
    sends a final to a dead socket. uvicorn raises RuntimeError('Unexpected ASGI
    message ... after ... close'); the send must be swallowed, not surface as an
    error traceback."""
    import asyncio

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


def test_stream_handshake_idle_timeout_frees_slot(client, app_module, monkeypatch):
    # A client that connects + passes auth but never sends its config handshake
    # must not hold a session slot forever: the server abandons the wait after
    # STREAMING_IDLE_TIMEOUT_S and closes with the idle close code (4408).
    from faster_whisper_backend.streaming.routes import _WS_IDLE_TIMEOUT
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 0.3, raising=False)
    with stream_ws(client) as ws:
        with pytest.raises(WebSocketDisconnect) as ei:
            ws.receive_json()   # send nothing → idle close
    assert ei.value.code == _WS_IDLE_TIMEOUT
    # The close code alone would pass with a leaked slot: both the per-user
    # count and the active-session set must be empty again.
    assert streaming_routes._stream_sessions._counts == {}
    assert streaming_routes._active_sessions == set()


def test_stream_session_idle_timeout_closes_and_notifies(
        client, app_module, monkeypatch, energy_vad):
    # After a successful handshake, a connection that goes silent mid-session is
    # closed once the idle timeout elapses, with an idle_timeout notice first.
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 0.3, raising=False)
    with ready_ws(client, response_format="json") as ws:
        msgs = _drain(ws)   # send nothing further → idle close
    assert any(m.get("code") == "idle_timeout" for m in msgs)


def test_stream_idle_timeout_zero_disables(client, app_module, monkeypatch, energy_vad):
    # 0 disables the idle timeout: the normal stop/close flow still works and no
    # idle_timeout notice is emitted (the receive falls through to a plain await).
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 0.0, raising=False)
    with ready_ws(client, response_format="json") as ws:
        msgs = stop_and_drain(ws)[0]
    assert not any(m.get("code") == "idle_timeout" for m in msgs)


def test_stream_idle_timeout_not_re_armed_by_non_audio_frames(
        client, app_module, monkeypatch, energy_vad):
    """The idle deadline is anchored to the last AUDIO byte, not the last frame.

    A client alternating an unrecognised control JSON with an empty binary
    frame (a no-op sink write) must not hold its session slot open while
    decoding nothing — enough such sockets would pin STREAMING_MAX_SESSIONS.

    The assertion is on the DRAIN duration, not on the notice alone: a deadline
    re-armed on every frame also emits an idle_timeout notice, just one full
    timeout AFTER the last frame.
    Here the socket must already be closed by the time we stop sending.
    """
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 0.5, raising=False)
    with ready_ws(client, response_format="json") as ws:
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
    assert drained_in < 0.25, f"socket was still armed when we stopped sending ({drained_in:.2f}s)"


def test_stream_idle_timeout_not_tripped_while_audio_flows(
        client, app_module, monkeypatch, energy_vad):
    """Real audio DOES re-arm the idle deadline, so a live mic (which streams
    PCM continuously, even through silence) is never cut short even though the
    session runs far longer than one idle timeout."""
    # 2 s timeout with a frame every 0.25 s: 1.75 s of slack per frame, so a
    # stalled sleep on a loaded CI runner cannot turn into a spurious idle
    # close (a 0.5 s timeout with 0.25 s frames flaked in CI).
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_S", 2.0, raising=False)
    with ready_ws(client, response_format="json") as ws:
        for _ in range(12):           # 3 s of streaming: 1.5x the idle timeout
            time.sleep(0.25)
            ws.send_bytes(_pcm(0, 100))   # silent PCM is still AUDIO
        msgs = stop_and_drain(ws)[0]
    assert not any(m.get("code") == "idle_timeout" for m in msgs)


def test_per_user_cap_refuses_while_the_global_cap_has_headroom(
        client, app_module, monkeypatch, caplog):
    """One client must not be able to fill the pool. The global cap is left
    wide open, so a refusal here can only come from the per-user one."""
    import logging

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS", 10,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 1,
                        raising=False)
    with caplog.at_level(logging.INFO, logger=streaming_routes.logger.name):
        with ready_ws(client):
            assert len(streaming_routes._active_sessions) == 1
            with pytest.raises(WebSocketDisconnect) as ei:
                with stream_ws(client) as ws2:
                    ws2.receive_json()
            assert ei.value.code == streaming_routes._WS_TOO_MANY
    assert any("STREAMING_MAX_SESSIONS_PER_USER" in r.getMessage()
               for r in caplog.records)


def test_per_user_cap_zero_is_unlimited(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 0,
                        raising=False)
    with stream_ws(client) as a, stream_ws(client) as b, stream_ws(client) as c:
        for ws in (a, b, c):
            assert _handshake(ws)["type"] == "ready"
        assert len(streaming_routes._active_sessions) == 3
        # limit<=0 short-circuits in InFlight.acquire, so no key is ever booked
        assert streaming_routes._stream_sessions._counts == {}


def test_session_admitted_with_the_cap_off_releases_no_slot(client, app_module,
                                                             monkeypatch):
    """A session admitted while STREAMING_MAX_SESSIONS_PER_USER=0 took no
    slot. When the cap is raised mid-session and a second session takes the
    identity's one real slot, the first one's teardown must leave that slot
    alone — an unconditional release there frees a slot a live session holds."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 0,
                        raising=False)
    counts = streaming_routes._stream_sessions._counts
    b_cm = None
    try:
        with ready_ws(client):
            assert counts == {}
            monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER",
                                1, raising=False)
            b_cm = stream_ws(client)
            b = b_cm.__enter__()
            assert _handshake(b)["type"] == "ready"
            assert list(counts.values()) == [1]
        # The cap-off session is gone; the second one still holds its slot.
        assert list(counts.values()) == [1]
    finally:
        if b_cm is not None:
            b_cm.__exit__(None, None, None)
    assert counts == {}


def test_global_cap_refusal_with_the_cap_off_releases_no_slot(client, app_module,
                                                               monkeypatch):
    """Same guard at the server-wide refusal: a connection refused there while
    the per-user cap is off took no slot, so it must not give back the one a
    live session of the same identity holds."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS", 1, raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 1,
                        raising=False)
    counts = streaming_routes._stream_sessions._counts
    with ready_ws(client):
        assert list(counts.values()) == [1]
        monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 0,
                            raising=False)
        with pytest.raises(WebSocketDisconnect) as ei:
            with stream_ws(client) as ws2:
                ws2.receive_json()
        assert ei.value.code == streaming_routes._WS_TOO_MANY
        assert list(counts.values()) == [1]
    assert counts == {}


def test_serial_sessions_never_leak_a_slot(client, app_module, monkeypatch):
    """The leak guard: open and close cap+2 sessions one at a time. Each must
    succeed, and the gauge must be empty afterwards — a release skipped on any
    teardown path would strand a slot and refuse the next connection."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 2,
                        raising=False)
    for _ in range(4):
        with stream_ws(client) as ws:
            assert _handshake(ws)["type"] == "ready"
    assert streaming_routes._stream_sessions._counts == {}


def test_per_user_cap_is_per_user(client, make_user_key, app_module,
                                  monkeypatch):
    """Open mode resolves every caller to one synthetic admin, so real keys
    are what prove the cap is keyed per identity."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 1,
                        raising=False)
    _uid_a, key_a = locked_down_user(make_user_key)
    _uid_b, key_b = make_user_key("bob")

    with ready_ws(client, key_a):
        with pytest.raises(WebSocketDisconnect) as ei:
            with stream_ws(client, key_a) as dup:
                dup.receive_json()
        assert ei.value.code == streaming_routes._WS_TOO_MANY
        # bob's own slot is untouched by alice holding hers.
        with stream_ws(client, key_b) as b:
            assert _handshake(b)["type"] == "ready"


def test_aborted_handshake_releases_the_per_user_slot(client, app_module,
                                                       monkeypatch):
    """ws.accept() raises when the client is already gone. The per-user slot
    is taken BEFORE accept and the releasing finally only starts after it, so
    without its own release an aborted handshake burns one slot for the life
    of the process."""
    import starlette.websockets

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 1,
                        raising=False)

    real_accept = starlette.websockets.WebSocket.accept

    async def _gone(self, *args, **kwargs):
        raise RuntimeError("client went away during the handshake")
    monkeypatch.setattr(starlette.websockets.WebSocket, "accept", _gone)
    # Exactly the injected error: a handshake idle-timeout close (4408) would
    # also empty _counts, without ever taking the aborted-accept path.
    with pytest.raises(RuntimeError, match="client went away during the handshake"):
        with stream_ws(client) as ws:
            ws.receive_json()
    assert streaming_routes._stream_sessions._counts == {}
    assert streaming_routes._active_sessions == set()

    # The slot is free again: the very next handshake from the same identity
    # goes through instead of being refused with 4429.
    monkeypatch.setattr(starlette.websockets.WebSocket, "accept", real_accept)
    with stream_ws(client) as ws:
        assert _handshake(ws)["type"] == "ready"


def test_cancel_during_a_completed_load_still_releases_the_lease(
        client, app_module, monkeypatch):
    """A cancellation delivered while the keepalive loop waits on a load that
    has ALREADY returned: the lease was taken, the record of it never made.
    The teardown must still find and release it."""
    import asyncio

    _real = tx_models._get_or_load_model

    async def _loader(name, *, lease=False):
        model = await _real(name, lease=False)
        # Cached like a real load: the release tells it from a drained one.
        monkeypatch.setitem(tx_models._loaded_models, name, model)
        if lease:
            tx_models._model_leases[name] = \
                tx_models._model_leases.get(name, 0) + 1
        return model
    monkeypatch.setattr(tx_models, "_get_or_load_model", _loader)

    real_wait = asyncio.wait

    async def _wait_then_cancel(fs, *args, **kwargs):
        done, pending = await real_wait(fs, *args, **kwargs)
        if done and all(t.done() for t in fs):
            raise asyncio.CancelledError()
        return done, pending
    monkeypatch.setattr(streaming_routes.asyncio, "wait", _wait_then_cancel)

    with pytest.raises(BaseException):
        with stream_ws(client) as ws:
            _handshake(ws)
            ws.receive_json()
    assert tx_models._model_leases == {}
    assert streaming_routes._active_sessions == set()


def test_stream_client_job_names_the_usage_job(client, energy_vad):
    """The handshake's `client_job` becomes the usage job id, so the outcome
    the client posts afterwards lands on the session's own utterances."""
    from faster_whisper_backend.stats import usage_store
    dictate_session(client, client_job="ab" * 16)
    rows = usage_store._require_conn().execute(
        "SELECT job_id, kind, utterances FROM usage_jobs").fetchall()
    assert [tuple(r) for r in rows] == [("ab" * 16, "dictation", 1)]


def test_stream_malformed_client_job_falls_back_to_session_id(client, energy_vad):
    from faster_whisper_backend.stats import usage_store
    dictate_session(client, client_job="../not-hex")
    rows = usage_store._require_conn().execute(
        "SELECT job_id, kind FROM usage_jobs").fetchall()
    assert len(rows) == 1 and rows[0]["kind"] == "dictation"
    assert len(rows[0]["job_id"]) == 32 and rows[0]["job_id"] != "../not-hex"


def test_dictate_page_keeps_earlier_documents_across_a_hard_break(client):
    """StreamSession._hard_break (STREAMING_HARD_BREAK_SILENCE_MS, 5 s by
    default) emits {"type":"boundary"} and resets the document, so the next
    final's committed/tail cover only the NEW document. The page must fold
    what is on screen into a prefix on that frame, or the full-replace final
    handler wipes everything dictated before the pause the moment the user
    resumes speaking."""
    body = client.get("/dictate").text
    assert 'm.type === "boundary"' in body
    assert "docPrefix + (m.committed" in body
    assert 'docPrefix = ""' in body          # reset per start()


def test_dictate_page_resampler_seam_uses_a_real_neighbour(client):
    """pushSamples() must interpolate a block seam between the previous block's
    last sample (carry, index -1) and this block's first — not use the carry as
    the right neighbour of the LAST sample (128 samples stale). Invisible at
    48 kHz (integer ratio) but a ~124 Hz click train at 44.1 kHz."""
    body = client.get("/dictate").text
    assert "i < 0 ? resampleCarry : float32[i]" in body
    assert "float32[i + 1] : resampleCarry" not in body


def test_stream_final_arms_the_residual_stop_and_reports_it(
        client, app_module, monkeypatch, caplog, energy_vad):
    """The FINAL decode resolves DECODE_SKIP_RESIDUAL_WINDOWS through cfg_for,
    hands it to the trace capture, and the receipt's guards block shows the
    value (with the non-default marker once it is switched off)."""
    import logging
    from faster_whisper_backend.transcription import decode_trace as dt

    seen = []
    real = dt.capture

    def spy(kwargs=None, *, skip_residual=False, **kw):
        seen.append(skip_residual)
        return real(kwargs, skip_residual=skip_residual, **kw)
    monkeypatch.setattr(streaming_routes.decode_trace, "capture", spy)

    with caplog.at_level(logging.INFO, logger=streaming_routes.logger.name):
        dictate_session(client)
    assert seen == [True], "one final decode, stop rule armed"
    assert guard_line(caplog, "skip_residual_windows").endswith("true")

    seen.clear(); caplog.clear()
    monkeypatch.setattr(app_module.cfg, "DECODE_SKIP_RESIDUAL_WINDOWS", False, raising=False)
    with caplog.at_level(logging.INFO, logger=streaming_routes.logger.name):
        dictate_session(client)
    assert seen == [False]
    assert guard_line(caplog, "skip_residual_windows").endswith("false *")


def _final_and_partial_kwargs(fake_model):
    """Record every transcribe call; the FINAL is the one carrying the
    temperature ladder (partials decode at a single temperature)."""
    calls = []
    real = fake_model.transcribe

    def rec(path, **kw):
        calls.append(kw)
        return real(path, **kw)
    fake_model.transcribe = rec
    return calls


def _is_final(kw):
    return isinstance(kw.get("temperature"), (tuple, list))


def test_stream_final_caps_tokens_and_samples_one_candidate(
        client, app_module, fake_model, monkeypatch, caplog, energy_vad):
    """A looping rung once ran to 224 tokens (18.6 s), then a sampled rung's
    17-token answer waited 13.9 s for a looping sibling. The FINAL
    decode gets the token cap (through the trace capture) and
    STREAMING_FINAL_BEST_OF; partials keep their own knobs."""
    import logging
    from faster_whisper_backend.transcription import decode_trace as dt

    caps = []
    real = dt.capture

    def spy(kwargs=None, **kw):
        caps.append(kw.get("token_cap_per_s"))
        return real(kwargs, **kw)
    monkeypatch.setattr(streaming_routes.decode_trace, "capture", spy)
    calls = _final_and_partial_kwargs(fake_model)

    with caplog.at_level(logging.INFO, logger=streaming_routes.logger.name):
        dictate_session(client)
    assert caps == [10.0]
    finals = [kw for kw in calls if _is_final(kw)]
    assert len(finals) == 1 and finals[0]["best_of"] == 1
    assert all(kw["best_of"] == app_module.cfg.BEST_OF
               for kw in calls if not _is_final(kw)), "partials untouched"
    assert guard_line(caplog, "token_cap_per_second").endswith("10.0")

    caps.clear(); calls.clear()
    monkeypatch.setattr(app_module.cfg, "DECODE_TOKEN_CAP_PER_SECOND", 0.0, raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_FINAL_BEST_OF", 3, raising=False)
    dictate_session(client)
    assert caps == [0.0]
    assert [kw["best_of"] for kw in calls if _is_final(kw)] == [3]


def test_stream_final_best_of_yields_to_a_client_override(client, fake_model, energy_vad):
    calls = _final_and_partial_kwargs(fake_model)
    dictate_session(client, decode_overrides={"best_of": 4})
    assert [kw["best_of"] for kw in calls if _is_final(kw)] == [4]


@pytest.mark.parametrize("value", [None, "abc"])
def test_stream_final_best_of_pins_when_the_client_value_does_not_apply(
        client, app_module, fake_model, monkeypatch, value, energy_vad):
    """A null / unparseable best_of is dropped by the assembler (the batch
    BEST_OF stays in kwargs), so the key's mere presence must not skip the
    STREAMING_FINAL_BEST_OF pin."""
    monkeypatch.setattr(app_module.cfg, "BEST_OF", 5, raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_FINAL_BEST_OF", 1, raising=False)
    calls = _final_and_partial_kwargs(fake_model)
    dictate_session(client, decode_overrides={"best_of": value})
    assert [kw["best_of"] for kw in calls if _is_final(kw)] == [1]


# Tail cuts inside a segment (transcription/segment_guards.py) on finals and
# previews. _loop_segment's timings are the real ones the production server
# returned on 2026-09-19.

def _loop_segment():
    from tests.conftest import FakeSegment, FakeWord
    real = [(" Besser", 0.32, 0.74), (" wäre", 0.74, 1.0), (" zwei", 1.0, 1.4),
            (" Tabletten", 1.4, 2.0)]
    tail = [(" zu", 2.0, 2.04), (" nehmen", 2.04, 2.04), (" und", 2.04, 2.4)] + [
        (w, 2.4, 2.4) for w in (" die", " Doppelpunkte", " abzuschliessen") * 3]
    ws = [FakeWord(*x) for x in real + tail]
    return FakeSegment("".join(w.word for w in ws), 0.32, 2.4, words=ws)


def test_stream_final_and_previews_cut_the_made_up_tail(client, fake_model, caplog, energy_vad):
    from faster_whisper_backend.stats import metrics
    fake_model._segments = [_loop_segment()]
    with caplog.at_level("INFO"):
        msgs = dictate_session(client)
    finals = [m for m in msgs if m["type"] == "final"]
    partials = [m for m in msgs if m["type"] == "partial"]
    text = "".join(m["committed"] + m.get("tail", "") for m in finals[-1:])
    assert "Tabletten" in text and "abzuschliessen" not in text and "nehmen" not in text
    # previews never show (and LocalAgreement never commits) the tail either
    assert partials
    assert all("abzuschliessen" not in m["committed"] + m.get("pending", "")
               for m in partials)
    assert metrics.guard_hits["burst"] == 1          # one final; previews uncounted
    assert "cut made-up tail of final segment (burst+zero_tail+repeat" in caplog.text
    assert "tail_cut" in caplog.text


def test_stream_preview_drops_a_segment_cut_to_nothing(client, fake_model, monkeypatch,
                                                       energy_vad):
    """A preview whose only segment is one zero-length pile: handed on as
    (start, made-up end, "") two previews agree on "" and LocalAgreement
    commits an empty word at the made-up end — every later real word before
    that time is then dropped. The preview must hand on nothing instead."""
    from tests.conftest import FakeSegment, FakeWord
    from faster_whisper_backend.streaming import localagreement
    ws = [FakeWord(f" x{i}", 2.4, 2.4) for i in range(12)]
    fake_model._segments = [FakeSegment("".join(w.word for w in ws), 0.3, 2.4, words=ws)]
    seen = []
    orig = localagreement.LocalAgreementProcessor.insert_hypothesis

    def spy(self, words, offset):
        words = list(words)
        seen.append(words)
        return orig(self, words, offset)

    monkeypatch.setattr(localagreement.LocalAgreementProcessor, "insert_hypothesis", spy)
    msgs = dictate_session(client)
    assert seen, "no preview was decoded"
    assert all(h == [] for h in seen)
    assert all(m["committed"] == "" for m in msgs if m["type"] == "partial")


def test_stream_tail_cuts_off_keep_the_tail(client, app_module, fake_model, monkeypatch,
                                            energy_vad):
    for name in ("SEGMENT_MAX_WORD_BURST_PER_S", "SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS",
                 "SEGMENT_REPEAT_COLLAPSE_MIN_REPEATS", "SEGMENT_MAX_WORDS_PER_S"):
        monkeypatch.setattr(app_module.cfg, name, 0, raising=False)
    fake_model._segments = [_loop_segment()]
    msgs = dictate_session(client)
    finals = [m for m in msgs if m["type"] == "final"]
    assert "abzuschliessen" in "".join(m["committed"] + m.get("tail", "") for m in finals)


def test_stream_formats_the_first_final_in_the_detected_language(client, fake_model, monkeypatch,
                                                                 energy_vad):
    """Auto language: the utterance's own final is formatted in the language
    its decode detected, not with every language's rules (a German question
    got the Spanish '¿'), so the closing final extends it unchanged. The
    previews guess Spanish on purpose: _fmt_lang falls back to the partials'
    guess, so a matching preview language would hide a missing detection."""
    from tests.conftest import FakeInfo, FakeSegment
    fake_model._segments = [FakeSegment(" Hat der Patient Fieber Fragezeichen", 0.0, 1.0)]
    real_transcribe = fake_model.transcribe
    seen = {"preview": 0, "final": 0}

    def transcribe(path, **kwargs):
        segs, info = real_transcribe(path, **kwargs)
        if isinstance(kwargs.get("temperature"), (list, tuple)):
            seen["final"] += 1          # the final decode's temperature ladder
            return segs, info
        seen["preview"] += 1
        return segs, FakeInfo(language="es")
    monkeypatch.setattr(fake_model, "transcribe", transcribe)
    msgs = dictate_session(client)
    assert seen["preview"] and seen["final"]
    docs = [m["committed"] + m.get("tail", "") for m in msgs if m["type"] == "final"]
    assert len(docs) >= 2                                # the utterance's final + the close
    assert docs[0] == "Hat der Patient Fieber?"
    assert all("¿" not in d for d in docs)
    for a, b in zip(docs, docs[1:]):
        assert b.startswith(a)


def test_stream_unknown_language_skips_language_tagged_rules(client, fake_model, energy_vad):
    """No language requested, none detected: "und" — every language-tagged rule
    is skipped (the German map and the Spanish marks alike) instead of all of
    them running."""
    from tests.conftest import FakeInfo, FakeSegment
    fake_model._segments = [FakeSegment(" Hat der Patient Fieber Fragezeichen", 0.0, 1.0)]
    fake_model._info = FakeInfo(language=None)
    msgs = dictate_session(client)
    docs = [m["committed"] + m.get("tail", "") for m in msgs if m["type"] == "final"]
    assert docs and all("¿" not in d for d in docs)
    assert "Fragezeichen" in docs[-1]


def test_stream_holds_a_split_dictation_phrase_until_the_close(client, fake_model, energy_vad):
    """The route wires the real hold-back: the trailing 'neue' of a final is
    held back (the rest is sent) until the close releases it."""
    from tests.conftest import FakeSegment
    fake_model._segments = [FakeSegment(" Befund unauffällig Punkt neue", 0.0, 1.0)]
    msgs = dictate_session(client)
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals[0]["committed"] + finals[0]["tail"] == "Befund unauffällig."
    assert finals[-1].get("last") is True
    assert finals[-1]["committed"] == "Befund unauffällig. Neue"


def test_preview_decode_does_not_charge_the_session_wait():
    """A live preview's GPU-gate wait must not land on the dictation row's
    wait_s: decode_partial unsets the session accumulator around its decode
    (the contract itself is in tests/stats/test_metrics.py)."""
    import inspect

    src = inspect.getsource(streaming_routes)
    body = src[src.index("async def decode_partial("):]
    body = body[:body.index("\n        async def ", 1)]
    unset = body.index("metrics.WAIT_ACC.set(None)")
    assert unset < body.index("await _transcribe(partial_model_obj")
    assert "metrics.WAIT_ACC.reset(_wait_tok)" in body


def test_final_decode_starts_from_a_clean_session_wait():
    """decode_final zeroes the session wait accumulator (metrics.take_wait())
    after the revoked-credential check and before its own decode, so a wait
    left over from an earlier decode cannot land on this utterance's wait_s."""
    import inspect

    src = inspect.getsource(streaming_routes)
    body = src[src.index("async def decode_final("):]
    body = body[:body.index("\n        async def ", 1)]
    take = body.index("metrics.take_wait()")
    assert body.index('_require_live_credential()') < take
    assert take < body.index("await _transcribe(")
