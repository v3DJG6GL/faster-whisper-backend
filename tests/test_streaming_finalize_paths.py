"""The finalize paths that reach on_final WITHOUT a final decode, and the
handshake refusals' close codes.

The near-silence gate (streaming_session._finalize) skips the decoder but
still reports the partial-committed text. on_final then has no decode of its
own to describe, so every sink that used to assume "on_final follows
decode_final" is exercised here: the diagnostics snapshot, the word list,
the stage/usage row, and the revocation guard. The handshake refusals must
ACCEPT before they close, or the application close code never reaches a
browser (a pre-accept close is a bare HTTP 403 on the wire).
"""

import asyncio
import logging
import time

import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from conftest import bearer
from streaming_session import StreamConfig, StreamSession
from streaming_vad import FRAME_MS, SAMPLE_RATE as SR

_STREAM_URL = "/v1/audio/transcriptions/stream"


def _pcm(level, ms, sr=16000):
    return np.full(sr * ms // 1000, level, dtype="<i2").tobytes()


def _drain(ws, limit=200):
    msgs = []
    code = None
    try:
        for _ in range(limit):
            msgs.append(ws.receive_json())
    except WebSocketDisconnect as exc:
        code = exc.code
    return msgs, code


def _config(ws, **extra):
    ws.send_json({"type": "config", "model": "whisper-1",
                  "audio": {"format": "pcm_s16le", "sample_rate": 16000}, **extra})
    return ws.receive_json()


# ---- session level: the gate path's payload ----------------------------------


class _AlwaysSpeech:
    def is_speech(self, frame):
        return True

    def reset(self) -> None:
        pass


def _grid_words(start_s: float, end_s: float):
    out = []
    t = int(start_s * 10)
    while t < int(end_s * 10):
        out.append((t / 10.0, (t + 1) / 10.0, f" w{t}"))
        t += 1
    return out


def _run_gate_path():
    """Loud speech long enough for a trim to bank words, then quiet frames
    until the max_buffer_sec ceiling forces a finalize whose live buffer is
    near-silence: the gate skips the decode, on_final still fires."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-42.0, preroll_keep_ms=100, max_buffer_sec=6.0,
    )
    finals_meta: list[dict] = []
    final_decodes: list[float] = []
    box: list[StreamSession] = []

    async def emit(m):
        pass

    async def on_final(info):
        finals_meta.append(info)

    async def decode_partial(audio, prompt):
        off = box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]

    async def decode_final(audio, prompt):
        final_decodes.append(audio.shape[0] / SR)
        return (" should-not-run.", [], False)

    s = StreamSession(
        config=cfg, endpointer=_AlwaysSpeech(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit, on_final=on_final,
    )
    box.append(s)

    async def run():
        for _ in range(3500 // FRAME_MS):
            await s.feed_pcm(_pcm(8000, FRAME_MS))
        assert s._trimmed_text, "setup failed: no trim banked any text"
        for _ in range(5000 // FRAME_MS):
            await s.feed_pcm(_pcm(1, FRAME_MS))
            if finals_meta:
                break

    asyncio.run(run())
    assert finals_meta, "the ceiling never forced a finalize"
    assert final_decodes == []
    return finals_meta[0]


def test_gate_path_flags_no_decode_and_hands_over_whole_utterance_words():
    info = _run_gate_path()
    assert info["decoded"] is False
    assert info["proc_dur"] == 0.0
    assert info["trimmed_sec"] > 0.0
    words = info["words"]
    # Not empty, not just the trim-banked prefix: the whole utterance.
    assert words, "gate path handed captures an empty word list"
    assert "".join(w["word"] for w in words) == info["raw_text"]
    assert words[0]["word"] == " w0"
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    texts = [w["word"] for w in words]
    assert len(texts) == len(set(texts)), f"duplicated words: {texts!r}"
    # Absolute utterance time: every word lies inside the whole utterance's
    # audio, and the banked prefix was not re-based or prepended twice.
    assert words[-1]["end"] <= info["audio_dur"] + 1e-6
    assert words[-1]["end"] > info["trimmed_sec"]


def test_decoded_path_flags_decode_and_rebases_words():
    """The ordinary path keeps its contract: decoded=True and the final
    decode's buffer-relative words re-based behind the banked prefix."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    finals_meta: list[dict] = []
    box: list[StreamSession] = []

    async def emit(m):
        pass

    async def on_final(info):
        finals_meta.append(info)

    async def decode_partial(audio, prompt):
        off = box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]

    async def decode_final(audio, prompt):
        off = box[0]._buffer_offset
        dur = audio.shape[0] / SR
        gw = _grid_words(off, off + dur)
        return ("".join(t for _, _, t in gw),
                [{"word": t, "start": a - off, "end": b - off} for a, b, t in gw],
                False)

    s = StreamSession(
        config=cfg, endpointer=__import__("streaming_vad").EnergyEndpointer(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit, on_final=on_final,
    )
    box.append(s)

    async def run():
        await s.feed_pcm(_pcm(8000, 3500))
        await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    info = finals_meta[-1]
    assert info["decoded"] is True
    assert info["trimmed_sec"] > 0.0
    words = info["words"]
    assert words[0]["word"] == " w0"
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert len({w["word"] for w in words}) == len(words)


# ---- route level ----------------------------------------------------------------


def _gate_second_utterance(client, app_module, monkeypatch, ws, *, before_second=None):
    """Utterance 1 decodes normally; utterance 2 commits text through the
    partials and is then finalized with the RMS gate forced shut, so on_final
    runs for it without a decode."""
    import streaming_session

    ws.send_bytes(_pcm(8000, 2500))
    ws.send_bytes(_pcm(0, 1500))          # utterance 1 → decoded final
    time.sleep(0.5)
    ws.send_bytes(_pcm(8000, 2500))       # utterance 2: partials commit text
    time.sleep(0.5)
    if before_second is not None:
        before_second()
    # From here the gate judges every buffer as silence: the pending finalize
    # skips its decode but still reports the committed text.
    monkeypatch.setattr(streaming_session, "rms_dbfs", lambda a: -999.0)
    ws.send_bytes(_pcm(0, 1500))
    time.sleep(0.5)


def test_gate_skipped_utterance_carries_no_previous_decode_diagnostics(
        client, app_module, monkeypatch, caplog):
    import recent_transcriptions_store

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        with client.websocket_connect(_STREAM_URL) as ws:
            assert _config(ws)["type"] == "ready"
            _gate_second_utterance(client, app_module, monkeypatch, ws)
            ws.send_json({"type": "stop"})
            msgs, _code = _drain(ws)
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) >= 2, msgs
    rows = sorted(recent_transcriptions_store.list_recent(limit=10),
                  key=lambda r: r.get("created_ts") or 0)
    rows = [r for r in rows if r.get("source") == "stream"]
    assert len(rows) >= 2, rows
    first, second = rows[0], rows[-1]
    # The first utterance decoded with the fake model (language "de") and got
    # its transcribing stage; the gate-skipped one must not inherit either.
    assert first["language"] == "de"
    assert any(st.get("name") == "transcribing" for st in (first.get("stages") or []))
    assert second["language"] == ""
    assert not (second.get("stages") or []), second.get("stages")
    lines = [r.getMessage() for r in caplog.records]
    assert any("near-silence gate skipped the final decode" in ln for ln in lines), lines
    # The decoded-trim wording must not be claimed for the skipped decode.
    skipped = [ln for ln in lines if "near-silence gate skipped" in ln]
    assert not any("final decode heard" in ln for ln in skipped)


def test_revoked_identity_gets_no_rows_through_the_gate_path(
        client, make_user_key, app_module, monkeypatch):
    """decode_final's pre-model raise used to be the only revocation guard;
    the gate path reaches on_final with no decode at all, so on_final has to
    check for itself. A revoked identity must write no trace/usage row."""
    import api_keys_store
    import recent_transcriptions_store
    from streaming_routes import _WS_UNAUTH

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)
    uid, raw_alice = make_user_key("alice")
    counted = {}

    def revoke():
        counted["before"] = recent_transcriptions_store.count()
        api_keys_store.revoke_user(uid)

    with client.websocket_connect(_STREAM_URL, headers=bearer(raw_alice)) as ws:
        assert _config(ws)["type"] == "ready"
        _gate_second_utterance(client, app_module, monkeypatch, ws, before_second=revoke)
        ws.send_bytes(_pcm(0, 200))
        try:
            ws.send_json({"type": "stop"})
        except Exception:  # noqa: BLE001 — already closed
            pass
        msgs, code = _drain(ws)
    assert code == _WS_UNAUTH, (code, msgs)
    assert recent_transcriptions_store.count() == counted["before"], \
        "the revoked identity still got a trace/usage row through the gate path"


# ---- translate_expect on a translation-disabled server ----------------------


def test_translate_expect_is_ignored_when_translation_is_disabled(
        client, app_module, monkeypatch):
    """/v1/text/translations refuses before it can claim a receipt, so a
    declaration on a TRANSLATION_ENABLED=0 server must park nothing: the
    receipt is logged inline and no `captured` frame invites a claim."""
    import captures_store
    import receipt_hold

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", False, raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True, raising=False)
    monkeypatch.setattr(captures_store, "create_capture", lambda **kw: "cap-test-id")

    with client.websocket_connect(_STREAM_URL) as ws:
        ready = _config(ws, translate_expect={"targets": ["en"], "per_utterance": True})
        assert ready["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))
        ws.send_json({"type": "stop"})
        msgs, _code = _drain(ws)
    assert any(m["type"] == "final" for m in msgs)
    assert not any(m.get("type") == "captured" for m in msgs), msgs
    assert receipt_hold.pending() == 0


def test_translate_expect_parks_when_translation_is_enabled(
        client, app_module, monkeypatch):
    """The other direction, so the gate above is not just disabling the hold."""
    import captures_store
    import receipt_hold

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True, raising=False)
    monkeypatch.setattr(captures_store, "create_capture", lambda **kw: "cap-test-id")

    with client.websocket_connect(_STREAM_URL) as ws:
        ready = _config(ws, translate_expect={"targets": ["en"], "per_utterance": True})
        assert ready["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))
        ws.send_json({"type": "stop"})
        msgs, _code = _drain(ws)
    assert any(m.get("type") == "captured" for m in msgs), msgs
    assert receipt_hold.pending() >= 1


# ---- refusals carry their close code ---------------------------------------


def test_disabled_refusal_closes_after_accept_with_4503(app_module, monkeypatch):
    from streaming_routes import _WS_DISABLED

    monkeypatch.setattr(app_module.cfg, "STREAMING_ENABLED", False, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with pytest.raises(WebSocketDisconnect) as ei:
            with client.websocket_connect(_STREAM_URL) as ws:
                ws.receive_json()
    # A pre-accept close surfaces as a handshake error (HTTP 403), never as a
    # close with this code — so the code arriving proves the accept happened.
    assert ei.value.code == _WS_DISABLED
    assert "disabled" in (ei.value.reason or "")


def test_unauth_refusal_closes_after_accept_with_4401(app_module):
    from streaming_routes import _WS_UNAUTH

    with TestClient(app_module.app, client=("203.0.113.9", 1234)) as c:
        with pytest.raises(WebSocketDisconnect) as ei:
            with c.websocket_connect(_STREAM_URL) as ws:
                ws.receive_json()
    assert ei.value.code == _WS_UNAUTH


def test_origin_refusal_closes_after_accept_with_4403(client, app_module):
    from streaming_routes import _WS_BAD_ORIGIN

    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect(
                _STREAM_URL, headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
    assert ei.value.code == _WS_BAD_ORIGIN


def test_cap_refusal_closes_after_accept_with_4429(client, app_module, monkeypatch):
    from streaming_routes import _WS_TOO_MANY

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS", 1, raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 0, raising=False)
    with client.websocket_connect(_STREAM_URL) as ws:
        assert _config(ws)["type"] == "ready"
        with pytest.raises(WebSocketDisconnect) as ei:
            with client.websocket_connect(_STREAM_URL) as ws2:
                ws2.receive_json()
        assert ei.value.code == _WS_TOO_MANY
        assert "live sessions" in (ei.value.reason or "")


# ---- handshake reporting / jobs row / queue item cap ---------------------------


def test_condition_on_previous_text_override_is_reported_ignored(client, app_module):
    with client.websocket_connect(_STREAM_URL) as ws:
        ready = _config(ws, decode_overrides={"condition_on_previous_text": True})
        assert ready["type"] == "ready"
        assert "condition_on_previous_text" in ready.get("overrides_ignored", [])


def test_running_dictate_job_names_its_model(client, app_module):
    import jobs

    with client.websocket_connect(_STREAM_URL) as ws:
        assert _config(ws)["type"] == "ready"
        rows = [r for r in jobs.jobs_snapshot() if r["kind"] == "dictate"]
        assert rows, "no running dictate job"
        assert rows[0]["model"] == app_module._resolve_model_name("whisper-1")


def test_queue_item_cap_sheds_tiny_frames_and_flushes(app_module, monkeypatch, caplog):
    """Two-byte PCM frames alternated with flushes queue two items per ~0 bytes,
    so the byte cap never trips; the item cap must shed them instead."""
    import streaming_routes
    import streaming_session

    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(streaming_routes, "_HARD_CAP_ITEMS", 32)
    orig_flush = streaming_session.StreamSession.flush_utterance
    held = []

    async def slow_flush(self):
        # Hold the pump on the first flush so everything after it queues.
        if not held:
            held.append(True)
            await asyncio.sleep(2.0)
        await orig_flush(self)

    monkeypatch.setattr(streaming_session.StreamSession, "flush_utterance", slow_flush)
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
            with client.websocket_connect(_STREAM_URL) as ws:
                assert _config(ws)["type"] == "ready"
                ws.send_bytes(_pcm(8000, 200))
                ws.send_json({"type": "flush"})   # pump held here
                for _ in range(100):
                    ws.send_bytes(b"\x00\x00")
                    ws.send_json({"type": "flush"})
                ws.send_json({"type": "stop"})
                _drain(ws)
    assert any("audio backlog over cap" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]
