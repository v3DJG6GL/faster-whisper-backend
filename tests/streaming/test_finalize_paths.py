"""The finalize paths that reach on_final WITHOUT a final decode, the
handshake refusals' close codes, and the route's smaller per-utterance
contracts (translate_expect parking, the capture gate, handshake reporting,
the queue item cap).

The near-silence gate (StreamSession._finalize_inner) skips the decoder but
still reports the partial-committed text. on_final then has no decode of its
own to describe, so every sink that used to assume "on_final follows
decode_final" is exercised here: the diagnostics snapshot, the word list,
the stage/usage row, and the revocation guard. The handshake refusals must
ACCEPT before they close, or the application close code never reaches a
browser (a pre-accept close is a bare HTTP 403 on the wire).
"""

import asyncio
import logging
import threading
import time

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests._streaming_helpers import (
    AlwaysSpeech as _AlwaysSpeech, collector, const_final, const_pcm as _pcm, fast_config,
    dictate_session, handshake as _config, ready_ws, speak_and_stop, stop_and_drain,
    timed_final, ws_drain as _drain)
from tests.streaming._feed_helpers import feed_levels
from faster_whisper_backend.streaming.session import StreamSession
from faster_whisper_backend.streaming.vad import FRAME_MS, SAMPLE_RATE as SR, EnergyEndpointer
from faster_whisper_backend.transcription import models as tx_models

_STREAM_URL = "/v1/audio/transcriptions/stream"


def _grid_words(start_s: float, end_s: float):
    out = []
    t = int(start_s * 10)
    while t < int(end_s * 10):
        out.append((t / 10.0, (t + 1) / 10.0, f" w{t}"))
        t += 1
    return out


def _grid_partial(box):
    """Fake partial decoder: the grid words of exactly the buffer it receives."""
    async def decode_partial(audio, prompt):
        off = box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]
    return decode_partial


async def _discard(m):
    pass


def _grid_session(box, cfg, decode_final, *, endpointer=None, emit=_discard, **kw):
    """Session whose partials transcribe the grid of their buffer; appended to box."""
    s = StreamSession(
        config=cfg, endpointer=endpointer or EnergyEndpointer(),
        decode_partial=_grid_partial(box), decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit, **kw,
    )
    box.append(s)
    return s


def _run_gate_path():
    """Loud speech long enough for a trim to bank words, then quiet frames
    until the max_buffer_sec ceiling forces a finalize whose live buffer is
    near-silence: the gate skips the decode, on_final still fires."""
    cfg = fast_config(buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
                      rms_gate_dbfs=-42.0, max_buffer_sec=6.0)
    finals_meta, on_final = collector()
    final_decodes, decode_final = timed_final(" should-not-run.", SR)

    s = _grid_session([], cfg, decode_final, endpointer=_AlwaysSpeech(), on_final=on_final)

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
    cfg = fast_config(buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0)
    finals_meta, on_final = collector()
    box: list[StreamSession] = []

    async def decode_final(audio, prompt):
        off = box[0]._buffer_offset
        dur = audio.shape[0] / SR
        gw = _grid_words(off, off + dur)
        return ("".join(t for _, _, t in gw),
                [{"word": t, "start": a - off, "end": b - off} for a, b, t in gw],
                False)

    s = _grid_session(box, cfg, decode_final, on_final=on_final)

    feed_levels(s, (8000, 3500), (0, 400))
    info = finals_meta[-1]
    assert info["decoded"] is True
    assert info["trimmed_sec"] > 0.0
    words = info["words"]
    assert words[0]["word"] == " w0"
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert len({w["word"] for w in words}) == len(words)


@pytest.mark.parametrize("trim", [False, True], ids=["no_trim", "trim"])
def test_empty_final_decode_takes_text_and_words_from_localagreement(trim):
    """A final decode that returns nothing (its own VAD trimmed the buffer)
    without dropping anything falls back to the partial-agreed transcript —
    for the text AND the words (not the decode's empty list, nor only the
    trim-banked prefix), so the captured text and words match."""
    cfg = fast_config(buffer_trim_sec=2.0 if trim else 100.0, buffer_trim_keep_sec=1.0)
    finals_meta, on_final = collector()
    s = _grid_session([], cfg, const_final(), on_final=on_final)

    feed_levels(s, (8000, 3500), (0, 400))
    info = finals_meta[-1]
    assert (info["trimmed_sec"] > 0.0) is trim
    assert info["raw_text"].strip(), "setup failed: the partials agreed nothing"
    words = info["words"]
    assert "".join(w["word"] for w in words) == info["raw_text"]
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert words[-1]["end"] <= info["audio_dur"] + 1e-6


def _gate_second_utterance(monkeypatch, ws, *, before_second=None):
    """Utterance 1 decodes normally; utterance 2 commits text through the
    partials and is then finalized with the RMS gate forced shut, so on_final
    runs for it without a decode."""
    from faster_whisper_backend.streaming import session as streaming_session

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
        client, app_module, monkeypatch, caplog, energy_vad):
    from faster_whisper_backend.stats import recent_transcriptions_store

    with caplog.at_level(logging.INFO, logger="faster_whisper_backend.streaming.routes"):
        with ready_ws(client) as ws:
            _gate_second_utterance(monkeypatch, ws)
            msgs, _code = stop_and_drain(ws)
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) >= 2, msgs
    rows = sorted(recent_transcriptions_store.list_recent(limit=10),
                  key=lambda r: r.get("created_ts") or 0)
    rows = [r for r in rows if r.get("source") == "stream"]
    assert len(rows) >= 2, rows
    first, second = rows[0], rows[-1]
    # The first utterance decoded with the fake model (language "de") and got
    # its transcribing stage; the gate-skipped one must not inherit the stage.
    # Its language is the session's detected one, not a blank.
    assert first["language"] == "de"
    assert any(st.get("name") == "transcribing" for st in (first.get("stages") or []))
    assert second["language"] == "de"
    assert not (second.get("stages") or []), second.get("stages")
    lines = [r.getMessage() for r in caplog.records]
    assert any("near-silence gate skipped the final decode" in ln for ln in lines), lines
    # The decoded-trim wording must not be claimed for the skipped decode.
    skipped = [ln for ln in lines if "near-silence gate skipped" in ln]
    assert not any("final decode heard" in ln for ln in skipped)


def test_gate_skipped_utterance_receipt_says_no_final_decode(
        client, app_module, monkeypatch, caplog, energy_vad):
    """The gate-skipped utterance's receipt says "no final decode", not
    "[!] empty output" blaming the decode thresholds: its RAW line holds text."""
    with caplog.at_level(logging.INFO, logger="faster_whisper_backend.streaming.routes"):
        with ready_ws(client) as ws:
            _gate_second_utterance(monkeypatch, ws)
            stop_and_drain(ws)
    receipts = [r.getMessage() for r in caplog.records
                if "/v1/audio/transcriptions/stream" in r.getMessage()
                and "═" in r.getMessage()]
    assert len(receipts) >= 2, receipts
    skipped = receipts[-1]
    assert "[!] no final decode" in skipped
    assert "near-silence gate skipped the final decode" in skipped
    assert "empty output" not in skipped
    assert "thresholds suppressed" not in skipped
    assert "no final decode" not in receipts[0]


def test_revoked_identity_gets_no_rows_through_the_gate_path(
        client, make_user_key, app_module, monkeypatch, energy_vad):
    """The gate path reaches on_final without decode_final's pre-model
    revocation raise, so on_final checks for itself: a revoked identity must
    write no trace/usage row."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.stats import recent_transcriptions_store
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH

    make_user_key("admin", is_admin=True)
    uid, raw_alice = make_user_key("alice")
    counted = {}

    def revoke():
        counted["before"] = recent_transcriptions_store.count()
        api_keys_store.revoke_user(uid)

    with ready_ws(client, raw_alice) as ws:
        _gate_second_utterance(monkeypatch, ws, before_second=revoke)
        ws.send_bytes(_pcm(0, 200))
        try:
            ws.send_json({"type": "stop"})
        except Exception:  # noqa: BLE001 — already closed
            pass
        msgs, code = _drain(ws)
    assert code == _WS_UNAUTH, (code, msgs)
    assert recent_transcriptions_store.count() == counted["before"], \
        "the revoked identity still got a trace/usage row through the gate path"


def test_translate_expect_is_ignored_when_translation_is_disabled(
        client, app_module, monkeypatch, energy_vad):
    """/v1/text/translations refuses before it can claim a receipt, so a
    declaration on a TRANSLATION_ENABLED=0 server must park nothing: the
    receipt is logged inline and no `captured` frame invites a claim."""
    from faster_whisper_backend.captures import store as captures_store
    from faster_whisper_backend.transcription import receipt_hold

    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", False, raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True, raising=False)
    monkeypatch.setattr(captures_store, "create_capture", lambda **kw: "cap-test-id")

    msgs = dictate_session(client, translate_expect={"targets": ["en"], "per_utterance": True})
    assert any(m["type"] == "final" for m in msgs)
    assert not any(m.get("type") == "captured" for m in msgs), msgs
    assert receipt_hold.pending() == 0


def test_translate_expect_parks_when_translation_is_enabled(
        client, app_module, monkeypatch, energy_vad):
    """The other direction, so the gate above is not just disabling the hold."""
    from faster_whisper_backend.captures import store as captures_store
    from faster_whisper_backend.transcription import receipt_hold

    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True, raising=False)
    monkeypatch.setattr(captures_store, "create_capture", lambda **kw: "cap-test-id")

    msgs = dictate_session(client, translate_expect={"targets": ["en"], "per_utterance": True})
    assert any(m.get("type") == "captured" for m in msgs), msgs
    assert receipt_hold.pending() >= 1


def test_a_full_capture_store_still_records_and_leaves_eviction_to_the_store(
        client, app_module, monkeypatch, energy_vad, created_captures):
    """At CAPTURES_MAX the gate still hands the utterance to create_capture:
    its _evict_to_cap does the documented oldest-first rotation, and refusing
    first would silently record nothing from then on."""
    from faster_whisper_backend.captures import store as captures_store

    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_MAX", 10, raising=False)
    monkeypatch.setattr(captures_store, "count_evictable", lambda: 10)
    # Not a store full of "ready" rows (the real DB state must not decide).
    monkeypatch.setattr(captures_store, "ready_fills_cap", lambda: False)

    msgs = dictate_session(client)
    assert any(m["type"] == "final" for m in msgs)
    assert len(created_captures) == 1


def test_a_store_full_of_ready_captures_skips_the_stream_capture(
        client, app_module, monkeypatch, energy_vad, created_captures):
    """Ready rows alone fill CAPTURES_MAX: the new row would be evicted by its
    own insert, so the utterance is not handed to create_capture (no WAV)."""
    from faster_whisper_backend.captures import store as captures_store

    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True, raising=False)
    monkeypatch.setattr(captures_store, "ready_fills_cap", lambda: True)

    msgs = dictate_session(client)
    assert any(m["type"] == "final" for m in msgs)
    assert created_captures == []


def test_turning_capture_recording_off_mid_session_stops_recording(
        client, app_module, monkeypatch, energy_vad, created_captures):
    """CAPTURES_RECORDING_ENABLED is a privacy switch: switched off while a
    dictation socket is open, the next utterance on that socket is not
    written to the captures store."""
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True, raising=False)

    with ready_ws(client) as ws:
        monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", False,
                            raising=False)
        msgs, _code = speak_and_stop(ws)
    assert any(m["type"] == "final" for m in msgs)
    assert created_captures == []


@pytest.fixture
def ws_sends(monkeypatch):
    """Every ASGI message the server-side WebSocket sends, as
    ``(socket id, type, close code)``. Starlette's TestClient reports a
    PRE-accept close as the same WebSocketDisconnect(code, reason) a
    post-accept one raises, so the close code arriving proves nothing about
    the accept — a real server answers a pre-accept close with a bare HTTP 403
    and the browser never sees the code. This spy is the proof instead."""
    from starlette.websockets import WebSocket

    sent: list[tuple[int, str, "int | None"]] = []
    real_send = WebSocket.send

    async def _spy(self, message):
        sent.append((id(self), message["type"], message.get("code")))
        return await real_send(self, message)

    monkeypatch.setattr(WebSocket, "send", _spy)
    return sent


def _assert_accepted_before_close(sent, code):
    closes = [i for i, (_sid, typ, c) in enumerate(sent)
              if typ == "websocket.close" and c == code]
    assert closes, sent
    sid = sent[closes[0]][0]
    assert any(s == sid and typ == "websocket.accept"
               for s, typ, _c in sent[:closes[0]]), sent


def test_disabled_refusal_closes_after_accept_with_4503(client, app_module, monkeypatch, ws_sends):
    from faster_whisper_backend.streaming.routes import _WS_DISABLED

    monkeypatch.setattr(app_module.cfg, "STREAMING_ENABLED", False, raising=False)
    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect(_STREAM_URL) as ws:
            ws.receive_json()
    assert ei.value.code == _WS_DISABLED
    assert "disabled" in (ei.value.reason or "")
    _assert_accepted_before_close(ws_sends, _WS_DISABLED)


def test_unauth_refusal_closes_after_accept_with_4401(app_module, ws_sends):
    from faster_whisper_backend.streaming.routes import _WS_UNAUTH

    with TestClient(app_module.app, client=("203.0.113.9", 1234)) as c:
        with pytest.raises(WebSocketDisconnect) as ei:
            with c.websocket_connect(_STREAM_URL) as ws:
                ws.receive_json()
    assert ei.value.code == _WS_UNAUTH
    _assert_accepted_before_close(ws_sends, _WS_UNAUTH)


def test_origin_refusal_closes_after_accept_with_4403(client, app_module, ws_sends):
    from faster_whisper_backend.streaming.routes import _WS_BAD_ORIGIN

    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect(
                _STREAM_URL, headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
    assert ei.value.code == _WS_BAD_ORIGIN
    _assert_accepted_before_close(ws_sends, _WS_BAD_ORIGIN)


def test_cap_refusal_closes_after_accept_with_4429(client, app_module, monkeypatch, ws_sends):
    from faster_whisper_backend.streaming.routes import _WS_TOO_MANY

    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS", 1, raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_MAX_SESSIONS_PER_USER", 0, raising=False)
    with ready_ws(client):
        with pytest.raises(WebSocketDisconnect) as ei:
            with client.websocket_connect(_STREAM_URL) as ws2:
                ws2.receive_json()
        assert ei.value.code == _WS_TOO_MANY
        assert "live sessions" in (ei.value.reason or "")
    _assert_accepted_before_close(ws_sends, _WS_TOO_MANY)


def test_condition_on_previous_text_override_is_reported_ignored(client, app_module):
    with client.websocket_connect(_STREAM_URL) as ws:
        ready = _config(ws, decode_overrides={"condition_on_previous_text": True})
        assert ready["type"] == "ready"
        assert "condition_on_previous_text" in ready.get("overrides_ignored", [])


def test_null_condition_on_previous_text_is_not_reported_ignored(client, app_module):
    """null is the client's "inherit": the assembler skips it, so it overrode
    nothing and must not be listed as an ignored override."""
    with client.websocket_connect(_STREAM_URL) as ws:
        ready = _config(ws, decode_overrides={"condition_on_previous_text": None})
        assert ready["type"] == "ready"
        assert "condition_on_previous_text" not in ready.get("overrides_ignored", [])


def test_multilingual_with_a_set_language_is_reported_ignored(client, app_module,
                                                             monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DEFAULT_LANGUAGE", "", raising=False)
    with client.websocket_connect(_STREAM_URL) as ws:
        ready = _config(ws, language="de", decode_overrides={"multilingual": True})
        assert "multilingual" in ready.get("overrides_ignored", [])
    with client.websocket_connect(_STREAM_URL) as ws:
        ready = _config(ws, decode_overrides={"multilingual": True})
        assert "multilingual" not in ready.get("overrides_ignored", [])


def test_multilingual_override_reaches_the_final_decode(app_module, fake_model):
    from faster_whisper_backend.streaming import routes as sr
    # The handshake drops any override not in _CLIENT_OVERRIDE_KEYS, so
    # multilingual must be one of them.
    assert "multilingual" in sr._CLIENT_OVERRIDE_KEYS
    kw = sr._build_transcribe_kwargs(
        "some-model", final=True, prompt="", want_words=False,
        language="", model_obj=fake_model, overrides={"multilingual": True})
    assert kw["multilingual"] is True


def test_running_dictate_job_names_its_model(client, app_module):
    from faster_whisper_backend.core import jobs

    with ready_ws(client):
        rows = [r for r in jobs.jobs_snapshot() if r["kind"] == "dictate"]
        assert rows, "no running dictate job"
        assert rows[0]["model"] == tx_models._resolve_model_name("whisper-1")


def test_queue_item_cap_sheds_tiny_frames_and_flushes(
        client, app_module, monkeypatch, caplog, energy_vad):
    """Two-byte PCM frames alternated with flushes queue two items per ~0 bytes,
    so the byte cap never trips; the item cap must shed them instead."""
    from faster_whisper_backend.streaming import routes as streaming_routes
    from faster_whisper_backend.streaming import session as streaming_session

    monkeypatch.setattr(streaming_routes, "_HARD_CAP_ITEMS", 32)
    orig_flush = streaming_session.StreamSession.flush_utterance
    held = []
    released = threading.Event()

    async def slow_flush(self):
        # Hold the pump on the first flush so everything after it queues —
        # until the test has seen the cap trip (no fixed sleep to race).
        if not held:
            held.append(True)
            while not released.is_set():
                await asyncio.sleep(0.01)
        await orig_flush(self)

    monkeypatch.setattr(streaming_session.StreamSession, "flush_utterance", slow_flush)
    with caplog.at_level(logging.WARNING, logger="faster_whisper_backend.streaming.routes"):
        with ready_ws(client) as ws:
            ws.send_bytes(_pcm(8000, 200))
            ws.send_json({"type": "flush"})   # pump held here
            for _ in range(100):
                ws.send_bytes(b"\x00\x00")
                ws.send_json({"type": "flush"})
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not any(
                    "audio backlog over cap" in r.getMessage() for r in caplog.records):
                time.sleep(0.02)
            released.set()
            stop_and_drain(ws)
    assert any("audio backlog over cap" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]
