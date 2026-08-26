"""Cooperative cancellation of in-flight batch transcriptions: the cancel
endpoint flags a progress id, and the handler's stage checks abort the
request with 499 instead of soft-failing onward."""

import bgm_separation
import diarization

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
_PID = "cafe" * 8  # 32 hex chars — passes _PROGRESS_ID_RE


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


# --- the cancel endpoint -----------------------------------------------------

def test_cancel_malformed_id_is_422(client):
    assert client.post("/v1/audio/transcriptions/cancel/NOPE").status_code == 422


def test_cancel_unknown_id_is_a_noop(client, app_module):
    r = client.post(f"/v1/audio/transcriptions/cancel/{_PID}")
    assert r.status_code == 200
    assert r.json() == {"cancelled": False}
    assert _PID not in app_module._BATCH_CANCELLED


def test_cancel_flags_an_in_flight_id(client, app_module):
    app_module._BATCH_PROGRESS[_PID] = {"stage": "transcribing", "updated": 0}
    try:
        r = client.post(f"/v1/audio/transcriptions/cancel/{_PID}")
        assert r.status_code == 200
        assert r.json() == {"cancelled": True}
        assert _PID in app_module._BATCH_CANCELLED
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)
        app_module._BATCH_CANCELLED.discard(_PID)


# --- the handler's cooperative abort -----------------------------------------

def test_pre_flagged_request_aborts_with_499(client, app_module):
    # The decode-loop check fires on the first segment: a request whose id is
    # already flagged never returns a transcript. (In real use the flag lands
    # mid-flight; pre-flagging just makes the race deterministic.)
    app_module._BATCH_CANCELLED.add(_PID)
    try:
        r = _post(client, progress_id=_PID)
        assert r.status_code == 499, r.text
        # The finally cleaned both registries.
        assert _PID not in app_module._BATCH_CANCELLED
        assert _PID not in app_module._BATCH_PROGRESS
    finally:
        app_module._BATCH_CANCELLED.discard(_PID)


def test_bgm_cancel_is_not_a_soft_fail(client, app_module, monkeypatch):
    # BgmCancelled must abort the request — never the "transcribing the
    # original audio" warning path that BgmSeparationError takes.
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        async def _cancelled(path, *, progress_cb=None, cancel_check=None):
            raise bgm_separation.BgmCancelled()
        monkeypatch.setattr(bgm_separation, "separate", _cancelled)
        r = _post(client, separate_bgm="1", progress_id=_PID)
        assert r.status_code == 499, r.text
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_diarize_cancel_is_not_a_soft_fail(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        async def _cancelled(path, **kw):
            raise diarization.DiarizeCancelled()
        monkeypatch.setattr(diarization, "diarize", _cancelled)
        r = _post(client, diarize="1", progress_id=_PID)
        assert r.status_code == 499, r.text
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False
