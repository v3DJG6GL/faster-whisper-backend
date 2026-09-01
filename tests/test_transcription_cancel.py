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
        async def _cancelled(path, **kw):
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


# --- recorded status ---------------------------------------------------------

def test_cancelled_run_records_status_cancelled(client, app_module,
                                                monkeypatch):
    # The 499 raised by the _ClientCancelled arm lands in the OUTER except
    # Exception, which must not clobber the "cancelled" status the inner arm
    # set — /stats would otherwise count user cancels as server failures.
    recorded = []
    _orig = app_module.metrics.record_transcription

    def _spy(**kw):
        recorded.append(kw)
        return _orig(**kw)
    monkeypatch.setattr(app_module.metrics, "record_transcription", _spy)
    app_module._BATCH_CANCELLED.add(_PID)
    try:
        r = _post(client, progress_id=_PID)
        assert r.status_code == 499, r.text
    finally:
        app_module._BATCH_CANCELLED.discard(_PID)
    assert recorded and recorded[-1]["status"] == "cancelled"


# --- early-reject orphan entries ---------------------------------------------

def test_validation_reject_leaves_no_progress_entry(client, app_module):
    # A 422 on `task` must not leak a "waiting" entry (pollable forever) or
    # a cancellable id — the seed happens inside the try whose finally pops.
    r = _post(client, task="nope", progress_id=_PID)
    assert r.status_code == 422
    assert _PID not in app_module._BATCH_PROGRESS
    assert client.get(
        f"/v1/audio/transcriptions/progress/{_PID}").json() == {
            "stage": "unknown"}
    r = client.post(f"/v1/audio/transcriptions/cancel/{_PID}")
    assert r.json() == {"cancelled": False}
    assert _PID not in app_module._BATCH_CANCELLED


# --- owner binding -----------------------------------------------------------

def test_progress_and_cancel_are_owner_bound(client, app_module,
                                             make_user_key):
    from tests.conftest import bearer
    uid_alice, raw_alice = make_user_key("alice")
    _, raw_bob = make_user_key("bob")
    _, raw_admin = make_user_key("root", is_admin=True)
    app_module._BATCH_PROGRESS[_PID] = {
        "stage": "transcribing", "owner": uid_alice, "updated": 0}
    try:
        # Another user's id reads exactly like a miss — no existence oracle.
        r = client.get(f"/v1/audio/transcriptions/progress/{_PID}",
                       headers=bearer(raw_bob))
        assert r.json() == {"stage": "unknown"}
        r = client.post(f"/v1/audio/transcriptions/cancel/{_PID}",
                        headers=bearer(raw_bob))
        assert r.json() == {"cancelled": False}
        assert _PID not in app_module._BATCH_CANCELLED
        # The owner sees and cancels their own run.
        r = client.get(f"/v1/audio/transcriptions/progress/{_PID}",
                       headers=bearer(raw_alice))
        assert r.json()["stage"] == "transcribing"
        # An admin may cancel anyone's run (the /stats activity popover).
        r = client.post(f"/v1/audio/transcriptions/cancel/{_PID}",
                        headers=bearer(raw_admin))
        assert r.json() == {"cancelled": True}
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)
        app_module._BATCH_CANCELLED.discard(_PID)


# --- job mirror stage transitions --------------------------------------------

def test_stage_transition_clears_mirrored_progress(app_module):
    import jobs
    pid = "feedf00d" * 4
    jid = jobs.job_start("transcribe", id="job-mirror-test", model="m")
    app_module._JOB_BY_PID[pid] = "job-mirror-test"
    try:
        app_module._progress_set(pid, stage="transcribing", progress=0.97,
                                 step="dec", model="large-v3")
        row = jobs.jobs_snapshot()[0]
        assert row["progress"] == 0.97 and row["step"] == "dec"
        # A stage transition resets the derived columns the new stage did not
        # report — no more stale ~100% bar during diarizing/translating.
        app_module._progress_set(pid, stage="diarizing", progress=None,
                                 step=None, model=None)
        row = jobs.jobs_snapshot()[0]
        assert row["stage"] == "diarizing"
        assert row["progress"] is None and row["step"] is None
        assert row["model"] is None
        # Same-stage ticks must NOT clear what they don't carry.
        app_module._progress_set(pid, stage="diarizing", step="embedding")
        app_module._progress_set(pid, stage="diarizing", progress=0.5)
        row = jobs.jobs_snapshot()[0]
        assert row["step"] == "embedding" and row["progress"] == 0.5
    finally:
        app_module._JOB_BY_PID.pop(pid, None)
        app_module._BATCH_PROGRESS.pop(pid, None)
        jobs.job_end("job-mirror-test")
