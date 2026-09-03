"""Cooperative cancellation of in-flight batch transcriptions: the cancel
endpoint flags a progress id, and the handler's stage checks abort the
request with 499 instead of soft-failing onward."""

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import diarization

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
    from faster_whisper_backend.core import jobs
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


# --- a cancel that lands INSIDE a soft-fail stage -----------------------------
# Every post-decode stage polls _check_cancelled inside its own try, and that
# raises _ClientCancelled — a plain Exception the stage's trailing soft-fail
# arm used to swallow as "<stage> failed", turning a cancel into a 200 with a
# false warning (or, for separation, a bogus error log before the next check
# finally aborted).

def _flag_on(app_module, monkeypatch, pid, pred):
    """Flag `pid` cancelled the moment a _progress_set call matches `pred`."""
    orig = app_module._progress_set

    def spy(p, **fields):
        orig(p, **fields)
        if p == pid and pred(fields):
            app_module._BATCH_CANCELLED.add(pid)
    monkeypatch.setattr(app_module, "_progress_set", spy)


def test_cancel_inside_separation_stage_is_not_a_soft_fail(
        client, app_module, monkeypatch, caplog):
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    _flag_on(app_module, monkeypatch, _PID,
             lambda f: f.get("stage") == "separating")

    async def _never(path, **kw):
        raise AssertionError("separate() must not run after a cancel")
    monkeypatch.setattr(bgm_separation, "separate", _never)
    try:
        r = _post(client, separate_bgm="1", progress_id=_PID)
        assert r.status_code == 499, r.text
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False
        app_module._BATCH_CANCELLED.discard(_PID)
    assert "[bgm] unexpected failure" not in caplog.text


def test_cancel_inside_diarization_stage_is_not_a_soft_fail(
        client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    _flag_on(app_module, monkeypatch, _PID,
             lambda f: f.get("stage") == "diarizing")

    async def _never(path, **kw):
        raise AssertionError("diarize() must not run after a cancel")
    monkeypatch.setattr(diarization, "diarize", _never)
    try:
        r = _post(client, diarize="1", progress_id=_PID)
        assert r.status_code == 499, r.text
        assert "diarization failed" not in r.text
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False
        app_module._BATCH_CANCELLED.discard(_PID)


def test_cancel_inside_translation_stage_is_not_a_soft_fail(
        client, app_module, monkeypatch):
    from faster_whisper_backend.audio import translation
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    # The decode loop's own check runs BEFORE each segment's progress tick,
    # so a flag raised on the (only) segment's tail is first seen by the
    # translation stage's in-try check.
    _flag_on(app_module, monkeypatch, _PID,
             lambda f: f.get("last_text") is not None)

    async def _never(*args, **kw):
        raise AssertionError("translate_segments() must not run after a cancel")
    monkeypatch.setattr(translation, "translate_segments", _never)
    try:
        r = _post(client, translate_to="en", progress_id=_PID)
        assert r.status_code == 499, r.text
        assert "translation failed" not in r.text
    finally:
        app_module._BATCH_CANCELLED.discard(_PID)


# --- colliding progress ids ---------------------------------------------------

def test_in_flight_progress_id_is_treated_as_absent(client, app_module):
    # A client-chosen id that is already live belongs to another request:
    # this one gets no progress rail instead of sharing (and popping) the
    # other's entry and cancel flag.
    app_module._BATCH_PROGRESS[_PID] = {
        "stage": "transcribing", "owner": "other", "updated": 0}
    try:
        r = _post(client, progress_id=_PID)
        assert r.status_code == 200, r.text
        entry = app_module._BATCH_PROGRESS.get(_PID)
        assert entry is not None
        assert entry["owner"] == "other" and entry["stage"] == "transcribing"
        assert _PID not in app_module._JOB_BY_PID
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)
        app_module._PROGRESS_OWNER.pop(_PID, None)


def test_text_translations_in_flight_progress_id_is_treated_as_absent(
        client, app_module, monkeypatch):
    from faster_whisper_backend.audio import translation
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_DEFAULT_MODEL",
                        "org/default-GGUF:Q4", raising=False)

    async def _fake(segments, targets, **kw):
        return ([{t: "x" for t in targets} for _ in segments], [],
                {"model": "org/default-GGUF:Q4", "source": "de",
                 "mode": "fluent"})
    monkeypatch.setattr(translation, "translate_segments", _fake)
    app_module._BATCH_PROGRESS[_PID] = {
        "stage": "transcribing", "owner": "other", "updated": 0}
    try:
        r = client.post("/v1/text/translations", json={
            "segments": [{"text": "hallo"}], "targets": ["en"],
            "progress_id": _PID})
        assert r.status_code == 200, r.text
        entry = app_module._BATCH_PROGRESS.get(_PID)
        assert entry is not None
        assert entry["owner"] == "other" and entry["stage"] == "transcribing"
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)
        app_module._PROGRESS_OWNER.pop(_PID, None)


# --- owner survives a re-created entry ---------------------------------------

def test_recreated_progress_entry_keeps_its_owner(client, app_module,
                                                  make_user_key):
    from tests.conftest import bearer
    uid_alice, _ = make_user_key("alice")
    _, raw_bob = make_user_key("bob")
    make_user_key("root", is_admin=True)  # locks the app down: bob is bob
    try:
        app_module._progress_set(_PID, stage="waiting", owner=uid_alice)
        # An executor-thread stage tick after the handler's finally popped
        # the entry (or after the cap eviction took it) re-creates it —
        # with the owner stamp, not as an open, owner-less entry.
        app_module._BATCH_PROGRESS.pop(_PID)
        app_module._progress_set(_PID, progress=0.5, last_text="x")
        assert app_module._BATCH_PROGRESS[_PID]["owner"] == uid_alice
        r = client.get(f"/v1/audio/transcriptions/progress/{_PID}",
                       headers=bearer(raw_bob))
        assert r.json() == {"stage": "unknown"}
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)
        app_module._PROGRESS_OWNER.pop(_PID, None)


def test_owner_map_is_pruned_with_the_stale_sweep(app_module):
    app_module._progress_set(_PID, stage="waiting", owner="u1")
    try:
        app_module._BATCH_PROGRESS[_PID]["updated"] = -1e9
        # Any entry creation runs the stale sweep; it must prune both maps.
        app_module._progress_set("d00d" * 8, stage="waiting")
        assert _PID not in app_module._BATCH_PROGRESS
        assert _PID not in app_module._PROGRESS_OWNER
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)
        app_module._BATCH_PROGRESS.pop("d00d" * 8, None)
        app_module._PROGRESS_OWNER.pop(_PID, None)


# --- the first seed is not a stage transition --------------------------------

def test_first_seed_keeps_job_start_model(app_module):
    from faster_whisper_backend.core import jobs
    pid = "f00dfeed" * 4
    jobs.job_start("transcribe", id="job-seed-test", model="large-v3")
    app_module._JOB_BY_PID[pid] = "job-seed-test"
    try:
        # No previous stage → nothing stale to clear; job_start's model must
        # survive the whole pre-decode phase (waiting/downloading/...).
        app_module._progress_set(pid, stage="waiting", progress=None)
        row = jobs.jobs_snapshot()[0]
        assert row["stage"] == "waiting" and row["model"] == "large-v3"
        # A real transition still clears the previous stage's columns.
        app_module._progress_set(pid, stage="transcribing", progress=0.9,
                                 step="dec")
        app_module._progress_set(pid, stage="diarizing")
        row = jobs.jobs_snapshot()[0]
        assert row["stage"] == "diarizing"
        assert row["progress"] is None and row["step"] is None
        assert row["model"] is None
    finally:
        app_module._JOB_BY_PID.pop(pid, None)
        app_module._BATCH_PROGRESS.pop(pid, None)
        jobs.job_end("job-seed-test")


def test_stage_transition_clears_mirrored_total_bytes(app_module):
    from faster_whisper_backend.core import jobs
    pid = "ba5eba11" * 4
    jobs.job_start("transcribe", id="job-bytes-test", model="m")
    app_module._JOB_BY_PID[pid] = "job-bytes-test"
    try:
        app_module._progress_set(pid, stage="downloading", progress=0.5,
                                 total_bytes=4096)
        assert jobs.jobs_snapshot()[0]["total_bytes"] == 4096
        # The download's byte total is a per-stage column too — it must not
        # stick to the row through transcribing/diarizing/translating.
        app_module._progress_set(pid, stage="transcribing", progress=None,
                                 total_bytes=None)
        assert jobs.jobs_snapshot()[0]["total_bytes"] is None
    finally:
        app_module._JOB_BY_PID.pop(pid, None)
        app_module._BATCH_PROGRESS.pop(pid, None)
        jobs.job_end("job-bytes-test")


def test_task_cancellation_records_status_cancelled(client, app_module,
                                                    monkeypatch):
    # A client disconnect / lifespan shutdown unwinds the handler with
    # asyncio.CancelledError — a BaseException that skips every
    # `except Exception` arm. The finally still records the run, and it must
    # say "cancelled", not the seeded "ok" with zero words.
    import asyncio

    import pytest

    async def _cancelled_loader(name, *, lease=False):
        raise asyncio.CancelledError()
    monkeypatch.setattr(app_module, "_get_or_load_model", _cancelled_loader)
    recorded = []
    _orig = app_module.metrics.record_transcription

    def _spy(**kw):
        recorded.append(kw)
        return _orig(**kw)
    monkeypatch.setattr(app_module.metrics, "record_transcription", _spy)
    with pytest.raises(BaseException):
        _post(client)
    assert recorded and recorded[-1]["status"] == "cancelled"


def test_cancelled_request_is_classified_cancelled(client, app_module):
    from faster_whisper_backend.stats import recent_transcriptions_store
    app_module._BATCH_CANCELLED.add(_PID)
    try:
        r = _post(client, progress_id=_PID)
        assert r.status_code == 499, r.text
    finally:
        app_module._BATCH_CANCELLED.discard(_PID)
    row = recent_transcriptions_store.list_recent(limit=1)[0]
    assert row["status"] == "cancelled"
    assert row["error_class"] == "cancelled"
