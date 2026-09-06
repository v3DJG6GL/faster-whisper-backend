"""GET/DELETE /v1/jobs* — the durable job resource for batch runs posted
with a progress_id (core/jobs_store.py)."""

import pytest

from faster_whisper_backend.core import jobs_store as js
from tests.conftest import bearer

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
_PID = "cafe" * 8


def _post(client, headers=None, **data):
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data,
                       headers=headers or {})


# --- rows land on every POST path ------------------------------------------

@pytest.mark.parametrize("fmt", ["json", "text", "verbose_json"])
def test_post_with_progress_id_stores_the_verbatim_payload(client, fmt):
    r = _post(client, progress_id=_PID, response_format=fmt)
    assert r.status_code == 200, r.text
    j = client.get(f"/v1/jobs/{_PID}")
    assert j.status_code == 200, j.text
    row = j.json()
    assert row["state"] == "done" and row["result_available"] is True
    assert row["kind"] == "transcribe" and row["task"] == "transcribe"
    assert row["source_kind"] == "file" and row["source_name"] == "a.wav"
    assert row["response_format"] == fmt and row["error"] is None
    assert row["progress"] is None  # the run closed
    assert row["finished_at"] >= row["created_at"]
    res = client.get(f"/v1/jobs/{_PID}/result")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    assert res.json() == r.json()


def test_post_without_progress_id_creates_no_row(client):
    assert _post(client).status_code == 200
    assert js.count() == 0
    assert client.get("/v1/jobs").json() == {"jobs": []}


def test_audio_translations_forward_lands_with_task_translate(client):
    r = client.post("/v1/audio/translations", files=_FILE,
                    data={"model": "whisper-1", "progress_id": _PID})
    assert r.status_code == 200, r.text
    row = client.get(f"/v1/jobs/{_PID}").json()
    assert row["kind"] == "transcribe" and row["task"] == "translate"


def test_failed_run_stores_the_error_and_no_result(client, fake_model):
    def _boom(path, **kw):
        raise RuntimeError("gpu on fire")
    fake_model.transcribe = _boom
    r = _post(client, progress_id=_PID, response_format="verbose_json")
    assert r.status_code == 500
    row = client.get(f"/v1/jobs/{_PID}").json()
    assert row["state"] == "failed" and row["error"] == "transcription failed"
    assert row["result_available"] is False
    assert client.get(f"/v1/jobs/{_PID}/result").status_code == 404


def test_validation_4xx_after_the_seed_stores_the_curated_detail(
        client, app_module, monkeypatch):
    # The upload ceiling is checked AFTER the progress seed (the seed sits
    # right above it), so the rejected run has a row carrying the detail.
    monkeypatch.setattr(app_module.cfg, "MEDIA_MAX_BYTES", 1, raising=False)
    r = _post(client, progress_id=_PID)
    assert r.status_code == 413
    row = client.get(f"/v1/jobs/{_PID}").json()
    assert row["state"] == "failed"
    assert row["error"] == "upload too large"


def test_cancelled_run_lands_as_cancelled(client, app_module):
    app_module._BATCH_CANCELLED.add(_PID)
    try:
        assert _post(client, progress_id=_PID).status_code == 499
    finally:
        app_module._BATCH_CANCELLED.discard(_PID)
    row = client.get(f"/v1/jobs/{_PID}").json()
    assert row["state"] == "cancelled" and row["error"] is None
    assert client.get(f"/v1/jobs/{_PID}/result").status_code == 404


def test_text_translation_run_gets_a_translate_row(client, app_module,
                                                    monkeypatch):
    from tests.audio.test_routes_text_translations import (
        _body, _enable, _stub_translate)
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    r = client.post("/v1/text/translations",
                    json=_body(targets=["en", "fr"], progress_id=_PID))
    assert r.status_code == 200, r.text
    row = client.get(f"/v1/jobs/{_PID}").json()
    assert row["kind"] == "translate" and row["source_kind"] == "text"
    assert row["source_name"] == "2 segments → en,fr"
    assert row["state"] == "done"
    assert client.get(f"/v1/jobs/{_PID}/result").json() == r.json()


def test_text_translation_failure_records_failed(client, app_module,
                                                  monkeypatch):
    from faster_whisper_backend.audio import translation
    from tests.audio.test_routes_text_translations import _body, _enable
    _enable(app_module, monkeypatch)

    async def _fail(*a, **kw):
        raise translation.TranslationError("model refused")
    monkeypatch.setattr(translation, "translate_segments", _fail)
    r = client.post("/v1/text/translations", json=_body(progress_id=_PID))
    assert r.status_code == 400
    row = client.get(f"/v1/jobs/{_PID}").json()
    assert row["state"] == "failed" and row["error"] == "model refused"


# --- the live merge ------------------------------------------------------------

def _seed_running(app_module, pid=_PID, owner=None):
    js.start(job_id=pid, request_id="req", kind="transcribe",
             user_id=owner, key_id=None, model="whisper-1",
             source_kind="file", source_name="a.wav", ttl_s=3600,
             max_rows=100, max_bytes=0)
    entry = {"stage": "transcribing", "progress": 0.4, "updated": 0}
    if owner:
        entry["owner"] = owner
    app_module._BATCH_PROGRESS[pid] = entry


def test_running_row_merges_live_progress_and_result_is_409(client, app_module):
    _seed_running(app_module)
    try:
        row = client.get(f"/v1/jobs/{_PID}").json()
        assert row["state"] == "running" and row["result_available"] is False
        assert row["progress"]["stage"] == "transcribing"
        assert row["progress"]["progress"] == 0.4
        assert "plan" in row["progress"]
        r = client.get(f"/v1/jobs/{_PID}/result")
        assert r.status_code == 409
        assert r.json()["detail"] == "job still running"
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)


def test_delete_running_flags_cancel_then_the_post_lands_cancelled(client,
                                                                     app_module):
    _seed_running(app_module)
    try:
        r = client.delete(f"/v1/jobs/{_PID}")
        assert r.json() == {"cancelled": True}
        assert _PID in app_module._BATCH_CANCELLED
        # The cooperative flag is what the handler polls: with it set, the
        # (re)posted run aborts and the row lands `cancelled`.
        app_module._BATCH_PROGRESS.pop(_PID, None)
        assert _post(client, progress_id=_PID).status_code == 499
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)
        app_module._BATCH_CANCELLED.discard(_PID)
    assert client.get(f"/v1/jobs/{_PID}").json()["state"] == "cancelled"


def test_delete_running_without_a_live_entry_answers_false(client, app_module):
    js.start(job_id=_PID, request_id="req", kind="transcribe", user_id=None,
             key_id=None, model="m", source_kind="file", source_name="a",
             ttl_s=3600, max_rows=100, max_bytes=0)
    assert client.delete(f"/v1/jobs/{_PID}").json() == {"cancelled": False}
    assert _PID not in app_module._BATCH_CANCELLED


def test_delete_finished_removes_the_row(client):
    assert _post(client, progress_id=_PID).status_code == 200
    assert client.delete(f"/v1/jobs/{_PID}").json() == {"deleted": True}
    assert client.get(f"/v1/jobs/{_PID}").status_code == 404
    assert client.delete(f"/v1/jobs/{_PID}").status_code == 404


# --- gates, ownership, listing ----------------------------------------------

def test_malformed_id_is_422_and_unknown_is_404(client):
    assert client.get("/v1/jobs/NOPE").status_code == 422
    assert client.get(f"/v1/jobs/{_PID}").status_code == 404
    assert client.get(f"/v1/jobs/{_PID}/result").status_code == 404
    assert client.delete(f"/v1/jobs/{_PID}").status_code == 404


def test_disabled_feature_is_403_and_writes_no_rows(client, app_module,
                                                     monkeypatch):
    monkeypatch.setattr(app_module.cfg, "JOBS_ENABLED", False, raising=False)
    assert _post(client, progress_id=_PID).status_code == 200
    assert js.count() == 0
    for r in (client.get("/v1/jobs"), client.get(f"/v1/jobs/{_PID}"),
              client.get(f"/v1/jobs/{_PID}/result"),
              client.delete(f"/v1/jobs/{_PID}")):
        assert r.status_code == 403
    assert client.get("/v1/me").json()["jobs_enabled"] is False


def test_me_reports_jobs_caps(client, app_module, monkeypatch):
    j = client.get("/v1/me").json()
    assert j["jobs_enabled"] is True and j["jobs"] == {"ttl_s": 259200}
    monkeypatch.setattr(app_module.cfg, "JOBS_ENABLED", False, raising=False)
    j = client.get("/v1/me").json()
    assert j["jobs_enabled"] is False and "jobs" not in j


def test_expired_row_reads_as_unknown(client):
    js.start(job_id=_PID, request_id="req", kind="transcribe", user_id=None,
             key_id=None, model="m", source_kind="file", source_name="a",
             ttl_s=0.0, max_rows=100, max_bytes=0)
    import time; time.sleep(0.01)
    assert client.get(f"/v1/jobs/{_PID}").status_code == 404
    assert client.get("/v1/jobs").json() == {"jobs": []}


def test_list_filters_state_and_rejects_unknown_state(client, app_module,
                                                      monkeypatch):
    assert _post(client, progress_id=_PID).status_code == 200
    monkeypatch.setattr(app_module.cfg, "MEDIA_MAX_BYTES", 1, raising=False)
    assert _post(client, progress_id="b" * 32).status_code == 413
    monkeypatch.undo()
    rows = client.get("/v1/jobs").json()["jobs"]
    assert [r["job_id"] for r in rows] == ["b" * 32, _PID]  # newest first
    assert "result_json" not in rows[0] and "progress" not in rows[0]
    done = client.get("/v1/jobs?state=done").json()["jobs"]
    assert [r["job_id"] for r in done] == [_PID]
    assert client.get("/v1/jobs?state=weird").status_code == 422
    assert len(client.get("/v1/jobs?limit=1").json()["jobs"]) == 1


def test_rows_are_owner_bound_and_admins_read_all(client, make_user_key):
    _, admin = make_user_key("root", is_admin=True)
    _, alice = make_user_key("alice")
    _, bob = make_user_key("bob")
    assert _post(client, bearer(alice), progress_id=_PID).status_code == 200
    # Foreign caller: the row reads like a miss, on every verb.
    assert client.get(f"/v1/jobs/{_PID}", headers=bearer(bob)).status_code == 404
    assert client.get(f"/v1/jobs/{_PID}/result", headers=bearer(bob)).status_code == 404
    assert client.delete(f"/v1/jobs/{_PID}", headers=bearer(bob)).status_code == 404
    assert client.get("/v1/jobs", headers=bearer(bob)).json() == {"jobs": []}
    # The owner and an admin see it; only `?all=1` lists it for the admin.
    assert client.get(f"/v1/jobs/{_PID}", headers=bearer(alice)).status_code == 200
    assert client.get(f"/v1/jobs/{_PID}", headers=bearer(admin)).status_code == 200
    assert client.get("/v1/jobs", headers=bearer(admin)).json() == {"jobs": []}
    rows = client.get("/v1/jobs?all=1", headers=bearer(admin)).json()["jobs"]
    assert [r["job_id"] for r in rows] == [_PID]
    # A non-admin's ?all=1 is silently their own list.
    assert client.get("/v1/jobs?all=1", headers=bearer(bob)).json() == {"jobs": []}
    # Owner-less rows (open mode) never leak to a keyed caller.
    js.start(job_id="d" * 32, request_id="req", kind="transcribe",
             user_id=None, key_id=None, model="m", source_kind="file",
             source_name="a", ttl_s=3600, max_rows=100, max_bytes=0)
    assert client.get(f"/v1/jobs/{'d' * 32}", headers=bearer(alice)).status_code == 404


def test_rate_limited_answers_429(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "JOBS_RATE_PER_MIN", 1, raising=False)
    assert client.get("/v1/jobs").status_code == 200
    r = client.get("/v1/jobs")
    assert r.status_code == 429 and r.headers.get("retry-after")


# --- stored results and retained media ---------------------------------------

def test_result_drops_dangling_media_ids(client):
    assert _post(client, progress_id=_PID).status_code == 200
    js.finish(job_id=_PID, state="done", ttl_s=3600, result={
        "text": "hallo", "source_media_id": "ab" * 16,
        "source_media_expires_at": 1, "source_video_media_id": "cd" * 16,
        "source_video_expires_at": 2, "source_video_pending": True})
    body = client.get(f"/v1/jobs/{_PID}/result").json()
    assert body == {"text": "hallo"}


def test_result_refreshes_a_live_media_expiry(client, tmp_path):
    from faster_whisper_backend.url import media_store as ums
    src = tmp_path / "x.m4a"; src.write_bytes(b"x" * 10)
    mid = ums.register(str(src), user_id=None)
    assert mid
    assert _post(client, progress_id=_PID).status_code == 200
    js.finish(job_id=_PID, state="done", ttl_s=3600, result={
        "text": "hallo", "source_media_id": mid, "source_media_expires_at": 1})
    body = client.get(f"/v1/jobs/{_PID}/result").json()
    assert body["source_media_id"] == mid
    assert body["source_media_expires_at"] == ums.expires_at_unix(mid)


# --- lifecycle -----------------------------------------------------------------

def test_lifespan_marks_interrupted_runs_failed(client, app_module):
    js.start(job_id=_PID, request_id="req", kind="transcribe", user_id=None,
             key_id=None, model="m", source_kind="file", source_name="a",
             ttl_s=3600, max_rows=100, max_bytes=0)
    from starlette.testclient import TestClient
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as c:
        row = c.get(f"/v1/jobs/{_PID}").json()
    assert row["state"] == "failed" and row["error"] == "server restarted"


def test_closed_progress_id_is_not_resurrected_by_a_late_tick(app_module):
    app_module._BATCH_PROGRESS[_PID] = {"stage": "transcribing", "updated": 0,
                                        "owner": "alice"}
    app_module._PROGRESS_OWNER[_PID] = "alice"
    try:
        app_module._progress_close(_PID)
        assert _PID not in app_module._BATCH_PROGRESS
        # A straggling stage-thread tick after the handler's finally.
        app_module._progress_set(_PID, stage="diarizing", progress=0.5)
        assert _PID not in app_module._BATCH_PROGRESS
        # A fresh owner-stamped seed re-opens the id.
        app_module._progress_set(_PID, stage="waiting", owner="bob")
        assert app_module._BATCH_PROGRESS[_PID]["owner"] == "bob"
        assert _PID not in app_module._PROGRESS_CLOSED
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)
        app_module._PROGRESS_OWNER.pop(_PID, None)
        app_module._PROGRESS_CLOSED.pop(_PID, None)


def test_jobs_sweep_runs_off_the_loop_thread(app_module, monkeypatch):
    from tests.main.test_retention_loops import _drive
    calls = _drive(monkeypatch, app_module._jobs_retention_loop, js)
    assert calls == [False]
