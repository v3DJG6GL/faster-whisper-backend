"""Unit tests for jobs.py — the central running-jobs registry."""

from faster_whisper_backend.core import jobs


def test_start_update_end_roundtrip():
    jid = jobs.job_start("translate", model="org/m:Q4", user="u" * 32,
                         key="k" * 32, detail="3 segs → en")
    assert jid
    jobs.job_update(jid, progress=0.5, step="en 1/2", stage="translating")
    snap = jobs.jobs_snapshot(include_identity=True)
    assert len(snap) == 1
    row = snap[0]
    assert row["kind"] == "translate"
    assert row["model"] == "org/m:Q4"
    assert row["progress"] == 0.5
    assert row["step"] == "en 1/2"
    assert row["stage"] == "translating"
    assert row["user"] == "u" * 32
    assert row["detail"] == "3 segs → en"
    assert row["elapsed_s"] >= 0
    jobs.job_end(jid)
    assert jobs.jobs_snapshot() == []


def test_snapshot_scrubs_identity_by_default():
    jobs.job_start("transcribe", model="large-v3", user="uid",
                   key="kid", detail="secret.wav")
    row = jobs.jobs_snapshot()[0]
    assert "user" not in row and "key" not in row and "detail" not in row
    assert row["model"] == "large-v3"          # model is not identity


def test_update_merges_only_non_none_and_ignores_unknown_ids():
    jid = jobs.job_start("download", model="gguf:org/m")
    jobs.job_update(jid, progress=0.3)
    jobs.job_update(jid, progress=None, step="fetch")
    row = jobs.jobs_snapshot()[0]
    assert row["progress"] == 0.3 and row["step"] == "fetch"
    jobs.job_update("nope", progress=1.0)               # no-op, no raise
    jobs.job_end(jid)
    jobs.job_end(jid)                                   # idempotent
    jobs.job_end(None)


def test_snapshot_sorted_by_start_and_caller_id_wins():
    jobs.job_start("transcribe", id="a" * 32)
    jobs.job_start("dictate", id="b" * 32)
    snap = jobs.jobs_snapshot(include_identity=True)
    assert [r["id"] for r in snap] == ["a" * 32, "b" * 32]
    assert snap[1]["kind"] == "dictate"


def test_registry_is_bounded():
    for _ in range(jobs._MAX_JOBS + 10):
        jobs.job_start("transcribe")
    assert len(jobs.jobs_snapshot()) == jobs._MAX_JOBS


def test_update_clear_sentinel_writes_none():
    jid = jobs.job_start("transcribe", model="large-v3")
    jobs.job_update(jid, progress=0.9, step="s", stage="transcribing")
    # None still means "leave alone"; CLEAR explicitly writes None.
    jobs.job_update(jid, progress=jobs.CLEAR, step=None, stage="diarizing")
    row = jobs.jobs_snapshot()[0]
    assert row["progress"] is None
    assert row["step"] == "s"
    assert row["stage"] == "diarizing"


def test_snapshot_filters_by_user_id():
    """Own-scope viewers get only rows they own; rows started without an
    owner (model downloads, standalone stage runs) never show for them."""
    a = jobs.job_start("transcribe", model="m", user="alice", user_id="ua")
    b = jobs.job_start("dictate", model="m", user="bob", user_id="ub")
    d = jobs.job_start("download", model="gguf:org/m")
    ids = {r["id"] for r in jobs.jobs_snapshot(include_identity=True,
                                               user_id="ua")}
    assert ids == {a}
    assert {r["id"] for r in jobs.jobs_snapshot(include_identity=True)} == {
        a, b, d}
    assert jobs.jobs_snapshot(user_id="nobody") == []


def test_user_id_is_never_emitted():
    jobs.job_start("transcribe", model="m", user="alice", user_id="ua")
    for row in (jobs.jobs_snapshot()[0],
                jobs.jobs_snapshot(include_identity=True)[0],
                jobs.jobs_snapshot(user_id="ua")[0]):
        assert "user_id" not in row


def test_progress_id_visible_to_owner_and_admin_only():
    """The cancel handle rides the row for admins (include_identity) and for
    the row's own owner (viewer_user_id) — never for another non-admin."""
    jid = jobs.job_start("transcribe", model="m", user="alice", user_id="ua")
    jobs.job_update(jid, progress_id="pid-1")
    assert jobs.jobs_snapshot(include_identity=True)[0]["progress_id"] == "pid-1"
    assert jobs.jobs_snapshot(viewer_user_id="ua")[0]["progress_id"] == "pid-1"
    assert "progress_id" not in jobs.jobs_snapshot(viewer_user_id="ub")[0]
    assert "progress_id" not in jobs.jobs_snapshot()[0]


def test_server_id_visible_to_owner_and_admin_only():
    """The row id is the run's server request/session id, which is also its
    usage job id while the client named none: a foreign viewer holding it
    could claim that job via POST /v1/usage/outcome before the run records.
    So it rides the row under the cancel handle's gate."""
    jid = jobs.job_start("transcribe", model="m", user="alice", user_id="ua")
    assert jobs.jobs_snapshot(include_identity=True)[0]["id"] == jid
    assert jobs.jobs_snapshot(viewer_user_id="ua")[0]["id"] == jid
    assert "id" not in jobs.jobs_snapshot(viewer_user_id="ub")[0]
    assert "id" not in jobs.jobs_snapshot()[0]
