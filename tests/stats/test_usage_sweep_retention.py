"""usage_store.sweep: a job retention shorter than the unreported grace period
must keep an unreported dictation job until the grace ends, then close it
out and delete it — counted once either way."""

from tests.stats._usage_helpers import outcome as _outcome, rec as _rec, utc_doc as _doc


def test_sweep_marks_unreported_jobs_once_the_grace_ends(us):
    _rec(us, audio_s=10.0, words=30, kind="dictation", job_id="1" * 32,
         hour=us.now_hour() - 31 * 24)

    counts = us.sweep(unreported_after_h=720, jobs_retention_days=7,
                      app_retention_days=0, hourly_retention_days=0)

    assert counts["marked"] == 1
    assert counts["jobs"] == 1
    doc = _doc(us, days=60)
    assert doc["dictation"]["delivery"]["unreported"] == 1


def test_late_outcome_inside_the_grace_counts_the_session_once(us):
    _rec(us, audio_s=10.0, words=30, kind="dictation", job_id="2" * 32, hour=us.now_hour() - 8 * 24)

    counts = us.sweep(unreported_after_h=720, jobs_retention_days=7,
                      app_retention_days=0, hourly_retention_days=0)
    assert counts == {"marked": 0, "jobs": 0, "apps": 0, "hourly": 0}

    assert _outcome(us, "2" * 32, activation="hold", delivery="typed") == "accepted"

    doc = _doc(us, days=30)
    assert doc["dictation"]["delivery"]["typed"] == 1
    assert doc["dictation"]["delivery"]["unreported"] == 0
    assert doc["dictation"]["activation"] == {"hold": 1, "handsfree": 0, "unreported": 0}


def test_retention_still_prunes_reported_and_non_dictation_jobs(us):
    old = us.now_hour() - 8 * 24
    _rec(us, audio_s=5.0, words=10, kind="file", job_id="3" * 32, hour=old)
    _rec(us, audio_s=5.0, words=10, kind="dictation", job_id="4" * 32, hour=old)
    _outcome(us, "4" * 32)

    counts = us.sweep(unreported_after_h=720, jobs_retention_days=7,
                      app_retention_days=0, hourly_retention_days=0)

    assert counts["jobs"] == 2


def test_stale_unreported_scan_uses_the_partial_index(us):
    plan = us._require_conn().execute(
        "EXPLAIN QUERY PLAN " + us._STALE_UNREPORTED_SQL, (0.0,)).fetchall()
    assert any("idx_usage_jobs_unreported" in row[-1] for row in plan), plan


def test_app_name_scrub_uses_the_partial_index(us):
    """The hourly scrub runs under _lock; on idx_usage_jobs_created it walks
    every job past the app clock even when none still names a program."""
    plan = us._require_conn().execute(
        "EXPLAIN QUERY PLAN " + us._APP_SCRUB_SQL, (0.0,)).fetchall()
    assert any("idx_usage_jobs_app" in row[-1] for row in plan), plan


def test_app_retention_scrubs_the_program_name_off_old_job_rows(us):
    """USAGE_APP_RETENTION_DAYS is a privacy clock: the job rows (kept for the
    longer job retention) must not keep naming the program, or a with=
    document reads it back after the app rollup dropped it."""
    tr = [{"name": "translating", "secs": 1.0, "targets": ["de"]}]
    for jid, age_h in (("5" * 32, 200 * 24), ("6" * 32, 10 * 24)):
        _rec(us, audio_s=5.0, words=20, kind="dictation", job_id=jid, hour=us.now_hour() - age_h,
             stages=tr)
        _outcome(us, jid, translation="translated",
                 app_id="secret.app" if jid[0] == "5" else "vim")

    counts = us.sweep(unreported_after_h=24, jobs_retention_days=365,
                      app_retention_days=90, hourly_retention_days=0)

    assert counts["apps"] == 1
    assert counts["jobs"] == 0
    doc = _doc(us, days=365, with_stages=("translating",))
    assert doc["apps"] == [{"app_id": "vim", "sessions": 1, "words": 20}]
    conn = us._require_conn()
    rows = conn.execute("SELECT job_id, app_id FROM usage_jobs ORDER BY job_id")
    assert [tuple(r) for r in rows] == [("5" * 32, None), ("6" * 32, "vim")]
