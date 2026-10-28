"""usage_store.sweep: a job retention shorter than the unreported grace period
must keep an unreported dictation job until the grace ends, then close it
out and delete it — counted once either way."""

import zoneinfo

_UTC = zoneinfo.ZoneInfo("UTC")


def test_sweep_marks_unreported_jobs_once_the_grace_ends(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=10.0, words=30, status="ok",
                    kind="dictation", job_id="1" * 32, hour=us.now_hour() - 31 * 24)

    counts = us.sweep(unreported_after_h=720, jobs_retention_days=7,
                      app_retention_days=0, hourly_retention_days=0)

    assert counts["marked"] == 1
    assert counts["jobs"] == 1
    doc = us.document("u", days=60, tz=_UTC, tz_name="UTC")
    assert doc["dictation"]["delivery"]["unreported"] == 1


def test_late_outcome_inside_the_grace_counts_the_session_once(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=10.0, words=30, status="ok",
                    kind="dictation", job_id="2" * 32, hour=us.now_hour() - 8 * 24)

    counts = us.sweep(unreported_after_h=720, jobs_retention_days=7,
                      app_retention_days=0, hourly_retention_days=0)
    assert counts == {"marked": 0, "jobs": 0, "apps": 0, "hourly": 0}

    assert us.record_outcome(user_id="u", job_id="2" * 32, activation="hold",
                             delivery="typed", translation="not_asked") == "accepted"

    doc = us.document("u", days=30, tz=_UTC, tz_name="UTC")
    assert doc["dictation"]["delivery"]["typed"] == 1
    assert doc["dictation"]["delivery"]["unreported"] == 0
    assert doc["dictation"]["activation"] == {"hold": 1, "handsfree": 0, "unreported": 0}


def test_retention_still_prunes_reported_and_non_dictation_jobs(usage_store_db):
    us = usage_store_db
    old = us.now_hour() - 8 * 24
    us.record_usage(key_id="k", user_id="u", audio_s=5.0, words=10, status="ok",
                    kind="file", job_id="3" * 32, hour=old)
    us.record_usage(key_id="k", user_id="u", audio_s=5.0, words=10, status="ok",
                    kind="dictation", job_id="4" * 32, hour=old)
    us.record_outcome(user_id="u", job_id="4" * 32, activation="hold",
                      delivery="typed", translation="not_asked")

    counts = us.sweep(unreported_after_h=720, jobs_retention_days=7,
                      app_retention_days=0, hourly_retention_days=0)

    assert counts["jobs"] == 2
