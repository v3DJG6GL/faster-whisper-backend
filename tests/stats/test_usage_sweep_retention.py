"""usage_store.sweep: a job retention shorter than the unreported grace period
must still close out the dictation jobs it is about to delete."""

import zoneinfo

_UTC = zoneinfo.ZoneInfo("UTC")


def test_sweep_marks_unreported_jobs_the_retention_delete_removes(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=10.0, words=30, status="ok",
                    kind="dictation", job_id="1" * 32, hour=us.now_hour() - 10 * 24)

    counts = us.sweep(unreported_after_h=720, jobs_retention_days=7,
                      app_retention_days=0, hourly_retention_days=0)

    assert counts["marked"] == 1
    assert counts["jobs"] == 1
    doc = us.document("u", days=30, tz=_UTC, tz_name="UTC")
    assert doc["dictation"]["delivery"]["unreported"] == 1
