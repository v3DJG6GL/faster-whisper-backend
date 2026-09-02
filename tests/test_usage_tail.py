"""usage_store.tail() and its parts: wait quantiles, turnaround histogram,
failures by stage / class, per-model table, compare window."""

import datetime
import zoneinfo

import pytest

_UTC = zoneinfo.ZoneInfo("UTC")
_EPOCH = datetime.date(1970, 1, 1)


def _D(iso):
    return (datetime.date.fromisoformat(iso) - _EPOCH).days


def _ts(iso, hh=12):
    return datetime.datetime.fromisoformat(iso).replace(hour=hh, tzinfo=_UTC).timestamp()


def _hour(iso, hh=12):
    return int(_ts(iso, hh) // 3600)


NOW = _ts("2025-06-11", 15)


def _seed(us):
    # Current window 06-02..06-11: five alice files with waits 0,1,2,4,40,
    # two of them failed (cuda_oom in transcribing; a policy block in
    # downloading), one soft-failed diarization on an ok job; a bob
    # dictation. Previous window (05-23..06-01): two ok files, wait 10.
    waits = [0.0, 1.0, 2.0, 4.0, 40.0]
    for i, w in enumerate(waits):
        us.record_usage(key_id="k1", user_id="alice", audio_s=100.0, words=10,
                        status="error" if i in (1, 4) else "ok", kind="file",
                        hour=_hour("2025-06-03") + i * 24, proc_s=10.0 + i,
                        job_id=f"a{i}", model="large-v3" if i < 3 else "medium",
                        wait_s=w,
                        error_class=("cuda_oom" if i == 4 else "policy_blocked" if i == 1 else None),
                        error_stage=("transcribing" if i == 4 else "downloading" if i == 1 else None),
                        stages=([{"name": "diarizing", "secs": 3.0, "error": "timeout"}]
                                if i == 2 else None))
    us.record_usage(key_id="k2", user_id="bob", audio_s=5.0, words=5, status="ok",
                    kind="dictation", hour=_hour("2025-06-05"), proc_s=0.5,
                    job_id="b1", model="large-v3", wait_s=0.2)
    for i in range(2):
        us.record_usage(key_id="k1", user_id="alice", audio_s=50.0, words=10,
                        status="ok", kind="file", hour=_hour("2025-05-25") + i * 24,
                        proc_s=5.0, job_id=f"p{i}", model="large-v3", wait_s=10.0)


def test_wait_quantiles_nearest_rank(usage_store_db):
    us = usage_store_db
    _seed(us)
    s, e = _ts("2025-06-02", 0), _ts("2025-06-12", 0)
    q = us.wait_quantiles(start_ts=s, end_ts=e)
    assert q["n"] == 6 and q["max"] == 40.0
    # sorted waits: 0, 0.2, 1, 2, 4, 40 → p50 = index round(0.5*5)=2 → 1.0;
    # p95 = index round(0.95*5)=5 → 40.
    assert q["p50"] == 1.0 and q["p95"] == 40.0
    assert us.wait_quantiles(start_ts=s, end_ts=e, user_id="bob")["n"] == 1
    assert us.wait_quantiles(start_ts=s, end_ts=e, kind="dictation")["p50"] == 0.2
    assert us.wait_quantiles(start_ts=s, end_ts=e, model="medium")["p50"] == 4.0
    assert us.wait_quantiles(start_ts=s, end_ts=e, user_id="nobody") == {
        "n": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    days = us.wait_series_by_day(start_ts=s, end_ts=e, tz=_UTC, user_id="alice")
    assert [d["day"] for d in days] == [_D("2025-06-03") + i for i in range(5)]
    assert days[4] == {"day": _D("2025-06-07"), "n": 1, "p50": 40.0, "p95": 40.0}


def test_turnaround_histogram_buckets_and_wait_share(usage_store_db):
    us = usage_store_db
    _seed(us)
    h = us.turnaround_histogram(start_ts=_ts("2025-06-02", 0), end_ts=_ts("2025-06-12", 0))
    assert h["edges_s"] == [0, 1, 2, 5, 10, 30, 60, 120, 300, 900]
    assert h["n"] == 6 and sum(h["counts"]) == 6
    # turnarounds: 10, 12, 14, 17, 54, 0.7 → buckets 10-30 ×4, 30-60 ×1, 0-1 ×1
    assert h["counts"][4] == 4 and h["counts"][5] == 1 and h["counts"][0] == 1
    assert h["wait_share"][5] == round(40.0 / 54.0, 3)
    assert h["p50"] == 12.0 and h["max"] == 54.0


def test_failures_union_terminal_and_soft(usage_store_db):
    us = usage_store_db
    _seed(us)
    f = us.failures(start_ts=_ts("2025-06-02", 0), end_ts=_ts("2025-06-12", 0))
    assert (f["jobs"], f["failed"]) == (6, 2)
    assert f["by_stage"] == {"transcribing": {"cuda_oom": 1},
                             "downloading": {"policy_blocked": 1},
                             "diarizing": {"timeout": 1}}
    assert f["by_class"] == {"cuda_oom": 1, "policy_blocked": 1, "timeout": 1}
    own = us.failures(start_ts=_ts("2025-06-02", 0), end_ts=_ts("2025-06-12", 0),
                      user_id="bob")
    assert own == {"jobs": 1, "failed": 0, "by_stage": {}, "by_class": {}}


def test_by_model_with_wait_p50(usage_store_db):
    us = usage_store_db
    _seed(us)
    rows = {m["model"]: m for m in us.by_model(
        start_ts=_ts("2025-06-02", 0), end_ts=_ts("2025-06-12", 0))}
    assert rows["large-v3"]["runs"] == 4 and rows["large-v3"]["errors"] == 1
    assert rows["medium"]["runs"] == 2 and rows["medium"]["wait_p50"] == 4.0
    assert rows["large-v3"]["rtf"] == round(33.5 / 305.0, 3)


def test_tail_document_and_compare(usage_store_db, app_module):
    us = usage_store_db
    _seed(us)
    doc = us.tail(user_id=None, tz=_UTC, tz_name="UTC", from_day=_D("2025-06-02"),
                  to_day=_D("2025-06-11"), jobs_retention_days=365, now=NOW)
    assert doc["range"]["days"] == 10 and doc["range"]["truncated_to_days"] is None
    assert doc["wait"]["p95"] == 40.0 and len(doc["wait"]["by_day"]) == 5
    assert doc["turnaround"]["n"] == 6
    assert doc["failures"]["failed"] == 2
    assert [m["model"] for m in doc["models"]][0] == "large-v3"
    c = doc["compare"]
    assert (c["from"], c["to"]) == (_D("2025-05-23"), _D("2025-06-01"))
    assert c["wait_p50"] == {"cur": 1.0, "prev": 10.0, "delta": -9.0}
    assert c["runs"] == {"cur": 6, "prev": 2, "delta": 4}
    assert c["errors"]["prev"] == 0 and c["audio_s"]["prev"] == 100.0
    old = us.tail(user_id=None, tz=_UTC, tz_name="UTC", days=3650,
                  jobs_retention_days=30, now=NOW)
    assert old["range"]["truncated_to_days"] == 30
    assert us.truncated_to_days(NOW - 10 * 86400, 30, now=NOW) is None
    assert us.truncated_to_days(NOW - 40 * 86400, 0, now=NOW) is None
