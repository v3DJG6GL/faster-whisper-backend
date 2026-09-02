"""usage_store.overview(): the /stats/usage v2 gather — dense axis,
breakdowns, leaderboard, compare window, per-model table."""

import datetime
import zoneinfo

import pytest

_UTC = zoneinfo.ZoneInfo("UTC")
_EPOCH = datetime.date(1970, 1, 1)


def _D(iso):
    return (datetime.date.fromisoformat(iso) - _EPOCH).days


def _ts(iso, hh=12):
    return datetime.datetime.fromisoformat(iso).replace(
        hour=hh, tzinfo=_UTC).timestamp()


def _hour(iso, hh=12):
    return int(_ts(iso, hh) // 3600)


def _seed(us):
    # 2025-06-02 (Mon) .. 2025-06-11: alice files on k1/k2, bob dictation on k3,
    # one diarized file from carol on 06-05, one text job from alice.
    us.record_usage(key_id="k1", user_id="alice", audio_s=60.0, words=100,
                    status="ok", kind="file", hour=_hour("2025-06-02"),
                    proc_s=6.0, job_id="a1", model="large-v3")
    us.record_usage(key_id="k2", user_id="alice", audio_s=30.0, words=50,
                    status="ok", kind="file", hour=_hour("2025-06-04"),
                    proc_s=9.0, job_id="a2", model="medium")
    us.record_usage(key_id="k3", user_id="bob", audio_s=10.0, words=40,
                    status="ok", kind="dictation", hour=_hour("2025-06-04"),
                    proc_s=1.0, job_id="b1", model="large-v3")
    us.record_usage(key_id="k3", user_id="bob", audio_s=10.0, words=40,
                    status="error", kind="dictation", hour=_hour("2025-06-10"),
                    proc_s=1.0, job_id="b2", model="large-v3")
    us.record_usage(key_id="k4", user_id="carol", audio_s=120.0, words=300,
                    status="ok", kind="url", hour=_hour("2025-06-05"),
                    proc_s=30.0, job_id="c1", model="large-v3",
                    stages=[{"name": "diarizing", "secs": 12.0, "speakers": 3}])
    us.record_usage(key_id="k1", user_id="alice", audio_s=0.0, words=0,
                    status="ok", kind="text", hour=_hour("2025-06-06"),
                    proc_s=0.5, job_id="a3", model=None)


NOW = _ts("2025-06-11", 15)


def _ov(us, **kw):
    kw.setdefault("user_id", None)
    kw.setdefault("tz", _UTC)
    kw.setdefault("tz_name", "UTC")
    kw.setdefault("now", NOW)
    return us.overview(**kw)


def test_overview_dense_axis_and_buckets(usage_store_db):
    us = usage_store_db
    _seed(us)
    ten = _ov(us, from_day=_D("2025-06-02"), to_day=_D("2025-06-11"))
    assert ten["v"] == 2 and ten["bucket"] == "day"
    assert ten["days"] == list(range(_D("2025-06-02"), _D("2025-06-11") + 1))
    assert all(len(ln["values"]) == 10 for ln in ten["lines"])
    wk = _ov(us, from_day=_D("2025-06-04"), to_day=_D("2025-06-20"), bucket="week")
    assert wk["days"] == [_D("2025-06-02"), _D("2025-06-09"), _D("2025-06-16")]
    mo = _ov(us, from_day=_D("2025-04-15"), to_day=_D("2025-06-11"), bucket="month")
    assert mo["days"] == [_D("2025-04-01"), _D("2025-05-01"), _D("2025-06-01")]
    assert _ov(us, days=120)["bucket"] == "day"
    assert _ov(us, days=121)["bucket"] == "week"
    assert _ov(us, days=731)["bucket"] == "month"
    assert us.bucket_mode(730) == "week"


def test_overview_by_kind_lines_sum_to_totals(usage_store_db):
    us = usage_store_db
    _seed(us)
    o = _ov(us, from_day=_D("2025-06-02"), to_day=_D("2025-06-11"), by="kind")
    ids = [ln["id"] for ln in o["lines"]]
    assert ids[:3] == ["url", "file", "dictation"]        # ranked by audio_s
    assert "text" in ids and "__others__" not in ids
    assert sum(sum(ln["values"]) for ln in o["lines"]) == o["totals"]["all"]["audio_s"]
    file_line = next(ln for ln in o["lines"] if ln["id"] == "file")
    assert file_line["values"][0] == 60.0 and file_line["values"][2] == 30.0
    board = {r["id"]: r for r in o["leaderboard"]}
    assert board["file"]["totals"]["sessions"] == 2
    assert board["file"]["rtf"] == round(15.0 / 90.0, 3)
    assert board["text"]["rtf"] is None
    assert o["breakdown"] == {"source": "rollups", "key_scoped": True}


def test_overview_by_user_top_k_and_others(usage_store_db):
    us = usage_store_db
    _seed(us)
    o = _ov(us, from_day=_D("2025-06-02"), to_day=_D("2025-06-11"), by="user",
            top_k=2, limit=2)
    assert [ln["id"] for ln in o["lines"]] == ["carol", "alice", "__others__"]
    assert o["lines"][-1]["label"] == "others (1)"
    assert sum(o["lines"][-1]["values"]) == 20.0            # bob's two dictations
    assert [r["id"] for r in o["leaderboard"]] == ["carol", "alice"]
    assert o["leaderboard"][1]["user_id"] == "alice"
    own = _ov(us, user_id="alice", from_day=_D("2025-06-02"),
              to_day=_D("2025-06-11"), by="key")
    assert sorted(ln["id"] for ln in own["lines"]) == ["k1", "k2"]
    assert all(ln["user_id"] == "alice" for ln in own["lines"])
    assert own["totals"]["all"]["audio_s"] == 90.0
    one = _ov(us, user_id="alice", key_id="k1", from_day=_D("2025-06-02"),
              to_day=_D("2025-06-11"), by="key")
    assert [ln["id"] for ln in one["lines"]] == ["k1"]
    assert one["filter"] == {"user_id": "alice", "key_id": "k1", "key_scoped": True}


def test_overview_by_model_from_jobs(usage_store_db):
    us = usage_store_db
    _seed(us)
    o = _ov(us, from_day=_D("2025-06-02"), to_day=_D("2025-06-11"), by="model",
            metric="sessions")
    assert o["breakdown"]["source"] == "jobs"
    board = {r["id"]: r for r in o["leaderboard"]}
    assert board["large-v3"]["totals"]["sessions"] == 4
    assert board["large-v3"]["totals"]["errors"] == 1
    assert board["medium"]["totals"]["sessions"] == 1
    assert board["(unknown)"]["totals"]["sessions"] == 1
    models = {m["model"]: m for m in o["models"]}
    assert models["large-v3"]["audio_s"] == 200.0 and models["large-v3"]["errors"] == 1
    assert models["medium"]["rtf"] == 0.3
    narrowed = _ov(us, from_day=_D("2025-06-02"), to_day=_D("2025-06-11"),
                   by="model", with_stages=("diarizing",))
    assert [ln["id"] for ln in narrowed["lines"]] == ["large-v3"]
    assert narrowed["models"] == [{"model": "large-v3", "sessions": 1, "requests": 1,
                                   "errors": 0, "words": 300, "audio_s": 120.0,
                                   "proc_s": 30.0, "rtf": 0.25}]


def test_overview_by_stage_from_stage_hourly(usage_store_db):
    us = usage_store_db
    _seed(us)
    o = _ov(us, from_day=_D("2025-06-02"), to_day=_D("2025-06-11"), by="stage",
            metric="proc_s")
    assert [ln["id"] for ln in o["lines"]] == ["diarizing"]
    assert sum(o["lines"][0]["values"]) == 12.0
    assert o["leaderboard"][0]["totals"]["audio_s"] == 120.0
    assert o["leaderboard"][0]["totals"]["sessions"] == 1
    assert o["breakdown"]["key_scoped"] is False
    keyed = _ov(us, user_id="carol", key_id="k4", from_day=_D("2025-06-02"),
                to_day=_D("2025-06-11"), by="stage")
    assert keyed["filter"]["key_scoped"] is False


def test_overview_compare_prev_and_yoy_aligned(usage_store_db):
    us = usage_store_db
    _seed(us)
    # An earlier window with one file job for alice, and the same dates a
    # year before with a dictation for bob.
    us.record_usage(key_id="k1", user_id="alice", audio_s=15.0, words=1,
                    status="ok", kind="file", hour=_hour("2025-05-24"),
                    proc_s=1.0, job_id="p1")
    us.record_usage(key_id="k3", user_id="bob", audio_s=7.0, words=1,
                    status="ok", kind="dictation", hour=_hour("2024-06-03"),
                    proc_s=1.0, job_id="y1")
    o = _ov(us, from_day=_D("2025-06-02"), to_day=_D("2025-06-11"), by="kind",
            compare="prev")
    c = o["compare"]
    assert c["mode"] == "prev"
    assert c["range"] == {"from": _D("2025-05-23"), "to": _D("2025-06-01"), "days": 10}
    assert c["totals"]["all"]["audio_s"] == 15.0
    lines = {ln["id"]: ln["values"] for ln in c["lines"]}
    assert set(lines) == {ln["id"] for ln in o["lines"]}
    assert lines["file"][1] == 15.0 and len(lines["file"]) == 10
    y = _ov(us, from_day=_D("2025-06-02"), to_day=_D("2025-06-11"), by="kind",
            compare="yoy")
    assert y["compare"]["range"] == {"from": _D("2024-06-02"), "to": _D("2024-06-11"),
                                     "days": 10}
    assert y["compare"]["totals"]["all"]["audio_s"] == 7.0
    assert {ln["id"]: ln["values"] for ln in y["compare"]["lines"]}["dictation"][1] == 7.0
    # Feb 29 clamps to Feb 28 a year earlier.
    assert us._year_back(_D("2024-02-29")) == _D("2023-02-28")
    assert _ov(us, days=3, compare="off")["compare"] is None


def test_overview_unknown_params_fall_back(usage_store_db):
    us = usage_store_db
    o = _ov(us, by="zzz", metric="zzz", bucket="zzz", compare="zzz")
    assert (o["by"], o["metric"], o["bucket"], o["compare"]) == ("kind", "audio_s", "day", None)
    assert o["lines"] == [] and o["leaderboard"] == [] and o["models"] == []
