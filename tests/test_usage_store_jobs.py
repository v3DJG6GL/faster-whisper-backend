"""usage_store: the per-kind / per-job / per-stage rollups behind the desktop
app's statistics — the migration from the per-request rollup, session
counting, the outcome + sweep write paths, and the document() read path
(tz-aware day reckoning, stage meters, streak, time saved).
"""

import datetime
import sqlite3
import zoneinfo


def _D(iso):
    """days-since-epoch of an ISO date — the wire form of `day`."""
    return (datetime.date.fromisoformat(iso) - datetime.date(1970, 1, 1)).days

_ZH = zoneinfo.ZoneInfo("Europe/Zurich")
_UTC = zoneinfo.ZoneInfo("UTC")


def _ts(y, m, d, hh=12, mm=0, tz=_UTC) -> float:
    return datetime.datetime(y, m, d, hh, mm, tzinfo=tz).timestamp()


def _hour(y, m, d, hh=12, mm=0, tz=_UTC) -> int:
    return int(_ts(y, m, d, hh, mm, tz) // 3600)


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------

_LEGACY_SCHEMA = """
CREATE TABLE usage_hourly (
  hour INTEGER NOT NULL, key_id TEXT NOT NULL, user_id TEXT NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0, errors INTEGER NOT NULL DEFAULT 0,
  words INTEGER NOT NULL DEFAULT 0, audio_s REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, key_id)
);
CREATE INDEX idx_usage_user_hour ON usage_hourly(user_id, hour);
CREATE INDEX idx_usage_hour      ON usage_hourly(hour);
"""


def test_init_rebuilds_legacy_hourly_with_kind_and_sessions(tmp_path):
    """A pre-kind DB (PK hour×key) is rebuilt to hour×key×kind; old rows read
    as kind='unknown' with one session per request, and the parking table
    is gone. A second init is a no-op."""
    import usage_store
    path = str(tmp_path / "legacy.sqlite3")
    raw = sqlite3.connect(path)
    raw.executescript(_LEGACY_SCHEMA)
    raw.execute("INSERT INTO usage_hourly VALUES (100, 'k', 'u', 3, 1, 42, 9.5)")
    raw.execute("INSERT INTO usage_hourly VALUES (101, 'k', 'u', 1, 0, 7, 1.0)")
    raw.commit()
    raw.close()

    usage_store.init_db(path)
    try:
        conn = usage_store._require_conn()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(usage_hourly)")}
        assert {"kind", "sessions", "proc_s"} <= cols
        rows = conn.execute(
            "SELECT hour, kind, requests, sessions, words, audio_s"
            " FROM usage_hourly ORDER BY hour").fetchall()
        assert [tuple(r) for r in rows] == [
            (100, "unknown", 3, 3, 42, 9.5), (101, "unknown", 1, 1, 7, 1.0)]
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='usage_hourly_legacy'"
        ).fetchone() is None
        idx = {r["name"] for r in conn.execute("PRAGMA index_list(usage_hourly)")}
        assert {"idx_usage_user_hour", "idx_usage_hour"} <= idx
        # The old rows still feed the folded admin views.
        assert usage_store.totals_for_user("u")["words"] == 49
        conn.close()
        usage_store.init_db(path)
        assert usage_store.totals_for_user("u")["requests"] == 4
    finally:
        usage_store._require_conn().close()
        usage_store._conn = None


# --------------------------------------------------------------------------
# Sessions, kinds, stages
# --------------------------------------------------------------------------

def test_sessions_counted_once_per_job_across_utterances(usage_store_db):
    us = usage_store_db
    for words in (5, 7, 9):
        us.record_usage(key_id="k", user_id="u", audio_s=2.0, words=words,
                        status="ok", kind="dictation", job_id="a" * 32,
                        proc_s=0.5)
    row = us._require_conn().execute(
        "SELECT requests, sessions, words, proc_s FROM usage_hourly").fetchone()
    assert tuple(row) == (3, 1, 21, 1.5)
    job = us._require_conn().execute(
        "SELECT utterances, words, audio_s, kind FROM usage_jobs").fetchone()
    assert tuple(job) == (3, 21, 6.0, "dictation")
    # Without a job id every request is its own session.
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok",
                    kind="file")
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok",
                    kind="file")
    doc = us.document("u", days=7, tz=_UTC, tz_name="UTC")
    assert doc["today"]["dictation"]["sessions"] == 1
    assert doc["today"]["dictation"]["requests"] == 3
    assert doc["today"]["file"]["sessions"] == 2
    assert doc["today"]["all"]["sessions"] == 3


def test_colliding_job_id_of_another_user_is_not_merged(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="ka", user_id="alice", audio_s=1.0, words=10,
                    status="ok", kind="dictation", job_id="c" * 32)
    us.record_usage(key_id="kb", user_id="bob", audio_s=1.0, words=20,
                    status="ok", kind="dictation", job_id="c" * 32)
    job = us._require_conn().execute(
        "SELECT user_id, words, utterances FROM usage_jobs").fetchone()
    assert tuple(job) == ("alice", 10, 1)
    # Bob's work is still counted — as a session of its own.
    assert us.document("bob", days=1, tz=_UTC,
                       tz_name="UTC")["today"]["dictation"]["sessions"] == 1


def test_kind_totals_and_unknown_folds_into_all_only(usage_store_db):
    us = usage_store_db
    for kind, words in (("dictation", 1), ("file", 2), ("url", 4), ("text", 8),
                        (None, 16), ("bogus", 32)):
        us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=words,
                        status="ok", kind=kind)
    doc = us.document("u", days=1, tz=_UTC, tz_name="UTC")
    total = doc["total"]
    assert set(total) == {"all", "dictation", "file", "url", "text"}
    assert [total[k]["words"] for k in ("dictation", "file", "url", "text")] == [1, 2, 4, 8]
    assert total["all"]["words"] == 63
    assert total["all"]["requests"] == 6


def test_stage_and_target_rollups(usage_store_db):
    us = usage_store_db
    us.record_usage(
        key_id="k", user_id="u", audio_s=100.0, words=50, status="ok",
        kind="file", job_id="f" * 32, stages=[
            {"name": "transcribing", "secs": 4.0, "model": "large-v3"},
            {"name": "diarizing", "secs": 3.0, "speakers": 3,
             "detail": "3 speakers"},
            {"name": "vad", "secs": 1.0, "retained": 0.5},
            {"name": "translating", "secs": 2.0, "targets": ["de", "fr"],
             "kept_original": 2},
        ])
    # The text endpoint names its stage 'translate' — same row.
    us.record_usage(
        key_id="k", user_id="u", audio_s=0.0, words=0, status="ok",
        kind="text", job_id="t" * 32,
        stages=[{"name": "translate", "secs": 1.5, "targets": ["de"]}])
    # A dictation cannot have been diarized — it must not widen that meter.
    us.record_usage(key_id="k", user_id="u", audio_s=5.0, words=9,
                    status="ok", kind="dictation", job_id="d" * 32)

    doc = us.document("u", days=1, tz=_UTC, tz_name="UTC")
    stages = {s["stage"]: s for s in doc["stages"]}
    assert list(stages) == ["translating", "diarizing", "vad"]
    tr = stages["translating"]
    assert (tr["runs"], tr["of_runs"], tr["secs"], tr["kept_original"]) == (2, 3, 3.5, 2)
    assert tr["targets"] == [{"code": "de", "runs": 2}, {"code": "fr", "runs": 1}]
    di = stages["diarizing"]
    assert (di["runs"], di["of_runs"], di["speakers_avg"], di["audio_s"]) == (1, 1, 3.0, 100.0)
    assert stages["vad"]["retained_avg"] == 0.5
    assert "speakers_avg" not in tr and "kept_original" not in di
    per_job = us._require_conn().execute(
        "SELECT stage, targets, speakers, retained FROM usage_job_stages"
        " WHERE job_id = ? ORDER BY stage", ("f" * 32,)).fetchall()
    assert [tuple(r) for r in per_job] == [
        ("diarizing", None, 3, None), ("transcribing", None, None, None),
        ("translating", "de,fr", None, None), ("vad", None, None, 0.5)]


# --------------------------------------------------------------------------
# Day reckoning
# --------------------------------------------------------------------------

def test_days_reckoned_in_caller_zone_across_dst_end(usage_store_db):
    """Europe/Zurich leaves DST on 2025-10-26. An utterance at 00:30 local on
    the 26th (22:30 UTC on the 25th) and one at 00:30 local on the 27th
    (23:30 UTC on the 26th — the offset has changed) must land on the local
    dates, and on the UTC dates for a UTC caller."""
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=10, status="ok",
                    kind="dictation", hour=_hour(2025, 10, 26, 0, 30, _ZH))
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=20, status="ok",
                    kind="dictation", hour=_hour(2025, 10, 27, 0, 30, _ZH))
    now = _ts(2025, 10, 27, 12, 0, _UTC)

    zh = us.document("u", days=7, tz=_ZH, tz_name="Europe/Zurich",
                     now=now)
    assert [(p["day"], p["all"]["words"]) for p in zh["series"]] == [
        (_D("2025-10-26"), 10), (_D("2025-10-27"), 20)]
    assert zh["today"]["all"]["words"] == 20
    assert zh["streak"]["all"] == {"current": 2, "best": 2}

    utc = us.document("u", days=7, tz=_UTC, tz_name="UTC", now=now)
    assert [(p["day"], p["all"]["words"]) for p in utc["series"]] == [
        (_D("2025-10-25"), 10), (_D("2025-10-26"), 20)]
    assert utc["today"]["all"]["words"] == 0
    # Yesterday's streak still counts while today is empty.
    assert utc["streak"]["all"] == {"current": 2, "best": 2}


def test_server_local_fallback_matches_tz_none(usage_store_db, set_tz):
    set_tz("America/New_York")
    us = usage_store_db
    # 23:00 New York on the 3rd = 04:00 UTC on the 4th.
    ny = zoneinfo.ZoneInfo("America/New_York")
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=5, status="ok",
                    kind="file", hour=_hour(2025, 6, 3, 23, 0, ny))
    doc = us.document("u", days=30, tz=None, tz_name="local",
                      now=_ts(2025, 6, 4, 12, 0, _UTC))
    assert doc["series"][0]["day"] == _D("2025-06-03")
    assert doc["tz"] == "local"


def test_window_forms_days_from_to_and_all(usage_store_db):
    """`days` ends today; `from`/`to` are inclusive caller-local days; `all`
    starts at the first day with usage. total/series/calendar/hours cover the
    window; the streak runs over the whole history."""
    us = usage_store_db
    now = _ts(2025, 3, 20)
    for days_ago, words in ((0, 1), (5, 2), (40, 4), (100, 8)):
        us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=words,
                        status="ok", kind="text",
                        hour=int(now // 3600) - days_ago * 24)
    today = _D("2025-03-20")
    doc = us.document("u", days=7, tz=_UTC, tz_name="UTC", now=now)
    assert doc["range"] == {"from": today - 6, "to": today, "days": 7,
                            "first_day": _D("2024-12-10"), "source": "rollups",
                            "jobs_retention_days": 365}
    assert doc["total"]["all"]["words"] == 3
    assert [c["all"] for c in doc["calendar"]] == [2, 1]
    assert [c["text"] for c in doc["calendar"]] == [2, 1]
    assert doc["streak"]["all"] == {"current": 1, "best": 1}
    assert doc["streak"]["text"] == {"current": 1, "best": 1}
    assert doc["streak"]["file"] == {"current": 0, "best": 0}

    doc = us.document("u", from_day=today - 45, to_day=today - 30, tz=_UTC,
                      tz_name="UTC", now=now)
    assert doc["range"]["days"] == 16
    assert doc["total"]["all"]["words"] == 4
    assert [p["day"] for p in doc["series"]] == [today - 40]
    assert doc["today"]["all"]["words"] == 1  # today is today whatever the window

    doc = us.document("u", all_time=True, tz=_UTC, tz_name="UTC", now=now)
    assert doc["range"]["from"] == _D("2024-12-10") and doc["range"]["to"] == today
    assert doc["total"]["all"]["words"] == 15
    assert len(doc["calendar"]) == 4

    # `to` alone ends the default 30-day window there; the span is clamped.
    doc = us.document("u", to_day=today - 30, tz=_UTC, tz_name="UTC", now=now)
    assert doc["range"] ["from"] == today - 59
    doc = us.document("u", from_day=0, to_day=today, tz=_UTC, tz_name="UTC", now=now)
    assert doc["range"]["days"] == us.MAX_WINDOW_DAYS
    import pytest
    with pytest.raises(ValueError):
        us.document("u", from_day=today, to_day=today - 1, tz=_UTC, tz_name="UTC",
                    now=now)
    with pytest.raises(ValueError):
        us.document("u", with_stages=("decoding",), tz=_UTC, tz_name="UTC", now=now)
    # No usage at all: all-time is today, first_day is null.
    doc = us.document("nobody", all_time=True, tz=_UTC, tz_name="UTC", now=now)
    assert doc["range"]["from"] == today and doc["range"]["first_day"] is None


def test_streak_is_per_kind_and_never_capped_by_the_window(usage_store_db):
    us = usage_store_db
    now = _ts(2025, 3, 20)
    for days_ago in range(0, 12):
        us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1,
                        status="ok", kind="dictation",
                        hour=int(now // 3600) - days_ago * 24)
    for days_ago in (0, 3, 4, 5, 6, 9):
        us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1,
                        status="ok", kind="file",
                        hour=int(now // 3600) - days_ago * 24)
    doc = us.document("u", days=7, tz=_UTC, tz_name="UTC", now=now)
    assert doc["streak"]["all"] == {"current": 12, "best": 12}
    assert doc["streak"]["dictation"] == {"current": 12, "best": 12}
    assert doc["streak"]["file"] == {"current": 1, "best": 4}
    assert doc["streak"]["url"] == {"current": 0, "best": 0}


def test_hours_grid_is_local_weekday_by_hour_per_kind(usage_store_db):
    """Europe/Zurich leaves DST on 2025-10-26: 09:30 local is 07:30 UTC on
    the 25th (Sat) and 08:30 UTC on the 27th (Mon). Both land in the local
    09:00 slot of their local weekday; a UTC caller sees them at 07 and 08."""
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=10, status="ok",
                    kind="dictation", hour=_hour(2025, 10, 25, 9, 30, _ZH))
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=20, status="ok",
                    kind="file", hour=_hour(2025, 10, 27, 9, 30, _ZH))
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=5, status="ok",
                    kind="dictation", hour=_hour(2025, 10, 27, 9, 45, _ZH))
    now = _ts(2025, 10, 27, 12, 0, _UTC)
    zh = us.document("u", days=7, tz=_ZH, tz_name="Europe/Zurich", now=now)
    assert zh["hours"] == [
        {"dow": 0, "hour": 9, "all": 25, "dictation": 5, "file": 20, "url": 0, "text": 0},
        {"dow": 5, "hour": 9, "all": 10, "dictation": 10, "file": 0, "url": 0, "text": 0},
    ]
    utc = us.document("u", days=7, tz=_UTC, tz_name="UTC", now=now)
    assert [(h["dow"], h["hour"], h["all"]) for h in utc["hours"]] == [
        (0, 8, 25), (5, 7, 10)]


def test_with_stages_narrows_to_jobs_that_ran_all_of_them(usage_store_db):
    us = usage_store_db
    now = _ts(2025, 6, 10, 12, 0, _UTC)
    h = int(now // 3600)
    # f1: translated + diarized; f2: diarized only; d1: dictation translated
    # (via the text stage), t1: text translate.
    us.record_usage(key_id="k", user_id="u", audio_s=100.0, words=400, status="ok",
                    kind="file", job_id="f1" * 16, hour=h - 2, stages=[
                        {"name": "translating", "secs": 3.0, "targets": ["de"]},
                        {"name": "diarizing", "secs": 5.0, "speakers": 2}])
    us.record_usage(key_id="k", user_id="u", audio_s=50.0, words=200, status="error",
                    kind="file", job_id="f2" * 16, hour=h - 30, stages=[
                        {"name": "diarizing", "secs": 2.0, "speakers": 4}])
    for _ in range(2):
        us.record_usage(key_id="k", user_id="u", audio_s=10.0, words=30, status="ok",
                        kind="dictation", job_id="d1" * 16, hour=h - 1,
                        stages=[{"name": "translate", "secs": 1.0, "targets": ["fr"]}])
    us.record_outcome(user_id="u", job_id="d1" * 16, activation="hold",
                      delivery="typed", translation="kept_original", app_id="vim")
    us.record_usage(key_id="k", user_id="u", audio_s=0.0, words=0, status="ok",
                    kind="text", job_id="t1" * 16, hour=h - 3,
                    stages=[{"name": "translate", "secs": 0.5, "targets": ["de"]}])

    doc = us.document("u", days=7, with_stages=("translating",), tz=_UTC,
                      tz_name="UTC", now=now)
    assert doc["range"]["source"] == "jobs"
    tot = doc["total"]
    assert (tot["all"]["sessions"], tot["all"]["requests"], tot["all"]["errors"]) == (3, 4, 0)
    assert tot["file"]["words"] == 400 and tot["dictation"]["words"] == 60
    assert tot["text"]["sessions"] == 1
    assert doc["today"]["all"]["sessions"] == 3
    stages = {s["stage"]: s for s in doc["stages"]}
    # The chosen stage covers every narrowed run; diarizing co-occurred once
    # among the two narrowed batch jobs.
    assert (stages["translating"]["runs"], stages["translating"]["of_runs"]) == (3, 3)
    assert stages["translating"]["targets"] == [{"code": "de", "runs": 2}, {"code": "fr", "runs": 1}]
    assert stages["translating"]["kept_original"] == 1
    assert (stages["diarizing"]["runs"], stages["diarizing"]["of_runs"]) == (1, 1)
    assert stages["diarizing"]["speakers_avg"] == 2.0
    assert doc["dictation"]["sessions"] == 1
    assert doc["dictation"]["delivery"]["typed"] == 1
    assert doc["apps"] == [{"app_id": "vim", "sessions": 1, "words": 60}]
    assert [c["all"] for c in doc["calendar"]] == [460]
    assert doc["calendar"][0]["file"] == 400
    assert sorted(h["all"] for h in doc["hours"]) == [60, 400]  # 10:00 and 11:00 slots
    assert doc["streak"]["all"] == {"current": 1, "best": 1}
    assert doc["time_saved_s"] == 60 / 40 * 60 - 20

    both = us.document("u", days=7, with_stages=("translating", "diarizing"),
                       tz=_UTC, tz_name="UTC", now=now)
    assert both["total"]["all"]["sessions"] == 1
    assert both["total"]["file"]["words"] == 400
    assert {s["stage"]: (s["runs"], s["of_runs"]) for s in both["stages"]} == {
        "translating": (1, 1), "diarizing": (1, 1)}

    # A window that excludes f1: the error job f2 is diarized-only.
    dia = us.document("u", from_day=_D("2025-06-08"), to_day=_D("2025-06-09"),
                      with_stages=("diarizing",), tz=_UTC, tz_name="UTC", now=now)
    assert dia["total"]["all"] == {"sessions": 1, "requests": 1, "errors": 1,
                                   "words": 200, "audio_s": 50.0, "proc_s": 0.0}
    assert dia["today"]["all"]["sessions"] == 1  # f1 is today, whatever the window
    assert dia["range"]["first_day"] == _D("2025-06-09")


def test_time_saved_and_wpm_over_dictation_only(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=300.0, words=400,
                    status="ok", kind="dictation", job_id="a" * 32)
    # A file's words are not time the user saved by speaking.
    us.record_usage(key_id="k", user_id="u", audio_s=3600.0, words=9000,
                    status="ok", kind="file")
    doc = us.document("u", days=1, tz=_UTC, tz_name="UTC")
    assert doc["time_saved_s"] == 400 / 40 * 60 - 300
    assert doc["dictation"]["wpm"] == 80.0
    assert doc["dictation"]["words"] == 400
    us.record_usage(key_id="k", user_id="slow", audio_s=600.0, words=10,
                    status="ok", kind="dictation")
    assert us.document("slow", days=1, tz=_UTC,
                       tz_name="UTC")["time_saved_s"] == 0.0


# --------------------------------------------------------------------------
# Outcomes + sweep
# --------------------------------------------------------------------------

def test_outcome_is_idempotent_and_rolls_once(usage_store_db):
    us = usage_store_db
    jid = "b" * 32
    for _ in range(2):
        us.record_usage(key_id="k", user_id="u", audio_s=15.0, words=25,
                        status="ok", kind="dictation", job_id=jid)
    assert us.record_outcome(user_id="u", job_id=jid, activation="hold",
                             delivery="typed", translation="not_asked",
                             app_id="thunderbird") == "accepted"
    assert us.record_outcome(user_id="u", job_id=jid, activation="handsfree",
                             delivery="clipboard", translation="translated",
                             app_id="thunderbird") == "duplicate"
    doc = us.document("u", days=1, tz=_UTC, tz_name="UTC")
    d = doc["dictation"]
    assert d["activation"] == {"hold": 1, "handsfree": 0}
    assert d["delivery"] == {"typed": 1, "clipboard": 0, "none": 0, "unreported": 0}
    assert d["translation"]["not_asked"] == 1 and d["translation"]["translated"] == 0
    assert doc["apps"] == [{"app_id": "thunderbird", "sessions": 1, "words": 50}]
    job = us._require_conn().execute(
        "SELECT activation, delivery, app_id FROM usage_jobs").fetchone()
    assert tuple(job) == ("hold", "typed", "thunderbird")


def test_outcome_for_unknown_job_creates_a_stub(usage_store_db):
    us = usage_store_db
    assert us.record_outcome(user_id="u", job_id="e" * 32, activation="hold",
                             delivery="none", translation="aborted") == "accepted"
    job = us._require_conn().execute(
        "SELECT kind, words, audio_s, utterances, reported_ts FROM usage_jobs"
    ).fetchone()
    assert tuple(job)[:4] == ("dictation", 0, 0.0, 0)
    assert job["reported_ts"] is not None
    doc = us.document("u", days=1, tz=_UTC, tz_name="UTC")
    assert doc["dictation"]["delivery"]["none"] == 1
    assert doc["apps"] == []
    assert us.record_outcome(user_id="u", job_id="e" * 32, activation="hold",
                             delivery="none", translation="aborted") == "duplicate"


def test_outcome_scoped_to_owner(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="alice", audio_s=1.0, words=1,
                    status="ok", kind="dictation", job_id="a" * 32)
    assert us.record_outcome(user_id="bob", job_id="a" * 32, activation="hold",
                             delivery="typed", translation="not_asked") == "duplicate"
    assert us._require_conn().execute(
        "SELECT reported_ts FROM usage_jobs").fetchone()["reported_ts"] is None
    assert us.document("bob", days=1, tz=_UTC,
                       tz_name="UTC")["dictation"]["delivery"]["typed"] == 0


def test_sweep_marks_unreported_and_prunes(usage_store_db):
    us = usage_store_db
    now_h = us.now_hour()
    # 48 h old dictation with no outcome; a file job of the same age never
    # reports and must not be marked.
    us.record_usage(key_id="k", user_id="u", audio_s=10.0, words=30, status="ok",
                    kind="dictation", job_id="1" * 32, hour=now_h - 48)
    us.record_usage(key_id="k", user_id="u", audio_s=10.0, words=30, status="ok",
                    kind="file", job_id="2" * 32, hour=now_h - 48,
                    stages=[{"name": "vad", "secs": 1.0}])
    # Fresh dictation: inside the grace period.
    us.record_usage(key_id="k", user_id="u", audio_s=10.0, words=30, status="ok",
                    kind="dictation", job_id="3" * 32)
    us.record_outcome(user_id="u", job_id="3" * 32, activation="hold",
                      delivery="typed", translation="not_asked", app_id="vim")
    # An app row from long ago (past the 90-day clock).
    us._require_conn().execute(
        "INSERT INTO usage_app_hourly VALUES (?, 'u', 'old-app', 1, 1)",
        (now_h - 100 * 24,))

    counts = us.sweep(unreported_after_h=24, jobs_retention_days=0,
                      app_retention_days=90, hourly_retention_days=0)
    assert counts == {"marked": 1, "jobs": 0, "apps": 1, "hourly": 0}
    doc = us.document("u", days=7, tz=_UTC, tz_name="UTC")
    assert doc["dictation"]["delivery"] == {"typed": 1, "clipboard": 0,
                                            "none": 0, "unreported": 1}
    assert doc["dictation"]["translation"]["unreported"] == 1
    assert doc["apps"] == [{"app_id": "vim", "sessions": 1, "words": 30}]
    # Marked once: a second sweep finds nothing.
    assert us.sweep(unreported_after_h=24, jobs_retention_days=0,
                    app_retention_days=90, hourly_retention_days=0)["marked"] == 0

    # Job retention drops the two old jobs and their stage detail; the hourly
    # rollups (and so the totals) survive a job prune.
    counts = us.sweep(unreported_after_h=24, jobs_retention_days=1,
                      app_retention_days=90, hourly_retention_days=0)
    assert counts["jobs"] == 2
    conn = us._require_conn()
    assert conn.execute("SELECT COUNT(*) FROM usage_jobs").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM usage_job_stages").fetchone()[0] == 0
    assert us.totals_for_user("u")["words"] == 90

    # Hourly retention takes the old rollup rows in every hour-keyed table.
    counts = us.sweep(unreported_after_h=24, jobs_retention_days=0,
                      app_retention_days=0, hourly_retention_days=1)
    assert counts["hourly"] >= 3
    assert us.totals_for_user("u")["words"] == 30
    assert conn.execute(
        "SELECT COUNT(*) FROM usage_dictation_hourly").fetchone()[0] == 1


def test_empty_document_shape_matches_populated(usage_store_db):
    us = usage_store_db
    empty = us.empty_document(from_day=_D("2025-01-01"), to_day=_D("2025-01-30"), tz="UTC")
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok",
                    kind="dictation", job_id="a" * 32)
    us.record_outcome(user_id="u", job_id="a" * 32, activation="hold",
                      delivery="typed", translation="not_asked")
    full = us.document("u", days=30, tz=_UTC, tz_name="UTC")
    assert set(empty) == set(full)
    assert set(empty["dictation"]) == set(full["dictation"])
    for k in ("today", "total"):
        assert set(empty[k]) == set(full[k])
        assert set(empty[k]["all"]) == set(full[k]["all"])
    assert set(empty["range"]) == set(full["range"])
    assert set(empty["streak"]) == set(full["streak"]) == {"all", "dictation", "file", "url", "text"}
    assert set(full["calendar"][0]) == {"day", "all", "dictation", "file", "url", "text"}
    assert set(full["hours"][0]) == {"dow", "hour", "all", "dictation", "file", "url", "text"}


def test_fold_runs_on_every_init_so_a_crash_mid_migration_heals(usage_store_db):
    """The rename and the copy are separate steps; if the process dies in
    between, the next init finds the parking table and finishes the job."""
    us = usage_store_db
    conn = us._require_conn()
    # The parked table as _park_legacy_hourly leaves it: renamed, indexes dropped.
    table_only = _LEGACY_SCHEMA.split("CREATE INDEX")[0]
    conn.executescript(table_only.replace("usage_hourly", "usage_hourly_legacy"))
    conn.execute("INSERT INTO usage_hourly_legacy VALUES (7, 'k', 'u', 2, 0, 5, 1.0)")
    us._fold_legacy_hourly(conn)
    assert us.totals_for_user("u")["requests"] == 2
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='usage_hourly_legacy'").fetchone() is None
