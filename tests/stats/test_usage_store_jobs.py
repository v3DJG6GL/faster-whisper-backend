"""usage_store: the per-kind / per-job / per-stage rollups behind the desktop
app's statistics — the migration from the per-request rollup, session
counting, the outcome + sweep write paths, and the document() read path
(tz-aware day reckoning, stage meters, streak, time saved).
"""

import datetime
import sqlite3
import time
import zoneinfo

import pytest

from tests.conftest import _close_store
from tests.stats._usage_helpers import (
    _UTC, epoch_day as _D, outcome as _outcome, rec as _rec, utc_doc as _doc)

_ZH = zoneinfo.ZoneInfo("Europe/Zurich")


def _ts(y, m, d, hh=12, mm=0, tz=_UTC) -> float:
    return datetime.datetime(y, m, d, hh, mm, tzinfo=tz).timestamp()


def _hour(y, m, d, hh=12, mm=0, tz=_UTC) -> int:
    return int(_ts(y, m, d, hh, mm, tz) // 3600)


def _row(us, sql):
    return us._require_conn().execute(sql).fetchone()


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
    as kind='dictation' with one session per request, and the parking table
    is gone. A second init is a no-op."""
    from faster_whisper_backend.stats import usage_store
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
        assert {"kind", "sessions", "processing_s"} <= cols
        rows = conn.execute(
            "SELECT hour, kind, requests, sessions, words, audio_s"
            " FROM usage_hourly ORDER BY hour").fetchall()
        assert [tuple(r) for r in rows] == [
            (100, "dictation", 3, 3, 42, 9.5), (101, "dictation", 1, 1, 7, 1.0)]
        assert _row(
            usage_store, "SELECT 1 FROM sqlite_master WHERE name='usage_hourly_legacy'"
        ) is None
        idx = {r["name"] for r in conn.execute("PRAGMA index_list(usage_hourly)")}
        assert {"idx_usage_user_hour", "idx_usage_hour"} <= idx
        # The old rows still feed the folded admin views.
        assert usage_store.totals_for_user("u")["words"] == 49
        conn.close()
        usage_store.init_db(path)
        assert usage_store.totals_for_user("u")["requests"] == 4
    finally:
        _close_store(usage_store)


def test_failed_init_leaves_the_store_closed(tmp_path, monkeypatch):
    """A migration step that raises (a lock held past the busy timeout, a
    full disk) must not leave the half-migrated connection behind: main.py
    logs the store as down and starts no sweep, so readers must get the
    RuntimeError the routes turn into 503, not a partly live store."""
    from faster_whisper_backend.stats import usage_store
    monkeypatch.setattr(usage_store, "_conn", None)

    def locked(conn):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(usage_store, "_fold_legacy_hourly", locked)
    with pytest.raises(sqlite3.OperationalError):
        usage_store.init_db(str(tmp_path / "usage.sqlite3"))
    with pytest.raises(RuntimeError):
        usage_store.totals_by_key()


def test_sessions_counted_once_per_job_across_utterances(us):
    for words in (5, 7, 9):
        _rec(us, audio_s=2.0, words=words, kind="dictation", job_id="a" * 32, processing_s=0.5)
    row = _row(us, "SELECT requests, sessions, words, processing_s FROM usage_hourly")
    assert tuple(row) == (3, 1, 21, 1.5)
    job = _row(us, "SELECT utterances, words, audio_s, kind FROM usage_jobs")
    assert tuple(job) == (3, 21, 6.0, "dictation")
    # Without a job id every request is its own session.
    _rec(us, audio_s=1.0, words=1, kind="file")
    _rec(us, audio_s=1.0, words=1, kind="file")
    doc = _doc(us, days=7)
    assert doc["today"]["dictation"]["sessions"] == 1
    assert doc["today"]["dictation"]["requests"] == 3
    assert doc["today"]["file"]["sessions"] == 2
    assert doc["today"]["all"]["sessions"] == 3


def test_colliding_job_id_of_another_user_is_not_merged(us):
    _rec(us, key_id="ka", user_id="alice", audio_s=1.0, words=10, kind="dictation", job_id="c" * 32)
    _rec(us, key_id="kb", user_id="bob", audio_s=1.0, words=20, kind="dictation", job_id="c" * 32)
    job = _row(us, "SELECT user_id, words, utterances FROM usage_jobs")
    assert tuple(job) == ("alice", 10, 1)
    # Bob's work is still counted — as a session of its own.
    assert _doc(us, "bob", days=1)["today"]["dictation"]["sessions"] == 1


def test_job_language_is_capped(us):
    # The streaming fallback hands the client's handshake value through;
    # stored at most 32 chars, like recent_transcriptions_store.
    _rec(us, audio_s=1.0, words=1, kind="dictation", job_id="l" * 32, language="x" * 500)
    row = _row(us, "SELECT language FROM usage_jobs")
    assert row["language"] == "x" * 32


def test_job_model_is_capped(us):
    # A refused stream handshake hands its raw `model` string (up to the
    # socket's max frame size) through; stored at most 128 chars, like
    # jobs_store.
    _rec(us, audio_s=1.0, words=1, status="error", kind="dictation", job_id="m" * 32,
         model="m" * 10000)
    row = _row(us, "SELECT model FROM usage_jobs")
    assert len(row["model"]) == 128


def test_init_reclassifies_unknown_rows_as_dictation(tmp_path):
    """A DB migrated by the first per-kind build holds its history as
    'unknown'; the next start folds it into 'dictation', summing into a
    dictation row that shares the hour×key, and a further start is a no-op."""
    from faster_whisper_backend.stats import usage_store
    path = str(tmp_path / "unknown.sqlite3")
    usage_store.init_db(path)
    conn = usage_store._require_conn()
    conn.executescript(
        "INSERT INTO usage_hourly (hour, key_id, user_id, kind, requests, errors,"
        " words, audio_s, processing_s, sessions) VALUES"
        " (100, 'k', 'u', 'unknown', 3, 1, 42, 9.5, 0, 3),"
        " (101, 'k', 'u', 'unknown', 1, 0, 7, 1.0, 0, 1),"
        " (101, 'k', 'u', 'dictation', 2, 0, 10, 2.0, 0.5, 1),"
        " (102, 'k', 'u', 'file', 1, 0, 5, 3.0, 1.0, 1);")
    conn.commit()
    conn.close()
    try:
        for _ in range(2):
            usage_store.init_db(path)
            conn = usage_store._require_conn()
            rows = conn.execute(
                "SELECT hour, kind, requests, errors, sessions, words, audio_s, processing_s"
                " FROM usage_hourly ORDER BY hour, kind").fetchall()
            assert [tuple(r) for r in rows] == [
                (100, "dictation", 3, 1, 3, 42, 9.5, 0),
                (101, "dictation", 3, 0, 2, 17, 3.0, 0.5),
                (102, "file", 1, 0, 1, 5, 3.0, 1.0)]
            conn.close()
    finally:
        _close_store(usage_store)


def test_kind_totals_and_unknown_folds_into_all_only(us):
    for kind, words in (("dictation", 1), ("file", 2), ("url", 4), ("text", 8),
                        (None, 16), ("bogus", 32)):
        _rec(us, audio_s=1.0, words=words, kind=kind)
    doc = _doc(us, days=1)
    total = doc["total"]
    assert set(total) == {"all", "dictation", "file", "url", "text"}
    assert [total[k]["words"] for k in ("dictation", "file", "url", "text")] == [1, 2, 4, 8]
    assert total["all"]["words"] == 63
    assert total["all"]["requests"] == 6


def test_dictation_targets_per_language_with_kept_count(us):
    """`dictation.targets` lists the languages dictations translated into,
    busiest first, with how many of those runs kept the original; file runs
    and jobs outside the window do not count; the `with=` path agrees."""
    now = time.time()
    tr = lambda *codes: [{"name": "translating", "secs": 1.0, "targets": list(codes)}]
    for jid, codes in (("a", ("de",)), ("b", ("de", "en")), ("c", ("de",))):
        _rec(us, audio_s=5.0, words=20, kind="dictation", job_id=jid * 32, stages=tr(*codes))
    _rec(us, audio_s=5.0, words=20, kind="file", job_id="f" * 32, stages=tr("de"))
    _rec(us, audio_s=5.0, words=20, kind="dictation", job_id="n" * 32)
    _outcome(us, "a" * 32, translation="translated")
    _outcome(us, "b" * 32, translation="kept_original")
    # An old translated dictation, outside a one-day window.
    us._require_conn().execute(
        "UPDATE usage_jobs SET created_ts = ? WHERE job_id = ?",
        (now - 10 * 86400, "c" * 32))
    us._require_conn().commit()
    doc = _doc(us, days=1)
    assert doc["dictation"]["targets"] == [
        {"code": "de", "runs": 2, "kept_original": 1},
        {"code": "en", "runs": 1, "kept_original": 1}]
    wide = _doc(us, days=30)
    assert wide["dictation"]["targets"][0] == {"code": "de", "runs": 3, "kept_original": 1}
    narrowed = _doc(us, days=1, with_stages=("translating",))
    assert narrowed["dictation"]["targets"] == doc["dictation"]["targets"]


def test_translation_attaches_to_its_dictation_session(us):
    """A dictation's follow-up translation is its own POST under a fresh
    progress id. With session_job_id naming the same user's dictation job,
    its stage attaches there (dictation.targets filled, no second text
    session); another user's session id, an unknown one and a failed
    translation keep their own text job."""
    sess = "s" * 32
    tr = [{"name": "translate", "secs": 1.0, "targets": ["de"]}]
    _rec(us, audio_s=5.0, words=20, kind="dictation", job_id=sess)
    _rec(us, audio_s=0.0, words=0, kind="text", job_id="p" * 32, stages=tr, session_job_id=sess)
    doc = _doc(us, days=1)
    assert doc["dictation"]["targets"] == [
        {"code": "de", "runs": 1, "kept_original": 0}]
    meter = next(s for s in doc["stages"] if s["stage"] == "translating")
    assert (meter["runs"], meter["of_runs"]) == (1, 1)
    assert doc["total"]["text"]["requests"] == 1
    assert doc["total"]["text"]["sessions"] == 0
    conn = us._require_conn()
    assert conn.execute("SELECT 1 FROM usage_jobs WHERE job_id = ?",
                        ("p" * 32,)).fetchone() is None
    # Someone else's session, an unknown id, a failed run: own text job.
    _rec(us, key_id="k2", user_id="v", audio_s=0.0, words=0, kind="text", job_id="q" * 32,
         stages=tr, session_job_id=sess)
    _rec(us, audio_s=0.0, words=0, kind="text", job_id="r" * 32, stages=tr, session_job_id="x" * 32)
    _rec(us, audio_s=0.0, words=0, status="error", kind="text", job_id="e" * 32,
         stages=tr, session_job_id=sess)
    for jid in ("q", "r", "e"):
        assert conn.execute("SELECT kind FROM usage_jobs WHERE job_id = ?",
                            (jid * 32,)).fetchone()["kind"] == "text", jid
    assert _doc(us, days=1)["dictation"]["targets"][0]["runs"] == 1


def test_per_utterance_translations_count_one_run_per_session(us):
    """Per-utterance translation sends one POST per utterance, all naming
    the same session: on both paths the session counts one translating run
    (not "3 of 1") and lists each language once (en, en, fr -> en, fr)."""
    sess = "s" * 32
    tr = lambda code: [{"name": "translate", "secs": 1.0, "targets": [code]}]
    for i, code in enumerate(("en", "en", "fr")):
        _rec(us, audio_s=5.0, words=20, kind="dictation", job_id=sess)
        _rec(us, audio_s=0.0, words=0, kind="text", job_id=f"{i}" * 32, stages=tr(code),
             session_job_id=sess)
    doc = _doc(us, days=1)
    meter = next(s for s in doc["stages"] if s["stage"] == "translating")
    assert (meter["runs"], meter["of_runs"]) == (1, 1)
    assert meter["targets"] == [{"code": "en", "runs": 1},
                                {"code": "fr", "runs": 1}]
    assert doc["dictation"]["targets"] == [
        {"code": "en", "runs": 1, "kept_original": 0},
        {"code": "fr", "runs": 1, "kept_original": 0}]
    narrowed = _doc(us, days=1, with_stages=("translating",))
    wmeter = next(s for s in narrowed["stages"] if s["stage"] == "translating")
    assert (wmeter["runs"], wmeter["of_runs"]) == (1, 1)
    assert wmeter["targets"] == meter["targets"]
    assert narrowed["dictation"]["targets"] == doc["dictation"]["targets"]


def test_dictation_utterance_stage_counts_one_run_per_session(us):
    """Every dictation utterance carries a "transcribing" stage under the
    session's job id; the meter and the by=stage leaderboard still count one
    run per session (not "3 of 1")."""
    for _ in range(3):
        _rec(us, audio_s=5.0, words=20, kind="dictation", job_id="s" * 32,
             stages=[{"name": "transcribing", "secs": 1.0,
                      "model": "large-v3"}])
    doc = _doc(us, days=1, all_stages=True)
    meter = next(s for s in doc["stages"] if s["stage"] == "transcribing")
    assert (meter["runs"], meter["of_runs"], meter["secs"]) == (1, 1, 3.0)
    o = us.overview(user_id="u", tz=_UTC, tz_name="UTC", days=1, by="stage")
    board = {r["id"]: r["totals"] for r in o["leaderboard"]}
    assert board["transcribing"]["sessions"] == 1


def test_attached_translation_audio_is_the_same_on_both_paths(us):
    """A per-utterance translation attached to a dictation has no audio of
    its own: the rollup, the with= path and the by=stage board all read 0 s
    for translating, not the joined dictation's audio (a made-up RTF)."""
    sess = "s" * 32
    _rec(us, audio_s=30.0, words=20, kind="dictation", job_id=sess,
         stages=[{"name": "vad", "secs": 0.1, "retained": 0.9}])
    _rec(us, audio_s=0.0, words=0, kind="text", job_id="p" * 32, session_job_id=sess,
         stages=[{"name": "translate", "secs": 1.0, "targets": ["de"]}])
    for with_stages in ((), ("translating",)):
        doc = _doc(us, days=1, with_stages=with_stages)
        st = {s["stage"]: s for s in doc["stages"]}
        assert st["translating"]["audio_s"] == 0.0, with_stages
        assert st["vad"]["audio_s"] == 30.0, with_stages
    for with_stages in ((), ("translating",)):
        o = us.overview(user_id="u", tz=_UTC, tz_name="UTC", days=1, by="stage",
                        with_stages=with_stages)
        board = {r["id"]: r["totals"] for r in o["leaderboard"]}
        assert board["translating"]["audio_s"] == 0.0, with_stages


def test_dictation_utterance_vad_retained_averages_over_one_run(us):
    """Each dictation utterance carries a "vad" stage under the session's
    job id. retained averages over one run per session, and both paths keep
    the session's first utterance: 0.9 / 0.5 / 0.4 read 0.9, never a summed
    1.8 or the latest 0.4."""
    for retained in (0.9, 0.5, 0.4):
        _rec(us, audio_s=5.0, words=20, kind="dictation", job_id="J" * 32,
             stages=[{"name": "vad", "secs": 0.1, "retained": retained}])
    for with_stages in ((), ("vad",)):
        doc = _doc(us, days=1, with_stages=with_stages)
        meter = next(s for s in doc["stages"] if s["stage"] == "vad")
        assert (meter["runs"], meter["retained_avg"]) == (1, 0.9), with_stages


def test_attached_targets_past_the_cap_count_once(us):
    """The per-job target union is capped; a target past the cap was never
    stored, read as new on every later attach and counted again each time."""
    sess = "s" * 32
    cap = us._JOB_TARGETS_MAX
    codes = [f"l{i}" for i in range(cap)]
    _rec(us, audio_s=5.0, words=20, kind="dictation", job_id=sess)
    tr = lambda cs: [{"name": "translate", "secs": 1.0, "targets": cs}]
    _rec(us, audio_s=0.0, words=0, kind="text", job_id="p" * 32, stages=tr(codes),
         session_job_id=sess)
    for i in range(3):
        _rec(us, audio_s=0.0, words=0, kind="text", job_id=f"{i}" * 32, stages=tr(["xx"]),
             session_job_id=sess)
    assert _row(us, "SELECT COUNT(*) FROM usage_target_hourly WHERE target = 'xx'")[0] == 0
    assert _row(us, "SELECT SUM(runs) FROM usage_target_hourly")[0] == cap


def test_translating_kept_original_is_the_segment_count_on_both_paths(us):
    """The translating stage's kept_original is the guard's segment fallback
    count on both paths, never the dictation outcome's kept-original jobs."""
    jid = "d" * 32
    _rec(us, audio_s=5.0, words=20, kind="dictation", job_id=jid,
         stages=[{"name": "translating", "secs": 1.0,
                  "targets": ["de"], "kept_original": 3}])
    _outcome(us, jid, translation="translated")
    for with_stages in ((), ("translating",)):
        doc = _doc(us, days=1, with_stages=with_stages)
        meter = next(s for s in doc["stages"] if s["stage"] == "translating")
        assert meter["kept_original"] == 3, with_stages


def test_outcome_before_a_file_run_does_not_make_it_a_dictation(us):
    """record_outcome only refuses a non-dictation job that already exists,
    so an early outcome leaves a dictation stub; a file run under that id
    must take it over as kind='file', outside the dictation rollups."""
    jid = "f" * 32
    assert _outcome(us, jid, delivery="typed", app_id="vim") == "accepted"
    _rec(us, audio_s=100.0, words=500, kind="file", job_id=jid)
    job = _row(us, "SELECT kind, utterances, reported_ts, activation FROM usage_jobs")
    assert tuple(job) == ("file", 1, None, None)
    assert _row(us, "SELECT COUNT(*) FROM usage_dictation_hourly")[0] == 0
    assert _row(us, "SELECT COUNT(*) FROM usage_app_hourly")[0] == 0
    doc = _doc(us, days=1)
    assert doc["total"]["file"]["sessions"] == 1
    assert doc["dictation"]["sessions"] == 0
    assert doc["dictation"]["delivery"]["typed"] == 0


def test_foreign_outcome_stub_does_not_drop_a_file_run(us):
    """Another key holder who saw a batch run's request_id (a live trace, a
    capture row) can post an outcome under it before the run records usage.
    That bare dictation stub must not cost the run its job and stage rows:
    the run takes the row over and the stub's session leaves the poster's
    dictation / app buckets."""
    jid = "a" * 32
    assert _outcome(us, jid, user_id="mallory", app_id="vim") == "accepted"
    _rec(us, user_id="alice", key_id="ka", audio_s=100.0, words=500,
         kind="file", job_id=jid,
         stages=[{"name": "transcribing", "secs": 3.0, "model": "m"}])
    job = _row(us, "SELECT user_id, key_id, kind, utterances, reported_ts,"
                   " activation, app_id, words FROM usage_jobs")
    assert tuple(job) == ("alice", "ka", "file", 1, None, None, None, 500)
    stage = _row(us, "SELECT job_id, stage FROM usage_job_stages")
    assert tuple(stage) == (jid, "transcribing")
    assert _row(us, "SELECT COUNT(*) FROM usage_dictation_hourly")[0] == 0
    assert _row(us, "SELECT COUNT(*) FROM usage_app_hourly")[0] == 0
    assert _doc(us, user_id="alice", days=1)["total"]["file"]["sessions"] == 1
    assert _doc(us, user_id="mallory", days=1)["dictation"]["sessions"] == 0


def test_foreign_stub_stage_rows_stay_out_of_the_run_that_takes_it_over(us):
    """The stub's owner can attach translation stage rows to it (a
    translation's session_job_id). The run that takes the stub over gets
    only its own stage rows; the poster's hourly stage rollup keeps the
    translation, like any request whose job id collided."""
    jid = "b" * 32
    _outcome(us, jid, user_id="mallory")
    _rec(us, user_id="mallory", audio_s=1.0, words=3, kind="text",
         job_id="c" * 32, session_job_id=jid,
         stages=[{"name": "translating", "secs": 2.0, "targets": ["de"]}])
    _rec(us, user_id="alice", audio_s=100.0, words=500, kind="file",
         job_id=jid, stages=[{"name": "transcribing", "secs": 3.0}])
    stages = us._require_conn().execute(
        "SELECT stage FROM usage_job_stages WHERE job_id = ?", (jid,)).fetchall()
    assert [r["stage"] for r in stages] == ["transcribing"]
    runs = us._require_conn().execute(
        "SELECT user_id, stage, runs FROM usage_stage_hourly"
        " ORDER BY user_id").fetchall()
    assert [tuple(r) for r in runs] == [
        ("alice", "transcribing", 1), ("mallory", "translating", 1)]


def test_foreign_dictation_rows_are_never_merged(us):
    """Collisions with a foreign dictation stub or session never merge into
    it: a dictation still counts its work without a job row, a file run
    keeps a job of its own under its derived id."""
    _outcome(us, "s" * 32, user_id="mallory")
    _rec(us, user_id="alice", audio_s=5.0, words=20, kind="dictation",
         job_id="s" * 32)
    _rec(us, user_id="mallory", audio_s=5.0, words=20, kind="dictation",
         job_id="t" * 32)
    _rec(us, user_id="alice", audio_s=5.0, words=20, kind="file",
         job_id="t" * 32)
    rows = us._require_conn().execute(
        "SELECT job_id, user_id, kind, utterances FROM usage_jobs"
        " ORDER BY job_id").fetchall()
    assert [tuple(r) for r in rows] == [
        ("s" * 32, "mallory", "dictation", 0),
        ("t" * 32, "mallory", "dictation", 1),
        ("t" * 32 + "~alice", "alice", "file", 1)]
    assert _doc(us, user_id="mallory", days=1)["dictation"]["sessions"] == 2


@pytest.mark.parametrize("first_kind", ["dictation", "text"])
def test_foreign_session_under_the_id_does_not_drop_a_file_run(us, first_kind):
    """Bob's dictation session or text request that named Alice's batch id
    first keeps its row; Alice's file run keeps its own job and stage rows
    under an id derived from hers, which no client-minted (hex) id can
    claim."""
    jid = "e" * 32
    _rec(us, user_id="bob", audio_s=1.0, words=2, kind=first_kind, job_id=jid)
    _rec(us, user_id="alice", audio_s=100.0, words=500, kind="file",
         job_id=jid, stages=[{"name": "transcribing", "secs": 3.0}])
    rows = us._require_conn().execute(
        "SELECT job_id, user_id, kind, utterances, words FROM usage_jobs"
        " ORDER BY job_id").fetchall()
    assert [tuple(r) for r in rows] == [
        (jid, "bob", first_kind, 1, 2),
        (f"{jid}~alice", "alice", "file", 1, 500)]
    stage = _row(us, "SELECT job_id, stage FROM usage_job_stages")
    assert tuple(stage) == (f"{jid}~alice", "transcribing")
    # A re-post of the same batch by Alice accumulates on her derived row.
    _rec(us, user_id="alice", audio_s=1.0, words=5, kind="file", job_id=jid)
    assert _row(us, "SELECT utterances, words FROM usage_jobs"
                    f" WHERE job_id = '{jid}~alice'")[:] == (2, 505)
    assert _doc(us, user_id="alice", days=1)["total"]["file"]["sessions"] == 1


def _stub_at(us, monkeypatch, jid, ts, **kw):
    """record_outcome's stub for `jid`, stamped at epoch `ts`."""
    with monkeypatch.context() as m:
        m.setattr(us.time, "time", lambda: ts)
        assert _outcome(us, jid, **kw) == "accepted"


def test_foreign_stub_older_than_the_run_is_left_to_its_owner(us, monkeypatch):
    """Only a stub that appeared while the run was under way can be a squat
    on its id. Bob's own dictation outcome stub from before Alice's run
    started stays his: his session keeps counting and Alice's run keeps its
    job and stage rows under its derived id."""
    jid = "d" * 32
    _stub_at(us, monkeypatch, jid, time.time() - 600, user_id="bob", app_id="vim")
    _rec(us, user_id="alice", audio_s=100.0, words=500, kind="file",
         job_id=jid, processing_s=30.0, wait_s=5.0,
         stages=[{"name": "transcribing", "secs": 3.0}])
    job = _row(us, "SELECT user_id, kind, utterances, app_id, reported_ts IS NOT NULL"
                   f" FROM usage_jobs WHERE job_id = '{jid}'")
    assert tuple(job) == ("bob", "dictation", 0, "vim", 1)
    mine = _row(us, "SELECT user_id, kind, utterances FROM usage_jobs"
                    f" WHERE job_id = '{jid}~alice'")
    assert tuple(mine) == ("alice", "file", 1)
    stage = _row(us, "SELECT job_id, stage FROM usage_job_stages")
    assert tuple(stage) == (f"{jid}~alice", "transcribing")
    assert _row(us, "SELECT SUM(sessions) FROM usage_dictation_hourly")[0] == 1
    assert _row(us, "SELECT SUM(sessions) FROM usage_app_hourly")[0] == 1
    assert _doc(us, user_id="alice", days=1)["total"]["file"]["sessions"] == 1


def test_foreign_stub_posted_while_the_run_was_under_way_is_taken_over(
        us, monkeypatch):
    """A stub younger than the run's start (record time minus processing and
    queue time) is a squat on the run's id: the run takes it over."""
    jid = "e" * 32
    _stub_at(us, monkeypatch, jid, time.time() - 120, user_id="mallory")
    _rec(us, user_id="alice", audio_s=100.0, words=500, kind="file",
         job_id=jid, processing_s=200.0, wait_s=5.0)
    job = _row(us, "SELECT user_id, kind, utterances FROM usage_jobs")
    assert tuple(job) == ("alice", "file", 1)
    assert _row(us, "SELECT COUNT(*) FROM usage_dictation_hourly")[0] == 0


def test_with_document_dictation_targets_honour_the_kind_filter(us):
    """The with= document's dictation.targets honours the kind filter like
    every other figure: kind=file lists none next to dictation.sessions 0."""
    tr = lambda code: [{"name": "translating", "secs": 1.0, "targets": [code]}]
    _rec(us, audio_s=5.0, words=20, kind="dictation", job_id="d" * 32, stages=tr("de"))
    _rec(us, audio_s=5.0, words=20, kind="file", job_id="f" * 32, stages=tr("fr"))
    files = _doc(us, days=1, with_stages=("translating",), kinds=("file",))
    assert "kind_scoped" not in files["range"]
    assert files["dictation"]["sessions"] == 0
    assert files["total"]["all"]["sessions"] == 1
    assert files["dictation"]["targets"] == []
    dicts = _doc(us, days=1, with_stages=("translating",), kinds=("dictation",))
    assert dicts["dictation"]["targets"] == [
        {"code": "de", "runs": 1, "kept_original": 0}]


def test_rollup_document_dictation_targets_honour_the_kind_filter(us):
    """Rollup-path twin: without with_stages, dictation.sessions is
    kind-filtered, so the per-job targets read must be too."""
    _rec(us, audio_s=5.0, words=20, kind="dictation", job_id="d" * 32,
         stages=[{"name": "translating", "secs": 1.0, "targets": ["de"]}])
    files = _doc(us, days=1, kinds=("file",))
    assert files["dictation"]["sessions"] == 0
    assert files["dictation"]["targets"] == []
    dicts = _doc(us, days=1, kinds=("dictation",))
    assert dicts["dictation"]["targets"] == [
        {"code": "de", "runs": 1, "kept_original": 0}]


def test_stage_and_target_rollups(us):
    _rec(
        us, audio_s=100.0, words=50, kind="file", job_id="f" * 32, stages=[
            {"name": "transcribing", "secs": 4.0, "model": "large-v3"},
            {"name": "diarizing", "secs": 3.0, "speakers": 3,
             "detail": "3 speakers"},
            {"name": "vad", "secs": 1.0, "retained": 0.5},
            {"name": "translating", "secs": 2.0, "targets": ["de", "fr"],
             "kept_original": 2},
        ])
    # The text endpoint names its stage 'translate' — same row.
    _rec(
        us, audio_s=0.0, words=0, kind="text", job_id="t" * 32,
        stages=[{"name": "translate", "secs": 1.5, "targets": ["de"]}])
    # A dictation cannot have been diarized — it must not widen that meter.
    _rec(us, audio_s=5.0, words=9, kind="dictation", job_id="d" * 32)

    doc = _doc(us, days=1)
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


def test_days_reckoned_in_caller_zone_across_dst_end(us):
    """Europe/Zurich leaves DST on 2025-10-26. An utterance at 00:30 local on
    the 26th (22:30 UTC on the 25th) and one at 00:30 local on the 27th
    (23:30 UTC on the 26th — the offset has changed) must land on the local
    dates, and on the UTC dates for a UTC caller."""
    _rec(us, audio_s=1.0, words=10, kind="dictation", hour=_hour(2025, 10, 26, 0, 30, _ZH))
    _rec(us, audio_s=1.0, words=20, kind="dictation", hour=_hour(2025, 10, 27, 0, 30, _ZH))
    now = _ts(2025, 10, 27, 12, 0, _UTC)

    zh = us.document("u", days=7, tz=_ZH, tz_name="Europe/Zurich",
                     now=now)
    assert [(p["day"], p["all"]["words"]) for p in zh["series"]] == [
        (_D("2025-10-26"), 10), (_D("2025-10-27"), 20)]
    assert zh["today"]["all"]["words"] == 20
    assert zh["streak"]["all"] == {"current": 2, "best": 2}

    utc = _doc(us, days=7, now=now)
    assert [(p["day"], p["all"]["words"]) for p in utc["series"]] == [
        (_D("2025-10-25"), 10), (_D("2025-10-26"), 20)]
    assert utc["today"]["all"]["words"] == 0
    # Yesterday's streak still counts while today is empty.
    assert utc["streak"]["all"] == {"current": 2, "best": 2}



def test_half_hour_zone_files_the_boundary_hour_inside_the_window(us):
    """Asia/Kolkata is +5:30, so one UTC hour straddles each local midnight
    and is filed under the day it starts in. The hour from 23:30 on the 9th
    is the 9th's and stays out of a 10..11 window; the one from 23:30 on the
    11th is the 11th's and counts. A floored bound did the reverse: the
    series carried a day before range.from and dropped the last day's
    evening. Same on the per-job (with=) path, and 'today' agrees."""
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    for d, words in ((9, 10), (11, 5)):
        _rec(us, audio_s=1.0, words=words, kind="file", job_id=f"j{d}",
             hour=_hour(2025, 3, d, 23, 30, ist), stages=[{"name": "diarizing", "secs": 1.0}])
    now = _ts(2025, 3, 11, 12, 0, _UTC)
    for with_stages in ((), ("diarizing",)):
        doc = us.document("u", from_day=_D("2025-03-10"), to_day=_D("2025-03-11"),
                          tz=ist, tz_name="Asia/Kolkata", now=now,
                          with_stages=with_stages)
        assert [(p["day"], p["all"]["words"]) for p in doc["series"]] == [
            (_D("2025-03-11"), 5)], with_stages
        assert doc["total"]["all"]["words"] == 5
        assert doc["today"]["all"]["words"] == 5

def test_server_local_fallback_matches_tz_none(us, set_tz):
    set_tz("America/New_York")
    # 23:00 New York on the 3rd = 03:00 UTC on the 4th.
    ny = zoneinfo.ZoneInfo("America/New_York")
    _rec(us, audio_s=1.0, words=5, kind="file", hour=_hour(2025, 6, 3, 23, 0, ny))
    doc = us.document("u", days=30, tz=None, tz_name="local",
                      now=_ts(2025, 6, 4, 12, 0, _UTC))
    assert doc["series"][0]["day"] == _D("2025-06-03")
    assert doc["tz"] == "local"


def test_window_forms_days_from_to_and_all(us):
    """`days` ends today; `from`/`to` are inclusive caller-local days; `all`
    starts at the first day with usage. total/series/calendar/hours cover the
    window; the streak runs over the whole history."""
    now = _ts(2025, 3, 20)
    for days_ago, words in ((0, 1), (5, 2), (40, 4), (100, 8)):
        _rec(us, audio_s=1.0, words=words, kind="text", hour=int(now // 3600) - days_ago * 24)
    today = _D("2025-03-20")
    doc = _doc(us, days=7, now=now)
    assert doc["range"] == {"from": today - 6, "to": today, "days": 7,
                            "first_day": _D("2024-12-10"), "source": "rollups",
                            "jobs_retention_days": 365}
    assert doc["total"]["all"]["words"] == 3
    assert [c["all"] for c in doc["calendar"]] == [2, 1]
    assert [c["text"] for c in doc["calendar"]] == [2, 1]
    assert doc["streak"]["all"] == {"current": 1, "best": 1}
    assert doc["streak"]["text"] == {"current": 1, "best": 1}
    assert doc["streak"]["file"] == {"current": 0, "best": 0}

    doc = _doc(us, from_day=today - 45, to_day=today - 30, now=now)
    assert doc["range"]["days"] == 16
    assert doc["total"]["all"]["words"] == 4
    assert [p["day"] for p in doc["series"]] == [today - 40]
    assert doc["today"]["all"]["words"] == 1  # today is today whatever the window

    doc = _doc(us, all_time=True, now=now)
    assert doc["range"]["from"] == _D("2024-12-10") and doc["range"]["to"] == today
    assert doc["total"]["all"]["words"] == 15
    assert len(doc["calendar"]) == 4

    # `to` alone ends the default 30-day window there; the span is clamped.
    doc = _doc(us, to_day=today - 30, now=now)
    assert doc["range"] ["from"] == today - 59
    doc = _doc(us, from_day=0, to_day=today, now=now)
    assert doc["range"]["days"] == us.MAX_WINDOW_DAYS
    with pytest.raises(ValueError):
        _doc(us, from_day=today, to_day=today - 1, now=now)
    with pytest.raises(ValueError):
        _doc(us, with_stages=("decoding",), now=now)
    doc = _doc(us, "nobody", all_time=True, now=now)
    assert doc["range"]["from"] == today and doc["range"]["first_day"] is None


def test_streak_is_per_kind_and_never_capped_by_the_window(us):
    now = _ts(2025, 3, 20)
    for days_ago in range(0, 12):
        _rec(us, audio_s=1.0, words=1, kind="dictation", hour=int(now // 3600) - days_ago * 24)
    for days_ago in (0, 3, 4, 5, 6, 9):
        _rec(us, audio_s=1.0, words=1, kind="file", hour=int(now // 3600) - days_ago * 24)
    doc = _doc(us, days=7, now=now)
    assert doc["streak"]["all"] == {"current": 12, "best": 12}
    assert doc["streak"]["dictation"] == {"current": 12, "best": 12}
    assert doc["streak"]["file"] == {"current": 1, "best": 4}
    assert doc["streak"]["url"] == {"current": 0, "best": 0}


def test_hours_grid_is_local_weekday_by_hour_per_kind(us):
    """Europe/Zurich leaves DST on 2025-10-26: 09:30 local is 07:30 UTC on
    the 25th (Sat) and 08:30 UTC on the 27th (Mon). Both land in the local
    09:00 slot of their local weekday; a UTC caller sees them at 07 and 08."""
    _rec(us, audio_s=1.0, words=10, kind="dictation", hour=_hour(2025, 10, 25, 9, 30, _ZH))
    _rec(us, audio_s=1.0, words=20, kind="file", hour=_hour(2025, 10, 27, 9, 30, _ZH))
    _rec(us, audio_s=1.0, words=5, kind="dictation", hour=_hour(2025, 10, 27, 9, 45, _ZH))
    now = _ts(2025, 10, 27, 12, 0, _UTC)
    zh = us.document("u", days=7, tz=_ZH, tz_name="Europe/Zurich", now=now)
    flat = [{k: v for k, v in h.items() if k not in us.SLOT_MEASURES}
            for h in zh["hours"]]
    assert flat == [
        {"dow": 0, "hour": 9, "all": 25, "dictation": 5, "file": 20, "url": 0, "text": 0},
        {"dow": 5, "hour": 9, "all": 10, "dictation": 10, "file": 0, "url": 0, "text": 0},
    ]
    utc = _doc(us, days=7, now=now)
    assert [(h["dow"], h["hour"], h["all"]) for h in utc["hours"]] == [
        (0, 8, 25), (5, 7, 10)]


def test_with_stages_narrows_to_jobs_that_ran_all_of_them(us):
    now = _ts(2025, 6, 10, 12, 0, _UTC)
    h = int(now // 3600)
    # f1: translated + diarized; f2: diarized only; d1: dictation translated
    # (via the text stage), t1: text translate.
    _rec(us, audio_s=100.0, words=400, kind="file", job_id="f1" * 16, hour=h - 2, stages=[
             {"name": "translating", "secs": 3.0, "targets": ["de"]},
             {"name": "diarizing", "secs": 5.0, "speakers": 2}])
    _rec(us, audio_s=50.0, words=200, status="error", kind="file", job_id="f2" * 16, hour=h - 30,
         stages=[{"name": "diarizing", "secs": 2.0, "speakers": 4}])
    for _ in range(2):
        _rec(us, audio_s=10.0, words=30, kind="dictation", job_id="d1" * 16, hour=h - 1,
             stages=[{"name": "translate", "secs": 1.0, "targets": ["fr"]}])
    _outcome(us, "d1" * 16, delivery="typed", translation="kept_original", app_id="vim")
    _rec(us, audio_s=0.0, words=0, kind="text", job_id="t1" * 16, hour=h - 3,
         stages=[{"name": "translate", "secs": 0.5, "targets": ["de"]}])

    doc = _doc(us, days=7, with_stages=("translating",), now=now)
    assert doc["range"]["source"] == "jobs"
    tot = doc["total"]
    assert (tot["all"]["sessions"], tot["all"]["requests"], tot["all"]["errors"]) == (3, 4, 0)
    assert tot["file"]["words"] == 400 and tot["dictation"]["words"] == 60
    assert tot["text"]["sessions"] == 1
    assert doc["today"]["all"]["sessions"] == 3
    stages = {s["stage"]: s for s in doc["stages"]}
    # The chosen stage covers every narrowed run; diarizing co-occurred on
    # the one narrowed batch job (f1).
    assert (stages["translating"]["runs"], stages["translating"]["of_runs"]) == (3, 3)
    assert stages["translating"]["targets"] == [{"code": "de", "runs": 2}, {"code": "fr", "runs": 1}]
    # The reported "kept original" outcome is dictation.translation's; the
    # stage row counts the guard's fallback segments, none recorded here.
    assert stages["translating"]["kept_original"] == 0
    assert doc["dictation"]["translation"]["kept_original"] == 1
    assert (stages["diarizing"]["runs"], stages["diarizing"]["of_runs"]) == (1, 1)
    assert stages["diarizing"]["speakers_avg"] == 2.0
    assert doc["dictation"]["sessions"] == 1
    assert doc["dictation"]["delivery"]["typed"] == 1
    assert doc["apps"] == [{"app_id": "vim", "sessions": 1, "words": 60}]
    assert [c["all"] for c in doc["calendar"]] == [460]
    assert doc["calendar"][0]["file"] == 400
    # The text job's slot has no words and no processing time but still
    # counts as a session in the busy-hours grid.
    assert sorted(h["all"] for h in doc["hours"]) == [0, 60, 400]
    assert sorted(h["sessions"]["all"] for h in doc["hours"]) == [1, 1, 1]
    assert sum(h["sessions"]["text"] for h in doc["hours"]) == 1
    assert doc["streak"]["all"] == {"current": 1, "best": 1}
    assert doc["time_saved_s"] == 60 / 40 * 60 - 20

    both = _doc(us, days=7, with_stages=("translating", "diarizing"), now=now)
    assert both["total"]["all"]["sessions"] == 1
    assert both["total"]["file"]["words"] == 400
    assert {s["stage"]: (s["runs"], s["of_runs"]) for s in both["stages"]} == {
        "translating": (1, 1), "diarizing": (1, 1)}

    # A window that excludes f1: the error job f2 is diarized-only.
    dia = _doc(us, from_day=_D("2025-06-08"), to_day=_D("2025-06-09"), with_stages=("diarizing",),
               now=now)
    assert dia["total"]["all"] == {"sessions": 1, "requests": 1, "errors": 1,
                                   "words": 200, "audio_s": 50.0, "processing_s": 0.0}
    assert dia["today"]["all"]["sessions"] == 1  # f1 is today, whatever the window
    assert dia["range"]["first_day"] == _D("2025-06-09")


def test_time_saved_and_wpm_over_dictation_only(us):
    _rec(us, audio_s=300.0, words=400, kind="dictation", job_id="a" * 32)
    # A file's words are not time the user saved by speaking.
    _rec(us, audio_s=3600.0, words=9000, kind="file")
    doc = _doc(us, days=1)
    assert doc["time_saved_s"] == 400 / 40 * 60 - 300
    assert doc["dictation"]["wpm"] == 80.0
    assert doc["dictation"]["words"] == 400
    _rec(us, user_id="slow", audio_s=600.0, words=10, kind="dictation")
    assert _doc(us, "slow", days=1)["time_saved_s"] == 0.0


def test_outcome_is_idempotent_and_rolls_once(us):
    jid = "b" * 32
    for _ in range(2):
        _rec(us, audio_s=15.0, words=25, kind="dictation", job_id=jid)
    assert _outcome(us, jid, activation="hold", delivery="typed", translation="not_asked",
                    app_id="thunderbird") == "accepted"
    assert _outcome(us, jid, activation="handsfree", delivery="clipboard",
                    translation="translated", app_id="thunderbird") == "duplicate"
    doc = _doc(us, days=1)
    d = doc["dictation"]
    assert d["activation"] == {"hold": 1, "handsfree": 0, "unreported": 0}
    assert d["delivery"] == {"typed": 1, "clipboard": 0, "none": 0, "unreported": 0}
    assert d["translation"]["not_asked"] == 1 and d["translation"]["translated"] == 0
    assert doc["apps"] == [{"app_id": "thunderbird", "sessions": 1, "words": 50}]
    job = _row(us, "SELECT activation, delivery, app_id FROM usage_jobs")
    assert tuple(job) == ("hold", "typed", "thunderbird")


def test_outcome_for_unknown_job_creates_a_stub(us):
    assert _outcome(us, "e" * 32, delivery="none", translation="aborted") == "accepted"
    job = _row(us, "SELECT kind, words, audio_s, utterances, reported_ts FROM usage_jobs")
    assert tuple(job)[:4] == ("dictation", 0, 0.0, 0)
    assert job["reported_ts"] is not None
    doc = _doc(us, days=1)
    assert doc["dictation"]["delivery"]["none"] == 1
    assert doc["apps"] == []
    assert _outcome(us, "e" * 32, delivery="none", translation="aborted") == "duplicate"


def test_outcome_scoped_to_owner(us):
    _rec(us, user_id="alice", audio_s=1.0, words=1, kind="dictation", job_id="a" * 32)
    assert _outcome(us, "a" * 32, user_id="bob", delivery="typed") == "duplicate"
    assert _row(us, "SELECT reported_ts FROM usage_jobs")["reported_ts"] is None
    assert _doc(us, "bob", days=1)["dictation"]["delivery"]["typed"] == 0


def _outcome_rollups(us):
    conn = us._require_conn()
    dictation = conn.execute(
        "SELECT SUM(sessions) AS s, SUM(words) AS w, SUM(audio_s) AS a"
        " FROM usage_dictation_hourly").fetchone()
    app = conn.execute(
        "SELECT SUM(sessions) AS s, SUM(words) AS w FROM usage_app_hourly").fetchone()
    return (dictation["s"], dictation["w"], dictation["a"]), (app["s"], app["w"])


def test_utterance_after_the_outcome_matches_the_usual_order(us):
    """An outcome that races ahead of its utterance leaves a record_outcome
    stub; the late utterance must still count its session in usage_hourly,
    give the job its own key (not the open-mode one) and fill the outcome
    rollups."""
    jid = "c" * 32
    assert _outcome(us, jid, user_id="alice", app_id="thunderbird") == "accepted"
    _rec(us, key_id="k", user_id="alice", audio_s=15.0, words=25, kind="dictation", job_id=jid)
    conn = us._require_conn()
    hourly = conn.execute(
        "SELECT key_id, requests, sessions, words FROM usage_hourly").fetchall()
    assert [tuple(r) for r in hourly] == [("k", 1, 1, 25)]
    job = _row(us, "SELECT key_id, utterances FROM usage_jobs")
    assert tuple(job) == ("k", 1)
    assert _outcome_rollups(us) == ((1, 25, 15.0), (1, 25))
    # A second utterance adds words, never a session (and keeps the key).
    _rec(us, key_id="other", user_id="alice", audio_s=5.0, words=5, kind="dictation", job_id=jid)
    assert _row(us, "SELECT SUM(sessions) FROM usage_hourly")[0] == 1
    assert _row(us, "SELECT key_id FROM usage_jobs")[0] == "k"
    assert _outcome_rollups(us) == ((1, 30, 20.0), (1, 30))


def test_outcome_naming_a_non_dictation_job_is_a_duplicate(us):
    """Only a dictation reports an outcome: naming one's own file job id
    must not fold its words and audio into the dictation and app rollups."""
    _rec(us, audio_s=600.0, words=5000, kind="file", job_id="aaaaaaaa")
    assert _outcome(us, "aaaaaaaa", app_id="thunderbird") == "duplicate"
    assert _row(us, "SELECT COUNT(*) FROM usage_dictation_hourly")[0] == 0
    assert _row(us, "SELECT COUNT(*) FROM usage_app_hourly")[0] == 0
    assert _row(us, "SELECT reported_ts FROM usage_jobs")[0] is None


def test_sweep_marks_unreported_and_prunes(us):
    now_h = us.now_hour()
    # 48 h old dictation with no outcome; a file job of the same age never
    # reports and must not be marked.
    _rec(us, audio_s=10.0, words=30, kind="dictation", job_id="1" * 32, hour=now_h - 48)
    _rec(us, audio_s=10.0, words=30, kind="file", job_id="2" * 32, hour=now_h - 48,
         stages=[{"name": "vad", "secs": 1.0}])
    # Fresh dictation: inside the grace period.
    _rec(us, audio_s=10.0, words=30, kind="dictation", job_id="3" * 32)
    _outcome(us, "3" * 32, delivery="typed", app_id="vim")
    # An app row from long ago (past the 90-day clock).
    us._require_conn().execute(
        "INSERT INTO usage_app_hourly VALUES (?, 'u', 'old-app', 1, 1)",
        (now_h - 100 * 24,))

    counts = us.sweep(unreported_after_h=24, jobs_retention_days=0,
                      app_retention_days=90, hourly_retention_days=0)
    assert counts == {"marked": 1, "jobs": 0, "apps": 1, "hourly": 0}
    doc = _doc(us, days=7)
    assert doc["dictation"]["delivery"] == {"typed": 1, "clipboard": 0,
                                            "none": 0, "unreported": 1}
    assert doc["dictation"]["translation"]["unreported"] == 1
    assert doc["apps"] == [{"app_id": "vim", "sessions": 1, "words": 30}]
    assert us.sweep(unreported_after_h=24, jobs_retention_days=0,
                    app_retention_days=90, hourly_retention_days=0)["marked"] == 0

    # Job retention drops the two old jobs and their stage detail; the hourly
    # rollups (and so the totals) survive a job prune.
    counts = us.sweep(unreported_after_h=24, jobs_retention_days=1,
                      app_retention_days=90, hourly_retention_days=0)
    assert counts["jobs"] == 2
    assert _row(us, "SELECT COUNT(*) FROM usage_jobs")[0] == 1
    assert _row(us, "SELECT COUNT(*) FROM usage_job_stages")[0] == 0
    assert us.totals_for_user("u")["words"] == 90

    # Hourly retention takes the old rollup rows in every hour-keyed table.
    counts = us.sweep(unreported_after_h=24, jobs_retention_days=0,
                      app_retention_days=0, hourly_retention_days=1)
    assert counts["hourly"] == 4
    assert _row(us, "SELECT COUNT(*) FROM usage_stage_hourly")[0] == 0
    assert us.totals_for_user("u")["words"] == 30
    assert _row(us, "SELECT COUNT(*) FROM usage_dictation_hourly")[0] == 1


def test_empty_document_shape_matches_populated(us):
    empty = us.empty_document(from_day=_D("2025-01-01"), to_day=_D("2025-01-30"), tz="UTC")
    _rec(us, audio_s=1.0, words=1, kind="dictation", job_id="a" * 32)
    _outcome(us, "a" * 32)
    full = _doc(us, days=30)
    assert set(empty) == set(full)
    assert set(empty["dictation"]) == set(full["dictation"])
    for k in ("today", "total"):
        assert set(empty[k]) == set(full[k])
        assert set(empty[k]["all"]) == set(full[k]["all"])
    assert set(empty["range"]) == set(full["range"])
    assert set(empty["streak"]) == set(full["streak"]) == {"all", "dictation", "file", "url", "text"}
    assert set(full["calendar"][0]) == {"day", "all", "dictation", "file", "url", "text"}
    assert set(full["hours"][0]) == {"dow", "hour", "all", "dictation", "file",
                                     "url", "text", "processing_s", "audio_s", "sessions",
                                     "requests", "errors"}


def test_fold_runs_on_every_init_so_a_crash_mid_migration_heals(
        us, tmp_path):
    """The rename and the copy are separate steps; if the process dies in
    between, the next init finds the parking table and finishes the job."""
    conn = us._require_conn()
    # The parked table as _park_legacy_hourly leaves it: renamed, indexes dropped.
    table_only = _LEGACY_SCHEMA.split("CREATE INDEX")[0]
    conn.executescript(table_only.replace("usage_hourly", "usage_hourly_legacy"))
    conn.execute("INSERT INTO usage_hourly_legacy VALUES (7, 'k', 'u', 2, 0, 5, 1.0)")
    conn.commit()
    conn.close()
    us.init_db(str(tmp_path / "usage.sqlite3"))
    assert us.totals_for_user("u")["requests"] == 2
    assert _row(us, "SELECT 1 FROM sqlite_master WHERE name='usage_hourly_legacy'") is None


def test_hours_carry_proc_s_and_sessions(us):
    """Each hour slot also carries nested processing_s / sessions splits (the
    backend's busy-hours grid measures GPU seconds); the flat words keys
    the desktop app reads are unchanged. Both fill paths agree."""
    h = _hour(2025, 10, 27, 9, 30, _UTC)          # Monday 09:xx UTC
    _rec(us, audio_s=10.0, words=10, kind="file", hour=h, processing_s=2.5, job_id="f1",
         stages=[{"name": "diarizing", "secs": 1.0}])
    _rec(us, audio_s=5.0, words=4, kind="dictation", hour=h, processing_s=0.5, job_id="d1")
    now = _ts(2025, 10, 27, 12, 0, _UTC)
    doc = _doc(us, days=7, now=now)
    slot = doc["hours"][0]
    assert (slot["dow"], slot["hour"], slot["all"]) == (0, 9, 14)
    assert slot["processing_s"] == {"all": 3.0, "dictation": 0.5, "file": 2.5,
                              "url": 0.0, "text": 0.0}
    assert slot["sessions"] == {"all": 2, "dictation": 1, "file": 1,
                                "url": 0, "text": 0}
    narrowed = _doc(us, days=7, now=now, with_stages=("diarizing",))
    assert narrowed["range"]["source"] == "jobs"
    assert narrowed["hours"][0]["processing_s"]["all"] == 2.5
    assert narrowed["hours"][0]["sessions"] == {"all": 1, "dictation": 0,
                                                "file": 1, "url": 0, "text": 0}


def test_document_user_id_none_aggregates_all_users(us):
    """user_id=None is the whole server: totals, series, stages and hours
    sum over every user; key_id narrows the key-bearing tables only and
    says so in range.key_scoped."""
    h = _hour(2025, 6, 10, 9, 0, _UTC)
    _rec(us, key_id="ka", user_id="alice", audio_s=10.0, words=10, kind="file", hour=h,
         processing_s=1.0, job_id="a1", stages=[{"name": "diarizing", "secs": 1.0, "speakers": 2}])
    _rec(us, key_id="ka2", user_id="alice", audio_s=5.0, words=5, kind="file", hour=h,
         processing_s=1.0, job_id="a2")
    _rec(us, key_id="kb", user_id="bob", audio_s=20.0, words=20, kind="dictation", hour=h,
         processing_s=2.0, job_id="b1")
    now = _ts(2025, 6, 10, 12, 0, _UTC)
    everyone = _doc(us, None, days=7, now=now)
    alice = _doc(us, "alice", days=7, now=now)
    bob = _doc(us, "bob", days=7, now=now)
    assert everyone["total"]["all"]["audio_s"] == 35.0
    assert everyone["total"]["all"]["sessions"] == 3
    assert (everyone["total"]["file"]["words"], everyone["total"]["dictation"]["words"]) == (15, 20)
    assert everyone["series"][0]["all"]["processing_s"] == 4.0
    assert everyone["hours"][0]["sessions"]["all"] == 3
    assert everyone["stages"][0]["stage"] == "diarizing"
    assert everyone["stages"][0]["runs"] == 1 and everyone["stages"][0]["of_runs"] == 2
    assert alice["total"]["all"]["audio_s"] + bob["total"]["all"]["audio_s"] == 35.0
    assert "key_scoped" not in everyone["range"]
    one_key = _doc(us, "alice", days=7, now=now, key_id="ka")
    assert one_key["total"]["all"]["audio_s"] == 10.0
    assert one_key["range"]["key_scoped"] is False
    # The stage tables carry no key: `runs` stays at user scope while the
    # denominator follows the key-scoped window — hence key_scoped=False.
    assert one_key["stages"][0]["runs"] == 1 and one_key["stages"][0]["of_runs"] == 1
    # with= over everyone: the jobs path scopes the same way.
    narrowed = _doc(us, None, days=7, now=now, with_stages=("diarizing",))
    assert narrowed["total"]["all"]["sessions"] == 1
    # The jobs path reads key and kind off the job rows, so a key or kinds
    # filter there is fully applied and no flag says otherwise.
    keyed_jobs = _doc(us, "alice", days=7, now=now, key_id="ka", kinds=["file"],
                      with_stages=("diarizing",))
    assert "key_scoped" not in keyed_jobs["range"]
    assert "kind_scoped" not in keyed_jobs["range"]
    assert narrowed["range"]["first_day"] == everyone["range"]["first_day"]


def test_parse_window_params_matches_v1_route(us):
    w = us.parse_window_params(days=9999, tz="Europe/Zurich")
    assert w.days == 3650 and w.tz_name == "Europe/Zurich" and w.tz is not None
    w = us.parse_window_params(days=-3, tz="Mars/Olympus")
    assert w.days == 1 and w.tz is None and w.tz_name == "local"
    w = us.parse_window_params(with_="translating, diarizing,translating")
    assert w.with_stages == ("translating", "diarizing") and w.source == "jobs"
    assert us.parse_window_params().source == "rollups"
    with pytest.raises(ValueError, match="decoding"):
        us.parse_window_params(with_="decoding")
    with pytest.raises(ValueError, match="from"):
        us.parse_window_params(from_day=20007, to_day=20006)
    # `to` defaults to today: a future `from` alone is the same error here,
    # not resolve_window's ValueError inside the store call (a 500).
    future = us._epoch_day(datetime.date.today()) + 5
    with pytest.raises(ValueError, match="'from' is after 'to'"):
        us.parse_window_params(from_day=future)
    assert us.parse_window_params(from_day=future, all_time=True).all_time
    with pytest.raises(ValueError, match="'to' out of range"):
        us.parse_window_params(to_day=10**9)
    with pytest.raises(ValueError, match="'from' out of range"):
        us.parse_window_params(from_day=-1)
    assert us.parse_window_params(from_day=0, to_day=us._MAX_EPOCH_DAY).to_day == us._MAX_EPOCH_DAY
    # The month axis steps one month past to_day: December 9999 has none, so
    # the last accepted day is in November and the axis there still builds.
    with pytest.raises(ValueError, match="'to' out of range"):
        us.parse_window_params(to_day=_D("9999-12-15"))
    us.overview(user_id=None, tz=_UTC, tz_name="UTC", from_day=us._MAX_EPOCH_DAY - 100,
                to_day=us._MAX_EPOCH_DAY, bucket="month")


def test_wait_s_accumulates_and_error_class_is_kept(us):
    h = _hour(2025, 6, 10, 9, 0, _UTC)
    _rec(us, audio_s=1.0, words=1, status="error", kind="dictation", job_id="d1", hour=h,
         wait_s=1.5, error_class="timeout", error_stage="transcribing")
    _rec(us, audio_s=1.0, words=1, kind="dictation", job_id="d1", hour=h, wait_s=0.5)
    row = _row(us, "SELECT wait_s, error_class, error_stage, utterances FROM usage_jobs"
                   " WHERE job_id = 'd1'")
    assert (row["wait_s"], row["error_class"], row["error_stage"],
            row["utterances"]) == (2.0, "timeout", "transcribing", 2)
    # A negative wait is a clock artefact, stored as 0; a job without an
    # error keeps NULLs.
    _rec(us, audio_s=1.0, words=1, kind="file", job_id="f1", hour=h, wait_s=-2)
    row = _row(us, "SELECT wait_s, error_class FROM usage_jobs WHERE job_id = 'f1'")
    assert (row["wait_s"], row["error_class"]) == (0.0, None)


def test_a_failed_utterance_keeps_the_dictation_session_failed(us):
    """One failed utterance mid-session must not be flipped back to ok by the
    next good one: failures() and by_model count status, and the row kept
    the error class."""
    h = _hour(2025, 6, 10, 9, 0, _UTC)
    for status, ecls in (("ok", None), ("error", "cuda_oom"), ("ok", None)):
        _rec(us, audio_s=1.0, words=1, status=status, kind="dictation", job_id="s1", hour=h,
             error_class=ecls, error_stage="transcribing" if ecls else None)
    row = _row(us, "SELECT status, error_class, utterances FROM usage_jobs WHERE job_id = 's1'")
    assert (row["status"], row["error_class"], row["utterances"]) == (
        "error", "cuda_oom", 3)
    out = us.failures(start_ts=h * 3600 - 1, end_ts=h * 3600 + 3600)
    assert (out["jobs"], out["failed"]) == (1, 1)
    # A batch id re-posted after a failure: the run that finished last wins.
    _rec(us, audio_s=1.0, words=1, status="error", kind="file", job_id="f1", hour=h,
         error_class="timeout")
    _rec(us, audio_s=1.0, words=1, kind="file", job_id="f1", hour=h)
    assert _row(us, "SELECT status FROM usage_jobs WHERE job_id = 'f1'")[0] == "ok"


def test_stage_error_lands_on_the_stage_row_and_bumps_hourly_errors(us):
    """A soft-failed stage (the job went on without it) carries `error` on
    the per-job stage row and counts in usage_stage_hourly.errors while
    the job's own status stays ok."""
    h = _hour(2025, 6, 10, 9, 0, _UTC)
    _rec(us, audio_s=10.0, words=5, kind="file", job_id="f1", hour=h,
         stages=[{"name": "diarizing", "secs": 2.0, "error": "timeout"},
                 {"name": "translating", "secs": 1.0, "targets": ["de"]}])
    _rec(us, audio_s=10.0, words=5, kind="file", job_id="f2", hour=h,
         stages=[{"name": "diarizing", "secs": 2.0, "speakers": 2}])
    conn = us._require_conn()
    rows = {r["stage"]: r for r in conn.execute(
        "SELECT stage, error FROM usage_job_stages WHERE job_id = 'f1'")}
    assert rows["diarizing"]["error"] == "timeout"
    assert rows["translating"]["error"] is None
    hourly = {r["stage"]: r for r in conn.execute(
        "SELECT stage, runs, errors FROM usage_stage_hourly WHERE user_id = 'u'")}
    assert (hourly["diarizing"]["runs"], hourly["diarizing"]["errors"]) == (2, 1)
    assert hourly["translating"]["errors"] == 0
    assert _row(us, "SELECT status FROM usage_jobs WHERE job_id = 'f1'")["status"] == "ok"


_PRE_V2_JOBS_SCHEMA = """
CREATE TABLE usage_hourly (
  hour INTEGER NOT NULL, key_id TEXT NOT NULL, user_id TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'unknown', requests INTEGER NOT NULL DEFAULT 0,
  errors INTEGER NOT NULL DEFAULT 0, words INTEGER NOT NULL DEFAULT 0,
  audio_s REAL NOT NULL DEFAULT 0, proc_s REAL NOT NULL DEFAULT 0,
  sessions INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, key_id, kind)
);
CREATE TABLE usage_jobs (
  job_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, key_id TEXT NOT NULL,
  kind TEXT NOT NULL, created_ts REAL NOT NULL, status TEXT NOT NULL,
  audio_s REAL NOT NULL DEFAULT 0, words INTEGER NOT NULL DEFAULT 0,
  proc_s REAL NOT NULL DEFAULT 0, utterances INTEGER NOT NULL DEFAULT 0,
  model TEXT, language TEXT, activation TEXT, delivery TEXT, app_id TEXT,
  translation TEXT, reported_ts REAL
);
CREATE TABLE usage_job_stages (
  job_id TEXT NOT NULL, stage TEXT NOT NULL, secs REAL NOT NULL DEFAULT 0,
  model TEXT, targets TEXT, speakers INTEGER, retained REAL,
  PRIMARY KEY (job_id, stage)
);
CREATE TABLE usage_stage_hourly (
  hour INTEGER NOT NULL, user_id TEXT NOT NULL, stage TEXT NOT NULL,
  runs INTEGER NOT NULL DEFAULT 0, audio_s REAL NOT NULL DEFAULT 0,
  secs REAL NOT NULL DEFAULT 0, speakers INTEGER NOT NULL DEFAULT 0,
  retained_sum REAL NOT NULL DEFAULT 0, kept_original INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, user_id, stage)
);
"""


def test_init_adds_v2_ledger_columns_to_a_pre_v2_db(tmp_path):
    """A DB whose ledger tables predate wait_s / error_class / stage errors
    (and still spell processing_s proc_s) gains the columns on init_db and
    keeps the renamed column's data; old rows read with the defaults; a
    second init changes nothing; a fresh DB has the columns from the start."""
    from faster_whisper_backend.stats import usage_store
    path = str(tmp_path / "prev2.sqlite3")
    raw = sqlite3.connect(path)
    raw.executescript(_PRE_V2_JOBS_SCHEMA)
    raw.execute("INSERT INTO usage_jobs (job_id, user_id, key_id, kind, created_ts,"
                " status, audio_s, words, proc_s, utterances)"
                " VALUES ('old', 'u', 'k', 'file', 1000.0, 'ok', 5, 2, 1, 1)")
    raw.execute("INSERT INTO usage_job_stages (job_id, stage, secs)"
                " VALUES ('old', 'diarizing', 1.0)")
    raw.execute("INSERT INTO usage_hourly (hour, key_id, user_id, kind, requests,"
                " words, audio_s, proc_s, sessions)"
                " VALUES (0, 'k', 'u', 'file', 1, 2, 5, 1.5, 1)")
    raw.execute("INSERT INTO usage_stage_hourly (hour, user_id, stage, runs)"
                " VALUES (0, 'u', 'diarizing', 1)")
    raw.commit(); raw.close()
    usage_store.init_db(path)
    try:
        conn = usage_store._require_conn()
        cols = lambda t: {r["name"] for r in conn.execute(f"PRAGMA table_info({t})")}
        assert {"wait_s", "error_class", "error_stage"} <= cols("usage_jobs")
        assert {"error", "kept_original", "audio_s"} <= cols("usage_job_stages")
        assert "errors" in cols("usage_stage_hourly")
        for t in ("usage_jobs", "usage_hourly"):
            assert "processing_s" in cols(t) and "proc_s" not in cols(t), t
        assert _row(usage_store, "SELECT processing_s FROM usage_jobs"
                                 " WHERE job_id = 'old'")[0] == 1.0
        assert _row(usage_store, "SELECT processing_s FROM usage_hourly")[0] == 1.5
        row = _row(usage_store, "SELECT wait_s, error_class FROM usage_jobs")
        assert (row["wait_s"], row["error_class"]) == (0.0, None)
        assert _row(usage_store, "SELECT errors FROM usage_stage_hourly")["errors"] == 0
        idx = {r["name"] for r in conn.execute("PRAGMA index_list(usage_jobs)")}
        assert "idx_usage_jobs_user_created" in idx
        conn.close()
        usage_store._conn = None
        usage_store.init_db(path)                   # a second init: no change
        conn = usage_store._require_conn()
        counts = [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("usage_hourly", "usage_jobs", "usage_job_stages",
                            "usage_stage_hourly")]
        assert counts == [1, 1, 1, 1]
        assert _row(usage_store, "SELECT processing_s FROM usage_jobs")[0] == 1.0
        conn.close()
    finally:
        usage_store._conn = None
    usage_store.init_db(str(tmp_path / "fresh.sqlite3"))
    try:
        conn = usage_store._require_conn()
        assert {"wait_s", "error_class", "error_stage"} <= {
            r["name"] for r in conn.execute("PRAGMA table_info(usage_jobs)")}
        conn.close()
    finally:
        usage_store._conn = None


def test_document_lists_every_stage_only_when_asked(us):
    """The rollup records the decode and the download too; document() keeps
    the desktop app's optional-stage list unless all_stages=True (the /stats
    console shows the whole pipeline with transcription as the reference row)."""
    _rec(us, audio_s=60.0, words=10, kind="url", job_id="d" * 32, processing_s=6.0, stages=[
             {"name": "downloading", "secs": 2.0},
             {"name": "transcribing", "secs": 4.0, "model": "large-v3"},
             {"name": "translate", "secs": 1.0, "targets": ["de"]}])
    plain = _doc(us, days=1)
    assert [s["stage"] for s in plain["stages"]] == ["translating"]
    full = _doc(us, days=1, all_stages=True)
    assert [s["stage"] for s in full["stages"]] == ["downloading", "transcribing", "translating"]
    tr = {s["stage"]: s for s in full["stages"]}
    assert tr["transcribing"]["runs"] == 1 and tr["transcribing"]["of_runs"] == 1
    assert tr["downloading"]["of_runs"] == 1          # a URL job: eligible
    over = us.overview(user_id="u", days=1, tz=_UTC, tz_name="UTC")
    assert "transcribing" in {s["stage"] for s in over["stages"]}


def test_dom_hours_is_the_same_slots_by_day_of_month(us):
    """`dom_hours` folds the hourly slots by day of month × hour (the
    console's busy-days rhythm) with the same per-kind splits as `hours`;
    the two grids sum to the same totals."""
    _rec(us, audio_s=10.0, words=10, kind="file", hour=_hour(2025, 10, 25, 9, 30, _UTC),
         processing_s=2.0)
    _rec(us, audio_s=5.0, words=5, kind="dictation", hour=_hour(2025, 10, 27, 9, 30, _UTC),
         processing_s=1.0)
    now = _ts(2025, 10, 27, 12, 0, _UTC)
    doc = _doc(us, days=7, now=now)
    dom = {(h["dom"], h["hour"]): h for h in doc["dom_hours"]}
    assert set(dom) == {(25, 9), (27, 9)}
    assert dom[(25, 9)]["file"] == 10 and dom[(25, 9)]["processing_s"]["all"] == 2.0
    assert dom[(27, 9)]["dictation"] == 5 and dom[(27, 9)]["sessions"]["all"] == 1
    assert set(dom[(25, 9)]) == set(doc["hours"][0]) - {"dow"} | {"dom"}
    assert sum(h["all"] for h in doc["hours"]) == sum(h["all"] for h in doc["dom_hours"]) == 15


def test_outcome_stubs_are_capped_per_user_per_hour(us, monkeypatch):
    # Reports for jobs the server never saw make stub rows; one identity must
    # not fabricate them at the post limit (6 000 a minute) forever.
    monkeypatch.setattr(us, "_MAX_STUBS_PER_HOUR", 3)
    assert [_outcome(us, f"{i:08x}", user_id="alice") for i in range(4)] == [
        "accepted", "accepted", "accepted", "duplicate"]
    assert _row(us, "SELECT COUNT(*) FROM usage_jobs WHERE user_id = 'alice'")[0] == 3
    assert _outcome(us, "b0b0b0b0", user_id="bob") == "accepted"
    # A server-minted job still takes its outcome past the cap.
    _rec(us, user_id="alice", audio_s=1.0, words=3, kind="dictation", job_id="feedf00d")
    assert _outcome(us, "feedf00d", user_id="alice") == "accepted"
