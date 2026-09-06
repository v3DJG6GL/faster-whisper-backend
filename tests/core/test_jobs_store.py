"""core/jobs_store — the durable job resource behind GET/DELETE /v1/jobs*."""

import sqlite3
import time

import pytest

from faster_whisper_backend.core import jobs_store as js

_TTL = 3600.0


@pytest.fixture
def db(tmp_path):
    js._reset_for_tests()
    js.init_db(str(tmp_path / "jobs.sqlite3"))
    yield js
    js._reset_for_tests()


def _start(db, job_id="a" * 32, **kw):
    args = dict(job_id=job_id, request_id="req", kind="transcribe",
                user_id="u1", key_id="k1", model="large-v3",
                source_kind="file", source_name="meeting.m4a",
                task="transcribe", response_format="verbose_json",
                ttl_s=_TTL, max_rows=100, max_bytes=0, prune_every=0)
    args.update(kw)
    db.start(**args)
    return args["job_id"]


def test_schema_and_migration_are_idempotent(tmp_path):
    path = str(tmp_path / "old.sqlite3")
    # A DB from a build that predated the side-blob columns.
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, request_id TEXT NOT NULL,"
        " kind TEXT NOT NULL, user_id TEXT, key_id TEXT, state TEXT NOT NULL"
        " DEFAULT 'running', created_ts REAL NOT NULL, finished_ts REAL,"
        " expires_ts REAL NOT NULL, model TEXT, source_kind TEXT NOT NULL"
        " DEFAULT 'file', source_name TEXT, error TEXT, result_json TEXT,"
        " result_bytes INTEGER NOT NULL DEFAULT 0);")
    conn.commit(); conn.close()
    js._reset_for_tests()
    js.init_db(path)
    js.init_db(path)  # second call must not raise
    cols = {r["name"] for r in js._conn.execute("PRAGMA table_info(jobs)")}
    assert {"task", "response_format", "stages_json", "plan_json"} <= cols
    js._reset_for_tests()


def test_start_and_finish_round_trip_dict_result(db):
    jid = _start(db)
    row = db.get(jid)
    assert row["state"] == "running" and row["result_available"] is False
    assert row["expires_ts"] >= row["created_ts"] + _TTL - 1
    assert db.finish(job_id=jid, state="done", result={"text": "hallo"},
                     stages=[{"name": "transcribing", "secs": 1.5}],
                     plan=[{"stage": "transcribing"}], ttl_s=_TTL)
    row = db.get(jid)
    assert row["state"] == "done" and row["result_available"] is True
    assert row["error"] is None and row["finished_ts"] is not None
    assert row["result_bytes"] == len(b'{"text": "hallo"}')
    assert row["stages"] == [{"name": "transcribing", "secs": 1.5}]
    assert row["plan"] == [{"stage": "transcribing"}]
    assert "result_json" not in row  # get() never loads the blob
    assert db.get_result(jid) == {"text": "hallo"}


def test_str_result_round_trips_as_the_string(db):
    jid = _start(db, response_format="text")
    db.finish(job_id=jid, state="done", result="hallo welt", ttl_s=_TTL)
    assert db.get_result(jid) == "hallo welt"


def test_failed_and_cancelled_store_no_result(db):
    a = _start(db, job_id="a" * 32)
    b = _start(db, job_id="b" * 32)
    db.finish(job_id=a, state="failed", error="upload too large\n\x00x",
              result={"text": "ignored"}, ttl_s=_TTL)
    db.finish(job_id=b, state="cancelled", ttl_s=_TTL)
    ra, rb = db.get(a), db.get(b)
    assert ra["state"] == "failed" and ra["error"] == "upload too large??x"
    assert ra["result_available"] is False and db.get_result(a) is None
    assert rb["state"] == "cancelled" and rb["error"] is None


def test_finish_rejects_running_and_unknown_states(db):
    jid = _start(db)
    with pytest.raises(ValueError):
        db.finish(job_id=jid, state="running", ttl_s=_TTL)
    with pytest.raises(ValueError):
        db.finish(job_id=jid, state="weird", ttl_s=_TTL)
    assert db.finish(job_id="0" * 32, state="done", ttl_s=_TTL) is False


def test_start_replaces_a_finished_row_under_the_same_id(db):
    jid = _start(db)
    db.finish(job_id=jid, state="done", result={"text": "1"}, ttl_s=_TTL)
    _start(db, job_id=jid, request_id="req2")
    row = db.get(jid)
    assert row["state"] == "running" and row["request_id"] == "req2"
    assert db.get_result(jid) is None


def test_ownership_user_key_and_open_mode(db):
    _start(db, job_id="1" * 32, user_id="alice", key_id="ka")
    _start(db, job_id="2" * 32, user_id=None, key_id="kb")
    _start(db, job_id="3" * 32, user_id=None, key_id=None)
    ids = lambda rows: sorted(r["job_id"][0] for r in rows)
    assert ids(db.list_jobs(user_id="alice", key_id="ka")) == ["1"]
    # A key without a user sees its own key-bound rows only.
    assert ids(db.list_jobs(user_id=None, key_id="kb")) == ["2"]
    # Open mode (no identity at all) sees only owner-less rows.
    assert ids(db.list_jobs(user_id=None, key_id=None)) == ["3"]
    assert ids(db.list_jobs(user_id="x", key_id="y", all_users=True)) == ["1", "2", "3"]
    r1, r2, r3 = db.get("1" * 32), db.get("2" * 32), db.get("3" * 32)
    assert db.is_owner(r1, user_id="alice", key_id="other") is True
    assert db.is_owner(r1, user_id="bob", key_id="ka") is False
    assert db.is_owner(r2, user_id=None, key_id="kb") is True
    # A caller presenting the key owns key-bound rows whatever its user.
    assert db.is_owner(r2, user_id="alice", key_id="kb") is True
    assert db.is_owner(r2, user_id="alice", key_id="ka") is False
    assert db.is_owner(r3, user_id=None, key_id=None) is True
    assert db.is_owner(r3, user_id="alice", key_id=None) is False


def test_list_is_newest_first_filtered_and_limited(db):
    for i, st in enumerate(("done", "failed", None)):
        jid = _start(db, job_id=str(i) * 32)
        if st:
            db.finish(job_id=jid, state=st, ttl_s=_TTL)
        time.sleep(0.01)
    rows = db.list_jobs(user_id="u1", key_id="k1")
    assert [r["job_id"][0] for r in rows] == ["2", "1", "0"]
    assert [r["job_id"][0] for r in db.list_jobs(user_id="u1", key_id="k1",
                                                 state="done")] == ["0"]
    assert len(db.list_jobs(user_id="u1", key_id="k1", limit=2)) == 2


def test_prune_ttl_rows_and_bytes(db):
    # Expired.
    old = _start(db, job_id="e" * 32, ttl_s=0.0)
    time.sleep(0.01)
    assert db.prune(ttl_s=_TTL, max_rows=0, max_bytes=0) == 1
    assert db.get(old) is None
    # Row cap: newest kept, a running row never evicted.
    running = _start(db, job_id="r" * 32)
    time.sleep(0.01)
    for i in range(4):
        jid = _start(db, job_id=str(i) * 32)
        db.finish(job_id=jid, state="done", result={"t": "x" * 100}, ttl_s=_TTL)
        time.sleep(0.01)
    assert db.count() == 5
    # Newest two (done 3, done 2) stay; done 0/1 go; the running row is
    # older than all of them and still survives.
    assert db.prune(ttl_s=_TTL, max_rows=2, max_bytes=0) == 2
    assert db.get(running) is not None
    assert db.get("3" * 32) is not None and db.get("2" * 32) is not None
    assert db.get("0" * 32) is None and db.get("1" * 32) is None
    # Byte cap: oldest finished results dropped until the total fits.
    total = db.total_result_bytes()
    assert total > 0
    assert db.prune(ttl_s=_TTL, max_rows=0, max_bytes=total - 1) == 1
    assert db.get("2" * 32) is None and db.get("3" * 32) is not None


def test_lazy_prune_runs_every_nth_insert(db):
    _start(db, job_id="e" * 32, ttl_s=0.0, prune_every=3)  # counter 1
    time.sleep(0.01)
    _start(db, job_id="1" * 32, prune_every=3)  # counter 2
    assert db.get("e" * 32) is not None
    _start(db, job_id="2" * 32, prune_every=3)  # counter 3 → prune
    assert db.get("e" * 32) is None


def test_mark_running_as_failed_flips_only_running_rows(db):
    a = _start(db, job_id="a" * 32)
    b = _start(db, job_id="b" * 32)
    db.finish(job_id=b, state="done", result={"text": "1"}, ttl_s=_TTL)
    assert db.mark_running_as_failed("server restarted") == 1
    assert db.get(a)["state"] == "failed"
    assert db.get(a)["error"] == "server restarted"
    assert db.get(b)["state"] == "done"


def test_delete_and_clear(db):
    a = _start(db, job_id="a" * 32)
    assert db.delete(a) is True and db.delete(a) is False
    _start(db, job_id="b" * 32)
    assert db.clear_all() == 1 and db.count() == 0


def test_sweep_retention_reads_live_config(db, monkeypatch):
    from faster_whisper_backend import config as cfg
    _start(db, job_id="a" * 32)
    time.sleep(0.01)
    monkeypatch.setattr(cfg, "JOBS_TTL_S", 1, raising=False)
    monkeypatch.setattr(cfg, "JOBS_MAX_ROWS", 0, raising=False)
    monkeypatch.setattr(cfg, "JOBS_MAX_BYTES", 0, raising=False)
    # Row expiry is stamped at start (ttl 3600) — the sweep honours the
    # stored expires_ts, so nothing goes yet.
    assert db.sweep_retention() == 0
    db.finish(job_id="a" * 32, state="done", result={"t": 1}, ttl_s=0.0)
    time.sleep(0.01)
    assert db.sweep_retention() == 1
