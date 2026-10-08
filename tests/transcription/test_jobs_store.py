"""transcription/jobs_store — the durable job resource behind GET/DELETE /v1/jobs*."""

import sqlite3
import time

import pytest

from faster_whisper_backend.transcription import jobs_store as js

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
    # The running floor, not the TTL: _TTL is below it, so this pins it.
    assert row["expires_ts"] == pytest.approx(
        row["created_ts"] + js._RUNNING_FLOOR_S, abs=1)
    assert db.finish(job_id=jid, state="done", result={"text": "hallo"},
                     stages=[{"name": "transcribing", "secs": 1.5}],
                     plan=[{"stage": "transcribing"}], ttl_s=_TTL)
    row = db.get(jid, side_blobs=True)
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


def test_finish_with_an_unstorable_result_stamps_failed_not_running(db):
    # A lone surrogate (echoed from a client body) cannot be encoded as
    # UTF-8: the run still ends, as `failed`, instead of finish() raising
    # and leaving the row `running` until the 24 h floor.
    jid = _start(db)
    assert db.finish(job_id=jid, state="done",
                     result={"segments": [{"id": "\ud800"}]}, ttl_s=_TTL)
    row = db.get(jid)
    assert row["state"] == "failed"
    assert row["error"] == "result could not be stored"
    assert row["result_available"] is False and db.get_result(jid) is None


def test_start_replaces_a_finished_row_under_the_same_id(db):
    jid = _start(db)
    db.finish(job_id=jid, state="done", result={"text": "1"}, ttl_s=_TTL)
    # The owner's own re-use: start() reports the row written.
    args = dict(job_id=jid, request_id="req2", kind="transcribe",
                user_id="u1", key_id="k1", model="large-v3",
                source_kind="file", source_name="meeting.m4a",
                ttl_s=_TTL, max_rows=100, max_bytes=0, prune_every=0)
    assert db.start(**args) is True
    row = db.get(jid)
    assert row["state"] == "running" and row["request_id"] == "req2"
    assert db.get_result(jid) is None


def test_start_refuses_a_foreign_row(db):
    # The job id is client-chosen: bob posting with alice's id must not
    # destroy her stored result or take the id over.
    jid = _start(db, user_id="alice", key_id="ka")
    db.finish(job_id=jid, state="done", result={"text": "alice"}, ttl_s=_TTL)
    args = dict(job_id=jid, request_id="req-bob", kind="transcribe",
                user_id="bob", key_id="kb", model="large-v3",
                source_kind="file", source_name="x.m4a",
                ttl_s=_TTL, max_rows=100, max_bytes=0, prune_every=0)
    assert db.start(**args) is False
    row = db.get(jid)
    assert row["user_id"] == "alice" and row["state"] == "done"
    assert row["request_id"] == "req"
    assert db.get_result(jid) == {"text": "alice"}
    # An open-mode caller does not own it either.
    assert db.start(**dict(args, user_id=None, key_id=None)) is False
    # An EXPIRED foreign row no longer blocks the id.
    db.finish(job_id=jid, state="done", result={"text": "alice"}, ttl_s=0.0)
    time.sleep(0.01)
    assert db.start(**args) is True
    assert db.get(jid)["user_id"] == "bob" and db.get_result(jid) is None


def test_init_db_twice_closes_the_previous_connection(tmp_path):
    js._reset_for_tests()
    path = str(tmp_path / "jobs.sqlite3")
    js.init_db(path)
    first = js._conn
    js.init_db(path)
    assert js._conn is not first
    with pytest.raises(sqlite3.ProgrammingError):
        first.execute("SELECT 1")  # closed, not leaked
    js._reset_for_tests()


def test_list_jobs_skips_the_side_blobs(db):
    jid = _start(db)
    db.finish(job_id=jid, state="done", result={"text": "x"},
              stages=[{"name": "transcribing"}], plan=[{"stage": "t"}],
              ttl_s=_TTL)
    assert db.get(jid, side_blobs=True)["stages"] == [{"name": "transcribing"}]
    assert db.get(jid)["stages"] is None, "a poll never decodes them"
    (row,) = db.list_jobs(user_id="u1", key_id="k1")
    assert row["stages"] is None and row["plan"] is None
    assert row["result_available"] is True


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
    old = _start(db, job_id="e" * 32)
    db.finish(job_id=old, state="failed", ttl_s=0.0)
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


def test_byte_cap_never_eats_rows_that_hold_no_bytes(db):
    # Two old failures in front of the byte-carrying rows: deleting them
    # frees nothing, so the byte cap must walk past them.
    for i, st in enumerate(("failed", "cancelled")):
        db.finish(job_id=_start(db, job_id=str(i) * 32), state=st, ttl_s=_TTL)
        time.sleep(0.01)
    for i in (2, 3):
        db.finish(job_id=_start(db, job_id=str(i) * 32), state="done",
                  result={"t": "x" * 100}, ttl_s=_TTL)
        time.sleep(0.01)
    total = db.total_result_bytes()
    assert db.prune(ttl_s=_TTL, max_rows=0, max_bytes=total - 1) == 1
    assert db.get("0" * 32) is not None and db.get("1" * 32) is not None
    assert db.get("2" * 32) is None and db.get("3" * 32) is not None


def test_lazy_prune_runs_every_nth_insert(db):
    _start(db, job_id="e" * 32, prune_every=3)  # counter 1
    db.finish(job_id="e" * 32, state="failed", ttl_s=0.0)
    time.sleep(0.01)
    _start(db, job_id="1" * 32, prune_every=3)  # counter 2
    assert db.get("e" * 32) is not None
    _start(db, job_id="2" * 32, prune_every=3)  # counter 3 → prune
    assert db.get("e" * 32) is None


def test_a_run_longer_than_a_short_ttl_stays_listed_and_finishes(
        db, monkeypatch):
    """JOBS_TTL_S counts from the finish: a running row outliving a short
    TTL is still listed, fetchable and spared by the prune, and its result
    is stored when it finishes."""
    jid = _start(db, job_id="f" * 32, ttl_s=600.0)
    real = time.time
    monkeypatch.setattr(js.time, "time", lambda: real() + 700)
    assert [r["job_id"] for r in db.list_jobs(user_id="u1", key_id="k1")] \
        == [jid]
    assert float(db.get(jid)["expires_ts"]) >= js.time.time()
    db.prune(ttl_s=600.0, max_rows=0, max_bytes=0)
    assert db.get(jid) is not None
    assert db.finish(job_id=jid, state="done", result={"text": "ok"},
                     ttl_s=600.0) is True
    row = db.get(jid)
    assert row["expires_ts"] == pytest.approx(js.time.time() + 600.0, abs=5)


def test_mark_running_as_failed_flips_only_running_rows(db, monkeypatch):
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.setattr(cfg, "SERVER_WORKERS", 1, raising=False)
    a = _start(db, job_id="a" * 32)
    b = _start(db, job_id="b" * 32)
    db.finish(job_id=b, state="done", result={"text": "1"}, ttl_s=_TTL)
    assert db.mark_running_as_failed("server restarted") == 1
    assert db.get(a)["state"] == "failed"
    assert db.get(a)["error"] == "server restarted"
    assert db.get(b)["state"] == "done"


def test_mark_running_as_failed_restamps_the_expiry_from_the_flip(
        db, monkeypatch):
    # The flip is the run's finish: a short TTL counts from it, not from
    # start()'s 24 h running floor.
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.setattr(cfg, "SERVER_WORKERS", 1, raising=False)
    monkeypatch.setattr(cfg, "JOBS_TTL_S", 600, raising=False)
    a = _start(db, job_id="a" * 32, ttl_s=600.0)
    assert db.get(a)["expires_ts"] > js.time.time() + 3600
    assert db.mark_running_as_failed("server restarted") == 1
    row = db.get(a)
    assert row["expires_ts"] == pytest.approx(row["finished_ts"] + 600.0,
                                              abs=1)


def test_finish_with_a_stale_request_id_leaves_the_re_posted_row_alone(db):
    # Run A closed its progress entry, a same-id re-post B started a fresh
    # row, then A's late finish lands: it must not stamp B's row.
    jid = _start(db, request_id="req-a")
    _start(db, request_id="req-b")
    assert db.finish(job_id=jid, state="done", result={"text": "A's"},
                     ttl_s=_TTL, request_id="req-a") is False
    row = db.get(jid)
    assert row["request_id"] == "req-b" and row["state"] == "running"
    assert db.get_result(jid) is None
    assert db.finish(job_id=jid, state="done", result={"text": "B's"},
                     ttl_s=_TTL, request_id="req-b") is True
    assert db.get_result(jid) == {"text": "B's"}


def test_mark_running_as_failed_is_skipped_with_several_workers(db, monkeypatch):
    # The DB is shared: a respawned worker must not fail its siblings' runs.
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.setattr(cfg, "SERVER_WORKERS", 4, raising=False)
    a = _start(db, job_id="a" * 32)
    assert db.mark_running_as_failed("server restarted") == 0
    assert db.get(a)["state"] == "running"


def test_delete_and_clear(db):
    a = _start(db, job_id="a" * 32)
    assert db.delete(a) is True and db.delete(a) is False
    _start(db, job_id="b" * 32)
    assert db.clear_all() == 1 and db.count() == 0


def test_delete_finished_only_spares_a_running_row(db):
    a = _start(db, job_id="a" * 32)
    assert db.delete(a, finished_only=True) is False
    assert db.get(a)["state"] == "running"
    db.finish(job_id=a, state="failed", error="x", ttl_s=_TTL)
    assert db.delete(a, finished_only=True) is True


def test_patch_result_patches_outside_the_lock_and_retries_a_lost_race(db):
    a = _start(db, job_id="a" * 32)
    db.finish(job_id=a, state="done", result={"text": "hi", "n": 0},
              ttl_s=_TTL)
    calls = []

    def _patch(payload):
        # The JSON work runs without the writer lock held …
        assert not db._lock._is_owned()
        calls.append(dict(payload))
        if len(calls) == 1:
            # … so a writer can land in between: this patch must not clobber
            # it, and is re-applied on the fresh payload.
            db.finish(job_id=a, state="done", result={"text": "hi", "n": 1},
                      ttl_s=_TTL)
        payload["video"] = True
        return True
    assert db.patch_result(a, _patch) is True
    assert [c["n"] for c in calls] == [0, 1]
    assert db.get_result(a) == {"text": "hi", "n": 1, "video": True}


def test_jobs_start_async_keeps_the_loop_free_while_the_lock_is_held(db):
    import asyncio
    import threading
    from faster_whisper_backend.transcription import progress as tx_progress
    held, release = threading.Event(), threading.Event()

    def _holder():
        with db._lock:
            held.set()
            release.wait(5)
    threading.Thread(target=_holder, daemon=True).start()
    assert held.wait(5)

    async def _main():
        gaps, last = [], time.monotonic()

        async def _ticker():
            nonlocal last
            while True:
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now
        tick = asyncio.create_task(_ticker())
        asyncio.get_running_loop().call_later(0.3, release.set)
        ok = await tx_progress._jobs_start_async(
            "a" * 32, request_id="req", kind="transcribe", user_id="u1",
            key_id=None, model="m", source_kind="file", source_name="a")
        tick.cancel()
        return ok, gaps
    ok, gaps = asyncio.run(_main())
    assert ok is True and db.get("a" * 32)["state"] == "running"
    assert len(gaps) >= 10 and max(gaps) < 0.2


def test_sweep_retention_reads_live_config(db, monkeypatch):
    """A lowered row / byte cap applies on the next sweep tick."""
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.setattr(cfg, "JOBS_MAX_ROWS", 100, raising=False)
    monkeypatch.setattr(cfg, "JOBS_MAX_BYTES", 0, raising=False)
    ids = []
    for jid in ("a" * 32, "b" * 32, "c" * 32):
        _start(db, job_id=jid)
        db.finish(job_id=jid, state="done", result={"t": 1}, ttl_s=_TTL)
        ids.append(jid)
        time.sleep(0.01)                    # created_ts apart: a oldest
    a, b, c = ids
    assert db.sweep_retention() == 0
    monkeypatch.setattr(cfg, "JOBS_MAX_ROWS", 2, raising=False)
    assert db.sweep_retention() == 1
    assert db.get(a) is None and db.get(b) is not None
    monkeypatch.setattr(cfg, "JOBS_MAX_ROWS", 0, raising=False)
    monkeypatch.setattr(cfg, "JOBS_MAX_BYTES", db.get(c)["result_bytes"],
                        raising=False)
    assert db.sweep_retention() == 1
    assert db.get(b) is None and db.get(c) is not None
