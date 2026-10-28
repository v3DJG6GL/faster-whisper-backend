"""Reads on the shared store connections run under the store lock: two threads
running the same SELECT text share pysqlite's cached statement, and each can
read the other's rows (another user's, under an own-scope filter)."""
import threading
import time


def _held_elsewhere(lock) -> bool:
    """True when some other thread holds `lock` (an RLock held by the caller
    still counts: the probe runs in a fresh thread)."""
    out = []

    def probe():
        got = lock.acquire(blocking=False)
        if got:
            lock.release()
        out.append(not got)

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return out[0]


class _LockCheckingConn:
    def __init__(self, real, lock):
        self._real = real
        self._lock = lock
        self.unlocked: list[str] = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def execute(self, sql, *a):
        if not _held_elsewhere(self._lock):
            self.unlocked.append(sql)
        return self._real.execute(sql, *a)


def test_recent_transcriptions_reads_hold_the_lock(tx_store, monkeypatch):
    tx_store.record_timing(request_id="r1", model="m", audio_s=1.0,
                           processing_s=0.5, status="ok", words=2)
    conn = _LockCheckingConn(tx_store._conn, tx_store._lock)
    monkeypatch.setattr(tx_store, "_conn", conn)

    assert len(tx_store.list_recent(limit=5, user_id_filter=None)) == 1
    assert tx_store.count() == 1
    assert conn.unlocked == []


def test_system_metrics_list_series_holds_the_lock(sm_store, monkeypatch):
    sm_store.record([{"ts": 1000, "gpu_util": 1.0}])
    conn = _LockCheckingConn(sm_store._conn, sm_store._lock)
    monkeypatch.setattr(sm_store, "_conn", conn)

    out = sm_store.list_series(metric="gpu_util", from_ts=0, to_ts=2000, step_s=10)
    assert out["avg"] == [1.0]
    assert conn.unlocked == []


def test_jobs_store_reads_hold_the_lock(tmp_path, monkeypatch):
    from faster_whisper_backend.transcription import jobs_store as js

    js._reset_for_tests()
    js.init_db(str(tmp_path / "jobs.sqlite3"))
    try:
        js.start(job_id="a" * 32, request_id="r", kind="transcribe",
                 user_id="u1", key_id="k1", model="m", source_kind="file",
                 source_name="a.wav", ttl_s=3600.0, max_rows=100, max_bytes=0,
                 prune_every=0)
        js.finish(job_id="a" * 32, state="done", result={"text": "hi"},
                  ttl_s=3600.0)
        conn = _LockCheckingConn(js._conn, js._lock)
        monkeypatch.setattr(js, "_conn", conn)

        assert len(js.list_jobs(user_id="u1", key_id="k1")) == 1
        assert js.get("a" * 32)["state"] == "done"
        assert js.get_result("a" * 32) == {"text": "hi"}
        assert js.count() == 1
        assert js.total_result_bytes() > 0
        assert js.patch_result("a" * 32, lambda p: p.update(x=1) or True)
        assert conn.unlocked == []
    finally:
        monkeypatch.undo()
        js._reset_for_tests()


def test_usage_store_reads_hold_the_lock(usage_store_db, monkeypatch):
    us = usage_store_db
    us.record_usage(user_id="u1", key_id="k1", model="m", audio_s=1.0,
                    processing_s=0.5, words=2, status="ok", kind="transcribe",
                    job_id="j1")
    conn = _LockCheckingConn(us._conn, us._lock)
    monkeypatch.setattr(us, "_conn", conn)
    now = time.time()
    win = dict(start_ts=0.0, end_ts=now + 1)

    us.totals_by_key()
    us.totals_by_user()
    us.totals_for_user("u1")
    us.series()
    us.leaderboard()
    us.distinct_ids("user_id")
    us.wait_quantiles(**win, user_id="u1")
    us.wait_series_by_day(**win, tz=None, user_id="u1")
    us.turnaround_histogram(**win, user_id="u1")
    us.failures(**win, user_id="u1")
    us.by_model(**win, user_id="u1")
    us.overview(user_id="u1", tz=None, tz_name="UTC", days=7)
    us.document("u1", tz=None, tz_name="UTC", days=7)
    us.tail(user_id="u1", tz=None, tz_name="UTC", days=7)
    assert conn.unlocked == []
