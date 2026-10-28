"""Reads on the shared store connections run under the store lock: two threads
running the same SELECT text share pysqlite's cached statement, and each can
read the other's rows (another user's, under an own-scope filter)."""
import threading


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
