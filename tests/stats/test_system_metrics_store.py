"""system_metrics_store: the /stats history rows, own SQLite file."""
import sqlite3

import pytest


def test_round_trip_downsample_and_prune(sm_store):
    import time
    # Minute-aligned so the six 10-s samples fall into exactly two 30-s buckets.
    now = int(time.time()) // 60 * 60
    rows = [{"ts": now - 60 + i * 10, "gpu_util": 10.0 * i, "gpu_mem_mb": 100.0,
             "gpu_temp": None, "cpu_pct": 5.0, "ram_pct": 50.0, "slot_busy": 0.5}
            for i in range(6)]
    assert sm_store.record(rows) == 6
    sm_store.record([{"ts": now - 60, "gpu_util": 99.0}])   # replace
    fine = sm_store.list_series(metric="gpu_util", from_ts=now - 60,
                                to_ts=now + 1, step_s=10)
    assert fine["t"] == [now - 60 + i * 10 for i in range(6)]
    assert fine["avg"][0] == 99.0 and fine["max"][-1] == 50.0
    coarse = sm_store.list_series(metric="gpu_util", from_ts=now - 60,
                                to_ts=now + 1, step_s=30)
    assert len(coarse["t"]) == 2
    assert coarse["max"][0] == 99.0 and coarse["avg"][1] == pytest.approx(40.0)
    # NULL samples (no NVML) are skipped, not zeroed.
    assert sm_store.list_series(metric="gpu_temp", from_ts=now - 60,
                                to_ts=now + 1, step_s=10)["t"] == []
    with pytest.raises(ValueError):
        sm_store.list_series(metric="ts; DROP TABLE x", from_ts=0,
                                  to_ts=now, step_s=10)
    sm_store.record([{"ts": now - 40 * 86400, "gpu_util": 1.0}])
    assert sm_store.prune(30) == 1
    assert sm_store.prune(0) == 0


def test_adopt_legacy_moves_sys_samples_once(sm_store, tmp_path):
    """The table lived in the recent-transcriptions DB as sys_samples before
    the split. adopt_legacy copies the rows, drops the old table, and is a
    no-op on a DB that has nothing to adopt."""
    legacy = sqlite3.connect(str(tmp_path / "recent.sqlite3"))
    legacy.row_factory = sqlite3.Row
    legacy.execute("CREATE TABLE sys_samples (ts INTEGER PRIMARY KEY, gpu_util REAL,"
                   " gpu_mem_mb REAL, gpu_temp REAL, cpu_pct REAL, ram_pct REAL,"
                   " slot_busy REAL)")
    legacy.executemany("INSERT INTO sys_samples VALUES (?,?,?,?,?,?,?)",
                       [(1000 + i * 10, 1.0 * i, None, None, None, None, 0.0)
                        for i in range(4)])
    legacy.commit()
    assert sm_store.adopt_legacy(legacy) == 4
    assert sm_store.list_series(metric="gpu_util", from_ts=0, to_ts=2000,
                                step_s=10)["avg"] == [0.0, 1.0, 2.0, 3.0]
    assert legacy.execute("SELECT name FROM sqlite_master WHERE name='sys_samples'"
                          ).fetchone() is None
    assert sm_store.adopt_legacy(legacy) == 0
