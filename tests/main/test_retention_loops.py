"""The hourly captures / reports retention sweeps run OFF the event loop:
a backlog sweep (retention day boundary, a lowered *_RETENTION_DAYS) does
hundreds of unlinks plus per-sample dissolves, all synchronous SQLite +
filesystem work that would otherwise block every in-flight transcription,
dictation frame and SSE tick."""

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from tests.conftest import _close_store


def _drive(monkeypatch, loop_fn, store_mod):
    """Run `loop_fn` for one tick; the sweep must run once, off the loop thread."""
    calls = []
    ticks = {"n": 0}
    intervals = []

    def _sweep():
        calls.append(threading.current_thread() is threading.main_thread())
        return 0
    monkeypatch.setattr(store_mod, "sweep_retention", _sweep)

    _real_sleep = asyncio.sleep

    async def _fake_sleep(secs):
        intervals.append(secs)
        ticks["n"] += 1
        if ticks["n"] > 1:
            raise asyncio.CancelledError()
        await _real_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await loop_fn()
    asyncio.run(_run())
    assert intervals and intervals[0] == 3600
    assert calls == [False]


def test_captures_sweep_runs_off_the_loop_thread(app_module, monkeypatch):
    from faster_whisper_backend.captures import store as captures_store
    _drive(monkeypatch, app_module._captures_retention_loop, captures_store)


def test_reports_sweep_runs_off_the_loop_thread(app_module, monkeypatch):
    from faster_whisper_backend.reports import store as reports_store
    _drive(monkeypatch, app_module._reports_retention_loop, reports_store)


def test_recent_tx_sweep_runs_off_the_loop_thread(app_module, monkeypatch):
    from faster_whisper_backend.stats import recent_transcriptions_store
    _drive(monkeypatch, app_module._recent_tx_retention_loop, recent_transcriptions_store)


def test_startup_ages_out_recent_rows_past_retention(app_module, monkeypatch):
    """The lazy prune counts inserts per process, so a row written before a
    restart outlived RECENT_TRANSCRIPTIONS_RETENTION_DAYS until 50 more
    arrived. The lifespan sweeps once before serving."""
    import time

    from faster_whisper_backend.stats import recent_transcriptions_store as rts

    monkeypatch.setattr(app_module.cfg, "RECENT_TRANSCRIPTIONS_RETENTION_DAYS", 30,
                        raising=False)
    rts.init_db(app_module.cfg.RECENT_TRANSCRIPTIONS_DB)
    rts.record_trace(request_id="stale", model="m", raw="x", final="y",
                     created_ts=time.time() - 40 * 86400)
    rts.record_trace(request_id="fresh", model="m", raw="x", final="y")
    _close_store(rts)

    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        ids = {r["request_id"] for r in rts.list_recent(limit=10)}
    assert ids == {"fresh"}


def test_startup_sweeps_usage_rows_past_retention(app_module, monkeypatch):
    """The hourly usage loop sleeps before its first sweep, so a server
    restarted more often than hourly kept per-app rows past
    USAGE_APP_RETENTION_DAYS. The lifespan sweeps once before serving."""
    from faster_whisper_backend.stats import usage_store as us

    monkeypatch.setattr(app_module.cfg, "USAGE_APP_RETENTION_DAYS", 7, raising=False)
    us.init_db(app_module.cfg.USAGE_DB)
    now_h = us.now_hour()
    us._require_conn().executemany(
        "INSERT INTO usage_app_hourly VALUES (?, 'u', ?, 1, 1)",
        [(now_h - 10 * 24, "old-app"), (now_h, "new-app")])
    _close_store(us)

    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        apps = {r[0] for r in us._require_conn().execute(
            "SELECT app_id FROM usage_app_hourly")}
    assert apps == {"new-app"}
