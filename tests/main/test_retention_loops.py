"""The hourly captures / reports retention sweeps run OFF the event loop:
a backlog sweep (retention day boundary, a lowered *_RETENTION_DAYS) does
hundreds of unlinks plus per-sample dissolves, all synchronous SQLite +
filesystem work that would otherwise block every in-flight transcription,
dictation frame and SSE tick."""

import asyncio
import threading

import pytest


def _drive(monkeypatch, loop_fn, store_mod):
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
    return calls


def test_captures_sweep_runs_off_the_loop_thread(app_module, monkeypatch):
    from faster_whisper_backend.captures import store as captures_store
    calls = _drive(monkeypatch, app_module._captures_retention_loop, captures_store)
    assert calls == [False]


def test_reports_sweep_runs_off_the_loop_thread(app_module, monkeypatch):
    from faster_whisper_backend.reports import store as reports_store
    calls = _drive(monkeypatch, app_module._reports_retention_loop, reports_store)
    assert calls == [False]


def test_recent_tx_sweep_runs_off_the_loop_thread(app_module, monkeypatch):
    from faster_whisper_backend.stats import recent_transcriptions_store
    calls = _drive(monkeypatch, app_module._recent_tx_retention_loop,
                   recent_transcriptions_store)
    assert calls == [False]


def test_startup_ages_out_recent_rows_past_retention(app_module, monkeypatch):
    """The lazy prune counts inserts per process, so a row written before a
    restart outlived RECENT_TRANSCRIPTIONS_RETENTION_DAYS until 50 more
    arrived. The lifespan sweeps once before serving."""
    import time

    from fastapi.testclient import TestClient
    from faster_whisper_backend.stats import recent_transcriptions_store as rts

    monkeypatch.setattr(app_module.cfg, "RECENT_TRANSCRIPTIONS_RETENTION_DAYS", 30,
                        raising=False)
    rts.init_db(app_module.cfg.RECENT_TRANSCRIPTIONS_DB)
    rts.record_trace(request_id="stale", model="m", raw="x", final="y",
                     created_ts=time.time() - 40 * 86400)
    rts.record_trace(request_id="fresh", model="m", raw="x", final="y")
    rts._require_conn().close()
    rts._conn = None

    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        ids = {r["request_id"] for r in rts.list_recent(limit=10)}
    assert ids == {"fresh"}
