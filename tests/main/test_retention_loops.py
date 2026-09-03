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
    from faster_whisper_backend.admin import reports_store
    calls = _drive(monkeypatch, app_module._reports_retention_loop, reports_store)
    assert calls == [False]
