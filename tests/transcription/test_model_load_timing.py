"""Whisper load accounting and the idle evictor's log line.

`load_secs` is the receipt's named cold-load cost, so a request that merely
queued behind another model's construction must not be billed that wait; and
the evictor must only claim an unload that actually happened.
"""

import asyncio
import logging
import sys
import time
import types

from faster_whisper_backend.runtime import system_stats
from tests.conftest import FakeModel
from faster_whisper_backend.transcription import models as tx_models


def _stub_load(monkeypatch):
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = lambda path, **kw: FakeModel()
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)

    async def _same(name):
        return name
    monkeypatch.setattr(tx_models, "_ensure_ct2_model", _same)
    monkeypatch.setattr(tx_models.cfg, "ALLOWED_MODELS", set(), raising=False)
    monkeypatch.setattr(tx_models.cfg, "LOCAL_FILES_ONLY", True, raising=False)
    monkeypatch.setattr(tx_models.cfg, "MAX_LOADED_MODELS", 4, raising=False)
    tx_models._loaded_models.clear()
    system_stats._loaded_models.clear()


def test_lock_wait_is_not_billed_as_load_time(monkeypatch):
    _stub_load(monkeypatch)
    # A throwaway lock: contending the production global inside this test's
    # own asyncio.run() loop would leave it bound to a closed loop.
    monkeypatch.setattr(tx_models, "_model_load_lock", asyncio.Lock())
    recorded = {}
    monkeypatch.setattr(tx_models.metrics, "record_model_load",
                        lambda name, secs: recorded.__setitem__(name, secs))

    async def run():
        await tx_models._model_load_lock.acquire()
        try:
            task = asyncio.create_task(tx_models._get_or_load_model("x"))
            await asyncio.sleep(0.25)
        finally:
            tx_models._model_load_lock.release()
        await task

    try:
        asyncio.run(run())
        assert "x" in recorded
        assert recorded["x"] < 0.1
        assert system_stats._loaded_models["x"]["load_secs"] < 0.1
    finally:
        tx_models._loaded_models.clear()
        system_stats._loaded_models.clear()


def _run_one_evictor_tick(monkeypatch):
    calls = {"n": 0}

    async def _fake_sleep(_secs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(tx_models.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(tx_models.cfg, "MODEL_IDLE_TIMEOUT_S", 1, raising=False)
    asyncio.run(tx_models._idle_evictor())


def _register_stale(name):
    tx_models._loaded_models[name] = FakeModel()
    system_stats.register_loaded_model(name, 0, "cpu", "int8")
    system_stats._loaded_models[name]["last_used_monotonic"] = (
        time.monotonic() - 3600)


def test_evictor_does_not_claim_an_unload_it_was_refused(monkeypatch, caplog):
    _register_stale("a")
    tx_models._model_leases["a"] = 1
    try:
        with caplog.at_level(logging.INFO, logger="whisper-api"):
            _run_one_evictor_tick(monkeypatch)
        assert "a" in tx_models._loaded_models
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("[idle-evict] unload" in m for m in msgs)
        assert any("eviction deferred" in m for m in msgs)
    finally:
        tx_models._model_leases.pop("a", None)
        tx_models._loaded_models.clear()
        system_stats._loaded_models.clear()


def test_evictor_logs_the_unload_it_performed(monkeypatch, caplog):
    _register_stale("b")
    tx_models._model_leases.pop("b", None)
    try:
        with caplog.at_level(logging.INFO, logger="whisper-api"):
            _run_one_evictor_tick(monkeypatch)
        assert "b" not in tx_models._loaded_models
        assert any("[idle-evict] unloaded b after 1s idle" == r.getMessage()
                   for r in caplog.records)
    finally:
        tx_models._loaded_models.clear()
        system_stats._loaded_models.clear()
