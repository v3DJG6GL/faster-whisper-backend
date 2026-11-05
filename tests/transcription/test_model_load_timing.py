"""Whisper load accounting and the idle evictor's log line.

`load_secs` is the receipt's named cold-load cost, so a request that merely
queued behind another model's construction must not be billed that wait; and
the evictor must only claim an unload that actually happened.
"""

import asyncio
import logging
import sys
import threading
import types

import pytest

from faster_whisper_backend.runtime import model_registry
from tests._transcription_helpers import (
    blocked_register,
    cancel_while_register_held,
    cancel_with_register_queued,
)
from tests.conftest import FakeModel
from faster_whisper_backend.transcription import models as tx_models
from tests.transcription._load_helpers import register_stale_model as _register_stale
from tests.transcription._load_helpers import stub_gated_load as _stub_gated
from tests.transcription._load_helpers import stub_hub_snapshot as _stub_hub
from tests.transcription._load_helpers import stub_whisper_load as _stub_load


def test_lock_wait_is_not_billed_as_load_time(monkeypatch):
    _stub_load(monkeypatch, max_loaded=4)
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

    asyncio.run(run())
    assert "x" in recorded
    assert recorded["x"] < 0.1
    assert model_registry._loaded_models["x"]["load_secs"] < 0.1


def test_load_in_progress_covers_the_conversion_outside_the_lock(monkeypatch):
    """A cold load spends its CT2 conversion / Hub pre-download with
    _model_load_lock free; load_in_progress() must still report it, or a
    preload is admitted and the job's constructor queues behind the warm-up."""
    _stub_load(monkeypatch, max_loaded=4)
    seen = []

    async def _converting(name):
        seen.append((tx_models._model_load_lock.locked(),
                     tx_models.load_in_progress()))
        return name
    monkeypatch.setattr(tx_models, "_ensure_ct2_model", _converting)

    assert tx_models.load_in_progress() is False
    asyncio.run(tx_models._get_or_load_model("x"))
    assert seen == [(False, True)]
    assert tx_models.load_in_progress() is False
    # A failed load releases the count too.
    async def _boom(name):
        raise RuntimeError("convert failed")
    monkeypatch.setattr(tx_models, "_ensure_ct2_model", _boom)
    with pytest.raises(RuntimeError):
        asyncio.run(tx_models._get_or_load_model("y"))
    assert tx_models._loads_in_flight == 0


def test_cancel_during_register_keeps_the_model_cached_and_frees_the_lease(
        monkeypatch):
    """The registry write runs on a thread a cancellation cannot stop: the
    model must already be cached by then (no phantom registry entry), and
    the lease the cancelled caller will never release is given back."""
    _stub_load(monkeypatch, max_loaded=4)
    entered, go = blocked_register(monkeypatch)
    cancel_while_register_held(
        lambda: tx_models._get_or_load_model("x", lease=True), entered, go)
    assert "x" in tx_models._loaded_models
    assert "x" in model_registry._loaded_models
    assert tx_models._model_leases.get("x", 0) == 0


def test_cancel_while_register_is_queued_still_registers(monkeypatch):
    """The default executor is shared with every other to_thread call: with
    its workers busy the registry write is still queued when the cancel
    lands, and a plain cancelled to_thread future drops the queued item. The
    cached model must still be registered (the idle evictor and /stats walk
    the registry, never the cache)."""
    built = _stub_load(monkeypatch, max_loaded=4)
    cancel_with_register_queued(
        monkeypatch, built, lambda: tx_models._get_or_load_model("x"))
    assert "x" in tx_models._loaded_models
    assert "x" in model_registry._loaded_models


def test_cancelled_load_holds_the_lock_until_its_queued_register_lands(
        monkeypatch):
    """A cancelled load must not free the load lock while its registry write
    is still queued: an eviction in that window pops the cached model and
    unregisters nothing, then the late write leaves a registry entry with no
    model behind it (/stats and the admin page list it as loaded)."""
    import concurrent.futures
    _stub_load(monkeypatch, max_loaded=4)
    gate, queued = threading.Event(), []

    def _record_then_busy_the_pool(name, secs):
        asyncio.get_running_loop().run_in_executor(None, gate.wait, 5)
        queued.append(name)
    monkeypatch.setattr(tx_models.metrics, "record_model_load",
                        _record_then_busy_the_pool)

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1))
        task = asyncio.create_task(tx_models._get_or_load_model("x"))
        while not queued:
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.05)
        evicting = asyncio.create_task(tx_models.evict("x"))
        await asyncio.sleep(0.05)
        assert not evicting.done()      # the cancelled load still holds it
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await evicting is True
        await loop.run_in_executor(None, lambda: None)

    try:
        asyncio.run(run())
        assert "x" not in tx_models._loaded_models
        assert "x" not in model_registry._loaded_models
    finally:
        gate.set()


def test_cancel_during_construction_finishes_the_build_under_the_lock(
        monkeypatch):
    """A cancelled load must not release the load lock while its constructor
    is still running on the executor thread: the next waiter would start a
    second build of the same model. The build is finished, cached unleased,
    and the retry gets that model without a second constructor call."""
    entered, go, built = _stub_gated(monkeypatch, max_loaded=4)

    async def run():
        a = asyncio.create_task(tx_models._get_or_load_model("x", lease=True))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        a.cancel()
        await asyncio.sleep(0.05)
        b = asyncio.create_task(tx_models._get_or_load_model("x", lease=True))
        await asyncio.sleep(0.05)
        assert len(built) == 1          # B waits on the lock A still holds
        go.set()
        with pytest.raises(asyncio.CancelledError):
            await a
        return await b

    try:
        got = asyncio.run(run())
        assert built == ["x"]
        assert got is tx_models._loaded_models["x"]
        assert "x" in model_registry._loaded_models
        assert tx_models._model_leases.get("x", 0) == 1    # B's lease only
    finally:
        go.set()


def test_predownload_capture_scope_runs_on_the_executor_thread(monkeypatch):
    """The whisper pre-download's capture scope lives inside the executor
    callable, like the GGUF and pyannote fetches: a scope held on the loop
    across the await logged a cancelled caller's still-running download as
    aborted and wrote its done row on the event loop."""
    import contextlib

    from faster_whisper_backend.runtime import download_progress
    _stub_load(monkeypatch, max_loaded=4)
    seen = {}

    @contextlib.contextmanager
    def _spy_capture(label, cb=None, **kw):
        seen["enter"] = threading.get_ident()
        try:
            yield types.SimpleNamespace(tqdm_kwargs={"tqdm_class": "T"})
        finally:
            seen["exit"] = threading.get_ident()
    monkeypatch.setattr(download_progress, "capture", _spy_capture)

    def _fake_snapshot(**kw):
        seen["snapshot"] = threading.get_ident()
        seen["kwargs"] = kw
    _stub_hub(monkeypatch, _fake_snapshot)

    async def run():
        seen["loop"] = threading.get_ident()
        await tx_models._get_or_load_model("org/x")

    asyncio.run(run())
    assert seen["enter"] == seen["snapshot"] == seen["exit"]
    assert seen["enter"] != seen["loop"]
    assert seen["kwargs"]["repo_id"] == "org/x"
    assert seen["kwargs"]["tqdm_class"] == "T"


def test_hardware_change_during_a_queued_load_reaches_the_constructor(
        monkeypatch):
    """A MODEL_DEVICE change saved while a cold load is still downloading or
    queued behind another model's load finds nothing cached for
    drain_then_evict to drop; the load must still build the model with the
    NEW device, not the one it would have read before it waited."""
    built = []
    _stub_load(monkeypatch,
               whisper_model=lambda path, **kw: built.append(kw) or FakeModel(),
               max_loaded=4)
    monkeypatch.setattr(tx_models.cfg, "MODEL_DEVICE", "cpu", raising=False)

    async def run():
        await tx_models._model_load_lock.acquire()
        try:
            task = asyncio.create_task(tx_models._get_or_load_model("x"))
            await asyncio.sleep(0.05)       # the load now waits on the lock
            assert "x" not in tx_models._loaded_models  # nothing to evict yet
            monkeypatch.setattr(tx_models.cfg, "MODEL_DEVICE", "cuda",
                                raising=False)
        finally:
            tx_models._model_load_lock.release()
        await task

    asyncio.run(run())
    assert [kw["device"] for kw in built] == ["cuda"]


def test_vram_delta_is_sampled_on_the_load_device_index(monkeypatch):
    """With DEVICE_INDEX=1 on a two-GPU host the delta must come from GPU 1:
    sampling GPU 0 would record ~0 (or another process's usage) as the
    model's measured footprint."""
    _stub_load(monkeypatch, max_loaded=4)
    monkeypatch.setattr(tx_models.cfg, "DEVICE_INDEX", 1, raising=False)
    reads = {0: iter([500, 500]), 1: iter([1000, 4000])}
    monkeypatch.setattr(tx_models.system_stats, "gpu_mem_used_bytes",
                        lambda index=0: next(reads[index]))
    seen = {}
    monkeypatch.setattr(model_registry, "register_loaded_model",
                        lambda name, **kw: seen.__setitem__(name, kw))
    asyncio.run(tx_models._get_or_load_model("x"))
    assert seen["x"]["vram_bytes"] == 3000


def _run_one_evictor_tick(monkeypatch):
    """One _idle_evictor sweep with gc.collect stubbed; returns one entry per
    gc.collect() call."""
    import gc
    collected = []
    monkeypatch.setattr(gc, "collect", lambda *a: collected.append(1))
    calls = {"n": 0}

    async def _fake_sleep(_secs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(tx_models.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(tx_models.cfg, "MODEL_IDLE_TIMEOUT_S", 1, raising=False)
    asyncio.run(tx_models._idle_evictor())
    return collected


def test_evictor_does_not_claim_an_unload_it_was_refused(monkeypatch, caplog):
    """A lease taken between the pre-scan and the locked drop: the drop is
    refused, so no unload is logged and no full gc runs on the loop."""
    _register_stale("a")
    seen = {"n": 0}

    def _is_warm(name):
        # Called once in the pre-scan and once under the lock: the lease
        # lands between the two, after "a" was listed as stale.
        seen["n"] += 1
        if seen["n"] == 2:
            tx_models._model_leases["a"] = 1
        return False
    monkeypatch.setattr(model_registry, "is_warm", _is_warm)
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        collected = _run_one_evictor_tick(monkeypatch)
    assert seen["n"] == 2, "the tick reached the locked drop"
    assert "a" in tx_models._loaded_models
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("[idle-evict] unload" in m for m in msgs)
    assert any("eviction deferred" in m for m in msgs)
    assert collected == []


def test_evictor_prescan_skips_a_leased_model(monkeypatch, caplog):
    _register_stale("a", leases=1)
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        collected = _run_one_evictor_tick(monkeypatch)
    assert "a" in tx_models._loaded_models
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("[idle-evict] unload" in m for m in msgs)
    # The pre-scan skips a leased model outright: a long decode must not
    # log a deferred eviction (and take the load lock) on every tick.
    assert not any("eviction deferred" in m for m in msgs)
    # Nothing was unloaded, so nothing to reclaim: no full gc on the loop.
    assert collected == []


def test_evictor_logs_the_unload_it_performed(monkeypatch, caplog):
    _register_stale("b")
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        collected = _run_one_evictor_tick(monkeypatch)
    assert "b" not in tx_models._loaded_models
    assert collected == [1]
    assert any("[idle-evict] unloaded b after 1s idle" == r.getMessage()
               for r in caplog.records)


class _ImportRecorder:
    """A meta-path finder that only records which modules were looked up."""

    def __init__(self):
        self.names = []

    def find_spec(self, name, path=None, target=None):
        self.names.append(name)
        return None


def test_evictor_does_not_cold_import_torch(monkeypatch):
    """A whisper-only process never imported torch: the unload must not pull
    libtorch + CUDA in on the event loop just to empty a cache it never had."""
    _register_stale("c")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    recorder = _ImportRecorder()
    monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
    _run_one_evictor_tick(monkeypatch)
    assert "c" not in tx_models._loaded_models
    assert "torch" not in recorder.names
    assert "torch" not in sys.modules


def test_evictor_empties_an_already_imported_torch_cache(monkeypatch):
    _register_stale("d")
    emptied = []
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: emptied.append(1))
    monkeypatch.setitem(sys.modules, "torch", torch)
    _run_one_evictor_tick(monkeypatch)
    assert "d" not in tx_models._loaded_models
    assert emptied == [1]
