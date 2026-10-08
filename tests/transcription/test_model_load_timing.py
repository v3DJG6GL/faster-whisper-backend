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

from faster_whisper_backend.runtime import model_registry
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
    model_registry._loaded_models.clear()


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
        assert model_registry._loaded_models["x"]["load_secs"] < 0.1
    finally:
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()


def test_cancel_during_register_keeps_the_model_cached_and_frees_the_lease(
        monkeypatch):
    """The registry write runs on a thread a cancellation cannot stop: the
    model must already be cached by then (no phantom registry entry), and
    the lease the cancelled caller will never release is given back."""
    import threading
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models, "_model_load_lock", asyncio.Lock())
    entered, go = threading.Event(), threading.Event()
    real_register = model_registry.register_loaded_model

    def _slow_register(*a, **kw):
        entered.set()
        go.wait(5)
        real_register(*a, **kw)
    monkeypatch.setattr(model_registry, "register_loaded_model",
                        _slow_register)

    async def run():
        task = asyncio.create_task(
            tx_models._get_or_load_model("x", lease=True))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        go.set()
        while "x" not in model_registry._loaded_models:
            await asyncio.sleep(0.01)

    try:
        asyncio.run(run())
        assert "x" in tx_models._loaded_models
        assert "x" in model_registry._loaded_models
        assert tx_models._model_leases.get("x", 0) == 0
    finally:
        go.set()
        tx_models._model_leases.pop("x", None)
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()


def test_cancel_during_construction_finishes_the_build_under_the_lock(
        monkeypatch):
    """A cancelled load must not release the load lock while its constructor
    is still running on the executor thread: the next waiter would start a
    second build of the same model. The build is finished, cached unleased,
    and the retry gets that model without a second constructor call."""
    import threading
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models, "_model_load_lock", asyncio.Lock())
    entered, go = threading.Event(), threading.Event()
    built = []

    def _slow_model(path, **kw):
        built.append(path)
        entered.set()
        go.wait(5)
        return FakeModel()
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = _slow_model
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)

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
        try:
            await a
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("the cancelled load must re-raise")
        return await b

    try:
        got = asyncio.run(run())
        assert built == ["x"]
        assert got is tx_models._loaded_models["x"]
        assert "x" in model_registry._loaded_models
        assert tx_models._model_leases.get("x", 0) == 1    # B's lease only
    finally:
        go.set()
        tx_models._model_leases.pop("x", None)
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()


def test_predownload_capture_scope_runs_on_the_executor_thread(monkeypatch):
    """The whisper pre-download's capture scope lives inside the executor
    callable, like the GGUF and pyannote fetches: a scope held on the loop
    across the await logged a cancelled caller's still-running download as
    aborted and wrote its done row on the event loop."""
    import contextlib
    import threading

    import huggingface_hub
    from faster_whisper_backend.runtime import download_progress
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models, "_model_load_lock", asyncio.Lock())
    monkeypatch.setattr(tx_models.cfg, "LOCAL_FILES_ONLY", False,
                        raising=False)
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
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot)

    async def run():
        seen["loop"] = threading.get_ident()
        await tx_models._get_or_load_model("org/x")

    try:
        asyncio.run(run())
        assert seen["enter"] == seen["snapshot"] == seen["exit"]
        assert seen["enter"] != seen["loop"]
        assert seen["kwargs"]["repo_id"] == "org/x"
        assert seen["kwargs"]["tqdm_class"] == "T"
    finally:
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()


def test_hardware_change_during_a_queued_load_reaches_the_constructor(
        monkeypatch):
    """A MODEL_DEVICE change saved while a cold load is still downloading or
    queued behind another model's load finds nothing cached for
    drain_then_evict to drop; the load must still build the model with the
    NEW device, not the one it would have read before it waited."""
    _stub_load(monkeypatch)
    built = []
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = lambda path, **kw: built.append(kw) or FakeModel()
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setattr(tx_models, "_model_load_lock", asyncio.Lock())
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

    try:
        asyncio.run(run())
        assert [kw["device"] for kw in built] == ["cuda"]
    finally:
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()


def test_vram_delta_is_sampled_on_the_load_device_index(monkeypatch):
    """With DEVICE_INDEX=1 on a two-GPU host the delta must come from GPU 1:
    sampling GPU 0 would record ~0 (or another process's usage) as the
    model's measured footprint."""
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models, "_model_load_lock", asyncio.Lock())
    monkeypatch.setattr(tx_models.cfg, "DEVICE_INDEX", 1, raising=False)
    # GPU 0 stays flat; GPU 1 grows by 3000 across the load.
    reads = {0: iter([500, 500]), 1: iter([1000, 4000])}
    monkeypatch.setattr(tx_models.system_stats, "gpu_mem_used_bytes",
                        lambda index=0: next(reads[index]))
    seen = {}
    monkeypatch.setattr(model_registry, "register_loaded_model",
                        lambda name, **kw: seen.__setitem__(name, kw))
    try:
        asyncio.run(tx_models._get_or_load_model("x"))
        assert seen["x"]["vram_bytes"] == 3000
    finally:
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()


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
    model_registry.register_loaded_model(name, 0, "cpu", "int8")
    model_registry._loaded_models[name]["last_used_monotonic"] = (
        time.monotonic() - 3600)


def test_evictor_does_not_claim_an_unload_it_was_refused(monkeypatch, caplog):
    """A lease taken between the pre-scan and the locked drop: the drop is
    refused, so no unload is logged and no full gc runs on the loop."""
    import gc
    _register_stale("a")
    tx_models._model_leases.pop("a", None)
    seen = {"n": 0}

    def _is_warm(name):
        # Called once in the pre-scan and once under the lock: the lease
        # lands between the two, after "a" was listed as stale.
        seen["n"] += 1
        if seen["n"] == 2:
            tx_models._model_leases["a"] = 1
        return False
    monkeypatch.setattr(model_registry, "is_warm", _is_warm)
    collected = []
    monkeypatch.setattr(gc, "collect", lambda *a: collected.append(1))
    try:
        with caplog.at_level(logging.INFO, logger="whisper-api"):
            _run_one_evictor_tick(monkeypatch)
        assert seen["n"] == 2, "the tick reached the locked drop"
        assert "a" in tx_models._loaded_models
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("[idle-evict] unload" in m for m in msgs)
        assert any("eviction deferred" in m for m in msgs)
        # Nothing was unloaded, so nothing to reclaim: no full gc on the loop.
        assert collected == []
    finally:
        tx_models._model_leases.pop("a", None)
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()


def test_evictor_prescan_skips_a_leased_model(monkeypatch, caplog):
    import gc
    _register_stale("a")
    tx_models._model_leases["a"] = 1
    collected = []
    monkeypatch.setattr(gc, "collect", lambda *a: collected.append(1))
    try:
        with caplog.at_level(logging.INFO, logger="whisper-api"):
            _run_one_evictor_tick(monkeypatch)
        assert "a" in tx_models._loaded_models
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("[idle-evict] unload" in m for m in msgs)
        # The pre-scan skips a leased model outright: a long decode must not
        # log a deferred eviction (and take the load lock) on every tick.
        assert not any("eviction deferred" in m for m in msgs)
        # Nothing was unloaded, so nothing to reclaim: no full gc on the loop.
        assert collected == []
    finally:
        tx_models._model_leases.pop("a", None)
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()


def test_evictor_logs_the_unload_it_performed(monkeypatch, caplog):
    import gc
    _register_stale("b")
    tx_models._model_leases.pop("b", None)
    collected = []
    monkeypatch.setattr(gc, "collect", lambda *a: collected.append(1))
    try:
        with caplog.at_level(logging.INFO, logger="whisper-api"):
            _run_one_evictor_tick(monkeypatch)
        assert "b" not in tx_models._loaded_models
        assert collected == [1]
        assert any("[idle-evict] unloaded b after 1s idle" == r.getMessage()
                   for r in caplog.records)
    finally:
        tx_models._loaded_models.clear()
        model_registry._loaded_models.clear()
