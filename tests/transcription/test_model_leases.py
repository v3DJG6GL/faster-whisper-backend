"""Whisper job leases: a model a request is decoding on is never freed.

faster_whisper is never imported — the load path's only heavy dependency is
stubbed in sys.modules, exactly the boundary _get_or_load_model uses.
"""

import asyncio
import threading
import logging
import sys
import time
import types

import pytest
from fastapi import HTTPException

from faster_whisper_backend.runtime import model_registry
from tests.conftest import FakeModel
from faster_whisper_backend.transcription import models as tx_models

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


@pytest.fixture(autouse=True)
def _clean_cache():
    """These cases seed the model cache directly; leave it as we found it."""
    tx_models._loaded_models.clear()
    model_registry._loaded_models.clear()
    yield
    tx_models._loaded_models.clear()
    model_registry._loaded_models.clear()


def _register(name):
    """Put a fake model in the cache the way a successful load would."""
    tx_models._loaded_models[name] = FakeModel()
    model_registry.register_loaded_model(name, 0, "cpu", "int8")


def _stub_load(monkeypatch):
    """Neutralise everything _get_or_load_model needs from the native stack."""
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = lambda path, **kw: FakeModel()
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)

    async def _same(name):
        return name
    monkeypatch.setattr(tx_models, "_ensure_ct2_model", _same)
    monkeypatch.setattr(tx_models.cfg, "ALLOWED_MODELS", set(), raising=False)
    # Skips the Hub pre-download block (no huggingface_hub in the test env).
    monkeypatch.setattr(tx_models.cfg, "LOCAL_FILES_ONLY", True, raising=False)


# --- _drop_loaded_model ------------------------------------------------------

def test_drop_refuses_while_leased():
    _register("a")
    tx_models._model_leases["a"] = 1
    assert tx_models._drop_loaded_model("a") is False
    assert "a" in tx_models._loaded_models
    assert model_registry._loaded_models.get("a") is not None


def test_drop_force_evicts_a_leased_model():
    _register("a")
    tx_models._model_leases["a"] = 1
    assert tx_models._drop_loaded_model("a", force=True) is True
    assert "a" not in tx_models._loaded_models
    assert "a" not in model_registry._loaded_models


def test_drop_unleased_still_drops():
    _register("a")
    assert tx_models._drop_loaded_model("a") is True
    assert "a" not in tx_models._loaded_models


# --- LRU eviction ------------------------------------------------------------

def test_lru_skips_a_leased_entry(monkeypatch):
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models.cfg, "MAX_LOADED_MODELS", 2, raising=False)
    _register("a")
    _register("b")
    tx_models._model_leases["a"] = 1  # oldest, but in use

    asyncio.run(tx_models._get_or_load_model("c"))

    # "b" was the first UNLEASED entry, so it paid instead of "a".
    assert set(tx_models._loaded_models) == {"a", "c"}


def test_all_leased_overflows_the_cap(monkeypatch, caplog):
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models.cfg, "MAX_LOADED_MODELS", 1, raising=False)
    _register("a")
    tx_models._model_leases["a"] = 1

    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        asyncio.run(tx_models._get_or_load_model("b"))

    assert set(tx_models._loaded_models) == {"a", "b"}
    assert any("temporarily exceeding MAX_LOADED_MODELS" in r.getMessage()
               for r in caplog.records)


# --- lease acquisition -------------------------------------------------------

def test_lease_taken_on_cache_hit_and_on_load(monkeypatch):
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models.cfg, "MAX_LOADED_MODELS", 4, raising=False)
    _register("a")

    asyncio.run(tx_models._get_or_load_model("a", lease=True))   # lock-free hit
    asyncio.run(tx_models._get_or_load_model("b", lease=True))   # fresh load
    assert tx_models._model_leases == {"a": 1, "b": 1}

    asyncio.run(tx_models._get_or_load_model("a"))               # no lease asked
    assert tx_models._model_leases == {"a": 1, "b": 1}


def test_rejected_names_take_no_lease(monkeypatch):
    _stub_load(monkeypatch)
    # Allowlist gate: it sits before the cache and the traversal guard.
    monkeypatch.setattr(tx_models.cfg, "ALLOWED_MODELS", {"a"}, raising=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(tx_models._get_or_load_model("b", lease=True))
    assert exc.value.status_code == 400
    assert tx_models._model_leases == {}

    # No allowlist: the id-shape guard is the only thing between the request
    # string and os.path.isdir() / the converter (DEFAULT_MODEL stays exempt).
    monkeypatch.setattr(tx_models.cfg, "ALLOWED_MODELS", set(), raising=False)
    monkeypatch.setattr(tx_models.cfg, "DEFAULT_MODEL", "base", raising=False)
    for bad in ("../etc/passwd", "http://x/y"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(tx_models._get_or_load_model(bad, lease=True))
        assert exc.value.status_code == 400
        assert tx_models._model_leases == {}


# --- _release_model_lease ----------------------------------------------------

def test_release_restamps_last_used():
    _register("a")
    tx_models._model_leases["a"] = 1
    info = model_registry._loaded_models["a"]
    info["last_used_monotonic"] = time.monotonic() - 500
    stale = info["last_used_monotonic"]

    tx_models._release_model_lease("a")

    assert tx_models._model_leases == {}
    assert model_registry._loaded_models["a"]["last_used_monotonic"] > stale


def test_release_decrements_before_zero():
    _register("a")
    tx_models._model_leases["a"] = 2
    tx_models._release_model_lease("a")
    assert tx_models._model_leases == {"a": 1}
    assert tx_models._drop_loaded_model("a") is False


def test_release_tolerates_a_dropped_model():
    _register("a")
    tx_models._model_leases["a"] = 1
    tx_models._drop_loaded_model("a", force=True)   # drain_then_evict shape
    tx_models._release_model_lease("a")             # must not KeyError
    assert tx_models._model_leases == {}


# --- idle evictor ------------------------------------------------------------

def test_idle_evictor_defers_then_evicts(monkeypatch):
    _register("a")
    tx_models._model_leases["a"] = 1
    model_registry._loaded_models["a"]["last_used_monotonic"] = \
        time.monotonic() - 500
    monkeypatch.setattr(tx_models.cfg, "MODEL_IDLE_TIMEOUT_S", 1, raising=False)

    ticks = {"n": 0}
    seen: "list[bool]" = []

    async def _fake_sleep(_secs):
        ticks["n"] += 1
        if ticks["n"] == 2:
            # After the first (refused) sweep.
            seen.append("a" in tx_models._loaded_models)
            tx_models._release_model_lease("a")
            model_registry._loaded_models["a"]["last_used_monotonic"] = \
                time.monotonic() - 500
        if ticks["n"] > 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(tx_models.asyncio, "sleep", _fake_sleep)
    asyncio.run(tx_models._idle_evictor())

    assert seen == [True]                      # the lease deferred the evict
    assert "a" not in tx_models._loaded_models      # the release let it through


# --- the batch handler balances its lease ------------------------------------

def _leasing_loader(app_module, model, held):
    """`held` records every name leased, so the tests can key the mid-decode
    snapshot off the RESOLVED name the handler asked for, not "whisper-1"."""
    async def _loader(name, *, lease=False):
        if lease:
            held.append(name)
            tx_models._model_leases[name] = \
                tx_models._model_leases.get(name, 0) + 1
        return model
    return _loader


class _SnapshotModel(FakeModel):
    """Records the live lease map at the moment the decode runs — the
    property the lease exists for is "held DURING the decode"."""

    def __init__(self, app_module, boom=None):
        super().__init__()
        self._app = app_module
        self._boom = boom
        self.leases_during = None

    def transcribe(self, path, **kwargs):
        self.leases_during = dict(tx_models._model_leases)
        if self._boom is not None:
            raise self._boom
        return super().transcribe(path, **kwargs)


def test_transcribe_releases_the_lease_on_success(client, app_module,
                                                  monkeypatch):
    held: "list[str]" = []
    model = _SnapshotModel(app_module)
    monkeypatch.setattr(tx_models, "_get_or_load_model",
                        _leasing_loader(app_module, model, held))
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1"})
    assert r.status_code == 200
    assert held and model.leases_during == {held[0]: 1}
    assert tx_models._model_leases == {}


def test_transcribe_releases_the_lease_on_error(client, app_module,
                                                monkeypatch):
    held: "list[str]" = []
    model = _SnapshotModel(app_module, boom=RuntimeError("decode blew up"))
    monkeypatch.setattr(tx_models, "_get_or_load_model",
                        _leasing_loader(app_module, model, held))
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1"})
    assert r.status_code == 500
    assert held and model.leases_during == {held[0]: 1}
    assert tx_models._model_leases == {}


# --- persistence runs off the event loop -------------------------------------

def test_register_loaded_model_runs_off_the_loop(monkeypatch):
    """register_loaded_model persists the measurement (model_sizes.record ->
    atomic_json.save_lock + fsync) and can block for the lock timeout when
    a peer worker holds the ledger; the load path must hand it to a thread."""
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models.cfg, "MAX_LOADED_MODELS", 4, raising=False)
    seen = {}

    def fake_register(name, **kw):
        seen[name] = threading.current_thread()
    monkeypatch.setattr(model_registry, "register_loaded_model", fake_register)

    asyncio.run(tx_models._get_or_load_model("a"))

    assert "a" in tx_models._loaded_models
    assert seen["a"] is not threading.main_thread()
