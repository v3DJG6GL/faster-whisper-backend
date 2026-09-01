"""Whisper job leases: a model a request is decoding on is never freed.

faster_whisper is never imported — the load path's only heavy dependency is
stubbed in sys.modules, exactly the boundary _get_or_load_model uses.
"""

import asyncio
import logging
import sys
import time
import types

import pytest
from fastapi import HTTPException

import main
import system_stats
from conftest import FakeModel

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


@pytest.fixture(autouse=True)
def _clean_cache():
    """These cases seed the model cache directly; leave it as we found it."""
    main._loaded_models.clear()
    system_stats._loaded_models.clear()
    yield
    main._loaded_models.clear()
    system_stats._loaded_models.clear()


def _register(name):
    """Put a fake model in the cache the way a successful load would."""
    main._loaded_models[name] = FakeModel()
    system_stats.register_loaded_model(name, 0, "cpu", "int8")


def _stub_load(monkeypatch):
    """Neutralise everything _get_or_load_model needs from the native stack."""
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = lambda path, **kw: FakeModel()
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)

    async def _same(name):
        return name
    monkeypatch.setattr(main, "_ensure_ct2_model", _same)
    monkeypatch.setattr(main.cfg, "ALLOWED_MODELS", set(), raising=False)
    # Skips the Hub pre-download block (no huggingface_hub in the test env).
    monkeypatch.setattr(main.cfg, "LOCAL_FILES_ONLY", True, raising=False)


# --- _drop_loaded_model ------------------------------------------------------

def test_drop_refuses_while_leased():
    _register("a")
    main._model_leases["a"] = 1
    assert main._drop_loaded_model("a") is False
    assert "a" in main._loaded_models
    assert system_stats._loaded_models.get("a") is not None


def test_drop_force_evicts_a_leased_model():
    _register("a")
    main._model_leases["a"] = 1
    assert main._drop_loaded_model("a", force=True) is True
    assert "a" not in main._loaded_models
    assert "a" not in system_stats._loaded_models


def test_drop_unleased_still_drops():
    _register("a")
    assert main._drop_loaded_model("a") is True
    assert "a" not in main._loaded_models


# --- LRU eviction ------------------------------------------------------------

def test_lru_skips_a_leased_entry(monkeypatch):
    _stub_load(monkeypatch)
    monkeypatch.setattr(main.cfg, "MAX_LOADED_MODELS", 2, raising=False)
    _register("a")
    _register("b")
    main._model_leases["a"] = 1  # oldest, but in use

    asyncio.run(main._get_or_load_model("c"))

    # "b" was the first UNLEASED entry, so it paid instead of "a".
    assert set(main._loaded_models) == {"a", "c"}


def test_all_leased_overflows_the_cap(monkeypatch, caplog):
    _stub_load(monkeypatch)
    monkeypatch.setattr(main.cfg, "MAX_LOADED_MODELS", 1, raising=False)
    _register("a")
    main._model_leases["a"] = 1

    with caplog.at_level(logging.WARNING, logger="whisper-server"):
        asyncio.run(main._get_or_load_model("b"))

    assert set(main._loaded_models) == {"a", "b"}
    assert any("temporarily exceeding MAX_LOADED_MODELS" in r.getMessage()
               for r in caplog.records)


# --- lease acquisition -------------------------------------------------------

def test_lease_taken_on_cache_hit_and_on_load(monkeypatch):
    _stub_load(monkeypatch)
    monkeypatch.setattr(main.cfg, "MAX_LOADED_MODELS", 4, raising=False)
    _register("a")

    asyncio.run(main._get_or_load_model("a", lease=True))   # lock-free hit
    asyncio.run(main._get_or_load_model("b", lease=True))   # fresh load
    assert main._model_leases == {"a": 1, "b": 1}

    asyncio.run(main._get_or_load_model("a"))               # no lease asked
    assert main._model_leases == {"a": 1, "b": 1}


def test_rejected_names_take_no_lease(monkeypatch):
    _stub_load(monkeypatch)
    # Allowlist gate: it sits before the cache and the traversal guard.
    monkeypatch.setattr(main.cfg, "ALLOWED_MODELS", {"a"}, raising=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main._get_or_load_model("b", lease=True))
    assert exc.value.status_code == 400
    assert main._model_leases == {}

    # No allowlist: the id-shape guard is the only thing between the request
    # string and os.path.isdir() / the converter (DEFAULT_MODEL stays exempt).
    monkeypatch.setattr(main.cfg, "ALLOWED_MODELS", set(), raising=False)
    monkeypatch.setattr(main.cfg, "DEFAULT_MODEL", "base", raising=False)
    for bad in ("../etc/passwd", "http://x/y"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main._get_or_load_model(bad, lease=True))
        assert exc.value.status_code == 400
        assert main._model_leases == {}


# --- _release_model_lease ----------------------------------------------------

def test_release_restamps_last_used():
    _register("a")
    main._model_leases["a"] = 1
    info = system_stats._loaded_models["a"]
    info["last_used_monotonic"] = time.monotonic() - 500
    stale = info["last_used_monotonic"]

    main._release_model_lease("a")

    assert main._model_leases == {}
    assert system_stats._loaded_models["a"]["last_used_monotonic"] > stale


def test_release_decrements_before_zero():
    _register("a")
    main._model_leases["a"] = 2
    main._release_model_lease("a")
    assert main._model_leases == {"a": 1}
    assert main._drop_loaded_model("a") is False


def test_release_tolerates_a_dropped_model():
    _register("a")
    main._model_leases["a"] = 1
    main._drop_loaded_model("a", force=True)   # drain_then_evict shape
    main._release_model_lease("a")             # must not KeyError
    assert main._model_leases == {}


# --- idle evictor ------------------------------------------------------------

def test_idle_evictor_defers_then_evicts(monkeypatch):
    _register("a")
    main._model_leases["a"] = 1
    system_stats._loaded_models["a"]["last_used_monotonic"] = \
        time.monotonic() - 500
    monkeypatch.setattr(main.cfg, "MODEL_IDLE_TIMEOUT_S", 1, raising=False)

    ticks = {"n": 0}
    seen: "list[bool]" = []

    async def _fake_sleep(_secs):
        ticks["n"] += 1
        if ticks["n"] == 2:
            # After the first (refused) sweep.
            seen.append("a" in main._loaded_models)
            main._release_model_lease("a")
            system_stats._loaded_models["a"]["last_used_monotonic"] = \
                time.monotonic() - 500
        if ticks["n"] > 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(main.asyncio, "sleep", _fake_sleep)
    asyncio.run(main._idle_evictor())

    assert seen == [True]                      # the lease deferred the evict
    assert "a" not in main._loaded_models      # the release let it through


# --- the batch handler balances its lease ------------------------------------

def _leasing_loader(app_module, model, held):
    """`held` records every name leased, so the tests can key the mid-decode
    snapshot off the RESOLVED name the handler asked for, not "whisper-1"."""
    async def _loader(name, *, lease=False):
        if lease:
            held.append(name)
            app_module._model_leases[name] = \
                app_module._model_leases.get(name, 0) + 1
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
        self.leases_during = dict(self._app._model_leases)
        if self._boom is not None:
            raise self._boom
        return super().transcribe(path, **kwargs)


def test_transcribe_releases_the_lease_on_success(client, app_module,
                                                  monkeypatch):
    held: "list[str]" = []
    model = _SnapshotModel(app_module)
    monkeypatch.setattr(app_module, "_get_or_load_model",
                        _leasing_loader(app_module, model, held))
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1"})
    assert r.status_code == 200
    assert held and model.leases_during == {held[0]: 1}
    assert app_module._model_leases == {}


def test_transcribe_releases_the_lease_on_error(client, app_module,
                                                monkeypatch):
    held: "list[str]" = []
    model = _SnapshotModel(app_module, boom=RuntimeError("decode blew up"))
    monkeypatch.setattr(app_module, "_get_or_load_model",
                        _leasing_loader(app_module, model, held))
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1"})
    assert r.status_code == 500
    assert held and model.leases_during == {held[0]: 1}
    assert app_module._model_leases == {}
