"""Helpers shared by the whisper load-path test modules
(test_model_leases, test_model_load_timing).

Import as ``from tests.transcription._load_helpers import ...``.

The conftest reset hook already gives each test empty caches, no leases and
a fresh _model_load_lock; only a test that runs a second asyncio.run() after
a contended one needs its own lock.
"""

import sys
import threading
import time
import types

from faster_whisper_backend.runtime import model_registry
from tests.conftest import FakeModel
from faster_whisper_backend.transcription import models as tx_models


def stub_whisper_load(monkeypatch, *, whisper_model=None, max_loaded=None):
    """Neutralise everything _get_or_load_model needs from the native stack;
    `whisper_model` stands in for faster_whisper.WhisperModel. Returns the
    paths the default stand-in built."""
    built = []
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = (whisper_model if whisper_model is not None
                       else lambda path, **kw: built.append(path) or FakeModel())
    # The short names these cases load, standing in for the size list.
    fw.available_models = lambda: ["a", "b", "c", "x", "y"]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)

    async def _same(name):
        return name
    monkeypatch.setattr(tx_models, "_ensure_ct2_model", _same)
    monkeypatch.setattr(tx_models.cfg, "ALLOWED_MODELS", set(), raising=False)
    # Skips the Hub pre-download, so the stubbed load never touches the network.
    monkeypatch.setattr(tx_models.cfg, "LOCAL_FILES_ONLY", True, raising=False)
    if max_loaded is not None:
        monkeypatch.setattr(tx_models.cfg, "MAX_LOADED_MODELS", max_loaded,
                            raising=False)
    return built


def stub_gated_load(monkeypatch, *, max_loaded, only=None):
    """stub_whisper_load with a WhisperModel that holds on a gate (5 s at most)
    while building `only` (None: any name); returns (entered, gate, built)."""
    entered, gate = threading.Event(), threading.Event()
    built = []

    def _build(path, **kw):
        built.append(path)
        if only is None or path == only:
            entered.set()
            gate.wait(5)
        return FakeModel()
    stub_whisper_load(monkeypatch, whisper_model=_build, max_loaded=max_loaded)
    return entered, gate, built


def register_fake_model(name, leases=0):
    """Put a fake model in the cache the way a successful load would, held by
    `leases` requests."""
    tx_models._loaded_models[name] = FakeModel()
    model_registry.register_loaded_model(name, 0, "cpu", "int8")
    if leases:
        tx_models._model_leases[name] = leases


def register_stale_model(name, leases=0, idle_s=3600):
    """register_fake_model, last used `idle_s` seconds ago."""
    register_fake_model(name, leases)
    model_registry._loaded_models[name]["last_used_monotonic"] = (
        time.monotonic() - idle_s)


def stub_hub_snapshot(monkeypatch, snapshot):
    """Let the load's Hub pre-download run (LOCAL_FILES_ONLY off) against
    `snapshot`, the huggingface_hub.snapshot_download stand-in."""
    import huggingface_hub
    monkeypatch.setattr(tx_models.cfg, "LOCAL_FILES_ONLY", False, raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
