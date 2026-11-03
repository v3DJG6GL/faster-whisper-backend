"""Helpers shared by the whisper load-path test modules
(test_model_leases, test_model_load_timing).

Import as ``from tests.transcription._load_helpers import ...``.
"""

import sys
import types

from tests.conftest import FakeModel
from faster_whisper_backend.transcription import models as tx_models


def stub_whisper_load(monkeypatch, *, whisper_model=None, max_loaded=None):
    """Neutralise everything _get_or_load_model needs from the native stack;
    `whisper_model` stands in for faster_whisper.WhisperModel."""
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = (whisper_model if whisper_model is not None
                       else lambda path, **kw: FakeModel())
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
