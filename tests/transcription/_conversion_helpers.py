"""Helpers shared by the AUTO_CONVERT_HF_MODELS test modules
(test_auto_convert, test_pinned_snapshot_weights).

Import as ``from tests.transcription._conversion_helpers import ...``.
"""

import sys
import types

from faster_whisper_backend.transcription import models as tx_models


class NoopFileLock:
    def __init__(self, path, timeout=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def stub_conversion_stack(monkeypatch, converter, *, converted_dir, overrides):
    """Stub filelock + ctranslate2's converter (+ transformers/torch) and turn
    AUTO_CONVERT_HF_MODELS on at float16; `converter` is the
    TransformersConverter stand-in."""
    for key, val in (("AUTO_CONVERT_HF_MODELS", True),
                     ("CONVERT_QUANTIZATION", "float16"),
                     ("CONVERTED_MODELS_DIR", str(converted_dir)),
                     ("MODEL_OVERRIDES", overrides)):
        monkeypatch.setattr(tx_models.cfg, key, val, raising=False)
    monkeypatch.setattr(tx_models, "_convert_locks", {})
    fl = types.ModuleType("filelock")
    fl.FileLock = NoopFileLock
    fl.Timeout = type("Timeout", (Exception,), {})
    conv = types.ModuleType("ctranslate2.converters")
    conv.TransformersConverter = converter
    monkeypatch.setitem(sys.modules, "filelock", fl)
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", conv)
    for mod in ("ctranslate2", "transformers", "torch"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
