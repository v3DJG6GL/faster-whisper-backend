"""AUTO_CONVERT_HF_MODELS with a REVISION pin: the local snapshot the
converter reads must hold only the weight set from_pretrained loads, not every
*.safetensors / *.bin variant the repo ships.

The conversion stack is stubbed in sys.modules, as in test_auto_convert.py.
"""

import asyncio
import os
import sys
import types
from fnmatch import fnmatch

from faster_whisper_backend.transcription import models as tx_models

# openai/whisper-large-v3's top level, plus a Trainer-style subfolder.
_LARGE_V3 = [
    ".gitattributes", "README.md", "added_tokens.json", "config.json",
    "generation_config.json", "merges.txt", "model.safetensors",
    "model.fp32-00001-of-00002.safetensors",
    "model.fp32-00002-of-00002.safetensors",
    "model.safetensors.index.fp32.json", "normalizer.json",
    "preprocessor_config.json", "pytorch_model.bin",
    "pytorch_model.fp32-00001-of-00002.bin",
    "pytorch_model.fp32-00002-of-00002.bin",
    "pytorch_model.bin.index.fp32.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "training_args.bin", "checkpoint-500/model.safetensors",
    "checkpoint-500/config.json",
]


def _fetched(files, patterns):
    return {f for f in files if any(fnmatch(f, p) for p in patterns)}


def test_pinned_convert_snapshots_one_weight_set(monkeypatch, tmp_path):
    monkeypatch.setattr(tx_models.cfg, "AUTO_CONVERT_HF_MODELS", True,
                        raising=False)
    monkeypatch.setattr(tx_models.cfg, "CONVERT_QUANTIZATION", "float16",
                        raising=False)
    monkeypatch.setattr(tx_models.cfg, "CONVERTED_MODELS_DIR", str(tmp_path),
                        raising=False)
    monkeypatch.setattr(tx_models.cfg, "MODEL_OVERRIDES",
                        {"openai/whisper-large-v3": {"REVISION": "abc"}},
                        raising=False)
    monkeypatch.setattr(tx_models, "_convert_locks", {})
    patterns = []

    def _snapshot_download(repo_id, revision=None, allow_patterns=None):
        patterns.extend(allow_patterns)
        snap = tmp_path / "snap"
        snap.mkdir()
        return str(snap)

    class _Converter:
        def __init__(self, model_name_or_path, **kw):
            pass

        def convert(self, out, quantization, force):
            os.makedirs(out)
            open(os.path.join(out, "model.bin"), "w").close()

    class _FileLock:
        def __init__(self, path, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    hub = types.ModuleType("huggingface_hub")
    hub.list_repo_files = lambda repo_id, revision=None: list(_LARGE_V3)
    hub.snapshot_download = _snapshot_download
    fl = types.ModuleType("filelock")
    fl.FileLock = _FileLock
    fl.Timeout = type("Timeout", (Exception,), {})
    conv = types.ModuleType("ctranslate2.converters")
    conv.TransformersConverter = _Converter
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "filelock", fl)
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", conv)
    for mod in ("ctranslate2", "transformers", "torch"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))

    asyncio.run(tx_models._ensure_ct2_model("openai/whisper-large-v3"))

    fetched = _fetched(_LARGE_V3, patterns)
    weights = {f for f in fetched if f.endswith((".safetensors", ".bin"))}
    assert weights == {"model.safetensors"}
    assert {"config.json", "tokenizer.json",
            "preprocessor_config.json", "vocab.json",
            "merges.txt"} <= fetched
    assert not any("/" in f for f in fetched)


def test_pinned_snapshot_takes_shards_or_falls_back_to_pytorch_bin():
    sharded = ["config.json", "model.safetensors.index.json",
               "model-00001-of-00002.safetensors",
               "model-00002-of-00002.safetensors",
               "model.fp32-00001-of-00002.safetensors",
               "pytorch_model.bin"]
    assert set(tx_models._pinned_snapshot_files(sharded)) == {
        "config.json", "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors"}
    legacy = ["config.json", "pytorch_model.bin", "training_args.bin"]
    assert set(tx_models._pinned_snapshot_files(legacy)) == {
        "config.json", "pytorch_model.bin"}
