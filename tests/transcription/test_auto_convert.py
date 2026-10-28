"""AUTO_CONVERT_HF_MODELS: the CT2 conversion of a transformers checkpoint.

The conversion stack (huggingface_hub, filelock, ctranslate2's converter,
transformers, torch) is stubbed in sys.modules: what is pinned here is which
revision gets probed, converted and cached, not the conversion itself.
"""

import asyncio
import os
import sys
import types

from faster_whisper_backend.transcription import models as tx_models


def test_pinned_revision_is_converted_not_answered_by_the_head_cache(
        monkeypatch, tmp_path):
    """A REVISION pin on an HF transformers repo must reach the file-list
    probe and the converter, and must not be served the HEAD conversion an
    earlier, unpinned load left under the same id. The converter is handed a
    local snapshot of the pinned revision: it forwards its own `revision` to
    the weights only, so config, tokenizer and preprocessor files would
    otherwise come from HEAD."""
    monkeypatch.setattr(tx_models.cfg, "AUTO_CONVERT_HF_MODELS", True,
                        raising=False)
    monkeypatch.setattr(tx_models.cfg, "CONVERT_QUANTIZATION", "float16",
                        raising=False)
    monkeypatch.setattr(tx_models.cfg, "CONVERTED_MODELS_DIR", str(tmp_path),
                        raising=False)
    monkeypatch.setattr(tx_models.cfg, "MODEL_OVERRIDES",
                        {"org/ft": {"REVISION": "v2"}}, raising=False)
    monkeypatch.setattr(tx_models, "_convert_locks", {})
    head = tmp_path / "org__ft" / "float16"
    head.mkdir(parents=True)
    (head / "model.bin").write_bytes(b"HEAD")

    probed, snapshots, converted = [], [], []

    def _list_repo_files(repo_id, revision=None):
        probed.append(revision)
        return ["config.json", "model.safetensors"]

    def _snapshot_download(repo_id, revision=None, **kw):
        snapshots.append((repo_id, revision))
        snap = tmp_path / "snapshots" / (revision or "HEAD")
        snap.mkdir(parents=True)
        return str(snap)

    class _Converter:
        def __init__(self, model_name_or_path, revision=None, **kw):
            converted.append((model_name_or_path, revision))
            self._src = model_name_or_path

        def convert(self, out, quantization, force):
            os.makedirs(out)
            with open(os.path.join(out, "model.bin"), "w") as f:
                f.write(os.path.basename(self._src))

    class _FileLock:
        def __init__(self, path, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    hub = types.ModuleType("huggingface_hub")
    hub.list_repo_files = _list_repo_files
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

    out = asyncio.run(tx_models._ensure_ct2_model("org/ft"))

    assert probed == ["v2"]
    assert snapshots == [("org/ft", "v2")]
    assert converted == [(str(tmp_path / "snapshots" / "v2"), None)]
    with open(os.path.join(out, "model.bin")) as f:
        assert f.read() == "v2"
    assert (head / "model.bin").read_bytes() == b"HEAD"


def test_local_dir_is_reconverted_once_its_weights_change(
        monkeypatch, tmp_path):
    """A local HF-format dir has no revision to key a fresh cache dir on: a
    retrain into the same dir must be reconverted, not answered by the
    model.bin of the previous fine-tune. An unchanged dir keeps the cache."""
    monkeypatch.setattr(tx_models.cfg, "AUTO_CONVERT_HF_MODELS", True,
                        raising=False)
    monkeypatch.setattr(tx_models.cfg, "CONVERT_QUANTIZATION", "float16",
                        raising=False)
    monkeypatch.setattr(tx_models.cfg, "CONVERTED_MODELS_DIR",
                        str(tmp_path / "ct2"), raising=False)
    monkeypatch.setattr(tx_models.cfg, "MODEL_OVERRIDES", {}, raising=False)
    monkeypatch.setattr(tx_models, "_convert_locks", {})
    src = tmp_path / "my-ft"
    src.mkdir()
    (src / "config.json").write_text("{}")
    weights = src / "model.safetensors"
    weights.write_bytes(b"old")
    out_dir = tx_models._converted_dir_for(str(src), "float16")
    os.makedirs(out_dir)
    cached = os.path.join(out_dir, "model.bin")
    with open(cached, "w") as f:
        f.write("old")
    os.utime(src / "config.json", (1_000, 1_000))
    os.utime(weights, (1_000, 1_000))
    os.utime(cached, (2_000, 2_000))

    converted = []

    class _Converter:
        def __init__(self, model_name_or_path, **kw):
            converted.append(model_name_or_path)
            self._src = model_name_or_path

        def convert(self, out, quantization, force):
            os.makedirs(out)
            with open(os.path.join(out, "model.bin"), "w") as f:
                f.write(open(os.path.join(self._src,
                                          "model.safetensors")).read())

    class _FileLock:
        def __init__(self, path, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fl = types.ModuleType("filelock")
    fl.FileLock = _FileLock
    fl.Timeout = type("Timeout", (Exception,), {})
    conv = types.ModuleType("ctranslate2.converters")
    conv.TransformersConverter = _Converter
    monkeypatch.setitem(sys.modules, "filelock", fl)
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", conv)
    for mod in ("ctranslate2", "transformers", "torch"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))

    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert converted == []

    weights.write_bytes(b"new")
    os.utime(weights, (3_000, 3_000))
    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert converted == [str(src)]
    with open(cached) as f:
        assert f.read() == "new"
