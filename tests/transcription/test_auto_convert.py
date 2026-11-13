"""AUTO_CONVERT_HF_MODELS: the CT2 conversion of a transformers checkpoint.

The conversion stack (huggingface_hub, filelock, ctranslate2's converter,
transformers, torch) is stubbed in sys.modules: what is pinned here is which
revision gets probed, converted and cached, not the conversion itself.
"""

import asyncio
import os
from pathlib import Path

from faster_whisper_backend.transcription import models as tx_models
from tests.transcription._conversion_helpers import stub_conversion_stack, stub_hub


def test_pinned_revision_is_converted_not_answered_by_the_head_cache(
        monkeypatch, tmp_path):
    """A REVISION pin on an HF transformers repo must reach the file-list
    probe and the converter, and must not be served the HEAD conversion an
    earlier, unpinned load left under the same id. The converter is handed a
    local snapshot of the pinned revision: it forwards its own `revision` to
    the weights only, so config, tokenizer and preprocessor files would
    otherwise come from HEAD."""
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
            Path(out, "model.bin").write_text(os.path.basename(self._src))

    stub_hub(monkeypatch, list_repo_files=_list_repo_files,
             snapshot_download=_snapshot_download)
    stub_conversion_stack(monkeypatch, _Converter, converted_dir=tmp_path,
                          overrides={"org/ft": {"REVISION": "v2"}})

    out = asyncio.run(tx_models._ensure_ct2_model("org/ft"))

    assert probed == ["v2"]
    assert snapshots == [("org/ft", "v2")]
    assert converted == [(str(tmp_path / "snapshots" / "v2"), None)]
    assert Path(out, "model.bin").read_text() == "v2"
    assert (head / "model.bin").read_bytes() == b"HEAD"


def test_local_dir_is_reconverted_once_its_weights_change(
        monkeypatch, tmp_path):
    """A local HF-format dir has no revision to key a fresh cache dir on: a
    retrain into the same dir must be reconverted, not answered by the
    model.bin of the previous fine-tune. An unchanged dir keeps the cache."""
    converted = _stub_local_conversion(monkeypatch, tmp_path)
    src, weights = _local_ft(tmp_path, b"old")
    out_dir = tx_models._converted_dir_for(str(src), "float16")
    os.makedirs(out_dir)
    cached = os.path.join(out_dir, "model.bin")
    Path(cached).write_text("old")
    os.utime(src / "config.json", (1_000, 1_000))
    os.utime(weights, (1_000, 1_000))
    os.utime(cached, (2_000, 2_000))

    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert converted == []

    weights.write_bytes(b"new")
    os.utime(weights, (3_000, 3_000))
    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert converted == [str(src)]
    assert Path(cached).read_text() == "new"


def _stub_local_conversion(monkeypatch, tmp_path, during_convert=None):
    """Stub the converter + filelock for a local-dir auto-convert; returns
    the list of sources converted. `during_convert` runs after the converter
    has read the weights, before model.bin is written."""
    converted = []

    class _Converter:
        def __init__(self, model_name_or_path, **kw):
            converted.append(model_name_or_path)
            self._src = model_name_or_path

        def convert(self, out, quantization, force):
            data = open(os.path.join(self._src, "model.safetensors")).read()
            if during_convert is not None:
                during_convert()
            os.makedirs(out)
            Path(out, "model.bin").write_text(data)

    stub_conversion_stack(monkeypatch, _Converter,
                          converted_dir=tmp_path / "ct2", overrides={})
    return converted


def _local_ft(tmp_path, data):
    """A local HF-format fine-tune dir (config.json + model.safetensors =
    `data`); returns (src, weights)."""
    src = tmp_path / "my-ft"
    src.mkdir()
    (src / "config.json").write_text("{}")
    weights = src / "model.safetensors"
    weights.write_bytes(data)
    return src, weights


def test_local_dir_swapped_in_with_older_mtimes_is_reconverted(
        monkeypatch, tmp_path):
    """rsync -a / tar keep the source mtime: new weights older than the
    conversion's model.bin are still a change, not the cached fine-tune."""
    converted = _stub_local_conversion(monkeypatch, tmp_path)
    src, weights = _local_ft(tmp_path, b"v1")
    out_dir = tx_models._converted_dir_for(str(src), "float16")
    cached = os.path.join(out_dir, "model.bin")

    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert converted == [str(src)]

    os.utime(src / "config.json", (1_000, 1_000))
    os.utime(cached, (2_000, 2_000))
    weights.write_bytes(b"v2")
    os.utime(weights, (1_500, 1_500))
    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert converted == [str(src), str(src)]
    assert Path(cached).read_text() == "v2"


def test_local_dir_written_during_conversion_is_reconverted(
        monkeypatch, tmp_path):
    """Weights saved while a conversion runs are older than the model.bin it
    writes at the end; the conversion of the earlier weights must not count
    as current."""
    src, weights = _local_ft(tmp_path, b"mid")
    os.utime(weights, (1_000, 1_000))
    first = []

    def _final_save_lands():
        if not first:
            first.append(True)
            weights.write_bytes(b"final")
            os.utime(weights, (1_100, 1_100))

    converted = _stub_local_conversion(monkeypatch, tmp_path,
                                       during_convert=_final_save_lands)
    out_dir = tx_models._converted_dir_for(str(src), "float16")

    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert converted == [str(src), str(src)]
    assert Path(out_dir, "model.bin").read_text() == "final"


def test_converted_models_dir_tilde_is_expanded(monkeypatch, tmp_path):
    """A "~/..." CONVERTED_MODELS_DIR (the form its own default is written in;
    dotenv does not expand it) resolves under the home dir like DOWNLOAD_ROOT,
    not to a relative literal "~" dir under the cwd."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # expanduser on Windows
    monkeypatch.setattr(tx_models.cfg, "CONVERTED_MODELS_DIR", "~/models/ct2",
                        raising=False)
    out = tx_models._converted_dir_for("org/ft", "float16")
    # normpath: on Windows expanduser("~/x") keeps the "/".
    assert os.path.normpath(out) == os.path.join(
        str(tmp_path), "models", "ct2", "org__ft", "float16")


def test_concurrent_misses_share_one_conversion_probe(monkeypatch, tmp_path):
    """Every cold miss of one model used to run its own Hub file-list probe,
    each holding a default-executor worker for as long as the Hub takes to
    answer. Concurrent misses await one probe; a dropped caller does not
    cancel it for the rest, and a later miss probes afresh."""
    import threading

    stub_conversion_stack(monkeypatch, None, converted_dir=tmp_path,
                          overrides={"org/pinned": {"REVISION": "v2"}})
    calls, release = [], threading.Event()

    def _probe(model_id, revision=None):
        calls.append((model_id, revision))
        release.wait(5)
        return None     # an already-CT2 repo: nothing to convert

    monkeypatch.setattr(tx_models, "_conversion_repo_files", _probe)

    async def _misses():
        same = [asyncio.create_task(tx_models._ensure_ct2_model("org/ct2"))
                for _ in range(8)]
        other = asyncio.create_task(tx_models._ensure_ct2_model("org/pinned"))
        while len(calls) < 2:
            await asyncio.sleep(0.005)
        same[0].cancel()
        await asyncio.sleep(0)
        release.set()
        done = await asyncio.gather(*same, other, return_exceptions=True)
        assert tx_models._convert_probes == {}
        return done

    done = asyncio.run(_misses())
    assert isinstance(done[0], asyncio.CancelledError)
    assert done[1:] == ["org/ct2"] * 7 + ["org/pinned"]
    assert sorted(calls) == [("org/ct2", None), ("org/pinned", "v2")]

    assert asyncio.run(tx_models._ensure_ct2_model("org/ct2")) == "org/ct2"
    assert calls.count(("org/ct2", None)) == 2


def _recording_hub_conversion(monkeypatch, tmp_path, overrides=None):
    """A Hub whose every repo is a transformers checkpoint, plus a converter:
    returns (probed repo ids, converted sources)."""
    probed, converted = [], []

    def _list_repo_files(repo_id, revision=None):
        probed.append(repo_id)
        return ["config.json", "model.safetensors"]

    class _Converter:
        def __init__(self, model_name_or_path, **kw):
            converted.append(model_name_or_path)

        def convert(self, out, quantization, force):
            os.makedirs(out)
            Path(out, "model.bin").write_text("converted")

    stub_hub(monkeypatch, list_repo_files=_list_repo_files)
    stub_conversion_stack(monkeypatch, _Converter,
                          converted_dir=tmp_path / "ct2",
                          overrides=overrides or {})
    return probed, converted


def test_local_files_only_skips_the_hub_probe_and_conversion(
        monkeypatch, tmp_path):
    """LOCAL_FILES_ONLY promises no network: with auto-convert on too, a load
    miss of a Hub id must not ask the Hub for its file list, nor fetch and
    convert its weights. A conversion made earlier still answers."""
    probed, converted = _recording_hub_conversion(monkeypatch, tmp_path)
    monkeypatch.setattr(tx_models.cfg, "LOCAL_FILES_ONLY", True, raising=False)

    assert asyncio.run(tx_models._ensure_ct2_model("org/ft")) == "org/ft"
    assert (probed, converted) == ([], [])

    out_dir = tx_models._converted_dir_for("org/ft", "float16")
    os.makedirs(out_dir)
    Path(out_dir, "model.bin").write_text("earlier")
    assert asyncio.run(tx_models._ensure_ct2_model("org/ft")) == out_dir
    assert (probed, converted) == ([], [])


def test_local_files_only_per_model_override_skips_only_that_model(
        monkeypatch, tmp_path):
    probed, converted = _recording_hub_conversion(
        monkeypatch, tmp_path, overrides={"org/ft": {"LOCAL_FILES_ONLY": True}})
    monkeypatch.setattr(tx_models.cfg, "LOCAL_FILES_ONLY", False, raising=False)

    assert asyncio.run(tx_models._ensure_ct2_model("org/ft")) == "org/ft"
    assert (probed, converted) == ([], [])
    other = asyncio.run(tx_models._ensure_ct2_model("org/other"))
    assert other == tx_models._converted_dir_for("org/other", "float16")
    assert (probed, converted) == (["org/other"], ["org/other"])


def test_local_files_only_still_converts_a_local_dir(monkeypatch, tmp_path):
    """A local HF-format dir needs no network: it converts as before."""
    converted = _stub_local_conversion(monkeypatch, tmp_path)
    monkeypatch.setattr(tx_models.cfg, "LOCAL_FILES_ONLY", True, raising=False)
    src, _weights = _local_ft(tmp_path, b"w")
    out_dir = tx_models._converted_dir_for(str(src), "float16")

    assert asyncio.run(tx_models._ensure_ct2_model(str(src))) == out_dir
    assert converted == [str(src)]
    assert Path(out_dir, "model.bin").read_text() == "w"
