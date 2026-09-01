"""model_sizes: the persisted measured-size ledger and the fit check.

Everything runs against a real temp file — the atomic-write/lock path is the
part most likely to break, so stubbing it out would test nothing.
"""
import json
import os

import pytest

import model_sizes


GB = 1024 ** 3


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Repoint PATH *and* the path default ARG of _read/_write (bound at def
    time — the same trap conftest documents for config_store)."""
    p = str(tmp_path / "model_sizes.json")
    monkeypatch.setattr(model_sizes, "PATH", p, raising=False)
    for fn in (model_sizes._read, model_sizes._write,
               model_sizes._write_locked):
        defaults = list(fn.__defaults__ or ())
        defaults[-1] = p
        monkeypatch.setattr(fn, "__defaults__", tuple(defaults), raising=False)
    model_sizes._reset_for_tests()
    yield p
    model_sizes._reset_for_tests()


def test_record_estimate_roundtrip_survives_a_restart(ledger):
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB
    assert os.path.exists(ledger)

    # Simulate a fresh process: the module-level cache is empty, the file isn't.
    model_sizes._reset_for_tests()
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB


def test_smaller_resample_never_lowers_the_stored_size(ledger):
    model_sizes.record("large-v3", "cuda", "float16", 4 * GB)
    # CTranslate2's caching allocator makes a re-load under-report; the ledger
    # must keep the high-water mark or the fit check turns into an OOM.
    model_sizes.record("large-v3", "cuda", "float16", 1 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 4 * GB

    doc = json.loads(open(ledger, encoding="utf-8").read())
    assert doc["models"]["large-v3|cuda|float16"]["n"] == 2


def test_placement_is_part_of_the_key(ledger):
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    model_sizes.record("large-v3", "cpu", "int8", 1 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB
    assert model_sizes.estimate("large-v3", "cpu", "int8") == 1 * GB


def test_any_device_fallback_then_none(ledger):
    model_sizes.record("gguf:some/repo", "cpu", "int8", 2 * GB)
    assert model_sizes.estimate("gguf:some/repo", "cuda", "float16") == 2 * GB
    assert model_sizes.estimate("never-loaded", "cuda", "float16") is None


def test_one_percent_drift_does_not_rewrite_the_file(ledger, monkeypatch):
    model_sizes.record("large-v3", "cuda", "float16", 1000 * 1000 * 1000)

    writes = []
    real_write = model_sizes._write_locked
    monkeypatch.setattr(model_sizes, "_write_locked",
                        lambda *a, **k: (writes.append(1), real_write(*a, **k))[1])

    model_sizes.record("large-v3", "cuda", "float16", 1010 * 1000 * 1000)  # +1%
    assert writes == []
    model_sizes.record("large-v3", "cuda", "float16", 1200 * 1000 * 1000)  # +20%
    assert len(writes) == 1
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 1200 * 1000 * 1000


def test_fits_cuda_verdicts_and_the_reserve_boundary(ledger, monkeypatch):
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    reserve = 1 * GB

    def free(n):
        monkeypatch.setattr(model_sizes.system_stats, "gpu_mem_free_bytes",
                            lambda: n, raising=False)

    free(8 * GB)
    assert model_sizes.fits("large-v3", "cuda", "float16",
                            reserve_bytes=reserve) == (True, None)
    # free - need == reserve is still a fit; one byte less is not.
    free(4 * GB)
    assert model_sizes.fits("large-v3", "cuda", "float16",
                            reserve_bytes=reserve) == (True, None)
    free(4 * GB - 1)
    assert model_sizes.fits("large-v3", "cuda", "float16",
                            reserve_bytes=reserve) == (False, "insufficient_vram")


def test_fits_reports_unknowns_apart(ledger, monkeypatch):
    monkeypatch.setattr(model_sizes.system_stats, "gpu_mem_free_bytes",
                        lambda: 8 * GB, raising=False)
    # Never measured: "cannot say", NOT a refusal.
    assert model_sizes.fits("never-loaded", "cuda", "float16",
                            reserve_bytes=0) == (None, "size_unknown")

    # NVML absent: we refuse rather than guess on the GPU...
    monkeypatch.setattr(model_sizes.system_stats, "gpu_mem_free_bytes",
                        lambda: None, raising=False)
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    assert model_sizes.fits("large-v3", "cuda", "float16",
                            reserve_bytes=0) == (False, "vram_unknown")


def test_fits_cpu_uses_psutil_even_without_nvml(ledger, monkeypatch):
    monkeypatch.setattr(model_sizes.system_stats, "gpu_mem_free_bytes",
                        lambda: None, raising=False)
    model_sizes.record("large-v3", "cpu", "int8", 2 * GB)

    class _VM:
        available = 5 * GB
    monkeypatch.setattr(model_sizes.psutil, "virtual_memory", lambda: _VM())

    assert model_sizes.fits("large-v3", "cpu", "int8",
                            reserve_bytes=3 * GB) == (True, None)
    assert model_sizes.fits("large-v3", "cpu", "int8",
                            reserve_bytes=3 * GB + 1) == (False, "insufficient_ram")


@pytest.mark.parametrize("body", [
    "{not json at all",
    '{"version": 1, "models": {"a|cuda|f16": {"byt',   # truncated write
    '{"version": 99, "models": {"a|cuda|f16": {"bytes": 1}}}',
    '["not", "an", "object"]',
])
def test_unreadable_file_degrades_to_no_data(ledger, body):
    with open(ledger, "w", encoding="utf-8") as f:
        f.write(body)
    model_sizes._reset_for_tests()
    assert model_sizes.estimate("a", "cuda", "f16") is None
    assert model_sizes.fits("a", "cuda", "f16", reserve_bytes=0)[1] is not None


# ---------------------------------------------------------------------------
# Disk-size prior
#
# Without it, a model that has never been loaded cannot be sized, so preload
# refuses it, so it is never loaded, so it is never measured. On a fresh
# install that deadlock covers every model and fails completely silently.
# ---------------------------------------------------------------------------

def test_disk_size_reads_a_uvr_onnx_file(ledger, tmp_path, monkeypatch):
    import config as cfg
    root = tmp_path / "dl"
    (root / "audio-separator").mkdir(parents=True)
    blob = root / "audio-separator" / "UVR-MDX-NET-Inst_HQ_4.onnx"
    blob.write_bytes(b"x" * 4096)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(root), raising=False)

    assert model_sizes.disk_size("uvr:UVR-MDX-NET-Inst_HQ_4") == 4096
    # The `.onnx` suffix is implied by the friendly name, as elsewhere.
    assert model_sizes.disk_size("uvr:UVR-MDX-NET-Inst_HQ_4.onnx") == 4096


def test_disk_size_sums_a_hf_repo_dir(ledger, tmp_path, monkeypatch):
    hf = tmp_path / "hf"
    d = hf / "hub" / "models--tencent--HY-MT1.5-7B-GGUF" / "blobs"
    d.mkdir(parents=True)
    (d / "a").write_bytes(b"x" * 1000)
    (d / "b").write_bytes(b"x" * 2000)
    monkeypatch.setenv("HF_HOME", str(hf))

    assert model_sizes.disk_size("gguf:tencent/HY-MT1.5-7B-GGUF:Q4_K_M") == 3000


def test_disk_size_is_none_when_nothing_is_there(ledger, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nope"))
    assert model_sizes.disk_size("gguf:no/such") is None


def test_estimate_falls_back_to_disk_then_prefers_a_measurement(
        ledger, tmp_path, monkeypatch):
    hf = tmp_path / "hf"
    d = hf / "hub" / "models--openai--whisper-tiny"
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x" * 8192)
    monkeypatch.setenv("HF_HOME", str(hf))

    # Never measured → the prior stands in, and fits() can now decide.
    assert model_sizes.estimate("openai/whisper-tiny", "cuda", "float16") == 8192
    assert model_sizes.fits("openai/whisper-tiny", "cpu", "float16",
                            reserve_bytes=0)[1] != "size_unknown"

    # A real measurement supersedes it.
    model_sizes.record("openai/whisper-tiny", "cuda", "float16", 99 * 1024)
    assert model_sizes.estimate("openai/whisper-tiny", "cuda",
                                "float16") == 99 * 1024


def test_disk_size_uvr_falls_back_to_tempdir_without_download_root(
        ledger, tmp_path, monkeypatch):
    """bgm_separation loads from <tempdir>/audio-separator when DOWNLOAD_ROOT
    is unset — the sizing must look in the same place, not give up."""
    import config as cfg
    (tmp_path / "audio-separator").mkdir()
    (tmp_path / "audio-separator" / "Foo.onnx").write_bytes(b"x" * 2048)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "", raising=False)
    monkeypatch.setattr(model_sizes.tempfile, "gettempdir",
                        lambda: str(tmp_path))

    assert model_sizes.disk_size("uvr:Foo") == 2048


def test_measurement_replaces_a_larger_disk_prior(ledger):
    """The disk walk over-counts (every revision, fp32 blobs); the first REAL
    measurement must win even when it is smaller, or the prior is unbeatable
    and fits() refuses a model that would load fine."""
    model_sizes.record("large-v3", "cuda", "float16", 6 * GB, measured=False)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 6 * GB

    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB
    doc = json.loads(open(ledger, encoding="utf-8").read())
    assert doc["models"]["large-v3|cuda|float16"]["src"] == "measured"

    # ...and a later disk prior never drags a measured row anywhere.
    model_sizes.record("large-v3", "cuda", "float16", 9 * GB, measured=False)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB
    # Measured-vs-measured still keeps the high-water mark.
    model_sizes.record("large-v3", "cuda", "float16", 1 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB


def test_disk_size_defaults_to_the_hub_cache_without_any_root(
        ledger, tmp_path, monkeypatch):
    """Empty DOWNLOAD_ROOT + no HF_HOME is the documented default install
    (.env.example: 'empty = standard HF cache ~/.cache/huggingface'); the
    prior must look there instead of giving up."""
    import config as cfg
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".cache" / "huggingface" / "hub" / "models--org--repo"
    d.mkdir(parents=True)
    (d / "x.bin").write_bytes(b"x" * 1234)

    assert model_sizes.disk_size("gguf:org/repo") == 1234


def test_disk_size_finds_whisper_under_download_root(
        ledger, tmp_path, monkeypatch):
    """main passes DOWNLOAD_ROOT itself as snapshot_download's cache_dir and
    maps 'large-v3' through faster_whisper's _MODELS table — the prior must
    look where the bytes actually land."""
    pytest.importorskip("faster_whisper")
    import config as cfg
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(tmp_path), raising=False)
    d = tmp_path / "models--Systran--faster-whisper-large-v3"
    d.mkdir()
    (d / "model.bin").write_bytes(b"x" * 5000)

    assert model_sizes.disk_size("large-v3") == 5000
    assert model_sizes.disk_size("Systran/faster-whisper-large-v3") == 5000


def test_record_merges_a_peer_workers_row_instead_of_clobbering_it(ledger):
    """Two workers share the file. A peer's row written while this process
    holds a stale in-memory copy (same mtime — writes within one clock
    tick) must survive our next record(), which rewrites the whole
    document."""
    model_sizes.record("a", "cuda", "float16", 1 * GB)      # warms the cache
    doc = json.loads(open(ledger, encoding="utf-8").read())
    doc["models"]["peer|cpu|int8"] = {"bytes": 7, "ts": 0, "n": 1,
                                      "src": "measured"}
    with open(ledger, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.utime(ledger, (model_sizes._cache_mtime, model_sizes._cache_mtime))
    assert model_sizes._read() is model_sizes._cache      # cache looks fresh

    model_sizes.record("b", "cuda", "float16", 2 * GB)
    models = json.loads(open(ledger, encoding="utf-8").read())["models"]
    assert set(models) >= {"a|cuda|float16", "peer|cpu|int8", "b|cuda|float16"}


def test_record_holds_the_cross_process_save_lock(ledger, monkeypatch):
    import config_store
    entered = []
    real = config_store._save_lock

    def spy(path):
        entered.append(path)
        return real(path)
    monkeypatch.setattr(config_store, "_save_lock", spy)
    model_sizes.record("a", "cuda", "float16", 1 * GB)
    # Once around the whole read-modify-write — never re-acquired inside
    # the write (the per-path lock is not reentrant).
    assert entered == [ledger]
