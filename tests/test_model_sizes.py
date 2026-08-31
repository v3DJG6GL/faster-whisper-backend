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
    for fn in (model_sizes._read, model_sizes._write):
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
    real_write = model_sizes._write
    monkeypatch.setattr(model_sizes, "_write",
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
