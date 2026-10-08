"""Tests for system_stats: snapshot shape (including the registry's models
row), and idempotent shutdown. The registry itself: test_model_registry.py.

system_stats has import-time side effects (psutil priming, an NVML init
attempt that degrades gracefully). On this CI box NVML is absent, so the
snapshot's gpu is None and gpu_error is a non-empty string; assertions accept
either the absent-fallback (the real CI condition) or a present GPU.

The snapshot's models row reads model_registry's process-global registry; a
local autouse fixture clears it between tests so cases don't observe each
other's writes.
"""

import os

import pytest

from faster_whisper_backend.runtime import model_registry
from faster_whisper_backend.runtime import system_stats


@pytest.fixture(autouse=True)
def _reset_registry(model_sizes_ledger):
    """Reset model_registry's module-global loaded-model registry. A
    registration with a positive VRAM delta records into the measured-size
    ledger, so that is repointed at tmp_path too."""
    with model_registry._loaded_models_lock:
        model_registry._loaded_models.clear()
    yield
    with model_registry._loaded_models_lock:
        model_registry._loaded_models.clear()


# ---------------------------------------------------------------------------
# system_snapshot
# ---------------------------------------------------------------------------

def test_snapshot_shape():
    snap = system_stats.system_snapshot()
    assert set(snap.keys()) == {"gpu", "gpu_error", "host", "process", "models"}
    assert isinstance(snap["host"], dict)
    assert isinstance(snap["process"], dict)
    assert isinstance(snap["models"], list)
    # On this box NVML is absent: gpu None + non-empty error. Accept a present
    # GPU too (dict + error None) so the test isn't box-specific.
    if snap["gpu"] is None:
        assert isinstance(snap["gpu_error"], str) and snap["gpu_error"]
    else:
        assert isinstance(snap["gpu"], dict)
        assert snap["gpu_error"] is None


def test_snapshot_host_fields():
    host = system_stats.system_snapshot()["host"]
    for key in ("cpu_pct", "cpu_per_core", "ram_used_mb", "ram_total_mb",
                "ram_pct"):
        assert key in host
    assert "disk_free_gb" in host
    assert isinstance(host["cpu_per_core"], list)
    assert host["ram_total_mb"] > 0


def test_snapshot_process_fields():
    import os
    proc = system_stats.system_snapshot()["process"]
    assert proc["pid"] == os.getpid()
    assert proc["uptime_sec"] >= 0
    for key in ("rss_mb", "cpu_pct", "threads"):
        assert key in proc


def test_snapshot_models_reflects_registry(model_sizes_ledger):
    model_registry.register_loaded_model("base", 1024 * 1024, "cuda", "int8")
    models = system_stats.system_snapshot()["models"]
    assert len(models) == 1
    assert models[0]["name"] == "base"
    # The measurement went to the repointed ledger, not the session default.
    assert "base|cuda|int8" in open(model_sizes_ledger, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# gpu_mem_used_bytes
# ---------------------------------------------------------------------------

def test_gpu_mem_used_bytes():
    val = system_stats.gpu_mem_used_bytes()
    # NVML absent here -> None; if present, a non-negative int.
    if system_stats.NVML_OK:
        assert val is None or val >= 0
    else:
        assert val is None


class _FakeNvml:
    """Two cards: NVML index 0 is busy, index 1 is empty."""
    class _Mem:
        def __init__(self, used, free):
            self.used, self.free = used, free

    def nvmlDeviceGetHandleByIndex(self, i):
        if i not in (0, 1):
            raise RuntimeError("no such device")
        return f"h{i}"

    def nvmlDeviceGetMemoryInfo(self, h):
        return {"h0": self._Mem(7000, 1000), "h1": self._Mem(100, 9000)}[h]

    shutdowns = 0

    def nvmlShutdown(self):
        self.shutdowns += 1


@pytest.fixture
def two_gpus(monkeypatch):
    monkeypatch.setattr(system_stats, "pynvml", _FakeNvml(), raising=False)
    monkeypatch.setattr(system_stats, "NVML_OK", True)
    monkeypatch.setattr(system_stats, "_handles", {})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_vram_reads_follow_the_device_index(two_gpus, monkeypatch):
    """A per-model DEVICE_INDEX=1 loads onto the second card; its VRAM
    delta and fit check must read that card, not GPU 0."""
    assert system_stats.gpu_mem_used_bytes() == 7000
    assert system_stats.gpu_mem_free_bytes() == 1000
    assert system_stats.gpu_mem_used_bytes(1) == 100
    assert system_stats.gpu_mem_free_bytes(1) == 9000
    # NVML ignores CUDA_VISIBLE_DEVICES: CUDA ordinal 0 is NVML card 1 here.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setattr(system_stats, "_handles", {})
    assert system_stats.gpu_mem_free_bytes(0) == 9000
    assert system_stats.gpu_mem_free_bytes(1) == 1000
    # A UUID entry cannot be mapped: the raw ordinal stands.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    assert system_stats._nvml_index(0) == 0
    # A card NVML does not have degrades to "unknown", never a raise.
    assert system_stats.gpu_mem_free_bytes(5) is None


def test_fit_check_reads_the_target_card(two_gpus, model_sizes_ledger,
                                         monkeypatch):
    from faster_whisper_backend.runtime import model_sizes
    model_sizes.record("m", "cuda", "float16", 4000)
    assert model_sizes.fits("m", "cuda", "float16", reserve_bytes=0) == (
        False, "insufficient_vram")
    assert model_sizes.fits("m", "cuda", "float16", reserve_bytes=0,
                            device_index=1) == (True, None)


def test_preload_sizes_whisper_on_its_device_index(monkeypatch):
    from faster_whisper_backend.runtime import preload
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.setattr(cfg, "DEVICE_INDEX", 0, raising=False)
    monkeypatch.setattr(cfg, "MODEL_OVERRIDES",
                        {"tiny": {"DEVICE_INDEX": 1}}, raising=False)
    assert preload._device_index("whisper", "tiny") == 1
    assert preload._device_index("whisper", "large-v3") == 0
    assert preload._device_index("diarization", "p/x") == 0


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------

def test_shutdown_safe_and_idempotent(two_gpus):
    # Under the fake NVML, never the real one: shutdown flips the module
    # global NVML_OK, which would change every later VRAM read on a GPU box
    # (two_gpus' monkeypatch restores NVML_OK and _handles). It also makes
    # the NVML_OK branch run where no real NVML exists (CI).
    system_stats.gpu_mem_used_bytes(1)          # binds a handle
    assert system_stats._handles
    system_stats.shutdown()
    assert system_stats.NVML_OK is False
    assert system_stats._handles == {}
    system_stats.shutdown()  # no raise on second call
    assert system_stats.NVML_OK is False
    assert system_stats.pynvml.shutdowns == 1


# ---------------------------------------------------------------------------
# _build_host regressions
# ---------------------------------------------------------------------------

def test_disk_free_reads_the_download_root_drive(monkeypatch, tmp_path):
    """/stats labels it "disk free (model cache)": with HF_HOME unset the
    cache is <DOWNLOAD_ROOT>/hf/hub, not the OS drive's ~/.cache/huggingface —
    and a not-yet-created hf/hub (or hf/) dir still resolves to the nearest
    existing ancestor's drive."""
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(tmp_path), raising=False)
    seen = []
    real = system_stats.psutil.disk_usage

    def _spy(path):
        seen.append(path)
        return real(path)
    monkeypatch.setattr(system_stats.psutil, "disk_usage", _spy)
    host = system_stats._build_host()
    assert seen == [str(tmp_path)]           # hf/hub, hf/ absent -> tmp_path
    assert host["disk_free_gb"] is not None
    (tmp_path / "hf").mkdir()
    system_stats._build_host()
    assert seen[-1] == str(tmp_path / "hf")  # hf/hub absent -> hf
    # A set HF_HOME wins over DOWNLOAD_ROOT (which stays set): a distinct
    # dir, or both precedences would land on the same path.
    (tmp_path / "other").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "other"))
    system_stats._build_host()
    assert seen[-1] == str(tmp_path / "other")


def test_disk_free_walks_a_relative_download_root_up_to_cwd(monkeypatch,
                                                            tmp_path):
    """A relative DOWNLOAD_ROOT whose dir does not exist yet still reads the
    drive it will land on (the cwd's), not None."""
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "models", raising=False)
    seen = []
    real = system_stats.psutil.disk_usage

    def _spy(path):
        seen.append(path)
        return real(path)
    monkeypatch.setattr(system_stats.psutil, "disk_usage", _spy)
    host = system_stats._build_host()
    assert seen == [os.getcwd()]
    assert host["disk_free_gb"] is not None
