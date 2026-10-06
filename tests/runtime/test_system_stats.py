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

import pytest

from faster_whisper_backend.runtime import model_registry
from faster_whisper_backend.runtime import system_stats


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset model_registry's module-global loaded-model registry."""
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


def test_snapshot_models_reflects_registry():
    model_registry.register_loaded_model("base", 1024 * 1024, "cpu", "int8")
    models = system_stats.system_snapshot()["models"]
    assert len(models) == 1
    assert models[0]["name"] == "base"


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


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------

def test_shutdown_safe_and_idempotent():
    # Safe whether NVML inited or not; safe to call twice. After shutdown
    # NVML_OK must be False.
    system_stats.shutdown()
    assert system_stats.NVML_OK is False
    system_stats.shutdown()  # no raise on second call
    assert system_stats.NVML_OK is False


# ---------------------------------------------------------------------------
# _build_host regressions
# ---------------------------------------------------------------------------

def test_disk_free_reads_the_download_root_drive(monkeypatch, tmp_path):
    """/stats labels it "disk free (model cache)": with HF_HOME unset the
    cache is <DOWNLOAD_ROOT>/hf, not the OS drive's ~/.cache/huggingface —
    and a not-yet-created hf/ dir still resolves to its parent's drive."""
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
    assert seen == [str(tmp_path)]           # hf/ absent -> its parent
    assert host["disk_free_gb"] is not None
    (tmp_path / "hf").mkdir()
    system_stats._build_host()
    assert seen[-1] == str(tmp_path / "hf")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "", raising=False)
    system_stats._build_host()
    assert seen[-1] == str(tmp_path / "hf")  # HF_HOME wins
