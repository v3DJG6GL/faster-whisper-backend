"""
System / GPU snapshot for the /stats dashboard.

Two libraries: nvidia-ml-py (NVML, optional) and psutil. Both are imported
defensively — on a host without an NVIDIA GPU or without the package, the
GPU panel disappears from the UI and the rest still works.

The loaded-model registry (and its per-model VRAM accounting, which samples
gpu_mem_used_bytes() around each load) is runtime.model_registry.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from faster_whisper_backend.runtime import model_registry

# --- NVML init (optional, defensive) -----------------------------------------
NVML_OK = False
NVML_ERR: str | None = None
_nvml_handle: Any = None
try:
    import pynvml  # type: ignore[import-not-found]
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception as e:                  # ImportError, NVMLError, ...
    NVML_ERR = f"{type(e).__name__}: {e}"

# --- psutil ------------------------------------------------------------------
import psutil

# Prime the non-blocking cpu_percent calls so the first /stats fetch returns
# real numbers (the documented psutil contract: first call returns 0.0).
psutil.cpu_percent(interval=None)
_proc = psutil.Process()
_proc.cpu_percent(interval=None)
_PROC_START_TS = _proc.create_time()


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Wrap NVML calls so a transient driver hiccup degrades to `default`
    instead of taking the whole /stats request down with a 500."""
    if not NVML_OK:
        return default
    try:
        return fn()
    except Exception:
        return default


def gpu_mem_used_bytes() -> int | None:
    """Return current global VRAM used (in bytes), or None if NVML unavailable."""
    if not NVML_OK:
        return None
    try:
        return int(pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used)
    except Exception:
        return None


def gpu_mem_free_bytes() -> int | None:
    """Return currently free VRAM (in bytes), or None if NVML unavailable.

    `.free` is the DRIVER's global view, which is exactly why it is the right
    number for a pre-load fit check: it accounts for every other process on the
    machine (a second worker, a game, the desktop compositor), not just the
    models we registered above."""
    if not NVML_OK:
        return None
    try:
        return int(pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).free)
    except Exception:
        return None


def gpu_name() -> str | None:
    """The GPU's marketing name ("NVIDIA GeForce RTX 3080"), or None when NVML
    found no device. Older pynvml returns bytes — normalized to str."""
    raw = _safe(lambda: pynvml.nvmlDeviceGetName(_nvml_handle))
    if raw is None:
        return None
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def _build_gpu() -> dict[str, Any] | None:
    if not NVML_OK:
        return None
    h = _nvml_handle
    name = _safe(lambda: pynvml.nvmlDeviceGetName(h))
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
    util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
    cuda_int = _safe(lambda: pynvml.nvmlSystemGetCudaDriverVersion_v2())
    cuda_str = (
        f"{cuda_int // 1000}.{(cuda_int % 1000) // 10}"
        if isinstance(cuda_int, int) else None
    )
    p_state = _safe(lambda: pynvml.nvmlDeviceGetPerformanceState(h))
    return {
        "name": name,
        "driver": _safe(lambda: pynvml.nvmlSystemGetDriverVersion()),
        "cuda": cuda_str,
        "mem_used_mb": round(mem.used / (1024 * 1024), 1) if mem else None,
        "mem_total_mb": round(mem.total / (1024 * 1024), 1) if mem else None,
        "util_pct": util.gpu if util else None,
        "temp_c": _safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
        "power_w": _safe(lambda: round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)),
        "power_limit_w": _safe(
            lambda: round(pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0, 1)
        ),
        "sm_clock_mhz": _safe(
            lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        ),
        "p_state": f"P{p_state}" if isinstance(p_state, int) else None,
    }


def _build_host() -> dict[str, Any]:
    vmem = psutil.virtual_memory()
    # Disk free on the drive containing the model cache — the same
    # precedence model_sizes._model_path resolves (HF_HOME, else
    # DOWNLOAD_ROOT/hf where every whisper / GGUF / UVR download actually
    # lands on bare metal, else the hub's default). Walk up to the nearest
    # existing ancestor so a not-yet-created cache dir still reads its drive.
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        try:
            from faster_whisper_backend.settings import config as _cfg
            root = (getattr(_cfg, "DOWNLOAD_ROOT", "") or "").strip()
        except Exception:  # noqa: BLE001 — stats only
            root = ""
        cache_dir = os.path.join(root, "hf") if root else \
            os.path.expanduser("~/.cache/huggingface")
    while not os.path.exists(cache_dir) and \
            os.path.dirname(cache_dir) not in ("", cache_dir):
        cache_dir = os.path.dirname(cache_dir)
    try:
        disk_free_gb = round(psutil.disk_usage(cache_dir).free / (1024 ** 3), 1)
    except (OSError, FileNotFoundError):
        disk_free_gb = None
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "cpu_per_core": psutil.cpu_percent(interval=None, percpu=True),
        "ram_used_mb": round(vmem.used / (1024 * 1024), 1),
        "ram_total_mb": round(vmem.total / (1024 * 1024), 1),
        "ram_pct": vmem.percent,
        "disk_free_gb": disk_free_gb,
    }


def _build_process() -> dict[str, Any]:
    try:
        rss_mb = round(_proc.memory_info().rss / (1024 * 1024), 1)
        cpu = _proc.cpu_percent(interval=None)
        threads = _proc.num_threads()
    except psutil.Error:
        rss_mb = cpu = threads = None  # type: ignore[assignment]
    return {
        "pid": os.getpid(),
        "rss_mb": rss_mb,
        "cpu_pct": cpu,
        "threads": threads,
        "uptime_sec": round(time.time() - _PROC_START_TS, 1),
    }


def system_snapshot() -> dict[str, Any]:
    """Build a snapshot of GPU + host + process + loaded models.

    No TTL cache: the dominant consumer is the /stats/stream SSE generator
    which rebuilds every second (longer than any sub-second cache could
    survive), so a shared cache only helped the rare multi-tab snapshot
    burst, at the cost of an unsafe RMW on a module global."""
    return {
        "gpu": _build_gpu(),
        "gpu_error": NVML_ERR if not NVML_OK else None,
        "host": _build_host(),
        "process": _build_process(),
        "models": model_registry.loaded_models_snapshot(),
    }


def shutdown() -> None:
    """Best-effort NVML shutdown — call from FastAPI's lifespan exit handler.

    Safe to call when NVML didn't init; safe to call twice."""
    global NVML_OK
    if NVML_OK:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        NVML_OK = False
