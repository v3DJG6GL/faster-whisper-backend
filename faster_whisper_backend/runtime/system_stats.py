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
import sys
import time
from typing import Any, Callable

from faster_whisper_backend.runtime import model_registry

NVML_OK = False
NVML_ERR: str | None = None
_nvml_handle: Any = None
# NVML index → device handle, bound lazily: a per-model DEVICE_INDEX can name
# any card, not just the first. Handles are process-constant once bound.
_handles: dict[int, Any] = {}


def _nvml_index(cuda_index: int) -> int:
    """The NVML index of CUDA ordinal `cuda_index`. NVML enumerates every
    GPU and ignores CUDA_VISIBLE_DEVICES, so ordinal N is the N-th entry of
    that list when it is set. A UUID / MIG entry, or an unset or empty
    list, keeps the raw ordinal. Exact only when CUDA's own order matches
    NVML's (CUDA_DEVICE_ORDER=PCI_BUS_ID, or identical cards)."""
    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cvd:
        parts = [p.strip() for p in cvd.split(",")]
        if 0 <= cuda_index < len(parts) and parts[cuda_index].isdigit():
            return int(parts[cuda_index])
    return cuda_index


def _handle(index: int = 0) -> Any:
    """NVML handle of CUDA ordinal `index` (see _nvml_index). Raises like
    the NVML call it wraps; callers already guard every NVML read."""
    nv = _nvml_index(int(index or 0))
    h = _handles.get(nv)
    if h is None:
        h = _handles[nv] = pynvml.nvmlDeviceGetHandleByIndex(nv)
    return h


try:
    import pynvml  # type: ignore[import-not-found]
    pynvml.nvmlInit()
    try:
        # The /stats panel describes the card CUDA calls 0.
        _nvml_handle = _handle(0)
    except Exception:                   # a CUDA_VISIBLE_DEVICES NVML lacks
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception as e:                  # ImportError, NVMLError, ...
    NVML_ERR = f"{type(e).__name__}: {e}"

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


def gpu_mem_used_bytes(index: int = 0) -> int | None:
    """Return current global VRAM used (in bytes) on CUDA device `index`
    (a model's DEVICE_INDEX), or None if NVML unavailable."""
    return _safe(lambda: int(pynvml.nvmlDeviceGetMemoryInfo(_handle(index)).used))


def gpu_mem_free_bytes(index: int = 0) -> int | None:
    """Return currently free VRAM (in bytes) on CUDA device `index`, or None
    if NVML unavailable.

    `.free` is the DRIVER's global view, which is exactly why it is the right
    number for a pre-load fit check: it accounts for every other process on the
    machine (a second worker, a game, the desktop compositor), not just the
    models runtime.model_registry knows about."""
    return _safe(lambda: int(pynvml.nvmlDeviceGetMemoryInfo(_handle(index)).free))


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
    name = gpu_name()
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


_CGROUP_ROOT = "/sys/fs/cgroup"


def _read_text(path: str) -> str:
    with open(path, encoding="ascii") as f:
        return f.read()


def _stat_field(text: str, key: str) -> int:
    """`key`'s value in a cgroup memory.stat; ValueError when it is absent."""
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        if name == key:
            return int(value)
    raise ValueError(f"memory.stat has no {key}")


def _cgroup_headroom(host_total: "int | None") -> "int | None":
    """Bytes this process's memory cgroup — its own or any ancestor's limit,
    whichever is tightest — still allows before the kernel OOM-kills inside
    it: the limit minus what the group holds, its reclaimable inactive page
    cache excluded. None when no limit applies or the files cannot be read.
    cgroup v2 first (memory.max, "max" = no limit); v1 (memory controller)
    only for a limit below the host's RAM, v1's "no limit" being a huge
    number."""
    try:
        groups = [line.split(":", 2)
                  for line in _read_text("/proc/self/cgroup").splitlines()]
        v2 = next((rel for hid, ctrls, rel in groups
                   if hid == "0" and not ctrls), None)
        v1 = next((rel for _hid, ctrls, rel in groups
                   if "memory" in ctrls.split(",")), None)
        for rel, base, files in (
                (v2, _CGROUP_ROOT, ("memory.max", "memory.current",
                                    "inactive_file")),
                (v1, _CGROUP_ROOT + "/memory", ("memory.limit_in_bytes",
                                                "memory.usage_in_bytes",
                                                "total_inactive_file"))):
            if rel is None:
                continue
            # The process's own group up to the root of the mount: inside a
            # container's cgroup namespace that is the container's group;
            # a group the mount does not show is skipped.
            parts = [p for p in rel.split("/") if p]
            found, heads = False, []
            for depth in range(len(parts), -1, -1):
                d = os.path.join(base, *parts[:depth])
                try:
                    limit = _read_text(os.path.join(d, files[0])).strip()
                except FileNotFoundError:
                    continue
                found = True
                if limit == "max" or (files[0] == "memory.limit_in_bytes"
                                      and (host_total is None
                                           or int(limit) >= host_total)):
                    continue
                used = (int(_read_text(os.path.join(d, files[1])))
                        - _stat_field(_read_text(os.path.join(d, "memory.stat")),
                                      files[2]))
                heads.append(max(0, int(limit) - used))
            if found:
                return min(heads) if heads else None
    except (OSError, ValueError):
        return None
    return None


def available_ram_bytes() -> int:
    """psutil's available RAM, capped by what this process's memory cgroup
    still allows (see _cgroup_headroom): psutil reads /proc/meminfo, which
    inside a container is the HOST's, so under `docker run --memory` it
    reports RAM the kernel would OOM-kill the container for using. Linux
    only; elsewhere, or with no limit set, psutil's figure as is. Raises
    what psutil raises."""
    vm = psutil.virtual_memory()
    free = int(vm.available)
    if not sys.platform.startswith("linux"):
        return free
    total = getattr(vm, "total", None)
    head = _cgroup_headroom(int(total) if total else None)
    return free if head is None else min(free, head)


def _build_host() -> dict[str, Any]:
    vmem = psutil.virtual_memory()
    # Disk free on the drive holding the hub cache (hf_cache.hub_lookup_dir(),
    # where model_sizes looks for GGUF / pyannote). Whisper and UVR live under
    # DOWNLOAD_ROOT beside it, so without a hub override it is their drive too.
    # Walk up to the nearest existing ancestor so a not-yet-created cache dir
    # still reads its drive.
    try:
        from faster_whisper_backend.runtime import hf_cache
        cache_dir = hf_cache.hub_lookup_dir()
    except Exception:  # noqa: BLE001 — stats only
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    # Absolute first: a relative DOWNLOAD_ROOT ("models") would otherwise
    # stop the walk at its own missing first component, never reaching cwd.
    cache_dir = os.path.abspath(cache_dir)
    while not os.path.exists(cache_dir) and \
            os.path.dirname(cache_dir) != cache_dir:
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

    No TTL cache: the /stats/stream SSE generator rebuilds every second,
    past any sub-second cache's life, and a shared cache would need an
    unsafe read-modify-write on a module global."""
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
        _handles.clear()
