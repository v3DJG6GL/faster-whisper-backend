"""
System / GPU snapshot for the /stats dashboard.

Two libraries: nvidia-ml-py (NVML, optional) and psutil. Both are imported
defensively — on a host without an NVIDIA GPU or without the package, the
GPU panel disappears from the UI and the rest still works.

Per-model VRAM accounting: an NVML delta sample taken under main.py's
existing _model_load_lock around WhisperModel(...) construction. We have to
do it this way because per-PID VRAM via nvmlDeviceGetComputeRunningProcesses
returns NVML_VALUE_NOT_AVAILABLE on Windows WDDM (the default driver mode for
consumer cards with a display attached). Documented at:
  https://forums.developer.nvidia.com/t/nvml-problems-for-windows-not-available-in-wddm-driver-model/77557
This is the same constraint DeepSpeed and vLLM hit; the workaround pattern
(delta around construction, serialize loads under a lock) is theirs too.

The CTranslate2 caching allocator can make subsequent loads of the same
size under-report VRAM (cached freed memory gets reused). We don't try to
fight this — the first load's number is the trustworthy one; later
re-reports get whatever the delta showed.
"""

from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any, Callable

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


# --- Per-model VRAM tracking -------------------------------------------------
# Populated by main.py around the WhisperModel(...) construction. Removed on
# LRU eviction. NOT a measurement loop — this is a registry of what the
# delta sample said at construction time.
_loaded_models_lock = Lock()
_loaded_models: dict[str, dict[str, Any]] = {}


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


def register_loaded_model(name: str, vram_bytes: int | None,
                          device: str, compute_type: str,
                          load_secs: float | None = None) -> None:
    """Called from main._get_or_load_model after a successful load. The VRAM
    delta sample comes from the caller — see main.py for the before/after
    dance under _model_load_lock.

    `load_secs` is how long the load itself took. Every family already
    measures this for its "loaded on %s in %.1fs" line; recording it here
    is what lets a request receipt split a stage's wall time into load vs
    run, so a cold start reads as a cost rather than as an unexplained gap
    between two timestamps. Optional — an omitting caller just gets no
    split."""
    with _loaded_models_lock:
        _loaded_models[name] = {
            "name": name,
            "device": device,
            "compute_type": compute_type,
            "vram_bytes": vram_bytes,
            "load_secs": load_secs,
            "loaded_at": time.time(),
            "last_used": time.time(),
            # Monotonic counterpart for the idle-evictor's safe time math
            # (wall-clock can jump on NTP correction; monotonic cannot).
            "last_used_monotonic": time.monotonic(),
        }
    # Persist the measurement so a fresh process can size this model BEFORE
    # loading it. All four families (whisper, pyannote, UVR, GGUF) come through
    # here, so this one hook covers the lot; `device` is the ACTUAL placement
    # (whisper's cuda->cpu fallback in main._get_or_load_model passes the real
    # one), so the ledger inherits that correctness for free. Imported lazily:
    # model_sizes imports this module. Never fatal to a load.
    # `if vram_bytes:` used to guard this, which quietly excluded every CPU
    # load and every load whose NVML delta came back 0 or None — those models
    # then had no ledger row, so preload could not size them, so it refused
    # to load them, so they were never measured. Fall back to the on-disk
    # footprint instead: a rough number that lets an admission decision be
    # made beats no row at all, and a real measurement supersedes it (record
    # replaces a disk-sourced row outright — the disk walk can over-count).
    try:
        from faster_whisper_backend.runtime import model_sizes
        size = vram_bytes or model_sizes.disk_size(name)
        if size:
            model_sizes.record(name, device, compute_type, size,
                               measured=bool(vram_bytes))
    except Exception:
        pass


def load_secs_since(name: str, since_ts: float) -> float:
    """How much of a stage's wall time went into loading this model.

    Returns the recorded load duration when the model was (re)loaded at or
    after `since_ts` — i.e. inside the stage that is asking — and 0.0 when
    it was already resident. That 0.0 is the interesting answer: it is the
    receipt's own proof that preloading did its job."""
    with _loaded_models_lock:
        info = _loaded_models.get(name)
        if info is None:
            return 0.0
        if float(info.get("loaded_at") or 0.0) < since_ts:
            return 0.0
        return float(info.get("load_secs") or 0.0)


def touch_loaded_model(name: str) -> None:
    """Bump last_used timestamp on cache hit — drives the warm/cold UI badge
    and the idle-evictor's eviction decision."""
    with _loaded_models_lock:
        info = _loaded_models.get(name)
        if info is not None:
            info["last_used"] = time.time()
            info["last_used_monotonic"] = time.monotonic()


def unregister_loaded_model(name: str) -> None:
    """Called from main._get_or_load_model when LRU eviction happens."""
    with _loaded_models_lock:
        _loaded_models.pop(name, None)


# --- Warm-lease predicate (dependency inversion for preload.py) --------------
# preload.py holds "warm leases": a model that some live plan expects to use
# soon and that the idle evictors must therefore leave alone. All four evictors
# live in modules preload itself imports (main, diarization, bgm_separation,
# translation), so asking preload directly would close an import cycle in every
# one of them. They all already import THIS module, so the predicate is
# registered here instead and the dependency points the safe way.
_warm_predicate: "Callable[[str], bool] | None" = None


def set_warm_predicate(fn: "Callable[[str], bool] | None") -> None:
    """Install (or clear, with None) the warm-lease predicate. Called by
    preload.start(); cleared by preload._reset_for_tests()."""
    global _warm_predicate
    _warm_predicate = fn


def is_warm(name: str) -> bool:
    """True when a live preload plan holds a warm lease on this stats key.

    NEVER raises and defaults to False: an unregistered predicate (preload
    disabled, or a unit test importing only this module) and a predicate that
    throws must both degrade to "not warm" — the fail-safe direction, since the
    only consequence is that the idle evictor is free to reclaim the VRAM."""
    fn = _warm_predicate
    if fn is None:
        return False
    try:
        return bool(fn(name))
    except Exception:  # noqa: BLE001 — eviction must never break on this
        return False


# What a loaded model is for, told apart by the registration prefix each
# family uses (diarization._STATS_PREFIX, bgm_separation._STATS_PREFIX,
# translation._STATS_PREFIX); an unprefixed name is a faster-whisper decode
# model. The role names match the pipeline-stage vocabulary the stats page
# colours (transcribing / diarizing / separating / translating).
MODEL_ROLE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("pyannote:", "diarizing"),
    ("uvr:", "separating"),
    ("gguf:", "translating"),
)


def model_role(name: str) -> tuple[str, str]:
    """(role, display label) for a registry name: the prefix picks the role
    and is stripped from the label."""
    for prefix, role in MODEL_ROLE_PREFIXES:
        if name.startswith(prefix):
            return role, name[len(prefix):]
    return "transcribing", name


def loaded_models_snapshot() -> list[dict[str, Any]]:
    """Returned in /stats/snapshot. Sorted by load order (oldest first).
    Each entry carries `role` (transcribing / diarizing / separating /
    translating) and `label` (the name without the family prefix)."""
    with _loaded_models_lock:
        out = []
        now = time.time()
        for info in _loaded_models.values():
            mb = (info["vram_bytes"] / (1024 * 1024)) if info["vram_bytes"] is not None else None
            role, label = model_role(info["name"])
            out.append({
                "name": info["name"],
                "role": role,
                "label": label,
                "device": info["device"],
                "compute_type": info["compute_type"],
                "vram_mb": round(mb, 1) if mb is not None else None,
                "age_sec": round(now - info["loaded_at"], 1),
                "idle_sec": round(now - info["last_used"], 1),
            })
        return out


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
    # Disk free on the drive containing the model cache (HF default location).
    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
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
        "models": loaded_models_snapshot(),
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
