"""
Loaded-model registry: what every model cache has resident right now, for
the /stats dashboard, the request receipts and the preloader.

All four families (whisper, pyannote, UVR, GGUF) register here after a load
and unregister on eviction; the name carries the family prefix
(``pyannote:``, ``uvr:``, ``gguf:``; bare id = whisper). The warm-lease
predicate lives here too, so the idle evictors can ask runtime.preload about
warmth without importing it.

Per-model VRAM accounting: an NVML delta sample taken under the whisper
cache's _model_load_lock around WhisperModel(...) construction (the other
families take the same delta around their own loads). We have to do it this
way because per-PID VRAM via nvmlDeviceGetComputeRunningProcesses
returns NVML_VALUE_NOT_AVAILABLE on Windows WDDM (the default driver mode for
consumer cards with a display attached). Documented at:
  https://forums.developer.nvidia.com/t/nvml-problems-for-windows-not-available-in-wddm-driver-model/77557
This is the same constraint DeepSpeed and vLLM hit; the workaround pattern
(delta around construction, serialize loads under a lock) is theirs too.

The CTranslate2 caching allocator can make subsequent loads of the same
size under-report VRAM (cached freed memory gets reused). We don't try to
fight this — the first load's number is the trustworthy one; later
re-reports get whatever the delta showed.

The GPU / host / process snapshot (NVML, psutil) is runtime.system_stats.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Callable


# --- Per-model VRAM tracking -------------------------------------------------
# Populated by each model cache around its load. Removed on LRU eviction.
# NOT a measurement loop — this is a registry of what the delta sample said at
# construction time.
_loaded_models_lock = Lock()
_loaded_models: dict[str, dict[str, Any]] = {}


def register_loaded_model(name: str, vram_bytes: int | None,
                          device: str, compute_type: str,
                          load_secs: float | None = None) -> None:
    """Called from transcription.models._get_or_load_model after a successful load. The VRAM
    delta sample comes from the caller — see transcription.models for the
    before/after dance under _model_load_lock.

    `load_secs` is how long the load itself took. Every family already
    measures this for its "loaded on %s in %.1fs" line; recording it here
    is what lets a request receipt split a stage's wall time into load vs
    run, so a cold start reads as a cost rather than as an unexplained gap
    between two timestamps. Optional — an omitting caller just gets no
    split."""
    # A NEGATIVE delta (a concurrent free elsewhere on the GPU during the
    # before/after window) is not a measurement: never store or display it.
    if vram_bytes is not None and vram_bytes < 0:
        vram_bytes = None
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
    # (whisper's cuda->cpu fallback in transcription.models._get_or_load_model passes the real
    # one), so the ledger inherits that correctness for free. Imported lazily:
    # model_sizes imports system_stats, which imports this module. Never fatal
    # to a load.
    # `if vram_bytes:` used to guard this, which quietly excluded every CPU
    # load and every load whose NVML delta came back 0 or None — those models
    # then had no ledger row, so preload could not size them, so it refused
    # to load them, so they were never measured. Fall back to the on-disk
    # footprint instead: a rough number that lets an admission decision be
    # made beats no row at all, and a real measurement supersedes it (record
    # replaces a disk-sourced row outright — the disk walk can over-count).
    try:
        from faster_whisper_backend.runtime import model_sizes
        # A NEGATIVE delta (a concurrent free elsewhere on the GPU) is not a
        # measurement either: record() would refuse it and leave no row.
        measured = bool(vram_bytes and vram_bytes > 0)
        size = (vram_bytes if measured else None) or model_sizes.disk_size(name)
        if size:
            model_sizes.record(name, device, compute_type, size,
                               measured=measured)
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
    """Called from transcription.models._get_or_load_model when LRU eviction happens."""
    with _loaded_models_lock:
        _loaded_models.pop(name, None)


# --- Warm-lease predicate (dependency inversion for preload.py) --------------
# preload.py holds "warm leases": a model that some live plan expects to use
# soon and that the idle evictors must therefore leave alone. All four evictors
# live in modules preload itself imports (transcription.models, diarization,
# bgm_separation, translation), so asking preload directly would close an
# import cycle in every one of them. They all already import THIS module, so
# the predicate is registered here instead and the dependency points the safe
# way.
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


def _reset_for_tests() -> None:
    """Test-only: clear the warm-lease predicate (one left installed would
    keep a later eviction test's model pinned by a plan an earlier test
    owned), empty the registry stage tests register stubs in, and rebind its
    lock."""
    global _loaded_models_lock
    set_warm_predicate(None)
    _loaded_models.clear()
    _loaded_models_lock = Lock()
