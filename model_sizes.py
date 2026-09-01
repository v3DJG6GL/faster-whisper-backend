"""Persisted ledger of MEASURED model sizes, plus a free-memory fit check.

Why a file at all: system_stats.register_loaded_model() already measures every
model's footprint as an NVML delta at load time, but that number lives in a
process-local dict and dies with the worker. To decide whether a model can be
preloaded we need its size BEFORE loading it — i.e. carried across restarts.

Why the fit check reads the DRIVER's free VRAM rather than summing our own
registry: other processes on the machine (a second worker, a game, a desktop
compositor) consume the same card, and our bookkeeping cannot see them.

Import-light on purpose: it is reached from system_stats, which main imports
very early. config_store (pydantic, ~3 k lines) is imported LAZILY inside the
write path only.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time

# Both are hard dependencies already imported unguarded by system_stats (psutil
# is in requirements.txt); NVML is the optional one, and system_stats hides that
# behind gpu_mem_free_bytes() returning None.
import psutil

import system_stats

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Same two-line precedence rule config_store states for config.local.json:
# WHISPER_MODEL_SIZES_PATH > WHISPER_DATA_DIR/model_sizes.json >
# /data/model_sizes.json (Windows: <repo>\data — bare metal by definition, and
# "/data" would be drive-relative there; keep in sync with config._DATA_DIR).
# Computed here rather than imported from config: config imports config_store,
# and system_stats reaches this module, so importing config would close a cycle.
PATH = os.environ.get("WHISPER_MODEL_SIZES_PATH") or os.path.normpath(
    os.path.join(
        (os.environ.get("WHISPER_DATA_DIR") or "").strip()
        or (os.path.join(_REPO_DIR, "data") if os.name == "nt" else "/data"),
        "model_sizes.json"))

# Bumped only on an incompatible layout change; an unknown version reads as
# "no data" so a downgrade cannot mis-parse a newer file into a wrong estimate.
SCHEMA_VERSION = 1

# A re-measurement within this band is noise, not news. Without the band every
# single load would rewrite the file (the NVML delta jitters by a few MB), and
# the file is on the same disk as the model cache.
_REWRITE_THRESHOLD = 0.05

_lock = threading.Lock()
_cache: dict[str, dict] | None = None
_cache_mtime: float | None = None


def _key(name: str, device: str, compute_type: str) -> str:
    # The system_stats names are already family-namespaced (bare id = whisper,
    # `pyannote:`, `uvr:`, `gguf:`). Placement is appended because the same
    # weights at cuda/float16 and cpu/int8 differ by several gigabytes.
    return f"{name}|{device or ''}|{compute_type or ''}"


def _read(path: str = PATH) -> dict[str, dict]:
    """Return the models map, re-reading only when the file's mtime moved.

    NEVER raises: a truncated write, hand-editing, or a file from a future
    schema all degrade to "no data" — a missing estimate merely disables the
    fit check, while a raised exception here would break a model load."""
    global _cache, _cache_mtime
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _cache, _cache_mtime = {}, None
        return {}
    if _cache is not None and _cache_mtime == mtime:
        return _cache
    models: dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if isinstance(doc, dict) and doc.get("version") == SCHEMA_VERSION:
            raw = doc.get("models")
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict) and isinstance(v.get("bytes"), (int, float)):
                        models[k] = v
    except (OSError, ValueError):
        models = {}
    _cache, _cache_mtime = models, mtime
    return models


def _write(models: dict[str, dict], path: str = PATH) -> None:
    global _cache, _cache_mtime
    # Lazy: config_store drags in pydantic and the whole AdminConfig schema,
    # which this module's importers (system_stats) must not pay for.
    from config_store import _atomic_write_json, _save_lock
    doc = {"version": SCHEMA_VERSION, "models": models}
    with _save_lock(path):
        _atomic_write_json(doc, path, sort_keys=True, tmp_prefix=".model_sizes")
    import store_common
    store_common.secure_file(path)
    _cache = models
    try:
        _cache_mtime = os.path.getmtime(path)
    except OSError:
        _cache_mtime = None


def record(name: str, device: str, compute_type: str, vram_bytes: int, *,
           measured: bool = True) -> None:
    """Note a footprint. Cheap no-op when the value is already known to
    within _REWRITE_THRESHOLD, so a long-running worker writes the file a
    handful of times rather than once per load.

    ``measured=False`` marks an on-disk PRIOR (system_stats falls back to it
    when the NVML delta is unusable). A prior never overrides a measurement,
    and the first measurement REPLACES a prior outright: the disk walk sums
    every revision and fp32 blob in a hub dir, so letting it into the
    high-water mark below would make an inflated prior unbeatable forever."""
    if not name or not vram_bytes or vram_bytes <= 0:
        return
    k = _key(name, device, compute_type)
    src = "measured" if measured else "disk"
    with _lock:
        models = dict(_read())
        old = models.get(k)
        if old is not None:
            prev = int(old.get("bytes") or 0)
            old_measured = old.get("src") != "disk"
            if old_measured and not measured:
                return
            if measured and not old_measured:
                size = int(vram_bytes)
            else:
                if prev and abs(vram_bytes - prev) <= prev * _REWRITE_THRESHOLD:
                    return
                # max(), not last-write: CTranslate2's caching allocator makes
                # RE-loads under-report (the freed blocks it kept get reused,
                # see system_stats.py's module docstring). Under-estimating is
                # the dangerous direction — it is exactly what turns a "fits"
                # verdict into an OOM — so the ledger keeps the high-water mark.
                size = max(prev, int(vram_bytes))
            models[k] = {
                "bytes": size,
                "ts": time.time(),
                "n": int(old.get("n") or 0) + 1,
                "src": src,
            }
        else:
            models[k] = {"bytes": int(vram_bytes), "ts": time.time(), "n": 1,
                         "src": src}
        _write(models)


def estimate(name: str, device: str, compute_type: str) -> int | None:
    """Best known size in bytes, or None when this model was never measured."""
    models = _read()
    rec = models.get(_key(name, device, compute_type))
    if rec is not None:
        return int(rec["bytes"])
    # Any-device fallback: a cpu/int8 measurement is a poor proxy for a
    # cuda/float16 load, but a rough number beats no check at all — and the
    # exact record replaces it the first time that placement is measured.
    prefix = f"{name}|"
    for k, v in models.items():
        if k.startswith(prefix):
            return int(v["bytes"])
    # Never measured anywhere. Fall back to what the model WEIGHS ON DISK,
    # which for a GGUF or an ONNX file is a solid lower bound on its resident
    # size, and for a CT2 directory is close enough to decide whether a load
    # is even plausible. Without this, a model that has never been loaded
    # cannot be sized, so preload refuses it, so it is never loaded, so it is
    # never measured — the deadlock this fallback exists to break.
    return disk_size(name)


def disk_size(name: str) -> int | None:
    """On-disk footprint for a namespaced stats key, or None if not found.

    Deliberately best-effort and exception-swallowing: this is a prior for an
    admission heuristic, not an accounting figure, and a stat() that fails
    must degrade to "unknown" rather than break a load."""
    try:
        path = _model_path(name)
        if not path:
            return None
        if os.path.isfile(path):
            return int(os.path.getsize(path))
        if os.path.isdir(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for fn in files:
                    try:
                        total += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
            return total or None
    except Exception:  # noqa: BLE001 — a prior is never worth an exception
        return None
    return None


def _model_path(name: str) -> "str | None":
    """Filesystem location for a namespaced stats key.

    The prefixes are the ones preload.stats_key mints: `uvr:`, `gguf:`,
    `pyannote:`, and a bare id for whisper."""
    import config as _cfg
    root = (getattr(_cfg, "DOWNLOAD_ROOT", "") or "").strip()
    if name.startswith("uvr:"):
        model = name[4:]
        if "." not in model:
            model += ".onnx"
        # Same fallback bgm_separation._load_blocking uses for model_file_dir,
        # so a default install (no DOWNLOAD_ROOT) is still sizeable.
        return os.path.join(root or tempfile.gettempdir(), "audio-separator",
                            model)
    hf_home = os.environ.get("HF_HOME") or (
        os.path.join(root, "hf") if root else "")
    if not hf_home:
        return None
    if name.startswith("gguf:"):
        repo = name[5:].split(":", 1)[0]
        return _hf_repo_dir(hf_home, repo)
    if name.startswith("pyannote:"):
        return _hf_repo_dir(hf_home, name[9:])
    # Whisper: the HF cache dir. A transformers checkpoint that main converts
    # to CT2 lives under a separate root keyed by quantisation (see
    # main._converted_dir_for), which this name-only lookup cannot address;
    # the source repo is an adequate prior for it.
    return _hf_repo_dir(hf_home, name)


def _hf_repo_dir(hf_home: str, repo: str) -> "str | None":
    """`<HF_HOME>/hub/models--org--repo`, the layout huggingface_hub uses."""
    if not repo:
        return None
    return os.path.join(hf_home, "hub",
                        "models--" + repo.replace("/", "--"))


def fits(name: str, device: str, compute_type: str, *,
         reserve_bytes: int) -> tuple[bool | None, str | None]:
    """Would loading this model still leave `reserve_bytes` free?

    Returns (True, None) / (False, reason) / (None, "size_unknown") — None is
    "cannot say", distinct from a definite no, so callers can choose to try
    anyway rather than refusing a model they have simply never seen."""
    if (device or "").startswith("cuda"):
        free = system_stats.gpu_mem_free_bytes()
        if free is None:
            return (False, "vram_unknown")
        need = estimate(name, device, compute_type)
        if need is None:
            return (None, "size_unknown")
        return (True, None) if free - need >= reserve_bytes else (False, "insufficient_vram")
    free = int(psutil.virtual_memory().available)
    need = estimate(name, device, compute_type)
    if need is None:
        return (None, "size_unknown")
    return (True, None) if free - need >= reserve_bytes else (False, "insufficient_ram")


def _reset_for_tests() -> None:
    global _cache, _cache_mtime
    with _lock:
        _cache = None
        _cache_mtime = None
