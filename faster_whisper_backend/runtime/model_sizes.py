"""Persisted ledger of MEASURED model sizes, plus a free-memory fit check.

Why a file at all: model_registry.register_loaded_model() already measures every
model's footprint as an NVML delta at load time, but that number lives in a
process-local dict and dies with the worker. To decide whether a model can be
preloaded we need its size BEFORE loading it — i.e. carried across restarts.

Why the fit check reads the DRIVER's free VRAM rather than summing our own
registry: other processes on the machine (a second worker, a game, a desktop
compositor) consume the same card, and our bookkeeping cannot see them.

Import-light on purpose: model_registry.register_loaded_model imports it
lazily, and it imports system_stats (which imports model_registry). The writer
(core/atomic_json) is stdlib-only.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import math
import os
import tempfile
import threading
import time

# Both are hard dependencies already imported unguarded by system_stats (psutil
# is in requirements.txt); NVML is the optional one, and system_stats hides that
# behind gpu_mem_free_bytes() returning None.
import psutil

from faster_whisper_backend.core import atomic_json
from faster_whisper_backend.runtime import hf_cache
from faster_whisper_backend.runtime import system_stats
from faster_whisper_backend.paths import REPO_ROOT

_REPO_DIR = REPO_ROOT  # the checkout, not this package — see paths.py
# Same two-line precedence rule config_store states for config.local.json:
# WHISPER_MODEL_SIZES_PATH > WHISPER_DATA_DIR/model_sizes.json >
# /data/model_sizes.json (Windows: <repo>\data — bare metal by definition, and
# "/data" would be drive-relative there; keep in sync with config._DATA_DIR).
# Computed here rather than imported from config: config imports config_store,
# and model_registry reaches this module, so importing config would close a cycle.
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
    # The model_registry names are already family-namespaced (bare id = whisper,
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
                    b = v.get("bytes") if isinstance(v, dict) else None
                    # json.load accepts Infinity / NaN, and a bool is an
                    # int: none of them, nor a non-positive size, is a
                    # footprint. Normalised here, once, like
                    # stage_rates._read: lookup() and the merge trust
                    # `bytes` and `n`, and int(Infinity) or int("many")
                    # there would 500 /stats.
                    if (isinstance(b, (int, float)) and not isinstance(b, bool)
                            and math.isfinite(b) and b > 0):
                        row = {"bytes": int(b), "n": _count(v.get("n"))}
                        if "ts" in v:
                            row["ts"] = v["ts"]
                        if isinstance(v.get("src"), str):
                            row["src"] = v["src"]
                        models[k] = row
    except (OSError, ValueError, OverflowError, RecursionError):
        # OverflowError: a 400-digit integer literal is a valid JSON int
        # that math.isfinite() cannot convert; RecursionError: a deeply
        # nested document.
        models = {}
    _cache, _cache_mtime = models, mtime
    return models


def _count(n) -> int:
    if (isinstance(n, (int, float)) and not isinstance(n, bool)
            and math.isfinite(n) and n > 0):
        return int(n)
    return 0


def _write_locked(models: dict[str, dict], path: str = PATH) -> None:
    """The write itself; the caller holds atomic_json.save_lock(path)
    (a plain threading.Lock per path, NOT reentrant — never nest)."""
    global _cache, _cache_mtime
    doc = {"version": SCHEMA_VERSION, "models": models}
    atomic_json.atomic_write_json(doc, path, sort_keys=True, tmp_prefix=".model_sizes")
    from faster_whisper_backend.core import store_common
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

    ``measured=False`` marks an on-disk PRIOR (model_registry falls back to it
    when the NVML delta is unusable). A prior never overrides a measurement,
    and the first measurement REPLACES a prior outright: the disk walk sums
    every revision and fp32 blob in a hub dir, so letting it into the
    high-water mark below would make an inflated prior unbeatable forever."""
    if not name or not vram_bytes or vram_bytes <= 0:
        return
    k = _key(name, device, compute_type)
    src = "measured" if measured else "disk"
    # PATH resolved once, at call time, and handed to the lock, the read and
    # the write alike: a def-time default would let them name different files.
    path = PATH
    with _lock:
        # The read-modify-write below rewrites the WHOLE ledger, so with
        # SERVER_WORKERS > 1 two workers measuring different models would
        # each write back their own stale document and lose the other's
        # row. atomic_json.save_lock serialises it across workers; a lock
        # timeout (OSError) falls back to the unlocked path — record() must
        # never break a load.
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(atomic_json.save_lock(path))
            except OSError:
                pass
            _record_locked(k, vram_bytes, measured, src, path)


def _is_measured(k: str, row: dict) -> bool:
    """A ledger row counts as a measurement only on a cuda placement. NVML
    cannot measure a cpu load (model_registry no longer records one), so a
    non-cuda "measured" row is a stray VRAM delta from an older build: it is
    treated like a disk prior — a fresh disk prior replaces it, and the
    any-device fallback ranks it last."""
    return (row.get("src") != "disk"
            and k.rsplit("|", 2)[-2].startswith("cuda"))


def _record_locked(k: str, vram_bytes: int, measured: bool, src: str,
                   path: str) -> None:
    """The merge half of record(). Writes through _write_locked without
    taking the lock: the caller may already hold
    atomic_json.save_lock(path), which is not reentrant."""
    global _cache_mtime
    # Drop the mtime cache so the merge sees a peer's just-written rows.
    _cache_mtime = None
    models = dict(_read(path))
    old = models.get(k)
    if old is not None:
        prev = int(old.get("bytes") or 0)
        old_measured = _is_measured(k, old)
        if old_measured and not measured:
            return
        if measured and not old_measured:
            size = int(vram_bytes)
        elif not measured:
            # Disk vs disk: last write. A disk walk is deterministic, so a
            # newer prior replaces an older one — a high-water mark here
            # would keep an inflated pre-fix walk (symlink double count,
            # every GGUF quant) forever.
            if prev and abs(vram_bytes - prev) <= prev * _REWRITE_THRESHOLD:
                return
            size = int(vram_bytes)
        else:
            # max(), not last-write: CTranslate2's caching allocator makes
            # RE-loads under-report (the freed blocks it kept get reused,
            # see model_registry.py's module docstring). Under-estimating is
            # the dangerous direction — it is exactly what turns a "fits"
            # verdict into an OOM — so the ledger keeps the high-water mark,
            # and a re-measurement at or below it is no news at all.
            if prev and vram_bytes <= prev * (1 + _REWRITE_THRESHOLD):
                return
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
    _write_locked(models, path)


def lookup(name: str, device: str, compute_type: str) -> dict | None:
    """Best known size WITH its provenance: `{bytes, src, n, ts}` where src is
    "measured" (an NVML delta for exactly this placement), "disk" (the
    on-disk prior recorded for this placement, or the live disk walk when
    nothing was ever recorded), or "proxy" (a measurement of the same model
    on another device / compute type). None when nothing is known at all.
    /stats shows the source so an estimate is never mistaken for a
    measurement."""
    models = _read()
    exact = _key(name, device, compute_type)
    rec = models.get(exact)
    if rec is not None:
        # The same provenance rule as the fallback below: a stray non-cuda
        # "measured" row is only a disk-grade prior, never a measurement.
        return {"bytes": int(rec["bytes"]),
                "src": "measured" if _is_measured(exact, rec) else "disk",
                "n": int(rec.get("n") or 0), "ts": rec.get("ts")}
    # Any-device fallback: a cpu/int8 measurement is a poor proxy for a
    # cuda/float16 load, but a rough number beats no check at all — and the
    # exact record replaces it the first time that placement is measured.
    # Ranked, not first-match: the ledger is written sorted, so "cpu" would
    # always beat "cuda" and a cpu DISK row would shadow a measurement on the
    # same card. A measurement on the same device family first, then one on
    # any device; a disk row (or a stray non-cuda "measured" row, see
    # _is_measured) only after the live disk walk below. max() within a rank
    # stays conservative.
    prefix = f"{name}|"
    cuda = (device or "").startswith("cuda")
    best = None
    for k, v in models.items():
        if not k.startswith(prefix):
            continue
        peer_cuda = k[len(prefix):].startswith("cuda")
        if not _is_measured(k, v):
            rank = 2
        else:
            rank = 0 if peer_cuda == cuda else 1
        cand = (-rank, int(v["bytes"]))
        if best is None or cand > best[0]:
            best = (cand, v)
    if best is not None and best[0][0] > -2:   # rank 0/1: a measurement
        v = best[1]
        return {"bytes": int(v["bytes"]), "src": "proxy",
                "n": int(v.get("n") or 0), "ts": v.get("ts")}
    # Never measured anywhere. Fall back to what the model WEIGHS ON DISK,
    # which for a GGUF or an ONNX file is a solid lower bound on its resident
    # size, and for a CT2 directory is close enough to decide whether a load
    # is even plausible. Without this, a model that has never been loaded
    # cannot be sized, so preload refuses it, so it is never loaded, so it is
    # never measured — the deadlock this fallback exists to break.
    #
    # Also ahead of a stored disk row of ANOTHER placement: disk_size() is
    # name-only, so such a row can never be more accurate than a fresh
    # walk, only staler — a pre-fix build's inflated prior (symlink double
    # count, every GGUF quant) survives for any placement that is never
    # loaded again. The stored row is kept only for a model no longer on
    # disk.
    size = disk_size(name)
    if size is None:
        if best is None:
            return None
        v = best[1]
        return {"bytes": int(v["bytes"]), "src": "disk",
                "n": int(v.get("n") or 0), "ts": v.get("ts")}
    return {"bytes": int(size), "src": "disk", "n": 0, "ts": None}


def estimate(name: str, device: str, compute_type: str) -> int | None:
    """Best known size in bytes, or None when this model was never measured
    (see lookup() for the provenance-carrying variant)."""
    rec = lookup(name, device, compute_type)
    return None if rec is None else int(rec["bytes"])


def disk_size(name: str) -> int | None:
    """On-disk footprint for a namespaced stats key, or None if not found.

    Deliberately best-effort and exception-swallowing: this is a prior for an
    admission heuristic, not an accounting figure, and a stat() that fails
    must degrade to "unknown" rather than break a load."""
    try:
        path = _model_path(name)
        if not path:
            return None
        if name.startswith("gguf:"):
            # Quant-less too: translation.engine resolves it with "*.gguf"
            # and wants exactly one match, so the walk over the whole repo
            # dir (every revision, every quant fetched) is never the size.
            return _gguf_quant_size(path, _gguf_quant(name))
        if os.path.isfile(path):
            return int(os.path.getsize(path))
        if os.path.isdir(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for fn in files:
                    p = os.path.join(root, fn)
                    # A hub cache's snapshots/<rev>/* are symlinks into
                    # blobs/, and getsize follows them: counting both would
                    # double every weight. blobs/ holds the real bytes;
                    # without symlink support the hub writes real files
                    # and there are no links to skip.
                    if os.path.islink(p):
                        continue
                    try:
                        total += os.path.getsize(p)
                    except OSError:
                        pass
            return total or None
    except Exception:  # noqa: BLE001 — a prior is never worth an exception
        return None
    return None


def _gguf_quant(name: str) -> str:
    """The `:QUANT` suffix of a `gguf:org/repo:QUANT` key, else ''."""
    if not name.startswith("gguf:"):
        return ""
    return name[5:].partition(":")[2]


def _gguf_quant_size(repo_dir: str, quant: str) -> "int | None":
    """Size of the ONE file in the newest snapshot matching `*{quant}.gguf`
    (the glob translation.engine resolves the quant with), or None when
    there is no single match. Summing the repo dir instead would charge an
    operator who ever fetched a second quant for both files."""
    snaps = os.path.join(repo_dir, "snapshots")
    try:
        revs = [p for p in (os.path.join(snaps, d) for d in os.listdir(snaps))
                if os.path.isdir(p)]
    except OSError:
        return None
    if not revs:
        return None
    newest = max(revs, key=os.path.getmtime)
    pattern = f"*{quant}.gguf".lower()
    matches = []
    for root, _dirs, files in os.walk(newest):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, newest).replace(os.sep, "/")
            if fnmatch.fnmatchcase(rel.lower(), pattern):
                matches.append(p)
    if len(matches) != 1:
        return None
    # os.stat follows the snapshot symlink to its blob, once.
    return int(os.stat(matches[0]).st_size) or None


def _model_path(name: str) -> "str | None":
    """Filesystem location for a namespaced stats key.

    The prefixes are the ones preload.stats_key mints: `uvr:`, `gguf:`,
    `pyannote:`, and a bare id for whisper."""
    from faster_whisper_backend.settings import config as _cfg
    root = (getattr(_cfg, "DOWNLOAD_ROOT", "") or "").strip()
    if name.startswith("uvr:"):
        model = name[4:]
        if "." not in model:
            model += ".onnx"
        # The directory bgm_separation.model_file_dir() loads from, tempdir
        # fallback included, so a default install (no DOWNLOAD_ROOT) is still
        # sizeable. Mirrored rather than called: this module is reached from
        # the loaded-model registry on every load and stays import-light, and
        # a runtime → audio import would deepen the audio/runtime pair (audio
        # imports runtime). tests/runtime/test_model_sizes.py pins the match.
        return os.path.join(root or tempfile.gettempdir(), "audio-separator",
                            model)
    # Where translation / diarization downloads land (hf_cache owns the
    # precedence: HF_HUB_CACHE, else HF_HOME/hub, else <DOWNLOAD_ROOT>/hf,
    # else the hub default).
    hub = hf_cache.hub_lookup_dir()
    if name.startswith("gguf:"):
        repo = name[5:].split(":", 1)[0]
        return _hf_repo_dir(hub, repo)
    if name.startswith("pyannote:"):
        return _hf_repo_dir(hub, name[9:])
    # Whisper: main resolves a bare id ('large-v3') through faster_whisper's
    # _MODELS table and passes DOWNLOAD_ROOT itself as snapshot_download's
    # cache_dir (no `/hf` sub-dir — that convention belongs to
    # runtime.hf_cache.hub_cache_dir()), so the repo dir sits
    # directly under the root; without a root the hub cache is used. A
    # transformers checkpoint that main converts to CT2 lives under a
    # separate root keyed by quantisation (see transcription.models._converted_dir_for),
    # which this name-only lookup cannot address; the source repo is an
    # adequate prior for it.
    # DEFAULT_MODEL may be a local CT2 directory, which the loader opens in
    # place: size that directory, not a hub leaf that never exists.
    local = os.path.expanduser(name)
    if os.path.isdir(local):
        return local
    try:
        from faster_whisper.utils import _MODELS
        repo = _MODELS.get(name) or name
    except Exception:  # noqa: BLE001 — faster_whisper absent = bare repo id
        repo = name
    leaf = "models--" + repo.replace("/", "--")
    # snapshot_download expands "~" in the cache_dir it is handed, so a .env
    # DOWNLOAD_ROOT=~/models lands under the home dir. (UVR above is not
    # expanded: bgm_separation.model_file_dir() passes the root verbatim.)
    candidates = [os.path.join(os.path.expanduser(root), leaf)] if root else []
    candidates.append(os.path.join(hub, leaf))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _hf_repo_dir(hub: str, repo: str) -> "str | None":
    """`<hub cache>/models--org--repo`, the layout huggingface_hub uses."""
    if not repo:
        return None
    return os.path.join(hub, "models--" + repo.replace("/", "--"))


def fits(name: str, device: str, compute_type: str, *,
         reserve_bytes: int,
         device_index: int = 0) -> tuple[bool | None, str | None]:
    """Would loading this model still leave `reserve_bytes` free?
    `device_index` is the CUDA ordinal the load would land on (whisper's
    per-model DEVICE_INDEX): the free VRAM of THAT card is what counts.

    Returns (True, None) / (False, reason) / (None, "size_unknown") — None is
    "cannot say", distinct from a definite no, so callers can choose to try
    anyway rather than refusing a model they have simply never seen."""
    if (device or "").startswith("cuda"):
        free = system_stats.gpu_mem_free_bytes(device_index)
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
