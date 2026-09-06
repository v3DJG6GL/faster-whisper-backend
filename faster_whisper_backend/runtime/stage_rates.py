"""Persisted ledger of LEARNED pipeline-stage throughput rates.

The run plan (core/run_plan.py) estimates how long each stage of a job will
take from a cost driver (bytes for a download, audio seconds for the GPU
stages, segments × targets for translation) divided by a rate. Rates are
learned from finished stages and keyed by what actually decides them — the
model, the device it ran on and (whisper) its compute type / (translation)
its mode — so a 7B GGUF on CPU and a 1.5B on CUDA never share a number.

Seeds cover a key until its first measured sample; from then on the ledger
row wins outright and later samples are folded in with an EWMA.

Same shape as runtime/model_sizes.py (a JSON file under the data dir, an
mtime-cached read that never raises, an atomic locked write) with one
deliberate difference: the path is resolved at CALL time through _path(),
never bound as a default argument. model_sizes binds `path=PATH` at def
time, which is why tests must patch three functions' __defaults__ to keep
it out of the real /data — and still miss the one record() calls.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time

from faster_whisper_backend.paths import REPO_ROOT

# WHISPER_STAGE_RATES_PATH > WHISPER_DATA_DIR/stage_rates.json >
# /data/stage_rates.json (Windows: <repo>\data) — the model_sizes rule.
PATH = os.environ.get("WHISPER_STAGE_RATES_PATH") or os.path.normpath(
    os.path.join(
        (os.environ.get("WHISPER_DATA_DIR") or "").strip()
        or (os.path.join(REPO_ROOT, "data") if os.name == "nt" else "/data"),
        "stage_rates.json"))

SCHEMA_VERSION = 1

# Weight of the newest sample. 0.5 forgets a one-off stall within a few
# runs but still lets a genuinely slower placement settle quickly.
ALPHA = 0.5

# Until a key has a measured sample. Units per stage:
#   downloading   bytes per second
#   separating    × realtime (audio seconds per wall second)
#   transcribing  × realtime, over the audio the VAD kept
#   diarizing     × realtime
#   translating   translation units (segments) per second, per target
SEEDS: dict[str, float] = {
    "downloading": 3_000_000.0,
    "separating": 8.0,
    "transcribing": 6.0,
    "diarizing": 11.0,
    "translating": 1.6,
    # pyannote's steps, × realtime each: one forward pass of the
    # segmentation model is seconds, the speaker embeddings are the wall
    # clock, the clustering that follows is seconds again. These split the
    # diarizing row's bar; the stage key above still estimates the whole.
    "diarizing.segmentation": 600.0,
    "diarizing.embeddings": 14.0,
    "diarizing.clustering": 400.0,
}

_lock = threading.Lock()
_cache: dict[str, dict] | None = None
_cache_mtime: float | None = None


def _path() -> str:
    return PATH


def key_for(stage: str, model: str | None, device: str | None,
            compute: str | None = None) -> str:
    return f"{stage}|{model or ''}|{device or ''}|{compute or ''}"


def _read() -> dict[str, dict]:
    """The rates map, re-read only when the file's mtime moved. NEVER
    raises: a truncated or hand-edited file degrades to "no data", which
    merely means seeds — an exception here would break a transcription."""
    global _cache, _cache_mtime
    path = _path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _cache, _cache_mtime = {}, None
        return {}
    if _cache is not None and _cache_mtime == mtime:
        return _cache
    rates: dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if isinstance(doc, dict) and doc.get("version") == SCHEMA_VERSION:
            raw = doc.get("rates")
            if isinstance(raw, dict):
                for k, v in raw.items():
                    r = v.get("rate") if isinstance(v, dict) else None
                    if (isinstance(r, (int, float)) and math.isfinite(r)
                            and r > 0):
                        rates[k] = v
    except (OSError, ValueError):
        rates = {}
    _cache, _cache_mtime = rates, mtime
    return rates


def _write_locked(rates: dict[str, dict]) -> None:
    global _cache, _cache_mtime
    from faster_whisper_backend.config_store import _atomic_write_json
    from faster_whisper_backend.core import store_common
    path = _path()
    _atomic_write_json({"version": SCHEMA_VERSION, "rates": rates}, path,
                       sort_keys=True, tmp_prefix=".stage_rates")
    store_common.secure_file(path)
    _cache = rates
    try:
        _cache_mtime = os.path.getmtime(path)
    except OSError:
        _cache_mtime = None


def lookup(stage: str, model: str | None, device: str | None,
           compute: str | None = None) -> dict:
    """`{rate, src, n}` — the measured row for this exact key when one
    exists, else the stage seed (`src == "seed"`, `n == 0`). A stage with no
    seed at all answers rate None."""
    rec = _read().get(key_for(stage, model, device, compute))
    if rec is not None:
        return {"rate": float(rec["rate"]), "src": "measured",
                "n": int(rec.get("n") or 0)}
    seed = SEEDS.get(stage)
    return {"rate": seed, "src": "seed", "n": 0}


def record(stage: str, model: str | None, device: str | None,
           compute: str | None, rate: float) -> None:
    """Fold one measured rate into the ledger. The first sample REPLACES the
    seed; later ones EWMA in. Non-finite or non-positive samples are
    dropped — a stage that took 0 s or ran backwards is bookkeeping noise,
    never evidence. Never raises: recording is a nicety after a finished
    job, and a locked or unwritable file must not fail the request."""
    try:
        r = float(rate)
    except (TypeError, ValueError):
        return
    if not math.isfinite(r) or r <= 0 or not stage:
        return
    k = key_for(stage, model, device, compute)
    with _lock:
        with contextlib.ExitStack() as stack:
            try:
                from faster_whisper_backend.config_store import _save_lock
                stack.enter_context(_save_lock(_path()))
            except OSError:
                pass   # lock timeout: write unlocked rather than lose the sample
            try:
                _record_locked(k, r)
            except Exception:  # noqa: BLE001 — see docstring
                pass


def _record_locked(k: str, r: float) -> None:
    global _cache_mtime
    _cache_mtime = None   # see a peer worker's just-written rows
    rates = dict(_read())
    old = rates.get(k)
    if old is None:
        rates[k] = {"rate": r, "n": 1, "ts": time.time()}
    else:
        prev = float(old.get("rate") or r)
        rates[k] = {"rate": ALPHA * r + (1.0 - ALPHA) * prev,
                    "n": int(old.get("n") or 0) + 1, "ts": time.time()}
    _write_locked(rates)


def _reset_for_tests() -> None:
    global _cache, _cache_mtime
    with _lock:
        _cache = None
        _cache_mtime = None
