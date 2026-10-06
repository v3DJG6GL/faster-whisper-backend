"""
Atomic JSON file writes and the per-path save lock around them.

Shared by the settings persistence layer (config.local.json / config.json in
settings/config_store.py) and the small runtime ledgers (model_sizes,
stage_rates). Import-light on purpose: stdlib only at import; `filelock` is
imported lazily inside save_lock().

Atomic writes: tmp file in the same directory, then os.replace. Retry loop
covers Windows sharing-violations from AV scanners briefly holding the file.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from typing import Any


# Per-path in-process guard, held around the cross-process FileLock below.
# The common case is two threadpool threads in ONE worker
# (`asyncio.to_thread(config_store.save_overrides, ...)` from two requests) — that is the
# reproduced lost update, and this lock settles it without depending on
# filelock's per-fd/per-process semantics or on filelock being installed at
# all. Keyed by absolute path so config.json and config.local.json don't block
# each other.
_SAVE_LOCKS: dict[str, threading.Lock] = {}
_SAVE_LOCKS_GUARD = threading.Lock()

# Generous relative to the measured worst case: the guard_regex ReDoS probe
# runs OUT OF PROCESS between the read and the write and was measured at
# 2.3-2.6 s with both schema maxima (200 rules x 200 entries), so a holder can
# legitimately occupy the lock for ~3 s. 15 s leaves room for a few queued
# savers plus a slow disk before we give up; longer would let a wedged peer
# stall an admin save indefinitely.
SAVE_LOCK_TIMEOUT_S = 15.0


@contextlib.contextmanager
def save_lock(path: str):
    """Serialise the read-modify-write of a config file across threads AND
    across uvicorn workers (SERVER_WORKERS > 1 — the file is the shared medium).

    Without this, config_store.save_overrides() is an unlocked read-modify-write whose
    window spans the out-of-process regex guard: a concurrent save that lands
    inside that window is silently reverted when the slower saver writes back
    its whole stale merged document (measured: an admin's
    ADMIN_WEBUI_ALLOWED_HOSTS edit reverted 2.2 s later by a non-admin
    PATCH /v1/pipeline-rules).

    The lock file is `<path>.lock`; it never collides with
    atomic_write_json()'s `.config*.tmp` tempfiles and nothing in this module
    scans the directory. Raises OSError on timeout so the callers' existing
    `except OSError` handling covers it. A missing `filelock` (a transitive of
    faster-whisper via huggingface_hub, and already used directly in main.py)
    degrades to the in-process lock only, rather than failing the save."""
    key = os.path.abspath(path)
    with _SAVE_LOCKS_GUARD:
        lk = _SAVE_LOCKS.setdefault(key, threading.Lock())
    if not lk.acquire(timeout=SAVE_LOCK_TIMEOUT_S):
        raise OSError(f"timed out waiting to write {path} (peer save in progress)")
    try:
        try:
            from filelock import FileLock, Timeout as FileLockTimeout
        except ImportError:      # pragma: no cover - filelock is a hard transitive
            yield
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        try:
            with FileLock(key + ".lock", timeout=SAVE_LOCK_TIMEOUT_S):
                yield
        except FileLockTimeout as e:
            raise OSError(
                f"timed out waiting to write {path} (peer worker save in progress)"
            ) from e
    finally:
        lk.release()


def atomic_write_json(obj: Any, path: str, *, sort_keys: bool, tmp_prefix: str) -> None:
    """Atomically write `obj` as pretty JSON to `path`.

    Write to a tempfile in the same directory, fsync, then os.replace. The
    rename is retried a few times — on Windows an AV scanner can briefly hold
    the destination open and raise PermissionError.

    `sort_keys`: True for config.local.json (stable diff of a flat settings
    dict); False for config.json so the committed factory rules keep their
    authored order and git diffs stay minimal.
    """
    dst_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=tmp_prefix, suffix=".tmp", dir=dst_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=sort_keys)
            f.flush()
            os.fsync(f.fileno())
        last_err: Exception | None = None
        for _ in range(5):
            try:
                os.replace(tmp, path)
                tmp = ""  # consumed
                break
            except PermissionError as e:
                last_err = e
                time.sleep(0.1)
        else:
            raise last_err if last_err else OSError("os.replace failed")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
