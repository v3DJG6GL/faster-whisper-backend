"""Central registry of running jobs (transcribe / dictate / translate /
download / preload).

One process-wide dict guarded by a threading.Lock: handlers run on the
asyncio loop, but model loads and downloads report from executor threads,
so plain-dict-plus-GIL (the _BATCH_PROGRESS stance) is not enough here —
job_start/job_update/job_end race across threads.

Lifecycle: `job_start(kind, ...)` → `job_update(job_id, ...)` (any number
of times, merging only non-None fields) → `job_end(job_id)` in the owning
code path's finally. `jobs_snapshot()` feeds /stats (payload key "jobs")
and the WebUI header activity cluster.

Identity scrubbing: the snapshot's non-admin variant omits `user`, `key`
and `detail` — mirroring the recent-transcriptions projection philosophy
(/stats has no 'own' scope, so a non-admin holder of pages.stats must not
read other users' identities). No transcript text ever enters a job entry
(last_text stays in _BATCH_PROGRESS only).

Bounded: a job leak (a code path that misses its job_end) is capped at
_MAX_JOBS entries; past that the oldest entry is dropped on insert.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

# An unknown kind is silently coerced to "transcribe" in job_start below, so a
# family missing from this tuple does not merely lose its label — it shows up
# in the /stats activity cluster as somebody transcribing. "preload" earns its
# place for exactly that reason: warming a model is not a transcription.
# Adding a kind here needs matching render entries: the /stats `.kindchip.<k>`
# CSS + `#rj-kind` filter button (stats_routes.py) and the header activity
# cluster's `kindCls` map (web_common.py) — otherwise it renders as an
# unstyled, unfilterable grey chip.
KINDS = ("transcribe", "dictate", "translate", "download", "preload")

_MAX_JOBS = 200

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def job_start(
    kind: str,
    *,
    id: "str | None" = None,
    model: "str | None" = None,
    user: "str | None" = None,
    key: "str | None" = None,
    detail: "str | None" = None,
    total_bytes: "int | None" = None,
) -> str:
    """Register a running job; returns its id (a fresh uuid4 hex unless the
    caller passes its own request/session id). Auto-stamps monotonic +
    wall-clock start times."""
    job_id = id or uuid.uuid4().hex
    entry = {
        "id": job_id,
        "kind": kind if kind in KINDS else "transcribe",
        "model": model,
        "user": user,
        "key": key,
        "detail": detail,
        "total_bytes": total_bytes,
        "progress": None,
        "stage": None,
        "step": None,
        # Set via job_update by handlers whose client opted into the
        # progress/cancel registry — lets an admin's activity popover POST
        # the cancel endpoint for this job.
        "progress_id": None,
        "started_ts": time.time(),
        "started_mono": time.monotonic(),
    }
    with _lock:
        if job_id not in _jobs and len(_jobs) >= _MAX_JOBS:
            oldest = min(_jobs, key=lambda k: _jobs[k]["started_mono"])
            _jobs.pop(oldest, None)
        _jobs[job_id] = entry
    return job_id


def job_update(job_id: "str | None", **fields: Any) -> None:
    """Merge non-None `fields` into the job entry (no-op on unknown ids —
    progress mirroring must never break the request that feeds it)."""
    if not job_id:
        return
    with _lock:
        entry = _jobs.get(job_id)
        if entry is None:
            return
        for k, v in fields.items():
            if v is not None:
                entry[k] = v


def job_end(job_id: "str | None") -> None:
    """Drop the job entry. Idempotent."""
    if not job_id:
        return
    with _lock:
        _jobs.pop(job_id, None)


def jobs_snapshot(include_identity: bool = False) -> list[dict[str, Any]]:
    """List of running jobs, oldest first, ready for JSON. Each row carries
    kind/model/progress/stage/step/total_bytes plus elapsed seconds; the
    identity fields (user, key, detail) only when `include_identity` (admin
    viewers)."""
    now = time.monotonic()
    with _lock:
        entries = sorted(_jobs.values(), key=lambda e: e["started_mono"])
        out = []
        for e in entries:
            row = {
                "id": e["id"],
                "kind": e["kind"],
                "model": e["model"],
                "progress": e["progress"],
                "stage": e["stage"],
                "step": e["step"],
                "total_bytes": e["total_bytes"],
                "started_ts": e["started_ts"],
                "elapsed_s": round(now - e["started_mono"], 1),
            }
            if include_identity:
                row["user"] = e["user"]
                row["key"] = e["key"]
                row["detail"] = e["detail"]
                # Cancel handle (admin viewers only): the id the cancel
                # endpoint accepts, when the job's client registered one.
                row["progress_id"] = e.get("progress_id")
            out.append(row)
        return out


def _reset_for_tests() -> None:
    with _lock:
        _jobs.clear()
