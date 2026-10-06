"""Batch-run progress and the durable job resource: the progress / cancel
routes for a run posted with a ``progress_id`` and GET/DELETE /v1/jobs*
(core/jobs_store.py). The registries themselves live in
transcription/progress.py.
"""
import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse as _JSONResponse

from faster_whisper_backend.auth import rate_limit as _rl
from faster_whisper_backend.auth.dependencies import get_current_user as _get_current_user_dep
from faster_whisper_backend.core import jobs_store as _jobs_store
from faster_whisper_backend.media import media_store as url_media_store
from faster_whisper_backend.transcription import progress as tx_progress

logger = logging.getLogger("whisper-api")

router = APIRouter()


@router.get("/v1/audio/transcriptions/progress/{progress_id}")
async def transcription_progress(progress_id: str,
                                 user: dict = Depends(_get_current_user_dep)):
    """Live progress of an in-flight file transcription that was posted with a
    matching `progress_id` form field. Stages: waiting (semaphore queue) →
    [resolving → downloading (URL flow: `progress` 0..1 when the size is
    known, with `total_bytes`)] → separating → analyzing (audio decode +
    VAD, inside transcribe()) → transcribing (with `progress` 0..1 and the
    audio `duration`) → diarizing → translating. A requested-but-declined
    stage lands in `skipped` instead ("separating" / "diarizing" /
    "translating"). An unknown/finished id answers stage "unknown" — the
    POST's own response is the completion signal, so the poller just
    stops."""
    if not tx_progress._PROGRESS_ID_RE.match(progress_id):
        raise HTTPException(status_code=422, detail="malformed progress_id")
    entry = tx_progress._progress_entry_for(progress_id, user)
    if entry is None:
        return {"stage": "unknown"}
    return tx_progress._progress_payload(progress_id, entry)


@router.post("/v1/audio/transcriptions/cancel/{progress_id}")
async def transcription_cancel(progress_id: str,
                               user: dict = Depends(_get_current_user_dep)):
    """Abort the in-flight transcription posted with this `progress_id`.

    Closing the upload connection does NOT stop the server-side work (the
    stages run in executor threads that outlive the handler task), so a
    client's Cancel button calls this too. The flag is checked cooperatively
    between demix chunks / decoded segments / pyannote steps, so the abort
    lands within a chunk, not instantly. Only ids currently in flight are
    accepted; an unknown/finished id answers cancelled=false."""
    if not tx_progress._PROGRESS_ID_RE.match(progress_id):
        raise HTTPException(status_code=422, detail="malformed progress_id")
    if tx_progress._progress_entry_for(progress_id, user) is None:
        return {"cancelled": False}
    tx_progress._BATCH_CANCELLED.add(progress_id)
    logger.info("[batch] cancel requested for an in-flight transcription")
    return {"cancelled": True}


# ── Server jobs: GET/DELETE /v1/jobs* ───────────────────────────────────────
# The durable job resource (core/jobs_store.py): every batch run posted with
# a progress_id is a row a client can list, poll, fetch the result of and
# cancel/delete — the connection that carried the POST is no longer the only
# way to get the transcript. Owner-gated like the progress route: a foreign,
# expired or unknown id is one 404 (no existence oracle); admins read all.

_jobs_rate = _rl.FixedWindow(
    config_field="JOBS_RATE_PER_MIN",
    window_s=60.0,
    default_max=120,
    message="too many job requests — slow down "
            "({limit}/min; retry in {retry_after}s)",
)

_JOB_STATES = ("running", "done", "failed", "cancelled")


def _job_wire(row: dict) -> dict:
    """A job row's client shape (never the result blob)."""
    return {
        "job_id": row["job_id"],
        "kind": row.get("kind"),
        "state": row.get("state"),
        "created_at": row.get("created_ts"),
        "finished_at": row.get("finished_ts"),
        "expires_at": row.get("expires_ts"),
        "model": row.get("model"),
        "source_kind": row.get("source_kind"),
        "source_name": row.get("source_name"),
        "task": row.get("task"),
        "response_format": row.get("response_format"),
        "error": row.get("error"),
        "result_bytes": int(row.get("result_bytes") or 0),
        "result_available": bool(row.get("result_available")),
    }


def _jobs_gate(user: dict, request: Request) -> None:
    if not tx_progress._jobs_enabled():
        raise HTTPException(status_code=403,
                            detail="server jobs are not enabled on this server")
    _jobs_rate.hit(_rl.identity_key(user, request))


def _job_for_caller(job_id: str, user: dict, request: Request) -> dict:
    """The row for `job_id` if this caller may see it, else 404 (unknown,
    expired and foreign all read the same); 422 on a malformed id."""
    _jobs_gate(user, request)
    if not tx_progress._PROGRESS_ID_RE.match(job_id):
        raise HTTPException(status_code=422, detail="malformed job id")
    row = _jobs_store.get(job_id)
    if (row is None or float(row.get("expires_ts") or 0) < time.time()
            or (not user.get("is_admin") and not _jobs_store.is_owner(
                row, user_id=user.get("user_id"), key_id=user.get("key_id")))):
        raise HTTPException(status_code=404, detail="job not found")
    return row


def _scrub_media_refs(payload: dict, *, user_id: "str | None") -> dict:
    """A stored result may name retained media (source_media_id and the
    video twin) that has since expired or died with a restart — re-validate
    each id against the media store and refresh its expiry, or drop the pair
    so the client never receives a dangling id. A video still pending when
    the run finished never got its id into the payload; the flag goes too."""
    from faster_whisper_backend.media import media_store as _ums
    for id_key, exp_key in (("source_media_id", "source_media_expires_at"),
                            ("source_video_media_id",
                             "source_video_expires_at")):
        if id_key not in payload and exp_key not in payload:
            continue
        mid = payload.get(id_key)
        ok = (isinstance(mid, str) and url_media_store.MEDIA_ID_RE.match(mid)
              and _ums.resolve_entry(mid, user_id=user_id) is not None)
        if ok:
            payload[exp_key] = _ums.expires_at_unix(mid)
        else:
            payload.pop(id_key, None)
            payload.pop(exp_key, None)
    payload.pop("source_video_pending", None)
    return payload


@router.get("/v1/jobs")
async def jobs_list(request: Request,
                    state: "str | None" = None,
                    limit: int = 50,
                    all_users: int = Query(0, alias="all"),
                    user: dict = Depends(_get_current_user_dep)):
    """The caller's job rows, newest first (`?state=` filters; admins may
    pass `?all=1` for every user's). No result blobs, no live progress —
    poll GET /v1/jobs/{id} for those."""
    _jobs_gate(user, request)
    if state is not None and state not in _JOB_STATES:
        raise HTTPException(status_code=422, detail="unknown job state")
    rows = _jobs_store.list_jobs(
        user_id=user.get("user_id"), key_id=user.get("key_id"),
        all_users=bool(all_users and user.get("is_admin")),
        state=state, limit=max(1, min(int(limit), 200)))
    return _JSONResponse({"jobs": [_job_wire(r) for r in rows]},
                         headers={"Cache-Control": "no-store"})


@router.get("/v1/jobs/{job_id}")
async def job_get(job_id: str, request: Request,
                  user: dict = Depends(_get_current_user_dep)):
    """One job row plus, while its run is in flight in THIS process, the
    live progress under `progress` (the progress route's shape) — one poll
    serves a re-attached client. `progress` is null once the run closed, or
    when it runs in a sibling worker process."""
    row = _job_for_caller(job_id, user, request)
    out = _job_wire(row)
    entry = tx_progress._progress_entry_for(job_id, user)
    out["progress"] = (tx_progress._progress_payload(job_id, entry)
                       if entry is not None else None)
    return _JSONResponse(out, headers={"Cache-Control": "no-store"})


@router.get("/v1/jobs/{job_id}/result")
async def job_result(job_id: str, request: Request,
                     user: dict = Depends(_get_current_user_dep)):
    """The run's response payload, byte-for-byte what the POST returned
    (the `text` format comes back as the JSON string it was). 409 while the
    run is still going; 404 when there is none (failed / cancelled)."""
    row = _job_for_caller(job_id, user, request)
    if row.get("state") == "running":
        raise HTTPException(status_code=409, detail="job still running")
    if row.get("state") != "done" or not row.get("result_available"):
        raise HTTPException(status_code=404, detail="no result for this job")
    payload = await asyncio.to_thread(_jobs_store.get_result, job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="no result for this job")
    if isinstance(payload, dict):
        payload = _scrub_media_refs(payload, user_id=user.get("user_id"))
    return _JSONResponse(payload, headers={"Cache-Control": "no-store"})


@router.delete("/v1/jobs/{job_id}")
async def job_delete(job_id: str, request: Request,
                     user: dict = Depends(_get_current_user_dep)):
    """Running → cooperative cancel (same flag as the cancel route; the row
    turns `cancelled` when the handler unwinds). Finished → delete the row
    and its stored result."""
    row = _job_for_caller(job_id, user, request)
    if row.get("state") == "running":
        if tx_progress._progress_entry_for(job_id, user) is None:
            # Hosted by a sibling worker, or the entry was cap-evicted.
            return {"cancelled": False}
        tx_progress._BATCH_CANCELLED.add(job_id)
        logger.info("[jobs] cancel requested for an in-flight run")
        return {"cancelled": True}
    return {"deleted": bool(_jobs_store.delete(job_id))}
