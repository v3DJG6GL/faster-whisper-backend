"""Per-run progress, cancellation and plan state for the batch routes, keyed
on the client's opt-in ``progress_id``: the live progress registry
(_BATCH_PROGRESS + owner stamps + closed-id tombstones, written by
_progress_set from every stage, retired by _progress_close), the cooperative
cancel flag (_BATCH_CANCELLED, _check_cancelled / _ClientCancelled), the
progress_id → job / preload-plan / run-plan bindings that _progress_set
mirrors every tick into, the job-ledger writes for runs posted with an id
(_jobs_start / _jobs_finish, transcription/jobs_store.py), and the progress route's
wire shape (_progress_entry_for, _progress_payload).

Callers go through the module attribute (``tx_progress._progress_set(...)``)
so a test patching a name here reaches every caller. Must not import main or
the url/media helpers (the keep_video state reaches the entry as a plain
``video=`` field).
"""
import asyncio
import logging
import re
import time

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder as _jsonable_encoder

from faster_whisper_backend.core import jobs
from faster_whisper_backend.transcription import jobs_store as _jobs_store
from faster_whisper_backend.transcription import run_plan as _run_plan
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.settings import config as cfg

logger = logging.getLogger("whisper-api")


# ── Batch progress registry ──────────────────────────────────────────────────
# Optional per-request progress for the file-upload path: a client that sends a
# `progress_id` form field can poll GET /v1/audio/transcriptions/progress/<id>
# while its POST is in flight. Entries live only for the request (popped in the
# handler's finally); the cap + stale sweep below bound a client that invents
# ids and never posts. Plain dict + GIL: every writer does a single dict-entry
# update, and the poller only reads; the entry-creation sweep/eviction in
# _progress_set iterates over a snapshot (list(...)) so a concurrent pop on
# the loop thread can't blow up an executor-thread iteration.
_BATCH_PROGRESS: "dict[str, dict]" = {}
# progress_id → owner, remembered from the handler's seed so an entry that an
# executor-thread stage re-creates after a cap eviction keeps its owner stamp
# instead of coming back owner-less and readable/cancellable by any caller.
# Popped in the handlers' finally (alongside _BATCH_PROGRESS) and in the
# stale sweep.  Deliberately NOT popped in the cap eviction — that is
# exactly the moment an executor thread can re-create the entry and needs
# the stamp.
_PROGRESS_OWNER: "dict[str, str]" = {}
_BATCH_PROGRESS_MAX = 200
_BATCH_PROGRESS_STALE_S = 2 * 3600
# progress_id → monotonic time its run CLOSED (the handler's finally popped
# the entry). A stage thread that outlives the handler (a decode the client
# cancelled, a diarization step mid-flight on disconnect) keeps calling
# _progress_set; without this it would re-create the entry — owner-less,
# since _PROGRESS_OWNER was popped too — and leave it readable by any
# authenticated caller until the stale sweep. A closed id is a no-op in
# _progress_set until a fresh owner-stamped seed re-opens it. Bounded by the
# TTL sweep in _progress_close (every run end); each suppressed tick restamps
# the tombstone, so a stage thread still ticking keeps its id closed however
# long it outlives the close, and the TTL is never shorter than the stale
# sweep's.
_PROGRESS_CLOSED: "dict[str, float]" = {}
_PROGRESS_CLOSED_TTL_S = _BATCH_PROGRESS_STALE_S


def _progress_close(pid: "str | None") -> None:
    """Retire `pid`'s live progress entry: pop the three registries and
    tombstone the id so a straggling stage-thread tick cannot resurrect it.
    Every handler finally that used to pop the trio calls this instead."""
    if not pid:
        return
    _BATCH_PROGRESS.pop(pid, None)
    _PROGRESS_OWNER.pop(pid, None)
    _BATCH_CANCELLED.discard(pid)
    now = time.monotonic()
    _PROGRESS_CLOSED[pid] = now
    if len(_PROGRESS_CLOSED) > 64:
        for k in [k for k, t in list(_PROGRESS_CLOSED.items())
                  if now - t > _PROGRESS_CLOSED_TTL_S]:
            _PROGRESS_CLOSED.pop(k, None)
_PROGRESS_ID_RE = re.compile(r"\A[0-9a-f]{8,64}\Z")


# progress_id → job id: handlers that registered a job in jobs.py bind their
# progress_id here so every _progress_set call (all stages already flow
# through it) mirrors stage/progress into the central registry for free.
# last_text is deliberately NOT mirrored — job rows never carry transcript
# text (see jobs.jobs_snapshot's scrubbing contract).
_JOB_BY_PID: "dict[str, str]" = {}
_JOB_MIRROR_FIELDS = ("stage", "progress", "step", "model", "total_bytes")


def _claim_progress_id(progress_id) -> "str | None":
    """A request's opt-in progress id (see _BATCH_PROGRESS), or None. A
    malformed id is treated as absent — progress is a convenience, never a
    422. So is an id that is already in flight: the registries keyed on it
    are bare overwrites, and a colliding id would share one entry and one
    cancel flag with (and let this caller cancel) another request."""
    if not (isinstance(progress_id, str) and _PROGRESS_ID_RE.match(progress_id)):
        return None
    if progress_id in _BATCH_PROGRESS:
        logger.info("[progress] id already in flight — progress/cancel "
                    "disabled for this request")
        return None
    return progress_id


def _bind_job_pid(pid: "str | None", request_id: str) -> None:
    """Bind a claimed progress id to its jobs.py row (the mirror above); the
    handler's outer finally pops it."""
    if pid:
        _JOB_BY_PID[pid] = request_id
        jobs.job_update(request_id, progress_id=pid)

# progress_id → preload plan id, the same shape and lifetime as _JOB_BY_PID
# above and popped in the same finally. Bound once, right after the batch
# handler has fully resolved the stage plan.
#
# ONE hook, here, rather than four per-stage ones: every stage transition in
# the pipeline already flows through _progress_set, and it already carries a
# side effect of exactly this shape (the job mirror). Four hooks placed at the
# four stage entry points would drift the first time a stage moved.
_PLAN_BY_PID: "dict[str, str]" = {}

# progress_id → the run's server-owned plan (transcription/run_plan.py): expected and
# actual seconds per stage, per-language translation units, and the overall
# fraction + ETA the progress route publishes. Same shape and lifetime as
# _JOB_BY_PID (popped in the handlers' outer finally); _progress_set feeds
# every tick into it from whichever thread reported.
_RUN_PLAN_BY_PID: "dict[str, _run_plan.RunPlan]" = {}


def _plan_fields(pid: str) -> dict:
    """`plan` / `overall` / `eta_s` for the progress route — all None when
    the id has no plan (the admin prompt lab seeds entries without one)."""
    rp = _RUN_PLAN_BY_PID.get(pid)
    if rp is None:
        return {"plan": None, "overall": None, "eta_s": None}
    snap = rp.snapshot()
    return {
        "plan": snap["plan"],
        "overall": (round(snap["overall"], 4)
                    if snap["overall"] is not None else None),
        "eta_s": (round(snap["eta_s"], 1)
                  if snap["eta_s"] is not None else None),
    }


# ── Server jobs (durable job resource, transcription/jobs_store.py) ──────────────────
# Every batch run posted WITH a progress_id gets a row: `running` right after
# the progress seed, then its terminal state + the verbatim response payload
# from the handler's outer finally, so a client that lost its connection can
# list / re-attach to / fetch the run via GET /v1/jobs*. A ledger write never
# fails a run: every helper here swallows and logs.

def _jobs_enabled() -> bool:
    return bool(getattr(cfg, "JOBS_ENABLED", True))


def _jobs_ttl_s() -> float:
    return float(getattr(cfg, "JOBS_TTL_S", 259_200))


def _jobs_start(pid: "str | None", *, request_id: str, kind: str,
                user_id: "str | None", key_id: "str | None",
                model: "str | None", source_kind: str,
                source_name: "str | None", task: "str | None" = None,
                response_format: "str | None" = None) -> bool:
    """Insert the `running` row for `pid`. False when no row was written
    (no id, feature off, the store is unavailable, or the id already names
    another caller's row — the store refuses to replace that one)."""
    if not pid or not _jobs_enabled():
        return False
    try:
        # prune_every=0: retention is `_jobs_retention_loop`'s job (hourly,
        # off the loop); the store's lazy prune would run on the event loop.
        return bool(_jobs_store.start(
            job_id=pid, request_id=request_id, kind=kind,
            user_id=user_id, key_id=key_id, model=model,
            source_kind=source_kind, source_name=source_name, task=task,
            response_format=response_format, ttl_s=_jobs_ttl_s(),
            max_rows=int(getattr(cfg, "JOBS_MAX_ROWS", 2000)),
            max_bytes=int(getattr(cfg, "JOBS_MAX_BYTES", 2_000_000_000)),
            prune_every=0))
    except Exception as e:  # noqa: BLE001 — never fail a run on the ledger
        logger.warning("[jobs] could not record job start: %s", e)
        return False


async def _jobs_start_async(pid: "str | None", **kw) -> bool:
    """Off-loop `_jobs_start`: the insert takes jobs_store's lock, which a
    worker thread can hold for a while (a big finish, patch_result, the
    hourly prune) — on the loop that would stall every request and stream.
    A cancellation landing on the await cannot stop the insert already on
    the thread, and the caller never learns it has a row to finish, so the
    row is closed here once the insert lands."""
    if not pid or not _jobs_enabled():
        return False
    fut = asyncio.ensure_future(asyncio.to_thread(_jobs_start, pid, **kw))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        def _close(f: "asyncio.Future") -> None:
            if not f.cancelled() and f.exception() is None and f.result():
                f.get_loop().run_in_executor(None, lambda: _jobs_finish_sync(
                    pid, status="error", error="request aborted"))
        fut.add_done_callback(_close)
        raise


def _job_error_text(status: str, exc: "BaseException | None",
                    fallback: str = "transcription failed") -> "str | None":
    """Client-safe error for a job row: a curated 4xx detail verbatim, any
    other failure as the generic text the response carried. None when the
    run was cancelled (not an error)."""
    if status == "cancelled":
        return None
    if (isinstance(exc, HTTPException) and isinstance(exc.detail, str)
            and exc.status_code < 500):
        return exc.detail
    return fallback


def _jobs_finish_sync(pid: str, *, status: str, payload=None,
                      error: "str | None" = None, stages=None, plan=None,
                      model: "str | None" = None,
                      task: "str | None" = None) -> None:
    """Stamp the terminal state. `payload` is the response object exactly as
    the handler returned it (dict, or the `text` format's str); it is stored
    only for status "ok". Blocking SQLite — call off the loop for big runs."""
    state = {"ok": "done", "cancelled": "cancelled"}.get(status, "failed")
    try:
        result = None
        if state == "done" and payload is not None:
            result = _jsonable_encoder(payload)
        if not _jobs_store.finish(job_id=pid, state=state, error=error,
                                  result=result, stages=stages, plan=plan,
                                  model=model, task=task, ttl_s=_jobs_ttl_s()):
            # Evicted meanwhile (row cap / byte cap / TTL): a re-attaching
            # client gets 404, and this line is what explains it.
            logger.info("[jobs] row %s gone before finish — result not stored",
                        pid[:8])
    except Exception as e:  # noqa: BLE001
        logger.warning("[jobs] could not record job end: %s", e)


async def _jobs_finish(pid: str, **kw) -> None:
    """Off-loop `_jobs_finish_sync` (a verbose_json payload can be MBs).
    shield: a handler being unwound must not abort a write already on the
    thread — the row would stay `running` forever."""
    try:
        await asyncio.shield(asyncio.to_thread(_jobs_finish_sync, pid, **kw))
    except Exception as e:  # noqa: BLE001
        logger.warning("[jobs] could not record job end: %s", e)


def _provisional_stages(*, is_url: bool, separate: "bool | None",
                        diarize: "bool | None",
                        translate_to: "str | None") -> "list[str]":
    """The stage list a request implies BEFORE its knobs are resolved
    against the caller's identity — the plan needs a denominator from the
    first poll on; set_stages() replaces it once the verdicts are in."""
    sep = separate if separate is not None else bool(
        getattr(cfg, "SEPARATE_BGM", False))
    diar = diarize if diarize is not None else bool(
        getattr(cfg, "DIARIZE", False))
    tt = (translate_to.strip() if translate_to is not None
          else (getattr(cfg, "TRANSLATE_TO", "") or ""))
    return [*(["downloading"] if is_url else []),
            *(["separating"] if sep else []),
            "transcribing",
            *(["diarizing"] if diar else []),
            *(["translating"] if tt else [])]


def _progress_set(pid: "str | None", **fields) -> None:
    """Merge `fields` into the progress entry for `pid` (no-op without one)."""
    if not pid:
        return
    if fields.get("owner") is not None:
        # A fresh handler seed re-opens an id its previous run closed.
        _PROGRESS_CLOSED.pop(pid, None)
    elif pid in _PROGRESS_CLOSED:
        # A stage thread of a run that already closed: nothing to update,
        # and re-creating the entry would leave it owner-less.
        _PROGRESS_CLOSED[pid] = time.monotonic()
        return
    _job_id = _JOB_BY_PID.get(pid)
    if _job_id:
        _mirror = {k: fields[k] for k in _JOB_MIRROR_FIELDS
                   if fields.get(k) is not None}
        _new_stage = fields.get("stage")
        _prev = _BATCH_PROGRESS.get(pid, {}).get("stage")
        if _new_stage and _prev and _new_stage != _prev:
            # A stage TRANSITION resets the derived columns the new stage did
            # not (yet) report — without this the job row keeps the previous
            # stage's last progress/step/model/total_bytes (a stale ~100%
            # bar, the decode model shown for the whole of diarizing/
            # translating, the download's byte total for the rest of the
            # request), because both this mirror and jobs.job_update skip
            # plain Nones. The first seed is NOT a transition: there is no
            # previous stage's stale state to clear, only job_start's model
            # to preserve.
            for k in ("progress", "step", "model", "total_bytes"):
                _mirror.setdefault(k, jobs.CLEAR)
        jobs.job_update(_job_id, **_mirror)
    _stage = fields.get("stage") or _BATCH_PROGRESS.get(pid, {}).get("stage")
    if _stage and pid in _PLAN_BY_PID:
        # Advances the plan's cursor and warms the next stage's model. Sync,
        # never awaits, and swallows everything internally — this runs on
        # executor threads (the decode, the demix, the pyannote hook) and
        # progress must never break a request.
        # Every tick, not only transitions: on_stage_start restamps the TTL
        # before its monotone-cursor check, so a stage longer than
        # MODEL_PRELOAD_WARM_TTL_S keeps the plan alive without enqueueing
        # anything.
        preload.on_stage_start(_PLAN_BY_PID[pid], _stage)
    if fields.get("owner") is not None:
        _PROGRESS_OWNER[pid] = fields["owner"]
    entry = _BATCH_PROGRESS.get(pid)
    if entry is None:
        if pid in _PROGRESS_CLOSED and fields.get("owner") is None:
            # The loop-thread close landed after the check at the top (this
            # tick ran job_update / on_stage_start on an executor thread in
            # between): same no-op, never an owner-less resurrection.
            return
        # Snapshot (list(...)) before iterating: this branch runs on executor
        # threads too, and the handler's finally pops entries on the loop
        # thread — iterating the live dict would raise "dictionary changed
        # size during iteration" out of a healthy stage callback.
        now = time.monotonic()
        for k in [k for k, v in list(_BATCH_PROGRESS.items())
                  if now - v.get("updated", 0) > _BATCH_PROGRESS_STALE_S]:
            _BATCH_PROGRESS.pop(k, None)
            _PROGRESS_OWNER.pop(k, None)
        if len(_BATCH_PROGRESS) >= _BATCH_PROGRESS_MAX:
            _snap = list(_BATCH_PROGRESS.items())
            if _snap:
                oldest = min(_snap, key=lambda kv: kv[1].get("updated", 0))[0]
                _BATCH_PROGRESS.pop(oldest, None)
        entry = _BATCH_PROGRESS[pid] = (
            {"owner": _PROGRESS_OWNER[pid]} if pid in _PROGRESS_OWNER else {})
    entry.update(fields)
    entry["updated"] = time.monotonic()
    _rp = _RUN_PLAN_BY_PID.get(pid)
    if _rp is not None:
        # Same stance as the preload hook above: runs on executor threads,
        # and progress must never break a request.
        try:
            _rp.tick(stage=entry.get("stage"),
                     progress=fields.get("progress"),
                     target=fields.get("target"),
                     target_progress=fields.get("target_progress"),
                     total_bytes=fields.get("total_bytes"),
                     # Sticky like `stage`: a write that omits `step` (the
                     # keep_video task's `video=` ticks) means "unchanged",
                     # not "no step" — every step ends with an explicit None.
                     step=entry.get("step"))
        except Exception:  # noqa: BLE001
            pass


# Cooperative cancellation for in-flight batch requests: POST
# /v1/audio/transcriptions/cancel/<id> flags the id here, and the handler's
# stage callbacks (which already fire every demix chunk / decoded segment /
# pyannote step) poll the flag and abort. Closing the HTTP connection alone
# does NOT stop the work — the stages run in executor threads that outlive a
# cancelled handler task. Only ids with a live _BATCH_PROGRESS entry can be
# flagged, and the handler's finally discards, so the set stays bounded by
# the number of in-flight requests.
_BATCH_CANCELLED: "set[str]" = set()


class _ClientCancelled(Exception):
    """The client cancelled this request via the cancel endpoint."""


def _cancel_requested(pid: "str | None") -> bool:
    return bool(pid) and pid in _BATCH_CANCELLED


def _check_cancelled(pid: "str | None") -> None:
    if _cancel_requested(pid):
        raise _ClientCancelled()


def _progress_entry_for(progress_id: str, user: dict) -> "dict | None":
    """The _BATCH_PROGRESS entry for `progress_id` IF this caller may see it.
    An entry stamped with an `owner` that is not this caller reads exactly
    like a miss (no existence oracle) unless the caller is an admin (the
    /stats activity popover cancels other users' jobs); an owner-less entry
    (tests / legacy seeds) stays accessible to any authenticated caller."""
    entry = _BATCH_PROGRESS.get(progress_id)
    if entry is None:
        return None
    _owner = entry.get("owner")
    if (_owner and not user.get("is_admin")
            and _owner not in (user.get("user_id"), user.get("key_id"))):
        return None
    return entry


def _progress_payload(pid: str, entry: dict) -> dict:
    """The progress route's wire shape for a live entry — also embedded
    under `progress` by GET /v1/jobs/{id} while the run is in flight."""
    return {
        "stage": entry.get("stage"),
        "progress": entry.get("progress"),
        "duration": entry.get("duration"),
        # Rich run-panel fields (all optional, stage-scoped): seconds of
        # audio decoded, the diarization pipeline's current step, the last
        # decoded segment's text, and the active stage's model/device.
        "position": entry.get("position"),
        "step": entry.get("step"),
        "last_text": entry.get("last_text"),
        "model": entry.get("model"),
        "device": entry.get("device"),
        "compute": entry.get("compute"),
        # Fraction of the audio the VAD kept (0..1), set once decoding starts;
        # null when the filter was off. Persists for the rest of the run.
        "vad_retained": entry.get("vad_retained"),
        # URL flow, downloading stage: bytes expected (progress is the
        # downloaded fraction when this is known; null on fragmented streams).
        "total_bytes": entry.get("total_bytes"),
        # Requested stages this server declined to run (feature disabled) —
        # "separating" / "diarizing" / "translating". Set the moment the skip
        # is known, so the client's rail can say "skipped" instead of guessing.
        "skipped": entry.get("skipped"),
        # Translation ticks: the language being translated and how far
        # along it is (0..1 within that language).
        "target": entry.get("target"),
        "target_progress": entry.get("target_progress"),
        # keep_video runs: the secondary video download's own state (see
        # _video_state) — null unless a video was requested.
        "video": entry.get("video"),
        # The server-owned plan (transcription/run_plan.py): per-stage expected /
        # actual seconds, per-language units, and the overall fraction +
        # ETA the client renders verbatim.
        **_plan_fields(pid),
    }


def _reset_for_tests() -> None:
    """Empty every progress / cancel / binding registry in place (a caller
    holding a reference sees it empty)."""
    _BATCH_PROGRESS.clear()
    _PROGRESS_OWNER.clear()
    _PROGRESS_CLOSED.clear()
    _JOB_BY_PID.clear()
    _PLAN_BY_PID.clear()
    _RUN_PLAN_BY_PID.clear()
    _BATCH_CANCELLED.clear()
