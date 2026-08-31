"""Model preloading: plan registry, warm leases and the single load worker.

The point is to hide a model load inside work the job is already doing. A
transcribe job that will diarize spends minutes decoding first; loading the
pyannote pipeline during those minutes costs nothing, while loading it after
the decode adds its full wall-clock to the job. Two callers drive the same
mechanism: a client that POSTs its intended pipeline up front
(``preload_routes``) and the server itself, which registers the resolved stage
plan of every batch job and advances it from ``main._progress_set``.

Precedence: JOB LEASE > WARM LEASE > nothing. A warm lease makes a model
ineligible for the idle evictors and for preload-driven eviction, and that is
ALL it does. It never forces residency, never pins a model against a job that
needs the memory, and never gates a loader — so a job is never delayed by
warmth, and the worst case for the whole feature is that it silently does
nothing and every stage loads in-band exactly as before.

Threading: guarded by a ``threading.Lock``, deliberately NOT an asyncio lock.
The stage-ahead entry point ``on_stage_start`` is called from ``_progress_set``,
a SYNC function reached from executor threads (the decode, the demix, the
pyannote hook all report from there). The lock is never held across an await:
every path snapshots under it and acts outside it. Combined with the single
worker task — which holds at most one FAMILY lock at a time, and no code path
in the tree holds two — there is no lock ordering to get wrong.

Families are ``whisper`` / ``diarization`` / ``separation`` / ``translation``.
There is deliberately NO vad family: Silero ships inside faster-whisper, runs
on the CPU, has no registry entry and no evictor, so there is nothing to warm
and nothing a warm lease could protect.

Failure stance: ``register_plan`` never raises and worker exceptions are caught
and logged. Either way the plan degrades to "the job loads it in-band as
today". The most likely real-world failure is therefore the feature quietly
doing nothing, which is why ``diagnostics()`` is surfaced in /stats and the
worker announces itself at startup.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field

import config as cfg
import jobs
import model_sizes
import system_stats

logger = logging.getLogger("whisper-server")

FAMILIES = ("whisper", "diarization", "separation", "translation")

# Canonical pipeline order. Also the stage-ahead cursor's coordinate system:
# the worker only ever looks FORWARD from the running stage.
_FAMILY_STAGE = {"separation": 0, "whisper": 1, "diarization": 2,
                 "translation": 3}

# Progress stage name → stage index. `waiting` and `analyzing` are sub-stages
# of the decode (main.py's semaphore wait and its post-decode analysis), not
# stages of their own — mapping them anywhere else would make them read as
# unknown stages and stall the cursor mid-pipeline.
STAGE_INDEX = {
    "separating": 0,
    "transcribing": 1,
    "waiting": 1,
    "analyzing": 1,
    "diarizing": 2,
    "translating": 3,
}

# Both hardcoded. A plan is one job's pipeline (at most four models), and the
# queue only ever holds work the single worker will get to in seconds — a
# larger bound would just defer the same "the box is behind" answer.
_MAX_PLANS = 8
_MAX_QUEUE = 8

_lock = threading.Lock()
_plans: "dict[str, Plan]" = {}
# stats_key → expiry (monotonic). Derived state: recomputed from _plans as
# their union, never refcounted. That is what makes the cascade correct — a
# plan expiring drops exactly the keys no OTHER live plan still wants, with no
# per-key counter that a lost decrement could leak.
_warm: "dict[str, float]" = {}

_queue: "asyncio.Queue | None" = None
_loop: "asyncio.AbstractEventLoop | None" = None
_worker: "asyncio.Task | None" = None
# True while the worker is inside a load: the difference between reporting
# "loading" and "queued" to the client.
_busy = False

_SWEEP_S = 15


@dataclass
class Plan:
    """One job's (or one client's) intended pipeline.

    ``entries`` are (family, id) pairs in canonical stage order and ``stages``
    is the matching stage index for each, so the cursor can be compared
    against an entry without re-deriving anything.
    """
    plan_id: str
    user_id: str
    entries: "list[tuple[str, str]]"
    stages: "list[int]"
    cursor: int = -1
    warmed: "set[str]" = field(default_factory=set)
    expires_mono: float = 0.0
    # Keys handed to the queue and not yet acted on — the double-enqueue
    # guard for a re-POST and for a replayed stage.
    inflight: "set[str]" = field(default_factory=set)
    job_id: "str | None" = None
    dead: bool = False
    # False = the client drives its own pipeline and will POST again; the plan
    # still holds its warm leases, the server just never advances it.
    stage_ahead: bool = True


# =============================================================================
# Keys and residency
# =============================================================================

def normalize_id(family: str, model_id: str) -> str:
    """The id as the owning module CACHES it.

    Only separation differs from what a client types: audio-separator keys its
    singleton by on-disk FILENAME while the allowlist holds friendly names.
    That mapping lives here (and nowhere else) precisely because /v1/me open-
    coded it once already — a second copy would disagree for exactly the UVR
    case the moment one of them changed."""
    model_id = (model_id or "").strip()
    if family == "separation":
        return model_id if "." in model_id else f"{model_id}.onnx"
    return model_id


def stats_key(family: str, model_id: str) -> str:
    """The system_stats registry key — the shared namespace the warm leases,
    the size ledger and the idle evictors all speak."""
    mid = normalize_id(family, model_id)
    if family == "diarization":
        return f"pyannote:{mid}"
    if family == "separation":
        return f"uvr:{mid}"
    if family == "translation":
        return f"gguf:{mid}"
    return mid


def is_resident(family: str, model_id: str) -> bool:
    """Is this model loaded right now?

    The single answer for all four families. /v1/me and /v1/models open-coded
    four different versions of this question; they call here now, so a `loaded`
    flag cannot drift from what the preloader believes."""
    mid = normalize_id(family, model_id)
    if not mid:
        return False
    try:
        if family == "whisper":
            import main
            return mid in main._loaded_models
        if family == "translation":
            import translation
            return mid in translation._models
        if family == "diarization":
            import diarization
            key = diarization._pipeline_key
            return bool(key) and key[0] == mid
        if family == "separation":
            import bgm_separation
            key = bgm_separation._separator_key
            return bool(key) and key[0] == mid
    except Exception:  # noqa: BLE001 — a residency probe must never raise
        return False
    return False


# =============================================================================
# Warm leases
# =============================================================================

def _ttl() -> int:
    return int(getattr(cfg, "MODEL_PRELOAD_WARM_TTL_S", 180) or 180)


def _enabled() -> bool:
    return bool(getattr(cfg, "MODEL_PRELOAD_ENABLED", True))


def _recompute_warm_locked() -> None:
    """Rebuild _warm as the union of the live plans' keys. Caller holds _lock.

    Cheap (at most 8 plans x 4 entries) and total, which is the whole point:
    no incremental bookkeeping means no way to leak a lease."""
    fresh: "dict[str, float]" = {}
    for plan in _plans.values():
        if plan.dead:
            continue
        for fam, mid in plan.entries:
            k = stats_key(fam, mid)
            if plan.expires_mono > fresh.get(k, 0.0):
                fresh[k] = plan.expires_mono
    _warm.clear()
    _warm.update(fresh)


def is_warm(key: str) -> bool:
    """True while some live plan still expects to use this stats key.

    Registered as ``system_stats.set_warm_predicate`` so the four idle evictors
    can consult it without importing this module (which would close a cycle in
    all four). Never raises."""
    try:
        with _lock:
            exp = _warm.get(key)
            return exp is not None and exp > time.monotonic()
    except Exception:  # noqa: BLE001 — an evictor must never break on this
        return False


# =============================================================================
# Admission
# =============================================================================

def _stage_enabled(family: str) -> bool:
    if family == "diarization":
        return bool(getattr(cfg, "DIARIZATION_ENABLED", False))
    if family == "separation":
        return bool(getattr(cfg, "BGM_SEPARATION_ENABLED", False))
    if family == "translation":
        return bool(getattr(cfg, "TRANSLATION_ENABLED", False))
    return family == "whisper"


def _placement(family: str) -> "tuple[str, str]":
    """(device, compute_type) exactly as the family's loader registers it —
    the size ledger is keyed on that tuple, so a mismatch here silently turns
    every fit check into `size_unknown`."""
    if family == "whisper":
        return ((getattr(cfg, "MODEL_DEVICE", "cpu") or "cpu"),
                (getattr(cfg, "MODEL_COMPUTE_TYPE", "") or ""))
    if family == "diarization":
        import diarization
        return (diarization._resolve_device(), "torch")
    if family == "separation":
        import bgm_separation
        return (bgm_separation._resolve_device(), "onnx")
    import translation
    return (translation._resolve_device(), "gguf")


def _reserve_bytes(device: str) -> int:
    if (device or "").startswith("cuda"):
        mb = int(getattr(cfg, "MODEL_PRELOAD_VRAM_RESERVE_MB", 1024) or 0)
    else:
        mb = int(getattr(cfg, "MODEL_PRELOAD_RAM_RESERVE_MB", 2048) or 0)
    return mb * 1024 * 1024


def _family_busy(family: str, model_id: str) -> bool:
    """Would admitting this model collide with a RUNNING job?

    The singletons are the sharp case: loading a different pipeline while the
    resident one is job-leased goes down `_drop_locked(force=True)`'s ORPHAN
    path, which keeps both in memory until the job drains. That is the correct
    behaviour for a real request and exactly the wrong one for a speculative
    warm-up, so the ladder refuses instead."""
    mid = normalize_id(family, model_id)
    try:
        if family == "whisper":
            import main
            # Held across a whisper load. A preload must never be the reason a
            # job waits on it, so a busy lock is a refusal, not a queue.
            return main._model_load_lock.locked()
        if family == "diarization":
            import diarization
            key = diarization._pipeline_key
            if not key or key[0] == mid:
                return False
            return bool(diarization._leases.get(key[0], 0)
                        or diarization._orphans.get(key[0], 0))
        if family == "separation":
            import bgm_separation
            key = bgm_separation._separator_key
            if not key or key[0] == mid:
                return False
            return bool(bgm_separation._leases.get(key[0], 0)
                        or bgm_separation._orphans.get(key[0], 0))
        if family == "translation":
            import translation
            cap = max(1, int(getattr(cfg, "TRANSLATION_MAX_LOADED_MODELS", 1) or 1))
            if len(translation._models) < cap:
                return False
            return all(translation._active.get(r, 0)
                       for r in translation._models)
    except Exception:  # noqa: BLE001 — unknown state is not a reason to load
        return True
    return False


def _idle_peer(family: str, model_id: str) -> "str | None":
    """An idle, UNLEASED, UNWARMED peer of the same family we could drop to
    make room, or None. Returns the peer's own id (not its stats key)."""
    mid = normalize_id(family, model_id)
    try:
        if family == "whisper":
            import main
            for name in main._loaded_models:
                if name == mid or main._model_leases.get(name, 0):
                    continue
                if not system_stats.is_warm(stats_key(family, name)):
                    return name
            return None
        if family == "translation":
            import translation
            for ref in translation._models:
                if ref == mid or translation._active.get(ref, 0):
                    continue
                if not system_stats.is_warm(stats_key(family, ref)):
                    return ref
            return None
        if family == "diarization":
            import diarization
            key = diarization._pipeline_key
        else:
            import bgm_separation
            key = bgm_separation._separator_key
        if not key or key[0] == mid:
            return None
        peer = key[0]
        if _family_busy(family, model_id):
            return None
        if system_stats.is_warm(stats_key(family, peer)):
            return None
        return peer
    except Exception:  # noqa: BLE001 — no peer is the conservative answer
        return None


def _admit(family: str, model_id: str) -> "tuple[str, str | None]":
    """The admission ladder: (state, reason) with state in
    resident|loading|queued|deferred.

    Run TWICE per entry — once at enqueue so the response can say something
    honest, and again at dequeue, because residency, leases and free VRAM all
    move in between. The DEQUEUE verdict is the one that decides what actually
    happens; the enqueue verdict is a forecast."""
    if not _enabled():
        return ("deferred", "disabled")
    if family not in FAMILIES:
        return ("deferred", "not_allowed")
    if not normalize_id(family, model_id):
        return ("deferred", "not_allowed")
    if not _stage_enabled(family):
        return ("deferred", "stage_disabled")

    if is_resident(family, model_id):
        # Residency is the answer AND a courtesy: the touch restarts the idle
        # clock, so a plan naming an already-loaded model keeps it loaded even
        # if the job that needs it is still minutes away.
        key = stats_key(family, model_id)
        system_stats.touch_loaded_model(key)
        with _lock:
            if _warm.get(key, 0.0) < time.monotonic() + _ttl():
                _warm[key] = time.monotonic() + _ttl()
        return ("resident", None)

    if _family_busy(family, model_id):
        return ("deferred", "family_busy")

    device, compute = _placement(family)
    ok, reason = model_sizes.fits(
        stats_key(family, model_id), device, compute,
        reserve_bytes=_reserve_bytes(device))
    if ok is True:
        return (_pending_state(), None)
    if (reason in ("insufficient_vram", "insufficient_ram")
            and bool(getattr(cfg, "MODEL_PRELOAD_EVICT_IDLE_MODELS", True))
            and _idle_peer(family, model_id) is not None):
        return (_pending_state(), None)
    return ("deferred", reason or "size_unknown")


def _pending_state() -> str:
    q = _queue
    if _busy or (q is not None and not q.empty()):
        return "queued"
    return "loading"


# =============================================================================
# Registration
# =============================================================================

def derive_plan_id(user_id: str, entries: "list[tuple[str, str]]") -> str:
    """Stable id for a caller that did not supply one, so a client repeating
    the same intent restamps its plan instead of accumulating plans."""
    raw = (user_id or "") + "|" + "|".join(
        f"{f}:{normalize_id(f, m)}" for f, m in sorted(entries))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def register_plan(user_id: "str | None",
                  entries: "list[tuple[str, str]]",
                  *,
                  plan_id: "str | None" = None,
                  denied: "dict[tuple[str, str], str] | None" = None,
                  stage_ahead: bool = True) -> dict:
    """Create or restamp a plan; returns the endpoint's response body.

    NEVER raises. Every failure mode — an unknown family, a full registry, a
    disabled feature, a loader that will later blow up — resolves to a
    `deferred` row, because the fallback is the behaviour the server had
    before this module existed.

    `denied` carries per-entry verdicts the CALLER already made (a model the
    request's allowlist refuses); those entries are reported with the caller's
    reason and never join the plan."""
    try:
        return _register_plan(user_id, entries, plan_id=plan_id,
                              denied=denied or {}, stage_ahead=stage_ahead)
    except Exception as e:  # noqa: BLE001 — a preload must never fail a request
        logger.error("[preload] register_plan failed: %s", e)
        return {
            "plan_id": plan_id or "",
            "expires_in_s": 0,
            "models": [{"family": f, "id": m, "state": "deferred",
                        "reason": "disabled"} for f, m in entries],
        }


def _register_plan(user_id, entries, *, plan_id, denied, stage_ahead) -> dict:
    now = time.monotonic()
    ttl = _ttl()
    kept = [(f, m) for f, m in entries if (f, m) not in denied]
    kept.sort(key=lambda e: _FAMILY_STAGE.get(e[0], 99))
    pid = (plan_id or "").strip() or derive_plan_id(user_id or "", kept)

    results: "list[dict]" = []
    to_enqueue: "list[tuple[str, str, str]]" = []
    with _lock:
        plan = _plans.get(pid)
        if plan is None:
            if len(_plans) >= _MAX_PLANS:
                # Drop the plan closest to expiry rather than refusing: the
                # newest intent is the one a job is about to act on.
                oldest = min(_plans, key=lambda k: _plans[k].expires_mono)
                _drop_plan_locked(oldest, "registry full")
            plan = Plan(plan_id=pid, user_id=user_id or "", entries=list(kept),
                        stages=[_FAMILY_STAGE.get(f, 99) for f, _ in kept],
                        expires_mono=now + ttl, stage_ahead=stage_ahead)
            _plans[pid] = plan
        else:
            # A repeat POST restamps and merges; it does not reset the cursor,
            # so a server-driven plan already three stages in is not rewound by
            # a late client POST of the same intent.
            plan.dead = False
            plan.expires_mono = now + ttl
            for f, m in kept:
                if (f, m) not in plan.entries:
                    plan.entries.append((f, m))
                    plan.stages.append(_FAMILY_STAGE.get(f, 99))
        _recompute_warm_locked()

    for fam, mid in kept:
        key = stats_key(fam, mid)
        state, reason = _admit(fam, mid)
        if state == "resident":
            with _lock:
                plan.warmed.add(key)
        elif state in ("loading", "queued"):
            with _lock:
                q = _queue
                depth = q.qsize() if q is not None else 0
                if key in plan.warmed or key in plan.inflight:
                    # Already warmed or already queued by an earlier POST /
                    # stage advance — restamping must not enqueue it twice.
                    state, reason = ("resident" if key in plan.warmed
                                     else "queued"), None
                elif q is None or depth >= _MAX_QUEUE:
                    state, reason = "deferred", "queue_full"
                else:
                    plan.inflight.add(key)
                    to_enqueue.append((pid, fam, mid))
        row = {"family": fam, "id": mid, "state": state}
        if reason:
            row["reason"] = reason
        results.append(row)

    for fam, mid in entries:
        if (fam, mid) in denied:
            results.append({"family": fam, "id": mid, "state": "deferred",
                            "reason": denied[(fam, mid)]})

    with _lock:
        if plan.job_id is None and not _all_warmed_locked(plan):
            # Visible in /stats: warming is real GPU work and an operator
            # watching the activity cluster must be able to see it happen.
            plan.job_id = jobs.job_start(
                "preload", user=user_id,
                model=", ".join(stats_key(f, m) for f, m in plan.entries)[:200],
                detail=f"plan {pid[:8]}")
        _end_job_if_settled_locked(plan)

    for item in to_enqueue:
        _enqueue_threadsafe(item)

    return {"plan_id": pid, "expires_in_s": ttl, "models": results}


def cancel_plan(plan_id: str) -> bool:
    """Mark a plan dead. The worker skips dead-plan items at dequeue rather
    than the queue being drained: an asyncio.Queue has no removal primitive and
    the items are three-tuples the worker discards in microseconds.

    A load already inside an executor thread is NOT cancellable and is not
    cancelled — it finishes and registers normally, which at worst leaves a
    model loaded that nobody asked for and at best hands the size ledger a free
    measurement of it."""
    with _lock:
        plan = _plans.get(plan_id)
        if plan is None:
            return False
        _drop_plan_locked(plan_id, "cancelled")
        return True


def _drop_plan_locked(plan_id: str, why: str) -> None:
    plan = _plans.pop(plan_id, None)
    if plan is None:
        return
    plan.dead = True
    if plan.job_id:
        jobs.job_end(plan.job_id)
        plan.job_id = None
    logger.info("[preload] plan %s dropped (%s)", plan_id[:8], why)
    _recompute_warm_locked()


def _all_warmed_locked(plan: Plan) -> bool:
    return all(stats_key(f, m) in plan.warmed for f, m in plan.entries)


def _end_job_if_settled_locked(plan: Plan) -> None:
    """Close the /stats job row once every entry is loaded. A DEFERRED entry
    is not settled — a later stage advance retries it, and the row should stay
    up while that is still possible."""
    if plan.job_id and _all_warmed_locked(plan):
        jobs.job_end(plan.job_id)
        plan.job_id = None


# =============================================================================
# Stage-ahead
# =============================================================================

def on_stage_start(plan_id: str, stage: str) -> None:
    """A stage of the owning job just started: advance the cursor and warm the
    next model.

    MUST NOT await — the only caller is ``main._progress_set``, which runs on
    executor threads. Everything after the lock is a ``call_soon_threadsafe``
    hand-off to the loop, and the whole body is wrapped: progress reporting
    must never break a request, the stance every progress callback in this tree
    already takes."""
    try:
        idx = STAGE_INDEX.get(stage)
        if idx is None:
            return
        item = None
        with _lock:
            plan = _plans.get(plan_id)
            if plan is None or plan.dead or not plan.stage_ahead:
                return
            # Restamp on every stage start, advancing or not: a long job keeps
            # its plan alive for free, which is the whole reason the TTL can be
            # as short as three minutes.
            plan.expires_mono = time.monotonic() + _ttl()
            _recompute_warm_locked()
            if idx <= plan.cursor:
                # Monotone cursor: a replayed or out-of-order stage enqueues
                # nothing, so a job that re-reports "separating" after
                # diarizing cannot walk the plan backwards.
                return
            plan.cursor = idx
            for i, (fam, mid) in enumerate(plan.entries):
                if plan.stages[i] <= plan.cursor:
                    continue
                key = stats_key(fam, mid)
                if key in plan.warmed or key in plan.inflight:
                    continue
                q = _queue
                if q is None or q.qsize() >= _MAX_QUEUE:
                    return
                plan.inflight.add(key)
                item = (plan_id, fam, mid)
                break
        if item is not None:
            _enqueue_threadsafe(item)
    except Exception:  # noqa: BLE001 — never break the progress callback
        pass


def _enqueue_threadsafe(item: "tuple[str, str, str]") -> None:
    loop, q = _loop, _queue
    if loop is None or q is None or loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(q.put_nowait, item)
    except RuntimeError:
        # The loop closed between the check and the call — shutdown racing a
        # last progress callback. Nothing to warm on a dying process.
        pass


# =============================================================================
# Worker
# =============================================================================

async def _worker_loop() -> None:
    """Single consumer. One task, not a pool: serialised loads keep the NVML
    delta clean (the same reason ``main._model_load_lock`` exists), bound
    preload concurrency to exactly one, and give the lifespan one cancellation
    point."""
    global _busy
    while True:
        item = await _queue.get()
        _busy = True
        try:
            await _handle(item)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the worker must survive
            logger.error("[preload] load failed for %s/%s: %s",
                         item[1], item[2], e)
        finally:
            _busy = False
            plan_id, fam, mid = item
            with _lock:
                plan = _plans.get(plan_id)
                if plan is not None:
                    plan.inflight.discard(stats_key(fam, mid))
                    _end_job_if_settled_locked(plan)


async def _handle(item: "tuple[str, str, str]") -> None:
    plan_id, family, model_id = item
    key = stats_key(family, model_id)
    with _lock:
        plan = _plans.get(plan_id)
        if plan is None or plan.dead or key in plan.warmed:
            return
    # Second run of the ladder. State moved while the item sat in the queue —
    # the job may have loaded the model itself, a peer may have taken the VRAM,
    # or a lease may have appeared. This verdict is the one that acts.
    state, reason = _admit(family, model_id)
    if state == "resident":
        with _lock:
            plan = _plans.get(plan_id)
            if plan is not None:
                plan.warmed.add(key)
        return
    if state == "deferred":
        logger.info("[preload] %s deferred: %s", key, reason)
        return

    if not is_resident(family, model_id):
        peer = None
        if bool(getattr(cfg, "MODEL_PRELOAD_EVICT_IDLE_MODELS", True)):
            device, compute = _placement(family)
            ok, _r = model_sizes.fits(
                key, device, compute, reserve_bytes=_reserve_bytes(device))
            if ok is not True:
                peer = _idle_peer(family, model_id)
        if peer is not None:
            await _evict(family, peer)

    t0 = time.perf_counter()
    await _load(family, model_id)
    logger.info("[preload] warmed %s in %.1fs", key, time.perf_counter() - t0)
    with _lock:
        plan = _plans.get(plan_id)
        if plan is not None:
            plan.warmed.add(key)


async def _load(family: str, model_id: str) -> None:
    """Load without a lease. A preload deliberately takes no job lease: the
    model must stay evictable the moment a real request needs the memory."""
    mid = normalize_id(family, model_id)
    if family == "whisper":
        import main
        await main._get_or_load_model(mid)
    elif family == "diarization":
        import diarization
        await diarization._get_pipeline(mid)
    elif family == "separation":
        import bgm_separation
        await bgm_separation._get_separator(mid)
    elif family == "translation":
        import translation
        await translation._get_model(mid)


async def _evict(family: str, peer_id: str) -> None:
    """Drop one idle peer. Each branch takes at most its OWN family lock, and
    nothing in the tree holds two — that is the deadlock proof."""
    logger.info("[preload] evicting idle %s to make room",
                stats_key(family, peer_id))
    if family == "whisper":
        import main
        async with main._model_load_lock:
            main._drop_loaded_model(peer_id)
    elif family == "translation":
        import translation
        async with translation._lock:
            translation._drop_locked(peer_id)
    elif family == "diarization":
        import diarization
        await diarization.drop_pipeline(force=False)
    elif family == "separation":
        import bgm_separation
        await bgm_separation.drop_separator(force=False)


# =============================================================================
# Lifecycle
# =============================================================================

async def start() -> None:
    """Bind the loop, open the queue and start the worker. Called from
    lifespan; safe to call twice."""
    global _loop, _queue, _worker
    _loop = asyncio.get_running_loop()
    if _queue is None:
        _queue = asyncio.Queue()
    if _worker is None or _worker.done():
        _worker = asyncio.create_task(_worker_loop())
    system_stats.set_warm_predicate(is_warm)
    logger.info(
        "[preload] worker started (enabled=%s, ttl=%ds, reserve=%d/%d MB, "
        "evict_idle=%s)",
        _enabled(), _ttl(),
        int(getattr(cfg, "MODEL_PRELOAD_VRAM_RESERVE_MB", 0) or 0),
        int(getattr(cfg, "MODEL_PRELOAD_RAM_RESERVE_MB", 0) or 0),
        bool(getattr(cfg, "MODEL_PRELOAD_EVICT_IDLE_MODELS", True)))


async def stop() -> None:
    """Cancel the worker and unregister the warm predicate, so a stopped
    preloader cannot keep models pinned against the idle evictors."""
    global _worker
    system_stats.set_warm_predicate(None)
    task, _worker = _worker, None
    if task is not None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def sweeper_loop() -> None:
    """Drop expired plans and recompute the warm set. Started/cancelled by
    lifespan beside the four idle evictors, on the same shape of loop."""
    while True:
        await asyncio.sleep(_SWEEP_S)
        try:
            now = time.monotonic()
            with _lock:
                for pid in [p for p, pl in _plans.items()
                            if pl.expires_mono <= now]:
                    _drop_plan_locked(pid, "expired")
                # Unconditional: the recompute inside _drop_plan_locked only
                # runs when something expired, and _warm must also converge
                # after a plan's own entries changed.
                _recompute_warm_locked()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the loop must survive
            logger.error("[preload] sweeper error: %s", e)


def diagnostics() -> dict:
    """Surfaced in the /stats payload next to the loaded-model list. The most
    likely real failure of this feature is silence — no worker, no plans, no
    loads — and four integers make that visible instead of invisible."""
    q = _queue
    with _lock:
        plans, warm = len(_plans), len(_warm)
    return {
        "enabled": _enabled(),
        "worker_alive": bool(_worker is not None and not _worker.done()),
        "plans": plans,
        "warm": warm,
        "queue_depth": q.qsize() if q is not None else 0,
    }


def _reset_for_tests() -> None:
    """Same contract as jobs._reset_for_tests: drop every scrap of module
    state so one test cannot observe another's plans."""
    global _queue, _loop, _worker, _busy
    with _lock:
        _plans.clear()
        _warm.clear()
    if _queue is not None:
        while True:
            try:
                _queue.get_nowait()
            except Exception:  # noqa: BLE001 — QueueEmpty, or a closed loop
                break
    _queue = None
    _loop = None
    _worker = None
    _busy = False
