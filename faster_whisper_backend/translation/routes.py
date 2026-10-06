"""POST /v1/text/translations: text-to-text translation of an existing
transcript through the llama.cpp engine (translation/engine.py), with its
in-flight gauge and request-shape ceilings. /v1/audio/translations stays in
main next to the transcription handler.
"""
import asyncio
import functools
import logging
import re
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from faster_whisper_backend.auth import rate_limit as _rl
from faster_whisper_backend.auth.dependencies import get_current_user as _get_current_user_dep
from faster_whisper_backend.core import jobs
from faster_whisper_backend.core import receipt_hold
from faster_whisper_backend.core import run_plan as _run_plan
from faster_whisper_backend.core import store_common
from faster_whisper_backend.core.languages import TRANSLATE_CODE_RE as _TRANSLATE_CODE_RE
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.stats import metrics
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import progress as tx_progress
from faster_whisper_backend.transcription import receipt as tx_receipt
from faster_whisper_backend.translation import engine as _tr
from faster_whisper_backend.translation import gating as tr_gating

logger = logging.getLogger("whisper-api")
_log_safe = store_common.log_safe

router = APIRouter()


# ── Text-to-text translation (translate an existing transcript) ─────────────

# Every accepted request is real llama.cpp inference, possibly minutes of it,
# so the meaningful limit is CONCURRENCY, not rate: what hurts is one client
# holding the model while everyone else waits, and a per-minute counter cannot
# express that (a single request can outlive its own window). The in-flight
# gauge is the protection; the window below is only a backstop against a
# runaway loop that never reaches the gauge because each attempt fails
# validation first.
_translate_inflight = _rl.InFlight(
    config_field="TRANSLATE_MAX_INFLIGHT_PER_USER",
    default_max=2,
    message="you already have {limit} translations running — "
            "wait for one to finish",
)
_text_translate_rate = _rl.FixedWindow(
    config_field="TRANSLATE_RATE_PER_MIN",
    window_s=60.0,
    default_max=120,
    message="too many translation requests — slow down "
            "({limit}/min; retry in {retry_after}s)",
)


# Request-shape ceilings: entry count and total characters. The 4 MiB JSON
# body cap (main._max_body_mw) bounds the wire size before either check runs.
_TEXT_TRANSLATE_MAX_SEGMENTS = 2000
_TEXT_TRANSLATE_MAX_CHARS = 200_000

# translate_segments' warning strings name segments by 1-based POSITION
# ("segment 2", "segments 1-3"). On this endpoint clients address segments by
# their own ids — rewrite the references before returning.
_TR_SEG_WARN_RE = re.compile(r"segments (\d+)-(\d+)|segment (\d+)")


def _client_id_warnings(warnings: "list[str]", ids: "list") -> "list[str]":
    """Rewrite positional 1-based segment references in translation warnings
    to the CLIENT-supplied segment ids (a group span expands to the member
    ids, which need not be sequential)."""
    def _sub(m: "re.Match") -> str:
        if m.group(3) is not None:
            i = int(m.group(3)) - 1
            return f"segment {ids[i]}" if 0 <= i < len(ids) else m.group(0)
        a, b = int(m.group(1)) - 1, int(m.group(2)) - 1
        if 0 <= a <= b < len(ids):
            return "segments " + ", ".join(str(ids[j])
                                           for j in range(a, b + 1))
        return m.group(0)
    return [_TR_SEG_WARN_RE.sub(_sub, w) for w in warnings]


@router.post("/v1/text/translations")
async def translate_text(request: Request,
                         user: dict = Depends(_get_current_user_dep)):
    """Translate already-transcribed segments (text→text via GGUF models) —
    the standalone twin of the batch handler's `translate_to` stage, for
    translating a transcript the client already holds without re-uploading
    the audio. Body: {"segments": [{"id", "text", "speaker"?}], "targets":
    [codes], "source"?, "translation_model"?, "translation_mode"?,
    "translation_glossary"?, "context_segments"?, "progress_id"?}. The
    optional progress_id plugs into the same GET progress / POST cancel
    endpoints as a batch transcription. Answers each input segment's id in
    input order with its {target: text} translations."""
    if not getattr(cfg, "TRANSLATION_ENABLED", False):
        raise HTTPException(status_code=403,
                            detail="translation is disabled on this server")
    # The per-minute backstop fires just below, INSIDE the release-on-reject
    # bracket: it costs one dict lookup and needs nothing parsed, but its 429
    # (like every validation exit) must release a parked dictation receipt,
    # and the receipt key only exists once the body is parsed (the body cap
    # middleware bounds the wire cost of parsing first). The in-flight slot
    # is taken much later, right before the work starts — see below.
    _inflight_key = _rl.identity_key(user, request)
    # Canonical job id — stamped on every log line of this run (req=<id8>)
    # so a multi-minute translation can be followed through the log.
    request_id = uuid.uuid4().hex
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed body is a caller error
        raise HTTPException(status_code=422, detail="expected a JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="expected a JSON object")

    # A dictation utterance whose receipt is being held open for us. The
    # stream logged nothing for it and is waiting on this request to complete
    # the block, so EVERY exit below has to release it — including the
    # validation rejects, which is why the key is parsed FIRST and the whole
    # validation ladder runs inside one release-on-reject bracket. (The 403 /
    # malformed-body exits above precede the parse and cannot know the key;
    # a stream on a translation-disabled server parks nothing.)
    _cap = body.get("captured_id")
    _held_key = (_cap.strip()[:64]
                 if isinstance(_cap, str) and _cap.strip() else None)
    # The dictation session this translation belongs to (the same client-
    # minted id the stream handshake carried as `client_job`). A stop-timing
    # one-shot claims no receipt, so this is the only handle that ties its
    # standalone receipt to the utterances it translated. Malformed → absent.
    _cj = body.get("client_job")
    _client_job = (_cj if isinstance(_cj, str) and tx_progress._PROGRESS_ID_RE.match(_cj)
                   else None)

    try:
        _text_translate_rate.hit(_inflight_key)
        segments = body.get("segments")
        if not isinstance(segments, list) or not segments:
            raise HTTPException(status_code=422,
                                detail="segments must be a non-empty list")
        if len(segments) > _TEXT_TRANSLATE_MAX_SEGMENTS:
            raise HTTPException(
                status_code=422,
                detail=f"segments is capped at {_TEXT_TRANSLATE_MAX_SEGMENTS} "
                       "entries")
        seg_in: "list[dict]" = []
        ids: "list" = []
        total_chars = 0
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict) or not isinstance(seg.get("text"), str):
                raise HTTPException(
                    status_code=422,
                    detail=f"segments[{i}] must be an object with a string "
                           "'text'")
            total_chars += len(seg["text"])
            speaker = seg.get("speaker")
            seg_in.append({"text": seg["text"],
                           "speaker": speaker if isinstance(speaker, str) else None})
            ids.append(seg.get("id", i))
        if total_chars > _TEXT_TRANSLATE_MAX_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"segments exceed {_TEXT_TRANSLATE_MAX_CHARS} total "
                       "characters")

        # Per-identity policy (locks + overrides) applies here exactly as on the
        # batch path — reading bare cfg would let a locked-down key bypass its
        # profile by using this endpoint instead of the transcription form.
        ident = effective_config.build_ident(user, None)
        _ignored: "list[str]" = []

        # The module-level locked-wins / request-wins / config-inherits ladder,
        # bound to this request. _NO_DEFAULT: this endpoint resolves numeric
        # knobs (TRANSLATION_MAX_TARGETS, context segments) through the same
        # ladder, where the batch sites' `or ""` would corrupt a legitimate 0.
        _knob = functools.partial(effective_config._resolve_request_knob, None, ident, _ignored,
                                  default=effective_config._NO_DEFAULT)

        raw_targets = body.get("targets")
        max_targets = int(_knob("TRANSLATION_MAX_TARGETS", "", None) or 1)
        if not isinstance(raw_targets, list) or not raw_targets:
            raise HTTPException(status_code=422,
                                detail="targets must be a non-empty list of "
                                       "language codes")
        targets: "list[str]" = []
        for t in raw_targets:
            code = t.strip() if isinstance(t, str) else ""
            if not _TRANSLATE_CODE_RE.match(code):
                raise HTTPException(
                    status_code=422,
                    detail=f"targets contains an invalid language code: {t!r}")
            if code not in targets:
                targets.append(code)
        if len(targets) > max_targets:
            raise HTTPException(
                status_code=422,
                detail=f"targets is capped at TRANSLATION_MAX_TARGETS "
                       f"({max_targets})")

        source = body.get("source")
        if source is not None and not isinstance(source, str):
            raise HTTPException(status_code=422, detail="source must be a string")
        mode = body.get("translation_mode")
        if mode is not None and mode not in ("fluent", "faithful"):
            raise HTTPException(
                status_code=422,
                detail="translation_mode must be 'fluent' or 'faithful'")
        mode = _knob("TRANSLATION_MODE", "translation_mode", mode) or "fluent"
        glossary = body.get("translation_glossary")
        if glossary is not None and not isinstance(glossary, str):
            raise HTTPException(status_code=422,
                                detail="translation_glossary must be a string")
        glossary = (_knob("TRANSLATION_GLOSSARY", "translation_glossary",
                          glossary) or "")[:4000]
        context_segments = body.get("context_segments")
        # bool is an int subclass: JSON true must not pass as 1.
        if context_segments is not None and (not isinstance(context_segments, int)
                                             or isinstance(context_segments, bool)):
            raise HTTPException(status_code=422,
                                detail="context_segments must be an integer")
        _ctx_resolved = _knob("TRANSLATION_CONTEXT_SEGMENTS", "context_segments",
                              tx_models._clamp_context_segments(context_segments))
        context_segments = int(_ctx_resolved) if _ctx_resolved is not None else None

        model_ref = body.get("translation_model")
        if model_ref is not None and not isinstance(model_ref, str):
            raise HTTPException(status_code=422,
                                detail="translation_model must be a string")
        _tm_req = (model_ref or "").strip() or None
        _tr_model = (_knob("TRANSLATION_MODEL", "translation_model",
                           _tm_req) or "").strip() or tr_gating._translation_default_model()
        # Shared allowlist gate; like the batch stage, it constrains only the
        # CLIENT-requested value — an admin-pinned per-identity/per-model
        # TRANSLATION_MODEL is policy and passes.
        _tm_inherited = (effective_config.cfg_for(None, "TRANSLATION_MODEL", ident)
                         or "").strip() or None
        if not tr_gating._translation_model_allowed(_tr_model, requested=_tm_req,
                                          inherited=_tm_inherited):
            raise HTTPException(
                status_code=400,
                detail="requested translation model is not in "
                       "TRANSLATION_ALLOWED_MODELS on this server")

        # Optional progress/cancel plumbing: a valid id joins _BATCH_PROGRESS so
        # the existing GET progress and POST cancel endpoints work unchanged
        # (cancel only accepts ids it can see in flight).
        _pid = tx_progress._claim_progress_id(body.get("progress_id"))
        _rplan = _run_plan.RunPlan(kind="text")
    except BaseException:
        # Any rejection above (422/413/429/400) — or a client disconnect
        # mid-validation — must hand the parked receipt back NOW, or the
        # sweeper logs it ~90 s later as "no result within 90s", out of
        # order and with the wrong reason. Mirrors the acquire-refusal
        # release below.
        tx_receipt._release_held_receipt(_held_key, "request rejected")
        raise

    # ── Canonical job logging ────────────────────────────────────────────
    # Start receipt now, throttled heartbeats from the progress wrapper,
    # and a mirrored terminal line (✓ done / ✗ failed / ✗ cancelled) on
    # every exit path — this endpoint used to log only on cancel/failure.
    _uid = (user.get("user_id") or "")
    logger.info(
        "[translate] req=%s start: %d segments × %d targets (%s) model=%s "
        "mode=%s user=%s",
        request_id[:8], len(seg_in), len(targets), ",".join(targets),
        _tr_model or "?", mode, _uid[:8] or "-")
    _t0 = time.perf_counter()
    # Heartbeat + load-time bookkeeping shared with the progress wrapper.
    # first_cb approximates "model ready" — good enough to split load from
    # infer on the completion line when the model was cold.
    _hb = {"last_log": _t0, "last_pct": 0, "first_cb": None}
    _was_loaded = _tr_model in _tr._models
    # Take the in-flight slot HERE rather than next to the rate check at
    # the top: a dozen `raise HTTPException` validation exits sit between
    # the two, and each one would have to remember to release a slot it
    # never actually used. From this line to the finally there is exactly
    # one path out.
    metrics.seed_wait()
    try:
        _translate_inflight.acquire(_inflight_key)
    except HTTPException:
        # Outside the try below, so the generic `except HTTPException`
        # release there never sees this refusal — without this the parked
        # dictation receipt would sit until the sweeper logged it as
        # "no result within 90s" instead of the rejection it actually was.
        tx_receipt._release_held_receipt(_held_key, "request rejected — too many in flight")
        raise
    # Set only AFTER a successful acquire — the acquire itself sits
    # outside the try, so a refused request never releases a slot it does
    # not hold.
    _inflight_held: "str | None" = _inflight_key
    # Set once the translation itself has returned. From that point the
    # success path below OWNS the held receipt and will claim it — and the
    # claim happens after the finally, so the finally has to know not to
    # release the receipt out from under it.
    _receipt_claimed_below = False
    # Server jobs row (kind "translate"): written after the progress seed,
    # finished by _record_run on every non-ok exit and by the success tail.
    _job_row = False
    _job_finished = False
    try:
        # Central running-jobs registry entry. Progress feeds in directly from
        # _on_progress below (works whether or not the client sent a progress_id).
        jobs.job_start("translate", id=request_id, model=(_tr_model or None),
                       user=(user.get("username") or _uid or None),
                       key=user.get("key_id"), user_id=_uid,
                       detail=f"{len(seg_in)} segs → {','.join(targets)}",
                       )
        jobs.job_update(request_id, stage="translating", progress_id=_pid)

        def _on_progress(f, step=None, last_text=None, target=None,
                         target_progress=None):
            now = time.perf_counter()
            if _hb["first_cb"] is None:
                _hb["first_cb"] = now
            # Restamp the held receipt's IDLE timer. This is what makes the
            # hold safe to keep short: a cold GGUF load that takes two minutes
            # keeps its receipt alive because it keeps reporting, while a
            # wedged one stops reporting and is released on schedule.
            if _held_key:
                receipt_hold.touch(_held_key)
            pct = int(max(0.0, min(1.0, f or 0.0)) * 100)
            # Log on every crossed 10% boundary, and at least every 30 s.
            if (now - _hb["last_log"] >= 30.0
                    or pct // 10 > _hb["last_pct"] // 10):
                _hb["last_log"] = now
                _hb["last_pct"] = pct
                logger.info("[translate] req=%s %s %d%%",
                            request_id[:8], step or "translating", pct)
            # stage is re-asserted on every tick so the first real batch flips
            # a "downloading" entry (cold model fetch) back to "translating".
            jobs.job_update(request_id, stage="translating", progress=f,
                            step=step)
            fields = {"stage": "translating", "progress": f, "step": step,
                      "target": target, "target_progress": target_progress}
            if last_text:
                fields["last_text"] = last_text
            tx_progress._progress_set(_pid, **fields)

        def _on_download(done, total):
            frac = (done / total) if total else None
            jobs.job_update(request_id, stage="downloading", progress=frac,
                            total_bytes=total or None)
            tx_progress._progress_set(_pid, stage="downloading", progress=frac,
                          total_bytes=total or None)

        def _record_run(status: str, exc: "BaseException | None" = None,
                        *, folded_into: "str | None" = None) -> None:
            """Persist this run as a recent-jobs row (kind='translate') on every
            terminal path. No audio duration; segment count lives in the stage
            detail (words=0 — a segment count is not a word count).

            `folded_into` names the dictation utterance row this translation
            was appended to as a stage; then only the usage rollup is written
            here — a second recent-jobs row would show the one job twice."""
            nonlocal _job_finished
            secs = round(time.perf_counter() - _t0, 3)
            if _job_row and status != "ok":
                # The success tail stamps `done` with the payload itself.
                tx_progress._jobs_finish_sync(
                    _pid, status=status,
                    error=(str(exc) if isinstance(exc, _tr.TranslationError)
                           else tx_progress._job_error_text(status, exc,
                                                "translation failed")),
                    plan=_rplan.snapshot()["plan"], model=(_tr_model or None))
                _job_finished = True
            if status == "ok":
                # The clean run's ledger write is blocking file IO: the
                # success tail awaits it off the loop as its last step.
                _rplan.stage_done("translating")
            else:
                try:
                    _rplan.finish_run(status)   # no IO: nothing is learned
                except Exception:  # noqa: BLE001 — never fail on a ledger write
                    pass
            _ec, _es = metrics.classify_error(exc, status=status,
                                              stage="translating")
            metrics.record_transcription(
                error_class=_ec,
                error_stage=_es,
                model=(_tr_model or ""),
                audio_dur=0.0,
                proc_dur=secs,
                status=status,
                words=0,
                request_id=request_id,
                user_id=(_uid or None),
                key_id=user.get("key_id"),
                username=user.get("username"),
                key_label=user.get("key_label"),
                kind="translate",
                stages=[{"name": "translate", "secs": secs,
                         "model": (_tr_model or None),
                         "detail": f"{len(seg_in)} segs → {','.join(targets)}",
                         "targets": list(targets)}],
                job_id=_pid or request_id,
                wait_s=metrics.take_wait(),
                recent_row=folded_into is None,
            )

        # `owner` binds the entry to this caller, exactly like the batch
        # seed — the progress/cancel endpoints treat a mismatch as unknown.
        # A text run is a one-stage plan: its units are the targets.
        _rplan.set_stages(["translating"])
        _rplan.set_segments(len(seg_in))
        _rplan.set_translation(list(targets), model=(_tr_model or None),
                               device=_tr._resolve_device(), mode=mode,
                               source_lang=(source or None))
        if _pid:
            tx_progress._RUN_PLAN_BY_PID[_pid] = _rplan
        tx_progress._progress_set(_pid, stage="translating", progress=0.0,
                      model=(_tr_model or None),
                      device=_tr._resolve_device(), compute="gguf",
                      owner=(user.get("user_id") or user.get("key_id")))
        _job_row = tx_progress._jobs_start(
            _pid, request_id=request_id, kind="translate",
            user_id=(user.get("user_id") or None), key_id=user.get("key_id"),
            model=(_tr_model or None), source_kind="text",
            source_name=f"{len(seg_in)} segments → {','.join(targets)}")
        try:
            tx_progress._check_cancelled(_pid)

            async def _run_translation():
                return await _tr.translate_segments(
                    seg_in, targets,
                    source_lang=(source or None),
                    model_ref=_tr_model,
                    mode=mode,
                    glossary=glossary,
                    context_segments=context_segments,
                    progress_cb=_on_progress,
                    cancel_check=lambda: tx_progress._cancel_requested(_pid),
                    download_cb=_on_download,
                )
            # Same policy as the batch stage: the GPU inference semaphore is
            # held only when translation actually runs on cuda — a llama.cpp
            # CPU run must not occupy a GPU slot for its duration.
            if _tr._resolve_device() == "cuda":
                async with tx_models.get_inference_semaphore():
                    tx_progress._check_cancelled(_pid)
                    per_seg, warnings, meta = await _run_translation()
            else:
                per_seg, warnings, meta = await _run_translation()
            _receipt_claimed_below = True
        except _tr.TranslationCancelled:
            raise tx_progress._ClientCancelled() from None
    except tx_progress._ClientCancelled:
        logger.info("[translate] req=%s ✗ cancelled after %.1fs",
                    request_id[:8], time.perf_counter() - _t0)
        tx_receipt._release_held_receipt(
            _held_key,
            f"cancelled by client after {time.perf_counter() - _t0:.1f}s")
        _record_run("cancelled")
        raise HTTPException(status_code=499, detail="cancelled by the client")
    except _tr.TranslationError as e:
        # str(e) is client-safe by the module's contract.
        logger.info("[translate] req=%s ✗ failed after %.1fs (%s)",
                    request_id[:8], time.perf_counter() - _t0,
                    _log_safe(str(e)))
        tx_receipt._release_held_receipt(
            _held_key,
            f"failed after {time.perf_counter() - _t0:.1f}s — {_log_safe(str(e))}")
        _record_run("error", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        tx_receipt._release_held_receipt(_held_key, "request rejected")
        raise
    except Exception as e:  # noqa: BLE001 — never forward raw errors
        logger.error("[translate] req=%s ✗ failed after %.1fs: %s",
                     request_id[:8], time.perf_counter() - _t0,
                     _log_safe(str(e)))
        tx_receipt._release_held_receipt(
            _held_key,
            f"failed after {time.perf_counter() - _t0:.1f}s")
        _record_run("error", e)
        raise HTTPException(status_code=500, detail="translation failed")
    finally:
        # Release FIRST — before any await and before job_end. A client
        # disconnect raises CancelledError, which is a BaseException, so every
        # `except Exception` arm above is skipped and only `finally` runs; a
        # release sitting after an await in here would in turn be skipped if
        # the cancellation landed on that await. That is exactly the bug class
        # the streaming teardown documents (streaming/routes.py, the `finally`
        # of the websocket handler that releases _stream_sessions / model
        # leases before any await), where it permanently burned one of
        # STREAMING_MAX_SESSIONS. Nullable local: only a
        # successful acquire sets it, so a refused request releases nothing.
        if _inflight_held is not None:
            _translate_inflight.release(_inflight_held)
            _inflight_held = None
        # Same reasoning for the held dictation receipt: the five except arms
        # above cover every exception, but a dropped connection raises
        # CancelledError past all of them, and the receipt would then survive
        # only until the 90 s idle sweep — long after the utterance it belongs
        # to scrolled off. Releasing here is the catch-all, not a sixth
        # duplicate: release on an already-released or already-claimed key is a
        # no-op, so the five paths above keep their own, more specific notes.
        # The success path claims the receipt AFTER this finally runs, which is
        # what the flag guards.
        if _held_key and not _receipt_claimed_below:
            tx_receipt._release_held_receipt(
                _held_key,
                f"connection closed after {time.perf_counter() - _t0:.1f}s")
        jobs.job_end(request_id)
        if _pid:
            tx_progress._progress_close(_pid)
            tx_progress._RUN_PLAN_BY_PID.pop(_pid, None)
        # Catch-all for the paths that never reached _record_run: the
        # HTTPException arm and a disconnect unwinding past every arm. The
        # success path claims below, so it is excluded here.
        if _job_row and not _job_finished and not _receipt_claimed_below:
            tx_progress._jobs_finish_sync(_pid, status="error", error="request aborted",
                              model=(_tr_model or None))
            _job_finished = True

    _elapsed = time.perf_counter() - _t0
    # Cold model: everything up to the first progress callback is load (the
    # cache layer logs the exact load line too); warm model: all infer.
    _load_s = (max(0.0, _hb["first_cb"] - _t0)
               if (not _was_loaded and _hb["first_cb"] is not None) else 0.0)
    _chars_out = sum(len(t) for d in per_seg for t in d.values())
    logger.info(
        "[translate] req=%s ✓ done in %.1fs (load %.1fs · infer %.1fs) · "
        "%d segs → %s · %d chars in / %d out · %d guard fallbacks",
        request_id[:8], _elapsed, _load_s, max(0.0, _elapsed - _load_s),
        len(seg_in), ",".join(targets), total_chars, _chars_out,
        len(warnings))

    # Complete the dictation receipt this request was holding open, so the
    # utterance and its translation read as ONE block instead of a receipt
    # and four orphan [translate] lines with nothing linking them. Claimed
    # BEFORE the run is recorded: the utterance's recent-jobs row is where
    # this translation lands as a second stage (like a batch job's), and
    # _record_run must know that so it does not write a second row.
    _folded_into: "str | None" = None
    _held = None
    if _held_key:
        _used_model = meta.get("model") or _tr_model
        _tr_key = preload.stats_key("translation", _used_model or "")
        # ONE stage row for both consumers (the /stats recent-jobs row and
        # the receipt's Pipeline table), so they cannot print different
        # seconds for the same stage.
        _tr_stage = {"name": "translating", "secs": round(_elapsed, 3),
                     "model": _used_model or None,
                     "load_secs": round(_load_s, 3),
                     "device": tx_receipt._model_compute_device(_tr_key)[1],
                     "detail": f"{len(seg_in)} segs → {','.join(targets)}",
                     "targets": list(targets)}
        _held = receipt_hold.claim(_held_key)
        if _held is not None and _held.get("request_id"):
            try:
                from faster_whisper_backend.stats import recent_transcriptions_store as _rts
                if _rts.append_stage(
                        str(_held["request_id"]), dict(_tr_stage),
                        add_processing_s=_elapsed):
                    _folded_into = str(_held["request_id"])
            except Exception as _fe:  # noqa: BLE001 — a stats miss never fails the request
                logger.warning("[translate] could not fold into utterance row: %s", _fe)
    _record_run("ok", folded_into=_folded_into)

    if _held_key:
        if _held is not None:
            _held["translation"] = {
                "model": _used_model or None,
                "device": tx_receipt._model_compute_device(_tr_key)[1] or tx_receipt._OMIT,
                "targets": list(targets),
                "source": (source or "").strip() or tx_receipt._OMIT,
                "mode": mode,
                "result": (f"{len(seg_in)} segs · {total_chars} chars in / "
                           f"{_chars_out} out · {len(warnings)} guard fallbacks"),
            }
            # Append the translate row to the stage table the utterance was
            # parked with, so the Pipeline section shows both halves and the
            # cold-load cost lands where a reader looks for it.
            _held["stages"] = list(_held.get("stages") or []) + [_tr_stage]
            try:
                logger.info(tx_receipt._format_request_block(**_held))
            except Exception as _me:  # noqa: BLE001 — never fail on a receipt
                logger.warning("[translate] held receipt render failed: %s", _me)
        else:
            _held_key = None   # nothing to claim → standalone receipt below
    if not _held_key:
        # No held utterance to merge into (a stop-timing one-shot, a plain
        # API caller, or a capture that was already swept): still log ONE
        # receipt, so the translate is not just four progress lines with
        # no model / targets / user attached.
        _used_model = meta.get("model") or _tr_model
        _tr_key = preload.stats_key("translation", _used_model or "")
        try:
            logger.info(tx_receipt._format_translate_block(
                request_id=request_id, model_name=_used_model,
                device=tx_receipt._model_compute_device(_tr_key)[1],
                targets=list(targets), source=source, mode=mode,
                result=(f"{len(seg_in)} segs · {total_chars} chars in / "
                        f"{_chars_out} out · {len(warnings)} guard fallbacks"),
                secs=_elapsed, load_secs=_load_s, client_job=_client_job,
                user_id=user.get("user_id"), key_id=user.get("key_id"),
                username=user.get("username"), key_label=user.get("key_label")))
        except Exception as _me:  # noqa: BLE001 — never fail on a receipt
            logger.warning("[translate] receipt render failed: %s", _me)

    # kept_original: targets for which the guard fallback returned the SOURCE
    # text — without it a kept German line under translations["en"] is
    # indistinguishable from a real translation. Absent when clean. Also
    # emitted as `translations_kept`, the name the batch endpoint's segments
    # use for the same fact, so a client can read one key on both endpoints
    # (kept_original stays for existing consumers).
    _kept = meta.get("kept") or {}
    _result = {
        "segments": [{"id": ids[i], "translations": per_seg[i],
                      **({"kept_original": list(_kept[i]),
                          "translations_kept": list(_kept[i])}
                         if _kept.get(i) else {})}
                     for i in range(len(ids))],
        "translation": {"model": meta.get("model"), "targets": targets,
                        "source": meta.get("source"), "mode": meta.get("mode")},
        "plan": _rplan.snapshot()["plan"],
        "warnings": _client_id_warnings(warnings, ids) + [
            f"{name} is locked on this server — your value was ignored"
            for name in _ignored if name
        ],
    }
    if _job_row:
        # Text results are small (segments in, translations out) — inline.
        tx_progress._jobs_finish_sync(_pid, status="ok", payload=_result,
                          plan=_result["plan"], model=(meta.get("model") or _tr_model or None))
    # Teach the rates ledger — off the loop (a locked, fsync'd file rewrite)
    # and LAST, so a cancellation landing on this await skips nothing.
    try:
        await asyncio.to_thread(_rplan.finish_run, "ok")
    except Exception:  # noqa: BLE001 — never fail on a ledger write
        pass
    return _result
