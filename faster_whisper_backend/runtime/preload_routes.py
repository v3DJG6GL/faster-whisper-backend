"""POST /v1/models/preload — ask the server to warm the models a job is about
to need.

Mounted always-on in main.py (like /v1/synced-client-settings), so a
route-level 404 keeps its client-side meaning of "this backend build doesn't
have the endpoint" rather than "preloading is switched off here" — the latter
is a 202 with every entry `deferred`.

Security model:
  - User-tier bearer auth ONLY: Depends(get_current_user), the same tier as
    /v1/models and /v1/me, both of which already publish `loaded` flags for
    every family this endpoint can warm. Deliberately NO require_page gate and
    NO host allowlist, for the reason client_settings/routes.py states: a key
    that may transcribe must be able to prepare the server for its own
    transcription, and remote desktop clients must reach it.
  - The endpoint cannot load anything a transcribe request could not: the
    per-family allowlists are applied exactly as the batch handler applies
    them, including the "an empty allowlist means the configured model only,
    never anything" rule for diarization/separation, and the caller's
    identity/per-model EFFECTIVE stage models, which that handler admits as
    admin policy.

There is NO 4xx path beyond pydantic's 422 for a structurally invalid body.
A disallowed model, a disabled stage and a disabled feature all answer 202
with `deferred` plus a reason, because every one of them is a statement about
the server's state, not about the request being wrong — and the client's
fallback for all three is identical: let the stage load its model in-band.

Idempotency: a `plan_id` from the client (else one derived from the caller and
the sorted entries) means a repeat POST restamps the plan's TTL and re-admits
only what is not already warmed, instead of accumulating duplicate plans.

Expiry: there is no cancel endpoint. A plan dies when its TTL runs out — the
sweeper drops it, and the worker skips dead-plan items at dequeue rather than
the queue being drained (an asyncio.Queue has no removal primitive). A load
already inside an executor thread is NOT cancellable — it finishes and
registers normally, which at worst leaves a model loaded nobody asked for and
at best hands the size ledger a free measurement of it.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.translation import gating as tr_gating
from faster_whisper_backend.auth.dependencies import get_current_user

# Same logger name the rest of the model machinery uses, so a preload and the
# load it causes sit under one name in the log. This module had no logging at
# all, which meant a POST that landed left no trace and was indistinguishable
# from one that was never sent.
logger = logging.getLogger("whisper-server")

router = APIRouter(prefix="/v1")


class PreloadModel(BaseModel):
    model_config = {"extra": "forbid"}

    family: Literal["whisper", "diarization", "separation", "translation"]
    # 128 is comfortably above the longest real id (a GGUF `org/repo:QUANT`);
    # the point is a bound, not a schema — ids are matched against the
    # allowlists below, never used to build a path.
    id: str = Field(min_length=1, max_length=128)


class PreloadRequest(BaseModel):
    model_config = {"extra": "forbid"}

    # 6, not 4: a client may legitimately name two whisper models (the one it
    # will use plus the one its next queued job will).
    models: list[PreloadModel] = Field(min_length=1, max_length=6)
    # Hex is a shape bound only: a derived id is 32 hex chars too, so a client
    # CAN name one. What stops adopting another caller's plan is the owner
    # check in preload._register_plan (a foreign plan_id is re-derived).
    # ^…$ rather than the \A…\Z tx_progress._PROGRESS_ID_RE uses: pydantic v2 compiles
    # patterns with the Rust regex engine, which rejects \A/\Z outright (a
    # SchemaError at import, not a failed match). In that engine ^/$ are
    # end-of-TEXT anchors without a multiline flag, so the two are equivalent.
    plan_id: "str | None" = Field(default=None, pattern=r"^[0-9a-f]{8,64}$")
    stage_ahead: bool = True
    # Free-text label naming the client path that fired this plan
    # (dictation / transcribe / viewer). Echoed into the log receipt so an
    # operator can tell WHICH trigger ran, not just that one did. Bounded
    # and never used for anything but display.
    trigger: "str | None" = Field(default=None, max_length=32)


# The cfg field that picks each non-whisper family's model: the key into the
# effective values and into the caller's ident.locked.
_FAMILY_CFG_KEY = {"diarization": "DIARIZATION_MODEL",
                   "separation": "BGM_SEPARATION_UVR_MODEL",
                   "translation": "TRANSLATION_MODEL"}


def _allowed(family: str, model_id: str,
             effective: "dict[str, str] | None" = None,
             locked: "frozenset[str] | set[str]" = frozenset()) -> bool:
    """The batch handler's allowlist rules.

    whisper: judged on the RESOLVED id (`whisper-1` → DEFAULT_MODEL, the
    same mapping the transcribe route applies before its gate). An EMPTY
    ALLOWED_MODELS admits the configured default plus any WELL-FORMED id —
    the same `_MODEL_ID_RE` / no-".." guard `tx_models._get_or_load_model`
    applies, so a path-shaped id is deferred here instead of wasting a queue
    slot on a load the guard rejects anyway. A non-empty one admits exactly
    its members, as that gate does (the default is NOT implied: a load of it
    would 400 there).
    diarization/separation: the allowlist plus the configured model (global
    AND the caller's effective value), and an empty allowlist therefore means
    "the configured model only", never "anything". translation:
    `tr_gating._translation_model_allowed`, the rule the batch stage and the
    job plan share, with the effective TRANSLATION_MODEL as `inherited`.
    `effective` holds those identity/per-model values ({} = global only).
    A family whose model field is in `locked` (the caller's ident.locked)
    admits its effective value only: the batch handler resolves a locked
    knob to the server value and merely reports a different request
    ignored, so the preload must not load one either."""
    effective = effective or {}
    _cfg_key = _FAMILY_CFG_KEY.get(family)
    if _cfg_key is not None and _cfg_key in locked:
        want = effective.get(_cfg_key) or ""
        if family == "translation":
            # A locked "" still runs the server default (the job resolves
            # an empty TRANSLATION_MODEL to it), so that is what to warm.
            want = want or tr_gating._translation_default_model()
        return bool(model_id) and model_id == want
    if family == "whisper":
        model_id = preload.normalize_id(family, model_id)
        if not model_id:
            return False
        allow = set(getattr(cfg, "ALLOWED_MODELS", None) or ())
        if not allow:
            return model_id == getattr(cfg, "DEFAULT_MODEL", "") or (
                ".." not in model_id
                and bool(tx_models._MODEL_ID_RE.match(model_id)))
        return model_id in allow
    if family == "diarization":
        allow = set(getattr(cfg, "DIARIZATION_ALLOWED_MODELS", None) or ())
        # Only a SET value joins: an unset one must not make "" allowed.
        allow.update(v for v in (getattr(cfg, "DIARIZATION_MODEL", ""),
                                 effective.get("DIARIZATION_MODEL")) if v)
        return bool(model_id) and model_id in allow
    if family == "separation":
        allow = set(getattr(cfg, "BGM_SEPARATION_ALLOWED_MODELS", None) or ())
        allow.update(v for v in (getattr(cfg, "BGM_SEPARATION_UVR_MODEL", ""),
                                 effective.get("BGM_SEPARATION_UVR_MODEL"))
                     if v)
        return bool(model_id) and model_id in allow
    # requested=model_id makes it the CLIENT-value rule: a bare call would
    # let any ref through as admin policy.
    return tr_gating._translation_model_allowed(
        model_id, requested=model_id,
        inherited=effective.get("TRANSLATION_MODEL") or None)


def _effective_stage_models(
        user: dict, body: "PreloadRequest",
) -> "tuple[dict[str, str], frozenset[str]]":
    """The caller's identity/per-model effective stage models, resolved
    against the whisper model the plan names (else DEFAULT_MODEL) — the ident
    the batch handler builds for the same job — plus which of those fields
    the ident locks. ({}, empty) on any failure, which degrades to the
    global-only rule."""
    resolved_model = next(
        (preload.normalize_id("whisper", m.id) for m in body.models
         if m.family == "whisper"), "") or getattr(cfg, "DEFAULT_MODEL", "")
    try:
        ident = effective_config.build_ident(user, resolved_model)
        keys = _FAMILY_CFG_KEY.values()
        return ({k: (effective_config.cfg_for(resolved_model, k, ident)
                     or "").strip() for k in keys},
                frozenset(k for k in keys if k in ident.locked))
    except Exception:  # noqa: BLE001 — never worse than the global rule
        return {}, frozenset()


@router.post("/models/preload", status_code=status.HTTP_202_ACCEPTED)
async def preload_models(body: PreloadRequest,
                         user: dict = Depends(get_current_user)) -> dict:
    """Register a preload plan. Always 202."""
    entries: "list[tuple[str, str]]" = []
    denied: "dict[tuple[str, str], str]" = {}
    effective, locked = _effective_stage_models(user, body)
    for m in body.models:
        pair = (m.family, m.id.strip())
        if pair in entries:
            continue
        # A blank id (whitespace passes min_length) names no model, and a
        # non-printable one names no real model either; both would only
        # ever be deferred, and would hold the plan's /stats row open.
        if (not pair[1] or not pair[1].isprintable()
                or not _allowed(m.family, pair[1], effective, locked)):
            denied[pair] = "not_allowed"
        entries.append(pair)

    # stage_ahead=False still registers the plan (the entries are admitted
    # once and the warm leases exist); it only opts out of the server
    # advancing the plan from job progress — for a client that drives its own
    # pipeline and will POST again at each step.
    # Display-only, but it lands verbatim in a multi-line log receipt: strip
    # newlines and other non-printables so a client cannot forge log rows.
    trigger = "".join(ch for ch in (body.trigger or "")
                      if ch.isprintable()) or None
    # "job" is reserved for the server's own job binding (register_plan
    # rewinds the cursor for it and the receipt says from=job): a client
    # sending it must not pass for that.
    if trigger == "job":
        trigger = "client:job"
    logger.debug("[preload] POST %d model(s) from=%s user=%s",
                 len(entries), trigger or "-",
                 (user.get("user_id") or "-")[:8])
    return preload.register_plan(user.get("user_id"), entries,
                                 plan_id=body.plan_id, denied=denied,
                                 stage_ahead=body.stage_ahead,
                                 trigger=trigger,
                                 user_name=user.get("username"))
