"""Dictation-outcome reporting from the desktop client.

Mounted always-on in main.py (like the settings-sync router), so a
route-level 404 keeps its client-side meaning of "this backend build doesn't
have the endpoint". One endpoint:

  POST /v1/usage/outcome   {"outcomes": [{job_id, activation, delivery,
                            translation, app_id?}, …]}  (≤ 100 items)
                           → {"results": [{job_id, status}, …]}

The server already holds every number about a dictation session (words,
audio seconds, utterances — its own rows); what only the client knows is
HOW the session was run and WHERE the text went: hold-to-talk vs
hands-free, typed vs clipboard vs dropped, whether a translation was taken,
and the app id it was typed into. That is all this endpoint accepts — a
client cannot inflate its own word counts through it.

Security model:
  - User-tier bearer auth ONLY: Depends(get_current_user). No require_page
    gate (outcomes belong to the key that dictated, not to a WebUI page) and
    no host allowlist, for the same reason /v1/usage and the settings sync
    have none: remote desktop clients must reach it.
  - Strictly self-scoped: an outcome only ever attaches to a job the caller
    owns. Someone else's job id reads as `duplicate` — the same answer as a
    re-send, so the endpoint confirms nothing about other users' sessions.
  - Idempotent: the first report per job is `accepted`, every later one
    `duplicate` (nothing changes). The client queues outcomes offline and
    retries, so a replay must be harmless.
  - Flood guard: 60 posts per minute per identity (rate_limit.FixedWindow),
    which is far above what a human dictating can produce and keeps a
    retry loop gone wrong from hammering the store.
  - CSRF: bearer clients are exempt from the double-submit middleware; a
    cookie-authenticated browser POST needs X-CSRF-Token (main.py).

An app_id names a program on the user's machine; it is stored, never logged.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from faster_whisper_backend.auth import rate_limit as _rl
from faster_whisper_backend.stats import usage_store
from faster_whisper_backend.auth.dependencies import get_current_user

router = APIRouter(prefix="/v1")

# No matching AdminConfig field: the ceiling is a flood guard, not a quota
# an operator would tune, so FixedWindow's config lookup falls through to
# the default here.
_outcome_rate = _rl.FixedWindow(
    config_field="USAGE_OUTCOME_RATE_PER_MIN",
    window_s=60.0,
    default_max=60,
    message="too many usage outcome posts — slow down "
            "({limit}/min; retry in {retry_after}s)",
)


class Outcome(BaseModel):
    """One session's outcome. `job_id` is the client-minted id it opened
    the stream with (or the server's session id when it sent none); the
    same alphabet the batch progress id uses."""
    model_config = {"extra": "forbid"}
    job_id: str = Field(pattern=r"^[0-9a-f]{8,64}$")
    activation: Literal["hold", "handsfree"]
    delivery: Literal["typed", "clipboard", "none"]
    translation: Literal["translated", "kept_original", "not_asked", "aborted"]
    app_id: str | None = Field(default=None, min_length=1, max_length=64)


class OutcomesBody(BaseModel):
    model_config = {"extra": "forbid"}
    outcomes: list[Outcome] = Field(min_length=1, max_length=100)


@router.post("/usage/outcome")
async def post_usage_outcomes(
    payload: OutcomesBody,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Attach outcomes to the caller's dictation jobs. Always 200 with one
    result per item in input order: `accepted` on the first report of a
    job (a job the server never saw — a session with no finished utterance
    — gets a stub so it still counts), `duplicate` when it was reported
    before. 503 when the store is not open on this server."""
    _outcome_rate.hit(_rl.identity_key(user, request))
    uid = user.get("user_id") or ""
    if not uid:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "usage outcomes need a user-bound key")

    def _apply() -> list[dict[str, str]]:
        # Off the loop: up to 100 small transactions behind the store lock.
        return [
            {"job_id": o.job_id,
             "status": usage_store.record_outcome(
                 user_id=uid, job_id=o.job_id, activation=o.activation,
                 delivery=o.delivery, translation=o.translation,
                 app_id=o.app_id)}
            for o in payload.outcomes
        ]

    try:
        results = await asyncio.to_thread(_apply)
    except RuntimeError:
        # usage_store.init_db failed at startup; the reason is in the log.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "usage store unavailable on this server — check the server log "
            "for the startup error (USAGE_DB path/permissions)",
        ) from None
    return {"results": results}
