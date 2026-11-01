"""Per-account settings-sync endpoints for the desktop client.

Mounted always-on in main.py (like the pipeline-rules /v1 router), so a
route-level 404 keeps its client-side meaning of "this backend build
doesn't have the endpoint". Endpoints:

  GET    /v1/synced-client-settings   Current blob (200 {version:0, blob:null} when empty)
  PUT    /v1/synced-client-settings   Optimistic write; 409 carries the current state
  DELETE /v1/synced-client-settings   Drop the stored blob

The old /v1/client-settings path stays as a deprecated alias of all three
(main.py's _LEGACY_PATHS stamps the Deprecation/Link headers).

Security model:
  - User-tier bearer auth ONLY: Depends(get_current_user). Deliberately NO
    require_page gate (unlike quick_config's v1_router) — settings sync is
    account infrastructure, not a page permission; a key that may transcribe
    must be able to sync its client's settings. No host allowlist for the
    same reason /v1/usage has none: remote desktop clients must reach it.
  - Open mode: only callers inside `ADMIN_WEBUI_ALLOWED_HOSTS` (loopback by
    default) resolve to the synthetic "(open-mode)" user and share one row;
    every other caller falls through to bearer/cookie auth and gets 401
    until a real key exists — so remote sync needs a key. Acceptable for
    the single-user self-host case; documented in the README.
  - CSRF: bearer clients are exempt from the double-submit middleware; a
    cookie-authenticated browser PUT/DELETE needs X-CSRF-Token (main.py).

The blob is opaque, SENSITIVE client JSON (may contain the client's own
backend API keys). Never log or interpret its contents — the store logs
only sizes/versions, and this module logs nothing about the payload.
"""

from __future__ import annotations

import asyncio

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from faster_whisper_backend.client_settings import store as client_settings_store
from faster_whisper_backend.auth.dependencies import get_current_user

router = APIRouter(prefix="/v1")


class PutClientSettings(BaseModel):
    """PUT body. `blob` must be a JSON OBJECT (not any scalar) so a stored
    blob can never be confused with the `blob: null` zero-state on GET.
    The server never interprets its keys. `base_version` echoes the version
    the client last saw; 0 (or absent row semantics) means "create"."""
    model_config = {"extra": "forbid"}
    blob: dict[str, Any]
    # Bounded: pydantic accepts arbitrary-precision JSON ints, and anything
    # past SQLite's INTEGER would OverflowError (a 500) in the store's CAS.
    base_version: int = Field(le=2**63 - 1)
    device: str | None = None


# Deepest nesting a stored blob may have (the blob object itself is level 1).
# GET renders through pydantic's response serializer, which 500s from about
# 255 levels while PUT's JSONResponse took them — so an over-deep PUT
# answered 200 and every later pull of that account 500'd. The desktop
# client's serde_json stops at 128 levels of the whole response anyway; its
# real blob is a handful of levels deep.
_MAX_BLOB_DEPTH = 64


class _BlobTooDeep(ValueError):
    """Raised by _put_checked; the route maps it to 422."""


def _put_checked(
    user_id: str, blob: dict[str, Any], base_version: int, device: str | None,
) -> tuple[bool, dict[str, Any] | None]:
    """client_settings_store.put after the nesting check — both walk the
    whole blob, so both run off the loop."""
    stack: list[tuple[Any, int]] = [(blob, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_BLOB_DEPTH:
            raise _BlobTooDeep
        children = node.values() if isinstance(node, dict) else node
        stack.extend((c, depth + 1) for c in children if isinstance(c, (dict, list)))
    return client_settings_store.put(user_id, blob, base_version, device=device)


def _store_unavailable() -> HTTPException:
    """503 for StoreUnavailable: init_db failed at startup (the reason is in
    the server log — typically a CLIENT_SETTINGS_DB path that isn't writable
    in this deployment). A bare 500 here told the operator nothing."""
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "client-settings store unavailable on this server — check the server "
        "log for the startup error (CLIENT_SETTINGS_DB path/permissions)",
    )


def _state_body(row: dict[str, Any] | None) -> dict[str, Any]:
    """The one wire shape shared by GET, PUT-200, and PUT-409 (so a client
    parses a single schema everywhere): absent row = the version-0 zero-state."""
    if row is None:
        return {"version": 0, "blob": None, "updated_ts": None, "device": None}
    return {
        "version": row["version"],
        "blob": row["blob"],
        "updated_ts": row["updated_ts"],
        "device": row["device"],
    }


@router.get("/synced-client-settings")
@router.get("/client-settings", deprecated=True)
async def get_client_settings(
    response: Response,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The caller's stored settings blob. Empty store is 200 with
    `{version: 0, blob: null}` — NOT 404/204, because the desktop client
    reads a route-level 404 as "backend too old for sync" and a 204 would
    force a bodyless special case."""
    # Sensitive blob (may hold the client's API keys): never in a shared cache.
    response.headers["Cache-Control"] = "no-store"
    try:
        return _state_body(
            await asyncio.to_thread(client_settings_store.get, user["user_id"]))
    except client_settings_store.StoreUnavailable:
        raise _store_unavailable() from None


@router.put("/synced-client-settings")
@router.put("/client-settings", deprecated=True)
async def put_client_settings(
    payload: PutClientSettings,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Optimistic write. 200 = stored (body carries the new version);
    409 = `base_version` is stale — the body carries the CURRENT state so
    the client can 3-way merge and re-PUT without another GET round-trip.
    413 = blob over the server cap; 422 = blob is not strict JSON (a
    non-finite float, or a lone UTF-16 surrogate in the blob or the device
    label) or is nested past _MAX_BLOB_DEPTH — nothing is stored.
    Force-push is just a PUT echoing the version fetched a moment ago;
    there is no bypass flag."""
    try:
        # Off the loop: the depth walk and put()'s json.dumps + encode both
        # touch the whole blob BEFORE the cap can reject it — up to
        # MAX_JSON_BODY_BYTES (4 MiB default, main.py's _max_body_mw).
        ok, row = await asyncio.to_thread(
            _put_checked,
            user["user_id"],
            payload.blob,
            payload.base_version,
            payload.device,
        )
    except client_settings_store.StoreUnavailable:
        raise _store_unavailable() from None
    except _BlobTooDeep:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"client settings blob nested deeper than {_MAX_BLOB_DEPTH} levels",
        )
    except client_settings_store.InvalidBlob:
        # Before `except ValueError` — InvalidBlob subclasses it.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "client settings blob must be strict JSON (no NaN/Infinity, "
            "no lone surrogates)",
        )
    except ValueError:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "client settings blob too large",
        )
    if not ok:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={**_state_body(row), "detail": "version conflict"},
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content=_state_body(row))


@router.delete("/synced-client-settings")
@router.delete("/client-settings", deprecated=True)
async def delete_client_settings(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Drop the stored blob. After this, GET reads the version-0 zero-state
    and a device still holding version N gets a 409 on its next PUT — the
    deletion surfaces instead of being silently overwritten."""
    try:
        removed = await asyncio.to_thread(
            client_settings_store.delete, user["user_id"])
    except client_settings_store.StoreUnavailable:
        raise _store_unavailable() from None
    return {"ok": True, "deleted": removed}
