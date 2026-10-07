"""/settings/api-keys — admin UI for per-user API key management.

Endpoints (all admin-only):

  GET    /settings/api-keys                       HTML page
  GET    /settings/api-keys/api/users             list users + key counts
  POST   /settings/api-keys/api/users             { username, is_admin }
  DELETE /settings/api-keys/api/users/{uid}       soft-revoke (cascades to keys)
  GET    /settings/api-keys/api/users/{uid}/keys  list keys for one user
  POST   /settings/api-keys/api/users/{uid}/keys  { label }   -> show-once raw key
  DELETE /settings/api-keys/api/users/{uid}/keys/{kid}        soft-revoke
  GET    /settings/api-keys/api/client-settings   per-account sync metadata map
  GET    /settings/api-keys/api/users/{uid}/client-settings/export   file download
  POST   /settings/api-keys/api/users/{uid}/client-settings/import   { blob }
  DELETE /settings/api-keys/api/users/{uid}/client-settings          remove server copy

Last-admin guard: revoking the only admin key (or only admin user)
returns 409. Prevents accidental lockout.

The HTML page is a single-file React-less app: vanilla JS + an HttpOnly
session cookie (same pattern as /settings). Generates keys with a show-once modal —
the raw key is copied to the clipboard, then never retrievable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from faster_whisper_backend.auth import api_keys_store
from faster_whisper_backend.core import web_common
from faster_whisper_backend.core.web_common import require_admin_webui_host
from faster_whisper_backend.auth.dependencies import require_admin
from faster_whisper_backend.core import templates
from faster_whisper_backend.client_settings import store as client_settings_store
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.stats import usage_store

logger = logging.getLogger("whisper-api")

# Host gate on the router constructor so every present + future sub-route
# inherits the check (the auth.require_page convention) — closes the
# "forgot to gate this endpoint" hole. Router-level dependencies run before
# route-level ones, so behaviour is identical to the per-route copies.
router = APIRouter(
    prefix="/settings/api-keys",
    dependencies=[Depends(require_admin_webui_host)],
)


# ---------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------

class CreateUserIn(BaseModel):
    model_config = {"extra": "forbid"}
    username: str = Field(min_length=1, max_length=128)
    is_admin: bool = False


class CreateKeyIn(BaseModel):
    model_config = {"extra": "forbid"}
    label: str = Field(default="", max_length=128)


class RenameKeyIn(BaseModel):
    """Wire shape for PATCH /api/users/{uid}/keys/{kid}/label. A label is
    required (the store rejects blank) — renaming only touches the display
    label, never the secret."""
    model_config = {"extra": "forbid"}
    label: str = Field(max_length=128)


class ConfigBindingIn(BaseModel):
    """Wire shape for a per-identity config binding — shared by the per-user
    `config` field on PatchPermissionsIn and the per-key /config endpoint.
    `overrides` is a flat OverrideProfile-shaped dict (decode/streaming fields
    + PIPELINE_RULES_EXCLUDE/INCLUDE); `profiles` is the ordered FORCED-applied
    profile list (earlier wins); `locks` names the fields the client may not
    override.

    Request-gate fields (all optional; None = inherit the next scope → global,
    and they can only NARROW the global gate, never widen it):
    `allow_request_override_profile` — may this identity NAME a profile per
    request; `allow_request_decode_overrides` — may it send inline decode
    tweaks; `allowed_override_profiles` — the request allowlist (distinct from
    `profiles`): None = all, ["*"] = all, an explicit list restricts, [] = none.

    `apply_no_profiles` is an ADMIN FORCE, not a request gate (set per-key in the
    WebUI): True suppresses every bound + requested profile for the identity, so
    it resolves to plain defaults. It does NOT inherit and is NOT bound by
    ALLOW_REQUEST_OVERRIDE_PROFILE; None/absent = off.
    """
    model_config = {"extra": "forbid"}
    overrides: dict[str, Any] = Field(default_factory=dict)
    profiles: list[str] = Field(default_factory=list)
    locks: list[str] = Field(default_factory=list)
    allow_request_override_profile: bool | None = None
    allow_request_decode_overrides: bool | None = None
    allowed_override_profiles: list[str] | None = None
    apply_no_profiles: bool | None = None


class PatchPermissionsIn(BaseModel):
    """Payload for PATCH /api/users/{uid}/permissions.

    `pages` is a partial map — only the cells the admin changed need
    appear; the store's `set_user_permissions` merges with the existing
    shape (omitted pages keep their stored scope). Cell-by-cell save
    is also fine; full-row save (what the matrix UI sends) just lists
    every page.

    `quick_config_tags` is the user's per-rule tag set for the new
    tag-based /quick-config visibility filter. `None` means "leave the
    stored value untouched" — useful for cell-level saves that only
    touched pages. An empty list `[]` is explicit "clear all tags"
    (user sees only untagged rules)."""
    model_config = {"extra": "forbid"}
    pages: dict[str, str] = Field(default_factory=dict)
    quick_config_tags: list[str] | None = None
    # Per-user config binding (override profiles + direct override blob + per-
    # field locks). `None` = leave the stored value untouched; an explicit empty
    # binding ({} overrides, [] profiles) clears it. Validated by
    # config_store.validate_binding inside set_user_permissions.
    config: ConfigBindingIn | None = None


class ImportClientSettingsIn(BaseModel):
    """Wire shape for POST /api/users/{uid}/client-settings/import — the
    WebUI's restore path. `blob` is a full opaque settings document (the
    desktop export's `categories` object, or a file previously downloaded
    with Export here); the route force-writes it, so no base_version."""
    model_config = {"extra": "forbid"}
    blob: dict[str, Any]


# ---------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------

@router.get("", response_class=HTMLResponse)
async def api_keys_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_API_KEYS_HTML, current="api-keys"),
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------
# JSON APIs
# ---------------------------------------------------------------------

@router.get(
    "/api/users",
    dependencies=[Depends(require_admin)],
)
async def list_users_api() -> JSONResponse:
    # Snapshot of every exposed (non-terminal) rule's tag list. Lets
    # the matrix UI render the "Will see: N of M rules" preview live
    # as the admin edits a user's tags — no extra roundtrip.
    exposed_rule_tags: list[list[str]] = []
    for r in (cfg.PIPELINE_RULES or []):
        rd = r.model_dump() if hasattr(r, "model_dump") else r
        if not isinstance(rd, dict):
            continue
        if rd.get("type") == "terminal":
            continue
        if not rd.get("exposed"):
            continue
        exposed_rule_tags.append(list(rd.get("tags") or []))
    users = api_keys_store.list_users()
    # Annotate each user with their active key count for the card header.
    # Batched: one GROUP BY query instead of N list_keys() roundtrips.
    counts = api_keys_store.active_key_counts()
    # Newest last_used_ts across each user's non-revoked keys — feeds the
    # header's "last active" timestamp + activity dot (renderUser builds the
    # header synchronously, before the per-user /keys fetch, so this can't be
    # derived client-side from the rendered key cards). Same batched shape as
    # the counts above.
    last_used = api_keys_store.last_used_by_user()
    out = [
        {
            **u,
            "active_key_count": counts.get(u["id"], 0),
            "last_used_ts": last_used.get(u["id"]),
            # permissions is already in `u` via _row_to_user_dict — keep
            # the canonical key name so the matrix UI can read it
            # directly without a second roundtrip.
        }
        for u in users
    ]
    return JSONResponse({
        "users": out,
        "open_mode": not api_keys_store.is_locked_down(),
        # Surface the page model so the front-end matrix can render
        # column headers without hardcoding them — keeps server +
        # client in sync when a new page is added.
        "pages": list(api_keys_store.PAGES),
        "scoped_pages": sorted(api_keys_store.SCOPED_PAGES),
        "access_only_pages": sorted(api_keys_store.ACCESS_ONLY_PAGES),
        # Union of every tag currently used by any rule. The matrix's
        # tag-picker uses this for autocomplete + "tags actually in
        # use" hints. Empty list means no rule is tagged yet, which
        # is the day-0 migration state.
        "available_tags": config_store.pipeline_rule_tags(cfg.PIPELINE_RULES),
        # Tag list per exposed rule — for the "Will see: N of M" preview.
        "exposed_rule_tags": exposed_rule_tags,
    })


@router.post(
    "/api/users",
    dependencies=[Depends(require_admin)],
)
async def create_user_api(payload: CreateUserIn) -> JSONResponse:
    try:
        uid = api_keys_store.create_user(payload.username, payload.is_admin)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"user_id": uid})


@router.delete(
    "/api/users/{uid}",
    dependencies=[Depends(require_admin)],
)
async def revoke_user_api(uid: str) -> JSONResponse:
    user = api_keys_store.get_user(uid)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    if user["revoked_ts"] is not None:
        return JSONResponse({"ok": True, "already_revoked": True})
    try:
        api_keys_store.revoke_user(uid)
    except api_keys_store.LastAdminError as e:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{e} — create another admin first",
        )
    return JSONResponse({"ok": True})


@router.get(
    "/api/users/{uid}/keys",
    dependencies=[Depends(require_admin)],
)
async def list_user_keys_api(uid: str) -> JSONResponse:
    if api_keys_store.get_user(uid) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return JSONResponse({"keys": api_keys_store.list_keys(user_id=uid)})


@router.get(
    "/api/usage",
    dependencies=[Depends(require_admin)],
)
async def usage_api(days: int = 0) -> JSONResponse:
    """Per-user and per-key usage rollup for the cards. `days=0` (default)
    is lifetime; `days=N` is the trailing N-day window. Returned as id-keyed
    maps so the front-end can join onto the users + keys it already renders
    (no server-side name resolution needed). Per-user totals include every
    one of that user's keys plus any pre-feature backfilled usage; per-key
    totals cover only real keys (backfill has no key id)."""
    # Window in UTC epoch-hours; the N-day window is reckoned in the server's
    # local timezone (admin/operator perspective). days=0 => lifetime.
    start_hour = None
    if days and days > 0:
        start_hour = usage_store.local_day_start_hour(days_ago=int(days) - 1)
    # Off the loop: with days=0 (what every admin keys-page load sends) there is
    # no WHERE clause, so these are two full-table GROUP BY scans over the
    # never-pruned `usage_hourly` — a stall that grows monotonically with
    # deployment age and freezes in-flight transcriptions and WebSockets.
    def _gather() -> tuple[Any, dict[str, Any]]:
        return (
            usage_store.totals_by_user(start_hour=start_hour),
            {
                r["key_id"]: r
                for r in usage_store.totals_by_key(start_hour=start_hour)
            },
        )

    by_user, by_key = await asyncio.to_thread(_gather)
    return JSONResponse({"by_user": by_user, "by_key": by_key, "days": days})


@router.post(
    "/api/users/{uid}/keys",
    dependencies=[Depends(require_admin)],
)
async def create_user_key_api(uid: str, payload: CreateKeyIn) -> JSONResponse:
    """Show-once raw key on creation. Subsequent reads via list_user_keys
    never return the raw value. A label is mandatory at this boundary (the
    store stays lenient for internal/test callers); blank/whitespace -> 400."""
    label = payload.label.strip()
    if not label:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "label is required")
    try:
        raw_key, rec = api_keys_store.create_key(uid, label=label)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"key": raw_key, "record": rec})


@router.delete(
    "/api/users/{uid}/keys/{kid}",
    dependencies=[Depends(require_admin)],
)
async def revoke_key_api(uid: str, kid: str) -> JSONResponse:
    key = api_keys_store.get_key(kid)
    if key is None or key["user_id"] != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    if key["revoked_ts"] is not None:
        return JSONResponse({"ok": True, "already_revoked": True})
    try:
        api_keys_store.revoke_key(kid)
    except api_keys_store.LastAdminError as e:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{e} — generate another admin key first",
        )
    return JSONResponse({"ok": True})


@router.patch(
    "/api/users/{uid}/permissions",
    dependencies=[Depends(require_admin)],
)
async def patch_user_permissions_api(
    uid: str, payload: PatchPermissionsIn,
) -> JSONResponse:
    """PATCH-merge per-page permissions onto a user. Returns the
    canonical post-merge shape so the matrix UI can echo it back into
    its rendered state without a second GET.

    Admins still validate + persist (so a future demote-to-non-admin
    path picks up the saved defaults) but their `is_admin` flag
    short-circuits all page checks at request time — the matrix UI
    greys their row out for clarity."""
    target = api_keys_store.get_user(uid)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    if target["revoked_ts"] is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "user is revoked")
    try:
        merge_payload: dict[str, Any] = {"pages": payload.pages}
        if payload.quick_config_tags is not None:
            merge_payload["quick_config_tags"] = payload.quick_config_tags
        if payload.config is not None:
            merge_payload["config"] = payload.config.model_dump()
        merged = api_keys_store.set_user_permissions(uid, merge_payload)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"ok": True, "permissions": merged})


@router.patch(
    "/api/users/{uid}/keys/{kid}/label",
    dependencies=[Depends(require_admin)],
)
async def rename_key_api(
    uid: str, kid: str, payload: RenameKeyIn,
) -> JSONResponse:
    """Rename an existing key's display label. 404 if the key isn't this
    user's; 409 if revoked (revoked keys are read-only); 400 on a blank or
    over-long label. Renaming never exposes or rotates the secret."""
    key = api_keys_store.get_key(kid)
    if key is None or key["user_id"] != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    if key["revoked_ts"] is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "key is revoked")
    try:
        rec = api_keys_store.update_key_label(kid, payload.label)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if rec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    return JSONResponse({"ok": True, "record": rec})


@router.patch(
    "/api/users/{uid}/keys/{kid}/config",
    dependencies=[Depends(require_admin)],
)
async def patch_key_config_api(
    uid: str, kid: str, payload: ConfigBindingIn,
) -> JSONResponse:
    """Validate + persist a per-key config binding (override profiles + direct
    overrides + locks). Returns the stored {"direct": …, "profiles": …} so the
    drawer can echo it back. 404 if the key isn't this user's; 409 if revoked."""
    key = api_keys_store.get_key(kid)
    if key is None or key["user_id"] != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    if key["revoked_ts"] is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "key is revoked")
    try:
        binding = api_keys_store.set_key_config(uid, kid, payload.model_dump())
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"ok": True, "config": binding})


# ---------------------------------------------------------------------
# Synced client settings (desktop sync) — per-account admin management.
#
# The desktop app syncs its settings through the user-tier
# /v1/synced-client-settings endpoint (client_settings_routes). These admin
# endpoints add per-account visibility + management on the keys page.
# The stored blob is OPAQUE, SENSITIVE client JSON (by user choice it can
# include the account's saved API keys), so the metadata endpoint never
# returns blob contents — the blob itself only moves as an explicit
# export download or import upload, and the page warns before both.
# ---------------------------------------------------------------------

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


async def _cs_user_or_404(uid: str) -> dict[str, Any] | None:
    """Resolve the account a synced-settings row belongs to. Accept any
    known user id — revoked included, their stored blob stays exportable/
    deletable — plus the "(open-mode)" sentinel row the desktop writes
    while the server runs without admin keys. Unknown ids 404 so a typoed
    path can't create or expose rows."""
    if uid == api_keys_store.OPEN_MODE_USER["user_id"]:
        return None
    user = await asyncio.to_thread(api_keys_store.get_user, uid)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return user


@router.get(
    "/api/client-settings",
    dependencies=[Depends(require_admin)],
)
async def client_settings_meta_api() -> JSONResponse:
    """Per-account synced-settings metadata for the header chips + drawers,
    id-keyed like /api/usage so the client joins it onto the users it
    already renders. v1 desktop clients always use the default (profile='')
    set; named sets stay out of this map until they exist.
    Store never initialized (init_db failed at boot) → empty map rather than
    a 503 (the page must still render its users), with `unavailable: true`
    so the drawers say "store down" instead of the falsehood "nothing
    stored" — the flag, not the drawer endpoints, is what tells the admin."""
    unavailable = False
    try:
        # Off the loop like the /v1 siblings (client_settings/routes.py).
        rows = await asyncio.to_thread(client_settings_store.list_meta)
    except client_settings_store.StoreUnavailable:
        rows = []
        unavailable = True
    by_user = {
        r["user_id"]: {
            "version": r["version"],
            "bytes": r["bytes"],
            "updated_ts": r["updated_ts"],
            "device": r["device"],
        }
        for r in rows
        if r["profile"] == ""
    }
    return JSONResponse({"by_user": by_user, "unavailable": unavailable})


@router.get(
    "/api/users/{uid}/client-settings/export",
    dependencies=[Depends(require_admin)],
)
async def export_client_settings_api(uid: str) -> Response:
    """Download the stored settings document as a JSON file. Pretty-printed
    — a human inspecting/restoring it beats byte-fidelity, and it re-parses
    to the identical document (it re-imports through the WebUI's Import
    button, which recognises this endpoint's filename). The file may
    include the account's saved API keys; the WebUI's two-press guard warns
    before this endpoint is ever hit."""
    user = await _cs_user_or_404(uid)
    try:
        row = await asyncio.to_thread(client_settings_store.get, uid)
    except client_settings_store.StoreUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from None
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no synced settings stored")
    name = user["username"] if user else "open-mode"
    name = _FILENAME_SAFE.sub("-", name).strip("-") or "user"
    return Response(
        json.dumps(row["blob"], ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition":
                f'attachment; filename="client-settings_{name}_v{row["version"]}.json"',
            "Cache-Control": "no-store",
        },
    )


@router.post(
    "/api/users/{uid}/client-settings/import",
    dependencies=[Depends(require_admin)],
)
async def import_client_settings_api(
    uid: str, payload: ImportClientSettingsIn,
) -> JSONResponse:
    """Force-write a settings document for the account. The version bumps
    past whatever is stored, so every device's next sync sees a newer
    server copy and applies it through its normal merge path — no
    device-side changes needed."""
    await _cs_user_or_404(uid)
    try:
        # Off the loop: force_put json.dumps + encodes the whole blob BEFORE
        # the 512 KB cap can reject it (see client_settings_routes.put_client_settings).
        state = await asyncio.to_thread(
            client_settings_store.force_put,
            uid, payload.blob, device="WebUI import",
        )
    except client_settings_store.StoreUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from None
    except client_settings_store.InvalidBlob:
        # Before `except ValueError` — InvalidBlob subclasses it.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "settings blob must be strict JSON (no NaN/Infinity, no lone surrogates)",
        )
    except ValueError:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, "settings blob too large",
        )
    return JSONResponse({
        "ok": True,
        "version": state["version"],
        "updated_ts": state["updated_ts"],
    })


@router.delete(
    "/api/users/{uid}/client-settings",
    dependencies=[Depends(require_admin)],
)
async def delete_client_settings_api(uid: str) -> JSONResponse:
    """Remove the account's server copy. Devices keep their local settings;
    a device still holding version N gets a 409 on its next push, correctly
    surfacing the deletion (see client_settings_store.delete)."""
    await _cs_user_or_404(uid)
    try:
        deleted = await asyncio.to_thread(client_settings_store.delete, uid)
    except client_settings_store.StoreUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from None
    return JSONResponse({"ok": True, "deleted": deleted})


# ---------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------

_API_KEYS_HTML = templates.load(__file__, "api_keys.html")
