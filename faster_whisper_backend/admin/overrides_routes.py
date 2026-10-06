"""
Admin API for the layered per-identity config-override feature.

Serves the dedicated /settings/overrides page's JSON contract:
  GET  /settings/overrides/state    profiles + field metadata + groups + rules + usage
  POST /settings/overrides/state    save the OVERRIDE_PROFILES dirty diff (hot-applied)
  GET  /settings/overrides/resolve  the effective-config WATERFALL for a user/key/model
                                    (drives the Explorer + the in-context preview)

The HTML page + nav wiring live in this module too once the UI lands (phase 6);
for now this is the backend the binding editors and the Explorer call. All
endpoints are admin-only (host allowlist + admin key).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from faster_whisper_backend.auth import api_keys_store
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.core import web_common
from faster_whisper_backend.pipeline import apply as pl_apply
from faster_whisper_backend.auth.dependencies import require_admin
from faster_whisper_backend.core import templates

logger = logging.getLogger("whisper.overrides")

require_admin_webui_host = web_common.require_admin_webui_host

router = APIRouter(prefix="/settings/overrides")

@functools.lru_cache(maxsize=1)
def _build_field_meta() -> dict[str, dict[str, Any]]:
    """Widget metadata (kind / min / max / opts) for every overridable field,
    derived from the OverrideProfile JSON schema (via the shared
    settings_schema.override_field_meta) so it can never drift from the Pydantic
    bounds. Drives the profile editor + direct-override sub-editor. `locks`
    and `requestable` are profile-level metadata, not per-field overrides —
    rendered by dedicated controls, never in the field grid."""
    return settings_schema.override_field_meta(
        settings_schema.OverrideProfile, exclude={"locks", "requestable"})


def _build_defaults() -> dict[str, Any]:
    """Live effective global value (getattr(cfg, name), after config.local.json +
    env) for every overridable field — what a profile inherits when it doesn't set
    the field. Reuses the per-model page's serializer (pl_apply.resolved_value),
    so the overrides page's `inherits <value>` hint matches /settings byte-for-byte.
    Rulelist fields resolve to None and are never read by the scalar field rows."""
    return {name: pl_apply.resolved_value(name) for name in _build_field_meta()}


def _build_groups() -> list[dict[str, Any]]:
    """Section layout for the profile editor: the global /settings field groups
    filtered to the per-identity overridable scalars, so section names + order
    match the rest of the admin UI (Decode / Advanced / VAD / Live streaming /
    Output …). Load-time + server sections drop out entirely."""
    target = settings_schema.LOCKABLE_FIELDS
    out: list[dict[str, Any]] = []
    for section, subs in settings_schema.FIELD_GROUPS:
        subgroups = []
        for sub_title, names in subs:
            fields = [n for n in names if n in target]
            if fields:
                subgroups.append({"title": sub_title, "fields": fields})
        if subgroups:
            out.append({"title": section, "subgroups": subgroups})
    return out


def _build_rules() -> list[dict[str, Any]]:
    """Non-terminal pipeline rules (name/label/enabled/languages/card_no) for
    the per-profile force-on/off checklist. `card_no` is the rule's 1-based
    position in cfg.PIPELINE_RULES (the /settings/pipeline card ordinal) so the
    row can show `#N`, matching the pipeline page + the /logs trace.
    `languages` lets the row flag language-scoped rules: a force-on here still
    won't fire on a non-matching language (see pipeline.engine._postprocess_text).
    `tags` are the quick-config visibility tags, shown read-only so the admin
    can tell the two scopes apart (who sees it vs. when it runs)."""
    out = []
    for i, r in enumerate(getattr(cfg, "PIPELINE_RULES", None) or [], start=1):
        if not isinstance(r, dict) or r.get("type") == "terminal":
            continue
        out.append({
            "name": r.get("name"),
            "label": r.get("label") or r.get("name"),
            "enabled": bool(r.get("enabled", True)),
            "languages": list(r.get("languages") or []),
            "tags": list(r.get("tags") or []),
            "card_no": i,
        })
    return out


def _build_usage() -> dict[str, dict[str, list[str]]]:
    """Reverse index: profile name → {users:[id…], keys:[id…]} that reference
    it, either FORCED (the binding `profiles` list, auto-applied) or merely
    REQUESTABLE (the `allowed_override_profiles` allowlist). Both fields count
    so the delete guard refuses ANY referenced profile — an allowlist-only name
    would otherwise look unused and delete silently, stranding a dangling
    reference that the binding's next save then rejects. Mirrors rename, which
    also follows both fields. Powers the sidebar usage counts and the
    usage-aware delete guard."""
    usage: dict[str, dict[str, list[str]]] = {
        name: {"users": [], "keys": []} for name in (getattr(cfg, "OVERRIDE_PROFILES", None) or {})
    }

    def _refs(binding: "dict[str, Any] | None") -> set[str]:
        b = binding or {}
        return {p for field in ("profiles", "allowed_override_profiles")
                for p in (b.get(field) or []) if isinstance(p, str)}

    for u in api_keys_store.list_users():
        uid = u["id"]
        for p in _refs(api_keys_store.get_user_config(uid)):
            usage.setdefault(p, {"users": [], "keys": []})["users"].append(uid)
        for k in api_keys_store.list_keys(uid):
            for p in _refs(k.get("config")):
                usage.setdefault(p, {"users": [], "keys": []})["keys"].append(k["id"])
    return usage


def _models() -> list[str]:
    """Model ids for the Explorer's model picker (allowlist, else default)."""
    allowed = sorted(getattr(cfg, "ALLOWED_MODELS", None) or [])
    if allowed:
        return allowed
    default = getattr(cfg, "DEFAULT_MODEL", "") or ""
    return [default] if default else []


@router.get("/state",
            dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def get_state() -> dict[str, Any]:
    """Profiles + the metadata the editors need. Off the loop: _build_usage
    walks users/keys in SQLite — the same work post_state already moved to
    a thread. (_build_field_meta is lru_cached, so it only pays the Pydantic
    schema build on the first call.)"""
    return await asyncio.to_thread(_state_payload)


def _state_payload() -> dict[str, Any]:
    return {
        "profiles": dict(getattr(cfg, "OVERRIDE_PROFILES", None) or {}),
        "field_meta": _build_field_meta(),
        "defaults": _build_defaults(),
        "groups": _build_groups(),
        "rules": _build_rules(),
        "usage": _build_usage(),
        "models": _models(),
        # Read-only echo of the two global request gates so the page can show
        # whether requesting is enabled at all (they're edited on /settings).
        "globals": {
            "ALLOW_REQUEST_OVERRIDE_PROFILE":
                bool(getattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True)),
            "ALLOW_REQUEST_DECODE_OVERRIDES":
                bool(getattr(cfg, "ALLOW_REQUEST_DECODE_OVERRIDES", True)),
        },
    }


@router.get("", response_class=HTMLResponse,
            dependencies=[Depends(require_admin_webui_host)])
async def overrides_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_OVERRIDES_HTML, current="overrides"),
        headers={"Cache-Control": "no-store"},
    )


@router.post("/state",
             dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def post_state(payload: dict[str, Any], request: Request) -> JSONResponse:
    """Persist the OVERRIDE_PROFILES dirty diff (the client sends the full
    profiles dict under that key). Same validate → save → hot-apply contract as
    /settings/state; OVERRIDE_PROFILES is a hot field (resolved per-request, so
    no cache rebuild / model eviction needed)."""
    # Only the OVERRIDE_PROFILES key is accepted here — this page never edits
    # any other global setting.
    unknown = set(payload) - {"OVERRIDE_PROFILES"}
    if unknown:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"unexpected fields for this page: {sorted(unknown)}")
    # The "profile is in use — unbind it first" rule existed ONLY in the page
    # JS (startDelete), and _build_usage's own docstring already calls itself
    # "the usage-aware delete guard". save_overrides performs no such check —
    # validate_profile_refs enforces existence on the BINDING save path, the
    # opposite direction — so a POST from curl, or from a tab whose usage map
    # went stale against a concurrent binding change, deleted a referenced
    # profile outright. That fails OPEN rather than merely dangling:
    # effective_config resolves a missing name to None and silently DROPS the
    # layer, so every `locks` entry it contributed disappears and fields the
    # admin had pinned against per-request decode_override become
    # client-overridable again, with no error anywhere.
    # Presence-keyed, not isinstance-keyed: `{"OVERRIDE_PROFILES": null}` is
    # the save_overrides remove-the-whole-key sentinel, which deletes EVERY
    # profile — so a non-dict value must count all current profiles as removed
    # rather than skip the guard.
    if "OVERRIDE_PROFILES" in payload:
        incoming = payload["OVERRIDE_PROFILES"]
        keep = set(incoming) if isinstance(incoming, dict) else set()
        current = getattr(cfg, "OVERRIDE_PROFILES", None) or {}
        removed = set(current) - keep
        if removed:
            usage = await asyncio.to_thread(_build_usage)
            in_use = sorted(
                name for name in removed
                if (usage.get(name, {}).get("users")
                    or usage.get(name, {}).get("keys"))
            )
            if in_use:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "still bound to a user or key — unbind on the API keys "
                    f"page first: {in_use}",
                )
    try:
        # Off the loop: save_overrides re-validates the merged config, which
        # re-runs regex_guard (a child process with a 2 s budget) and then
        # rewrites config.local.json. Inline that froze every request, SSE
        # stream and streaming WebSocket on the worker for its duration.
        written = await asyncio.to_thread(config_store.save_overrides, payload)
    except ValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"errors": settings_schema.format_validation_errors(e)},
        )
    except OSError as e:
        logger.error("[overrides] save failed: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"could not write config.local.json: {e}")

    applied = await pl_apply.apply_hot_changes(written)
    client_host = request.client.host if request.client else "?"
    logger.info("[overrides] profiles update from=%s saved=%s",
                client_host, sorted(written.keys()))
    return JSONResponse({
        "saved": sorted(written.keys()),
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
    })


class _RenameProfileIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    old: str = Field(min_length=1)
    new: str = Field(min_length=1)


@router.post("/profiles/rename",
             dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def rename_profile(payload: _RenameProfileIn, request: Request) -> JSONResponse:
    """Rename an override profile (its dict key in OVERRIDE_PROFILES) and cascade
    the new name through every per-user / per-key binding that references it —
    the binding `profiles` list and `allowed_override_profiles` allowlist. The
    profile's overrides are preserved untouched.

    Unlike delete (which refuses an in-use profile and asks the admin to unbind
    first), rename FOLLOWS the references so in-use bindings keep resolving — the
    whole point of renaming is usually to retitle a profile that is already in
    use. The library's dict order is irrelevant (the page sorts for display), so
    the rename keeps the surviving overrides exactly and only swaps the key."""
    profiles = dict(getattr(cfg, "OVERRIDE_PROFILES", None) or {})
    old = payload.old.strip().lower()
    new = payload.new.strip().lower()
    if old not in profiles:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown profile {old!r}")
    if new == old:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "new name is the same as the current name")
    if not settings_schema.TAG_RE.match(new):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "invalid profile name (a-z 0-9 -, max 32, must start "
                            "with a letter or digit)")
    if new in profiles:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"a profile named {new!r} already exists")

    # 1) Rename the key in OVERRIDE_PROFILES (preserve insertion order), then
    #    persist + hot-apply through the same path /state uses.
    renamed = {(new if k == old else k): v for k, v in profiles.items()}
    try:
        # Off the loop, same reasoning as the /state save above.
        written = await asyncio.to_thread(
            config_store.save_overrides, {"OVERRIDE_PROFILES": renamed},
        )
    except ValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"errors": settings_schema.format_validation_errors(e)},
        )
    except OSError as e:
        logger.error("[overrides] rename save failed: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"could not write config.local.json: {e}")

    # 2) Cascade the new name through all bindings (now that OVERRIDE_PROFILES
    #    holds the new key, the binding set stays referentially consistent).
    #    Off the loop like the save above: the cascade is a full users+keys
    #    scan with per-row UPDATEs under the store lock.
    #    The save above reached disk only: the running cfg still holds `old`
    #    alone until apply_hot_changes, so a request resolving a binding the
    #    cascade already moved to `new` would drop that profile layer and its
    #    locks (fail-open). Alias both names to the same bundle meanwhile so a
    #    binding resolves under either name; the hot-apply leaves only `new`.
    aliased = dict(getattr(cfg, "OVERRIDE_PROFILES", None) or {})
    if old in aliased:
        aliased[new] = aliased[old]
        setattr(cfg, "OVERRIDE_PROFILES", aliased)
    try:
        affected = await asyncio.to_thread(api_keys_store.rename_profile_refs, old, new)
    finally:
        applied = await pl_apply.apply_hot_changes(written)
    client_host = request.client.host if request.client else "?"
    logger.info("[overrides] profile renamed %r->%r from=%s bindings=%d",
                old, new, client_host, affected)
    return JSONResponse({
        "ok": True, "old": old, "new": new, "bindings_updated": affected,
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
    })


def _resolve_model(model: str | None) -> str | None:
    model = (model or "").strip()
    if not model or model == "whisper-1":
        return getattr(cfg, "DEFAULT_MODEL", "") or None
    return model


@router.get("/resolve",
            dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def resolve(user_id: str = "", key_id: str = "", model: str = "",
                  sim: str = "") -> dict[str, Any]:
    """Effective-config waterfall for (user_id [, key_id], model), optionally
    simulating a client per-request decode_override (`sim` = JSON object of
    lowercase decode keys). Returns, per field, the ordered layer stack with the
    winner, lock state, and the simulated client outcome."""
    sim_dict: dict[str, Any] = {}
    if sim.strip():
        try:
            parsed = json.loads(sim)
            if isinstance(parsed, dict):
                sim_dict = parsed
        except json.JSONDecodeError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"sim is not valid JSON: {e}")

    r = effective_config.resolve(
        _resolve_model(model),
        user_id=user_id or None, key_id=key_id or None,
        request_overrides=sim_dict, with_provenance=True,
    )

    fields: dict[str, Any] = {}
    for fname, stack in (r.provenance or {}).items():
        winner = next((h for h in stack if h["is_winner"]), None)
        client_sim = None
        ck = effective_config._CONFIG_TO_CLIENT_KEY.get(fname)
        if ck and ck in sim_dict:
            # Reflect the SAME gate the live decode path applies (transcription.
            # models._apply_decode_overrides / batch+streaming overrides_ignored key off
            # locked_client_keys): a field-level lock OR the per-identity decode
            # master gate being off (which locks every client key) → ignored.
            # Checking only r.locked would report "applied" for a key the server
            # actually drops whenever the master gate is off.
            client_sim = {
                "value": sim_dict[ck],
                "outcome": ("ignored_locked"
                            if (fname in r.locked or ck in r.locked_client_keys)
                            else "applied"),
            }
        fields[fname] = {
            "winner_value": winner["value"] if winner else None,
            "winner_layer": winner["layer_id"] if winner else None,
            "locked": fname in r.locked,
            "client_sim": client_sim,
            "layers": stack,
        }
    return {
        "fields": fields,
        "rules": r.rule_provenance or {},
        "profiles_applied": r.profiles_applied,
    }


# ---------------------------------------------------------------------
# HTML page — Profiles manager (master-detail) + effective-config Explorer
# ---------------------------------------------------------------------

_OVERRIDES_HTML = templates.load(__file__, "overrides.html")
