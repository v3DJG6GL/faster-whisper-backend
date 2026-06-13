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

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import api_keys_store
import config as cfg
import config_store
import effective_config
import web_common
from auth import require_admin

logger = logging.getLogger("whisper.overrides")

require_admin_webui_host = web_common.require_admin_webui_host

router = APIRouter(prefix="/settings/overrides")

_TYPE_KIND = {
    "integer": "int", "number": "float", "boolean": "bool",
    "string": "str", "array": "list",
}


def _build_field_meta() -> dict[str, dict[str, Any]]:
    """Widget metadata (kind / min / max / opts) for every overridable field,
    derived from the OverrideProfile JSON schema so it can never drift from the
    Pydantic bounds. Drives the profile editor + direct-override sub-editor."""
    schema = config_store.OverrideProfile.model_json_schema()
    out: dict[str, dict[str, Any]] = {}
    for name, spec in schema.get("properties", {}).items():
        if name == "locks":
            continue
        variants = spec.get("anyOf") or [spec]
        v = next((x for x in variants if x.get("type") != "null"), variants[0])
        info: dict[str, Any] = {}
        if name in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE"):
            info["kind"] = "rulelist"
        elif "enum" in v:
            info["kind"] = "enum"
            info["opts"] = v["enum"]
        else:
            info["kind"] = _TYPE_KIND.get(v.get("type"), "str")
        if "minimum" in v:
            info["min"] = v["minimum"]
        if "maximum" in v:
            info["max"] = v["maximum"]
        if "maxLength" in v:
            info["maxlen"] = v["maxLength"]
        out[name] = info
    return out


def _build_groups() -> list[dict[str, Any]]:
    """Section layout for the profile editor: the global /settings field groups
    filtered to the per-identity overridable scalars, so section names + order
    match the rest of the admin UI (Decode / Advanced / VAD / Live streaming /
    Output …). Load-time + server sections drop out entirely."""
    import admin_routes
    target = config_store.LOCKABLE_FIELDS
    out: list[dict[str, Any]] = []
    for section, subs in admin_routes._FIELD_GROUPS:
        subgroups = []
        for sub_title, names in subs:
            fields = [n for n in names if n in target]
            if fields:
                subgroups.append({"title": sub_title, "fields": fields})
        if subgroups:
            out.append({"title": section, "subgroups": subgroups})
    return out


def _build_rules() -> list[dict[str, Any]]:
    """Non-terminal pipeline rules (name/label/enabled) for the per-profile
    force-on/off checklist."""
    out = []
    for r in (getattr(cfg, "PIPELINE_RULES", None) or []):
        if not isinstance(r, dict) or r.get("type") == "terminal":
            continue
        out.append({
            "name": r.get("name"),
            "label": r.get("label") or r.get("name"),
            "enabled": bool(r.get("enabled", True)),
        })
    return out


def _build_usage() -> dict[str, dict[str, list[str]]]:
    """Reverse index: profile name → {users:[id…], keys:[id…]} that reference
    it. Powers the sidebar usage counts and the usage-aware delete guard."""
    usage: dict[str, dict[str, list[str]]] = {
        name: {"users": [], "keys": []} for name in (getattr(cfg, "OVERRIDE_PROFILES", None) or {})
    }
    for u in api_keys_store.list_users():
        uid = u["id"]
        for p in api_keys_store.get_user_config(uid).get("profiles", []):
            usage.setdefault(p, {"users": [], "keys": []})["users"].append(uid)
        for k in api_keys_store.list_keys(uid):
            for p in (k.get("config") or {}).get("profiles", []):
                usage.setdefault(p, {"users": [], "keys": []})["keys"].append(k["id"])
    return usage


@router.get("/state",
            dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def get_state() -> dict[str, Any]:
    """Profiles + the metadata the editors need."""
    return {
        "profiles": dict(getattr(cfg, "OVERRIDE_PROFILES", None) or {}),
        "field_meta": _build_field_meta(),
        "groups": _build_groups(),
        "rules": _build_rules(),
        "usage": _build_usage(),
    }


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
    try:
        written = config_store.save_overrides(payload)
    except ValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"errors": config_store.format_validation_errors(e)},
        )
    except OSError as e:
        logger.error("[overrides] save failed: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"could not write config.local.json: {e}")

    import admin_routes
    applied = await admin_routes._apply_hot_changes(written)
    client_host = request.client.host if request.client else "?"
    logger.info("[overrides] profiles update from=%s saved=%s",
                client_host, sorted(written.keys()))
    return JSONResponse({
        "saved": sorted(written.keys()),
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
            client_sim = {
                "value": sim_dict[ck],
                "outcome": "ignored_locked" if fname in r.locked else "applied",
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
