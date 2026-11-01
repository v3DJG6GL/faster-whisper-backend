"""
Admin API + page for the layered per-identity config-override feature.

  GET  /settings/overrides                  the page shell (admin host allowlist only)
  GET  /settings/overrides/state            profiles + field metadata + groups + rules + usage
  POST /settings/overrides/state            save the OVERRIDE_PROFILES dirty diff (hot-applied)
  POST /settings/overrides/profiles/rename  rename a profile + cascade it through bindings
  GET  /settings/overrides/resolve          the effective-config WATERFALL for a user/key/model
                                            (drives the Explorer + the in-context preview)

The JSON endpoints are admin-only (host allowlist + admin key).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import threading
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
from faster_whisper_backend.core.loop_lock import LoopLock
from faster_whisper_backend.pipeline import apply as pl_apply
from faster_whisper_backend.auth.dependencies import require_admin
from faster_whisper_backend.core import templates
from faster_whisper_backend.transcription import models as tx_models

logger = logging.getLogger("whisper.overrides")

require_admin_webui_host = web_common.require_admin_webui_host

router = APIRouter(prefix="/settings/overrides")

# Serializes the library's read-modify-write: post_state and rename_profile
# each snapshot cfg.OVERRIDE_PROFILES, save, and hot-apply by RE-READING disk
# (apply_hot_changes). Interleaved, a /state save still naming `old` landed
# between a rename's save and its apply, so the cascade moved every binding to
# `new` while cfg reloaded a library without it: the layer and its locks
# dropped (fail-open). Single-worker assumption, like pl_apply.rules_lock().
_PROFILES_LOCK = LoopLock()

# The third writer of profile names: the per-user / per-key binding PATCH
# routes (api_keys_routes, plain-def threadpool handlers) validate names
# against the running cfg and then write. Inside the window above, cfg still
# holds the profile a /state save is deleting, or a rename's `old` alias, so a
# binding save naming it passed and landed after the in-use guard or the
# cascade: dangling, and fail-open as above. A threading lock, so the sync
# routes can hold it; the async writers poll for it (_hold_profile_refs) rather
# than block the event loop behind a threadpool holder.
PROFILE_REFS_LOCK = threading.Lock()


@contextlib.asynccontextmanager
async def _hold_profile_refs():
    while not PROFILE_REFS_LOCK.acquire(blocking=False):
        await asyncio.sleep(0.01)
    try:
        yield
    finally:
        PROFILE_REFS_LOCK.release()


@functools.lru_cache(maxsize=1)
def _build_field_meta() -> dict[str, dict[str, Any]]:
    """Widget metadata (kind / min / max / opts) for every overridable field,
    derived from the OverrideProfile JSON schema (via the shared
    settings_schema.override_field_meta) so it can never drift from the Pydantic
    bounds. Drives the profile editor + direct-override sub-editor. `locks`
    and `requestable` are profile-level metadata, not per-field overrides —
    rendered by dedicated controls, never in the field grid."""
    meta = settings_schema.override_field_meta(
        settings_schema.OverrideProfile, exclude={"locks", "requestable"})
    # The multi-line text fields every editor renders as a <textarea>: a
    # single-line <input> strips the newlines, so editing a multi-line
    # glossary or prompt here would save it as one line.
    for name in settings_schema.MULTILINE_TEXT_FIELDS & meta.keys():
        meta[name]["kind"] = "textarea"
    return meta


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
    unknown = set(payload) - {"OVERRIDE_PROFILES"}
    if unknown:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"unexpected fields for this page: {sorted(unknown)}")
    # An env-pinned library (WHISPER_OVERRIDE_PROFILES) keeps its running
    # value: apply_hot_changes skips the pinned field, so the page would
    # reload the env value under a green "saved" while config.local.json
    # quietly holds a divergent library that takes over once the env var is
    # unset. Refuse like rename does; save() shows this detail.
    pinned_by = config_store.env_pinned_fields().get("OVERRIDE_PROFILES")
    if pinned_by and "OVERRIDE_PROFILES" in payload:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"OVERRIDE_PROFILES is pinned by {pinned_by}; "
                            "edit the profiles there")

    async def _locked() -> JSONResponse:
        async with _PROFILES_LOCK, _hold_profile_refs():
            # The server-side in-use guard: save_overrides does not check
            # references (validate_profile_refs covers only the binding save), so a
            # curl POST or a tab with a stale usage map could delete a bound
            # profile. That fails OPEN: effective_config drops the missing layer,
            # and every lock it contributed silently goes away.
            # Presence-keyed, not isinstance-keyed: `{"OVERRIDE_PROFILES": null}` is
            # save_overrides' remove-the-whole-key sentinel, so a non-dict value
            # counts every current profile as removed.
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

    # Lock + save + apply as one task a dropped connection cannot cut short
    # (see pl_apply.run_to_end).
    return await pl_apply.run_to_end(_locked())


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
    # An env-pinned library (WHISPER_OVERRIDE_PROFILES) keeps its running
    # value and comes back at the next restart: the cascade would then leave
    # every binding naming a profile the env never defines, dropping that
    # layer and its locks (fail-open). Refuse before any save or cascade.
    pinned_by = config_store.env_pinned_fields().get("OVERRIDE_PROFILES")
    if pinned_by:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"OVERRIDE_PROFILES is pinned by {pinned_by}; "
                            "rename the profile there")

    async def _locked() -> JSONResponse:
        async with _PROFILES_LOCK, _hold_profile_refs():
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

            # Cascade only after the save, so disk never names a missing profile.
            # The cascade is a full users+keys scan with per-row UPDATEs under the
            # store lock, hence off the loop. The save reached disk only: until
            # apply_hot_changes the running cfg holds `old` alone, so a binding the
            # cascade already moved to `new` would drop that layer and its locks
            # (fail-open). Alias both names to the same bundle meanwhile; the
            # hot-apply leaves only `new`.
            aliased = dict(getattr(cfg, "OVERRIDE_PROFILES", None) or {})
            if old in aliased:
                aliased[new] = aliased[old]
                setattr(cfg, "OVERRIDE_PROFILES", aliased)
            # A failed cascade must not hot-apply the rename: the bindings still
            # name `old` (the cascade is one transaction), so a cfg holding only
            # `new` would drop their profile layer and its locks. Write the
            # original profiles back and re-apply them instead; if even that
            # fails, the alias above keeps both names resolving in this process.
            try:
                affected = await asyncio.to_thread(api_keys_store.rename_profile_refs, old, new)
            except Exception as e:
                logger.error("[overrides] rename %r->%r cascade failed: %s", old, new, e)
                try:
                    restored = await asyncio.to_thread(
                        config_store.save_overrides, {"OVERRIDE_PROFILES": profiles},
                    )
                    await pl_apply.apply_hot_changes(restored)
                except Exception as rb_err:  # noqa: BLE001 — the alias stays live
                    logger.error("[overrides] rename rollback failed: %s", rb_err)
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    f"could not update the profile bindings: {e}")
            applied = await pl_apply.apply_hot_changes(written)
            client_host = request.client.host if request.client else "?"
            logger.info("[overrides] profile renamed %r->%r from=%s bindings=%d",
                        old, new, client_host, affected)
            return JSONResponse({
                "ok": True, "old": old, "new": new, "bindings_updated": affected,
                **applied,
                "requires_restart": bool(applied["cold_pending"]),
            })

    # Lock + save + apply as one task a dropped connection cannot cut short
    # (see pl_apply.run_to_end).
    return await pl_apply.run_to_end(_locked())


# Client keys the live decode path (tx_models._apply_decode_overrides) parses;
# the VAD sub-params land inside its vad_parameters dict. The live-dictation
# keys (settings_schema.STREAM_ONLY_CLIENT_KEYS) go through the streaming
# handshake's own parser instead; the rest (suppress_chars,
# output_prefix/suffix) are read elsewhere and stay reported as sent.
_VAD_SUB_KEYS = {"vad_min_silence_duration_ms": "min_silence_duration_ms",
                 "vad_speech_pad_ms": "speech_pad_ms",
                 "vad_threshold": "threshold"}
_DECODE_KEYS = frozenset({*tx_models._DECODE_INT_BOUNDS, *tx_models._DECODE_FLOAT_BOUNDS,
                          *tx_models._DECODE_STR_CAPS, "temperature", "suppress_tokens",
                          "condition_on_previous_text", "multilingual", "vad_filter"})
_STREAM_FIELDS = {ck: f for f, ck in settings_schema.CONFIG_TO_CLIENT_KEY.items()
                  if ck in settings_schema.STREAM_ONLY_CLIENT_KEYS}


def _sim_client_value(ck: str, raw: Any, model_id: str | None,
                      ident: Any, stream: tuple[dict, list[str]]) -> tuple[Any, str]:
    """(value, outcome) for one unlocked simulated client key, as the live
    decode path would use it: "applied" (as sent, or merely parsed — "5",
    "true", "0.2,0.4"), "clamped" (bounded / capped / fitted to `value`),
    "ignored_invalid" (unparsable, dropped) or, for a live-dictation outer
    silence that a LOCKED partner gate cannot fit, "ignored_partner_locked"
    (the outer itself is not locked).
    `stream` is stream_routes._client_stream_values over the whole simulated
    request: the silence gates are fitted as a set, not one key at a time."""
    if ck in settings_schema.STREAM_ONLY_CLIENT_KEYS:
        (out, notes), key = stream, _STREAM_FIELDS[ck]
        if (key not in out and key == "STREAMING_VAD_OUTER_SILENCE_MS"
                and any(n.startswith("outer ") for n in notes)):
            return raw, "ignored_partner_locked"
    elif ck in _VAD_SUB_KEYS:
        # VAD seeded on with a non-empty params dict, so the sub-param is
        # judged on its own value rather than on the config's VAD toggle.
        out = tx_models._apply_decode_overrides(
            {"vad_filter": True, "vad_parameters": {"_": None}}, model_id,
            {ck: raw}, ident=ident)["vad_parameters"]
        key = _VAD_SUB_KEYS[ck]
    elif ck in _DECODE_KEYS:
        out, key = tx_models._apply_decode_overrides({}, model_id, {ck: raw}, ident=ident), ck
    else:
        return raw, "applied"
    if key not in out:
        # A blank hotwords is the client's "clear DEFAULT_HOTWORDS".
        if ck == "hotwords" and isinstance(raw, str):
            return "", "applied"
        return raw, "ignored_invalid"
    value = out[key]
    if isinstance(value, tuple):
        value = list(value)
    # None = an explicit suppress_tokens clear.
    if value is None or value == raw:
        return value, "applied"
    if isinstance(raw, str) and not isinstance(value, str):
        for text in (raw.strip().lower(), f"[{raw}]"):
            with contextlib.suppress(ValueError):
                if json.loads(text) == value:
                    return value, "applied"
    return value, "clamped"


def _resolve_model(model: str | None) -> str | None:
    model = (model or "").strip()
    if not model or model == "whisper-1":
        return getattr(cfg, "DEFAULT_MODEL", "") or None
    return model


@router.get("/resolve",
            dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
def resolve(user_id: str = "", key_id: str = "", model: str = "",
            sim: str = "") -> dict[str, Any]:
    """Effective-config waterfall for (user_id [, key_id], model), optionally
    simulating a client per-request decode_override (`sim` = JSON object of
    lowercase decode keys). Returns, per field, the ordered layer stack with the
    winner, lock state, and the simulated client outcome.

    Plain `def` (FastAPI threadpool): the resolve reads key/user bindings from
    SQLite under api_keys_store._lock, which a profile-rename cascade holds for
    a whole scan — that wait must not land on the event loop."""
    sim_dict: dict[str, Any] = {}
    if sim.strip():
        try:
            parsed = json.loads(sim)
            if isinstance(parsed, dict):
                sim_dict = parsed
        except json.JSONDecodeError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"sim is not valid JSON: {e}")

    model_id = _resolve_model(model)
    r = effective_config.resolve(
        model_id,
        user_id=user_id or None, key_id=key_id or None,
        request_overrides=sim_dict, with_provenance=True,
    )
    # The live-dictation knobs as the streaming handshake would set them
    # (clamped, the silence gates fitted as a pair, locked keys skipped).
    # Imported here, not at module level: api_keys_routes imports this
    # module, so a streaming router that fails to import costs only this
    # call instead of /settings/overrides and /settings/api-keys.
    from faster_whisper_backend.streaming import routes as stream_routes
    stream = stream_routes._client_stream_values(sim_dict, r)

    fields: dict[str, Any] = {}
    for fname, stack in (r.provenance or {}).items():
        winner = next((h for h in stack if h["is_winner"]), None)
        client_sim = None
        ck = effective_config._CONFIG_TO_CLIENT_KEY.get(fname)
        if ck and ck in sim_dict:
            # The SAME gate the live decode path applies (transcription.
            # models._apply_decode_overrides / batch+streaming overrides_ignored key off
            # locked_client_keys): a field-level lock OR the per-identity decode
            # master gate being off (it locks every client key via
            # locked_client_keys, not r.locked) → ignored. An unlocked value is
            # reported as production would use it: clamped to the client bounds,
            # or dropped when unparsable.
            if fname in r.locked or ck in r.locked_client_keys:
                value, outcome = sim_dict[ck], "ignored_locked"
            else:
                value, outcome = _sim_client_value(ck, sim_dict[ck], model_id, r, stream)
            client_sim = {"value": value, "sent": sim_dict[ck], "outcome": outcome}
        elif fname in stream[0]:
            # A silence gate the client did not send, moved to fit one it did.
            client_sim = {"value": stream[0][fname], "sent": None, "outcome": "clamped"}
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
        # A binding-store fault locks every field fail-closed with no
        # owning layer; say so instead of "everything locked, by no one".
        "binding_fault": bool(getattr(r, "binding_fault", False)),
    }


_OVERRIDES_HTML = templates.load(__file__, "overrides.html")
