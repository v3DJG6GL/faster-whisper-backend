"""
Admin WebUI for faster-whisper-backend.

Mounted at /settings when WHISPER_ADMIN_UI=1. Endpoints:

  GET  /settings               HTML page (loopback / ADMIN_WEBUI_ALLOWED_HOSTS)
  GET  /settings/state         Resolved config + provenance + hot/cold tags
  POST /settings/state         Save overrides (validation errors -> 422)
  POST /settings/test-pipeline Dry-run PIPELINE_RULES against a sample
  POST /settings/restart       Detach a self-restart helper (Windows only)

Security model (layered):
  1. Allowlist gate:   require_admin_webui_host rejects callers not in
                       cfg.ADMIN_WEBUI_ALLOWED_HOSTS (loopback always permitted)
  2. API key:          Depends(require_admin) — bearer must resolve to a
                       user with is_admin=True. In OPEN mode (no admin
                       key exists yet) the dep yields a synthetic admin
                       so the operator can bootstrap.
  3. Pydantic schema:  AdminConfig validates body shape, types, bounds
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from typing import Annotated, Any, Literal, get_args, get_origin

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from faster_whisper_backend import build_info
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.settings import descriptions as field_descriptions
from faster_whisper_backend.runtime import system_stats
from faster_whisper_backend.runtime import model_registry
from faster_whisper_backend.translation import engine as translation
from faster_whisper_backend.core.languages import WHISPER_LANGUAGE_NAMES
from faster_whisper_backend.core import web_common
from faster_whisper_backend.pipeline import dictation_map
from faster_whisper_backend.pipeline import apply as pl_apply
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.pipeline import regex_guard
from faster_whisper_backend.transcription import progress as tx_progress
from faster_whisper_backend.translation import gating as tr_gating
from faster_whisper_backend.auth.dependencies import require_admin
from faster_whisper_backend.core import templates

logger = logging.getLogger("whisper-api")

# ---------------------------------------------------------------------------
# Per-model pane constants injected into the /settings page at render time
# ---------------------------------------------------------------------------
# The page JS needs the ModelOverride load-time field set (reload badge +
# drain-then-evict hint) and per-field widget metadata. Both used to be
# hand-maintained JS literals that mirrored config_store; they are now derived
# from the schemas here and substituted into the {{MO_*_JSON}} placeholders.

# Widget extras the JSON schema can't express, overlaid onto the schema-derived
# metadata: richer widget kinds ('textarea' for prompt-length text,
# 'nullable_float' for empty-clears-override floats, 'string' for plain text),
# spinner step sizes, and placeholder hints.
_MO_FIELD_META_OVERLAY: dict[str, dict[str, Any]] = {
    "REVISION": {"kind": "string", "placeholder": "main | <git-sha>"},
    "DEFAULT_LANGUAGE": {"kind": "string",
                         "placeholder": "e.g. en, de (empty = auto)"},
    "DEFAULT_PROMPT": {"kind": "textarea"},
    "DEFAULT_HOTWORDS": {"kind": "textarea"},
    # One "source = target" pair per line (translation engine splitlines);
    # a single-line <input> would join the pairs on the first keystroke.
    "TRANSLATION_GLOSSARY": {"kind": "textarea"},
    "NO_SPEECH_THRESHOLD": {"kind": "nullable_float", "step": 0.05},
    "LOG_PROB_THRESHOLD": {"kind": "nullable_float", "step": 0.1},
    "COMPRESSION_RATIO_THRESHOLD": {"kind": "nullable_float", "step": 0.1},
    "TEMPERATURE": {"kind": "string",
                    "placeholder": "0.0,0.2,0.4,0.6,0.8,1.0"},
    "PATIENCE": {"step": 0.1},
    "LENGTH_PENALTY": {"step": 0.1},
    "REPETITION_PENALTY": {"step": 0.05},
    "PROMPT_RESET_ON_TEMPERATURE": {"step": 0.05},
    "VAD_THRESHOLD": {"step": 0.05},
    "LANGUAGE_DETECTION_THRESHOLD": {"step": 0.05},
    "HALLUCINATION_SILENCE_THRESHOLD": {"kind": "nullable_float", "step": 0.5},
    "SUPPRESS_TOKENS": {"kind": "string",
                        "placeholder": "-1 | comma-ints | (empty = none)"},
    "SUPPRESS_CHARS": {"kind": "string",
                       "placeholder": "chars to mask, e.g. .,?!:;"},
    "PREPEND_PUNCTUATIONS": {"kind": "string"},
    "APPEND_PUNCTUATIONS": {"kind": "string"},
    "OUTPUT_PREFIX": {"kind": "string"},
    "OUTPUT_SUFFIX": {"kind": "string"},
}


def _mo_field_meta() -> dict[str, dict[str, Any]]:
    """Schema-derived ModelOverride widget metadata merged with the overlay."""
    meta = settings_schema.override_field_meta(settings_schema.ModelOverride)
    for name, extra in _MO_FIELD_META_OVERLAY.items():
        meta[name] = {**meta.get(name, {}), **extra}
    return meta


# Both are static per process (schemas are fixed at import), so serialize once.
_MO_LOAD_TIME_FIELDS_JSON: str = json.dumps(sorted(
    settings_schema.LOAD_TIME_FIELDS & set(settings_schema.ModelOverride.model_fields)
))
_MO_FIELD_META_JSON: str = json.dumps(_mo_field_meta())
_WHISPER_LANGS_JSON: str = json.dumps(
    [{"code": c, "name": n} for c, n in WHISPER_LANGUAGE_NAMES.items()])

# Fields the WebUI is allowed to surface — drives section grouping in the HTML
# and the /settings/state endpoint's provenance map. Generated in settings/schema.py
# from the per-field registry metadata (group/subgroup/order on each
# AdminConfig field); see settings_schema._GROUP_ORDER for the section layout.
# Section groups: each section can have one or more SUB-groups. A subgroup
# title of None means "no subheader" — fields render directly under the
# section.
_FIELD_GROUPS: list[tuple[str, list[tuple[str | None, list[str]]]]] = (
    settings_schema.FIELD_GROUPS
)

def _all_fields() -> list[str]:
    """Flat list of every field name across all sections + subgroups, in
    display order. Used by the /state endpoint and post_state echo paths."""
    out: list[str] = []
    for _section, subs in _FIELD_GROUPS:
        for _sub_title, names in subs:
            out.extend(names)
    return out


# --- auth deps ---------------------------------------------------------------
#
# /settings is gated by an IP/CIDR allowlist (cfg.ADMIN_WEBUI_ALLOWED_HOSTS,
# loopback always implicit) AND by `Depends(require_admin)` — an API key
# resolving to is_admin=True. In open mode (no admin key configured yet)
# require_admin yields the synthetic admin so the operator can bootstrap.
# The concrete host gate lives in web_common (shared, always-loaded home).
require_admin_webui_host = web_common.require_admin_webui_host


# --- router ------------------------------------------------------------------

router = APIRouter(prefix="/settings")


def _provenance(field: str, env_pinned: dict[str, str], saved: dict[str, Any]) -> str:
    """Where the current effective value came from: 'env', 'local.json', or 'default'."""
    if field in env_pinned:
        return "env"
    if field in saved:
        return "local.json"
    return "default"


def _baseline_value(name: str) -> Any:
    """The in-repo default captured in cfg._BASELINE before local.json + env
    overrides apply. Used by the WebUI's "↺ Reset" button and by post_state's
    prune-on-default logic. Convert non-JSON-serializable types (set, frozenset,
    tuple of tuples) the same way pl_apply.resolved_value does so the round-trip is clean.
    """
    baseline = getattr(cfg, "_BASELINE", {}) or {}
    v = baseline.get(name)
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    if isinstance(v, tuple):
        return [list(p) if isinstance(p, tuple) else p for p in v]
    return v


def _field_choices(field: str) -> list[str] | None:
    """Dropdown options for a field, derived from its AdminConfig `Literal` type —
    the single source of truth for enum settings. Returns None for non-Literal
    fields. Unwraps `Literal[...] | None`. The WebUI renders a <select> whenever
    this is non-None (main form AND the per-model pane), so the dropdown options
    can never drift from the schema — they ARE the schema."""
    fld = settings_schema.AdminConfig.model_fields.get(field)
    if fld is None:
        return None
    for cand in (fld.annotation, *get_args(fld.annotation)):
        if get_origin(cand) is Literal:
            return list(get_args(cand))
    return None


def _values_equal(a: Any, b: Any) -> bool:
    """True if `a` and `b` are the same config value. Uses Python equality so
    numbers compare numerically — JSON int 1 == baseline float 1.0, the
    REPETITION_PENALTY case where the JS client submits a whole-number float
    without its decimal. That matches the client, whose JS has no int/float
    distinction (JSON.stringify(1.0) === JSON.stringify(1) === "1"). Containers
    compare structurally (dict order-independent, list order-sensitive, with the
    same numeric rule applied to nested values). json.dumps string comparison
    can NOT be used here: it renders 1 as "1" but 1.0 as "1.0"."""
    return a == b


# Structured fields with bespoke override management, exempt from
# prune-on-default. PIPELINE_RULES: a local copy equal to the factory rules
# intentionally SHADOWS config.json (pins against future factory edits) and is
# cleared via the pipeline page's dedicated "clear local override" action — not
# the scalar "↺ Reset" button. (Its wire form is also pl_apply.canon_rules-normalized,
# so a submitted-value vs raw-baseline compare here would be unreliable anyway.)
_PRUNE_EXEMPT = frozenset({"PIPELINE_RULES"})


def _prune_defaults_to_removal(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert any field whose submitted value equals its in-repo baseline into
    the None removal sentinel, so save_overrides() DROPS the key from
    config.local.json instead of persisting a default-valued override.

    Without this, the WebUI's "↺ Reset to default" button — which submits the
    default *value*, not a removal — would leave the key in config.local.json
    and the "local.json" provenance badge would wrongly persist after a reset.
    Matches the client's reset-link equality (numbers compared numerically, as
    JS does) so the two sides agree on what counts as an override. None values
    already mean "remove", so they pass through untouched. Fields in
    _PRUNE_EXEMPT manage their own overrides and are never auto-pruned.
    """
    cleaned: dict[str, Any] = {}
    for k, v in payload.items():
        if k not in _PRUNE_EXEMPT and v is not None and _values_equal(v, _baseline_value(k)):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


# Stats-registry prefixes of the NON-decode model families, which share
# runtime/model_registry's loaded-model registry with whisper (whisper registers bare
# model names): translation._STATS_PREFIX / diarization._STATS_PREFIX /
# bgm_separation._STATS_PREFIX.
_NON_DECODE_PREFIXES = ("gguf:", "pyannote:", "uvr:")


def _server_ident_fields() -> dict[str, str]:
    """The "This server" identity card's values (build + runtime facts), built
    per request so the uptime is fresh. They ride the admin-gated
    GET /settings/state and are written into the card client-side: the page
    shell itself is only host-gated (no key), so versions/engines/paths must
    never be baked into it. Values never include secrets.
    cfg._DATA_DIR/_DB_DIR keep their underscore on purpose: uppercase config
    globals get swept into the admin-editable _BASELINE, and these resolved
    layout facts must stay out of it (see config.py)."""
    # Device word from what the server actually decodes on, not from NVML
    # merely finding a card: a box with an NVIDIA GPU but MODEL_DEVICE=cpu
    # (or a CUDA-less CTranslate2 build after the cuda→cpu fallback) must not
    # claim "gpu — …" in the card or in pasted bug reports. Prefer a loaded
    # model's observed device; fall back to the configured one before any
    # model has loaded.
    gpu = system_stats.gpu_name()
    # The snapshot is shared by EVERY model family in load order; only a
    # decode (whisper) model's observed device may drive the device word — a
    # cuda pyannote pipeline on a MODEL_DEVICE=cpu box must not flip the card
    # (and a cpu gguf translator loaded first must not hide a cuda decode).
    loaded = model_registry.loaded_models_snapshot()
    _dec = next(
        (e for e in loaded
         if not str(e.get("name") or "").startswith(_NON_DECODE_PREFIXES)),
        None)
    device = str((_dec.get("device") if _dec else None)
                 or getattr(cfg, "MODEL_DEVICE", "") or "")
    if device.startswith("cuda"):
        device_word = f"gpu — {gpu}" if gpu else "gpu"
    else:
        device_word = "cpu" + (f" ({gpu} present)" if gpu else "")
    runs = f"{build_info.runs_as()} · {device_word}"
    engines = build_info.engine_versions()
    boot = (
        f"{build_info.BOOT_ID[:8]} · started {build_info.STARTED_UTC}"
        f" · up {build_info.uptime_str()}"
    )
    # Resolve once and reuse for both the card fields and the report line, so
    # the two can never disagree: DOWNLOAD_ROOT is `str | None` (None = the
    # standard HF cache), and f-stringing the raw value handed the copy button
    # a literal "models None".
    data_dir = cfg._DATA_DIR or "(unset)"
    db_dir = cfg._DB_DIR or "(unset)"
    models_dir = cfg.DOWNLOAD_ROOT or "(HF default cache)"
    return {
        "version": build_info.APP_VERSION,
        "runs_as": runs,
        "engine": engines,
        "boot": boot,
        "data_dir": data_dir,
        "db_dir": db_dir,
        "models_dir": models_dir,
        # Pre-joined copy-report payload — the page hands it straight to the
        # card's copy button (_fwCopyBuild reads it from data-build).
        "report": "\n".join([
            f"{build_info.SERVER_NAME} {build_info.APP_VERSION}",
            f"runs-as: {runs} · {engines}",
            f"boot {build_info.BOOT_ID[:8]} · started {build_info.STARTED_UTC}",
            f"data {data_dir} · db {db_dir} · models {models_dir}",
        ]),
    }


# Static chrome for the identity card — no runtime facts, so it is safe in the
# keyless shell. The <dl> is filled (and .ready set, which the CSS requires to
# reveal it) by renderServerIdent() once /settings/state resolves. The copy
# button reuses _fwCopyBuild from the header vtag fragment.
_SERVER_IDENT_SHELL = (
    '<div id="srv-ident"><section aria-label="Server identity">'
    '<div class="si-head">'
    '<span class="si-title">This server</span>'
    '<span class="si-sub">build &amp; runtime identity</span>'
    '<button type="button" class="si-copy" data-build="" '
    'onclick="_fwCopyBuild(this)">copy report</button>'
    '</div><dl id="si-facts"></dl></section></div>'
)


@router.get("", response_class=HTMLResponse, dependencies=[Depends(require_admin_webui_host)])
async def settings_page() -> HTMLResponse:
    """The admin HTML page. Allowlist-gated (loopback always allowed) — no
    token required to LOAD the page; the page itself collects the token and
    attaches it on every fetch. `no-store` so browsers never serve a stale
    build after a service restart."""
    return HTMLResponse(
        web_common.render_page(
            _SETTINGS_VIEWER_HTML.replace("{{SETTINGS_VIEW}}", "settings")
            .replace("{{SERVER_IDENT}}", _SERVER_IDENT_SHELL)
            .replace("{{MO_LOAD_TIME_FIELDS_JSON}}", _MO_LOAD_TIME_FIELDS_JSON)
            .replace("{{MO_FIELD_META_JSON}}", _MO_FIELD_META_JSON)
            .replace("{{WHISPER_LANGS_JSON}}", _WHISPER_LANGS_JSON),
            current="settings"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/pipeline", response_class=HTMLResponse,
            dependencies=[Depends(require_admin_webui_host)])
async def pipeline_page() -> HTMLResponse:
    """The dedicated Pipeline-rules page. Serves the SAME settings shell + rule
    editor as /settings, but the IIFE reads the `settings-view=pipeline` meta
    and renders ONLY the Pipeline section (global rule list + dry-run panel).
    /settings renders everything else and shows a link here in its place. Same
    /settings/state + /settings/factory-rules + /settings/test-pipeline
    contracts — no separate backend. Allowlist-gated exactly like /settings."""
    return HTMLResponse(
        web_common.render_page(
            # No identity card on the focused pipeline editor — settings only.
            _SETTINGS_VIEWER_HTML.replace("{{SETTINGS_VIEW}}", "pipeline")
            .replace("{{SERVER_IDENT}}", "")
            .replace("{{MO_LOAD_TIME_FIELDS_JSON}}", _MO_LOAD_TIME_FIELDS_JSON)
            .replace("{{MO_FIELD_META_JSON}}", _MO_FIELD_META_JSON)
            .replace("{{WHISPER_LANGS_JSON}}", _WHISPER_LANGS_JSON),
            current="pipeline"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/state", dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def get_state(response: Response) -> dict[str, Any]:
    """Return the resolved config (current effective values) plus provenance
    flags so the WebUI can render badges. Does NOT include the saved-only
    overrides — the form fills from effective values; the badge tells the
    user where the value is coming from."""
    # The field map carries resolved secret values (HF_TOKEN is the
    # HuggingFace credential) and the absolute data/db/models paths. The page
    # shell and every sibling data endpoint already send this; without it the
    # payload is heuristically cacheable by the browser and any intermediary.
    response.headers["Cache-Control"] = "no-store"
    saved = await asyncio.to_thread(config_store.load_overrides)
    env_pinned = config_store.env_pinned_fields()
    field_descs = field_descriptions.FIELD_DESCRIPTIONS
    pyd_fields = settings_schema.AdminConfig.model_fields

    fields: dict[str, dict[str, Any]] = {}
    for name in _all_fields():
        # Description preference: Pydantic schema > FIELD_DESCRIPTIONS dict
        # (they're the same string in practice; schema wins so reload picks
        # up live edits to FIELD_DESCRIPTIONS without a service restart).
        desc = ""
        if name in pyd_fields and pyd_fields[name].description:
            desc = pyd_fields[name].description
        elif name in field_descs:
            desc = field_descs[name]
        fields[name] = {
            "value": pl_apply.resolved_value(name),
            "default_value": _baseline_value(name),
            "description": desc,
            "provenance": _provenance(name, env_pinned, saved),
            "env_var": env_pinned.get(name),
            "restart_required": name in settings_schema.RESTART_REQUIRED_FIELDS,
            "choices": _field_choices(name),
        }

    # PIPELINE_RULES: canonicalize key order on both sides of the wire so
    # the WebUI's deep-equal compare (JSON.stringify) is reliable on first
    # paint. See pl_apply.canon_rules() for the why.
    if "PIPELINE_RULES" in fields:
        fields["PIPELINE_RULES"]["value"] = pl_apply.canon_rules(fields["PIPELINE_RULES"]["value"])
        fields["PIPELINE_RULES"]["default_value"] = pl_apply.canon_rules(fields["PIPELINE_RULES"]["default_value"])

    # Surface the nested group structure to the client. Each group has a list
    # of subgroups: {title, subgroups: [{title: str | None, fields: [...]}]}.
    groups_payload = [
        {
            "title": section,
            "subgroups": [
                {"title": sub_title, "fields": names}
                for sub_title, names in subs
            ],
        }
        for section, subs in _FIELD_GROUPS
    ]

    return {
        "fields": fields,
        "groups": groups_payload,
        "service_name": "WhisperAPI",
        # Build/runtime facts for the "This server" card. Served here (not in
        # the keyless page shell) because this route requires admin.
        "server_ident": _server_ident_fields(),
    }


@router.post("/state", dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def post_state(payload: dict[str, Any], request: Request) -> JSONResponse:
    """Validate and persist overrides. Returns the diff (which fields were
    saved) plus a `requires_restart` flag for the WebUI to act on. Hot fields
    are applied to the running cfg module immediately and any derived caches
    are rebuilt; cold fields stick around in the JSON file for the restart to
    pick up."""
    # OVERRIDE_PROFILES is edited only on /settings/overrides (group=None, so
    # this page never sends it), and that route owns the "profile still bound
    # to a user or key" 409 guard. Accepting it here let a curl POST of
    # {"OVERRIDE_PROFILES": null} delete bound profiles and their locks.
    if "OVERRIDE_PROFILES" in payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "edit OVERRIDE_PROFILES on /settings/overrides")
    # A field submitted at its in-repo default is NOT an override — drop it from
    # config.local.json instead of rewriting it with the default value, so the
    # "↺ Reset to default" button actually clears the "local.json" badge.
    payload = _prune_defaults_to_removal(payload)
    # A PIPELINE_RULES save is a whole-key write racing /quick-config's
    # read-modify-write patches: hold the shared pipeline lock across the
    # save + hot-apply so a concurrent quick-config patch snapshots the
    # post-save list instead of silently reverting this edit. Unrelated
    # scalar saves stay off the shared lock.
    _lock = (pl_apply.rules_lock() if "PIPELINE_RULES" in payload
             else contextlib.nullcontext())
    _prev_model_overrides = (
        getattr(cfg, "MODEL_OVERRIDES", None) or {}
    ) if "MODEL_OVERRIDES" in payload else None
    async with _lock:
        try:
            # Off the loop: save_overrides validates PIPELINE_RULES through
            # regex_guard, which spawns a child interpreter and waits up to
            # _GUARD_TIMEOUT (2.0 s). test_pipeline next door already uses this
            # idiom for the same reason; the save path was left behind.
            written = await asyncio.to_thread(config_store.save_overrides, payload)
        except ValidationError as e:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"errors": settings_schema.format_validation_errors(e)},
            )
        except OSError as e:
            logger.error("[config] save failed: %s", e)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                                f"could not write config.local.json: {e}")

        applied = await pl_apply.apply_hot_changes(written, _prev_model_overrides)

    client_host = request.client.host if request.client else "?"
    logger.info(
        "[config] admin update from=%s saved=%d hot=%s cold=%s pinned=%s evicted=%s",
        client_host, len(written), applied["hot_applied"], applied["cold_pending"],
        applied["env_pinned_ignored"], applied["evicted"],
    )

    return JSONResponse({
        "saved": sorted(written.keys()),
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
    })


@router.get("/factory-rules",
            dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def get_factory_rules() -> dict[str, Any]:
    """Return the committed factory pipeline rules (config.json).

    The WebUI fetches this just before a "promote" so the diff dialog compares
    against the truly-current config.json. Distinct from GET /settings/state,
    which returns the EFFECTIVE rules (config.json overlaid by config.local.json).
    """
    try:
        rules = config_store.load_factory_rules()
    except RuntimeError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))
    return {"PIPELINE_RULES": pl_apply.canon_rules(rules)}


@router.post("/factory-rules",
             dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def post_factory_rules(payload: dict[str, Any], request: Request) -> JSONResponse:
    """Validate and persist the factory pipeline rules to the committed
    config.json. Whole-list replace — the WebUI's promote actions send the full
    intended config.json array (one rule spliced in, or the whole effective list
    for "promote all").

    config.json is git-tracked, so the save surfaces as a working-tree change
    the admin then commits + pushes to ship the fix to every deployment.

    After the write: refresh cfg._BASELINE (the "reset to default" baseline),
    recompute the effective cfg.PIPELINE_RULES (config.local.json still wins if
    it carries its own PIPELINE_RULES — unchanged local-override behaviour),
    and rebuild the pipeline cache. The response carries the canonicalized saved
    rules so the editor can refresh its in-memory `factoryRules` snapshot.
    """
    rules = payload.get("PIPELINE_RULES")
    # Element type matters, not just the container: save_factory_rules does
    # `[{**r, "seeded": True} for r in rules]` BEFORE model_validate, so a
    # non-mapping element raises TypeError, which neither `except
    # ValidationError` nor `except OSError` below catches — an unhandled 500
    # with a stack trace instead of this 400.
    if not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "payload must contain a 'PIPELINE_RULES' array",
        )
    # Under the shared pipeline lock: this write races /quick-config's
    # read-modify-write patches the same way post_state's does.
    async with pl_apply.rules_lock():
        try:
            saved = await asyncio.to_thread(config_store.save_factory_rules, rules)
        except ValidationError as e:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"errors": settings_schema.format_validation_errors(e)},
            )
        except OSError as e:
            logger.error("[config] factory-rules save failed: %s", e)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                                f"could not write config.json: {e}")

        # config.json IS the factory baseline — refresh the in-memory snapshot the
        # WebUI's "↺ reset to default" and /settings/state `default_value` rely on.
        if isinstance(getattr(cfg, "_BASELINE", None), dict):
            cfg._BASELINE["PIPELINE_RULES"] = [dict(r) for r in saved]

        # Recompute the EFFECTIVE rule list. config.local.json's PIPELINE_RULES
        # still wins if present (per-deployment local override — unchanged); only
        # when there is no local override does the factory list run directly.
        overrides = await asyncio.to_thread(config_store.load_overrides)
        local_rules = overrides.get("PIPELINE_RULES")
        shadowed = isinstance(local_rules, list)
        cfg.PIPELINE_RULES = local_rules if shadowed else [dict(r) for r in saved]

        await pl_apply.rebuild_caches_off_loop("factory-rules save")

    client_host = request.client.host if request.client else "?"
    logger.info("[config] factory-rules update from=%s rules=%d shadowed_by_local=%s",
                client_host, len(saved), shadowed)

    return JSONResponse({
        "saved": len(saved),
        "shadowed_by_local": shadowed,
        "rules": pl_apply.canon_rules(saved),
    })


@router.post("/factory-rules/clear-local-override",
             dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def clear_local_pipeline_override(request: Request) -> JSONResponse:
    """Remove the PIPELINE_RULES override from config.local.json so the committed
    config.json becomes the live pipeline on this deployment.

    Offered after a "promote all": once the factory file holds everything, the
    local snapshot is redundant and only shadows config.json. Clearing it makes
    config.json the runtime source here too.

    Kept as a dedicated route (not POST /settings/state with a None sentinel)
    so the clear re-reads config.json into cfg directly, instead of depending
    on cfg._BASELINE["PIPELINE_RULES"] being in sync with the file
    post_factory_rules just wrote. The None sentinel DOES revert to the
    baseline these days (pl_apply.apply_hot_changes' removal branch reads
    cfg._BASELINE), so this route is a convenience + freshness guarantee,
    not a workaround.
    """
    # Shared pipeline lock: same whole-key write race as post_state.
    async with pl_apply.rules_lock():
        try:
            await asyncio.to_thread(
                config_store.save_overrides, {"PIPELINE_RULES": None})
            factory = await asyncio.to_thread(config_store.load_factory_rules)
        except (ValidationError, RuntimeError, OSError) as e:
            logger.error("[config] clear-local-override failed: %s", e)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                                f"could not clear local override: {e}")

        cfg.PIPELINE_RULES = factory
        await pl_apply.rebuild_caches_off_loop("clearing local override")

    client_host = request.client.host if request.client else "?"
    logger.info("[config] local PIPELINE_RULES override cleared from=%s — "
                "config.json (%d rules) is now live", client_host, len(factory))
    return JSONResponse({"ok": True, "rules": len(factory)})


# Dry-run input bounds. Every rule can burn a full 2 s of a worker thread, so
# both the list length and the sample have to be bounded before the loop runs.
# The rule cap is read off AdminConfig.PIPELINE_RULES' schema so anything the
# editor is allowed to SAVE is still testable here; the sample cap is far
# above any realistic transcript the test panel sends.


def _pipeline_rules_max() -> int:
    """AdminConfig.PIPELINE_RULES' max_length, read off the schema.

    The bound sits inside an `Annotated[list[...], Field(max_length=...)]`
    union arm (`... | None`), so `model_fields[...].metadata` is EMPTY and
    the derivation has to walk the annotation's arms for the FieldInfo
    carrying a MaxLen constraint. Falls back to the historical 200 if the
    metadata layout ever changes (a test pins the derivation against that)."""
    field = settings_schema.AdminConfig.model_fields["PIPELINE_RULES"]
    for arm in (field.annotation, *get_args(field.annotation)):
        for fi in getattr(arm, "__metadata__", ()):
            for m in (*getattr(fi, "metadata", ()), fi):
                ml = getattr(m, "max_length", None)
                if isinstance(ml, int):
                    return ml
    return 200


_TEST_PIPELINE_MAX_RULES = _pipeline_rules_max()
_TEST_PIPELINE_MAX_SAMPLE = 8192


@router.post("/test-pipeline",
             dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def test_pipeline(payload: dict[str, Any]) -> JSONResponse:
    """Dry-run the full pipeline-rules list against a sample. Used by the
    WebUI's per-row live-validation badge AND the inline test panel.

    Payload: { sample: str, rules: list[dict] }
    `rules` is the live (dirty+saved) overlay from the WebUI so unsaved edits
    are testable. Each rule is the same dict shape as in cfg.PIPELINE_RULES.

    Response:
      {
        "steps": [
          { ordinal, label, type, before, after, matches, skipped, error, slow }, ...
          { ordinal, label: "Trim edges", type: "terminal", ... }
        ],
        "final": str,
      }

    Over-long lists (> _TEST_PIPELINE_MAX_RULES) and samples
    (> _TEST_PIPELINE_MAX_SAMPLE chars) are rejected with 400.

    Each rule is compiled and run under a 2 s threading-timer guard against
    the sample, off the event loop. Disabled rules render as `skipped: true`
    (not run). Rules with empty patterns also `skipped: true`. Compile errors
    → `error: "<msg>"` and the pipeline continues with the un-modified text.
    The terminal trim is appended at the end; if no terminal row is present,
    the trim is still applied (matching the engine's behaviour).
    """
    import threading

    # Advisory shown instead of starting a thread on an exponential shape: a
    # timed-out guard thread here is ABANDONED, not killed (CPython cannot
    # interrupt re.sub), so each one would pin a core for the life of the
    # process. A NEW save of the same shape is refused (config_store ->
    # regex_guard.validate) — but a rule saved BEFORE the guard tightened
    # still compiles and runs in the engine (pl_engine.rebuild_caches does no
    # structural screen), so the message must not claim engine parity, and
    # `not_run` marks the step so the panel renders it as a warning, not as
    # "the engine skips this too".
    _NESTED_REP_MSG = (
        "nested repetition (catastrophic backtracking risk) — not executed "
        "here: a timed-out dry-run thread cannot be interrupted. A NEW save "
        "of this pattern would be rejected; if the rule is already saved it "
        "IS still running in the live pipeline, so this step is not what "
        "production produces"
    )

    sample = str(payload.get("sample") or "")
    rules = payload.get("rules") or []
    if not isinstance(rules, list):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "rules must be a list of rule dicts"},
        )
    if len(rules) > _TEST_PIPELINE_MAX_RULES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"too many rules (max {_TEST_PIPELINE_MAX_RULES})"},
        )
    if len(sample) > _TEST_PIPELINE_MAX_SAMPLE:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"sample too long (max {_TEST_PIPELINE_MAX_SAMPLE} chars)"},
        )

    def _run_rule(text: str, rule: dict) -> dict[str, Any]:
        """Apply one rule to `text`. Returns the step dict for the response."""
        # Dry-run contract is parity with the pipeline engine — use its replacer
        # factories rather than a second copy that could drift.
        rtype = rule.get("type", "?")
        label = rule.get("label", rule.get("name", "?"))
        common = {"label": label, "type": rtype, "before": text, "matches": 0,
                  "skipped": False, "error": None, "slow": False}

        if rtype == "terminal":
            after = text.lstrip(" \t\r").rstrip(" \t\r")
            return {**common, "after": after}

        if not rule.get("enabled", True):
            return {**common, "after": text, "skipped": True}

        # regex-list: apply each entry in order to a running buffer (entry N's
        # output feeds N+1), all inside ONE 2 s guard. Same mt.expand backref
        # idiom as the single-pattern path → parity with the engine / the streaming
        # equivalence guard. Reports one card-level step (summed matches).
        if rtype == "regex-list":
            entries = rule.get("entries", []) or []
            if not entries:
                return {**common, "after": text, "skipped": True}
            lout: dict[str, Any] = {"done": False, "after": text, "matches": 0}
            def _do_list() -> None:
                try:
                    cur = text
                    total = 0
                    bad: str | None = None
                    for entry in entries:
                        ep = entry.get("pattern", "") or ""
                        if not ep:
                            continue
                        if regex_guard._nested_repetition(str(ep)):
                            # Wins over an earlier compile error: `not_run`
                            # needs its "not executed here" caveat shown.
                            bad = _NESTED_REP_MSG
                            lout["not_run"] = True
                            continue
                        try:
                            ecre = re.compile(ep)
                        except re.error as e:
                            # Engine parity (pl_engine.rebuild_caches): a bad entry is
                            # SKIPPED, not the whole card — the valid entries still
                            # apply. Surface the first bad pattern as an advisory.
                            if bad is None:
                                bad = str(e)
                            continue
                        er = entry.get("replacement", "") or ""
                        erep = (lambda mt, _r=er: mt.expand(_r) if "\\" in _r else _r)
                        total += sum(1 for _ in ecre.finditer(cur))
                        cur = ecre.sub(erep, cur)
                    lout["after"] = cur
                    lout["matches"] = total
                    if bad is not None:
                        lout["err"] = bad
                    lout["done"] = True
                except Exception as e:  # noqa: BLE001
                    lout["err"] = str(e)
                    lout["done"] = True
            tl = threading.Thread(target=_do_list, daemon=True)
            tl.start()
            tl.join(timeout=2.0)
            if not lout["done"]:
                return {**common, "after": text, "slow": True}
            # A bad entry no longer blanks the card (the engine skips it per-entry),
            # so show the valid entries' result + the bad pattern as an advisory.
            return {**common, "after": lout["after"], "matches": lout["matches"],
                    "error": lout.get("err"),
                    **({"not_run": True} if lout.get("not_run") else {})}

        try:
            if rtype == "callback:map":
                m = rule.get("map", {}) or {}
                # Editor rules skip pl_apply._PIPELINE_RULE_ADAPTER: a list/str map
                # would survive to m.items() below and 500 the whole dry run.
                if not isinstance(m, dict):
                    return {**common, "after": text,
                            "error": "map must be an object"}
                if not m:
                    return {**common, "after": text, "skipped": True}
                # The engine's own compile (pl_engine.rebuild_caches uses it too):
                # ß/ss variants and the dictated-punctuation prefix included,
                # so the dry run shows what a transcription really gets.
                cre, replacer, _lookup = dictation_map.compile_map(m)
            else:
                pattern = rule.get("pattern", "") or ""
                if not pattern:
                    return {**common, "after": text, "skipped": True}
                if regex_guard._nested_repetition(str(pattern)):
                    return {**common, "after": text, "error": _NESTED_REP_MSG,
                            "not_run": True}
                cre = re.compile(pattern)
                if rtype == "callback:lowercase-wordlist":
                    replacer = pl_engine._make_lowercase_wordlist_replacer(
                        frozenset(w.lower() for w in (rule.get("wordlist") or [])))
                elif rtype == "callback:dedup":
                    replacer = pl_engine._dedup_callback
                elif rtype == "callback:upper":
                    replacer = pl_engine._upper_callback
                else:
                    return {**common, "after": text, "skipped": True,
                            "error": f"unknown rule type: {rtype}"}
        # TypeError/ValueError as well as re.error: the rule dicts arrive from
        # the editor without going through pl_apply._PIPELINE_RULE_ADAPTER, so a non-string
        # `pattern` (a half-typed row, or a hand-rolled request body) reaches
        # re.compile as e.g. an int and raises TypeError — which re.error does not
        # cover, turning a dry run into an unhandled 500 instead of a per-step
        # error the panel already knows how to render.
        except (re.error, AttributeError, TypeError, ValueError) as e:
            return {**common, "after": text, "error": str(e)}

        out: dict[str, Any] = {"done": False, "after": text, "matches": 0}
        def _do() -> None:
            try:
                out["matches"] = sum(1 for _ in cre.finditer(text))
                out["after"] = cre.sub(replacer, text)
                out["done"] = True
            except Exception as e:  # noqa: BLE001
                out["err"] = str(e)
                out["done"] = True
        t = threading.Thread(target=_do, daemon=True)
        t.start()
        t.join(timeout=2.0)
        if not out["done"]:
            return {**common, "after": text, "slow": True}
        if "err" in out:
            return {**common, "after": text, "error": out["err"]}
        return {**common, "after": out["after"], "matches": out["matches"]}

    text = sample
    steps: list[dict[str, Any]] = []
    saw_terminal = False
    # Whole-request budget: each timed-out rule strands an uninterruptible
    # guard thread, so once the budget is burnt the remaining rules are
    # reported `slow` WITHOUT starting more threads.
    deadline = time.monotonic() + 5.0
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        # `terminal` is a plain strip in _run_rule (no regex, no guard
        # thread), so it is exempt from the skip: reporting it `slow` and
        # suppressing the implicit final trim would misreport `final`.
        if time.monotonic() > deadline and rule.get("type") != "terminal":
            step = {"label": rule.get("label", rule.get("name", "?")),
                    "type": rule.get("type", "?"), "before": text,
                    "after": text, "matches": 0, "skipped": False,
                    "error": None, "slow": True}
            step["ordinal"] = idx + 1
            steps.append(step)
            if rule.get("type") == "terminal":
                saw_terminal = True
            continue
        # _run_rule blocks for up to 2 s on its guard thread's join — run it on
        # a worker so one pathological pattern cannot stall unrelated requests.
        step = await asyncio.to_thread(_run_rule, text, rule)
        step["ordinal"] = idx + 1
        steps.append(step)
        text = step["after"]
        if rule.get("type") == "terminal":
            saw_terminal = True
    if not saw_terminal:
        # No terminal row in the payload — apply the implicit trim.
        before = text
        text = text.lstrip(" \t\r").rstrip(" \t\r")
        if before != text:
            steps.append({
                "ordinal": len(steps) + 1,
                "label": "Trim edges (always-last)",
                "type": "terminal",
                "before": before, "after": text, "matches": 0,
                "skipped": False, "error": None, "slow": False,
            })

    return JSONResponse({"steps": steps, "final": text})


class _TranslationTestBody(BaseModel):
    """POST /settings/translation-test payload — pydantic-shaped so malformed
    bodies fail 422 like POST /settings/state."""
    text: Annotated[str, Field(min_length=1, max_length=2000)]
    target: Annotated[str, Field(min_length=2, max_length=16)]
    source: Annotated[str, Field(min_length=2, max_length=16)] | None = None
    # Overrides cfg.TRANSLATION_PROMPT_TEMPLATE for THIS call only (unsaved
    # textarea value from the WebUI's custom-template editor).
    template: Annotated[str, Field(max_length=8000)] | None = None
    # Prompt lab (all optional, unsaved WebUI values): which model to load
    # (allowlist-gated like the request path), which family to force, a
    # glossary to inject, and preview=True to RENDER the prompt without
    # loading any model.
    model: Annotated[str, Field(max_length=160)] | None = None
    family: Annotated[str, Field(max_length=32)] | None = None
    glossary: Annotated[str, Field(max_length=4000)] | None = None
    preview: bool = False
    # Opt-in live progress: a hex id the lab JS polls at
    # GET /v1/audio/transcriptions/progress/{id} while the test runs
    # (download → load → translate). Malformed → ignored, like the
    # request path's stance.
    progress_id: Annotated[str, Field(max_length=64)] | None = None


@router.post("/translation-test",
             dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def translation_test(
        body: _TranslationTestBody,
        user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    """Run ONE sample segment through the translation stage — the WebUI's
    "▶ Test with loaded model" button next to the custom-template preview.
    Uses the configured default model (loading it on first use, exactly like
    a real request); a supplied `template` is threaded through
    translate_segments(template_override=...) so the admin can test the
    UNSAVED textarea value. 403 when the stage is off; TranslationError →
    400 with its client-safe message (same contract as the request path)."""
    if not getattr(cfg, "TRANSLATION_ENABLED", False):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "translation is disabled (TRANSLATION_ENABLED)")
    fam = (body.family or "").strip().lower() or None
    if fam is not None and fam not in translation._FAMILIES:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"error": f"unknown prompt family '{fam}'"})
    # Same admission rule as the request path — tr_gating._translation_model_allowed
    # is the one home for the allowlist semantics (an empty allowlist admits
    # any well-formed ref; a non-empty one admits members + the configured
    # default).
    ref = (body.model or "").strip()
    default = (getattr(cfg, "TRANSLATION_DEFAULT_MODEL", "") or "").strip()
    if ref and not tr_gating._translation_model_allowed(ref, requested=ref):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "model is not in TRANSLATION_ALLOWED_MODELS"})
    # Custom-family template only — a stale textarea must not leak into a
    # built-in family's test. Resolve the EFFECTIVE family first: the lab
    # sends family=null for "auto", and render_prompt/translate_segments
    # would otherwise treat any non-None template as the custom family,
    # so the panel would preview/test a family a real request never uses.
    eff = fam or translation.resolve_family(ref or default)
    template = body.template if eff == "custom" else None
    prompt = translation.render_prompt(
        body.text, body.target, source=body.source, model_ref=ref or None,
        family=fam, glossary=body.glossary or "", template=template)
    if body.preview:
        # Render-only: what the model WOULD receive; nothing is loaded.
        return JSONResponse({"prompt": prompt, "warnings": []})
    cold = not prompt.get("model_loaded", False)
    # Optional progress plumbing: joins the shared _BATCH_PROGRESS registry
    # so the lab JS can poll the existing progress endpoint while a cold
    # test downloads (multi-GB) + loads + translates.
    _pid = (body.progress_id
            if (body.progress_id
                and tx_progress._PROGRESS_ID_RE.match(body.progress_id))
            else None)
    # Owner-stamped like the batch/stage-ahead seeds in main: an owner-less
    # entry is readable/cancellable by ANY authenticated caller holding the id.
    tx_progress._progress_set(_pid, stage="starting", progress=None,
                              model=(ref or default or None), compute="gguf",
                              owner=(user.get("user_id") or user.get("key_id")))
    t0 = time.perf_counter()
    try:
        results, warnings, meta = await translation.translate_segments(
            [{"text": body.text}], [body.target],
            source_lang=body.source, mode="faithful",
            model_ref=ref or None, glossary=body.glossary or "",
            template_override=template, family_override=fam,
            # target_progress is the only intra-target signal: a one-target
            # test keeps `f` at 0.0 until the end, so forward it (as main does).
            progress_cb=lambda f, step=None, last_text=None, target=None,
                target_progress=None, **_kw:
                tx_progress._progress_set(_pid, stage="translating",
                                          progress=f, step=step, target=target,
                                          target_progress=target_progress),
            download_cb=lambda done, total:
                tx_progress._progress_set(
                    _pid, stage="downloading",
                    progress=(done / total) if total else None,
                    total_bytes=total or None))
    except translation.TranslationError as e:
        # str(e) is client-safe by the module's failure contract.
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"error": str(e)})
    finally:
        # The helper also leaves the _PROGRESS_CLOSED tombstone, so a straggling
        # download_cb tick from the load thread cannot re-create the entry
        # owner-less.
        tx_progress._progress_close(_pid)
    ms = int((time.perf_counter() - t0) * 1000)
    # A guard-failed test FALLS BACK to the untranslated source text — the
    # warnings are the only signal, so they MUST reach the admin (otherwise a
    # broken template reads as a fast success).
    return JSONResponse({
        "output": results[0].get(body.target, "") if results else "",
        "ms": ms,
        "model": meta.get("model", ""),
        "warnings": warnings,
        "cold": cold,
        "prompt": prompt,
    })


@router.post("/restart", dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def post_restart(request: Request) -> JSONResponse:
    """Trigger a self-restart of the backend (cross-platform).

    Windows: spawns `WhisperAPI.exe restart!` (WinSW's documented self-restart
    command) and schedules `os._exit(0)` ~1.5 s out. Other OSes: re-execs the
    process in place (os.execv) ~1.5 s out — works bare, under systemd, or in
    a container. Either way this returns 200 first; the 1.5 s delay lets the
    response flush over loopback before the process restarts. End-to-end
    downtime is ~3-4 s for a no-preload deployment.

    See restart_service.py for the per-platform mechanics (and why Windows
    uses WinSW's explicit `restart!` rather than <onfailure>).
    """
    try:
        from faster_whisper_backend.admin.restart_service import trigger_self_restart
    except ImportError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"restart_service module unavailable: {e}")

    client_host = request.client.host if request.client else "?"
    logger.info("[config] admin restart requested from=%s", client_host)
    try:
        method = trigger_self_restart()
    except Exception as e:
        logger.error("[config] self-restart scheduling failed: %s", e)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"could not schedule self-restart: {e}",
        )
    delay_sec = 1.5
    logger.info("[config] self-restart scheduled via method=%s; "
                "process will exit in %.1f s", method, delay_sec)

    return JSONResponse({
        "status": "restarting",
        "method": method,
        "delay_sec": delay_sec,
    })


# --- HTML template ------------------------------------------------------------
# Vanilla JS, no build step. Mirrors the /logs viewer styling. Sections,
# per-rule PIPELINE_RULES editor, textarea-per-line editors for list/set
# fields, save flow with restart modal + post-restart polling.

_SETTINGS_VIEWER_HTML = templates.load(__file__, "settings.html")
