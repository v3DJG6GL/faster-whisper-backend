"""
Admin settings schema: the Pydantic models behind config.local.json.

Holds the field type aliases, the pipeline-rule models, AdminConfig (one
Optional field per editable setting, `extra="forbid"`), the field-registry
tables generated from its per-field metadata, and the generated override
models (ModelOverride / OverrideProfile). Persistence (load / save of
config.local.json and config.json) lives in settings/config_store.py.

Validation uses Pydantic v2. Every field is Optional — missing means "use the
config.py default". `model_config = {"extra": "forbid"}` rejects unknown keys
so typos and probing surface as 422 errors instead of silent no-ops.
"""

from __future__ import annotations

import functools
import ipaddress
import re
from pathlib import PurePath, PureWindowsPath
from typing import Annotated, Any, Literal

from faster_whisper_backend.core.languages import WHISPER_LANGUAGE_NAMES
from faster_whisper_backend.settings.descriptions import FIELD_DESCRIPTIONS

from pydantic import (
    BaseModel, Field, ValidationError, ValidationInfo, create_model,
    field_validator, model_validator,
)


# ENV_VAR_MAPPING / RESTART_REQUIRED_FIELDS / LOAD_TIME_FIELDS /
# CACHE_REBUILD_FIELDS / _POST_LOAD_COERCERS / FIELD_GROUPS /
# CONFIG_TO_CLIENT_KEY / LOCKABLE_FIELDS are GENERATED from the per-field
# registry metadata declared on each AdminConfig field via _F(...) — see the
# "Generated field-registry tables" block below the AdminConfig class body.


# Valid values for the per-field `scope` registry key:
#   server      — server-wide only; never overridable per-model or per-request
#   per_model   — load-time ModelOverride-only fields that are also global
#                 AdminConfig fields (MODEL_DEVICE, NUM_WORKERS, …)
#   per_request — call-time / streaming fields shared with ModelOverride and
#                 OverrideProfile (the lockable decode/streaming scalars)
_SCOPES = ("server", "per_model", "per_request")
# Derived-extras buckets a field may name via `evict=`. admin_routes._EVICTORS
# must carry a dropper for each — an unknown name would otherwise ship as a
# bucket whose eviction only ever logs a KeyError on edit.
_EVICT_BUCKETS = ("diarization", "bgm", "translation")


def _F(
    name: str,
    *,
    scope: str | None = None,
    group: str | None,
    subgroup: str | None = None,
    order: int | None = None,
    env: str | None = None,
    restart: bool = False,
    load_time: bool = False,
    cache_rebuild: bool = False,
    coerce: Any = None,
    client_key: str | None = None,
    model_override: bool = True,
    evict: str | None = None,
    **kwargs: Any,
) -> Any:
    """`Field(default=None, description=FIELD_DESCRIPTIONS[name], **kwargs)`
    plus the per-field REGISTRY METADATA (stored in json_schema_extra
    ["x_registry"]) that the generated module-level tables below the
    AdminConfig class body are derived from:

      scope          REQUIRED — one of _SCOPES (see above). ValueError at
                     import when missing/unknown, so a new field cannot ship
                     without declaring its override surface.
      group          /settings form section title. None = hidden from the
                     form (edited on a dedicated page, e.g. OVERRIDE_PROFILES).
      subgroup       Sub-section title within `group` (None = no subheader).
      order          Explicit position within the (group, subgroup) bucket for
                     the few subgroups whose display order deviates from the
                     AdminConfig declaration order. Fields without it sort by
                     declaration order.
      env            Env var pinning the field. Default WHISPER_<NAME>; pass
                     only where the historical name differs (TRACE_ENABLED →
                     WHISPER_TRACE).
      restart        Cold setting — service restart required (→ RESTART_
                     REQUIRED_FIELDS).
      load_time      Read at WhisperModel(...) construction — edit triggers
                     drain-then-evict (→ LOAD_TIME_FIELDS).
      cache_rebuild  Edit requires main.rebuild_caches() (→ CACHE_REBUILD_
                     FIELDS).
      evict          Derived-extras bucket dropped when this field is edited
                     — one of _EVICT_BUCKETS (→ EXTRAS_EVICTION, dispatched
                     via admin_routes._EVICTORS).
      coerce         Post-load JSON coercion callable, e.g. `set`
                     (→ _POST_LOAD_COERCERS).
      client_key     Lowercase per-request decode_override key the field
                     governs (→ CONFIG_TO_CLIENT_KEY).
      model_override True (default) = a per_request field is also a
                     per-model ModelOverride field; False marks the
                     streaming-only fields excluded from ModelOverride.

    Single-source-of-truth helper: every editable field passes its name to
    this and gets its description wired up automatically. Raises KeyError
    at import time if a name is missing — keeps schema and descriptions
    in lockstep.
    """
    if scope not in _SCOPES:
        raise ValueError(
            f"AdminConfig field {name!r}: scope={scope!r} is missing or "
            f"unknown — must be one of {_SCOPES}")
    if coerce is not None and coerce is not set:
        raise ValueError(
            f"AdminConfig field {name!r}: coerce={coerce!r} — only `set` "
            f"(or None) is supported")
    if evict is not None and evict not in _EVICT_BUCKETS:
        raise ValueError(
            f"AdminConfig field {name!r}: evict={evict!r} — must be one of "
            f"{_EVICT_BUCKETS}")
    # Everything in the registry must stay JSON-serializable: pydantic runs
    # to_jsonable_python over json_schema_extra whenever model_json_schema()
    # is generated, so the coercion CALLABLE is stored by name and mapped
    # back through _COERCERS_BY_NAME when _POST_LOAD_COERCERS is built.
    registry: dict[str, Any] = {
        "scope": scope,
        "group": group,
        "subgroup": subgroup,
        "order": order,
        "env": env or f"WHISPER_{name}",
        "restart": restart,
        "load_time": load_time,
        "cache_rebuild": cache_rebuild,
        "coerce": "set" if coerce is set else None,
        "client_key": client_key,
        "model_override": model_override,
        "evict": evict,
    }
    return Field(default=None, description=FIELD_DESCRIPTIONS[name],
                 json_schema_extra={"x_registry": registry}, **kwargs)

# faster-whisper short name OR HuggingFace repo id (org/name).
_MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(/[A-Za-z0-9_.\-]+)?$"
ModelId = Annotated[str, Field(min_length=1, max_length=96, pattern=_MODEL_ID_PATTERN)]
# Same shape, but "" is legal ("" = no separate model, use the request's).
OptionalModelId = Annotated[
    str, Field(max_length=96, pattern=r"^(" + _MODEL_ID_PATTERN.strip("^$") + r")?$")]

# Config-profile name — same shape as a tag (lowercase a-z0-9-, 1-32 chars,
# no leading hyphen), so profile names and the existing visibility
# tags share one normalisation contract.
ProfileName = Annotated[str, Field(min_length=1, max_length=32,
                                   pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")]

LogLevel = Literal["debug", "info", "warning", "error", "critical"]
DeviceLit = Literal["cuda", "cpu"]
# Whisper task. "translate" targets English only — Whisper has no other target.
TaskLit = Literal["transcribe", "translate"]
# Diarization: "auto" follows MODEL_DEVICE (incl. its fallback semantics).
DiarizationDeviceLit = Literal["auto", "cuda", "cpu"]
# The two supported pyannote pipelines (both HF-gated; community-1 needs
# pyannote.audio 4.x).
DiarizationModelLit = Literal[
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]
# GGUF translation model reference: "org/repo" with an optional ":quant"
# suffix selecting a quantization file inside the repo (e.g.
# "tencent/HY-MT1.5-7B-GGUF:Q4_K_M"). Empty string = unset.
_TRANSLATION_MODEL_REF_PATTERN = (
    r"^([A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9_.\-]+(:[A-Za-z0-9_.\-]+)?)?$")
TranslationModelRef = Annotated[str, Field(
    max_length=160, pattern=_TRANSLATION_MODEL_REF_PATTERN)]
# Non-empty variant for the allow/preload list entries.
TranslationModelRefItem = Annotated[str, Field(
    min_length=1, max_length=160, pattern=_TRANSLATION_MODEL_REF_PATTERN)]
# Comma-separated target language codes: "en" / "fr-CA" / "en,de,pt-BR".
_TRANSLATE_TO_PATTERN = (
    r"^([a-z]{2,3}(-[A-Za-z0-9]{2,8})?(,[a-z]{2,3}(-[A-Za-z0-9]{2,8})?)*)?$")
# Runtime compute_type — the full CTranslate2 set (verified vs ctranslate2 4.7.2
# + the CT2 docs). "auto" lets CT2 pick the fastest type supported on the device;
# "default" keeps the model's converted type. A choice unsupported on the
# hardware falls back to MODEL_COMPUTE_TYPE_FALLBACK at load.
ComputeLit = Literal[
    "auto", "default", "float32", "float16", "bfloat16",
    "int16", "int8", "int8_float32", "int8_float16", "int8_bfloat16",
]
# CONVERT_QUANTIZATION accepts CT2's full conversion set (ctranslate2.specs
# .model_spec.ACCEPTED_MODEL_TYPES) — wider than ComputeLit (which is the runtime
# compute_type). Verified against ctranslate2 4.7.2 + the CTranslate2 docs.
ConvertQuantLit = Literal[
    "float32", "float16", "bfloat16", "int16",
    "int8", "int8_float32", "int8_float16", "int8_bfloat16",
]


# =============================================================================
# Pipeline rule schema (discriminated union on `type`)
# =============================================================================
# Each rule is one row in the unified post-processing pipeline. See
# config.py:PIPELINE_RULES for the canonical seeded list.
RuleSlug = Annotated[str, Field(min_length=1, max_length=64,
                                pattern=r"^[a-z0-9-]+$")]
RuleLabel = Annotated[str, Field(min_length=1, max_length=80)]

# Tag format — Kubernetes label-style: lowercase letters/digits/hyphens,
# 1-32 chars, no leading hyphen. Tags filter which users see
# which rules on /quick-config. Re-used by api_keys_store for the
# per-user `quick_config_tags` validator so admins can't drift the two
# schemas apart.
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}\Z")   # \Z: `$` also matches before a trailing "\n"

# Reserved override-profile name meaning "apply NO profile — plain server
# defaults". A client sends this (via override_profile / the WS handshake) to
# SUPPRESS every identity-bound profile layer and fall straight through to
# per-model + global config. Underscores can't match TAG_RE, so it can never
# collide with a real profile name. MUST stay in sync with the frontend constant
# NO_OVERRIDE_PROFILE (faster-whisper-frontend src/lib/types.ts).
NO_PROFILE_SENTINEL = "__none__"


# Whisper's language codes (tests pin the table to faster-whisper's own list).
WHISPER_LANGUAGE_CODES: frozenset[str] = frozenset(WHISPER_LANGUAGE_NAMES)


def normalize_languages(raw: Any) -> list[str]:
    """Canonicalise a raw language-code list: trim, lowercase, drop empties,
    dedup, sort. Raises ValueError on codes not in Whisper's language set.
    Empty list is permitted (means 'apply to all languages')."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("languages must be a list of strings")
    seen: set[str] = set()
    out: list[str] = []
    for code in raw:
        if not isinstance(code, str):
            raise ValueError(
                f"language code must be a string, got {type(code).__name__}"
            )
        norm = code.strip().lower()
        if not norm:
            continue
        if norm not in WHISPER_LANGUAGE_CODES:
            raise ValueError(
                f"unknown language code {code!r} — must be one of Whisper's "
                f"{len(WHISPER_LANGUAGE_CODES)} supported codes (ISO 639-1)"
            )
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    out.sort()
    return out


def normalize_tags(raw: Any) -> list[str]:
    """Canonicalise a raw tag list: trim, lowercase, drop empties, dedup,
    sort. Raises ValueError on any tag that doesn't match TAG_RE. Empty
    list is permitted (semantic varies by call site)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("tags must be a list of strings")
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            raise ValueError(f"tag must be a string, got {type(t).__name__}")
        norm = t.strip().lower()
        if not norm:
            continue
        if not TAG_RE.match(norm):
            raise ValueError(
                f"invalid tag {t!r} — lowercase a-z0-9- only, max 32 chars,"
                " no leading hyphen"
            )
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    out.sort()
    return out


# Curated palette of card-tint tokens for the pipeline editor. "" = no tint.
# Semantic names (not hex) so the theme / dark-mode own the actual colour.
RULE_CARD_COLORS = ("red", "amber", "green", "teal", "blue", "purple", "pink")


class _RuleBase(BaseModel):
    """Common fields for every PipelineRule row."""
    model_config = {"extra": "forbid"}
    name: RuleSlug
    label: RuleLabel
    enabled: bool = True
    languages: list[str] = Field(default_factory=list, max_length=99)
    locked: bool = False
    seeded: bool = False
    # When True, the rule is shown on /quick-config so end-users (non-admin
    # session) can edit its body fields. Toggle is admin-only — see the
    # per-type allow-list enforcement in quick_config/routes.py.
    exposed: bool = False
    # Tag list for per-user visibility. Asymmetric semantics: an empty
    # list means "visible to every authenticated user" (zero-config
    # migration); a populated list means "visible only to users whose
    # `quick_config_tags` intersects this list". Admins always see
    # everything. See auth.Permissions.can_see_rule().
    tags: list[str] = Field(default_factory=list, max_length=32)
    # Free-text rationale for the rule — why it exists, ordering constraints,
    # tradeoffs. Lives in config.json so a rule's "why" travels with it (this
    # was inline config.py commentary before the factory defaults moved to
    # config.json). Optional; defaults to "" so older config.local.json files
    # that predate this field still validate.
    note: Annotated[str, Field(max_length=4000)] = ""
    # Optional card tint for the pipeline editor (one of RULE_CARD_COLORS, or
    # "" for none). Forgiving by design: an unknown/typo'd token normalises to
    # "" rather than raising, so this cosmetic value can never trip
    # load_overrides (which drops ALL overrides on any validation error).
    color: str = ""
    # Fingerprint of the config.json rule this LOCAL copy was last in sync
    # with (set by the pipeline editor on add / reset / promote, and for a
    # rule that equals config.json when the list is saved). A saved local
    # list replaces the factory list, so without it the editor cannot tell
    # "edited on this server" from "config.json changed in an update" — both
    # are just "differs". Editor bookkeeping only: never evaluated by the
    # pipeline, never written to config.json (see save_factory_rules), and
    # forgiving like `color` (a malformed value becomes None, not an error).
    config_rev: str | None = None

    @field_validator("config_rev", mode="before")
    @classmethod
    def _normalize_config_rev(cls, v: Any) -> "str | None":
        if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{6,32}", v):
            return v
        return None

    @field_validator("languages", mode="before")
    @classmethod
    def _normalize_languages(cls, v: Any) -> list[str]:
        return normalize_languages(v)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: Any) -> list[str]:
        return normalize_tags(v)

    @field_validator("color", mode="before")
    @classmethod
    def _normalize_color(cls, v: Any) -> str:
        s = ("" if v is None else str(v)).strip().lower()
        return s if s in RULE_CARD_COLORS else ""


class RegexListEntry(BaseModel):
    """One find→replace pair inside a regex-list rule. `pattern` is required; an
    empty `replacement` deletes the match. `label`/`note` are optional human
    annotations (a blank label falls back to showing the pattern in the editor).
    Optional fields default to "" — never None — so model_dump(exclude_none=True)
    on the export/env round-trip keeps them."""
    model_config = {"extra": "forbid"}
    pattern: Annotated[str, Field(max_length=512)]
    replacement: Annotated[str, Field(max_length=512)] = ""
    label: Annotated[str, Field(max_length=80)] = ""
    note: Annotated[str, Field(max_length=4000)] = ""


class RegexListRule(_RuleBase):
    """Ordered batch of (pattern, replacement) entries — the unified find→replace
    rule. Entries apply in list order (entry N's output feeds N+1), exactly like a
    run of standalone re.sub rules. Replaces the legacy single `regex` type (a
    one-entry list == an old `regex` rule). An empty `entries` list is a no-op."""
    type: Literal["regex-list"]
    entries: list[RegexListEntry] = Field(default_factory=list, max_length=200)


class LowercaseWordlistRule(_RuleBase):
    """Strip terminator and lowercase the next word if it's in the wordlist."""
    type: Literal["callback:lowercase-wordlist"]
    pattern: Annotated[str, Field(max_length=512)]
    wordlist: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[A-Za-zÄÖÜäöüß]+$")]],
        Field(max_length=2000),
    ]


class MapRule(_RuleBase):
    """Spoken-word → symbol lookup. Pattern auto-built from `map` keys
    (longest-first alternation, case-insensitive) at compile time."""
    type: Literal["callback:map"]
    map: dict[
        Annotated[str, Field(min_length=1, max_length=64,
                             pattern=r"^[\w \-.,!?ßẞÄÖÜäöü]{1,64}$")],
        Annotated[str, Field(max_length=64)],
    ] = Field(default_factory=dict, max_length=10_000)
    # Server-owned: epoch seconds per `map` key, set when an entry is added or
    # its value last changed (see quick_config_routes.post_state). Never written
    # by the client — drives newest-first ordering + the inline date column on
    # /quick-config. Keys not present in `map` are dropped by the validator so
    # the two stay consistent even when an admin edits the map via /settings.
    map_meta: dict[str, int] = Field(default_factory=dict, max_length=10_000)

    @field_validator("map_meta", mode="after")
    @classmethod
    def _prune_map_meta(cls, v: dict[str, int], info: Any) -> dict[str, int]:
        keys = info.data.get("map") or {}
        return {k: ts for k, ts in v.items() if k in keys}


class DedupRule(_RuleBase):
    """Pattern-only row. Callback collapses each match: prefer last
    non-comma in the run; pure-comma run collapses to a single comma."""
    type: Literal["callback:dedup"]
    pattern: Annotated[str, Field(max_length=512)]


class UpperRule(_RuleBase):
    """Pattern-only row. Callback uppercases group(2) (or the entire match
    if the pattern has fewer than 2 groups)."""
    type: Literal["callback:upper"]
    pattern: Annotated[str, Field(max_length=512)]


class TerminalRule(_RuleBase):
    """Hardcoded final lstrip(' \\t\\r') + rstrip(' \\t\\r'). Always last;
    never user-editable. At most one terminal row, and it must be the last entry."""
    type: Literal["terminal"]


PipelineRule = Annotated[
    RegexListRule | LowercaseWordlistRule | MapRule | DedupRule | UpperRule | TerminalRule,
    Field(discriminator="type"),
]


# =============================================================================
# Per-model overrides
# =============================================================================
# A ModelOverride bundle lives at MODEL_OVERRIDES[model_id]. Every field is
# Optional — absent means "inherit the global default". The runtime helper
# effective_config.cfg_for(model_id, field) walks: per-model override > global > faster-
# whisper default. Same precedence as everywhere else, just with one more
# layer interposed.
#
# Validation: models may only carry override values that pass the same
# constraints as the corresponding global field. Pipeline rule scoping uses
# PIPELINE_RULES_EXCLUDE: a flat list of rule slugs to skip for this model.
# Rule bodies are NEVER per-model — they stay in the single global PIPELINE_RULES
# list, edited in the global pipeline editor. The per-model pane only toggles
# inclusion via a checklist.

# Override-schema fields that are NOT AdminConfig fields — they exist ONLY on
# the override models (ModelOverride / the call-time mixin), never as a global
# setting. Declared here with the same Annotated types they carry in the
# override schemas so the generated tables (and, eventually, the generated
# override models) can source them alongside the AdminConfig registry:
#   REVISION                — per-model HF snapshot pin; ModelOverride-only,
#                             load-time (edit → drain-then-evict).
#   PIPELINE_RULES_EXCLUDE/ — per-bundle rule scoping lists; call-time-mixin-
#   PIPELINE_RULES_INCLUDE    only and NOT lockable (rules resolve via the
#                             rule path, not the scalar-lock path).
_VIRTUAL_OVERRIDE_FIELDS: dict[str, dict[str, Any]] = {
    "REVISION": {
        "annotation": Annotated[str, Field(min_length=1, max_length=128)] | None,
        "scope": "per_model",
        "load_time": True,
        "lockable": False,
    },
    "PIPELINE_RULES_EXCLUDE": {
        "annotation": Annotated[list[RuleSlug], Field(max_length=200)] | None,
        "scope": "per_request",
        "load_time": False,
        "lockable": False,
    },
    "PIPELINE_RULES_INCLUDE": {
        "annotation": Annotated[list[RuleSlug], Field(max_length=200)] | None,
        "scope": "per_request",
        "load_time": False,
        "lockable": False,
    },
}


# The override models — _CallTimeOverrideMixin, _StreamingOverrideMixin,
# ModelOverride, OverrideProfile — are GENERATED from the per-field registry
# metadata (scope / model_override on each AdminConfig field) plus
# _VIRTUAL_OVERRIDE_FIELDS. They are defined in the "Generated override
# schemas" block BELOW the AdminConfig class body, because generation needs
# AdminConfig.model_fields; AdminConfig.model_rebuild() afterwards resolves
# its ModelOverride / OverrideProfile forward references.


class AdminConfig(BaseModel):
    """Pydantic schema for config.local.json. Every field is Optional; absent
    means "do not override". Bounds and patterns enforce resource caps and
    cheap input hygiene at validation time. Per-field user-facing descriptions
    live in FIELD_DESCRIPTIONS above (single source of truth — change there,
    every consumer reflects it on next reload)."""

    # `protected_namespaces=()` disables Pydantic's "model_*" reserved-prefix
    # warning so we can use MODEL_DEVICE / MODEL_COMPUTE_TYPE field names.
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    # --- Models ---
    DEFAULT_MODEL: ModelId | None = _F(
        "DEFAULT_MODEL", scope="server", group="Models", order=1)
    # Sets serialize as JSON arrays; convert back on load. List type here lets
    # us validate per-element via the ModelId Annotated type.
    ALLOWED_MODELS: list[ModelId] | None = _F(
        "ALLOWED_MODELS", scope="server", group="Models", order=2,
        coerce=set)
    MAX_LOADED_MODELS: Annotated[int, Field(ge=1, le=8)] | None = _F(
        "MAX_LOADED_MODELS", scope="server", group="Models", order=4)
    MODEL_IDLE_TIMEOUT_S: Annotated[int, Field(ge=0, le=86400)] | None = _F(
        "MODEL_IDLE_TIMEOUT_S", scope="server", group="Models", order=5)
    PRELOAD_MODELS: list[ModelId] | None = _F(
        "PRELOAD_MODELS", scope="server", group="Models", order=3,
        restart=True)
    MODEL_DEVICE: DeviceLit | None = _F(
        "MODEL_DEVICE", scope="per_model", group="Models", order=6,
        load_time=True)
    MODEL_COMPUTE_TYPE: ComputeLit | None = _F(
        "MODEL_COMPUTE_TYPE", scope="per_model", group="Models", order=7,
        load_time=True)
    MODEL_DEVICE_FALLBACK: DeviceLit | None = _F(
        "MODEL_DEVICE_FALLBACK", scope="per_model", group="Models", order=8,
        load_time=True)
    MODEL_COMPUTE_TYPE_FALLBACK: ComputeLit | None = _F(
        "MODEL_COMPUTE_TYPE_FALLBACK", scope="per_model", group="Models",
        order=9, load_time=True)

    # --- Decode params (transcribe-time) ---
    DEFAULT_LANGUAGE: Annotated[str, Field(pattern=r"^([a-z]{2,3})?$")] | None = _F(
        "DEFAULT_LANGUAGE", scope="per_request", group="Decode params",
        order=1)
    DEFAULT_PROMPT: Annotated[str, Field(max_length=2048)] | None = _F(
        "DEFAULT_PROMPT", scope="per_request", group="Decode params",
        order=2)
    BEAM_SIZE: Annotated[int, Field(ge=1, le=20)] | None = _F(
        "BEAM_SIZE", scope="per_request", group="Decode params", order=5,
        client_key="beam_size")
    BEST_OF: Annotated[int, Field(ge=1, le=20)] | None = _F(
        "BEST_OF", scope="per_request", group="Decode params", order=6,
        client_key="best_of")
    VAD_FILTER: bool | None = _F(
        "VAD_FILTER", scope="per_request", group="Decode params", order=7,
        client_key="vad_filter")
    VAD_MIN_SILENCE_MS: Annotated[int, Field(ge=0, le=10000)] | None = _F(
        "VAD_MIN_SILENCE_MS", scope="per_request", group="Decode params",
        order=8, client_key="vad_min_silence_duration_ms")
    VAD_SPEECH_PAD_MS: Annotated[int, Field(ge=0, le=2000)] | None = _F(
        "VAD_SPEECH_PAD_MS", scope="per_request", group="Decode params",
        order=9, client_key="vad_speech_pad_ms")
    VAD_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F(
        "VAD_THRESHOLD", scope="per_request", group="Decode params",
        order=10, client_key="vad_threshold")
    LEADING_SILENCE_PAD_MS: Annotated[int, Field(ge=0, le=5000)] | None = _F(
        "LEADING_SILENCE_PAD_MS", scope="per_request", group="Decode params",
        order=11)
    CONDITION_ON_PREVIOUS_TEXT: bool | None = _F(
        "CONDITION_ON_PREVIOUS_TEXT", scope="per_request",
        group="Decode params", order=12,
        client_key="condition_on_previous_text")
    WORD_TIMESTAMPS_ENABLED: bool | None = _F(
        "WORD_TIMESTAMPS_ENABLED", scope="per_request",
        group="Decode params", order=13)
    NO_SPEECH_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F(
        "NO_SPEECH_THRESHOLD", scope="per_request", group="Decode params",
        order=14, client_key="no_speech_threshold")
    LOG_PROB_THRESHOLD: Annotated[float, Field(ge=-10.0, le=0.0)] | None = _F(
        "LOG_PROB_THRESHOLD", scope="per_request", group="Decode params",
        order=15, client_key="log_prob_threshold")
    COMPRESSION_RATIO_THRESHOLD: Annotated[float, Field(ge=0.0, le=10.0)] | None = _F(
        "COMPRESSION_RATIO_THRESHOLD", scope="per_request",
        group="Decode params", order=16,
        client_key="compression_ratio_threshold")

    # --- Decode params (advanced) ---
    DEFAULT_HOTWORDS: Annotated[str, Field(max_length=2048)] | None = _F(
        "DEFAULT_HOTWORDS", scope="per_request", group="Decode params",
        order=3, client_key="hotwords")
    TASK: TaskLit | None = _F(
        "TASK", scope="per_request", group="Decode params", order=4)
    TEMPERATURE: Annotated[str, Field(max_length=64)] | None = _F(
        "TEMPERATURE", scope="per_request", group="Decode params",
        subgroup="Advanced — beam & sampling", client_key="temperature")
    PATIENCE: Annotated[float, Field(ge=0.5, le=5.0)] | None = _F(
        "PATIENCE", scope="per_request", group="Decode params",
        subgroup="Advanced — beam & sampling", client_key="patience")
    LENGTH_PENALTY: Annotated[float, Field(ge=0.1, le=5.0)] | None = _F(
        "LENGTH_PENALTY", scope="per_request", group="Decode params",
        subgroup="Advanced — beam & sampling", client_key="length_penalty")
    REPETITION_PENALTY: Annotated[float, Field(ge=0.5, le=5.0)] | None = _F(
        "REPETITION_PENALTY", scope="per_request", group="Decode params",
        subgroup="Advanced — beam & sampling",
        client_key="repetition_penalty")
    NO_REPEAT_NGRAM_SIZE: Annotated[int, Field(ge=0, le=10)] | None = _F(
        "NO_REPEAT_NGRAM_SIZE", scope="per_request", group="Decode params",
        subgroup="Advanced — beam & sampling",
        client_key="no_repeat_ngram_size")
    PROMPT_RESET_ON_TEMPERATURE: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F(
        "PROMPT_RESET_ON_TEMPERATURE", scope="per_request",
        group="Decode params", subgroup="Advanced — beam & sampling")

    # --- Language detection (active when DEFAULT_LANGUAGE is empty) ---
    MULTILINGUAL: bool | None = _F(
        "MULTILINGUAL", scope="per_request", group="Decode params",
        subgroup="Advanced — language detection (active when DEFAULT_LANGUAGE empty)",
        client_key="multilingual")
    LANGUAGE_DETECTION_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F(
        "LANGUAGE_DETECTION_THRESHOLD", scope="per_request",
        group="Decode params",
        subgroup="Advanced — language detection (active when DEFAULT_LANGUAGE empty)",
        client_key="language_detection_threshold")
    LANGUAGE_DETECTION_SEGMENTS: Annotated[int, Field(ge=1, le=10)] | None = _F(
        "LANGUAGE_DETECTION_SEGMENTS", scope="per_request",
        group="Decode params",
        subgroup="Advanced — language detection (active when DEFAULT_LANGUAGE empty)",
        client_key="language_detection_segments")

    # --- Anti-hallucination & token control ---
    HALLUCINATION_SILENCE_THRESHOLD: Annotated[float, Field(ge=0.0, le=60.0)] | None = _F(
        "HALLUCINATION_SILENCE_THRESHOLD", scope="per_request",
        group="Decode params",
        subgroup="Advanced — anti-hallucination & token control",
        client_key="hallucination_silence_threshold")
    SEGMENT_MAX_WORDS_PER_S: Annotated[float, Field(ge=0.0, le=100.0)] | None = _F(
        "SEGMENT_MAX_WORDS_PER_S", scope="per_request",
        group="Decode params",
        subgroup="Advanced — anti-hallucination & token control")
    SEGMENT_MAX_WORD_BURST_PER_S: Annotated[float, Field(ge=0.0, le=100.0)] | None = _F(
        "SEGMENT_MAX_WORD_BURST_PER_S", scope="per_request",
        group="Decode params",
        subgroup="Advanced — anti-hallucination & token control")
    SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS: Annotated[int, Field(ge=0, le=20)] | None = _F(
        "SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS", scope="per_request",
        group="Decode params",
        subgroup="Advanced — anti-hallucination & token control")
    SEGMENT_REPEAT_COLLAPSE_MIN_REPEATS: Annotated[int, Field(ge=0, le=20)] | None = _F(
        "SEGMENT_REPEAT_COLLAPSE_MIN_REPEATS", scope="per_request",
        group="Decode params",
        subgroup="Advanced — anti-hallucination & token control")
    SEGMENT_HEAD_ECHO_MIN_WORDS: Annotated[int, Field(ge=0, le=32)] | None = _F(
        "SEGMENT_HEAD_ECHO_MIN_WORDS", scope="per_request",
        group="Decode params",
        subgroup="Advanced — anti-hallucination & token control")
    DECODE_SKIP_RESIDUAL_WINDOWS: bool | None = _F(
        "DECODE_SKIP_RESIDUAL_WINDOWS", scope="per_request",
        group="Decode params",
        subgroup="Advanced — anti-hallucination & token control")
    DECODE_TOKEN_CAP_PER_SECOND: Annotated[float, Field(ge=0.0, le=50.0)] | None = _F(
        "DECODE_TOKEN_CAP_PER_SECOND", scope="per_request",
        group="Decode params",
        subgroup="Advanced — anti-hallucination & token control")
    SUPPRESS_BLANK: bool | None = _F(
        "SUPPRESS_BLANK", scope="per_request", group="Decode params",
        subgroup="Advanced — anti-hallucination & token control")
    SUPPRESS_TOKENS: Annotated[str, Field(max_length=256)] | None = _F(
        "SUPPRESS_TOKENS", scope="per_request", group="Decode params",
        subgroup="Advanced — anti-hallucination & token control",
        client_key="suppress_tokens")
    SUPPRESS_CHARS: Annotated[str, Field(max_length=64)] | None = _F(
        "SUPPRESS_CHARS", scope="per_request", group="Decode params",
        subgroup="Advanced — anti-hallucination & token control",
        cache_rebuild=True,
        client_key="suppress_chars")
    PREPEND_PUNCTUATIONS: Annotated[str, Field(max_length=64)] | None = _F(
        "PREPEND_PUNCTUATIONS", scope="per_request", group="Decode params",
        subgroup="Advanced — anti-hallucination & token control",
        client_key="prepend_punctuations")
    APPEND_PUNCTUATIONS: Annotated[str, Field(max_length=64)] | None = _F(
        "APPEND_PUNCTUATIONS", scope="per_request", group="Decode params",
        subgroup="Advanced — anti-hallucination & token control",
        client_key="append_punctuations")

    # --- Output wrappers (NOT a faster-whisper param; backend-level) ---
    OUTPUT_PREFIX: Annotated[str, Field(max_length=512)] | None = _F(
        "OUTPUT_PREFIX", scope="per_request", group="Output wrappers",
        client_key="output_prefix")
    OUTPUT_SUFFIX: Annotated[str, Field(max_length=512)] | None = _F(
        "OUTPUT_SUFFIX", scope="per_request", group="Output wrappers",
        client_key="output_suffix")

    # --- Live streaming (WebSocket dictation) ---
    STREAMING_ENABLED: bool | None = _F(
        "STREAMING_ENABLED", scope="server", group="Live streaming", order=1)
    STREAMING_MAX_SESSIONS: Annotated[int, Field(ge=1, le=256)] | None = _F(
        "STREAMING_MAX_SESSIONS", scope="server", group="Live streaming",
        order=2)
    STREAMING_IDLE_TIMEOUT_S: Annotated[float, Field(ge=0.0, le=3600.0)] | None = _F(
        "STREAMING_IDLE_TIMEOUT_S", scope="per_request",
        group="Live streaming", order=4, model_override=False)
    STREAMING_WS_PING_INTERVAL_S: Annotated[float, Field(ge=0.0, le=300.0)] | None = _F(
        "STREAMING_WS_PING_INTERVAL_S", scope="server",
        group="Live streaming", order=5, restart=True)
    STREAMING_WS_PING_TIMEOUT_S: Annotated[float, Field(ge=0.0, le=300.0)] | None = _F(
        "STREAMING_WS_PING_TIMEOUT_S", scope="server",
        group="Live streaming", order=6, restart=True)
    INFERENCE_CONCURRENCY: Annotated[int, Field(ge=1, le=64)] | None = _F(
        "INFERENCE_CONCURRENCY", scope="server", group="Live streaming",
        order=3, restart=True)
    STREAMING_PARTIAL_MODEL: OptionalModelId | None = _F(
        "STREAMING_PARTIAL_MODEL", scope="server", group="Live streaming",
        subgroup="Partial decoding (live preview)")
    STREAMING_PARTIAL_BEAM: Annotated[int, Field(ge=1, le=20)] | None = _F(
        "STREAMING_PARTIAL_BEAM", scope="per_request",
        group="Live streaming", subgroup="Partial decoding (live preview)",
        model_override=False)
    STREAMING_PARTIAL_TEMPERATURE: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F(
        "STREAMING_PARTIAL_TEMPERATURE", scope="per_request",
        group="Live streaming", subgroup="Partial decoding (live preview)",
        model_override=False)
    STREAMING_PARTIAL_CONDITION_ON_PREVIOUS_TEXT: bool | None = _F(
        "STREAMING_PARTIAL_CONDITION_ON_PREVIOUS_TEXT", scope="per_request",
        group="Live streaming", subgroup="Partial decoding (live preview)",
        model_override=False)
    STREAMING_VAD_BACKEND: Literal["auto", "silero", "energy"] | None = _F(
        "STREAMING_VAD_BACKEND", scope="per_request", group="Live streaming",
        subgroup="Endpointing (VAD) & speech gates", order=1,
        model_override=False)
    STREAMING_VAD_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F(
        "STREAMING_VAD_THRESHOLD", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=2, model_override=False,
        client_key="streaming_vad_threshold")
    STREAMING_GATE_RMS_DBFS: Annotated[float, Field(ge=-90.0, le=0.0)] | None = _F(
        "STREAMING_GATE_RMS_DBFS", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=4, model_override=False)
    STREAMING_PARTIAL_INTERVAL_MS: Annotated[int, Field(ge=200, le=5000)] | None = _F(
        "STREAMING_PARTIAL_INTERVAL_MS", scope="per_request",
        group="Live streaming", subgroup="Partial decoding (live preview)",
        model_override=False)
    STREAMING_GATE_MIN_SPEECH_MS: Annotated[int, Field(ge=0, le=5000)] | None = _F(
        "STREAMING_GATE_MIN_SPEECH_MS", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=3, model_override=False)
    STREAMING_FINAL_DROP_MIN_AVG_LOGPROB: Annotated[float, Field(ge=-100.0, le=0.0)] | None = _F(
        "STREAMING_FINAL_DROP_MIN_AVG_LOGPROB", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=7, model_override=False)
    STREAMING_FINAL_DROP_TEMPERATURE: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F(
        "STREAMING_FINAL_DROP_TEMPERATURE", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=8, model_override=False)
    STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT: bool | None = _F(
        "STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=9, model_override=False)
    STREAMING_FINAL_BEST_OF: Annotated[int, Field(ge=1, le=20)] | None = _F(
        "STREAMING_FINAL_BEST_OF", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=10, model_override=False)
    STREAMING_TAIL_TRIM_PAD_MS: Annotated[int, Field(ge=0, le=5000)] | None = _F(
        "STREAMING_TAIL_TRIM_PAD_MS", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=11, model_override=False)
    STREAMING_VAD_INNER_SILENCE_MS: Annotated[int, Field(ge=0, le=5000)] | None = _F(
        "STREAMING_VAD_INNER_SILENCE_MS", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=5, model_override=False,
        client_key="streaming_vad_inner_silence_ms")
    STREAMING_VAD_OUTER_SILENCE_MS: Annotated[int, Field(ge=100, le=10000)] | None = _F(
        "STREAMING_VAD_OUTER_SILENCE_MS", scope="per_request",
        group="Live streaming", subgroup="Endpointing (VAD) & speech gates",
        order=6, model_override=False,
        client_key="streaming_vad_outer_silence_ms")
    STREAMING_HARD_BREAK_SILENCE_MS: Annotated[int, Field(ge=0, le=120000)] | None = _F(
        "STREAMING_HARD_BREAK_SILENCE_MS", scope="per_request",
        group="Live streaming", subgroup="Finalize & document breaks",
        order=2, model_override=False,
        client_key="streaming_hard_break_silence_ms")
    STREAMING_HARD_BREAK_SEPARATOR: Annotated[str, Field(max_length=8)] | None = _F(
        "STREAMING_HARD_BREAK_SEPARATOR", scope="per_request",
        group="Live streaming", subgroup="Finalize & document breaks",
        order=3, model_override=False,
        client_key="streaming_hard_break_separator")
    STREAMING_FORCED_COMMIT_S: Annotated[float, Field(ge=5.0, le=29.0)] | None = _F(
        "STREAMING_FORCED_COMMIT_S", scope="per_request",
        group="Live streaming", subgroup="Finalize & document breaks",
        order=1, model_override=False)
    STREAMING_BUFFER_TRIM_S: Annotated[float, Field(ge=5.0, le=29.0)] | None = _F(
        "STREAMING_BUFFER_TRIM_S", scope="per_request",
        group="Live streaming", subgroup="Buffer management",
        model_override=False)
    STREAMING_BUFFER_TRIM_KEEP_S: Annotated[float, Field(ge=2.0, le=29.0)] | None = _F(
        "STREAMING_BUFFER_TRIM_KEEP_S", scope="per_request",
        group="Live streaming", subgroup="Buffer management",
        model_override=False)
    # Lower bound sits above the FORCED_COMMIT_S ceiling (29) so this can never
    # be tuned down into the range where it would fire during real dictation.
    STREAMING_MAX_BUFFER_S: Annotated[float, Field(ge=60.0, le=3600.0)] | None = _F(
        "STREAMING_MAX_BUFFER_S", scope="server", group="Live streaming",
        subgroup="Buffer management")
    STREAMING_PROMPT_WORDS: Annotated[int, Field(ge=0, le=400)] | None = _F(
        "STREAMING_PROMPT_WORDS", scope="per_request",
        group="Live streaming", subgroup="Finalize & document breaks",
        order=4, model_override=False)

    # --- Load-time, hardware (advanced) ---
    DOWNLOAD_ROOT: Annotated[str, Field(max_length=512)] | None = _F(
        "DOWNLOAD_ROOT", scope="server", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)
    LOCAL_FILES_ONLY: bool | None = _F(
        "LOCAL_FILES_ONLY", scope="server", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)
    HF_TOKEN: Annotated[str, Field(max_length=256)] | None = _F(
        "HF_TOKEN", scope="server", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)
    AUTO_CONVERT_HF_MODELS: bool | None = _F(
        "AUTO_CONVERT_HF_MODELS", scope="server", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)
    CONVERT_QUANTIZATION: ConvertQuantLit | None = _F(
        "CONVERT_QUANTIZATION", scope="server", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)
    CONVERTED_MODELS_DIR: Annotated[str, Field(max_length=512)] | None = _F(
        "CONVERTED_MODELS_DIR", scope="server", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)
    CPU_THREADS: Annotated[int, Field(ge=0, le=128)] | None = _F(
        "CPU_THREADS", scope="server", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)
    NUM_WORKERS: Annotated[int, Field(ge=1, le=8)] | None = _F(
        "NUM_WORKERS", scope="per_model", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)
    DEVICE_INDEX: Annotated[int, Field(ge=0, le=15)] | None = _F(
        "DEVICE_INDEX", scope="per_model", group="Models",
        subgroup="Advanced — load-time hardware", load_time=True)

    # --- Preload & warm cache (advanced) ---
    # All HOT: preload.py re-reads every one of these per admission decision,
    # so an operator raising a reserve or switching the feature off applies to
    # the next plan with no restart. None is load_time — a warm lease changes
    # only eviction eligibility, never how a model is constructed.
    MODEL_PRELOAD_ENABLED: bool | None = _F(
        "MODEL_PRELOAD_ENABLED", scope="server", group="Models",
        subgroup="Advanced — preload & warm cache")
    MODEL_PRELOAD_WARM_TTL_S: Annotated[int, Field(ge=30, le=3600)] | None = _F(
        "MODEL_PRELOAD_WARM_TTL_S", scope="server", group="Models",
        subgroup="Advanced — preload & warm cache")
    MODEL_PRELOAD_VRAM_RESERVE_MB: Annotated[
        int, Field(ge=0, le=32768)
    ] | None = _F(
        "MODEL_PRELOAD_VRAM_RESERVE_MB", scope="server", group="Models",
        subgroup="Advanced — preload & warm cache")
    MODEL_PRELOAD_RAM_RESERVE_MB: Annotated[
        int, Field(ge=0, le=131072)
    ] | None = _F(
        "MODEL_PRELOAD_RAM_RESERVE_MB", scope="server", group="Models",
        subgroup="Advanced — preload & warm cache")
    MODEL_PRELOAD_EVICT_IDLE_MODELS: bool | None = _F(
        "MODEL_PRELOAD_EVICT_IDLE_MODELS", scope="server", group="Models",
        subgroup="Advanced — preload & warm cache")

    # --- Speaker diarization ---
    DIARIZATION_ENABLED: bool | None = _F(
        "DIARIZATION_ENABLED", scope="server", group="Diarization")
    # Per-request since the stage-model-allowlist work: a caller may pick the
    # pipeline (subject to DIARIZATION_ALLOWED_MODELS); the evict metadata
    # stays so a global edit still drops the cached pipeline.
    DIARIZATION_MODEL: DiarizationModelLit | None = _F(
        "DIARIZATION_MODEL", scope="per_request", group="Diarization",
        evict="diarization")
    DIARIZATION_ALLOWED_MODELS: list[DiarizationModelLit] | None = _F(
        "DIARIZATION_ALLOWED_MODELS", scope="server", group="Diarization",
        evict="diarization")
    DIARIZATION_PRELOAD: bool | None = _F(
        "DIARIZATION_PRELOAD", scope="server", group="Diarization",
        restart=True)
    DIARIZATION_DEVICE: DiarizationDeviceLit | None = _F(
        "DIARIZATION_DEVICE", scope="server", group="Diarization",
        evict="diarization")
    DIARIZATION_IDLE_TIMEOUT_S: Annotated[int, Field(ge=0, le=86400)] | None = _F(
        "DIARIZATION_IDLE_TIMEOUT_S", scope="server", group="Diarization")
    DIARIZATION_EMBEDDING_BATCH_SIZE: Annotated[int, Field(ge=1, le=64)] | None = _F(
        "DIARIZATION_EMBEDDING_BATCH_SIZE", scope="server",
        group="Diarization", subgroup="Advanced — speaker bounds & VRAM",
        order=4, evict="diarization")
    DIARIZE: bool | None = _F(
        "DIARIZE", scope="per_request", group="Diarization")
    DIARIZATION_NUM_SPEAKERS: Annotated[int, Field(ge=1, le=32)] | None = _F(
        "DIARIZATION_NUM_SPEAKERS", scope="per_request", group="Diarization",
        subgroup="Advanced — speaker bounds & VRAM", order=1)
    DIARIZATION_MIN_SPEAKERS: Annotated[int, Field(ge=1, le=32)] | None = _F(
        "DIARIZATION_MIN_SPEAKERS", scope="per_request", group="Diarization",
        subgroup="Advanced — speaker bounds & VRAM", order=2)
    DIARIZATION_MAX_SPEAKERS: Annotated[int, Field(ge=1, le=32)] | None = _F(
        "DIARIZATION_MAX_SPEAKERS", scope="per_request", group="Diarization",
        subgroup="Advanced — speaker bounds & VRAM", order=3)

    # --- Background-music separation ---
    BGM_SEPARATION_ENABLED: bool | None = _F(
        "BGM_SEPARATION_ENABLED", scope="server", group="Music separation")
    # Per-request since the stage-model-allowlist work (see DIARIZATION_MODEL).
    BGM_SEPARATION_UVR_MODEL: Annotated[str, Field(min_length=1, max_length=128)] | None = _F(
        "BGM_SEPARATION_UVR_MODEL", scope="per_request",
        group="Music separation", evict="bgm")
    BGM_SEPARATION_ALLOWED_MODELS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=128)]],
        Field(max_length=64),
    ] | None = _F(
        "BGM_SEPARATION_ALLOWED_MODELS", scope="server",
        group="Music separation", evict="bgm")
    BGM_SEPARATION_PRELOAD: bool | None = _F(
        "BGM_SEPARATION_PRELOAD", scope="server", group="Music separation",
        restart=True)
    BGM_SEPARATION_DEVICE: DiarizationDeviceLit | None = _F(
        "BGM_SEPARATION_DEVICE", scope="server", group="Music separation",
        evict="bgm")
    BGM_SEPARATION_IDLE_TIMEOUT_S: Annotated[int, Field(ge=0, le=86400)] | None = _F(
        "BGM_SEPARATION_IDLE_TIMEOUT_S", scope="server",
        group="Music separation")
    SEPARATE_BGM: bool | None = _F(
        "SEPARATE_BGM", scope="per_request", group="Music separation")

    # --- Translation (T2T) ---
    TRANSLATION_ENABLED: bool | None = _F(
        "TRANSLATION_ENABLED", scope="server", group="Translation")
    # evict="translation": editing the default/allowlist can orphan a loaded
    # model the request path could no longer reach; editing device/cap — or
    # the prompt family, which sets the load-time n_ctx — changes the load
    # parameters. Drop the LRU on save in all these cases.
    TRANSLATION_DEFAULT_MODEL: TranslationModelRef | None = _F(
        "TRANSLATION_DEFAULT_MODEL", scope="server", group="Translation",
        evict="translation")
    # Sets serialize as JSON arrays; convert back on load (mirrors
    # ALLOWED_MODELS — list type here for per-element validation).
    TRANSLATION_ALLOWED_MODELS: list[TranslationModelRefItem] | None = _F(
        "TRANSLATION_ALLOWED_MODELS", scope="server", group="Translation",
        coerce=set, evict="translation")
    TRANSLATION_PRELOAD_MODELS: list[TranslationModelRefItem] | None = _F(
        "TRANSLATION_PRELOAD_MODELS", scope="server", group="Translation",
        restart=True)
    TRANSLATION_MAX_LOADED_MODELS: Annotated[int, Field(ge=1, le=4)] | None = _F(
        "TRANSLATION_MAX_LOADED_MODELS", scope="server", group="Translation",
        evict="translation")
    TRANSLATION_DEVICE: DiarizationDeviceLit | None = _F(
        "TRANSLATION_DEVICE", scope="server", group="Translation",
        evict="translation")
    TRANSLATION_IDLE_TIMEOUT_S: Annotated[int, Field(ge=0, le=86400)] | None = _F(
        "TRANSLATION_IDLE_TIMEOUT_S", scope="server", group="Translation")
    TRANSLATION_BATCH_SEGMENTS: Annotated[int, Field(ge=1, le=50)] | None = _F(
        "TRANSLATION_BATCH_SEGMENTS", scope="server", group="Translation")
    TRANSLATION_PROMPT_FAMILY: Literal[
        "auto", "hunyuan", "gemma-translate", "milmmt", "seedx", "chatml",
        "custom",
    ] | None = _F(
        "TRANSLATION_PROMPT_FAMILY", scope="server", group="Translation",
        evict="translation")
    TRANSLATION_PROMPT_TEMPLATE: Annotated[str, Field(max_length=8000)] | None = _F(
        "TRANSLATION_PROMPT_TEMPLATE", scope="server", group="Translation")
    TRANSLATION_LANGUAGES: Annotated[
        str, Field(max_length=2000, pattern=_TRANSLATE_TO_PATTERN)
    ] | None = _F(
        "TRANSLATION_LANGUAGES", scope="server", group="Translation")
    # Call-time defaults (per-identity > per-model > global; lockable).
    TRANSLATE_TO: Annotated[
        str, Field(max_length=64, pattern=_TRANSLATE_TO_PATTERN)
    ] | None = _F(
        "TRANSLATE_TO", scope="per_request", group="Translation",
        subgroup="Per-request defaults")
    TRANSLATION_MODEL: TranslationModelRef | None = _F(
        "TRANSLATION_MODEL", scope="per_request", group="Translation",
        subgroup="Per-request defaults")
    TRANSLATION_CONTEXT_SEGMENTS: Annotated[int, Field(ge=0, le=10)] | None = _F(
        "TRANSLATION_CONTEXT_SEGMENTS", scope="per_request",
        group="Translation", subgroup="Per-request defaults")
    TRANSLATION_MAX_TARGETS: Annotated[int, Field(ge=1, le=10)] | None = _F(
        "TRANSLATION_MAX_TARGETS", scope="per_request", group="Translation",
        subgroup="Per-request defaults")
    TRANSLATION_MODE: Literal["fluent", "faithful"] | None = _F(
        "TRANSLATION_MODE", scope="per_request", group="Translation",
        subgroup="Per-request defaults")
    TRANSLATION_GLOSSARY: Annotated[str, Field(max_length=4000)] | None = _F(
        "TRANSLATION_GLOSSARY", scope="per_request", group="Translation",
        subgroup="Per-request defaults")

    # --- Transcribe-from-URL (yt-dlp) ---
    URL_DOWNLOAD_ENABLED: bool | None = _F(
        "URL_DOWNLOAD_ENABLED", scope="server", group="Transcribe from URL")
    URL_ALLOWED_EXTRACTORS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=128),
    ] | None = _F(
        "URL_ALLOWED_EXTRACTORS", scope="server",
        group="Transcribe from URL")
    URL_ALLOW_DIRECT_MEDIA: bool | None = _F(
        "URL_ALLOW_DIRECT_MEDIA", scope="server",
        group="Transcribe from URL")
    URL_ALLOW_GENERIC: bool | None = _F(
        "URL_ALLOW_GENERIC", scope="server", group="Transcribe from URL")
    URL_MAX_DURATION_S: Annotated[int, Field(ge=1, le=86400 * 7)] | None = _F(
        "URL_MAX_DURATION_S", scope="server", group="Transcribe from URL")
    URL_VIDEO_ENABLED: bool | None = _F(
        "URL_VIDEO_ENABLED", scope="server", group="Transcribe from URL")
    URL_SUBTITLES_ENABLED: bool | None = _F(
        "URL_SUBTITLES_ENABLED", scope="server", group="Transcribe from URL")
    URL_LANGUAGE_CHECK_ENABLED: bool | None = _F(
        "URL_LANGUAGE_CHECK_ENABLED", scope="server",
        group="Transcribe from URL")
    URL_DOWNLOAD_TIMEOUT_S: Annotated[int, Field(ge=10, le=86400)] | None = _F(
        "URL_DOWNLOAD_TIMEOUT_S", scope="server",
        group="Transcribe from URL",
        subgroup="Advanced — timeouts, concurrency & retention")
    URL_VIDEO_DOWNLOAD_TIMEOUT_S: Annotated[int, Field(ge=10, le=86400)] | None = _F(
        "URL_VIDEO_DOWNLOAD_TIMEOUT_S", scope="server",
        group="Transcribe from URL",
        subgroup="Advanced — timeouts, concurrency & retention")
    URL_PREVIEW_TIMEOUT_S: Annotated[int, Field(ge=1, le=300)] | None = _F(
        "URL_PREVIEW_TIMEOUT_S", scope="server",
        group="Transcribe from URL",
        subgroup="Advanced — timeouts, concurrency & retention")
    URL_SOCKET_TIMEOUT_S: Annotated[int, Field(ge=1, le=600)] | None = _F(
        "URL_SOCKET_TIMEOUT_S", scope="server",
        group="Transcribe from URL",
        subgroup="Advanced — timeouts, concurrency & retention")
    URL_DOWNLOAD_CONCURRENCY: Annotated[int, Field(ge=1, le=16)] | None = _F(
        "URL_DOWNLOAD_CONCURRENCY", scope="server",
        group="Transcribe from URL",
        subgroup="Advanced — timeouts, concurrency & retention",
        restart=True)
    URL_MEDIA_DIR: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "URL_MEDIA_DIR", scope="server", group="Transcribe from URL",
        subgroup="Advanced — timeouts, concurrency & retention",
        restart=True)
    URL_MEDIA_TTL_S: Annotated[int, Field(ge=10, le=86400 * 7)] | None = _F(
        "URL_MEDIA_TTL_S", scope="server", group="Transcribe from URL",
        subgroup="Advanced — timeouts, concurrency & retention")
    RETAINED_MEDIA_MAX_BYTES: Annotated[int, Field(ge=0, le=500_000_000_000)] | None = _F(
        "RETAINED_MEDIA_MAX_BYTES", scope="server", group="Transcribe from URL",
        subgroup="Advanced — timeouts, concurrency & retention")

    # --- Media export (subtitle packaging) ---
    MEDIA_PACKAGE_ENABLED: bool | None = _F(
        "MEDIA_PACKAGE_ENABLED", scope="server", group="Media export")
    MEDIA_PACKAGE_TIMEOUT_S: Annotated[int, Field(ge=30, le=7200)] | None = _F(
        "MEDIA_PACKAGE_TIMEOUT_S", scope="server", group="Media export")

    # --- Per-model overrides ---
    MODEL_OVERRIDES: dict[ModelId, ModelOverride] | None = _F(
        "MODEL_OVERRIDES", scope="server", group="Per-model overrides")

    # --- Per-identity config profiles (reusable, name → override bundle) ---
    OVERRIDE_PROFILES: dict[ProfileName, OverrideProfile] | None = _F(
        "OVERRIDE_PROFILES", scope="server", group=None)
    ALLOW_REQUEST_OVERRIDE_PROFILE: bool | None = _F(
        "ALLOW_REQUEST_OVERRIDE_PROFILE", scope="server",
        group="Access & sessions", order=5)
    ALLOW_REQUEST_DECODE_OVERRIDES: bool | None = _F(
        "ALLOW_REQUEST_DECODE_OVERRIDES", scope="server",
        group="Access & sessions", order=6)
    STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS: bool | None = _F(
        "STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS", scope="server",
        group="Access & sessions", order=7)

    # --- Pipeline ---
    PIPELINE_RULES: Annotated[list[PipelineRule], Field(max_length=200)] | None = _F(
        "PIPELINE_RULES", scope="server", group="Pipeline",
        cache_rebuild=True)
    TRACE_ENABLED: bool | None = _F(
        "TRACE_ENABLED", scope="server", group="Logging", order=6,
        env="WHISPER_TRACE")

    # --- Logging ---
    LOG_FILE: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "LOG_FILE", scope="server", group="Logging", order=1, restart=True)
    LOG_MAX_BYTES: Annotated[int, Field(ge=1024 * 1024, le=1024 * 1024 * 1024)] | None = _F(
        "LOG_MAX_BYTES", scope="server", group="Logging", order=2,
        restart=True)
    LOG_BACKUP_COUNT: Annotated[int, Field(ge=1, le=100)] | None = _F(
        "LOG_BACKUP_COUNT", scope="server", group="Logging", order=3,
        restart=True)
    LOG_VIEWER_INITIAL_LINES: Annotated[int, Field(ge=10, le=100_000)] | None = _F(
        "LOG_VIEWER_INITIAL_LINES", scope="server", group="Logging", order=4)
    LOG_VIEWER_DOM_MAX: Annotated[int, Field(ge=0, le=1_000_000)] | None = _F(
        "LOG_VIEWER_DOM_MAX", scope="server", group="Logging", order=5)
    LOG_SEGMENT_ROWS_MAX: Annotated[int, Field(ge=0, le=100_000)] | None = _F(
        "LOG_SEGMENT_ROWS_MAX", scope="server", group="Logging", order=7)
    LOG_SEGMENT_ROWS_SHOWN: Annotated[int, Field(ge=1, le=1000)] | None = _F(
        "LOG_SEGMENT_ROWS_SHOWN", scope="server", group="Logging", order=8)
    LOG_RECEIPT_HOLD_S: Annotated[int, Field(ge=5, le=3600)] | None = _F(
        "LOG_RECEIPT_HOLD_S", scope="server", group="Logging", order=9)
    LOG_STAGE_COLORS: bool | None = _F(
        "LOG_STAGE_COLORS", scope="server", group="Logging", order=10)
    CONSOLE_LOG_LEVEL: LogLevel | None = _F(
        "CONSOLE_LOG_LEVEL", scope="server", group="Logging", order=11)

    # --- Server ---
    SERVER_HOST: Annotated[str, Field(min_length=1, max_length=64)] | None = _F(
        "SERVER_HOST", scope="server", group="Server", restart=True)
    SERVER_PORT: Annotated[int, Field(ge=1, le=65535)] | None = _F(
        "SERVER_PORT", scope="server", group="Server", restart=True)
    SERVER_WORKERS: Annotated[int, Field(ge=1, le=8)] | None = _F(
        "SERVER_WORKERS", scope="server", group="Server", restart=True)
    SERVER_LOG_LEVEL: LogLevel | None = _F(
        "SERVER_LOG_LEVEL", scope="server", group="Server", restart=True)
    MEDIA_MAX_BYTES: Annotated[int, Field(ge=1024, le=50_000_000_000)] | None = _F(
        "MEDIA_MAX_BYTES", scope="server", group="Server")
    MAX_REQUEST_BYTES: Annotated[int, Field(ge=1024, le=50_000_000_000)] | None = _F(
        "MAX_REQUEST_BYTES", scope="server", group="Server")

    # --- WebUI access control (host allowlists, bucketed by privilege tier) ---
    # Each entry must be parseable by ipaddress.ip_network(strict=False) — bare
    # IPs (v4 or v6) and CIDRs are both accepted. See _validate_hosts below.
    ADMIN_WEBUI_ALLOWED_HOSTS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=64),
    ] | None = _F(
        "ADMIN_WEBUI_ALLOWED_HOSTS", scope="server",
        group="Access & sessions", order=1)
    USER_WEBUI_ALLOWED_HOSTS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=64),
    ] | None = _F(
        "USER_WEBUI_ALLOWED_HOSTS", scope="server",
        group="Access & sessions", order=2)
    # CORS allowlist — each entry is a browser origin (scheme://host[:port]) or
    # "*". Validated by _validate_cors_origins below.
    CORS_ALLOW_ORIGINS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=256)]],
        Field(max_length=64),
    ] | None = _F(
        "CORS_ALLOW_ORIGINS", scope="server", group="Access & sessions",
        order=3, restart=True)
    # Extra origins the unsafe-method same-origin check accepts. Same entry
    # shape as CORS_ALLOW_ORIGINS but "*" is rejected — see
    # _validate_trusted_origins below.
    TRUSTED_ORIGINS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=256)]],
        Field(max_length=64),
    ] | None = _F(
        "TRUSTED_ORIGINS", scope="server", group="Access & sessions",
        order=4, restart=True)
    API_KEYS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "API_KEYS_DB", scope="server", group="Access & sessions",
        order=8, restart=True)
    # --- Browser sessions ---
    SESSIONS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "SESSIONS_DB", scope="server", group="Access & sessions",
        subgroup="Browser sessions (cookie auth)", restart=True)
    SESSION_COOKIE_SECURE: bool | None = _F(
        "SESSION_COOKIE_SECURE", scope="server", group="Access & sessions",
        subgroup="Browser sessions (cookie auth)")
    SESSION_TTL_S: Annotated[
        int, Field(ge=300, le=31_536_000)
    ] | None = _F(
        "SESSION_TTL_S", scope="server", group="Access & sessions",
        subgroup="Browser sessions (cookie auth)")
    SESSION_COOKIE_NAME: Annotated[
        str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    ] | None = _F(
        "SESSION_COOKIE_NAME", scope="server", group="Access & sessions",
        subgroup="Browser sessions (cookie auth)")
    SESSION_CSRF_COOKIE_NAME: Annotated[
        str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    ] | None = _F(
        "SESSION_CSRF_COOKIE_NAME", scope="server",
        group="Access & sessions", subgroup="Browser sessions (cookie auth)")
    # --- Concurrency & request limits ---
    # All hot: the limiters re-read cfg on every call, so an edit takes effect
    # on the next request with no restart and no bucket reset. Every one of
    # them treats 0 as "unlimited" — the escape hatch for a single-user box.
    # Budgets are per PROCESS: SERVER_WORKERS > 1 splits each one N ways.
    TRANSLATE_MAX_INFLIGHT_PER_USER: Annotated[
        int, Field(ge=0, le=64)
    ] | None = _F(
        "TRANSLATE_MAX_INFLIGHT_PER_USER", scope="server",
        group="Concurrency & Request Limits", order=1)
    TRANSLATE_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "TRANSLATE_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=2)
    STREAMING_MAX_SESSIONS_PER_USER: Annotated[
        int, Field(ge=0, le=256)
    ] | None = _F(
        "STREAMING_MAX_SESSIONS_PER_USER", scope="server",
        group="Concurrency & Request Limits", order=3)
    URL_PREVIEW_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "URL_PREVIEW_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=4)
    URL_VIDEO_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "URL_VIDEO_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=5)
    URL_SUBTITLES_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "URL_SUBTITLES_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=5)
    URL_LANGUAGE_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "URL_LANGUAGE_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=5)
    MEDIA_UPLOAD_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "MEDIA_UPLOAD_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=5)
    MEDIA_PACKAGE_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "MEDIA_PACKAGE_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=5)
    MEDIA_PACKAGE_MAX_INFLIGHT_PER_USER: Annotated[
        int, Field(ge=0, le=64)
    ] | None = _F(
        "MEDIA_PACKAGE_MAX_INFLIGHT_PER_USER", scope="server",
        group="Concurrency & Request Limits", order=5)
    JOBS_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "JOBS_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=5)
    CAPTURES_AUDIO_RATE_PER_MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "CAPTURES_AUDIO_RATE_PER_MIN", scope="server",
        group="Concurrency & Request Limits", order=6)
    REPORTS_SUBMIT_RATE_PER_10MIN: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "REPORTS_SUBMIT_RATE_PER_10MIN", scope="server",
        group="Concurrency & Request Limits", order=7)
    LOGIN_FAILURE_RATE: Annotated[
        int, Field(ge=0, le=100_000)
    ] | None = _F(
        "LOGIN_FAILURE_RATE", scope="server",
        group="Concurrency & Request Limits", order=8)

    # --- Reports store ---
    REPORTS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "REPORTS_DB", scope="server", group="Reports", restart=True)
    REPORTS_MAX: Annotated[int, Field(ge=10, le=100_000)] | None = _F(
        "REPORTS_MAX", scope="server", group="Reports")
    REPORTS_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F(
        "REPORTS_RETENTION_DAYS", scope="server", group="Reports")
    REPORTS_ALLOW_USER_SUBMIT: bool | None = _F(
        "REPORTS_ALLOW_USER_SUBMIT", scope="server", group="Reports")

    # --- Recent transcriptions store (persistent /quick-config + /stats) ---
    # MAX/TTL/PRUNE_EVERY accept 0 to mean "disabled"; combined bound is
    # "tighter of MAX and TTL wins."
    RECENT_TRANSCRIPTIONS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "RECENT_TRANSCRIPTIONS_DB", scope="server",
        group="Recent transcriptions", restart=True)
    RECENT_TRANSCRIPTIONS_MAX: Annotated[int, Field(ge=0, le=100_000)] | None = _F(
        "RECENT_TRANSCRIPTIONS_MAX", scope="server",
        group="Recent transcriptions")
    RECENT_TRANSCRIPTIONS_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F(
        "RECENT_TRANSCRIPTIONS_RETENTION_DAYS", scope="server",
        group="Recent transcriptions")
    RECENT_TRANSCRIPTIONS_PAGE_SIZE: Annotated[int, Field(ge=10, le=1000)] | None = _F(
        "RECENT_TRANSCRIPTIONS_PAGE_SIZE", scope="server",
        group="Recent transcriptions")
    QUICK_CONFIG_MAP_COLLAPSE_AFTER: Annotated[int, Field(ge=0, le=100_000)] | None = _F(
        "QUICK_CONFIG_MAP_COLLAPSE_AFTER", scope="server",
        group="Recent transcriptions")
    QUICK_CONFIG_WORD_SUGGESTIONS_MAX: Annotated[int, Field(ge=0, le=10_000)] | None = _F(
        "QUICK_CONFIG_WORD_SUGGESTIONS_MAX", scope="server",
        group="Recent transcriptions")
    RECENT_TRANSCRIPTIONS_PRUNE_EVERY: Annotated[int, Field(ge=0, le=10_000)] | None = _F(
        "RECENT_TRANSCRIPTIONS_PRUNE_EVERY", scope="server",
        group="Recent transcriptions")
    STATS_RECENT_TRANSCRIPTIONS_COUNT: Annotated[int, Field(ge=1, le=100)] | None = _F(
        "STATS_RECENT_TRANSCRIPTIONS_COUNT", scope="server",
        group="Recent transcriptions")

    # --- Server jobs (durable job resource, core/jobs_store.py) ---
    JOBS_ENABLED: bool | None = _F(
        "JOBS_ENABLED", scope="server", group="Jobs")
    JOBS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "JOBS_DB", scope="server", group="Jobs", restart=True)
    JOBS_TTL_S: Annotated[int, Field(ge=600, le=2_592_000)] | None = _F(
        "JOBS_TTL_S", scope="server", group="Jobs")
    JOBS_MAX_ROWS: Annotated[int, Field(ge=10, le=100_000)] | None = _F(
        "JOBS_MAX_ROWS", scope="server", group="Jobs")
    JOBS_MAX_BYTES: Annotated[int, Field(ge=0, le=500_000_000_000)] | None = _F(
        "JOBS_MAX_BYTES", scope="server", group="Jobs")
    STATS_SYSTEM_METRICS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "STATS_SYSTEM_METRICS_DB", scope="server",
        group="System metrics", restart=True)
    STATS_SYSTEM_METRICS_SAMPLE_S: Annotated[int, Field(ge=1, le=3600)] | None = _F(
        "STATS_SYSTEM_METRICS_SAMPLE_S", scope="server",
        group="System metrics")
    STATS_SYSTEM_METRICS_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F(
        "STATS_SYSTEM_METRICS_RETENTION_DAYS", scope="server",
        group="System metrics")

    # --- Usage statistics (the desktop app's /v1/usage + admin /stats) ---
    USAGE_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "USAGE_DB", scope="server", group="Usage statistics", restart=True)
    USAGE_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F(
        "USAGE_RETENTION_DAYS", scope="server", group="Usage statistics")
    USAGE_JOBS_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F(
        "USAGE_JOBS_RETENTION_DAYS", scope="server", group="Usage statistics")
    USAGE_APP_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F(
        "USAGE_APP_RETENTION_DAYS", scope="server", group="Usage statistics")
    USAGE_UNREPORTED_AFTER_H: Annotated[int, Field(ge=1, le=720)] | None = _F(
        "USAGE_UNREPORTED_AFTER_H", scope="server", group="Usage statistics")

    # --- Client settings sync (the desktop app's /v1/synced-client-settings) ---
    CLIENT_SETTINGS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "CLIENT_SETTINGS_DB", scope="server", group="Client settings sync",
        restart=True)

    # --- Captures (fine-tuning data store) ---
    CAPTURES_RECORDING_ENABLED: bool | None = _F(
        "CAPTURES_RECORDING_ENABLED", scope="server", group="Captures",
        order=1)
    CAPTURES_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "CAPTURES_DB", scope="server", group="Captures", subgroup="Storage",
        restart=True)
    CAPTURES_DIR: Annotated[str, Field(min_length=1, max_length=512)] | None = _F(
        "CAPTURES_DIR", scope="server", group="Captures", subgroup="Storage")
    CAPTURES_MAX: Annotated[int, Field(ge=10, le=1_000_000)] | None = _F(
        "CAPTURES_MAX", scope="server", group="Captures", subgroup="Storage")
    CAPTURES_MAX_MB: Annotated[int, Field(ge=1, le=10_000_000)] | None = _F(
        "CAPTURES_MAX_MB", scope="server", group="Captures",
        subgroup="Storage")
    CAPTURES_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F(
        "CAPTURES_RETENTION_DAYS", scope="server", group="Captures", order=3)
    CAPTURES_RECORDING_SAMPLE_RATE: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F(
        "CAPTURES_RECORDING_SAMPLE_RATE", scope="server", group="Captures",
        order=2)
    CAPTURES_RECORDING_MIN_DURATION_S: Annotated[float, Field(ge=0.0, le=3600.0)] | None = _F(
        "CAPTURES_RECORDING_MIN_DURATION_S", scope="server",
        group="Captures", subgroup="Duration & size guards")
    CAPTURES_RECORDING_MAX_DURATION_S: Annotated[float, Field(ge=0.1, le=86400.0)] | None = _F(
        "CAPTURES_RECORDING_MAX_DURATION_S", scope="server",
        group="Captures", subgroup="Duration & size guards")
    CAPTURES_RECORDING_AUDIO_BYTES_HARD_LIMIT: Annotated[int, Field(ge=1024, le=10_000_000_000)] | None = _F(
        "CAPTURES_RECORDING_AUDIO_BYTES_HARD_LIMIT", scope="server",
        group="Captures", subgroup="Duration & size guards")
    # Captures-specific pipeline-rule exclusion (set of rule slugs).
    # Stored as a list in JSON; coerced back to set at use time. The
    # admin UI surfaces this as the same rule-checklist widget used for
    # per-model PIPELINE_RULES_EXCLUDE so the editing affordance is
    # identical.
    CAPTURES_PIPELINE_RULES_EXCLUDE: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=64),
    ] | None = _F(
        "CAPTURES_PIPELINE_RULES_EXCLUDE", scope="server", group="Captures",
        subgroup="Training-form pipeline", coerce=set)
    CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES: bool | None = _F(
        "CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES", scope="server",
        group="Captures", subgroup="Silence trim (Silero VAD)")
    CAPTURES_VAD_MARGIN_SAMPLE_EDGE_MS: Annotated[int, Field(ge=0, le=2000)] | None = _F(
        "CAPTURES_VAD_MARGIN_SAMPLE_EDGE_MS", scope="server",
        group="Captures", subgroup="Silence trim (Silero VAD)")
    CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS: Annotated[int, Field(ge=0, le=2000)] | None = _F(
        "CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS", scope="server",
        group="Captures", subgroup="Silence trim (Silero VAD)")
    CAPTURES_SAMPLE_MIN_DURATION_S: Annotated[float, Field(ge=0, le=30)] | None = _F(
        "CAPTURES_SAMPLE_MIN_DURATION_S", scope="server", group="Captures",
        subgroup="Sample sizing")
    CAPTURES_SAMPLE_MAX_DURATION_S: Annotated[float, Field(gt=0, le=30)] | None = _F(
        "CAPTURES_SAMPLE_MAX_DURATION_S", scope="server", group="Captures",
        subgroup="Sample sizing")
    CAPTURES_SAMPLE_JOIN_STRATEGY: Literal["space", "period_space"] | None = _F(
        "CAPTURES_SAMPLE_JOIN_STRATEGY", scope="server", group="Captures",
        subgroup="Sample sizing")
    CAPTURES_PROPOSER_TARGET_S: Annotated[float, Field(gt=0, le=30)] | None = _F(
        "CAPTURES_PROPOSER_TARGET_S", scope="server", group="Captures",
        subgroup="Sample sizing")
    CAPTURES_PROPOSER_SESSION_GAP_S: Annotated[int, Field(ge=1, le=86400)] | None = _F(
        "CAPTURES_PROPOSER_SESSION_GAP_S", scope="server", group="Captures",
        subgroup="Sample sizing")
    CAPTURES_PROPOSER_DUP_THRESHOLD: Annotated[float, Field(ge=0, le=1)] | None = _F(
        "CAPTURES_PROPOSER_DUP_THRESHOLD", scope="server", group="Captures",
        subgroup="Sample sizing")
    CAPTURES_PROPOSER_MAX_PROPOSALS: Annotated[int, Field(ge=1, le=200)] | None = _F(
        "CAPTURES_PROPOSER_MAX_PROPOSALS", scope="server", group="Captures",
        subgroup="Sample sizing")

    @model_validator(mode="after")
    def _validate_sample_sizing(self) -> "AdminConfig":
        # Enforce MIN ≤ TARGET ≤ MAX ≤ 30 on the EFFECTIVE values (a None
        # override means "revert to default", so fall back to the in-repo
        # default for the comparison — catches e.g. lowering MAX below the
        # target). Use config._BASELINE (the pre-override snapshot), NOT the
        # live config attribute: the live value already carries any applied
        # override, so at save time (server running) it reflects the OLD
        # override while at load time (config import) it is the bare default.
        # That asymmetry let a save pass validation, then the next restart's
        # load fail it and silently drop EVERY override on disk. _BASELINE is
        # identical at both times, so the two validations always agree.
        from faster_whisper_backend.settings import config as _cfg
        _base = getattr(_cfg, "_BASELINE", {})

        def _default(name: str) -> float:
            # Prefer the immutable baseline; fall back to the live attribute
            # only if the snapshot is somehow unavailable (partial import).
            return float(_base[name] if name in _base else getattr(_cfg, name))

        mn = self.CAPTURES_SAMPLE_MIN_DURATION_S
        tg = self.CAPTURES_PROPOSER_TARGET_S
        mx = self.CAPTURES_SAMPLE_MAX_DURATION_S
        mn = mn if mn is not None else _default("CAPTURES_SAMPLE_MIN_DURATION_S")
        tg = tg if tg is not None else _default("CAPTURES_PROPOSER_TARGET_S")
        mx = mx if mx is not None else _default("CAPTURES_SAMPLE_MAX_DURATION_S")
        if not (mn <= tg <= mx):
            raise ValueError(
                "require CAPTURES_SAMPLE_MIN_DURATION_S ≤ "
                "CAPTURES_PROPOSER_TARGET_S ≤ CAPTURES_SAMPLE_MAX_DURATION_S "
                f"(got {mn} ≤ {tg} ≤ {mx})"
            )
        return self

    @model_validator(mode="after")
    def _validate_body_caps(self) -> "AdminConfig":
        # MAX_REQUEST_BYTES is documented (config.py, FIELD_DESCRIPTIONS,
        # main._max_body_mw) as sitting ABOVE MEDIA_MAX_BYTES so an oversized
        # media POST hits the media-specific 413 that names the right setting.
        # Enforce it on the EFFECTIVE values with the same _BASELINE fallback
        # as _validate_sample_sizing (see the rationale there).
        from faster_whisper_backend.settings import config as _cfg
        _base = getattr(_cfg, "_BASELINE", {})

        def _default(name: str) -> int:
            return int(_base[name] if name in _base else getattr(_cfg, name))

        up = self.MEDIA_MAX_BYTES
        rq = self.MAX_REQUEST_BYTES
        up = up if up is not None else _default("MEDIA_MAX_BYTES")
        rq = rq if rq is not None else _default("MAX_REQUEST_BYTES")
        if rq < up:
            raise ValueError(
                "require MAX_REQUEST_BYTES >= MEDIA_MAX_BYTES "
                f"(got {rq} < {up})"
            )
        return self

    @model_validator(mode="after")
    def _validate_recording_duration(self) -> "AdminConfig":
        from faster_whisper_backend.settings import config as _cfg
        _base = getattr(_cfg, "_BASELINE", {})

        def _default(name: str) -> float:
            return float(_base[name] if name in _base else getattr(_cfg, name))

        mn = self.CAPTURES_RECORDING_MIN_DURATION_S
        mx = self.CAPTURES_RECORDING_MAX_DURATION_S
        mn = mn if mn is not None else _default("CAPTURES_RECORDING_MIN_DURATION_S")
        mx = mx if mx is not None else _default("CAPTURES_RECORDING_MAX_DURATION_S")
        if mn > mx:
            raise ValueError(
                "require CAPTURES_RECORDING_MIN_DURATION_S <= "
                "CAPTURES_RECORDING_MAX_DURATION_S "
                f"(got {mn} > {mx})"
            )
        return self

    @model_validator(mode="after")
    def _validate_buffer_trim_order(self) -> "AdminConfig":
        from faster_whisper_backend.settings import config as _cfg
        _base = getattr(_cfg, "_BASELINE", {})

        def _default(name: str) -> float:
            return float(_base[name] if name in _base else getattr(_cfg, name))

        trim = self.STREAMING_BUFFER_TRIM_S
        keep = self.STREAMING_BUFFER_TRIM_KEEP_S
        trim = trim if trim is not None else _default("STREAMING_BUFFER_TRIM_S")
        keep = keep if keep is not None else _default("STREAMING_BUFFER_TRIM_KEEP_S")
        if keep >= trim:
            raise ValueError(
                "require STREAMING_BUFFER_TRIM_KEEP_S < "
                "STREAMING_BUFFER_TRIM_S "
                f"(got {keep} >= {trim})"
            )
        return self

    @model_validator(mode="after")
    def _validate_custom_template(self) -> "AdminConfig":
        from faster_whisper_backend.settings import config as _cfg
        _base = getattr(_cfg, "_BASELINE", {})
        family = (self.TRANSLATION_PROMPT_FAMILY
                  if self.TRANSLATION_PROMPT_FAMILY is not None
                  else _base.get("TRANSLATION_PROMPT_FAMILY",
                                 getattr(_cfg, "TRANSLATION_PROMPT_FAMILY", "auto")))
        tpl = (self.TRANSLATION_PROMPT_TEMPLATE
               if self.TRANSLATION_PROMPT_TEMPLATE is not None
               else _base.get("TRANSLATION_PROMPT_TEMPLATE",
                              getattr(_cfg, "TRANSLATION_PROMPT_TEMPLATE", "")))
        if family == "custom" and not (tpl or "").strip():
            raise ValueError(
                "TRANSLATION_PROMPT_FAMILY is 'custom' but "
                "TRANSLATION_PROMPT_TEMPLATE is blank — translations would "
                "be sent as empty prompts; set a template or choose another family"
            )
        return self

    # Login sets the session cookie and then the CSRF cookie with the same
    # path/samesite; the browser keeps the LAST Set-Cookie per name, so equal
    # names make every cookie login fail (the CSRF token is read as the session).
    @model_validator(mode="after")
    def _validate_cookie_names_differ(self) -> "AdminConfig":
        from faster_whisper_backend.settings import config as _cfg
        _base = getattr(_cfg, "_BASELINE", {})
        sess = (self.SESSION_COOKIE_NAME
                if self.SESSION_COOKIE_NAME is not None
                else _base.get("SESSION_COOKIE_NAME",
                               getattr(_cfg, "SESSION_COOKIE_NAME", "whisper_session")))
        csrf = (self.SESSION_CSRF_COOKIE_NAME
                if self.SESSION_CSRF_COOKIE_NAME is not None
                else _base.get("SESSION_CSRF_COOKIE_NAME",
                               getattr(_cfg, "SESSION_CSRF_COOKIE_NAME", "whisper_csrf")))
        if sess == csrf:
            raise ValueError(
                f"SESSION_COOKIE_NAME and SESSION_CSRF_COOKIE_NAME must differ "
                f"(both {sess!r}) — the CSRF Set-Cookie would overwrite the "
                f"session cookie and every browser login would fail"
            )
        return self

    @field_validator("SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS")
    @classmethod
    def _zero_tail_not_one(cls, v: int | None) -> int | None:
        # A single zero-length last word is always kept (a real last word can
        # come out zero-length), so 1 could never fire; refuse it rather than
        # show a rule as on that is off.
        if v == 1:
            raise ValueError("must be 0 (off) or at least 2")
        return v

    @field_validator("SEGMENT_HEAD_ECHO_MIN_WORDS")
    @classmethod
    def _head_echo_not_one(cls, v: int | None) -> int | None:
        # One repeated word ("und", "die") starts sentences all the time, so the
        # guard treats 1 as off; refuse it rather than show a rule as on that
        # is off.
        if v == 1:
            raise ValueError("must be 0 (off) or at least 2")
        return v

    @field_validator("LOG_FILE")
    @classmethod
    def _safe_log_path(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Reject UNC and \\?\ extended paths; cheap to enforce, removes a class
        # of footguns where an admin types a network share by accident.
        if v.startswith("\\\\") or v.startswith("//"):
            raise ValueError("UNC / network paths are not allowed")
        # Reject path traversal segments. We use both PurePath and PureWindowsPath
        # because the deploy target is Windows but the dev machine may be Linux.
        if ".." in PurePath(v).parts or ".." in PureWindowsPath(v).parts:
            raise ValueError("'..' segments are not allowed")
        return v

    @field_validator("SERVER_HOST")
    @classmethod
    def _safe_host(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # IPv4 / IPv6 / hostname / 0.0.0.0 / ::. Loose check — the actual bind
        # error will surface on restart if the address is invalid.
        if not re.fullmatch(r"[A-Za-z0-9._:\-\[\]]+", v):
            raise ValueError("invalid host string")
        return v

    @field_validator("ALLOWED_MODELS", "PRELOAD_MODELS",
                     "TRANSLATION_ALLOWED_MODELS",
                     "TRANSLATION_PRELOAD_MODELS")
    @classmethod
    def _cap_list(cls, v: list[Any] | None) -> list[Any] | None:
        if v is None:
            return v
        if len(v) > 1000:
            raise ValueError(f"capped at 1000 entries (got {len(v)})")
        return v

    @field_validator("PIPELINE_RULES")
    @classmethod
    def _validate_pipeline_rules(
            cls, v: list[Any] | None, info: ValidationInfo) -> list[Any] | None:
        """Validate the unified pipeline rules list:
          1. Every regex pattern compiles (always). On an explicit SAVE of
             user/admin-submitted rules (the `guard_regex` validation context)
             each pattern ALSO survives an OUT-OF-PROCESS catastrophic-
             backtracking guard against a 1 KB fixture — for callback:* rows the
             single `pattern`, for regex-list rows EACH entry's `pattern`. The
             probe runs in a killable subprocess (see regex_guard) because
             CPython's re engine can't be interrupted in-process. (Empty
             patterns/entries are OK — just a no-op.)
             When the context also carries `guard_slugs` (a set of rule names),
             the guard probes ONLY those rules. The quick-config patch path
             passes the slugs it actually changed: a save merges the patch into
             the FULL rule list, and re-probing every untouched (admin) rule on
             each user save both burned the shared guard budget and meant one
             pre-existing rule the CURRENT guard refuses — saved before a guard
             tightening, loaded fine ever since — bricked every user's save of
             any rule. Compile + template checks still cover the whole list.
          2. Every regex-list entry's replacement TEMPLATE parses against its
             pattern (always, in-process — catches bad backrefs like `\\3` when
             only 2 groups exist; CPython parses templates eagerly, so this is
             cheap and has no backtracking risk).
          3. Slug uniqueness across the list.
          4. At most one terminal rule, and it must be the last entry.

        The load/startup/diff paths validate WITHOUT the `guard_regex` context,
        so a normal config load runs only the compile + template checks (fast)
        and never spawns the helper subprocess.
        """
        if v is None:
            return v
        # Collect every (where, pattern, replacement, slug) that needs a regex
        # smoke-test. The actual re.sub probe runs OUT OF PROCESS at the end
        # (only on save, via the guard_regex context) so a catastrophic-
        # backtracking pattern can be killed — CPython's re can't be interrupted
        # in-process. The slug rides along so `guard_slugs` can scope the probe
        # to the rules a patch actually changed; it is stripped before the
        # 3-tuple regex_guard.validate call.
        checks: list[tuple[str, str, str, str]] = []
        seen: set[str] = set()
        terminal_idx: int | None = None
        for idx, rule in enumerate(v):
            slug = getattr(rule, "name", None)
            rtype = getattr(rule, "type", None)
            if slug in seen:
                raise ValueError(f"duplicate rule name '{slug}' at index {idx}")
            if slug is not None:
                seen.add(slug)
            if rtype == "terminal":
                if terminal_idx is not None:
                    raise ValueError(f"only one terminal rule allowed (already at index {terminal_idx})")
                terminal_idx = idx
                continue
            # callback:map has no pattern field (auto-built from map keys at
            # compile time); skip it here.  On save, reject case-collisions:
            # the compiler lowercases all keys, so two keys differing only by
            # case silently shadow each other.
            if rtype == "callback:map":
                if (info.context or {}).get("guard_regex"):
                    m = getattr(rule, "map", None) or {}
                    lc: dict[str, list[str]] = {}
                    for k in m:
                        lc.setdefault(k.lower(), []).append(k)
                    for lower, originals in lc.items():
                        if len(originals) > 1:
                            raise ValueError(
                                f"rule {idx} ({slug!r}): map keys "
                                f"{originals} collide when lowercased")
                continue
            # regex-list: compile + guard each entry's pattern + replacement.
            if rtype == "regex-list":
                for eidx, entry in enumerate(getattr(rule, "entries", None) or []):
                    epat = getattr(entry, "pattern", None)
                    if not epat:
                        continue
                    try:
                        re.compile(epat)
                    except re.error as e:
                        raise ValueError(
                            f"rule {idx} ({slug!r}) entry {eidx}: invalid regex: {e}")
                    erepl = getattr(entry, "replacement", "") or ""
                    checks.append(
                        (f"rule {idx} ({slug!r}) entry {eidx}", epat, erepl,
                         str(slug or "")))
                continue
            # callback:* pattern-only rows (lowercase-wordlist / dedup / upper):
            # smoke-test the pattern with an empty replacement.
            pattern = getattr(rule, "pattern", None)
            if not pattern:
                continue
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"rule {idx} ({slug!r}): invalid regex: {e}")
            checks.append((f"rule {idx} ({slug!r})", pattern, "", str(slug or "")))
        if terminal_idx is not None and terminal_idx != len(v) - 1:
            raise ValueError(
                f"terminal rule must be the last entry "
                f"(found at index {terminal_idx}, list has {len(v)} rules)"
            )
        # Replacement-template sanity runs IN-PROCESS on every path: CPython
        # parses the template eagerly (before any matching), so a bad backref
        # (`\3` with two groups) is caught cheaply with zero backtracking
        # risk. This must not be save-only — a bad backref in a hand-edited
        # config.local.json has to keep failing validation at LOAD (the
        # documented fail-safe whole-file drop) instead of loading cleanly
        # and raising re.error on every request at match time.
        for where, pat, repl, _slug in checks:
            if not repl:
                continue
            try:
                re.compile(pat).sub(repl, "")
            except (re.error, IndexError) as e:
                raise ValueError(f"{where}: regex test failed: {e}")
        # Out-of-process catastrophic-backtracking guard (a real .sub run
        # against the 1 KB fixture). Runs ONLY on an explicit save of
        # user/admin-submitted rules (guard_regex context), so load/startup/
        # diff validations never spawn the helper subprocess. `guard_slugs`
        # (when present) narrows the probe to the rules the save changed —
        # see the docstring; the chained probe then only threads across the
        # probed rules' own entries, an accepted trade for not letting an
        # untouched rule fail (or time-budget-starve) someone else's save.
        if checks and (info.context or {}).get("guard_regex"):
            guard_slugs = (info.context or {}).get("guard_slugs")
            to_guard = (checks if guard_slugs is None
                        else [c for c in checks if c[3] in guard_slugs])
            if to_guard:
                from faster_whisper_backend.core import regex_guard
                regex_guard.validate([c[:3] for c in to_guard])
        return v

    @model_validator(mode="after")
    def _no_orphan_overrides(self) -> "AdminConfig":
        """Refuse to save if ALLOWED_MODELS is being shrunk in a way that
        orphans entries in MODEL_OVERRIDES. Admin must clean up overrides
        first (or keep the model in the allowlist). Never silent data loss.

        Only fires when both ALLOWED_MODELS *and* MODEL_OVERRIDES are
        present in the same payload. If only one is being saved, the cross-
        check is skipped — the merged-with-existing payload that
        save_overrides() builds will catch the conflict instead.
        """
        if self.ALLOWED_MODELS is None or self.MODEL_OVERRIDES is None:
            return self
        allowed = set(self.ALLOWED_MODELS)
        if not allowed:
            # Empty allowlist = "anything goes" per config.py convention;
            # we don't need to enforce overrides being a subset.
            return self
        orphans = sorted(set(self.MODEL_OVERRIDES.keys()) - allowed)
        if orphans:
            raise ValueError(
                f"MODEL_OVERRIDES references models not in ALLOWED_MODELS: "
                f"{orphans}. Remove the override(s) first or add them back "
                f"to the allowlist."
            )
        return self

    @model_validator(mode="after")
    def _validate_pipeline_rule_slugs(self, info: ValidationInfo) -> "AdminConfig":
        """Reject any per-model EXCLUDE / INCLUDE that references a rule slug
        not present in the canonical PIPELINE_RULES list. Closes the silent-
        typo footgun where 'dictashion-map' would save cleanly and quietly do
        nothing at runtime.

        Fires when PIPELINE_RULES is in the payload, or when save_overrides()
        supplies the live/factory slug set via the `canonical_slugs`
        validation context; a bare partial validation with neither still skips.
        """
        if self.MODEL_OVERRIDES is None:
            return self
        if self.PIPELINE_RULES is not None:
            canonical = {r.name for r in self.PIPELINE_RULES}
        else:
            canonical = (info.context or {}).get("canonical_slugs") or set()
        if not canonical:
            return self
        for model_id, override in self.MODEL_OVERRIDES.items():
            for list_name in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE"):
                slugs = getattr(override, list_name, None) or []
                unknown = [s for s in slugs if s not in canonical]
                if unknown:
                    raise ValueError(
                        f"MODEL_OVERRIDES[{model_id!r}].{list_name} "
                        f"references unknown rule slugs: {unknown}. "
                        f"Valid: {sorted(canonical)}."
                    )
        return self

    @model_validator(mode="after")
    def _validate_profile_pipeline_slugs(self, info: ValidationInfo) -> "AdminConfig":
        """Reject any OVERRIDE_PROFILES EXCLUDE / INCLUDE that references a rule
        slug not present in the canonical PIPELINE_RULES list — same silent-typo
        guard as the per-model check, applied to config profiles. Fires when
        PIPELINE_RULES is in the payload, or when save_overrides() supplies
        the live/factory slug set via the `canonical_slugs` validation
        context; a bare partial validation with neither still skips."""
        if self.OVERRIDE_PROFILES is None:
            return self
        if self.PIPELINE_RULES is not None:
            canonical = {r.name for r in self.PIPELINE_RULES}
        else:
            canonical = (info.context or {}).get("canonical_slugs") or set()
        if not canonical:
            return self
        for pname, prof in self.OVERRIDE_PROFILES.items():
            for list_name in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE"):
                slugs = getattr(prof, list_name, None) or []
                unknown = [s for s in slugs if s not in canonical]
                if unknown:
                    raise ValueError(
                        f"OVERRIDE_PROFILES[{pname!r}].{list_name} "
                        f"references unknown rule slugs: {unknown}. "
                        f"Valid: {sorted(canonical)}."
                    )
        return self

    @model_validator(mode="after")
    def _validate_captures_pipeline_slugs(self, info: ValidationInfo) -> "AdminConfig":
        """Same silent-typo guard as the per-model and per-profile checks,
        applied to CAPTURES_PIPELINE_RULES_EXCLUDE. Fires when PIPELINE_RULES
        is in the payload, or when save_overrides() supplies the live/factory
        slug set via the `canonical_slugs` validation context; a bare partial
        validation with neither still skips."""
        if self.CAPTURES_PIPELINE_RULES_EXCLUDE is None:
            return self
        if self.PIPELINE_RULES is not None:
            canonical = {r.name for r in self.PIPELINE_RULES}
        else:
            canonical = (info.context or {}).get("canonical_slugs") or set()
        if not canonical:
            return self
        unknown = [s for s in self.CAPTURES_PIPELINE_RULES_EXCLUDE
                   if s not in canonical]
        if unknown:
            raise ValueError(
                f"CAPTURES_PIPELINE_RULES_EXCLUDE references unknown "
                f"rule slugs: {unknown}. Valid: {sorted(canonical)}."
            )
        return self

    @field_validator("CONSOLE_LOG_LEVEL", mode="before")
    @classmethod
    def _lowercase_console_log_level(cls, v: Any) -> Any:
        """Accept Python's own spelling (WHISPER_CONSOLE_LOG_LEVEL=WARNING);
        the stored value is the lowercase LogLevel literal."""
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("CONVERT_QUANTIZATION", mode="before")
    @classmethod
    def _empty_convert_quant_is_unset(cls, v: Any) -> Any:
        """Treat an explicit empty string as 'unset' (-> None = use the runtime
        default). The allowed set itself is now enforced by the ConvertQuantLit
        type, which mirrors CT2's ACCEPTED_MODEL_TYPES."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("TEMPERATURE")
    @classmethod
    def _validate_temperature(cls, v: str | None) -> str | None:
        """temperature is stored as a comma-separated string (e.g. '0,0.2,0.4').
        Empty / None = library default. Validate parseable floats, ascending
        order is NOT enforced (faster-whisper accepts any order)."""
        if v is None or not v.strip():
            return v
        for token in v.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                f = float(token)
            except ValueError:
                raise ValueError(
                    f"temperature must be comma-separated floats; got '{token}'"
                )
            if not (0.0 <= f <= 1.0):
                raise ValueError(
                    f"temperature values must be in [0.0, 1.0]; got {f}"
                )
        return v

    @field_validator("TRANSLATION_PROMPT_TEMPLATE")
    @classmethod
    def _validate_translation_template(cls, v: str | None) -> str | None:
        """A non-empty custom template must carry the two mandatory slots the
        renderer substitutes — a template without them would silently translate
        nothing (no {text}) or into nowhere (no {target_language})."""
        if v is None or not v.strip():
            return v
        missing = [s for s in ("{text}", "{target_language}") if s not in v]
        if missing:
            raise ValueError(
                f"TRANSLATION_PROMPT_TEMPLATE must contain the "
                f"{' and '.join(missing)} placeholder"
                f"{'s' if len(missing) > 1 else ''} (optional slots: "
                "{source_language}, {context}, {glossary})"
            )
        return v

    @field_validator("SUPPRESS_TOKENS")
    @classmethod
    def _validate_suppress_tokens(cls, v: str | None) -> str | None:
        """suppress_tokens is stored as a comma-separated string of ints.
        '-1' is the library sentinel for default suppression set; '' = no
        suppression."""
        if v is None or not v.strip():
            return v
        for token in v.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                int(token)
            except ValueError:
                raise ValueError(
                    f"suppress_tokens must be comma-separated ints; got '{token}'"
                )
        return v

    @field_validator("ADMIN_WEBUI_ALLOWED_HOSTS", "USER_WEBUI_ALLOWED_HOSTS")
    @classmethod
    def _validate_hosts(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        for entry in v:
            try:
                ipaddress.ip_network(entry, strict=False)
            except ValueError as e:
                raise ValueError(
                    f"'{entry}' is not a valid IP or CIDR (e.g. '127.0.0.1' or "
                    f"'192.168.1.0/24'): {e}"
                )
        return v

    @field_validator("CORS_ALLOW_ORIGINS")
    @classmethod
    def _validate_cors_origins(cls, v: list[str] | None) -> list[str] | None:
        """Each entry must be '*' or a bare browser origin: scheme://host[:port]
        with NO path/query (matching what the browser sends in the Origin header);
        scheme and host are lowercased to match the browser's serialisation."""
        if v is None:
            return v
        out: list[str] = []
        for entry in v:
            if entry == "*":
                out.append(entry)
                continue
            m = re.fullmatch(r"https?://[^/?#\s*]+", entry)
            if not m:
                raise ValueError(
                    f"'{entry}' is not a valid CORS origin — use 'scheme://host[:port]' "
                    f"(e.g. 'https://app.example.com' or 'http://192.168.1.50:8000') "
                    f"or '*'; no trailing path/slash, and no wildcard host."
                )
            out.append(entry.lower())
        return out

    @field_validator("TRUSTED_ORIGINS")
    @classmethod
    def _validate_trusted_origins(cls, v: list[str] | None) -> list[str] | None:
        """Same entry shape as CORS_ALLOW_ORIGINS, minus '*': a wildcard here
        would accept every cross-site Origin and disable the guard outright.
        Scheme and host are lowercased to match the browser's serialisation."""
        if v is None:
            return v
        out: list[str] = []
        for entry in v:
            if not re.fullmatch(r"https?://[^/?#\s*]+", entry):
                raise ValueError(
                    f"'{entry}' is not a valid trusted origin — use "
                    f"'scheme://host[:port]' (e.g. 'https://whisper.example.com' "
                    f"or 'http://192.168.1.50:8000'); no trailing path/slash, "
                    f"and no '*'."
                )
            out.append(entry.lower())
        return out


# =============================================================================
# Generated field-registry tables
# =============================================================================
# Every table below is DERIVED from the per-field metadata declared on each
# AdminConfig field via _F(..., scope=…, group=…, …) — see the _F docstring.
# Public names and types are unchanged from the era when these were
# hand-written literals, so every consumer keeps working; the metadata now
# lives ON the field it describes and cannot drift from the schema.

def _registry(name: str) -> dict[str, Any]:
    """The x_registry metadata dict attached to an AdminConfig field."""
    extra = AdminConfig.model_fields[name].json_schema_extra
    reg = extra.get("x_registry") if isinstance(extra, dict) else None
    if not isinstance(reg, dict):
        raise ValueError(f"AdminConfig.{name} is missing x_registry metadata")
    return reg


_REGISTRY: dict[str, dict[str, Any]] = {
    name: _registry(name) for name in AdminConfig.model_fields
}

# Map AdminConfig field name -> env var that pins it. Mirrors the override
# block at the bottom of config.py. Used by the WebUI to mark fields as
# "currently overridden by WHISPER_X" with a badge.
ENV_VAR_MAPPING: dict[str, str] = {
    name: reg["env"] for name, reg in _REGISTRY.items()
}

# Cold settings — editing these requires a service restart for the new value
# to take effect. The WebUI shows a 'restart' badge and offers to trigger a
# self-restart after save. Note: MODEL_DEVICE / MODEL_COMPUTE_TYPE are hot —
# admin save triggers drain-then-evict on the affected loaded models so they
# reload with the new values.
RESTART_REQUIRED_FIELDS: frozenset[str] = frozenset(
    name for name, reg in _REGISTRY.items() if reg["restart"]
)

# Load-time fields. Editing these (globally OR per-model in MODEL_OVERRIDES)
# triggers drain-then-evict on the affected loaded models so the next request
# reloads them with the new values. These are read at WhisperModel(...)
# construction time; changes only take effect after re-load. Includes the
# virtual (ModelOverride-only) REVISION field.
LOAD_TIME_FIELDS: frozenset[str] = frozenset(
    name for name, reg in _REGISTRY.items() if reg["load_time"]
) | frozenset(
    name for name, spec in _VIRTUAL_OVERRIDE_FIELDS.items()
    if spec.get("load_time")
)

# Hot settings whose derived caches need rebuild after edit. The admin route
# calls main.rebuild_caches() when any of these change.
CACHE_REBUILD_FIELDS: frozenset[str] = frozenset(
    name for name, reg in _REGISTRY.items() if reg["cache_rebuild"]
)

# Types that don't survive JSON round-trip natively. Convert after model_dump
# so consumers (config.py, main.py) get the same Python types as if the values
# were defined inline in config.py. Mirrored by config._SET_FIELDS (kept there
# too: config.py needs it before this module can be imported). The registry
# stores the coercion by NAME (JSON-safe, see _F); mapped back here.
_COERCERS_BY_NAME: dict[str, Any] = {"set": set}
_POST_LOAD_COERCERS: dict[str, Any] = {
    name: _COERCERS_BY_NAME[reg["coerce"]] for name, reg in _REGISTRY.items()
    if reg["coerce"] is not None
}

# Map an UPPER_CASE config field → the lowercase client decode_override key it
# governs. Mirrors the allow-list enforced by main._apply_decode_overrides, so
# a lock on the config field blocks the matching client key. Fields absent here
# are not client-overridable, so a lock on them is a no-op for the client gate.
# Consumed by effective_config (which keeps its historical module-level alias
# _CONFIG_TO_CLIENT_KEY pointing at this dict).
CONFIG_TO_CLIENT_KEY: dict[str, str] = {
    name: reg["client_key"] for name, reg in _REGISTRY.items()
    if reg["client_key"]
}

# The client keys only live dictation reads: the STREAMING_* fields (not
# per-model, so never a model.transcribe kwarg). They ride decode_overrides
# like every client key and are applied by the streaming handshake; a batch
# request has nothing to apply them to, so it neither uses nor reports them.
STREAM_ONLY_CLIENT_KEYS: frozenset[str] = frozenset(
    reg["client_key"] for reg in _REGISTRY.values()
    if reg["client_key"] and not reg["model_override"]
)

# Fields whose resolved value a client per-request decode_override may be
# LOCKED against — every overridable scalar, i.e. everything except the
# pipeline include/exclude lists (which are virtual, not AdminConfig fields,
# and not client-overridable). scope="per_request" IS that set.
LOCKABLE_FIELDS: frozenset[str] = frozenset(
    name for name, reg in _REGISTRY.items() if reg["scope"] == "per_request"
)

# The /settings form layout, generated from the per-field group/subgroup/order
# metadata. Section order and subgroup order are pinned by _GROUP_ORDER below
# (display order is editorial, not schema-derivable); field order within a
# subgroup is the AdminConfig declaration order unless a field carries an
# explicit `order` int. Section titles mirror the per-request log block phases
# (Decode params / Pipeline / …) so an operator reading a log can find the
# matching config knobs by section name with no translation. A subgroup title
# of None means "no subheader" — fields render directly under the section.
_GROUP_ORDER: list[tuple[str, list[str | None]]] = [
    ("Models", [None, "Advanced — load-time hardware",
                "Advanced — preload & warm cache"]),
    ("Decode params", [
        None,
        "Advanced — beam & sampling",
        "Advanced — language detection (active when DEFAULT_LANGUAGE empty)",
        "Advanced — anti-hallucination & token control",
    ]),
    ("Output wrappers", [None]),
    ("Live streaming", [
        None,
        "Partial decoding (live preview)",
        "Endpointing (VAD) & speech gates",
        "Finalize & document breaks",
        "Buffer management",
    ]),
    ("Diarization", [None, "Advanced — speaker bounds & VRAM"]),
    ("Music separation", [None]),
    ("Translation", [None, "Per-request defaults"]),
    ("Transcribe from URL", [
        None,
        "Advanced — timeouts, concurrency & retention",
    ]),
    ("Media export", [None]),
    ("Per-model overrides", [None]),
    ("Pipeline", [None]),
    ("Logging", [None]),
    ("Server", [None]),
    ("Access & sessions", [None, "Browser sessions (cookie auth)"]),
    ("Concurrency & Request Limits", [None]),
    ("Reports", [None]),
    ("Recent transcriptions", [None]),
    ("Jobs", [None]),
    ("System metrics", [None]),
    ("Usage statistics", [None]),
    ("Client settings sync", [None]),
    ("Captures", [
        None,
        "Storage",
        "Duration & size guards",
        "Sample sizing",
        "Training-form pipeline",
        "Silence trim (Silero VAD)",
    ]),
]


def _build_field_groups() -> list[tuple[str, list[tuple[str | None, list[str]]]]]:
    """Assemble the _GROUP_ORDER × per-field metadata into the nested
    section/subgroup/fields structure admin_routes renders. Raises at import
    on any drift: a field pointing at an unlisted (group, subgroup), or a
    listed subgroup no field belongs to."""
    decl_idx = {name: i for i, name in enumerate(AdminConfig.model_fields)}
    listed: set[tuple[str, str | None]] = {
        (section, sub) for section, subs in _GROUP_ORDER for sub in subs
    }
    for name, reg in _REGISTRY.items():
        if reg["group"] is not None and (reg["group"], reg["subgroup"]) not in listed:
            raise ValueError(
                f"AdminConfig.{name} declares group/subgroup "
                f"({reg['group']!r}, {reg['subgroup']!r}) not listed in "
                f"_GROUP_ORDER")
    out: list[tuple[str, list[tuple[str | None, list[str]]]]] = []
    for section, subs in _GROUP_ORDER:
        rendered: list[tuple[str | None, list[str]]] = []
        for sub in subs:
            names = sorted(
                (n for n, r in _REGISTRY.items()
                 if r["group"] == section and r["subgroup"] == sub),
                key=lambda n: (
                    _REGISTRY[n]["order"]
                    if _REGISTRY[n]["order"] is not None else decl_idx[n],
                    decl_idx[n],
                ),
            )
            if not names:
                raise ValueError(
                    f"_GROUP_ORDER lists empty subgroup ({section!r}, {sub!r})")
            rendered.append((sub, names))
        out.append((section, rendered))
    return out


FIELD_GROUPS: list[tuple[str, list[tuple[str | None, list[str]]]]] = (
    _build_field_groups()
)

# Derived-extras eviction: editing any of the fields in a bucket drops the
# matching cached extra (diarization pipeline / BGM separator) so its VRAM
# frees now instead of at the idle timeout. Generated from the per-field
# `evict=` metadata; admin_routes.post_state dispatches each bucket name
# through its _EVICTORS table.
EXTRAS_EVICTION: dict[str, frozenset[str]] = {
    bucket: frozenset(
        name for name, reg in _REGISTRY.items() if reg["evict"] == bucket
    )
    for bucket in sorted({r["evict"] for r in _REGISTRY.values() if r["evict"]})
}


# =============================================================================
# Generated override schemas
# =============================================================================
# A ModelOverride bundle lives at MODEL_OVERRIDES[model_id]. Every field is
# Optional — absent means "inherit the global default". The runtime helper
# effective_config.cfg_for(model_id, field) walks: per-model override > global > faster-
# whisper default. Same precedence as everywhere else, just with one more
# layer interposed.
#
# Validation: models may only carry override values that pass the same
# constraints as the corresponding global field — guaranteed structurally:
# each override field's annotation is copied verbatim from the AdminConfig
# field it overrides (the bounds live inside the Annotated type), selected by
# the per-field registry metadata:
#   scope="per_request" + model_override=True  → call-time mixin
#   scope="per_request" + model_override=False → streaming mixin
#   scope="per_model"                          → ModelOverride load-time block
# plus _VIRTUAL_OVERRIDE_FIELDS (REVISION → ModelOverride; the pipeline
# include/exclude lists → call-time mixin). Pipeline rule scoping uses
# PIPELINE_RULES_EXCLUDE: a flat list of rule slugs to skip for this model.
# Rule bodies are NEVER per-model — they stay in the single global
# PIPELINE_RULES list, edited in the global pipeline editor. The per-model
# pane only toggles inclusion via a checklist.

def _override_defs(pred: Any) -> dict[str, Any]:
    """create_model field definitions `(annotation, None)` for every
    AdminConfig field whose registry entry satisfies `pred`. The annotation is
    the field's exact Optional annotated type (bounds included); the AdminConfig
    field-level FieldInfo (description, x_registry) is deliberately NOT copied,
    so the override schemas stay as minimal as the hand-written classes they
    replaced."""
    return {
        name: (AdminConfig.model_fields[name].annotation, None)
        for name, reg in _REGISTRY.items() if pred(reg)
    }


def _virtual_defs(scope: str) -> dict[str, Any]:
    """create_model field definitions for the _VIRTUAL_OVERRIDE_FIELDS of the
    given scope (fields that exist ONLY on the override schemas)."""
    return {
        name: (spec["annotation"], None)
        for name, spec in _VIRTUAL_OVERRIDE_FIELDS.items()
        if spec["scope"] == scope
    }


class _CallTimeOverrideBase(BaseModel):
    """Config + validator carrier for the generated _CallTimeOverrideMixin
    (create_model can't attach validators directly)."""
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    @model_validator(mode="after")
    def _no_overlap_include_exclude(self) -> "_CallTimeOverrideBase":
        """A rule slug cannot be both force-disabled AND force-enabled in the
        same bundle — admin must pick one. Catches obvious misconfiguration
        (e.g. typed both lists then forgot to clean one up)."""
        ex = set(getattr(self, "PIPELINE_RULES_EXCLUDE", None) or [])
        inc = set(getattr(self, "PIPELINE_RULES_INCLUDE", None) or [])
        overlap = ex & inc
        if overlap:
            raise ValueError(
                f"PIPELINE_RULES_EXCLUDE and PIPELINE_RULES_INCLUDE overlap: "
                f"{sorted(overlap)} — a rule cannot be both force-disabled "
                f"and force-enabled in the same bundle. Remove from one of "
                f"the lists."
            )
        return self


class _StreamingOverrideBase(BaseModel):
    """Config carrier for the generated _StreamingOverrideMixin."""
    model_config = {"extra": "forbid", "protected_namespaces": ()}


# Call-time (decode + post-processing) override fields, shared by ModelOverride
# (per-model) and OverrideProfile (per-identity) so the bounds are
# single-sourced and the two layers can never drift apart. All fields optional;
# absent = inherit the next layer down. The diarization / BGM capacity switches
# (DIARIZATION_ENABLED, BGM_SEPARATION_ENABLED) are deliberately server-scoped
# so they never appear here; DIARIZE / the speaker bounds / SEPARATE_BGM are
# per-caller policy and therefore do.
_CallTimeOverrideMixin = create_model(
    "_CallTimeOverrideMixin",
    __base__=_CallTimeOverrideBase,
    **_override_defs(
        lambda r: r["scope"] == "per_request" and r["model_override"]),
    **_virtual_defs("per_request"),
)
_CallTimeOverrideMixin.__doc__ = (
    "Call-time (decode + post-processing) override fields, shared by "
    "ModelOverride (per-model) and OverrideProfile (per-identity) so the "
    "bounds are single-sourced and the two layers can never drift apart. "
    "All fields optional; absent = inherit the next layer down. Generated "
    "from the AdminConfig registry (scope=per_request, model_override=True) "
    "+ the virtual pipeline include/exclude lists."
)

# Live-streaming (WebSocket dictation) override fields that are meaningful
# per-identity — partial-decode knobs, VAD / speech gates, finalize &
# document-break, buffer trimming, idle timeout. Hard server-capacity caps
# (STREAMING_ENABLED, STREAMING_MAX_SESSIONS, INFERENCE_CONCURRENCY) and the
# partial-model selector are deliberately NOT here — they are server-wide, not
# per-caller (scope="server"). The idle timeout IS here: it is a per-caller
# policy (a trusted profile can be granted a longer silence grace than an
# anonymous one). Bounds are copied verbatim from the AdminConfig STREAMING_*
# fields so the two can never drift. All optional; absent = inherit the next
# layer down.
_StreamingOverrideMixin = create_model(
    "_StreamingOverrideMixin",
    __base__=_StreamingOverrideBase,
    **_override_defs(
        lambda r: r["scope"] == "per_request" and not r["model_override"]),
)
_StreamingOverrideMixin.__doc__ = (
    "Live-streaming override fields that are meaningful per-identity. "
    "Generated from the AdminConfig registry (scope=per_request, "
    "model_override=False)."
)

# Per-model override bundle: the call-time fields + the load-time fields
# (editing any of the latter drains-then-evicts the affected loaded model).
# All fields optional; absent = inherit global.
ModelOverride = create_model(
    "ModelOverride",
    __base__=_CallTimeOverrideMixin,
    **_override_defs(lambda r: r["scope"] == "per_model"),
    **_virtual_defs("per_model"),
)
ModelOverride.__doc__ = (
    "Per-model override bundle. Inherits the call-time fields from "
    "_CallTimeOverrideMixin; adds the load-time fields (scope=per_model + "
    "the virtual REVISION — editing any of these drains-then-evicts the "
    "affected loaded model). All fields optional; absent = inherit global."
)


class OverrideProfile(_CallTimeOverrideMixin, _StreamingOverrideMixin):
    """A reusable, named per-identity override bundle — the config-bearing
    "profile" (the evolution of the visibility-only tag concept). Carries the
    call-time + streaming override fields (absent = inherit the next layer
    down) plus `locks`: the field names whose resolved value a client's
    per-request decode_override may NOT replace."""

    locks: Annotated[list[str], Field(max_length=200)] | None = None

    # Whether a client may NAME this profile in a per-request `override_profile`.
    # None / True = requestable (clients can select it); False = internal-only —
    # the profile may still be admin-applied via a per-key/per-user binding, but
    # is never offered to clients and a request naming it is silently refused.
    # This is profile-library metadata, NOT a per-field override or a lockable
    # field, so it is excluded from the per-identity `direct` blob (validate_
    # binding strips it) and rendered as a profile-level control, not in the
    # decode-field grid.
    requestable: bool | None = None

    @field_validator("locks")
    @classmethod
    def _validate_locks(cls, v: list[str] | None) -> list[str] | None:
        if not v:
            return v
        bad = sorted(f for f in v if f not in LOCKABLE_FIELDS)
        if bad:
            raise ValueError(
                f"locks references non-lockable field(s): {bad}. Lockable "
                f"fields are the overridable decode/streaming scalars."
            )
        return sorted(set(v))


# AdminConfig's MODEL_OVERRIDES / OVERRIDE_PROFILES annotations forward-
# reference the two models generated above; resolve them now that both exist.
AdminConfig.model_rebuild()


# JSON-schema "type" → widget kind for override_field_meta below.
_JSON_TYPE_TO_KIND = {
    "integer": "int", "number": "float", "boolean": "bool",
    "string": "str", "array": "list",
}


def override_field_meta(
    model: type[BaseModel],
    *,
    exclude: "frozenset[str] | set[str]" = frozenset(),
) -> dict[str, dict[str, Any]]:
    """Widget metadata (kind / min / max / maxlen / opts) for every field of an
    override model, derived from its JSON schema so it can never drift from
    the Pydantic bounds. Shared by overrides_routes (OverrideProfile → the
    profile editor + direct-override sub-editor) and admin_routes
    (ModelOverride → the /settings per-model pane's injected FIELD_META)."""
    schema = model.model_json_schema()
    out: dict[str, dict[str, Any]] = {}
    for name, spec in schema.get("properties", {}).items():
        if name in exclude:
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
            info["kind"] = _JSON_TYPE_TO_KIND.get(v.get("type"), "str")
        if "minimum" in v:
            info["min"] = v["minimum"]
        if "maximum" in v:
            info["max"] = v["maximum"]
        if "maxLength" in v:
            info["maxlen"] = v["maxLength"]
        out[name] = info
    return out


@functools.lru_cache(maxsize=1)
def field_bounds() -> dict[str, dict[str, Any]]:
    """Per-identity overridable field → its widget metadata (kind / min / max
    / maxlen), from the same JSON schema the profile editor reads
    (override_field_meta over OverrideProfile) — so a request-side clamp can
    never drift from the admin bounds. Cached: the schema is fixed at import.
    Callers must not mutate the result."""
    return override_field_meta(OverrideProfile)


@functools.lru_cache(maxsize=1)
def client_key_bounds() -> dict[str, dict[str, Any]]:
    """field_bounds() keyed by client decode key instead of config field."""
    meta = field_bounds()
    return {client_key: meta[field]
            for field, client_key in CONFIG_TO_CLIENT_KEY.items()
            if field in meta}


def format_validation_errors(err: ValidationError) -> list[dict[str, str]]:
    """Shape a Pydantic ValidationError into compact JSON for the WebUI.

    Each entry: {"loc": "FIELD.SUBPATH", "msg": "human-readable explanation"}.
    No traceback or input-value leaking — failure messages stay terse.
    """
    out: list[dict[str, str]] = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()))
        msg = e.get("msg", "invalid value")
        out.append({"loc": loc, "msg": msg})
    return out
