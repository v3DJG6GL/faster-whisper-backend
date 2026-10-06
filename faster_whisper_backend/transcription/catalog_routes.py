"""What a client may ask for: GET /v1/models, /v1/me (capabilities and
feature flags), /v1/override-profiles* and /v1/request-default-settings (with
its deprecated /v1/decode-defaults alias).
"""
import time

from fastapi import APIRouter, Depends, HTTPException

from faster_whisper_backend.auth.dependencies import get_current_user as _get_current_user_dep
from faster_whisper_backend.build_info import APP_VERSION, BOOT_ID, SERVER_NAME
from faster_whisper_backend.core.languages import ALL_LANGUAGE_NAMES, language_codes
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.runtime import model_registry
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import progress as tx_progress
from faster_whisper_backend.translation import engine as _tr
from faster_whisper_backend.translation import gating as tr_gating

router = APIRouter()


@router.get("/v1/models", dependencies=[Depends(_get_current_user_dep)])
async def list_models():
    """OpenAI-style model listing — currently-loaded models plus the configured
    default. Useful for clients to discover what's available without trial.

    User-tier auth like its /v1 siblings: the payload carries the build version,
    the per-process boot_id and the whole ALLOWED_MODELS list, so it is a
    fingerprint, not public metadata."""
    now = int(time.time())
    # The device each loaded model actually sits on — register_loaded_model
    # records the fallback device after a failed primary load.
    loaded_devices = {m["name"]: m["device"]
                      for m in model_registry.loaded_models_snapshot()}
    names: list[str] = list(tx_models._loaded_models.keys())
    if cfg.DEFAULT_MODEL not in names:
        names.append(cfg.DEFAULT_MODEL)
    if cfg.ALLOWED_MODELS:
        for n in sorted(cfg.ALLOWED_MODELS):
            if n not in names:
                names.append(n)
    return {
        "object": "list",
        "boot_id": BOOT_ID,
        # Build identity (non-standard, like boot_id): lets clients show
        # "faster-whisper-backend · v0.1.0" instead of a generic detection tag.
        "server_name": SERVER_NAME,
        "server_version": APP_VERSION,
        "data": [
            {
                "id": n,
                "object": "model",
                "created": now,
                "owned_by": "local",
                # preload.is_resident, not `n in _loaded_models`: one
                # residency predicate for all four families, so this flag and
                # the preloader's admission ladder can never disagree.
                "loaded": preload.is_resident("whisper", n),
                "device": _model_device(n, loaded_devices.get(n)),
            }
            for n in names
        ],
    }


def _model_device(name: str, loaded_device: "str | None") -> str:
    """The device a whisper model runs on, as a short lowercase token: where
    it is loaded, else where a load would put it (MODEL_DEVICE, per-model
    override > global). "auto" resolves like CTranslate2 does: cuda when it
    sees a GPU, else cpu."""
    device = str(loaded_device or effective_config.cfg_for(name, "MODEL_DEVICE") or "cpu").lower()
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001 — no CT2 / no driver reads as cpu
            device = "cpu"
    return device[:16]


# llama-cpp-python version cache, url_download.yt_dlp_version-style: the
# distribution finder walks sys.path (listdir/stat per entry) for a value
# that cannot change without a restart, and /v1/me is the WebUI's per-page-
# load capability probe — resolve once, behind a sentinel so "not installed"
# (None) is cached too.
_LLAMA_CPP_VERSION_UNSET = object()
_LLAMA_CPP_VERSION: "str | None | object" = _LLAMA_CPP_VERSION_UNSET


def _llama_cpp_version() -> "str | None":
    global _LLAMA_CPP_VERSION
    if _LLAMA_CPP_VERSION is _LLAMA_CPP_VERSION_UNSET:
        try:
            import importlib.metadata
            _LLAMA_CPP_VERSION = importlib.metadata.version("llama-cpp-python")
        except Exception:  # noqa: BLE001 — absence is a supported state
            _LLAMA_CPP_VERSION = None
    return _LLAMA_CPP_VERSION


@router.get("/v1/me")
async def whoami_capabilities(user: dict = Depends(_get_current_user_dep)):
    """The caller's effective request-override capabilities — drives the client
    UI (hide the decode editor / override-profile picker it isn't permitted to
    use). The client UI is convenience only; the server enforces everything in
    effective_config.resolve regardless. User-tier auth: any valid key; 401
    without one when the server is locked down.

    Returns: {can_request_override_profile, can_request_decode_overrides,
    allowed_override_profiles: ["*"] | [names…] | []} plus the feature flags
    below. The decode values a caller inherits are GET
    /v1/request-default-settings."""
    caps = effective_config.resolve_capabilities(
        user_id=user.get("user_id"), key_id=user.get("key_id"))
    # The caller's identity layers, for the per_request values below
    # (TRANSLATE_TO, TRANSLATION_MAX_TARGETS) — resolved once.
    _ident = effective_config.build_ident(user, None)
    # Additive: whether the optional pipeline stages exist on this server at
    # all, so the client can disable its "Separate music" / "Speaker
    # diarization" toggles pre-flight instead of letting a request soft-fail
    # into a warning. Server-wide feature switches, not per-identity grants.
    caps["bgm_separation_enabled"] = bool(
        getattr(cfg, "BGM_SEPARATION_ENABLED", False))
    caps["diarization_enabled"] = bool(
        getattr(cfg, "DIARIZATION_ENABLED", False))
    # Additive: transcribe-from-URL capacity switch + the installed yt-dlp
    # version (or null). The version is deliberately visible to user-tier
    # callers: "is the downloader stale?" is the first question when a site
    # stops working, and clients surface it in their error guidance.
    caps["url_download_enabled"] = bool(
        getattr(cfg, "URL_DOWNLOAD_ENABLED", False))
    if caps["url_download_enabled"]:
        from faster_whisper_backend.media import download as _udl
        caps["yt_dlp_version"] = _udl.yt_dlp_version()
    # The one media ceiling, so the client can label a preview's rungs over
    # the cap, size its own local copies and refuse an oversized upload
    # before sending it. Always present: uploads are capped too.
    caps["media_max_bytes"] = int(getattr(cfg, "MEDIA_MAX_BYTES", 10_000_000_000))
    # Additive: whether a link's VIDEO can be kept/fetched (keep_video on the
    # transcription form, POST /v1/audio/url-media/video). Always present;
    # the detail key rides only when on — same discipline as yt_dlp_version.
    caps["url_video_enabled"] = bool(
        caps["url_download_enabled"] and getattr(cfg, "URL_VIDEO_ENABLED", False))
    if caps["url_video_enabled"]:
        # No server-side height ceiling in this release: null = best available.
        caps["url_video_default_max_height"] = None
    # Additive: whether the spoken-language check of a link is offered
    # (POST /v1/audio/url-language).
    caps["url_language_check_enabled"] = bool(
        caps["url_download_enabled"]
        and getattr(cfg, "URL_LANGUAGE_CHECK_ENABLED", False))
    # Additive: whether a link's own subtitle tracks are listed by the preview
    # and fetchable (POST /v1/audio/url-subtitles).
    caps["url_subtitles_enabled"] = bool(
        caps["url_download_enabled"]
        and getattr(cfg, "URL_SUBTITLES_ENABLED", False))
    # Additive: the durable job resource (GET/DELETE /v1/jobs*). The flag is
    # always present; the detail block rides only when on.
    caps["jobs_enabled"] = tx_progress._jobs_enabled()
    if caps["jobs_enabled"]:
        caps["jobs"] = {"ttl_s": int(tx_progress._jobs_ttl_s())}
    # Additive: subtitle packaging (POST /v1/audio/media{,/{id}/package}).
    # The flag is always present; the detail block rides only when the
    # feature is on — its `reason` says why ffmpeg cannot (a stripped build).
    from faster_whisper_backend.media import subtitle_mux as _pk
    _pk_on = bool(getattr(cfg, "MEDIA_PACKAGE_ENABLED", True))
    _pk_caps = _pk.ffmpeg_capabilities() if _pk_on else None
    caps["media_package_enabled"] = bool(_pk_on and _pk_caps and _pk_caps.available)
    if _pk_on:
        caps["media_package"] = {
            "containers": [c for c, ok in (("mkv", _pk_caps.mkv), ("mp4", _pk_caps.mp4)) if ok],
            "max_tracks": _pk.MAX_TRACKS,
            "max_srt_bytes": _pk.MAX_SRT_BYTES,
            "max_upload_bytes": int(getattr(cfg, "MEDIA_MAX_BYTES", 10_000_000_000)),
            "reason": None if _pk_caps.available else _pk_caps.reason,
            "ffmpeg_version": _pk_caps.version,
        }
    # Stage-model list builder shared by all three optional stages below.
    # Configured model FIRST, then the allowlist (de-duplicated, order kept):
    # an empty allowlist means "the configured model only" for the
    # diarization/separation stages, so building from the allowlist alone
    # published [] for a server that accepts one model — and a picker
    # pre-flighting on it showed nothing. This is exactly those stages'
    # admission rule (allowlist ∪ configured). Translation differs: its
    # EMPTY allowlist is permissive at request time (_translation_model_
    # allowed admits any well-formed ref), so translation_models is "what
    # the picker offers", not "all the server accepts". The list
    # deliberately never consults residency: a loaded-but-no-longer-
    # allowed model must not be offered (every request naming it would be
    # refused); residency is reported per row by the "loaded" flag instead.
    def _stage_refs(configured, allowed) -> "list[str]":
        refs = [configured] if configured else []
        for _m in (allowed or []):
            if _m and _m not in refs:
                refs.append(_m)
        return refs

    # Additive: text-to-text translation capability surface. The flag is
    # always present (pre-flight for the client's Translate control); the
    # detail keys ride only when the stage exists — same shape discipline as
    # yt_dlp_version above.
    caps["translation_enabled"] = bool(
        getattr(cfg, "TRANSLATION_ENABLED", False))
    if caps["translation_enabled"]:
        _t_default = tr_gating._translation_default_model()
        # "languages": the codes the model supports (TRANSLATION_LANGUAGES
        # override, else its family table ∪ model card), null = unknown.
        caps["translation_models"] = [
            {"id": _ref, "loaded": preload.is_resident("translation", _ref),
             "languages": _tr.languages_for(_ref)}
            for _ref in _stage_refs(
                _t_default,
                sorted(getattr(cfg, "TRANSLATION_ALLOWED_MODELS", None)
                       or set()))]
        # DEPRECATED — clients read translation_models[].languages. Kept for
        # older clients: the default model's list, or every named code when
        # that is unknown.
        caps["translation_languages"] = (
            _tr.languages_for(_t_default) or sorted(ALL_LANGUAGE_NAMES))
        # The CALLER's effective TRANSLATE_TO default (per-identity overrides
        # respected), parsed csv → list like the transcribe handler does.
        caps["translate_to_default"] = language_codes(
            effective_config.cfg_for(None, "TRANSLATE_TO", _ident))
        # Engine version, yt_dlp_version-style best-effort (null when the
        # optional dependency set isn't installed); cached at module level —
        # it cannot change without a restart.
        caps["llama_cpp_version"] = _llama_cpp_version()
    # Additive: the stage-model allowlists with a loaded flag, mirroring
    # translation_models — the client's model pickers pre-flight on these.
    # "Loaded" = the module's single cached instance is exactly this model
    # (both stages cache one pipeline/separator at a time). preload.is_resident
    # owns that comparison for every family, including the separator's
    # friendly-name → ".onnx" filename mapping this used to open-code.
    caps["diarization_models"] = [
        {"id": _m, "loaded": preload.is_resident("diarization", _m)}
        for _m in _stage_refs(
            (getattr(cfg, "DIARIZATION_MODEL", "") or "").strip(),
            getattr(cfg, "DIARIZATION_ALLOWED_MODELS", None))]
    caps["separation_models"] = [
        {"id": _m, "loaded": preload.is_resident("separation", _m)}
        for _m in _stage_refs(
            (getattr(cfg, "BGM_SEPARATION_UVR_MODEL", "") or "").strip(),
            getattr(cfg, "BGM_SEPARATION_ALLOWED_MODELS", None))]
    caps["server_info"] = _server_info(caps, _ident)
    return caps


def _server_info(caps: dict, ident) -> dict:
    """/v1/me's read-only policy view: the limits a request runs into and
    what this server keeps about the caller's work. Privacy-positive
    disclosure (SECURITY-REVIEW_NOTES, "server_info is an admin policy
    view"): retention knobs and caps only, never data. Every value is
    scope="server" (plain cfg) except TRANSLATION_MAX_TARGETS, which is
    per_request and resolved for the caller exactly like the text route."""
    limits: dict = {
        "translation_max_targets": int(
            effective_config.cfg_for(None, "TRANSLATION_MAX_TARGETS", ident) or 1),
    }
    if caps.get("url_download_enabled"):
        limits["url_max_duration_s"] = int(getattr(cfg, "URL_MAX_DURATION_S", 0) or 0)
        # [] = every dedicated extractor (the allowlist is off).
        limits["url_allowed_extractors"] = list(
            getattr(cfg, "URL_ALLOWED_EXTRACTORS", None) or [])
        limits["url_allow_direct_media"] = bool(
            getattr(cfg, "URL_ALLOW_DIRECT_MEDIA", False))
    return {
        "limits": limits,
        "keeps": {
            # Captures are only written while word timestamps are on (the
            # batch route and the live final decode both gate on it); the
            # switch is still what the operator chose.
            "captures": {
                "enabled": bool(getattr(cfg, "CAPTURES_RECORDING_ENABLED", False)),
                "retention_days": int(getattr(cfg, "CAPTURES_RETENTION_DAYS", 0) or 0),
                "sample_fraction": float(
                    getattr(cfg, "CAPTURES_RECORDING_SAMPLE_RATE", 1.0)),
                "max": int(getattr(cfg, "CAPTURES_MAX", 0) or 0),
            },
            "server_log": {
                "max_bytes": int(getattr(cfg, "LOG_MAX_BYTES", 0) or 0),
                "backup_count": int(getattr(cfg, "LOG_BACKUP_COUNT", 0) or 0),
            },
            "recent_transcriptions": {
                "retention_days": int(
                    getattr(cfg, "RECENT_TRANSCRIPTIONS_RETENTION_DAYS", 0) or 0),
                "max": int(getattr(cfg, "RECENT_TRANSCRIPTIONS_MAX", 0) or 0),
            },
            # 0 = kept forever (the retention loops skip a 0).
            "usage_app_retention_days": int(
                getattr(cfg, "USAGE_APP_RETENTION_DAYS", 0) or 0),
            "usage_retention_days": int(getattr(cfg, "USAGE_RETENTION_DAYS", 0) or 0),
            "usage_jobs_retention_days": int(
                getattr(cfg, "USAGE_JOBS_RETENTION_DAYS", 0) or 0),
            "url_media_ttl_s": int(getattr(cfg, "URL_MEDIA_TTL_S", 0) or 0),
        },
    }


@router.get("/v1/override-profiles")
async def list_override_profiles(user: dict = Depends(_get_current_user_dep)):
    """Names of the server-side OVERRIDE_PROFILES THIS caller may reference via the
    per-request `override_profile` field — filtered by the global gate, the
    caller's per-identity gate + allowlist, and each profile's `requestable` flag.
    Names only — never the profile contents; empty list when the caller may not
    request any. User-tier auth: any valid key (admin not required); 401 without
    one when the server is locked down."""
    names = effective_config.allowed_profile_names(
        user_id=user.get("user_id"), key_id=user.get("key_id"))
    return {"profiles": names}


@router.get("/v1/override-profiles/{name}")
async def get_override_profile(name: str,
                               user: dict = Depends(_get_current_user_dep)):
    """The decode-relevant values + locked client keys of a single override-
    profile THIS caller may request — for the client to preview as inherited
    defaults. 404 when the profile doesn't exist OR the caller may not request it
    (don't leak internal / disallowed profiles). The returned `values` are the
    profile's OWN contribution projected to the client decode keys; admin locks
    elsewhere can still win at request time (reported then via overrides_ignored).
    User-tier auth."""
    allowed = effective_config.allowed_profile_names(
        user_id=user.get("user_id"), key_id=user.get("key_id"))
    if name not in allowed:
        raise HTTPException(status_code=404, detail="override-profile not found")
    profiles = getattr(cfg, "OVERRIDE_PROFILES", None) or {}
    blob = profiles.get(name)
    values, locked = effective_config.project_profile_to_client(blob)
    # `prompt` is exposed SEPARATELY (not in `values`, which is exactly the
    # client decode keys): the client's "Vocabulary / prompt" maps to the server's
    # DEFAULT_PROMPT, which has no client decode key, so the editor needs it here to
    # ghost the profile's prompt as an inherited default.
    prompt = None
    prompt_locked = False
    if isinstance(blob, dict):
        _p = blob.get("DEFAULT_PROMPT")
        if isinstance(_p, str) and _p:
            prompt = _p
        prompt_locked = "DEFAULT_PROMPT" in (blob.get("locks") or [])
    return {"name": name, "values": values, "locked": locked,
            "prompt": prompt, "prompt_locked": prompt_locked}


# The longest model id GET /v1/request-default-settings looks at; a real one is far shorter.
_DECODE_DEFAULTS_MODEL_MAX = 200


def _provenance_source(rows: "list[dict] | None") -> "tuple[str, str]":
    """(source, label) of the row that supplies a field's value, from
    effective_config's provenance stack. Coarse categories for the client's
    tooltip: an identity layer bound to the caller's key or user is "account",
    the request-named profile "override_profile", then "model" / "server";
    nothing set anywhere is faster-whisper's own default ("builtin"). A field
    without a provenance stack (not identity-overridable) reads from global."""
    if rows is None:
        return "server", "global default"
    for row in rows:
        if not (row.get("is_winner") and row.get("is_set")):
            continue
        layer_id = str(row.get("layer_id") or "")
        label = str(row.get("label") or "")
        if layer_id.startswith("request.profile:"):
            return "override_profile", label
        if layer_id.startswith(("key.", "user.")):
            return "account", label
        if layer_id == "per-model":
            return "model", label
        if layer_id == "global":
            return "server", label
    return "builtin", "faster-whisper default"


@router.get("/v1/request-default-settings")
@router.get("/v1/decode-defaults", deprecated=True)
async def get_decode_defaults(model: str = "", override_profile: str = "",
                              user: dict = Depends(_get_current_user_dep)):
    """The decode values THIS caller's requests get when they send no
    decode_overrides — each of the client decode keys with its value, where it
    comes from and whether an admin locked it, plus the prompt and the two
    values live dictation pins on its final decode. Drives the client's
    "Inherit · <value>" labels.

    Resolved exactly like a request: identity layers (key/user direct values,
    their bound profiles, then ``override_profile`` when the caller may request
    it; ``__none__`` drops the bound profiles) > per-model > global. ``model``
    "" or "whisper-1" is DEFAULT_MODEL; a model the server would refuse is 400,
    like a transcription naming it. Nothing is loaded.

    Exposes the global / per-model / bound DEFAULT_PROMPT and DEFAULT_HOTWORDS to
    any key holder — intended: they shape every transcript that key gets back
    (SECURITY-REVIEW_NOTES, "decode defaults are readable"). User-tier auth."""
    model = (model or "").strip()
    if len(model) > _DECODE_DEFAULTS_MODEL_MAX:
        raise HTTPException(status_code=400, detail="Model id is too long.")
    model_name = tx_models._resolve_model_name(model)
    tx_models._check_model_name(model_name)
    request_profile = (override_profile or "").strip() or None
    ident = effective_config.build_ident(user, model_name, request_profile=request_profile,
                        with_provenance=True)
    provenance = ident.provenance or {}

    def _entry(field: str, value, locked: bool) -> dict:
        source, label = _provenance_source(provenance.get(field))
        return {"value": value, "source": source, "label": label,
                "locked": bool(locked)}

    settings: dict = {}
    for field, client_key in settings_schema.CONFIG_TO_CLIENT_KEY.items():
        value = effective_config.cfg_for(model_name, field, ident)
        # Blank text is "unset" to the decoder (hotwords are only sent when
        # non-blank) — say so, rather than ghosting an empty string.
        if isinstance(value, str) and not value.strip() and client_key == "hotwords":
            value = None
        settings[client_key] = _entry(field, value,
                                      client_key in ident.locked_client_keys)
    prompt = effective_config.cfg_for(model_name, "DEFAULT_PROMPT", ident)
    prompt = prompt if isinstance(prompt, str) and prompt.strip() else None
    return {
        "model": model_name,
        "profile_applied": ident.request_profile_applied,
        "settings": settings,
        "prompt": _entry("DEFAULT_PROMPT", prompt, "DEFAULT_PROMPT" in ident.locked),
        # The per-run defaults a file/link run inherits for its form fields
        # (not decode keys — locked by config name like the prompt), resolved
        # the way the run resolves them.
        "language": _entry("DEFAULT_LANGUAGE",
                           effective_config.cfg_for(model_name, "DEFAULT_LANGUAGE", ident) or "",
                           "DEFAULT_LANGUAGE" in ident.locked),
        "word_timestamps": _entry(
            "WORD_TIMESTAMPS_ENABLED",
            bool(effective_config.cfg_for(model_name, "WORD_TIMESTAMPS_ENABLED", ident)),
            "WORD_TIMESTAMPS_ENABLED" in ident.locked),
        "diarize": _entry("DIARIZE", bool(effective_config.cfg_for(model_name, "DIARIZE", ident)),
                          "DIARIZE" in ident.locked),
        "separate_bgm": _entry("SEPARATE_BGM",
                               bool(effective_config.cfg_for(model_name, "SEPARATE_BGM", ident)),
                               "SEPARATE_BGM" in ident.locked),
        "diarization_model": _entry(
            "DIARIZATION_MODEL",
            (effective_config.cfg_for(model_name, "DIARIZATION_MODEL", ident) or "").strip(),
            "DIARIZATION_MODEL" in ident.locked),
        "separation_model": _entry(
            "BGM_SEPARATION_UVR_MODEL",
            (effective_config.cfg_for(model_name, "BGM_SEPARATION_UVR_MODEL", ident) or "").strip(),
            "BGM_SEPARATION_UVR_MODEL" in ident.locked),
        "translation": {
            "context_segments": _entry(
                "TRANSLATION_CONTEXT_SEGMENTS",
                int(effective_config.cfg_for(model_name, "TRANSLATION_CONTEXT_SEGMENTS", ident) or 0),
                "TRANSLATION_CONTEXT_SEGMENTS" in ident.locked),
        },
        # Live dictation's final decode pins condition_on_previous_text (a
        # client override is ignored, streaming/routes.py) and defaults best_of
        # to its own value (a client override still wins).
        "streaming": {
            "condition_on_previous_text": {
                "final": bool(effective_config.cfg_for(model_name,
                                      "STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT", ident)),
                "partial": bool(effective_config.cfg_for(model_name,
                                        "STREAMING_PARTIAL_CONDITION_ON_PREVIOUS_TEXT", ident)),
                "pinned": True,
            },
            "best_of": {"value": int(effective_config.cfg_for(model_name, "STREAMING_FINAL_BEST_OF", ident))},
        },
    }


def _reset_for_tests() -> None:
    """Forget the cached llama-cpp-python version, so a test that fakes the
    distribution lookup sees its own answer."""
    global _LLAMA_CPP_VERSION
    _LLAMA_CPP_VERSION = _LLAMA_CPP_VERSION_UNSET
