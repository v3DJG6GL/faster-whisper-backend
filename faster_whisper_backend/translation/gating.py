"""Admission rule for translation model refs: the configured default
(_translation_default_model) and the one allowlist check every translation
entry point shares (_translation_model_allowed) — the batch stage, the
stage-ahead plan, /v1/text/translations, the startup / on-demand preload and
the admin prompt lab. Stateless; imports only config.
"""
import re

from faster_whisper_backend.settings import config as cfg


# Mirrors settings_schema._TRANSLATION_MODEL_REF_PATTERN — org/repo[:quant].
_TRANSLATION_REF_RE = re.compile(
    r"\A[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9_.\-]+(:[A-Za-z0-9_.\-]+)?\Z")


def _translation_default_model() -> str:
    """The configured server-wide default translation model ref ("" unset)."""
    return (getattr(cfg, "TRANSLATION_DEFAULT_MODEL", "") or "").strip()


def _translation_model_allowed(ref: str,
                               requested: "str | None" = None,
                               inherited: "str | None" = None) -> bool:
    """The one admission rule the startup preload, the batch stage, the
    stage-ahead plan and /v1/text/translations all share: a non-empty
    TRANSLATION_ALLOWED_MODELS admits its members plus the configured
    default, while an EMPTY allowlist admits any well-formed ref (an opt-in
    allowlist — deliberately laxer than the diarization/separation gates,
    which admit allowlist ∪ {default} only). Like those gates, the allowlist
    constrains only the CLIENT-requested value: pass `requested` (the raw
    client ref, or None when the client sent none) and a config/identity-
    inherited `ref` is admin policy and always passes. `inherited` is the
    config/identity-effective TRANSLATION_MODEL: a client that merely ECHOES
    it (or is locked to it) has not chosen anything, so it passes exactly
    like a request that sent no model — the diarization/separation gates
    admit the effective value the same way."""
    allowed = getattr(cfg, "TRANSLATION_ALLOWED_MODELS", set()) or set()
    if inherited and ref == inherited:
        return True
    # "Any well-formed ref" — the client value reaches hf_hub_download /
    # Llama.from_pretrained as a repo id, so shape-check it the way the
    # whisper path does; an inherited ref is admin policy (config_store
    # already validates it at save time).
    if (requested is not None and ref == requested
            and (len(ref) > 160 or ".." in ref
                 or not _TRANSLATION_REF_RE.match(ref))):
        return False
    if not allowed or ref in allowed or ref == _translation_default_model():
        return True
    return requested is None or ref != requested
