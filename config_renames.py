"""Config keys that were renamed, old spelling -> current spelling.

One table serves both entry points so a deployment from before a rename
keeps working unchanged:

* env vars — ``WHISPER_<old>`` (and ``WHISPER_<old>_FILE``) are copied onto
  the new name at config import unless the new name is already set
  (``alias_env``);
* ``config.local.json`` — a stored old key is moved to the new key before
  validation (``migrate_keys``). AdminConfig forbids unknown keys and a
  validation failure drops ALL overrides, so without this one stale key
  would silently lose every setting.

Dependency-free on purpose: config.py imports it before pydantic exists.
"""
from __future__ import annotations

from typing import Any, MutableMapping

RENAMED_KEYS: dict[str, str] = {
    # HF_TOKEN matches the env var huggingface_hub itself reads.
    "USE_AUTH_TOKEN": "HF_TOKEN",
    # Every other rolling store says RETENTION_DAYS.
    "RECENT_TRANSCRIPTIONS_TTL_DAYS": "RECENT_TRANSCRIPTIONS_RETENTION_DAYS",
}

ENV_PREFIX = "WHISPER_"


def alias_env(environ: MutableMapping[str, str]) -> list[str]:
    """Copy each set ``WHISPER_<old>[_FILE]`` onto ``WHISPER_<new>[_FILE]``
    when the new name is unset. A set new-name value always wins. Returns
    one warning line per alias applied, for the startup log."""
    warnings: list[str] = []
    for old, new in RENAMED_KEYS.items():
        for sfx in ("", "_FILE"):
            o, n = ENV_PREFIX + old + sfx, ENV_PREFIX + new + sfx
            if environ.get(o) and not environ.get(n):
                environ[n] = environ[o]
                warnings.append(f"{o} was renamed to {n}; the old name still "
                                f"works but will be removed in a later release.")
    return warnings


def migrate_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Move every renamed key in a stored overrides dict to its new name
    (in place; also returned). An already-present new key wins."""
    for old, new in RENAMED_KEYS.items():
        if old in raw:
            raw.setdefault(new, raw[old])
            del raw[old]
    return raw
