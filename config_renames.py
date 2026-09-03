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
    # Seconds are _S everywhere (_MS, _MIN, _H are the other unit symbols);
    # _SEC and _SECONDS were two more spellings of the same unit.
    # CAPTURE_RECORDINGS_* was the only singular prefix in the captures
    # subsystem; every sibling says CAPTURES_.
    "SEGMENT_MAX_WORDS_PER_SEC": "SEGMENT_MAX_WORDS_PER_S",
    "STREAMING_IDLE_TIMEOUT_SEC": "STREAMING_IDLE_TIMEOUT_S",
    "URL_MEDIA_TTL_SEC": "URL_MEDIA_TTL_S",
    "URL_SOCKET_TIMEOUT_SEC": "URL_SOCKET_TIMEOUT_S",
    "SESSION_TTL_SECONDS": "SESSION_TTL_S",
    "URL_PREVIEW_TIMEOUT_SEC": "URL_PREVIEW_TIMEOUT_S",
    "URL_MAX_DURATION_SEC": "URL_MAX_DURATION_S",
    "URL_DOWNLOAD_TIMEOUT_SEC": "URL_DOWNLOAD_TIMEOUT_S",
    "STREAMING_WS_PING_TIMEOUT_SEC": "STREAMING_WS_PING_TIMEOUT_S",
    "STREAMING_WS_PING_INTERVAL_SEC": "STREAMING_WS_PING_INTERVAL_S",
    "STREAMING_FORCED_COMMIT_SEC": "STREAMING_FORCED_COMMIT_S",
    "STREAMING_BUFFER_TRIM_SEC": "STREAMING_BUFFER_TRIM_S",
    "STREAMING_BUFFER_TRIM_KEEP_SEC": "STREAMING_BUFFER_TRIM_KEEP_S",
    "STREAMING_MAX_BUFFER_SEC": "STREAMING_MAX_BUFFER_S",
    "CAPTURE_RECORDINGS_ENABLED": "CAPTURES_RECORDING_ENABLED",
    "CAPTURE_RECORDINGS_MIN_DURATION_SEC": "CAPTURES_RECORDING_MIN_DURATION_S",
    "CAPTURE_RECORDINGS_MAX_DURATION_SEC": "CAPTURES_RECORDING_MAX_DURATION_S",
    "CAPTURE_RECORDINGS_SAMPLE_RATE": "CAPTURES_RECORDING_SAMPLE_RATE",
    "CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT": "CAPTURES_RECORDING_AUDIO_BYTES_HARD_LIMIT",
    # The machine-load table moved out of the recent-transcriptions DB into
    # its own system_metrics store, and its keys say what they sample.
    "STATS_HISTORY_SAMPLE_S": "STATS_SYSTEM_METRICS_SAMPLE_S",
    "STATS_HISTORY_RETENTION_DAYS": "STATS_SYSTEM_METRICS_RETENTION_DAYS",
    "STATS_OWN_SHOWS_MACHINE": "STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS",
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
