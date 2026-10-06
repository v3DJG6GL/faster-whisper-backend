"""Config keys that were renamed, old spelling -> current spelling.

One table serves both entry points so a deployment from before a rename
keeps working unchanged:

* env vars — ``WHISPER_<old>`` (and ``WHISPER_<old>_FILE``) are copied onto
  the new name at config import unless the new name is already set
  (``alias_env``);
* ``config.local.json`` — a stored old key is moved to the new key before
  validation (``migrate_keys``; a key in ``REMOVED_KEYS`` is dropped
  instead). AdminConfig forbids unknown keys and a
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
    # One size cap for uploads and URL downloads (MEDIA_MAX_BYTES); the
    # retention cap says what it bounds.
    "MAX_UPLOAD_BYTES": "MEDIA_MAX_BYTES",
    "URL_MEDIA_MAX_BYTES": "RETAINED_MEDIA_MAX_BYTES",
}

# Keys that were removed with no successor. A stored one is dropped before
# validation for the same reason a renamed one is moved (an unknown key
# loses every override). URL_MAX_BYTES folded into MEDIA_MAX_BYTES, but its
# shipped value 0 ("inherit the upload cap") is not a legal MEDIA_MAX_BYTES,
# so it cannot be mapped. Not env-aliased: an unknown WHISPER_* var is inert.
REMOVED_KEYS: frozenset[str] = frozenset({"URL_MAX_BYTES"})

# Pipeline rule slugs that were renamed, old -> current. A slug is referenced
# by name from stored data (a local PIPELINE_RULES copy, the exclude / include
# lists of MODEL_OVERRIDES / OVERRIDE_PROFILES / per-identity bindings,
# CAPTURES_PIPELINE_RULES_EXCLUDE), and an unknown slug fails validation —
# which drops ALL local overrides. Language-specific rules carry their
# language prefix (de-, ch-, es-).
RENAMED_RULES: dict[str, str] = {
    "dictation-map": "de-dictation-map",
}

# Factory regex-list entries whose text was CHANGED in config.json, keyed by
# the entry label: (old pattern, old replacement) -> (new pattern, new
# replacement). A saved PIPELINE_RULES copy (config.local.json, or the
# WHISPER_PIPELINE_RULES JSON env var) holds the whole list as it was when the
# admin last saved, so a factory fix never reached it — the stored copy wins
# over config.json entry for entry. upgrade_rule_entries rewrites an entry ONLY
# when both its pattern and its replacement still equal the old factory text
# exactly: an entry the admin edited is theirs and stays untouched.
UPGRADED_RULE_ENTRIES: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    # The old pair matched `"…"` greedily from any quote, so a quote that was
    # still open at the end of one utterance's text was paired differently
    # once the next utterance closed it — already-typed text changed and live
    # dictation wrote the quoted words twice. The new one pairs quotes left to
    # right (an unclosed quote at the end already counts as opening) and may
    # span a line break.
    "tighten-quote-spacing": (
        ('"[ \\t]*([^"\\n]+?)[ \\t]*"', '"\\1"'),
        ('"[ \\t]*([^"]*?)[ \\t]*("|\\Z)', '"\\1\\2'),
    ),
}

# Keys holding a list of rule slugs, at the top level and inside each
# MODEL_OVERRIDES / OVERRIDE_PROFILES bundle.
_SLUG_LIST_KEYS = ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE",
                   "CAPTURES_PIPELINE_RULES_EXCLUDE")

ENV_PREFIX = "WHISPER_"


def rename_slugs(slugs: Any) -> Any:
    """Map renamed rule slugs in a list / set / tuple of slugs (order kept,
    a slug present under both names collapses to one). Other values pass
    through untouched."""
    if not isinstance(slugs, (list, tuple, set, frozenset)):
        return slugs
    out: list[Any] = []
    for s in slugs:
        s = RENAMED_RULES.get(s, s) if isinstance(s, str) else s
        if s not in out:
            out.append(s)
    return type(slugs)(out) if not isinstance(slugs, list) else out


def migrate_rule_slugs(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply RENAMED_RULES to a stored overrides dict (in place; also
    returned): the `name` of each stored PIPELINE_RULES entry and every slug
    list. A stored rule already using the new name wins over the old one."""
    rules = raw.get("PIPELINE_RULES")
    if isinstance(rules, list):
        # str only: a hand-edited non-string name must reach the schema's
        # error, not crash the dict lookups here (load_overrides never raises).
        names = {r.get("name") for r in rules
                 if isinstance(r, dict) and isinstance(r.get("name"), str)}
        for r in rules:
            if (isinstance(r, dict) and isinstance(r.get("name"), str)
                    and r["name"] in RENAMED_RULES):
                new = RENAMED_RULES[r["name"]]
                if new not in names:
                    r["name"] = new
                    names.add(new)
    bundles = [raw]
    for key in ("OVERRIDE_PROFILES", "MODEL_OVERRIDES"):
        group = raw.get(key)
        if isinstance(group, dict):
            bundles.extend(b for b in group.values() if isinstance(b, dict))
    for b in bundles:
        for key in _SLUG_LIST_KEYS:
            if key in b:
                b[key] = rename_slugs(b[key])
    return raw


def upgrade_rule_entries(raw: dict[str, Any]) -> list[str]:
    """Apply UPGRADED_RULE_ENTRIES to the stored PIPELINE_RULES of an
    overrides dict (in place). Returns the labels of the entries rewritten,
    for the caller's log line; empty when nothing matched. Idempotent: an
    upgraded entry no longer equals the old text."""
    rules = raw.get("PIPELINE_RULES")
    done: list[str] = []
    if not isinstance(rules, list):
        return done
    for r in rules:
        if not isinstance(r, dict) or r.get("type") != "regex-list":
            continue
        entries = r.get("entries")
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("label"), str):
                continue
            up = UPGRADED_RULE_ENTRIES.get(e["label"])
            if up is None:
                continue
            (old_pat, old_rep), (new_pat, new_rep) = up
            if e.get("pattern") == old_pat and (e.get("replacement") or "") == old_rep:
                e["pattern"] = new_pat
                e["replacement"] = new_rep
                done.append(f"{r.get('name')}/{e.get('label')}")
    return done


def alias_env(environ: MutableMapping[str, str]) -> list[str]:
    """Copy each set ``WHISPER_<old>[_FILE]`` onto ``WHISPER_<new>[_FILE]``
    when the new name is unset. A set new-name value always wins — in EITHER
    spelling: a stale ``WHISPER_USE_AUTH_TOKEN`` must not be copied onto
    ``WHISPER_HF_TOKEN`` when the operator moved to ``WHISPER_HF_TOKEN_FILE``
    (the plain var would then shadow the file). Returns one warning line per
    alias applied or old name ignored, for the startup log."""
    warnings: list[str] = []
    for old, new in RENAMED_KEYS.items():
        new_set = [ENV_PREFIX + new + sfx for sfx in ("", "_FILE")
                   if ENV_PREFIX + new + sfx in environ]
        for sfx in ("", "_FILE"):
            o, n = ENV_PREFIX + old + sfx, ENV_PREFIX + new + sfx
            if not environ.get(o):
                continue
            if new_set:
                warnings.append(f"{o} is ignored: {new_set[0]} (its new name) "
                                f"is set.")
                continue
            environ[n] = environ[o]
            warnings.append(f"{o} was renamed to {n}; the old name still "
                            f"works but will be removed in a later release.")
    return warnings


def migrate_bundle_keys(raw: dict[str, Any]) -> list[str]:
    """Apply migrate_keys, and the same rename to each ``locks`` entry, inside
    every OVERRIDE_PROFILES / MODEL_OVERRIDES bundle of an overrides dict (in
    place). The bundles are extra="forbid", so one pre-rename key there fails
    validation of the whole value. Returns the old names that were renamed,
    for the caller's log line; empty when nothing matched."""
    renamed: list[str] = []
    for group_key in ("OVERRIDE_PROFILES", "MODEL_OVERRIDES"):
        group = raw.get(group_key)
        if not isinstance(group, dict):
            continue
        for bundle in group.values():
            if not isinstance(bundle, dict):
                continue
            renamed.extend(k for k in RENAMED_KEYS if k in bundle)
            migrate_keys(bundle)
            locks = bundle.get("locks")
            if isinstance(locks, list):
                renamed.extend(lk for lk in locks
                               if isinstance(lk, str) and lk in RENAMED_KEYS)
                bundle["locks"] = [RENAMED_KEYS.get(lk, lk) for lk in locks]
    return sorted(set(renamed))


def migrate_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Move every renamed key in a stored overrides dict to its new name
    (in place; also returned). An already-present new key wins. Keys in
    REMOVED_KEYS are dropped."""
    for old, new in RENAMED_KEYS.items():
        if old in raw:
            raw.setdefault(new, raw[old])
            del raw[old]
    for old in REMOVED_KEYS:
        raw.pop(old, None)
    return raw
