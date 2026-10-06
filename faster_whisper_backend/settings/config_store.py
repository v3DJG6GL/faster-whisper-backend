"""
Persistence layer for the admin WebUI.

Stores user-edited overrides in config.local.json inside the data dir
(default /data, on Windows the repo's data dir; WHISPER_DATA_DIR moves the dir,
WHISPER_CONFIG_LOCAL moves just this file). The file is loaded by config.py
BETWEEN the in-file defaults and the env-var override block, so precedence
stays:  ENV  >  config.local.json  >  config.py defaults.

Validation uses the Pydantic schema in settings/schema.py (AdminConfig and the
override models). Atomic writes and the per-path save lock come from
core/atomic_json.py; the config version counter lives in settings/version.py.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from faster_whisper_backend.settings import config_renames as _renames
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.settings import version as settings_version
from faster_whisper_backend.core import atomic_json
from faster_whisper_backend.paths import REPO_ROOT

from pydantic import ValidationError


_REPO_DIR = REPO_ROOT  # the checkout, not this package — see paths.py
# Admin-edited overrides file. Defaults into the data dir like every other
# runtime-state path (mirrors config._DATA_DIR — computed locally, the rule is
# two lines): WHISPER_CONFIG_LOCAL > WHISPER_DATA_DIR/config.local.json >
# /data/config.local.json (Windows: <repo>\data — bare metal by definition, and
# "/data" would be drive-relative there; keep in sync with config._DATA_DIR).
OVERRIDES_PATH = os.environ.get("WHISPER_CONFIG_LOCAL") or os.path.normpath(
    os.path.join(
        (os.environ.get("WHISPER_DATA_DIR") or "").strip()
        or (os.path.join(_REPO_DIR, "data") if os.name == "nt" else "/data"),
        "config.local.json"))
# Committed factory-default pipeline rules. Unlike config.local.json this file
# IS version-controlled — the admin WebUI's "Defaults" mode edits it so rule
# fixes can be git-pushed to every deployment. See load_factory_rules().
FACTORY_PATH = os.path.join(_REPO_DIR, "config.json")


def migrate_tightened_bundle(bundle: dict[str, Any], label: str) -> list[str]:
    """Rewrite one OverrideProfile-shaped bundle's values that the validators
    started refusing after it was stored (in place) — see
    _migrate_tightened_bundle_values for the rules. Returns one note per
    change, prefixed with `label`; the caller decides where they go (stderr at
    config load, _ENV_WARNINGS for an env bundle, nowhere for a per-identity
    binding re-parsed on every decode)."""
    notes: list[str] = []
    for guard in ("SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS",
                  "SEGMENT_HEAD_ECHO_MIN_WORDS"):
        if bundle.get(guard) == 1 and not isinstance(bundle[guard], bool):
            bundle[guard] = 0
            notes.append(f"{label}.{guard}=1 is no longer accepted; stored "
                         f"as 0 (off, as 1 already behaved)")
    for key, check in (("TEMPERATURE", settings_schema._temperature_csv),
                       ("SUPPRESS_TOKENS", settings_schema._suppress_tokens_csv),
                       ("DEFAULT_LANGUAGE", settings_schema._language_code)):
        val = bundle.get(key)
        if not isinstance(val, str):
            continue
        try:
            check(val)
        except ValueError as e:
            del bundle[key]
            locks = bundle.get("locks")
            if isinstance(locks, list):
                bundle["locks"] = [lk for lk in locks if lk != key]
            notes.append(f"dropped {label}.{key} {val!r}: {e}")
    return notes


def _migrate_tightened_bundle_values(raw: dict[str, Any]) -> list[str]:
    """Rewrite stored values that the validators started refusing after they
    were stored, mostly inside OVERRIDE_PROFILES / MODEL_OVERRIDES bundles
    (in place). Returns one note per change for the caller's log.

    The bundles gained AdminConfig's value rules late: the two segment guards
    refuse 1, TEMPERATURE / SUPPRESS_TOKENS must parse as CSV. The per-profile
    / per-model grid accepted those values before, and one of them left in a
    stored bundle fails load_overrides' whole-file validation — every admin
    override is then ignored at boot, and every save 422s. A guard at 1 was
    already off at runtime, so it becomes 0 (off); an unparseable CSV is
    dropped (inherit), with its lock. DEFAULT_LANGUAGE (top level too) gained
    a Whisper-code membership check: an unknown code failed every decode
    that relied on it, so it is dropped (inherit). SUPPRESS_TOKENS (top
    level too) gained a token-id range: an id far past the vocab crashed
    every decode, so it is dropped the same way. Per-identity bindings
    (api_keys_store._parse_binding) and the env JSON bundles (config.py) run
    the same per-bundle rules via migrate_tightened_bundle."""
    notes: list[str] = []
    for key, check in (("DEFAULT_LANGUAGE", settings_schema._language_code),
                       ("SUPPRESS_TOKENS", settings_schema._suppress_tokens_csv)):
        val = raw.get(key)
        if isinstance(val, str):
            try:
                check(val)
            except ValueError as e:
                del raw[key]
                notes.append(f"dropped {key} {val!r}: {e}")
    for group_key in ("OVERRIDE_PROFILES", "MODEL_OVERRIDES"):
        group = raw.get(group_key)
        if not isinstance(group, dict):
            continue
        for name, bundle in group.items():
            if isinstance(bundle, dict):
                notes.extend(migrate_tightened_bundle(
                    bundle, f"{group_key}[{name!r}]"))
    return notes


def _migrate_legacy_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """One-time key migration (config_renames.RENAMED_KEYS). AdminConfig forbids
    unknown keys and a validation failure drops ALL overrides, so a stored
    file from before the rename would otherwise silently lose every setting.
    Called on BOTH the load path and the save path's raw re-read — save
    merges the payload atop the raw file, so a surviving legacy key would
    make every write raise ValidationError forever (and no save could ever
    clean the file)."""
    gone = sorted(k for k in _renames.REMOVED_KEYS if k in raw)
    if gone:
        print(f"[config_store] dropped removed keys {gone} — they no longer "
              f"exist and have no successor", file=sys.stderr)
    _renames.migrate_keys(raw)
    _renames.migrate_rule_slugs(raw)
    # A stored PIPELINE_RULES copy still carrying a factory entry's OLD text
    # gets the fixed one (config_renames.UPGRADED_RULE_ENTRIES); an entry the
    # admin edited is left alone.
    upgraded = _renames.upgrade_rule_entries(raw)
    if upgraded:
        print(f"[config_store] upgraded stored factory rule entries {upgraded} "
              f"to the current factory text", file=sys.stderr)
    _renames.migrate_bundle_keys(raw)
    for note in _migrate_tightened_bundle_values(raw):
        print(f"[config_store] {note}", file=sys.stderr)
    # Wildcard-host origins ('https://*.example.com') used to pass the origin
    # validators but never matched anything (both the CORS middleware and the
    # trusted-origin guard compare the Origin header by exact string). They
    # are rejected now; strip the already-inert entries from a stored file so
    # the tightened rule cannot wipe every other override at boot. A bare '*'
    # stays legal for CORS only.
    # The env reader (config.py) strips the same entries from
    # WHISPER_TRUSTED_ORIGINS / WHISPER_CORS_ALLOW_ORIGINS.
    for key in ("TRUSTED_ORIGINS", "CORS_ALLOW_ORIGINS"):
        entries = raw.get(key)
        if not isinstance(entries, list):
            continue
        kept, dropped = _renames.strip_wildcard_origins(key, entries)
        if dropped:
            print(f"[config_store] dropped wildcard {key} entries "
                  f"{dropped} — never matched "
                  f"an Origin header and are no longer accepted", file=sys.stderr)
            if kept:
                raw[key] = kept
            else:
                del raw[key]
    return raw


def load_overrides(path: str = OVERRIDES_PATH) -> dict[str, Any]:
    """Load and validate the overrides file. NEVER raises — returns {} on any
    error (missing file, malformed JSON, validation failure). Logs to stderr
    because the standard logger isn't fully wired at config-import time.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        # ValueError covers json.JSONDecodeError AND UnicodeDecodeError — a
        # hand edit saved as cp1252/latin-1 must not stop config from importing.
        print(f"[config_store] cannot read {path}: {e}", file=sys.stderr)
        return {}
    if not isinstance(raw, dict):
        print(f"[config_store] {path} must contain a JSON object", file=sys.stderr)
        return {}
    raw = _migrate_legacy_keys(raw)
    try:
        validated = settings_schema.AdminConfig.model_validate(raw)
    except ValidationError as e:
        print(f"[config_store] {path} failed validation; ignoring overrides:\n{e}",
              file=sys.stderr)
        return {}
    out = validated.model_dump(exclude_none=True)
    for key, coerce in settings_schema._POST_LOAD_COERCERS.items():
        if key in out:
            out[key] = coerce(out[key])
    return out

def load_factory_rules(path: str = FACTORY_PATH) -> list[dict[str, Any]]:
    """Load and validate the committed factory pipeline rules from config.json.

    Unlike load_overrides(), this RAISES on any problem — config.json is a
    required, committed file and the pipeline has no rules without it. The
    caller surfaces the failure as a fatal startup error.

    Returns the validated PIPELINE_RULES list (list of plain dicts).
    """
    if not os.path.exists(path):
        raise RuntimeError(
            f"factory rules file not found: {path} — it is required and "
            f"committed to the repository. Restore it with "
            f"'git checkout config.json'."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:   # ValueError: JSON or UTF-8 decode
        raise RuntimeError(f"cannot read factory rules file {path}: {e}") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("PIPELINE_RULES"), list):
        raise RuntimeError(
            f"{path} must be a JSON object with a 'PIPELINE_RULES' key"
        )
    try:
        validated = settings_schema.AdminConfig.model_validate({"PIPELINE_RULES": raw["PIPELINE_RULES"]})
    except ValidationError as e:
        raise RuntimeError(f"{path} failed validation:\n{e}") from e
    return validated.model_dump(exclude_none=True, mode="json")["PIPELINE_RULES"]


def save_factory_rules(rules: list[Any], path: str = FACTORY_PATH, *,
                       overrides_path: str | None = None) -> list[dict[str, Any]]:
    """Validate `rules` and atomically write them to config.json.

    The WebUI's "Defaults" mode always sends the FULL rule list (not a dirty
    diff like save_overrides()), so this replaces the entire PIPELINE_RULES
    array wholesale — but it read-modify-writes config.json (see below) to
    preserve the sibling scalar factory defaults that now also live there.

    Every rule is normalised to `seeded=True` — a rule living in the committed
    factory file IS a factory default by definition; this keeps the editor's
    seeded/custom distinction consistent and prevents a promoted local rule
    from landing in config.json marked `seeded:false`.

    When these rules are the ones in force (no local or env PIPELINE_RULES
    copy), a rule still named by a stored MODEL_OVERRIDES / OVERRIDE_PROFILES
    / CAPTURES_PIPELINE_RULES_EXCLUDE (in `overrides_path`, default
    OVERRIDES_PATH) cannot be removed or renamed: the dangling slug would make
    every later save of that key 422, as save_overrides refuses it too.

    Returns the validated, coerced rule list. Raises ValidationError on bad
    input — the route handler converts that to a 422 response.
    """
    rules = [{**{k: v for k, v in r.items() if k != "config_rev"}, "seeded": True}
             for r in rules]
    validated = settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": rules}, context={"guard_regex": True})
    out_rules = validated.model_dump(exclude_none=True, mode="json")["PIPELINE_RULES"]
    if "PIPELINE_RULES" not in env_pinned_fields():
        stored = load_overrides(overrides_path or OVERRIDES_PATH)
        refs = {k: stored[k] for k in ("MODEL_OVERRIDES", "OVERRIDE_PROFILES",
                                       "CAPTURES_PIPELINE_RULES_EXCLUDE")
                if k in stored}
        if refs and "PIPELINE_RULES" not in stored:
            settings_schema.AdminConfig.model_validate(
                refs, context={"canonical_slugs": frozenset(
                    r["name"] for r in out_rules)})
    # config.json now holds ALL factory defaults, not just PIPELINE_RULES, so
    # read-modify-write to preserve the sibling scalar keys. A blind whole-file
    # replace (as before) would wipe every other default on a rules promote.
    # The read-through-write runs under atomic_json.save_lock: an unlocked RMW here loses
    # a concurrent save's scalar edits the same way save_overrides() did.
    with atomic_json.save_lock(path):
        merged: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    raw = json.load(f)
                except ValueError:   # malformed JSON or not UTF-8
                    raw = None
            if isinstance(raw, dict):
                merged = raw
        merged.setdefault("schema_version", 1)
        merged["PIPELINE_RULES"] = out_rules
        atomic_json.atomic_write_json(merged, path, sort_keys=False, tmp_prefix=".config.")
    return out_rules


def _strip_env_model_overrides(submitted: dict[str, Any],
                               stored: Any) -> dict[str, Any]:
    """Keep WHISPER_MODEL_OVERRIDE__<id>__<FIELD> values out of a saved
    MODEL_OVERRIDES. The per-model editor round-trips the live dict, which
    carries the env layer; persisting it would leave each env value in force
    from config.local.json after the operator unsets the var. A submitted
    value still equal to the live env value is put back to the stored one
    (or dropped when the file had none); a different value is the admin's
    edit and is kept. Returns a new dict; `submitted` is not modified."""
    from faster_whisper_backend.settings import config as _cfg  # deferred — config imports this module at import
    env_vals = getattr(_cfg, "_ENV_OVERRIDE_VALUES", None) or {}
    if not env_vals:
        return submitted
    stored = stored if isinstance(stored, dict) else {}
    out = dict(submitted)
    for mid, fields in env_vals.items():
        entry = out.get(mid)
        if not isinstance(entry, dict):
            continue
        stored_entry = stored.get(mid) if isinstance(stored.get(mid), dict) else {}
        entry = dict(entry)
        for field, env_value in fields.items():
            if field in entry and entry[field] == env_value:
                if field in stored_entry:
                    entry[field] = stored_entry[field]
                else:
                    del entry[field]
        if entry or mid in stored:
            out[mid] = entry
        else:
            del out[mid]
    return out


def with_env_model_overrides(model_overrides: Any) -> Any:
    """`model_overrides` (a MODEL_OVERRIDES dict as loaded from the file) with
    the live WHISPER_MODEL_OVERRIDE__ values laid on top — the inverse of
    _strip_env_model_overrides, for the hot-apply path that reloads the file
    into the running cfg. A non-dict passes through."""
    from faster_whisper_backend.settings import config as _cfg  # deferred — config imports this module at import
    env_vals = getattr(_cfg, "_ENV_OVERRIDE_VALUES", None) or {}
    if not env_vals or not isinstance(model_overrides, dict):
        return model_overrides
    out = {mid: dict(e) if isinstance(e, dict) else e
           for mid, e in model_overrides.items()}
    for mid, fields in env_vals.items():
        entry = out.setdefault(mid, {})
        if isinstance(entry, dict):
            entry.update(fields)
    return out


def save_overrides(
    payload: dict[str, Any],
    path: str = OVERRIDES_PATH,
    *,
    guard_slugs: "frozenset[str] | set[str] | None" = None,
) -> dict[str, Any]:
    """Validate `payload` against AdminConfig and atomically write it to disk.

    `guard_slugs` scopes the out-of-process regex guard to the named pipeline
    rules (see _validate_pipeline_rules). The quick-config patch path passes
    the slugs the patch actually changed; admin full saves omit it and keep
    guarding the whole list. Compile + template validation always covers
    everything regardless.

    `payload` may contain ONLY the fields the user just edited — the WebUI
    sends a "dirty" diff, not the full state. We MERGE on top of whatever is
    already in `config.local.json` so partial saves preserve previously-saved
    settings. Without this, saving one field would wipe every other override
    on disk and the next restart would revert those values to the in-repo
    defaults.

    Sentinels in `payload`:
      - any key with value `None`  → REMOVE the override (revert to default)
      - any key absent from payload → KEEP the existing value on disk

    Returns a dict containing ONLY the fields actually changed by THIS call
    (after validation/coercion). The route handler uses this for "what needs
    a restart" / "what needs a cache rebuild" decisions — without this
    distinction, every save would re-flag every previously-saved cold setting
    as "restart required."

    Raises ValidationError on bad input — the route handler converts to a 422
    JSON response.

    Atomicity: write to a tempfile in the same directory, then os.replace. On
    Windows AV scanners can briefly hold the destination open; we retry the
    rename a few times with a short backoff.
    """
    # Read existing file (raw, no Pydantic) so we don't lose fields the caller
    # didn't include in `payload`. load_overrides() applies coercions that
    # don't round-trip through model_validate cleanly (set, frozenset, tuple),
    # so we read raw JSON here.
    # The read → validate → write sequence is a read-modify-write that rewrites
    # the WHOLE merged document, and the guard_regex probe below runs out of
    # process for seconds in the middle of it. atomic_json.save_lock() serialises it across
    # threads and workers; unlocked, a save that lands inside another save's
    # window is silently reverted (see atomic_json.save_lock's docstring).
    with atomic_json.save_lock(path):
        existing: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    raw = json.load(f)
                except ValueError:   # malformed JSON or not UTF-8
                    raw = None
            if isinstance(raw, dict):
                existing = _migrate_legacy_keys(raw)

        if isinstance(payload.get("MODEL_OVERRIDES"), dict):
            payload = {**payload, "MODEL_OVERRIDES": _strip_env_model_overrides(
                payload["MODEL_OVERRIDES"], existing.get("MODEL_OVERRIDES"))}

        # Merge: payload wins over existing. None means "remove this override."
        merged = dict(existing)
        for k, v in payload.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v

        # guard_regex: run the out-of-process catastrophic-backtracking check —
        # this is a SAVE of (possibly non-admin) submitted rules.
        # Screen regex rules only when this save actually submits rules —
        # otherwise one stored legacy pattern that fails today's structural
        # screen would brick every unrelated settings save.
        context: dict[str, Any] = {"guard_regex": True} if "PIPELINE_RULES" in payload else {}
        if guard_slugs is not None:
            context["guard_slugs"] = frozenset(guard_slugs)
        # Cross-field checks must see the env-pinned siblings that will be in
        # force, as the restart-time env pass does (see schema._effective).
        env_effective = _env_effective_values()
        if env_effective:
            context["env_effective"] = env_effective
        # The local file usually carries no PIPELINE_RULES copy (factory rules
        # live in config.json), so the merged pass would never see the
        # canonical slug list and a typo'd slug would persist silently. Scoped
        # to saves that touch a slug-bearing key so a rule renamed in
        # config.json cannot brick unrelated settings saves.
        if "PIPELINE_RULES" not in merged and any(
                k in payload for k in ("MODEL_OVERRIDES", "OVERRIDE_PROFILES",
                                       "CAPTURES_PIPELINE_RULES_EXCLUDE")):
            context["canonical_slugs"] = _save_canonical_slugs()
        validated = settings_schema.AdminConfig.model_validate(merged, context=context)
        if env_effective:
            # load_overrides validates the file ON ITS OWN (no env context),
            # at the next boot and in the hot-apply right after this save, and
            # drops EVERY override on a failure. A file that is consistent only
            # thanks to an env pin would be written and then thrown away, so
            # it must also pass the bare check. The regex guard already ran.
            settings_schema.AdminConfig.model_validate(merged, context={
                k: v for k, v in context.items()
                if k not in ("env_effective", "guard_regex", "guard_slugs")})
        to_write = validated.model_dump(exclude_none=True, mode="json")

        atomic_json.atomic_write_json(to_write, path, sort_keys=True, tmp_prefix=".config.local.")
        # Enforce the documented 0600 guarantee rather than inheriting it from
        # mkstemp's default mode on the replaced tempfile.
        from faster_whisper_backend.core import store_common
        store_common.secure_file(path)
    settings_version.bump_config_version()   # let live consumers (streaming idents) re-resolve

    # Return only the fields that actually changed in this call. Compare
    # against `existing` (what was on disk before) using the validated form
    # so type coercions don't show up as spurious diffs.
    changed: dict[str, Any] = {}
    for k in payload:
        new_v = to_write.get(k)             # post-validation value, or None if removed
        old_v = existing.get(k)
        if new_v != old_v:
            changed[k] = new_v
    return changed


def _env_effective_values() -> dict[str, Any]:
    """{field: live value} for every field a WHISPER_* var currently pins —
    the save-time validation context schema._effective prefers over the
    submitted / baseline value. Save-time only: load_overrides validates the
    file without it, so save_overrides also runs that bare check."""
    from faster_whisper_backend.settings import config as _cfg  # deferred — config imports this module at import
    return {f: getattr(_cfg, f) for f in env_pinned_fields() if hasattr(_cfg, f)}


def pipeline_rule_tags(rules: Any) -> list[str]:
    """Return the deduped, sorted union of every tag across the given
    rule list. Used by `/settings/state` + `/settings/api-keys/api/users`
    to populate autocomplete in the tag-picker widget so admins don't
    have to remember the exact spelling.

    Accepts both dicts (post pipeline.apply.canon_rules) and Pydantic models."""
    seen: set[str] = set()
    for r in (rules or []):
        if hasattr(r, "model_dump"):
            r = r.model_dump()
        if not isinstance(r, dict):
            continue
        for t in (r.get("tags") or []):
            if isinstance(t, str) and t:
                seen.add(t)
    return sorted(seen)


def env_pinned_fields() -> dict[str, str]:
    """Return {field_name: env_var_name} for fields currently pinned by env.

    The WebUI uses this to render an 'env-pinned' badge so the admin can see
    that their saved value won't take effect until the env var is unset.

    A field whose env value was REJECTED and reverted by config's validation
    pass is not pinned: the var no longer controls it, and the /settings
    apply path skips pinned names, so a stale badge would also stop an
    admin's edit from ever reaching the live cfg.
    """
    from faster_whisper_backend.settings import config as _cfg  # deferred — config imports this module at import
    _rejected = getattr(_cfg, "_ENV_REJECTED", ())
    # An explicitly EMPTY var pins the fields whose reader treats "" as a
    # value (None / "" / [] / empty set — e.g. WHISPER_ALLOWED_MODELS="" is
    # "any model", WHISPER_DEFAULT_LANGUAGE="" is auto-detect); for every
    # other reader "" means "keep current" and controls nothing.
    _empty_is_value = getattr(_cfg, "_EMPTY_IS_VALUE", ())
    out: dict[str, str] = {}
    for field, env in settings_schema.ENV_VAR_MAPPING.items():
        raw = os.environ.get(env)
        if raw is None or field in _rejected:
            continue
        if raw.strip() or field in _empty_is_value:
            out[field] = env
    return out


# =============================================================================
# Per-identity binding validation (per-user / per-API-key config blobs)
# =============================================================================
# A binding = {"direct": <OverrideProfile-shaped dict>, "profiles": [name, …]}.
# The wire shape sent by the WebUI is {"overrides": {...}, "profiles": [...],
# "locks": [...]}; validate_binding() turns it into the stored shape, applying
# the same OverrideProfile schema (bounds + lock-field check) the profiles use.

def _canonical_rule_slugs() -> set[str]:
    """The set of rule slugs in the live PIPELINE_RULES list (post-load dicts
    or rule objects), for cross-checking per-identity include/exclude."""
    from faster_whisper_backend.settings import config as _cfg
    out: set[str] = set()
    for r in (getattr(_cfg, "PIPELINE_RULES", None) or []):
        name = r.get("name") if isinstance(r, dict) else getattr(r, "name", None)
        if name:
            out.add(name)
    return out


def _save_canonical_slugs() -> set[str]:
    """Slug set for a save that does not carry PIPELINE_RULES: the live list
    (config.json + env + any local copy), falling back to the committed
    factory file; empty set = unknown → validators skip."""
    live = _canonical_rule_slugs()
    if live:
        return live
    try:
        return {r["name"] for r in load_factory_rules()}
    except Exception:
        return set()


def validate_profile_refs(names: Any) -> list[str]:
    """Validate an ORDERED list of profile names referenced by a user/key
    binding. Each must match the profile-name shape AND exist in the current
    OVERRIDE_PROFILES (save rejects dangling references). Order is preserved
    (precedence is positional, earlier-wins); duplicates are dropped. Empty /
    None → []. Raises ValueError on any bad / unknown name."""
    if names is None:
        return []
    if not isinstance(names, list):
        raise ValueError("profiles must be a list of strings")
    from faster_whisper_backend.settings import config as _cfg
    available = set((getattr(_cfg, "OVERRIDE_PROFILES", None) or {}).keys())
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        if not isinstance(n, str):
            raise ValueError(f"profile name must be a string, got {type(n).__name__}")
        nm = n.strip()
        if not nm:
            continue
        if not settings_schema.TAG_RE.match(nm):
            raise ValueError(
                f"invalid profile name {n!r} — lowercase a-z0-9- only, "
                "max 32 chars, no leading hyphen")
        if nm not in available:
            raise ValueError(f"unknown profile {nm!r} — create it first")
        if nm in seen:
            continue
        seen.add(nm)
        out.append(nm)
    return out


# Wildcard sentinel for an override-profile allowlist meaning "all profiles".
# Distinct from any real profile name (ProfileName forbids "*").
ALLOWED_PROFILES_WILDCARD = "*"


def validate_allowed_profiles(raw: Any) -> list[str] | None:
    """Validate a per-identity ALLOWLIST of override-profile names a client may
    REQUEST. This is distinct from `profiles` (which are forced-applied layers);
    the allowlist only restricts which names a per-request `override_profile`
    may select. None = inherit / no restriction (every requestable profile is
    allowed). The wildcard "*" (alone) = explicitly all. An explicit list
    restricts to those names — each must match the profile-name shape AND exist;
    [] = allow none. Raises ValueError on a bad / unknown name."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("allowed_override_profiles must be a list of strings")
    if any(p == ALLOWED_PROFILES_WILDCARD for p in raw):
        return [ALLOWED_PROFILES_WILDCARD]
    return validate_profile_refs(raw)


def validate_binding(raw: Any) -> dict[str, Any]:
    """Validate a per-identity config binding sent by the WebUI and return the
    stored shape {"direct": {...}, "profiles": [...], + optional request gates}.

    Input: {"overrides": {field: value, …, PIPELINE_RULES_*: [...]},
            "locks": [field, …], "profiles": [name, …],
            "allow_request_override_profile": bool|None,
            "allow_request_decode_overrides": bool|None,
            "allowed_override_profiles": [name…]|["*"]|[]|None,
            "apply_no_profiles": bool|None}.
    The three request-gate fields are stored only when explicitly set (a bool, or
    a non-None allowlist) — absent = inherit the next scope down → global. They
    can only NARROW the global gates, never widen them (enforced at resolve time).
    `apply_no_profiles` is a different beast — an ADMIN FORCE, not a request gate:
    True suppresses every bound/requested profile for the identity (plain
    defaults); it does NOT inherit and is NOT bound by ALLOW_REQUEST_OVERRIDE_PROFILE.
    Raises ValueError on any invalid field / bound / lock target / unknown
    profile reference / unknown pipeline slug."""
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("config.overrides must be a JSON object")
    locks = raw.get("locks") or []
    blob = dict(overrides)
    if locks:
        blob["locks"] = locks
    try:
        direct = settings_schema.OverrideProfile.model_validate(blob).model_dump(
            exclude_none=True, mode="json")
    except ValidationError as e:
        raise ValueError("; ".join(
            f"{x['loc']}: {x['msg']}" for x in settings_schema.format_validation_errors(e)
        )) from e
    # `requestable` is profile-library metadata, meaningless on an inline direct
    # override blob — never persist it inside a binding.
    direct.pop("requestable", None)
    canonical = _canonical_rule_slugs()
    if canonical:
        for list_name in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE"):
            unknown = [s for s in (direct.get(list_name) or []) if s not in canonical]
            if unknown:
                raise ValueError(
                    f"{list_name} references unknown rule slugs: {unknown}")
    out: dict[str, Any] = {
        "direct": direct,
        "profiles": validate_profile_refs(raw.get("profiles")),
    }
    for fld in ("allow_request_override_profile", "allow_request_decode_overrides"):
        v = raw.get(fld)
        if isinstance(v, bool):
            out[fld] = v
        elif v is not None:
            raise ValueError(f"{fld} must be a boolean or null")
    allow = raw.get("allowed_override_profiles")
    if allow is not None:
        out["allowed_override_profiles"] = validate_allowed_profiles(allow)
    # Admin force — NOT a request gate: does not inherit, is not bound by the
    # override-profile gate. True = suppress every bound/requested profile for
    # this identity. Stored only when explicitly set.
    anp = raw.get("apply_no_profiles")
    if isinstance(anp, bool):
        out["apply_no_profiles"] = anp
    elif anp is not None:
        raise ValueError("apply_no_profiles must be a boolean or null")
    return out


def binding_is_empty(binding: Any) -> bool:
    """True if a stored binding carries no direct override, no profiles, no
    request-gate setting, and no apply-no-profiles force — i.e. contributes
    nothing and need not be persisted. A request gate set to False, an explicit
    (even empty) allowlist, or apply_no_profiles set counts as a meaningful
    setting and keeps the binding alive."""
    if not isinstance(binding, dict):
        return True
    return not (
        binding.get("direct")
        or binding.get("profiles")
        or binding.get("allow_request_override_profile") is not None
        or binding.get("allow_request_decode_overrides") is not None
        or binding.get("allowed_override_profiles") is not None
        or binding.get("apply_no_profiles") is not None
    )
