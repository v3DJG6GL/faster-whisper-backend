"""
Persistence layer for the admin WebUI.

Stores user-edited overrides in config.local.json inside the data dir
(default /data, on Windows the repo's data dir; WHISPER_DATA_DIR moves the dir,
WHISPER_CONFIG_LOCAL moves just this file). The file is loaded by config.py
BETWEEN the config.json factory defaults and the env-var override block, so
precedence stays:  ENV  >  config.local.json  >  config.json.

Validation uses the Pydantic schema in settings/schema.py (AdminConfig and the
override models). Atomic writes and the per-path save lock come from
core/atomic_json.py; the config version counter lives in settings/version.py.
"""

from __future__ import annotations

import array
import copy
import functools
import json
import os
import re
import sys
from typing import Any

from faster_whisper_backend.settings import config_renames as _renames
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.settings import version as settings_version
from faster_whisper_backend.core import atomic_json
from faster_whisper_backend.paths import REPO_ROOT, default_data_dir

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError


_REPO_DIR = REPO_ROOT  # the checkout, not this package — see paths.py
# Admin-edited overrides file, in the data dir like every other runtime-state
# path. Windows uses <repo>\data: bare metal by definition, and "/data" would
# be drive-relative there (paths.default_data_dir, shared with config._DATA_DIR).
OVERRIDES_PATH = os.environ.get("WHISPER_CONFIG_LOCAL") or os.path.normpath(
    os.path.join(default_data_dir(), "config.local.json"))
# Committed factory defaults (every setting, PIPELINE_RULES included). Unlike
# config.local.json this file IS version-controlled — the admin WebUI's
# "Defaults" mode edits its rules so rule fixes can be git-pushed to every
# deployment. See load_factory_rules().
FACTORY_PATH = os.path.join(_REPO_DIR, "config.json")


def _migrate_segment_guards(d: dict[str, Any], prefix: str) -> list[str]:
    """Rewrite a segment guard stored at 1 to 0 (in place): the validators
    refuse 1 since it was stored, and 1 already behaved as off. Returns one
    note per change, each key prefixed with `prefix`."""
    notes: list[str] = []
    for guard in ("SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS",
                  "SEGMENT_HEAD_ECHO_MIN_WORDS"):
        if d.get(guard) == 1 and not isinstance(d[guard], bool):
            d[guard] = 0
            notes.append(f"{prefix}{guard}=1 is no longer accepted; stored "
                         f"as 0 (off, as 1 already behaved)")
    return notes


def migrate_tightened_bundle(bundle: dict[str, Any], label: str) -> list[str]:
    """Rewrite one OverrideProfile-shaped bundle's values that the validators
    started refusing after it was stored (in place) — see
    _migrate_tightened_bundle_values for the rules. Returns one note per
    change, prefixed with `label`; the caller decides where they go (stderr at
    config load, _ENV_WARNINGS for an env bundle, nowhere for a per-identity
    binding re-parsed on every decode)."""
    notes = _migrate_segment_guards(bundle, f"{label}.")
    for key, check in (("TEMPERATURE", settings_schema._temperature_csv),
                       ("SUPPRESS_TOKENS", settings_schema._suppress_tokens_csv),
                       ("DEFAULT_LANGUAGE", settings_schema._language_code)):
        val = bundle.get(key)
        if not isinstance(val, str):
            continue
        try:
            check(val)
        except ValueError as e:
            # A lock stays: a value-less lock pins the inherited value, so
            # the admin's "the client may not change this" still holds.
            del bundle[key]
            notes.append(f"dropped {label}.{key} {val!r}: {e}")
    return notes


def _migrate_tightened_bundle_values(raw: dict[str, Any]) -> list[str]:
    """Rewrite stored values that the validators started refusing after they
    were stored (in place). Returns one note per change for the caller's log.

    One such value left in the file fails load_overrides' whole-file
    validation: every admin override is then ignored at boot and every save
    422s. So a segment guard at 1 (already off at runtime) becomes 0; inside
    OVERRIDE_PROFILES / MODEL_OVERRIDES an unparseable TEMPERATURE /
    SUPPRESS_TOKENS CSV is dropped (inherit) and its lock stays; an unknown
    DEFAULT_LANGUAGE code or an out-of-vocab SUPPRESS_TOKENS id (both failed
    every decode) is dropped, top level too; a path setting or HF_TOKEN
    holding a NUL (schema.NUL_FREE_FIELDS) and a whitespace-only
    PIPELINE_RULES callback:map key are dropped. Per-identity bindings
    (api_keys_store._parse_binding) and the env JSON bundles (config.py) run
    the per-bundle rules via migrate_tightened_bundle."""
    notes = _migrate_segment_guards(raw, "")
    for key, check in (("DEFAULT_LANGUAGE", settings_schema._language_code),
                       ("SUPPRESS_TOKENS", settings_schema._suppress_tokens_csv)):
        val = raw.get(key)
        if isinstance(val, str):
            try:
                check(val)
            except ValueError as e:
                del raw[key]
                notes.append(f"dropped {key} {val!r}: {e}")
    for key in settings_schema.NUL_FREE_FIELDS:
        val = raw.get(key)
        if isinstance(val, str) and "\x00" in val:
            del raw[key]
            # The value is not echoed: HF_TOKEN is a secret.
            notes.append(f"dropped {key}: it holds a NUL character, "
                         f"which is no longer accepted")
    for group_key, name, bundle in _renames.iter_bundles(raw):
        notes.extend(migrate_tightened_bundle(bundle, f"{group_key}[{name!r}]"))
    # A callback:map key gained a non-blank rule: a whitespace-only key has
    # no word edge, so it matched every space. Drop it (and its map_meta).
    rules = raw.get("PIPELINE_RULES")
    for r in rules if isinstance(rules, list) else ():
        if not (isinstance(r, dict) and r.get("type") == "callback:map"
                and isinstance(r.get("map"), dict)):
            continue
        blank = [k for k in r["map"] if isinstance(k, str) and not k.strip()]
        for k in blank:
            del r["map"][k]
            if isinstance(r.get("map_meta"), dict):
                r["map_meta"].pop(k, None)
        if blank:
            notes.append(f"dropped whitespace-only map keys {blank!r} from "
                         f"PIPELINE_RULES[{r.get('name')!r}]")
    return notes


def _env_rule_names() -> set[str] | None:
    """The rule names of the WHISPER_PIPELINE_RULES JSON list as config.py
    applies it (renamed slugs mapped, upgraded, validated); None when the var
    is unset or config.py rejects it — the rules in force are then the file's
    and config.json's. config.py applies that list AFTER this file loads, so
    with no local rule copy it, not config.json, decides whether a stored slug
    list still names a card split since (config_renames.rename_slugs)."""
    text = os.environ.get(settings_schema.ENV_VAR_MAPPING["PIPELINE_RULES"])
    if text is None or not text.strip():
        return None
    try:
        # The steps of config.py's JSON env loop, in its order.
        wrapped = _renames.migrate_rule_slugs({"PIPELINE_RULES": json.loads(text)})
        _renames.upgrade_rule_entries(wrapped)
        _migrate_tightened_bundle_values(wrapped)
        rules = settings_schema.AdminConfig.model_validate(
            wrapped).model_dump(exclude_none=True).get("PIPELINE_RULES")
    except Exception:  # noqa: BLE001 — config.py rejects on any failure too
        return None
    return _renames._rule_names(rules)


@functools.cache
def _rule_adapter() -> TypeAdapter:
    return TypeAdapter(settings_schema.PipelineRule)


def _editor_view(rule: Any) -> dict[str, Any] | None:
    """A rule as the /settings pipeline editor compares and fingerprints it
    (settings.html _ruleStrip on the /settings/state copy): the schema dump,
    without the `seeded` flag, the server-owned map_meta and config_rev
    itself. None for a rule the schema refuses."""
    try:
        out = _rule_adapter().validate_python(rule).model_dump(exclude_none=True)
    except (ValidationError, TypeError, ValueError):
        return None
    for k in ("seeded", "map_meta", "config_rev"):
        out.pop(k, None)
    return out


# A JS object lists array-index keys ("0", "17") first, in numeric order, then
# the others in insertion order — the order JSON.stringify writes them in.
_JS_INDEX_KEY = re.compile(r"0|[1-9][0-9]{0,9}")


def _js_json(v: Any) -> str:
    """JSON.stringify(_sortDeep(v)) as settings.html writes it: keys sorted by
    UTF-16 code unit, then array-index keys moved to the front."""
    if isinstance(v, dict):
        keys = sorted(v, key=lambda k: k.encode("utf-16-be", "surrogatepass"))
        idx = sorted((k for k in keys if _JS_INDEX_KEY.fullmatch(k)
                      and int(k) < 2 ** 32 - 1), key=int)
        idx_set = set(idx)
        keys = idx + [k for k in keys if k not in idx_set]
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _js_json(v[k])
                              for k in keys) + "}"
    if isinstance(v, list):
        return "[" + ",".join(_js_json(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


def _editor_rule_hash(view: dict[str, Any]) -> str:
    """settings.html _ruleHash (cyrb53 over the UTF-16 code units of the
    stringified rule, as 14 hex digits) of an _editor_view."""
    units = array.array("H")
    units.frombytes(_js_json(view).encode("utf-16-le", "surrogatepass"))
    if sys.byteorder == "big":
        units.byteswap()
    m = 0xFFFFFFFF
    h1, h2 = 0xDEADBEEF, 0x41C6CE57
    for ch in units:
        h1 = ((h1 ^ ch) * 2654435761) & m
        h2 = ((h2 ^ ch) * 1597334677) & m
    h1 = (((h1 ^ (h1 >> 16)) * 2246822507) & m) ^ (((h2 ^ (h2 >> 13)) * 3266489909) & m)
    h2 = (((h2 ^ (h2 >> 16)) * 2246822507) & m) ^ (((h1 ^ (h1 >> 13)) * 3266489909) & m)
    return format(4294967296 * (2097151 & h2) + h1, "x").zfill(14)


def _resync_untouched_factory_copies(raw: dict[str, Any]) -> list[str]:
    """Replace each stored PIPELINE_RULES rule that is still exactly the
    config.json version it was last in sync with (the editor's sync record:
    its config_rev equals its own fingerprint) by the current config.json
    rule, rev included (in place) — what the editor's "↓ Update" would save.
    upgrade_rule_entries fixes only an old entry's text, so such a copy kept
    the other changes of its rule (a new entry, a new note) out and, its
    content no longer matching its rev, read "edited here · config.json
    changed too". A rule the admin edited no longer matches its rev and is
    left to upgrade_rule_entries. Returns the names replaced."""
    rules = raw.get("PIPELINE_RULES")
    if not isinstance(rules, list) or not any(
            isinstance(r, dict) and r.get("config_rev") for r in rules):
        return []
    try:
        factory = {r["name"]: r for r in load_factory_rules(FACTORY_PATH)}
    except RuntimeError:
        return []
    done: list[str] = []
    for i, r in enumerate(rules):
        if not isinstance(r, dict) or not isinstance(r.get("config_rev"), str):
            continue
        base = factory.get(r.get("name"))
        if base is None or base.get("type") == "terminal":
            continue
        mine, theirs = _editor_view(r), _editor_view(base)
        if (mine is None or theirs is None or mine == theirs
                or _editor_rule_hash(mine) != r["config_rev"]):
            continue
        rules[i] = {**copy.deepcopy(base), "config_rev": _editor_rule_hash(theirs)}
        done.append(base["name"])
    return done


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
    # A local rule copy decides on its own: the file is validated against it.
    known = (None if isinstance(raw.get("PIPELINE_RULES"), list)
             else _env_rule_names())
    _renames.migrate_rule_slugs(raw, known)
    resynced = _resync_untouched_factory_copies(raw)
    if resynced:
        print(f"[config_store] updated stored rules {resynced} you never edited "
              f"to the current config.json version", file=sys.stderr)
    upgraded = _renames.upgrade_rule_entries(raw)
    if upgraded:
        print(f"[config_store] upgraded stored factory rule entries {upgraded} "
              f"to the current factory text", file=sys.stderr)
    _renames.migrate_bundle_keys(raw, known)
    for note in _migrate_tightened_bundle_values(raw):
        print(f"[config_store] {note}", file=sys.stderr)
    # Strip the inert wildcard-host entries (config_renames.strip_wildcard_origins)
    # so their refusal cannot wipe every other override at boot.
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
        # utf-8-sig: a file saved "UTF-8 with BOM" (Notepad, PowerShell 5.1
        # Set-Content -Encoding UTF8) is still readable; BOM-less reads the same.
        with open(path, "r", encoding="utf-8-sig") as f:
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
    required, committed file and the pipeline has no rules without it.

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
    array wholesale; the sibling scalar factory defaults in config.json stay.

    Every rule is normalised to `seeded=True` — a rule living in the committed
    factory file IS a factory default by definition; this keeps the editor's
    seeded/custom distinction consistent and prevents a promoted local rule
    from landing in config.json marked `seeded:false`.

    When these rules are the ones in force (no local or env PIPELINE_RULES
    copy), a rule still named by a stored MODEL_OVERRIDES / OVERRIDE_PROFILES
    / CAPTURES_PIPELINE_RULES_EXCLUDE (in `overrides_path`, default
    OVERRIDES_PATH) cannot be removed or renamed: the pages draw rows only for
    live rules, so the dangling slug could be neither seen nor cleared.

    Returns the validated, coerced rule list. Raises ValidationError on bad
    input or an unreadable config.json — the route handler converts that to
    a 422 response.
    """
    rules = [{**{k: v for k, v in r.items() if k != "config_rev"}, "seeded": True}
             for r in rules]
    validated = settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": rules}, context={"guard_regex": True})
    out_rules = validated.model_dump(exclude_none=True, mode="json")["PIPELINE_RULES"]
    if "PIPELINE_RULES" not in env_pinned_fields():
        stored = load_overrides(overrides_path or OVERRIDES_PATH)
        refs = {k: stored[k] for k in _SLUG_REF_KEYS if k in stored}
        # Refuse only the slugs THIS save removes from the factory list: a ref
        # that already dangles (a local-copy rule, a git-pulled rename) is not
        # this edit's doing and must not brick every Defaults save.
        out_names = {r["name"] for r in out_rules}
        removed = _factory_slugs(path) - out_names
        if refs and removed and "PIPELINE_RULES" not in stored:
            settings_schema.AdminConfig.model_validate(
                refs, context={"canonical_slugs": frozenset(
                    out_names | (_stored_ref_slugs(refs) - removed))})
    # Read-modify-write: a whole-file replace would wipe the sibling scalar
    # defaults. Under save_lock, or a concurrent save's scalar edits are lost.
    with atomic_json.save_lock(path):
        merged: dict[str, Any] = {}
        if os.path.exists(path):
            # An unreadable config.json (a conflicted git pull, a half-saved
            # hand edit) is refused, not replaced: rewriting it with only the
            # rules would drop every scalar default and fail the next boot.
            with open(path, "r", encoding="utf-8") as f:
                try:
                    raw = json.load(f)
                except ValueError as e:   # malformed JSON or not UTF-8
                    raise _unreadable_overrides_error(
                        path, str(e), "fix it (git checkout config.json)") from e
            if not isinstance(raw, dict):
                raise _unreadable_overrides_error(
                    path, f"top level is {type(raw).__name__}, not an object",
                    "fix it (git checkout config.json)")
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


# A save touching any of these runs schema's __Host-/__Secure- cookie check.
_COOKIE_PREFIX_KEYS = ("SESSION_COOKIE_NAME", "SESSION_CSRF_COOKIE_NAME",
                       "SESSION_COOKIE_SECURE")
# A save touching any of these runs schema's live-dictation silence-pair check.
_SILENCE_PAIR_KEYS = ("STREAMING_VAD_INNER_SILENCE_MS",
                      "STREAMING_VAD_OUTER_SILENCE_MS",
                      "STREAMING_HARD_BREAK_SILENCE_MS")


def _unreadable_overrides_error(path: str, why: str,
                                remedy: str = "fix or remove it") -> ValidationError:
    """The 422 a save over an unparseable config.local.json or config.json
    fails with: merging onto {} and rewriting would silently wipe every
    hand-edited value that fixing the encoding or a stray comma would have
    recovered."""
    return ValidationError.from_exception_data("AdminConfig", [{
        "type": PydanticCustomError(
            "config_file_unreadable",
            "{file} cannot be read ({why}); {remedy}, then save again",
            {"file": os.path.basename(path), "why": why, "remedy": remedy}),
        "loc": (), "input": None}])


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

    `payload` may hold ONLY the fields the user just edited (the WebUI sends
    a dirty diff); it is merged onto the stored file so a partial save keeps
    every other override.

    Sentinels in `payload`:
      - any key with value `None`  → REMOVE the override (revert to default)
      - any key absent from payload → KEEP the existing value on disk

    Returns a dict containing ONLY the fields actually changed by THIS call
    (after validation/coercion). The route handler uses this for "what needs
    a restart" / "what needs a cache rebuild" decisions — without this
    distinction, every save would re-flag every previously-saved cold setting
    as "restart required."

    Raises ValidationError on bad input — the route handler converts to a 422
    JSON response. An existing file that is not a readable JSON object fails
    the same way and is left untouched; a missing file is the empty case.
    """
    # Raw JSON, not load_overrides(): its coercions (set, frozenset, tuple)
    # don't round-trip through model_validate.
    # This read-modify-write rewrites the WHOLE document while the guard_regex
    # probe runs out of process for seconds; save_lock serialises it across
    # threads and workers, or a save landing in that window is silently reverted.
    with atomic_json.save_lock(path):
        existing: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8-sig") as f:   # as load_overrides
                try:
                    text = f.read()
                    # An empty (touched) file holds nothing a rewrite could lose.
                    raw = json.loads(text) if text.strip() else {}
                except ValueError as e:   # malformed JSON or not UTF-8
                    raise _unreadable_overrides_error(path, str(e)) from e
            if not isinstance(raw, dict):
                raise _unreadable_overrides_error(
                    path, f"top level is {type(raw).__name__}, not an object")
            existing = _migrate_legacy_keys(raw)

        if isinstance(payload.get("MODEL_OVERRIDES"), dict):
            payload = {**payload, "MODEL_OVERRIDES": _strip_env_model_overrides(
                payload["MODEL_OVERRIDES"], existing.get("MODEL_OVERRIDES"))}

        merged = dict(existing)
        for k, v in payload.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
        # Removing the local rules copy makes the factory list the one in
        # force: a slug list naming a card the copy still held under its
        # pre-split slug names that card's successors now (as the next load
        # would map it), so the check below and the written file see those.
        drops_local_rules = (
            "PIPELINE_RULES" in payload and "PIPELINE_RULES" not in merged
            and "PIPELINE_RULES" in existing
            and "PIPELINE_RULES" not in env_pinned_fields())
        if drops_local_rules:
            factory = _factory_slugs() or None
            for k in ("MODEL_OVERRIDES", "OVERRIDE_PROFILES"):
                if k in merged:   # migrated in place; `existing` stays as read
                    merged[k] = copy.deepcopy(merged[k])
            _renames.migrate_rule_slugs(merged, factory)
            _renames.migrate_bundle_keys(merged, factory)

        # The out-of-process regex guard runs only when this save submits
        # rules: one stored legacy pattern that fails today's screen would
        # otherwise brick every unrelated settings save.
        context: dict[str, Any] = {}
        if "PIPELINE_RULES" in payload:
            context["guard_regex"] = True
        if guard_slugs is not None:
            context["guard_slugs"] = frozenset(guard_slugs)
        # Same scoping for the __Host-/__Secure- cookie check: a name stored
        # before that check existed (or hand-edited) must not 422 a save of
        # unrelated keys. A None (revert) still touches the key. Likewise
        # the live-dictation silence pairs, scoped to the touched keys.
        if any(k in payload for k in _COOKIE_PREFIX_KEYS):
            context["cookie_prefix_check"] = True
        silence_keys = frozenset(k for k in _SILENCE_PAIR_KEYS if k in payload)
        if silence_keys:
            context["silence_pair_check"] = silence_keys
        if "TRUSTED_ORIGINS" in payload or "CORS_ALLOW_ORIGINS" in payload:
            context["strict_origins"] = True   # schema._origin_entry
        # A stored or submitted PIPELINE_RULES of [] leaves no valid slug: a
        # save touching the rules or a slug ref checks against it instead of
        # skipping as "unknown" (schema._canonical_slugs; load still skips).
        if any(k in payload for k in ("PIPELINE_RULES", *_SLUG_REF_KEYS)):
            context["empty_rules_known"] = True
        # Cross-field checks must see the env-pinned siblings that will be in
        # force, as the restart-time env pass does (see schema._effective).
        env_effective = _env_effective_values()
        if env_effective:
            context["env_effective"] = env_effective
        validated = settings_schema.AdminConfig.model_validate(merged, context=context)
        # The local file usually carries no PIPELINE_RULES copy (factory rules
        # live in config.json), so the merged pass never sees the canonical
        # slug list and a typo'd slug would persist silently. Checked in a
        # separate pass over ONLY the slug-bearing values this save changes, so
        # a rule renamed or dropped in config.json cannot brick a save of the
        # other keys or entries (the merged pass would check every stored one).
        if "PIPELINE_RULES" not in merged:
            if drops_local_rules:
                # The local copy is being removed: the factory list becomes
                # the one in force, so EVERY stored ref must survive it, or
                # it is left naming a rule the pages never show or clear.
                slug_refs = {k: merged[k] for k in _SLUG_REF_KEYS
                             if merged.get(k) is not None}
                slugs = factory if slug_refs else None
            else:
                slug_refs = _changed_slug_refs(payload, merged, existing)
                slugs = _save_canonical_slugs() if slug_refs else None
            if slugs is not None:
                settings_schema.AdminConfig.model_validate(
                    slug_refs, context={"canonical_slugs": frozenset(slugs)})
        if env_effective:
            # load_overrides validates the file ON ITS OWN (no env context),
            # at the next boot and in the hot-apply right after this save, and
            # drops EVERY override on a failure. A file that is consistent only
            # thanks to an env pin would be written and then thrown away, so
            # it must also pass the bare check. The regex guard already ran,
            # and the cookie-prefix and silence-pair checks are save-only
            # (load never runs them), so re-running them without the env pins
            # would refuse a cookie that IS Secure or a pair that DOES fit.
            settings_schema.AdminConfig.model_validate(merged, context={
                k: v for k, v in context.items()
                if k not in ("env_effective", "guard_regex", "guard_slugs",
                             "cookie_prefix_check", "empty_rules_known",
                             "strict_origins", "silence_pair_check")})
        to_write = validated.model_dump(exclude_none=True, mode="json")

        # Enforce the documented 0600 guarantee rather than inheriting it from
        # mkstemp's default mode on the replaced tempfile.
        atomic_json.atomic_write_private_json(
            to_write, path, sort_keys=True, tmp_prefix=".config.local.")
    settings_version.bump_config_version()   # let live consumers (streaming idents) re-resolve

    # Dropping the local rules copy also rewrote the slug refs migrated above,
    # so they count as changed too: the caller puts them on the running cfg.
    changed: dict[str, Any] = {}
    for k in dict.fromkeys((*payload, *(_SLUG_REF_KEYS if drops_local_rules else ()))):
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
    """Deduped, sorted union of every tag across `rules` (dicts or Pydantic
    models), for the tag-picker autocomplete of /settings/api-keys/api/users."""
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


def _canonical_rule_slugs() -> set[str] | None:
    """The set of rule slugs in the live PIPELINE_RULES list (post-load dicts
    or rule objects), for cross-checking per-identity include/exclude. An
    explicit empty list gives an empty set (no valid slug); None = unknown
    (no list at all), skip."""
    from faster_whisper_backend.settings import config as _cfg
    rules = getattr(_cfg, "PIPELINE_RULES", None)
    if not isinstance(rules, list):
        return None
    out: set[str] = set()
    for r in rules:
        name = r.get("name") if isinstance(r, dict) else getattr(r, "name", None)
        if name:
            out.add(name)
    return out


# Stored keys that name pipeline rules by slug (validated against the
# canonical rule list on save; see save_overrides / save_factory_rules).
_SLUG_REF_KEYS = ("MODEL_OVERRIDES", "OVERRIDE_PROFILES",
                  "CAPTURES_PIPELINE_RULES_EXCLUDE")


def _factory_slugs(path: str | None = None) -> set[str]:
    """Slugs of the committed factory rules; empty set (= unknown, skip)
    when config.json cannot be read."""
    try:
        return {r["name"] for r in load_factory_rules(path or FACTORY_PATH)}
    except Exception:  # noqa: BLE001 — a broken factory file is reported elsewhere
        return set()


def _stored_ref_slugs(refs: dict[str, Any]) -> set[str]:
    """Every rule slug named by the slug-bearing keys of a stored dict."""
    out: set[str] = set(refs.get("CAPTURES_PIPELINE_RULES_EXCLUDE") or [])
    for _group, _name, bundle in _renames.iter_bundles(refs):
        for k in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE"):
            out.update(bundle.get(k) or [])
    return {s for s in out if isinstance(s, str)}


def _changed_slug_refs(payload: dict[str, Any], merged: dict[str, Any],
                       existing: dict[str, Any]) -> dict[str, Any]:
    """The slug-bearing values a save changes, for its slug check: of each
    _SLUG_REF_KEYS key the payload carries, the MODEL_OVERRIDES /
    OVERRIDE_PROFILES entries that differ from the stored ones (the WebUI and
    the profile rename send the whole dict, so one stale stored entry must not
    block an edit of its siblings), and CAPTURES_PIPELINE_RULES_EXCLUDE when
    it differs. Only the slugs a value ADDS are kept for the check, as
    validate_binding does: a slug the stored entry with the same id already
    names (any entry of the stored group, for a renamed id), or the stored
    captures list, is dropped, because the pages draw rows only for live rules
    and send an unseen stale slug back unchanged, so it could not be cleared."""
    out: dict[str, Any] = {}
    for k in _SLUG_REF_KEYS:
        new = merged.get(k)
        if k not in payload or new is None:
            continue
        old = existing.get(k)
        if isinstance(new, dict):
            old = old if isinstance(old, dict) else {}
            entries = {}
            for eid, v in new.items():
                if old.get(eid) == v:
                    continue
                stale = _stored_ref_slugs({k: {eid: old[eid]} if eid in old else old})
                if isinstance(v, dict) and stale:
                    v = {f: ([s for s in x if s not in stale] if (
                        f in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE")
                        and isinstance(x, list)) else x) for f, x in v.items()}
                entries[eid] = v
            if entries:
                out[k] = entries
        elif new != old:
            stale = _stored_ref_slugs({k: old}) if isinstance(old, list) else set()
            out[k] = [s for s in new if s not in stale] if isinstance(new, list) else new
    return out


def _save_canonical_slugs() -> set[str] | None:
    """Slug set for a save that does not carry PIPELINE_RULES: the live list
    (config.json + env + any local copy; an explicit [] = no valid slug),
    falling back to the committed factory file only when the live list is
    unknown; None = unknown → validators skip."""
    live = _canonical_rule_slugs()
    if live is not None:
        return live
    return _factory_slugs() or None


def validate_profile_refs(names: Any, stored: Any = ()) -> list[str]:
    """Validate an ORDERED list of profile names referenced by a user/key
    binding. Each must match the profile-name shape AND exist in the current
    OVERRIDE_PROFILES (save rejects dangling references) — unless `stored`
    (the same list of the binding before this save) already names it: a
    profile that vanished without the delete guard (env pin, hand edit, a
    dropped config.local.json) must not block an unrelated edit, as
    validate_binding's slug check. Order is preserved (precedence is
    positional, earlier-wins); duplicates are dropped. Empty / None → [].
    Raises ValueError on any bad / unknown name."""
    if names is None:
        return []
    if not isinstance(names, list):
        raise ValueError("profiles must be a list of strings")
    from faster_whisper_backend.settings import config as _cfg
    available = set((getattr(_cfg, "OVERRIDE_PROFILES", None) or {}).keys())
    kept = set(stored) if isinstance(stored, list) else set()
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
        if nm not in available and nm not in kept:
            raise ValueError(f"unknown profile {nm!r} — create it first")
        if nm in seen:
            continue
        seen.add(nm)
        out.append(nm)
    return out


# Wildcard sentinel for an override-profile allowlist meaning "all profiles".
# Distinct from any real profile name (ProfileName forbids "*").
ALLOWED_PROFILES_WILDCARD = "*"


def validate_allowed_profiles(raw: Any, stored: Any = ()) -> list[str] | None:
    """Validate a per-identity ALLOWLIST of override-profile names a client may
    REQUEST. This is distinct from `profiles` (which are forced-applied layers);
    the allowlist only restricts which names a per-request `override_profile`
    may select. None = inherit / no restriction (every requestable profile is
    allowed). A "*" anywhere in the list = explicitly all (the other names are
    not checked). An explicit list restricts to those names — each must match
    the profile-name shape AND exist;
    [] = allow none. A name `stored` (the binding's allowlist before this
    save) already holds passes as in validate_profile_refs. Raises
    ValueError on a bad / unknown name."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("allowed_override_profiles must be a list of strings")
    if any(p == ALLOWED_PROFILES_WILDCARD for p in raw):
        return [ALLOWED_PROFILES_WILDCARD]
    return validate_profile_refs(raw, stored)


def validate_binding(raw: Any, previous: Any = None) -> dict[str, Any]:
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
    `previous` is the binding stored before this save (the get_*_config
    shape): a rule slug its direct blob already names is not refused, since
    the drawer sends the whole blob back and a slug a later rules edit
    removed must not block an unrelated edit of the same binding — only the
    slugs this save adds are checked, as save_overrides does. The same holds
    for a profile name its `profiles` / `allowed_override_profiles` lists
    already hold.
    Raises ValueError on any invalid field / bound / lock target / unknown
    profile reference / unknown pipeline slug."""
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    # Default only on None: a falsy wrong type ([] / "" / 0) must reach the
    # type checks, not pass as "empty" and wipe the stored overrides / locks.
    overrides = raw.get("overrides")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("config.overrides must be a JSON object")
    locks = raw.get("locks")
    blob = dict(overrides)
    if locks is not None and locks != []:
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
    if canonical is not None:
        old_direct = previous.get("direct") if isinstance(previous, dict) else None
        if not isinstance(old_direct, dict):
            old_direct = {}
        for list_name in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE"):
            stored = old_direct.get(list_name)
            known = canonical | ({x for x in stored if isinstance(x, str)}
                                 if isinstance(stored, list) else set())
            unknown = [s for s in (direct.get(list_name) or []) if s not in known]
            if unknown:
                raise ValueError(
                    f"{list_name} references unknown rule slugs: {unknown}")
    prev = previous if isinstance(previous, dict) else {}
    out: dict[str, Any] = {
        "direct": direct,
        "profiles": validate_profile_refs(raw.get("profiles"),
                                          prev.get("profiles")),
    }
    for fld in ("allow_request_override_profile", "allow_request_decode_overrides"):
        v = raw.get(fld)
        if isinstance(v, bool):
            out[fld] = v
        elif v is not None:
            raise ValueError(f"{fld} must be a boolean or null")
    allow = raw.get("allowed_override_profiles")
    if allow is not None:
        out["allowed_override_profiles"] = validate_allowed_profiles(
            allow, prev.get("allowed_override_profiles"))
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
