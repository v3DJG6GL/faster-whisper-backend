"""Config hot-apply: push a saved config into the running process.

After /settings/state, /settings/overrides and /quick-config/state (and
PATCH /v1/pipeline-rules) persist a change, apply_hot_changes() copies the
saved values into the running cfg module, rebuilds the pipeline engine's
derived caches, evicts models and stage extras whose load-time settings
changed, and bumps the config version. rules_lock() is the one lock every
PIPELINE_RULES read-modify-write takes, whichever router it comes from.
canon_rules() / resolved_value() give config values the wire form the
admin and quick-config pages compare against.

Imports no routes module, so every router imports it eagerly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Coroutine
from typing import Any, TypeVar

from pydantic import TypeAdapter

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import diarization
from faster_whisper_backend.core import log_setup
from faster_whisper_backend.core.loop_lock import LoopLock
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.settings import version as settings_version
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.translation import engine as translation

logger = logging.getLogger("whisper-api")

_T = TypeVar("_T")

# See rules_lock() for the lost update this closes. Single-worker only: with
# SERVER_WORKERS > 1 the cross-process case still relies on
# atomic_json.save_lock + the per-slug fingerprints.
# A LoopLock (one asyncio.Lock per event loop): an asyncio.Lock binds to the
# first loop that awaits it, and this module (unlike main) is not reloaded
# per test, so a plain Lock would raise "bound to a different event loop"
# on the second TestClient. In production one worker = one loop = one lock.
_RULES_LOCK = LoopLock()

# Serializes apply_hot_changes' reload + setattr (see there). Not rules_lock():
# the PIPELINE_RULES writers already hold that one around the call, and a
# LoopLock is not re-entrant.
_APPLY_LOCK = LoopLock()

# The HF_TOKEN the service environment set (systemd Environment=, a shell
# export), which clearing the config field restores instead of removing.
# main.py promotes cfg.HF_TOKEN over it before this module is imported, so it
# first records the environment's value under SERVICE_HF_TOKEN_ENV ("" when
# none was set); a uvicorn worker inherits that record with the promoted
# environment. Without main (a script, a test) nothing was promoted.
SERVICE_HF_TOKEN_ENV = "_WHISPER_SERVICE_HF_TOKEN"
_ENV_HF_TOKEN: "str | None" = os.environ.get(
    SERVICE_HF_TOKEN_ENV, os.environ.get("HF_TOKEN")) or None


def rules_lock() -> LoopLock:
    """The loop-keyed lock that serializes every PIPELINE_RULES
    read-modify-write: quick_config_routes.apply_rules_patch and the admin
    routes post_state (when it carries PIPELINE_RULES), post_factory_rules
    and clear_local_pipeline_override. apply_rules_patch snapshots
    cfg.PIPELINE_RULES from memory, holds it across the offloaded save, and
    writes the WHOLE key back, so an admin save landing inside that window
    would be persisted, applied, answered 200 — and then silently reverted by
    the stale quick-config document. Both routers queueing on this one lock
    closes the cross-endpoint half of that lost update as well as the
    intra-endpoint half."""
    return _RULES_LOCK


async def run_to_end(coro: Coroutine[Any, Any, _T]) -> _T:
    """Await `coro` as a task a cancelled caller does not stop.

    For a config writer's lock + save + apply: a dropped connection unwinds
    the handler as a cancellation, but the offloaded save keeps writing
    config.local.json on its thread. Cancelled inside the body, the hot-apply
    was skipped (disk new, running cfg old) and the lock was released before
    the apply landed, so a queued /quick-config patch snapshotted the stale
    cfg list and wrote it back over the edit."""
    return await asyncio.shield(asyncio.ensure_future(coro))


# Async dropper per settings_schema.EXTRAS_EVICTION bucket (diarization /
# BGM / translation), awaited when a save touches a bucket field: a load
# parameter or the stage's on/off switch. It only frees the VRAM now instead
# of at the idle timeout, since each module re-keys on its load params per
# request. All three are lazy-import-safe: their heavy optional deps load on
# first pipeline use, not at module import.
EVICTORS: dict[str, Any] = {
    "diarization": diarization.drop_pipeline,
    "bgm": bgm_separation.drop_separator,
    "translation": translation.drop_models,
}

# Discriminated-union adapter for PIPELINE_RULES canonicalization. Built once
# at import time — TypeAdapter construction walks every rule subclass and is
# the dominant cost of canon_rules, called twice per /settings/state request.
_PIPELINE_RULE_ADAPTER: TypeAdapter = TypeAdapter(settings_schema.PipelineRule)


def resolved_value(field: str) -> Any:
    """Read the current effective value of a config field by attribute name."""
    val = getattr(cfg, field, None)
    # Convert un-JSON-able types so the WebUI gets clean data.
    if isinstance(val, (set, frozenset)):
        return sorted(val)
    if isinstance(val, tuple):
        return [list(p) if isinstance(p, tuple) else p for p in val]
    return val


# Pydantic re-validates each rule so model_dump() emits keys in the
# discriminated-union's declaration order — same on both `value` and
# `default_value` so JSON.stringify on each yields identical strings
# when the rule contents match. Without this, _BASELINE keeps source
# order while the resolved value (after a local.json overlay) carries
# Pydantic's parent-first MRO order, and the WebUI's dirty / origin-badge
# comparisons would report a spurious diff on first paint.
def canon_rules(rules: Any) -> Any:
    if not isinstance(rules, list):
        return rules
    out: list[Any] = []
    for r in rules:
        try:
            dumped = _PIPELINE_RULE_ADAPTER.validate_python(r).model_dump(exclude_none=True)
            out.append(_sort_dicts(dumped))
        except Exception:
            out.append(r)  # malformed — pass through; save-time validator catches it
    return out


# Same reason for nested dicts: model_dump() keeps their insertion order (a
# callback:map's `map`), which can differ between value and default_value
# when the contents are equal, a false diff that also makes reset visibly
# re-sort the rows. Alphabetical is safe: the map's match order (longest
# first) is rebuilt server-side from the keys.
def _sort_dicts(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_dicts(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_dicts(x) for x in obj]
    return obj


async def rebuild_caches_off_loop(reason: str) -> None:
    """Recompile the pipeline engine's derived caches, off the event loop.

    Off the loop because rebuild_caches recompiles every rule, and a
    callback:map compiles all its keys into one large pattern (tens to
    hundreds of ms at a few thousand keys). A *changed* map, which is what
    every caller here produces, is never in a cache.

    rebuild_caches serialises itself on engine._REBUILD_LOCK. Each caller's
    cfg setattr lands before its own rebuild starts, so the last rebuild to
    run compiles the latest cfg."""
    try:
        await asyncio.to_thread(pl_engine.rebuild_caches)
        logger.info("[config] rebuilt pipeline caches after %s", reason)
    except Exception as e:
        logger.error("[config] cache rebuild failed after %s: %s", reason, e)


async def apply_hot_changes(
    written: dict[str, Any],
) -> dict[str, Any]:
    """Apply hot edits from a config save to the running cfg module, rebuild
    caches, and evict load-time-affected models.

    Returns a dict suitable to splat into the JSON response envelope:
      hot_applied, cold_pending, env_pinned_ignored, evicted.

    The per-model eviction diff snapshots the running MODEL_OVERRIDES itself,
    right before the setattr. A caller's snapshot would be taken before its
    awaited save, so two overlapping saves both diffed against the same stale
    value and could leave a model loaded under the settings in between.
    """
    # Shielded: every caller awaits this after its save already wrote
    # config.local.json. A cancelled await would skip the setattr loop, the
    # evictions, the HF_TOKEN re-sync and the log-level push, so the running
    # cfg would keep the old values while disk holds the new ones.
    return await run_to_end(_apply_hot_changes(written))


async def _apply_hot_changes(written: dict[str, Any]) -> dict[str, Any]:
    # Re-loaded from disk so the in-memory values get load_overrides'
    # coercions (set/frozenset/tuple). Off the loop: a blocking disk read +
    # full Pydantic pass that the non-admin /quick-config save reaches too.
    # Load + setattr under one lock: two saves that skip rules_lock() (scalar
    # admin saves) could otherwise interleave, and the older save's load could
    # setattr() its stale value last while disk holds the newer one.
    async with _APPLY_LOCK:
        coerced = await asyncio.to_thread(config_store.load_overrides)
        hot_changed: list[str] = []
        cold_changed: list[str] = []
        needs_cache_rebuild = False
        env_pinned = config_store.env_pinned_fields()
        # What the loaded models were built from: taken below with no await before
        # the setattr, so a concurrent save cannot slip in between.
        prev_model_overrides: "dict[str, Any] | None" = None

        for name in written:
            if name in env_pinned:
                # Save persists, but the running cfg won't change until the env
                # var is unset. Don't include in `hot_changed` — nothing changed
                # in memory.
                continue
            if name in coerced:
                new_val = coerced[name]
            else:
                # Override removed (reset to default): revert to the in-repo
                # baseline. getattr(cfg, name) still holds the stale override, so
                # it can't be the fallback; _BASELINE holds the native-typed
                # default (matching load_overrides' coercions).
                baseline = getattr(cfg, "_BASELINE", {}) or {}
                new_val = baseline.get(name, getattr(cfg, name, None))
            if name == "MODEL_OVERRIDES":
                # save_overrides keeps WHISPER_MODEL_OVERRIDE__ env values out of
                # the file; lay them back on the running cfg.
                new_val = config_store.with_env_model_overrides(new_val)
                prev_model_overrides = dict(getattr(cfg, "MODEL_OVERRIDES", None) or {})
            setattr(cfg, name, new_val)
            if name in settings_schema.CACHE_REBUILD_FIELDS:
                needs_cache_rebuild = True
            if name in settings_schema.RESTART_REQUIRED_FIELDS:
                cold_changed.append(name)
            else:
                hot_changed.append(name)

    if needs_cache_rebuild:
        await rebuild_caches_off_loop("admin update")

    # save_overrides already bumped the config version when the FILE was
    # written, but the running cfg only got the new values in the setattr loop
    # above — two awaits later. A streaming session whose _refresh_ident ran
    # in that window stamped the new version while resolving from the OLD cfg
    # and would never re-resolve. Bump again now that cfg is current; consumers
    # only compare for inequality, so the cost is one redundant re-resolve.
    # Here, not at the end: drain_then_evict below waits on the model-load
    # lock, which a concurrent load holds for its whole constructor, and
    # nothing after this point changes what a session resolves from cfg.
    settings_version.bump_config_version()

    # Eviction-on-edit: in-flight transcribes finish on the old WhisperModel
    # instance via Python ref-counting (drain-then-evict).
    evicted: list[str] = []
    # Env-pinned names were skipped above: their running value did not change,
    # so they are no reason to drop a loaded model or extra.
    effective = {n for n in written if n not in env_pinned}
    try:
        load_time_changed_globally = bool(
            effective & settings_schema.LOAD_TIME_FIELDS
        )
        if load_time_changed_globally:
            # Affects every loaded model that doesn't have a per-model
            # override winning over the changed global field. Conservative
            # fallback: evict ALL models. They reload lazily so this is cheap.
            ev = await tx_models.drain_then_evict(None)
            evicted.extend(ev)
        if prev_model_overrides is not None:
            # Per-model override changed for one or more model ids — evict
            # only those whose LOAD-TIME subset (added, changed or removed
            # key) differs between the running value replaced above and the
            # new bundle (None: env-pinned, nothing changed in memory).
            # A removed id whose bundle held only decode-time keys needs no
            # reload, matching the global-field rule.
            new_overrides = getattr(cfg, "MODEL_OVERRIDES", None) or {}
            old_overrides = prev_model_overrides
            lt = settings_schema.LOAD_TIME_FIELDS
            for model_id in set(old_overrides) | set(new_overrides):
                o = old_overrides.get(model_id)
                n = new_overrides.get(model_id)
                o = o if isinstance(o, dict) else {}
                n = n if isinstance(n, dict) else {}
                keys = lt & (set(o) | set(n))
                if keys and any(o.get(k) != n.get(k) for k in keys):
                    ev = await tx_models.drain_then_evict(model_id)
                    evicted.extend(ev)
        if "MAX_LOADED_MODELS" in effective:
            # The cap is enforced on a load miss only: a lowered cap must trim
            # the resident models now, or cache hits keep them all in VRAM.
            evicted.extend(n for n in await tx_models.trim_to_cap()
                           if n not in evicted)
    except Exception as e:
        # Never let eviction failure break the save response. The user's
        # change still persisted; worst case they restart manually.
        logger.error("[config] eviction-on-edit failed: %s", e)

    # Buckets and trigger fields come from settings_schema.EXTRAS_EVICTION
    # (per-field `evict=`; see EVICTORS). A failed drop never breaks the save
    # response.
    for _extra, _extra_fields in settings_schema.EXTRAS_EVICTION.items():
        if not effective & _extra_fields:
            continue
        try:
            await EVICTORS[_extra]()
        except Exception as e:
            logger.error("[config] %s eviction-on-edit failed: %s", _extra, e)

    # Re-sync os.environ["HF_TOKEN"] whenever cfg.HF_TOKEN changed. The
    # token is set process-wide at startup (main.py) so non-WhisperModel HF
    # calls (Silero VAD, tokenizer fetches) inherit it; live edits via the
    # admin UI need to re-set the env var or those callers stay on the old
    # value until next service restart.
    if "HF_TOKEN" in written:
        new_token = getattr(cfg, "HF_TOKEN", None) or ""
        if new_token:
            os.environ["HF_TOKEN"] = new_token
            logger.info("[config] HF_TOKEN env updated from admin edit")
        elif _ENV_HF_TOKEN:
            os.environ["HF_TOKEN"] = _ENV_HF_TOKEN
            logger.info("[config] HF_TOKEN env restored to the service environment's")
        else:
            os.environ.pop("HF_TOKEN", None)
            logger.info("[config] HF_TOKEN env cleared (config field unset)")

    # The console handler's level is read once at import; push the new one.
    # From cfg, not `written`, so an env pin or a reset to baseline wins.
    if "CONSOLE_LOG_LEVEL" in written:
        log_setup.apply_console_log_level(getattr(cfg, "CONSOLE_LOG_LEVEL", "warning"))

    return {
        "hot_applied": hot_changed,
        "cold_pending": cold_changed,
        "env_pinned_ignored": sorted(n for n in written if n in env_pinned),
        "evicted": evicted,
    }
