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
from typing import Any

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

# Serializes quick_config_routes.apply_rules_patch (and, via rules_lock(), the
# admin PIPELINE_RULES writers): it snapshots cfg.PIPELINE_RULES, awaits the
# offloaded save (up to the guard timeout), and writes the WHOLE key back —
# so two overlapping patches would both snapshot the pre-update list and the
# second save would silently revert the first caller's edit. This closes the
# single-worker window only; with SERVER_WORKERS > 1 the cross-process case
# still relies on atomic_json.save_lock + the per-slug fingerprints.
# A LoopLock (one asyncio.Lock per event loop): an asyncio.Lock binds to the
# first loop that awaits it, and this module (unlike main) is not reloaded
# per test, so a plain module-level Lock would raise "bound to a different
# event loop" on the second TestClient. In production one worker = one loop
# = one lock.
_RULES_LOCK = LoopLock()


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


# Async dropper per settings_schema.EXTRAS_EVICTION bucket: when a save touches
# any field in a bucket, apply_hot_changes awaits the matching dropper so the cached
# extra's VRAM frees now instead of at the idle timeout. (Correctness doesn't
# depend on this — both modules re-key on their load params per request.)
# Both modules are lazy-import-safe: their heavy optional deps load on first
# pipeline use, not at module import.
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


# `model_dump()` preserves insertion order on nested dict fields (e.g. the
# `map` on a callback:map rule). The resolved value (after a local.json
# overlay) and the baseline `default_value` (from cfg._BASELINE) can carry
# different insertion orders even when contents are equal — which makes
# JSON.stringify(value) !== JSON.stringify(default_value) so the WebUI's
# dirty / origin-badge checks falsely report a diff, AND clicking reset
# visibly re-sorts the rows. Recursively sorting nested dict keys (applied
# identically to value AND default_value) makes the equality check reliable.
# Forced alphabetical is the right canonical order for `cb:map` rules: the
# longest-first word-bounded regex is rebuilt server-side from these keys,
# so display order has no functional meaning.
def _sort_dicts(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_dicts(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_dicts(x) for x in obj]
    return obj


async def rebuild_caches_off_loop(reason: str) -> None:
    """Recompile the pipeline engine's derived caches, off the event loop.

    Off the loop because rebuild_caches recompiles every rule; a
    callback:map builds a \\b(alt|alt|...)\\b alternation and
    re.compile()s it — measured at 68 ms for 500 random 13-char keys, 116 ms
    at 1000, 250 ms at 2000, and a *changed* map (exactly what every caller
    here produces) always misses re._cache.

    rebuild_caches mutates module globals, so this is only safe under a
    single-writer assumption. That assumption already had to hold:
    save_overrides — the far heavier writer one frame up the same call
    chain — has been offloaded to a worker thread for a while, so the write
    side of this path was already running off the loop. Moving the rebuild
    alongside it does not widen the window; both are awaited in sequence, so
    no two rebuilds from a single request's chain can overlap, and
    concurrent requests were already able to interleave at the
    save_overrides await."""
    try:
        await asyncio.to_thread(pl_engine.rebuild_caches)
        logger.info("[config] rebuilt pipeline caches after %s", reason)
    except Exception as e:
        logger.error("[config] cache rebuild failed after %s: %s", reason, e)


async def apply_hot_changes(
    written: dict[str, Any],
    prev_model_overrides: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    """Apply hot edits from a config save to the running cfg module, rebuild
    caches, and evict load-time-affected models.

    Shared by /settings/state and /settings/overrides (admin) and
    /quick-config/state (end-user). For
    /quick-config only PIPELINE_RULES can change, so most branches here
    simply skip — but the helper handles all cases uniformly so the two
    paths stay in lockstep.

    Returns a dict suitable to splat into the JSON response envelope:
      hot_applied, cold_pending, env_pinned_ignored, evicted.
    """
    # Apply hot edits to the running cfg module so the next request sees them.
    # We re-load from disk so the in-memory values get the same coercions
    # (set/frozenset/tuple) load_overrides applies.
    # Off the event loop: this helper is NOT admin-only in practice — POST
    # /quick-config/state carries only require_user_webui_host +
    # require_page("quick_config") and awaits apply_rules_patch, which awaits
    # us. load_overrides is a blocking disk read + full Pydantic pass.
    coerced = await asyncio.to_thread(config_store.load_overrides)
    hot_changed: list[str] = []
    cold_changed: list[str] = []
    needs_cache_rebuild = False
    env_pinned = config_store.env_pinned_fields()

    for name in written:
        if name in env_pinned:
            # Save persists, but the running cfg won't change until the env
            # var is unset. Don't include in `hot_changed` — nothing changed
            # in memory.
            continue
        if name in coerced:
            new_val = coerced[name]
        else:
            # The override was removed (reset to default): revert the running
            # cfg to the in-repo baseline. getattr(cfg, name) would still hold
            # the stale override value (set when it was first saved) until we
            # overwrite it here, so it can't be the fallback. _BASELINE holds
            # the native-typed default (matching load_overrides' coercions).
            baseline = getattr(cfg, "_BASELINE", {}) or {}
            new_val = baseline.get(name, getattr(cfg, name, None))
        setattr(cfg, name, new_val)
        if name in settings_schema.CACHE_REBUILD_FIELDS:
            needs_cache_rebuild = True
        if name in settings_schema.RESTART_REQUIRED_FIELDS:
            cold_changed.append(name)
        else:
            hot_changed.append(name)

    if needs_cache_rebuild:
        await rebuild_caches_off_loop("admin update")

    # Eviction-on-edit: when a load-time field changed (globally or per-model),
    # drop the affected loaded model(s) from the cache so the next request
    # reloads them with the new settings. In-flight transcribes finish on the
    # old WhisperModel instance via Python ref-counting (drain-then-evict).
    evicted: list[str] = []
    try:
        load_time_changed_globally = bool(
            set(written.keys()) & settings_schema.LOAD_TIME_FIELDS
        )
        if load_time_changed_globally:
            # Affects every loaded model that doesn't have a per-model
            # override winning over the changed global field. Conservative
            # fallback: evict ALL models. They reload lazily so this is cheap.
            ev = await tx_models.drain_then_evict(None)
            evicted.extend(ev)
        if "MODEL_OVERRIDES" in written:
            # Per-model override changed for one or more model ids — evict
            # only those whose LOAD-TIME subset (added, changed or removed
            # key) differs between the pre-save snapshot and the new bundle.
            # A removed id whose bundle held only decode-time keys needs no
            # reload, matching the global-field rule.
            new_overrides = coerced.get("MODEL_OVERRIDES") or {}
            old_overrides = prev_model_overrides or {}
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
    except Exception as e:
        # Never let eviction failure break the save response. The user's
        # change still persisted; worst case they restart manually.
        logger.error("[config] eviction-on-edit failed: %s", e)

    # Drop the cached extras (pyannote pipeline / BGM separator) when their
    # load parameters changed, so the VRAM frees now instead of at the idle
    # timeout. Buckets and their trigger fields come from the generated
    # settings_schema.EXTRAS_EVICTION (per-field `evict=` registry metadata); a
    # failed drop never breaks the save response.
    for _extra, _extra_fields in settings_schema.EXTRAS_EVICTION.items():
        if not set(written.keys()) & _extra_fields:
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
        else:
            os.environ.pop("HF_TOKEN", None)
            logger.info("[config] HF_TOKEN env cleared (config field unset)")

    # The console handler's level is read once at import; push the new one.
    # From cfg, not `written`, so an env pin or a reset to baseline wins.
    if "CONSOLE_LOG_LEVEL" in written:
        log_setup.apply_console_log_level(getattr(cfg, "CONSOLE_LOG_LEVEL", "warning"))

    # save_overrides already bumped the config version when the FILE was
    # written, but the running cfg only got the new values in the setattr loop
    # above — two awaits later. A streaming session whose _refresh_ident ran
    # in that window stamped the new version while resolving from the OLD cfg
    # and would never re-resolve. Bump again now that cfg is current; consumers
    # only compare for inequality, so the cost is one redundant re-resolve.
    settings_version.bump_config_version()

    return {
        "hot_applied": hot_changed,
        "cold_pending": cold_changed,
        "env_pinned_ignored": sorted(n for n in written if n in env_pinned),
        "evicted": evicted,
    }
