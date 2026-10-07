"""Every PIPELINE_RULES read-modify-write queues on ONE lock: pl_apply.rules_lock().

/quick-config's apply_rules_patch snapshots cfg.PIPELINE_RULES, holds it across
the offloaded save and writes the whole key back; an admin save landing inside
that window would be persisted, applied, answered 200 and then silently
reverted. Before pipeline/apply.py existed the routers shared the lock by
importing each other (admin borrowed quick_config's _patch_lock). These tests
pin the property that cross-import provided: while the shared lock is held,
neither router's writer gets past it.
"""

import asyncio

import pytest
from fastapi import HTTPException

from faster_whisper_backend.admin import routes as admin_routes
from faster_whisper_backend.core.loop_lock import LoopLock
from faster_whisper_backend.pipeline import apply as pl_apply
from faster_whisper_backend.quick_config import routes as quick_config_routes
from faster_whisper_backend.settings import config_store


def test_rules_lock_is_one_module_level_lock():
    assert isinstance(pl_apply.rules_lock(), LoopLock)
    assert pl_apply.rules_lock() is pl_apply.rules_lock()
    # Neither router keeps a lock of its own that a writer could take instead.
    for mod in (admin_routes, quick_config_routes):
        own = [n for n, v in vars(mod).items() if isinstance(v, LoopLock)]
        assert not own, (mod.__name__, own)


def test_admin_and_quick_config_writers_wait_on_the_same_lock(monkeypatch):
    entered: list[str] = []

    async def _qc_locked(user, rules_patch, fingerprints=None, *, client_host="?"):
        entered.append("quick_config.apply_rules_patch")
        return 200, {}

    def _save_overrides(payload):
        entered.append("save_overrides")
        raise OSError("stop here")

    def _save_factory_rules(rules):
        entered.append("save_factory_rules")
        raise OSError("stop here")

    async def _sync(fn, *a, **k):
        return fn(*a, **k)

    monkeypatch.setattr(quick_config_routes, "_apply_rules_patch_locked", _qc_locked)
    monkeypatch.setattr(config_store, "save_overrides", _save_overrides)
    monkeypatch.setattr(config_store, "save_factory_rules", _save_factory_rules)
    # Run the writers' offloaded save inline: a writer that bypassed the lock
    # then records itself before the next yield, instead of whenever an
    # executor thread happens to get scheduled — the probe below is no
    # longer a race against thread start-up.
    monkeypatch.setattr(admin_routes.asyncio, "to_thread", _sync)

    async def scenario():
        lock = pl_apply.rules_lock()
        await lock.acquire()
        tasks = [
            asyncio.create_task(quick_config_routes.apply_rules_patch({}, {})),
            asyncio.create_task(admin_routes.post_state(
                {"PIPELINE_RULES": [{"name": "x"}]}, None)),
            asyncio.create_task(admin_routes.post_factory_rules(
                {"PIPELINE_RULES": [{"name": "x"}]}, None)),
            asyncio.create_task(admin_routes.clear_local_pipeline_override(None)),
        ]
        for _ in range(20):
            await asyncio.sleep(0)
        held_out = list(entered)
        lock.release()
        # Bounded, so a writer wedged on the lock fails here instead of
        # hanging the suite (no pytest-timeout).
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=5)
        return held_out, results

    held_out, results = asyncio.run(scenario())

    assert held_out == [], f"writers got past the held rules lock: {held_out}"
    assert sorted(entered) == sorted([
        "quick_config.apply_rules_patch", "save_overrides",
        "save_factory_rules", "save_overrides",
    ])
    assert results[0] == (200, {})
    for r in results[1:]:
        assert isinstance(r, HTTPException) and r.status_code == 500, r


def test_scalar_admin_save_stays_off_the_rules_lock(monkeypatch):
    """Unrelated scalar saves do not queue behind a PIPELINE_RULES write.

    The lock is a plain, non-reentrant asyncio lock held by this very task,
    so a regression that puts scalar saves on it would block forever —
    bounded by wait_for, it fails as a TimeoutError instead of hanging the
    suite (no pytest-timeout)."""
    entered: list[str] = []

    def _save_overrides(payload):
        entered.append("save_overrides")
        raise OSError("stop here")

    monkeypatch.setattr(config_store, "save_overrides", _save_overrides)

    async def scenario():
        async with pl_apply.rules_lock():
            with pytest.raises(HTTPException):
                await asyncio.wait_for(
                    admin_routes.post_state({"BEAM_SIZE": 3}, None), timeout=5)
            return list(entered)

    assert asyncio.run(scenario()) == ["save_overrides"]
