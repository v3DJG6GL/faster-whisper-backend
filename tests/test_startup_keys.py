"""WHISPER_BOOTSTRAP_ADMIN_KEY at startup: an already-registered key is never
re-judged, bootstrap failures keep their own wording through the lifespan,
and a fatal API-keys init leaves nothing running behind it."""

import asyncio
import logging

import pytest

from faster_whisper_backend.runtime import system_stats

_KEY = "bootstrap-key-with-enough-entropy-1234"


def test_existing_key_below_todays_floor_is_silently_accepted(
        api_keys_db, monkeypatch, caplog):
    from faster_whisper_backend import main
    main._bootstrap_admin_from_env(_KEY)
    h = api_keys_db.hash_key(_KEY)
    assert api_keys_db._KEY_INDEX.get(h) is not None
    # The floor moves up; the key is now "weak" — but it already exists.
    monkeypatch.setattr(main, "_BOOTSTRAP_KEY_MIN_LEN", 999)
    assert not main._bootstrap_key_is_strong(_KEY)
    with caplog.at_level(logging.ERROR, logger="whisper-server"):
        main._bootstrap_admin_from_env(_KEY)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert api_keys_db._KEY_INDEX.get(h) is not None
    assert api_keys_db.is_locked_down() is True


def test_new_weak_key_is_still_refused(api_keys_db, monkeypatch, caplog):
    from faster_whisper_backend import main
    with caplog.at_level(logging.ERROR, logger="whisper-server"):
        main._bootstrap_admin_from_env("short")
    assert any("too weak" in r.getMessage() for r in caplog.records)
    assert api_keys_db.is_locked_down() is False


def test_revoked_key_raises_bootstrap_admin_error(api_keys_db):
    from faster_whisper_backend import main
    main._bootstrap_admin_from_env(_KEY)
    h = api_keys_db.hash_key(_KEY)
    uid2 = api_keys_db.create_user("second-admin", is_admin=True)
    api_keys_db.create_key(uid2)
    api_keys_db.revoke_key(api_keys_db._KEY_INDEX[h]["key_id"])
    with pytest.raises(main.BootstrapAdminError, match="REVOKED"):
        main._bootstrap_admin_from_env(_KEY)


def test_lifespan_passes_bootstrap_errors_through_unwrapped(
        app_module, monkeypatch, caplog):
    monkeypatch.setattr(app_module.cfg, "BOOTSTRAP_ADMIN_KEY", _KEY,
                        raising=False)

    def _boom(_raw):
        raise app_module.BootstrapAdminError(
            "WHISPER_BOOTSTRAP_ADMIN_KEY matches an API key that has been "
            "REVOKED.")
    monkeypatch.setattr(app_module, "_bootstrap_admin_from_env", _boom)

    async def run():
        async with app_module.lifespan(app_module.app):
            pass

    with caplog.at_level(logging.CRITICAL, logger="whisper-server"):
        with pytest.raises(app_module.BootstrapAdminError) as ei:
            asyncio.run(run())
    assert "REVOKED" in str(ei.value)
    assert "API keys store unavailable" not in str(ei.value)
    crit = [r.getMessage() for r in caplog.records]
    assert any("REVOKED" in m for m in crit)
    assert not any("Failed to initialize the API keys store" in m for m in crit)


def test_fatal_api_keys_init_leaves_no_tasks_behind(app_module, monkeypatch):
    from faster_whisper_backend.auth import api_keys_store

    def _boom(_path):
        raise OSError("read-only")
    monkeypatch.setattr(api_keys_store, "init_db", _boom)
    leaked = {}

    async def run():
        before = {t for t in asyncio.all_tasks() if not t.done()}
        try:
            async with app_module.lifespan(app_module.app):
                pass
        finally:
            leaked["tasks"] = {t for t in asyncio.all_tasks()
                               if not t.done()} - before

    with pytest.raises(RuntimeError, match="API keys store unavailable"):
        asyncio.run(run())
    assert leaked["tasks"] == set()
    assert system_stats._warm_predicate is None
