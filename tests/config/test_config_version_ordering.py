"""admin_routes._apply_hot_changes must move config_version() AFTER the running
cfg module holds the new values.

save_overrides bumps the counter when the FILE is written; the setattr loop
that updates the live cfg runs two awaits later. A streaming session whose
_refresh_ident ran in that window stamped the new version while resolving
from the OLD cfg — and, with no later bump, kept pre-edit config for the rest
of its life. The trailing bump in _apply_hot_changes closes that gap."""

import asyncio

from faster_whisper_backend import config as cfg
from faster_whisper_backend import config_store
from faster_whisper_backend.admin import routes as admin_routes


def test_apply_hot_changes_bumps_version_after_cfg_is_current(monkeypatch):
    monkeypatch.setattr(cfg, "BEAM_SIZE", 10, raising=False)
    monkeypatch.setattr(config_store, "env_pinned_fields", lambda: frozenset())
    # Simulate a stale consumer racing the save: the file-write bump has
    # already happened, and a _refresh_ident latches the version while the
    # running cfg still carries the pre-edit value.
    config_store.bump_config_version()
    latched = {}

    def _load_overrides():
        latched["version"] = config_store.config_version()
        latched["beam"] = cfg.BEAM_SIZE
        return {"BEAM_SIZE": 3}
    monkeypatch.setattr(config_store, "load_overrides", _load_overrides)

    asyncio.run(admin_routes._apply_hot_changes({"BEAM_SIZE": 3}))

    assert latched["beam"] == 10                       # consumer saw OLD cfg…
    assert cfg.BEAM_SIZE == 3                          # …cfg is current now…
    assert config_store.config_version() > latched["version"]   # …so it re-resolves
