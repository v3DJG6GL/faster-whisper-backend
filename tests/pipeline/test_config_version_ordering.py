"""pl_apply.apply_hot_changes must move config_version() AFTER the running
cfg module holds the new values.

save_overrides bumps the counter when the FILE is written; the setattr loop
that updates the live cfg runs two awaits later. A streaming session whose
_refresh_ident ran in that window stamped the new version while resolving
from the OLD cfg — and, with no later bump, kept pre-edit config for the rest
of its life. The trailing bump in apply_hot_changes closes that gap."""

import asyncio

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.settings import version as settings_version
from faster_whisper_backend.pipeline import apply as pl_apply


def test_apply_hot_changes_bumps_version_after_cfg_is_current(monkeypatch):
    monkeypatch.setattr(cfg, "BEAM_SIZE", 10, raising=False)
    monkeypatch.setattr(config_store, "env_pinned_fields", lambda: frozenset())
    # Simulate a stale consumer racing the save: the file-write bump has
    # already happened, and a _refresh_ident latches the version while the
    # running cfg still carries the pre-edit value.
    settings_version.bump_config_version()
    latched = {}

    def _load_overrides():
        latched["version"] = settings_version.config_version()
        latched["beam"] = cfg.BEAM_SIZE
        return {"BEAM_SIZE": 3}
    monkeypatch.setattr(config_store, "load_overrides", _load_overrides)

    asyncio.run(pl_apply.apply_hot_changes({"BEAM_SIZE": 3}))

    assert latched["beam"] == 10                       # consumer saw OLD cfg…
    assert cfg.BEAM_SIZE == 3                          # …cfg is current now…
    assert settings_version.config_version() > latched["version"]   # …so it re-resolves


def test_apply_hot_changes_bumps_version_before_model_eviction(monkeypatch):
    # drain_then_evict waits on the model-load lock, which a concurrent load
    # holds for its whole constructor. The re-resolve bump must not queue
    # behind that: by the time eviction starts, cfg is current and the
    # version already says so.
    from faster_whisper_backend.transcription import models as tx_models
    monkeypatch.setattr(cfg, "MODEL_DEVICE", "cuda", raising=False)
    monkeypatch.setattr(config_store, "env_pinned_fields", lambda: frozenset())
    monkeypatch.setattr(config_store, "load_overrides",
                        lambda: {"MODEL_DEVICE": "cpu"})
    monkeypatch.setattr(pl_apply, "EVICTORS",
                        {k: (lambda: asyncio.sleep(0)) for k in pl_apply.EVICTORS})
    seen = {}

    async def _drain(model_id=None):
        seen["version"] = settings_version.config_version()
        return []
    monkeypatch.setattr(tx_models, "drain_then_evict", _drain)

    before = settings_version.config_version()
    asyncio.run(pl_apply.apply_hot_changes({"MODEL_DEVICE": "cpu"}))

    assert "version" in seen                    # MODEL_DEVICE is load-time
    assert seen["version"] > before
