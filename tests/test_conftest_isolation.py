"""conftest.isolate_app_env must not layer the operator's config.local.json
onto cfg.

config.py calls config_store.load_overrides() with no argument at import, so
the per-test overrides path has to be in place BEFORE the config reload. On a
dev box that exports WHISPER_DATA_DIR or WHISPER_CONFIG_LOCAL (a sourced
deploy .env), config_store's import-time path names the real file.
"""

import importlib
import json

from tests.conftest import _repoint_path_default, isolate_app_env


def test_config_reload_never_reads_the_operator_overrides_file(tmp_path,
                                                               monkeypatch):
    from faster_whisper_backend.settings import config as cfg
    from faster_whisper_backend.settings import config_store
    seeded = tmp_path / "operator" / "config.local.json"
    seeded.parent.mkdir()
    seeded.write_text(json.dumps({"BEAM_SIZE": 7}), encoding="utf-8")
    # Stand-in for an exported WHISPER_DATA_DIR: the path config_store froze
    # at its first import names the operator's file.
    monkeypatch.setattr(config_store, "OVERRIDES_PATH", str(seeded))
    _repoint_path_default(monkeypatch, (config_store.load_overrides,
                                        config_store.save_overrides),
                          str(seeded))
    assert config_store.load_overrides().get("BEAM_SIZE") == 7   # live file
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    try:
        isolate_app_env(app_dir, monkeypatch)
        assert cfg.BEAM_SIZE != 7
    finally:
        # Leave cfg as the session environment built it.
        monkeypatch.undo()
        importlib.reload(cfg)


def test_repoint_path_default_targets_the_named_parameter(monkeypatch):
    """A trailing default added after `path` must not take the rewrite (it
    once left a function on the real repo config)."""
    import pytest

    def fn(payload, path="/real", indent=2, *, sort=False):
        return path, indent

    def kw(payload, *, path="/real"):
        return path

    _repoint_path_default(monkeypatch, (fn, kw), "/tmp/x")
    assert fn(None) == ("/tmp/x", 2)
    assert kw(None) == "/tmp/x"

    def no_path(payload, other="/real"):
        return other
    with pytest.raises(AssertionError):
        _repoint_path_default(monkeypatch, (no_path,), "/tmp/x")
