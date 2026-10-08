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


def test_app_env_keeps_factory_rule_writes_off_the_repo_config(app_module,
                                                               tmp_path):
    """POST /settings/factory-rules saves through save_factory_rules' `path`
    default: under the app fixture both it and FACTORY_PATH must name a
    per-test copy, never the committed config.json."""
    import inspect
    import os
    from faster_whisper_backend.settings import config_store
    root = os.path.realpath(str(tmp_path))
    assert os.path.realpath(config_store.FACTORY_PATH).startswith(root)
    for fn in (config_store.load_factory_rules,
               config_store.save_factory_rules):
        default = inspect.signature(fn).parameters["path"].default
        assert os.path.realpath(default).startswith(root)
    # ...and the copy still holds the shipped rules.
    assert config_store.load_factory_rules()


def test_session_data_dir_is_a_throwaway_dir_even_when_exported():
    """conftest re-roots WHISPER_DATA_DIR unconditionally and drops the
    narrower path knobs, so a sourced deploy .env cannot point the
    import-time defaults at the operator's live data."""
    import os
    import tempfile

    from faster_whisper_backend.runtime import model_sizes, stage_rates
    from faster_whisper_backend.settings import config as cfg
    from tests import conftest

    data_dir = os.path.realpath(os.environ["WHISPER_DATA_DIR"])
    assert data_dir == os.path.realpath(conftest._TEST_DATA_DIR)
    assert data_dir.startswith(os.path.realpath(tempfile.gettempdir()))
    assert {"WHISPER_DB_DIR", "WHISPER_JOBS_DB",
            "WHISPER_URL_MEDIA_DIR"} <= conftest._DATA_PATH_ENV
    # Present-but-empty, not popped: config.py's load_dotenv() fills only
    # ABSENT keys, so a popped knob would come back from a repo-local .env.
    # LOG_FILE alone is popped ("" there switches file logging off).
    for k in conftest._DATA_PATH_ENV - {"WHISPER_LOG_FILE"}:
        assert os.environ.get(k) == "", k
    # All three froze at import, before any fixture could repoint them.
    for path in (cfg._DB_DIR, model_sizes.PATH, stage_rates.PATH):
        assert os.path.realpath(path).startswith(data_dir + os.sep), path
