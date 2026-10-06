"""Tests for config.py's environment-variable layer: the coercion helpers,
the per-model override decoder, and the stdlib factory-rules loader.

The helper functions read os.environ at call time, so they're tested directly
with monkeypatched env vars. The module-level per-model scanner is exercised
once via a guarded importlib.reload (restored in a finally).
"""

import importlib
import json
import os

import pytest

from faster_whisper_backend.settings import config


# ---------------------------------------------------------------------------
# Scalar coercion helpers
# ---------------------------------------------------------------------------

def test_truthy():
    for v in ["1", "true", "TRUE", "Yes", "on", "  on  "]:
        assert config._truthy(v) is True
    for v in ["0", "false", "no", "off", "", "maybe"]:
        assert config._truthy(v) is False


def test_env_int(monkeypatch):
    monkeypatch.setenv("X_INT", "42")
    assert config._env_int("X_INT", 7) == 42
    monkeypatch.setenv("X_INT", "  ")
    assert config._env_int("X_INT", 7) == 7          # blank -> current
    monkeypatch.setenv("X_INT", "notanint")
    assert config._env_int("X_INT", 7) == 7          # invalid -> current
    monkeypatch.delenv("X_INT")
    assert config._env_int("X_INT", 7) == 7          # unset -> current


def test_env_float(monkeypatch):
    monkeypatch.setenv("X_F", "1.5")
    assert config._env_float("X_F", 0.0) == 1.5
    monkeypatch.setenv("X_F", "bad")
    assert config._env_float("X_F", 0.25) == 0.25


def test_env_bool(monkeypatch):
    monkeypatch.setenv("X_B", "yes")
    assert config._env_bool("X_B", False) is True
    monkeypatch.setenv("X_B", "")
    assert config._env_bool("X_B", True) is True     # blank -> current
    monkeypatch.delenv("X_B")
    assert config._env_bool("X_B", True) is True
    # The common short / verbose spellings parse symmetrically.
    monkeypatch.setenv("X_B", "y")
    assert config._env_bool("X_B", False) is True
    monkeypatch.setenv("X_B", "enabled")
    assert config._env_bool("X_B", False) is True
    monkeypatch.setenv("X_B", "n")
    assert config._env_bool("X_B", True) is False
    monkeypatch.setenv("X_B", "disabled")
    assert config._env_bool("X_B", True) is False
    # A genuinely unparseable value keeps the current setting and warns.
    monkeypatch.setattr(config, "_ENV_WARNINGS", [])
    monkeypatch.setattr(config, "_ENV_UNPARSED", set())
    monkeypatch.setenv("X_B", "maybe")
    assert config._env_bool("X_B", True) is True
    assert config._env_bool("X_B", False) is False
    assert any("X_B='maybe'" in w for w in config._ENV_WARNINGS)
    assert "X_B" in config._ENV_UNPARSED


def test_env_str(monkeypatch):
    monkeypatch.setenv("X_S", "  hi ")
    assert config._env_str("X_S", "cur") == "hi"
    monkeypatch.setenv("X_S", "   ")
    assert config._env_str("X_S", "cur") == "cur"    # blank -> current


def test_env_str_or_none(monkeypatch):
    # explicit empty string -> None (disable)
    monkeypatch.setenv("X_SON", "")
    assert config._env_str_or_none("X_SON", "cur") is None
    monkeypatch.setenv("X_SON", "val")
    assert config._env_str_or_none("X_SON", "cur") == "val"
    monkeypatch.delenv("X_SON")
    assert config._env_str_or_none("X_SON", "cur") == "cur"


def test_env_str_passthrough(monkeypatch):
    # empty string is preserved as a real value (NOT None / current)
    monkeypatch.setenv("X_SP", "")
    assert config._env_str_passthrough("X_SP", "cur") == ""
    monkeypatch.delenv("X_SP")
    assert config._env_str_passthrough("X_SP", "cur") == "cur"


def test_env_csv_list(monkeypatch):
    monkeypatch.setenv("X_L", "a, b ,,c")
    assert config._env_csv_list("X_L", ["z"]) == ["a", "b", "c"]
    monkeypatch.setenv("X_L", "")          # explicit empty -> empty list
    assert config._env_csv_list("X_L", ["z"]) == []
    monkeypatch.delenv("X_L")
    assert config._env_csv_list("X_L", ["z"]) == ["z"]   # unset -> current


# ---------------------------------------------------------------------------
# Per-model override decode helpers
# ---------------------------------------------------------------------------

def test_decode_model_id():
    assert config._decode_model_id("org__SLASH__name__DOT__ct2") == "org/name.ct2"
    assert config._decode_model_id("plain") == "plain"


def test_coerce_override_value_types():
    assert config._coerce_override_value("VAD_FILTER", "true") is True
    assert config._coerce_override_value("BEAM_SIZE", "7") == 7
    assert config._coerce_override_value("BEAM_SIZE", "x") == "x"   # invalid -> raw
    assert config._coerce_override_value("VAD_THRESHOLD", "0.5") == 0.5
    assert config._coerce_override_value("PIPELINE_RULES_EXCLUDE", "a, b ,c") == ["a", "b", "c"]
    # TEMPERATURE is unclassified -> raw string passthrough
    assert config._coerce_override_value("TEMPERATURE", "0,0.2") == "0,0.2"


def test_per_model_env_scanner_end_to_end(monkeypatch):
    # WHISPER_MODEL_OVERRIDE__<encoded id>__<FIELD> populates MODEL_OVERRIDES.
    # NOTE: the encoded id is UPPERCASE on purpose. Windows normalises
    # os.environ keys to uppercase, so a lowercase id in the var NAME would not
    # round-trip there; an uppercase id is case-stable on every platform and
    # still exercises the right-to-left "__" boundary scanner + _decode_model_id.
    monkeypatch.setenv(
        "WHISPER_MODEL_OVERRIDE__ORG__SLASH__NAME__DOT__CT2__BEAM_SIZE", "7"
    )
    try:
        importlib.reload(config)
        assert config.MODEL_OVERRIDES.get("ORG/NAME.CT2", {}).get("BEAM_SIZE") == 7
    finally:
        monkeypatch.undo()
        importlib.reload(config)  # restore from the clean environment


def test_env_value_is_stored_as_the_schema_normalizes_it(monkeypatch):
    # The schema accepts Python's own spelling; the runtime value (and so the
    # /settings page) is the lowercase literal, not the raw env text.
    monkeypatch.setenv("WHISPER_CONSOLE_LOG_LEVEL", "WARNING")
    try:
        importlib.reload(config)
        assert config.CONSOLE_LOG_LEVEL == "warning"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# ---------------------------------------------------------------------------
# _load_defaults (stdlib loader: config.json is the single source of factory
# defaults; config.py reads every value from it at import)
# ---------------------------------------------------------------------------

def _write_cfg(tmp_path, **extra):
    """Write a minimal-but-valid config.json into tmp_path and return its path."""
    data = {"schema_version": 1,
            "PIPELINE_RULES": [{"name": "trim", "label": "Trim", "type": "terminal"}],
            **extra}
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_load_defaults_reads_committed_config():
    d = config._load_defaults()
    # Returns ALL defaults, not just the rules: scalars + the rules list. Assert
    # structure/types, not specific values (those are deployment-tunable).
    assert isinstance(d, dict)
    assert isinstance(d["PIPELINE_RULES"], list) and len(d["PIPELINE_RULES"]) >= 2
    assert isinstance(d["DEFAULT_MODEL"], str) and d["DEFAULT_MODEL"]
    assert isinstance(d["BEST_OF"], int)
    assert len(d) > 50                        # the full settings set, not a handful
    assert "schema_version" not in d          # stripped — it's metadata, not a setting


def test_load_defaults_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_defaults()


def test_load_defaults_corrupt_raises(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_defaults()


def test_load_defaults_missing_rules_raises(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"schema_version": 1}),
                                          encoding="utf-8")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_defaults()


def test_load_defaults_is_the_source(monkeypatch, tmp_path):
    # A value placed in config.json is what _load_defaults returns — proving
    # config.json (not config.py) is the source of truth.
    _write_cfg(tmp_path, BEST_OF=9, DEFAULT_MODEL="my-model")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    d = config._load_defaults()
    assert d["BEST_OF"] == 9
    assert d["DEFAULT_MODEL"] == "my-model"


def test_load_defaults_resolves_data_dir_placeholders(monkeypatch, tmp_path):
    # {DATA_DIR}/{DB_DIR}/{MODELS_DIR} placeholders resolve against the data
    # layout knobs (WHISPER_DATA_DIR/WHISPER_DB_DIR/WHISPER_MODELS_DIR —
    # captured at import into _DATA_DIR/_DB_DIR/_MODELS_DIR), NOT the repo
    # dir. See also tests/settings/test_data_dir.py for the end-to-end env → path
    # matrix.
    _write_cfg(tmp_path,
               LOG_FILE="{DATA_DIR}/logs/whisper.log",
               API_KEYS_DB="{DB_DIR}/api_keys.local.sqlite3",
               DOWNLOAD_ROOT="{MODELS_DIR}")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    monkeypatch.setattr(config, "_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(config, "_DB_DIR", str(tmp_path / "state" / "db"))
    monkeypatch.setattr(config, "_MODELS_DIR", str(tmp_path / "models"))
    d = config._load_defaults()
    assert d["LOG_FILE"] == os.path.normpath(
        os.path.join(str(tmp_path), "state", "logs/whisper.log"))
    assert d["API_KEYS_DB"] == os.path.normpath(
        os.path.join(str(tmp_path), "state", "db", "api_keys.local.sqlite3"))
    assert d["DOWNLOAD_ROOT"] == os.path.normpath(
        os.path.join(str(tmp_path), "models"))
    assert "{DATA_DIR}" not in d["LOG_FILE"] and "{DB_DIR}" not in d["API_KEYS_DB"]


def test_load_defaults_coerces_set_fields(monkeypatch, tmp_path):
    _write_cfg(tmp_path,
               ALLOWED_MODELS=["large-v2", "large-v3"],
               CAPTURES_PIPELINE_RULES_EXCLUDE=["de-dictation-map"])
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    d = config._load_defaults()
    assert d["ALLOWED_MODELS"] == {"large-v2", "large-v3"}
    assert isinstance(d["ALLOWED_MODELS"], set)
    assert isinstance(d["CAPTURES_PIPELINE_RULES_EXCLUDE"], set)


def test_baseline_comes_from_config_json():
    # _BASELINE (what "↺ Reset to default" reverts to) must equal the values in
    # config.json, with the same set-coercion + {DATA_DIR}/{DB_DIR}/{MODELS_DIR}
    # path resolution applied.
    # Locks "config.json is the single source of truth for factory defaults".
    expected = config._load_defaults()
    for k, v in expected.items():
        assert config._BASELINE[k] == v, k
    assert set(config._BASELINE) == set(expected)


def test_env_float_or_none(monkeypatch):
    # explicit empty string -> None (disable the check)
    monkeypatch.setenv("X_FON", "")
    assert config._env_float_or_none("X_FON", 0.6) is None
    monkeypatch.setenv("X_FON", "0.3")
    assert config._env_float_or_none("X_FON", 0.6) == 0.3
    monkeypatch.setenv("X_FON", "bad")
    assert config._env_float_or_none("X_FON", 0.6) == 0.6   # invalid -> current
    monkeypatch.delenv("X_FON")
    assert config._env_float_or_none("X_FON", 0.6) == 0.6   # unset -> current


# ---------------------------------------------------------------------------
# Env var names (ENV_VAR_MAPPING covers every AdminConfig field by
# construction: schema.py builds it over the field registry)
# ---------------------------------------------------------------------------

def test_env_var_names_are_unique():
    from faster_whisper_backend.settings import schema as settings_schema
    names = list(settings_schema.ENV_VAR_MAPPING.values())
    assert len(names) == len(set(names)), "duplicate WHISPER_* env var names"
    assert all(n.startswith("WHISPER_") for n in names)


# ---------------------------------------------------------------------------
# End-to-end schema-driven env application (importlib.reload)
# ---------------------------------------------------------------------------

def _reload_with_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    importlib.reload(config)


def test_scalar_env_overrides_apply(monkeypatch):
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_BEAM_SIZE="3",
            WHISPER_SERVER_PORT="8123",
            WHISPER_VAD_FILTER="0",
            WHISPER_MODEL_DEVICE="cpu",
        )
        assert config.BEAM_SIZE == 3
        assert config.SERVER_PORT == 8123
        assert config.VAD_FILTER is False
        assert config.MODEL_DEVICE == "cpu"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_optional_threshold_empty_disables(monkeypatch):
    try:
        _reload_with_env(monkeypatch, WHISPER_NO_SPEECH_THRESHOLD="")
        assert config.NO_SPEECH_THRESHOLD is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_default_language_empty_means_autodetect(monkeypatch):
    # DEFAULT_LANGUAGE="" is a meaningful literal (auto-detect), not None.
    try:
        _reload_with_env(monkeypatch, WHISPER_DEFAULT_LANGUAGE="")
        assert config.DEFAULT_LANGUAGE == ""
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_set_typed_special_cases_stay_sets(monkeypatch):
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_ALLOWED_MODELS="a,b",
            WHISPER_CAPTURES_PIPELINE_RULES_EXCLUDE=
            "strip-stray-symbols,strip-trailing-period",
        )
        assert config.ALLOWED_MODELS == {"a", "b"}
        assert config.CAPTURES_PIPELINE_RULES_EXCLUDE == {
            "strip-stray-symbols", "strip-trailing-period"}
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_json_model_overrides_env(monkeypatch):
    # JSON blob validates + normalises to plain dicts; per-model var merges atop.
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_MODEL_OVERRIDES='{"large-v2": {"BEAM_SIZE": 4}}',
            # Uppercase id on purpose: Windows normalises os.environ keys to
            # uppercase, so a lowercase id in the var NAME wouldn't round-trip
            # there (same reason as test_per_model_env_scanner_end_to_end).
            WHISPER_MODEL_OVERRIDE__LARGE__DOT__V3__BEAM_SIZE="7",
        )
        assert config.MODEL_OVERRIDES["large-v2"] == {"BEAM_SIZE": 4}
        assert isinstance(config.MODEL_OVERRIDES["large-v2"], dict)
        assert config.MODEL_OVERRIDES["LARGE.V3"]["BEAM_SIZE"] == 7
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_json_pipeline_rules_env(monkeypatch):
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_PIPELINE_RULES='[{"name": "x", "label": "X", "type": "terminal"}]',
        )
        assert isinstance(config.PIPELINE_RULES, list)
        assert config.PIPELINE_RULES[0]["name"] == "x"
        assert isinstance(config.PIPELINE_RULES[0], dict)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_json_pipeline_rules_env_upgrades_an_old_factory_entry(monkeypatch):
    """A WHISPER_PIPELINE_RULES copy written before a factory entry was fixed
    gets the fixed text (config_renames.UPGRADED_RULE_ENTRIES), with a warning."""
    import json as _json
    from faster_whisper_backend.settings import config_renames
    (old_pat, old_rep), new = config_renames.UPGRADED_RULE_ENTRIES["tighten-quote-spacing"]
    rules = [
        {"name": "tidy", "label": "Tidy", "type": "regex-list",
         "entries": [{"pattern": old_pat, "replacement": old_rep,
                      "label": "tighten-quote-spacing"}]},
        {"name": "x", "label": "X", "type": "terminal"},
    ]
    try:
        _reload_with_env(monkeypatch, WHISPER_PIPELINE_RULES=_json.dumps(rules))
        e = config.PIPELINE_RULES[0]["entries"][0]
        assert (e["pattern"], e["replacement"]) == new
        assert any("upgraded factory rule entries" in m for m in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_invalid_json_keeps_default_and_warns(monkeypatch):
    try:
        _reload_with_env(monkeypatch, WHISPER_PIPELINE_RULES="not json")
        # factory rules remain in place
        assert len(config.PIPELINE_RULES) >= 2
        assert any("WHISPER_PIPELINE_RULES" in m for m in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_bad_scalar_records_warning(monkeypatch):
    try:
        _reload_with_env(monkeypatch, WHISPER_BEAM_SIZE="ten")
        assert config.BEAM_SIZE == 10   # default kept
        assert any("WHISPER_BEAM_SIZE" in m for m in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_secret_file_indirection(monkeypatch, tmp_path):
    secret = tmp_path / "key"
    secret.write_text("  sk-from-file  \n", encoding="utf-8")
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_BOOTSTRAP_ADMIN_KEY_FILE=str(secret),
        )
        assert config.BOOTSTRAP_ADMIN_KEY == "sk-from-file"
    finally:
        monkeypatch.undo()
        # The *_FILE prepass writes the resolved secret straight into os.environ
        # (so both the explicit reader and the schema loop see it); monkeypatch
        # can't undo that, so clear it before the restoring reload.
        os.environ.pop("WHISPER_BOOTSTRAP_ADMIN_KEY", None)
        importlib.reload(config)


def test_rejected_secret_warning_is_redacted(monkeypatch):
    """An over-long HF_TOKEN fails AdminConfig's max_length=256 and the
    field reverts — but the warning must NOT carry the repr of the value that
    stays in force. _ENV_WARNINGS is drained into the logger and that log is
    served by the /logs viewer and /logs/stream."""
    try:
        _reload_with_env(monkeypatch, WHISPER_HF_TOKEN="hf_" + "z" * 300)
        warn = [m for m in config._ENV_WARNINGS
                if "WHISPER_HF_TOKEN" in m and "not a valid" in m]
        assert warn, config._ENV_WARNINGS
        msg = warn[0]
        # The operator still learns which var was rejected and why...
        assert "HF_TOKEN" in msg
        # ...but the retained value is not echoed.
        assert "keeping <redacted>" in msg
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_use_auth_token_env_alias(monkeypatch):
    """The pre-rename env spelling still works: WHISPER_USE_AUTH_TOKEN is
    aliased onto WHISPER_HF_TOKEN at config import, and a set new-name value
    wins over the alias."""
    try:
        _reload_with_env(monkeypatch, WHISPER_USE_AUTH_TOKEN="hf_old")
        assert config.HF_TOKEN == "hf_old"
        # The alias writes the NEW var straight into os.environ (so the _FILE
        # loop and env_pinned_fields see it); monkeypatch can't undo that —
        # clear it before the next reload (same caveat as the bootstrap-key
        # test above).
        os.environ.pop("WHISPER_HF_TOKEN", None)
        _reload_with_env(monkeypatch, WHISPER_USE_AUTH_TOKEN="hf_old",
                         WHISPER_HF_TOKEN="hf_new")
        assert config.HF_TOKEN == "hf_new"
    finally:
        monkeypatch.undo()
        os.environ.pop("WHISPER_HF_TOKEN", None)
        importlib.reload(config)


def test_local_overrides_migrate_use_auth_token(tmp_path):
    """A config.local.json from before the rename still carries USE_AUTH_TOKEN;
    load_overrides must migrate the key instead of failing validation (which
    would silently drop EVERY stored override)."""
    from faster_whisper_backend.settings import config_store
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"USE_AUTH_TOKEN": "hf_stored", "BEAM_SIZE": 7}),
                 encoding="utf-8")
    out = config_store.load_overrides(str(p))
    assert out.get("HF_TOKEN") == "hf_stored"
    assert "USE_AUTH_TOKEN" not in out
    assert out.get("BEAM_SIZE") == 7


def test_rejected_nonsecret_warning_still_shows_value(monkeypatch):
    """The redaction is limited to the credential fields — an ordinary field
    still reports the value left in force, which is what makes the warning
    actionable."""
    try:
        _reload_with_env(monkeypatch, WHISPER_BEAM_SIZE="9999")
        warn = [m for m in config._ENV_WARNINGS if "WHISPER_BEAM_SIZE" in m]
        assert warn, config._ENV_WARNINGS
        # BEAM_SIZE is the reverted (pre-env) value after the rejection.
        assert f"keeping {config.BEAM_SIZE}" in warn[0]
        assert "<redacted>" not in warn[0]
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_per_model_env_override_coerces_newer_field_types(monkeypatch):
    """DIARIZE (bool) and DIARIZATION_NUM_SPEAKERS (int) are ModelOverride
    fields that postdate the coercion frozensets — a raw string surviving here
    means bool("false") is True downstream and speaker hints reach pyannote
    as strings."""
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_MODEL_OVERRIDE__TINY__DIARIZE="false",
            WHISPER_MODEL_OVERRIDE__TINY__DIARIZATION_NUM_SPEAKERS="3",
        )
        entry = config.MODEL_OVERRIDES.get("TINY", {})
        assert entry.get("DIARIZE") is False
        assert entry.get("DIARIZATION_NUM_SPEAKERS") == 3
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_per_model_env_override_unparseable_bool_is_skipped(monkeypatch):
    """The per-model path used to coerce bools with _truthy, so DIARIZE=enabled
    silently became False (a valid bool the revalidation cannot catch). It must
    warn and leave that one field out, keeping the model's other overrides."""
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_MODEL_OVERRIDE__TINY__DIARIZE="maybe",
            WHISPER_MODEL_OVERRIDE__TINY__BEAM_SIZE="3",
        )
        entry = config.MODEL_OVERRIDES.get("TINY", {})
        assert "DIARIZE" not in entry
        assert entry.get("BEAM_SIZE") == 3
        assert any("DIARIZE" in w and "not a valid boolean" in w
                   for w in config._ENV_WARNINGS), config._ENV_WARNINGS
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_per_model_env_override_accepts_verbose_bool_spellings(monkeypatch):
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_MODEL_OVERRIDE__TINY__DIARIZE="enabled",
            WHISPER_MODEL_OVERRIDE__TINY__VAD_FILTER="n",
        )
        entry = config.MODEL_OVERRIDES.get("TINY", {})
        assert entry.get("DIARIZE") is True
        assert entry.get("VAD_FILTER") is False
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_reader_level_env_rejection_clears_the_pin(monkeypatch):
    """A value the reader could not parse controls nothing, exactly like one
    the schema pass reverted — so it must land in _ENV_REJECTED and drop out
    of env_pinned_fields(), or the admin's /settings edit never applies."""
    from faster_whisper_backend.settings import config_store
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_BEAM_SIZE="abc",
            WHISPER_SESSION_COOKIE_SECURE="maybe",
        )
        assert config.BEAM_SIZE == config._BASELINE["BEAM_SIZE"]
        assert "BEAM_SIZE" in config._ENV_REJECTED
        assert "SESSION_COOKIE_SECURE" in config._ENV_REJECTED
        pinned = config_store.env_pinned_fields()
        assert "BEAM_SIZE" not in pinned
        assert "SESSION_COOKIE_SECURE" not in pinned
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_save_overrides_migrates_use_auth_token(tmp_path):
    """The save path re-reads the raw file and merges the payload on top; a
    surviving pre-rename USE_AUTH_TOKEN key would make AdminConfig
    (extra=forbid) reject EVERY save forever. The first successful write must
    migrate the key and self-heal the file."""
    from faster_whisper_backend.settings import config_store
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"USE_AUTH_TOKEN": "hf_stored", "BEAM_SIZE": 7}),
                 encoding="utf-8")
    changed = config_store.save_overrides({"BEAM_SIZE": 5}, path=str(p))
    assert changed.get("BEAM_SIZE") == 5
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk.get("HF_TOKEN") == "hf_stored"
    assert "USE_AUTH_TOKEN" not in on_disk
    assert on_disk.get("BEAM_SIZE") == 5


def test_env_cross_field_triple_validates_as_a_batch(monkeypatch):
    """A self-consistent TARGET/MAX pair set together via env must apply.
    Per-field validation filled the unset members from _BASELINE (TARGET
    26.0), so MAX=20 alone was mis-rejected even though the operator also
    set TARGET=15 in the same environment."""
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_CAPTURES_PROPOSER_TARGET_S="15",
            WHISPER_CAPTURES_SAMPLE_MAX_DURATION_S="20",
        )
        assert config.CAPTURES_PROPOSER_TARGET_S == 15
        assert config.CAPTURES_SAMPLE_MAX_DURATION_S == 20
        assert not [m for m in config._ENV_WARNINGS
                    if "CAPTURES_" in m], config._ENV_WARNINGS
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_env_cross_field_triple_inconsistent_pair_is_reverted(monkeypatch):
    """The cross-field validator reports loc=(), so the attribution pass
    cannot name a field and the per-field fallback passes each half in
    isolation. The group backstop must still revert the inconsistent pair."""
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_CAPTURES_SAMPLE_MIN_DURATION_S="20",
            WHISPER_CAPTURES_PROPOSER_TARGET_S="10",
        )
        assert (config.CAPTURES_SAMPLE_MIN_DURATION_S
                == config._BASELINE["CAPTURES_SAMPLE_MIN_DURATION_S"])
        assert (config.CAPTURES_PROPOSER_TARGET_S
                == config._BASELINE["CAPTURES_PROPOSER_TARGET_S"])
        assert "CAPTURES_SAMPLE_MIN_DURATION_S" in config._ENV_REJECTED
        assert "CAPTURES_PROPOSER_TARGET_S" in config._ENV_REJECTED
        assert [m for m in config._ENV_WARNINGS if "CAPTURES_" in m]
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_legacy_in_repo_state_warns_when_ignored(tmp_path, monkeypatch):
    """An in-place upgrade that still has runtime state under the checkout,
    with nothing at the configured (data-dir) location, must say so instead of
    silently starting from factory config / an empty key store."""
    monkeypatch.delenv("WHISPER_CONFIG_LOCAL", raising=False)
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "config.local.json").write_text("{}", encoding="utf-8")
    api_db = str(data / "db" / "api_keys.local.sqlite3")
    usage_db = str(data / "db" / "usage.local.sqlite3")
    configured = {
        "config.local.json": str(data / "config.local.json"),
        "api_keys.local.sqlite3": api_db,
        "usage.local.sqlite3": usage_db,
    }

    warns = config._legacy_state_warnings(str(repo), str(data), configured)
    assert len(warns) == 1                      # config.local.json only
    assert str(repo / "config.local.json") in warns[0]
    assert str(data / "config.local.json") in warns[0]
    assert "IGNORED" in warns[0]

    # Once the configured file exists the warning goes away.
    (data / "config.local.json").write_text("{}", encoding="utf-8")
    assert config._legacy_state_warnings(str(repo), str(data), configured) == []

    # And the legacy api-keys DB warns the same way.
    (repo / "api_keys.local.sqlite3").write_text("", encoding="utf-8")
    warns = config._legacy_state_warnings(str(repo), str(data), configured)
    assert len(warns) == 1
    assert "api_keys.local.sqlite3" in warns[0]

    # Every relocated store is covered, not just config + api keys: a usage
    # DB left under the checkout (empty usage history after the upgrade).
    (data / "db").mkdir()
    (data / "db" / "api_keys.local.sqlite3").write_text("", encoding="utf-8")
    (repo / "usage.local.sqlite3").write_text("", encoding="utf-8")
    warns = config._legacy_state_warnings(str(repo), str(data), configured)
    assert len(warns) == 1
    assert "usage.local.sqlite3" in warns[0]


def test_legacy_state_mapping_uses_the_loader_overrides_path():
    """The startup call feeds config_store.OVERRIDES_PATH (the file the loader
    actually reads) into the mapping — not a re-derived copy of the rule."""
    from faster_whisper_backend.settings import config_store
    assert config._overrides_path == config_store.OVERRIDES_PATH

def test_legacy_data_dir_root_db_warns_when_ignored(tmp_path):
    """Pre-db-layout compose installs kept SQLite stores at the data-dir
    ROOT; the upgrade note relies on this probe catching them too."""
    from faster_whisper_backend.settings import config as cfg
    legacy = tmp_path / "api_keys.local.sqlite3"
    legacy.write_bytes(b"")
    warns = cfg._legacy_state_warnings(
        str(tmp_path / "repo"), str(tmp_path),
        {"api_keys.local.sqlite3": str(tmp_path / "db" / "api_keys.local.sqlite3")},
    )
    assert len(warns) == 1 and str(legacy) in warns[0]
    # and silent when the configured path exists
    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "api_keys.local.sqlite3").write_bytes(b"")
    assert cfg._legacy_state_warnings(
        str(tmp_path / "repo"), str(tmp_path),
        {"api_keys.local.sqlite3": str(tmp_path / "db" / "api_keys.local.sqlite3")},
    ) == []


def test_renamed_keys_env_alias_is_table_driven(monkeypatch):
    """Every entry in config_renames.RENAMED_KEYS is honoured as an env alias
    (not only the hand-written HF_TOKEN shim it replaced), and each applied
    alias leaves one startup warning naming both spellings."""
    from faster_whisper_backend.settings import config_renames
    old, new = "RECENT_TRANSCRIPTIONS_TTL_DAYS", "RECENT_TRANSCRIPTIONS_RETENTION_DAYS"
    assert config_renames.RENAMED_KEYS[old] == new
    try:
        _reload_with_env(monkeypatch, WHISPER_RECENT_TRANSCRIPTIONS_TTL_DAYS="7")
        assert config.RECENT_TRANSCRIPTIONS_RETENTION_DAYS == 7
        assert any(old in w and new in w for w in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        os.environ.pop("WHISPER_" + new, None)
        importlib.reload(config)


def test_local_overrides_migrate_renamed_keys(tmp_path):
    """A stored config.local.json still carrying a renamed key is migrated
    for every RENAMED_KEYS entry; a present new key wins over the old one."""
    from faster_whisper_backend.settings import config_store
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"RECENT_TRANSCRIPTIONS_TTL_DAYS": 5,
                             "USE_AUTH_TOKEN": "hf_old", "HF_TOKEN": "hf_new"}),
                 encoding="utf-8")
    out = config_store.load_overrides(str(p))
    assert out.get("RECENT_TRANSCRIPTIONS_RETENTION_DAYS") == 5
    assert "RECENT_TRANSCRIPTIONS_TTL_DAYS" not in out
    assert out.get("HF_TOKEN") == "hf_new"


@pytest.mark.parametrize("old,new,value", [
    ("MAX_UPLOAD_BYTES", "MEDIA_MAX_BYTES", 200000000),
    ("URL_MEDIA_MAX_BYTES", "RETAINED_MEDIA_MAX_BYTES", 2000000000),
])
def test_local_overrides_survive_legacy_size_cap_keys(tmp_path, old, new, value):
    """The size caps that were folded into MEDIA_MAX_BYTES /
    RETAINED_MEDIA_MAX_BYTES shipped in config.json, so a stored file holds
    them. One such key must not make validation drop every other override."""
    from faster_whisper_backend.settings import config_store
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({old: value, "BEAM_SIZE": 5,
                             "DEFAULT_LANGUAGE": "de"}), encoding="utf-8")
    out = config_store.load_overrides(str(p))
    assert out.get(new) == value
    assert old not in out
    assert out.get("BEAM_SIZE") == 5 and out.get("DEFAULT_LANGUAGE") == "de"


def test_local_overrides_drop_removed_key_and_keep_siblings(tmp_path, capsys):
    """URL_MAX_BYTES has no successor (its shipped 0 = 'inherit' is not a
    legal MEDIA_MAX_BYTES): it is dropped with a stderr note, never mapped,
    and the rest of the file survives."""
    from faster_whisper_backend.settings import config_renames, config_store
    assert "URL_MAX_BYTES" in config_renames.REMOVED_KEYS
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"URL_MAX_BYTES": 0, "MAX_UPLOAD_BYTES": 200000000,
                             "BEAM_SIZE": 5}), encoding="utf-8")
    out = config_store.load_overrides(str(p))
    assert out == {"MEDIA_MAX_BYTES": 200000000, "BEAM_SIZE": 5}
    assert "URL_MAX_BYTES" in capsys.readouterr().err


def test_removed_keys_are_really_gone():
    """A REMOVED_KEYS entry that is still a config attribute (or also a
    rename source) would be silently discarded from every stored file."""
    from faster_whisper_backend.settings import config_renames
    from faster_whisper_backend.settings import schema as settings_schema
    for key in config_renames.REMOVED_KEYS:
        assert not hasattr(config, key), key
        assert key not in settings_schema.AdminConfig.model_fields, key
        assert key not in config_renames.RENAMED_KEYS, key


def test_legacy_upload_cap_env_var_is_aliased():
    """WHISPER_MAX_UPLOAD_BYTES in an existing .env keeps bounding uploads
    instead of silently giving way to the MEDIA_MAX_BYTES default."""
    from faster_whisper_backend.settings import config_renames
    env = {"WHISPER_MAX_UPLOAD_BYTES": "200000000",
           "WHISPER_URL_MEDIA_MAX_BYTES": "2000000000",
           "WHISPER_URL_MAX_BYTES": "0"}
    warns = config_renames.alias_env(env)
    assert env["WHISPER_MEDIA_MAX_BYTES"] == "200000000"
    assert env["WHISPER_RETAINED_MEDIA_MAX_BYTES"] == "2000000000"
    assert len(warns) == 2


def test_every_renamed_key_targets_a_live_field():
    """Each RENAMED_KEYS value must be a real config attribute, and no old
    spelling may still be one — otherwise the alias points into the void or
    the two names silently coexist."""
    from faster_whisper_backend.settings import config_renames
    for old, new in config_renames.RENAMED_KEYS.items():
        assert hasattr(config, new), new
        assert not hasattr(config, old), old


def test_per_model_env_override_invalid_keeps_stored_entry(monkeypatch):
    """One bad WHISPER_MODEL_OVERRIDE__ value must revert just that field,
    not wipe the whole stored entry from config.local.json for the process
    lifetime (every other env field reverts to its pre-env value)."""
    from faster_whisper_backend.settings import config_store
    # config.py re-imports load_overrides from config_store on reload and the
    # path default is bound at def time, so patch the function, not the path.
    monkeypatch.setattr(
        config_store, "load_overrides",
        lambda path=None: {"MODEL_OVERRIDES": {"TINY": {"BEAM_SIZE": 3, "VAD_FILTER": False}}})
    try:
        _reload_with_env(monkeypatch, WHISPER_MODEL_OVERRIDE__TINY__BEAM_SIZE="9999")
        assert config.MODEL_OVERRIDES["TINY"] == {"BEAM_SIZE": 3, "VAD_FILTER": False}
        # The pre-env snapshot must not share the mutated nested entry dict.
        assert config._ENV_PRE["MODEL_OVERRIDES"]["TINY"]["BEAM_SIZE"] == 3
        assert any("TINY" in m and "keeping the stored entry" in m
                   for m in config._ENV_WARNINGS), config._ENV_WARNINGS
        assert not any("was dropped" in m for m in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_per_model_env_override_invalid_unknown_field_keeps_stored_entry(monkeypatch):
    from faster_whisper_backend.settings import config_store
    monkeypatch.setattr(
        config_store, "load_overrides",
        lambda path=None: {"MODEL_OVERRIDES": {"TINY": {"BEAM_SIZE": 3, "VAD_FILTER": False}}})
    try:
        _reload_with_env(monkeypatch, WHISPER_MODEL_OVERRIDE__TINY__BOGUS="1")
        assert config.MODEL_OVERRIDES["TINY"] == {"BEAM_SIZE": 3, "VAD_FILTER": False}
        assert "BOGUS" not in config.MODEL_OVERRIDES["TINY"]
        assert any("TINY" in m and "keeping the stored entry" in m
                   for m in config._ENV_WARNINGS), config._ENV_WARNINGS
        assert not any("was dropped" in m for m in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_per_model_env_override_invalid_without_stored_entry_is_dropped(monkeypatch):
    """An env-only bad entry still never goes live."""
    try:
        _reload_with_env(monkeypatch, WHISPER_MODEL_OVERRIDE__TINY__BEAM_SIZE="9999")
        assert "TINY" not in config.MODEL_OVERRIDES
        assert any("TINY" in m for m in config._ENV_WARNINGS), config._ENV_WARNINGS
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def _point_overrides_at(monkeypatch, path):
    """Repoint config_store.load_overrides at `path` the way tests/conftest.py
    does: the module constant AND the def-time-bound default argument."""
    from faster_whisper_backend.settings import config_store
    monkeypatch.setattr(config_store, "OVERRIDES_PATH", str(path), raising=False)
    _defaults = list(config_store.load_overrides.__defaults__ or ())
    _defaults[-1] = str(path)
    monkeypatch.setattr(config_store.load_overrides, "__defaults__",
                        tuple(_defaults), raising=False)


def test_env_cross_field_inconsistent_with_local_override_is_reverted(tmp_path, monkeypatch):
    """env TARGET=28 + local MAX=27 is inconsistent against the EFFECTIVE
    config; the validators used to compare against _BASELINE (MAX 29.9) and
    let TARGET > MAX boot silently."""
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"CAPTURES_SAMPLE_MAX_DURATION_S": 27}), encoding="utf-8")
    _point_overrides_at(monkeypatch, p)
    try:
        _reload_with_env(monkeypatch, WHISPER_CAPTURES_PROPOSER_TARGET_S="28")
        assert config.CAPTURES_PROPOSER_TARGET_S == 26.0
        assert config.CAPTURES_SAMPLE_MAX_DURATION_S == 27
        assert "CAPTURES_PROPOSER_TARGET_S" in config._ENV_REJECTED
        warns = [m for m in config._ENV_WARNINGS if "CAPTURES_" in m]
        assert warns and any("27" in m for m in warns), config._ENV_WARNINGS
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_env_cross_field_consistent_with_local_override_applies(tmp_path, monkeypatch):
    """env TARGET=29.95 + local MAX=30 is self-consistent and must apply; the
    stale-baseline comparison (MAX 29.9) used to mis-revert it."""
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"CAPTURES_SAMPLE_MAX_DURATION_S": 30}), encoding="utf-8")
    _point_overrides_at(monkeypatch, p)
    try:
        _reload_with_env(monkeypatch, WHISPER_CAPTURES_PROPOSER_TARGET_S="29.95")
        assert config.CAPTURES_PROPOSER_TARGET_S == 29.95
        assert config.CAPTURES_SAMPLE_MAX_DURATION_S == 30
        assert "CAPTURES_PROPOSER_TARGET_S" not in config._ENV_REJECTED
        assert not [m for m in config._ENV_WARNINGS if "CAPTURES_" in m], config._ENV_WARNINGS
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_explicitly_empty_env_pins_fields_where_empty_is_a_value(monkeypatch):
    """WHISPER_X="" is a real value for the readers that map "" to None / ""
    / [] / set() — the var controls the field, so /settings must badge it
    as env-pinned (and the hot-apply path must skip it) instead of letting
    an edit "work" until the next restart. A reader that treats "" as
    "keep current" (BEAM_SIZE) still does not pin."""
    from faster_whisper_backend.settings import config_store
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_NO_SPEECH_THRESHOLD="",
            WHISPER_PRELOAD_MODELS="",
            WHISPER_ALLOWED_MODELS="",
            WHISPER_DEFAULT_LANGUAGE="",
            WHISPER_CONVERT_QUANTIZATION="",
            WHISPER_BEAM_SIZE="",
        )
        assert config.NO_SPEECH_THRESHOLD is None
        assert config.PRELOAD_MODELS == []
        assert config.ALLOWED_MODELS == set()
        assert config.DEFAULT_LANGUAGE == ""
        assert config.CONVERT_QUANTIZATION == "float16"
        pinned = config_store.env_pinned_fields()
        for f in ("NO_SPEECH_THRESHOLD", "PRELOAD_MODELS", "ALLOWED_MODELS",
                  "DEFAULT_LANGUAGE", "CONVERT_QUANTIZATION"):
            assert f in pinned, f
        assert "BEAM_SIZE" not in pinned
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_env_supplied_override_with_unknown_rule_slug_is_rejected(monkeypatch):
    """The env entry points validate MODEL_OVERRIDES / OVERRIDE_PROFILES /
    CAPTURES_PIPELINE_RULES_EXCLUDE with the live slug set, like a /settings
    save: a typo'd slug is dropped with a warning instead of booting silently
    and doing nothing at runtime. A real slug still loads."""
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_MODEL_OVERRIDES=
            '{"tiny": {"PIPELINE_RULES_EXCLUDE": ["dictashion-map"]}}',
            WHISPER_OVERRIDE_PROFILES=
            '{"ok": {"PIPELINE_RULES_EXCLUDE": ["strip-stray-symbols"]}}',
            WHISPER_CAPTURES_PIPELINE_RULES_EXCLUDE="no-such-rule",
        )
        assert "tiny" not in config.MODEL_OVERRIDES
        assert config.OVERRIDE_PROFILES["ok"]["PIPELINE_RULES_EXCLUDE"] == [
            "strip-stray-symbols"]
        assert "no-such-rule" not in config.CAPTURES_PIPELINE_RULES_EXCLUDE
        assert "CAPTURES_PIPELINE_RULES_EXCLUDE" in config._ENV_REJECTED
        assert any("dictashion-map" in m for m in config._ENV_WARNINGS)
        assert any("no-such-rule" in m for m in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# ---------------------------------------------------------------------------
# Renamed keys / slugs on every env entry point; pre-env validation errors
# ---------------------------------------------------------------------------

def test_env_rules_that_break_a_stored_exclude_do_not_revert_every_scalar(monkeypatch):
    """A WHISPER_PIPELINE_RULES without the rules CAPTURES_PIPELINE_RULES_
    EXCLUDE names fails validation with every env scalar reverted. That error
    used to fail each isolation pass too, so every env scalar (the cookie
    flag included) was reverted and blamed for it."""
    with open(os.path.join(config._REPO_DIR, "config.json"), encoding="utf-8") as f:
        rules = json.load(f)["PIPELINE_RULES"]
    one = [r for r in rules if r["name"] == "strip-stray-symbols"]
    try:
        _reload_with_env(monkeypatch, WHISPER_PIPELINE_RULES=json.dumps(one),
                         WHISPER_BEAM_SIZE="3",
                         WHISPER_SESSION_COOKIE_SECURE="true")
        assert config.BEAM_SIZE == 3
        assert config.SESSION_COOKIE_SECURE is True
        assert "BEAM_SIZE" not in config._ENV_REJECTED
        assert "SESSION_COOKIE_SECURE" not in config._ENV_REJECTED
        culprit = [m for m in config._ENV_WARNINGS
                   if "CAPTURES_PIPELINE_RULES_EXCLUDE" in m]
        assert len(culprit) == 1, config._ENV_WARNINGS
        assert "independently of the WHISPER_*" in culprit[0]
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_env_scalar_still_reverted_next_to_a_pre_env_error(monkeypatch):
    with open(os.path.join(config._REPO_DIR, "config.json"), encoding="utf-8") as f:
        rules = json.load(f)["PIPELINE_RULES"]
    one = [r for r in rules if r["name"] == "strip-stray-symbols"]
    try:
        _reload_with_env(monkeypatch, WHISPER_PIPELINE_RULES=json.dumps(one),
                         WHISPER_BEAM_SIZE="9999",
                         WHISPER_SESSION_COOKIE_SECURE="true")
        assert "BEAM_SIZE" in config._ENV_REJECTED
        assert config.SESSION_COOKIE_SECURE is True
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_env_captures_exclude_renamed_slug_keeps_working(monkeypatch):
    try:
        _reload_with_env(monkeypatch,
                         WHISPER_CAPTURES_PIPELINE_RULES_EXCLUDE="dictation-map")
        assert config.CAPTURES_PIPELINE_RULES_EXCLUDE == {"de-dictation-map"}
        assert "CAPTURES_PIPELINE_RULES_EXCLUDE" not in config._ENV_REJECTED
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_per_model_env_renamed_slug_keeps_the_whole_entry(monkeypatch):
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_MODEL_OVERRIDE__TINY__PIPELINE_RULES_EXCLUDE="dictation-map",
            WHISPER_MODEL_OVERRIDE__TINY__SEGMENT_MAX_WORDS_PER_SEC="4")
        assert config.MODEL_OVERRIDES["TINY"] == {
            "PIPELINE_RULES_EXCLUDE": ["de-dictation-map"],
            "SEGMENT_MAX_WORDS_PER_S": 4.0}
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_env_json_bundles_migrate_renamed_keys_and_locks(monkeypatch):
    """WHISPER_OVERRIDE_PROFILES / WHISPER_MODEL_OVERRIDES bundles are
    extra="forbid": one pre-rename key used to drop the whole env value."""
    bundle = {"STREAMING_IDLE_TIMEOUT_SEC": 30,
              "locks": ["STREAMING_IDLE_TIMEOUT_SEC"]}
    try:
        _reload_with_env(monkeypatch,
                         WHISPER_OVERRIDE_PROFILES=json.dumps({"x": bundle}),
                         WHISPER_MODEL_OVERRIDES=json.dumps(
                             {"TINY": {"SEGMENT_MAX_WORDS_PER_SEC": 4}}))
        assert config.OVERRIDE_PROFILES["x"]["STREAMING_IDLE_TIMEOUT_S"] == 30
        assert config.OVERRIDE_PROFILES["x"]["locks"] == ["STREAMING_IDLE_TIMEOUT_S"]
        assert config.MODEL_OVERRIDES["TINY"]["SEGMENT_MAX_WORDS_PER_S"] == 4
        assert any("renamed field" in m and "STREAMING_IDLE_TIMEOUT_SEC" in m
                   for m in config._ENV_WARNINGS), config._ENV_WARNINGS
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_alias_env_ignores_the_old_name_when_either_new_spelling_is_set():
    """A stale WHISPER_USE_AUTH_TOKEN must not become WHISPER_HF_TOKEN when the
    operator moved to WHISPER_HF_TOKEN_FILE: the plain var would then make the
    _FILE indirection skip the file."""
    from faster_whisper_backend.settings import config_renames
    env = {"WHISPER_USE_AUTH_TOKEN": "a", "WHISPER_HF_TOKEN_FILE": "/p"}
    warns = config_renames.alias_env(env)
    assert "WHISPER_HF_TOKEN" not in env
    assert any("WHISPER_USE_AUTH_TOKEN is ignored" in w for w in warns)
    env = {"WHISPER_USE_AUTH_TOKEN_FILE": "/old", "WHISPER_HF_TOKEN": "b"}
    config_renames.alias_env(env)
    assert "WHISPER_HF_TOKEN_FILE" not in env
    # Neither new spelling set: both old spellings still alias.
    env = {"WHISPER_USE_AUTH_TOKEN": "a", "WHISPER_USE_AUTH_TOKEN_FILE": "/old"}
    config_renames.alias_env(env)
    assert env["WHISPER_HF_TOKEN"] == "a" and env["WHISPER_HF_TOKEN_FILE"] == "/old"


def test_per_model_env_override_outside_a_stored_allowlist_is_dropped(monkeypatch):
    """An env per-model entry for a model the stored ALLOWED_MODELS excludes
    used to boot silently and make every later per-model save 422."""
    from faster_whisper_backend.settings import config_store
    monkeypatch.setattr(config_store, "load_overrides",
                        lambda path=None: {"ALLOWED_MODELS": {"large-v3"}})
    try:
        _reload_with_env(monkeypatch, WHISPER_MODEL_OVERRIDE__TINY__BEAM_SIZE="3")
        assert "TINY" not in config.MODEL_OVERRIDES
        assert any("TINY" in m and "ALLOWED_MODELS" in m
                   for m in config._ENV_WARNINGS), config._ENV_WARNINGS
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_per_model_env_values_are_recorded_for_the_save_path(monkeypatch):
    try:
        _reload_with_env(monkeypatch, WHISPER_MODEL_OVERRIDE__TINY__BEAM_SIZE="3")
        assert config._ENV_OVERRIDE_VALUES == {"TINY": {"BEAM_SIZE": 3}}
    finally:
        monkeypatch.undo()
        importlib.reload(config)
