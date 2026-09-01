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

import config


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
    # dir. See also tests/test_data_dir.py for the end-to-end env → path
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
               CAPTURES_PIPELINE_RULES_EXCLUDE=["dictation-map"])
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    d = config._load_defaults()
    assert d["ALLOWED_MODELS"] == {"large-v2", "large-v3"}
    assert isinstance(d["ALLOWED_MODELS"], set)
    assert isinstance(d["CAPTURES_PIPELINE_RULES_EXCLUDE"], set)


def test_baseline_comes_from_config_json():
    # _BASELINE (what "↺ Reset to default" reverts to) must equal the values in
    # config.json, with the same set-coercion + {REPO_DIR} resolution applied.
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
# Single-source-of-truth invariant: every AdminConfig field is env-configurable
# ---------------------------------------------------------------------------

def test_every_admin_field_is_env_mapped():
    # ENV_VAR_MAPPING is the source of truth driving config.py's schema loop,
    # the WebUI "env-pinned" badge, and env > GUI precedence. Every editable
    # AdminConfig field MUST be present (and vice-versa) or it silently loses
    # env-configurability / badging. This guards against future drift.
    import config_store as cs
    fields = set(cs.AdminConfig.model_fields)
    mapped = set(cs.ENV_VAR_MAPPING)
    assert fields == mapped, (
        f"missing from ENV_VAR_MAPPING: {sorted(fields - mapped)}; "
        f"mapping entries not in schema: {sorted(mapped - fields)}")


def test_env_var_names_are_unique():
    import config_store as cs
    names = list(cs.ENV_VAR_MAPPING.values())
    assert len(names) == len(set(names)), "duplicate WHISPER_* env var names"


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
            WHISPER_CAPTURES_PIPELINE_RULES_EXCLUDE="x,y",
        )
        assert config.ALLOWED_MODELS == {"a", "b"}
        assert config.CAPTURES_PIPELINE_RULES_EXCLUDE == {"x", "y"}
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
    import config_store
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
    import config_store
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
    import config_store
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
    import config_store
    assert config._overrides_path == config_store.OVERRIDES_PATH

def test_legacy_data_dir_root_db_warns_when_ignored(tmp_path):
    """Pre-db-layout compose installs kept SQLite stores at the data-dir
    ROOT; the upgrade note relies on this probe catching them too."""
    import config as cfg
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
