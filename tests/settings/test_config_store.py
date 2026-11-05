"""Tests for config_store + the settings schema: AdminConfig field bounds &
validators, ModelOverride, normalize_tags, the overrides load/save layer (incl.
its save lock) and the small helpers. The atomic writer is covered in
tests/core/test_atomic_json.py, factory-rule round-trips in
test_factory_rules.py.
"""

import json
import os
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from faster_whisper_backend.pipeline import regex_guard
from faster_whisper_backend.settings import config as cs_config
from faster_whisper_backend.settings import config_store as cs
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.core import atomic_json
from tests.settings._rules_helpers import (
    load_saved,
    map_rule as _map_rule,
    old_strip_whisper_punctuation as _old_strip_whisper_punctuation,
    regex_list_rules,
    regex_rule as _regex,
    seed_local as _seed_local,
    terminal_rule as _terminal,
    validate_rules,
)


def _ok(**fields):
    return settings_schema.AdminConfig.model_validate(fields)


def _bad(**fields):
    with pytest.raises(ValidationError):
        settings_schema.AdminConfig.model_validate(fields)


def _write_json(p, doc):
    Path(p).write_text(json.dumps(doc), encoding="utf-8")


def _read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


@pytest.fixture
def local_json(tmp_path):
    return str(tmp_path / "config.local.json")


def test_unknown_key_rejected():
    _bad(NOT_A_REAL_FIELD=1)


def test_empty_payload_ok():
    m = _ok()
    assert m.BEAM_SIZE is None


@pytest.mark.parametrize("field,lo,hi", [
    ("MAX_LOADED_MODELS", 1, 8),
    ("MODEL_IDLE_TIMEOUT_S", 0, 86400),
    ("BEAM_SIZE", 1, 20),
    ("BEST_OF", 1, 20),
    ("VAD_MIN_SILENCE_MS", 0, 10000),
    ("VAD_SPEECH_PAD_MS", 0, 2000),
    ("NO_REPEAT_NGRAM_SIZE", 0, 10),
    ("LANGUAGE_DETECTION_SEGMENTS", 1, 10),
    ("CPU_THREADS", 0, 128),
    ("NUM_WORKERS", 1, 8),
    ("DEVICE_INDEX", 0, 15),
    ("LOG_BACKUP_COUNT", 1, 100),
    ("LOG_VIEWER_INITIAL_LINES", 10, 100_000),
    ("LOG_VIEWER_DOM_MAX", 0, 1_000_000),
    ("LOG_SEGMENT_ROWS_MAX", 0, 100_000),
    ("LOG_SEGMENT_ROWS_SHOWN", 1, 1000),
    ("LOG_RECEIPT_HOLD_S", 5, 3600),
    ("SERVER_PORT", 1, 65535),
    ("SERVER_WORKERS", 1, 8),
    ("REPORTS_MAX", 10, 100_000),
    ("REPORTS_RETENTION_DAYS", 0, 3650),
    ("RECENT_TRANSCRIPTIONS_MAX", 0, 100_000),
    ("JOBS_TTL_S", 600, 2_592_000),
    ("JOBS_MAX_ROWS", 10, 100_000),
    ("JOBS_RATE_PER_MIN", 0, 100_000),
    ("STATS_RECENT_TRANSCRIPTIONS_COUNT", 1, 100),
    ("CAPTURES_MAX", 10, 1_000_000),
    ("CAPTURES_MAX_MB", 1, 10_000_000),
    ("LOG_MAX_BYTES", 1024 * 1024, 1024 * 1024 * 1024),
])
def test_int_bounds(field, lo, hi):
    _ok(**{field: lo})
    _ok(**{field: hi})
    _bad(**{field: lo - 1})
    _bad(**{field: hi + 1})


@pytest.mark.parametrize("field,lo,hi", [
    ("VAD_THRESHOLD", 0.0, 1.0),
    ("NO_SPEECH_THRESHOLD", 0.0, 1.0),
    ("LOG_PROB_THRESHOLD", -10.0, 0.0),
    ("COMPRESSION_RATIO_THRESHOLD", 0.0, 10.0),
    ("PATIENCE", 0.5, 5.0),
    ("LENGTH_PENALTY", 0.1, 5.0),
    ("REPETITION_PENALTY", 0.5, 5.0),
    ("PROMPT_RESET_ON_TEMPERATURE", 0.0, 1.0),
    ("LANGUAGE_DETECTION_THRESHOLD", 0.0, 1.0),
    ("HALLUCINATION_SILENCE_THRESHOLD", 0.0, 60.0),
    ("CAPTURES_RECORDING_SAMPLE_RATE", 0.0, 1.0),
    # 600 is the cross-field ceiling (baseline MAX = 600), not the field's
    # own le=3600 bound — see test_recording_duration_true_upper_bound.
    ("CAPTURES_RECORDING_MIN_DURATION_S", 0.0, 600.0),
])
def test_float_bounds(field, lo, hi):
    _ok(**{field: lo})
    _ok(**{field: hi})
    _bad(**{field: lo - 0.1})
    _bad(**{field: hi + 0.1})


def test_zero_length_tail_min_words_is_off_or_at_least_two():
    # The help text promises a single zero-length last word is kept; 1 would
    # cut exactly that word.
    _ok(SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS=0)
    _ok(SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS=2)
    _ok(SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS=20)
    _bad(SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS=1)
    _bad(SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS=21)


def test_capture_max_duration_min_is_0_1():
    # Asymmetric: MIN allows 0.0 but MAX requires ge=0.1.
    # Pair with MIN=0 so the cross-field validator passes.
    _ok(CAPTURES_RECORDING_MAX_DURATION_S=0.1,
        CAPTURES_RECORDING_MIN_DURATION_S=0.0)
    # Paired with MIN=0 too: alone, the baseline MIN (1.0) > MAX would trip the
    # cross-field check and hide a loosened field bound.
    with pytest.raises(ValidationError) as ei:
        settings_schema.AdminConfig.model_validate(
            {"CAPTURES_RECORDING_MAX_DURATION_S": 0.0,
             "CAPTURES_RECORDING_MIN_DURATION_S": 0.0})
    assert [e["loc"] for e in ei.value.errors()] == [
        ("CAPTURES_RECORDING_MAX_DURATION_S",)]


def test_recording_duration_min_le_max():
    _ok(CAPTURES_RECORDING_MIN_DURATION_S=1.0,
        CAPTURES_RECORDING_MAX_DURATION_S=600.0)
    _bad(CAPTURES_RECORDING_MIN_DURATION_S=60.0,
         CAPTURES_RECORDING_MAX_DURATION_S=30.0)


def test_recording_duration_true_upper_bound():
    _ok(CAPTURES_RECORDING_MIN_DURATION_S=3600.0,
        CAPTURES_RECORDING_MAX_DURATION_S=3600.0)
    _bad(CAPTURES_RECORDING_MIN_DURATION_S=3600.1,
         CAPTURES_RECORDING_MAX_DURATION_S=3600.1)


def test_buffer_trim_keep_lt_trim():
    _ok(STREAMING_BUFFER_TRIM_S=20.0, STREAMING_BUFFER_TRIM_KEEP_S=5.0)
    _bad(STREAMING_BUFFER_TRIM_S=20.0, STREAMING_BUFFER_TRIM_KEEP_S=20.0)
    _bad(STREAMING_BUFFER_TRIM_S=20.0, STREAMING_BUFFER_TRIM_KEEP_S=25.0)
    # KEEP-only override above the baseline TRIM.
    base = getattr(cs_config, "_BASELINE", {})
    trim_default = float(base.get("STREAMING_BUFFER_TRIM_S", cs_config.STREAMING_BUFFER_TRIM_S))
    _bad(STREAMING_BUFFER_TRIM_KEEP_S=trim_default + 1.0)


def test_session_cookie_names_must_differ():
    _ok(SESSION_COOKIE_NAME="a_sess", SESSION_CSRF_COOKIE_NAME="a_csrf")
    _bad(SESSION_COOKIE_NAME="same", SESSION_CSRF_COOKIE_NAME="same")
    # Single override colliding with the effective (baseline) session name.
    base = getattr(cs_config, "_BASELINE", {})
    sess_default = base.get("SESSION_COOKIE_NAME", cs_config.SESSION_COOKIE_NAME)
    _bad(SESSION_CSRF_COOKIE_NAME=sess_default)


def test_prefixed_cookie_name_needs_the_secure_flag():
    # A browser drops a __Host- / __Secure- cookie set without Secure, so the
    # login answered 200 and the gate looped with no cookie and no error.
    # Refused on save only: at load a failure would drop every override.
    def save(**fields):
        return settings_schema.AdminConfig.model_validate(
            fields, context={"cookie_prefix_check": True})
    with pytest.raises(ValidationError):
        save(SESSION_COOKIE_NAME="__Host-s")
    with pytest.raises(ValidationError):
        save(SESSION_CSRF_COOKIE_NAME="__secure-c", SESSION_COOKIE_SECURE=False)
    save(SESSION_COOKIE_NAME="__Host-s", SESSION_COOKIE_SECURE=True)
    _ok(SESSION_COOKIE_NAME="__Host-s")


def test_silence_gates_must_fit_outer_on_save():
    # Outer 6000 with the default hard break 5000 broke after every dictated
    # sentence; inner 1500 vs the default outer 1200 stretched every commit.
    # Refused on save only: at load a failure would drop every override.
    def save(**fields):
        return settings_schema.AdminConfig.model_validate(
            fields, context={"silence_pair_check": frozenset(fields)})
    for bad in ({"STREAMING_VAD_OUTER_SILENCE_MS": 6000},
                {"STREAMING_VAD_INNER_SILENCE_MS": 1500},
                {"STREAMING_HARD_BREAK_SILENCE_MS": 1200}):
        with pytest.raises(ValidationError, match="does not fit"):
            save(**bad)
        _ok(**bad)
    save(STREAMING_VAD_OUTER_SILENCE_MS=6000, STREAMING_HARD_BREAK_SILENCE_MS=0)
    save(STREAMING_VAD_OUTER_SILENCE_MS=6000,
         STREAMING_HARD_BREAK_SILENCE_MS=7000)


@pytest.mark.parametrize("val", ["large-v2", "org/name", "a", "A1_.-"])
def test_model_id_valid(val):
    assert _ok(DEFAULT_MODEL=val).DEFAULT_MODEL == val


@pytest.mark.parametrize("val", ["/leading", "trailing/", "has space", "", "a/b/c", "x" * 97])
def test_model_id_invalid(val):
    _bad(DEFAULT_MODEL=val)


@pytest.mark.parametrize("val", ["", "de", "en", "yue"])
def test_default_language_valid(val):
    assert _ok(DEFAULT_LANGUAGE=val).DEFAULT_LANGUAGE == val


@pytest.mark.parametrize("val", ["DE", "d", "d1", "abcd"])
def test_default_language_invalid(val):
    _bad(DEFAULT_LANGUAGE=val)


def test_device_and_compute_literals():
    _ok(MODEL_DEVICE="cuda", MODEL_COMPUTE_TYPE="float16")
    _ok(MODEL_DEVICE="cpu", MODEL_COMPUTE_TYPE="int8")
    _bad(MODEL_DEVICE="rocm")
    _bad(MODEL_COMPUTE_TYPE="int4")


def test_compute_vs_convert_quant_literals():
    # MODEL_COMPUTE_TYPE is CT2's runtime set, incl. the "auto"/"default"
    # selectors; CONVERT_QUANTIZATION is the conversion set — concrete
    # precisions only, so auto/default are refused there.
    _ok(MODEL_COMPUTE_TYPE="auto")
    _ok(MODEL_COMPUTE_TYPE="int16")
    _ok(MODEL_COMPUTE_TYPE="int8_bfloat16")
    _bad(CONVERT_QUANTIZATION="auto")
    _bad(CONVERT_QUANTIZATION="default")
    assert _ok(CONVERT_QUANTIZATION="int16").CONVERT_QUANTIZATION == "int16"


def test_server_log_level_literal():
    _ok(SERVER_LOG_LEVEL="debug")
    _bad(SERVER_LOG_LEVEL="verbose")


def test_console_log_level_literal():
    _ok(CONSOLE_LOG_LEVEL="debug")
    # Python's own spelling is accepted and stored as the lowercase literal.
    assert _ok(CONSOLE_LOG_LEVEL=" WARNING ").CONSOLE_LOG_LEVEL == "warning"
    _bad(CONSOLE_LOG_LEVEL="verbose")


@pytest.mark.parametrize("val", ["", "org/repo", "org/repo:Q4_K_M",
                                 "tencent/HY-MT1.5-7B-GGUF:Q4_K_M",
                                 "org/" + "x" * 156])   # exactly max_length
def test_translation_model_ref_valid(val):
    assert _ok(TRANSLATION_DEFAULT_MODEL=val).TRANSLATION_DEFAULT_MODEL == val
    assert _ok(TRANSLATION_MODEL=val).TRANSLATION_MODEL == val


@pytest.mark.parametrize("val", ["no-slash-model:q4", "no-slash-model",
                                 "/leading", "org/repo:", "a/b:c:d",
                                 "has space/repo",
                                 # Well-formed but over max_length=160: a
                                 # slash-less string is rejected by the
                                 # pattern alone and never reaches the cap.
                                 "org/" + "x" * 200])
def test_translation_model_ref_invalid(val):
    _bad(TRANSLATION_DEFAULT_MODEL=val)
    _bad(TRANSLATION_MODEL=val)


@pytest.mark.parametrize("val", ["", "de", "en,de", "fr-CA", "en,pt-BR,uk",
                                 "yue"])
def test_translate_to_valid(val):
    assert _ok(TRANSLATE_TO=val).TRANSLATE_TO == val


@pytest.mark.parametrize("val", ["german", "DE", "en,", ",en", "en de",
                                 "en;de", "d"])
def test_translate_to_invalid(val):
    _bad(TRANSLATE_TO=val)


def test_translation_prompt_template_placeholders():
    # Empty = unset (family "custom" then has nothing to render — a runtime
    # concern, not a schema one).
    _ok(TRANSLATION_PROMPT_TEMPLATE="")
    _ok(TRANSLATION_PROMPT_TEMPLATE=(
        "Translate into {target_language}:\n\n{text}"))
    _bad(TRANSLATION_PROMPT_TEMPLATE="Translate into {target_language}: hi")
    _bad(TRANSLATION_PROMPT_TEMPLATE="Just do it: {text}")
    _bad(TRANSLATION_PROMPT_TEMPLATE="no placeholders at all")


def test_translation_list_fields_validate_entries():
    _ok(TRANSLATION_ALLOWED_MODELS=["org/repo", "org/repo:Q8_0"])
    _ok(TRANSLATION_PRELOAD_MODELS=["org/repo:Q4_K_M"])
    _bad(TRANSLATION_ALLOWED_MODELS=[""])
    _bad(TRANSLATION_ALLOWED_MODELS=["no-slash:q4"])
    _bad(TRANSLATION_PRELOAD_MODELS=["no-slash"])


def test_lowercase_wordlist_accepts_uppercase_umlauts():
    r = settings_schema.LowercaseWordlistRule(name="w", label="W", type="callback:lowercase-wordlist",
                                 pattern="x", wordlist=["Ärger", "Österreich", "Übung", "ärger", "Und"])
    assert r.wordlist == ["Ärger", "Österreich", "Übung", "ärger", "Und"]
    with pytest.raises(ValidationError):
        settings_schema.LowercaseWordlistRule(name="w", label="W", type="callback:lowercase-wordlist",
                                 pattern="x", wordlist=["Ärger!"])


def test_translation_bounds_and_literals():
    _ok(TRANSLATION_MAX_LOADED_MODELS=1)
    _ok(TRANSLATION_MAX_LOADED_MODELS=4)
    _bad(TRANSLATION_MAX_LOADED_MODELS=0)
    _bad(TRANSLATION_MAX_LOADED_MODELS=5)
    _ok(TRANSLATION_BATCH_SEGMENTS=1)
    _bad(TRANSLATION_BATCH_SEGMENTS=51)
    _ok(TRANSLATION_CONTEXT_SEGMENTS=0)
    _bad(TRANSLATION_CONTEXT_SEGMENTS=11)
    _ok(TRANSLATION_MAX_TARGETS=10)
    _bad(TRANSLATION_MAX_TARGETS=0)
    _ok(TRANSLATION_MODE="faithful")
    _bad(TRANSLATION_MODE="literal")
    _ok(TRANSLATION_PROMPT_FAMILY="gemma-translate")
    _bad(TRANSLATION_PROMPT_FAMILY="alpaca")
    _ok(TRANSLATION_DEVICE="auto")
    _bad(TRANSLATION_DEVICE="rocm")


def test_load_overrides_coerces_translation_allowed_models_to_set(tmp_path):
    out = load_saved(tmp_path, {"TRANSLATION_ALLOWED_MODELS": ["org/a", "org/b:Q4_0"]})
    assert out["TRANSLATION_ALLOWED_MODELS"] == {"org/a", "org/b:Q4_0"}
    assert isinstance(out["TRANSLATION_ALLOWED_MODELS"], set)


def test_convert_quantisation():
    for v in ["float32", "float16", "bfloat16", "int16",
              "int8", "int8_float32", "int8_float16", "int8_bfloat16"]:
        _ok(CONVERT_QUANTIZATION=v)
    # Empty string -> treated as unset (None = use the runtime default).
    assert _ok(CONVERT_QUANTIZATION="").CONVERT_QUANTIZATION is None
    _bad(CONVERT_QUANTIZATION="int4")


def test_temperature():
    _ok(TEMPERATURE="")
    _ok(TEMPERATURE="0,0.2,0.4,0.6,0.8,1.0")
    _ok(TEMPERATURE="0.8,0.2")          # descending allowed (order not enforced)
    _ok(TEMPERATURE="0.5 , 0.5")
    _bad(TEMPERATURE="1.1")             # out of range
    _bad(TEMPERATURE="abc")


def test_suppress_tokens():
    _ok(SUPPRESS_TOKENS="-1")
    _ok(SUPPRESS_TOKENS="1, 2 ,3")
    _ok(SUPPRESS_TOKENS="")
    _bad(SUPPRESS_TOKENS="1.5")
    _bad(SUPPRESS_TOKENS="x")


def test_allowed_hosts_ip_cidr():
    _ok(ADMIN_WEBUI_ALLOWED_HOSTS=["127.0.0.1", "::1", "192.168.1.0/24"])
    _ok(USER_WEBUI_ALLOWED_HOSTS=["10.0.0.0/8", "0.0.0.0/0", "::/0"])
    _bad(ADMIN_WEBUI_ALLOWED_HOSTS=["not-an-ip"])
    _bad(USER_WEBUI_ALLOWED_HOSTS=["example.com"])  # hostname is not an IP/CIDR


def test_allowed_hosts_rewrites_ipv4_mapped_entries(local_json):
    """auth/hosts folds a "::ffff:a.b.c.d" client to a.b.c.d, so a mapped
    entry copied from the "::" bind's log saved cleanly and then 403ed that
    client. It is stored as the IPv4 spelling; other entries stay as written."""
    from types import SimpleNamespace
    from faster_whisper_backend.auth.hosts import host_in_allowlist
    got = _ok(ADMIN_WEBUI_ALLOWED_HOSTS=[
        "::ffff:192.168.1.5", "::ffff:10.0.0.0/120", "::ffff:0:0/95",
        "::1", "fd00::/8", "192.168.2.0/24",
    ]).ADMIN_WEBUI_ALLOWED_HOSTS
    assert got == ["192.168.1.5", "10.0.0.0/24", "::ffff:0:0/95",
                   "::1", "fd00::/8", "192.168.2.0/24"]
    req = SimpleNamespace(client=SimpleNamespace(host="::ffff:192.168.1.5"))
    assert host_in_allowlist(req, got)
    req = SimpleNamespace(client=SimpleNamespace(host="::ffff:10.0.0.77"))
    assert host_in_allowlist(req, got)
    cs.save_overrides({"USER_WEBUI_ALLOWED_HOSTS": ["::ffff:192.168.1.5"]}, local_json)
    on_disk = _read_json(local_json)
    assert on_disk["USER_WEBUI_ALLOWED_HOSTS"] == ["192.168.1.5"]


def test_server_host_loose_charset():
    _ok(SERVER_HOST="0.0.0.0")
    _ok(SERVER_HOST="::")
    _ok(SERVER_HOST="my-host.local")
    _bad(SERVER_HOST="bad host")
    _bad(SERVER_HOST="has/slash")


def test_server_host_bracketed_ipv6_saves_bindable():
    # "[::]" is the URL spelling; uvicorn/getaddrinfo refuse it, so a save
    # would leave a server that cannot start after the offered restart.
    assert _ok(SERVER_HOST="[::]").SERVER_HOST == "::"
    assert _ok(SERVER_HOST="[::1]").SERVER_HOST == "::1"
    _bad(SERVER_HOST="[::")
    _bad(SERVER_HOST="[[::]]")


def test_log_file_rejects_unc_and_traversal():
    _ok(LOG_FILE="logs/whisper.log")
    _bad(LOG_FILE="\\\\server\\share\\x.log")  # UNC
    _bad(LOG_FILE="//server/share/x.log")       # posix UNC-ish
    _bad(LOG_FILE="../etc/passwd")              # .. segment
    _bad(LOG_FILE="logs/../../x")               # .. mid-path


@pytest.mark.parametrize("field", [
    "LOG_FILE", "CAPTURES_DIR", "CAPTURES_DB", "DOWNLOAD_ROOT",
    "CONVERTED_MODELS_DIR", "URL_MEDIA_DIR", "API_KEYS_DB", "SESSIONS_DB",
    "REPORTS_DB", "RECENT_TRANSCRIPTIONS_DB", "JOBS_DB",
    "STATS_SYSTEM_METRICS_DB", "USAGE_DB", "CLIENT_SETTINGS_DB"])
def test_path_settings_refuse_an_embedded_nul(field):
    """os / sqlite path calls raise ValueError on a NUL, which the OSError
    fallbacks miss: a saved LOG_FILE with one crash-looped the next boot in
    log_setup.install(), so the save refuses it."""
    _ok(**{field: "data/x.log"})
    with pytest.raises(ValidationError, match="NUL character"):
        _ok(**{field: "data/x\x00.log"})


def test_hf_token_refuses_an_embedded_nul():
    """os.environ raises ValueError on a NUL: a saved HF_TOKEN with one 500ed
    the save in apply_hot_changes and crash-looped the next boot in main.py."""
    _ok(HF_TOKEN="hf_abc")
    with pytest.raises(ValidationError, match="NUL character is not allowed in a token"):
        _ok(HF_TOKEN="hf_ab\x00c")


def test_hf_token_drops_surrounding_whitespace():
    """models.py sends HF_TOKEN as the explicit Hub token, which the hub does
    not clean: "hf_abc " became the illegal header "Bearer hf_abc " and every
    Hub model load failed. A blank token turns into "", which falls back to
    the service environment's HF_TOKEN."""
    assert _ok(HF_TOKEN=" hf_abc\n").HF_TOKEN == "hf_abc"
    assert _ok(HF_TOKEN=" \t\n").HF_TOKEN == ""


def test_cap_list_over_1000():
    _bad(ALLOWED_MODELS=[f"m{i}" for i in range(1001)])
    _ok(ALLOWED_MODELS=[f"m{i}" for i in range(1000)])


def test_normalize_tags_basic():
    assert settings_schema.normalize_tags(None) == []
    assert settings_schema.normalize_tags([]) == []
    assert settings_schema.normalize_tags(["B", "a", "a", " c "]) == ["a", "b", "c"]


def test_normalize_tags_drops_empty():
    assert settings_schema.normalize_tags(["", "   ", "ok"]) == ["ok"]


def test_normalize_tags_rejects_bad():
    with pytest.raises(ValueError):
        settings_schema.normalize_tags("notalist")
    with pytest.raises(ValueError):
        settings_schema.normalize_tags([123])
    with pytest.raises(ValueError):
        settings_schema.normalize_tags(["-leadinghyphen"])
    with pytest.raises(ValueError):
        settings_schema.normalize_tags(["x" * 33])


def _rl(*entries, **extra):
    return {"name": "rl", "label": "RL", "type": "regex-list",
            "entries": list(entries), **extra}


def test_pipeline_callback_map_skips_pattern_validation():
    rule = _map_rule("m", {"Komma": ","})
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].type == "callback:map"


def test_pipeline_deeply_nested_regex_is_invalid_not_a_crash(tmp_path):
    # sre's parser recurses per open group: '(' * 512 (within the pattern
    # cap) raised RecursionError out of model_validate — a bare 500 on a
    # save, and load_overrides raised instead of ignoring the file at boot.
    deep = regex_list_rules("deep", "(" * 512, "y")
    with pytest.raises(ValidationError, match="invalid regex"):
        _ok(PIPELINE_RULES=deep)
    p = _seed_local(tmp_path, {"PIPELINE_RULES": deep})
    assert cs.load_overrides(str(p)) == {}
    with pytest.raises(RuntimeError, match="invalid regex"):
        cs.load_factory_rules(str(p))


def _ok_on_save(**fields):
    """Validate as a SAVE would (guard_regex context) — runs the out-of-process
    regex probe (backref + catastrophic-backtracking guard)."""
    return settings_schema.AdminConfig.model_validate(fields, context={"guard_regex": True})


def test_pipeline_bad_backref_reported():
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=regex_list_rules("b", "(a)", r"\3"))
    assert "regex test failed" in str(ei.value)


def test_pipeline_catastrophic_regex_rejected_on_save(monkeypatch):
    # The structural screen refuses a catastrophic-backtracking pattern on save
    # before any timed probe runs; the shortened timeout only bounds a regression.
    monkeypatch.setattr(regex_guard, "_GUARD_TIMEOUT", 0.5)
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=regex_list_rules("boom", "(.*a)+$", ""))
    assert "catastrophic backtracking" in str(ei.value)


def test_pipeline_replacement_growth_rejected_when_fixture_never_matches():
    # The growth probe measures rx.sub() against a FIXED German-prose fixture,
    # so a pattern the fixture never matches scored a growth ratio of 1.0 no
    # matter how large its replacement. ("n", "n"*512) passed the guard and
    # then amplified a real transcript 512x per match on EVERY transcription
    # — two such entries allocate ~1 GB inside a single re.sub.
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=regex_list_rules("blow", "n", "n" * 512))
    assert "regex test failed" in str(ei.value)


def test_pipeline_backref_only_replacement_rejected():
    # The growth bound charges each group reference the shortest string the
    # pattern can match: ("(n+)", r"\1" * 256) amplifies exactly as hard as
    # ("n", "n" * 256), which the sibling test above rejects.
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=regex_list_rules("blow", "(n+)", "\\1" * 256))
    assert "regex test failed" in str(ei.value)


def test_pipeline_prefix_ambiguous_alternation_rejected():
    # A prefix-ambiguous repeated alternation splits one run of input many ways
    # and backtracks exponentially. (The full _nested_repetition family is
    # unit-tested in tests/pipeline/test_regex_guard.py.)
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=regex_list_rules("boom", "(n|d|nd)+#", "X"))
    assert "catastrophic backtracking" in str(ei.value)


def test_pipeline_backrefs_and_alternation_still_accepted():
    # Guard against over-correction: ordinary backreference replacements and
    # UNrepeated alternations are the bread and butter of these rules.
    _ok_on_save(PIPELINE_RULES=[
        _regex("decimal", pattern=r"(\d+),(\d+)", replacement=r"\1.\2"),
        _regex("anrede", pattern="(Herr|Frau) ", replacement=r"\1 "),
        _terminal()])


def test_pipeline_ordinary_expansion_still_accepted():
    # Counterpart to the replacement-growth rejections: a rule whose
    # replacement is longer than its match is normal and must keep validating.
    _ok_on_save(PIPELINE_RULES=regex_list_rules("expand", r"z\.B\.", "zum Beispiel"))


def test_pipeline_regex_guard_skipped_without_save_context(monkeypatch):
    # Load / diff validations (no guard_regex context) must NOT run the probe —
    # so a normal config load never spawns the helper and never hangs on a
    # stored pattern. A pattern that only the BACKTRACKING probe would flag
    # (compiles fine; pathological only against real input) validates cleanly.
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1

    monkeypatch.setattr(regex_guard, "validate", _spy)
    _ok(PIPELINE_RULES=regex_list_rules("b", "(.*a)+$", "x"))
    assert calls["n"] == 0


def test_pipeline_guard_scoped_to_guard_slugs():
    # guard_slugs narrows the probe to the rules a patch actually changed. A
    # rule the CURRENT guard refuses can sit on disk (saved before a guard
    # tightening — the load path never probes), and unscoped it 422'd every
    # user's save of ANY rule. Scoped to the untouched sibling, the save
    # passes; scoped to (or including) the bad rule itself, it still fails.
    bad = _regex("legacy-boom", pattern="(n|d|nd)+#", replacement="X")
    good = _regex("harmless", pattern="Komma", replacement=",")

    def _save(slugs):
        return validate_rules([bad, good, _terminal()],
                              guard_regex=True, guard_slugs=frozenset(slugs))

    _save({"harmless"})
    with pytest.raises(ValidationError) as ei:
        _save({"harmless", "legacy-boom"})
    assert "catastrophic backtracking" in str(ei.value)
    # No guard_slugs in the context -> unchanged full-list behaviour.
    with pytest.raises(ValidationError):
        _ok_on_save(PIPELINE_RULES=[bad, good, _terminal()])


def test_save_overrides_guard_slugs_reaches_the_validator(tmp_path):
    # Same scenario through the public entry point: the `guard_slugs` kwarg
    # must become the `guard_slugs` validation-context key the pipeline
    # validator reads, and omitting it must keep the full-list probe.
    bad = _regex("legacy-boom", pattern="(n|d|nd)+#", replacement="X")
    good = _regex("harmless", pattern="Komma", replacement=",")
    p = str(_seed_local(tmp_path, {"PIPELINE_RULES": [bad, good, _terminal()]}))

    edited = _regex("harmless", pattern="Komma", replacement=";")
    changed = cs.save_overrides({"PIPELINE_RULES": [bad, edited, _terminal()]},
                                p, guard_slugs=frozenset({"harmless"}))
    assert "PIPELINE_RULES" in changed
    on_disk = _read_json(p)
    assert on_disk["PIPELINE_RULES"][1]["entries"][0]["replacement"] == ";"

    with pytest.raises(ValidationError) as ei:
        cs.save_overrides({"PIPELINE_RULES": [bad, good, _terminal()]}, p,
                          guard_slugs=frozenset({"harmless", "legacy-boom"}))
    assert "catastrophic backtracking" in str(ei.value)
    with pytest.raises(ValidationError):
        cs.save_overrides({"PIPELINE_RULES": [bad, good, _terminal()]}, p)


def test_pipeline_guard_scoping_never_skips_compile_and_template_checks():
    # Scoping narrows only the out-of-process probe. A rule that fails the
    # ALWAYS-on in-process checks (bad backref template) must keep failing
    # even when guard_slugs points at a different rule.
    with pytest.raises(ValidationError) as ei:
        validate_rules([_regex("b", pattern="(a)", replacement=r"\3"),
                        _regex("harmless"), _terminal()],
                       guard_regex=True, guard_slugs=frozenset({"harmless"}))
    assert "regex test failed" in str(ei.value)


def test_pipeline_bad_backref_rejected_on_load_without_subprocess(monkeypatch):
    # A bad replacement backref must keep failing on EVERY path (the eager
    # in-process template parse) — a hand-edited config.local.json with \3
    # against one group has to fail-safe at LOAD, not load cleanly and then
    # raise re.error on every request at match time. And detecting it must
    # not need the subprocess helper.

    def _boom(*a, **k):
        raise AssertionError("subprocess guard must not run on load")

    monkeypatch.setattr(regex_guard, "validate", _boom)
    with pytest.raises(ValidationError) as ei:
        _ok(PIPELINE_RULES=regex_list_rules("b", "(a)", r"\3"))
    assert "regex test failed" in str(ei.value)


def test_pipeline_duplicate_slug():
    _bad(PIPELINE_RULES=[_regex("dup"), _regex("dup"), _terminal()])


def test_pipeline_terminal_must_be_last():
    _bad(PIPELINE_RULES=[_terminal(), _regex("after")])


def test_regex_list_validates_and_keeps_order():
    rule = _rl({"pattern": "a", "replacement": "b"},
               {"pattern": "b", "replacement": "c", "label": "x", "note": "n"})
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].type == "regex-list"
    assert [e.pattern for e in m.PIPELINE_RULES[0].entries] == ["a", "b"]


def test_regex_list_requires_pattern_per_entry():
    _bad(PIPELINE_RULES=[_rl({"replacement": "b"}), _terminal()])


def test_regex_list_entry_extra_forbid():
    _bad(PIPELINE_RULES=[_rl({"pattern": "a", "bogus": 1}), _terminal()])


def test_regex_list_optional_fields_default_and_survive_exclude_none():
    m = _ok(PIPELINE_RULES=[_rl({"pattern": "a"}), _terminal()])
    e = m.PIPELINE_RULES[0].entries[0]
    assert (e.replacement, e.label, e.note) == ("", "", "")
    # exclude_none must KEEP the "" defaults (they are "" not None).
    dumped = m.model_dump(exclude_none=True, mode="json")["PIPELINE_RULES"][0]["entries"][0]
    assert dumped == {"pattern": "a", "replacement": "", "label": "", "note": ""}


def test_regex_list_entry_bad_regex_reports_index():
    with pytest.raises(ValidationError) as ei:
        _ok(PIPELINE_RULES=[_rl({"pattern": "("}), _terminal()])
    assert "entry 0" in str(ei.value)


def test_regex_list_entry_conflicting_flags_reports_index():
    # re.compile raises a plain ValueError (not re.error) for conflicting
    # inline flags: the 422 said only "ASCII and UNICODE flags are
    # incompatible", naming no rule or entry.
    with pytest.raises(ValidationError) as ei:
        _ok(PIPELINE_RULES=[_rl({"pattern": "a"}, {"pattern": "(?a)(?u)x"}), _terminal()])
    assert "rule 0 ('rl') entry 1: invalid regex" in str(ei.value)


def test_map_meta_pruned_to_map_keys():
    rule = _map_rule("m", {"Komma": ","}, map_meta={"Komma": 5, "ghost": 9})
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].map_meta == {"Komma": 5}


def test_regex_list_nested_fixed_counts_refused_before_the_template_sub():
    # The in-process template check (`.sub(repl, "")`) runs on EVERY
    # validation path, LOAD included, before the out-of-process guard:
    # `(?:(?:a{0}){N}){N}` loops N*N times even on "", so 1e5 x 1e5 never
    # returned. The counts are refused match-free first. Run in a thread
    # so a regression fails here instead of hanging the suite.
    out = {}

    def _validate():
        try:
            _ok(PIPELINE_RULES=[_rl({"pattern": "(?:(?:a{0}){100000}){100000}",
                                     "replacement": "y"}), _terminal()])
            out["result"] = "accepted"
        except ValidationError as e:
            out["result"] = str(e)

    t = threading.Thread(target=_validate, daemon=True)
    t.start()
    t.join(5.0)
    assert not t.is_alive(), "template check hung on nested fixed counts"
    assert "nested fixed counts" in out["result"]


def test_nested_fixed_counts_refused_at_load_without_a_replacement():
    # The count screen runs before the empty-replacement skip: a "delete the
    # match" entry or a callback pattern row must fail at LOAD too (the
    # out-of-process guard runs only on a save), or it loops N*N on every request.
    pat = "(?:(?:a{0}){100000}){100000}"
    for rule in (
            _rl({"pattern": pat, "replacement": ""}),
            {"name": "dd", "label": "DD", "type": "callback:dedup",
             "pattern": pat}):
        with pytest.raises(ValidationError, match="nested fixed counts"):
            validate_rules([rule, _terminal()])


def test_map_key_collision_check_is_scoped_to_guard_slugs():
    """An untouched dictionary with colliding keys (it loads fine: load runs
    without the guard context) must not 422 a save scoped to another rule."""
    rule = _map_rule("m", {"Passwort": "x", "passwort": "y"})
    rules = [rule, _terminal()]
    validate_rules(rules, guard_regex=True, guard_slugs={"other"})
    for ctx in ({"guard_regex": True, "guard_slugs": {"m"}}, {"guard_regex": True}):
        with pytest.raises(ValidationError, match="collide when lowercased"):
            validate_rules(rules, **ctx)


def test_rule_scoped_to_every_whisper_language_validates():
    # The languages cap tracks the code table (100 with yue).
    langs = sorted(settings_schema.WHISPER_LANGUAGE_CODES)
    rule = _rl({"pattern": "a"}, languages=langs)
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].languages == langs


def test_default_language_must_be_a_whisper_code():
    for model in (settings_schema.AdminConfig, settings_schema.ModelOverride,
                  settings_schema.OverrideProfile):
        with pytest.raises(ValidationError, match="unknown language code"):
            model.model_validate({"DEFAULT_LANGUAGE": "xx"})
        for ok in ("yue", "de", ""):
            model.model_validate({"DEFAULT_LANGUAGE": ok})


def test_model_override_include_exclude_overlap():
    with pytest.raises(ValidationError):
        settings_schema.ModelOverride.model_validate({
            "PIPELINE_RULES_EXCLUDE": ["a"],
            "PIPELINE_RULES_INCLUDE": ["a"],
        })


def test_model_override_bounds_inherit_global():
    settings_schema.ModelOverride.model_validate({"BEAM_SIZE": 20})
    with pytest.raises(ValidationError):
        settings_schema.ModelOverride.model_validate({"BEAM_SIZE": 21})


def test_admin_extra_forbid_on_override():
    with pytest.raises(ValidationError):
        settings_schema.ModelOverride.model_validate({"NONSENSE": 1})


def test_empty_override_for_a_nonempty_str_field_means_no_override():
    # The /settings per-model pane seeds `+ override` with '' for string
    # kinds (REVISION has no global to seed from) and a cleared string widget
    # stores '': for a field needing >= 1 character '' means "no override"
    # (absent = inherit), not a 422 for the whole save.
    m = settings_schema.ModelOverride.model_validate(
        {"REVISION": "", "BGM_SEPARATION_UVR_MODEL": "", "BEAM_SIZE": 3})
    assert m.REVISION is None and m.BGM_SEPARATION_UVR_MODEL is None
    assert "REVISION" not in m.model_fields_set
    assert m.BEAM_SIZE == 3
    p = settings_schema.OverrideProfile.model_validate(
        {"BGM_SEPARATION_UVR_MODEL": ""})
    assert p.BGM_SEPARATION_UVR_MODEL is None
    m = settings_schema.ModelOverride.model_validate(
        {"REVISION": "main", "BGM_SEPARATION_UVR_MODEL": "UVR-MDX-NET-Inst_HQ_3"})
    assert (m.REVISION, m.BGM_SEPARATION_UVR_MODEL) == (
        "main", "UVR-MDX-NET-Inst_HQ_3")


def test_empty_override_keeps_an_explicit_empty_where_empty_is_valid():
    # Only fields where '' can never validate are dropped: an explicit ''
    # elsewhere still beats the inherited value (test_explicit_empty_overrides).
    m = settings_schema.ModelOverride.model_validate({"DEFAULT_LANGUAGE": ""})
    assert m.DEFAULT_LANGUAGE == "" and "DEFAULT_LANGUAGE" in m.model_fields_set


def test_no_orphan_overrides_fires_only_with_both():
    # Both present + non-empty allowlist + orphan -> reject.
    _bad(ALLOWED_MODELS=["a"], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    # Empty allowlist = anything goes -> skip check.
    _ok(ALLOWED_MODELS=[], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    # Only overrides present -> cross-check skipped.
    _ok(MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    _ok(ALLOWED_MODELS=["a", "b"], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})


def test_pipeline_rule_slugs_cross_check():
    rules = [_regex("known"), _terminal()]
    # Unknown slug in a per-model EXCLUDE -> reject (both keys present).
    _bad(PIPELINE_RULES=rules,
         MODEL_OVERRIDES={"m": {"PIPELINE_RULES_EXCLUDE": ["bogus"]}})
    _ok(PIPELINE_RULES=rules,
        MODEL_OVERRIDES={"m": {"PIPELINE_RULES_EXCLUDE": ["known"]}})
    # Only MODEL_OVERRIDES present -> skipped.
    _ok(MODEL_OVERRIDES={"m": {"PIPELINE_RULES_EXCLUDE": ["bogus"]}})


def test_pipeline_rule_slugs_from_context():
    ctx = {"canonical_slugs": {"known"}}
    with pytest.raises(ValidationError):
        settings_schema.AdminConfig.model_validate(
            {"MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["bogus"]}}}, context=ctx)
    with pytest.raises(ValidationError):
        settings_schema.AdminConfig.model_validate(
            {"OVERRIDE_PROFILES": {"p": {"PIPELINE_RULES_EXCLUDE": ["bogus"]}}}, context=ctx)
    with pytest.raises(ValidationError):
        settings_schema.AdminConfig.model_validate(
            {"CAPTURES_PIPELINE_RULES_EXCLUDE": ["bogus"]}, context=ctx)
    settings_schema.AdminConfig.model_validate(
        {"MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["known"]}}}, context=ctx)
    # An explicit PIPELINE_RULES in the payload wins over the context.
    with pytest.raises(ValidationError):
        settings_schema.AdminConfig.model_validate(
            {"PIPELINE_RULES": [_regex("known"), _terminal()],
             "MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["bogus"]}}},
            context={"canonical_slugs": {"bogus"}})


def test_load_overrides_missing_returns_empty(tmp_path):
    assert cs.load_overrides(str(tmp_path / "nope.json")) == {}


def test_load_overrides_corrupt_returns_empty(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ not json", encoding="utf-8")
    assert cs.load_overrides(str(p)) == {}


def test_load_overrides_non_object_returns_empty(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text("[1,2,3]", encoding="utf-8")
    assert cs.load_overrides(str(p)) == {}


@pytest.mark.parametrize("rules", [
    [{"name": ["x"], "type": "regex-list", "entries": []}],
    [{"name": "r", "type": "regex-list",
      "entries": [{"label": ["x"], "pattern": "a", "replacement": "b"}]}],
])
def test_load_overrides_hand_edited_unhashable_rule_fields_never_raise(tmp_path, rules):
    # The rename/upgrade migrations run before validation; a list where a
    # string belongs must end in the schema's rejection, not a TypeError.
    assert load_saved(tmp_path, {"PIPELINE_RULES": rules}) == {}


@pytest.mark.parametrize("rule", [
    _regex("big", pattern="a{4294967296}"),
    {"name": "dd", "label": "dd", "type": "callback:dedup", "pattern": "a{1,99999999999}"},
])
def test_overflowing_repeat_count_is_a_validation_error(tmp_path, rule):
    # re.compile raises OverflowError (not re.error) past MAXREPEAT. It escaped
    # pydantic raw: a 500 instead of a 422 on save, and a hand-edited file
    # broke load_overrides' never-raises promise, so the server could not boot.
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=[rule, _terminal()])
    assert "invalid regex" in str(ei.value)
    assert load_saved(tmp_path, {"PIPELINE_RULES": [rule, _terminal()]}) == {}


def test_override_models_apply_the_global_value_rules():
    for model in (settings_schema.ModelOverride, settings_schema.OverrideProfile):
        for bad in ({"TEMPERATURE": "banana"}, {"SUPPRESS_TOKENS": "x,y"},
                    {"SEGMENT_HEAD_ECHO_MIN_WORDS": 1},
                    {"SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS": 1}):
            with pytest.raises(ValidationError):
                model.model_validate(bad)
        model.model_validate({"TEMPERATURE": "0,0.2", "SEGMENT_HEAD_ECHO_MIN_WORDS": 2})


def test_suppress_tokens_ids_must_be_real_token_ids():
    """An id far past the vocab segfaults CTranslate2 on every decode; the
    global, per-model and profile validators all refuse it."""
    for model in (settings_schema.AdminConfig, settings_schema.ModelOverride,
                  settings_schema.OverrideProfile):
        for bad in ("100000000", "-1,100000000", "-2",
                    str(settings_schema.SUPPRESS_TOKEN_ID_MAX)):
            with pytest.raises(ValidationError):
                model.model_validate({"SUPPRESS_TOKENS": bad})
        model.model_validate({"SUPPRESS_TOKENS": "-1,50257"})
        model.model_validate({"SUPPRESS_TOKENS": ""})


def test_stored_out_of_range_suppress_tokens_is_dropped_not_fatal(tmp_path):
    out = load_saved(tmp_path, {
        "SERVER_PORT": 9000, "SUPPRESS_TOKENS": "100000000",
        "OVERRIDE_PROFILES": {"p": {"SUPPRESS_TOKENS": "-1,100000000",
                                    "BEAM_SIZE": 2}}})
    assert out["SERVER_PORT"] == 9000
    assert "SUPPRESS_TOKENS" not in out
    assert out["OVERRIDE_PROFILES"]["p"] == {"BEAM_SIZE": 2}


def test_load_overrides_unknown_key_ignored_whole_file(tmp_path):
    assert load_saved(tmp_path, {"BEAM_SIZE": 5, "BOGUS": 1}) == {}


def test_load_overrides_coerces_allowed_models_to_set(tmp_path):
    out = load_saved(tmp_path, {"ALLOWED_MODELS": ["a", "b"]})
    assert isinstance(out["ALLOWED_MODELS"], set)
    assert out["ALLOWED_MODELS"] == {"a", "b"}


def test_load_overrides_coerces_captures_excludes_to_set(tmp_path):
    # config.json default + ENV paths already yield a set (config._SET_FIELDS);
    # the config.local.json override path must match, or the pipeline engine's
    # _postprocess_text does `set | list` and raises TypeError on every captures consumer.
    out = load_saved(tmp_path, {"CAPTURES_PIPELINE_RULES_EXCLUDE": ["r1", "r2"]})
    assert isinstance(out["CAPTURES_PIPELINE_RULES_EXCLUDE"], set)
    assert out["CAPTURES_PIPELINE_RULES_EXCLUDE"] == {"r1", "r2"}


def test_save_overrides_rejects_unknown_slug_without_local_rules(local_json, monkeypatch):
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"known"})
    with pytest.raises(ValidationError):
        cs.save_overrides(
            {"MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["dictashion-map"]}}}, local_json)
    assert not os.path.exists(local_json)
    cs.save_overrides({"MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["known"]}}}, local_json)
    assert os.path.exists(local_json)
    # A stale slug on disk must not brick unrelated saves.
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"renamed"})
    cs.save_overrides({"BEAM_SIZE": 5}, local_json)
    assert _read_json(local_json)["BEAM_SIZE"] == 5


def test_empty_rule_list_leaves_no_valid_slug_on_save(local_json):
    """An explicit PIPELINE_RULES of [] read as "unknown, skip", so every
    EXCLUDE / INCLUDE ref saved with it dangled silently. On a save touching
    the rules or a ref it now means "no valid slug"; a load (no context)
    stays permissive, since a load failure drops every override."""
    bad = {"PIPELINE_RULES": [],
           "MODEL_OVERRIDES": {"small": {"PIPELINE_RULES_EXCLUDE": ["nope"]}}}
    with pytest.raises(ValidationError):
        settings_schema.AdminConfig.model_validate(
            bad, context={"empty_rules_known": True})
    with pytest.raises(ValidationError):
        settings_schema.AdminConfig.model_validate(
            {"PIPELINE_RULES": [], "CAPTURES_PIPELINE_RULES_EXCLUDE": ["nope"]},
            context={"empty_rules_known": True})
    settings_schema.AdminConfig.model_validate(bad)      # load: still skips
    # A partial validation with neither rules nor context still skips.
    _ok(MODEL_OVERRIDES={"small": {"PIPELINE_RULES_EXCLUDE": ["nope"]}})
    with pytest.raises(ValidationError):
        cs.save_overrides(bad, local_json)
    assert not os.path.exists(local_json)
    # A stored [] with a dangling ref does not refuse an unrelated save.
    _write_json(local_json, bad)
    cs.save_overrides({"BEAM_SIZE": 3}, local_json)
    assert cs.load_overrides(local_json)["BEAM_SIZE"] == 3


def test_validate_binding_checks_slugs_against_an_empty_live_rule_list(monkeypatch):
    """An explicit empty live rule list skipped the binding slug check (and
    the save fell back to factory slugs not in force): any typo saved."""
    monkeypatch.setattr(cs_config, "PIPELINE_RULES", [])
    with pytest.raises(ValueError, match="typo"):
        cs.validate_binding({"overrides": {"PIPELINE_RULES_EXCLUDE": ["typo"]}})
    assert cs._save_canonical_slugs() == set()
    # No list at all = unknown: skipped, as before.
    monkeypatch.setattr(cs_config, "PIPELINE_RULES", None)
    cs.validate_binding({"overrides": {"PIPELINE_RULES_EXCLUDE": ["typo"]}})


def test_stale_stored_slug_only_blocks_saves_of_its_own_key(tmp_path, monkeypatch):
    # The slug check covers only the keys a save submits, not the whole merged
    # document, so a stale exclude stored under MODEL_OVERRIDES must not 422
    # an OVERRIDE_PROFILES / captures save.
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"known"})
    p = _seed_local(tmp_path, {"MODEL_OVERRIDES": {"m": {
        "PIPELINE_RULES_EXCLUDE": ["gone"]}}})
    cs.save_overrides({"OVERRIDE_PROFILES": {"p": {"BEAM_SIZE": 2}}}, str(p))
    cs.save_overrides({"CAPTURES_PIPELINE_RULES_EXCLUDE": ["known"]}, str(p))
    on_disk = _read_json(p)
    assert on_disk["OVERRIDE_PROFILES"] == {"p": {"BEAM_SIZE": 2}}
    # A typo in the key a save DOES submit is still refused.
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValidationError, match="typo"):
        cs.save_overrides({"OVERRIDE_PROFILES": {"p": {
            "PIPELINE_RULES_EXCLUDE": ["typo"]}}}, str(p))
    with pytest.raises(ValidationError, match="typo"):
        cs.save_overrides({"MODEL_OVERRIDES": {"m": {
            "PIPELINE_RULES_EXCLUDE": ["typo"]}}}, str(p))
    assert p.read_text(encoding="utf-8") == before


def test_stale_stored_slug_only_blocks_saves_of_its_own_entry(tmp_path, monkeypatch):
    # The WebUI /state save and the profile rename send the WHOLE profiles
    # dict, so checking every entry of a submitted key let one stale stored
    # exclude in profile b 422 every edit of profile a.
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"known"})
    p = _seed_local(tmp_path, {"OVERRIDE_PROFILES": {
        "a": {"BEAM_SIZE": 2},
        "b": {"PIPELINE_RULES_EXCLUDE": ["gone"]}}})
    cs.save_overrides({"OVERRIDE_PROFILES": {
        "a": {"BEAM_SIZE": 3},
        "b": {"PIPELINE_RULES_EXCLUDE": ["gone"]}}}, str(p))
    on_disk = _read_json(p)
    assert on_disk["OVERRIDE_PROFILES"]["a"] == {"BEAM_SIZE": 3}
    # A typo in an entry the save changes or adds is still refused.
    before = p.read_text(encoding="utf-8")
    for profiles in (
            {"a": {"BEAM_SIZE": 3}, "b": {"PIPELINE_RULES_EXCLUDE": ["typo"]}},
            {"a": {"BEAM_SIZE": 3}, "b": {"PIPELINE_RULES_EXCLUDE": ["gone"]},
             "c": {"PIPELINE_RULES_EXCLUDE": ["typo"]}}):
        with pytest.raises(ValidationError, match="typo"):
            cs.save_overrides({"OVERRIDE_PROFILES": profiles}, str(p))
    assert p.read_text(encoding="utf-8") == before


def test_stale_stored_slug_does_not_block_edits_of_its_own_entry(tmp_path, monkeypatch):
    # The pages draw rows only for live rules and send an unseen stale slug
    # back unchanged, so a rename or any edit of the entry carrying it (or a
    # toggle in a captures list holding one) 422'd with no way to clear it.
    # Only the slugs a save adds are checked, as validate_binding does.
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"known"})
    p = _seed_local(tmp_path, {
        "OVERRIDE_PROFILES": {"a": {"BEAM_SIZE": 3},
                              "b": {"PIPELINE_RULES_EXCLUDE": ["gone"]}},
        "CAPTURES_PIPELINE_RULES_EXCLUDE": ["gone"]})
    cs.save_overrides({"OVERRIDE_PROFILES": {
        "a": {"BEAM_SIZE": 3},
        "b": {"PIPELINE_RULES_EXCLUDE": ["gone"], "BEAM_SIZE": 2}}}, str(p))
    cs.save_overrides({"OVERRIDE_PROFILES": {
        "a": {"BEAM_SIZE": 3},
        "c": {"PIPELINE_RULES_EXCLUDE": ["gone"], "BEAM_SIZE": 2}}}, str(p))
    cs.save_overrides({"CAPTURES_PIPELINE_RULES_EXCLUDE": ["gone", "known"]}, str(p))
    on_disk = _read_json(p)
    assert on_disk["OVERRIDE_PROFILES"]["c"] == {
        "PIPELINE_RULES_EXCLUDE": ["gone"], "BEAM_SIZE": 2}
    assert "b" not in on_disk["OVERRIDE_PROFILES"]
    assert on_disk["CAPTURES_PIPELINE_RULES_EXCLUDE"] == ["gone", "known"]
    # A new unknown slug is still refused, next to the stale one too.
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValidationError, match="typo"):
        cs.save_overrides({"OVERRIDE_PROFILES": {
            "a": {"BEAM_SIZE": 3},
            "c": {"PIPELINE_RULES_EXCLUDE": ["gone", "typo"]}}}, str(p))
    with pytest.raises(ValidationError, match="typo"):
        cs.save_overrides({"CAPTURES_PIPELINE_RULES_EXCLUDE": [
            "gone", "known", "typo"]}, str(p))
    assert p.read_text(encoding="utf-8") == before


def test_stored_top_level_guard_at_one_is_migrated_not_fatal(tmp_path, capsys):
    # The top-level segment guards started refusing 1 after releases that
    # stored it; one left in config.local.json dropped every override at boot
    # and 422'd every save, the failure the bundle migration already covers.
    p = _seed_local(tmp_path, {"SERVER_PORT": 9000,
                               "SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS": 1,
                               "SEGMENT_HEAD_ECHO_MIN_WORDS": 1})
    out = cs.load_overrides(str(p))
    assert out["SERVER_PORT"] == 9000
    assert out["SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS"] == 0
    assert out["SEGMENT_HEAD_ECHO_MIN_WORDS"] == 0
    assert "SEGMENT_HEAD_ECHO_MIN_WORDS=1" in capsys.readouterr().err
    cs.save_overrides({"BEST_OF": 3}, str(p))
    on_disk = _read_json(p)
    assert on_disk["SERVER_PORT"] == 9000 and on_disk["BEST_OF"] == 3
    assert on_disk["SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS"] == 0


def test_binding_save_keeps_a_stored_slug_a_rules_edit_removed(monkeypatch):
    # The binding drawer sends the whole overrides blob back, so a slug the
    # binding already stored (since removed from the rules) must not 422 an
    # edit of an unrelated field; a slug the save ADDS is still checked.
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"known"})
    previous = {"direct": {"PIPELINE_RULES_EXCLUDE": ["x"]}, "profiles": []}
    out = cs.validate_binding(
        {"overrides": {"PIPELINE_RULES_EXCLUDE": ["x"], "BEAM_SIZE": 3}},
        previous=previous)
    assert out["direct"] == {"PIPELINE_RULES_EXCLUDE": ["x"], "BEAM_SIZE": 3}
    with pytest.raises(ValueError, match="typo"):
        cs.validate_binding(
            {"overrides": {"PIPELINE_RULES_EXCLUDE": ["x", "typo"]}},
            previous=previous)
    # Without the stored binding the stale slug is refused as before.
    with pytest.raises(ValueError, match="'x'"):
        cs.validate_binding({"overrides": {"PIPELINE_RULES_EXCLUDE": ["x"]}})


@pytest.mark.parametrize("bad", [[], "", 0, False])
def test_binding_refuses_a_falsy_wrong_type_for_overrides(bad):
    # A falsy non-object must be refused like a non-empty one, not pass as
    # "empty" and wipe the key's stored direct overrides.
    with pytest.raises(ValueError, match="config.overrides must be a JSON object"):
        cs.validate_binding({"overrides": bad})


@pytest.mark.parametrize("bad", ["", 0, False, {}])
def test_binding_refuses_a_falsy_wrong_type_for_locks(bad):
    with pytest.raises(ValueError, match="locks"):
        cs.validate_binding({"overrides": {}, "locks": bad})


def test_binding_defaults_missing_or_null_overrides_and_locks_to_empty():
    for raw in ({}, {"overrides": None, "locks": None}, {"locks": []}):
        assert cs.validate_binding(raw)["direct"] == {}


def test_removing_the_local_rules_copy_refuses_refs_it_would_strand(
        tmp_path, monkeypatch):
    # Resetting PIPELINE_RULES makes the factory list canonical; a stored
    # exclude naming a local-only rule would then dangle silently (no page
    # draws a row for it), so the reset is refused instead and the file kept.
    factory = [{"name": "factory-rule"}, {"name": "trim-edges"}]
    monkeypatch.setattr(cs, "load_factory_rules", lambda path=None: factory)
    monkeypatch.setattr(cs, "_canonical_rule_slugs",
                        lambda: {"factory-rule", "trim-edges", "my-local-rule"})
    local = {
        "PIPELINE_RULES": [
            {"name": "my-local-rule", "label": "mine", "type": "regex-list",
             "entries": [{"pattern": "x", "replacement": "y"}]},
            {"name": "trim-edges", "label": "Trim edges", "type": "terminal"}],
        "MODEL_OVERRIDES": {"large-v3": {
            "PIPELINE_RULES_EXCLUDE": ["my-local-rule"]}},
    }
    p = _seed_local(tmp_path, local)
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValidationError, match="my-local-rule"):
        cs.save_overrides({"PIPELINE_RULES": None}, str(p))
    assert p.read_text(encoding="utf-8") == before
    # Without the stranded ref the reset goes through.
    local["MODEL_OVERRIDES"]["large-v3"]["PIPELINE_RULES_EXCLUDE"] = ["factory-rule"]
    _write_json(p, local)
    cs.save_overrides({"PIPELINE_RULES": None}, str(p))
    assert "PIPELINE_RULES" not in _read_json(p)


def test_removing_a_local_copy_of_a_split_card_writes_its_refs_as_the_successors(
        tmp_path):
    # A copy from before June still runs strip-whisper-punctuation, and the
    # excludes name it. The factory list has its two successor cards, which
    # the next load maps the slug to: the clear must not 422 on it.
    split = ["strip-trailing-period", "strip-auto-punctuation"]
    rules = [r for r in cs.load_factory_rules() if r["name"] not in split]
    rules.insert(-1, _old_strip_whisper_punctuation())
    p = _seed_local(tmp_path, {
        "PIPELINE_RULES": rules,
        "CAPTURES_PIPELINE_RULES_EXCLUDE": ["strip-whisper-punctuation"],
        "MODEL_OVERRIDES": {"large-v3": {
            "PIPELINE_RULES_EXCLUDE": ["strip-whisper-punctuation"], "BEAM_SIZE": 3}},
        "OVERRIDE_PROFILES": {"p": {"PIPELINE_RULES_EXCLUDE": ["strip-whisper-punctuation"]}},
    })
    cs.save_overrides({"PIPELINE_RULES": None}, str(p))
    on_disk = _read_json(p)
    assert "PIPELINE_RULES" not in on_disk
    assert sorted(on_disk["CAPTURES_PIPELINE_RULES_EXCLUDE"]) == sorted(split)
    assert on_disk["MODEL_OVERRIDES"]["large-v3"] == {
        "PIPELINE_RULES_EXCLUDE": split, "BEAM_SIZE": 3}
    assert on_disk["OVERRIDE_PROFILES"]["p"]["PIPELINE_RULES_EXCLUDE"] == split


def test_save_overrides_roundtrip_and_merge(local_json):
    changed = cs.save_overrides({"BEAM_SIZE": 5}, local_json)
    assert changed == {"BEAM_SIZE": 5}
    cs.save_overrides({"BEST_OF": 3}, local_json)
    on_disk = _read_json(local_json)
    assert on_disk["BEAM_SIZE"] == 5 and on_disk["BEST_OF"] == 3


def test_save_overrides_none_removes(local_json):
    cs.save_overrides({"BEAM_SIZE": 5, "BEST_OF": 3}, local_json)
    changed = cs.save_overrides({"BEAM_SIZE": None}, local_json)
    assert "BEAM_SIZE" in changed and changed["BEAM_SIZE"] is None
    on_disk = _read_json(local_json)
    assert "BEAM_SIZE" not in on_disk and on_disk["BEST_OF"] == 3


def test_save_overrides_changed_excludes_unchanged(local_json):
    cs.save_overrides({"BEAM_SIZE": 5}, local_json)
    changed = cs.save_overrides({"BEAM_SIZE": 5}, local_json)
    assert changed == {}


def test_save_factory_rules_preserves_sibling_defaults(tmp_path):
    # config.json holds ALL factory defaults, not just PIPELINE_RULES, so a
    # rules "Promote to factory" must read-modify-write, not clobber the
    # sibling scalar defaults.
    p = str(tmp_path / "config.json")
    _write_json(p, {"schema_version": 1, "DEFAULT_MODEL": "keep-me", "BEST_OF": 7,
                    "PIPELINE_RULES": [{"name": "t", "label": "T", "type": "terminal"}]})
    cs.save_factory_rules([{"name": "trim", "label": "Trim", "type": "terminal"}], p)
    on_disk = _read_json(p)
    assert on_disk["DEFAULT_MODEL"] == "keep-me"
    assert on_disk["BEST_OF"] == 7
    assert [r["name"] for r in on_disk["PIPELINE_RULES"]] == ["trim"]


def test_sample_sizing_absent_field_uses_baseline_not_live_override(monkeypatch):
    # _validate_sample_sizing must fall back to config._BASELINE (the in-repo
    # default) for an absent field, NOT the live attribute: that carries the
    # applied override at save time but the bare default at load time, so a
    # save passed and the next restart's load failed it and silently dropped
    # EVERY override on disk.
    monkeypatch.setattr(cs_config, "CAPTURES_PROPOSER_TARGET_S", 5.0, raising=False)
    assert cs_config._BASELINE["CAPTURES_PROPOSER_TARGET_S"] > 6.0

    # TARGET absent: the 26.0 baseline is outside 1..6; the live 5.0 would pass.
    _bad(CAPTURES_SAMPLE_MIN_DURATION_S=1.0, CAPTURES_SAMPLE_MAX_DURATION_S=6.0)


@pytest.mark.parametrize("content", [
    b'{"BEAM_SIZE": 7, "DEFAULT_LANGUAGE": "de",}',   # trailing comma
    b'{"DEFAULT_LANGUAGE": "de", "DEFAULT_PROMPT": "Gr\xfc\xdfe"}',  # cp1252
    b'["DEFAULT_LANGUAGE", "de"]',                    # valid JSON, not an object
], ids=["malformed", "not-utf8", "not-object"])
def test_save_overrides_refuses_unreadable_existing_file(tmp_path, content):
    """Boot ignores an unreadable file, so the WebUI shows defaults; a save
    merged onto {} then replaced it, wiping every hand-edited override. It
    must 422 like a schema-invalid file and leave the bytes alone."""
    p = tmp_path / "config.local.json"
    p.write_bytes(content)
    with pytest.raises(ValidationError) as ei:
        cs.save_overrides({"BEAM_SIZE": 5}, str(p))
    assert p.read_bytes() == content
    [err] = settings_schema.format_validation_errors(ei.value)
    assert "config.local.json cannot be read" in err["msg"]


def test_save_overrides_empty_existing_file_is_the_empty_case(tmp_path):
    p = tmp_path / "config.local.json"
    p.write_text("\n", encoding="utf-8")
    cs.save_overrides({"BEAM_SIZE": 7}, str(p))
    assert _read_json(p) == {"BEAM_SIZE": 7}


def test_load_overrides_non_utf8_file_is_ignored_not_raised(tmp_path):
    """load_overrides NEVER raises: config imports it with only `except
    ImportError` around it, so a hand edit saved as cp1252 (UnicodeDecodeError
    is a ValueError, not a JSONDecodeError) used to stop the server booting."""
    p = tmp_path / "config.local.json"
    p.write_bytes(b'{"DEFAULT_PROMPT": "Gr\xfc\xdfe"}')
    assert cs.load_overrides(str(p)) == {}


def test_overrides_file_with_utf8_bom_is_read_and_saved_onto(tmp_path):
    """Notepad / PowerShell 5.1 `Set-Content -Encoding UTF8` write a BOM:
    the file must still load and a save must merge onto it, not 422."""
    p = tmp_path / "config.local.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"DEFAULT_PROMPT": "x"}).encode())
    assert cs.load_overrides(str(p)) == {"DEFAULT_PROMPT": "x"}
    cs.save_overrides({"BEAM_SIZE": 7}, str(p))
    assert _read_json(p) == {"DEFAULT_PROMPT": "x", "BEAM_SIZE": 7}


def test_save_overrides_checks_cross_field_pairs_against_env_pins(local_json,
                                                                  monkeypatch):
    """A save must see the env-pinned sibling that will be in force: with env
    KEEP_S=12 pinned, TRIM_S=11 passed against the baseline KEEP_S=10, ran
    inconsistent, and the next restart reverted the env value."""
    monkeypatch.setattr(cs, "env_pinned_fields", lambda: {
        "STREAMING_BUFFER_TRIM_KEEP_S": "WHISPER_STREAMING_BUFFER_TRIM_KEEP_S"})
    monkeypatch.setattr(cs_config, "STREAMING_BUFFER_TRIM_KEEP_S", 12.0)
    with pytest.raises(ValidationError):
        cs.save_overrides({"STREAMING_BUFFER_TRIM_S": 11}, local_json)
    assert not os.path.exists(local_json)
    cs.save_overrides({"STREAMING_BUFFER_TRIM_S": 13}, local_json)   # consistent: saved
    # The import-time path (no context) still checks the bare baseline.
    settings_schema.AdminConfig.model_validate({"STREAMING_BUFFER_TRIM_S": 11})


def test_save_overrides_refuses_a_file_only_an_env_pin_makes_valid(local_json,
                                                                   monkeypatch):
    """load_overrides validates the file without the env context and drops
    EVERY override when it fails (at boot and in the hot-apply right after a
    save). A save consistent only thanks to an env-pinned sibling must 422
    instead of being written and then thrown away with the rest of the file."""
    monkeypatch.setattr(cs, "env_pinned_fields", lambda: {
        "MEDIA_MAX_BYTES": "WHISPER_MEDIA_MAX_BYTES",
        "STREAMING_BUFFER_TRIM_KEEP_S": "WHISPER_STREAMING_BUFFER_TRIM_KEEP_S"})
    monkeypatch.setattr(cs_config, "MEDIA_MAX_BYTES", 2_000_000_000)
    monkeypatch.setattr(cs_config, "STREAMING_BUFFER_TRIM_KEEP_S", 5.0)
    with pytest.raises(ValidationError):
        cs.save_overrides({"MAX_REQUEST_BYTES": 3_000_000_000}, local_json)
    with pytest.raises(ValidationError):
        cs.save_overrides({"STREAMING_BUFFER_TRIM_S": 8}, local_json)
    assert not os.path.exists(local_json)
    # What a save does accept round-trips through the bare load.
    cs.save_overrides({"DEFAULT_PROMPT": "hello", "STREAMING_BUFFER_TRIM_S": 13}, local_json)
    assert cs.load_overrides(local_json) == {"DEFAULT_PROMPT": "hello",
                                             "STREAMING_BUFFER_TRIM_S": 13.0}


def test_save_overrides_prefixed_cookie_name_with_env_pinned_secure(tmp_path, local_json,
                                                                  monkeypatch):
    """An env-pinned SESSION_COOKIE_SECURE=True counts for the save-only
    __Host- prefix check: the bare re-check an env pin triggers must not fall
    back to the baseline Secure=False."""
    monkeypatch.setattr(cs, "_env_effective_values",
                        lambda: {"SESSION_COOKIE_SECURE": True})
    cs.save_overrides({"SESSION_COOKIE_NAME": "__Host-s"}, local_json)
    assert cs.load_overrides(local_json)["SESSION_COOKIE_NAME"] == "__Host-s"
    cs.save_overrides({"BEAM_SIZE": 3}, local_json)   # later unrelated save passes
    # Without the env pin the same save is still refused.
    monkeypatch.setattr(cs, "_env_effective_values", lambda: {})
    q = str(tmp_path / "other.json")
    with pytest.raises(ValidationError):
        cs.save_overrides({"SESSION_COOKIE_NAME": "__Host-s"}, q)
    assert not os.path.exists(q)


def test_save_overrides_stored_prefixed_cookie_blocks_only_cookie_saves(tmp_path,
                                                                       monkeypatch):
    """A __Host- name stored with Secure off (pre-check file or hand edit)
    must not 422 every unrelated save — only a save touching the cookie
    keys, as guard_regex is scoped to saves that submit rules."""
    monkeypatch.setattr(cs, "_env_effective_values", lambda: {})
    p = _seed_local(tmp_path, {"SESSION_COOKIE_NAME": "__Host-s"})
    cs.save_overrides({"BEAM_SIZE": 3}, str(p))
    assert _read_json(p)["BEAM_SIZE"] == 3
    with pytest.raises(ValidationError):
        cs.save_overrides({"SESSION_COOKIE_SECURE": False}, str(p))
    with pytest.raises(ValidationError):
        cs.save_overrides({"SESSION_COOKIE_NAME": "__Host-s"}, str(p))
    # Saving the fix itself goes through.
    cs.save_overrides({"SESSION_COOKIE_SECURE": True}, str(p))


def test_save_overrides_silence_pair_checked_on_save_not_load(tmp_path,
                                                              monkeypatch):
    """A save of outer 6000 against the default hard break 5000 is refused,
    but the same pair already on disk still loads and blocks only saves that
    touch the pair."""
    monkeypatch.setattr(cs, "_env_effective_values", lambda: {})
    p = tmp_path / "config.local.json"
    with pytest.raises(ValidationError, match="STREAMING_HARD_BREAK_SILENCE_MS"):
        cs.save_overrides({"STREAMING_VAD_OUTER_SILENCE_MS": 6000}, str(p))
    assert not p.exists()
    _write_json(p, {"STREAMING_VAD_OUTER_SILENCE_MS": 6000})
    assert cs.load_overrides(str(p)) == {"STREAMING_VAD_OUTER_SILENCE_MS": 6000}
    cs.save_overrides({"BEAM_SIZE": 3}, str(p))
    with pytest.raises(ValidationError):
        cs.save_overrides({"STREAMING_HARD_BREAK_SILENCE_MS": 4000}, str(p))
    cs.save_overrides({"STREAMING_HARD_BREAK_SILENCE_MS": 0}, str(p))


def test_save_overrides_invalid_raises(local_json):
    with pytest.raises(ValidationError):
        cs.save_overrides({"BEAM_SIZE": 999}, local_json)
    assert not os.path.exists(local_json)


def test_pipeline_rule_tags_union():
    rules = [
        {"tags": ["b", "a"]},
        {"tags": ["a", "c"]},
        {"tags": []},
        "not-a-dict",
    ]
    assert cs.pipeline_rule_tags(rules) == ["a", "b", "c"]


def test_env_pinned_fields(monkeypatch):
    monkeypatch.delenv("WHISPER_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_BEAM_SIZE", raising=False)
    # A mapped field whose env var is set IS reported as pinned...
    monkeypatch.setenv("WHISPER_DEFAULT_MODEL", "large-v3")
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "5")
    # ...but a WHISPER_* var with no AdminConfig field (not in the mapping) is not.
    monkeypatch.setenv("WHISPER_BOOTSTRAP_ADMIN_KEY", "wk_x")
    pinned = cs.env_pinned_fields()
    assert pinned.get("DEFAULT_MODEL") == "WHISPER_DEFAULT_MODEL"
    assert pinned.get("BEAM_SIZE") == "WHISPER_BEAM_SIZE"
    assert "BOOTSTRAP_ADMIN_KEY" not in pinned


def test_env_pinned_fields_excludes_rejected_env_values(monkeypatch):
    """A field whose env value was rejected and reverted by config's
    validation pass must NOT be badged as pinned — the var no longer controls
    it, and the /settings apply path skips pinned names, so a stale badge
    would stop an admin's edit from ever reaching the live cfg."""
    import importlib

    try:
        monkeypatch.setenv("WHISPER_BEAM_SIZE", "9999")   # fails Field(le=...)
        monkeypatch.setenv("WHISPER_BEST_OF", "3")        # valid
        importlib.reload(cs_config)
        assert "BEAM_SIZE" in cs_config._ENV_REJECTED
        assert "BEST_OF" not in cs_config._ENV_REJECTED
        pinned = cs.env_pinned_fields()
        assert "BEAM_SIZE" not in pinned
        assert pinned.get("BEST_OF") == "WHISPER_BEST_OF"
    finally:
        monkeypatch.undo()
        importlib.reload(cs_config)  # restore from the clean environment


def test_format_validation_errors_shape():
    with pytest.raises(ValidationError) as ei:
        settings_schema.AdminConfig.model_validate({"BEAM_SIZE": 999})
    out = settings_schema.format_validation_errors(ei.value)
    assert isinstance(out, list) and out
    assert set(out[0]) == {"loc", "msg"}
    assert "BEAM_SIZE" in out[0]["loc"]


def test_concurrent_save_overrides_keep_both_keys(local_json, monkeypatch):
    """Two saves touching DIFFERENT keys must both survive.

    save_overrides() is a read-modify-write that rewrites the WHOLE merged
    document, and validation (the out-of-process regex guard) runs between the
    read and the write — measured at 2.3-2.6 s for a max-size PIPELINE_RULES
    payload. Unlocked, a save landing inside that window was silently reverted:
    the slow saver wrote back its stale snapshot. Both callers reach this via
    asyncio.to_thread, and PATCH /v1/pipeline-rules is non-admin-reachable, so
    a non-admin save could revert an admin's ADMIN_WEBUI_ALLOWED_HOSTS edit.

    A real barrier (not a sleep) proves the serialisation: the slow save is
    parked inside its window and only released once the fast save has run to
    completion, which is the exact interleaving that lost the update.
    """
    cs.save_overrides({"BEAM_SIZE": 5}, local_json)

    slow_inside = threading.Event()
    fast_started = threading.Event()
    fast_done = threading.Event()
    real_validate = settings_schema.AdminConfig.model_validate

    def slow_validate(payload, **kw):
        # Stand in for the guard_regex subprocess: a long window between the
        # read and the write, but only for the slow saver's payload.
        if isinstance(payload, dict) and payload.get("BEST_OF") == 3:
            slow_inside.set()
            fast_done.wait(10)
        return real_validate(payload, **kw)

    monkeypatch.setattr(settings_schema.AdminConfig, "model_validate", slow_validate)

    errors = []

    def slow():
        try:
            cs.save_overrides({"BEST_OF": 3}, local_json)
        except Exception as e:               # noqa: BLE001
            errors.append(e)

    def fast():
        try:
            # Only starts once the slow saver is parked mid-window. With the
            # lock it blocks here; unlocked it read the pre-BEST_OF file and
            # was overwritten by the slow saver's stale document.
            slow_inside.wait(10)
            fast_started.set()
            cs.save_overrides({"ADMIN_WEBUI_ALLOWED_HOSTS": ["10.0.0.1"]}, local_json)
        except Exception as e:               # noqa: BLE001
            errors.append(e)
        finally:
            fast_done.set()

    ts = [threading.Thread(target=slow), threading.Thread(target=fast)]
    for t in ts:
        t.start()
    # The fast saver must not be able to finish while the slow one holds the
    # lock, so release the slow saver on a timer if the lock did its job. The
    # timer is armed only once the fast saver has actually entered
    # save_overrides: armed on the wall clock from here it could release the
    # slow saver before the fast one contends for the lock, and an unlocked
    # save_overrides would then pass by accident on a loaded box.
    assert fast_started.wait(10)
    timer = threading.Timer(0.5, fast_done.set)
    timer.start()
    try:
        for t in ts:
            t.join(30)
    finally:
        timer.cancel()
    assert not [t for t in ts if t.is_alive()]
    assert errors == []

    on_disk = _read_json(local_json)
    assert on_disk["BEST_OF"] == 3
    assert on_disk["ADMIN_WEBUI_ALLOWED_HOSTS"] == ["10.0.0.1"]
    assert on_disk["BEAM_SIZE"] == 5          # the pre-existing key survived too


def test_save_lock_timeout_surfaces_as_oserror(local_json, monkeypatch):
    """Callers wrap save_overrides in `except OSError`. A same-process peer
    holding the save lock trips the in-process threading.Lock timeout, which
    must surface as an OSError naming the peer save."""
    monkeypatch.setattr(atomic_json, "SAVE_LOCK_TIMEOUT_S", 0.05)
    holder_in = threading.Event()
    release = threading.Event()

    def hold():
        with atomic_json.save_lock(local_json):
            holder_in.set()
            release.wait(10)

    t = threading.Thread(target=hold)
    t.start()
    try:
        holder_in.wait(10)
        with pytest.raises(OSError, match="peer save in progress"):
            cs.save_overrides({"BEAM_SIZE": 5}, local_json)
    finally:
        release.set()
        t.join(10)


def test_save_lock_cross_process_timeout_surfaces_as_plain_oserror(local_json, monkeypatch):
    """The cross-worker path: another process holds `<path>.lock`. filelock's
    Timeout already subclasses OSError, so `pytest.raises(OSError)` alone
    would not prove the conversion — pin the plain OSError type and the
    'peer worker' message the except-branch produces."""
    import subprocess
    import sys

    pytest.importorskip("filelock")
    lock_path = os.path.abspath(local_json) + ".lock"
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, filelock\n"
         "lk = filelock.FileLock(sys.argv[1]); lk.acquire()\n"
         "print('ready', flush=True); sys.stdin.readline()\n",
         lock_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        monkeypatch.setattr(atomic_json, "SAVE_LOCK_TIMEOUT_S", 0.2)
        with pytest.raises(OSError, match="peer worker save in progress") as ei:
            cs.save_overrides({"BEAM_SIZE": 5}, local_json)
        assert type(ei.value) is OSError
    finally:
        try:
            child.stdin.close()
            child.wait(10)
        except Exception:                    # noqa: BLE001
            child.kill()


def test_save_lock_file_does_not_disturb_the_config_dir(tmp_path, local_json):
    """The lock file must not look like atomic_write_json's tempfiles (or the
    config itself) to anything scanning the data dir."""
    cs.save_overrides({"BEAM_SIZE": 5}, local_json)
    names = sorted(os.listdir(tmp_path))
    assert "config.local.json" in names
    assert not [n for n in names if n.endswith(".tmp")]
    # POSIX filelock leaves the released .lock file behind; the Windows
    # implementation deletes it on release. Either way nothing but the lock
    # may sit next to the config.
    assert [n for n in names if n != "config.local.json"] in (
        [], ["config.local.json.lock"])


def test_media_cap_above_baseline_request_cap_rejected():
    # 20 GB media cap alone: the baseline request cap (10 GiB) would answer
    # first with the generic body-too-large 413 instead of the media 413.
    _bad(MEDIA_MAX_BYTES=20_000_000_000)


def test_media_and_request_caps_raised_together_ok():
    _ok(MEDIA_MAX_BYTES=20_000_000_000, MAX_REQUEST_BYTES=21_000_000_000)


def test_request_cap_below_baseline_upload_cap_rejected():
    _bad(MAX_REQUEST_BYTES=1024)


# Wildcard hosts never matched, so refusing them takes nothing away.
@pytest.mark.parametrize("bad", ["https://*.example.com", "https://*"])
def test_trusted_origins_rejects_wildcard_host(bad):
    _bad(TRUSTED_ORIGINS=[bad])


@pytest.mark.parametrize("bad", ["https://*.example.com", "https://*"])
def test_cors_origins_rejects_wildcard_host(bad):
    _bad(CORS_ALLOW_ORIGINS=[bad])


def test_cors_origins_bare_star_still_allowed():
    _ok(CORS_ALLOW_ORIGINS=["*"])


def test_cors_origins_lowercased():
    assert _ok(CORS_ALLOW_ORIGINS=["*", "https://Example.COM:8000"]).CORS_ALLOW_ORIGINS == [
        "*", "https://example.com:8000"]


def test_trusted_origins_lowercased():
    assert _ok(TRUSTED_ORIGINS=["https://MyHost.local"]).TRUSTED_ORIGINS == [
        "https://myhost.local"]


def test_origins_uppercase_scheme_is_lowercased_not_refused():
    assert _ok(TRUSTED_ORIGINS=["HTTPS://A.com"]).TRUSTED_ORIGINS == ["https://a.com"]
    assert _ok(CORS_ALLOW_ORIGINS=["HTTP://B.com:8080"]).CORS_ALLOW_ORIGINS == [
        "http://b.com:8080"]


def test_origins_default_port_dropped_on_save_and_load(local_json):
    # The browser omits the scheme's default port from Origin, so a stored
    # 'https://x:443' would never match exactly: every POST/WebSocket from
    # that proxy origin would get the CSRF 403.
    cs.save_overrides({"TRUSTED_ORIGINS": ["https://w.example.com:443",
                                           "http://[::1]:80", "https://a.com:80",
                                           "http://a.com:8443"]}, local_json)
    assert _read_json(local_json)["TRUSTED_ORIGINS"] == [
        "https://w.example.com", "http://[::1]", "https://a.com:80",
        "http://a.com:8443"]
    # A hand-written stored entry is normalised at load too.
    assert _ok(CORS_ALLOW_ORIGINS=["HTTPS://B.com:443"]).CORS_ALLOW_ORIGINS == [
        "https://b.com"]


@pytest.mark.parametrize("bad", ["http://a.com:99999", "http://a.com:080",
                                 "http://a.com:0", "https://bücher.de"])
def test_origins_impossible_port_or_non_ascii_host_refused_on_save(local_json, bad):
    # A browser sends neither an out-of-range / zero-padded port nor a
    # non-punycode host, so the entry would save and never match.
    for key in ("TRUSTED_ORIGINS", "CORS_ALLOW_ORIGINS"):
        with pytest.raises(ValidationError):
            cs.save_overrides({key: [bad]}, local_json)
        _ok(**{key: [bad]})
    assert not os.path.exists(local_json)


@pytest.mark.parametrize("bad", ["https://u@a.com", "https://a.com:1:2",
                                 "https://a.com:x"])
def test_origins_with_userinfo_or_junk_port_refused_on_save(local_json, bad):
    # Such an entry never equals a browser Origin header, so it saved cleanly
    # and silently did nothing. Refused on a save that submits the list;
    # load keeps accepting it (a load failure drops every override).
    for key in ("TRUSTED_ORIGINS", "CORS_ALLOW_ORIGINS"):
        with pytest.raises(ValidationError):
            cs.save_overrides({key: [bad]}, local_json)
        _ok(**{key: [bad]})
    assert not os.path.exists(local_json)
    cs.save_overrides({"TRUSTED_ORIGINS": ["https://a.com:8443", "http://[::1]:8000"]}, local_json)
    # A stored one does not block an unrelated save.
    _write_json(local_json, {"TRUSTED_ORIGINS": [bad]})
    cs.save_overrides({"BEAM_SIZE": 3}, local_json)


def test_load_overrides_strips_wildcard_origins_keeps_rest(tmp_path):
    # A stored file from before wildcard hosts were rejected must not wipe
    # every other override at boot: only the inert wildcard entries go.
    out = load_saved(tmp_path, {"TRUSTED_ORIGINS": ["https://*.example.com"],
                                "CORS_ALLOW_ORIGINS": ["*", "https://*.example.com",
                                                       "https://app.example.com"],
                                "BEAM_SIZE": 7})
    assert out["BEAM_SIZE"] == 7
    assert "TRUSTED_ORIGINS" not in out            # list emptied → key dropped
    assert out["CORS_ALLOW_ORIGINS"] == ["*", "https://app.example.com"]


def test_load_overrides_drops_blank_map_key_keeps_rest(tmp_path):
    # The old map-key pattern accepted " "; a file stored then must lose only
    # that key, not every override — and must stay saveable.
    rule = _map_rule("m", {" ": "x", "Komma": ","}, map_meta={" ": 1, "Komma": 2})
    p = str(_seed_local(tmp_path, {"PIPELINE_RULES": [rule, _terminal()], "BEAM_SIZE": 7}))
    out = cs.load_overrides(p)
    assert out["BEAM_SIZE"] == 7
    assert out["PIPELINE_RULES"][0]["map"] == {"Komma": ","}
    assert out["PIPELINE_RULES"][0]["map_meta"] == {"Komma": 2}
    cs.save_overrides({"BEAM_SIZE": 5}, path=p)
    assert cs.load_overrides(p)["BEAM_SIZE"] == 5


def test_extras_eviction_buckets_are_declared():
    assert set(settings_schema.EXTRAS_EVICTION) <= set(settings_schema._EVICT_BUCKETS)


def test_field_helper_rejects_unknown_evict_bucket():
    with pytest.raises(ValueError, match="evict="):
        settings_schema._F("DEFAULT_MODEL", scope="server", group="Models", evict="diarizaton")


def test_stored_bundle_values_refused_since_are_migrated_not_fatal(tmp_path, capsys):
    # Values a later-tightened validator refuses must not make load drop every override.
    p = _seed_local(tmp_path, {
        "SERVER_PORT": 9000,
        "DEFAULT_LANGUAGE": "jp",
        "OVERRIDE_PROFILES": {"p": {"SEGMENT_HEAD_ECHO_MIN_WORDS": 1,
                                    "SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS": 1,
                                    "TEMPERATURE": "0,abc", "BEAM_SIZE": 2,
                                    "locks": ["TEMPERATURE", "BEAM_SIZE"]}},
        "MODEL_OVERRIDES": {"large-v3": {"TEMPERATURE": "banana",
                                         "SUPPRESS_TOKENS": "x,1",
                                         "DEFAULT_LANGUAGE": "xx",
                                         "BEAM_SIZE": 3}},
    })
    out = cs.load_overrides(str(p))
    assert out["SERVER_PORT"] == 9000
    assert "DEFAULT_LANGUAGE" not in out
    assert out["OVERRIDE_PROFILES"]["p"]["SEGMENT_HEAD_ECHO_MIN_WORDS"] == 0
    assert out["OVERRIDE_PROFILES"]["p"]["SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS"] == 0
    prof = out["OVERRIDE_PROFILES"]["p"]
    # The lock outlives the dropped value: it pins the inherited one.
    assert "TEMPERATURE" not in prof
    assert sorted(prof["locks"]) == ["BEAM_SIZE", "TEMPERATURE"]
    mo = out["MODEL_OVERRIDES"]["large-v3"]
    assert "TEMPERATURE" not in mo and "SUPPRESS_TOKENS" not in mo
    assert "DEFAULT_LANGUAGE" not in mo
    assert mo["BEAM_SIZE"] == 3
    assert "banana" in capsys.readouterr().err
    # ...and a later save merges onto the migrated file instead of 422ing.
    cs.save_overrides({"BEST_OF": 3}, str(p))
    on_disk = _read_json(p)
    assert on_disk["SERVER_PORT"] == 9000 and on_disk["BEST_OF"] == 3


def test_stored_nul_in_a_path_or_token_is_dropped_not_fatal(tmp_path, capsys):
    # A file saved before the NUL rule must lose only those keys at boot,
    # not every override — and an unrelated save must not 422 on them.
    p = _seed_local(tmp_path, {"URL_MEDIA_DIR": "data/x\x00y",
                               "HF_TOKEN": "hf_secret\x00tail",
                               "DEFAULT_MODEL": "small"})
    out = cs.load_overrides(str(p))
    assert out == {"DEFAULT_MODEL": "small"}
    err = capsys.readouterr().err
    assert "dropped URL_MEDIA_DIR" in err and "dropped HF_TOKEN" in err
    assert "hf_secret" not in err
    cs.save_overrides({"BEAM_SIZE": 3}, str(p))
    assert _read_json(p) == {"DEFAULT_MODEL": "small", "BEAM_SIZE": 3}


def test_save_keeps_env_per_model_values_out_of_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cs_config, "_ENV_OVERRIDE_VALUES",
                        {"x": {"BEAM_SIZE": 3}, "y": {"BEAM_SIZE": 4}}, raising=False)
    p = _seed_local(tmp_path, {"MODEL_OVERRIDES": {"y": {"BEAM_SIZE": 2}}})
    # The editor round-trips the live dict, env values included.
    cs.save_overrides({"MODEL_OVERRIDES": {
        "x": {"BEAM_SIZE": 3, "PATIENCE": 1.0},
        "y": {"BEAM_SIZE": 4},
        "z": {"BEAM_SIZE": 5}}}, str(p))
    stored = _read_json(p)["MODEL_OVERRIDES"]
    assert stored["x"] == {"PATIENCE": 1.0}          # env value not persisted
    assert stored["y"] == {"BEAM_SIZE": 2}           # stored value restored
    assert stored["z"] == {"BEAM_SIZE": 5}           # not env-supplied
    # An admin edit that differs from the env value is theirs to keep.
    cs.save_overrides({"MODEL_OVERRIDES": {"x": {"BEAM_SIZE": 6}}}, str(p))
    assert _read_json(p)["MODEL_OVERRIDES"]["x"] == {"BEAM_SIZE": 6}
    # The hot-apply path lays the env layer back over the reloaded file.
    assert cs.with_env_model_overrides({"x": {"PATIENCE": 1.0}}) == {
        "x": {"PATIENCE": 1.0, "BEAM_SIZE": 3}, "y": {"BEAM_SIZE": 4}}
