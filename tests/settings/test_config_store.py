"""Exhaustive tests for config_store + the settings schema: AdminConfig field
bounds & validators, ModelOverride, normalize_tags, the overrides load/save
layer (incl. its save lock), and the small helper functions. The atomic writer
itself is covered in tests/core/test_atomic_json.py.

Factory-rule round-trips (load/save_factory_rules, terminal/dup/bad-regex)
are already covered by test_factory_rules.py; here we add the
override layer, the scalar/model validators, and the helpers it does not touch.
"""

import json
import os

import pytest
from pydantic import ValidationError

from faster_whisper_backend.settings import config as cs_config
from faster_whisper_backend.settings import config_store as cs
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.core import atomic_json


def _ok(**fields):
    """Validate a partial AdminConfig payload; return the model."""
    return settings_schema.AdminConfig.model_validate(fields)


def _bad(**fields):
    with pytest.raises(ValidationError):
        settings_schema.AdminConfig.model_validate(fields)


# ---------------------------------------------------------------------------
# extra=forbid
# ---------------------------------------------------------------------------

def test_unknown_key_rejected():
    _bad(NOT_A_REAL_FIELD=1)


def test_empty_payload_ok():
    m = _ok()
    assert m.BEAM_SIZE is None


# ---------------------------------------------------------------------------
# Numeric bounds (reject below / accept at / accept at / reject above)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Patterns / literals
# ---------------------------------------------------------------------------

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
    # MODEL_COMPUTE_TYPE is the full CT2 compute_type set (10), so it now allows
    # int16 and the "auto"/"default" selectors. CONVERT_QUANTIZATION is CT2's
    # conversion set (8) — the concrete precisions only, NOT auto/default.
    _ok(MODEL_COMPUTE_TYPE="auto")
    _ok(MODEL_COMPUTE_TYPE="int16")
    _ok(MODEL_COMPUTE_TYPE="int8_bfloat16")
    _bad(CONVERT_QUANTIZATION="auto")          # auto/default are runtime-only
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


# ---------------------------------------------------------------------------
# Translation (T2T) fields
# ---------------------------------------------------------------------------

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
    # Non-empty without the mandatory slots must be a 422 at save.
    _bad(TRANSLATION_PROMPT_TEMPLATE="Translate into {target_language}: hi")
    _bad(TRANSLATION_PROMPT_TEMPLATE="Just do it: {text}")
    _bad(TRANSLATION_PROMPT_TEMPLATE="no placeholders at all")


def test_translation_list_fields_validate_entries():
    _ok(TRANSLATION_ALLOWED_MODELS=["org/repo", "org/repo:Q8_0"])
    _ok(TRANSLATION_PRELOAD_MODELS=["org/repo:Q4_K_M"])
    _bad(TRANSLATION_ALLOWED_MODELS=[""])            # empty entry
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
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps(
        {"TRANSLATION_ALLOWED_MODELS": ["org/a", "org/b:Q4_0"]}),
        encoding="utf-8")
    out = cs.load_overrides(str(p))
    assert out["TRANSLATION_ALLOWED_MODELS"] == {"org/a", "org/b:Q4_0"}
    assert isinstance(out["TRANSLATION_ALLOWED_MODELS"], set)


# ---------------------------------------------------------------------------
# CONVERT_QUANTIZATION / TEMPERATURE / SUPPRESS_TOKENS validators
# ---------------------------------------------------------------------------

def test_convert_quantisation():
    # The full CT2 ACCEPTED_MODEL_TYPES set is valid (ConvertQuantLit) — wider
    # than ComputeLit (e.g. int16, int8_bfloat16, int8_float32 are NOT runtime
    # compute types but ARE valid conversion quantizations).
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
    _ok(TEMPERATURE="0.5 , 0.5")        # whitespace tolerated
    _bad(TEMPERATURE="1.1")             # out of range
    _bad(TEMPERATURE="abc")             # not a float


def test_suppress_tokens():
    _ok(SUPPRESS_TOKENS="-1")
    _ok(SUPPRESS_TOKENS="1, 2 ,3")
    _ok(SUPPRESS_TOKENS="")
    _bad(SUPPRESS_TOKENS="1.5")
    _bad(SUPPRESS_TOKENS="x")


# ---------------------------------------------------------------------------
# Host validators (two different ones!)
# ---------------------------------------------------------------------------

def test_allowed_hosts_ip_cidr():
    _ok(ADMIN_WEBUI_ALLOWED_HOSTS=["127.0.0.1", "::1", "192.168.1.0/24"])
    _ok(USER_WEBUI_ALLOWED_HOSTS=["10.0.0.0/8", "0.0.0.0/0", "::/0"])
    _bad(ADMIN_WEBUI_ALLOWED_HOSTS=["not-an-ip"])
    _bad(USER_WEBUI_ALLOWED_HOSTS=["example.com"])  # hostname is not an IP/CIDR


def test_server_host_loose_charset():
    _ok(SERVER_HOST="0.0.0.0")
    _ok(SERVER_HOST="::")
    _ok(SERVER_HOST="my-host.local")
    _bad(SERVER_HOST="bad host")       # space rejected
    _bad(SERVER_HOST="has/slash")


# ---------------------------------------------------------------------------
# LOG_FILE path safety
# ---------------------------------------------------------------------------

def test_log_file_rejects_unc_and_traversal():
    _ok(LOG_FILE="logs/whisper.log")
    _bad(LOG_FILE="\\\\server\\share\\x.log")  # UNC
    _bad(LOG_FILE="//server/share/x.log")       # posix UNC-ish
    _bad(LOG_FILE="../etc/passwd")              # .. segment
    _bad(LOG_FILE="logs/../../x")               # windows-style .. caught too


# ---------------------------------------------------------------------------
# _cap_list (ALLOWED_MODELS / PRELOAD_MODELS)
# ---------------------------------------------------------------------------

def test_cap_list_over_1000():
    _bad(ALLOWED_MODELS=[f"m{i}" for i in range(1001)])
    _ok(ALLOWED_MODELS=[f"m{i}" for i in range(1000)])


# ---------------------------------------------------------------------------
# normalize_tags
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline rule validators (the parts not covered via save_factory_rules)
# ---------------------------------------------------------------------------

def _regex(name, pattern="x", replacement="y"):
    # A one-entry regex-list == a former single `regex` rule.
    return {"name": name, "label": name, "type": "regex-list",
            "entries": [{"pattern": pattern, "replacement": replacement}]}


def _terminal():
    return {"name": "trim-edges", "label": "Trim", "type": "terminal"}


def test_pipeline_callback_map_skips_pattern_validation():
    rule = {"name": "m", "label": "m", "type": "callback:map",
            "map": {"Komma": ","}}
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].type == "callback:map"


def _ok_on_save(**fields):
    """Validate as a SAVE would (guard_regex context) — runs the out-of-process
    regex probe (backref + catastrophic-backtracking guard)."""
    return settings_schema.AdminConfig.model_validate(fields, context={"guard_regex": True})


def test_pipeline_bad_backref_reported():
    # Replacement \3 with one group -> re.sub raises -> "regex test failed".
    # The .sub probe runs out-of-process on SAVE (guard_regex context).
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=[_regex("b", pattern="(a)", replacement=r"\3"),
                                    _terminal()])
    assert "regex test failed" in str(ei.value)


def test_pipeline_catastrophic_regex_rejected_on_save(monkeypatch):
    # A catastrophic-backtracking pattern is rejected on save WITHOUT hanging:
    # the out-of-process guard is killed on timeout (shortened here).
    from faster_whisper_backend.pipeline import regex_guard
    monkeypatch.setattr(regex_guard, "_GUARD_TIMEOUT", 0.5)
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=[
            _regex("boom", pattern="(.*a)+$", replacement=""), _terminal()])
    assert "catastrophic backtracking" in str(ei.value)


def test_pipeline_replacement_growth_rejected_when_fixture_never_matches():
    # The growth probe measures rx.sub() against a FIXED German-prose fixture,
    # so a pattern the fixture never matches scored a growth ratio of 1.0 no
    # matter how large its replacement. ("n", "n"*512) passed the guard and
    # then amplified a real transcript 512x per match on EVERY transcription
    # — two such entries allocate ~1 GB inside a single re.sub.
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=[
            _regex("blow", pattern="n", replacement="n" * 512), _terminal()])
    assert "regex test failed" in str(ei.value)


def test_pipeline_backref_only_replacement_rejected():
    # The analytic growth bound used to DELETE group references from the
    # replacement before measuring it, so a replacement made only of
    # backreferences measured as zero growth and was always accepted —
    # ("(n+)", r"\1" * 256) amplifies exactly as hard as ("n", "n" * 256),
    # which the sibling test above proves is rejected. Each reference is now
    # charged the shortest string the pattern can match.
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=[
            _regex("blow", pattern="(n+)", replacement="\\1" * 256), _terminal()])
    assert "regex test failed" in str(ei.value)


def test_pipeline_prefix_ambiguous_alternation_rejected():
    # The overlap screen only caught byte-IDENTICAL branches, so the
    # prefix-ambiguous forms — one run of input that splits many ways — walked
    # through and then backtracked exponentially on real transcripts.
    # (The _nested_repetition unit assertions for the full family live in
    # tests/pipeline/test_regex_guard.py with the rest of the helper's coverage.)
    with pytest.raises(ValidationError) as ei:
        _ok_on_save(PIPELINE_RULES=[
            _regex("boom", pattern="(n|d|nd)+#", replacement="X"), _terminal()])
    assert "catastrophic backtracking" in str(ei.value)


def test_pipeline_backrefs_and_alternation_still_accepted():
    # Guard against over-correction: ordinary backreference replacements and
    # UNrepeated alternations are the bread and butter of these rules.
    _ok_on_save(PIPELINE_RULES=[
        _regex("decimal", pattern=r"(\d+),(\d+)", replacement=r"\1.\2"),
        _regex("anrede", pattern="(Herr|Frau) ", replacement=r"\1 "),
        _terminal()])


def test_pipeline_ordinary_expansion_still_accepted():
    # The counterpart to the test above: a rule whose replacement is longer
    # than its match is completely normal and must keep validating.
    _ok_on_save(PIPELINE_RULES=[
        _regex("expand", pattern=r"z\.B\.", replacement="zum Beispiel"),
        _terminal()])


def test_pipeline_regex_guard_skipped_without_save_context(monkeypatch):
    # Load / diff validations (no guard_regex context) must NOT run the probe —
    # so a normal config load never spawns the helper and never hangs on a
    # stored pattern. A pattern that only the BACKTRACKING probe would flag
    # (compiles fine; pathological only against real input) validates cleanly.
    from faster_whisper_backend.pipeline import regex_guard
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1

    monkeypatch.setattr(regex_guard, "validate", _spy)
    _ok(PIPELINE_RULES=[_regex("b", pattern="(.*a)+$", replacement="x"), _terminal()])
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
        return settings_schema.AdminConfig.model_validate(
            {"PIPELINE_RULES": [bad, good, _terminal()]},
            context={"guard_regex": True, "guard_slugs": frozenset(slugs)},
        )

    _save({"harmless"})  # bad rule not probed -> save succeeds
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
    p = str(tmp_path / "config.local.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"PIPELINE_RULES": [bad, good, _terminal()]}, fh)

    edited = _regex("harmless", pattern="Komma", replacement=";")
    changed = cs.save_overrides({"PIPELINE_RULES": [bad, edited, _terminal()]},
                                p, guard_slugs=frozenset({"harmless"}))
    assert "PIPELINE_RULES" in changed
    on_disk = json.loads(open(p, encoding="utf-8").read())
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
        settings_schema.AdminConfig.model_validate(
            {"PIPELINE_RULES": [_regex("b", pattern="(a)", replacement=r"\3"),
                                _regex("harmless"), _terminal()]},
            context={"guard_regex": True, "guard_slugs": frozenset({"harmless"})},
        )
    assert "regex test failed" in str(ei.value)


def test_pipeline_bad_backref_rejected_on_load_without_subprocess(monkeypatch):
    # A bad replacement backref must keep failing on EVERY path (the eager
    # in-process template parse) — a hand-edited config.local.json with \3
    # against one group has to fail-safe at LOAD, not load cleanly and then
    # raise re.error on every request at match time. And detecting it must
    # not need the subprocess helper.
    from faster_whisper_backend.pipeline import regex_guard

    def _boom(*a, **k):
        raise AssertionError("subprocess guard must not run on load")

    monkeypatch.setattr(regex_guard, "validate", _boom)
    with pytest.raises(ValidationError) as ei:
        _ok(PIPELINE_RULES=[_regex("b", pattern="(a)", replacement=r"\3"), _terminal()])
    assert "regex test failed" in str(ei.value)


def test_pipeline_duplicate_slug():
    _bad(PIPELINE_RULES=[_regex("dup"), _regex("dup"), _terminal()])


def test_pipeline_terminal_must_be_last():
    _bad(PIPELINE_RULES=[_terminal(), _regex("after")])


def test_regex_list_validates_and_keeps_order():
    rule = {"name": "rl", "label": "RL", "type": "regex-list",
            "entries": [{"pattern": "a", "replacement": "b"},
                        {"pattern": "b", "replacement": "c", "label": "x", "note": "n"}]}
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].type == "regex-list"
    assert [e.pattern for e in m.PIPELINE_RULES[0].entries] == ["a", "b"]


def test_regex_list_requires_pattern_per_entry():
    # `pattern` is required on every entry.
    _bad(PIPELINE_RULES=[{"name": "rl", "label": "RL", "type": "regex-list",
                          "entries": [{"replacement": "b"}]}, _terminal()])


def test_regex_list_entry_extra_forbid():
    # Unknown per-entry key rejected (RegexListEntry has extra="forbid").
    _bad(PIPELINE_RULES=[{"name": "rl", "label": "RL", "type": "regex-list",
                          "entries": [{"pattern": "a", "bogus": 1}]}, _terminal()])


def test_regex_list_optional_fields_default_and_survive_exclude_none():
    m = _ok(PIPELINE_RULES=[{"name": "rl", "label": "RL", "type": "regex-list",
                             "entries": [{"pattern": "a"}]}, _terminal()])
    e = m.PIPELINE_RULES[0].entries[0]
    assert (e.replacement, e.label, e.note) == ("", "", "")
    # exclude_none must KEEP the "" defaults (they are "" not None).
    dumped = m.model_dump(exclude_none=True, mode="json")["PIPELINE_RULES"][0]["entries"][0]
    assert dumped == {"pattern": "a", "replacement": "", "label": "", "note": ""}


def test_regex_list_entry_bad_regex_reports_index():
    with pytest.raises(ValidationError) as ei:
        _ok(PIPELINE_RULES=[{"name": "rl", "label": "RL", "type": "regex-list",
                             "entries": [{"pattern": "("}]}, _terminal()])
    assert "entry 0" in str(ei.value)


def test_map_meta_pruned_to_map_keys():
    rule = {"name": "m", "label": "m", "type": "callback:map",
            "map": {"Komma": ","}, "map_meta": {"Komma": 5, "ghost": 9}}
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].map_meta == {"Komma": 5}


def test_map_key_collision_check_is_scoped_to_guard_slugs():
    """An untouched dictionary with colliding keys (it loads fine: load runs
    without the guard context) must not 422 a save scoped to another rule."""
    rule = {"name": "m", "label": "m", "type": "callback:map",
            "map": {"Passwort": "x", "passwort": "y"}}
    payload = {"PIPELINE_RULES": [rule, _terminal()]}
    settings_schema.AdminConfig.model_validate(
        payload, context={"guard_regex": True, "guard_slugs": {"other"}})
    for ctx in ({"guard_regex": True, "guard_slugs": {"m"}}, {"guard_regex": True}):
        with pytest.raises(ValidationError, match="collide when lowercased"):
            settings_schema.AdminConfig.model_validate(payload, context=ctx)


def test_rule_scoped_to_every_whisper_language_validates():
    # The languages cap tracks the code table (100 with yue), not a stale 99.
    langs = sorted(settings_schema.WHISPER_LANGUAGE_CODES)
    rule = {"name": "rl", "label": "RL", "type": "regex-list",
            "entries": [{"pattern": "a"}], "languages": langs}
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].languages == langs


def test_default_language_must_be_a_whisper_code():
    for model in (settings_schema.AdminConfig, settings_schema.ModelOverride,
                  settings_schema.OverrideProfile):
        with pytest.raises(ValidationError, match="unknown language code"):
            model.model_validate({"DEFAULT_LANGUAGE": "xx"})
        for ok in ("yue", "de", ""):
            model.model_validate({"DEFAULT_LANGUAGE": ok})


# NOTE: the validator's "took > 2 s" catastrophic-backtracking branch is
# deliberately NOT tested here. Triggering it requires a pattern that never
# terminates (e.g. (.+)+# against the validator's fixed ~1 KB fixture); the
# validator abandons the work via a daemon thread join(timeout=2.0), but that
# daemon thread then runs the runaway regex forever, pinning a CPU core and
# contending the GIL for the rest of the pytest session. The error branch is
# covered by test_pipeline_bad_backref_reported above.


# ---------------------------------------------------------------------------
# ModelOverride validators
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model-level cross-field validators (only fire when both keys present)
# ---------------------------------------------------------------------------

def test_no_orphan_overrides_fires_only_with_both():
    # Both present + non-empty allowlist + orphan -> reject.
    _bad(ALLOWED_MODELS=["a"], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    # Empty allowlist = anything goes -> skip check.
    _ok(ALLOWED_MODELS=[], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    # Only overrides present -> cross-check skipped.
    _ok(MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    # Override model in allowlist -> ok.
    _ok(ALLOWED_MODELS=["a", "b"], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})


def test_pipeline_rule_slugs_cross_check():
    rules = [_regex("known"), _terminal()]
    # Unknown slug in a per-model EXCLUDE -> reject (both keys present).
    _bad(PIPELINE_RULES=rules,
         MODEL_OVERRIDES={"m": {"PIPELINE_RULES_EXCLUDE": ["bogus"]}})
    # Known slug -> ok.
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


# ---------------------------------------------------------------------------
# load_overrides / save_overrides
# ---------------------------------------------------------------------------

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
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"PIPELINE_RULES": rules}), encoding="utf-8")
    assert cs.load_overrides(str(p)) == {}


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
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({
        "SERVER_PORT": 9000, "SUPPRESS_TOKENS": "100000000",
        "OVERRIDE_PROFILES": {"p": {"SUPPRESS_TOKENS": "-1,100000000",
                                    "BEAM_SIZE": 2}}}), encoding="utf-8")
    out = cs.load_overrides(str(p))
    assert out["SERVER_PORT"] == 9000
    assert "SUPPRESS_TOKENS" not in out
    assert out["OVERRIDE_PROFILES"]["p"] == {"BEAM_SIZE": 2}


def test_load_overrides_unknown_key_ignored_whole_file(tmp_path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"BEAM_SIZE": 5, "BOGUS": 1}), encoding="utf-8")
    # Whole file is rejected on validation failure -> {}.
    assert cs.load_overrides(str(p)) == {}


def test_load_overrides_coerces_allowed_models_to_set(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"ALLOWED_MODELS": ["a", "b"]}), encoding="utf-8")
    out = cs.load_overrides(str(p))
    assert isinstance(out["ALLOWED_MODELS"], set)
    assert out["ALLOWED_MODELS"] == {"a", "b"}


def test_load_overrides_coerces_captures_excludes_to_set(tmp_path):
    # config.json default + ENV paths already yield a set (config._SET_FIELDS);
    # the config.local.json override path must match, or the pipeline engine's
    # _postprocess_text does `set | list` and raises TypeError on every captures consumer.
    p = tmp_path / "a.json"
    p.write_text(
        json.dumps({"CAPTURES_PIPELINE_RULES_EXCLUDE": ["r1", "r2"]}),
        encoding="utf-8",
    )
    out = cs.load_overrides(str(p))
    assert isinstance(out["CAPTURES_PIPELINE_RULES_EXCLUDE"], set)
    assert out["CAPTURES_PIPELINE_RULES_EXCLUDE"] == {"r1", "r2"}


def test_save_overrides_rejects_unknown_slug_without_local_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"known"})
    p = str(tmp_path / "config.local.json")
    with pytest.raises(ValidationError):
        cs.save_overrides({"MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["dictashion-map"]}}}, p)
    assert not os.path.exists(p)
    cs.save_overrides({"MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["known"]}}}, p)
    assert os.path.exists(p)
    # A stale slug on disk must not brick unrelated saves.
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"renamed"})
    cs.save_overrides({"BEAM_SIZE": 5}, p)
    assert json.loads(open(p, encoding="utf-8").read())["BEAM_SIZE"] == 5


def test_stale_stored_slug_only_blocks_saves_of_its_own_key(tmp_path, monkeypatch):
    # The slug check used to run over the whole merged document whenever the
    # save touched ANY slug-bearing key, so a stale exclude stored under
    # MODEL_OVERRIDES 422'd every OVERRIDE_PROFILES / captures save too.
    monkeypatch.setattr(cs, "_canonical_rule_slugs", lambda: {"known"})
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"MODEL_OVERRIDES": {"m": {
        "PIPELINE_RULES_EXCLUDE": ["gone"]}}}), encoding="utf-8")
    cs.save_overrides({"OVERRIDE_PROFILES": {"p": {"BEAM_SIZE": 2}}}, str(p))
    cs.save_overrides({"CAPTURES_PIPELINE_RULES_EXCLUDE": ["known"]}, str(p))
    on_disk = json.loads(p.read_text(encoding="utf-8"))
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
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"OVERRIDE_PROFILES": {
        "a": {"BEAM_SIZE": 2},
        "b": {"PIPELINE_RULES_EXCLUDE": ["gone"]}}}), encoding="utf-8")
    cs.save_overrides({"OVERRIDE_PROFILES": {
        "a": {"BEAM_SIZE": 3},
        "b": {"PIPELINE_RULES_EXCLUDE": ["gone"]}}}, str(p))
    on_disk = json.loads(p.read_text(encoding="utf-8"))
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


def test_stored_top_level_guard_at_one_is_migrated_not_fatal(tmp_path, capsys):
    # The top-level segment guards started refusing 1 after releases that
    # stored it; one left in config.local.json dropped every override at boot
    # and 422'd every save, the failure the bundle migration already covers.
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"SERVER_PORT": 9000,
                             "SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS": 1,
                             "SEGMENT_HEAD_ECHO_MIN_WORDS": 1}), encoding="utf-8")
    out = cs.load_overrides(str(p))
    assert out["SERVER_PORT"] == 9000
    assert out["SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS"] == 0
    assert out["SEGMENT_HEAD_ECHO_MIN_WORDS"] == 0
    assert "SEGMENT_HEAD_ECHO_MIN_WORDS=1" in capsys.readouterr().err
    cs.save_overrides({"BEST_OF": 3}, str(p))
    on_disk = json.loads(p.read_text(encoding="utf-8"))
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


def test_removing_the_local_rules_copy_refuses_refs_it_would_strand(
        tmp_path, monkeypatch):
    # Resetting PIPELINE_RULES makes the factory list canonical; a stored
    # exclude naming a local-only rule would then 422 every later save of
    # that key, so the reset is refused instead and the file kept.
    factory = [{"name": "factory-rule"}, {"name": "trim-edges"}]
    monkeypatch.setattr(cs, "load_factory_rules", lambda path=None: factory)
    monkeypatch.setattr(cs, "_canonical_rule_slugs",
                        lambda: {"factory-rule", "trim-edges", "my-local-rule"})
    p = tmp_path / "config.local.json"
    local = {
        "PIPELINE_RULES": [
            {"name": "my-local-rule", "label": "mine", "type": "regex-list",
             "entries": [{"pattern": "x", "replacement": "y"}]},
            {"name": "trim-edges", "label": "Trim edges", "type": "terminal"}],
        "MODEL_OVERRIDES": {"large-v3": {
            "PIPELINE_RULES_EXCLUDE": ["my-local-rule"]}},
    }
    p.write_text(json.dumps(local), encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValidationError, match="my-local-rule"):
        cs.save_overrides({"PIPELINE_RULES": None}, str(p))
    assert p.read_text(encoding="utf-8") == before
    # Without the stranded ref the reset goes through.
    local["MODEL_OVERRIDES"]["large-v3"]["PIPELINE_RULES_EXCLUDE"] = ["factory-rule"]
    p.write_text(json.dumps(local), encoding="utf-8")
    cs.save_overrides({"PIPELINE_RULES": None}, str(p))
    assert "PIPELINE_RULES" not in json.loads(p.read_text(encoding="utf-8"))


def test_save_overrides_roundtrip_and_merge(tmp_path):
    p = str(tmp_path / "config.local.json")
    changed = cs.save_overrides({"BEAM_SIZE": 5}, p)
    assert changed == {"BEAM_SIZE": 5}
    # Merge: a second partial save keeps the first field.
    cs.save_overrides({"BEST_OF": 3}, p)
    on_disk = json.loads(open(p, encoding="utf-8").read())
    assert on_disk["BEAM_SIZE"] == 5 and on_disk["BEST_OF"] == 3


def test_save_overrides_none_removes(tmp_path):
    p = str(tmp_path / "config.local.json")
    cs.save_overrides({"BEAM_SIZE": 5, "BEST_OF": 3}, p)
    changed = cs.save_overrides({"BEAM_SIZE": None}, p)
    assert "BEAM_SIZE" in changed and changed["BEAM_SIZE"] is None
    on_disk = json.loads(open(p, encoding="utf-8").read())
    assert "BEAM_SIZE" not in on_disk and on_disk["BEST_OF"] == 3


def test_save_overrides_changed_excludes_unchanged(tmp_path):
    p = str(tmp_path / "config.local.json")
    cs.save_overrides({"BEAM_SIZE": 5}, p)
    # Re-saving the same value reports no change for it.
    changed = cs.save_overrides({"BEAM_SIZE": 5}, p)
    assert changed == {}


def test_save_factory_rules_preserves_sibling_defaults(tmp_path):
    # config.json now holds ALL factory defaults, not just PIPELINE_RULES, so a
    # rules "Promote to factory" must read-modify-write — not clobber the sibling
    # scalar defaults (the old whole-file replace would wipe every other value).
    p = str(tmp_path / "config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "DEFAULT_MODEL": "keep-me", "BEST_OF": 7,
                   "PIPELINE_RULES": [{"name": "t", "label": "T", "type": "terminal"}]}, f)
    cs.save_factory_rules([{"name": "trim", "label": "Trim", "type": "terminal"}], p)
    on_disk = json.loads(open(p, encoding="utf-8").read())
    assert on_disk["DEFAULT_MODEL"] == "keep-me"     # sibling default preserved
    assert on_disk["BEST_OF"] == 7                    # sibling default preserved
    assert [r["name"] for r in on_disk["PIPELINE_RULES"]] == ["trim"]   # rules updated


def test_sample_sizing_absent_field_uses_baseline_not_live_override(monkeypatch):
    # Regression: _validate_sample_sizing must fall back to config._BASELINE
    # (the immutable in-repo default) for an absent field, NOT the live config
    # attribute. The live attribute already carries any applied override, so at
    # save time (server running) it would be the OLD override while at load
    # time (config import) it is the bare default — that asymmetry let a save
    # pass validation, then the next restart's load fail it and silently drop
    # EVERY override on disk.
    from faster_whisper_backend.settings import config as _cfg

    # Simulate a server running with a previously-applied TARGET override of 5.
    monkeypatch.setattr(_cfg, "CAPTURES_PROPOSER_TARGET_S", 5.0, raising=False)
    # _BASELINE keeps the real in-repo default (26.0), which exceeds MAX=6.
    assert _cfg._BASELINE["CAPTURES_PROPOSER_TARGET_S"] > 6.0

    # Removing TARGET reverts it to the 26.0 baseline → 1 ≤ 26 ≤ 6 is false.
    # Must reject regardless of the stale live value of 5.0.
    _bad(CAPTURES_SAMPLE_MIN_DURATION_S=1.0, CAPTURES_SAMPLE_MAX_DURATION_S=6.0)


def test_save_overrides_corrupt_existing_rewrites(tmp_path):
    p = str(tmp_path / "config.local.json")
    open(p, "w", encoding="utf-8").write("{ corrupt")
    cs.save_overrides({"BEAM_SIZE": 7}, p)
    assert json.loads(open(p, encoding="utf-8").read())["BEAM_SIZE"] == 7


def test_load_overrides_non_utf8_file_is_ignored_not_raised(tmp_path):
    """load_overrides NEVER raises: config imports it with only `except
    ImportError` around it, so a hand edit saved as cp1252 (UnicodeDecodeError
    is a ValueError, not a JSONDecodeError) used to stop the server booting."""
    p = tmp_path / "config.local.json"
    p.write_bytes(b'{"DEFAULT_PROMPT": "Gr\xfc\xdfe"}')
    assert cs.load_overrides(str(p)) == {}
    # The save path takes its corrupt-file branch instead of a 500.
    cs.save_overrides({"BEAM_SIZE": 7}, str(p))
    assert json.loads(p.read_text(encoding="utf-8"))["BEAM_SIZE"] == 7


def test_save_overrides_checks_cross_field_pairs_against_env_pins(tmp_path,
                                                                  monkeypatch):
    """A save must see the env-pinned sibling that will be in force: with env
    KEEP_S=12 pinned, TRIM_S=11 passed against the baseline KEEP_S=10, ran
    inconsistent, and the next restart reverted the env value."""
    from faster_whisper_backend.settings import config as _cfg
    monkeypatch.setattr(cs, "env_pinned_fields", lambda: {
        "STREAMING_BUFFER_TRIM_KEEP_S": "WHISPER_STREAMING_BUFFER_TRIM_KEEP_S"})
    monkeypatch.setattr(_cfg, "STREAMING_BUFFER_TRIM_KEEP_S", 12.0)
    p = str(tmp_path / "config.local.json")
    with pytest.raises(ValidationError):
        cs.save_overrides({"STREAMING_BUFFER_TRIM_S": 11}, p)
    assert not os.path.exists(p)
    cs.save_overrides({"STREAMING_BUFFER_TRIM_S": 13}, p)   # consistent: saved
    # The import-time path (no context) still checks the bare baseline.
    settings_schema.AdminConfig.model_validate({"STREAMING_BUFFER_TRIM_S": 11})


def test_save_overrides_refuses_a_file_only_an_env_pin_makes_valid(tmp_path,
                                                                   monkeypatch):
    """load_overrides validates the file without the env context and drops
    EVERY override when it fails (at boot and in the hot-apply right after a
    save). A save consistent only thanks to an env-pinned sibling must 422
    instead of being written and then thrown away with the rest of the file."""
    from faster_whisper_backend.settings import config as _cfg
    monkeypatch.setattr(cs, "env_pinned_fields", lambda: {
        "MEDIA_MAX_BYTES": "WHISPER_MEDIA_MAX_BYTES",
        "STREAMING_BUFFER_TRIM_KEEP_S": "WHISPER_STREAMING_BUFFER_TRIM_KEEP_S"})
    monkeypatch.setattr(_cfg, "MEDIA_MAX_BYTES", 2_000_000_000)
    monkeypatch.setattr(_cfg, "STREAMING_BUFFER_TRIM_KEEP_S", 5.0)
    p = str(tmp_path / "config.local.json")
    with pytest.raises(ValidationError):
        cs.save_overrides({"MAX_REQUEST_BYTES": 3_000_000_000}, p)
    with pytest.raises(ValidationError):
        cs.save_overrides({"STREAMING_BUFFER_TRIM_S": 8}, p)
    assert not os.path.exists(p)
    # What a save does accept round-trips through the bare load.
    cs.save_overrides({"DEFAULT_PROMPT": "hello", "STREAMING_BUFFER_TRIM_S": 13}, p)
    assert cs.load_overrides(p) == {"DEFAULT_PROMPT": "hello",
                                    "STREAMING_BUFFER_TRIM_S": 13.0}


def test_save_overrides_invalid_raises(tmp_path):
    p = str(tmp_path / "config.local.json")
    with pytest.raises(ValidationError):
        cs.save_overrides({"BEAM_SIZE": 999}, p)
    assert not os.path.exists(p)  # nothing written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    assert "BOOTSTRAP_ADMIN_KEY" not in pinned  # a secret constant, not an AdminConfig field


def test_env_pinned_fields_excludes_rejected_env_values(monkeypatch):
    """A field whose env value was rejected and reverted by config's
    validation pass must NOT be badged as pinned — the var no longer controls
    it, and the /settings apply path skips pinned names, so a stale badge
    would stop an admin's edit from ever reaching the live cfg."""
    import importlib

    from faster_whisper_backend.settings import config
    try:
        monkeypatch.setenv("WHISPER_BEAM_SIZE", "9999")   # fails Field(le=...)
        monkeypatch.setenv("WHISPER_BEST_OF", "3")        # valid
        importlib.reload(config)
        assert "BEAM_SIZE" in config._ENV_REJECTED
        assert "BEST_OF" not in config._ENV_REJECTED
        pinned = cs.env_pinned_fields()
        assert "BEAM_SIZE" not in pinned
        # A validly pinned field is unaffected by the exclusion.
        assert pinned.get("BEST_OF") == "WHISPER_BEST_OF"
    finally:
        monkeypatch.undo()
        importlib.reload(config)  # restore from the clean environment


def test_format_validation_errors_shape():
    with pytest.raises(ValidationError) as ei:
        settings_schema.AdminConfig.model_validate({"BEAM_SIZE": 999})
    out = settings_schema.format_validation_errors(ei.value)
    assert isinstance(out, list) and out
    assert set(out[0]) == {"loc", "msg"}
    assert "BEAM_SIZE" in out[0]["loc"]


# ---------------------------------------------------------------------------
# save_overrides concurrency — the lost update
# ---------------------------------------------------------------------------

def test_concurrent_save_overrides_keep_both_keys(tmp_path, monkeypatch):
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
    import threading

    p = str(tmp_path / "config.local.json")
    cs.save_overrides({"BEAM_SIZE": 5}, p)

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
            cs.save_overrides({"BEST_OF": 3}, p)
        except Exception as e:               # noqa: BLE001
            errors.append(e)

    def fast():
        try:
            # Only starts once the slow saver is parked mid-window. With the
            # lock it blocks here; unlocked it read the pre-BEST_OF file and
            # was overwritten by the slow saver's stale document.
            slow_inside.wait(10)
            fast_started.set()
            cs.save_overrides({"ADMIN_WEBUI_ALLOWED_HOSTS": ["10.0.0.1"]}, p)
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

    on_disk = json.loads(open(p, encoding="utf-8").read())
    assert on_disk["BEST_OF"] == 3
    assert on_disk["ADMIN_WEBUI_ALLOWED_HOSTS"] == ["10.0.0.1"]
    assert on_disk["BEAM_SIZE"] == 5          # the pre-existing key survived too


def test_save_lock_timeout_surfaces_as_oserror(tmp_path, monkeypatch):
    """Callers wrap save_overrides in `except OSError`. A same-process peer
    holding the save lock trips the in-process threading.Lock timeout, which
    must surface as an OSError naming the peer save."""
    import threading

    p = str(tmp_path / "config.local.json")
    monkeypatch.setattr(atomic_json, "SAVE_LOCK_TIMEOUT_S", 0.05)
    holder_in = threading.Event()
    release = threading.Event()

    def hold():
        with atomic_json.save_lock(p):
            holder_in.set()
            release.wait(10)

    t = threading.Thread(target=hold)
    t.start()
    try:
        holder_in.wait(10)
        with pytest.raises(OSError, match="peer save in progress"):
            cs.save_overrides({"BEAM_SIZE": 5}, p)
    finally:
        release.set()
        t.join(10)


def test_save_lock_cross_process_timeout_surfaces_as_plain_oserror(tmp_path, monkeypatch):
    """The cross-worker path: another process holds `<path>.lock`. filelock's
    Timeout already subclasses OSError, so `pytest.raises(OSError)` alone
    would not prove the conversion — pin the plain OSError type and the
    'peer worker' message the except-branch produces."""
    import subprocess
    import sys

    pytest.importorskip("filelock")
    p = str(tmp_path / "config.local.json")
    lock_path = os.path.abspath(p) + ".lock"
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
            cs.save_overrides({"BEAM_SIZE": 5}, p)
        assert type(ei.value) is OSError
    finally:
        try:
            child.stdin.close()
            child.wait(10)
        except Exception:                    # noqa: BLE001
            child.kill()


def test_save_lock_file_does_not_disturb_the_config_dir(tmp_path):
    """The lock file must not look like atomic_write_json's tempfiles (or the
    config itself) to anything scanning the data dir."""
    p = str(tmp_path / "config.local.json")
    cs.save_overrides({"BEAM_SIZE": 5}, p)
    names = sorted(os.listdir(tmp_path))
    assert "config.local.json" in names
    assert not [n for n in names if n.endswith(".tmp")]
    # POSIX filelock leaves the released .lock file behind; the Windows
    # implementation deletes it on release. Either way nothing but the lock
    # may sit next to the config.
    assert [n for n in names if n != "config.local.json"] in (
        [], ["config.local.json.lock"])


# ---------------------------------------------------------------------------
# MAX_REQUEST_BYTES must stay >= MEDIA_MAX_BYTES (effective values)
# ---------------------------------------------------------------------------

def test_media_cap_above_baseline_request_cap_rejected():
    # 20 GB media cap alone: the baseline request cap (10 GiB) would answer
    # first with the generic body-too-large 413 instead of the media 413.
    _bad(MEDIA_MAX_BYTES=20_000_000_000)


def test_media_and_request_caps_raised_together_ok():
    _ok(MEDIA_MAX_BYTES=20_000_000_000, MAX_REQUEST_BYTES=21_000_000_000)


def test_request_cap_below_baseline_upload_cap_rejected():
    _bad(MAX_REQUEST_BYTES=1024)


# ---------------------------------------------------------------------------
# Origin allowlists: wildcard hosts are rejected (they never matched anyway)
# ---------------------------------------------------------------------------

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


def test_load_overrides_strips_wildcard_origins_keeps_rest(tmp_path):
    # A stored file from before wildcard hosts were rejected must not wipe
    # every other override at boot: only the inert wildcard entries go.
    p = str(tmp_path / "config.local.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"TRUSTED_ORIGINS": ["https://*.example.com"],
                   "CORS_ALLOW_ORIGINS": ["*", "https://*.example.com",
                                          "https://app.example.com"],
                   "BEAM_SIZE": 7}, f)
    out = cs.load_overrides(p)
    assert out["BEAM_SIZE"] == 7
    assert "TRUSTED_ORIGINS" not in out            # list emptied → key dropped
    assert out["CORS_ALLOW_ORIGINS"] == ["*", "https://app.example.com"]


# ---------------------------------------------------------------------------
# _F(evict=...) is a closed set
# ---------------------------------------------------------------------------

def test_extras_eviction_buckets_are_declared():
    assert set(settings_schema.EXTRAS_EVICTION) <= set(settings_schema._EVICT_BUCKETS)


def test_field_helper_rejects_unknown_evict_bucket():
    with pytest.raises(ValueError, match="evict="):
        settings_schema._F("DEFAULT_MODEL", scope="server", group="Models", evict="diarizaton")


# ---------------------------------------------------------------------------
# Stored values a later-tightened validator refuses (load must not drop all)
# ---------------------------------------------------------------------------

def test_stored_bundle_values_refused_since_are_migrated_not_fatal(tmp_path, capsys):
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({
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
    }), encoding="utf-8")
    out = cs.load_overrides(str(p))
    assert out["SERVER_PORT"] == 9000
    assert "DEFAULT_LANGUAGE" not in out
    assert out["OVERRIDE_PROFILES"]["p"]["SEGMENT_HEAD_ECHO_MIN_WORDS"] == 0
    assert out["OVERRIDE_PROFILES"]["p"]["SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS"] == 0
    prof = out["OVERRIDE_PROFILES"]["p"]
    assert "TEMPERATURE" not in prof and prof["locks"] == ["BEAM_SIZE"]
    mo = out["MODEL_OVERRIDES"]["large-v3"]
    assert "TEMPERATURE" not in mo and "SUPPRESS_TOKENS" not in mo
    assert "DEFAULT_LANGUAGE" not in mo
    assert mo["BEAM_SIZE"] == 3
    assert "banana" in capsys.readouterr().err
    # ...and a later save merges onto the migrated file instead of 422ing.
    cs.save_overrides({"BEST_OF": 3}, str(p))
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["SERVER_PORT"] == 9000 and on_disk["BEST_OF"] == 3


# ---------------------------------------------------------------------------
# WHISPER_MODEL_OVERRIDE__ values stay env values on a per-model save
# ---------------------------------------------------------------------------

def test_save_keeps_env_per_model_values_out_of_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cs_config, "_ENV_OVERRIDE_VALUES",
                        {"x": {"BEAM_SIZE": 3}, "y": {"BEAM_SIZE": 4}}, raising=False)
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps({"MODEL_OVERRIDES": {"y": {"BEAM_SIZE": 2}}}),
                 encoding="utf-8")
    # The editor round-trips the live dict, env values included.
    cs.save_overrides({"MODEL_OVERRIDES": {
        "x": {"BEAM_SIZE": 3, "PATIENCE": 1.0},
        "y": {"BEAM_SIZE": 4},
        "z": {"BEAM_SIZE": 5}}}, str(p))
    stored = json.loads(p.read_text(encoding="utf-8"))["MODEL_OVERRIDES"]
    assert stored["x"] == {"PATIENCE": 1.0}          # env value not persisted
    assert stored["y"] == {"BEAM_SIZE": 2}           # stored value restored
    assert stored["z"] == {"BEAM_SIZE": 5}           # not env-supplied
    # An admin edit that differs from the env value is theirs to keep.
    cs.save_overrides({"MODEL_OVERRIDES": {"x": {"BEAM_SIZE": 6}}}, str(p))
    assert json.loads(p.read_text(encoding="utf-8"))["MODEL_OVERRIDES"]["x"] == {
        "BEAM_SIZE": 6}
    # The hot-apply path lays the env layer back over the reloaded file.
    assert cs.with_env_model_overrides({"x": {"PATIENCE": 1.0}}) == {
        "x": {"PATIENCE": 1.0, "BEAM_SIZE": 3}, "y": {"BEAM_SIZE": 4}}
