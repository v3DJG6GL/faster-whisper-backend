"""Exhaustive tests for config_store: AdminConfig field bounds & validators,
ModelOverride, normalize_tags, the overrides load/save layer, the atomic
writer, and the small helper functions.

Factory-rule round-trips (load/save_factory_rules, terminal/dup/bad-regex)
are already covered by test_factory_rules.py at the repo root; here we add the
override layer, the scalar/model validators, and the helpers it does not touch.
"""

import json
import os
import time

import pytest
from pydantic import ValidationError

from faster_whisper_backend import config_store as cs


def _ok(**fields):
    """Validate a partial AdminConfig payload; return the model."""
    return cs.AdminConfig.model_validate(fields)


def _bad(**fields):
    with pytest.raises(ValidationError):
        cs.AdminConfig.model_validate(fields)


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
    ("CAPTURES_RECORDING_MIN_DURATION_S", 0.0, 600.0),
])
def test_float_bounds(field, lo, hi):
    _ok(**{field: lo})
    _ok(**{field: hi})
    _bad(**{field: lo - 0.1})
    _bad(**{field: hi + 0.1})


def test_capture_max_duration_min_is_0_1():
    # Asymmetric: MIN allows 0.0 but MAX requires ge=0.1.
    # Pair with MIN=0 so the cross-field validator passes.
    _ok(CAPTURES_RECORDING_MAX_DURATION_S=0.1,
        CAPTURES_RECORDING_MIN_DURATION_S=0.0)
    _bad(CAPTURES_RECORDING_MAX_DURATION_S=0.0)


def test_recording_duration_min_le_max():
    _ok(CAPTURES_RECORDING_MIN_DURATION_S=1.0,
        CAPTURES_RECORDING_MAX_DURATION_S=600.0)
    _bad(CAPTURES_RECORDING_MIN_DURATION_S=60.0,
         CAPTURES_RECORDING_MAX_DURATION_S=30.0)


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
    assert cs.normalize_tags(None) == []
    assert cs.normalize_tags([]) == []
    assert cs.normalize_tags(["B", "a", "a", " c "]) == ["a", "b", "c"]


def test_normalize_tags_drops_empty():
    assert cs.normalize_tags(["", "   ", "ok"]) == ["ok"]


def test_normalize_tags_rejects_bad():
    with pytest.raises(ValueError):
        cs.normalize_tags("notalist")
    with pytest.raises(ValueError):
        cs.normalize_tags([123])
    with pytest.raises(ValueError):
        cs.normalize_tags(["-leadinghyphen"])
    with pytest.raises(ValueError):
        cs.normalize_tags(["x" * 33])


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
    return cs.AdminConfig.model_validate(fields, context={"guard_regex": True})


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
    from faster_whisper_backend.core import regex_guard
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
    # tests/test_regex_guard.py with the rest of the helper's coverage.)
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
    from faster_whisper_backend.core import regex_guard
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
        return cs.AdminConfig.model_validate(
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
        cs.AdminConfig.model_validate(
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
    from faster_whisper_backend.core import regex_guard

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
        cs.ModelOverride.model_validate({
            "PIPELINE_RULES_EXCLUDE": ["a"],
            "PIPELINE_RULES_INCLUDE": ["a"],
        })


def test_model_override_bounds_inherit_global():
    cs.ModelOverride.model_validate({"BEAM_SIZE": 20})
    with pytest.raises(ValidationError):
        cs.ModelOverride.model_validate({"BEAM_SIZE": 21})


def test_admin_extra_forbid_on_override():
    with pytest.raises(ValidationError):
        cs.ModelOverride.model_validate({"NONSENSE": 1})


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
    # the config.local.json override path must match, or main._postprocess_text
    # does `set | list` and raises TypeError on every captures consumer.
    p = tmp_path / "a.json"
    p.write_text(
        json.dumps({"CAPTURES_PIPELINE_RULES_EXCLUDE": ["r1", "r2"]}),
        encoding="utf-8",
    )
    out = cs.load_overrides(str(p))
    assert isinstance(out["CAPTURES_PIPELINE_RULES_EXCLUDE"], set)
    assert out["CAPTURES_PIPELINE_RULES_EXCLUDE"] == {"r1", "r2"}


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
    from faster_whisper_backend import config as _cfg

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


def test_save_overrides_invalid_raises(tmp_path):
    p = str(tmp_path / "config.local.json")
    with pytest.raises(ValidationError):
        cs.save_overrides({"BEAM_SIZE": 999}, p)
    assert not os.path.exists(p)  # nothing written


# ---------------------------------------------------------------------------
# _atomic_write_json
# ---------------------------------------------------------------------------

def test_atomic_write_unicode(tmp_path):
    p = str(tmp_path / "u.json")
    cs._atomic_write_json({"k": "Müller"}, p, sort_keys=True, tmp_prefix=".t")
    assert json.loads(open(p, encoding="utf-8").read())["k"] == "Müller"
    # ensure_ascii=False keeps the literal char on disk.
    assert "Müller" in open(p, encoding="utf-8").read()


def test_atomic_write_retries_then_succeeds(tmp_path, monkeypatch):
    p = str(tmp_path / "r.json")
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("AV lock")
        return real_replace(src, dst)

    monkeypatch.setattr(cs.os, "replace", flaky)
    monkeypatch.setattr(cs.time, "sleep", lambda *_: None)
    cs._atomic_write_json({"ok": 1}, p, sort_keys=True, tmp_prefix=".t")
    assert calls["n"] == 3
    assert json.loads(open(p, encoding="utf-8").read()) == {"ok": 1}


def test_atomic_write_gives_up_after_retries(tmp_path, monkeypatch):
    p = str(tmp_path / "x.json")

    def always_fail(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(cs.os, "replace", always_fail)
    monkeypatch.setattr(cs.time, "sleep", lambda *_: None)
    with pytest.raises(PermissionError):
        cs._atomic_write_json({"ok": 1}, p, sort_keys=True, tmp_prefix=".t")
    # The temp file is cleaned up in finally; only the (untouched) dir remains.
    leftovers = [f for f in os.listdir(tmp_path) if f != "x.json"]
    assert leftovers == []


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

    from faster_whisper_backend import config
    try:
        monkeypatch.setenv("WHISPER_BEAM_SIZE", "9999")   # fails Field(le=...)
        importlib.reload(config)
        assert "BEAM_SIZE" in config._ENV_REJECTED
        pinned = cs.env_pinned_fields()
        assert "BEAM_SIZE" not in pinned
        # A validly pinned field is unaffected by the exclusion.
        assert cs.ENV_VAR_MAPPING["BEAM_SIZE"] == "WHISPER_BEAM_SIZE"
    finally:
        monkeypatch.undo()
        importlib.reload(config)  # restore from the clean environment


def test_format_validation_errors_shape():
    try:
        cs.AdminConfig.model_validate({"BEAM_SIZE": 999})
    except ValidationError as e:
        out = cs.format_validation_errors(e)
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
    real_validate = cs.AdminConfig.model_validate

    def slow_validate(payload, **kw):
        # Stand in for the guard_regex subprocess: a long window between the
        # read and the write, but only for the slow saver's payload.
        if isinstance(payload, dict) and payload.get("BEST_OF") == 3:
            slow_inside.set()
            fast_done.wait(10)
        return real_validate(payload, **kw)

    monkeypatch.setattr(cs.AdminConfig, "model_validate", slow_validate)

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
    monkeypatch.setattr(cs, "_SAVE_LOCK_TIMEOUT_S", 0.05)
    holder_in = threading.Event()
    release = threading.Event()

    def hold():
        with cs._save_lock(p):
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
        monkeypatch.setattr(cs, "_SAVE_LOCK_TIMEOUT_S", 0.2)
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
    """The lock file must not look like _atomic_write_json's tempfiles (or the
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
# MAX_REQUEST_BYTES must stay >= MAX_UPLOAD_BYTES (effective values)
# ---------------------------------------------------------------------------

def test_upload_cap_above_baseline_request_cap_rejected():
    # 1 GB upload cap alone: the baseline request cap (256 MiB) would answer
    # first with the generic body-too-large 413 instead of the upload 413.
    _bad(MAX_UPLOAD_BYTES=1_000_000_000)


def test_upload_and_request_caps_raised_together_ok():
    _ok(MAX_UPLOAD_BYTES=1_000_000_000, MAX_REQUEST_BYTES=2_000_000_000)


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
    assert set(cs.EXTRAS_EVICTION) <= set(cs._EVICT_BUCKETS)


def test_field_helper_rejects_unknown_evict_bucket():
    with pytest.raises(ValueError, match="evict="):
        cs._F("DEFAULT_MODEL", scope="server", group="Models", evict="diarizaton")
