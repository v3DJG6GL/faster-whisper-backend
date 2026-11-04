"""Shared helpers for the tests/settings/ rule and config.local.json tests."""
import json
import os
import re
import time

from faster_whisper_backend import paths
from faster_whisper_backend.settings import schema as settings_schema


def factory_rules():
    """config.json's PIPELINE_RULES as shipped: a fresh, unvalidated parse on
    every call, so a caller may mutate the result."""
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["PIPELINE_RULES"]


def entry_by_label(rules, label):
    """The first entry labelled `label` in any rule of `rules`
    (StopIteration if none)."""
    return next(e for r in rules for e in (r.get("entries") or [])
                if e.get("label") == label)


def factory_rules_with_entry(label, pair):
    """The factory rules with entry `label` holding (pattern, replacement) `pair`."""
    rules = factory_rules()
    e = entry_by_label(rules, label)
    e["pattern"], e["replacement"] = pair
    return rules


def regex_list_rules(name, pattern, replacement="x"):
    """A one-entry regex-list rule `name` (labelled `name`), then the
    trim-edges terminal: the smallest valid PIPELINE_RULES list."""
    return [{"name": name, "label": name, "type": "regex-list",
             "entries": [{"pattern": pattern, "replacement": replacement}]},
            {"name": "trim-edges", "label": "Trim", "type": "terminal"}]


def validate_rules(rules, **context):
    """AdminConfig validation of a document holding only PIPELINE_RULES:
    a load with no context, a save with guard_regex=True (+ guard_slugs)."""
    return settings_schema.AdminConfig.model_validate(
        {"PIPELINE_RULES": rules}, context=context or None)


def load_saved(tmp_path, raw):
    """`raw` written as tmp_path/config.local.json, read back through
    config_store.load_overrides."""
    from faster_whisper_backend.settings import config_store as cs
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return cs.load_overrides(str(p))


def fast_postprocess(text, language="de"):
    """`text` through the installed rules, which must take under 1 s."""
    from faster_whisper_backend.pipeline import engine as pl_engine
    t0 = time.perf_counter()
    out = pl_engine._postprocess_text(text, model_name="", language=language)
    assert time.perf_counter() - t0 < 1.0
    return out


def assert_upgrades_to_factory_once(rules, paths):
    """Stored `rules` upgrade to config.json's factory list, reporting
    `paths`; a second pass changes nothing."""
    from faster_whisper_backend.settings import config_renames as renames
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == paths
    assert raw["PIPELINE_RULES"] == factory_rules()
    assert renames.upgrade_rule_entries(raw) == []


def assert_same_output_as_old(label, samples):
    """Factory entry `label` gives the same output on `samples` as the pair
    it was upgraded from."""
    from faster_whisper_backend.settings import config_renames as renames
    entry = entry_by_label(factory_rules(), label)
    old = renames.UPGRADED_RULE_ENTRIES[label][0]
    for s in samples:
        assert re.sub(entry["pattern"], entry["replacement"], s) == re.sub(*old, s), repr(s)


def old_strip_whisper_punctuation():
    """The combined card the June split replaced (config_renames.SPLIT_RULES)."""
    return {"name": "strip-whisper-punctuation", "label": "Strip Whisper punctuation",
            "type": "regex-list",
            "entries": [{"pattern": "\\s*\\.+\\s*$", "replacement": ""}]}


def pre_june_rules():
    """The factory rules as a copy saved before the June split holds them."""
    rules = [r for r in factory_rules()
             if r["name"] not in ("strip-trailing-period", "strip-auto-punctuation")]
    rules.insert(-1, old_strip_whisper_punctuation())
    return rules
