"""Shared helpers for the tests/settings/ rule and config.local.json tests."""
import json
import os

from faster_whisper_backend import paths


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


def regex_list_rules(name, pattern, replacement="x"):
    """A one-entry regex-list rule `name` (labelled `name`), then the
    trim-edges terminal: the smallest valid PIPELINE_RULES list."""
    return [{"name": name, "label": name, "type": "regex-list",
             "entries": [{"pattern": pattern, "replacement": replacement}]},
            {"name": "trim-edges", "label": "Trim", "type": "terminal"}]


def load_saved(tmp_path, raw):
    """`raw` written as tmp_path/config.local.json, read back through
    config_store.load_overrides."""
    from faster_whisper_backend.settings import config_store as cs
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return cs.load_overrides(str(p))


def assert_upgrades_to_factory_once(rules, paths):
    """Stored `rules` upgrade to config.json's factory list, reporting
    `paths`; a second pass changes nothing."""
    from faster_whisper_backend.settings import config_renames as renames
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == paths
    assert raw["PIPELINE_RULES"] == factory_rules()
    assert renames.upgrade_rule_entries(raw) == []


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
