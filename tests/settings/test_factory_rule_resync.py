"""A stored copy of a factory rule that is still exactly the config.json
version it was last in sync with is replaced by the current config.json rule
at load (config_store._resync_untouched_factory_copies).

upgrade_rule_entries fixes an old entry's text, but not the rest of a factory
change (a new entry, a new note) and not the copy's config_rev, so an untouched
copy read "edited here · config.json changed too" in the /settings pipeline
editor. The test for "untouched" is the editor's own: the copy's fingerprint
(settings.html _ruleHash) equals its config_rev.
"""

import copy
import json
import os
import shutil
import subprocess

import pytest

from faster_whisper_backend import paths
from faster_whisper_backend.settings import config_store as cs
from tests.settings._rules_helpers import factory_rules as _factory_rules
from tests.settings._rules_helpers import load_saved

_NODE = shutil.which("node")
_SETTINGS_HTML = os.path.join(paths.REPO_ROOT, "faster_whisper_backend", "admin",
                              "templates", "settings.html")
_QUOTE_LABELS = ("tighten-german-quote-open", "tighten-german-quote-close")


def _rev(rule):
    return cs._editor_rule_hash(cs._editor_view(rule))


def _older_tidy(rules):
    """digit-and-whitespace-tidy as shipped before cd619406 added the German
    quote entries, stamped in sync the way the editor stamps it."""
    tidy = next(r for r in rules if r["name"] == "digit-and-whitespace-tidy")
    tidy["entries"] = [e for e in tidy["entries"] if e["label"] not in _QUOTE_LABELS]
    tidy["config_rev"] = _rev(tidy)
    return tidy


def _load(tmp_path, rules):
    return load_saved(tmp_path, {"PIPELINE_RULES": rules})["PIPELINE_RULES"]


def test_an_untouched_older_factory_copy_becomes_the_current_rule(tmp_path):
    rules = copy.deepcopy(_factory_rules())
    _older_tidy(rules)
    out = _load(tmp_path, rules)
    tidy = next(r for r in out if r["name"] == "digit-and-whitespace-tidy")
    want = next(r for r in cs.load_factory_rules() if r["name"] == "digit-and-whitespace-tidy")
    assert tidy == {**want, "config_rev": _rev(want)}


def test_an_old_entry_text_in_an_untouched_copy_is_replaced_with_the_rule(tmp_path):
    """The entry upgrade alone would leave the quote entries out."""
    rules = copy.deepcopy(_factory_rules())
    tidy = _older_tidy(rules)
    e = next(e for e in tidy["entries"] if e["label"] == "tighten-digit-separators")
    e["pattern"] = "(\\d)\\s*([/,\\-])\\s*(\\d)"
    e["replacement"] = "\\1\\2\\3"
    tidy["config_rev"] = _rev(tidy)
    out = _load(tmp_path, rules)
    labels = [e["label"] for e in next(
        r for r in out if r["name"] == "digit-and-whitespace-tidy")["entries"]]
    assert set(_QUOTE_LABELS) <= set(labels)


def test_an_edited_copy_is_left_as_it_is(tmp_path):
    rules = copy.deepcopy(_factory_rules())
    tidy = _older_tidy(rules)
    tidy["entries"][-1]["note"] = "mine"
    edited = copy.deepcopy(tidy)
    out = _load(tmp_path, rules)
    got = next(r for r in out if r["name"] == "digit-and-whitespace-tidy")
    assert got["entries"] == edited["entries"]
    assert got["config_rev"] == edited["config_rev"]


def test_a_copy_without_a_sync_record_is_left_as_it_is(tmp_path):
    rules = copy.deepcopy(_factory_rules())
    del _older_tidy(rules)["config_rev"]
    out = _load(tmp_path, rules)
    got = next(r for r in out if r["name"] == "digit-and-whitespace-tidy")
    assert not set(_QUOTE_LABELS) & {e["label"] for e in got["entries"]}
    assert "config_rev" not in got


def test_the_rule_hash_matches_a_known_editor_value():
    """Pinned without node: settings.html _ruleHash of this rule."""
    rule = {"name": "a", "label": "A", "type": "callback:upper", "pattern": "x"}
    assert cs._editor_rule_hash(cs._editor_view(rule)) == _JS_KNOWN_HASH


_JS_KNOWN_HASH = "1cb72f88d12faf"


def _js_fn(src, name):
    i = src.index(f"  function {name}(")
    return src[i:src.index("\n  }\n", i) + 5]


@pytest.mark.skipif(_NODE is None, reason="node unavailable")
def test_the_rule_hash_is_the_editors():
    """Every factory rule, plus key orders JSON.stringify writes its own way
    (array-index keys first, UTF-16 order) and characters it escapes."""
    with open(_SETTINGS_HTML, encoding="utf-8") as fh:
        src = fh.read()
    js = "".join(_js_fn(src, n) for n in ("_sortDeep", "_withDefaults", "_ruleStrip", "_ruleHash"))
    rules = cs.load_factory_rules() + [
        {"name": "m", "label": "M", "type": "callback:map", "seeded": False,
         "map": {"10": "x", "2": "y", "b": "\t\"\\", "Ａ": "z", "\U0001d504": "q",
                 "Ä a": "\x01é"},
         "map_meta": {"b": 3}, "config_rev": "abcdef"},
        {"name": "u", "label": "U  ", "type": "callback:upper", "pattern": "(-)(\\w)",
         "tags": ["x"], "languages": ["de", "en"], "enabled": False},
    ]
    # The editor hashes the /settings/state copy: the schema dump.
    dumps = [cs._rule_adapter().validate_python(r).model_dump(exclude_none=True)
             for r in rules]
    p = subprocess.run([_NODE, "-"], capture_output=True, text=True, encoding="utf-8",
                       timeout=60, input=js + "\nconst rules = " + json.dumps(dumps)
                       + ";\nconsole.log(JSON.stringify(rules.map(_ruleHash)));\n")
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1]) == [_rev(r) for r in rules]
