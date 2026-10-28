"""A stored copy of the c4fefe2..00fd42f Spanish opening-mark entries is
upgraded to the factory text.

Their sentence-start group `[.!?…]\\s*` consumed the previous sentence's
terminator, so every second sentence got no ¿/¡; an admin who saved rules on
one of those releases kept that forever, because only UPGRADED_RULE_ENTRIES
carries a factory regex fix into a saved PIPELINE_RULES copy.
"""

import copy
import json
import os
import re

import pytest

from faster_whisper_backend import paths
from faster_whisper_backend.settings import config_renames as renames

# The entries as shipped in config.json at c4fefe2 (through 00fd42f^).
_OLD = {
    "es-opening-question-mark":
        ("(^\\s*|[.!?…]\\s*|\\n\\s*)(?!¿)([^.!?…¿\\n]+\\?)", "\\1¿\\2"),
    "es-opening-exclamation-mark":
        ("(^\\s*|[.!?…]\\s*|\\n\\s*)(?!¡)([^.!?…¡\\n]+!)", "\\1¡\\2"),
}


def _factory_rules():
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["PIPELINE_RULES"]


def _entry(rules, label):
    return next(e for r in rules for e in (r.get("entries") or [])
                if e.get("label") == label)


def test_stored_old_es_entries_are_upgraded_once():
    rules = copy.deepcopy(_factory_rules())
    for label, (pat, rep) in _OLD.items():
        e = _entry(rules, label)
        e["pattern"], e["replacement"] = pat, rep
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == [
        "es-punctuation/es-opening-question-mark",
        "es-punctuation/es-opening-exclamation-mark",
    ]
    assert raw["PIPELINE_RULES"] == _factory_rules()
    assert renames.upgrade_rule_entries(raw) == []


@pytest.mark.parametrize("label,text,want", [
    ("es-opening-question-mark", "Qué hora es? Dónde estás? Cómo te llamas?",
     "¿Qué hora es? ¿Dónde estás? ¿Cómo te llamas?"),
    ("es-opening-exclamation-mark", "Hola! Qué bien! Vamos!",
     "¡Hola! ¡Qué bien! ¡Vamos!"),
])
def test_upgraded_entry_marks_every_sentence(label, text, want):
    old_pat, old_rep = _OLD[label]
    assert re.sub(old_pat, old_rep, text) != want
    e = _entry(_factory_rules(), label)
    assert re.sub(e["pattern"], e["replacement"], text) == want
