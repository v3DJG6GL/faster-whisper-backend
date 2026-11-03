"""The factory tighten-quote-spacing entry is linear on a long space run
inside a quote, with the same output as before.

Its lazy quoted text tried the closing space run at every character it grew
by, so a dictated Gänsefüßchen before a hallucinated "* * * *" loop (one long
space run once strip-stray-symbols deleted the symbols) took quadratic time
on /transcribe (20000 items: 4 s).
"""

import copy
import random
import re
import time

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config_renames as renames
from tests.settings._rules_helpers import (
    assert_upgrades_to_factory_once as _assert_upgrades_to_factory_once,
    factory_rules as _factory_rules,
)

_LABEL = "tighten-quote-spacing"


def _entry(rules):
    return next(e for r in rules for e in (r.get("entries") or [])
                if e.get("label") == _LABEL)


def test_a_space_run_inside_a_dictated_quote_is_fast(app_module):
    app_module.cfg.PIPELINE_RULES = copy.deepcopy(_factory_rules())
    pl_engine.rebuild_caches()
    text = " Er sagte Gänsefüßchen Hallo" + " *" * 20000 + " Welt."
    t0 = time.perf_counter()
    out = pl_engine._postprocess_text(text, model_name="", language="de")
    assert time.perf_counter() - t0 < 1.0
    assert out == 'Er sagte "Hallo Welt'


def test_a_space_run_inside_a_quote_is_fast():
    e = _entry(_factory_rules())
    for text in ('Er sagte "x' + " " * 20000 + "y",
                 'Er sagte "x' + " \t" * 10000 + "y\n",
                 '"a' + " " * 20000 + 'b "c' + " " * 20000 + "d",
                 '"' + " " * 20000 + '"', '"' + " " * 20000):
        t0 = time.perf_counter()
        re.sub(e["pattern"], e["replacement"], text)
        assert time.perf_counter() - t0 < 0.5, repr(text[:12])


_SENTENCES = (
    'Er sagte "Hallo" zu mir.', 'Er sagte " Hallo " zu mir.', 'Sie rief "  Hilfe!  "',
    'Das Wort "Haus" und das Wort "Maus".', 'Das Wort " Haus "und das Wort" Maus ".',
    'Er sagte "Hallo', 'Er sagte " Hallo  ', 'Er sagte "', 'Er sagte "   ', '""',
    '" "', '"\t"', 'Er nannte es "den "großen" Tag".', '"Ja", sagte sie, "nein."',
    '" Ja ", sagte sie, " nein. "', 'Ein 3" Rohr und ein 5" Rohr.', 'Zitat: "Sein oder\nnicht sein"',
    'Zitat: " Sein oder \n nicht sein "', '"Erste Zeile\n"zweite Zeile', 'Er sagte "x"y"z" ende',
    'She said "hello" and "goodbye".', 'She said " hello world ".', 'The "quick" brown "fox',
    '"Why?" he asked. "Because!"', '" Why ? " he asked .', 'He wrote "a, b, c" (sic).',
    '("Klammer")', '[" eckig "]', '"Komma ", dann weiter', 'Er fragte: "Kommst du?"',
    '"Mehrere   Leerzeichen   innen"', '" \t gemischt \t "', 'Erst "eins" dann "zwei" dann "drei"',
    'Zeile eins\n"Zeile zwei"\nZeile drei', '"offen\nüber\nZeilen', 'Ohne Anführungszeichen.',
    '', ' ', '"', 'Er sagte "Hallo" " " "Welt"', '„Deutsch“ und "Englisch"',
    'Text "\n" Text', 'a "b \r\n c" d', 'Er sagte "Hallo  \n  Welt  " und ging.',
    'Preis: 5 " Zoll', 'Er sagte" Hallo" zu mir',
)


def test_the_output_is_unchanged():
    e = _entry(_factory_rules())
    old = renames.UPGRADED_RULE_ENTRIES[_LABEL][0]
    new, old_re = re.compile(e["pattern"]), re.compile(old[0])
    for s in _SENTENCES:
        assert new.sub(e["replacement"], s) == old_re.sub(old[1], s), repr(s)
    rnd = random.Random(0)
    alphabet = '"  \t\na.,!b\r'
    for _ in range(20000):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 12)))
        assert new.sub(e["replacement"], s) == old_re.sub(old[1], s), repr(s)


def test_a_stored_copy_of_either_earlier_entry_is_upgraded_once():
    for old in (renames.UPGRADED_RULE_ENTRIES[_LABEL][0],
                *renames.EARLIER_RULE_ENTRIES[_LABEL]):
        rules = copy.deepcopy(_factory_rules())
        _entry(rules)["pattern"], _entry(rules)["replacement"] = old
        _assert_upgrades_to_factory_once(rules, [f"digit-and-whitespace-tidy/{_LABEL}"])
