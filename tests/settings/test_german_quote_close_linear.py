"""The factory tighten-german-quote-close entry is linear on a run of „
openings with no closing “, with the same output as before.

Its quoted text could run past a later „, so from every „ the scan went on
to the end of the line looking for a “. A whole transcript is one line, so a
repeated dictated "Anführungszeichen unten" (or Whisper looping on it) took
quadratic time on /transcribe (10000 items: 10 s).
"""

import random
import re
import time

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config_renames as renames
from tests.settings._rules_helpers import (
    assert_same_output_as_old as _assert_same_output_as_old,
    assert_upgrades_to_factory_once as _assert_upgrades_to_factory_once,
    entry_by_label as _entry,
    factory_rules as _factory_rules,
    factory_rules_with_entry as _factory_rules_with_entry,
    fast_postprocess as _fast_postprocess,
    load_saved,
)

_LABEL = "tighten-german-quote-close"


def test_repeated_dictated_openings_are_fast(app_module):
    app_module.cfg.PIPELINE_RULES = _factory_rules()
    pl_engine.rebuild_caches()
    out = _fast_postprocess(" Anführungszeichen unten Hallo" * 5000)
    assert out.count("„") == 5000


def test_a_run_of_unclosed_openings_is_fast():
    e = _entry(_factory_rules(), _LABEL)
    for text in ("„Hallo " * 20000, "„ x " * 20000, "„Hallo " * 20000 + " “",
                 "„a „ \t" * 10000 + "b"):
        t0 = time.perf_counter()
        re.sub(e["pattern"], e["replacement"], text)
        assert time.perf_counter() - t0 < 0.5, repr(text[:12])


_SENTENCES = (
    "Er sagte „Hallo “ zu mir.", "Er sagte „Hallo“ zu mir.", "„Hallo  \t “",
    "„ “", "„“", "„", "“", "Er sagte “Hallo „ Welt “.", "„a „b “ c “",
    "„a „b“ c “", "„Hallo \n“", "„Hallo\n „Welt “", "„eins “ und „zwei “",
    "„eins “ „zwei “ „drei “", "She said “hello ” and „bye “.", "„a \t \t“b “",
    "„„ “", "„ „ “ “", "Text ohne Anführungszeichen.", "", " ", "„x“ “ “",
    "„Zitat: „innen “ außen “", "„a\r\n “", "„Hallo " * 50 + "“",
)


def test_the_output_is_unchanged():
    _assert_same_output_as_old(_LABEL, _SENTENCES)
    e = _entry(_factory_rules(), _LABEL)
    old = renames.UPGRADED_RULE_ENTRIES[_LABEL][0]
    new, old_re = re.compile(e["pattern"]), re.compile(old[0])
    rnd = random.Random(0)
    alphabet = "„“  \ta\nx"
    for _ in range(20000):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 16)))
        assert new.sub(e["replacement"], s) == old_re.sub(old[1], s), repr(s)


def test_a_stored_copy_of_the_old_entry_is_upgraded_once():
    _assert_upgrades_to_factory_once(
        _factory_rules_with_entry(_LABEL, renames.UPGRADED_RULE_ENTRIES[_LABEL][0]),
        [f"digit-and-whitespace-tidy/{_LABEL}"])


def test_an_edited_entry_is_left_as_it_is(tmp_path):
    mine = ("(„[^“\\n]*?)[ \\t]+“", "\\1 “")
    rules = _factory_rules_with_entry(_LABEL, mine)
    assert renames.upgrade_rule_entries({"PIPELINE_RULES": rules}) == []
    got = _entry(load_saved(tmp_path, {"PIPELINE_RULES": rules})["PIPELINE_RULES"], _LABEL)
    assert (got["pattern"], got["replacement"]) == mine
