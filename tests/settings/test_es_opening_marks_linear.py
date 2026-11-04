"""The Spanish ¿/¡ opening-mark entries are linear on a long whitespace run,
with the same output as before.

Their sentence-start `\\s*` and the sentence itself could split a space run
every way, so a hallucinated "* * * *" loop after a terminator (one long
space run once strip-stray-symbols deleted the symbols) took quadratic time
with language='es' (8000 spaces: 5 s), and every newline of a run of
paragraph breaks started a scan to its end.
"""

import copy
import time

import pytest

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config_renames as renames
from tests.settings._rules_helpers import (
    assert_same_output_as_old as _assert_same_output_as_old,
    assert_upgrades_to_factory_once as _assert_upgrades_to_factory_once,
    factory_rules as _factory_rules,
)

_LABELS = ("es-opening-question-mark", "es-opening-exclamation-mark")


@pytest.mark.parametrize("text, want", [
    ("Hola. " + "* " * 20000 + "gracias.", "Hola gracias"),
    ("Hola." + "\n" * 20000 + "Qué tal?", "Hola" + "\n" * 20000 + "Qué tal"),
    ("Hola." + "\n " * 10000 + "Vamos!", "Hola" + "\n" * 10000 + "Vamos"),
], ids=["symbol-loop", "paragraph-breaks", "spaced-paragraph-breaks"])
def test_a_long_whitespace_run_is_fast(app_module, text, want):
    app_module.cfg.PIPELINE_RULES = copy.deepcopy(_factory_rules())
    pl_engine.rebuild_caches()
    t0 = time.perf_counter()
    out = pl_engine._postprocess_text(text, model_name="", language="es")
    assert time.perf_counter() - t0 < 1.0
    assert out == want


_SPANISH = (
    "Qué hora es?", "Hola! Qué tal? Bien, gracias.", "Dónde estás? Estoy en casa! Vienes?",
    "Cuántos años tienes? Tengo 25.", "Cuesta 3.5 euros? Sí!", "Son las 10.23? Vale!",
    "Uno\nDos? Tres!", "¿Ya está? ¡Sí!", "Me dijo: vienes mañana? No lo sé.",
    "Hola... Qué pasa?", "Bueno… Y ahora?", "Increíble!!!", "En serio?!", "De verdad!?",
    "Llegó el 1.5.2024? No.", "Hola.  Qué tal?", "Hola.\n\nQué tal?", "Hola.\n \n Qué tal?",
    "  Qué hora es?  ", "Qué hora es ?", "Hola. ?", "Hola.\n?", "Hola. \n ?", "Qué? ? !",
    "Sabes qué? No me importa! Pero dime, por qué?", "Es el 3. Quién viene?",
    "No.\tPor qué?", "Hola,\nqué tal?", "¿Qué hora es? Las tres! ¿Y tú?",
    "Primero. Segundo? Tercero!\nCuarto? Quinto.", "Hola. * * * gracias?",
    "a ? b\x0b!\r\nc  ?",
)


@pytest.mark.parametrize("label", _LABELS)
def test_the_output_is_unchanged(label):
    _assert_same_output_as_old(label, _SPANISH)


def test_a_stored_copy_of_the_previous_entries_is_upgraded_once():
    rules = copy.deepcopy(_factory_rules())
    es = next(r for r in rules if r["name"] == "es-punctuation")
    for e in es["entries"]:
        e["pattern"], e["replacement"] = renames.UPGRADED_RULE_ENTRIES[e["label"]][0]
    _assert_upgrades_to_factory_once(rules, [f"es-punctuation/{x}" for x in _LABELS])
