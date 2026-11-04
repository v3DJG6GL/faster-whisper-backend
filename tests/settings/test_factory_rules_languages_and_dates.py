"""The shipped factory PIPELINE_RULES (config.json) and the stored-entry
upgrades / renames in config_renames.

"vom 3.-10 Oktober" once left the pipeline as "vom 310 Oktober" (final-cleanup
deleted ".-" between two numbers). A silently changed date or dose is the
worst kind of error in a medical dictation.
"""
import os
import re

import pytest

from faster_whisper_backend.settings import config_renames as renames
from faster_whisper_backend import paths
from faster_whisper_backend.pipeline import engine as pl_engine
from tests.settings._rules_helpers import (
    assert_same_output_as_old as _assert_same_output_as_old,
    assert_upgrades_to_factory_once as _assert_upgrades_to_factory_once,
    entry_by_label as _entry,
    factory_rules as _factory_rules,
    factory_rules_with_entry as _factory_rules_with_entry,
    fast_postprocess as _fast_postprocess,
    load_saved as _load_saved,
)


def _by_name():
    return {r["name"]: r for r in _factory_rules()}


_TRIM = {"name": "trim-edges", "label": "Trim", "type": "terminal"}


def _install(app_module, rules):
    app_module.cfg.PIPELINE_RULES = rules
    pl_engine.rebuild_caches()


def _install_factory(app_module):
    _install(app_module, _factory_rules())


def _install_one(app_module, name):
    _install(app_module, [_by_name()[name], dict(_TRIM)])


def _pp(text, language="de"):
    return pl_engine._postprocess_text(text, model_name="", language=language)


def _hyphen_pattern():
    entry = _entry(_factory_rules(), "delete-punct-before-hyphen")
    return entry["pattern"], entry["replacement"]


@pytest.mark.parametrize("text", [
    "Die Patientin ist vom 3.-10 Oktober abwesend.",
    "Termin am 12.-14.03.2026",
    "Dosis 2,-5 mg",
    "vom 3.- 10. Oktober",
])
def test_punct_before_hyphen_is_kept_after_a_digit(text):
    pat, repl = _hyphen_pattern()
    assert re.sub(pat, repl, text) == text


def test_punct_before_hyphen_still_removed_after_a_word():
    """The rule's documented purpose (a,-b → ab) is unchanged."""
    pat, repl = _hyphen_pattern()
    assert re.sub(pat, repl, "Belastung.- Neue Zeile") == "BelastungNeue Zeile"
    assert re.sub(pat, repl, "a,-b") == "ab"


def test_date_range_survives_the_whole_factory_pipeline(app_module):
    _install_factory(app_module)
    out = _pp(" Die Patientin ist vom 3.-10 Oktober abwesend Punkt")
    assert out == "Die Patientin ist vom 3.-10 Oktober abwesend."


def test_capitalize_after_terminator_is_fast_on_a_run_of_paragraph_breaks(app_module):
    """`\\n+\\s*` tried every split of a newline run at every start when no
    lowercase letter follows it, so a repeated "Neuer Absatz" (one paragraph
    break each) took cubic time: 2000 newlines ran 23 s on the event loop of
    /transcribe. `\\n\\s*` was still quadratic (32000 newlines: 22 s): it
    started at every newline of the run and scanned to its end. A match now
    starts only at the last newline of a run."""
    rule = _by_name()["capitalize-after-terminator"]
    _install(app_module, [rule, dict(_TRIM)])
    for text in ("Befund" + "\n" * 40000 + "Ende", "Befund" + "\n \n" * 20000 + "Ende"):
        assert _fast_postprocess(text) == text
    for old in (r"([.?!]\s+|\n+\s*)([a-zäöüß])", r"([.?!]\s+|\n\s*)([a-zäöüß])"):
        for s in ("Gut. danke\n\n  ja", "a\n \n\tb\n\n\n.c", "x.y! über\n1\nß",
                  "a\r\n\x0b\nb\n\x85\u2028c\n\x0c\x1cä"):
            assert (re.sub(rule["pattern"], pl_engine._upper_callback, s)
                    == re.sub(old, pl_engine._upper_callback, s))


def test_language_specific_rules_are_tagged():
    rules = _by_name()
    for name in ("de-strip-and-lowercase-non-noun", "de-split-punctuation-keywords",
                 "ch-symbol-cleanup", "de-dictation-map"):
        assert rules[name].get("languages") == ["de"], name
    assert rules["es-punctuation"]["languages"] == ["es"]
    # A language-prefixed slug without a language tag would run everywhere.
    for name, r in rules.items():
        if re.match(r"^(de|ch|es|fr|it|en)-", name):
            assert r.get("languages"), f"{name} has a language prefix but no tag"


def test_hyphen_capitalization_is_german_only(app_module):
    """It exists for the German ' Bindestrich ' compounds; untagged, it ran on
    every language and turned 'well-known' into 'well-Known'."""
    _install_one(app_module, "capitalize-letter-after-hyphen")
    run = _pp
    assert run("A well-known x-ray", "en") == "A well-known x-ray"
    assert run("Una niña franco-alemana", "es") == "Una niña franco-alemana"
    assert run("Eisen-infusion", "de") == "Eisen-Infusion"


@pytest.mark.parametrize("text, want", [
    (" Die 45-jährige Patientin.", "Die 45-jährige Patientin"),
    (" 3-mal pro Tag.", "3-mal pro Tag"),
    (" Ein 3-teiliges Set.", "Ein 3-teiliges Set"),
    (" Die Eisen-infusion.", "Die Eisen-Infusion"),
    (" Eisen Bindestrich infusion.", "Eisen-Infusion"),
])
def test_hyphen_capitalization_leaves_a_digit_compound_lowercase(app_module, text, want):
    """A German digit compound ("45-jährige", "3-mal") is lowercase after the
    hyphen; the rule capitalised it ("45-Jährige")."""
    _install_factory(app_module)
    assert _pp(text) == want


def test_spanish_opening_marks(app_module):
    _install_one(app_module, "es-punctuation")
    run = _pp
    assert run("Qué hora es? No lo sé. Qué bien!", "es") == "¿Qué hora es? No lo sé. ¡Qué bien!"
    assert run("¿Ya está? ¡Sí!", "es") == "¿Ya está? ¡Sí!", "existing marks are not doubled"
    assert run("Wie spät ist es?", "de") == "Wie spät ist es?", "German is untouched"


def test_spanish_opening_marks_on_consecutive_sentences(app_module):
    """The sentence-start anchor is zero-width: a consumed terminator made the
    scan resume past it, so only every other question got its opening mark."""
    _install_one(app_module, "es-punctuation")
    run = lambda t: _pp(t, "es")
    assert run("Qué? Cómo?") == "¿Qué? ¿Cómo?"
    assert run("Hola! Adiós!") == "¡Hola! ¡Adiós!"
    assert run("Qué hora es? Y luego? Vale.") == "¿Qué hora es? ¿Y luego? Vale."
    assert run("Uno\nDos? Tres!") == "Uno\n¿Dos? ¡Tres!"
    # digit guard: a decimal point is not a sentence end
    assert run("Vale. Cuesta 3.5 euros?") == "Vale. ¿Cuesta 3.5 euros?"
    assert run("Son las 10.23? Vale!") == "¿Son las 10.23? ¡Vale!"


def test_spanish_patterns_pass_the_regex_guard():
    from faster_whisper_backend.pipeline import regex_guard
    entries = _by_name()["es-punctuation"]["entries"]
    regex_guard.validate([[e["label"], e["pattern"], e["replacement"]] for e in entries])


def test_spanish_marks_survive_the_whole_factory_pipeline(app_module):
    """es-punctuation runs ahead of strip-auto-punctuation, which takes an
    opening mark along with its closing one: no ¿/¡ lands on the wrong
    sentence or is left orphaned."""
    _install_factory(app_module)
    run = lambda t: _pp(t, "es")
    assert run("Hola. Tienes 3?") == "Hola ¿Tienes 3?"
    assert run("Cuántos son? 25!") == "Cuántos son ¡25!"
    assert run("¿Qué hora es?") == "Qué hora es"
    off = _factory_rules()
    next(r for r in off if r["name"] == "strip-auto-punctuation")["enabled"] = False
    _install(app_module, off)
    assert run("Qué hora es? Qué bien!") == "¿Qué hora es? ¡Qué bien!"


def test_german_map_does_not_touch_spanish(app_module):
    _install_factory(app_module)
    assert "Punkt" in _pp("El Punkt rojo", "es")


def test_rule_rename_migrates_every_stored_reference():
    raw = {
        "PIPELINE_RULES": [{"name": "dictation-map", "label": "x"},
                           {"name": "other", "label": "y"}],
        "CAPTURES_PIPELINE_RULES_EXCLUDE": ["capitalize-after-terminator", "dictation-map"],
        "MODEL_OVERRIDES": {"m": {"PIPELINE_RULES_EXCLUDE": ["dictation-map"]}},
        "OVERRIDE_PROFILES": {"p": {"PIPELINE_RULES_INCLUDE": ["dictation-map", "de-dictation-map"]}},
    }
    # Both run on every stored / env path (config_store._migrate_legacy_keys,
    # config.py's JSON env loop): the bundles' slug lists are migrate_bundle's.
    renames.migrate_rule_slugs(raw)
    renames.migrate_bundle_keys(raw)
    assert [r["name"] for r in raw["PIPELINE_RULES"]] == ["de-dictation-map", "other"]
    assert raw["CAPTURES_PIPELINE_RULES_EXCLUDE"] == ["capitalize-after-terminator", "de-dictation-map"]
    assert raw["MODEL_OVERRIDES"]["m"]["PIPELINE_RULES_EXCLUDE"] == ["de-dictation-map"]
    assert raw["OVERRIDE_PROFILES"]["p"]["PIPELINE_RULES_INCLUDE"] == ["de-dictation-map"]


def test_rule_rename_keeps_an_already_migrated_rule():
    raw = {"PIPELINE_RULES": [{"name": "de-dictation-map", "label": "new"},
                              {"name": "dictation-map", "label": "old"}]}
    renames.migrate_rule_slugs(raw)
    assert [r["name"] for r in raw["PIPELINE_RULES"]] == ["de-dictation-map", "dictation-map"]


def test_rename_slugs_keeps_container_type():
    assert renames.rename_slugs({"dictation-map"}) == {"de-dictation-map"}
    assert renames.rename_slugs(["a", "dictation-map"]) == ["a", "de-dictation-map"]
    assert renames.rename_slugs(None) is None


def test_no_factory_default_names_a_renamed_rule():
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        text = fh.read()
    for old in renames.RENAMED_RULES:
        assert f'"{old}"' not in text, f"config.json still names {old}"


def test_upgraded_entries_match_config_json():
    """The NEW side of every upgrade is what config.json ships — an upgrade
    that wrote anything else would fork stored copies from the factory."""
    for label, (_old, new) in renames.UPGRADED_RULE_ENTRIES.items():
        e = _entry(_factory_rules(), label)
        assert (e["pattern"], e["replacement"]) == new
    assert set(renames.EARLIER_RULE_ENTRIES) <= set(renames.UPGRADED_RULE_ENTRIES)


def _assert_upgraded_once(label, path):
    rules = _factory_rules_with_entry(label, renames.UPGRADED_RULE_ENTRIES[label][0])
    _assert_upgrades_to_factory_once(rules, [path])


def test_stored_old_factory_entry_is_upgraded_once():
    _assert_upgraded_once("tighten-quote-spacing",
                          "digit-and-whitespace-tidy/tighten-quote-spacing")


def test_stored_old_hyphen_entry_is_upgraded_once():
    """A stored copy from before c4fefe2 still merged "3.-10" into "310"."""
    _assert_upgraded_once("delete-punct-before-hyphen",
                          "final-cleanup/delete-punct-before-hyphen")
    (old_pat, old_rep), (new_pat, new_rep) = \
        renames.UPGRADED_RULE_ENTRIES["delete-punct-before-hyphen"]
    assert re.sub(old_pat, old_rep, "vom 3.-10 Oktober") == "vom 310 Oktober"
    assert re.sub(new_pat, new_rep, "vom 3.-10 Oktober") == "vom 3.-10 Oktober"
    assert re.sub(new_pat, new_rep, "a,-b") == "ab"


def test_stored_old_strip_entry_is_upgraded_once():
    """A stored copy from before es-punctuation moved ahead of the strip left
    a Whisper-written "¿Qué hora es?" as an orphaned "¿Qué hora es"."""
    _assert_upgraded_once("Strip terminators & commas",
                          "strip-auto-punctuation/Strip terminators & commas")
    (old_pat, old_rep), (new_pat, new_rep) = \
        renames.UPGRADED_RULE_ENTRIES["Strip terminators & commas"]
    assert re.sub(old_pat, old_rep, "¿Qué hora es?") == "¿Qué hora es"
    assert re.sub(new_pat, new_rep, "¿Qué hora es?") == "Qué hora es"
    assert re.sub(new_pat, new_rep, "¿Vienes a las 10?") == "¿Vienes a las 10?"


def test_stray_symbols_strip_both_halves_of_a_typographic_quote(app_module):
    """The strip took the German opening „ but not its closing “, so a quote
    Whisper wrote as „…“ kept an orphaned closing mark; a stored copy of the
    old entry is upgraded."""
    _install_factory(app_module)
    run = _pp
    assert run(" Er sagte „Hallo“ zu mir.", "de") == "Er sagte Hallo zu mir"
    assert run(" Il a dit «Bonjour» à moi.", "fr") == "Il a dit Bonjour à moi"
    _assert_upgraded_once("strip-stray-symbols", "strip-stray-symbols/strip-stray-symbols")
    old_pat, old_rep = renames.UPGRADED_RULE_ENTRIES["strip-stray-symbols"][0]
    assert re.sub(old_pat, old_rep, "„Hallo“") == "Hallo“"


def test_edited_factory_entry_is_left_alone():
    old_pat, _old_rep = renames.UPGRADED_RULE_ENTRIES["tighten-quote-spacing"][0]
    # admin's own replacement
    rules = _factory_rules_with_entry("tighten-quote-spacing", (old_pat, '"\\1" '))
    raw = {"PIPELINE_RULES": rules}
    assert renames.upgrade_rule_entries(raw) == []
    assert _entry(raw["PIPELINE_RULES"], "tighten-quote-spacing")["pattern"] == old_pat
    assert renames.upgrade_rule_entries({}) == []
    assert renames.upgrade_rule_entries({"PIPELINE_RULES": "junk"}) == []


def test_load_overrides_upgrades_a_stored_old_quote_entry(tmp_path):
    old, new = renames.UPGRADED_RULE_ENTRIES["tighten-quote-spacing"]
    rules = _factory_rules_with_entry("tighten-quote-spacing", old)
    out = _load_saved(tmp_path, {"PIPELINE_RULES": rules})
    got = _entry(out["PIPELINE_RULES"], "tighten-quote-spacing")
    assert (got["pattern"], got["replacement"]) == new


@pytest.mark.parametrize("text, want", [
    (" Datum 3 Schrägstrich 4 Schrägstrich 2026.", "3/4/2026"),
    (" Aktenzeichen 12 Schrägstrich 3 Schrägstrich 4 Schrägstrich 5.", "12/3/4/5"),
    (" Datum 3 / 4 / 2026.", "3/4/2026"),
    (" Version 1 Komma 2 Komma 3.", "1,2,3"),
    (" Blutdruck 120 Schrägstrich 80.", "120/80"),
])
def test_every_separator_of_a_digit_chain_is_tightened(app_module, text, want):
    """tighten-digit-separators consumed the digit after the separator, so
    re.sub could not start the next match on it and every second separator
    of a chain kept its spaces: "3 Schrägstrich 4 Schrägstrich 2026" became
    "3/4 / 2026"."""
    _install_factory(app_module)
    out = _pp(text)
    assert want in out.split()


@pytest.mark.parametrize("text, want", [
    (" Base Excess Minus 2 Komma 5.", "Base Excess - 2,5"),
    (" BE -2 Komma 5.", "BE -2,5"),
    (" Sphäre -1 Komma 25.", "Sphäre -1,25"),
    (" Sphäre Minus 1 Komma 25, Zylinder Minus 0 Komma 5.",
     "Sphäre - 1,25 Zylinder - 0,5"),
    (" Kalium 3 Komma 5 Bindestrich 5 Komma 0.", "Kalium 3,5-5,0"),
    (" Temperatur 37 Komma 5 - 38 Komma 2.", "Temperatur 37,5/38,2"),
    (" am 1 2 / 0 3 / 2026.", "am 12/03/2026"),
    (" am 0 3 / 0 4 / 2026.", "am 03/04/2026"),
])
def test_a_dictated_decimal_or_split_date_is_joined_by_default(app_module, text, want):
    """strip-auto-punctuation deletes Whisper's list commas before the digit
    tidy, so every comma tighten-digit-separators sees is a dictated decimal
    mark. A comma arm that skipped a number after a minus or a range end left
    "- 2, 5" and "3,5-5, 0", and merge-digit-pair entries that skipped a
    number inside a / group left "12/0 3/2026"."""
    _install_factory(app_module)
    assert _pp(text) == want


def test_stored_old_digit_separator_entry_is_upgraded_once():
    _assert_upgraded_once("tighten-digit-separators",
                          "digit-and-whitespace-tidy/tighten-digit-separators")
    (old_pat, old_rep), (new_pat, new_rep) = \
        renames.UPGRADED_RULE_ENTRIES["tighten-digit-separators"]
    assert re.sub(old_pat, old_rep, "3 / 4 / 2026") == "3/4 / 2026"
    assert re.sub(new_pat, new_rep, "3 / 4 / 2026") == "3/4/2026"


@pytest.mark.parametrize("text, want", [
    (" Datum 3-4-2026.", "3/4/2026"),
    (" Datum 3 - 4 - 2026.", "3/4/2026"),
    (" Seiten 1-2-3.", "1/2/3"),
    (" Termin 10 - 12 Uhr.", "10/12"),
])
def test_every_hyphen_of_a_digit_chain_becomes_a_slash(app_module, text, want):
    """normalize-digit-range consumed the digit after the hyphen, so with a
    one-digit day or month every second hyphen stayed: "3-4-2026" became
    "3/4-2026"."""
    _install_factory(app_module)
    out = _pp(text)
    assert want in out.split()


@pytest.mark.parametrize("single_regex", [False, True])
def test_stored_old_digit_range_entry_is_upgraded(tmp_path, single_regex):
    """A copy saved before 8cde7fd holds the rule as a single `regex` card,
    which becomes a one-entry list labelled with its slug first."""
    rules = _factory_rules()
    (old_pat, old_rep), new = renames.UPGRADED_RULE_ENTRIES["normalize-digit-range"]
    i = next(i for i, r in enumerate(rules) if r["name"] == "normalize-digit-range")
    if single_regex:
        rules[i] = {"name": "normalize-digit-range", "label": "Normalize digit range",
                    "type": "regex", "pattern": old_pat, "replacement": old_rep}
    else:
        rules[i]["entries"][0].update(pattern=old_pat, replacement=old_rep)
    got = next(r for r in _load_saved(tmp_path, {"PIPELINE_RULES": rules})["PIPELINE_RULES"]
               if r["name"] == "normalize-digit-range")["entries"]
    assert [(e["label"], e["pattern"], e["replacement"]) for e in got] == [
        ("normalize-digit-range", *new)]
    assert re.sub(old_pat, old_rep, "3-4-2026") == "3/4-2026"
    assert re.sub(*new, "3-4-2026") == "3/4/2026"


@pytest.mark.parametrize("text, want", [
    (" Medikamente Neue Zeile Bindestrich Ramipril 5 Neue Zeile Bindestrich 2 Tabletten",
     "Medikamente\n-Ramipril 5\n-2 Tabletten"),
    (" Blutdruck 120 Schrägstrich 80 Neuer Absatz Bindestrich 3 Tage Pause",
     "Blutdruck 120/80\n\n-3 Tage Pause"),
    (" Natrium 140 neuer Absatz Minus 3 im Verlauf Punkt", "Natrium 140\n\n- 3 im Verlauf."),
    (" Score 12 Neue Zeile Schrägstrich 4", "Score 12\n/ 4"),
])
def test_a_dictated_line_break_between_two_numbers_is_kept(app_module, text, want):
    """tighten-digit-separators runs after de-dictation-map, and its `\\s*`
    took the line break a dictated "Neue Zeile" / "Neuer Absatz" left after a
    number: the next list item glued onto it as a range or fraction
    ("Ramipril 5-2 Tabletten", "Score 12/4")."""
    _install_factory(app_module)
    assert _pp(text) == want


def test_digit_range_keeps_a_line_break_between_two_numbers():
    entry = _by_name()["normalize-digit-range"]["entries"][0]
    sub = lambda t: re.sub(entry["pattern"], entry["replacement"], t)
    assert sub("Ramipril 5\n-2 Tabletten") == "Ramipril 5\n-2 Tabletten"
    assert sub("Termin 10 - 12 Uhr") == "Termin 10/12 Uhr"


def _assert_silence_loop_is_fast(app_module, unit, n):
    _install_factory(app_module)
    assert _fast_postprocess(" Das war es " + unit * n + "Danke.") == "Das war es Danke"


@pytest.mark.parametrize("unit", ["... ", " .", " ,"])
def test_space_before_closing_punct_is_fast_on_a_hallucinated_dot_loop(app_module, unit):
    """Whisper hallucinates "... ... ..." on silence; strip-auto-punctuation
    deletes the marks and leaves one long space run, which the "  +" collapse
    only reaches later in the card. `[ \\t]+` started a match at every space
    of the run and scanned to its end when no closing mark followed: 40000
    items took 7 s on /transcribe. A match now starts only at a run's first
    space (0.05 s)."""
    _assert_silence_loop_is_fast(app_module, unit, 40000)


def test_space_before_closing_punct_output_is_unchanged():
    _assert_same_output_as_old("space-before-closing-punct", (
        "a , b", "a \t \t. b  ;c", "( 45 Jahre ) klagt .", "x   y", " \t", "a  ,\n  .",
        "Wert 12 % ] } ! ?", "\t,\t.\t"))


def test_stored_old_closing_punct_entry_is_upgraded_once():
    _assert_upgraded_once("space-before-closing-punct",
                          "digit-and-whitespace-tidy/space-before-closing-punct")


def test_closing_punct_rule_saved_before_june_is_upgraded(tmp_path):
    """A copy saved before 8cde7fd holds the single `regex` rule with the
    pattern from before `%` joined the class, which UPGRADED_RULE_ENTRIES
    does not know: it kept the quadratic scan on a "... ... ..." loop."""
    old_pat, old_rep = renames.EARLIER_RULE_ENTRIES["space-before-closing-punct"][0]
    rules = _factory_rules()
    rules.insert(0, {"name": "space-before-closing-punct",
                     "label": "Remove space before closing punctuation",
                     "type": "regex", "pattern": old_pat, "replacement": old_rep})
    got = next(r for r in _load_saved(tmp_path, {"PIPELINE_RULES": rules})["PIPELINE_RULES"]
               if r["name"] == "space-before-closing-punct")["entries"]
    new = renames.UPGRADED_RULE_ENTRIES["space-before-closing-punct"][1]
    assert [(e["label"], e["pattern"], e["replacement"]) for e in got] == [
        ("space-before-closing-punct", *new)]


@pytest.mark.parametrize("unit, n", [("* ", 40000), ("( ) ", 20000)])
def test_trim_trailing_period_is_fast_on_a_hallucinated_symbol_loop(app_module, unit, n):
    """Whisper hallucinates "* * * *" (or "( ) ( )") on silence;
    strip-stray-symbols deletes the symbols and leaves one long space run.
    `\\s*\\.+\\s*$` started a match at every space of the run, took the rest
    of it and found no dot: 40000 spaces took 8 s on /transcribe. A match now
    starts only at a run's first space."""
    _assert_silence_loop_is_fast(app_module, unit, n)


def test_trim_trailing_period_is_fast_on_a_hallucinated_dot_run(app_module):
    """Whisper also writes "Das war es......" on silence, and the rule runs
    before strip-auto-punctuation deletes the dots. A match started at every
    dot of the run, took the rest of it and found text after it: 24000 dots
    took 8 s. A match now starts only at a run's first dot."""
    _assert_silence_loop_is_fast(app_module, ".", 20000)


def test_trim_trailing_period_output_is_unchanged():
    _assert_same_output_as_old("trim-trailing-period", (
        "Wert 5.", "gut. . . ", "x. .", "a  \t.\n", "z.B. morgen", " . ", "...",
        "Ende.\n\n", "a b", "  ", "1.5 mg .\t.", "Satz. Zwei", "es...... Danke",
        "5\n.\n\n.", ". .", "a.. b.."))


def test_stored_old_trailing_period_entry_is_upgraded_once():
    _assert_upgraded_once("trim-trailing-period",
                          "strip-trailing-period/trim-trailing-period")


def test_tidy_newlines_is_fast_on_a_symbol_loop_with_the_whitespace_card_off(app_module):
    """collapse-multi-space shrinks the space run that strip-stray-symbols
    leaves of a hallucinated "* * * *" loop, but an admin may switch its
    card off (or exclude it). tidy-newlines then got the run whole, and its
    two adjacent space runs tried every split of it at every space: 1000
    items took 2 s, 4000 took minutes."""
    rules = _factory_rules()
    next(r for r in rules if r.get("name") == "digit-and-whitespace-tidy")["enabled"] = False
    _install(app_module, rules)
    out = _fast_postprocess(" Das war es " + "* " * 2000 + "Danke.")
    assert out.split() == ["Das", "war", "es", "Danke"]


def test_tidy_newlines_output_is_unchanged():
    _assert_same_output_as_old("tidy-newlines", (
        "a ,\n, b", "Hallo   ,   \n   Welt", "x\n\n\ty", "\t\ta\n\t\n,a\t", "\n \n ",
        "a , , \n", ",\n,", "a  b", " \t ", "Zeile 1,\nZeile 2", "\n", "a ,b\n"))


def test_stored_old_tidy_newlines_entry_is_upgraded_once():
    _assert_upgraded_once("tidy-newlines", "final-cleanup/tidy-newlines")
