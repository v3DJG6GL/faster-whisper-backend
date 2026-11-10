"""dictation_map.compile_map: key word bounds and the "dictated punctuation
wins" prefix, which must stay linear on a long hallucinated punctuation run.

A plain greedy `[.,:;!?…]+` prefix was retried from every start position
inside the run, each attempt backtracking char by char into the failed
lookahead — 8000 × "…" took over a second, quadratic in the run length. The
factory strip rule keeps `…` and `:` runs, so such a run reaches the map on
/transcribe, every streaming final and every captures reapply.
"""

import copy
import re
import time

import pytest

from faster_whisper_backend.pipeline import dictation_map
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config as cfg
from tests.settings._rules_helpers import factory_rules as _factory_rules


@pytest.mark.parametrize("mark", ["…", ":", "."])
def test_long_punctuation_run_is_linear(mark):
    pattern, replace, _ = dictation_map.compile_map(
        {"Komma": ",", "Doppelpunkt": ":", "Hallo": "hi"})
    text = "a " + mark * 20000 + " b"
    t0 = time.perf_counter()
    assert pattern.sub(replace, text) == text
    assert time.perf_counter() - t0 < 0.2


def test_dictated_punctuation_still_replaces_whispers_marks():
    pattern, replace, _ = dictation_map.compile_map({"Komma": ",", "Punkt": "."})
    assert pattern.sub(replace, " HB 12... Komma 5") == " HB 12, 5"
    assert pattern.sub(replace, " Dosis 1, Komma, 5 mg") == " Dosis 1,, 5 mg"
    assert pattern.sub(replace, "12.5 und 12,5") == "12.5 und 12,5"
    assert pattern.sub(replace, "Ende…… Punkt") == "Ende."


@pytest.mark.parametrize("text, expected", [
    ("das z.B. hier", "das zB hier"),
    ("das z.B.", "das zB"),
    ("Z.B. so", "zB so"),
])
def test_key_ending_in_punctuation_matches(text, expected):
    """`\\b` after a key's trailing `.` needs a word character next, so a
    "z.B." key was saved and shown but never matched."""
    pattern, replace, _ = dictation_map.compile_map({"z.B.": "zB"})
    assert pattern.sub(replace, text) == expected


def test_word_bounded_key_still_needs_whole_word():
    pattern, replace, _ = dictation_map.compile_map({"Komma": ",", "Punkt": "."})
    assert pattern.sub(replace, "Kommando Punkt Punktzahl xKomma") == "Kommando . Punktzahl xKomma"
    assert pattern.sub(replace, "12... Komma 5") == "12, 5"


def test_key_does_not_match_between_combining_marks():
    """`\\w` has no combining marks, so the bounds saw a word edge at every
    vowel sign: a "लत" / "คร" key (both offered as autocomplete fragments)
    rewrote the inside of "बोलते" / "ครับ" to "बोXे" / "Yับ"."""
    pattern, replace, _ = dictation_map.compile_map(
        {"लत": "X", "คร": "Y", "बोलते": "Z"})
    assert pattern.sub(replace, "हम बोलते हैं ครับ") == "हम Z हैं ครับ"
    assert pattern.sub(replace, "लत คร") == "X Y"


def test_key_does_not_match_next_to_a_zero_width_joiner():
    """A ZWNJ / ZWJ is neither `\\w` nor a mark, so the bounds saw a word
    edge at it: a Persian "می" / "ها" key rewrote the verb prefix of
    می + ZWNJ + خواهم and the plural suffix of کتاب + ZWNJ + ها."""
    pattern, replace, _ = dictation_map.compile_map({"می": "X", "ها": "Y", "र्": "Z"})
    text = "من می\u200cخواهم کتاب\u200cها र्\u200dयाम"
    assert pattern.sub(replace, text) == text
    assert pattern.sub(replace, "می ها र्") == "X Y Z"


def test_large_non_ascii_map_stays_fast():
    """One mark class per key edge made compiling a 500-key Cyrillic map
    ~0.6 s and each sub of a 20k-char transcript ~1 s, 5-9x the plain `\\w`
    bounds; the class was walked for every key at every text position."""
    letters = "абвгдежзиклмнопрстуфхцчшщэюя"
    words = [letters[i % 28] + letters[i // 28 % 28] + letters[i // 784 % 28] + "ок"
             for i in range(1000)]
    keys, others = words[:500], words[500:]
    text = " ".join(f"{k} {o}," for k, o in zip(keys, others)) * 3
    t0 = time.perf_counter()
    pattern, replace, _ = dictation_map.compile_map({k: "x" for k in keys})
    out = pattern.sub(replace, text)
    assert time.perf_counter() - t0 < 0.6
    assert out == " ".join(f"x {o}," for o in others) * 3


def _factory_de_map():
    return next(r["map"] for r in _factory_rules() if r["name"] == "de-dictation-map")


@pytest.mark.parametrize("text, expected", [
    ("Eisen Bindestrich Infusion", "Eisen-Infusion"),
    ("Eisen Binde Strich Infusion", "Eisen-Infusion"),
    ("Das sind 50 Prozent mehr.", "Das sind 50% mehr."),
    ("5 Prozentzeichen", "5 %"),
    ("das Prozentual", "das Prozentual"),
])
def test_factory_keys_with_a_space_edge_match_in_running_text(text, expected):
    """The factory " Prozent" / " Bindestrich " keys sit right against the
    neighbouring word; a blanket `(?<!\\w)` / `(?!\\w)` around every key
    refused them, so every German transcript lost the dictated % and -."""
    pattern, replace, _ = dictation_map.compile_map(_factory_de_map())
    assert pattern.sub(replace, text) == expected


@pytest.mark.parametrize("key, text, expected", [
    ("-", "E-Mail und 5 - 6", "E-Mail und 5 – 6"),
    ("...", "Ende... und x...y", "Ende... und x...y"),
    ("...", "Ende ... und", "Ende – und"),
])
def test_punctuation_edge_keeps_its_word_bound(key, text, expected):
    """Only a space edge may sit against the neighbouring word: a "-" key
    must not rewrite the hyphen inside a compound."""
    pattern, replace, _ = dictation_map.compile_map({key: "–"})
    assert pattern.sub(replace, text) == expected


@pytest.mark.parametrize("key, value, text, expected", [
    (" Komma", ",", "12... Komma 5", "12, 5"),
    (" Punkt", ".", "Hallo, Punkt", "Hallo."),
    (" Punkt", ".", "Hallo. Punkt", "Hallo."),
])
def test_dictated_punctuation_wins_for_a_space_led_key(key, value, text, expected):
    """The possessive spaces after Whisper's marks ate the key's own leading
    space, so the prefix never applied and Whisper's mark stayed in front."""
    pattern, replace, _ = dictation_map.compile_map({key: value})
    assert pattern.sub(replace, text) == expected


@pytest.mark.parametrize("key", [" ", "  "])
def test_whitespace_only_key_is_ignored(key):
    """A stored config from before the schema refused it must not turn every
    space of every transcript into the key's value."""
    pattern, replace, lookup = dictation_map.compile_map({key: "x", "Komma": ","})
    assert pattern.sub(replace, "hallo Welt  Komma zwei") == "hallo Welt  , zwei"
    assert key not in lookup


@pytest.mark.parametrize("text, expected", [
    ("İstanbul ve", "IST ve"),
    ("izmir yolu", "IZM yolu"),
    ("ς x", "S x"),
])
def test_key_matched_by_sre_case_folding_is_applied(text, expected):
    """The pattern matches with sre's case folding, the value was looked up by
    `str.lower()`; the two disagree on "İ" (lowers to "i" + U+0307) and "ς",
    so the key matched and the text was silently left unchanged."""
    pattern, replace, _ = dictation_map.compile_map({"istanbul": "IST", "İzmir": "IZM", "σ": "S"})
    assert pattern.sub(replace, text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Treffen am 3. Schrägstrich 4. Mai in Bern.", "Treffen am 3. / 4. Mai in Bern"),
    ("Im 1. Schrägstrich 2. Quartal.", "Im 1. / 2. Quartal"),
    ("Siehe http: Schrägstrich Schrägstrich www.", "Siehe http: / / www"),
])
def test_dictated_slash_keeps_the_mark_in_front_of_it(app_module, text, expected):
    """A dictated Schrägstrich must not eat the mark in front of it: an
    ordinal pair "am 3. Schrägstrich 4. Mai" read as the fraction "3/4. Mai"
    once it did, and "http:" lost its colon. The pause period of
    "120. Schrägstrich 80" looks the same at the map, so it stays."""
    app_module.cfg.PIPELINE_RULES = copy.deepcopy(_factory_rules())
    pl_engine.rebuild_caches()
    assert pl_engine._postprocess_text(text, model_name="", language="de") == expected


def test_ascii_lead_bound_is_tested_before_the_keys():
    """Placed after the key group as a lookbehind over the key, the ASCII
    lead bound let sre walk every branch at every text position, mid-word
    included: a 2000-key map took ~4x as long per letter. Only the ~800-range
    mark class of a non-ASCII edge keeps the trailing placement."""
    pattern, _, _ = dictation_map.compile_map({"Komma": ",", "über": "x"})
    assert r"(?<!\w)(?:Komma)(?!\w)" in pattern.pattern
    assert r"(?:Komma)(?<!" not in pattern.pattern
    assert "(?:über)(?<![" in pattern.pattern


@pytest.mark.parametrize("text, expected", [
    ("Komma am Anfang", ", am Anfang"),
    ("xKomma 3Komma _Komma Komma", "xKomma 3Komma _Komma ,"),
    ("12... Komma 5", "12, 5"),
    ("e-Mail x.com und .com", "e-Mail x.com und DOT"),
    ("3D und a3D, 3d", "DREI und a3D, DREI"),
    ("Komma", ","),
    ("İstanbul xİstanbul", "IST xİstanbul"),
    ("die grösser als Grenze, xgrösser als", "die > Grenze, xgrösser als"),
    ("über, xüber, ́über", "U, xüber, ́über"),
])
def test_lead_word_bound_on_tricky_keys(text, expected):
    pattern, replace, _ = dictation_map.compile_map({
        "Komma": ",", ".com": "DOT", "3D": "DREI", "istanbul": "IST",
        "größer als": ">", "über": "U"})
    assert pattern.sub(replace, text) == expected


def _in_re_cache(source):
    # Reading re's private caches is fine in a test: the leak lives there.
    return any(key[1] == source for key in [*re._cache, *re._cache2])


def test_an_edited_map_leaves_no_old_pattern_in_res_cache(monkeypatch):
    """re.compile caches every pattern module-wide (up to 512), so each
    dictionary edit pinned the replaced map's whole alternation there:
    ~1 MB per edit at 2000 keys, never freed while the server ran."""
    sources = []
    try:
        for i in range(3):
            monkeypatch.setattr(cfg, "PIPELINE_RULES", [{
                "name": "m", "label": "m", "type": "callback:map", "enabled": True,
                "map": {"Komma": ",", f"wort{i}": "x"}}])
            pl_engine.rebuild_caches()
            sources.append(pl_engine._COMPILED_RULES[0].pattern.pattern)
        # The /settings/pipeline dry run compiles its edited map the same way.
        sources.append(dictation_map.compile_map({"Komma": ",", "probe": "x"})[0].pattern)
        assert len(set(sources)) == 4
        assert not any(_in_re_cache(s) for s in sources)
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()


def test_an_unchanged_map_is_not_recompiled():
    m = {"Komma": ",", "Punkt": "."}
    assert dictation_map.compile_map(dict(m))[0] is dictation_map.compile_map(dict(m))[0]


def _map_rules(maps):
    return [{"name": f"m{i}", "label": f"m{i}", "type": "callback:map",
             "enabled": True, "map": m} for i, m in enumerate(maps)]


def test_six_unchanged_maps_are_not_recompiled_by_each_rebuild(monkeypatch):
    """A 4-slot LRU cycled by 5+ maps (one per language) missed every time, so
    every rules save and every dry run recompiled every map cold."""
    maps = [{"Komma": ",", f"sechskarte{i}": "x"} for i in range(6)]
    real_compile = re.compile
    compiled = []

    def spy(source, flags=0):
        if "sechskarte" in source:
            compiled.append(source)
        return real_compile(source, flags)

    monkeypatch.setattr(re, "compile", spy)
    monkeypatch.setattr(cfg, "PIPELINE_RULES", _map_rules(maps))
    try:
        for _ in range(3):
            pl_engine.rebuild_caches()
        # The /settings/pipeline dry run compiles every stored map again.
        for m in maps:
            dictation_map.compile_map(dict(m))
        assert len(compiled) == 6
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()


def test_a_dropped_map_pattern_is_released(monkeypatch):
    import gc
    import weakref
    kept, dropped = {"Komma": ",", "bleibkarte": "x"}, {"Komma": ",", "wegkarte": "x"}
    try:
        monkeypatch.setattr(cfg, "PIPELINE_RULES", _map_rules([kept, dropped]))
        pl_engine.rebuild_caches()
        ref = weakref.ref(pl_engine._COMPILED_RULES[1].pattern)
        source = ref().pattern
        monkeypatch.setattr(cfg, "PIPELINE_RULES", _map_rules([kept]))
        pl_engine.rebuild_caches()
        gc.collect()
        assert ref() is None
        assert not _in_re_cache(source)
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()


def test_fold_fallback_cost_is_bounded_by_the_alphabet_not_the_key_count(monkeypatch):
    # A sre-fold miss ("sein" matched by the key "ſein") used to compile one
    # full-key pattern per same-length key on every occurrence: a page-scoped
    # non-admin's 10 000-key map made each such word cost ~140 ms on the loop.
    from faster_whisper_backend.pipeline import dictation_map as dm
    m = {f"{c}{i:03d}": "x" for c in "abcdefghij" for i in range(999)}
    m["ſein"] = "SEIN"
    pattern, replace, _lookup = dm.compile_map(m)
    if hasattr(dm, "_sre_char_eq"):
        dm._sre_char_eq.cache_clear()
    calls = []
    real = dm.re.fullmatch
    monkeypatch.setattr(dm.re, "fullmatch", lambda *a, **k: calls.append(1) or real(*a, **k))
    out = pattern.sub(replace, "Das ist sein Haus. " * 20)
    assert out.count("SEIN") == 20
    assert len(calls) <= 100
