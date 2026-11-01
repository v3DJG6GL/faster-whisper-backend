"""dictation_map: each key group is a prefix tree, and it matches exactly what
the flat alternation it replaced matched.

The flat ``(?:k1|...|kN)`` was walked branch by branch at every text
position, so a 10 000-key map (the schema cap) cost ~0.5-1 s per 20 KB
transcript. The oracle below is that old builder, kept only here.
"""

import itertools
import random
import re

import _sre

from faster_whisper_backend.pipeline import dictation_map


def _flat_alternation(keys):
    parts = []
    for (n, lead, tail), run in itertools.groupby(
            sorted(keys, key=dictation_map._shape, reverse=True), key=dictation_map._shape):
        out = "(?:" + "|".join(re.escape(k) for k in run) + ")"
        if lead == "ascii":
            out = r"(?<!\w)" + out
        elif lead:
            out += rf"(?<!{dictation_map._WORD_CLASS[lead]}(?s:.){{{n}}})"
        if tail:
            out += rf"(?!{dictation_map._WORD_CLASS[tail]})"
        parts.append(out)
    return "|".join(parts)


# A small alphabet so keys share prefixes, with the chars whose case folding
# sre and str.lower() disagree on: Kelvin sign, long s, İ / ı, ß / ẞ, σ / ς.
_KEY_CHARS = "kKoOmMaAsSäÄüÜßẞſKİıiIσΣς-.,!?1"
_TEXT_NOISE = ["", " ", "  ", ".", "...", "…", ", ", ": ", "x", "-", "́", "1"]
_VALUES = [",", ".", ":", "?!", "X", "Wort", ""]


def _random_map(rnd):
    m = {}
    for _ in range(rnd.randint(1, 40)):
        key = "".join(rnd.choice(_KEY_CHARS) for _ in range(rnd.randint(1, 5)))
        if rnd.random() < 0.25:
            key += " " + "".join(rnd.choice(_KEY_CHARS) for _ in range(rnd.randint(1, 3)))
        if rnd.random() < 0.15:
            key = " " + key
        m[key] = rnd.choice(_VALUES)
    return m


def _random_text(rnd, keys):
    out = []
    for _ in range(rnd.randint(1, 25)):
        out.append(rnd.choice(_TEXT_NOISE))
        word = rnd.choice(keys) if rnd.random() < 0.7 else rnd.choice(_KEY_CHARS)
        out.append(rnd.choice([word, word.upper(), word.lower(), word.swapcase()]))
    return "".join(out)


def test_prefix_tree_matches_the_flat_alternation(monkeypatch):
    rnd = random.Random(20261101)
    for _ in range(80):
        m = _random_map(rnd)
        new_pattern, new_replace, new_lookup = dictation_map.compile_map(m)
        with monkeypatch.context() as mp:
            mp.setattr(dictation_map, "_alternation", _flat_alternation)
            old_pattern, old_replace, old_lookup = dictation_map.compile_map(m)
        assert new_lookup == old_lookup
        keys = list(m) + [dictation_map._ss_variant(k) for k in m]
        for _ in range(30):
            text = _random_text(rnd, keys)
            assert new_pattern.sub(new_replace, text) == old_pattern.sub(old_replace, text), (m, text)


def test_keys_sharing_a_prefix_become_one_branch():
    pattern, replace, _ = dictation_map.compile_map({"Komma": ",", "Kolon": ":", "komma": ","})
    assert r"(?<!\w)(?:Ko(?:mma|lon))(?!\w)" in pattern.pattern
    assert pattern.sub(replace, "KOMMA kolon Komma") == ", : ,"


def test_lowercase_bucket_compiles_like_each_of_its_members():
    """_prefix_tree emits one member for a whole str.lower() bucket; that is
    only sound while sre's simple lowercase agrees with str.lower() on every
    char that lowers to one char (true for Python's Unicode tables today)."""
    for cp in range(0x110000):
        fold = chr(cp).lower()
        if len(fold) == 1 and fold != chr(cp):
            assert _sre.unicode_iscased(cp) and _sre.unicode_tolower(cp) == ord(fold), hex(cp)

