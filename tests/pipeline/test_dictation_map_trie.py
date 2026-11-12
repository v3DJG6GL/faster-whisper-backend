"""dictation_map: each key group is a prefix tree, and it matches exactly what
the flat alternation it replaced matched.

The flat ``(?:k1|...|kN)`` was walked branch by branch at every text
position, so a 10 000-key map (the schema cap) cost ~0.5-1 s per 20 KB
transcript. The oracle below is that old builder, kept only here.

The tree has to stay cheap for every key set a map can hold, not only for
words: one branch per sre case class, a wide node behind lookahead gates,
and no deeper group nesting than the keys are long.
"""

import itertools
import math
import random
import re
from re import _compiler, _parser

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


def _random_map(rnd, alphabet=_KEY_CHARS, most=40):
    m = {}
    for _ in range(rnd.randint(1, most)):
        key = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 5)))
        if rnd.random() < 0.25:
            key += " " + "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 3)))
        if rnd.random() < 0.15:
            key = " " + key
        m[key] = rnd.choice(_VALUES)
    return m


def _random_text(rnd, keys, alphabet=_KEY_CHARS):
    out = []
    for _ in range(rnd.randint(1, 25)):
        out.append(rnd.choice(_TEXT_NOISE))
        word = rnd.choice(keys) if rnd.random() < 0.7 else rnd.choice(alphabet)
        out.append(rnd.choice([word, word.upper(), word.lower(), word.swapcase()]))
    return "".join(out)


def _assert_same_as_flat(monkeypatch, rnd, m, texts, alphabet=_KEY_CHARS):
    new_pattern, new_replace, new_lookup = dictation_map.compile_map(m)
    with monkeypatch.context() as mp:
        mp.setattr(dictation_map, "_alternation", _flat_alternation)
        mp.setattr(dictation_map, "_lead_gate", lambda keys: "")
        old_pattern, old_replace, old_lookup = dictation_map.compile_map(m)
    assert new_lookup == old_lookup
    keys = list(m) + [dictation_map._ss_variant(k) for k in m]
    for _ in range(texts):
        text = _random_text(rnd, keys, alphabet)
        assert new_pattern.sub(new_replace, text) == old_pattern.sub(old_replace, text), (m, text)
        assert ([(mt.span(), mt.span(1)) for mt in new_pattern.finditer(text)]
                == [(mt.span(), mt.span(1)) for mt in old_pattern.finditer(text)]), (m, text)


def test_prefix_tree_matches_the_flat_alternation(monkeypatch):
    rnd = random.Random(20261101)
    for _ in range(80):
        _assert_same_as_flat(monkeypatch, rnd, _random_map(rnd), texts=30)


# Enough distinct case classes that a node outgrows _WIDE_NODE, with chars a
# gate's "[...]" has to escape.
_WIDE_CHARS = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
               "äöüÄÖÜßẞſKİıσΣςµμéÉçñøØåабвгдежзАБВГДЕЖЗαβγΑΒΓ中文字लत-.,!?]^\\[&|~")


def test_gated_wide_nodes_match_the_flat_alternation(monkeypatch):
    # The ~800-range mark class of a non-ASCII edge is most of a compile and
    # no part of the tree: a plain \w bound (in the oracle too) buys maps.
    monkeypatch.setitem(dictation_map._WORD_CLASS, "other", r"\w")
    rnd = random.Random(20261112)
    for _ in range(30):
        m = _random_map(rnd, _WIDE_CHARS, most=150)
        _assert_same_as_flat(monkeypatch, rnd, m, texts=15, alphabet=_WIDE_CHARS)
    # Every node of three branches gated: chunks at each depth of a
    # small-alphabet tree, the sre twins included.
    monkeypatch.setattr(dictation_map, "_WIDE_NODE", 2)
    for _ in range(60):
        _assert_same_as_flat(monkeypatch, rnd, _random_map(rnd), texts=15)


def test_keys_sharing_a_prefix_become_one_branch():
    pattern, replace, _ = dictation_map.compile_map({"Komma": ",", "Kolon": ":", "komma": ","})
    assert r"(?<!\w)(?:Ko(?:mma|lon))(?!\w)" in pattern.pattern
    assert pattern.sub(replace, "KOMMA kolon Komma") == ", : ,"


def test_lowercase_lookup_names_the_key_sre_matched():
    """The replacer looks a match up by str.lower() first; that names the
    key sre matched only while sre's simple lowercase agrees with
    str.lower() on every char that lowers to one char (true for Python's
    Unicode tables today)."""
    for cp in range(0x110000):
        fold = chr(cp).lower()
        if len(fold) == 1 and fold != chr(cp):
            assert _sre.unicode_iscased(cp) and _sre.unicode_tolower(cp) == ord(fold), hex(cp)


def _sre_case_classes():
    """Every sre IGNORECASE class with more than one member, by _sre_fold."""
    cased = set()
    for cp in range(0x110000):
        lo = _sre.unicode_tolower(cp)
        if lo != cp:
            cased.update((cp, lo))
    for lo, others in _compiler._EXTRA_CASES.items():
        cased.update((lo, *others))
    classes = {}
    for cp in sorted(cased):
        classes.setdefault(dictation_map._sre_fold(chr(cp)), []).append(chr(cp))
    return [members for members in classes.values() if len(members) > 1]


def test_one_member_stands_for_its_whole_case_class():
    """_prefix_tree emits one member's literal for a whole _sre_fold bucket,
    and a gate lists one member per bucket in a "[...]": both have to match
    every member of the class, and the literal no char of another class
    (two buckets that match one char are both walked)."""
    classes = _sre_case_classes()
    assert len(classes) > 1000
    assert {"s", "ſ"} <= set(next(c for c in classes if "s" in c))
    assert {"i", "I", "İ", "ı"} <= set(next(c for c in classes if "i" in c))
    every_cased_char = "".join(sorted(c for members in classes for c in members))
    for members in classes:
        for a in members:
            both = re.compile(re.escape(a) + "[" + re.escape(a) + "]", re.IGNORECASE)
            for b in members:
                assert both.fullmatch(b + b), (a, b)
        literal = re.compile(re.escape(members[0]), re.IGNORECASE)
        assert literal.findall(every_cased_char) == members


def test_sre_twins_share_one_branch():
    """Bucketed by str.lower(), "s" / "ſ" and "i" / "ı" (one class each to
    sre) were separate branches that each matched the other's text: a map of
    such spellings ("si si ..." in 2^n variants) walked every one of them at
    each "si", ~0.65 ms per char of that text."""
    assert dictation_map._prefix_tree(["s", "ſ"]) == "s"
    assert dictation_map._prefix_tree(["İx", "ix", "ıx", "Ix"]) == "İx"
    assert "|" not in dictation_map._prefix_tree(["si", "ſı", "sı", "ſi"])
    spellings = ["".join(p) for p in itertools.product(*["sſ", "iı"] * 4)]
    pattern, replace, _ = dictation_map.compile_map(
        {f"a {s} b": "A" if n % 2 else "B" for n, s in enumerate(spellings)})
    # 256 keys, no alternation left to walk.
    assert _widest_and_deepest(pattern.pattern) == (0, 0)
    assert (pattern.sub(replace, f"a {spellings[0]} b, A {spellings[1]} B, a SISISISI b, a sisi b")
            == "B, A, B, a sisi b")


def _widest_and_deepest(source):
    """(most alternatives of any alternation sre walks, deepest alternation
    nesting) of a pattern, read off re's own parse tree."""
    def walk(items, depth):
        widest = deepest = 0
        for op, av in items:
            if op is _parser.BRANCH:
                subs, widest, deepest = av[1], max(widest, len(av[1])), max(deepest, depth + 1)
                depth_in = depth + 1
            elif op is _parser.SUBPATTERN:
                subs, depth_in = [av[3]], depth
            elif op in (_parser.ASSERT, _parser.ASSERT_NOT):
                subs, depth_in = [av[1]], depth
            elif op in (_parser.MAX_REPEAT, _parser.MIN_REPEAT, _parser.POSSESSIVE_REPEAT):
                subs, depth_in = [av[2]], depth
            elif op is _parser.ATOMIC_GROUP:
                subs, depth_in = [av], depth
            else:
                continue
            for sub in subs:
                w, d = walk(sub, depth_in)
                widest, deepest = max(widest, w), max(deepest, d)
        return widest, deepest
    return walk(_parser.parse(source, re.IGNORECASE), 0)


_LEADS = ([chr(0x4E00 + i) for i in range(140)]
          + [chr(cp) for cp in range(0x430, 0x450)] + [chr(cp) for cp in range(0x3B1, 0x3C1)]
          + list("äöüßéçñøå"))


def test_wide_node_is_not_walked_linearly():
    """sre walks an alternation branch by branch, so a node with one key per
    distinct char cost one try per key at every text position that reached
    it: 158 lead chars x 63 key lengths (a map's 10 000 keys) took ~1.4 s per
    20 KB of any transcript. A wide node is cut into gated chunks, wherever
    it sits in the tree."""
    assert len(_LEADS) > 190
    bound = 2 * (math.isqrt(len(_LEADS)) + 1)
    for keys in ([c + "b" for c in _LEADS],
                 ["a" + c + "b" for c in _LEADS],
                 [" x" + c + "b" for c in _LEADS]):
        pattern, replace, _ = dictation_map.compile_map(dict.fromkeys(keys, "X"))
        widest, _deepest = _widest_and_deepest(pattern.pattern)
        assert widest <= bound, (widest, keys[0])
        text = " ".join(keys[::7]) + " " + keys[-1].upper() + " q"
        assert pattern.sub(replace, text).split() == ["X"] * (len(keys[::7]) + 1) + ["q"]


def test_wide_nodes_do_not_deepen_the_pattern():
    """re's parser and compiler recurse per nested group, and a map whose
    compile hits the recursion limit is skipped by the engine. A wide node
    can sit at every char of a key, so its gates must not add a group at
    each of them: the branch with the most keys below it stays ungated."""
    levels, width = 14, 12
    chars = [chr(0x4E00 + i) for i in range(width)]
    # Deepest first, so the spine ("z") is the first branch of every node.
    keys = ["z" * d + c + "y" * (levels - d - 1)
            for d in range(levels - 1, -1, -1) for c in chars]
    pattern, replace, _ = dictation_map.compile_map(dict.fromkeys(keys, "X"))
    widest, deepest = _widest_and_deepest(pattern.pattern)
    assert widest <= dictation_map._WIDE_NODE + width // dictation_map._WIDE_NODE + 1
    assert deepest <= levels + 2
    assert pattern.sub(replace, f"{keys[0]} {keys[-1]} {keys[40]}z") == f"X X {keys[40]}z"

