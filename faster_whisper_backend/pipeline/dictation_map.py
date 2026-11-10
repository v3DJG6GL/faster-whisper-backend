"""Compile a ``callback:map`` rule (spoken dictation word → symbol).

One implementation for every place that runs a map: the engine
(``engine.rebuild_caches``), the /settings/pipeline test panel's dry run
(``admin/routes.py``) and the factory-rule tests — so the preview can never
disagree with what a transcription actually gets.

Beyond the plain "whole word, any case, longest key first" lookup the map
always had, two things are decided here:

* **ß and ss are one spelling.** The Swiss cleanup (``ch-symbol-cleanup``)
  turns ß into ss BEFORE the map runs, so a key written with ß ("größer als")
  never saw its own words again. Every ß key gets a derived ss variant
  ("grösser als") with the same value; a key the admin wrote with ss
  explicitly wins over a derived one. This also covers keys added through
  /quick-config, which nobody should have to spell twice.
* **Dictated punctuation wins.** Whisper writes its own punctuation as it
  hears the pauses, and the strip rule keeps a mark right after a digit (so
  3.5 survives). "12... Komma 5" therefore reached the map as
  "12. Komma 5", became "12. , 5" and the dedup rule kept the period — the
  dictated comma lost. A run of ``. , : ; ! ? …`` (plus spaces) directly
  before a word whose value is punctuation is now consumed by the match and
  replaced together with it, so once the later spacing / dedup rules have
  run, "12... Komma 5" ends up "12,5" and "1, Komma, 5" "1,5". A mark
  between two digits ("12.5", "12,5") is never followed by a
  dictation word and is untouched.

Pure module: no import of ``main``.
"""
from __future__ import annotations

import collections
import functools
import itertools
import re
import threading
from typing import Callable, Iterable

from faster_whisper_backend.core.unicode_marks import JOINERS as _JOINERS
from faster_whisper_backend.core.unicode_marks import MARKS as _MARKS

# Values that count as punctuation for the "dictated punctuation wins" prefix.
_PUNCT_VALUE_CHARS = frozenset(".,:;!?")
# What Whisper puts before a dictated punctuation word: its own marks
# (the ellipsis as three periods or as one character); compile_map adds the
# spaces after them. Anchored at the start of a run and possessive: a greedy
# run would be retried from every position inside a long hallucinated
# "……" / "::::" run, each try backtracking into the failed lookahead — O(n²).
_WHISPER_PUNCT = r"(?<![.,:;!?…])[.,:;!?…]++"


def _is_punct_value(value: str) -> bool:
    v = (value or "").strip()
    return bool(v) and all(ch in _PUNCT_VALUE_CHARS for ch in v)


def _ss_variant(key: str) -> str:
    return key.replace("ß", "ss").replace("ẞ", "SS")


# Word edges, marks and joiners included. Only a non-ASCII edge gets the
# mark class: it is ~800 ranges, an ASCII letter next to a mark occurs only
# in NFD text, which whisper does not emit, and a joiner joins only
# non-Latin scripts.
_WORD_CLASS = {"ascii": r"\w", "other": rf"[\w{_MARKS}{_JOINERS}]"}


def _edge(ch: str) -> str:
    """The word class bounding this edge; "" for a whitespace edge."""
    if not ch or ch.isspace():
        return ""
    return "ascii" if ch.isascii() else "other"


def _shape(key: str) -> tuple[int, str, str]:
    return len(key), _edge(key[:1]), _edge(key[-1:])


def _prefix_tree(keys: list[str]) -> str:
    """Keys of one length as a prefix tree ("K(?:omma|olon)").

    sre has no trie: a flat ``(?:k1|...|kN)`` is walked branch by branch at
    every text position, so a 10 000-key map cost up to ~1 s per 20 KB
    transcript. Here each position tries one branch per distinct next char.
    Chars are bucketed by ``str.lower()`` when that is one char: sre's
    IGNORECASE compiles a cased literal from its simple lowercase alone, and
    the two agree on every such char (a test pins that), so any member
    stands for the whole bucket.
    "İ" (lowers to two chars) keeps its own bucket; "ſ" and "s", which
    sre folds together but ``lower()`` keeps apart, stay two buckets that
    are each tried — same span either way."""
    if not keys[0]:
        return ""
    buckets: dict[str, list[str]] = {}
    for k in keys:
        fold = k[0].lower()
        buckets.setdefault(fold if len(fold) == 1 else k[0], []).append(k)
    branches = [re.escape(run[0][0]) + _prefix_tree([k[1:] for k in run])
                for run in buckets.values()]
    return branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"


def _alternation(keys: list[str]) -> str:
    """``keys`` as one alternation, longest first, each word-bounded on
    every edge that is not whitespace.

    Not ``\\b``, and not one lookaround around the whole alternation: after a
    key's own trailing ``.`` ("z.B.") ``\\b`` needs a word character next, and
    a key that starts or ends with a space (" Prozent", " Bindestrich ") sits
    right against the neighbouring word, which a blanket ``(?<!\\w)`` /
    ``(?!\\w)`` refuses. A punctuation edge keeps the bound: a "-" key must
    not rewrite the hyphen inside "E-Mail". A combining mark counts as a
    word char (``\\w`` has none): a "लत" key must not rewrite the inside of
    "बोलते", nor "คร" that of "ครับ". So does a ZWNJ / ZWJ: a Persian "می"
    key must not rewrite the prefix of می + ZWNJ + خواهم.

    Keys of one length and edge kinds share one group. One mark class per
    key made each compile ~70x slower, and in front of the key it was walked
    for every key at every text position, so the sub of a 500-key Cyrillic
    or Hindi map ran 5-9x slower: a non-ASCII lead bound sits after the
    group, as a lookbehind over the key itself (sre folds case char by char,
    so the match is exactly n chars). An ASCII lead bound is one ``\\w`` test
    and stays in front: after the group, every branch was walked at every
    text position, mid-word included, before it could refuse — a 2000-key
    map ran ~4x slower. Reordering keys of one length is safe: at a given
    position they match the same span, so the same replacement — which is
    also why a group may be a prefix tree (``_prefix_tree``) instead of a
    flat list."""
    parts = []
    for (n, lead, tail), run in itertools.groupby(
            sorted(keys, key=_shape, reverse=True), key=_shape):
        out = "(?:" + _prefix_tree(list(run)) + ")"
        if lead == "ascii":
            out = r"(?<!\w)" + out
        elif lead:
            out += rf"(?<!{_WORD_CLASS[lead]}(?s:.){{{n}}})"
        if tail:
            out += rf"(?!{_WORD_CLASS[tail]})"
        parts.append(out)
    return "|".join(parts)


# The live rule set's map patterns (replaced wholesale by retain_live_maps at
# each engine rebuild) plus a few recent other ones (a dry run of an edited
# map), so N unchanged maps never recompile and a dropped map is released.
_LIVE: dict[str, "re.Pattern[str]"] = {}
_RECENT: "collections.OrderedDict[str, re.Pattern[str]]" = collections.OrderedDict()
_RECENT_MAX = 4
_CACHE_LOCK = threading.Lock()


def _compile_map_pattern(source: str) -> "re.Pattern[str]":
    with _CACHE_LOCK:
        pattern = _LIVE.get(source) or _RECENT.get(source)
        if pattern is not None:
            return pattern
    # re.compile keeps every pattern in re's module cache (hundreds of
    # entries), and a map's alternation is one big pattern (~1 MB at 2000
    # keys): each dictionary edit or edited-map dry run left the old one
    # pinned there long after the rule dropped it. Purge it right away.
    pattern = re.compile(source, re.IGNORECASE)
    re.purge()
    with _CACHE_LOCK:
        _RECENT[source] = pattern
        while len(_RECENT) > _RECENT_MAX:
            _RECENT.popitem(last=False)
    return pattern


def retain_live_maps(patterns: Iterable["re.Pattern[str]"]) -> None:
    """Keep exactly these map patterns (the rule set the engine just compiled)
    for reuse by the next rebuild or dry run; any other live one is dropped."""
    global _LIVE
    live = {p.pattern: p for p in patterns}
    with _CACHE_LOCK:
        _LIVE = live
        for source in live:
            _RECENT.pop(source, None)


@functools.lru_cache(maxsize=4096)
def _sre_char_eq(a: str, b: str) -> bool:
    """Whether sre's IGNORECASE matches literal char `a` against char `b`."""
    return a == b or re.fullmatch(re.escape(a), b, re.IGNORECASE) is not None


# Matched strings whose sre-fold lookup the replacer remembers (per map).
_FOLD_MEMO_MAX = 1024
# fold_memo.get default: a memoised None means "no key", not "not memoised".
_MISS = object()


def compile_map(m: dict) -> tuple["re.Pattern[str]", Callable[["re.Match[str]"], str], dict[str, str]]:
    """Compile a map into ``(pattern, replacer, lookup)``.

    ``lookup`` is keyed by the lower-cased key and holds the derived ss
    variants too, so a caller can reason about the effective key set (the
    streaming hold-back does). ``pattern.sub(replacer, text)`` applies the
    map: case-insensitively, longest key first, each key word-bounded on
    every edge that is not whitespace (``_alternation``). The optional
    punctuation prefix only ever precedes a key whose value is punctuation."""
    # A whitespace-only key has no word edge to bound it, so it would match
    # every space in every transcript; the schema refuses it and the stored-
    # config migration (config_store) drops it — this is the backstop.
    m = {k: v for k, v in m.items() if k.strip()}
    lookup: dict[str, str] = {k.lower(): v for k, v in m.items()}
    keys: list[str] = list(m)
    for k in m:
        derived = _ss_variant(k)
        if derived != k and derived.lower() not in lookup:
            lookup[derived.lower()] = m[k]
            keys.append(derived)
    alternation = _alternation(keys)
    punct_keys = [k for k in keys if _is_punct_value(lookup.get(k.lower(), ""))]
    if punct_keys:
        punct_alt = _alternation(punct_keys)
        # The spaces are possessive too, so a key that itself starts with a
        # space (" Komma") needs the second branch: the marks alone.
        ahead = r"(?=(?:" + punct_alt + r"))"
        prefix = r"(?:" + _WHISPER_PUNCT + r"(?:[ \t]*+" + ahead + r"|" + ahead + r"))?"
    else:
        prefix = ""
    pattern = _compile_map_pattern(prefix + r"(" + alternation + r")")
    keys_by_len: dict[int, list[str]] = {}
    for k in keys:
        keys_by_len.setdefault(len(k), []).append(k)
    fold_memo: dict[str, "str | None"] = {}

    def replace(mt: "re.Match[str]") -> str:
        key = mt.group(1)
        value = lookup.get(key.lower())
        if value is None:
            # sre's case folding is not str.lower(): "İ" matches "i" but
            # lowers to "i" + U+0307, "ς" matches "σ" but lowers to itself.
            # Ask sre which key it matched; it folds char by char, so only
            # keys of the matched length can be it, compared char by char
            # (cached per char pair: a 10 000-key map must not compile one
            # full-key pattern per candidate on every miss). First match in
            # key order wins, as before; the result is memoised per string.
            # One lookup, not a check then a read: other threads share this
            # memo and may clear it in between.
            value = fold_memo.get(key, _MISS)
            if value is _MISS:
                value = next((lookup[k.lower()] for k in keys_by_len.get(len(key), ())
                              if all(_sre_char_eq(a, b) for a, b in zip(k, key))), None)
                if len(fold_memo) >= _FOLD_MEMO_MAX:
                    fold_memo.clear()
                fold_memo[key] = value
        if value is None:
            return mt.group(0)
        lead = mt.start(1) - mt.start(0)
        if lead and not _is_punct_value(value):
            # The lookahead saw a punctuation key, but the longest-first
            # alternation matched a longer key that merely starts with it
            # ("Komma xyz" → "x"): keep Whisper's mark, it was not replaced.
            return mt.group(0)[:lead] + value
        return value

    return pattern, replace, lookup
