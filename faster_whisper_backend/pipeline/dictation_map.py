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
  3.5 survives). " HB 12... Komma 5" therefore reached the map as
  "12. Komma 5", became "12. , 5" and the dedup rule kept the period — the
  dictated comma lost. A run of ``. , : ; ! ? …`` (plus spaces) directly
  before a word whose value is punctuation is now consumed by the match and
  replaced together with it: "12... Komma 5" → "12,5", "1, Komma, 5" →
  "1,5". A mark between two digits ("12.5", "12,5") is never followed by a
  dictation word and is untouched.

Pure module: no import of ``main``.
"""
from __future__ import annotations

import re
from typing import Callable

# Values that count as punctuation for the "dictated punctuation wins" prefix.
_PUNCT_VALUE_CHARS = frozenset(".,:;!?")
# What Whisper puts before a dictated punctuation word: its own marks
# (the ellipsis as three periods or as one character), then spaces. Starts
# only at the beginning of a run and is possessive: a plain greedy run was
# retried from every position inside a long hallucinated "……" / "::::" run,
# each attempt backtracking char by char into the failed lookahead — O(n²).
_WHISPER_PUNCT = r"(?<![.,:;!?…])[.,:;!?…]++[ \t]*+"


def _is_punct_value(value: str) -> bool:
    v = (value or "").strip()
    return bool(v) and all(ch in _PUNCT_VALUE_CHARS for ch in v)


def _ss_variant(key: str) -> str:
    return key.replace("ß", "ss").replace("ẞ", "SS")


def compile_map(m: dict) -> tuple["re.Pattern[str]", Callable[["re.Match[str]"], str], dict[str, str]]:
    """Compile a map into ``(pattern, replacer, lookup)``.

    ``lookup`` is keyed by the lower-cased key and holds the derived ss
    variants too, so a caller can reason about the effective key set (the
    streaming hold-back does). ``pattern.sub(replacer, text)`` applies the
    map. Keys are matched word-bounded, case-insensitively, longest first —
    the historic behaviour; the optional punctuation prefix only ever
    precedes a key whose value is punctuation."""
    lookup: dict[str, str] = {k.lower(): v for k, v in m.items()}
    keys: list[str] = list(m)
    for k in m:
        derived = _ss_variant(k)
        if derived != k and derived.lower() not in lookup:
            lookup[derived.lower()] = m[k]
            keys.append(derived)
    alternation = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    punct_keys = [k for k in keys if _is_punct_value(lookup.get(k.lower(), ""))]
    if punct_keys:
        punct_alt = "|".join(re.escape(k) for k in sorted(punct_keys, key=len, reverse=True))
        prefix = r"(?:" + _WHISPER_PUNCT + r"(?=(?:" + punct_alt + r")\b))?"
    else:
        prefix = ""
    pattern = re.compile(prefix + r"\b(" + alternation + r")\b", re.IGNORECASE)

    def replace(mt: "re.Match[str]") -> str:
        key = mt.group(1)
        value = lookup.get(key.lower())
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
