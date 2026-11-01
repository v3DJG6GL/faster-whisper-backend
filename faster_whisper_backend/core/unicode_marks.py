"""Every Unicode combining mark as a regex char-class body.

`re` has no ``\\p{M}`` and ``\\w`` excludes marks (Mn/Mc/Me), so a word
boundary built from ``\\w`` alone sees one at every vowel sign: "बोलते"
reads as "बो" + "लते", "ครับ" as "คร" + "ับ". The autocomplete tokenizer
(quick_config/recent_feed) and the cb:map key bounds (pipeline/dictation_map)
both splice ``MARKS`` and ``JOINERS`` into their classes next to ``\\w``.
Built once at import (~0.1 s); pure module, no package imports.
"""
from __future__ import annotations

import re
import sys
import unicodedata


def _combining_marks_class() -> str:
    """Every Unicode combining mark (Mn/Mc/Me) as char-class ranges: Indic
    and Thai vowel signs, Arabic vowelling, NFD-decomposed Latin accents."""
    ranges: list[list[int]] = []
    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp))[0] == "M":
            if ranges and ranges[-1][1] == cp - 1:
                ranges[-1][1] = cp
            else:
                ranges.append([cp, cp])
    return "".join(
        re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}"
        for a, b in ranges
    )


MARKS = _combining_marks_class()

# ZERO WIDTH NON-JOINER / JOINER (category Cf) sit inside a word: Persian
# writes "I want" as می + ZWNJ + خواهم and "books" as کتاب + ZWNJ + ها, Indic
# scripts use them in conjuncts. Without them a boundary falls at the
# joiner and "می" / "ها" read as whole words.
JOINERS = "\u200c\u200d"
