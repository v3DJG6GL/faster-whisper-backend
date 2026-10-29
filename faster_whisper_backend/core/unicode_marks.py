"""Every Unicode combining mark as a regex char-class body.

`re` has no ``\\p{M}`` and ``\\w`` excludes marks (Mn/Mc/Me), so a word
boundary built from ``\\w`` alone sees one at every vowel sign: "बोलते"
reads as "बो" + "लते", "ครับ" as "คร" + "ับ". The autocomplete tokenizer
(quick_config/recent_feed) and the cb:map key bounds (pipeline/dictation_map)
both splice ``MARKS`` into their classes next to ``\\w``. Built once at
import (~0.1 s); pure module, no package imports.
"""
from __future__ import annotations

import re
import sys
import unicodedata


def _combining_marks_class() -> str:
    """Every Unicode combining mark (Mn/Mc/Me) as char-class ranges.

    Without them a token stopped at every vowel sign: "हिन्दी भाषा बोलते"
    gave only "लत", a fragment from inside "बोलते" that, picked as a cb:map
    key, rewrites the inside of other words. Same for Thai, Bengali,
    vowelled Arabic and NFD-decomposed Latin ("E" + U+0301 + "ric" gave
    "ric")."""
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
