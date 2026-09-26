"""Which trailing raw words live dictation must not format yet.

Streaming dictation formats the document utterance by utterance, and the
client types every new document's growth — text it has typed is never taken
back. The pipeline is not prefix-stable, though: a few rules decide what they
do by looking at the words that FOLLOW. Formatting "… Frau Kollegin, Komma,
neuer" and sending the result, then receiving " Absatz." with the next
utterance, would need the typed "neuer" to turn into a paragraph break — and
"Komma" into nothing, because the newline tidy eats a comma before a line
break. The fuzzing that found this (every word boundary of a German letter
corpus) traced almost every such rewrite to the dictation map:

* a multi-word key cut after one of its first words ("neue | Zeile",
  "Klammer | auf", "größer | als");
* a key whose value JOINS its neighbours — ``,`` (tidied away before a
  newline), ``-`` and ``/`` (digit ranges and hyphen compounds: "120
  Schrägstrich | 80" → "120/80", "Eisen Bindestrich | Infusion" →
  "Eisen-Infusion");
* Whisper's own trailing hyphen or slash ("-." once the decoder closes the
  sentence).

So the session holds those words back: they stay in the raw transcript, are
not formatted, not sent, and come out with the next utterance — or with a
release when nothing follows (hard break, flush, close, idle). The spec is
built from the ACTIVE map rules, so a key an admin adds is held too.

Pure module: no import of ``main``. The words are compared the way the map
matches them, loosely: NFKC, casefolded, ß as ss, Whisper's punctuation at
the word edges ignored (``-`` and ``/`` are part of a key like "Et-Zeichen"
and stay).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping

# Values of a key whose word is held when it ends the text: the pipeline joins
# these with what comes next.
_JOINER_VALUES = frozenset({",", "-", "/"})
# Tokens held per compose. Longer chains ("Komma neue …") are not dictation.
MAX_HELD_TOKENS = 8
# Whisper's own trailing hyphen / slash, optionally with the punctuation the
# decoder closes a sentence with ("12-", "Eisen-.", "120/").
_TRAILING_JOINER_RE = re.compile(r"[-/][.,;:!?]*$")
_TOKEN_RE = re.compile(r"\S+")


def norm(tok: str) -> str:
    """Comparison form of one word: NFKC, casefold, ß → ss, edge punctuation
    stripped except ``-`` and ``/``."""
    s = unicodedata.normalize("NFKC", tok or "").casefold().replace("ß", "ss")
    i, j = 0, len(s)
    while i < j and _edge_punct(s[i]):
        i += 1
    while j > i and _edge_punct(s[j - 1]):
        j -= 1
    return s[i:j]


def _edge_punct(ch: str) -> bool:
    if ch in "-/":
        return False
    return unicodedata.category(ch)[0] in "PS" or ch.isspace()


def _key_tokens(key: str) -> tuple[str, ...]:
    return tuple(t for t in (norm(w) for w in (key or "").split()) if t)


@dataclass(frozen=True)
class HoldSpec:
    """Token sequences that must not end the formatted text: every proper
    prefix of a multi-word key, and every whole key whose value is a joiner."""
    sequences: frozenset
    longest: int

    def __bool__(self) -> bool:
        return bool(self.sequences)


def build_spec(lookups: Iterable[Mapping[str, str]]) -> HoldSpec:
    """Build the spec from the lookups of the active ``callback:map`` rules
    (``core.dictation_map.compile_map``'s third value, ß/ss variants
    included)."""
    seqs: set[tuple[str, ...]] = set()
    for lookup in lookups:
        for key, value in (lookup or {}).items():
            toks = _key_tokens(key)
            if not toks:
                continue
            for n in range(1, len(toks)):
                seqs.add(toks[:n])
            if str(value).strip() in _JOINER_VALUES:
                seqs.add(toks)
    return HoldSpec(frozenset(seqs), max((len(s) for s in seqs), default=0))


def held_start(raw: str, spec: HoldSpec | None) -> int:
    """Index into ``raw`` where the held tail starts (``len(raw)`` when nothing
    is held). The boundary sits right after the last word that is formatted
    now, so the formatted part carries no trailing whitespace.

    Repeated: after a held tail is found, the words before it are checked
    again ("Kollegin, Komma, neuer" holds "neuer", then "Komma"), up to
    MAX_HELD_TOKENS words in all."""
    toks = [(m.start(), m.end(), m.group(0)) for m in _TOKEN_RE.finditer(raw or "")]
    n = len(toks)
    keep = n                       # tokens [0, keep) are formatted now
    longest = spec.longest if spec else 0
    while keep > 0 and n - keep < MAX_HELD_TOKENS:
        took = 0
        room = min(longest, keep, MAX_HELD_TOKENS - (n - keep))
        for j in range(room, 0, -1):
            seq = tuple(norm(t[2]) for t in toks[keep - j:keep])
            if seq in spec.sequences:  # type: ignore[union-attr]
                took = j
                break
        if not took and _TRAILING_JOINER_RE.search(toks[keep - 1][2]):
            took = 1
        if not took:
            break
        keep -= took
    if keep == n:
        return len(raw or "")
    return toks[keep - 1][1] if keep else 0
