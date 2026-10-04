"""Which language does a link speak? The pure half of POST /v1/audio/url-language.

The route downloads the whole audio (through the guarded download — never a
ranged ffmpeg fetch, which would bypass the SSRF guard), decodes a few short
pieces spread over it and lets Whisper's language detection listen to each;
these two functions pick the pieces and turn what was heard into a verdict.
"""
from __future__ import annotations

from collections import Counter

PIECE_SECONDS = 20.0
_PIECE_AT = (0.2, 0.5, 0.8)    # fractions of the duration
_ONE_PIECE_BELOW_S = 45.0      # shorter media: one piece from the start
CONFIDENT = 0.7


def piece_starts(duration: "float | None") -> "list[float]":
    """Where the pieces start: at 20/50/80 % of the media (each kept inside
    it), or one piece at 0 when the media is short or its length unknown."""
    if not duration or duration < _ONE_PIECE_BELOW_S:
        return [0.0]
    last = duration - PIECE_SECONDS
    return [round(min(duration * f, last), 1) for f in _PIECE_AT]


def vote(heard: "list[tuple[str | None, float]]") -> dict:
    """(language, probability) per piece → {language, probability, verdict,
    also}. `detected`: at least two pieces (the only one, for a single
    piece) agree with p ≥ CONFIDENT. `mixed`: detected, and another language
    is confident in some other piece (`also`). `unknown` otherwise — then
    `language` is just the most probable piece's guess."""
    confident = [(lang, p) for lang, p in heard if lang and p >= CONFIDENT]
    counts = Counter(lang for lang, _p in confident)
    need = 1 if len(heard) == 1 else 2
    top, n = counts.most_common(1)[0] if counts else (None, 0)
    if n >= need:
        also = [lang for lang in counts if lang != top]
        agree = [p for lang, p in confident if lang == top]
        return {"language": top, "probability": round(sum(agree) / len(agree), 3),
                "verdict": "mixed" if also else "detected", "also": also}
    guess = max(heard, key=lambda h: h[1], default=(None, 0.0))
    return {"language": guess[0], "probability": round(guess[1], 3),
            "verdict": "unknown", "also": []}
