"""Post-decode guards that cut a made-up TAIL inside one segment.

Whisper (large-v3 above all) sometimes does not emit end-of-text after the last
spoken word. It keeps writing a fluent continuation — often a phrase looping
3-4 times — and the real words and the made-up tail land in ONE segment.
2026-09-19: "Besser wäre wahrscheinlich einmal pro Tag zwei Tabletten" came back
with "zu nehmen und dann die Doppelpunktzutaten abzuschalten" appended three
times. Trigger was hotwords + the previous sentence as prompt; a temperature
retry only invented a different loop; cr 1.94 / alp -0.82 sat inside every
threshold; faster-whisper's hallucination_silence_threshold scores only the
first 8 words and drops whole segments only.

``main.segment_exceeds_word_rate`` cannot see it either: it averages over the
whole segment, so the real words dilute the tail (4.3 w/s there), and it can
only drop a whole segment — which would delete what was really said.

What gives the tail away: faster-whisper aligns words against the window's real
audio frames only (``find_alignment``), so every word emitted after the frames
run out lands on the last frame with EXACTLY zero length, stacked at the end of
the audio (measured: 12 word starts within one second, 10 zero-length words;
clean dictation: at most 3 starts per second, none zero-length). whisper-
timestamped (``remove_empty_words``) and stable-ts (``max_instant_words``) use
the same signal.

Three independent rules, each with its own setting, each 0 = off. All three are
TAIL-ANCHORED: they only ever cut ``words[idx:]``, never the middle of a
segment, and all three indices are computed on the ORIGINAL word list (cutting
the zero-length pile first would blind the burst rule) — the smallest wins.

  burst      SEGMENT_MAX_WORD_BURST_PER_S
  zero_tail  SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS
  repeat     SEGMENT_REPEAT_COLLAPSE_MIN_REPEATS

Pure module: no import of ``main`` (streaming imports main lazily).
"""

import re
import unicodedata

# A word counts as zero-length below this. The made-up words are EXACT zeros
# (start == end == last frame); a real one-frame word is 0.02 s and must stay,
# and ``0.02`` itself is not exactly representable, so compare well below it.
_ZERO_LEN_S = 0.005
# The burst rule looks at the segment's last second.
_BURST_WINDOW_S = 1.0
# Repeat rule: phrases shorter than this are never touched — dictation commands
# are legitimately repeated ("Neue Zeile Neue Zeile Neue Zeile").
_REPEAT_MIN_PHRASE_WORDS = 3
# Longest phrase (period) searched; bounds the cost at O(32·n).
_REPEAT_MAX_PERIOD = 32

_UNIT_RE = re.compile(r"\s*\S+")


def _start(w) -> float:
    return float(getattr(w, "start", 0.0) or 0.0)


def _end(w) -> float:
    return float(getattr(w, "end", 0.0) or 0.0)


def _is_zero(w) -> bool:
    return (_end(w) - _start(w)) < _ZERO_LEN_S


def _key(unit: str) -> str:
    """Comparison key for the repeat rule: NFKC, casefold, punctuation and
    whitespace stripped. A unit that is punctuation only keeps its raw form."""
    s = unicodedata.normalize("NFKC", unit or "").casefold()
    k = "".join(ch for ch in s
                if not ch.isspace() and not unicodedata.category(ch).startswith("P"))
    return k or s.strip()


def _burst_cut(words, limit: float) -> int | None:
    """More than ``limit`` words start within the segment's last second → cut at
    the first word of the pile: the first word in that window whose gap to the
    next word is below 1/limit s. Real words inside the look-back second keep
    their normal gaps and stay."""
    if not limit or limit <= 0 or len(words) < 2:
        return None
    t_last = _start(words[-1])
    lo = next(i for i, w in enumerate(words) if _start(w) > t_last - _BURST_WINDOW_S)
    if len(words) - lo <= limit:
        return None
    gap = 1.0 / float(limit)
    for i in range(lo, len(words) - 1):
        if _start(words[i + 1]) - _start(words[i]) < gap:
            return i
    return None


def _zero_tail_cut(words, min_words: int) -> int | None:
    """The segment ends with at least ``min_words`` zero-length words → cut the
    run. The leftover audio time must go to SOME word: when that absorber sits
    between zero-length words (``nehmen(0) → und → pile``) it is part of the
    tail and is cut too; a non-zero word with a spoken word before it is the
    real last word and is never cut."""
    if not min_words or min_words <= 0:
        return None
    i = len(words)
    while i > 0 and _is_zero(words[i - 1]):
        i -= 1
    if len(words) - i < min_words:
        return None
    while i >= 2 and _is_zero(words[i - 2]):
        i -= 1                      # the sandwiched absorber
        while i > 0 and _is_zero(words[i - 1]):
            i -= 1
    return i


def _repeat_cut(keys: list[str], min_repeats: int) -> int | None:
    """The segment ENDS with a phrase of >= 3 words repeated >= ``min_repeats``
    times back-to-back (an incomplete last copy counts towards the run) → keep
    the first copy, cut the rest. A repeat in the middle of a segment (a song
    refrain) is left alone: a missing-end-of-text loop always runs to the end.
    The smallest period wins, so "Neue Zeile" ×6 is period 2, not a 4-word
    phrase ×3, and is never touched."""
    n = len(keys)
    if not min_repeats or min_repeats < 2 or n < _REPEAT_MIN_PHRASE_WORDS * min_repeats:
        return None
    for p in range(1, min(_REPEAT_MAX_PERIOD, n // min_repeats) + 1):
        i = n - p - 1
        while i >= 0 and keys[i] == keys[i + p]:
            i -= 1
        run = n - 1 - i             # words covered by the periodic run
        if run >= min_repeats * p:
            if p < _REPEAT_MIN_PHRASE_WORDS:
                return None         # primitive period is a short phrase
            return n - run + p
    return None


def find_tail_cut(words, text: str, *, burst: float = 0.0, zero_tail: int = 0,
                  repeats: int = 0) -> tuple[int, list[str]] | None:
    """Index to cut at plus the rules that fired, or None. With ``words`` the
    index is into ``words``; without (word timestamps off) only the repeat rule
    runs and the index is into the whitespace units of ``text``."""
    hits: dict[str, int] = {}
    if words:
        b = _burst_cut(words, float(burst or 0))
        if b is not None:
            hits["burst"] = b
        z = _zero_tail_cut(words, int(zero_tail or 0))
        if z is not None:
            hits["zero_tail"] = z
        keys = [_key(getattr(w, "word", "") or "") for w in words]
    else:
        keys = [_key(u) for u in _UNIT_RE.findall(text or "")]
    r = _repeat_cut(keys, int(repeats or 0))
    if r is not None:
        hits["repeat"] = r
    if not hits:
        return None
    idx = min(hits.values())
    # Every rule whose own cut point lies inside the removed tail "fired".
    return idx, [name for name in ("burst", "zero_tail", "repeat") if name in hits]


def cut_text(text: str, kept_words) -> tuple[str, bool]:
    """``text`` shortened to the kept words. segment.text is NOT guaranteed to
    equal the joined words (faster-whisper merges punctuation across
    sub-segments), so walk the text word by word. Returns (kept_text,
    rebuilt) — rebuilt=True when the walk failed and the words were joined."""
    text = text or ""
    pos = 0
    for w in kept_words:
        tok = (getattr(w, "word", "") or "").strip()
        if not tok:
            continue
        idx = text.find(tok, pos)
        if idx < 0:
            return "".join(getattr(x, "word", "") or "" for x in kept_words), True
        pos = idx + len(tok)
    return text[:pos], False


def apply_tail_guards(seg, *, burst: float = 0.0, zero_tail: int = 0,
                      repeats: int = 0) -> dict | None:
    """Run the three rules on one decoded segment and cut it IN PLACE (words,
    text, end). In-place is safe: the objects come fresh from the decoder with
    this request as their only consumer. Returns None when nothing was cut, else
    ``{"rules", "n", "from", "text"}`` — ``text`` is what was REMOVED. Never
    raises: a guard must not break a decode."""
    try:
        words = list(getattr(seg, "words", None) or [])
        text = getattr(seg, "text", "") or ""
        found = find_tail_cut(words, text, burst=burst, zero_tail=zero_tail,
                              repeats=repeats)
        if found is None:
            return None
        idx, rules = found
        if words:
            kept = words[:idx]
            kept_text, rebuilt = cut_text(text, kept)
            info = {"rules": rules, "n": len(words) - idx,
                    "from": _start(words[idx]),
                    "text": "".join(getattr(w, "word", "") or "" for w in words[idx:])}
            if rebuilt:
                info["text_rebuilt"] = True
            seg.words = kept
            if kept:
                seg.end = max(_end(kept[-1]), float(getattr(seg, "start", 0.0) or 0.0))
        else:
            units = _UNIT_RE.findall(text)
            kept_text = "".join(units[:idx])
            info = {"rules": rules, "n": len(units) - idx, "from": None,
                    "text": "".join(units[idx:])}
        seg.text = kept_text
        return info
    except Exception:  # noqa: BLE001 — see docstring
        return None


def tail_words_diag(seg, last: int = 5) -> str | None:
    """Compact dump of a segment's last words when one of them is zero-length
    but nothing was cut — the data needed to tune the zero-length rule for short
    made-up endings (word, start, end, probability)."""
    words = list(getattr(seg, "words", None) or [])[-last:]
    if not any(_is_zero(w) for w in words):
        return None
    return " ".join(
        f"{(getattr(w, 'word', '') or '').strip()!r}@{_start(w):.2f}-{_end(w):.2f}"
        f"/p{float(getattr(w, 'probability', 0.0) or 0.0):.2f}" for w in words)


def describe_cut(info: dict) -> str:
    """One receipt line for a cut: rules · n words from t: 'removed…'."""
    removed = info.get("text") or ""
    if len(removed) > 60:
        removed = removed[:57] + "…"
    where = f" from {info['from']:.2f}s" if info.get("from") is not None else ""
    return f"{'+'.join(info.get('rules') or [])} · {info.get('n', 0)} words{where}: {removed!r}"
