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

One HEAD-anchored rule sits beside them, for the mirror image at the start of a
decode: the model repeats the last words of its PROMPT (the rolling dictation
context, or the hotwords) before the new speech, and the echo has no audio
behind it either — zero-length or squeezed words stacked on the first frames.
Streaming dictation sends the previous utterances as prompt, so an echo there
types the end of the last sentence a second time.

  head_echo  SEGMENT_HEAD_ECHO_MIN_WORDS

It cuts ``words[:k]`` of the first surviving segment only when those k words
equal the prompt's last k words AND every one of them looks made up — a real
repeat ("… zwei Tabletten" / "zwei Tabletten am Abend") is spoken words with a
length and a confidence, and stays.

Pure module: no import of ``main`` (streaming imports main lazily).
"""

import re
import unicodedata

# A word counts as zero-length below this. The made-up words are EXACT zeros
# (start == end == last frame); a real one-frame word is 0.02 s and must stay,
# and ``0.02`` itself is not exactly representable, so compare well below it.
_ZERO_LEN_S = 0.005
# Zero-length rule, short tails: a made-up word right before the zero-length end
# is either squeezed short AND very unsure, or it is the one word that absorbed
# the leftover audio time and is very unsure. Thresholds are OpenAI's word
# anomaly score (whisper PR #1838: probability < 0.15, duration < 0.133 s).
# Measured 2026-09-19 on the 3-word tail "zu nehmen?, Fragezeichen":
#   'Tabletten' 4.62-5.52 p0.99 | 'zu' 5.52-5.58 p0.01 | 'nehmen?,' 5.58-6.12
#   p0.10 | 'Fragezeichen' 6.12-6.12 p0.33   (6.12 s = end of the audio)
_LOW_PROB = 0.15
_SHORT_S = 0.133
# A CONFIDENT absorber sandwiched between zero-length words is only taken up to
# this length: leftover audio after the last word is short, a longer confident
# word is speech ("wirklich" 0.4-1.9 p0.99 between two collapsed words).
_ABSORBER_MAX_S = 1.0
# The burst rule looks at the segment's last second.
_BURST_WINDOW_S = 1.0
# Burst rule: a word at least this long is a spoken word, however close the next
# one starts — short function words ("und", "die") last 0.08-0.12 s, below the
# 1/8 s gap. The made-up words before the pile are squeezed far below it
# (" zu" 5.54-5.58) or zero-length.
_BURST_REAL_MIN_S = 0.07
# Repeat rule: phrases shorter than this are never touched — dictation commands
# are legitimately repeated ("Neue Zeile Neue Zeile Neue Zeile").
_REPEAT_MIN_PHRASE_WORDS = 3
# Longest phrase (period) searched; bounds the cost at O(32·n).
_REPEAT_MAX_PERIOD = 32
# Head-echo rule: longest echo searched (also the setting's ceiling), and the
# number of leading words a near-miss diagnostic line shows.
_HEAD_ECHO_MAX_WORDS = 32
_HEAD_DIAG_WORDS = 8

_UNIT_RE = re.compile(r"\s*\S+")


def _start(w) -> float:
    return float(getattr(w, "start", 0.0) or 0.0)


def _end(w) -> float:
    return float(getattr(w, "end", 0.0) or 0.0)


def _is_zero(w) -> bool:
    return (_end(w) - _start(w)) < _ZERO_LEN_S


def _low_prob(w) -> bool:
    p = getattr(w, "probability", None)
    return p is not None and float(p) < _LOW_PROB


def _squeezed(w) -> bool:
    return (_end(w) - _start(w)) < _SHORT_S and _low_prob(w)


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
    next word is below 1/limit s AND which is itself zero-length or squeezed
    below 0.07 s. Real words inside the look-back second stay — those with
    normal gaps, and fast short ones too (a spoken word has a length). Only
    when no close word looks made up does the first close word start the cut."""
    if not limit or limit <= 0 or len(words) < 2:
        return None
    t_last = _start(words[-1])
    lo = next((i for i, w in enumerate(words) if _start(w) > t_last - _BURST_WINDOW_S),
              len(words) - 1)
    if len(words) - lo <= limit:
        return None
    gap = 1.0 / float(limit)
    first_close = None
    for i in range(lo, len(words) - 1):
        if _start(words[i + 1]) - _start(words[i]) < gap:
            if (_end(words[i]) - _start(words[i])) < _BURST_REAL_MIN_S:
                return i
            if first_close is None:
                first_close = i
    return first_close


def _zero_tail_cut(words, min_words: int) -> int | None:
    """The segment ends with a zero-length word and the made-up tail it closes
    is at least ``min_words`` long → cut the tail. Walking backwards from the
    end, the tail is made of:
      * zero-length words,
      * squeezed words (shorter than 0.133 s AND probability < 0.15),
      * the absorber — the leftover audio time must go to SOME word. It belongs
        to the tail, once, when it sits between a zero-length word and a pile
        of at least two zero-length words and is shorter than 1 s or unsure
        (``nehmen(0) → und → pile``), or, once, when its probability is < 0.15
        (``zu → nehmen?, → Fragezeichen(0)``).
    A confident non-zero word with a spoken word before it is the real last word
    and is never cut, nor is one between two single collapsed words; a single
    zero-length last word is kept (``min_words`` 1 is treated as off)."""
    if not min_words or min_words < 2 or not words or not _is_zero(words[-1]):
        return None
    i = len(words)
    unsure_absorber_used = False
    sandwiched_absorber_used = False
    while i > 0:
        w = words[i - 1]
        if _is_zero(w) or _squeezed(w):
            i -= 1
        elif (not sandwiched_absorber_used and i >= 2 and _is_zero(words[i - 2])
              and i + 1 < len(words) and _is_zero(words[i]) and _is_zero(words[i + 1])
              and ((_end(w) - _start(w)) < _ABSORBER_MAX_S or _low_prob(w))):
            sandwiched_absorber_used = True
            i -= 1                  # the sandwiched absorber
        elif _low_prob(w) and not unsure_absorber_used:
            unsure_absorber_used = True
            i -= 1
        else:
            break
    if len(words) - i < min_words:
        return None
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
    if int(repeats or 0) >= 2:      # the keys are only worth building when on
        if words:
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
            seg_start = float(getattr(seg, "start", 0.0) or 0.0)
            # Nothing kept: no consumer may inherit the made-up end.
            seg.end = max(_end(kept[-1]), seg_start) if kept else seg_start
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


# ---- head echo of the decode prompt ------------------------------------------


def prompt_tail_text(kwargs) -> str:
    """The text the decoder was conditioned on, as far as an echo can come from
    it: ``initial_prompt`` when set (streaming sends the rolling dictation
    context there), else ``hotwords`` (faster-whisper puts them in the same
    prompt slot when there is no initial prompt). "" when neither is set."""
    kwargs = kwargs or {}
    ip = kwargs.get("initial_prompt")
    if isinstance(ip, str) and ip.strip():
        return ip
    hw = kwargs.get("hotwords")
    if isinstance(hw, str) and hw.strip():
        return hw
    return ""


def _head_matches(words, prompt_text: str, min_words: int):
    """Every k (largest first, ``min_words`` <= k <= 32) for which the first k
    words, normalized like the repeat rule's keys, equal the prompt's last k
    whitespace tokens."""
    if not min_words or min_words < 2 or not words:
        return
    tokens = (prompt_text or "").split()
    if len(tokens) < min_words:
        return
    top = min(_HEAD_ECHO_MAX_WORDS, len(words), len(tokens))
    if top < min_words:
        return
    wkeys = [_key(getattr(w, "word", "") or "") for w in words[:top]]
    pkeys = [_key(t) for t in tokens[-top:]]
    for k in range(top, min_words - 1, -1):
        if wkeys[:k] == pkeys[len(pkeys) - k:]:
            yield k


def _looks_made_up(head) -> bool:
    """True when no word of ``head`` has a spoken word's shape: each one is
    zero-length, squeezed (under 0.133 s AND probability < 0.15) or shorter than
    0.07 s, except at most ONE unsure word (probability < 0.15) of any length —
    the absorber that soaked up the audio time before the real speech starts,
    as the zero-length tail rule allows one at the end."""
    absorber_used = False
    for w in head:
        if _is_zero(w) or _squeezed(w) or (_end(w) - _start(w)) < _BURST_REAL_MIN_S:
            continue
        if _low_prob(w) and not absorber_used:
            absorber_used = True
            continue
        return False
    return True


def find_head_echo(words, prompt_text: str, min_words: int) -> int | None:
    """Number of leading words to cut as an echo of the prompt, or None.

    The largest k (at most 32) whose first k words equal the prompt's last k
    words (NFKC, casefold, punctuation stripped) is tried first; it is cut only
    when every one of those words looks made up (see _looks_made_up). If not,
    the next smaller matching k is tried, down to ``min_words`` — a long match
    whose last word is real speech can still hide a shorter made-up echo in
    front of it. ``min_words`` 0 is off; 1 is treated as off (a single repeated
    word is far too common to call an echo)."""
    for k in _head_matches(words, prompt_text, min_words):
        if _looks_made_up(words[:k]):
            return k
    return None


def cut_head_text(text: str, removed_words, kept_words) -> tuple[str, bool]:
    """``text`` without the removed leading words — the head twin of cut_text.
    Walks the removed words through the text; when the walk fails the kept
    words are joined instead. Returns (kept_text, rebuilt)."""
    text = text or ""
    pos = 0
    for w in removed_words:
        tok = (getattr(w, "word", "") or "").strip()
        if not tok:
            continue
        idx = text.find(tok, pos)
        if idx < 0:
            return "".join(getattr(x, "word", "") or "" for x in kept_words), True
        pos = idx + len(tok)
    return text[pos:], False


def apply_head_echo_guard(seg, prompt_text: str, min_words: int) -> dict | None:
    """Cut an echo of the prompt from the start of one decoded segment IN PLACE
    (words, text, start). Needs word timestamps — without them there is no way
    to tell an echo from a real repeat, so nothing is cut. Returns None when
    nothing was cut, else ``{"rules", "n", "from", "to", "text"}`` — ``text`` is
    what was REMOVED. Never raises: a guard must not break a decode."""
    try:
        words = list(getattr(seg, "words", None) or [])
        k = find_head_echo(words, prompt_text, int(min_words or 0))
        if not k:
            return None
        removed, kept = words[:k], words[k:]
        text = getattr(seg, "text", "") or ""
        kept_text, rebuilt = cut_head_text(text, removed, kept)
        info = {"rules": ["head_echo"], "n": k, "from": _start(removed[0]),
                "to": _end(removed[-1]),
                "text": "".join(getattr(w, "word", "") or "" for w in removed)}
        if rebuilt:
            info["text_rebuilt"] = True
        seg.words = kept
        seg_end = float(getattr(seg, "end", 0.0) or 0.0)
        # Nothing kept: no consumer may inherit the made-up start.
        seg.start = min(_start(kept[0]), seg_end) if kept else seg_end
        seg.text = kept_text
        return info
    except Exception:  # noqa: BLE001 — see docstring
        return None


def head_words_diag(seg, prompt_text: str, min_words: int) -> str | None:
    """Compact dump of a segment's first words when they repeat the prompt's
    last words (at least ``min_words``) but look spoken, so nothing was cut —
    the near misses needed to tune the rule (word, start, end, probability)."""
    try:
        words = list(getattr(seg, "words", None) or [])
        k = next(_head_matches(words, prompt_text, int(min_words or 0)), None)
        if not k:
            return None
        return f"{k} words match the prompt: " + " ".join(
            f"{(getattr(w, 'word', '') or '').strip()!r}@{_start(w):.2f}-{_end(w):.2f}"
            f"/p{float(getattr(w, 'probability', 0.0) or 0.0):.2f}"
            for w in words[:min(k, _HEAD_DIAG_WORDS)])
    except Exception:  # noqa: BLE001 — diagnostics only
        return None


def describe_head_cut(info: dict) -> str:
    """One receipt line for a head cut: n words up to t: 'removed…'."""
    removed = info.get("text") or ""
    if len(removed) > 60:
        removed = removed[:57] + "…"
    return (f"prompt echo · {info.get('n', 0)} words "
            f"{info.get('from', 0.0):.2f}-{info.get('to', 0.0):.2f}s: {removed!r}")
