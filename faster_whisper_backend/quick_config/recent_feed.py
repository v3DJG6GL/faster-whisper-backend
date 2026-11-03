"""
Tokenization + SSE broadcast layer for /quick-config recent transcriptions.

Durable storage lives in `recent_transcriptions_store` (SQLite, WAL). This
module:
  * tokenizes each transcription into single words + adjacent two-word
    phrases (bigrams) for the cb:map autocomplete datalist on
    /quick-config;
  * forwards each finished trace to the SQLite store via record_trace();
  * broadcasts the trace to live SSE subscribers so the UI updates in
    real time without polling.

The bigram pass exists for the common "whisper split a compound" case
— user dictates "Hanspeter", model emits "Hans Peter"; typing "Hans"
should offer both single-word and two-word completions so the user
can pick the phrase as a cb:map key in one step.

The trace + tokens carry literal dictation snippets, which can be
sensitive personal data — DO NOT log buffer contents. The on-disk log
file already holds the same trace via transcription.receipt._format_request_block;
the SQLite store is the canonical structured durable record.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from collections.abc import Iterable, Iterator
from typing import Any

from faster_whisper_backend.core.unicode_marks import JOINERS as _JOINERS
from faster_whisper_backend.core.unicode_marks import MARKS as _MARKS

logger = logging.getLogger("whisper-api")

# Token shape: a leading letter, then word-chars (letters/digits/marks/-),
# 2 to 64 chars. Drops single-char punctuation noise without aggressively
# splitting compound words (which is the whole point — the user's
# mis-recognition target IS the compound). The leading class is any
# Unicode letter, not just ASCII + umlauts: the tail is \w anyway, so a
# narrower start would cut "Éric" to "ric" — a key the word-bounded
# cb:map regex can never match. For the same reason the match may not
# start right after a word char: "5mg" / "_intern" must not yield the
# fragments "mg" / "intern".
_TOKEN_MIN_LEN = 2
_TOKEN_MAX_LEN = 64
# A mark or a ZWNJ / ZWJ counts as a word char on both sides: it extends
# the tail and, before a letter, means the match would start partway into
# a word (Persian می + ZWNJ + خواهم is one word, not "می" and "خواهم").
_TOKEN_RE = re.compile(rf"(?<![\w{_MARKS}{_JOINERS}])[^\W\d_][\w{_MARKS}{_JOINERS}\-]*")

# Small German stopword set — never useful as autocomplete candidates
# for cb:map keys. Lowercase compare; preserve original casing on display.
_STOPWORDS = frozenset({
    "der", "die", "das", "den", "dem", "und", "oder", "mit", "von", "zu",
    "im", "in", "an", "am", "auf", "für", "fuer", "ist", "sind", "wird",
    "wurde", "werden", "ein", "eine", "einen", "einer", "einem", "auch",
    "sich", "nicht", "nur", "aber", "als", "wie", "so", "bei", "aus",
    "nach", "vor", "über", "unter", "durch",
})


def _is_content(tok: str) -> bool:
    """In-range length, not a stopword, and a key the cb:map schema accepts.
    Python \\w takes category No (² ³ ₂ ½), the map-key pattern's Rust \\w
    does not, so "m²" / "CO₂" would 422 on Save. Drop the whole token: "m" /
    "km" are fragments the word-bounded map regex never matches in "km²"."""
    return (_TOKEN_MIN_LEN <= len(tok) <= _TOKEN_MAX_LEN
            and tok.lower() not in _STOPWORDS
            and not any(unicodedata.category(c) == "No" for c in tok))


# Per-transcription extraction cap: keep candidates relevant; bound
# memory and SSE payload size.
_TOKEN_CAP = 100

# Same idea for adjacent two-word phrases — kept lower because the
# multiplicative effect (N tokens → ~N-1 bigrams) plus per-trace
# datalist contribution would otherwise dominate the option list.
_BIGRAM_CAP = 50

# Each subscriber is a bounded asyncio.Queue. Bounded so a slow / stuck
# SSE client can't grow memory without limit; on overflow we drop the
# event for that client (the client misses a trace but stays connected).
_subscribers: list[asyncio.Queue] = []
_SUBSCRIBER_QUEUE_MAX = 64


def dedup_first_casing(words: Iterable[str], cap: int) -> list[str]:
    """Case-insensitive de-dup of `words` (first-seen casing wins, empty
    strings dropped), insertion-ordered, stopping once `cap` (>= 1) are kept.
    Pass a generator: the early stop then also stops producing candidates."""
    seen: dict[str, str] = {}
    for w in words:
        key = w.lower()
        if key and key not in seen:
            seen[key] = w
            if len(seen) >= cap:
                break
    return list(seen.values())


def _tokenize(text: str) -> list[str]:
    """Content tokens, de-duped case-insensitively (first-seen casing wins),
    insertion-ordered, capped at _TOKEN_CAP."""
    toks = (m.group(0) for m in _TOKEN_RE.finditer(text or ""))
    return dedup_first_casing((t for t in toks if _is_content(t)), _TOKEN_CAP)


def _extract_bigrams(text: str) -> list[str]:
    """Adjacent content-token pairs joined with a single space. A pair
    counts only when BOTH tokens are content (non-stopword, in-range
    length) AND the original text has nothing but whitespace between
    them — a comma or period between two words means they belong to
    different phrases and should NOT form a bigram.

    Insertion-ordered, lowercased de-duped: first-seen casing wins.
    Cap at _BIGRAM_CAP. Returns [] for empty/whitespace input.

    Example: "Hans Peter, und Anna Müller" → ["Hans Peter", "Anna Müller"]
    (the comma blocks the "Peter und" pair; "und" is a stopword
    anyway, which blocks "und Anna")."""
    src = text or ""
    matches = list(_TOKEN_RE.finditer(src))

    def _pairs() -> Iterator[str]:
        for a, b in zip(matches, matches[1:]):
            ta, tb = a.group(0), b.group(0)
            if not (_is_content(ta) and _is_content(tb)):
                continue
            between = src[a.end():b.start()]
            if between.strip():
                continue
            yield f"{ta} {tb}"

    return dedup_first_casing(_pairs(), _BIGRAM_CAP)


def record_trace(
    *,
    request_id: str | None = None,
    model: str,
    raw: str,
    steps: list,
    final: str,
    language: str | None = None,
    source: str = "file",
    user_id: str | None = None,
    status: str = "ok",
) -> None:
    """Persist a finished trace to the durable store and broadcast it to
    every live SSE subscriber. Called by main.py's transcribe handler and by
    streaming/routes.py for each live-dictation final.

    `request_id` is the request's id: it lets /quick-config correlate a
    user-submitted report with the text log (`req=<id[:8]>`), and lets the
    later metrics.record_transcription() call UPSERT timing fields onto the
    same row.

    `steps` is a list of (label, before, after) tuples; pass [] when tracing
    is off (autocomplete needs only raw + final + tokens).

    `status` only labels the live entry: the durable row's status is the
    one the later record_transcription() UPSERT writes, so a caller that
    will record an error passes it here too, or the live list disagrees
    with a reload."""
    if not request_id:
        # No request_id: nothing to key the UPSERT on, so only the live
        # SSE broadcast below runs.
        pass
    try:
        from faster_whisper_backend.auth import api_keys_store
        username = api_keys_store.get_username(user_id)
    except Exception:
        username = None
    from faster_whisper_backend.stats import recent_transcriptions_store as _rts
    # Same caps as the durable row: the live entry is what /quick-config's
    # Report posts, and the report route's edge bounds 422 anything past them.
    raw, final_text, steps = _rts.cap_trace_fields(raw or "", final or "", steps or [])
    tokens = _tokenize(final_text)
    bigrams = _extract_bigrams(final_text)
    created_ts = time.time()
    entry: dict[str, Any] = {
        "ts": created_ts,
        "created_ts": created_ts,
        "request_id": request_id,
        "model": model,
        "language": language,
        "source": source or "file",
        "raw": raw or "",
        "raw_text": raw or "",
        "steps": steps,
        "final": final_text,
        "final_text": final_text,
        "tokens": tokens,
        "bigrams": bigrams,
        "user_id": user_id,
        "username": username,
        "status": status,
    }
    if request_id:
        try:
            from faster_whisper_backend.settings import config as cfg
            from faster_whisper_backend.stats import recent_transcriptions_store
            recent_transcriptions_store.record_trace(
                request_id=request_id,
                model=model,
                raw=raw or "",
                final=final_text,
                steps=entry["steps"],
                tokens=tokens,
                bigrams=bigrams,
                language=language,
                source=source,
                user_id=user_id,
                username=username,
                created_ts=created_ts,
                prune_every=int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PRUNE_EVERY", 50)),
                max_rows=int(getattr(cfg, "RECENT_TRANSCRIPTIONS_MAX", 500)),
                ttl_days=float(getattr(cfg, "RECENT_TRANSCRIPTIONS_RETENTION_DAYS", 30)),
            )
        except Exception as e:
            logger.warning("[recent-tx] persist failed: %s", e)
    _broadcast({"event": "trace", "data": entry})


def subscribe() -> asyncio.Queue:
    """Register a new SSE subscriber. Caller MUST call unsubscribe()
    in a finally block when the connection ends."""
    q: asyncio.Queue = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    try:
        _subscribers.remove(q)
    except ValueError:
        pass


def _broadcast(item: dict[str, Any]) -> None:
    """Push to every subscriber; drop on overflow rather than blocking."""
    for q in list(_subscribers):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass
