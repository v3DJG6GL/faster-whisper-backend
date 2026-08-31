"""Hold a dictation's request receipt open until its translation lands.

WHY THIS EXISTS

A batch transcription translates in-band, so its receipt can describe the
whole pipeline. Dictation cannot: the stream finalizes an utterance, logs its
receipt, and the client THEN makes a separate HTTP request to translate the
text. The two were unlinked in the log — a receipt at 18:22:57, then four
orphan `[translate]` lines with nothing tying them to the utterance they
belonged to.

So the receipt is parked here instead of logged, and the translate request
claims it by `captured_id` and merges its own sections in. One block per
utterance, in the order a human reads it.

THE TIMEOUT IS AN IDLE TIMER, NOT A DEADLINE

A cold GGUF load is ~12 s, and on a busy GPU can be much worse; an absolute
deadline would split exactly the receipts that most need to be whole. The
translate job already heartbeats through its progress callback, and every
tick restamps the hold — so a slow load waits as long as it genuinely needs.
Only a crashed, wedged or never-sent translation trips the timeout.

NOTHING IS EVER SILENTLY LOST

Every exit releases the receipt exactly once, with a note saying why: the
client cancelled, the translation failed, nobody claimed it, the buffer
filled, or the server shut down. A held receipt that vanished would be worse
than the split one it replaces.

This module deliberately imports nothing from the app. It stores the
renderer's kwargs rather than rendered text (so the claim can add sections
rather than concatenate blocks), and the caller does the rendering and the
logging. That keeps it importable from both main and streaming_routes with
no cycle.
"""

from __future__ import annotations

import threading
import time
from typing import Any

# A receipt is a few KB of formatted strings. The cap exists so a live
# session that translates every phrase, with a client that then dies, cannot
# grow this without bound.
_MAX_HELD = 32

_lock = threading.Lock()
_held: "dict[str, dict[str, Any]]" = {}


def park(key: str, kwargs: "dict[str, Any]", *, hold_s: float) -> None:
    """Hold this receipt instead of logging it.

    Only ever called when the client DECLARED it will translate. Without a
    declaration the receipt is logged immediately, which is also every
    non-dictation caller's path — that is what keeps an old client's
    behaviour identical."""
    if not key:
        return
    now = time.monotonic()
    with _lock:
        if key not in _held and len(_held) >= _MAX_HELD:
            # Evict the least recently touched, not the oldest: a long,
            # actively-progressing translation must outlive a stalled one.
            victim = min(_held, key=lambda k: _held[k]["touched"])
            evicted = _held.pop(victim)
            evicted["note"] = "released — pending receipt buffer full"
            _overflow.append(evicted)
        _held[key] = {
            "kwargs": dict(kwargs),
            "touched": now,
            "hold_s": float(hold_s),
        }


# Receipts evicted by the cap, waiting for the caller to drain and log them.
# Never dropped on the floor: eviction is a reason to log, not to forget.
_overflow: "list[dict[str, Any]]" = []


def touch(key: str) -> bool:
    """Restamp the idle timer. Returns whether the key was actually held, so
    a caller can tell a real heartbeat from a stale one."""
    if not key:
        return False
    with _lock:
        entry = _held.get(key)
        if entry is None:
            return False
        entry["touched"] = time.monotonic()
        return True


def claim(key: str, **extra: Any) -> "dict[str, Any] | None":
    """Take the held receipt and merge `extra` into its render kwargs.

    Returns None when nothing was held — which is the normal case for every
    caller that is not a declared dictation, so it is not an error."""
    if not key:
        return None
    with _lock:
        entry = _held.pop(key, None)
    if entry is None:
        return None
    merged = entry["kwargs"]
    for k, v in extra.items():
        if v is not None:
            merged[k] = v
    return merged


def release(key: str, note: str) -> "dict[str, Any] | None":
    """Give up on a held receipt and return it for logging, with `note`
    appended to its Notes section so the log says why it is incomplete."""
    if not key:
        return None
    with _lock:
        entry = _held.pop(key, None)
    if entry is None:
        return None
    return _with_note(entry, note)


def _with_note(entry: "dict[str, Any]", note: str) -> "dict[str, Any]":
    kwargs = entry["kwargs"]
    notes = list(kwargs.get("skipped") or [])
    notes.append(f"translate  {note}")
    kwargs["skipped"] = notes
    return kwargs


def sweep() -> "list[dict[str, Any]]":
    """Release everything that has gone quiet for longer than its hold, plus
    anything the cap evicted. Returns render kwargs for the caller to log."""
    now = time.monotonic()
    out: "list[dict[str, Any]]" = []
    with _lock:
        stale = [k for k, e in _held.items()
                 if now - e["touched"] > e["hold_s"]]
        for k in stale:
            entry = _held.pop(k)
            out.append(_with_note(
                entry,
                f"no result within {entry['hold_s']:.0f}s — receipt released"))
        while _overflow:
            e = _overflow.pop(0)
            out.append(_with_note(e, e.get("note") or "released"))
    return out


def flush_all(note: str = "released at shutdown") -> "list[dict[str, Any]]":
    """Drain everything. Called on lifespan exit so a restart never eats a
    receipt that was merely waiting."""
    with _lock:
        entries = list(_held.values()) + list(_overflow)
        _held.clear()
        _overflow.clear()
    return [_with_note(e, e.get("note") or note) for e in entries]


def pending() -> int:
    with _lock:
        return len(_held)


def _reset_for_tests() -> None:
    with _lock:
        _held.clear()
        _overflow.clear()
