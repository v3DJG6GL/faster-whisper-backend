"""Shared schema for word-correction chips.

Both /reports (transcription-error reports) and /captures (fine-tuning
training samples) accept the same chip shape:

    {wrong: str, correct: str, idx: int?, idx_end: int?}

The cleaner and the merge live here so the two stores stay in lockstep
(a chip field added here reaches both surfaces). It is in core/, not in
either store, so neither store has to import the other.
"""
from __future__ import annotations

from typing import Any

# Caps applied server-side before insert. The route layer already validates
# via Pydantic, but accept-then-trim is what protects against future code
# paths that bypass the route (admin scripts, migrations).
CAP_CORRECTION_FIELD = 200
# Same as the captures PATCH schemas' list bound (max_length=200): a capture
# runs up to 10 minutes, and a chip the route accepted must not be dropped
# here without a word.
CAP_CORRECTIONS = 200


def clean_corrections(items: list[Any] | None) -> list[dict[str, Any]]:
    """Filter to entries with a non-empty `correct` field, apply length
    caps, and cap the list at CAP_CORRECTIONS. Anything malformed is
    dropped silently — this is end-user input, we tolerate it.

    Optional `idx_end` lets a chip span multiple adjacent words from
    the original final text. Stored only when it's a valid int with
    `idx <= idx_end < 10_000` and `idx_end != idx`; otherwise the entry
    stays single-word."""
    out: list[dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        wrong = str(it.get("wrong", "") or "").strip()[:CAP_CORRECTION_FIELD]
        correct = str(it.get("correct", "") or "").strip()[:CAP_CORRECTION_FIELD]
        if not correct:
            continue
        entry: dict[str, Any] = {"wrong": wrong, "correct": correct}
        idx = it.get("idx")
        if isinstance(idx, int) and 0 <= idx < 10_000:
            entry["idx"] = idx
            idx_end = it.get("idx_end")
            if (isinstance(idx_end, int)
                    and idx <= idx_end < 10_000
                    and idx_end != idx):
                entry["idx_end"] = idx_end
        out.append(entry)
        if len(out) >= CAP_CORRECTIONS:
            break
    return out


def over_cap(items: list[Any] | None) -> bool:
    """True when clean_corrections would cut valid entries (non-empty
    `correct`) at CAP_CORRECTIONS — lets a route refuse with a 422 instead
    of saving a silently shortened list (a three-way merge can grow past
    the schema's bound)."""
    n = 0
    for it in items or []:
        if isinstance(it, dict) and str(it.get("correct", "") or "").strip():
            n += 1
            if n > CAP_CORRECTIONS:
                return True
    return False


def three_way_merge_corrections(
    baseline: list[dict[str, Any]] | None,
    edited: list[dict[str, Any]] | None,
    current: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge chip lists across a concurrent-edit window.

    Inputs:
      baseline — chips the client loaded with its GET (snapshot at T=0).
      edited   — chips the client wants after the user's edits (T=save).
      current  — chips currently in the DB (post any concurrent writes
                 that landed between T=0 and T=save, e.g. another admin
                 saving the same capture/group from another tab).

    Algorithm:
      Start from `current` (post-concurrent state). Then for each chip
      key in (baseline ∪ edited), apply only the user's delta:

        * key in baseline AND NOT in edited  → user removed it: drop it
          from output (idempotent if also gone from `current`).
        * key in edited AND NOT in baseline  → user added it: insert,
          overwriting any concurrent chip at the same key.
        * key in both AND payload differs    → user edited it: payload
          wins over any concurrent edit.
        * key in both AND payload equal      → untouched by the user:
          keep whatever `current` has at that key (which may itself be
          a concurrent edit).

    Merge key is `(idx, idx_end)` with idx_end defaulting to idx for
    anchored chips. Anchorless chips (no integer `idx`) have no positional
    identity — keying them by position would collapse every (None, None)
    entry into one — so they are keyed by `(wrong, correct)` under the same
    delta rule; editing one's `correct` is a remove plus an add. With an
    empty baseline (the reports resubmit path) the result is the union of
    `current` and `edited`."""
    def key(c: dict[str, Any]) -> "tuple[int, int] | None":
        i = c.get("idx")
        if not isinstance(i, int):
            return None
        e = c.get("idx_end")
        return (i, e if isinstance(e, int) else i)

    def _anchorless_id(c: dict[str, Any]) -> tuple[str, str]:
        return (str(c.get("wrong") or ""), str(c.get("correct") or ""))

    def _split(items):
        anchored: dict[tuple[int, int], dict[str, Any]] = {}
        anchorless: dict[tuple[str, str], dict[str, Any]] = {}
        for c in (items or []):
            if not isinstance(c, dict):
                continue
            k = key(c)
            if k is None:
                anchorless[_anchorless_id(c)] = c
            else:
                anchored[k] = c
        return anchored, anchorless

    base_anc, base_free = _split(baseline)
    edit_anc, edit_free = _split(edited)
    cur_anc, cur_free = _split(current)

    out = dict(cur_anc)
    for k in set(base_anc) | set(edit_anc):
        in_b = k in base_anc
        in_e = k in edit_anc
        if in_b and not in_e:
            out.pop(k, None)
        elif in_e and not in_b:
            out[k] = edit_anc[k]
        elif base_anc[k] != edit_anc[k]:
            out[k] = edit_anc[k]

    def _sort_key(c: dict[str, Any]) -> tuple[int, int]:
        i = c.get("idx")
        try:
            return (0, int(i))
        except (TypeError, ValueError):
            return (1, 0)

    # Untouched anchorless chips (equal to baseline) are not written back,
    # so a concurrent removal stays removed.
    merged_free = {k: v for k, v in cur_free.items()
                   if not (k in base_free and k not in edit_free)}
    merged_free.update({k: v for k, v in edit_free.items()
                        if base_free.get(k) != v})
    return sorted(out.values(), key=_sort_key) + sorted(
        merged_free.values(), key=_sort_key,
    )
