"""Sample building for /captures: the merged-WAV build, the default
transcript join, the raw-word → final-text alignment behind the karaoke band,
and the per-sample rebuild lock. Shared by the captures routes and the two
background workers (reapply, vad_reprocess), which call through this module's
attributes so one test patch reaches every caller. Imports neither
captures.routes nor the workers.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from typing import Any

from fastapi import HTTPException, status

from faster_whisper_backend.captures import store as captures_store
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config as cfg


_JOIN_STR = {"space": " ", "period_space": ". "}


def _global_silence_ms() -> int:
    """Inter-member silence, sourced from the global VAD-internal knob
    (was a per-merge `silence_ms` payload field)."""
    try:
        return int(getattr(cfg, "CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS", 300))
    except (TypeError, ValueError):
        return 300


def _global_edge_ms() -> int:
    """Outer-margin silence (both ends of the merged WAV), sourced from the
    global VAD edge knob. Mirrors `_global_silence_ms()` so the edge default
    lives in one place instead of being repeated at each merge/preview site."""
    try:
        return int(getattr(cfg, "CAPTURES_VAD_MARGIN_SAMPLE_EDGE_MS", 300))
    except (TypeError, ValueError):
        return 300


def _apply_chips_to_text(text: str, corrections: list[dict[str, Any]]) -> str:
    """Substitute each chip's `wrong` text with its `correct` text in
    `text`. Walk in idx order so multi-word spans replace as a unit.
    If `wrong` isn't found verbatim (whitespace drift / regex specials),
    that chip is left alone. Mirrored byte-for-byte by the JS twin
    `_applyChipsToText` so server- and client-derived transcripts agree."""
    if not corrections:
        return text or ""
    out = text or ""
    def _sk(c):
        i = c.get("idx")
        try:
            return (0, int(i))
        except (TypeError, ValueError):
            return (1, 0)
    ordered = sorted(
        (c for c in corrections if isinstance(c, dict)), key=_sk,
    )
    for c in ordered:
        wrong = c.get("wrong") or ""
        correct = c.get("correct") or ""
        if not wrong or not correct:
            continue
        i = out.find(wrong)
        if i >= 0:
            out = out[:i] + correct + out[i + len(wrong):]
    return out


def _build_default_transcript(members: list[dict[str, Any]], strategy: str) -> str:
    """Concatenate member transcripts with chips applied. Each member's
    training-form text (`text_for_training`) gets its chip corrections
    layered on top before the join, so the merged result reflects what
    the export pipeline will actually produce. Falls back through `final`
    then `raw` for legacy members predating the `text_for_training`
    column."""
    parts: list[str] = []
    for m in members:
        base = m.get("text_for_training") or m.get("final") or m.get("raw") or ""
        t = _apply_chips_to_text(base, m.get("corrections") or []).strip()
        if t:
            parts.append(t)
    return _JOIN_STR.get(strategy, " ").join(parts)


def _build_merged_wav(
    *,
    sid: str,
    member_ids: list[str],
    silence_ms: int,
    member_paths: list[str] | None = None,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    """Resolve member audio paths (or accept pre-resolved ones), run the
    merge, return (duration_ms, member_hash_map, member_trims). When
    `member_paths` is None, looks them up via captures_store + validates each
    file exists. Caller must have validated member_ids belong to the same
    user and the total is within the configured cap.

    `member_trims` maps member_id → {lead_ms, new_duration_ms, segments} when
    CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES trims each member; _build_merged_words
    uses it to keep per-member karaoke timestamps in sync with the trimmed
    audio. Empty/identity when trimming is disabled or VAD is unavailable."""
    from faster_whisper_backend.captures import merge as audio_merge
    from faster_whisper_backend.captures import samples_store as capture_samples_store

    if member_paths is None:
        member_paths = []
        for mid in member_ids:
            cap = captures_store.get_capture(mid)
            if cap is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, f"capture {mid} not found",
                )
            abs_p = captures_store.abs_audio_path(cap["audio_relpath"])
            if not os.path.exists(abs_p):
                raise HTTPException(
                    status.HTTP_410_GONE, f"capture {mid} audio is missing",
                )
            member_paths.append(abs_p)

    hashes: dict[str, str] = {}
    for mid, abs_p in zip(member_ids, member_paths):
        hashes[mid] = audio_merge.hash_wav_pcm(abs_p)

    dst_relpath = capture_samples_store._relpath_for(sid)
    dst_abs = capture_samples_store.abs_path_for(dst_relpath)
    try:
        res = audio_merge.merge_wavs(
            member_paths, dst_abs, gap_ms=silence_ms,
            trim=bool(getattr(cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES", False)),
            edge_pad_ms=_global_edge_ms(),
            max_internal_gap_ms=_global_silence_ms(),
        )
    except audio_merge.WavFormatError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    duration_ms = int(res["duration_ms"])
    # Re-key the order-parallel per-member trims onto member ids so
    # _build_merged_words can look each member up regardless of ordering.
    member_trims = {
        mid: res["members"][i]
        for i, mid in enumerate(member_ids)
        if i < len(res["members"])
    }
    return duration_ms, hashes, member_trims


def _merged_wav_patch(
    duration_ms: int, hashes: dict[str, str],
    member_trims: dict[str, Any],
) -> dict[str, Any]:
    """Build the capture_samples patch fields from _build_merged_wav outputs.

    merged_lead/trail_trim_ms are forced to 0 under per-member trimming (the
    outer-edge offsets are now folded into the per-member segment maps); they
    survive only so legacy groups without member_trims keep rendering."""
    return {
        "merged_duration_ms":   int(duration_ms),
        "member_hashes":        json.dumps(hashes, sort_keys=True),
        "merged_lead_trim_ms":  0,
        "merged_trail_trim_ms": 0,
        "member_trims":         json.dumps(member_trims, sort_keys=True),
        "is_stale":             0,
    }


_ALIGN_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def _align_key(s: str) -> str:
    """Normalise a token for LCS comparison: strip surrounding whitespace
    AND leading/trailing non-word punctuation, then casefold. Internal
    punctuation (apostrophes, internal hyphens) is preserved so "don't"
    or "Sciene-fiction" still compare faithfully.

    Stripping edge punctuation is necessary because the inference pipeline
    (callback:map: "Komma" → ",", "Punkt" → ".") attaches the symbol to
    the preceding token when joining; without normalisation, "existiert"
    (raw) and "existiert," (final after Komma→, glue) won't LCS-match,
    falsely flagging the raw word as removed-by-rule in the corrections
    word-strip.

    The visual diff signal (rule changed the word) is still surfaced via
    `item["raw_word"]` when display != raw (see the raw_word branch in
    _align_words_to_final) — the user sees the dotted-underline + tooltip without
    the misleading strike-through."""
    s = (s or "").strip()
    s = _ALIGN_PUNCT_RE.sub("", s)
    return s.casefold()


def _align_words_to_final(
    words: list[dict[str, Any]],
    final: str,
    model_name: "str | None" = None,
    ident=None,
    language: "str | None" = None,
) -> list[dict[str, Any]]:
    """Project raw STT words onto post-pipeline `final` via LCS
    alignment. Replaces the all-or-nothing fallback that came before
    — most rule output IS faithfully attributable per word even when
    a cross-word rule (dedup, "Neue Zeile → \\n", etc.) also fires.

    Output items (one per raw word) carry:
      - `word`: the final token(s) attributed to this raw word — the
        display text for the karaoke band + the chip's `wrong` reference.
        A rule that EXPANDS one raw word into several tokens (e.g.
        "Nurtax" → "nur tags") yields the joined run ("nur tags") so the
        band reconstructs `final` losslessly instead of dropping the
        extra tokens.
      - `raw_word`: present when display != raw — powers the dotted
        underline + `title="raw: …"` tooltip
      - `removed`: True only when this raw word ends up owning ZERO final
        tokens (a cross-word rule deleted it, or it's the dropped side of
        a contraction). The UI fades + strikes-through these slots; chip
        creation is suppressed.

    Every final token is assigned to exactly one raw word, so
    `" ".join(item["word"] for non-removed items) == final`. Inserted
    tokens (no direct raw correspondent) attach to the raw word that
    expanded into them — sharing that word's audio timestamp — rather
    than being discarded.
    """
    src = list(words or [])
    if not src:
        return []
    # Memo: many captures repeat the same raw token (filler words, punctuation
    # carriers). Without the cache, _postprocess_text runs O(N) times per
    # caller and dominates karaoke-band assembly when /captures expands a
    # group with hundreds of words.
    post_cache: dict[str, str] = {}
    raw_keys: list[str] = []
    for w in src:
        raw_w = w.get("word") or ""
        post_w = post_cache.get(raw_w)
        if post_w is None:
            try:
                post_w = pl_engine._postprocess_text(raw_w, model_name=model_name, ident=ident, language=language)
            except Exception:
                post_w = raw_w
            post_cache[raw_w] = post_w
        raw_keys.append(_align_key(post_w) or _align_key(raw_w))

    fin_tokens = (final or "").split()
    fin_keys = [_align_key(t) for t in fin_tokens]

    n, m = len(raw_keys), len(fin_keys)
    matches: list[int] = [-1] * n
    if n and m:
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if raw_keys[i] and raw_keys[i] == fin_keys[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        i = j = 0
        while i < n and j < m:
            if raw_keys[i] and raw_keys[i] == fin_keys[j]:
                matches[i] = j
                i += 1
                j += 1
            elif dp[i + 1][j] >= dp[i][j + 1]:
                i += 1
            else:
                j += 1

    # Assign every final token to exactly one raw word so the per-word band
    # reconstructs `final` losslessly — even when a rule expands one raw word
    # into several tokens ("Nurtax" → "nur tags") or contracts several into one.
    # The matched raw words are monotonic anchors that split `fin_tokens` into
    # segments; the inserted (unmatched) tokens in a segment go to the unmatched
    # raw words sharing that segment, apportioned by each raw word's isolated
    # post-processed token count, with any remainder to the last one.
    owners: list[list[int]] = [[] for _ in range(n)]
    exp_count = [len((post_cache.get(w.get("word") or "") or "").split()) for w in src]
    anchors = [(idx, matches[idx]) for idx in range(n) if matches[idx] >= 0]
    seg_bounds = [(-1, -1)] + anchors + [(n, m)]
    for t in range(len(seg_bounds) - 1):
        ri0, fj0 = seg_bounds[t]
        ri1, fj1 = seg_bounds[t + 1]
        if ri0 >= 0:
            owners[ri0].append(fj0)  # the left anchor owns its matched token
        ins = list(range(fj0 + 1, fj1))       # inserted final tokens in this gap
        if not ins:
            continue
        unmatched = list(range(ri0 + 1, ri1))  # raw words sharing this gap
        if unmatched:
            k = 0
            for ri in unmatched:
                take = exp_count[ri]
                while take > 0 and k < len(ins):
                    owners[ri].append(ins[k]); k += 1; take -= 1
            last = unmatched[-1]
            while k < len(ins):  # leftover insertions → last raw word in the gap
                owners[last].append(ins[k]); k += 1
        else:
            # Pure insertion with no raw word in the gap: attach to an adjacent
            # anchor (it shares that word's audio time). Prefer the left anchor.
            target = ri0 if ri0 >= 0 else (ri1 if ri1 < n else None)
            if target is not None:
                owners[target].extend(ins)

    out: list[dict[str, Any]] = []
    for i, w in enumerate(src):
        item = _clone_word(w)
        raw_w = w.get("word") or ""
        toks = owners[i]
        if toks:
            disp = " ".join(fin_tokens[j] for j in toks)
            item["word"] = disp
            if disp.strip() != raw_w.strip():
                item["raw_word"] = raw_w
        else:
            item["word"] = raw_w
            item["raw_word"] = raw_w
            item["removed"] = True
        out.append(item)
    return out


def _clone_word(w: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of a words item with only the keys
    we care about preserved (word / start / end). Other fields
    (probability, etc.) are dropped — the UI doesn't use them and
    they bloat the JSON payload."""
    return {
        "word":  w.get("word") or "",
        "start": w.get("start"),
        "end":   w.get("end", w.get("start")),
    }



# Per-sid lock so a burst of audio requests for the same missing WAV
# at startup doesn't trigger N concurrent merges. The merge is dispatched
# with asyncio.to_thread and the VAD reprocess worker is a real thread, so
# a plain threading.Lock is the right primitive — but callers must never
# acquire it from the event loop itself (see the callers in captures/routes.py).
_rebuild_locks: dict[str, threading.Lock] = {}
_REBUILD_LOCKS_MAX = 512
_rebuild_locks_guard = threading.Lock()
# Sids whose Lock has been handed out but not necessarily acquired yet
# (value = number of such callers). `v.locked()` alone can't protect that
# window: the prune below would drop the entry and a later caller would mint
# a SECOND Lock for the same sid, letting two rebuilds run concurrently.
_rebuild_inflight: dict[str, int] = {}


def _get_rebuild_lock(sid: str) -> threading.Lock:
    with _rebuild_locks_guard:
        lock = _rebuild_locks.get(sid)
        if lock is None:
            # Opportunistic prune before growing. Sample ids are random and a
            # dissolved sample's id never returns, so without this the dict is
            # an unbounded create/regenerate/dissolve leak. Only idle locks
            # that are not in flight (handed out but not yet acquired) are
            # dropped, so a rebuild in flight is never orphaned.
            if len(_rebuild_locks) >= _REBUILD_LOCKS_MAX:
                for k in [k for k, v in _rebuild_locks.items()
                          if not v.locked() and k not in _rebuild_inflight]:
                    del _rebuild_locks[k]
            lock = threading.Lock()
            _rebuild_locks[sid] = lock
        return lock


@contextlib.contextmanager
def _rebuild_lock(sid: str):
    """Hold the per-sid rebuild lock, keeping the sid pinned against the
    prune in _get_rebuild_lock for the whole handout-to-release span."""
    with _rebuild_locks_guard:
        _rebuild_inflight[sid] = _rebuild_inflight.get(sid, 0) + 1
    try:
        with _get_rebuild_lock(sid):
            yield
    finally:
        with _rebuild_locks_guard:
            n = _rebuild_inflight.get(sid, 1) - 1
            if n <= 0:
                _rebuild_inflight.pop(sid, None)
            else:
                _rebuild_inflight[sid] = n


def _release_rebuild_lock(sid: str) -> None:
    """Drop a sample's lock entry once the sample itself is gone."""
    with _rebuild_locks_guard:
        if sid in _rebuild_inflight:
            return
        lock = _rebuild_locks.get(sid)
        if lock is not None and not lock.locked():
            del _rebuild_locks[sid]


def _reset_for_tests() -> None:
    """Drop every per-sid rebuild lock and in-flight pin, and rebind the
    guard (tests/conftest.py _RESET_HOOKS)."""
    global _rebuild_locks_guard
    _rebuild_locks.clear()
    _rebuild_inflight.clear()
    _rebuild_locks_guard = threading.Lock()
