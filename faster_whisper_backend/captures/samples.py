"""Sample building for /captures: the merged-WAV build, the default
transcript join, the raw-word → final-text alignment behind the karaoke band,
and the per-sample rebuild lock. Shared by the captures routes and the two
background workers (reapply, vad_reprocess), which call through this module's
attributes so one test patch reaches every caller. Imports neither
captures.routes nor the workers.
"""
from __future__ import annotations

import bisect
import contextlib
import difflib
import json
import os
import re
import threading
import unicodedata
from typing import Any

from fastapi import HTTPException, status

from faster_whisper_backend.captures import store as captures_store
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config as cfg


# Sentence-final punctuation (before any closing quotes/brackets): a member
# ending in one gets no extra "." under the period_space join. Covers German
# „…“ / ‚…‘ quotes and CJK / Arabic / Devanagari / Burmese / Khmer / Urdu ۔ /
# Armenian ։ / Amharic ። enders; the JS twin _endsSentence in captures.html
# must keep the same sets.
_SENTENCE_END = ".?!…。！？؟।॥။។៕۔։።‼⁇⁈⁉"
_SENTENCE_CLOSERS = "\"'”’»«“‘›‹)]}」』）］｝》〉"


def _ends_sentence(text: str) -> bool:
    # Spaces too: French typography puts one before the » ('Oui. »').
    tail = text.rstrip()
    while tail and tail[-1] in _SENTENCE_CLOSERS:
        tail = tail[:-1].rstrip()
    return bool(tail) and tail[-1] in _SENTENCE_END


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


# Scripts written without spaces between words: Thai, Lao, Myanmar, Khmer,
# Hiragana, Katakana and Han (the ideographic iteration / closing marks
# 々〆〇 included). The JS twins in captures.html (_applyChipsToText,
# _renderGroundSpans) must keep the same ranges.
_NO_SPACE_SCRIPT = re.compile(
    "[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u19e0-\u19ff"
    "\u3005-\u3007\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    "\ua9e0-\ua9ff\uaa60-\uaa7f\uf900-\ufaff\uff66-\uff9f"
    "\U00020000-\U0003134f]")
# The rest of the token from a position on (up to the next whitespace).
_TOKEN_REST = re.compile(r"\S*")


# No word starts with a Thai / Lao following vowel or repetition mark, a
# Japanese small kana, iteration mark or "ー", and none ends with a Thai /
# Lao leading vowel. Same sets in the JS twin _applyChipsToText.
_NO_WORD_START = ("\u0e30\u0e32\u0e33\u0e45\u0e46\u0eb0\u0eb2\u0eb3\u0ec6"
                  "\u3005\u3041\u3043\u3045\u3047\u3049\u3063\u3083\u3085"
                  "\u3087\u308e\u3095\u3096\u309d\u309e\u30a1\u30a3\u30a5"
                  "\u30a7\u30a9\u30c3\u30e3\u30e5\u30e7\u30ee\u30f5\u30f6"
                  "\u30fc\u30fd\u30fe\uff67\uff68\uff69\uff6a\uff6b\uff6c"
                  "\uff6d\uff6e\uff6f\uff70")
_NO_WORD_END = "\u0e40\u0e41\u0e42\u0e43\u0e44\u0ec0\u0ec1\u0ec2\u0ec3\u0ec4"


def _word_char(ch: str) -> bool:
    """A character that belongs to the word it sits in. A combining mark (a
    Devanagari or Thai vowel sign, an NFD accent) and ZWNJ/ZWJ do: "लत" is
    no whole word inside "बोलते", and "की" is not "क" with a stray sign."""
    return (ch.isalnum() or ch == "_" or ch in "\u200c\u200d"
            or unicodedata.category(ch)[0] == "M")


def _word_edge(a: str, b: str) -> bool:
    """Whether one word may end on `a` and the next begin on `b`. Two word
    characters belong to one word, except next to a script written without
    spaces (zh, ja, th, …), where two words meet with no space between them
    — but a letter never parts from the combining mark after it."""
    if not (_word_char(a) and _word_char(b)):
        return True
    return (bool(_NO_SPACE_SCRIPT.match(a) or _NO_SPACE_SCRIPT.match(b))
            and unicodedata.category(b)[0] != "M"
            and b not in _NO_WORD_START and a not in _NO_WORD_END)


def _apply_chips_to_text(
    text: str,
    corrections: list[dict[str, Any]],
    words: list[dict[str, Any]] | None = None,
) -> str:
    """Substitute each chip's `wrong` text with its `correct` text in
    `text`. Walk in idx order so multi-word spans replace as a unit.
    A verbatim match counts, and so does one where any whitespace run in
    `wrong` matches any whitespace run in the text (a multi-word chip's
    `wrong` joins its words with spaces, the text may have a dictated line
    break there); an anchorless chip takes the first occurrence, verbatim
    before flexible. If `wrong` still isn't found, a chip is left alone.
    Mirrored byte-for-byte by the JS twin `_applyChipsToText` so server- and
    client-derived transcripts agree.

    Only a whole-word occurrence counts: a chip on "cat" must not rewrite
    the "cat" inside "scat" (in a script written without spaces a word may
    sit between two letters, see _word_edge). An anchored chip searches past
    the previous anchored replacement (a `correct` that contains `wrong` is
    never hit twice) and, of several occurrences, takes the one whose word
    position is nearest its `idx` (in such a script, its char offset inside
    the run breaks a tie) — a chip on the second "die" must not rewrite the
    first. That nearest pick runs over the verbatim and the whitespace-
    flexible passes together. `idx` counts karaoke words (one per raw word,
    removed ones included), not text tokens. `words` — the aligned karaoke
    items the chips were made on — maps idx onto the text's token position:
    a filler or a dictated "Komma" owns no token, so without it the position
    is only a nearest match and drifts past such words. With `words`, an
    anchored chip is also matched by the words at idx..idx_end, the way the
    page re-derives `wrong` on display (a tie with the stored `wrong` keeps
    those words, which the page replaces, unless that `wrong` covers them):
    a chip saved before those words kept their spaces ("HalloWelt"), or on a
    form the pipeline has since changed, shows on the page at its idx and
    must be applied there too, not on another occurrence of the stale
    `wrong`."""
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
    # Text-token position of each karaoke word and its char offset inside
    # that token (`offs`; non-zero for a word inside a run of a script
    # written without spaces). The aligned words (training token when one is
    # attached) appear in the text in order, so each non-removed word is
    # located in it with a moving cursor: whether it starts a token comes
    # from the text, not from its own lead (a rule may put a line break or
    # a space between two words that had none). A removed word, or one not
    # found, sits at the token after the cursor.
    tok_starts = [m.start() for m in re.finditer(r"\S+", out)]
    starts: list[int] | None = None
    offs: list[int] = []
    if words:
        starts, at = [], 0
        for w in words:
            parts: list[str] = []
            if isinstance(w, dict):
                if w.get("train_word") is not None:
                    tok, gone = w.get("train_word") or "", w.get("train_removed")
                else:
                    tok, gone = w.get("word") or "", w.get("removed")
                if not gone:
                    parts = tok.split()
            p = out.find(parts[0], at) if parts else -1
            if p < 0:
                starts.append(bisect.bisect_left(tok_starts, at))
                offs.append(0)
                continue
            k = bisect.bisect_right(tok_starts, p) - 1
            starts.append(k)
            offs.append(p - tok_starts[k])
            at = p + len(parts[0])
            for part in parts[1:]:
                q = out.find(part, at)
                if q >= 0:
                    at = q + len(part)
        starts.append(bisect.bisect_left(tok_starts, at))
        offs.append(0)
    cursor = 0
    # Word- and char-count change of the anchored replacements so far (all
    # before the cursor), so a later chip's token position and its char
    # offset in that token are compared in the original text's frame, also
    # after a `correct` that put a space inside an unspaced run.
    shift = 0
    chars = 0
    for c in ordered:
        wrong = c.get("wrong") or ""
        correct = c.get("correct") or ""
        if not wrong or not correct:
            continue
        rank, idx = _sk(c)
        anchored = rank == 0
        pos, off = idx, 0
        if anchored and starts is not None:
            pos = starts[max(0, min(idx, len(starts) - 1))]
            off = offs[max(0, min(idx, len(offs) - 1))]
        needles = [wrong]
        if anchored and words:
            healed = _span_text(words, idx, c.get("idx_end"))
            if healed and healed != wrong:
                needles.insert(0, healed)
        # An anchored chip takes the occurrence nearest its idx over every
        # pass: the page strikes the words at idx, so a far match of the
        # stale `wrong` must not beat the words there. On a tie an
        # occurrence that contains the other wins, else the earlier pass:
        # the healed words win over a stale `wrong` that is a part of them,
        # and a stale `wrong` that covers them (a chip saved on a whole
        # unspaced sentence when its first word owned all of it) wins over
        # the words, so the corrected sentence does not replace one char.
        best, best_end, best_d = -1, -1, (0, 0)
        for needle, flex in ((n, f) for n in needles for f in (False, True)):
            for i, j in _occurrences(out, needle, cursor if anchored else 0,
                                     flex):
                if ((i > 0 and not _word_edge(out[i - 1], out[i]))
                        or (j < len(out) and not _word_edge(out[j - 1], out[j]))):
                    continue
                if not anchored:
                    best, best_end = i, j
                    break
                # Whole tokens before the occurrence: a "(" or „ it starts
                # inside belongs to its own word, not the one before. `ts`
                # is the start of the token it sits in.
                before = out[:i].split()
                k, ts = len(before), i
                if i > 0 and not out[i - 1].isspace():
                    k -= 1
                    ts = i - len(before[-1])
                # Inside a token of a script written without spaces, the
                # char offset in that token breaks a tie: a chip on the
                # third "的" of "我的猫的狗的" must not rewrite the first,
                # nor one on the second "Python" of "Python很好，Python很强"
                # (the whole token counts, not just up to the occurrence).
                te = _TOKEN_REST.match(out, j).end()
                d2 = 0
                if _NO_SPACE_SCRIPT.search(out, ts, te):
                    k0 = max(0, min(k + shift, len(tok_starts) - 1))
                    d2 = abs(i - chars - tok_starts[k0] - off)
                d = (abs(k + shift - pos), d2)
                if (best < 0 or d < best_d
                        or (d == best_d and i <= best and j >= best_end
                            and j - i > best_end - best)):
                    best, best_end, best_d = i, j, d
            if best >= 0 and not anchored:
                break
        if best < 0:
            continue
        hit = out[best:best_end]
        out = out[:best] + correct + out[best_end:]
        if anchored:
            cursor = best + len(correct)
            shift += len(hit.split()) - len(correct.split())
            chars += len(correct) - len(hit)
    return out


def _occurrences(text: str, needle: str, start: int, flex: bool):
    """(start, end) of each occurrence of `needle` in `text` from `start`,
    overlapping ones included. `flex` lets any whitespace run in `needle`
    match any whitespace run in `text`."""
    if not flex:
        i = text.find(needle, start)
        while i >= 0:
            yield i, i + len(needle)
            i = text.find(needle, i + 1)
        return
    if not needle.split():
        return
    rx = re.compile(r"\s+".join(map(re.escape, needle.split())))
    m = rx.search(text, start)
    while m:
        yield m.start(), m.end()
        m = rx.search(text, m.start() + 1)


def _span_text(words: list[dict[str, Any]], idx: int, idx_end: Any) -> str:
    """The text of the aligned karaoke words idx..idx_end (a removed word
    owns none), joined the way the page builds a chip's `wrong`: each token
    carries its own leading space. The training token wins when attached."""
    try:
        end = int(idx_end) if idx_end is not None else idx
    except (TypeError, ValueError):
        end = idx
    parts = []
    for w in words[max(0, idx):max(0, end) + 1]:
        if not isinstance(w, dict):
            continue
        if w.get("train_word") is not None:
            tok, gone = w.get("train_word") or "", w.get("train_removed")
        else:
            tok, gone = w.get("word") or "", w.get("removed")
        if not gone:
            parts.append(tok)
    return "".join(parts).lstrip()


def _build_default_transcript(
    members: list[dict[str, Any]],
    strategy: str,
    words_for=None,
) -> str:
    """Concatenate member transcripts with chips applied. Each member's
    training-form text (`text_for_training`) gets its chip corrections
    layered on top before the join, so the merged result reflects what
    the export pipeline will actually produce. Falls back through `final`
    then `raw` for legacy members predating the `text_for_training`
    column.

    `words_for(member)` (optional) returns that member's aligned karaoke
    words, so each chip lands on the word its idx marks; it is only called
    for a member that has corrections."""
    parts: list[str] = []
    for m in members:
        base = m.get("text_for_training") or m.get("final") or m.get("raw") or ""
        chips = m.get("corrections") or []
        words = words_for(m) if (words_for is not None and chips) else None
        t = _apply_chips_to_text(base, chips, words).strip()
        if t:
            parts.append(t)
    if strategy == "period_space":
        # ". " between members, unless the member already ends a sentence
        # ("Hallo Welt." must not become "Hallo Welt.. Wie …").
        parts = [p if _ends_sentence(p) else p + "." for p in parts[:-1]] + parts[-1:]
    return " ".join(parts)


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
        # The light projection, not get_capture: only audio_relpath is read
        # here, and this runs under the per-sid rebuild lock — a full row
        # json.loads the words + segments blobs per member.
        light = captures_store.get_captures_light(member_ids)
        for mid in member_ids:
            cap = light.get(mid)
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
    i, j = 0, len(s)
    while i < j and not _word_char(s[i]):
        i += 1
    while j > i and not _word_char(s[j - 1]):
        j -= 1
    return s[i:j].casefold()


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
    than being discarded. A script written without spaces (zh, ja, th, lo,
    my, km) puts many raw words in one final token; there the token is
    split among them character by character (see _split_unspaced), so the
    words' texts concatenate to that token instead.
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
    owners: list[list[str]] = [[] for _ in range(n)]
    exp_count = [len((post_cache.get(w.get("word") or "") or "").split()) for w in src]
    anchors = [(idx, matches[idx]) for idx in range(n) if matches[idx] >= 0]
    seg_bounds = [(-1, -1)] + anchors + [(n, m)]
    for t in range(len(seg_bounds) - 1):
        ri0, fj0 = seg_bounds[t]
        ri1, fj1 = seg_bounds[t + 1]
        if ri0 >= 0:
            owners[ri0].append(fin_tokens[fj0])  # the left anchor owns its matched token
        ins = fin_tokens[fj0 + 1:fj1]          # inserted final tokens in this gap
        if not ins:
            continue
        unmatched = list(range(ri0 + 1, ri1))  # raw words sharing this gap
        split = None
        if unmatched and any(_NO_SPACE_SCRIPT.search(t) for t in ins):
            split = _split_unspaced(
                ["".join((post_cache.get(src[ri].get("word") or "") or "").split())
                 for ri in unmatched], ins)
        if split is not None:
            for ri, parts in zip(unmatched, split):
                owners[ri].extend(parts)
        elif unmatched:
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
            disp = " ".join(toks)
            item["word"] = disp
            if disp.strip() != raw_w.strip():
                item["raw_word"] = raw_w
        else:
            item["word"] = raw_w
            item["raw_word"] = raw_w
            item["removed"] = True
        out.append(item)
    return out


def _split_unspaced(texts: list[str], toks: list[str]) -> list[list[str]] | None:
    """Split the final tokens `toks` of one alignment gap among the gap's raw
    words, whose isolated post-processed forms (whitespace removed) are
    `texts`, character by character. In a script written without spaces the
    whole sentence is one final token: handed to one raw word, it struck
    every other word as removed by a rule although none fired. A final
    character matched to a raw word's character goes to that word; the chars
    that replace raw chars (a substitution rule: 他 → 她) go to the words
    whose chars they replace, in proportion, the remainder to the last; an
    inserted one joins the word before it, or the first word. Returns each
    raw word's pieces (one per token it reaches, empty when a rule deleted
    it), or None when no character matches."""
    a_own = [r for r, t in enumerate(texts) for _ in t]
    tok_of = [k for k, t in enumerate(toks) for _ in t]
    b = "".join(toks)
    own = [-1] * len(b)
    sm = difflib.SequenceMatcher(None, "".join(texts), b, autojunk=False)
    matched = False
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal":
            matched = True
            own[b0:b1] = a_own[a0:a1]
        elif tag == "replace":
            # Each raw word's share of the replaced chars, by how many of
            # them it owns; the last takes the rounding remainder.
            counts: dict[int, int] = {}
            for r in a_own[a0:a1]:
                counts[r] = counts.get(r, 0) + 1
            x = b0
            for n, (r, c) in enumerate(counts.items()):
                take = b1 - x if n == len(counts) - 1 else (b1 - b0) * c // (a1 - a0)
                own[x:x + take] = [r] * take
                x += take
    if not matched:
        return None
    cur = 0
    pieces: list[list[str]] = [[] for _ in texts]
    prev = None
    for x, ch in enumerate(b):
        if own[x] >= 0:
            cur = own[x]
        if (cur, tok_of[x]) == prev:
            pieces[cur][-1] += ch
        else:
            pieces[cur].append(ch)
            prev = (cur, tok_of[x])
    return pieces


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
