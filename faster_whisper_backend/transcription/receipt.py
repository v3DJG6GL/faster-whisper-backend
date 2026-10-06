"""The per-request receipt: the fixed-width log block every transcription /
translation request ends with (_format_request_block and its section
helpers), the stage-row helpers the routes feed it (_stage_extras,
_failed_stage, _stage_ran / _stage_field), and the held-receipt release
path for dictation receipts that wait on a separate translate request
(_log_held_receipts, _release_held_receipt, _receipt_sweeper). Callers go
through the module attribute (``tx_receipt._format_request_block(...)``) so a
test patching it here reaches every caller.
"""
import asyncio
import logging
import re
import time

from faster_whisper_backend.core import receipt_hold
from faster_whisper_backend.core import store_common
from faster_whisper_backend.runtime import system_stats
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.stats import metrics

logger = logging.getLogger("whisper-api")


# =============================================================================
# Per-request log block
# =============================================================================
# Always emitted (regardless of cfg.TRACE_ENABLED) — surfaces the decode
# params actually applied + per-segment metadata so empty-output failures
# can be diagnosed from the log alone. The per-pipeline transformation
# trace is folded in only when TRACE_ENABLED.
#
# ANSI color is intentionally dropped: the service runs under WinSW (no TTY)
# and the SSE log viewer reads raw bytes — escape codes hurt both consumers.
_LOG_WIDTH = 78
_NAME_COL = 32        # value column starts at this character
_SEG_TEXT_MAX = 80    # truncate per-segment text in the table (full text in FINAL)

# "don't render this row" sentinel for the stage params sections. Distinct
# from None, which is a real value there and prints as `(none)` — an unset
# num_speakers genuinely means "auto" and is worth showing.
_OMIT = object()

# Single implementation lives in store_common so the stores can sanitise their
# own audit lines without importing main (which imports them).
_log_safe = store_common.log_safe


# Maps decode-kwarg name → cfg-default key in cfg._BASELINE. Used by the
# `*` non-default marker. Only scalar fields are listed; lists/dicts skipped.
# `temperature` and `suppress_tokens` are intentionally absent — their cfg
# baselines are strings ("0.0,0.2,…", "-1") while the kwargs are tuples/lists,
# so equality comparison is meaningless without parsing both sides.
_KWARG_TO_CFG = {
    # Task
    "task": "TASK",
    # Search / sampling
    "beam_size": "BEAM_SIZE",
    "best_of": "BEST_OF",
    "patience": "PATIENCE",
    "length_penalty": "LENGTH_PENALTY",
    "repetition_penalty": "REPETITION_PENALTY",
    "no_repeat_ngram_size": "NO_REPEAT_NGRAM_SIZE",
    "prompt_reset_on_temperature": "PROMPT_RESET_ON_TEMPERATURE",
    # VAD
    "vad_filter": "VAD_FILTER",
    "min_silence_duration_ms": "VAD_MIN_SILENCE_MS",
    "speech_pad_ms": "VAD_SPEECH_PAD_MS",
    "threshold": "VAD_THRESHOLD",
    # Output shape
    "word_timestamps": "WORD_TIMESTAMPS_ENABLED",
    # Prompt context
    "condition_on_previous_text": "CONDITION_ON_PREVIOUS_TEXT",
    "initial_prompt": "DEFAULT_PROMPT",
    "hotwords": "DEFAULT_HOTWORDS",
    # Safety / thresholds
    "no_speech_threshold": "NO_SPEECH_THRESHOLD",
    "log_prob_threshold": "LOG_PROB_THRESHOLD",
    "compression_ratio_threshold": "COMPRESSION_RATIO_THRESHOLD",
    "hallucination_silence_threshold": "HALLUCINATION_SILENCE_THRESHOLD",
    # Language detection
    "multilingual": "MULTILINGUAL",
    "language_detection_threshold": "LANGUAGE_DETECTION_THRESHOLD",
    "language_detection_segments": "LANGUAGE_DETECTION_SEGMENTS",
    # Token suppression / punctuation
    "suppress_blank": "SUPPRESS_BLANK",
    "prepend_punctuations": "PREPEND_PUNCTUATIONS",
    "append_punctuations": "APPEND_PUNCTUATIONS",
    # Post-decode guards (pseudo-kwargs: rendered in the log block's guards
    # section, never passed to model.transcribe)
    "segment_max_words_per_sec": "SEGMENT_MAX_WORDS_PER_S",
    "segment_max_word_burst_per_sec": "SEGMENT_MAX_WORD_BURST_PER_S",
    "segment_zero_length_tail_min_words": "SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS",
    "segment_repeat_collapse_min_repeats": "SEGMENT_REPEAT_COLLAPSE_MIN_REPEATS",
    "segment_head_echo_min_words": "SEGMENT_HEAD_ECHO_MIN_WORDS",
    "skip_residual_windows": "DECODE_SKIP_RESIDUAL_WINDOWS",
    "token_cap_per_second": "DECODE_TOKEN_CAP_PER_SECOND",
    "tail_trim_pad_ms": "STREAMING_TAIL_TRIM_PAD_MS",
    "final_drop_min_avg_logprob": "STREAMING_FINAL_DROP_MIN_AVG_LOGPROB",
    "final_drop_temperature": "STREAMING_FINAL_DROP_TEMPERATURE",
    # Post-decode STAGE params (pseudo-kwargs too: rendered in the block's
    # Separation / Diarization / Translation sections). Listing them here is
    # the whole wiring the `*` marker needs — _is_non_default is a whitelist
    # keyed by this dict, so without an entry a stage param can never be
    # marked non-default no matter how far it strays from the config.
    "num_speakers": "DIARIZATION_NUM_SPEAKERS",
    "min_speakers": "DIARIZATION_MIN_SPEAKERS",
    "max_speakers": "DIARIZATION_MAX_SPEAKERS",
    "embedding_batch_size": "DIARIZATION_EMBEDDING_BATCH_SIZE",
    "diarization_model": "DIARIZATION_MODEL",
    "separation_model": "BGM_SEPARATION_UVR_MODEL",
    "translation_model": "TRANSLATION_DEFAULT_MODEL",
    "mode": "TRANSLATION_MODE",
    "context_segments": "TRANSLATION_CONTEXT_SEGMENTS",
    "batch_segments": "TRANSLATION_BATCH_SEGMENTS",
}


# Canonical pipeline order for the Pipeline table. Stage names are the ones
# _stage_timings already uses, so the table is a straight render of that list
# rather than a second source of truth that can drift from it.
_STAGE_ORDER = ("downloading", "separating", "vad", "transcribing",
                "diarizing", "translating")


class PlainText:
    """Receipt value rendered verbatim (no repr quotes) — for composed rows
    such as `1.52s  (3.74s → 2.22s)` that are display text, not a config value."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = str(text)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"PlainText({self.text!r})"


def _pretty_value(v) -> str:
    """Compact display form for a config value: `true`/`false`, `(none)` for
    None, `(empty)` for "", trimmed-zero floats, repr'd strings."""
    if v is None:
        return "(none)"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        # Preserve at least one decimal so 0.0 / -1.0 / 0.5 still read as
        # floats (not as ints). Strip extra trailing zeros only.
        s = f"{v:.2f}"
        if "." in s and s.endswith("0"):
            s = s.rstrip("0")
            if s.endswith("."):
                s += "0"
        return s
    if isinstance(v, list):
        return "[" + ", ".join(_pretty_value(x) for x in v) + "]"
    if isinstance(v, str):
        if not v:
            return "(empty)"
        if len(v) > 60:
            return repr(v[:57] + "...")
        return repr(v)
    return str(v)


def _is_non_default(key: str, value) -> bool:
    """`*` marker test. True iff a known cfg-default exists and the current
    scalar value differs from it. Skips non-scalars to avoid surprises."""
    cfg_key = _KWARG_TO_CFG.get(key)
    if not cfg_key:
        return False
    baseline_dict = getattr(cfg, "_BASELINE", None)
    if baseline_dict is None:
        return False
    baseline = baseline_dict.get(cfg_key)
    scalar = (bool, int, float, str, type(None))
    if not isinstance(value, scalar) or not isinstance(baseline, scalar):
        return False
    if value is None and baseline == "":
        return False
    if value == "" and baseline is None:
        return False
    return value != baseline


def _param_row(indent: str, key: str, value) -> str:
    """`indent + key + spaces + value [*]` row. Value column lands at _NAME_COL
    regardless of indent depth so top-level and nested rows align."""
    star = " *" if _is_non_default(key, value) else ""
    pretty = _pretty_value(value)
    pad = max(1, _NAME_COL - len(indent) - len(key))
    return f"{indent}{key}{' ' * pad}{pretty}{star}"


def _section_rule(label: str) -> str:
    """`  ─── label ──────…` inner rule, padded to _LOG_WIDTH."""
    head = f"  ─── {label} "
    fill = max(0, _LOG_WIDTH - len(head))
    return head + ("─" * fill)


def _short_id(v):
    """A user / key id shortened for the receipt (8 chars; "(…)" markers kept)."""
    v = v or ""
    return v if v.startswith("(") else (v[:8] if v else "—")


def _format_translate_block(
    *,
    request_id: str | None,
    model_name: str | None,
    device: str | None,
    targets: list,
    source: str | None,
    mode: str | None,
    result: str,
    secs: float,
    load_secs: float,
    client_job: str | None = None,
    user_id: str | None = None,
    key_id: str | None = None,
    username: str | None = None,
    key_label: str | None = None,
) -> str:
    """Standalone receipt for a /v1/text/translations request that claimed
    no held dictation receipt — a stop-timing session's one-shot, a plain
    API caller, or a live client from before the handshake. The held case
    merges into the utterance's block instead (see receipt_hold); this is
    the fallback that keeps such a request from leaving only progress
    lines behind, with no model / targets / identity to tie it to a user."""
    title_rule = "═" * _LOG_WIDTH
    status = "✓ ok"
    if request_id:
        status = f"req={request_id[:8]}  {status}"
    title = "  /v1/text/translations"
    pad = max(1, _LOG_WIDTH - len(title) - len(status))
    lines: list[str] = ["", title_rule, f"{title}{' ' * pad}{status}", title_rule]
    model_line = f"  model  {model_name or '?'}"
    if device:
        model_line += f"   device={device}"
    lines.append(model_line)
    if client_job:
        # Same `job=` the session's utterance receipts carry on their file
        # line — grep for it and the whole session lines up.
        lines.append(f"  for    dictation job={client_job[:8]}")
    lines.append(_section_rule("Translation"))
    rows = [
        ("targets", ", ".join(targets) if targets else "—"),
        ("source_lang", (source or "").strip() or "auto"),
        ("mode", mode or "—"),
        ("result", result),
        ("wall", f"{secs:.1f}s" + (f"  (load {load_secs:.1f}s)" if load_secs > 0 else "")),
    ]
    for name, val in rows:
        lines.append(f"    {name:<{_NAME_COL - 4}}{val}")

    if user_id or key_id:
        lines.append(_section_rule("Identity"))
        _safe_name = _log_safe(username) if username else None
        who = f"{_safe_name} ({_short_id(user_id)})" if _safe_name else _short_id(user_id)
        lines.append(f"    {'user':<{_NAME_COL - 4}}{who}")
        if key_id:
            _safe_label = _log_safe(key_label) if key_label else None
            which = (f"{_safe_label} ({_short_id(key_id)})"
                     if _safe_label else _short_id(key_id))
            lines.append(f"    {'key':<{_NAME_COL - 4}}{which}")
    lines.append(title_rule)
    return "\n".join(lines)


def _format_decode_params(kwargs: dict) -> list[str]:
    """Render decode params as aligned rows, with VAD parameters indented
    under vad_filter to show the relationship visually. Fields are only
    printed when present in `kwargs` — most non-default knobs (patience,
    repetition_penalty, etc.) are conditionally added at request build
    time, so absence here means "at faster-whisper / config default".
    Order is grouped by intent (search → sampling → VAD → output → context
    → thresholds → language detection → suppression)."""
    out: list[str] = []
    order = (
        # Task (transcribe vs translate-to-English) — the most consequential
        # knob, so it leads
        "task",
        # Search / sampling
        "beam_size", "best_of", "patience", "length_penalty",
        "repetition_penalty", "no_repeat_ngram_size",
        "temperature", "prompt_reset_on_temperature",
        # VAD
        "vad_filter",
        # Output shape
        "word_timestamps",
        # Prompt context
        "condition_on_previous_text", "initial_prompt", "hotwords",
        # Safety / thresholds
        "no_speech_threshold", "log_prob_threshold",
        "compression_ratio_threshold", "hallucination_silence_threshold",
        # Language detection
        "multilingual", "language_detection_threshold",
        "language_detection_segments",
        # Token suppression / punctuation
        "suppress_blank", "suppress_tokens",
        "prepend_punctuations", "append_punctuations",
    )
    for k in order:
        if k not in kwargs:
            continue
        out.append(_param_row("    ", k, kwargs[k]))
        if k == "vad_filter" and kwargs[k] and kwargs.get("vad_parameters"):
            for vk, vv in kwargs["vad_parameters"].items():
                out.append(_param_row("      ", vk, vv))
    return out


def _short_speaker(label: str) -> str:
    """`SPEAKER_00` → `S0`. Keeps the segments table's new speaker column to
    the 5 characters the row budget can spare; anything unrecognised is just
    truncated."""
    m = re.search(r"(\d+)\s*$", label or "")
    return f"S{int(m.group(1))}" if m else (label or "")[:4]


# Windows shown in full when a decode has at most this many; a long file is
# summarised (totals + its slowest windows) so the receipt stays readable.
_TRACE_FULL_MAX_WINDOWS = 12
_TRACE_SLOWEST_SHOWN = 5


def _fmt_num(v, fmt="{:.2f}") -> str:
    return "-" if v is None else fmt.format(float(v))


def _format_decode_trace_section(trace: "dict | None") -> list[str]:
    """`Decode trace` section: one row per temperature rung, grouped by window.

    Header carries the totals that answer "where did the time go" at a glance:
    windows encoded, generate() calls, tokens generated, seconds inside
    generate(). Per rung: temperature, search (beamN / bestN), tokens, the
    window-level stats faster-whisper judged the rung by, wall time, and the
    outcome — `retry · <rule>` for a rung that failed the ladder, and for the
    last rung what became of the window (kept / skipped / no text)."""
    if not trace or not trace.get("windows"):
        return []
    n_w = trace.get("n_windows", len(trace["windows"]))
    head = (f"Decode trace  ({n_w} window{'s' if n_w != 1 else ''} · "
            f"{trace.get('n_rungs', 0)} generate call"
            f"{'s' if trace.get('n_rungs', 0) != 1 else ''} · "
            f"{trace.get('tokens', 0)} tokens · "
            f"{_fmt_secs(trace.get('generate_s'))} in generate")
    if trace.get("total_s") is not None:
        # Wall time of the whole decode: what is not "in generate" is
        # encoding, VAD and the segment bookkeeping between windows.
        head += f" · {_fmt_secs(trace['total_s'])} total"
    if trace.get("extra_encodes"):
        head += f" · +{trace['extra_encodes']} lang-detect encode"
    if trace.get("skipped_windows"):
        head += f" · {trace['skipped_windows']} residual skipped"
    head += ")"
    out = [_section_rule(head)]
    windows = list(trace["windows"])
    omitted = 0
    if len(windows) > _TRACE_FULL_MAX_WINDOWS:
        ranked = sorted(windows, key=lambda w: -(w.get("secs") or 0.0))
        keep = {id(w) for w in ranked[:_TRACE_SLOWEST_SHOWN]}
        shown = [w for w in windows if id(w) in keep]
        omitted = len(windows) - len(shown)
        windows = shown
        out.append(f"    slowest {len(shown)} of {n_w} windows shown")
    out.append(f"    {'w#':>3}  {'at':>7}  {'span':>6}  {'enc':>5}   "
               f"{'r#':>2}  {'T':>3}  {'search':<7}{'tokens':>6}  "
               f"{'alp':>6}  {'cr':>5}  {'nsp':>4}  {'secs':>6}  outcome")
    for w in windows:
        at = "-" if w.get("start_s") is None else f"{w['start_s']:.2f}s"
        span = "-" if w.get("len_s") is None else f"{w['len_s']:.2f}s"
        prefix = (f"    {w.get('n', '?'):>3}  {at:>7}  {span:>6}  "
                  f"{_fmt_secs(w.get('encode_s')):>5}   ")
        blank = " " * len(prefix)
        rungs = w.get("rungs") or []
        if not rungs:
            # A residual window the stop rule refused (no encode, no
            # generate) still gets its row: that is how a reader sees the
            # rule fire — and, should a last word ever go missing, whether
            # this rule was involved.
            out.append(prefix + (w.get("outcome") or "(no generate call)"))
            continue
        for j, r in enumerate(rungs):
            bs, nh = r.get("beam_size"), r.get("num_hypotheses")
            search = f"beam{bs}" if bs and bs > 1 else (f"best{nh}" if nh else "greedy")
            row = (f"{j + 1:>2}  {r.get('temperature', 0.0):>3.1f}  {search:<7}"
                   f"{r.get('tokens', 0):>6}  {_fmt_num(r.get('alp')):>6}  "
                   f"{_fmt_num(r.get('cr')):>5}  {_fmt_num(r.get('nsp')):>4}  "
                   f"{_fmt_secs(r.get('secs')):>6}  {r.get('outcome', '')}")
            if j == 0 and w.get("token_cap") is not None:
                # DECODE_TOKEN_CAP_PER_SECOND lowered this window's per-rung
                # token limit (a rung that ran into it says "hit cap").
                row += f"  [cap {w['token_cap']}]"
            out.append((prefix if j == 0 else blank) + row)
    if omitted:
        out.append(f"    … {omitted} more window{'s' if omitted != 1 else ''} omitted")
    return out


def _format_segments_section(seg_diag: list[dict], info, kwargs: dict,
                             speakers: "list | None" = None) -> list[str]:
    """Either a fixed-width segments table OR an empty-output diagnostic
    banner whose hint depends on `info.duration_after_vad` and `kwargs`.

    `speakers` (per-segment labels, when diarization ran) adds a `spk`
    column. It costs 5 of the text budget, so it is only ever added for runs
    that actually diarized — a decode-only receipt keeps the shape it has
    always had, and no existing log looks different."""
    n = len(seg_diag)
    if n == 0:
        out = [_section_rule("Segments  (n=0)  [!] no output produced")]
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        dav = getattr(info, "duration_after_vad", None)
        ip = kwargs.get("initial_prompt")
        if duration > 0 and dav is not None and float(dav) < 0.3 * duration:
            out.append(f"    likely cause: VAD ate audio  "
                       f"(duration_after_vad={float(dav):.2f}s vs {duration:.2f}s)")
            out.append("    next step:    set VAD_FILTER=false or "
                       "VAD_MIN_SILENCE_MS=250 in /settings")
        elif ip:
            out.append("    likely cause: initial_prompt may be poisoning decode")
            out.append("                  (tnfru/primeline finetunes); "
                       "clear DEFAULT_PROMPT in /settings")
        else:
            out.append("    likely cause: thresholds suppressed all segments")
            out.append("                  try disabling NO_SPEECH / LOG_PROB / "
                       "COMPRESSION_RATIO thresholds in /settings")
        return out

    dropped_n = sum(1 for s in seg_diag if s.get("dropped"))
    label = f"Segments  (n={n})"
    if dropped_n:
        label += f"  [✗ = {dropped_n} dropped by post-decode guard]"
    cut_n = sum(1 for s in seg_diag if s.get("cut") and not s.get("dropped"))
    if cut_n:
        label += f"  [✂ = {cut_n} made-up tail cut]"
    head_n = sum(1 for s in seg_diag if s.get("head_cut") and not s.get("dropped"))
    if head_n:
        label += f"  [✂ = {head_n} prompt echo cut from the start]"
    out = [_section_rule(label)]
    has_spk = bool(speakers) and any(speakers)
    spk_head = f"{'spk':>5}  " if has_spk else ""
    out.append(
        f"    {'#':>3}  {'start':>7}  {'end':>7}  {spk_head}"
        f"{'alp':>6}  {'nsp':>5}  {'cr':>5}  {'T':>4}    text"
    )
    # The FILE keeps up to LOG_SEGMENT_ROWS_MAX rows (0 = unlimited). The
    # /logs viewer folds them for display — but it can only ever reveal rows
    # that were written, which is exactly why the old inert "(+610 more)"
    # tail could never expand into anything.
    cap = int(getattr(cfg, "LOG_SEGMENT_ROWS_MAX", 0) or 0)
    rows = n if cap <= 0 else min(n, cap)
    text_max = _SEG_TEXT_MAX - 5 if has_spk else _SEG_TEXT_MAX
    for i in range(rows):
        s = seg_diag[i]
        text = s["text"]
        if len(text) > text_max:
            text = text[:text_max - 3] + "..."
        mark = "✗" if s.get("dropped") else (
            "✂" if s.get("cut") or s.get("head_cut") else " ")
        spk = ""
        if has_spk:
            raw_spk = speakers[i] if i < len(speakers) else ""
            spk = f"{_short_speaker(raw_spk or ''):>5}  "
        out.append(
            f"    {s['id']:>3d}  "
            f"{s['start']:>6.2f}s  {s['end']:>6.2f}s  {spk}"
            f"{s['alp']:>+6.2f}  {s['nsp']:>5.2f}  {s['cr']:>5.2f}  "
            f"{s['temp']:>4.1f}  {mark} {text}"
        )
    if n > rows:
        out.append(f"    … (+{n - rows} more, not logged — "
                   f"raise LOG_SEGMENT_ROWS_MAX)")
    return out


def _align_speakers_to_diag(seg_diag: list, speakers: "list | None") -> "list | None":
    """Speaker labels re-indexed to `seg_diag` rows for the receipt table.

    `seg_diag` keeps EVERY decoded segment (dropped ones included, flagged
    `dropped`), while assign_speakers labels only the kept `segments_list`.
    Indexing the former with the latter shifts every label after a dropped
    row by one; here each kept row consumes the next label and a dropped row
    gets "". None when there is nothing to label (keeps the column off)."""
    if not speakers:
        return None
    it = iter(speakers)
    return [("" if s.get("dropped") else next(it, "")) for s in seg_diag]


def _stage_extras(stats_key: "str | None", t0_perf: float) -> dict:
    """`device` + `load_secs` for one Pipeline-table row.

    Keyed by the NAMESPACED stats key (`pyannote:` / `uvr:` / `gguf:`
    prefixes), not the bare model name — _model_compute_device looks up by
    bare name and would silently miss every non-whisper family.

    The stage's wall-clock start is reconstructed from its perf_counter
    origin so `load_secs_since` can tell "this stage paid for the load" from
    "it was already resident", without threading a second clock through
    four call sites."""
    out: dict = {}
    if not stats_key:
        return out
    started_wall = time.time() - (time.perf_counter() - t0_perf)
    out["load_secs"] = system_stats.load_secs_since(stats_key, started_wall)
    for entry in system_stats.loaded_models_snapshot():
        if entry.get("name") == stats_key:
            out["device"] = entry.get("device")
            break
    return out


def _log_held_receipts(entries: "list[dict]") -> None:
    """Render and log receipts released without their translation.

    A held receipt that vanished would be strictly worse than the split one
    it replaces, so every release path funnels through here — including the
    sweeper's and the shutdown flush's."""
    for kwargs in entries:
        try:
            logger.info(_format_request_block(**kwargs))
        except Exception as e:  # noqa: BLE001 — a receipt is never fatal
            logger.warning("[receipt] release render failed: %s", e)


def _release_held_receipt(key: "str | None", note: str) -> None:
    """Release one held receipt on a translate failure/cancel path."""
    if not key:
        return
    entry = receipt_hold.release(key, note)
    if entry is not None:
        _log_held_receipts([entry])


async def _receipt_sweeper() -> None:
    """Release receipts whose translation went quiet.

    The hold is an IDLE timer restamped by the translate job's progress
    heartbeat, so this only fires for a translation that crashed, wedged, or
    was never sent at all — never for one that is merely slow."""
    while True:
        try:
            await asyncio.sleep(5.0)
            released = receipt_hold.sweep()
            if released:
                _log_held_receipts(released)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a sweeper must not die
            logger.warning("[receipt] sweeper error: %s", e)


def _stage_ran(stages: "list | None", name: str) -> bool:
    """Did this stage produce a SUCCESSFUL timing entry? Soft-failed stages
    (detail="failed", carries an error key) are excluded — the request FLAGS
    say what was asked for, which is a different question when a stage
    soft-fails or is disabled server-side."""
    return any(s.get("name") == name and "error" not in s
               for s in (stages or []))


def _stage_field(stages: "list | None", name: str, key: str):
    """One field off a stage's timing entry, or the _OMIT sentinel."""
    for s in (stages or []):
        if s.get("name") == name:
            v = s.get(key)
            return v if v is not None else _OMIT
    return _OMIT


def _fmt_secs(v) -> str:
    """`12.3s` / `-` for a missing timing, right-alignable."""
    if v is None:
        return "-"
    return f"{float(v):.1f}s"


def _format_pipeline_section(stages: "list[dict] | None") -> list[str]:
    """`─── Pipeline ───`: one row per stage that actually ran.

    Answers the question an operator opens the log with — where did the six
    minutes go — which no amount of per-stage params answers. The `load`
    column is the point: a cold model load becomes a named cost instead of
    an unexplained gap between two timestamps, and `load 0.0s` on a stage
    is the receipt's own proof that preloading worked.

    Reuses the fixed-width column idiom _format_segments_section
    established; there is no generic table helper and two of them would
    drift."""
    rows = [s for s in (stages or []) if s and s.get("name")]
    if not rows:
        return []
    rows.sort(key=lambda s: (_STAGE_ORDER.index(s["name"])
                             if s["name"] in _STAGE_ORDER else len(_STAGE_ORDER)))
    total = sum(float(s.get("secs") or 0.0) for s in rows)
    mins, secs = divmod(int(total), 60)
    wall = f"{mins}:{secs:02d}" if mins else f"{total:.1f}s"
    n = len(rows)
    out = [_section_rule(
        f"Pipeline  ({n} stage{'s' if n != 1 else ''} · {wall} wall)")]
    out.append(f"    {'#':>2}  {'stage':<12} {'model':<32} {'dev':<6} "
               f"{'load':>7} {'run':>8}")
    for i, s in enumerate(rows, 1):
        model = str(s.get("model") or "")
        if len(model) > 32:
            model = model[:29] + "..."
        load = s.get("load_secs")
        run = float(s.get("secs") or 0.0) - float(load or 0.0)
        out.append(
            f"    {i:>2}  {s['name'][:12]:<12} {model:<32} "
            f"{str(s.get('device') or '')[:6]:<6} "
            f"{_fmt_secs(load):>7} {_fmt_secs(max(0.0, run)):>8}"
        )
        detail = s.get("detail")
        if detail:
            out.append(f"        {detail}")
    return out


def _format_stage_section(label: str, rows: "list[tuple]") -> list[str]:
    """A `─── <Stage> ───` params block. `rows` are (key, value) pairs; a
    value of the sentinel `_OMIT` drops the row so callers can build a flat
    list without branching around every optional knob."""
    body = [_param_row("    ", k, v) for k, v in rows if v is not _OMIT]
    if not body:
        return []
    return [_section_rule(f"{label}  (* = non-default)"), *body]


def _format_notes_section(warnings: "list | None",
                          skipped: "list | None") -> list[str]:
    """`─── Notes ───`: soft failures and stages that didn't run.

    These already exist as log lines, but they fire minutes before the
    receipt and scroll away — and the receipt is the only place an operator
    looks. `( )` marks a skip and `[!]` a warning; deliberately NOT `✗`,
    which the segments table already owns for guard-dropped rows."""
    out: list[str] = []
    for s in (skipped or []):
        out.append(f"    ( ) {_log_safe(str(s))}")
    for w in (warnings or []):
        out.append(f"    [!] {_log_safe(str(w))}")
    if not out:
        return []
    return [_section_rule("Notes"), *out]


def _model_compute_device(name: str) -> "tuple[str | None, str | None]":
    """Look up the actual device + compute_type a model was loaded with —
    these may differ from cfg.MODEL_* if the fallback path was taken."""
    for entry in system_stats.loaded_models_snapshot():
        if entry.get("name") == name:
            return entry.get("compute_type"), entry.get("device")
    return None, None


def _format_request_block(
    *,
    file_label: str,
    model_name: str,
    info,
    kwargs: dict,
    seg_diag: list[dict],
    raw: str,
    final: str,
    steps: "list | None" = None,
    request_id: str | None = None,
    captured_id: str | None = None,
    endpoint: str = "/v1/audio/transcriptions",
    audio_source: str | None = None,
    ident=None,
    overrides_ignored: "list | None" = None,
    user_id: str | None = None,
    key_id: str | None = None,
    username: str | None = None,
    key_label: str | None = None,
    guards: "dict | None" = None,
    stages: "list | None" = None,
    separation: "dict | None" = None,
    diarization: "dict | None" = None,
    translation: "dict | None" = None,
    speakers: "list | None" = None,
    warnings: "list | None" = None,
    skipped: "list | None" = None,
    decode_trace: "dict | None" = None,
) -> str:
    """Full per-request log block. `steps` is the per-pipeline trace; passed
    in only when cfg.TRACE_ENABLED so the block stays a single message.

    `request_id` (uuid4 hex) is the cross-reference key between this
    durable log block and a report submitted via /quick-config. When
    present, the title line carries `req=<id[:8]>` so an admin reading
    a /reports row can grep the log for the matching block.

    `captured_id` is the capture row id when the capture pipeline fired
    for this request — admins can grep for `captured=<id[:8]>` to find
    the audio+timestamps row on /captures.

    `endpoint` is the route that produced the block — `/v1/audio/transcriptions`
    for the batch (file-upload) route, `…/stream` for live dictation — so the two
    sources are distinguishable in the log. `audio_source` (when given) describes
    the input transport/codec + rate, shown as an `input` line in the Audio
    section (the model itself always decodes at 16 kHz mono).

    `stages` / `separation` / `diarization` / `translation` / `speakers` /
    `warnings` / `skipped` describe the post-decode pipeline. All optional:
    this renderer was written when the pipeline was decode and nothing else,
    and callers that still are (live dictation, the test modules) pass none
    of them and get exactly the block they got before. A stage section is
    emitted only when that stage actually ran, so a decode-only receipt is
    unchanged and a four-stage one finally says what it did."""
    title_rule = "═" * _LOG_WIDTH
    rule = "─" * _LOG_WIDTH

    status = "[!] empty output" if len(seg_diag) == 0 else "✓ ok"
    if request_id:
        status = f"req={request_id[:8]}  {status}"
    if captured_id:
        status = f"captured={captured_id[:8]}  {status}"
    title = "  " + endpoint
    pad = max(1, _LOG_WIDTH - len(title) - len(status))
    title_line = f"{title}{' ' * pad}{status}"

    lines: list[str] = ["", title_rule, title_line, title_rule]

    lines.append(f"  file   {file_label}")
    model_line = f"  model  {model_name}"
    compute, device = _model_compute_device(model_name)
    extras = []
    if compute:
        extras.append(f"compute={compute}")
    if device:
        extras.append(f"device={device}")
    if extras:
        model_line += "   " + "  ".join(extras)
    lines.append(model_line)
    lines.extend(_format_pipeline_section(stages))

    lines.append(_section_rule("Audio"))
    if audio_source:
        lines.append(f"    {'input':<{_NAME_COL - 4}}{audio_source}")
    lang = getattr(info, "language", "?")
    lang_prob = getattr(info, "language_probability", None)
    lang_str = f"{lang}  (prob={lang_prob:.2f})" if lang_prob is not None else str(lang)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    lines.append(f"    {'language':<{_NAME_COL - 4}}{lang_str}")
    lines.append(f"    {'duration':<{_NAME_COL - 4}}{duration:.2f}s")
    dav = getattr(info, "duration_after_vad", None)
    if dav is not None:
        retained = (float(dav) / duration * 100) if duration > 0 else 0.0
        lines.append(
            f"    {'duration_after_vad':<{_NAME_COL - 4}}"
            f"{float(dav):.2f}s   ({retained:.0f} % retained)"
        )

    # Post-decode stage params, in pipeline order. Every one of these was in
    # scope at the call site all along; the block simply never asked for them.
    if separation:
        lines.extend(_format_stage_section("Separation", [
            ("separation_model", separation.get("model")),
            ("device", separation.get("device", _OMIT)),
            ("resample", separation.get("resample", _OMIT)),
            ("stem", separation.get("stem", _OMIT)),
        ]))
    if diarization:
        lines.extend(_format_stage_section("Diarization", [
            ("diarization_model", diarization.get("model")),
            ("device", diarization.get("device", _OMIT)),
            ("num_speakers", diarization.get("num_speakers")),
            ("min_speakers", diarization.get("min_speakers")),
            ("max_speakers", diarization.get("max_speakers")),
            ("embedding_batch_size", diarization.get("embedding_batch_size", _OMIT)),
            ("result", diarization.get("result", _OMIT)),
        ]))
    if translation:
        lines.extend(_format_stage_section("Translation", [
            ("translation_model", translation.get("model")),
            ("device", translation.get("device", _OMIT)),
            ("targets", translation.get("targets")),
            ("source_lang", translation.get("source", _OMIT)),
            ("mode", translation.get("mode")),
            ("context_segments", translation.get("context_segments", _OMIT)),
            ("glossary", translation.get("glossary", _OMIT)),
            ("result", translation.get("result", _OMIT)),
        ]))

    lines.append(_section_rule("Decode params  (* = non-default)"))
    lines.extend(_format_decode_params(kwargs))

    # Post-decode guards — applied AFTER model.transcribe (word-rate drop,
    # tail trim, streaming final-drop thresholds), so they are not kwargs and
    # would otherwise be invisible in the block. Rows marked ✗ in the segments
    # table were removed by one of these.
    if guards:
        lines.append(_section_rule("Post-decode guards  (* = non-default)"))
        for gk, gv in guards.items():
            lines.append(_param_row("    ", gk, gv))

    # What faster-whisper did INSIDE model.transcribe — windows, rungs,
    # tokens. The segments table cannot show this: a tail window that ran the
    # whole temperature ladder and was then skipped leaves no segment at all.
    lines.extend(_format_decode_trace_section(decode_trace))

    lines.extend(_format_segments_section(seg_diag, info, kwargs, speakers))

    lines.extend(_format_notes_section(warnings, skipped))

    # Identity section — always shown when the caller is known, so the resolved
    # user/key (and any applied per-identity overrides, or their ABSENCE) is
    # visible at a glance. This was previously suppressed for no-config
    # requests, which made per-identity mismatches invisible in the log.
    _ident_detail = ident is not None and (getattr(ident, "layers", None)
                                           or getattr(ident, "locked", None)
                                           or overrides_ignored)
    if user_id or key_id or _ident_detail:
        lines.append(_section_rule("Identity"))
        _safe_name = _log_safe(username) if username else None
        who = f"{_safe_name} ({_short_id(user_id)})" if _safe_name else _short_id(user_id)
        lines.append(f"    {'user':<{_NAME_COL - 4}}{who}")
        if key_id:
            _safe_label = _log_safe(key_label) if key_label else None
            which = (f"{_safe_label} ({_short_id(key_id)})"
                     if _safe_label else _short_id(key_id))
            lines.append(f"    {'key':<{_NAME_COL - 4}}{which}")
        if ident is not None and ident.profiles_applied:
            lines.append(f"    {'profiles':<{_NAME_COL - 4}}{' → '.join(ident.profiles_applied)}")
        if ident is not None and ident.layers:
            lines.append(f"    {'layers':<{_NAME_COL - 4}}{', '.join(ident.layers)}")
        elif user_id or key_id:
            # No identity layer resolved — call it out explicitly so a missing
            # binding (the classic "my override didn't apply") is obvious.
            lines.append(f"    {'overrides':<{_NAME_COL - 4}}(none — inherits per-model / global)")
        if ident is not None and ident.locked:
            lines.append(f"    {'locked':<{_NAME_COL - 4}}{', '.join(sorted(ident.locked))}")
        if overrides_ignored:
            lines.append(f"    {'overrides_ignored':<{_NAME_COL - 4}}{', '.join(overrides_ignored)}")

    lines.append(rule)
    lines.append(f"  RAW WHISPER  {raw!r}")
    lines.append(rule)
    if steps:
        # Count only steps that actually rewrote the text (before != after) as
        # "changed"; the rest (EXCLUDED for this model, globally disabled, no-op)
        # are "unchanged". This matches the /quick-config and /reports viewers,
        # which render the same trace and split it the same way — the header used
        # to print len(steps) and label them all "changed", overcounting skips.
        changed = sum(1 for _, before, after in steps if before != after)
        unchanged = len(steps) - changed
        plural = "s" if changed != 1 else ""
        header = f"  PIPELINE  ({changed} step{plural} changed text"
        if unchanged:
            header += f", {unchanged} unchanged"
        lines.append(header + ")")
        for name, before, after in steps:
            lines.append(f"    ▸ {name}")
            lines.append(f"        {before!r}")
            lines.append(f"     →  {after!r}")
        lines.append(rule)
    lines.append(f"  FINAL        {final!r}")
    lines.append(title_rule)

    return "\n".join(lines)


def _failed_stage(name: str, t0: float, model: "str | None",
                  exc: BaseException) -> dict:
    """A receipt row for a stage that soft-failed (the job went on without
    it): its wall time, and an `error` class the usage ledger counts —
    without it a failed stage left no row anywhere."""
    return {
        "name": name,
        "secs": round(time.perf_counter() - t0, 2),
        "model": model or None,
        "detail": "failed",
        "error": metrics.classify_error(exc, status="error", stage=name)[0],
    }
