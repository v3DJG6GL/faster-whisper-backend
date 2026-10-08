"""Post-decode guard helpers shared by the batch route and both streaming
decodes: the word-rate drop (segment_exceeds_word_rate), the tail-cut / head-
echo settings resolved per model + identity (tail_guard_limits,
head_echo_min_words), their receipt rows (tail_guard_rows, tail_cut_rows,
head_echo_rows) and the /stats counter (record_tail_cut). The cut rules
themselves live in transcription/segment_guards.py.
"""
from faster_whisper_backend.transcription import segment_guards
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.stats import metrics
from faster_whisper_backend.transcription.receipt import PlainText


# Below this many words a segment's rate is statistically meaningless (a single
# short interjection in a tight VAD chunk can legitimately look "fast").
_WORD_RATE_MIN_WORDS = 3


def _off_below_two(n: int) -> int:
    """A count rule the guard ignores below 2 reads 0 (off) on the receipt,
    never a 1 that looks on but never fires."""
    return n if n >= 2 else 0


def tail_guard_limits(model_name, ident) -> dict:
    """The three tail-cut settings (transcription/segment_guards.py) resolved for this
    model + identity, as apply_tail_guards kwargs. Shared by the batch route and
    both streaming decodes. The two counts treat 1 as off, like the guard does."""
    return {
        "burst": float(effective_config.cfg_for(model_name, "SEGMENT_MAX_WORD_BURST_PER_S", ident) or 0),
        "zero_tail": _off_below_two(int(effective_config.cfg_for(model_name, "SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS", ident) or 0)),
        "repeats": _off_below_two(int(effective_config.cfg_for(model_name, "SEGMENT_REPEAT_COLLAPSE_MIN_REPEATS", ident) or 0)),
    }


def tail_guard_rows(limits: dict) -> dict:
    """The receipt's "Post-decode guards" rows for the three tail-cut settings."""
    return {
        "segment_max_word_burst_per_sec": limits["burst"],
        "segment_zero_length_tail_min_words": limits["zero_tail"],
        "segment_repeat_collapse_min_repeats": limits["repeats"],
    }


# Tail cuts listed one per row; a long file that trips the guards on many
# segments gets the rest as a count, like the capped segments table.
_TAIL_CUT_ROWS_MAX = 10


def tail_cut_rows(cuts: list) -> dict:
    """One receipt row per tail cut that fired (`tail_cut`, `tail_cut_2`, …),
    the first `_TAIL_CUT_ROWS_MAX` of them; `tail_cut_more` counts the rest."""
    rows = {("tail_cut" if n == 0 else f"tail_cut_{n + 1}"):
            PlainText(segment_guards.describe_cut(c))
            for n, c in enumerate(cuts[:_TAIL_CUT_ROWS_MAX])}
    if len(cuts) > _TAIL_CUT_ROWS_MAX:
        rows["tail_cut_more"] = PlainText(
            f"{len(cuts) - _TAIL_CUT_ROWS_MAX} more not listed")
    return rows


def head_echo_min_words(model_name, ident) -> int:
    """SEGMENT_HEAD_ECHO_MIN_WORDS resolved for this model + identity (the head
    twin of tail_guard_limits; kept apart because it is not an
    apply_tail_guards kwarg). 1 is treated as off, like the guard does."""
    return _off_below_two(int(effective_config.cfg_for(model_name, "SEGMENT_HEAD_ECHO_MIN_WORDS", ident) or 0))


def head_echo_rows(min_words: int, cut: "dict | None") -> dict:
    """The receipt's "Post-decode guards" rows for the head-echo rule: the
    setting, plus a `head_cut` row when it fired. Only the first surviving
    segment is checked, so there is at most one cut per decode."""
    rows: dict = {"segment_head_echo_min_words": min_words}
    if cut:
        rows["head_cut"] = PlainText(segment_guards.describe_head_cut(cut))
    return rows


def record_tail_cut(cut: dict, *, emptied: bool) -> None:
    """Count one tail (or head-echo) cut on /stats: every rule that fired, plus
    "emptied" when nothing was left of the segment."""
    for rule in cut.get("rules") or []:
        metrics.record_guard_hit(rule)
    if emptied:
        metrics.record_guard_hit("emptied")


def segment_exceeds_word_rate(seg, max_wps: float) -> bool:
    """Post-decode anti-hallucination guard (SEGMENT_MAX_WORDS_PER_S), shared
    by the batch route and the streaming FINAL decode.

    When trailing non-speech audio survives the VAD into a decode, Whisper
    re-decodes the sub-second leftover after the last aligned word as its own
    zero-padded window and confidently replays its text context — segments of
    20+ words crammed into half a second. Those pass every confidence gate
    (high avg_logprob, temperature 0.0, no_speech_prob possibly below the
    threshold); the impossible word density is their one reliable signature.
    Real speech peaks around ~6 words/s, so the default limit of 10 has wide
    margin on both sides."""
    if not max_wps or max_wps <= 0:
        return False
    words = getattr(seg, "words", None)
    n = len(words) if words else len((getattr(seg, "text", "") or "").split())
    if n < _WORD_RATE_MIN_WORDS:
        return False
    duration = float(getattr(seg, "end", 0.0) or 0.0) - float(getattr(seg, "start", 0.0) or 0.0)
    if duration <= 0:
        return True
    return (n / duration) > float(max_wps)
