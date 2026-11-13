"""The post-processing rules engine: compiles cfg.PIPELINE_RULES into
_COMPILED_RULES (rebuild_caches — at import and again on any config save,
admin or quick-config, that touches a CACHE_REBUILD_FIELDS key) and applies
them to a transcript
(_postprocess_text), plus the live-dictation seam helpers holdback_start /
seam_culprit that read the same rule selection. Callers go through the module
attribute (``pl_engine._postprocess_text(...)``): rebuild_caches rebinds
_COMPILED_RULES, so a ``from ... import`` of it would go stale.

Once the app has started the helper pool (pipeline/rule_helper.py), every
walk that executes rule patterns on request text (_postprocess_text,
seam_culprit) runs in a helper process under a deadline: see "Time limit"
below. Without a pool the same walk runs in this process.
"""
import functools
import itertools
import logging
import pickle
import re
import threading
import time

from faster_whisper_backend.core.store_common import log_safe as _log_safe
from faster_whisper_backend.pipeline import dictation_map as _dictation_map
from faster_whisper_backend.pipeline import rule_helper as _rule_helper
from faster_whisper_backend.pipeline import seam_holdback as _seam_holdback
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import schema as settings_schema

logger = logging.getLogger("whisper-api")


# One ordered rule list (cfg.PIPELINE_RULES) runs over the joined transcript.
# Kinds: "regex-list" (ordered pattern→replacement entries),
# "callback:lowercase-wordlist" (German non-noun lowercaser), "callback:map"
# (dictation word→symbol), "callback:dedup" (collapse adjacent punctuation
# runs), "callback:upper" (capitalize after a sentence terminator) and
# "terminal" (final lstrip+rstrip; always last). Seeded list: config.json
# (settings/config.py PIPELINE_RULES); schema: settings/schema.py.
# The terminal row, empty patterns and invalid regexes are left out of the
# compiled list; globally-disabled rules stay in it (see _CompiledRule.enabled).

from dataclasses import dataclass, field as _dc_field


@dataclass(frozen=True)
class _CompiledRule:
    """One row of the compiled pipeline. `payload` is the second argument of
    the rule's `pattern.sub` (see _apply_rule):
      regex-list (per entry)      → replacement string
      callback:lowercase-wordlist → replacer closed over the lowercase wordlist
      callback:map                → replacer from dictation_map.compile_map
                                    (the lookup itself is in `map_lookup`)
      callback:dedup              → _dedup_callback
      callback:upper              → _upper_callback
    `name` is the rule slug used for per-model EXCLUDE / INCLUDE matching.
    `enabled` mirrors the global `rule.enabled` flag — checked at runtime
    rather than at compile time so per-model PIPELINE_RULES_INCLUDE can
    force-enable a globally-disabled rule.
    """
    name: str
    label: str
    type: str
    pattern: "re.Pattern[str]"
    payload: object
    enabled: bool
    # 1-based position of this rule's CARD in cfg.PIPELINE_RULES (terminal card
    # included) — identical to the /settings/pipeline card ordinal. `sub_no` is
    # the 1-based entry row within a regex-list card (None for single-compile
    # rules). Together they drive the `#P` / `#P.S` trace step number.
    card_no: int
    sub_no: "int | None" = None
    languages: frozenset[str] = frozenset()
    # The card's "Repeat until nothing changes" switch (schema
    # _RepeatableRule.repeat), on every row of a regex-list card: the walk
    # re-applies the whole card, not one entry (see _card_passes).
    repeat: bool = False
    # callback:map only: the effective lower-cased key → value lookup (ß keys'
    # ss variants included, see pipeline/dictation_map.py). The streaming
    # hold-back builds its spec from it. Kept out of eq/hash: a dict is
    # unhashable and the replacer already carries the same data.
    map_lookup: "dict[str, str] | None" = _dc_field(default=None, compare=False)
    # Upper bound on len(output) / (len(input) + 1) for one sub (see
    # _template_growth), or None when there is none: then _apply_bounded counts
    # the output exactly before building it.
    max_growth: "int | None" = _dc_field(default=None, compare=False)
    # regex-list only: the parsed replacement template, [literal, group,
    # literal, ...] (re._parser.parse_template), for that exact count.
    template: "list | None" = _dc_field(default=None, compare=False)
    # What a callback replacer was built from, when it is a closure: the map
    # (callback:map) or the lower-cased wordlist (callback:lowercase-wordlist).
    # A helper process rebuilds the replacer from it (_rule_spec): a closure
    # cannot cross a process boundary.
    source: object = _dc_field(default=None, compare=False)


_COMPILED_RULES: list[_CompiledRule] = []
# Bumped by every rebuild_caches(): the key under which per-rule-set derived
# data (the streaming hold-back spec) is cached.
_RULES_GEN: int = 0
# Hold-back specs rebuild_caches builds in its worker thread, keyed
# (gen, selection): one for each selection a session without per-identity or
# per-model scoping makes. holdback_start runs on the event loop, and a spec on
# a large map takes most of a second to build.
_WARM_SPECS: "dict[tuple[int, tuple[int, ...]], _seam_holdback.HoldSpec]" = {}
# Captured from the terminal rule's name/label in cfg.PIPELINE_RULES at cache
# build time. Falls back to the constants below if the user removed the
# terminal row. The slug is used to honor exclude-set membership (a captures
# pipeline can drop the trim by listing `trim-edges` in
# CAPTURES_PIPELINE_RULES_EXCLUDE).
_TERMINAL_NAME: str = "trim-edges"
_TERMINAL_LABEL: str = "Trim edges (always-last)"
# 1-based card position of the terminal trim, captured at rebuild time so its
# trace step number matches the /settings/pipeline card ordinal. Falls back to
# len(PIPELINE_RULES)+1 when the user deleted the terminal row.
_TERMINAL_CARD_NO: int = 0
# Ceiling on the pipeline's output, enforced on every rule before its sub
# builds the result (_apply_bounded). A 30-minute transcript is a few hundred
# KB, so 4 M chars only ever trips on a rule set that expands its own input
# (the pipeline runs on admin-editable regexes).
_POSTPROCESS_MAX_CHARS = 4_000_000
# The bound a walk actually gets is relative to its input (output_bound): a
# chain of expanding entries on a card a tagged non-admin may edit can grow a
# short final to just under the absolute bound, and every later rule then
# runs on MBs — seconds of CPU per walk, on the event loop for a streaming
# final. The floor keeps single expansions on a short text applied (a
# snippet of a few KB, a map value or regex-list replacement per match).
_POSTPROCESS_FLOOR = 16_384
_POSTPROCESS_MAX_RATIO = 32
# Longest step text the trace keeps. Every consumer cuts a step's text at this
# length (receipt._step_text and the recent-transcriptions / reports stores'
# _CAP_RAW), so a longer copy is never read, while a card of a few hundred
# expanding steps would otherwise hold one multi-MB string per step.
_TRACE_TEXT_MAX = 50_000
# Most trace rows that keep their texts: the last ones. Every consumer reads
# the texts of only the last _CAP_STEPS_ROWS rows (receipt, the
# recent-transcriptions and reports stores) and merely counts the earlier
# ones, so an earlier row keeps its label and whether it changed the text
# (_trace_push). A repeating card of a couple hundred changing entries
# otherwise holds one cut copy per row: over a thousand rows, hundreds of MB.
_TRACE_MAX_ROWS = 500
# Most passes a repeating rule gets per transcript (see the schema constant).
_REPEAT_MAX_PASSES = settings_schema.RULE_REPEAT_MAX_PASSES


def output_bound(n: int) -> int:
    """The output bound of a pipeline walk over an `n`-char text: the floor
    plus _POSTPROCESS_MAX_RATIO chars per input char, never past
    _POSTPROCESS_MAX_CHARS. The constants are read at call time."""
    return min(_POSTPROCESS_MAX_CHARS, _POSTPROCESS_FLOOR + _POSTPROCESS_MAX_RATIO * n)


class WalkBudget:
    """One output bound shared by many walks over pieces of one text (one
    walk per word or per subtitle cue): output_bound of the pieces' total
    length. Pass the same budget to every piece's _postprocess_text. Each
    walk is bounded by what is left and charged the longest text it built,
    not its result, so a rule set that expands and then shrinks back cannot
    hand every piece the full floor. A walk costs about rules x the longest
    text it built, so the pieces together cost no more than one walk over
    the whole text may. `left` (what remains, negative once overspent) is
    for the caller to skip a piece the budget no longer covers."""

    __slots__ = ("left",)

    def __init__(self, total_len: int) -> None:
        self.left = output_bound(total_len)


def _rule_ordinal(card_no: int, sub_no: "int | None" = None) -> str:
    """Format a pipeline step number to match the /settings/pipeline card
    position: '#P' for a single-compile rule, '#P.S' for a regex-list entry
    (S = the entry's 1-based row within that card)."""
    return f"#{card_no}.{sub_no}" if sub_no else f"#{card_no}"


def _dedup_callback(match: "re.Match[str]") -> str:
    """Pick the user-intended punct from a run of 2+ adjacent marks. Whisper
    emits its own commas as soft pauses around dictation keywords; after
    substitution we get ",." (Punkt) or ",;" (Semikolon). Prefer any non-
    comma; within non-commas prefer the LAST (dictation came after the
    Whisper pause). Pure commas → single comma."""
    run = match.group(0)
    non_comma = [c for c in run if c != ","]
    return non_comma[-1] if non_comma else ","


def _upper_callback(match: "re.Match[str]") -> str:
    """Uppercase group(2) if the pattern produced two groups; else uppercase
    the entire match. Default seeded pattern produces ([.?!]\\s+|\\n[^\\S\\n]*)
    + ([a-zäöüß]). Only group(2)'s span changes: text the match holds
    outside groups 1 and 2 (a literal between them, a third group) is kept.
    A group(2) captured in a lookaround lies outside the match, which a sub
    cannot rewrite: the match is kept as is instead of inserting a copy."""
    try:
        g1, g2 = match.group(1), match.group(2)
    except IndexError:
        return match.group(0).upper()
    if g1 is None or g2 is None:
        return match.group(0)
    m0, s = match.group(0), match.start(0)
    if match.start(2) < s or match.end(2) > match.end(0):
        return m0
    return m0[:match.start(2) - s] + g2.upper() + m0[match.end(2) - s:]


def _make_lowercase_wordlist_replacer(wordlist: frozenset):
    """Returns a regex-sub callback that strips the matched terminator and
    lowercases group(2) IFF (group(2)+group(3)).lower() is in `wordlist`.
    The default seeded pattern produces three groups:
      group(1) = whitespace between terminator and next word
      group(2) = first letter of the next word
      group(3) = rest of the next word
    A pattern with fewer than 3 groups keeps the match as is: a save refuses
    it, and stripping its whole match would delete the captured word with it.
    Only the text before group(1) is the terminator: what the match holds
    between or after the groups (an opening quote, a trailing space) is kept,
    as in _upper_callback.
    A group captured in a lookaround lies outside the match, which a sub
    keeps anyway: the match is kept as is instead of inserting a copy.
    """
    def replace(m: "re.Match[str]") -> str:
        try:
            ws, first, rest = m.group(1), m.group(2), m.group(3)
        except IndexError:
            return m.group(0)
        if ws is None or first is None or rest is None:
            return m.group(0)
        if any(m.start(g) < m.start(0) or m.end(g) > m.end(0) for g in (1, 2, 3)):
            return m.group(0)
        m0, s = m.group(0), m.start(0)
        keep = m.start(1) - s
        if (first + rest).lower() in wordlist and m.start(2) - s >= keep:
            return m0[keep:m.start(2) - s] + first.lower() + m0[m.end(2) - s:]
        return m0[keep:]
    return replace


# Serialises rebuild_caches: it runs in asyncio.to_thread (apply.py). Every
# hot-apply caller that rebuilds holds rules_lock() today (only PIPELINE_RULES
# triggers a rebuild), so this is defensive: it also covers the import-time and
# test rebuilds and any future caller that skips rules_lock(). Without it two
# rebuild threads could finish out of order — a stale compile overwriting the
# newer one — and the unsynchronised `_RULES_GEN += 1` could store the same gen
# twice. Every cfg setattr happens before its own rebuild thread starts, so the
# thread that compiles last also reads the latest cfg.
_REBUILD_LOCK = threading.Lock()


def rebuild_caches() -> None:
    """(Re)compile every rule in cfg.PIPELINE_RULES into _COMPILED_RULES.

    The terminal row is filtered out (it runs as the implicit final trim,
    not via the walker). Globally-DISABLED rules are still compiled — the
    runtime filter consults `rule.enabled` per-call so per-model
    PIPELINE_RULES_INCLUDE can force-enable a globally-disabled rule. A rule
    whose PATTERN does not compile (a hand-edited config file the save-time
    validator never saw) is logged + skipped; a regex-list replacement
    template is not checked here (re parses it at the first sub).
    """
    with _REBUILD_LOCK:
        _rebuild_caches_locked()


def _rebuild_caches_locked() -> None:
    global _COMPILED_RULES, _TERMINAL_NAME, _TERMINAL_LABEL, _TERMINAL_CARD_NO, _RULES_GEN
    global _WARM_SPECS, _QUARANTINE
    compiled: list[_CompiledRule] = []
    terminal_name = _TERMINAL_NAME
    terminal_label = _TERMINAL_LABEL
    terminal_card_no = len(cfg.PIPELINE_RULES) + 1
    for card_no, rule in enumerate(cfg.PIPELINE_RULES, start=1):
        rtype = rule.get("type")
        if rtype == "terminal":
            terminal_name = rule.get("name", terminal_name)
            terminal_label = rule.get("label", terminal_label)
            terminal_card_no = card_no
            continue
        rule_enabled = bool(rule.get("enabled", True))
        rule_langs = frozenset(rule.get("languages") or ())
        # `is True`: the schema stores only true (off is absent); a hand-edited
        # "yes" must not switch it on (the dry run reads it the same way).
        rule_repeat = rule.get("repeat") is True

        # regex-list: expand each entry into its own _CompiledRule row, all
        # sharing the card's name + enabled + languages (so per-model
        # EXCLUDE/INCLUDE and the global toggle flip the whole card together).
        # Entries run in list order — NO longest-first sort. A bad entry is
        # skipped (not the whole card); an empty-pattern entry is a no-op.
        if rtype == "regex-list":
            rname = rule.get("name", "?")
            rlabel = rule.get("label", rname)
            for sub_no, entry in enumerate(rule.get("entries", []) or [], start=1):
                epat = entry.get("pattern", "")
                if not epat:
                    continue
                try:
                    ecre = re.compile(epat)
                except settings_schema.REGEX_COMPILE_ERRORS as e:
                    logger.warning("[pipeline] rule %r entry has invalid regex "
                                   "(%s) — skipping entry", rname, e)
                    continue
                erepl = entry.get("replacement", "") or ""
                template, growth = _template_growth(ecre, erepl)
                # sub_no is the entry's row in the card editor (the enumerate index
                # advances across skipped empty/bad entries above), so the trace
                # number `#card_no.sub_no` lines up with what the admin sees.
                compiled.append(_CompiledRule(
                    rname, f"{rlabel} · {entry.get('label') or epat}",
                    "regex-list", ecre, erepl,
                    rule_enabled, card_no, sub_no, rule_langs, rule_repeat,
                    max_growth=growth, template=template))
            continue

        map_lookup = None
        source = None
        try:
            if rtype == "callback:map":
                # Alternation regex from the map keys, longest-first,
                # word-bounded, case-insensitive, plus the ß/ss variants and
                # the "dictated punctuation wins" prefix — built by
                # pipeline/dictation_map.py, which the /settings/pipeline dry run
                # shares so the preview is the engine.
                m = rule.get("map", {}) or {}
                if not m:
                    continue
                cre, payload, map_lookup = _dictation_map.compile_map(m)
                # A copy: the helper's replacer is built from it later, and
                # must be built from what this compile saw.
                source = dict(m)
            else:
                pattern = rule.get("pattern", "")
                if not pattern:
                    continue
                cre = re.compile(pattern)
                if rtype == "callback:lowercase-wordlist":
                    wordlist = frozenset(w.lower() for w in (rule.get("wordlist", []) or []))
                    payload = _make_lowercase_wordlist_replacer(wordlist)
                    source = wordlist
                elif rtype == "callback:dedup":
                    payload = _dedup_callback
                elif rtype == "callback:upper":
                    payload = _upper_callback
                else:
                    logger.warning("[pipeline] unknown rule type %r — skipping", rtype)
                    continue
        except settings_schema.REGEX_COMPILE_ERRORS as e:
            logger.warning("[pipeline] rule %r has invalid regex (%s) — skipping",
                           rule.get("name"), e)
            continue
        compiled.append(_CompiledRule(rule.get("name", "?"),
                                       rule.get("label", rule.get("name", "?")),
                                       rtype, cre, payload, rule_enabled, card_no,
                                       languages=rule_langs, repeat=rule_repeat,
                                       map_lookup=map_lookup,
                                       max_growth=_callback_growth(rtype, map_lookup),
                                       source=source))
    warm = _warm_holdback_specs(compiled)
    # Publish order: warm specs, rules, gen. holdback_start reads the gen,
    # then the rules, then the warm specs, so a caller that sees the new rules
    # finds only new-gen warm specs: under the old gen it misses and builds
    # from the current rules, never pairing new-rule indices with a spec built
    # from the old rule set. A caller that sees the new gen finds its spec.
    new_gen = _RULES_GEN + 1
    _WARM_SPECS = {(new_gen, sel): spec for sel, spec in warm.items()}
    _COMPILED_RULES = compiled
    _TERMINAL_NAME = terminal_name
    _TERMINAL_LABEL = terminal_label
    _TERMINAL_CARD_NO = terminal_card_no
    _RULES_GEN = new_gen
    # A quarantine names rows of the rule set it was made for (_quarantine):
    # the new set starts without one, and the old list is let go.
    _QUARANTINE = (None, frozenset())
    _dictation_map.retain_live_maps(
        r.pattern for r in compiled if r.type == "callback:map")
    # Older-gen specs are never hit again but hold MBs each on a large map;
    # drop them now rather than wait for the LRU to evict them.
    _holdback_spec.cache_clear()


def _apply_rule(rule: _CompiledRule, text: str) -> str:
    """Hot path, no dispatch: the payload is pre-bound at rebuild_caches()
    time (a replacement string for `regex-list`, a pre-built callable for
    every callback:* type), so every type is one pattern.sub call with no
    per-request closure allocation."""
    return rule.pattern.sub(rule.payload, text)  # type: ignore[arg-type]


def _captures_in_lookaround(pattern: "re.Pattern[str]") -> bool:
    """Whether a group of `pattern` sits inside a lookahead or lookbehind,
    where it can capture text outside the match. Parse-only."""
    try:
        parsed = re._parser.parse(pattern.pattern, pattern.flags)
    except Exception:  # noqa: BLE001 - it compiled, but assume the worst
        return True
    sub_t = re._parser.SubPattern

    def walk(x, inside: bool) -> bool:
        if isinstance(x, sub_t):
            for op, av in x:
                name = str(op)
                if name == "SUBPATTERN" and inside and av[0] is not None:
                    return True
                if walk(av, inside or name in ("ASSERT", "ASSERT_NOT")):
                    return True
        elif isinstance(x, (list, tuple)):
            return any(walk(y, inside) for y in x)
        return False

    return walk(parsed, False)


def _template_growth(pattern: "re.Pattern[str]", repl: str) -> "tuple[list | None, int | None]":
    """(parsed template, growth bound) for a regex-list entry.

    A match of length L becomes `lit` literal characters plus the groups it
    references, each at most L long when it lies inside the match, so a text
    of n characters grows to at most n * (lit + refs) + (n + 1) * lit (the
    second term: one empty match per position). No bound when a referenced
    group can sit in a lookaround (it copies text the match never consumed),
    and no template when the replacement does not parse: the sub raises the
    same error the engine always raised for it."""
    try:
        tmpl = re._parser.parse_template(repl, pattern)
    except Exception:  # noqa: BLE001 - _apply_rule reports it at the sub
        return None, None
    lit = sum(len(s) for s in tmpl[0::2])
    refs = len(tmpl[1::2])
    if refs and _captures_in_lookaround(pattern):
        return tmpl, None
    return tmpl, max(lit + refs, 1) + lit


def _callback_growth(rtype: "str | None", map_lookup: "dict[str, str] | None") -> "int | None":
    """Growth bound (see _template_growth) for a callback rule's replacer.
    A map match is a non-empty key (plus a kept punctuation lead) and becomes
    at most the lead and its value; str.upper / str.lower expand a character
    to at most 3; dedup returns one mark per match, even for an empty one."""
    if rtype == "callback:map":
        try:
            return max((len(v) for v in (map_lookup or {}).values()), default=0) + 2
        except TypeError:  # a non-string value: count exactly
            return None
    if rtype in ("callback:upper", "callback:lowercase-wordlist", "callback:dedup"):
        return 3
    return None


class _ClippedText(str):
    """A trace step text cut at _TRACE_TEXT_MAX. `full_len` is the uncut
    length and `changed` whether the step changed the text, which the cut
    texts of a step that edits past the cut no longer show. str() / repr() /
    json.dumps() give the same output as for the plain cut string."""
    full_len: int
    changed: bool


def _trace_text(s: str, changed: bool) -> str:
    """`s` as the trace keeps it: unchanged up to _TRACE_TEXT_MAX chars, else
    a _ClippedText of its first _TRACE_TEXT_MAX chars."""
    if len(s) <= _TRACE_TEXT_MAX:
        return s
    clipped = _ClippedText(s[:_TRACE_TEXT_MAX])
    clipped.full_len = len(s)
    clipped.changed = changed
    return clipped


def _trace_clip(s: str, changed: bool, memo: "list | None") -> str:
    """_trace_text, reusing the last cut copy when `s` is the same object.

    Consecutive rows trace one string (a skipped card's rows all trace the
    same text, and a changing step's after-text is the next step's
    before-text), so one cut copy serves them all. `memo` holds the source
    object itself, not its id, so a freed id cannot be matched."""
    if len(s) <= _TRACE_TEXT_MAX:
        return s
    if memo is not None and memo[0] is s and memo[1] == changed:
        return memo[2]
    clipped = _trace_text(s, changed)
    if memo is not None:
        memo[:] = [s, changed, clipped]
    return clipped


def _trace_row(label: str, before: str, after: str,
               memo: "list | None" = None) -> tuple:
    """One `(label, before, after)` trace row, its texts cut by _trace_text;
    whether the step changed the text is decided on the uncut strings.
    `memo` is the one-slot cache of a single _postprocess_text walk."""
    changed = before != after
    b = _trace_clip(before, changed, memo)
    return (label, b, b if after is before else _trace_clip(after, changed, memo))


def _trace_push(trace: list, row: tuple) -> None:
    """Append `row` to a _postprocess_text trace, dropping the texts of the
    row that thereby leaves the last _TRACE_MAX_ROWS. The dropped row keeps
    its label and, as `changed` on an empty _ClippedText, whether it changed
    the text: the receipt's changed / unchanged count reads exactly that."""
    trace.append(row)
    if len(trace) <= _TRACE_MAX_ROWS:
        return
    label, b, a = trace[-_TRACE_MAX_ROWS - 1]
    changed = bool(b != a or getattr(b, "changed", False)
                   or getattr(a, "changed", False))
    gone = _TRACE_GONE[changed]
    if b is not gone:
        trace[-_TRACE_MAX_ROWS - 1] = (label, gone, gone)


def _gone_text(changed: bool) -> "_ClippedText":
    t = _ClippedText("")
    t.full_len = 0
    t.changed = changed
    return t


# The shared text of a row past the trace window (_trace_push), indexed by
# `changed`.
_TRACE_GONE = (_gone_text(False), _gone_text(True))


class _PastBound(Exception):
    pass


def _apply_bounded(rule: _CompiledRule, text: str, bound: int) -> "str | None":
    """_apply_rule, or None when its output would pass `bound` (output_bound
    of the walk's input).

    The bound is enforced before the result exists: one entry that expands
    every match 60x turns a text just under the bound into hundreds of MB in
    a single sub, so checking the length afterwards is too late. A text short
    enough that even the rule's worst-case growth stays under the bound goes
    straight to _apply_rule; otherwise the output length is counted match by
    match first (a string template) or as the replacer runs (a callback)."""
    growth = rule.max_growth
    if growth is not None and (len(text) + 1) * growth <= bound:
        return _apply_rule(rule, text)
    size = len(text)
    payload = rule.payload
    if callable(payload):
        def counted(m: "re.Match[str]") -> str:
            nonlocal size
            r = payload(m)
            size += len(r) - (m.end() - m.start())
            if size > bound:
                raise _PastBound
            return r

        try:
            return rule.pattern.sub(counted, text)
        except _PastBound:
            return None
    tmpl = rule.template
    if tmpl is None:
        return _apply_rule(rule, text)
    lit = sum(len(s) for s in tmpl[0::2])
    groups = tmpl[1::2]
    for m in rule.pattern.finditer(text):
        size += lit - (m.end() - m.start()) + sum(m.end(g) - m.start(g) for g in groups)
        if size > bound:
            return None
    return _apply_rule(rule, text)



def _cards(rows: "list[_CompiledRule]") -> "list[list[_CompiledRule]]":
    """The compiled rows grouped by card (a regex-list card is one row per
    entry, consecutive and sharing its card_no), in pipeline order."""
    return [list(g) for _k, g in itertools.groupby(rows, key=lambda r: r.card_no)]


def _card_passes(card: "list[_CompiledRule]") -> int:
    """How many times the walk may apply `card`: once, or up to
    _REPEAT_MAX_PASSES for a repeating rule (the walk stops earlier at the
    first pass that leaves the text as it was)."""
    return _REPEAT_MAX_PASSES if card[0].repeat else 1


def _scoping(model_name: "str | None", ident=None,
             extra_excludes: "set[str] | None" = None) -> "tuple[set[str], set[str]]":
    """The (exclude, include) rule-slug sets _postprocess_text applies for this
    model / identity — see its docstring for the precedence."""
    exclude: "set[str]" = set()
    include: "set[str]" = set()
    if ident is not None:
        # The resolver already folded the per-model layer into these sets
        # (identity rules win first-mention; per-model is the fallback layer).
        # Don't re-read MODEL_OVERRIDES here or it would double-apply.
        exclude = set(ident.pipeline_exclude)
        include = set(ident.pipeline_include)
    elif model_name:
        overrides = getattr(cfg, "MODEL_OVERRIDES", None) or {}
        m_over = overrides.get(model_name) if isinstance(overrides, dict) else None
        if isinstance(m_over, dict):
            ex = m_over.get("PIPELINE_RULES_EXCLUDE") or []
            inc = m_over.get("PIPELINE_RULES_INCLUDE") or []
            if isinstance(ex, list):
                exclude = set(ex)
            if isinstance(inc, list):
                include = set(inc)
    if extra_excludes:
        exclude = exclude | extra_excludes
    return exclude, include


def _rule_runs(rule: _CompiledRule, exclude: "set[str] | frozenset[str]",
               include: "set[str] | frozenset[str]", language: "str | None") -> bool:
    """Whether _postprocess_text applies `rule` under this scoping: not
    excluded, not skipped for the language, and enabled or force-included."""
    if rule.name in exclude:
        return False
    if rule.languages and language and language not in rule.languages:
        return False
    return rule.enabled or rule.name in include


def _postprocess_text(text: str, model_name: "str | None" = None,
                       trace: "list | None" = None,
                       extra_excludes: "set[str] | None" = None,
                       ident=None,
                       language: "str | None" = None,
                       bound: "int | None" = None,
                       budget: "WalkBudget | None" = None) -> str:
    """Run the unified pipeline rule list on `text`. If `trace` is a list,
    each rule that changes the text, is skipped, or is force-included appends
    `(label_with_ordinal, before, after)` so the per-request log block can
    render a diff view; each text is kept to _TRACE_TEXT_MAX chars
    (_trace_row), and only the last _TRACE_MAX_ROWS rows keep their texts
    (_trace_push).

    The output is bounded by output_bound(len(text)): the rule that would
    pass it is not applied (traced "[OUTPUT BOUND — not applied]") and the
    remaining rules are skipped. `bound` tightens that bound further.
    A rule quarantined for running past the time limit (see "Time limit"
    below) is traced "[TIME LIMIT — not applied]"; the rules after it run.
    `budget` does too, for a caller walking many pieces of one text
    separately (one walk per word or per cue): every walk gets the same
    WalkBudget, is bounded by what is left of it and is charged the longest
    text it built.

    Per-model scoping (precedence top-down):
      1. PIPELINE_RULES_EXCLUDE — force-DISABLE for this model (highest priority).
      2. Language mismatch — rule.languages is non-empty AND the detected
         language is not in it → skip (even if force-INCLUDEd). When
         `language` is None (streaming partial before detection), no rule is
         skipped — safe default for ephemeral display text.
      3. PIPELINE_RULES_INCLUDE — force-ENABLE for this model, even if globally
         disabled.
      4. Otherwise inherit `rule.enabled` from the global PIPELINE_RULES list.

    _rule_runs applies this. `extra_excludes` adds slugs to skip on top of
    the per-model EXCLUDE: the /captures storage path uses it to produce a
    training-form transcript (cfg.CAPTURES_PIPELINE_RULES_EXCLUDE) without
    touching the /transcribe response. The per-model lists cannot share a
    slug (pydantic rejects that); extra_excludes can, and then the exclude
    wins over a per-model force-on.

    A rule with `repeat` on runs again on its own output (a regex-list card:
    all its entries) until a pass changes nothing, at most _REPEAT_MAX_PASSES
    passes; the scoping above decides for the card once.

    The terminal "trim-edges" step (filtered out of _COMPILED_RULES at
    rebuild time) runs as the always-last step here, gated by the same
    exclude set so a trainer can preserve trailing whitespace by adding
    the slug to CAPTURES_PIPELINE_RULES_EXCLUDE. The live /transcribe
    path applies an additional unconditional trim after the output
    wrappers, so per-model exclusion of trim-edges has no effect there.
    """
    exclude, include = _scoping(model_name, ident, extra_excludes)
    bound = output_bound(len(text)) if bound is None else min(bound, output_bound(len(text)))
    if budget is not None:
        bound = min(bound, budget.left)
    rules = _COMPILED_RULES
    terminal = (_TERMINAL_NAME, _TERMINAL_LABEL, _TERMINAL_CARD_NO)
    pool = _rule_helper._POOL
    if pool is not None:
        out = _walk_isolated(pool, text, rules, terminal, exclude, include,
                             model_name, language, bound, trace, budget)
        if out is not _IN_PROCESS:
            return out
    return _walk(text, rules, terminal, exclude, include, model_name, language,
                 bound, trace, budget, _skip_for(rules), None, logger.warning)


def _walk(text: str, rules: "list[_CompiledRule]", terminal: "tuple[str, str, int]",
          exclude: "set[str]", include: "set[str]", model_name: "str | None",
          language: "str | None", bound: int, trace: "list | None",
          budget: "WalkBudget | None", skip: "frozenset[int]", mark, warn) -> str:
    """The walk behind _postprocess_text, on an explicit rule set: this
    process's (_COMPILED_RULES) or, in a helper process, the one its parent
    sent. `skip` holds the indices into `rules` of the quarantined rows (see
    _quarantine); `mark` is the helper's progress slot, written with the
    index of each row before it runs, or None in-process; `warn` takes what
    the walk would log (a helper hands it to its parent to log)."""
    peak = len(text)
    if budget is not None:
        # Charged as the walk grows, so a rule that raises mid-walk leaves
        # what was built paid for.
        budget.left -= peak
    capped = False
    memo = [None, False, None] if trace is not None else None
    base = 0
    for card in _cards(rules):
        # A skipped card changes nothing, so it is traced once and left
        # after its first pass.
        for pass_no in range(1, _card_passes(card) + 1):
            card_in = text
            for idx, rule in enumerate(card, base):
                # Step number mirrors the /settings/pipeline card position
                # (`#P` / `#P.S`), NOT the flat index over the expanded
                # compiled list; the label of a repeat pass names the pass.
                ordinal = _rule_ordinal(rule.card_no, rule.sub_no)
                label = rule.label if pass_no == 1 else f"{rule.label} (pass {pass_no})"
                if not _rule_runs(rule, exclude, include, language):
                    # Trace why the rule was skipped, testing the reasons in
                    # _rule_runs' order so the label names the one that applied.
                    if trace is not None:
                        if rule.name in exclude:
                            why = f"EXCLUDED for {model_name}"
                        elif rule.languages and language and language not in rule.languages:
                            why = f"SKIPPED lang:{language} ∉ {sorted(rule.languages)}"
                        else:
                            why = "SKIPPED globally disabled"
                        _trace_push(trace, _trace_row(
                            f"{ordinal} {label} [{why}]", text, text, memo))
                    continue
                if idx in skip:
                    # Quarantined (_quarantine): it ran past the time limit
                    # on an earlier walk of this rule set.
                    if trace is not None and pass_no == 1:
                        _trace_push(trace, _trace_row(
                            f"{ordinal} {label} [TIME LIMIT — not applied]",
                            text, text, memo))
                    continue
                if mark is not None:
                    mark[0] = idx
                forced_in = rule.name in include
                before = text
                out = _apply_bounded(rule, before, bound)
                if out is None:
                    # This rule would pass the output bound (see below): it
                    # is not applied, and the pipeline stops on the text it
                    # was handed.
                    warn(
                        "[pipeline] output would exceed %d chars at rule %r — it and "
                        "the remaining rules skipped", bound, rule.name)
                    if trace is not None:
                        _trace_push(trace, _trace_row(
                            f"{ordinal} {label} [OUTPUT BOUND — not applied]",
                            before, before, memo))
                    capped = True
                    break
                text = out
                if budget is not None and len(text) > peak:
                    budget.left -= len(text) - peak
                    peak = len(text)
                if trace is not None:
                    # Force-included rule: tag the trace line so the admin sees
                    # the rule ran *because of* the per-model override, not the
                    # global state. Always emit (on its first pass) even when
                    # before == after, to make the override path visible.
                    if forced_in and not rule.enabled and (pass_no == 1 or before != text):
                        _trace_push(trace, _trace_row(
                            f"{ordinal} {label} [FORCED on for {model_name}]",
                            before, text, memo))
                    elif before != text:
                        _trace_push(trace, _trace_row(f"{ordinal} {label}", before, text, memo))
                # Output bound. Rules whose replacement template expands what
                # it matches compose: each one feeds the next, so a handful of
                # them can multiply a short transcript into gigabytes.
                # _apply_bounded refuses the rule that would pass it; this
                # length check is the backstop. No real rule set comes near it.
                if len(text) > bound:
                    warn(
                        "[pipeline] output exceeded %d chars at rule %r — remaining "
                        "rules skipped", bound, rule.name)
                    capped = True
                    break
            if capped or text == card_in:
                break
        if capped:
            break
        base += len(card)
    if mark is not None:
        mark[0] = -1
    terminal_name, terminal_label, terminal_card_no = terminal
    term_ordinal = _rule_ordinal(terminal_card_no)
    if terminal_name in exclude:
        if trace is not None:
            _trace_push(trace, _trace_row(
                f"{term_ordinal} {terminal_label} [EXCLUDED for {model_name}]",
                text, text, memo))
    else:
        before_trim = text
        text = text.lstrip(" \t\r").rstrip(" \t\r")
        if trace is not None and before_trim != text:
            _trace_push(trace, _trace_row(
                f"{term_ordinal} {terminal_label}", before_trim, text, memo))
    return text


# The streaming session formats its document utterance by utterance and never
# changes text it already sent. These two read the SAME rule selection as
# _postprocess_text (_scoping / _rule_runs), so what is held back and what is
# blamed always matches what the formatting actually runs.

def _warm_holdback_specs(
        compiled: "list[_CompiledRule]") -> "dict[tuple[int, ...], _seam_holdback.HoldSpec]":
    """The hold-back spec of every selection holdback_start makes with empty
    exclude/include, keyed by that selection (indices into `compiled`). Which
    maps run then depends only on the language: None runs every enabled map,
    a tagged language its own maps plus the untagged ones, and any other
    language ("\\x00" stands in for it) the untagged ones alone. Nothing to
    warm without a map row."""
    maps = [(i, r) for i, r in enumerate(compiled)
            if r.type == "callback:map" and r.map_lookup]
    if not maps:
        return {}
    langs: "set[str | None]" = {None, "\x00"}
    langs.update(lang for _i, r in maps for lang in r.languages)
    no_scope: "frozenset[str]" = frozenset()
    sels = {tuple(i for i, r in maps if _rule_runs(r, no_scope, no_scope, lang))
            for lang in langs}
    return {sel: _seam_holdback.build_spec(compiled[i].map_lookup for i in sel)
            for sel in sels}


@functools.lru_cache(maxsize=64)
def _holdback_spec(gen: int, sel: "tuple[int, ...]") -> "_seam_holdback.HoldSpec":
    # `gen` only keys the cache: a rebuild_caches() bumps _RULES_GEN, so a
    # spec built from an older rule set is never served again. `sel` holds
    # the _COMPILED_RULES indices of the callback:map rows the caller's
    # scoping and language run. Keying on those rows rather than on
    # (exclude, include, language) lets every caller that runs the same rows
    # share one spec: a spec on a large map takes long to build and holds
    # tens of MB, and a client controls its handshake language. A rebuild
    # between the caller's selection and this build can shift the indices;
    # such a spec lands under the stale gen, and the row checks keep it from
    # reading a non-map row.
    rules = _COMPILED_RULES
    return _seam_holdback.build_spec(
        rules[i].map_lookup for i in sel
        if i < len(rules) and rules[i].type == "callback:map" and rules[i].map_lookup)


def holdback_start(raw: str, *, model_name: "str | None" = None, ident=None,
                   language: "str | None" = None) -> int:
    """Index into `raw` from which live dictation holds the text back instead
    of formatting and sending it (len(raw) = nothing held) — the trailing
    words the active dictation maps would still join with the next utterance.
    See pipeline/seam_holdback.py."""
    exclude, include = _scoping(model_name, ident)
    # Read the gen BEFORE the rules: rebuild_caches rebinds _WARM_SPECS, then
    # _COMPILED_RULES, and bumps _RULES_GEN last, so a rebuild in between can
    # at worst file a newer rule set's spec under a gen nobody looks up again —
    # never old-rule indices under the live gen, nor an old warm spec for the
    # new rules.
    gen = _RULES_GEN
    rules = _COMPILED_RULES
    sel = tuple(i for i, rule in enumerate(rules)
                if rule.type == "callback:map" and rule.map_lookup
                and _rule_runs(rule, exclude, include, language))
    # `is None`, not `or`: a spec with no sequences is falsy yet still warm.
    spec = _WARM_SPECS.get((gen, sel))
    if spec is None:
        spec = _holdback_spec(gen, sel)
    return _seam_holdback.held_start(raw, spec)


def seam_culprit(prefix_raw: str, full_raw: str, *, model_name: "str | None" = None,
                 ident=None, language: "str | None" = None) -> str:
    """The rule after which formatting `full_raw` stops extending the
    formatting of its prefix `prefix_raw` — as '#P.S label' — for the
    streaming seam WARNING. Both texts are walked rule by rule with the same
    selection as _postprocess_text; edges are compared the way the terminal
    trim leaves them. Warning path only, so no caching — but the same output
    bound as _postprocess_text: the walk stops at the rule that would push
    either text past its output_bound (where the pipeline itself stops), so
    a compounding rule set cannot run on unbounded strings here."""
    exclude, include = _scoping(model_name, ident)
    rules = _COMPILED_RULES
    bounds = (output_bound(len(prefix_raw)), output_bound(len(full_raw)))
    pool = _rule_helper._POOL
    if pool is not None:
        out = _seam_isolated(pool, prefix_raw, full_raw, bounds, rules, exclude,
                             include, language)
        if out is not _IN_PROCESS:
            return out
    return _seam_walk(prefix_raw, full_raw, bounds, rules, exclude, include, language,
                      _skip_for(rules), None)


def _seam_walk(prefix_raw: str, full_raw: str, bounds: "tuple[int, int]",
               rules: "list[_CompiledRule]", exclude: "set[str]", include: "set[str]",
               language: "str | None", skip: "frozenset[int]", mark) -> str:
    """The walk behind seam_culprit on an explicit rule set; `bounds` are the
    two texts' output bounds, `skip` and `mark` as in _walk."""
    a, b = prefix_raw, full_raw
    ba, bb = bounds

    def extends(pa: str, pb: str) -> bool:
        return pb.lstrip(" \t\r").startswith(pa.lstrip(" \t\r").rstrip(" \t\r"))

    ok = extends(a, b)
    base = 0
    for card in _cards(rules):
        first, base = base, base + len(card)
        if not _rule_runs(card[0], exclude, include, language):
            continue
        # Both texts get the passes _postprocess_text gives each: a pass that
        # leaves one of them as it was leaves it so on every later pass too,
        # so walking both until neither changes equals two separate walks.
        for _pass in range(_card_passes(card)):
            a_in, b_in = a, b
            for idx, rule in enumerate(card, first):
                if idx in skip:
                    continue
                if mark is not None:
                    mark[0] = idx
                na, nb = _apply_bounded(rule, a, ba), _apply_bounded(rule, b, bb)
                if na is None or nb is None or len(na) > ba or len(nb) > bb:
                    return f"{_rule_ordinal(rule.card_no, rule.sub_no)} {rule.label} (output bound hit)"
                a, b = na, nb
                now = extends(a, b)
                if ok and not now:
                    return f"{_rule_ordinal(rule.card_no, rule.sub_no)} {rule.label}"
                ok = now
            if a == a_in and b == b_in:
                break
    return "none found" if ok else "raw text already diverges"


# ---------------------------------------------------------------------------
# Time limit: walks in a helper process
# ---------------------------------------------------------------------------
#
# A rule pattern is user-editable, `re` cannot be interrupted, and the
# save-time screen (pipeline/regex_guard.py) only predicts from a pattern's
# shape whether it can be slow. So with the helper pool running, a walk is one
# round trip to a helper process (pipeline/rule_helper.py) with a deadline:
#
#   * Scoping, the output bound and the WalkBudget are resolved here; the
#     helper gets plain values and sends back the text, the trace rows, what
#     to charge the budget and what to log. It runs the same _walk on the same
#     rules, so a walk that finishes in time gives what the in-process walk
#     gives.
#   * The helper's rules are this process's compiled rules, sent over the
#     connection (_shipment) and named by a token every request carries: a
#     helper that holds another rule set is sent the current one first, in
#     the same exchange, so no walk runs on stale rules and no helper reads
#     the config files for them.
#   * Before each rule the helper writes the rule's index into a shared slot.
#     A walk past its deadline has its helper killed, the rule in the slot
#     quarantined (_quarantine: skipped by every later walk of this rule set,
#     logged once, traced "[TIME LIMIT — not applied]") and is redone without
#     it, at most _WALK_ATTEMPTS times in all.
#   * No pool, no helper that can be started, or a rule set that cannot be
#     sent: the walk runs in this process as it always did (logged where it
#     is a fault). Helpers that are up but all held for the whole wait
#     (rule_helper.BUSY) are the slow rules' doing: the text is returned
#     without the rules applied, never walked in this process.
#
# In-process on purpose: holdback_start (no rule pattern runs there, only the
# map keys are read) and rebuild_caches (it compiles, it does not match).

# A walk's deadline: a floor plus an allowance per input character, capped.
# Measured in a helper, the factory rules take about 0.5 µs per character
# (median 0.16 / 1.1 / 10 / 26 ms at 200 / 2 000 / 20 000 / 50 000
# characters); with every core of the box busy at the same time the slowest
# walk seen took 20 / 18 / 88 / 256 ms. The deadline (2.0 / 2.4 / 6 / 12 s at
# those sizes) is 47x or more above that worst case, so only a rule that
# would not finish at all is cut: a quarantined rule changes what users get
# until the rules are next saved, and must not come from a busy box.
_DEADLINE_FLOOR_S = 2.0
_DEADLINE_PER_CHAR_S = 2e-4
_DEADLINE_MAX_S = 60.0
# How long a helper may take to compile a rule set it is sent.
_LOAD_TIMEOUT_S = 60.0
# Most attempts one walk makes in a helper. Each attempt that passes the
# deadline quarantines one rule, so a walk waits for at most this many
# deadlines; past them it returns its input, traced.
_WALK_ATTEMPTS = 3

# How many helpers one walk may lose (dead, not timed out) before it stops
# asking for another.
_LOST_HELPER_RETRIES = 6

# Returned by the isolated runners for "run it in this process".
_IN_PROCESS = object()
_NO_SKIP: "frozenset[int]" = frozenset()
# Guards _QUARANTINE and _SHIPPED updates (reads go without it).
_ISOLATION_LOCK = threading.Lock()
# (rule set, indices of its quarantined rows). The rule set is the
# _COMPILED_RULES list object itself: a rebuild binds a new list, which ends
# the quarantine without anything to clear.
_QUARANTINE: "tuple[list | None, frozenset[int]]" = (None, frozenset())
# The rule set last prepared for the helpers: (rules, terminal, token, load
# request), the request None when the set cannot be sent (_shipment).
_SHIPPED: "tuple | None" = None
_SHIP_TOKENS = itertools.count(1)
# Helper process only: (token, rules, terminal) of the rule set it holds.
_HELPER_SET: "tuple | None" = None


def walk_deadline(n: int) -> float:
    """Seconds a walk over `n` input characters may take in a helper. The
    constants are read at call time."""
    return min(_DEADLINE_MAX_S, _DEADLINE_FLOOR_S + _DEADLINE_PER_CHAR_S * n)


def _skip_for(rules: "list[_CompiledRule]") -> "frozenset[int]":
    """The quarantined row indices of `rules` (empty for any other rule set
    than the one the quarantine was made for)."""
    q_rules, q = _QUARANTINE
    return q if q_rules is rules else _NO_SKIP


def _quarantine(rules: "list[_CompiledRule]", idx: int, why: str) -> None:
    """Skip row `idx` of `rules` in every later walk of this rule set, and
    log it once. The quarantine lasts until the rules are compiled again (any
    save of the pipeline rules, or a restart)."""
    global _QUARANTINE
    if not 0 <= idx < len(rules):
        return
    with _ISOLATION_LOCK:
        q = _skip_for(rules)
        if idx in q:
            return
        _QUARANTINE = (rules, q | {idx})
    rule = rules[idx]
    logger.warning(
        "[pipeline] rule %s %r (%s) %s — it is skipped from now on, until the "
        "pipeline rules are saved again or the server restarts",
        _rule_ordinal(rule.card_no, rule.sub_no), _log_safe(rule.name),
        _log_safe(rule.label), why)


def _rule_spec(rule: _CompiledRule) -> tuple:
    """`rule` as plain values a helper rebuilds it from (_rule_from_spec).
    ValueError for a replacer that cannot be rebuilt there (a callable that
    is neither one of this module's nor built from `rule.source`)."""
    payload = rule.payload
    if isinstance(payload, str):
        how, src = "template", payload
    elif payload is _dedup_callback:
        how, src = "dedup", None
    elif payload is _upper_callback:
        how, src = "upper", None
    elif rule.type == "callback:map" and isinstance(rule.source, dict):
        how, src = "map", rule.source
    elif rule.type == "callback:lowercase-wordlist" and isinstance(rule.source, frozenset):
        how, src = "wordlist", rule.source
    else:
        raise ValueError(f"rule {rule.name!r} has a replacer a helper cannot rebuild")
    return (how, src, rule.pattern.pattern, rule.pattern.flags, rule.name, rule.label,
            rule.type, rule.enabled, rule.card_no, rule.sub_no, rule.languages,
            rule.repeat, rule.max_growth, rule.template)


def _rule_from_spec(spec: tuple) -> _CompiledRule:
    """Helper side of _rule_spec. The pattern is compiled from the parent's
    pattern source and flags, and a map's pattern must come out as the
    parent's did: the two processes run the same module, so anything else
    is refused rather than run."""
    (how, src, source, flags, name, label, rtype, enabled, card_no, sub_no,
     languages, repeat, max_growth, template) = spec
    payload: object
    if how == "map":
        pattern, payload, _lookup = _dictation_map.compile_map(src)
    else:
        pattern = re.compile(source, flags)
        if how == "template":
            payload = src
        elif how == "dedup":
            payload = _dedup_callback
        elif how == "upper":
            payload = _upper_callback
        elif how == "wordlist":
            payload = _make_lowercase_wordlist_replacer(src)
        else:
            raise ValueError(f"rule {name!r}: unknown replacer kind {how!r}")
    if pattern.pattern != source or pattern.flags != flags:
        raise ValueError(f"rule {name!r} compiles differently in the helper")
    return _CompiledRule(name, label, rtype, pattern, payload, enabled, card_no, sub_no,
                         languages, repeat, max_growth=max_growth, template=template)


def _shipment(rules: "list[_CompiledRule]",
              terminal: "tuple[str, str, int]") -> "tuple[int, bytes] | None":
    """(token, load request) for this rule set, built once per set; None when
    it cannot be sent, and its walks run in-process."""
    global _SHIPPED
    shipped = _SHIPPED
    if shipped is None or shipped[0] is not rules or shipped[1] != terminal:
        with _ISOLATION_LOCK:
            shipped = _SHIPPED
            if shipped is None or shipped[0] is not rules or shipped[1] != terminal:
                token = next(_SHIP_TOKENS)
                try:
                    request = pickle.dumps(
                        ("load", token, terminal, [_rule_spec(r) for r in rules]), 5)
                except Exception as exc:  # noqa: BLE001 - unsendable -> in-process
                    request = None
                    logger.warning(
                        "[pipeline] the rules cannot be sent to a helper process (%s) — "
                        "they run inside the server process without a time limit", exc)
                shipped = _SHIPPED = (rules, terminal, token, request)
    return None if shipped[3] is None else (shipped[2], shipped[3])


def _unshippable(rules: "list[_CompiledRule]", terminal: "tuple[str, str, int]",
                 why: str) -> None:
    """Stop sending this rule set to helpers: one refused or failed to load
    it. Its walks run in-process."""
    global _SHIPPED
    with _ISOLATION_LOCK:
        shipped = _SHIPPED
        if (shipped is None or shipped[0] is not rules or shipped[1] != terminal
                or shipped[3] is None):
            return
        _SHIPPED = (rules, terminal, shipped[2], None)
    logger.warning(
        "[pipeline] a helper process could not load the rules (%s) — they run "
        "inside the server process without a time limit", _log_safe(why))


def _in_helper(pool, rules: "list[_CompiledRule]", terminal: "tuple[str, str, int]",
               n_chars: int, request):
    """Run one request in a helper under walk_deadline(n_chars).

    `request(token, skip)` builds the request tuple; it is rebuilt for every
    attempt because a timed-out attempt adds to the quarantine. Returns the
    helper's reply tuple, _IN_PROCESS when the caller has to run the walk
    itself, or None when no attempt finished (the helpers all held for the
    whole wait included). Never raises."""
    shipment = _shipment(rules, terminal)
    if shipment is None:
        return _IN_PROCESS
    token, load_request = shipment
    deadline = walk_deadline(n_chars)
    died_at = None
    in_rule = False
    stale = lost = attempts = 0
    # One wait for a free helper per walk, shared by its attempts: a thread
    # is parked for at most that long on top of the walk's deadlines.
    wait_left = _rule_helper._ACQUIRE_TIMEOUT_S
    while attempts < _WALK_ATTEMPTS:
        asked = time.monotonic()
        helper = pool.acquire(max(0.0, wait_left))
        wait_left -= time.monotonic() - asked
        if helper is _rule_helper.BUSY:
            # Slow rules hold every helper. Walked here instead, the same
            # rules would hold the whole server, with no deadline at all.
            return None
        if helper is None:
            # No helper can be had (pool shut down under this walk, or none
            # startable). A walk that already lost a helper inside a rule is
            # not redone here: that rule may be one that never finishes.
            return None if in_rule else _IN_PROCESS
        loaded = helper.token == token
        try:
            if not loaded:
                reply = pickle.loads(_rule_helper.call(helper, load_request, _LOAD_TIMEOUT_S))
                if reply[0] != "ok":
                    pool.release(helper)
                    _unshippable(rules, terminal, str(reply[-1]))
                    return _IN_PROCESS
                helper.token = token
                loaded = True
            helper.mark[0] = -1
            reply = pickle.loads(_rule_helper.call(
                helper, pickle.dumps(request(token, _skip_for(rules)), 5), deadline))
        except _rule_helper.HelperTimeout:
            at = helper.mark[0]
            pool.discard(helper)
            if not loaded:
                _unshippable(rules, terminal, f"no answer in {_LOAD_TIMEOUT_S:g} s")
                return _IN_PROCESS
            attempts += 1
            if at >= 0:
                in_rule = True
                _quarantine(rules, at, f"ran past the {deadline:.3g} s time limit")
            else:
                logger.warning(
                    "[pipeline] a rule walk passed its %.3g s time limit outside "
                    "any rule", deadline)
            continue
        except _rule_helper.HelperGone as exc:
            # A helper that died between walks (or was killed from outside)
            # is replaced and the walk redone. One that dies twice in the
            # same rule of this walk is that rule's doing.
            at = helper.mark[0] if loaded else -1
            pool.discard(helper)
            in_rule = in_rule or at >= 0
            if at >= 0 and at == died_at:
                attempts += 1
                _quarantine(rules, at, "ended its helper process twice")
            else:
                lost += 1
                logger.warning("[pipeline] a rule helper process was lost (%s) — "
                               "replaced", exc)
                if lost > _LOST_HELPER_RETRIES:
                    # Helpers do not stay up. Outside any rule that is the
                    # platform's doing and the walk runs here; inside one it
                    # may be the rules', which then must not run here either.
                    return None if in_rule else _IN_PROCESS
            died_at = at
            continue
        except Exception as exc:  # noqa: BLE001 - plumbing fault -> in-process
            pool.discard(helper)
            logger.warning("[pipeline] rule helper request failed (%r) — this walk "
                           "runs inside the server process", exc)
            return _IN_PROCESS
        if reply[0] == "stale":
            # The helper lost its rule set (it cannot, short of a bug): send
            # it again, but not forever.
            helper.token = None
            pool.release(helper)
            stale += 1
            if stale > 2:
                return _IN_PROCESS
            continue
        pool.release(helper)
        if reply[0] == "error":
            logger.warning("[pipeline] rule helper request failed (%s) — this walk "
                           "runs inside the server process", _log_safe(str(reply[-1])))
            return _IN_PROCESS
        return reply
    return None


def _walk_isolated(pool, text: str, rules: "list[_CompiledRule]",
                   terminal: "tuple[str, str, int]", exclude: "set[str]",
                   include: "set[str]", model_name: "str | None",
                   language: "str | None", bound: int, trace: "list | None",
                   budget: "WalkBudget | None"):
    """_walk in a helper process: its result, or _IN_PROCESS. The helper's
    trace rows are pushed onto `trace` here (_trace_push, so the window over
    the caller's whole list is the in-process one), its warnings logged here
    and `budget` charged here."""
    limits = (_REPEAT_MAX_PASSES, _TRACE_TEXT_MAX, _TRACE_MAX_ROWS)
    reply = _in_helper(
        pool, rules, terminal, len(text),
        lambda token, skip: ("walk", token, text, exclude, include, model_name,
                             language, bound, trace is not None, skip, limits))
    if reply is _IN_PROCESS:
        return _IN_PROCESS
    if reply is None:
        # Out of attempts (or of helpers: all held, or gone under a walk
        # inside a rule): the text goes on as it came in. Charged like a
        # walk that changed nothing.
        logger.warning(
            "[pipeline] a rule walk did not finish in a helper process — the "
            "text is returned without the rules applied")
        if budget is not None:
            budget.left -= len(text)
        if trace is not None:
            _trace_push(trace, _trace_row(
                "pipeline [TIME LIMIT — rules not applied]", text, text))
        return text
    status, value, rows, charged, logs = reply
    for fmt, args in logs:
        logger.warning(fmt, *args)
    if budget is not None:
        budget.left -= charged
    if trace is not None:
        for row in rows:
            _trace_push(trace, row)
    if status == "raise":
        raise value
    return value


def _seam_isolated(pool, prefix_raw: str, full_raw: str, bounds: "tuple[int, int]",
                   rules: "list[_CompiledRule]", exclude: "set[str]",
                   include: "set[str]", language: "str | None"):
    """_seam_walk in a helper process: its result, or _IN_PROCESS."""
    # The terminal step plays no part in the seam walk, but the rule set a
    # helper holds is named together with it.
    terminal = (_TERMINAL_NAME, _TERMINAL_LABEL, _TERMINAL_CARD_NO)
    limits = (_REPEAT_MAX_PASSES, _TRACE_TEXT_MAX, _TRACE_MAX_ROWS)
    reply = _in_helper(
        pool, rules, terminal, len(prefix_raw) + len(full_raw),
        lambda token, skip: ("seam", token, prefix_raw, full_raw, bounds, exclude,
                             include, language, skip, limits))
    if reply is _IN_PROCESS:
        return _IN_PROCESS
    if reply is None:
        return "not determined (time limit hit)"
    status, value = reply[0], reply[1]
    if status == "raise":
        raise value
    return value


def _helper_handle(request: bytes, mark) -> bytes:
    """Helper process: answer one request of _in_helper. Every outcome is a
    reply; nothing a rule does may end the serving loop."""
    global _HELPER_SET, _REPEAT_MAX_PASSES, _TRACE_TEXT_MAX, _TRACE_MAX_ROWS
    try:
        req = pickle.loads(request)
        op = req[0]
        if op == "load":
            _op, token, terminal, specs = req
            # Dropped first: a failed load must not leave the previous set
            # answering under a token the parent no longer uses.
            _HELPER_SET = None
            _HELPER_SET = (token, [_rule_from_spec(spec) for spec in specs], terminal)
            return pickle.dumps(("ok",), 5)
        held = _HELPER_SET
        if held is None or held[0] != req[1]:
            return pickle.dumps(("stale",), 5)
        _token, rules, terminal = held
        if op == "walk":
            (_op, _tok, text, exclude, include, model_name, language, bound,
             want_trace, skip, limits) = req
            # The parent's values (tests lower them); read by _card_passes
            # and the trace helpers.
            _REPEAT_MAX_PASSES, _TRACE_TEXT_MAX, _TRACE_MAX_ROWS = limits
            trace: "list | None" = [] if want_trace else None
            logs: list = []
            # Starts at 0 and is charged by _walk: what it is down by is
            # what the parent's budget owes.
            charge = WalkBudget.__new__(WalkBudget)
            charge.left = 0
            try:
                out = _walk(text, rules, terminal, exclude, include, model_name,
                            language, bound, trace, charge, skip, mark,
                            lambda fmt, *args: logs.append((fmt, args)))
                reply = ("ok", out, trace, -charge.left, logs)
            except Exception as exc:  # noqa: BLE001 - the parent re-raises it
                reply = ("raise", _sendable(exc), trace, -charge.left, logs)
        elif op == "seam":
            (_op, _tok, prefix_raw, full_raw, bounds, exclude, include, language,
             skip, limits) = req
            _REPEAT_MAX_PASSES, _TRACE_TEXT_MAX, _TRACE_MAX_ROWS = limits
            try:
                reply = ("ok", _seam_walk(prefix_raw, full_raw, bounds, rules, exclude,
                                          include, language, skip, mark))
            except Exception as exc:  # noqa: BLE001 - the parent re-raises it
                reply = ("raise", _sendable(exc))
        else:
            reply = ("error", f"unknown request {op!r}")
        mark[0] = -1
        return pickle.dumps(reply, 5)
    except Exception as exc:  # noqa: BLE001 - reported, the loop goes on
        return pickle.dumps(("error", repr(exc)), 5)


def _sendable(exc: Exception) -> Exception:
    """`exc` if it survives a pickle round trip, else a RuntimeError naming
    it."""
    try:
        pickle.loads(pickle.dumps(exc, 5))
        return exc
    except Exception:  # noqa: BLE001 - any pickling fault
        return RuntimeError(f"{type(exc).__name__}: {exc}")


# Compile the configured rules once at import (callers never see an empty
# list). A helper process is sent its rules instead.
if not _rule_helper.IN_HELPER:
    rebuild_caches()


def _reset_for_tests() -> None:
    """Drop the hold-back spec cache and rebind _REBUILD_LOCK, so a lock a
    previous test's rebuild thread still holds cannot block the next test.
    The compiled rules themselves are rebuilt by the app_module fixture
    (rebuild_caches after reloading config). Also stops the helper pool a
    test (or a TestClient lifespan) started, and forgets the quarantine and
    the prepared rule set."""
    global _REBUILD_LOCK, _ISOLATION_LOCK, _QUARANTINE, _SHIPPED
    _holdback_spec.cache_clear()
    _WARM_SPECS.clear()
    _REBUILD_LOCK = threading.Lock()
    _rule_helper.shutdown()
    _ISOLATION_LOCK = threading.Lock()
    _QUARANTINE = (None, frozenset())
    _SHIPPED = None
