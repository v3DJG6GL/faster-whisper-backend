"""The post-processing rules engine: compiles cfg.PIPELINE_RULES into
_COMPILED_RULES (rebuild_caches — at import and again on an admin save that
touches a CACHE_REBUILD_FIELDS key) and applies them to a transcript
(_postprocess_text), plus the live-dictation seam helpers holdback_start /
seam_culprit that read the same rule selection. Callers go through the module
attribute (``pl_engine._postprocess_text(...)``): rebuild_caches rebinds
_COMPILED_RULES, so a ``from ... import`` of it would go stale.
"""
import functools
import logging
import re
import threading

from faster_whisper_backend.pipeline import dictation_map as _dictation_map
from faster_whisper_backend.pipeline import seam_holdback as _seam_holdback
from faster_whisper_backend.settings import config as cfg

logger = logging.getLogger("whisper-api")


# =============================================================================
# Text post-processing pipeline — unified rules list (cfg.PIPELINE_RULES)
# =============================================================================
# A single ordered list of rules is applied to the joined transcript. Each
# rule is one of: "regex-list" (ordered batch of pattern→replacement entries), "callback:lowercase-wordlist"
# (smart German non-noun lowercaser), "callback:map" (dictation word→symbol
# map), "callback:dedup" (collapse adjacent punctuation runs), "callback:upper"
# (capitalize after sentence terminator), or "terminal" (final lstrip+rstrip;
# always last). See config.py:PIPELINE_RULES for the canonical seeded list and
# settings/schema.py for the Pydantic schema.
#
# rebuild_caches() compiles each rule's regex pattern once at module load and
# again on admin WebUI save of a cache_rebuild field (settings/schema.py
# CACHE_REBUILD_FIELDS).
# Disabled rules and skipped types (terminal, empty patterns) are filtered
# out of the compiled list — the runtime walker is just a tight for-loop.

from dataclasses import dataclass, field as _dc_field


@dataclass(frozen=True)
class _CompiledRule:
    """One row of the compiled pipeline. `payload` carries type-specific data:
      regex-list (per entry)      → replacement string
      callback:lowercase-wordlist → frozenset[str] of lowercase words
      callback:map                → dict[str_lower, str] lookup
      callback:dedup              → None (callback hardcoded)
      callback:upper              → None (callback hardcoded)
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
    # callback:map only: the effective lower-cased key → value lookup (ß keys'
    # ss variants included, see pipeline/dictation_map.py). The streaming
    # hold-back builds its spec from it. Kept out of eq/hash: a dict is
    # unhashable and the replacer already carries the same data.
    map_lookup: "dict[str, str] | None" = _dc_field(default=None, compare=False)


_COMPILED_RULES: list[_CompiledRule] = []
# Bumped by every rebuild_caches(): the key under which per-rule-set derived
# data (the streaming hold-back spec) is cached.
_RULES_GEN: int = 0
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
# Ceiling on the pipeline's output, checked after every rule. A 30-minute
# transcript is a few hundred KB, so 4 M chars only ever trips on a rule set
# that expands its own input (the pipeline runs on admin-editable regexes).
_POSTPROCESS_MAX_CHARS = 4_000_000


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
    the entire match. Default seeded pattern produces ([.?!]\\s+|\\n+\\s*)
    + ([a-zäöüß])."""
    try:
        g1, g2 = match.group(1), match.group(2)
    except IndexError:
        return match.group(0).upper()
    if g1 is None or g2 is None:
        return match.group(0)
    return g1 + g2.upper()


def _make_lowercase_wordlist_replacer(wordlist: frozenset):
    """Returns a regex-sub callback that strips the matched terminator and
    lowercases group(2) IFF (group(2)+group(3)).lower() is in `wordlist`.
    The default seeded pattern produces three groups:
      group(1) = whitespace between terminator and next word
      group(2) = first letter of the next word
      group(3) = rest of the next word
    If the user's pattern has fewer than 3 groups we degrade to plain strip.
    """
    def replace(m: "re.Match[str]") -> str:
        try:
            ws, first, rest = m.group(1), m.group(2), m.group(3)
        except IndexError:
            return ""
        if ws is None or first is None or rest is None:
            return m.group(0)
        if (first + rest).lower() in wordlist:
            return ws + first.lower() + rest
        return ws + first + rest
    return replace


# Serialises rebuild_caches: it runs in asyncio.to_thread (apply.py), and not
# every hot-apply caller holds rules_lock(). Without it two rebuild threads
# could finish out of order — a stale compile overwriting the newer one — and
# the unsynchronised `_RULES_GEN += 1` could store the same gen twice. Every
# cfg setattr happens before its own rebuild thread starts, so the thread that
# compiles last also reads the latest cfg.
_REBUILD_LOCK = threading.Lock()


def rebuild_caches() -> None:
    """(Re)compile every rule in cfg.PIPELINE_RULES into _COMPILED_RULES.

    Called once at module load (bottom of this module) and again by the admin
    WebUI after a config change to PIPELINE_RULES (CACHE_REBUILD_FIELDS).

    The terminal row is filtered out (it runs as the implicit final trim,
    not via the walker). Globally-DISABLED rules are still compiled — the
    runtime filter consults `rule.enabled` per-call so per-model
    PIPELINE_RULES_INCLUDE can force-enable a globally-disabled rule. Rules
    with invalid regex are logged + skipped (the save-time validator
    usually catches these, but a hand-edited config.py or a runtime
    catastrophic-backtracking case might surface here).
    """
    with _REBUILD_LOCK:
        _rebuild_caches_locked()


def _rebuild_caches_locked() -> None:
    global _COMPILED_RULES, _TERMINAL_NAME, _TERMINAL_LABEL, _TERMINAL_CARD_NO, _RULES_GEN
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

        # regex-list: expand each entry into its own _CompiledRule row, all
        # sharing the card's name + enabled + languages (so per-model
        # EXCLUDE/INCLUDE and the global toggle flip the whole card together).
        # Entries run in list order — NO longest-first sort. A bad entry is
        # skipped (not the whole card); an empty-pattern entry is a no-op.
        # Each row is a plain string-replacement sub, exactly like the retired
        # single `regex` type.
        if rtype == "regex-list":
            rname = rule.get("name", "?")
            rlabel = rule.get("label", rname)
            for sub_no, entry in enumerate(rule.get("entries", []) or [], start=1):
                epat = entry.get("pattern", "")
                if not epat:
                    continue
                try:
                    ecre = re.compile(epat)
                except re.error as e:
                    logger.warning("[pipeline] rule %r entry has invalid regex "
                                   "(%s) — skipping entry", rname, e)
                    continue
                # sub_no is the entry's row in the card editor (the enumerate index
                # advances across skipped empty/bad entries above), so the trace
                # number `#card_no.sub_no` lines up with what the admin sees.
                compiled.append(_CompiledRule(
                    rname, f"{rlabel} · {entry.get('label') or epat}",
                    "regex-list", ecre, entry.get("replacement", "") or "",
                    rule_enabled, card_no, sub_no, rule_langs))
            continue

        map_lookup = None
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
                # Pre-bind the per-rule replacer once at compile time. _apply_rule
                # then becomes a uniform pattern.sub(payload, text) for every rule
                # type — no closure allocation on the hot path (twice per request).
                cre, payload, map_lookup = _dictation_map.compile_map(m)
            else:
                pattern = rule.get("pattern", "")
                if not pattern:
                    # Empty pattern on a callback rule → skip (no-op).
                    continue
                cre = re.compile(pattern)
                if rtype == "callback:lowercase-wordlist":
                    wordlist = frozenset(w.lower() for w in (rule.get("wordlist", []) or []))
                    payload = _make_lowercase_wordlist_replacer(wordlist)
                elif rtype == "callback:dedup":
                    payload = _dedup_callback
                elif rtype == "callback:upper":
                    payload = _upper_callback
                else:
                    logger.warning("[pipeline] unknown rule type %r — skipping", rtype)
                    continue
        except re.error as e:
            logger.warning("[pipeline] rule %r has invalid regex (%s) — skipping",
                           rule.get("name"), e)
            continue
        compiled.append(_CompiledRule(rule.get("name", "?"),
                                       rule.get("label", rule.get("name", "?")),
                                       rtype, cre, payload, rule_enabled, card_no,
                                       languages=rule_langs, map_lookup=map_lookup))
    _COMPILED_RULES = compiled
    _TERMINAL_NAME = terminal_name
    _TERMINAL_LABEL = terminal_label
    _TERMINAL_CARD_NO = terminal_card_no
    _RULES_GEN += 1


def _apply_rule(rule: _CompiledRule, text: str) -> str:
    """Dispatch on rule type. Hot path — payload is pre-bound at
    rebuild_caches() time (a replacement string for `regex-list`, a pre-built
    callable for every callback:* type), so every type collapses to a
    single pattern.sub call with no per-request closure allocation."""
    return rule.pattern.sub(rule.payload, text)  # type: ignore[arg-type]



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
                       language: "str | None" = None) -> str:
    """Run the unified pipeline rule list on `text`. If `trace` is a list,
    each rule that changes the text appends `(label_with_ordinal, before, after)`
    so the per-request log block can render a diff view.

    Per-model scoping (precedence top-down):
      1. PIPELINE_RULES_EXCLUDE — force-DISABLE for this model (highest priority).
      2. Language mismatch — rule.languages is non-empty AND the detected
         language is not in it → skip (even if force-INCLUDEd). When
         `language` is None (streaming partial before detection), no rule is
         skipped — safe default for ephemeral display text.
      3. PIPELINE_RULES_INCLUDE — force-ENABLE for this model, even if globally
         disabled.
      4. Otherwise inherit `rule.enabled` from the global PIPELINE_RULES list.

    Effective:  (rule.enabled AND slug NOT in EXCLUDE) OR (slug IN INCLUDE),
    AND (rule.languages is empty OR language matches or is unknown).
    A rule cannot appear in both lists — pydantic validator rejects that.

    `extra_excludes` is an additional set of rule slugs to skip on top of
    the per-model EXCLUDE. Used by the /captures storage path to produce
    a training-form transcript (cfg.CAPTURES_PIPELINE_RULES_EXCLUDE) while
    leaving the runtime /transcribe response untouched. Rules in
    `extra_excludes` are skipped even when they appear in INCLUDE — the
    captures-specific intent overrides the per-model force-on.

    The terminal "trim-edges" step (filtered out of _COMPILED_RULES at
    rebuild time) runs as the always-last step here, gated by the same
    exclude set so a trainer can preserve trailing whitespace by adding
    the slug to CAPTURES_PIPELINE_RULES_EXCLUDE. The live /transcribe
    path applies an additional unconditional trim after the output
    wrappers, so per-model exclusion of trim-edges has no effect there.
    """
    exclude, include = _scoping(model_name, ident, extra_excludes)
    for rule in _COMPILED_RULES:
        # Step number mirrors the /settings/pipeline card position (`#P` / `#P.S`),
        # NOT the flat index over the expanded compiled list.
        ordinal = _rule_ordinal(rule.card_no, rule.sub_no)
        if not _rule_runs(rule, exclude, include, language):
            # When tracing, surface the skip so the log explains why a rule
            # didn't run — the reasons in _rule_runs' order: force-EXCLUDE
            # wins outright (admin explicitly turned this off), then the
            # language, then globally disabled and not force-included.
            if trace is not None:
                if rule.name in exclude:
                    why = f"EXCLUDED for {model_name}"
                elif rule.languages and language and language not in rule.languages:
                    why = f"SKIPPED lang:{language} ∉ {sorted(rule.languages)}"
                else:
                    why = "SKIPPED globally disabled"
                trace.append((f"{ordinal} {rule.label} [{why}]", text, text))
            continue
        forced_in = rule.name in include
        before = text
        text = _apply_rule(rule, before)
        if trace is not None:
            # Force-included rule: tag the trace line so the admin sees the
            # rule ran *because of* the per-model override, not the global
            # state. Always emit even when before == after, to make the
            # override path visible.
            if forced_in and not rule.enabled:
                trace.append(
                    (f"{ordinal} {rule.label} [FORCED on for {model_name}]",
                     before, text)
                )
            elif before != text:
                trace.append((f"{ordinal} {rule.label}", before, text))
        # Absolute output bound. Rules whose replacement template expands what
        # it matches compose: each one feeds the next, so a handful of them can
        # multiply a short transcript into gigabytes. Stop the pipeline instead
        # and keep what we have — no real transcript comes near this size.
        if len(text) > _POSTPROCESS_MAX_CHARS:
            logger.warning(
                "[pipeline] output exceeded %d chars at rule %r — remaining "
                "rules skipped", _POSTPROCESS_MAX_CHARS, rule.name)
            break
    term_ordinal = _rule_ordinal(_TERMINAL_CARD_NO)
    if _TERMINAL_NAME in exclude:
        if trace is not None:
            trace.append(
                (f"{term_ordinal} {_TERMINAL_LABEL} [EXCLUDED for {model_name}]",
                 text, text)
            )
    else:
        before_trim = text
        text = text.lstrip(" \t\r").rstrip(" \t\r")
        if trace is not None and before_trim != text:
            trace.append((f"{term_ordinal} {_TERMINAL_LABEL}", before_trim, text))
    return text


# ---- live-dictation seam helpers (streaming/session.py) ----------------------
# The streaming session formats its document utterance by utterance and never
# changes text it already sent. These two read the SAME rule selection as
# _postprocess_text (_scoping / _rule_runs), so what is held back and what is
# blamed always matches what the formatting actually runs.

@functools.lru_cache(maxsize=64)
def _holdback_spec(gen: int, exclude: "frozenset[str]", include: "frozenset[str]",
                   language: "str | None") -> "_seam_holdback.HoldSpec":
    # `gen` only keys the cache: a rebuild_caches() bumps _RULES_GEN, so a
    # spec built from an older rule set is never served again.
    return _seam_holdback.build_spec(
        rule.map_lookup for rule in _COMPILED_RULES
        if rule.type == "callback:map" and rule.map_lookup
        and _rule_runs(rule, exclude, include, language))


def holdback_start(raw: str, *, model_name: "str | None" = None, ident=None,
                   language: "str | None" = None) -> int:
    """Index into `raw` from which live dictation holds the text back instead
    of formatting and sending it (len(raw) = nothing held) — the trailing
    words the active dictation maps would still join with the next utterance.
    See pipeline/seam_holdback.py."""
    exclude, include = _scoping(model_name, ident)
    spec = _holdback_spec(_RULES_GEN, frozenset(exclude), frozenset(include), language)
    return _seam_holdback.held_start(raw, spec)


def seam_culprit(prefix_raw: str, full_raw: str, *, model_name: "str | None" = None,
                 ident=None, language: "str | None" = None) -> str:
    """The rule after which formatting `full_raw` stops extending the
    formatting of its prefix `prefix_raw` — as '#P.S label' — for the
    streaming seam WARNING. Both texts are walked rule by rule with the same
    selection as _postprocess_text; edges are compared the way the terminal
    trim leaves them. Warning path only: no caching, no output bound."""
    exclude, include = _scoping(model_name, ident)
    a, b = prefix_raw, full_raw

    def extends(pa: str, pb: str) -> bool:
        return pb.lstrip(" \t\r").startswith(pa.lstrip(" \t\r").rstrip(" \t\r"))

    ok = extends(a, b)
    for rule in _COMPILED_RULES:
        if not _rule_runs(rule, exclude, include, language):
            continue
        a, b = _apply_rule(rule, a), _apply_rule(rule, b)
        now = extends(a, b)
        if ok and not now:
            return f"{_rule_ordinal(rule.card_no, rule.sub_no)} {rule.label}"
        ok = now
    return "none found" if ok else "raw text already diverges"


# Compile the configured rules once at import (callers never see an empty list).
rebuild_caches()


def _reset_for_tests() -> None:
    """Drop the hold-back spec cache. The compiled rules themselves are rebuilt
    by the app_module fixture (rebuild_caches after reloading config)."""
    global _REBUILD_LOCK
    _holdback_spec.cache_clear()
    _REBUILD_LOCK = threading.Lock()
