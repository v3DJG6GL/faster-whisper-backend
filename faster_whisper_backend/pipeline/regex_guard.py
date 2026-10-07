"""Out-of-process guard for user-authored pipeline regexes.

A catastrophic-backtracking pattern (e.g. ``(.*a)+$``) makes ``re.sub`` run
effectively forever, and CPython's regex engine CANNOT be interrupted
in-process: the match holds the GIL and a Python thread can't be killed. So an
in-process guard can reject the *save* but still leaves a thread pinning a CPU
core until the process restarts.

This module runs the ``re.sub`` probes in a SEPARATE process that the parent
KILLS on timeout (``subprocess`` ``timeout=`` -> ``proc.kill()``), which truly
frees the CPU. That lets any user the admin has granted a tag keep editing and
adding regex rules (web ``/quick-config`` and the ``faster-whisper-frontend``
``/v1/pipeline-rules`` editor) without the SAVE-time probe pinning a CPU core.

Two things run before a pattern is accepted: a structural screen that refuses
a repeat nested inside a repeated group (``_nested_repetition`` — the shape
whose match time is exponential, which no timed probe can reliably outrun),
and the timed probes themselves, run against the German fixture AND a few
short repetitive worst-case inputs (``_ADVERSARIAL``) so a pattern that is
fast on prose but slow on a long unpunctuated run is caught too.

Two of those probes exist because fixed fixtures alone are not enough:

  * a per-pattern SYNTHETIC run (``_synthetic_fixture``) — the fixed fixtures
    contain no `-`, `n`, `d`, `m`, `k` or `f`, so a pattern built around any of
    those matches nothing in any probe and returns in microseconds however
    explosive it is;
  * the CHAINED fixture (``_chain_advance``) — entry N is probed against the
    fixture as entries 0..N-1 have already rewritten it, because an earlier
    entry can MANUFACTURE the input that detonates a later one (`(Wetter)` ->
    60 dashes, then `(-+)(-+)(-+)…#` on the result: eight sibling quantified
    groups, no nesting for the structural screen to see, polynomial blowup).

A pattern crafted to blow up only on some other input shape can still pass and
hang the MAIN process at match time (rule application is in-process re.sub) —
accepted residual risk for a rule surface that is user-editable by design.

Two roles, one file:
  * parent  ->  ``regex_guard.validate(checks)`` (imported from the package)
  * child   ->  this file run as a script, ``[sys.executable, _SELF]`` (reads a
                JSON list of ``[pattern, replacement]`` pairs on stdin, writes a
                JSON verdict on stdout). Keep it stdlib-only: the child runs
                outside the package with this file's directory
                (faster_whisper_backend/pipeline/) as sys.path[0], so no
                sibling module there may shadow a stdlib name.
"""
from __future__ import annotations

import os

# ~1 KB realistic fixture (mirrors the former in-process guard's fixture).
FIXTURE = "Hallo. Wie geht's? 10.23 Uhr! Bitte. " * 32

# Whole-validation wall-clock budget: _GUARD_TIMEOUT plus a per-entry
# allowance, capped at _GUARD_TIMEOUT_MAX. Legit find->replace patterns finish
# in microseconds against a 1 KB input; only catastrophic backtracking
# approaches this. But the child runs ~7 probe inputs per entry (measured
# ~0.44 ms/entry on trivial rules), and an admin full save re-probes EVERY
# stored entry, so a flat 2 s would 422 a large legitimate rule set naming
# whichever innocent entry was in flight. Module-level so tests can lower it.
_GUARD_TIMEOUT = 2.0
# ~10x the measured per-entry cost: generous headroom for richer patterns.
_PER_CHECK_BUDGET = 5e-3
# Hard ceiling so a save can never block the admin thread indefinitely
# (quick_config/routes.py documents the wait around the save call).
_GUARD_TIMEOUT_MAX = 10.0

# Max output/input length ratio for ONE substitution against the fixture. A
# legit correction rule shrinks the text or barely grows it; an expanding
# replacement (`(.*)` -> `\1` repeated) costs microseconds but multiplies its
# input on EVERY chained rule, so a handful of entries turn a short transcript
# into gigabytes at match time. Anything above 1x compounds; 10x is generous.
_MAX_GROWTH = 10

# Absolute floor for the ANALYTIC growth check (run for every pattern, next to
# the measured one). A bounded literal expansion of a short token —
# ("°", "Grad Celsius"), (r"\bIT\b", "Informationstechnologie") — is a normal
# dictation rule and cannot compound: it contributes a fixed handful of
# characters per occurrence. Only replacements past this many characters (or
# past _MAX_GROWTH x the minimum match, whichever is larger) are refused. The
# floor does NOT cover group references: each one copies an unbounded capture,
# so more than _MAX_GROWTH of them is refused whatever the total length.
_MIN_ABS_GROWTH = 64

# Extra probe inputs, run after FIXTURE. The German fixture is full of
# punctuation and short sentences, so a pattern that only blows up on a long
# unpunctuated run (`(\w+ ?)+` on a dictated sentence) finishes on it in
# microseconds and slips through. These are the shapes that make a nested
# repetition go exponential: one long word run, one long letter run, one long
# digit run, one long alphanumeric run — all with no punctuation to anchor on.
# Each is ~240 chars (a quarter of FIXTURE), so all four together cost about
# the same as ONE extra fixture pass per entry and can't push a legitimate
# rule set anywhere near _GUARD_TIMEOUT; an exponential pattern, by contrast,
# already needs longer than the age of the universe at this length.
# Each ends in a `!` the run itself can't produce: blowup only shows up on a
# match that must FAIL, and a plain run often just matches and returns fast.
_ADVERSARIAL = (
    "wort " * 48 + "!",
    "a" * 240 + "!",
    "1234567890" * 24 + "!",
    "Wort123abc" * 24 + "!",
)

# Scaling probe (see _probe). The longer fixture repeats FIXTURE so the input
# SHAPE is identical and only the length changes — the comparison then measures
# how the pattern scales, not how it reacts to different text.
_SCALE_FACTOR = 4
_SCALE_FIXTURE = FIXTURE * _SCALE_FACTOR
# Reject when the longer run costs more than this multiple of the short one.
# Linear work costs _SCALE_FACTOR x; double that is generous headroom.
_SCALE_ALLOWANCE = 2 * _SCALE_FACTOR
# Ordinary rules finish in microseconds, where the timer's own resolution
# dominates the ratio. Compare against at least this to keep noise out.
_TIMER_FLOOR = 1e-4
# A ratio alone is not a verdict: legit rules run in tens of microseconds,
# where one cache miss can fake a large multiple, so a first-sample trip is
# re-measured this many times total and the BEST (min) of each side decides.
_SCALE_SAMPLES = 3
# ...and even a confirmed ratio only rejects when the scaled run costs at
# least this much CPU. A pattern that finishes the 4 KB fixture in under 5 ms
# cannot meaningfully stall a transcript whatever its growth curve; a real
# polynomial pattern (`.*.*#` ~ hundreds of ms at 4 KB) clears this floor by
# orders of magnitude. This is the deterministic backstop that keeps a busy
# host (the guard shares the box with transcription) from failing valid saves.
_SCALE_MIN_REJECT = 5e-3

# Chained-probe cap (see _probe). Entry N is probed against the fixture as
# entries 0..N-1 have already REWRITTEN it, because that is exactly what
# engine.rebuild_caches feeds it at match time — an earlier entry can manufacture
# the input shape that makes a later one explode, and the static fixtures never
# contain it. The running string is truncated to this many characters after
# each substitution so the chaining itself can never become the blowup.
_CHAIN_CAP = 4096

# Synthetic-probe shape (see _synthetic_fixture). Same length as _ADVERSARIAL
# so one extra probe costs about one extra fixture pass for a legitimate rule.
_SYNTH_RUN = 240
# Characters tried as the terminator that forces the engine to exhaust its
# split search. The first one the pattern does not mention literally is used.
_SYNTH_TERMINATORS = "#!@~%"
# What a shorthand escape can match, for the synthetic probe only.
_SHORTHAND_CHARS = {
    "w": "a", "W": "#", "d": "1", "D": "a", "s": " ", "S": "a",
    "n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v",
}
# Hex escapes and their exact digit counts (``\x2d``, ``\u2013``, ``\U…``).
_HEX_ESCAPES = {"x": 2, "u": 4, "U": 8}

# Cap on the product of nested minimum counts along any group-nesting path
# (see _fixed_count_product). Python runs a fixed count over a zero-width body all
# the same, so `(?:(?:a{0}){N}){N}` loops N*N times on EVERY input, the empty
# string included — an uninterruptible in-process hang wherever the pattern
# is first matched (the overlap check, the schema's template check). No
# dictation rule needs anything near this many repetitions.
_MAX_FIXED_COUNT = 10 ** 4

_SELF = os.path.abspath(__file__)


def _ascii_digits(s: str) -> bool:
    """CPython's quantifier parser accepts ASCII digits only, so any other
    brace run (`a{²}`, Arabic-Indic digits) is a literal `{` to the engine
    and must not reach ``int()`` — which would raise a raw, rule-less error."""
    return s.isascii() and s.isdigit()


def _read_quantifier(pat: str, i: int) -> "tuple[bool, bool, int, bool]":
    """Read a quantifier at ``pat[i]``.

    Returns (repeats, atomic, next_index, variable).

    ``repeats`` is True only for a quantifier that can match an atom MORE THAN
    ONCE (`*`, `+`, `{2,}`, `{0,3}`) — `?` and `{0,1}` are optional, not
    repetition, and are far too common in ordinary rules to treat as risky.
    ``atomic`` is True for a possessive quantifier (`*+`, `++`, `{n,m}+`),
    which cannot give back characters and therefore cannot backtrack.
    ``variable`` is True only when the quantifier admits MORE THAN ONE
    repetition count (`*`, `+`, `{n,}`, `{n,m}` with m > n). A fixed-count
    `{n}` / `{n,n}` matches exactly one way, so it cannot split the input
    ambiguously and is safe for the backtracking screen — `(\\d{4})+` and a
    thousands-separator `(?:\\.\\d{3})+` are ordinary rules, not blowups.
    """
    if i >= len(pat):
        return False, False, i, False
    c = pat[i]
    if c in "*+?":
        repeats = c != "?"
        variable = repeats
        i += 1
    elif c == "{":
        j = pat.find("}", i)
        if j == -1:
            return False, False, i, False  # a literal `{`, not a quantifier
        lo, comma, hi = pat[i + 1:j].partition(",")
        if comma:
            if not (_ascii_digits(lo) or not lo) or not (_ascii_digits(hi) or not hi):
                return False, False, i, False
            repeats = not hi or int(hi) > 1  # {n,} is unbounded
            variable = repeats and (not hi or int(hi) > (int(lo) if lo else 0))
        else:
            if not _ascii_digits(lo):
                return False, False, i, False
            repeats = int(lo) > 1
            variable = False  # {n} matches exactly one way
        i = j + 1
    else:
        return False, False, i, False
    atomic = i < len(pat) and pat[i] == "+"
    if i < len(pat) and pat[i] in "?+":
        i += 1  # lazy or possessive suffix
    return repeats, atomic, i, variable


def _fixed_count_product(pat: str) -> int:
    """The largest product of the MINIMUM counts of ``{n}`` / ``{n,m}`` /
    ``{n,}`` quantifiers along any group-nesting path of ``pat``:
    ``(?:(?:a{0}){30}){40}`` → 1200.

    Only the minimum is forced: past it the engine stops a loop whose body
    matched empty, so ``(?:(?:a{0}){0,100000}){0,100000}`` is instant while
    ``(?:(?:a{0}){3000}){3000}`` spins 9 million times (~3 s) even on "".
    Each count contributes at least 1 — ``a{0}`` still costs its enclosing
    loops their full iteration count — and ``*`` / ``+`` / ``?`` count as 1.
    Escapes and classes are skipped as in the screen; ``(?#...)`` comments are
    removed first (``_strip_comments``), so a count after one still counts.
    """
    pat = _strip_comments(_strip_verbose(pat))
    n = len(pat)

    def _count(j: int) -> "tuple[int, int]":
        if j < n and pat[j] == "{":
            end = pat.find("}", j)
            if end != -1:
                lo, comma, hi = pat[j + 1:end].partition(",")
                if ((_ascii_digits(lo) or (comma and not lo))
                        and (_ascii_digits(hi) or not hi)):
                    # Ten digits already exceed the engine's MAXREPEAT —
                    # but only SIGNIFICANT ones: the engine reads
                    # `{00000000003000}` as 3000, so leading zeros go first.
                    return max(int(lo.lstrip("0")[:10] or "0"), 1), end + 1
        return 1, j

    stack = [1]
    i = 0
    while i < n:
        c = pat[i]
        if c == "(":
            if pat.startswith("(?#", i):
                end = pat.find(")", i)
                i = n if end == -1 else end + 1
                continue
            stack.append(1)
            i += 1
            continue
        if c == ")" and len(stack) > 1:
            inner = stack.pop()
            mult, i = _count(i + 1)
            stack[-1] = max(stack[-1], inner * mult)
            continue
        if c == "\\":
            i += 2
        elif c == "[":
            i = _skip_class(pat, i)[1] + 1
        else:
            i += 1
        mult, i = _count(i)
        stack[-1] = max(stack[-1], mult)
    return max(stack)


def check_fixed_counts(where: str, pattern: str) -> None:
    """Raise ``ValueError(f"{where}: ...")`` when ``pattern`` nests fixed
    counts past ``_MAX_FIXED_COUNT`` (see there). Cheap and match-free, so it
    is safe to run before anything that matches the pattern in-process."""
    if pattern and _fixed_count_product(pattern) > _MAX_FIXED_COUNT:
        raise ValueError(
            f"{where}: regex test failed: nested fixed counts repeat more than "
            f"{_MAX_FIXED_COUNT} times (e.g. `(?:(?:x){{1000}}){{1000}}`), which "
            "can hang the server even on empty input — use smaller counts."
        )


def _skip_class(pat: str, i: int) -> "tuple[int, int, bool]":
    """Scan the character class opening at ``pat[i] == "["``.

    Returns ``(body_start, body_end, negated)`` where ``body_end`` is the index
    of the closing ``]`` (``len(pat)`` when unterminated). A leading ``^``
    negates, a ``]`` first in the body is a literal, and a backslash escapes
    the next character. The structural screen and the synthetic probe must
    agree on which ``]`` closes a class, so the rule lives here exactly once.
    """
    n = len(pat)
    j = i + 1
    negated = False
    if j < n and pat[j] == "^":
        negated, j = True, j + 1
    # The body starts BEFORE a leading literal `]`: it is a member, and the
    # synthetic probe's witness for `[]]` must be `]`, not an empty body.
    start = j
    if j < n and pat[j] == "]":
        j += 1  # a `]` first in the class is a literal
    while j < n and pat[j] != "]":
        j += 2 if pat[j] == "\\" else 1
    return start, j, negated


def _strip_outer_group(branch: str) -> str:
    """Strip every layer of parentheses that wraps a whole branch body.

    ``((a))`` → ``a``, ``(a)`` → ``a``, ``(?:13)`` → ``13``, ``abc`` → ``abc``.
    Only strips when the opening ``(`` is balanced by the LAST ``)`` (i.e. the
    parens wrap the entire branch, not just a prefix). A non-capturing, named,
    atomic or scoped-flag prefix (``?:``, ``?P<name>``, ``?<name>``, ``?>``,
    ``?i:``) goes with the parens — left in place it made the body uncompilable
    and its witness wrong, so wrapping an overlapping branch in ``(?:...)``
    slipped past the overlap check. A lookaround, conditional, comment or
    backreference group is not the plain body it holds and stays as it is.
    """
    import re
    while len(branch) >= 2 and branch[0] == "(":
        depth, i = 1, 1
        while i < len(branch) and depth:
            ch = branch[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "[":
                i = _skip_class(branch, i)[1] + 1
                continue
            depth += 1 if ch == "(" else (-1 if ch == ")" else 0)
            i += 1
        if not (depth == 0 and i == len(branch)):
            break
        inner = branch[1:-1]
        if inner.startswith("?"):
            m = re.match(r"\?(?:>|P?<(?![=!])\w+>|[aiLmsux]*(?:-[imsx]*)?:)", inner)
            if not m:
                break
            inner = inner[m.end():]
        branch = inner
    return branch


def _strip_verbose(pat: str) -> str:
    """``pat`` with its VERBOSE-mode whitespace and ``#`` comments removed.

    Under ``(?x)`` (or a scoped ``(?x:...)``) unescaped whitespace is ignored
    and ``#`` runs a comment to the end of the line, so the quantifier in
    ``(?x)(a+) +#`` really applies to the group — but a scanner that reads a
    quantifier only directly after ``)`` counted it as a root-level repeat,
    and a ``)`` inside a comment popped the wrong frame. Escapes, character
    classes and ``(?#...)`` are copied verbatim (whitespace and ``#`` are
    literal there). Returns ``pat`` unchanged when no verbose mode applies.
    """
    if "x" not in pat:
        return pat
    import re
    try:
        verbose = bool(re.compile(pat).flags & re.VERBOSE)
    except Exception:  # noqa: BLE001 - an uncompilable pattern is refused elsewhere
        return pat
    scoped = re.compile(r"\(\?([aiLmsux]*)(?:-([imsx]*))?:")
    if not verbose and not any("x" in m.group(1) for m in scoped.finditer(pat)):
        return pat
    n = len(pat)
    out: list[str] = []
    modes = [verbose]
    i = 0
    while i < n:
        c = pat[i]
        v = modes[-1]
        if c == "\\":
            out.append(pat[i:i + 2])
            i += 2
            continue
        if c == "[":
            end = _skip_class(pat, i)[1]
            out.append(pat[i:end + 1])
            i = end + 1
            continue
        if c == "(":
            if pat.startswith("(?#", i):
                end = _comment_end(pat, i)
                out.append(pat[i:end])
                i = end
                continue
            m = scoped.match(pat, i)
            if m:
                modes.append((v or "x" in m.group(1)) and "x" not in (m.group(2) or ""))
                out.append(m.group(0))
                i = m.end()
                continue
            modes.append(v)
        elif c == ")" and len(modes) > 1:
            modes.pop()
        elif v and c in " \t\n\r\f\v":
            i += 1
            continue
        elif v and c == "#":
            nl = pat.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _comment_end(pat: str, i: int) -> int:
    """The index just past the ``(?#...)`` comment opening at ``pat[i]``
    (``len(pat)`` when unterminated). Inside a comment the engine still reads
    a backslash as escaping the next character, so ``(?#\\))`` ends at the
    second ``)``."""
    n = len(pat)
    j = i + 3
    while j < n and pat[j] != ")":
        j += 2 if pat[j] == "\\" else 1
    return min(j + 1, n)


def _strip_comments(pat: str) -> str:
    """``pat`` with every ``(?#...)`` comment group removed.

    The engine drops a comment, so a quantifier after one binds to the atom
    BEFORE it: ``a(?#x){3}`` is ``a{3}`` and ``(a+)(?#x)+#`` is ``(a+)+#``. A
    scanner that skipped the comment and then read the quantifier as plain
    text missed both the fixed count and the nested repeat. Escapes and
    character classes are copied verbatim (``(?#`` is literal there).
    """
    if "(?#" not in pat:
        return pat
    n = len(pat)
    out: list[str] = []
    i = 0
    while i < n:
        c = pat[i]
        if c == "\\":
            out.append(pat[i:i + 2])
            i += 2
        elif c == "[":
            end = _skip_class(pat, i)[1]
            out.append(pat[i:end + 1])
            i = end + 1
        elif pat.startswith("(?#", i):
            i = _comment_end(pat, i)
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _nested_repetition(pat: str) -> bool:
    """True if ``pat`` contains a repeated group that itself repeats.

    `(x+)+`, `(x*)*`, `(x{2,})+` and the alternation-overlap `(a|a)*` are the
    shapes whose match time is EXPONENTIAL in the input length: every way of
    splitting the input between the inner and outer repetition is a distinct
    path the engine must try before it can fail. `(\\w+ ?)+#` needs 2**n steps
    on an n-word sentence — a fixture probe can't outrun that, so the shape
    itself has to be refused.

    The overlap test covers PREFIX-ambiguous alternatives too, not only
    byte-identical ones: `(a|ab)+`, `(x|xx)+y` and `(n|d|nd)+#` are all
    exponential for the same reason — one run of input splits many ways — and
    none of them repeats a branch verbatim. That test is a deliberate
    over-approximation: it also refuses a repeated `(km|km/h)+`, which is not
    actually explosive. Rejecting a writable rule is recoverable; hanging every
    transcription on an uninterruptible `re.sub` is not.

    Scans the pattern string once; there is no way to match this with a regex
    and no stdlib API that exposes the parse tree. `(`, `)`, `*`, `+` and `{`
    are LITERAL inside a character class and after a backslash, so both are
    skipped — a scanner that miscounts them rejects perfectly good rules.
    A verbose-mode pattern is scanned with its whitespace and comments
    removed (``_strip_verbose``), and every pattern without its ``(?#...)``
    comments (``_strip_comments``).
    """
    pat = _strip_comments(_strip_verbose(pat))
    n = len(pat)
    # One frame per open group plus a root frame. "rep": this level contains a
    # repetition; "start"/"alts": body slice + top-level `|` offsets, for the
    # (a|a)* overlap check; "atomic": (?>...) can't backtrack into itself;
    # "conditional": (?(1)yes|no), whose `|` picks a branch by the condition
    # and is not an overlapping alternation; "open": index of its `(`;
    # "inner": the branch slices of an alternation group that is this
    # group's WHOLE body — `(?:(a|ab))+` puts the `|` on the un-quantified
    # child, so the quantified parent must check the child's branches.
    stack = [{"rep": False, "start": 0, "alts": [], "atomic": False}]
    i = 0
    while i < n:
        c = pat[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            i = _skip_class(pat, i)[1] + 1
            continue
        if c == "(":
            atomic = False
            lookaround = False
            conditional = False
            j = i + 1
            if j < n and pat[j] == "?":
                k = j + 1
                if k < n and pat[k] == "#":  # (?#comment) — skip wholesale
                    end = pat.find(")", k)
                    i = n if end == -1 else end + 1
                    continue
                if k < n and pat[k] == ">":
                    atomic, j = True, k + 1
                elif k < n and pat[k] == ":":
                    j = k + 1
                elif k < n and pat[k] in "=!":
                    lookaround, j = True, k + 1
                elif k < n and pat[k] == "<" and k + 1 < n and pat[k + 1] in "=!":
                    lookaround, j = True, k + 2
                elif k < n and pat[k] == "(":
                    # (?(1)yes|no) / (?(name)...) — a real group whose body
                    # starts after the condition's `)`. Treating it as
                    # self-closing let its own `)` pop the ENCLOSING group,
                    # so `(x)?((?(1)\w+\s?|b))+#` slipped past the screen.
                    close = pat.find(")", k)
                    conditional, j = True, (n if close == -1 else close + 1)
                else:
                    # (?P<name>, (?P=name), (?<name>, inline flags (?i) / (?i:
                    end = pat.find(">", k)
                    close = pat.find(")", k)
                    colon = pat.find(":", k)
                    cand = [x for x in (end, colon) if x != -1 and (close == -1 or x < close)]
                    if not cand:
                        # (?P=name) / (?i) — self-closing, opens no group
                        i = close + 1 if close != -1 else n
                        continue
                    j = min(cand) + 1
            stack.append({"rep": False, "start": j, "alts": [], "atomic": atomic,
                          "lookaround": lookaround, "conditional": conditional,
                          "open": i})
            i = j
            continue
        if c == ")" and len(stack) > 1:
            frame = stack.pop()
            close = i
            i += 1
            repeats, possessive, i, variable = _read_quantifier(pat, i)
            spans = None
            if frame.get("conditional"):
                pass
            elif frame["alts"]:
                cuts = [frame["start"]] + [x + 1 for x in frame["alts"]]
                spans = list(zip(cuts, frame["alts"] + [close]))
            else:
                spans = frame.get("inner")
            if variable and not frame["atomic"] and not possessive:
                if frame["rep"]:
                    return True
                # (a|a)* — identical alternatives overlap, so the engine
                # re-tries the same split every way round. Prefix-ambiguous
                # branches are the same trap without being byte-identical:
                # (a|ab)+, (x|xx)+y, (n|d|nd)+# all let one run of input be
                # split many ways, which backtracks exponentially. An empty
                # branch — (|a)+ — is the degenerate case of the same thing.
                if spans and _spans_overlap(pat, spans):
                    return True
            # A lookaround is matched once and never backtracked into, so a
            # repeat inside it cannot split the enclosing group's input
            # ambiguously: (x(?=a+)y)+ is linear. Same for the BODY of an
            # atomic group, which never gives characters back — but a
            # quantifier on the group itself ((?>a)+) still repeats.
            if frame["lookaround"]:
                continue
            # An un-quantified alternation group that is its parent's whole
            # body hands its branches up: `(?:(a|ab))+` repeats them all the
            # same. Not through an atomic group, which never re-splits.
            if (spans and i == close + 1 and not frame["atomic"]
                    and len(stack) > 1 and frame["open"] == stack[-1]["start"]
                    and i < n and pat[i] == ")"):
                stack[-1]["inner"] = spans
            # Only a VARIABLE quantifier on the group marks the parent as
            # repeating, as for a single atom: a fixed-count `((?:ab){2})+`
            # matches each repetition exactly one way.
            if (frame["rep"] and not frame["atomic"]) or variable:
                stack[-1]["rep"] = True
            # An overlapping alternation splits its input many ways whether
            # or not it carries a quantifier of its own: `((?:a|aa){2})+$`
            # is `(?:a|aa)(?:a|aa)` repeated, and `(?:c|(n|nn))+#` repeats
            # the `(n|nn)` split exactly as `(n|nn)+#` does. Mark the parent
            # as repeating, so a variable quantifier on any enclosing group
            # refuses it. A possessive or atomic group never re-splits — and
            # neither does one whose split a mandatory, disjoint next atom
            # pins down: `(?:(?:um|umm), )+` (see _follower_fixes_split).
            elif (spans and not frame["atomic"] and not possessive
                    and _spans_overlap(pat, spans)
                    and not (i == close + 1
                             and _follower_fixes_split(pat, spans, i))):
                stack[-1]["rep"] = True
            continue
        if c == "|":
            stack[-1]["alts"].append(i)
            i += 1
            continue
        repeats, possessive, j, variable = _read_quantifier(pat, i)
        if j > i:
            # A possessive atom (`\w++`) never gives characters back, so —
            # like the body of an atomic group — it cannot re-split the
            # enclosing group's input: `(?:\w++ ?)+#` is linear.
            if variable and not possessive:
                stack[-1]["rep"] = True
            i = j
            continue
        i += 1
    return False


def _spans_overlap(pat: str, spans: "list[tuple[int, int]]") -> bool:
    """True if two of the alternation branches ``pat[a:b]`` in ``spans`` can
    match the same start of the input — the shape that lets one run of input
    split many ways once the alternation repeats.

    (a|a)* — identical alternatives overlap, so the engine re-tries the same
    split every way round. Prefix-ambiguous branches are the same trap without
    being byte-identical: (a|ab)+, (x|xx)+y, (n|d|nd)+# all let one run of
    input be split many ways, which backtracks exponentially. An empty branch
    — (|a)+ — is the degenerate case of the same thing.
    """
    stripped = [_strip_outer_group(pat[a:b]) for a, b in spans]
    if len(set(stripped)) < len(stripped):
        return True
    if any(b != a and b.startswith(a) for a in stripped for b in stripped):
        return True
    # Overlap through a class or shorthand is the same trap with no shared
    # text: (cx|[bc]x)+# splits "cxcx…" both ways. Refuse when one branch
    # matches the start of another branch's witness.
    return any(_branches_overlap(a, b) for a in stripped for b in stripped)


def _follower_fixes_split(pat: str, spans: "list[tuple[int, int]]", i: int) -> bool:
    """True if the atom at ``pat[i]``, right after an UN-quantified alternation
    group whose branches are ``spans``, leaves each match of that group exactly
    one way to split.

    `(?:(?:um|umm), )+` overlaps (`um` is a prefix of `umm`), but the `,` that
    must follow can't be matched by any branch character, so the group's match
    has to end right before the next `,` — and since no two branches have the
    same width, at most one branch fits that span. One parse per repetition is
    linear, so the enclosing repeat is not refused for it. Deliberately narrow:
    every branch must be a run of single-character, unquantified atoms; the
    follower a mandatory literal, escaped character or plain character class
    (`\\b` / `\\B` before it are skipped — they only restrict the match); and
    no follower character may be matched by any branch atom, case-insensitively
    and with DOTALL, so a flag elsewhere in the pattern cannot widen an atom
    past the check. Anything else answers False and keeps the refusal:
    `(?:(?:ax|[ab]x)c)+#` has two same-width branches matching `ax`, and
    `((?:a|aa)a?)+#` has an optional follower — both stay exponential.
    """
    import re
    n = len(pat)
    while pat.startswith(("\\b", "\\B"), i):
        i += 2
    if i >= n or pat[i] in "()|^$.*+?{":
        return False
    char, group, j = _next_atom(pat, i)
    if char is None or group is not None:
        return False
    k = _read_quantifier(pat, j)[2]
    if k > j:
        q = pat[j:k]
        lo = q[1:].split(",")[0].rstrip("}?+") if q[0] == "{" else ""
        if not (q[0] == "+" or (q[0] == "{" and _ascii_digits(lo) and int(lo) >= 1)):
            return False
    follower = _follower_chars(pat[i:j])
    if not follower:
        return False
    widths = set()
    atoms = []
    for a, b in spans:
        branch = _strip_outer_group(pat[a:b])
        count, p = 0, 0
        while p < len(branch):
            if branch[p] in "()|^$*+?{":
                return False
            c, g, e = _next_atom(branch, p)
            if c is None or g is not None or _read_quantifier(branch, e)[2] != e:
                return False
            atoms.append(branch[p:e])
            count += 1
            p = e
        if not count or count in widths:
            return False
        widths.add(count)
    try:
        compiled = [re.compile(x, re.IGNORECASE | re.DOTALL) for x in set(atoms)]
    except re.error:
        return False
    return not any(rx.fullmatch(ch) for rx in compiled for ch in follower)


def _follower_chars(atom: str) -> "set[str] | None":
    """Every character the single atom ``atom`` matches, or None when that set
    is not small and explicit: a literal, an escaped non-alphanumeric
    character, or a non-negated class of such characters and short ranges.
    Shorthands (``\\w``), negated classes and escaped letters answer None."""
    if atom.startswith("\\"):
        return {atom[1]} if len(atom) == 2 and not atom[1].isalnum() else None
    if not atom.startswith("["):
        return {atom} if len(atom) == 1 else None
    body = atom[1:-1]
    if not body or body[0] == "^":
        return None
    out: set[str] = set()
    p = 0
    while p < len(body):
        c = body[p]
        if c == "\\":
            if p + 1 >= len(body) or body[p + 1].isalnum():
                return None
            c, p = body[p + 1], p + 2
        else:
            p += 1
        if p + 1 < len(body) and body[p] == "-":
            hi = body[p + 1]
            if hi in "\\[" or ord(hi) < ord(c) or ord(hi) - ord(c) > 256:
                return None
            out.update(chr(x) for x in range(ord(c), ord(hi) + 1))
            p += 2
        else:
            out.add(c)
    # Each one with its case variants: under (?i) the follower matches those
    # too, and the branch atoms are compiled case-insensitively against them.
    return out | {v for c in out for v in (c.lower(), c.upper())}


def _loose_quantifier(b: str) -> bool:
    """True if ``b`` holds any quantifier other than a fixed count ``{n}`` /
    ``{n,n}`` — an optional ``?`` / ``{0,1}`` (or a lazy suffix) as well as a
    repeat. Escapes and character classes are skipped; the ``?`` that opens a
    ``(?:`` / ``(?P<`` / lookaround prefix is not a quantifier."""
    n = len(b)
    i = 0
    while i < n:
        c = b[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            i = _skip_class(b, i)[1] + 1
            continue
        if c == "(" and i + 1 < n and b[i + 1] == "?":
            i += 2
            continue
        if c in "*+?":
            return True
        if c == "{":
            j = b.find("}", i)
            if j != -1:
                lo, comma, hi = b[i + 1:j].partition(",")
                if comma and (not lo or lo != hi):
                    return True
        i += 1
    return False


def _branches_overlap(a: str, b: str) -> bool:
    """True if branch ``b`` matches a prefix of a string branch ``a`` matches.

    Best effort over ``_witness``: an empty witness, or a branch that does not
    compile on its own (a group reference, an inline flag), never counts as an
    overlap. This runs IN-PROCESS on a user-supplied fragment, so ``b`` is
    matched only when that match is linear: never when it holds a nested
    alternation — unquantified ``(?:x|xx)(?:x|xx)…`` backtracks exponentially
    even on a 64-character witness — and never when it holds any quantifier
    but a fixed count: optional atoms backtrack exponentially too
    (``a?a?…aa…`` against ``aa…``). A group without ``|`` and with only
    fixed counts, like ``(?:h)a``, matches one way and is still checked —
    unless its nested counts multiply past ``_MAX_FIXED_COUNT``:
    ``(?:(?:a{0}){N}){N}`` loops N*N times however short the witness is.
    """
    import re
    if (a == b or ("(" in b and "|" in b) or _loose_quantifier(b)
            or _fixed_count_product(b) > _MAX_FIXED_COUNT):
        return False
    w = _witness(a)
    if not w:
        return False
    try:
        return re.match(b, w) is not None
    except Exception:  # noqa: BLE001 - a fragment need not compile alone
        return False


def _class_char(body: str, negated: bool) -> "str | None":
    """A character the character class ``[body]`` can match, or None.

    Only needs ONE witness, so the first literal / range start / shorthand in
    the class body is enough. For a negated class, try a few ordinary
    candidates and return the first that the body does not obviously contain.
    """
    members = []
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == "\\" and i + 1 < n:
            char, i = _escape_char(body, i, in_class=True)
            if char is not None:
                members.append(char)
            continue
        # `a-z`: the range start is a member; a trailing `-` is a literal.
        members.append(c)
        i += 3 if (i + 2 < n and body[i + 1] == "-") else 1
    if not negated:
        return members[0] if members else None
    # Verify each candidate against the real class instead of guessing from
    # the expanded members: a shorthand like `\w` stands for a whole SET, so
    # its one representative ('a') banned only 'a' and let '1' through — a
    # character `[^\w]` can never match, silently no-op'ing the synthetic
    # probe. Fall back to the member heuristic if the body doesn't compile.
    import re
    try:
        rx = re.compile(f"[^{body}]")
    except re.error:
        rx = None
    if rx is not None:
        for cand in "a1 Zx.":
            if rx.match(cand):
                return cand
        return None
    banned = set(members)
    for cand in "a1 Zx.":
        if cand not in banned:
            return cand
    return None


def _escape_char(pat: str, i: int, in_class: bool = False) -> "tuple[str | None, int]":
    """Decode the escape at ``pat[i] == "\\"``.

    Returns ``(char, end)``: a character the escape matches (None for a
    zero-width assertion, a backreference or anything undecodable) and the
    index just past the WHOLE escape. A numeric or named escape is ONE
    character — ``\\x2d`` is ``-``, ``\\u2013`` and ``\\N{EN DASH}`` an en dash.
    Reading only the letter after the backslash took ``\\x2d`` for ``x`` and
    its ``2d`` for two more atoms, so the synthetic probe ran on a character
    the pattern never matches and waved a polynomial ``\\x2d*\\x2d*…#`` through.
    Digits follow the engine: outside a class ``\\0oo`` and three octal digits
    are a character and any other digit run a group reference; inside a class
    every digit escape is octal and ``\\b`` is a backspace.
    """
    n = len(pat)
    if i + 1 >= n:
        return None, i + 1
    esc = pat[i + 1]
    if esc in _SHORTHAND_CHARS:
        return _SHORTHAND_CHARS[esc], i + 2
    if esc == "a":
        return "\a", i + 2
    if esc in _HEX_ESCAPES:
        end = i + 2 + _HEX_ESCAPES[esc]
        digits = pat[i + 2:end]
        if len(digits) == _HEX_ESCAPES[esc] and all(d in "0123456789abcdefABCDEF" for d in digits):
            code = int(digits, 16)
            return (chr(code) if code <= 0x10FFFF else None), end
        return None, i + 2
    if esc == "N" and pat.startswith("{", i + 2):
        close = pat.find("}", i + 3)
        if close == -1:
            return None, i + 2
        import unicodedata
        try:
            return unicodedata.lookup(pat[i + 3:close]), close + 1
        except KeyError:
            return None, close + 1
    if esc in "0123456789":
        octal = "01234567"
        j = i + 2
        if in_class or esc == "0":
            if esc not in octal:
                return None, j
            while j < min(i + 4, n) and pat[j] in octal:
                j += 1
        elif (j + 1 < n and esc in octal and pat[j] in octal
                and pat[j + 1] in octal):
            j += 2
        else:
            # A group reference: one or two digits, matching nothing we know.
            if j < n and pat[j] in "0123456789":
                j += 1
            return None, j
        code = int(pat[i + 1:j], 8)
        return (chr(code) if code <= 0o377 else None), j
    if esc in "bBAZ":
        # Inside a class `\b` is a backspace; elsewhere these are zero-width.
        return ("\b" if in_class and esc == "b" else None), i + 2
    return esc, i + 2


def _next_atom(pat: str, i: int) -> "tuple[str | None, str | None, int]":
    """Parse the single atom at ``pat[i]``.

    Returns ``(char, group_body, end)`` where ``char`` is a character the atom
    can match (None if unknown or zero-width), ``group_body`` is the group's
    inner slice when the atom is a group (else None), and ``end`` is the index
    just past the atom and BEFORE any quantifier. ``end`` is always > ``i`` so
    every caller makes progress.
    """
    n = len(pat)
    c = pat[i]
    if c == "\\":
        char, end = _escape_char(pat, i)
        return char, None, end
    if c == "[":
        start, j, negated = _skip_class(pat, i)
        if j >= n:
            return None, None, n
        return _class_char(pat[start:j], negated), None, j + 1
    if c == "(":
        j = i + 1
        zero_width = False
        if j < n and pat[j] == "?":
            k = j + 1
            if k < n and pat[k] in ">:=!":
                zero_width = pat[k] in "=!"
                j = k + 1
            elif k < n and pat[k] == "<" and k + 1 < n and pat[k + 1] in "=!":
                zero_width, j = True, k + 2
            elif k < n and pat[k] in "aiLmsux":
                m = k
                while m < n and pat[m] in "aiLmsux-":
                    m += 1
                if m < n and pat[m] == ")":
                    return None, None, m + 1
                if m < n and pat[m] == ":":
                    j = m + 1
                else:
                    j = k
            else:
                end = pat.find(">", k)
                close = pat.find(")", k)
                colon = pat.find(":", k)
                cand = [x for x in (end, colon)
                        if x != -1 and (close == -1 or x < close)]
                if not cand:
                    # (?P=name) — self-closing, opens no group
                    return None, None, close + 1 if close != -1 else n
                j = min(cand) + 1
        # Find the matching `)`, skipping classes and escapes.
        depth, k = 1, j
        while k < n and depth:
            ch = pat[k]
            if ch == "\\":
                k += 2
                continue
            if ch == "[":
                k = _skip_class(pat, k)[1] + 1
                continue
            depth += 1 if ch == "(" else (-1 if ch == ")" else 0)
            k += 1
        if depth:
            return None, None, n
        # A lookaround consumes nothing, so it is not an atom for our purposes.
        return None, (None if zero_width else pat[j:k - 1]), k
    if c == ".":
        return "a", None, i + 1
    if c in "|^$*+?":
        return None, None, i + 1  # anchors / stray quantifiers carry no atom
    return c, None, i + 1


def _first_char(body: str, depth: int = 0) -> "str | None":
    """A character the FIRST atom of ``body`` can match, or None."""
    i = 0
    while i < len(body):
        char, group, j = _next_atom(body, i)
        if group is not None:
            if depth < 4:
                found = _first_char(group, depth + 1)
                if found:
                    return found
        elif char:
            return char
        _repeats, _atomic, k, _variable = _read_quantifier(body, j)
        i = max(k, j)
    return None


def _first_repeated_char(pat: str, depth: int = 0) -> "str | None":
    """A character matched by the first atom that carries a repeating
    quantifier (`*`, `+`, `{2,}`), or None if none can be determined.

    That atom is where an unbounded run of input gets consumed, so repeating
    the character it matches is the input shape most likely to make the
    pattern backtrack. Best effort by construction: returning None simply
    skips the synthetic probe for that pattern.
    """
    i = 0
    n = len(pat)
    while i < n:
        char, group, j = _next_atom(pat, i)
        repeats, _atomic, k, _variable = _read_quantifier(pat, j)
        if repeats:
            if group is not None:
                if depth < 4:
                    found = _first_char(group, depth + 1)
                    if found:
                        return found
            elif char:
                return char
        elif group is not None and depth < 4:
            found = _first_repeated_char(group, depth + 1)
            if found:
                return found
        i = max(k, j)
    return None


def _synthetic_fixture(pat: str) -> "str | None":
    """A per-pattern adversarial probe input, or None if none can be built.

    The fixed fixtures are German prose plus four repetitive runs, and between
    them they contain no `-`, `n`, `d`, `m`, `k` or `f`. A pattern built around
    a character none of them contains matches NOTHING in any probe and returns
    in microseconds however explosive it is. So synthesize the run this
    pattern actually cares about: ~240 repetitions of a character its first
    unbounded atom matches, plus a terminator the pattern does not mention, so
    the match is forced to FAIL after exhausting every way of splitting the
    run — the shape backtracking blows up on.
    """
    try:
        char = _first_repeated_char(_strip_comments(pat))
    except Exception:  # noqa: BLE001 - best effort, never fail the save
        return None
    if not char:
        return None
    for term in _SYNTH_TERMINATORS:
        if term not in pat and term != char:
            return char * _SYNTH_RUN + term
    return None


def _witness(pat: str, depth: int = 0) -> str:
    """A short string ``pat`` plausibly matches, or "" if none can be built.

    Used to keep the CHAINED probe reachable: an entry whose pattern matches
    nothing in the running fixture (`(Wetter)` against German small talk that
    never says "Wetter") substitutes nothing, so its replacement — which may be
    exactly the run that detonates a later entry — never reaches the chain.
    Injecting a witness makes the entry fire, and its output then travels
    downstream the way it will at match time.

    Best effort: one representative character per atom, the first alternative
    of a top-level alternation, optional atoms omitted. A wrong guess costs a
    few wasted characters in a probe input; it can never reject anything by
    itself, because the chained probe's only verdict is the parent's timeout.
    """
    out = []
    i = 0
    n = len(pat)
    while i < n and sum(len(p) for p in out) < 64:
        if pat[i] == "|":
            break  # first alternative only
        char, group, j = _next_atom(pat, i)
        repeats, _atomic, k, _variable = _read_quantifier(pat, j)
        optional = k > j and not repeats  # `?` / `{0,1}` — leave it out
        piece = ""
        if group is not None:
            piece = _witness(group, depth + 1) if depth < 4 else ""
        elif char:
            piece = char
        if piece and not optional:
            out.append(piece * (3 if repeats else 1))
        i = max(k, j)
    return "".join(out)[:64]


def _chain_advance(rx, pattern: str, replacement: str, chained: str) -> str:
    """Apply one entry to the running chained fixture and return the result.

    This is what makes the chained probe carry a MANUFACTURED input downstream:
    entry N is handed the text entries 0..N-1 produced, exactly as
    ``engine.rebuild_caches`` applies them (list order, no longest-first sort).
    Truncated to ``_CHAIN_CAP`` so the chaining itself can never become the
    expensive thing. Never raises: chaining is extra signal, and its only
    verdict is the parent's timeout — it must not invent a rejection reason.
    """
    try:
        seeded = False
        if not rx.search(chained):
            # An entry that matches nothing in the running fixture substitutes
            # nothing, so its replacement — possibly the very run that
            # detonates a later entry — never enters the chain. Seed it.
            seed = _witness(_strip_comments(pattern))
            if seed:
                # Trim BEFORE appending so the seed is inside the cap: a chain
                # already at the cap would otherwise drop it (and the
                # manufactured run it produces) off the tail.
                tail = (" " + seed) * 4
                chained = chained[:max(0, _CHAIN_CAP - len(tail))] + tail
                seeded = True
        out = rx.sub(replacement, chained)
        # A seeded entry touched nothing but the seed at the tail, so keep the
        # TAIL — a head cut would trim the very run the seed was planted to
        # produce once the replacement expands it. Otherwise keep the head.
        return out[-_CHAIN_CAP:] if seeded else out[:_CHAIN_CAP]
    except Exception:  # noqa: BLE001 - best effort; drop the chain, keep going
        return ""


def validate(checks: list, timeout: float | None = None) -> None:
    """Validate every regex in ``checks`` out-of-process.

    ``checks`` is a list of ``(where, pattern, replacement)`` tuples, spanning
    every entry of every rule in save order. Each pattern's ``re.sub`` is run
    against a fixed ~1 KB fixture (plus a few short repetitive worst-case
    inputs, a synthetic run built for that pattern, and the fixture as the
    PRECEDING entries rewrote it) inside a child process that is killed if it
    exceeds ``timeout`` seconds. Raises ``ValueError(f"{where}: ...")`` on a
    pattern that repeats an already-repeating group (`(x+)+`, `(a|a)*` — see
    ``_nested_repetition``), a bad regex/replacement (e.g. a backref to a
    non-existent group), a replacement that expands the fixture beyond
    ``_MAX_GROWTH``, or a catastrophic-backtracking timeout. No-op for an empty
    list.

    Fails OPEN: if the helper can't be launched / crashes, the save proceeds
    WITHOUT the backtracking check rather than blocking a legitimate edit — the
    guard is a safety improvement, never a gate that can break saving.
    """
    if not checks:
        return
    # Structural screen first, in-process: exponential shapes can be slower
    # than any wall-clock probe on inputs the fixtures don't happen to contain,
    # so they are refused on shape rather than on measured time.
    for where, pattern, _repl in checks:
        # Before the overlap check below, which matches branch fragments
        # in-process.
        check_fixed_counts(where, pattern)
        if pattern and _nested_repetition(pattern):
            raise ValueError(
                f"{where}: regex test failed: nested repetition "
                "(a repeat inside a repeated group, e.g. `(\\w+ ?)+`) or a "
                "repeated group whose alternatives overlap (e.g. `(a|ab)+`) "
                "causes catastrophic backtracking and can hang the server on "
                "ordinary input — rewrite without a repeat inside a repeat, "
                "and make the alternatives of a repeated group mutually "
                "exclusive."
            )
    import json
    import logging
    import subprocess
    import sys

    budget = (timeout if timeout is not None
              else min(_GUARD_TIMEOUT + _PER_CHECK_BUDGET * len(checks),
                       _GUARD_TIMEOUT_MAX))
    payload = json.dumps([[c[1], c[2]] for c in checks])
    try:
        proc = subprocess.run(
            [sys.executable, _SELF],
            input=payload, capture_output=True, text=True, timeout=budget,
        )
    except subprocess.TimeoutExpired as exc:
        idx = _last_index(getattr(exc, "stderr", None))
        where = checks[idx][0] if isinstance(idx, int) and 0 <= idx < len(checks) else "a rule"
        raise ValueError(
            f"{where}: regex took > {budget:.2g} s on the guard's test inputs "
            "(1 KB of prose, a 4x longer copy, short repetitive runs, and the "
            "text the preceding rules produce) — likely catastrophic "
            "backtracking. Simplify the pattern."
        )
    except Exception as exc:  # noqa: BLE001 - guard infra failure -> fail open
        logging.getLogger("whisper-api").warning(
            "regex guard skipped (could not run helper): %s", exc)
        return

    if proc.returncode != 0:
        logging.getLogger("whisper-api").warning(
            "regex guard skipped (helper exit %s): %s",
            proc.returncode, (proc.stderr or "").strip()[:200])
        return
    try:
        result = json.loads(proc.stdout or "")
    except Exception:  # noqa: BLE001 - unparseable verdict -> fail open
        return
    if not result.get("ok", False):
        idx = result.get("index")
        where = checks[idx][0] if isinstance(idx, int) and 0 <= idx < len(checks) else "a rule"
        raise ValueError(f"{where}: regex test failed: {result.get('error')}")


def _last_index(stderr_text: "str | bytes | None") -> int | None:
    """The last integer line the child emitted = the pattern it was testing
    when we killed it (catastrophic backtracking).

    On POSIX, ``TimeoutExpired.stderr`` is raw BYTES even under
    ``text=True`` (only the Windows branch of CPython re-decodes after a
    timeout) — normalize first so str-only handling can't TypeError inside
    the timeout path this function exists for."""
    if isinstance(stderr_text, bytes):
        stderr_text = stderr_text.decode("utf-8", "replace")
    if not stderr_text:
        return None
    last = None
    for line in stderr_text.splitlines():
        line = line.strip()
        if _ascii_digits(line):
            last = int(line)
    return last


def _probe(checks: list):
    """Child side: run each [pattern, replacement] against FIXTURE and then
    against the _ADVERSARIAL inputs, a per-pattern synthetic run, and the
    CHAINED fixture (the fixture as the preceding entries have rewritten it).
    Return (index, message) on the first failure, else None. Emit the index to
    stderr before each test so the parent can name the culprit if it kills us.
    """
    import re
    import sys
    import time
    # The fixture as it looks after entries 0..i-1 have been applied, in LIST
    # ORDER (engine.rebuild_caches applies them in list order, unsorted). Entry 0
    # sees the plain fixture; every later entry is additionally probed against
    # whatever its predecessors manufactured.
    chained = FIXTURE
    for i, item in enumerate(checks):
        sys.stderr.write("%d\n" % i)
        sys.stderr.flush()
        try:
            rx = re.compile(item[0])
            # process_time, not perf_counter: this child shares the box with
            # live transcription, and a wall clock charges the pattern for
            # every de-schedule the OS lands mid-probe — measured 20-36x fake
            # ratios on µs-scale factory rules under CPU load, which 422'd
            # perfectly valid saves. CPU time only counts what the regex burns.
            _t0 = time.process_time()
            out = rx.sub(item[1], FIXTURE)
            _t_base = time.process_time() - _t0
            # Scaling probe, measured back-to-back with the baseline so the
            # ratio reflects the pattern and not whatever the machine did in
            # between. The VERDICT stays below, after the growth checks, so
            # rejection priority is unchanged. See the comment there.
            _t0 = time.process_time()
            rx.sub(item[1], _SCALE_FIXTURE)
            _t_scaled = time.process_time() - _t0
        except Exception as exc:  # noqa: BLE001 - any compile/sub failure
            return i, str(exc)
        if len(out) > _MAX_GROWTH * len(FIXTURE):
            return i, (
                f"replacement grew the 1 KB fixture "
                f"{len(out) / len(FIXTURE):.0f}x (limit {_MAX_GROWTH}x). "
                "Simplify the replacement."
            )
        # The growth check above measures NOTHING when the pattern matches the
        # fixture zero times — and the fixtures are fixed German prose, so the
        # letters they happen to lack (n, d, m, ...) are a free pass. A rule
        # like ("n", "n"*512) scores a growth ratio of 1.0 here and then
        # amplifies a real transcript 512x per match on every transcription.
        # It measures almost nothing when the pattern matches the fixture only
        # a few times, either: `^Hallo|n` -> "x"*512 matches once via `^Hallo`
        # and then expands every "n" of a real transcript. So bound EVERY
        # replacement analytically as well: compare the characters it always
        # contributes against the shortest string the pattern can match.
        try:
            try:
                import re._parser as _reparser
            except ImportError:  # pragma: no cover - Python < 3.11
                import sre_parse as _reparser
            _min_match = _reparser.parse(item[0]).getwidth()[0]
        except Exception:  # noqa: BLE001 - width analysis is best effort
            _min_match = 0
        _mm = max(_min_match, 1)
        # A group reference is NOT free: it contributes whatever the group
        # captured, so ("(n+)", "\1"*256) amplifies exactly as hard as
        # ("n", "n"*256) — but deleting the references outright measures it
        # as an empty replacement and waves it through. Charge each one the
        # shortest string the pattern can match, which is the least it can
        # ever expand to, and count it alongside the literal characters.
        _refs = len(re.findall(r"\\(?:\d+|g<[^>]*>)", item[1]))
        _literal = re.sub(r"\\(?:\d+|g<[^>]*>)", "", item[1])
        _grown = len(_literal) + _refs * _mm
        # The _MIN_ABS_GROWTH floor is for bounded literal text only. A
        # reference copies an UNBOUNDED capture, so it is held to the ratio
        # alone: ("(n+)", "\1"*64) is under the floor and still 64x per match.
        if (_grown > max(_MAX_GROWTH * _mm, _MIN_ABS_GROWTH)
                or _refs > _MAX_GROWTH):
            return i, (
                f"replacement is {_grown} characters ({_refs} group "
                f"references) for a pattern that can match as few as "
                f"{_min_match} (limit {_MAX_GROWTH}x), so every match on a "
                "real transcript would grow by that much. Simplify the "
                "replacement."
            )
        # Timing probe only: a pattern that is fast on German prose but slow on
        # repetitive input hangs here and the parent's timeout kills us. The
        # growth check stays on FIXTURE alone, so these can't invent a new
        # rejection reason for an otherwise fine replacement.
        probes = list(_ADVERSARIAL)
        # Per-pattern synthetic run: closes the fixed-alphabet hole above for
        # any character, not just the ones the four fixtures happen to hold.
        synthetic = _synthetic_fixture(item[0])
        if synthetic:
            probes.append(synthetic)
        # The chained fixture: what this entry is actually handed at match time
        # once its predecessors have rewritten the text. An entry that is
        # harmless on prose but explodes on a long run some EARLIER entry
        # manufactures ((\w) -> 60 dashes, then (-+)(-+)... on the result) is
        # invisible to every static fixture and only shows up here.
        if chained and chained != FIXTURE:
            probes.append(chained)
        for fixture in probes:
            try:
                rx.sub(item[1], fixture)
            except Exception as exc:  # noqa: BLE001 - any sub failure
                return i, str(exc)
        # Scaling verdict (measured above). The structural screen catches
        # EXPONENTIAL shapes and the fixed-size probes catch anything already
        # slow at ~1 KB, but a merely POLYNOMIAL pattern is fast at 1 KB by
        # construction and only bites at transcript length — `.*.*#` costs
        # 0.2 s on FIXTURE and ~40 s at 6 KB. The longer fixture repeats
        # FIXTURE, so only the LENGTH changes: linear work grows with the
        # length multiplier, and anything growing far faster is superlinear in
        # the input it will actually be applied to.
        # _TIMER_FLOOR keeps timer noise on a microsecond-scale legit rule from
        # tripping the ratio; the allowance is double the length multiplier, so
        # ordinary rules have ample headroom while `.*.*#` (~130x) is caught.
        # A first-sample trip is confirmed before it rejects: re-measure both
        # sides and keep the best (min) of each — noise only ever inflates a
        # CPU-time reading, so the minima are the pattern's true cost. Legit
        # rules pay nothing for this (no trip, no resample); a genuinely
        # polynomial pattern trips every sample. The absolute _SCALE_MIN_REJECT
        # floor then keeps a sub-5 ms scaled run from rejecting on ratio alone.
        if _t_scaled > _SCALE_ALLOWANCE * max(_t_base, _TIMER_FLOOR):
            try:
                for _ in range(_SCALE_SAMPLES - 1):
                    _t0 = time.process_time()
                    rx.sub(item[1], FIXTURE)
                    _t_base = min(_t_base, time.process_time() - _t0)
                    _t0 = time.process_time()
                    rx.sub(item[1], _SCALE_FIXTURE)
                    _t_scaled = min(_t_scaled, time.process_time() - _t0)
            except Exception as exc:  # noqa: BLE001 - any sub failure
                return i, str(exc)
        if (_t_scaled > _SCALE_ALLOWANCE * max(_t_base, _TIMER_FLOOR)
                and _t_scaled > _SCALE_MIN_REJECT):
            return i, (
                f"match time grows faster than the input ({_t_scaled / max(_t_base, _TIMER_FLOOR):.0f}x "
                f"slower on a {_SCALE_FACTOR}x longer text). This pattern is fast "
                "on a short sample but stalls on a full transcript — simplify it "
                "(a leading or trailing `.*` is rarely needed; anchor instead)."
            )
        # This entry passed, so feed its OUTPUT to the next one, exactly as the
        # pipeline does. Truncated to _CHAIN_CAP so a legitimately expanding
        # rule set can't make the chained probe itself the expensive thing.
        chained = _chain_advance(rx, item[0], item[1], chained)
    return None


def _main() -> None:
    import json
    import sys
    try:
        checks = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        sys.stdout.write(json.dumps({"ok": False, "index": -1, "error": "bad input"}))
        return
    res = _probe(checks)
    if res is None:
        sys.stdout.write(json.dumps({"ok": True}))
    else:
        sys.stdout.write(json.dumps({"ok": False, "index": res[0], "error": res[1]}))


if __name__ == "__main__":
    _main()
