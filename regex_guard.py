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
fast on prose but slow on a long unpunctuated run is caught too. A pattern
crafted to blow up only on some other input shape can still pass and hang the
MAIN process at match time (rule application is in-process re.sub) — accepted
residual risk for a rule surface that is user-editable by design.

Two roles, one file:
  * parent  ->  ``import regex_guard; regex_guard.validate(checks)``
  * child   ->  ``python regex_guard.py``   (reads a JSON list of ``[pattern,
                replacement]`` pairs on stdin, writes a JSON verdict on stdout)
"""
from __future__ import annotations

import os

# ~1 KB realistic fixture (mirrors the former in-process guard's fixture).
FIXTURE = "Hallo. Wie geht's? 10.23 Uhr! Bitte. " * 32

# Whole-validation wall-clock budget. Legit find->replace patterns finish in
# microseconds against a 1 KB input; only catastrophic backtracking approaches
# this. Module-level so tests can lower it.
_GUARD_TIMEOUT = 2.0

# Max output/input length ratio for ONE substitution against the fixture. A
# legit correction rule shrinks the text or barely grows it; an expanding
# replacement (`(.*)` -> `\1` repeated) costs microseconds but multiplies its
# input on EVERY chained rule, so a handful of entries turn a short transcript
# into gigabytes at match time. Anything above 1x compounds; 10x is generous.
_MAX_GROWTH = 10

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

_SELF = os.path.abspath(__file__)


def _read_quantifier(pat: str, i: int) -> "tuple[bool, bool, int]":
    """Read a quantifier at ``pat[i]``. Returns (repeats, atomic, next_index).

    ``repeats`` is True only for a quantifier that can match an atom MORE THAN
    ONCE (`*`, `+`, `{2,}`, `{0,3}`) — `?` and `{0,1}` are optional, not
    repetition, and are far too common in ordinary rules to treat as risky.
    ``atomic`` is True for a possessive quantifier (`*+`, `++`, `{n,m}+`),
    which cannot give back characters and therefore cannot backtrack.
    """
    if i >= len(pat):
        return False, False, i
    c = pat[i]
    if c in "*+?":
        repeats = c != "?"
        i += 1
    elif c == "{":
        j = pat.find("}", i)
        if j == -1:
            return False, False, i  # a literal `{`, not a quantifier
        lo, comma, hi = pat[i + 1:j].partition(",")
        if comma:
            if not (lo.isdigit() or not lo) or not (hi.isdigit() or not hi):
                return False, False, i
            repeats = not hi or int(hi) > 1  # {n,} is unbounded
        else:
            if not lo.isdigit():
                return False, False, i
            repeats = int(lo) > 1
        i = j + 1
    else:
        return False, False, i
    atomic = i < len(pat) and pat[i] == "+"
    if i < len(pat) and pat[i] in "?+":
        i += 1  # lazy or possessive suffix
    return repeats, atomic, i


def _nested_repetition(pat: str) -> bool:
    """True if ``pat`` contains a repeated group that itself repeats.

    `(x+)+`, `(x*)*`, `(x{2,})+` and the alternation-overlap `(a|a)*` are the
    shapes whose match time is EXPONENTIAL in the input length: every way of
    splitting the input between the inner and outer repetition is a distinct
    path the engine must try before it can fail. `(\\w+ ?)+#` needs 2**n steps
    on an n-word sentence — a fixture probe can't outrun that, so the shape
    itself has to be refused.

    Scans the pattern string once; there is no way to match this with a regex
    and no stdlib API that exposes the parse tree. `(`, `)`, `*`, `+` and `{`
    are LITERAL inside a character class and after a backslash, so both are
    skipped — a scanner that miscounts them rejects perfectly good rules.
    """
    n = len(pat)
    # One frame per open group plus a root frame. "rep": this level contains a
    # repetition; "start"/"alts": body slice + top-level `|` offsets, for the
    # (a|a)* overlap check; "atomic": (?>...) can't backtrack into itself.
    stack = [{"rep": False, "start": 0, "alts": [], "atomic": False}]
    i = 0
    while i < n:
        c = pat[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            i += 1
            if i < n and pat[i] == "^":
                i += 1
            if i < n and pat[i] == "]":
                i += 1  # a `]` first in the class is a literal
            while i < n and pat[i] != "]":
                i += 2 if pat[i] == "\\" else 1
            i += 1
            continue
        if c == "(":
            atomic = False
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
                    j = k + 1
                elif k < n and pat[k] == "<" and k + 1 < n and pat[k + 1] in "=!":
                    j = k + 2
                else:
                    # (?P<name>, (?P=name), (?<name>, inline flags (?i) / (?i:
                    end = pat.find(">", k)
                    close = pat.find(")", k)
                    colon = pat.find(":", k)
                    cand = [x for x in (end, colon) if x != -1 and (close == -1 or x < close)]
                    j = (min(cand) + 1) if cand else (close + 1 if close != -1 else n)
            stack.append({"rep": False, "start": j, "alts": [], "atomic": atomic})
            i = j
            continue
        if c == ")" and len(stack) > 1:
            frame = stack.pop()
            body = pat[frame["start"]:i]
            i += 1
            repeats, possessive, i = _read_quantifier(pat, i)
            if repeats and not frame["atomic"] and not possessive:
                if frame["rep"]:
                    return True
                # (a|a)* — identical alternatives overlap, so the engine
                # re-tries the same split every way round.
                if frame["alts"]:
                    cuts = [frame["start"]] + [x + 1 for x in frame["alts"]]
                    ends = frame["alts"] + [frame["start"] + len(body)]
                    branches = [pat[a:b] for a, b in zip(cuts, ends)]
                    if len(set(branches)) < len(branches):
                        return True
            if frame["rep"] or repeats:
                stack[-1]["rep"] = True
            continue
        if c == "|":
            stack[-1]["alts"].append(i)
            i += 1
            continue
        repeats, _, j = _read_quantifier(pat, i)
        if j > i:
            if repeats:
                stack[-1]["rep"] = True
            i = j
            continue
        i += 1
    return False


def validate(checks: list, timeout: float | None = None) -> None:
    """Validate every regex in ``checks`` out-of-process.

    ``checks`` is a list of ``(where, pattern, replacement)`` tuples. Each
    pattern's ``re.sub`` is run against a fixed ~1 KB fixture (plus a few short
    repetitive worst-case inputs) inside a child process that is killed if it
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
    for where, pattern, _repl in ((c[0], c[1], c[2]) for c in checks):
        if pattern and _nested_repetition(pattern):
            raise ValueError(
                f"{where}: regex test failed: nested repetition "
                "(a repeat inside a repeated group, e.g. `(\\w+ ?)+`) causes "
                "catastrophic backtracking and can hang the server on ordinary "
                "input — rewrite without a repeat inside a repeat."
            )
    import json
    import logging
    import subprocess
    import sys

    budget = _GUARD_TIMEOUT if timeout is None else timeout
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
            f"{where}: regex took > {budget:.0f} s on a 1 KB fixture "
            "(likely catastrophic backtracking). Simplify the pattern."
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
        if line.isdigit():
            last = int(line)
    return last


def _probe(checks: list):
    """Child side: run each [pattern, replacement] against FIXTURE and then
    against the _ADVERSARIAL inputs. Return (index, message) on the first
    failure, else None. Emit the index to stderr before each test so the parent
    can name the culprit if it kills us."""
    import re
    import sys
    import time
    for i, item in enumerate(checks):
        sys.stderr.write("%d\n" % i)
        sys.stderr.flush()
        try:
            rx = re.compile(item[0])
            _t0 = time.perf_counter()
            out = rx.sub(item[1], FIXTURE)
            _t_base = time.perf_counter() - _t0
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
        # For that case only, bound the replacement analytically instead:
        # compare the characters it always contributes against the shortest
        # string the pattern can possibly match. Patterns the fixture DOES
        # exercise keep the measured check and are unaffected.
        if not rx.search(FIXTURE):
            try:
                try:
                    import re._parser as _reparser
                except ImportError:  # pragma: no cover - Python < 3.11
                    import sre_parse as _reparser
                _min_match = _reparser.parse(item[0]).getwidth()[0]
            except Exception:  # noqa: BLE001 - width analysis is best effort
                _min_match = 0
            # Drop group references: their length comes from the match, not
            # from the replacement text, so they can't be counted as growth.
            _literal = re.sub(r"\\(?:\d+|g<[^>]*>)", "", item[1])
            if len(_literal) > _MAX_GROWTH * max(_min_match, 1):
                return i, (
                    f"replacement is {len(_literal)} characters for a pattern "
                    f"that can match as few as {_min_match} "
                    f"(limit {_MAX_GROWTH}x). The test fixture never matches "
                    "this pattern, so the growth it would cause on a real "
                    "transcript cannot be measured. Simplify the replacement."
                )
        # Timing probe only: a pattern that is fast on German prose but slow on
        # repetitive input hangs here and the parent's timeout kills us. The
        # growth check stays on FIXTURE alone, so these can't invent a new
        # rejection reason for an otherwise fine replacement.
        for fixture in _ADVERSARIAL:
            try:
                rx.sub(item[1], fixture)
            except Exception as exc:  # noqa: BLE001 - any sub failure
                return i, str(exc)
        # Scaling probe. The structural screen catches EXPONENTIAL shapes and
        # the fixed-size probes above catch anything already slow at ~1 KB, but
        # a merely POLYNOMIAL pattern is fast at 1 KB by construction and only
        # bites at transcript length — `.*.*#` costs 0.2 s on FIXTURE and ~40 s
        # at 6 KB. Re-run on a longer fixture of the SAME shape and compare:
        # linear work grows with the length multiplier, so anything growing far
        # faster is superlinear in the input it will actually be applied to.
        try:
            _t0 = time.perf_counter()
            rx.sub(item[1], _SCALE_FIXTURE)
            _t_scaled = time.perf_counter() - _t0
        except Exception as exc:  # noqa: BLE001 - any sub failure
            return i, str(exc)
        # _TIMER_FLOOR keeps timer noise on a microsecond-scale legit rule from
        # tripping the ratio; the allowance is double the length multiplier, so
        # ordinary rules have ample headroom while `.*.*#` (~130x) is caught.
        if _t_scaled > _SCALE_ALLOWANCE * max(_t_base, _TIMER_FLOOR):
            return i, (
                f"match time grows faster than the input ({_t_scaled / max(_t_base, _TIMER_FLOOR):.0f}x "
                f"slower on a {_SCALE_FACTOR}x longer text). This pattern is fast "
                "on a short sample but stalls on a full transcript — simplify it "
                "(a leading or trailing `.*` is rarely needed; anchor instead)."
            )
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
