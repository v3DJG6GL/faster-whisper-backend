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
``/v1/pipeline-rules`` editor) without a crafted pattern being able to hang the
server.

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

_SELF = os.path.abspath(__file__)


def validate(checks: list, timeout: float | None = None) -> None:
    """Validate every regex in ``checks`` out-of-process.

    ``checks`` is a list of ``(where, pattern, replacement)`` tuples. Each
    pattern's ``re.sub`` is run against a fixed ~1 KB fixture inside a child
    process that is killed if it exceeds ``timeout`` seconds. Raises
    ``ValueError(f"{where}: ...")`` on a bad regex/replacement (e.g. a backref
    to a non-existent group) or a catastrophic-backtracking timeout. No-op for
    an empty list.

    Fails OPEN: if the helper can't be launched / crashes, the save proceeds
    WITHOUT the backtracking check rather than blocking a legitimate edit — the
    guard is a safety improvement, never a gate that can break saving.
    """
    if not checks:
        return
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


def _last_index(stderr_text: str | None) -> int | None:
    """The last integer line the child emitted = the pattern it was testing
    when we killed it (catastrophic backtracking)."""
    if not stderr_text:
        return None
    last = None
    for line in stderr_text.splitlines():
        line = line.strip()
        if line.isdigit():
            last = int(line)
    return last


def _probe(checks: list):
    """Child side: run each [pattern, replacement] against FIXTURE. Return
    (index, message) on the first failure, else None. Emit the index to stderr
    before each test so the parent can name the culprit if it kills us."""
    import re
    import sys
    for i, item in enumerate(checks):
        sys.stderr.write("%d\n" % i)
        sys.stderr.flush()
        try:
            re.compile(item[0]).sub(item[1], FIXTURE)
        except Exception as exc:  # noqa: BLE001 - any compile/sub failure
            return i, str(exc)
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
