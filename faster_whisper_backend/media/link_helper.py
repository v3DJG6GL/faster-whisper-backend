"""The helper process a pasted link is checked in.

Checking a link is regex work on text the caller or a site wrote: the URL is
matched against every yt-dlp site pattern (about 1 750 of them), and yt-dlp
scans the page a site answers with. Python's ``re`` holds the GIL for a whole
match and cannot be interrupted in-process, so one slow match would stop every
thread of the server for as long as it runs. A check that runs in a helper
has a deadline instead: the parent waits on the connection (GIL released) and
kills the helper when the deadline passes. What a check is and how its answer
is read are media/download.py's business; this module is the transport.

One helper serves ONE check and is then killed, so nothing a site sent stays
behind for the next link, and a check that is stopped takes no other check
with it. Starting a helper (the interpreter, yt-dlp, compiling its patterns)
takes most of a second, so one started, idle helper is kept ready (the
"spare") and replaced as soon as it is taken: a check then costs a few
milliseconds over running in-process, and only checks that arrive faster
than a spare starts wait for their own helper.

Two roles, one file (the shape of pipeline/rule_helper.py, whose transport
this follows):
  * parent -> ``start()`` / ``shutdown()`` (the app's lifespan), ``active()``
              and ``run()`` (media/download.py).
  * helper -> ``helper_main()``, started as ``python -P -c`` by ``_spawn``
              with the repo root added at the END of its import path: no
              directory of the repo can stand in for a module yt-dlp's
              extractors import.

The two sides talk over a ``multiprocessing.connection`` pair made by the
parent, the helper's end inherited at its start: ``poll(timeout)`` is the
same timed wait on every platform, and no address exists that another
process could connect to.

Stdlib-only at import: the helper imports this module before anything else.
"""
from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from multiprocessing import connection as _connection

from faster_whisper_backend import paths as _paths

logger = logging.getLogger("whisper-api")

# Whether start() starts anything. False only under the test suite
# (tests/conftest.py): hundreds of tests check links against a stand-in
# yt-dlp patched into the test process, which a helper process would not
# see. With no helper started the checks run in-process, the same code; the
# helper path has its own tests, which turn this on.
ISOLATE = True

# The helper's first message, sent once its imports are done.
_READY = b"ready"
# The first byte of a helper's reply: the check's own answer follows, or the
# text of what kept the helper from running the check at all.
_ANSWERED = b"+"
_FAILED = b"!"
# The largest request (a URL of at most 2 048 characters and the link
# settings) and the largest reply (a site's info dict as JSON: a few hundred
# KB for a real video, and every list in it is already capped by the yt-dlp
# guard) either side reads.
_REQUEST_MAX = 1 << 20
_REPLY_MAX = 64 << 20
# How long past its request's own time limit a helper may live before it
# ends itself. The parent enforces the real deadline and kills the helper;
# this only covers a parent that died without doing so.
_SELF_LIMIT_MARGIN_S = 30.0
# How much of a failed helper's own words are kept (they go to the log).
_FAILURE_TEXT_MAX = 500

_BOOT = ("import sys; sys.path.append(sys.argv[1]); "
         "from faster_whisper_backend.media import link_helper; "
         "link_helper.helper_main(int(sys.argv[2]))")


class HelperTimeout(Exception):
    """The helper did not answer within the caller's deadline (it is killed)."""


class HelperFailed(Exception):
    """No helper could be started, or it ended without an answer. str() is
    for the server log, never for a client."""


class _Helper:
    __slots__ = ("proc", "conn")

    def __init__(self, proc, conn) -> None:
        self.proc = proc
        self.conn = conn


def _child_end(conn) -> "tuple[int, dict]":
    """The helper's end of the pair as the helper will know it, and the Popen
    arguments that let exactly this one handle through to it (rule_helper's
    twin of the same name)."""
    end = conn.fileno()
    if os.name == "nt":
        os.set_handle_inheritable(end, True)
        startupinfo = subprocess.STARTUPINFO(lpAttributeList={"handle_list": [end]})
        return end, {"startupinfo": startupinfo}
    return end, {"pass_fds": (end,)}


def _spawn() -> _Helper:
    """Start one helper on its end of a connected pair. Returns as soon as
    the process exists: its start-up goes on in the background, and run()
    waits for its first message."""
    conn, theirs = _connection.Pipe(duplex=True)
    try:
        try:
            end, inherit = _child_end(theirs)
            proc = subprocess.Popen(
                [sys.executable, "-P", "-c", _BOOT, _paths.REPO_ROOT, str(end)],
                # Nothing of the helper's reaches the server's console: what
                # yt-dlp prints is site text, and the helper reports its own
                # failures in its reply.
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Its own process group (POSIX), so a program yt-dlp started
                # for a site (a JavaScript runtime) is killed with it.
                start_new_session=os.name != "nt",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **inherit)
        finally:
            # The helper holds its own copy; with this one closed, a helper
            # that ends shows as EOF on `conn`.
            theirs.close()
    except BaseException:
        conn.close()
        raise
    return _Helper(proc, conn)


def _stop(proc) -> None:
    """Kill the helper (and what it started) and reap it. SIGKILL at once: a
    helper inside a slow match cannot act on anything gentler, and it holds
    nothing to flush."""
    from faster_whisper_backend.core import proc as _proc  # noqa: PLC0415 - parent only
    # Only while it is unreaped: after that its pid may be another process's.
    # A helper that has answered is killed all the same: its group may still
    # hold a program it started.
    if proc.returncode is None:
        try:
            _proc.kill(proc)
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("[url-dl] link check helper %s did not exit after kill", proc.pid)


class _Helpers:
    """The spare and every helper that runs a check, so a shutdown finds
    them all."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spare: "_Helper | None" = None
        self._starting_spare = False
        self._live: "set[_Helper]" = set()
        self._closed = False
        self._spare_failure_logged = False

    def _start_one(self) -> _Helper:
        """A new helper, registered. HelperFailed when the pool is shut down
        or the process cannot be started."""
        with self._lock:
            if self._closed:
                raise HelperFailed("the link check helpers are shut down")
        try:
            helper = _spawn()
        except Exception as exc:  # noqa: BLE001 - any start failure: no check runs
            raise HelperFailed(
                f"could not start a link check helper: {type(exc).__name__}: {exc}") from None
        with self._lock:
            if not self._closed:
                self._live.add(helper)
                return helper
        self.end(helper)
        raise HelperFailed("the link check helpers are shut down")

    def refill(self) -> None:
        """Start the next spare unless one is there or on its way. Never
        raises: without a spare the next check starts its own helper."""
        with self._lock:
            if self._closed or self._spare is not None or self._starting_spare:
                return
            self._starting_spare = True
        helper = None
        try:
            helper = self._start_one()
        except HelperFailed as exc:
            if not self._spare_failure_logged:
                self._spare_failure_logged = True
                logger.warning("[url-dl] %s; the next link check starts its own", exc)
        with self._lock:
            self._starting_spare = False
            if helper is not None and not self._closed:
                self._spare_failure_logged = False
                self._spare, helper = helper, None
        if helper is not None:
            self.end(helper)

    def take(self) -> _Helper:
        """A helper for one check: the spare when it is alive, else a new
        one. The next spare is started before this returns."""
        with self._lock:
            helper, self._spare = self._spare, None
        if helper is not None and helper.proc.poll() is not None:
            self.end(helper)
            helper = None
        if helper is None:
            helper = self._start_one()
        self.refill()
        return helper

    def end(self, helper: _Helper) -> None:
        """Kill a helper, reap it and close this side's end of its pair."""
        _stop(helper.proc)
        try:
            helper.conn.close()
        except OSError:
            pass
        with self._lock:
            self._live.discard(helper)

    def shutdown(self) -> None:
        """Kill every helper, the spare and the ones inside a check (those
        checks end as failed)."""
        with self._lock:
            self._closed = True
            spare, self._spare = self._spare, None
            live = [h for h in self._live if h is not spare]
        if spare is not None:
            self.end(spare)
        # A check in flight closes its own end once it sees its helper gone.
        for helper in live:
            _stop(helper.proc)

    def pids(self) -> "list[int]":
        """The live helpers' process ids (tests, diagnostics)."""
        with self._lock:
            return sorted(h.proc.pid for h in self._live if h.proc.poll() is None)

    def spare_pid(self) -> "int | None":
        with self._lock:
            return self._spare.proc.pid if self._spare is not None else None


# None until the app has started the helpers: checks then run in-process.
# After shutdown() the closed object stays, and run() refuses.
_HELPERS: "_Helpers | None" = None
_HELPERS_LOCK = threading.Lock()
_atexit_registered = False


def active() -> bool:
    """Whether link checks go through a helper (start() has run)."""
    return _HELPERS is not None


def start(*, spare: bool = True) -> None:
    """Route link checks through helper processes from now on (idempotent).
    `spare`: start the first idle helper right away, so the first pasted
    link does not wait for one. Never raises: a helper that cannot be
    started fails the check that needed it (see run), never the startup."""
    global _HELPERS, _atexit_registered
    if not ISOLATE:
        return
    with _HELPERS_LOCK:
        if _HELPERS is None or _HELPERS._closed:
            _HELPERS = _Helpers()
            if not _atexit_registered:
                # For an exit that skips the app's shutdown: an idle helper
                # would end on its own (its connection closes), this reaps
                # it first.
                atexit.register(shutdown)
                _atexit_registered = True
        helpers = _HELPERS
    if spare:
        helpers.refill()


def shutdown() -> None:
    """Kill the helpers. Link checks are refused afterwards (never run
    in-process instead) until the next start()."""
    with _HELPERS_LOCK:
        helpers = _HELPERS
    if helpers is not None:
        helpers.shutdown()


def reset() -> None:
    """Back to the never-started state (tests): checks run in-process."""
    global _HELPERS
    shutdown()
    with _HELPERS_LOCK:
        _HELPERS = None


def run(payload: bytes, timeout: float) -> bytes:
    """Run one check in a helper of its own and return its answer. The
    helper is killed when this returns, however it returns. HelperTimeout
    when the answer did not come within `timeout` seconds (the helper's
    start counts); HelperFailed when no helper could be started or it ended
    without an answer: the caller refuses the link, it never runs the check
    in-process instead."""
    helpers = _HELPERS
    if helpers is None:
        raise HelperFailed("the link check helpers were never started")
    end = time.monotonic() + timeout
    helper = helpers.take()
    lost: "BaseException | None" = None
    reply = b""
    try:
        try:
            if not helper.conn.poll(max(0.0, end - time.monotonic())):
                raise HelperTimeout
            if helper.conn.recv_bytes(len(_READY)) != _READY:
                raise OSError("unexpected first message")
            helper.conn.send_bytes(struct.pack("!d", max(0.0, timeout)) + payload)
            if not helper.conn.poll(max(0.0, end - time.monotonic())):
                raise HelperTimeout
            # Larger than _REPLY_MAX is an OSError: nothing of it is read.
            reply = helper.conn.recv_bytes(_REPLY_MAX)
        except (EOFError, OSError) as exc:
            lost = exc
    finally:
        helpers.end(helper)
    if lost is not None:
        raise HelperFailed(
            f"the link check helper ended without an answer (exit code "
            f"{helper.proc.returncode}): {type(lost).__name__}: {lost}")
    if reply[:1] == _ANSWERED:
        return reply[1:]
    raise HelperFailed(
        "the link check helper could not run the check: "
        + reply[1:1 + _FAILURE_TEXT_MAX].decode("utf-8", "replace"))


def _keep_end_private(end: int) -> None:
    """Stop the helper's end of the pair from being inherited any further.
    It arrives inheritable (that is how the helper got it), and whoever
    holds it can write the one message the server reads as the check's
    answer: no program yt-dlp starts for a site gets it, whatever
    inheritance it is started with."""
    if os.name == "nt":
        os.set_handle_inheritable(end, False)
    else:
        os.set_inheritable(end, False)


def helper_main(end: int) -> None:
    """The helper process: take the inherited end of the parent's pair, load
    what a check needs, answer ONE request and exit."""
    # Before the check's code is loaded: that is what can start programs.
    _keep_end_private(end)
    # A Ctrl-C in the server's terminal reaches the whole console on
    # Windows; the parent decides when a helper stops.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    conn = (_connection.PipeConnection(end) if os.name == "nt"
            else _connection.Connection(end))
    download = None
    boot_error = ""
    try:
        from faster_whisper_backend.media import download  # noqa: PLC0415 - the helper's own load
    except BaseException as exc:  # noqa: BLE001 - told to the parent in the reply
        boot_error = f"{type(exc).__name__}: {exc}"
    else:
        try:
            download.helper_warm()
        except Exception:  # noqa: BLE001 - the check itself meets whatever is missing
            pass
    try:
        conn.send_bytes(_READY)
        # An idle spare waits here. The parent going away ends the wait
        # (EOF), and with it the helper.
        request = conn.recv_bytes(_REQUEST_MAX)
    except (EOFError, OSError):
        return
    (limit,) = struct.unpack("!d", request[:8])
    # faulthandler's watchdog is a C thread that needs no GIL, so it fires
    # inside a match that never returns, on every platform.
    faulthandler.dump_traceback_later(limit + _SELF_LIMIT_MARGIN_S, exit=True)
    if download is None:
        reply = _FAILED + boot_error.encode("utf-8", "replace")
    else:
        try:
            reply = _ANSWERED + download.helper_handle(request[8:])
        except BaseException as exc:  # noqa: BLE001 - told to the parent in the reply
            reply = _FAILED + f"{type(exc).__name__}: {exc}".encode("utf-8", "replace")
    if reply[:1] == _FAILED:
        reply = reply[:1 + 4 * _FAILURE_TEXT_MAX]
    try:
        conn.send_bytes(reply)
    except (EOFError, OSError, ValueError):
        pass
    # Not a normal exit: a thread still inside a site's answer must not hold
    # the process open (the parent kills it either way).
    os._exit(0)
