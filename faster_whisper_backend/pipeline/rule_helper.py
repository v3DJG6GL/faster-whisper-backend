"""A small pool of helper processes the rules engine walks its rules in.

Python's ``re`` holds the GIL for a whole match and cannot be interrupted
in-process, so one slow user-editable rule would freeze every thread of the
server for as long as it runs. A walk that runs in a helper can be given a
deadline instead: the parent waits on the connection (GIL released), and past
the deadline it kills the helper and starts another. Which rule to blame, what
a walk is and how its result comes back are the engine's business
(pipeline/engine.py); this module is the transport and the pool.

Two roles, one file:
  * parent -> ``start()`` / ``shutdown()`` (the app's lifespan) and the pool's
              ``acquire`` / ``call`` / ``wait`` / ``release`` / ``discard``
              (the engine).
  * helper -> ``helper_main()``, started as ``python -P -c`` by ``_spawn``
              (-P: the server's working directory is not an import root for
              it). It imports the engine (and with it the settings modules,
              never the model stack or the app), takes its rules from the
              parent over the connection and never reads them from the config
              files itself.

The helper is a plain child process (``subprocess``), not a
``multiprocessing.Process``: that one re-imports the parent's ``__main__`` in
the child, which under ``python main.py`` is the whole app. The two sides talk
over a ``multiprocessing.connection`` (a Unix socket, a named pipe on Windows),
whose ``poll(timeout)`` is the same timed wait on every platform. It is one
connected pair made by the parent, the helper's end inherited at its start:
no address exists that another process could find or connect to.

Stdlib-only at import: the engine imports this module, and the helper imports
it before anything else.
"""
from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from multiprocessing import connection as _connection
from multiprocessing import shared_memory as _shared_memory

from faster_whisper_backend import paths as _paths

logger = logging.getLogger("whisper-api")

# True only in a helper process, set by helper_main before it imports the
# engine: the engine then skips its import-time compile of the configured
# rules (a helper only ever runs the rules its parent sends).
IN_HELPER = False

# Helpers the pool keeps. One is enough for the time limit itself; the others
# keep walks flowing while one is held by a slow rule or is being replaced
# (about a quarter of a second), and serve walks from several threads at once.
POOL_SIZE = 3
# Longest wait for a helper's start: the interpreter and the engine import
# take about a quarter of a second on an idle box.
_START_TIMEOUT_S = 30.0
# Longest wait for a free helper. Walks take milliseconds, so only helpers
# held by slow rules (each for at most its deadline) make a walk wait at all.
_ACQUIRE_TIMEOUT_S = 30.0
# What acquire() answers when helpers are up but none came free in the wait:
# the walk must not run in the server process instead (the rules that hold
# the helpers would hold every thread of the server there). Also its answer
# while a helper was killed for the time limit and none has started since.
BUSY = object()
# After a helper failed to start, how long until the pool makes the start up
# (with none left, walks run in-process meanwhile, unless the time limit
# cost the pool a helper: see _Pool).
_RETRY_AFTER_S = 60.0
# How long a helper may stay inside one request before it ends itself. The
# parent enforces the real deadline and kills the helper; this only covers a
# parent that died without doing so, where a helper inside a slow match would
# otherwise hold a CPU core for good. Above the longest request the engine
# makes (a walk at its longest deadline plus the wait that settles which rule
# passed it).
_SELF_LIMIT_S = 180.0
# An idle helper wakes this often to push its own limit out.
_IDLE_POLL_S = 5.0
# The helper re-arms its limit at a request only when the last arming is
# older than this (arming costs more than a short walk).
_REARM_AFTER_S = 1.0

# The helper's first message, sent once its imports are done.
_READY = b"ready"

_BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); "
         "from faster_whisper_backend.pipeline import rule_helper; "
         "rule_helper.helper_main()")


class HelperTimeout(Exception):
    """The helper did not answer within the caller's deadline."""


class HelperGone(Exception):
    """The helper died or its connection broke."""


class _Helper:
    """One helper process as the parent holds it. `mark` is two shared
    signed 64-bit cells the helper writes its progress into (the engine: the
    index of the rule about to run, and time.monotonic_ns() when it
    started); `token` names the rule set the helper has loaded (the
    engine's)."""

    __slots__ = ("proc", "conn", "shm", "mark", "token")

    def __init__(self, proc, conn, shm) -> None:
        self.proc = proc
        self.conn = conn
        self.shm = shm
        self.mark = shm.buf.cast("q")
        self.token = None


def call(helper: _Helper, payload: bytes, timeout: float) -> bytes:
    """Send one request and wait up to `timeout` seconds for its reply.
    HelperTimeout past that (the caller may `wait` on, or discards the
    helper), HelperGone when the helper died (the caller discards it)."""
    try:
        helper.conn.send_bytes(payload)
    except (EOFError, OSError) as exc:
        raise HelperGone(str(exc) or type(exc).__name__) from exc
    return wait(helper, timeout)


def wait(helper: _Helper, timeout: float) -> bytes:
    """Wait up to `timeout` (more) seconds for the reply to the request
    `call` sent. Raises as `call` does."""
    try:
        if not helper.conn.poll(max(0.0, timeout)):
            raise HelperTimeout
        return helper.conn.recv_bytes()
    except (EOFError, OSError) as exc:
        raise HelperGone(str(exc) or type(exc).__name__) from exc


def _stop(proc) -> None:
    """Kill the helper and reap it. SIGKILL at once: a helper inside a slow
    match cannot act on anything gentler, and it holds nothing to flush. The
    wait is the reaping: a container's PID 1 collects no zombies, and the
    helper is this process's own child either way."""
    from faster_whisper_backend.core import proc as _proc  # noqa: PLC0415 - parent only
    # Only while it is unreaped: after that its pid may be another process's.
    if proc.poll() is None:
        try:
            _proc.kill(proc)
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("[pipeline] rule helper %s did not exit after kill", proc.pid)
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except OSError:
            pass


def _child_end(conn) -> "tuple[int, dict]":
    """The helper's end of the pair as the helper will know it, and the Popen
    arguments that let exactly this one handle through to it."""
    end = conn.fileno()
    if os.name == "nt":
        # Handles are not inherited unless marked so AND listed: with a
        # handle_list the child gets the listed ones (and its standard
        # streams), nothing else of this process.
        os.set_handle_inheritable(end, True)
        startupinfo = subprocess.STARTUPINFO(lpAttributeList={"handle_list": [end]})
        return end, {"startupinfo": startupinfo}
    return end, {"pass_fds": (end,)}


def _await_ready(proc, conn) -> None:
    """Wait for the helper's first message. RuntimeError when it does not
    come within _START_TIMEOUT_S or the helper ends first (its end of the
    pair closes with it)."""
    try:
        if not conn.poll(_START_TIMEOUT_S):
            raise RuntimeError(f"helper did not answer within {_START_TIMEOUT_S:g} s")
        if conn.recv_bytes(len(_READY)) != _READY:
            raise RuntimeError("helper sent an unexpected first message")
    except (EOFError, OSError) as exc:
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None
        raise RuntimeError(f"helper exited with code {code}") from exc


def _spawn(register) -> _Helper:
    """Start one helper on its end of a connected pair and wait until it is
    ready. `register(proc)` is called as soon as the process exists, so a
    shutdown in the middle of the start still finds it."""
    from faster_whisper_backend.core import proc as _proc  # noqa: PLC0415 - parent only
    shm = _shared_memory.SharedMemory(create=True, size=16, track=False)
    try:
        # Made here and inherited, not listened for: an endpoint with a name
        # (a socket address, a pipe name) is one any local process can
        # connect to first. A socketpair on POSIX; on Windows one pipe
        # instance with both ends already taken.
        conn, theirs = _connection.Pipe(duplex=True)
        try:
            try:
                end, inherit = _child_end(theirs)
                proc = subprocess.Popen(
                    [_proc.python_exe(), "-P", "-c", _BOOT, _paths.REPO_ROOT],
                    env=_proc.python_env(),
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **inherit)
            finally:
                # The helper holds its own copy; with this one closed, a
                # helper that ends shows as EOF on `conn`.
                theirs.close()
            try:
                register(proc)
                proc.stdin.write(pickle.dumps((end, shm.name)))
                proc.stdin.close()
                _await_ready(proc, conn)
            except BaseException:
                _stop(proc)
                raise
        except BaseException:
            conn.close()
            raise
        helper = _Helper(proc, conn, shm)
    except BaseException:
        _free_shm(shm, None)
        raise
    # Both sides have it mapped: without its name nothing is left behind if
    # this process dies without a shutdown. (A no-op on Windows, where the
    # mapping goes with its last handle.)
    _unlink_shm(shm)
    return helper


def _unlink_shm(shm) -> None:
    try:
        shm.unlink()
    except OSError:
        pass


def _free_shm(shm, mark) -> None:
    try:
        if mark is not None:
            mark.release()
        shm.close()
    except (OSError, BufferError):
        pass
    _unlink_shm(shm)


class _Pool:
    """A fixed number of helpers and the threads waiting for one. `_alive`
    counts helpers that exist or are being started; the idle ones wait in
    `_idle` (last in, first out: the warmest helper serves next). `_up` says
    that a helper has started since the pool was last without any: it tells
    helpers that are all held (by slow rules) from helpers the platform does
    not start. `_cut` says that a helper was killed for the time limit and
    none has started since: with no helper to be had then, the rules that
    passed the limit must not run in the server process instead, where
    nothing can stop them (a quarantine does not outlast the next save of
    the rules, and some slow walks have no rule to quarantine)."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._cond = threading.Condition()
        self._idle: list[_Helper] = []
        self._procs: set = set()
        self._starters: list[threading.Thread] = []
        self._alive = 0
        self._up = False
        self._cut = False
        self._closed = False
        self._retry_at = 0.0
        self._down_logged = False
        self._busy_logged_at = float("-inf")
        with self._cond:
            self._start_locked(size)

    @property
    def closed(self) -> bool:
        """Whether shutdown() ran: no helper is handed out any more."""
        return self._closed

    def _start_locked(self, n: int) -> None:
        """Start `n` helpers in the background. Never raises: a starter
        thread that cannot be started is a start that failed."""
        self._starters = [t for t in self._starters if t.is_alive()]
        for _ in range(n):
            try:
                t = threading.Thread(target=self._start_one, name="rule-helper-start",
                                     daemon=True)
                t.start()
            except Exception as exc:  # noqa: BLE001 - "can't start new thread"
                self._start_failed_locked(exc)
                break
            # Counted after the start (_start_one's own bookkeeping waits for
            # the lock this thread holds): a thread that never ran leaves
            # neither a count nobody takes back nor an entry shutdown() joins.
            self._alive += 1
            self._starters.append(t)

    def _start_failed_locked(self, exc: BaseException) -> None:
        """One helper start failed (its count is already taken back). The
        pool makes it up after _RETRY_AFTER_S (acquire); with no helper left
        the walks run in-process until then, or not at all when the time
        limit cost the pool a helper (`_cut`)."""
        if self._alive == 0:
            self._up = False
        if not self._closed:
            self._retry_at = time.monotonic() + _RETRY_AFTER_S
            if self._alive == 0 and not self._down_logged:
                self._down_logged = True
                if self._cut:
                    logger.warning(
                        "[pipeline] rule helper process could not be started (%s) "
                        "after a rule ran past its time limit — texts are returned "
                        "without the rules applied; retrying every %g s",
                        exc, _RETRY_AFTER_S)
                else:
                    logger.warning(
                        "[pipeline] rule helper process could not be started (%s) — "
                        "rules run inside the server process without a time limit; "
                        "retrying every %g s", exc, _RETRY_AFTER_S)
        self._cond.notify_all()

    def _register(self, proc) -> None:
        with self._cond:
            if self._closed:
                raise RuntimeError("pool is shut down")
            self._procs.add(proc)

    def _start_one(self) -> None:
        try:
            helper = _spawn(self._register)
        except Exception as exc:  # noqa: BLE001 - any start failure -> in-process walks
            with self._cond:
                self._alive -= 1
                self._procs = {p for p in self._procs if p.poll() is None}
                self._start_failed_locked(exc)
            return
        with self._cond:
            if not self._closed:
                if self._down_logged:
                    self._down_logged = False
                    logger.info("[pipeline] rule helper processes are back")
                self._up = True
                self._cut = False
                self._idle.append(helper)
                self._cond.notify()
                return
        self._close(helper)

    def acquire(self, timeout: "float | None" = None):
        """A free helper, waiting up to `timeout` seconds (_ACQUIRE_TIMEOUT_S
        when None) for one if all are busy or still starting. BUSY when
        helpers are up but none came free in that time: the caller gives the
        walk up. None when the walk has to run in-process: the pool is shut
        down, no helper could be started, or none has started in that time.
        The last two answer BUSY instead while `_cut` is set: no helper is
        there because of the time limit, and the walk is given up."""
        end = time.monotonic() + (_ACQUIRE_TIMEOUT_S if timeout is None else timeout)
        with self._cond:
            while True:
                if self._closed:
                    return None
                down = self._alive == 0
                if self._alive < self._size and time.monotonic() >= self._retry_at:
                    # Starts that failed are made up here, one round per
                    # _RETRY_AFTER_S (a failed start pushes _retry_at out).
                    self._start_locked(self._size - self._alive)
                if self._idle:
                    return self._idle.pop()
                if down:
                    return BUSY if self._cut else None
                left = end - time.monotonic()
                if left <= 0:
                    busy = self._up or self._cut
                    now = time.monotonic()
                    if now - self._busy_logged_at > _RETRY_AFTER_S:
                        self._busy_logged_at = now
                        if self._up:
                            logger.warning(
                                "[pipeline] no rule helper process came free in %g s "
                                "(rules running past their time limit hold them) — the "
                                "text is returned without the rules applied",
                                _ACQUIRE_TIMEOUT_S)
                        elif busy:
                            logger.warning(
                                "[pipeline] no rule helper process has started in %g s "
                                "after a rule ran past its time limit — the text is "
                                "returned without the rules applied", _ACQUIRE_TIMEOUT_S)
                        else:
                            logger.warning(
                                "[pipeline] no rule helper process has started in %g s "
                                "— this walk runs inside the server process without a "
                                "time limit", _ACQUIRE_TIMEOUT_S)
                    return BUSY if busy else None
                self._cond.wait(left)

    def release(self, helper: _Helper) -> None:
        """Hand a helper back after a request it answered."""
        with self._cond:
            if not self._closed:
                self._idle.append(helper)
                self._cond.notify()
                return
        self._close(helper)

    def discard(self, helper: _Helper, timed_out: bool = False) -> None:
        """Kill a helper that timed out or died, and start its replacement.
        `timed_out`: it is killed for the time limit, inside a walk (`_cut`
        until a helper starts)."""
        self._close(helper)
        with self._cond:
            self._alive -= 1
            if timed_out:
                self._cut = True
            if not self._closed:
                self._start_locked(1)
            self._cond.notify_all()

    def _close(self, helper: _Helper) -> None:
        _stop(helper.proc)
        try:
            helper.conn.close()
        except OSError:
            pass
        _free_shm(helper.shm, helper.mark)
        with self._cond:
            self._procs.discard(helper.proc)

    def shutdown(self) -> None:
        """Kill every helper, the busy and the starting ones included. A walk
        in flight sees its helper gone and finishes in-process, unless it
        was inside a rule: that one ends without its rules (the engine)."""
        with self._cond:
            self._closed = True
            idle, self._idle = self._idle, []
            procs = list(self._procs)
            starters = list(self._starters)
            self._cond.notify_all()
        for helper in idle:
            self._close(helper)
        for proc in procs:
            _stop(proc)
        # A start in flight fails as soon as its process is gone.
        for t in starters:
            t.join(timeout=_START_TIMEOUT_S)

    def pids(self) -> list[int]:
        """The live helpers' process ids (tests, diagnostics)."""
        with self._cond:
            return sorted(p.pid for p in self._procs if p.poll() is None)

    def wait_ready(self, timeout: float) -> bool:
        """Whether every helper has started within `timeout` seconds."""
        end = time.monotonic() + timeout
        with self._cond:
            while len(self._idle) < self._size:
                left = end - time.monotonic()
                if left <= 0 or self._closed or self._alive < self._size:
                    return False
                self._cond.wait(left)
            return True


# The running pool, or None: walks then run in-process. Read without the lock
# on every walk; start / shutdown swap it under _POOL_LOCK.
_POOL: "_Pool | None" = None
_POOL_LOCK = threading.Lock()
_atexit_registered = False


def start(size: int = POOL_SIZE) -> None:
    """Start the helper pool (idempotent). Returns at once: the helpers come
    up in the background, and a walk that arrives first waits for one. Never
    raises: helpers that cannot be started are logged and retried by the pool,
    and the walks run in-process meanwhile (not so once a helper was killed
    for the time limit: see _Pool)."""
    global _POOL, _atexit_registered
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = _Pool(size)
            if not _atexit_registered:
                # For an exit that skips the app's shutdown: the helpers
                # would end on their own (their connection closes), this
                # reaps them first.
                atexit.register(shutdown)
                _atexit_registered = True


def shutdown() -> None:
    """Stop the pool and kill its helpers; walks run in-process afterwards."""
    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        pool.shutdown()


def helper_main() -> None:
    """The helper process: take the inherited end of the parent's pair, then
    answer its requests one at a time with the engine's handler until the
    parent goes away."""
    global IN_HELPER
    IN_HELPER = True
    end, shm_name = pickle.loads(sys.stdin.buffer.read())
    # A Ctrl-C in the server's terminal reaches the whole foreground group;
    # the parent decides when a helper stops.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    armed_at = _arm_self_limit()
    from faster_whisper_backend.pipeline import engine  # noqa: PLC0415 - after IN_HELPER
    shm = _shared_memory.SharedMemory(name=shm_name, track=False)
    mark = shm.buf.cast("q")
    conn = (_connection.PipeConnection(end) if os.name == "nt"
            else _connection.Connection(end))
    try:
        conn.send_bytes(_READY)
    except (EOFError, OSError):
        return
    while True:
        # Idle: nothing to limit, so the limit is only pushed out. The parent
        # going away ends the wait (EOF), and with it the helper.
        while not conn.poll(_IDLE_POLL_S):
            armed_at = _arm_self_limit()
        try:
            request = conn.recv_bytes()
        except (EOFError, OSError):
            return
        if time.monotonic() - armed_at > _REARM_AFTER_S:
            armed_at = _arm_self_limit()
        try:
            conn.send_bytes(engine._helper_handle(request, mark))
        except (EOFError, OSError):
            return


def _arm_self_limit() -> float:
    """(Re)start the helper's own time limit and return when it was armed.
    faulthandler's watchdog is a C thread that needs no GIL, so it fires
    inside a match that never returns, on every platform."""
    faulthandler.dump_traceback_later(_SELF_LIMIT_S, exit=True)
    return time.monotonic()
