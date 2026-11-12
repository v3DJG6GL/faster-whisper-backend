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
              ``acquire`` / ``call`` / ``release`` / ``discard`` (the engine).
  * helper -> ``helper_main()``, started as ``python -c`` by ``_spawn``. It
              imports the engine (and with it the settings modules, never the
              model stack or the app), takes its rules from the parent over the
              connection and never reads them from the config files itself.

The helper is a plain child process (``subprocess``), not a
``multiprocessing.Process``: that one re-imports the parent's ``__main__`` in
the child, which under ``python main.py`` is the whole app. The two sides talk
over a ``multiprocessing.connection`` (a Unix socket, a named pipe on Windows),
whose ``poll(timeout)`` is the same timed wait on every platform.

Stdlib-only at import: the engine imports this module, and the helper imports
it before anything else.
"""
from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import pickle
import secrets
import signal
import subprocess
import sys
import tempfile
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
# Longest wait for a helper's start: the interpreter, the engine import and
# the connection handshake take about a quarter of a second on an idle box.
_START_TIMEOUT_S = 30.0
# Longest wait for a free helper. Walks take milliseconds, so only helpers
# held by slow rules (each for at most its deadline) make a walk wait at all.
_ACQUIRE_TIMEOUT_S = 30.0
# After every helper failed to start, how long walks run in-process before
# the pool tries again.
_RETRY_AFTER_S = 60.0
# How long a helper may stay inside one request before it ends itself. The
# parent enforces the real deadline and kills the helper; this only covers a
# parent that died without doing so, where a helper inside a slow match would
# otherwise hold a CPU core for good. Above the longest request the engine
# makes (its rule-set load timeout).
_SELF_LIMIT_S = 180.0
# An idle helper wakes this often to push its own limit out.
_IDLE_POLL_S = 5.0
# The helper re-arms its limit at a request only when the last arming is
# older than this (arming costs more than a short walk).
_REARM_AFTER_S = 1.0

_BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); "
         "from faster_whisper_backend.pipeline import rule_helper; "
         "rule_helper.helper_main()")


class HelperTimeout(Exception):
    """The helper did not answer within the caller's deadline."""


class HelperGone(Exception):
    """The helper died or its connection broke."""


class _Helper:
    """One helper process as the parent holds it. `mark` is one shared
    signed 64-bit slot the helper writes its progress into (the engine: the
    index of the rule about to run); `token` names the rule set the helper
    has loaded (the engine's)."""

    __slots__ = ("proc", "conn", "shm", "mark", "token")

    def __init__(self, proc, conn, shm) -> None:
        self.proc = proc
        self.conn = conn
        self.shm = shm
        self.mark = shm.buf.cast("q")
        self.token = None


def call(helper: _Helper, payload: bytes, timeout: float) -> bytes:
    """Send one request and wait up to `timeout` seconds for its reply.
    HelperTimeout past that, HelperGone when the helper died; either way the
    caller discards the helper."""
    try:
        helper.conn.send_bytes(payload)
        if not helper.conn.poll(timeout):
            raise HelperTimeout
        return helper.conn.recv_bytes()
    except (EOFError, OSError) as exc:
        raise HelperGone(str(exc) or type(exc).__name__) from exc


def _new_address() -> str:
    """A fresh address for one helper's listener: a named pipe on Windows, an
    abstract socket on Linux (no file, so a read-only filesystem is fine), a
    socket file in the temp dir elsewhere."""
    name = f"fwb-rules-{os.getpid()}-{secrets.token_hex(8)}"
    if os.name == "nt":
        return rf"\\.\pipe\{name}"
    if sys.platform.startswith("linux"):
        return "\0" + name
    return os.path.join(tempfile.gettempdir(), name + ".sock")


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


def _connect(proc, address: str, key: bytes):
    """The parent's end of the connection to a helper that is still starting:
    the helper listens once its imports are done, so the connect is retried
    until then."""
    end = time.monotonic() + _START_TIMEOUT_S
    while True:
        try:
            return _connection.Client(address, authkey=key)
        except (OSError, EOFError, _connection.AuthenticationError) as exc:
            if proc.poll() is not None:
                raise RuntimeError(f"helper exited with code {proc.returncode}") from exc
            if time.monotonic() > end:
                raise RuntimeError(
                    f"helper did not answer within {_START_TIMEOUT_S:g} s") from exc
            time.sleep(0.005)


def _spawn(register) -> _Helper:
    """Start one helper and connect to it. `register(proc)` is called as soon
    as the process exists, so a shutdown in the middle of the start still
    finds it."""
    shm = _shared_memory.SharedMemory(create=True, size=8, track=False)
    try:
        address = _new_address()
        key = secrets.token_bytes(32)
        proc = subprocess.Popen(
            [sys.executable, "-c", _BOOT, _paths.REPO_ROOT],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            register(proc)
            # Over stdin, not argv: the key must not show in the process
            # list, and an abstract socket name starts with a NUL.
            proc.stdin.write(pickle.dumps((address, key, shm.name)))
            proc.stdin.close()
            conn = _connect(proc, address, key)
        except BaseException:
            _stop(proc)
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
    `_idle` (last in, first out: the warmest helper serves next)."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._cond = threading.Condition()
        self._idle: list[_Helper] = []
        self._procs: set = set()
        self._starters: list[threading.Thread] = []
        self._alive = 0
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
        self._starters = [t for t in self._starters if t.is_alive()]
        for _ in range(n):
            self._alive += 1
            t = threading.Thread(target=self._start_one, name="rule-helper-start",
                                 daemon=True)
            self._starters.append(t)
            t.start()

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
                if self._alive == 0 and not self._closed:
                    self._retry_at = time.monotonic() + _RETRY_AFTER_S
                    if not self._down_logged:
                        self._down_logged = True
                        logger.warning(
                            "[pipeline] rule helper process could not be started (%s) — "
                            "rules run inside the server process without a time limit; "
                            "retrying every %g s", exc, _RETRY_AFTER_S)
                self._cond.notify_all()
            return
        with self._cond:
            if not self._closed:
                if self._down_logged:
                    self._down_logged = False
                    logger.info("[pipeline] rule helper processes are back")
                self._idle.append(helper)
                self._cond.notify()
                return
        self._close(helper)

    def acquire(self) -> "_Helper | None":
        """A free helper, waiting for one if all are busy or still starting.
        None when the walk has to run in-process: the pool is shut down, no
        helper could be started, or none came free in _ACQUIRE_TIMEOUT_S."""
        end = time.monotonic() + _ACQUIRE_TIMEOUT_S
        with self._cond:
            while True:
                if self._closed:
                    return None
                if self._idle:
                    return self._idle.pop()
                if self._alive == 0:
                    if time.monotonic() >= self._retry_at:
                        self._start_locked(self._size)
                    return None
                left = end - time.monotonic()
                if left <= 0:
                    now = time.monotonic()
                    if now - self._busy_logged_at > _RETRY_AFTER_S:
                        self._busy_logged_at = now
                        logger.warning(
                            "[pipeline] no rule helper process came free in %g s — "
                            "this walk runs inside the server process without a "
                            "time limit", _ACQUIRE_TIMEOUT_S)
                    return None
                self._cond.wait(left)

    def release(self, helper: _Helper) -> None:
        """Hand a helper back after a request it answered."""
        with self._cond:
            if not self._closed:
                self._idle.append(helper)
                self._cond.notify()
                return
        self._close(helper)

    def discard(self, helper: _Helper) -> None:
        """Kill a helper that timed out or died, and start its replacement."""
        self._close(helper)
        with self._cond:
            self._alive -= 1
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
        in flight sees its helper gone and finishes in-process."""
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
    up in the background, and a walk that arrives first waits for one."""
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
    """The helper process: connect back to the parent, then answer its
    requests one at a time with the engine's handler until the parent goes
    away."""
    global IN_HELPER
    IN_HELPER = True
    address, key, shm_name = pickle.loads(sys.stdin.buffer.read())
    # A Ctrl-C in the server's terminal reaches the whole foreground group;
    # the parent decides when a helper stops.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    armed_at = _arm_self_limit()
    from faster_whisper_backend.pipeline import engine  # noqa: PLC0415 - after IN_HELPER
    shm = _shared_memory.SharedMemory(name=shm_name, track=False)
    mark = shm.buf.cast("q")
    with _connection.Listener(address, authkey=key) as listener:
        conn = listener.accept()
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
