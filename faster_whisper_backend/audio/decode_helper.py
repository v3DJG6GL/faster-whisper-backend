"""The helper processes client audio is opened and decoded in.

A media file is opened and decoded by libav, and what that costs in memory
is the file's choice: a container can list elements by the million, one
packet can hold thousands of frames that all decode before the first is
handed out, and libav bounds a single allocation, never their sum. Inside
the server process the only end to that is the kernel ending the server.
A helper process has a memory limit instead: the open or decode that passes
it fails there (or the helper is ended), the file is refused, and nothing
else on the server notices. The same holds for a file that crashes the
decoder outright.

What an operation is and how its answer is read is the business of the
modules that own them (audio/transcode.py, media/subtitle_mux.py); this
module is the transport, the pool and the limit.

Two roles, one file (the shape of pipeline/rule_helper.py and
media/link_helper.py, whose transport this follows):
  * parent -> ``start()`` / ``shutdown()`` (the app's lifespan),
              ``routed()`` and ``run()`` (the modules above).
  * helper -> ``helper_main()``, started as ``python -P -c`` by ``_spawn``
              with the repo root added at the END of its import path.

Helpers are reused: starting one (the interpreter, PyAV, numpy) takes a
quarter of a second, and an upload is opened and decoded several times on
its way through the stages. One idle helper is kept ahead of demand, so an
operation normally costs a round trip on a local socket; a helper that
answered out of memory, grew, timed out or died is replaced.

The two sides talk over a ``multiprocessing.connection`` pair made by the
parent, the helper's end inherited at its start. Requests and answers are
JSON and raw sample bytes, never pickle: a helper reads bytes a client
chose, and what it sends back is parsed as data.

Stdlib-only at import: the helper imports this module before anything else.
"""
from __future__ import annotations

import atexit
import builtins
import gc
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from multiprocessing import connection as _connection

from faster_whisper_backend import paths as _paths

logger = logging.getLogger("whisper-api")

# Whether start() starts anything. False only under the test suite
# (tests/conftest.py): thousands of tests post a stub upload and patch the
# decode functions inside the test process, which a helper process would not
# see. With no helper started every operation runs in-process, the same
# code; the helper path has its own tests, which turn this on.
ISOLATE = True

# True only in a helper process: the operations then run where they are.
IN_HELPER = False

# How a helper holds itself to its budget. None picks by platform: "rlimit"
# on Linux (the kernel refuses the allocation past the limit), "watchdog"
# elsewhere (a thread of the helper reads its own memory use every
# _WATCH_POLL_S and ends the process past the limit). The tests name one.
LIMIT_MODE: "str | None" = None

# Helpers that may exist at once, the busy and the idle. The slow operations
# (whole-file decodes and transcodes) take at most _LONG_SLOTS of them, so
# the quick ones (a header open, a link's language-check piece) never queue
# behind a run of long transcodes.
MAX_LIVE = 8
_LONG_SLOTS = MAX_LIVE - 2
# Idle helpers kept whatever the demand; more than these retire once they
# have been idle for _IDLE_RETIRE_S.
KEEP_IDLE = 2
_IDLE_RETIRE_S = 120.0
# Longest wait for a helper's start.
_START_TIMEOUT_S = 30.0
# While no helper has ever started (the platform does not start them), how
# long the operations run in-process before the next try.
_RETRY_AFTER_S = 60.0

# What one operation may add to its helper's memory. A real file needs
# little: a header open builds an index of some tens of MB for a day of
# audio, and a decode holds a few frames and one group for the resampler,
# whatever the file's length (the samples leave for the parent as they are
# made). libav's own cap on a single allocation (audio/ffmpeg.py, 256 MiB)
# stays below it. With less memory free than twice this the budget is half
# of what is free, never under _BUDGET_FLOOR.
BUDGET_BYTES = 1 << 30
_BUDGET_FLOOR = 256 << 20
# A helper that ends an operation this much above where it started is
# replaced: what the allocator kept would count against the next file.
_RECYCLE_BYTES = 128 << 20
# The exit code of a helper the watchdog ended for its memory.
_EXIT_MEMORY = 77
_WATCH_POLL_S = 0.005
# How often a helper looks whether the server that started it is still its
# parent (POSIX): one inside a decode would otherwise finish it for nobody.
_PARENT_POLL_S = 2.0

# The first byte of every message a helper sends after its first one:
# sample bytes on the way, then one answer. A lower-case answer says the
# helper ends itself after it.
_DATA = b"D"
_RESULT, _RESULT_LAST = b"R", b"r"
_ERROR, _ERROR_LAST = b"E", b"e"
_MEMORY = b"M"
_READY = b"ready"
# The largest request (paths and a few numbers), answer (a dict of scalars
# or an error's text) and piece of sample bytes either side reads.
_REQUEST_MAX = 1 << 20
_REPLY_MAX = 1 << 20
_DATA_MAX = 64 << 20
_TEXT_MAX = 2000

# Builtin exceptions a helper's error comes back as, by the first of these
# in its class's bases; anything else is a RuntimeError. Callers tell a
# plain ValueError (a refusal of the file) from everything else, so the
# base matters more than the exact class.
_PLAIN = ("ValueError", "IndexError", "KeyError", "LookupError", "TypeError",
          "ImportError", "ArithmeticError", "AttributeError", "EOFError",
          "NotImplementedError", "AssertionError", "OSError", "RuntimeError")

# Environment variables a helper is started without (see _helper_env): the
# server's settings, and the model hub's token.
_SERVER_ENV_PREFIXES = ("WHISPER_", "HF_", "HUGGING_FACE_")

_BOOT = ("import sys; sys.path.append(sys.argv[1]); "
         "from faster_whisper_backend.audio import decode_helper; "
         "decode_helper.helper_main(int(sys.argv[2]), sys.argv[3])")


class HelperOutOfMemory(MemoryError):
    """The operation passed its helper's memory budget (the helper is
    replaced). A MemoryError: what the same file would have raised
    in-process on a server out of memory."""


class HelperDied(RuntimeError):
    """The helper ended without an answer while it read the file: the
    decoder crashed, or something killed the process. `returncode` is its
    exit code (negative: the signal), None when it is not known."""

    # stats/metrics.classify_error: the media could not be decoded.
    error_class = "decode_failed"

    def __init__(self, returncode: "int | None", what: str = "") -> None:
        super().__init__(
            what or f"the decode helper ended without an answer (exit code {returncode})")
        self.returncode = returncode


class HelperTimeout(Exception):
    """The helper did not answer within the caller's deadline (it is killed)."""


class HelperFailed(RuntimeError):
    """No helper could be had: the pool is shut down, or a helper could not
    be started although helpers have run before. The operation is refused;
    it never runs in-process instead. str() is for the server log."""


class NotStarted(Exception):
    """No helper has ever started on this server (see _RETRY_AFTER_S): the
    caller runs the operation in-process, as it did before helpers."""


class _Helper:
    __slots__ = ("proc", "conn", "limit")

    def __init__(self, proc, conn) -> None:
        self.proc = proc
        self.conn = conn
        self.limit = "none"


# --------------------------------------------------------------------------
# Parent: starting and stopping a helper
# --------------------------------------------------------------------------

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


def _stop(proc) -> None:
    """Kill the helper and reap it. SIGKILL at once: a helper inside a
    decode acts on nothing gentler, and it holds nothing to flush."""
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
        logger.warning("[decode] decode helper %s did not exit after kill", proc.pid)


def _await_ready(helper: _Helper) -> None:
    """Wait for the helper's first message and take what it says about
    itself. RuntimeError when it does not come within _START_TIMEOUT_S, the
    helper ends first, or it could not load what an operation needs."""
    try:
        if not helper.conn.poll(_START_TIMEOUT_S):
            raise RuntimeError(f"helper did not answer within {_START_TIMEOUT_S:g} s")
        first = helper.conn.recv_bytes(_REPLY_MAX)
    except (EOFError, OSError) as exc:
        try:
            code = helper.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None
        raise RuntimeError(f"helper exited with code {code}") from exc
    if first[:len(_READY)] != _READY:
        raise RuntimeError("helper sent an unexpected first message")
    try:
        said = json.loads(first[len(_READY):])
        boot, limit = str(said.get("boot") or ""), str(said.get("limit") or "none")
    except (ValueError, AttributeError) as exc:
        raise RuntimeError("helper sent an unexpected first message") from exc
    if boot:
        raise RuntimeError(f"helper could not load the decoder: {boot[:_TEXT_MAX]}")
    helper.limit = limit if limit in ("rlimit", "watchdog") else "none"


def _helper_env() -> dict:
    """The server's environment without its own settings: a helper reads
    none of them (it imports neither the settings nor the app), and they
    hold the server's secrets — keys, tokens, passwords — which a process
    that parses a client's file has no use for. Two malloc arenas at most:
    each further one reserves 64 MiB of address space, which the "rlimit"
    budget counts."""
    env = {name: value for name, value in os.environ.items()
           if not name.upper().startswith(_SERVER_ENV_PREFIXES)}
    env["MALLOC_ARENA_MAX"] = "2"
    return env


def _spawn(register) -> _Helper:
    """Start one helper on its end of a connected pair and wait until it is
    ready. `register(proc)` is called as soon as the process exists, so a
    shutdown in the middle of the start still finds it."""
    conn, theirs = _connection.Pipe(duplex=True)
    try:
        try:
            end, inherit = _child_end(theirs)
            env = _helper_env()
            proc = subprocess.Popen(
                [sys.executable, "-P", "-c", _BOOT, _paths.REPO_ROOT, str(end),
                 LIMIT_MODE or "auto"],
                # Nothing of the helper's reaches the server's console: what
                # libav prints quotes the file, and the helper reports its
                # own failures in its answers.
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=env,
                start_new_session=os.name != "nt",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **inherit)
        finally:
            # The helper holds its own copy; with this one closed, a helper
            # that ends shows as EOF on `conn`.
            theirs.close()
        helper = _Helper(proc, conn)
        try:
            register(proc)
            _await_ready(helper)
        except BaseException:
            _stop(proc)
            raise
    except BaseException:
        conn.close()
        raise
    return helper


# --------------------------------------------------------------------------
# Parent: the pool
# --------------------------------------------------------------------------

class _Pool:
    """The helpers and the threads waiting for one. `_count` is the helpers
    that exist or are being started; the idle ones wait in `_idle` with the
    time they came back (last in, first out: the warmest serves next, the
    oldest retire first). `proven` says a helper has started at least once:
    it tells a platform that does not start helpers (operations then run
    in-process, logged) from a start that failed on a server where they ran
    (the operation is refused)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._idle: "list[tuple[_Helper, float]]" = []
        self._procs: set = set()
        self._count = 0
        self._starting_spare = False
        self._retire_pending = False
        self._closed = False
        self.proven = False
        self.long_slots = threading.BoundedSemaphore(_LONG_SLOTS)

    @property
    def closed(self) -> bool:
        return self._closed

    def _register(self, proc) -> None:
        with self._cond:
            if self._closed:
                raise RuntimeError("the decode helpers are shut down")
            self._procs.add(proc)

    def _start_counted(self) -> _Helper:
        """Start the helper whose place `_count` already holds. HelperFailed
        when it cannot be started or the pool was shut down meanwhile."""
        try:
            helper = _spawn(self._register)
        except Exception as exc:  # noqa: BLE001 - any start failure
            with self._cond:
                self._count -= 1
                self._procs = {p for p in self._procs if p.poll() is None}
                self._cond.notify()
            raise HelperFailed(
                f"could not start a decode helper: {type(exc).__name__}: {exc}") from None
        with self._cond:
            if not self._closed:
                first, self.proven = not self.proven, True
                if first:
                    _note_started(helper)
                return helper
            self._count -= 1
        self._close(helper)
        raise HelperFailed("the decode helpers are shut down")

    def _start_spare(self) -> None:
        """A background start of one idle helper. Never raises: without a
        spare the next operation starts its own helper."""
        helper = None
        try:
            helper = self._start_counted()
        except HelperFailed as exc:
            if not self._closed:
                _note_start_failed(self, exc)
        with self._cond:
            self._starting_spare = False
        if helper is not None:
            self.release(helper)

    def spare(self) -> None:
        """Start an idle helper in the background unless one is there, one
        is on its way or MAX_LIVE exist."""
        with self._cond:
            if (self._closed or self._idle or self._starting_spare
                    or self._count >= MAX_LIVE):
                return
            self._starting_spare = True
            self._count += 1
        try:
            threading.Thread(target=self._start_spare, name="decode-helper-start",
                             daemon=True).start()
        except Exception:  # noqa: BLE001 - "can't start new thread"
            with self._cond:
                self._starting_spare = False
                self._count -= 1
                self._cond.notify()

    def acquire(self) -> _Helper:
        """A helper for one operation: an idle one, else a new one, else —
        with MAX_LIVE busy — the first that comes back. HelperFailed when
        the pool is shut down or a helper cannot be started."""
        dead: "list[_Helper]" = []
        helper = None
        try:
            with self._cond:
                while helper is None:
                    if self._closed:
                        raise HelperFailed("the decode helpers are shut down")
                    while self._idle and helper is None:
                        candidate, _since = self._idle.pop()
                        if candidate.proc.poll() is None:
                            helper = candidate
                        else:
                            self._count -= 1
                            dead.append(candidate)
                    if helper is not None:
                        break
                    if self._count < MAX_LIVE:
                        self._count += 1
                        break
                    self._cond.wait()
        finally:
            for gone in dead:
                self._close(gone)
        if helper is None:
            helper = self._start_counted()
        # The next operation's helper, ahead of it.
        self.spare()
        return helper

    def release(self, helper: _Helper) -> None:
        """Hand a helper back after an operation it answered and may repeat."""
        retire = False
        with self._cond:
            closed = self._closed
            if closed:
                self._count -= 1
            else:
                self._idle.append((helper, time.monotonic()))
                self._cond.notify()
                retire = len(self._idle) > KEEP_IDLE and not self._retire_pending
                self._retire_pending = self._retire_pending or retire
        if closed:
            self._close(helper)
        elif retire:
            self._retire_later(_IDLE_RETIRE_S)

    def _retire_later(self, after: float) -> None:
        try:
            timer = threading.Timer(after + 0.5, self._retire)
            timer.daemon = True
            timer.start()
        except Exception:  # noqa: BLE001 - "can't start new thread"
            with self._cond:
                self._retire_pending = False

    def _retire(self) -> None:
        """End the helpers that have been idle past _IDLE_RETIRE_S, down to
        KEEP_IDLE."""
        gone: "list[_Helper]" = []
        again = None
        with self._cond:
            self._retire_pending = False
            now = time.monotonic()
            while len(self._idle) > KEEP_IDLE:
                helper, since = self._idle[0]
                if now - since < _IDLE_RETIRE_S:
                    again = _IDLE_RETIRE_S - (now - since)
                    self._retire_pending = not self._closed
                    break
                del self._idle[0]
                self._count -= 1
                gone.append(helper)
        for helper in gone:
            self._close(helper)
        if again is not None and self._retire_pending:
            self._retire_later(again)

    def discard(self, helper: _Helper) -> None:
        """Kill a helper that must not serve again (out of memory, grown,
        timed out, dead) and make room for its replacement."""
        self._close(helper)
        with self._cond:
            self._count -= 1
            self._cond.notify()
        self.spare()

    def _close(self, helper: _Helper) -> None:
        _stop(helper.proc)
        try:
            helper.conn.close()
        except OSError:
            pass
        with self._cond:
            self._procs.discard(helper.proc)

    def shutdown(self) -> None:
        """Kill every helper, the busy and the starting ones included: an
        operation in flight sees its helper gone and fails."""
        with self._cond:
            self._closed = True
            idle, self._idle = self._idle, []
            self._count -= len(idle)
            procs = list(self._procs)
            self._cond.notify_all()
        for helper, _since in idle:
            self._close(helper)
        for proc in procs:
            _stop(proc)

    def pids(self) -> "list[int]":
        """The live helpers' process ids (tests, diagnostics)."""
        with self._cond:
            return sorted(p.pid for p in self._procs if p.poll() is None)

    def idle_count(self) -> int:
        with self._cond:
            return len(self._idle)

    def wait_idle(self, n: int, timeout: float) -> bool:
        """Whether `n` helpers are idle within `timeout` seconds (tests)."""
        end = time.monotonic() + timeout
        with self._cond:
            while len(self._idle) < n:
                left = end - time.monotonic()
                if left <= 0 or self._closed:
                    return False
                self._cond.wait(min(left, 0.05))
            return True


# The running pool, or None: every operation then runs in-process. After
# shutdown() the closed pool stays, and run() refuses.
_POOL: "_Pool | None" = None
_POOL_LOCK = threading.Lock()
_atexit_registered = False
# While no helper has ever started: the time until which operations run
# in-process (see NotStarted), and whether the log has said so.
_down_until = 0.0
_down_logged = False
_notes_lock = threading.Lock()


def _note_started(helper: _Helper) -> None:
    """The first helper of a pool is ready: say what holds it to its budget,
    once, and that helpers are back if the log said they were not."""
    global _down_until, _down_logged
    with _notes_lock:
        was_down, _down_logged, _down_until = _down_logged, False, 0.0
    if was_down:
        logger.info("[decode] decode helper processes are back")
    if helper.limit == "none":
        logger.warning(
            "[decode] the decode helper processes run without a memory limit "
            "(none could be set on this platform): a crafted file can still "
            "make a helper take gigabytes before it is refused.")


def _note_start_failed(pool: _Pool, exc: BaseException) -> None:
    """A helper start failed. While none has ever started the operations run
    in-process until the next try, and the log says so once."""
    global _down_until, _down_logged
    if pool.proven:
        return
    with _notes_lock:
        _down_until = time.monotonic() + _RETRY_AFTER_S
        logged, _down_logged = _down_logged, True
    if not logged:
        logger.warning(
            "[decode] %s — client audio is opened and decoded inside the "
            "server process, without a memory limit; retrying every %g s",
            exc, _RETRY_AFTER_S)


def routed() -> bool:
    """Whether an operation goes to a helper: start() has run, this is not a
    helper itself, and helpers are not known to be unstartable right now."""
    return (_POOL is not None and not IN_HELPER
            and (not _down_until or time.monotonic() >= _down_until))


def start(*, spare: bool = True) -> None:
    """Route the operations through helper processes from now on
    (idempotent). `spare`: start the first idle helper right away, so the
    first upload does not wait for one. Never raises."""
    global _POOL, _atexit_registered
    if not ISOLATE:
        return
    with _POOL_LOCK:
        if _POOL is None or _POOL.closed:
            _POOL = _Pool()
            if not _atexit_registered:
                # For an exit that skips the app's shutdown: an idle helper
                # would end on its own (its connection closes), this reaps
                # it first.
                atexit.register(shutdown)
                _atexit_registered = True
        pool = _POOL
    if spare:
        pool.spare()


def shutdown() -> None:
    """Kill the helpers. Operations are refused afterwards (never run
    in-process instead) until the next start()."""
    with _POOL_LOCK:
        pool = _POOL
    if pool is not None:
        pool.shutdown()


def reset() -> None:
    """Back to the never-started state (tests): operations run in-process."""
    global _POOL, _down_until, _down_logged
    shutdown()
    with _POOL_LOCK:
        _POOL = None
    with _notes_lock:
        _down_until, _down_logged = 0.0, False


def _budget() -> int:
    """The memory one operation may add to its helper right now."""
    try:
        from faster_whisper_backend.runtime import system_stats  # noqa: PLC0415 - parent only
        free = int(system_stats.available_ram_bytes())
    except Exception:  # noqa: BLE001 - unreadable: the full budget
        return BUDGET_BYTES
    return max(_BUDGET_FLOOR, min(BUDGET_BYTES, free // 2))


# --------------------------------------------------------------------------
# Parent: one operation
# --------------------------------------------------------------------------

def _av_error(name: str, args, text: str) -> BaseException:
    """PyAV's own error class `name` again, with the errno, message and file
    name the helper's had: callers tell av errors apart by class."""
    try:
        import av  # noqa: PLC0415 - optional dependency
        cls = getattr(av.error, name, None)
        if not (isinstance(cls, type) and issubclass(cls, av.error.FFmpegError)):
            cls = av.error.FFmpegError
        code, strerror, filename = (list(args) + [None, None, None])[:3]
        return cls(int(code) if isinstance(code, int) else 0,
                   str(strerror)[:_TEXT_MAX] if strerror is not None else text,
                   str(filename)[:_TEXT_MAX] if filename is not None else None)
    except Exception:  # noqa: BLE001 - no PyAV here, or a class with other arguments
        return RuntimeError(f"{name}: {text}")


def _rebuild(said, classes: dict, depth: int = 0) -> BaseException:
    """The exception a helper described, as one of: PyAV's class of that
    name, the caller's own class of that name (`classes`), the builtin the
    helper's class derives from (_PLAIN), a RuntimeError. Built from the
    description's scalars only."""
    if not isinstance(said, dict):
        return RuntimeError("the decode helper sent an unreadable error")
    name = str(said.get("name"))[:100]
    text = str(said.get("text") or "")[:_TEXT_MAX]
    base = said.get("base")
    if isinstance(said.get("av"), list):
        exc = _av_error(name, said["av"], text)
    elif said.get("local") and name in classes:
        exc = classes[name](text)
    elif base == "OSError":
        code = said.get("errno")
        exc = OSError(code, text) if isinstance(code, int) else OSError(text)
    elif isinstance(base, str) and base in _PLAIN:
        exc = getattr(builtins, base)(text)
    else:
        exc = RuntimeError(f"{name}: {text}")
    cause = said.get("cause")
    if cause is not None and depth < 2:
        exc.__cause__ = _rebuild(cause, classes, depth + 1)
    return exc


class _CallerStopped(Exception):
    """Carries an error of the caller's `on_data`, or this side's refusal of
    a message, out of _exchange's own handling of connection errors."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error


def _exchange(pool: _Pool, helper: _Helper, request: bytes,
              timeout: "float | None", on_data) -> "tuple[bytes, bytes]":
    """Send one request and read the helper's messages up to its answer:
    (tag, body). Sample bytes on the way go to `on_data`, whose own error
    passes through (the caller discards the helper). HelperTimeout past
    `timeout`: the helper is killed at the deadline by a timer, so the
    limit holds for a message that started to arrive and never ended as it
    does for one that never came. HelperOutOfMemory / HelperDied /
    HelperFailed when the helper ended (see _ended)."""
    late = threading.Event()
    timer = None
    if timeout is not None:
        def _kill() -> None:
            late.set()
            _stop(helper.proc)
        timer = threading.Timer(max(0.0, timeout), _kill)
        timer.daemon = True
    try:
        if timer is not None:
            timer.start()
        helper.conn.send_bytes(request)
        while True:
            # Longer than this is an OSError: nothing of it is read.
            message = helper.conn.recv_bytes(_DATA_MAX + 1)
            tag = message[:1]
            if tag != _DATA:
                if len(message) > _REPLY_MAX:
                    raise _CallerStopped(HelperDied(
                        None, "the decode helper sent an answer too long to read"))
                return tag, message[1:]
            if on_data is None:
                raise _CallerStopped(HelperDied(
                    None, "the decode helper sent samples nobody asked for"))
            try:
                on_data(memoryview(message)[1:])
            except BaseException as exc:  # noqa: BLE001 - the caller's own refusal
                raise _CallerStopped(exc) from None
    except _CallerStopped as stopped:
        raise stopped.error from None
    except (EOFError, OSError) as exc:
        if late.is_set():
            raise HelperTimeout from None
        raise _ended(pool, helper, exc) from None
    finally:
        if timer is not None:
            timer.cancel()


def _ended(pool: _Pool, helper: _Helper, exc: BaseException) -> BaseException:
    """What a connection that broke inside a request means, by how the
    helper ended."""
    if pool.closed:
        return HelperFailed("the decode helpers are shut down")
    # A helper that ended is reaped within milliseconds; one that is still
    # there sent what the transport does not take.
    try:
        code = helper.proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        code = None
    if code == _EXIT_MEMORY:
        return HelperOutOfMemory("the file needs more memory than a decode gets")
    if code is None:
        return HelperDied(None, f"the decode helper broke off: {type(exc).__name__}")
    if os.name != "nt" and code == -signal.SIGKILL:
        # Nothing in a decoder kills its own process: the kernel did, for
        # memory the whole machine ran out of (helpers go first, see
        # _first_to_go), or an operator. The server's fault as far as this
        # file goes, not a verdict on it.
        return HelperFailed("a decode helper was killed while it read a file "
                            "(out of memory on this machine?)")
    return HelperDied(code)


def _answer(body: bytes):
    """A helper's answer as data. HelperDied for what is no JSON, or JSON
    with a NaN / Infinity in it (no operation answers one)."""
    def _no_constant(name):
        raise ValueError(name)
    try:
        return json.loads(body, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        raise HelperDied(None, "the decode helper sent an unreadable answer") from None


def run(op: str, args: dict, *, timeout: "float | None" = None, long: bool = False,
        on_data=None, classes: "dict | None" = None):
    """Run operation `op` with `args` (JSON scalars) in a helper under its
    memory budget and return its answer (JSON scalars, lists, dicts — data
    the caller still checks). Sample bytes arrive through `on_data(view)`
    as they are decoded. `long`: a whole-file decode (see _LONG_SLOTS).

    Raises what the operation raised in the helper, rebuilt (see _rebuild;
    `classes` are the caller's own exception classes by name), and:
    HelperOutOfMemory past the budget, HelperDied when the helper ended
    without an answer, HelperTimeout past `timeout` seconds, HelperFailed
    when no helper could be had or the one that served was killed from
    outside, NotStarted when no helper has ever started here (the caller
    then runs the operation in-process)."""
    pool = _POOL
    if pool is None:
        raise NotStarted
    request = json.dumps({"op": op, "args": args, "budget": _budget()}).encode()
    if long:
        pool.long_slots.acquire()
    try:
        try:
            helper = pool.acquire()
        except HelperFailed as exc:
            if pool.proven or pool.closed:
                raise
            _note_start_failed(pool, exc)
            raise NotStarted from None
        keep = False
        try:
            tag, body = _exchange(pool, helper, request, timeout, on_data)
            said = None if tag == _MEMORY else _answer(body)
            # Only a helper whose answer could be read serves again.
            keep = tag in (_RESULT, _ERROR)
        finally:
            if keep:
                pool.release(helper)
            else:
                pool.discard(helper)
    finally:
        if long:
            pool.long_slots.release()
    if tag == _MEMORY:
        raise HelperOutOfMemory("the file needs more memory than a decode gets")
    if tag in (_RESULT, _RESULT_LAST):
        return said
    if tag in (_ERROR, _ERROR_LAST):
        raise _rebuild(said, classes or {})
    raise HelperDied(None, "the decode helper sent an unreadable answer")


# --------------------------------------------------------------------------
# Helper process
# --------------------------------------------------------------------------

class _Limit:
    """What holds a helper to an operation's budget. `mode` is "rlimit" (the
    kernel fails the allocation that would pass the limit on the process's
    address space), "watchdog" (a thread ends the process once its memory
    in use has passed the limit) or "none"."""

    def __init__(self, wanted: str) -> None:
        self.mode = "none"
        self._ceiling: "int | None" = None
        self._armed = threading.Event()
        if wanted in ("auto", "rlimit") and sys.platform.startswith("linux"):
            try:
                import resource  # noqa: PLC0415 - POSIX only
                self._resource = resource
                self._hard = resource.getrlimit(resource.RLIMIT_AS)[1]
                self._page = os.sysconf("SC_PAGE_SIZE")
                self.mode = "rlimit"
                self._used()
            except Exception:  # noqa: BLE001 - no such limit here
                self.mode = "none"
        if self.mode == "none" and wanted in ("auto", "watchdog"):
            try:
                import psutil  # noqa: PLC0415 - the server's own requirement
                self._process = psutil.Process()
                self.mode = "watchdog"
                self._used()
                threading.Thread(target=self._watch, name="memory-watch",
                                 daemon=True).start()
            except Exception:  # noqa: BLE001 - no reading, no limit
                self.mode = "none"
        self._base = self._used(resident=True) if self.mode != "none" else 0

    def _used(self, resident: bool = False) -> int:
        """Bytes this process holds, as the mode counts them: its address
        space ("rlimit"; with `resident` the part of it in memory), or the
        larger of what is resident and what is committed to it alone
        ("watchdog")."""
        if self.mode == "rlimit":
            with open("/proc/self/statm", "rb") as f:
                return int(f.read().split()[1 if resident else 0]) * self._page
        info = self._process.memory_info()
        return max(int(info.rss), int(getattr(info, "private", 0) or 0))

    def arm(self, budget: int) -> None:
        """Allow `budget` bytes on top of what the helper holds now."""
        if self.mode == "none":
            return
        ceiling = self._used() + budget
        if self.mode == "rlimit":
            unlimited = self._resource.RLIM_INFINITY
            soft = ceiling if self._hard == unlimited else min(ceiling, self._hard)
            self._resource.setrlimit(self._resource.RLIMIT_AS, (soft, self._hard))
        else:
            self._ceiling = ceiling
            self._armed.set()

    def lift(self) -> None:
        if self.mode == "rlimit":
            self._resource.setrlimit(self._resource.RLIMIT_AS, (self._hard, self._hard))
        elif self.mode == "watchdog":
            self._armed.clear()
            self._ceiling = None

    def grown(self) -> bool:
        """Whether the helper holds _RECYCLE_BYTES more in memory than at
        its start. Memory, not address space: thread stacks and arenas a
        library keeps reserved cost the next file nothing (its budget is
        counted from where the helper stands), and under a raised stack
        limit they alone would replace a helper after every request."""
        if self.mode == "none":
            return False
        try:
            return self._used(resident=True) - self._base > _RECYCLE_BYTES
        except Exception:  # noqa: BLE001 - unreadable: start a fresh one
            return True

    def _watch(self) -> None:
        # libav runs with the GIL released, so this thread gets its turns
        # inside an open or a decode call.
        while True:
            self._armed.wait()
            ceiling = self._ceiling
            try:
                if ceiling is not None and self._used() > ceiling:
                    os._exit(_EXIT_MEMORY)
            except Exception:  # noqa: BLE001 - a reading that failed once
                pass
            time.sleep(_WATCH_POLL_S)


def _first_to_go() -> None:
    """Linux: when the whole machine runs out of memory the kernel picks a
    process to end, and this says to pick a helper before the server."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def _end_with_parent() -> None:
    """POSIX: end this helper once the server that started it is gone. An
    idle helper sees its connection close; one inside a decode would not
    look until the decode is done."""
    if os.name == "nt":
        return
    parent = os.getppid()

    def _poll() -> None:
        while os.getppid() == parent:
            time.sleep(_PARENT_POLL_S)
        os._exit(0)

    threading.Thread(target=_poll, name="parent-watch", daemon=True).start()


def _describe(exc: BaseException, depth: int = 0) -> dict:
    """An exception as the scalars _rebuild takes."""
    cls = type(exc)
    module = (getattr(cls, "__module__", "") or "")
    said: dict = {"name": cls.__name__, "text": str(exc)[:_TEXT_MAX]}
    # PyAV's FFmpegError family only: its few other classes (a plain
    # ValueError for an unknown codec) go by their builtin base below.
    if (module.split(".")[0] == "av"
            and any(c.__name__ == "FFmpegError" for c in cls.__mro__)):
        said["av"] = [getattr(exc, "errno", None), getattr(exc, "strerror", None),
                      getattr(exc, "filename", None)]
        said["av"] = [v if isinstance(v, (int, str)) else None for v in said["av"]]
    else:
        said["local"] = module.startswith("faster_whisper_backend")
        said["base"] = next(
            (c.__name__ for c in cls.__mro__
             if c.__module__ == "builtins" and c.__name__ in _PLAIN), "RuntimeError")
        if isinstance(exc, OSError):
            said["errno"] = exc.errno if isinstance(exc.errno, int) else None
            said["text"] = str(exc.strerror or exc)[:_TEXT_MAX]
    if exc.__cause__ is not None and depth < 2:
        said["cause"] = _describe(exc.__cause__, depth + 1)
    return said


def _load_ops() -> dict:
    """The operations by name, from the modules that own them. Loads PyAV
    and numpy and fills the decoder lists, so the first request does not."""
    from faster_whisper_backend.audio import transcode  # noqa: PLC0415 - the helper's own load
    from faster_whisper_backend.media import subtitle_mux  # noqa: PLC0415
    transcode.helper_warm()
    return {**transcode.HELPER_OPS, **subtitle_mux.HELPER_OPS}


def helper_main(end: int, limit_mode: str) -> None:
    """The helper process: take the inherited end of the parent's pair, load
    the decoder, then answer the parent's requests one at a time, each under
    its memory budget, until the parent goes away or an answer says the
    helper must not serve again."""
    global IN_HELPER
    IN_HELPER = True
    # A Ctrl-C in the server's terminal reaches the whole console on
    # Windows; the parent decides when a helper stops.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    conn = (_connection.PipeConnection(end) if os.name == "nt"
            else _connection.Connection(end))
    _first_to_go()
    ops: dict = {}
    boot = ""
    try:
        ops = _load_ops()
    except BaseException as exc:  # noqa: BLE001 - told to the parent
        boot = f"{type(exc).__name__}: {exc}"[:_TEXT_MAX]
    limit = _Limit(limit_mode)
    try:
        conn.send_bytes(_READY + json.dumps({"limit": limit.mode, "boot": boot}).encode())
    except (EOFError, OSError):
        os._exit(0)
    if boot:
        os._exit(0)
    _end_with_parent()

    def _send(payload) -> None:
        conn.send_bytes(_DATA + bytes(payload))

    while True:
        try:
            # An idle helper waits here. The parent going away ends the
            # wait (EOF), and with it the helper.
            request = json.loads(conn.recv_bytes(_REQUEST_MAX))
            handler = ops[request["op"]]
            args, budget = request["args"], int(request["budget"])
        except (EOFError, OSError):
            os._exit(0)
        except Exception:  # noqa: BLE001 - not a request this helper knows
            os._exit(1)
        tag, body = _RESULT, None
        try:
            limit.arm(budget)
            try:
                body = handler(_send, **args)
            finally:
                # Before the answer is put together: that takes memory too.
                limit.lift()
        except MemoryError:
            tag = _MEMORY
        except BaseException as exc:  # noqa: BLE001 - told to the parent
            tag, body = _ERROR, _describe(exc)
            del exc
        last = tag == _MEMORY
        if not last and limit.grown():
            # Only what a collection does not give back counts as growth
            # (and a collection per request would cost more than a header
            # open does).
            gc.collect()
            last = limit.grown()
        try:
            if tag == _MEMORY:
                conn.send_bytes(_MEMORY)
            else:
                if last:
                    tag = _RESULT_LAST if tag == _RESULT else _ERROR_LAST
                conn.send_bytes(tag + json.dumps(body).encode())
        except (EOFError, OSError, ValueError, TypeError):
            os._exit(0)
        if last:
            # Not a normal exit: nothing of the file's must hold it open.
            os._exit(0)
