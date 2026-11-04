"""An asyncio lock that follows the running event loop.

``asyncio.Lock`` binds to the first loop that acquires it and raises
``RuntimeError: ... is bound to a different event loop`` from any other loop.
The stage caches (diarization, BGM separation, translation), the pipeline
rules and the profile overrides each guard their state with a module-level
lock, and the process normally has one loop for its whole life, so that never
mattered in production. Under the test suite every ``TestClient`` lifespan is
its own loop: a lock a previous loop left held (worker cancelled while
loading, loop torn down before the release ran) wedges the next lifespan's
shutdown. This lock keeps one ``asyncio.Lock`` per event loop, so a stale
lock from a dead loop can never be waited on, and
a lock one live loop holds is never swapped out from under it by another
(``release()`` always finds the object its own loop acquired). Closed loops
are pruned. It does NOT exclude two live loops from each other — nothing
built on ``asyncio.Lock`` can; with one loop it is exactly an
``asyncio.Lock``.

``CancelLatch`` and ``release_count`` are the model caches' other shared
primitives: the latch finishes an un-stoppable load / registry write under
such a lock and defers the caller's cancellation until it has landed;
``register_through_latch`` is the unwind around such a registry write;
``release_count`` drops a lease / in-flight count. ``cancel_and_wait`` is the
teardown of a background task (lifespan loops, the preload worker, a stream's
consumer); ``run_every`` is the body of those lifespan loops.
"""

from __future__ import annotations

import asyncio
import threading


class LoopLock:
    __slots__ = ("_locks", "_mutex")

    def __init__(self) -> None:
        self._locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
        self._mutex = threading.Lock()   # two loops = two threads

    def _current(self, *, create: bool = True) -> asyncio.Lock | None:
        loop = asyncio.get_running_loop()
        with self._mutex:
            lock = self._locks.get(loop)
            if lock is None and create:
                for dead in [lp for lp in self._locks if lp.is_closed()]:
                    del self._locks[dead]
                lock = self._locks[loop] = asyncio.Lock()
            return lock

    async def acquire(self) -> bool:
        return await self._current().acquire()

    def release(self) -> None:
        lock = self._current(create=False)
        if lock is None:
            raise RuntimeError("Lock is not acquired.")
        lock.release()

    def locked(self) -> bool:
        """This loop's lock; with no running loop, whether ANY loop's is
        held. Never creates or replaces a lock."""
        try:
            lock = self._current(create=False)
        except RuntimeError:          # no running loop
            with self._mutex:
                return any(lk.locked() for lk in self._locks.values())
        return bool(lock is not None and lock.locked())

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc) -> None:
        self.release()


class CancelLatch:
    """Await executor / to_thread futures to their end through cancellations.

    ``wait`` shields ``fut`` and keeps awaiting it when the caller is
    cancelled, remembering the FIRST such ``CancelledError`` in
    ``cancelled`` for the caller to re-raise once its cleanup is done; it
    re-raises at once when ``fut`` itself was cancelled, and ``fut``'s own
    exception propagates unchanged. One latch shared across several awaits
    keeps one first cancellation for all of them.
    """

    __slots__ = ("cancelled",)

    def __init__(self) -> None:
        self.cancelled: asyncio.CancelledError | None = None

    async def wait(self, fut):
        while True:
            try:
                return await asyncio.shield(fut)
            except asyncio.CancelledError as c:
                if fut.cancelled():
                    raise
                self.cancelled = self.cancelled or c


async def register_through_latch(latch: CancelLatch, register, *args,
                                 release, forget, **kwargs) -> None:
    """Run the blocking ``register(*args, **kwargs)`` on a thread, shielded and
    awaited to its end through cancellations, then unwind for the caller who
    will not receive the model: ``release`` undoes its lease on every failure
    or cancellation, then ``forget`` clears the cache entry after a failed
    register (a cancel keeps it). The latch is a parameter because a load
    shares it with its constructor, keeping one first cancellation for both.
    """
    fut = asyncio.ensure_future(asyncio.to_thread(register, *args, **kwargs))
    try:
        await latch.wait(fut)
    except BaseException as e:
        release()
        if not isinstance(e, asyncio.CancelledError):
            forget()
            if latch.cancelled is not None:
                raise latch.cancelled from e
        raise
    if latch.cancelled is not None:
        release()
        raise latch.cancelled


async def cancel_and_wait(task: asyncio.Task | None) -> None:
    """Cancel ``task`` (None is a no-op) and wait for it to unwind,
    swallowing its ``CancelledError`` or any exception it ends with."""
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def run_every(interval, tick, log, error_fmt: str) -> None:
    """Lifespan loop: sleep ``interval()`` seconds (re-read every round), await
    ``tick()``, repeat. A failed tick is logged via ``log`` with ``error_fmt``
    and the next still runs; cancellation is the only exit."""
    while True:
        await asyncio.sleep(interval())
        try:
            await tick()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the loop must survive
            log.error(error_fmt, e)


def release_count(counts: dict, key) -> int:
    """Decrement ``counts[key]`` (missing = 0) and return the new value;
    the key is dropped once it reaches 0 or below (a lease / in-flight
    count release). The caller holds whatever lock guards ``counts``."""
    n = counts.get(key, 0) - 1
    if n <= 0:
        counts.pop(key, None)
    else:
        counts[key] = n
    return n
