"""An asyncio lock that follows the running event loop.

``asyncio.Lock`` binds to the first loop that acquires it and raises
``RuntimeError: ... is bound to a different event loop`` from any other loop.
The stage caches (diarization, BGM separation, translation) guard their
pipeline with a module-level lock, and the process normally has one loop for
its whole life, so that never mattered in production. Under the test suite
every ``TestClient`` lifespan is its own loop: a lock a previous loop left
held (worker cancelled while loading, loop torn down before the release ran)
wedges the next lifespan's shutdown (observed on the Windows CI job from run
997 on, ``drop_pipeline`` at shutdown). This lock keeps one ``asyncio.Lock``
per event loop, so a stale lock from a dead loop can never be waited on, and
a lock one live loop holds is never swapped out from under it by another
(``release()`` always finds the object its own loop acquired). Closed loops
are pruned. It does NOT exclude two live loops from each other — nothing
built on ``asyncio.Lock`` can; with one loop it is exactly an
``asyncio.Lock``.
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
