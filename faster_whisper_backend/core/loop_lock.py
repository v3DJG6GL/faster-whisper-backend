"""An asyncio lock that follows the running event loop.

``asyncio.Lock`` binds to the first loop that acquires it and raises
``RuntimeError: ... is bound to a different event loop`` from any other loop.
The stage caches (diarization, BGM separation, translation) guard their
pipeline with a module-level lock, and the process normally has one loop for
its whole life, so that never mattered in production. Under the test suite
every ``TestClient`` lifespan is its own loop: a lock a previous loop left
held (worker cancelled while loading, loop torn down before the release ran)
wedges the next lifespan's shutdown (observed on the Windows CI job from run
997 on, ``drop_pipeline`` at shutdown). This lock is recreated whenever the
running loop changed, so a stale lock from a dead loop can never be waited
on. With one loop it is exactly an ``asyncio.Lock``.
"""

from __future__ import annotations

import asyncio


class LoopLock:
    __slots__ = ("_lock", "_loop")

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _current(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock

    async def acquire(self) -> bool:
        return await self._current().acquire()

    def release(self) -> None:
        self._current().release()

    def locked(self) -> bool:
        try:
            return self._current().locked()
        except RuntimeError:          # no running loop
            return bool(self._lock is not None and self._lock.locked())

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc) -> None:
        self.release()
