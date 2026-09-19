"""LoopLock: a module-level asyncio lock that survives the test suite's
loop-per-lifespan. A lock left held by a dead loop must not wedge the next
loop (the Windows CI failure at lifespan shutdown, run 997 onwards)."""

import asyncio
import threading

from faster_whisper_backend.core.loop_lock import LoopLock


def test_stale_lock_from_a_dead_loop_is_replaced():
    lk = LoopLock()

    async def hold_forever():
        await lk.acquire()          # never released: the loop dies with it held

    loop = asyncio.new_event_loop()
    loop.run_until_complete(hold_forever())
    loop.close()
    assert lk.locked()

    async def use():
        async with lk:
            return lk.locked()
    assert asyncio.run(use()) is True
    assert not lk.locked()


def test_behaves_like_a_lock_within_one_loop():
    lk = LoopLock()
    order = []

    async def worker(name):
        async with lk:
            order.append(name)
            await asyncio.sleep(0.01)
            order.append(name + "-done")

    async def main():
        await asyncio.gather(worker("a"), worker("b"))
    asyncio.run(main())
    assert order == ["a", "a-done", "b", "b-done"]


def test_a_second_live_loop_never_swaps_out_a_held_lock():
    """Two loops alive at once (a TestClient portal thread plus a test's
    own asyncio.run): the second loop's acquire must not replace the lock
    the first one holds — its release used to hit a fresh, unlocked object
    and raise "Lock is not acquired" out of the `async with`."""
    lk = LoopLock()
    held = threading.Event()
    go = threading.Event()
    errors = []

    async def first():
        async with lk:
            held.set()
            await asyncio.to_thread(go.wait, 5)
            assert lk.locked()

    def run_first():
        try:
            asyncio.run(first())
        except BaseException as e:  # noqa: BLE001 — reported to the test
            errors.append(e)

    t = threading.Thread(target=run_first)
    t.start()
    try:
        assert held.wait(5)

        async def second():
            assert not lk.locked()      # this loop's view; creates nothing
            async with lk:
                return lk.locked()
        assert asyncio.run(second()) is True
    finally:
        go.set()
        t.join(5)
    assert errors == []
    assert not lk.locked()


def test_release_without_acquire_raises():
    lk = LoopLock()

    async def main():
        try:
            lk.release()
        except RuntimeError:
            return True
        return False
    assert asyncio.run(main()) is True
