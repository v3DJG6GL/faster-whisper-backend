"""LoopLock: a module-level asyncio lock that survives the test suite's
loop-per-lifespan. A lock left held by a dead loop must not wedge the next
loop (the Windows CI failure at lifespan shutdown, run 997 onwards)."""

import asyncio

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
