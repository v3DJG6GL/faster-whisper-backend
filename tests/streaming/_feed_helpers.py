"""Helpers shared by the StreamSession test modules (test_buffer_trim,
test_equivalence, test_finalize_paths, test_seams, test_session,
test_utterance_frames).

Import as ``from tests.streaming._feed_helpers import ...``.
"""

import asyncio

from tests._streaming_helpers import const_pcm


def feed_levels(session, *chunks: tuple[int, int]) -> None:
    """asyncio.run one ``session.feed_pcm(const_pcm(level, ms))`` per
    ``(level, ms)`` chunk, in order."""
    async def run():
        for level, ms in chunks:
            await session.feed_pcm(const_pcm(level, ms))

    asyncio.run(run())


def feed_and_close(session, *chunks: tuple[int, int]) -> None:
    """``feed_levels``, then ``session.close()``, all on one event loop."""
    async def run():
        for level, ms in chunks:
            await session.feed_pcm(const_pcm(level, ms))
        await session.close()

    asyncio.run(run())


async def grow_document(session, *pieces: str) -> None:
    """Append each raw piece to ``session.raw_confirmed`` and emit its document,
    in order (no audio, no VAD)."""
    for piece in pieces:
        session.raw_confirmed += piece
        await session._emit_update()


def grow_and_close(session, *pieces: str) -> None:
    """asyncio.run ``grow_document(session, *pieces)``, then ``session.close()``
    (which commits the whole document)."""
    async def run():
        await grow_document(session, *pieces)
        await session.close()

    asyncio.run(run())
