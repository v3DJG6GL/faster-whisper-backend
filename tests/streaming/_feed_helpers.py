"""Helpers shared by the StreamSession test modules (test_buffer_trim,
test_finalize_paths, test_session, test_utterance_frames).

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
