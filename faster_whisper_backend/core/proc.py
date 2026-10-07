"""Child-process helpers shared by the modules that run ffmpeg / yt-dlp."""
from __future__ import annotations

import asyncio


async def terminate_then_kill(proc: "asyncio.subprocess.Process", grace: float = 5.0) -> None:
    """Stop an asyncio child: SIGTERM, wait up to `grace` seconds, then
    SIGKILL, and always reap it. A child that already exited (or vanished
    between the check and the signal) is a no-op."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        _kill(proc)
        await proc.wait()
    except asyncio.CancelledError:
        # Cancelled during the grace (a second cancel stacked on the one the
        # caller is handling): a child that ignores SIGTERM would otherwise
        # outlive the request. SIGKILL, reap, then let the cancel through.
        _kill(proc)
        try:
            await asyncio.shield(proc.wait())
        except asyncio.CancelledError:
            pass
        raise


def _kill(proc: "asyncio.subprocess.Process") -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass
