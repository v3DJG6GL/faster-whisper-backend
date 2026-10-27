"""Child-process helpers shared by the modules that run ffmpeg / yt-dlp."""
from __future__ import annotations

import asyncio
import os
import signal


async def terminate_then_kill(proc: "asyncio.subprocess.Process", grace: float = 5.0) -> None:
    """Stop an asyncio child: SIGTERM, wait up to `grace` seconds, then
    SIGKILL, and always reap it. A child that already exited (or vanished
    between the check and the signal) is a no-op. A child spawned with
    start_new_session is signalled with its whole process group."""
    if proc.returncode is not None:
        return
    try:
        pgid = _own_group(proc)
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        kill(proc)
        await proc.wait()
    except asyncio.CancelledError:
        # Cancelled during the grace (a second cancel stacked on the one the
        # caller is handling): a child that ignores SIGTERM would otherwise
        # outlive the request. SIGKILL, reap, then let the cancel through.
        kill(proc)
        try:
            await asyncio.shield(proc.wait())
        except asyncio.CancelledError:
            pass
        raise


def kill(proc: "asyncio.subprocess.Process") -> None:
    """SIGKILL the child (and its process group, see _own_group) now."""
    try:
        pgid = _own_group(proc)
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass


def _own_group(proc: "asyncio.subprocess.Process") -> "int | None":
    """The child's process group when it leads one (start_new_session), so
    what it spawned goes down with it: yt-dlp has no SIGTERM handler and
    dies without stopping the ffmpeg merge it runs. None on Windows (no
    killpg; per-PID there) and for a child sharing our group."""
    if os.name == "nt":
        return None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return None
    return pgid if pgid == proc.pid else None
