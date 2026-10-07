"""core/proc.terminate_then_kill: SIGTERM, grace, SIGKILL, always reap."""

import asyncio
import os
import signal
import sys

import pytest

from faster_whisper_backend.core import proc as core_proc

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="needs a POSIX shell and signals")


def test_terminate_then_kill_kills_a_term_ignoring_child():
    """`exec` keeps the ignored TERM on `sleep` itself (ignored dispositions
    survive exec), so the SIGKILL hits the process that ignores TERM and no
    grandchild is orphaned; the returncode proves SIGKILL ended it, not the
    30 s sleep running out."""
    async def run():
        p = await asyncio.create_subprocess_exec(
            "sh", "-c", "trap '' TERM; exec sleep 30")
        await asyncio.sleep(0.2)          # let the trap install
        await core_proc.terminate_then_kill(p, grace=0.3)
        return p
    p = asyncio.run(run())
    assert p.returncode == -signal.SIGKILL


def test_cancel_during_grace_still_kills_and_reaps_the_child():
    """A second cancel while waiting out the SIGTERM grace must not leave a
    child that ignores SIGTERM running."""
    async def run():
        p = await asyncio.create_subprocess_exec(
            "sh", "-c", "trap '' TERM; exec sleep 30")
        await asyncio.sleep(0.2)
        task = asyncio.ensure_future(core_proc.terminate_then_kill(p, grace=2))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return p
    p = asyncio.run(run())
    assert p.returncode == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.kill(p.pid, 0)
