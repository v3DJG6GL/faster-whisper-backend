"""core/proc.terminate_then_kill: SIGTERM, grace, SIGKILL, always reap."""

import asyncio
import os
import signal
import sys
import time

import psutil
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


def test_kill_takes_the_childs_process_group_down(tmp_path):
    """A start_new_session child is killed with its group: yt-dlp killed
    alone leaves its ffmpeg merge running. The backgrounded subshell
    outlives a per-PID kill of the `sleep 30` that sh exec'd into."""
    marker = tmp_path / "orphan-ran"

    async def run():
        p = await asyncio.create_subprocess_exec(
            "sh", "-c", f"(sleep 1.5; touch '{marker}') & exec sleep 30",
            start_new_session=True)
        await asyncio.sleep(0.3)          # let the subshell fork
        core_proc.kill(p)
        await asyncio.wait_for(p.wait(), 10)
    asyncio.run(run())
    time.sleep(2.5)
    assert not marker.exists()


def _gone_within(pid: int, seconds: float) -> bool:
    """True once ``pid`` has ended. A zombie counts as ended: the group member
    is a grandchild, so once its parent died the init process has to reap it,
    and in a container without an init (CI's job container runs `tail -f
    /dev/null` as PID 1) it stays a zombie forever."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def test_terminate_then_kill_reaps_a_term_ignoring_group_member(tmp_path):
    """The leader dies on SIGTERM, but a group member that ignores it (the
    shape of an ffmpeg merge yt-dlp runs) is SIGKILLed once the grace is up,
    through the group id taken before the leader was reaped."""
    pidfile = tmp_path / "gc.pid"
    member = ("import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
              "time.sleep(30)")

    async def run():
        p = await asyncio.create_subprocess_exec(
            "bash", "-c",
            f"'{sys.executable}' -c '{member}' </dev/null >/dev/null 2>&1 & "
            f"echo $! > '{pidfile}'; wait",
            start_new_session=True, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)          # let the member install its handler
        await core_proc.terminate_then_kill(p, grace=0.5)
        return p
    p = asyncio.run(run())
    gc_pid = int(pidfile.read_text())
    try:
        assert p.returncode == -signal.SIGTERM
        assert _gone_within(gc_pid, 1.5)
    finally:
        try:
            os.kill(gc_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_terminate_then_kill_lets_a_group_member_finish_within_the_grace(tmp_path):
    """A member that stops on SIGTERM by itself is never SIGKILLed: the sweep
    only fires for what is still alive when the grace runs out."""
    pidfile = tmp_path / "gc.pid"
    done = tmp_path / "flushed"
    member = ("import signal,sys,time,pathlib\n"
              "def h(*a):\n"
              "    time.sleep(0.3); pathlib.Path(sys.argv[1]).touch(); sys.exit(0)\n"
              "signal.signal(signal.SIGTERM,h)\n"
              "time.sleep(30)\n")
    script = tmp_path / "member.py"
    script.write_text(member)

    async def run():
        p = await asyncio.create_subprocess_exec(
            "bash", "-c",
            f"'{sys.executable}' '{script}' '{done}' </dev/null >/dev/null 2>&1 & "
            f"echo $! > '{pidfile}'; wait",
            start_new_session=True, stdin=asyncio.subprocess.DEVNULL)
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        await core_proc.terminate_then_kill(p, grace=3)
        return p
    asyncio.run(run())
    gc_pid = int(pidfile.read_text())
    assert _gone_within(gc_pid, 3)
    assert done.exists()


_ZOMBIE_MEMBER_RUN = r"""
import asyncio, ctypes, sys, time
from faster_whisper_backend.core import proc as core_proc
# Become the orphans' reaper and never reap them, as the container's
# PID 1 (`python main.py`) does: the member then lingers as a zombie.
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER

async def main():
    p = await asyncio.create_subprocess_exec(
        "bash", "-c", "sleep 30 & wait", start_new_session=True)
    await asyncio.sleep(0.3)
    t0 = time.monotonic()
    await core_proc.terminate_then_kill(p, grace=4)
    print(time.monotonic() - t0)
asyncio.run(main())
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="PR_SET_CHILD_SUBREAPER is Linux-only")
def test_group_sweep_does_not_wait_on_a_zombie_member():
    """A member that exited on SIGTERM but sits unreaped as a zombie is
    not a running member: the sweep returns at once instead of holding
    every abort for the rest of the grace."""
    import subprocess
    out = subprocess.run(
        [sys.executable, "-c", _ZOMBIE_MEMBER_RUN], capture_output=True,
        text=True, timeout=30, check=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    assert float(out.stdout.strip()) < 1.5, out
