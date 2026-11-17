"""Child-process helpers shared by the modules that run ffmpeg / yt-dlp."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Mapping

# How often the post-reap sweep looks for group members still running.
_SWEEP_POLL_S = 0.05


def _windows_venv() -> bool:
    """True in a Windows venv, where sys.executable is the venv's
    python.exe: a launcher that starts the base interpreter as ITS child."""
    base = getattr(sys, "_base_executable", None)
    return (sys.platform == "win32" and bool(base)
            and os.path.normcase(os.path.abspath(base))
            != os.path.normcase(os.path.abspath(sys.executable)))


def python_exe() -> str:
    """The interpreter to start a Python child with. In a Windows venv not
    sys.executable: killing what that starts kills only the launcher, and
    the real interpreter goes on running (a rule past its time limit too)
    with the pipes still open. The base interpreter, given
    __PYVENV_LAUNCHER__ by python_env(), is the venv's interpreter itself —
    multiprocessing starts its Windows children the same way (bpo-35797)."""
    return sys._base_executable if _windows_venv() else sys.executable


def python_env(env: "Mapping[str, str] | None" = None) -> "dict[str, str] | None":
    """The environment for a child started with python_exe(): `env` (None:
    this process's) as is, plus __PYVENV_LAUNCHER__ in a Windows venv, which
    makes the base interpreter take the venv's site-packages. None stays
    None (inherit) everywhere else."""
    if not _windows_venv():
        return None if env is None else dict(env)
    out = dict(os.environ if env is None else env)
    out["__PYVENV_LAUNCHER__"] = sys.executable
    return out


async def terminate_then_kill(proc: "asyncio.subprocess.Process", grace: float = 5.0) -> None:
    """Stop an asyncio child: SIGTERM, wait up to `grace` seconds, then
    SIGKILL, and always reap it. A child that already exited (or vanished
    between the check and the signal) is a no-op. A child spawned with
    start_new_session is signalled with its whole process group, and a
    member of that group still running when the grace is up is SIGKILLed
    too, even after the leader itself has exited."""
    if proc.returncode is not None:
        return
    loop = asyncio.get_running_loop()
    end = loop.time() + grace
    # Taken while the leader is alive: once it is reaped getpgid fails, and
    # the members it leaves behind are reachable only through this id.
    pgid = _own_group(proc)
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        _sigkill(proc, pgid)
        await proc.wait()
        return
    except asyncio.CancelledError:
        # Cancelled during the grace (a second cancel stacked on the one the
        # caller is handling): a child that ignores SIGTERM would otherwise
        # outlive the request. SIGKILL, reap, then let the cancel through.
        _sigkill(proc, pgid)
        try:
            await asyncio.shield(proc.wait())
        except asyncio.CancelledError:
            pass
        raise
    if pgid is not None:
        await _sweep_group(pgid, end)


async def _sweep_group(pgid: int, end: float) -> None:
    """The leader is reaped; give the rest of its group (yt-dlp's ffmpeg
    merge, which may ignore or be slow on SIGTERM) what is left of the
    grace, then SIGKILL whatever still runs. Nothing supervises those
    members once the leader is gone."""
    loop = asyncio.get_running_loop()
    try:
        while await asyncio.to_thread(_group_has_live_member, pgid):
            if loop.time() >= end:
                _killpg_quietly(pgid)
                return
            await asyncio.sleep(_SWEEP_POLL_S)
    except asyncio.CancelledError:
        _killpg_quietly(pgid)
        raise


def _group_has_live_member(pgid: int) -> bool:
    """Whether any process in group `pgid` still runs. A zombie member
    counts as gone: under a PID 1 that never reaps orphans (the container's
    `python main.py`) an exited member keeps the group id answering
    kill(0) for good. Without /proc the kill(0) answer stands."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        names = os.listdir("/proc")
    except OSError:
        return True
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                stat = f.read()
            # Fields after the parenthesised command: state, ppid, pgrp, ...
            fields = stat.rsplit(b")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in (b"Z", b"X"):
                return True
        except (OSError, IndexError, ValueError):
            continue
    return False


def _killpg_quietly(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def kill(proc: "asyncio.subprocess.Process") -> None:
    """SIGKILL the child (and its process group, see _own_group) now."""
    _sigkill(proc, _own_group(proc))


def _sigkill(proc: "asyncio.subprocess.Process", pgid: "int | None") -> None:
    try:
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
