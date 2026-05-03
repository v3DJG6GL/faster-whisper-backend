"""
Self-restart helper for the WhisperAPI Windows Service.

Strategy: schedule a clean process exit; let NSSM's default `AppExit=Restart`
relaunch the wrapped python process. This is the canonical NSSM pattern —
zero subprocess, zero job-object fight, zero PowerShell, zero Task
Scheduler. Restart latency is just NSSM's `AppRestartDelay` + python boot
(~3-4 s end-to-end on a no-preload deployment).

Why not spawn a detached PowerShell helper instead? On a service-jobbed
parent, `CREATE_BREAKAWAY_FROM_JOB` interacts badly with `DETACHED_PROCESS`:
`subprocess.Popen` returns a PID, but the child fails during process-init
(`STATUS_DLL_INIT_FAILED 0xC0000142`) and never actually runs. NSSM 2.24
doesn't even use Win32 job objects (the parent's job comes from SCM), so
breakaway is fighting the kernel, not NSSM.

NSSM defaults that this relies on (set explicitly in install-service.ps1):
  AppExit\\0       = Restart   (relaunch on clean exit)
  AppRestartDelay = 1500 ms   (gap so in-flight requests can drain)
  AppThrottle     = 1500 ms   (defends against boot-loop on broken config)
"""

from __future__ import annotations

import os
import sys
import threading


def trigger_self_restart(delay_sec: float = 1.5) -> str:
    """Schedule process exit so NSSM's AppExit=Restart relaunches us.

    Returns immediately with a method label (for the admin UI to display).
    The Timer thread fires after `delay_sec` so the caller's HTTP response
    has time to flush over loopback before uvicorn dies.

    Raises RuntimeError on non-Windows hosts — the linux dev environment
    has no NSSM to relaunch us, so a self-exit there would just take the
    server down for good.
    """
    if sys.platform != "win32":
        raise RuntimeError(
            "self-restart relies on NSSM's AppExit=Restart and is "
            f"Windows-only (current platform: {sys.platform})"
        )

    # os._exit (not sys.exit): we want to bypass uvicorn's signal handlers
    # and Python's atexit hooks. faster-whisper has no on-disk state that
    # needs flushing; the OS reclaims handles and VRAM on process death.
    threading.Timer(delay_sec, lambda: os._exit(0)).start()
    return "nssm-autoexit"
