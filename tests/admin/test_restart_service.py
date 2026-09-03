"""Tests for restart_service.trigger_self_restart.

trigger_self_restart ARMS a threading.Timer that, after the delay, calls
os.execv / subprocess.Popen / os._exit. We MUST NOT let that fire (it would
kill pytest). Every test monkeypatches threading.Timer to a capture-only stub
(records the callback, never starts a thread) and guards os.execv, os._exit,
and subprocess.Popen so even an accidental direct invocation can't escape.
"""

import os
import subprocess
import threading

import pytest

from faster_whisper_backend.admin import restart_service


class _FakeTimer:
    """Captures (delay, callback) instead of scheduling a real timer."""
    instances: list["_FakeTimer"] = []

    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.started = False
        _FakeTimer.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):  # pragma: no cover - never armed for real
        pass


@pytest.fixture(autouse=True)
def _guard(monkeypatch):
    """Capture Timer arming and neuter every process-killing primitive."""
    _FakeTimer.instances = []
    monkeypatch.setattr(restart_service.threading, "Timer", _FakeTimer)

    def _boom_execv(*a, **k):  # pragma: no cover - asserts it's never reached
        raise AssertionError("os.execv must not be called during tests")

    def _boom_exit(*a, **k):  # pragma: no cover
        raise AssertionError("os._exit must not be called during tests")

    monkeypatch.setattr(restart_service.os, "execv", _boom_execv)
    monkeypatch.setattr(restart_service.os, "_exit", _boom_exit)
    monkeypatch.setattr(
        restart_service.subprocess, "Popen",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Popen must not be called during tests")),
    )
    yield


# ---------------------------------------------------------------------------
# Non-win32 path (this Linux box)
# ---------------------------------------------------------------------------

def test_reexec_label_and_timer_armed(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    label = restart_service.trigger_self_restart(delay_sec=2.5)
    assert label == "execv-reexec"
    assert len(_FakeTimer.instances) == 1
    t = _FakeTimer.instances[0]
    assert t.interval == 2.5
    assert t.started is True
    assert callable(t.function)


def test_reexec_callback_invokes_execv(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    captured = {}

    def fake_execv(path, argv):
        captured["path"] = path
        captured["argv"] = argv

    monkeypatch.setattr(restart_service.os, "execv", fake_execv)
    restart_service.trigger_self_restart(delay_sec=0.1)
    # Manually fire the captured callback (the real Timer never runs it).
    _FakeTimer.instances[0].function()
    assert captured["path"] == restart_service.sys.executable
    assert captured["argv"][0] == restart_service.sys.executable


def test_reexec_replays_the_original_command_line_not_sys_argv(monkeypatch):
    """Under `python -m faster_whisper_backend`, sys.argv[0] is the path of
    the package's __main__.py; re-running THAT as a script puts the package
    dir on sys.path[0] and the exec'd process dies with ModuleNotFoundError.
    The re-exec must replay sys.orig_argv (module flag + interpreter flags)."""
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    monkeypatch.setattr(restart_service.sys, "orig_argv",
                        ["/usr/bin/python3", "-X", "dev", "-m", "faster_whisper_backend"])
    monkeypatch.setattr(restart_service.sys, "argv",
                        ["/repo/faster_whisper_backend/__main__.py"])
    captured = {}
    monkeypatch.setattr(restart_service.os, "execv",
                        lambda path, argv: captured.update(path=path, argv=argv))
    restart_service.trigger_self_restart(delay_sec=0.1)
    _FakeTimer.instances[0].function()
    assert captured["argv"] == [restart_service.sys.executable,
                                "-X", "dev", "-m", "faster_whisper_backend"]
    assert "__main__.py" not in " ".join(captured["argv"])


def test_reexec_callback_falls_back_to_exit_on_execv_failure(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "linux")

    def failing_execv(*a, **k):
        raise OSError("exec failed")

    exited = {}
    monkeypatch.setattr(restart_service.os, "execv", failing_execv)
    monkeypatch.setattr(restart_service.os, "_exit",
                        lambda code: exited.__setitem__("code", code))
    restart_service.trigger_self_restart()
    _FakeTimer.instances[0].function()
    assert exited["code"] == 0


# ---------------------------------------------------------------------------
# win32 branch: WinSW missing -> bare-exit fallback
# ---------------------------------------------------------------------------

def test_win32_winsw_missing_fallback(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "win32")
    # Force "WhisperAPI.exe not present".
    monkeypatch.setattr(restart_service.os.path, "isfile", lambda p: False)
    label = restart_service.trigger_self_restart(delay_sec=1.0)
    assert label == "winsw-missing-fallback"
    assert len(_FakeTimer.instances) == 1
    assert _FakeTimer.instances[0].interval == 1.0
    # Callback is the bare-exit lambda: firing it would call os._exit(0).
    exited = {}
    monkeypatch.setattr(restart_service.os, "_exit",
                        lambda code: exited.__setitem__("code", code))
    _FakeTimer.instances[0].function()
    assert exited["code"] == 0


# ---------------------------------------------------------------------------
# win32 branch: WinSW present -> restart! spawn
# ---------------------------------------------------------------------------

def test_win32_winsw_restart_bang(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "win32")
    monkeypatch.setattr(restart_service.os.path, "isfile", lambda p: True)
    # subprocess constants used in the flags only exist on Windows; stub them
    # so building the flags doesn't AttributeError on Linux.
    for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
        if not hasattr(restart_service.subprocess, name):
            monkeypatch.setattr(restart_service.subprocess, name, 0,
                                raising=False)
    label = restart_service.trigger_self_restart(delay_sec=0.5)
    assert label == "winsw-restart-bang"
    assert len(_FakeTimer.instances) == 1
    assert _FakeTimer.instances[0].interval == 0.5

    # Fire the callback: it should spawn WinSW then sleep+_exit. Stub Popen,
    # time.sleep, and os._exit so nothing real happens.
    # _flush_before_exit imports main, whose import chain shells out to
    # `git describe` through the SAME monkeypatched subprocess.Popen, so the
    # recorder must collect every call and select the WinSW spawn rather
    # than assume the callback makes exactly one Popen call.
    popen_calls = []

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))

    exited = {}
    monkeypatch.setattr(restart_service.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(restart_service.time, "sleep", lambda s: None)
    monkeypatch.setattr(restart_service.os, "_exit",
                        lambda code: exited.__setitem__("code", code))
    _FakeTimer.instances[0].function()

    winsw = [a for a, _ in popen_calls
             if a and str(a[0]).endswith("WhisperAPI.exe")]
    assert len(winsw) == 1
    assert winsw[0][1] == "restart!"
    assert exited["code"] == 0


# ---------------------------------------------------------------------------
# Shutdown obligations: every exec/exit path flushes the lossy lifespan
# duties (held receipts, NVML) FIRST — the restart bypasses ASGI shutdown.
# ---------------------------------------------------------------------------

def _order_probe(monkeypatch):
    order = []
    monkeypatch.setattr(restart_service, "_flush_before_exit",
                        lambda: order.append("flush"))
    return order


def test_reexec_flushes_before_execv(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    order = _order_probe(monkeypatch)
    monkeypatch.setattr(restart_service.os, "execv",
                        lambda path, argv: order.append("execv"))
    restart_service.trigger_self_restart()
    _FakeTimer.instances[0].function()
    assert order == ["flush", "execv"]


def test_winsw_missing_fallback_flushes_before_exit(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "win32")
    monkeypatch.setattr(restart_service.os.path, "isfile", lambda p: False)
    order = _order_probe(monkeypatch)
    monkeypatch.setattr(restart_service.os, "_exit",
                        lambda code: order.append("exit"))
    restart_service.trigger_self_restart()
    _FakeTimer.instances[0].function()
    assert order == ["flush", "exit"]


def test_winsw_restart_bang_flushes_before_exit(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "win32")
    monkeypatch.setattr(restart_service.os.path, "isfile", lambda p: True)
    for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
        if not hasattr(restart_service.subprocess, name):
            monkeypatch.setattr(restart_service.subprocess, name, 0,
                                raising=False)
    order = _order_probe(monkeypatch)
    monkeypatch.setattr(restart_service.subprocess, "Popen",
                        lambda *a, **k: order.append("popen"))
    monkeypatch.setattr(restart_service.time, "sleep", lambda s: None)
    monkeypatch.setattr(restart_service.os, "_exit",
                        lambda code: order.append("exit"))
    restart_service.trigger_self_restart()
    _FakeTimer.instances[0].function()
    assert order == ["popen", "flush", "exit"]


def test_flush_before_exit_swallows_failures(monkeypatch):
    """A broken flush must never block the restart itself."""
    from faster_whisper_backend.runtime import system_stats

    def _boom():
        raise RuntimeError("nvml exploded")

    monkeypatch.setattr(system_stats, "shutdown", _boom)
    restart_service._flush_before_exit()  # must not raise
