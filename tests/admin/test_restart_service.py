"""Tests for restart_service: trigger_self_restart and the startup TMPDIR sweep.

trigger_self_restart ARMS a threading.Timer that, after the delay, calls
os.execve / subprocess.Popen / os._exit. We MUST NOT let that fire (it would
kill pytest). Every test monkeypatches threading.Timer to a capture-only stub
(records the callback, never starts a thread) and guards os.execve, os._exit,
and subprocess.Popen so even an accidental direct invocation can't escape.
"""

import importlib
import os

import pytest

from faster_whisper_backend.admin import restart_service
from faster_whisper_backend.settings import config

# Bound at import, before the autouse _guard stubs the module attribute, so
# the direct-call test exercises the real function rather than the stub.
_REAL_FLUSH = restart_service._flush_before_exit


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
        raise AssertionError("os.execve must not be called during tests")

    def _boom_exit(*a, **k):  # pragma: no cover
        raise AssertionError("os._exit must not be called during tests")

    monkeypatch.setattr(restart_service.os, "execve", _boom_execv)
    monkeypatch.setattr(restart_service.os, "execv", _boom_execv)
    monkeypatch.setattr(restart_service.os, "_exit", _boom_exit)
    # The real flush drains receipt_hold, flushes the stats sampler and calls
    # system_stats.shutdown(), which flips the module-global NVML_OK for the
    # rest of the session.
    monkeypatch.setattr(restart_service, "_flush_before_exit", lambda: None)
    monkeypatch.setattr(
        restart_service.subprocess, "Popen",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Popen must not be called during tests")),
    )
    yield


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

    def fake_execv(path, argv, env):
        captured["path"] = path
        captured["argv"] = argv

    monkeypatch.setattr(restart_service.os, "execve", fake_execv)
    restart_service.trigger_self_restart(delay_sec=0.1)
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
    monkeypatch.setattr(restart_service.os, "execve",
                        lambda path, argv, env: captured.update(path=path, argv=argv))
    restart_service.trigger_self_restart(delay_sec=0.1)
    _FakeTimer.instances[0].function()
    assert captured["argv"] == [restart_service.sys.executable,
                                "-X", "dev", "-m", "faster_whisper_backend"]
    assert "__main__.py" not in " ".join(captured["argv"])


def test_reexec_rereads_rotated_secret_files(monkeypatch, tmp_path):
    """config writes a *_FILE secret's contents into os.environ and reads the
    file only while the plain var is unset; an exec that inherited that env
    kept the OLD bootstrap key after the operator rotated the file and pressed
    Restart. The re-exec drops exactly the file-derived vars."""
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    secret = tmp_path / "admin_key"
    secret.write_text("sk-old\n")
    monkeypatch.delenv("WHISPER_BOOTSTRAP_ADMIN_KEY", raising=False)
    monkeypatch.setenv("WHISPER_BOOTSTRAP_ADMIN_KEY_FILE", str(secret))
    monkeypatch.setenv("WHISPER_HF_TOKEN", "hf_plain")   # set directly: kept
    try:
        importlib.reload(config)
        assert os.environ["WHISPER_BOOTSTRAP_ADMIN_KEY"] == "sk-old"
        captured = {}
        monkeypatch.setattr(restart_service.os, "execve",
                            lambda path, argv, env: captured.update(env=env))
        restart_service.trigger_self_restart(delay_sec=0.1)
        _FakeTimer.instances[0].function()
        env = captured["env"]
        assert "WHISPER_BOOTSTRAP_ADMIN_KEY" not in env
        assert env["WHISPER_BOOTSTRAP_ADMIN_KEY_FILE"] == str(secret)
        assert env["WHISPER_HF_TOKEN"] == "hf_plain"
    finally:
        monkeypatch.undo()
        # The *_FILE prepass wrote the secret straight into os.environ, which
        # monkeypatch cannot undo; clear it before the restoring reload.
        os.environ.pop("WHISPER_BOOTSTRAP_ADMIN_KEY", None)
        importlib.reload(config)


def test_reexec_rereads_the_dotenv(monkeypatch):
    """load_dotenv copies .env into os.environ and never overrides a var that
    is already set; an exec that inherited those copies ran on the OLD .env
    after the operator edited or removed a line and pressed Restart. The
    re-exec drops exactly the .env-sourced vars."""
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    monkeypatch.setattr(config, "_ENV_FROM_DOTENV", {"WHISPER_BEAM_SIZE"})
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "5")
    monkeypatch.setenv("WHISPER_DEFAULT_PROMPT", "real")   # environment's: kept
    captured = {}
    monkeypatch.setattr(restart_service.os, "execve",
                        lambda path, argv, env: captured.update(env=env))
    restart_service.trigger_self_restart(delay_sec=0.1)
    _FakeTimer.instances[0].function()
    assert "WHISPER_BEAM_SIZE" not in captured["env"]
    assert captured["env"]["WHISPER_DEFAULT_PROMPT"] == "real"


def test_reexec_does_not_restore_a_dotenv_hf_token(monkeypatch):
    """A bare HF_TOKEN from .env was recorded as the service environment's
    and put back, so removing it from .env never took effect on Restart."""
    from faster_whisper_backend.pipeline import apply as pl_apply
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    monkeypatch.setattr(config, "_ENV_FROM_DOTENV", {"HF_TOKEN"})
    monkeypatch.setattr(pl_apply, "_ENV_HF_TOKEN", "hf_dotenv")
    monkeypatch.setenv("HF_TOKEN", "hf_dotenv")
    monkeypatch.setenv(pl_apply.SERVICE_HF_TOKEN_ENV, "hf_dotenv")
    captured = {}
    monkeypatch.setattr(restart_service.os, "execve",
                        lambda path, argv, env: captured.update(env=env))
    restart_service.trigger_self_restart(delay_sec=0.1)
    _FakeTimer.instances[0].function()
    assert "HF_TOKEN" not in captured["env"]
    assert pl_apply.SERVICE_HF_TOKEN_ENV not in captured["env"]


def test_reexec_drops_an_empty_hf_token_record(monkeypatch):
    """A process started without a token records "" for the environment's.
    Kept on exec, it blocked the record of a bare HF_TOKEN added to .env since
    (main.py only sets the record when absent), so clearing the config field
    later popped that .env token."""
    from faster_whisper_backend.pipeline import apply as pl_apply
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    monkeypatch.setattr(config, "_ENV_FROM_DOTENV", set())
    monkeypatch.setattr(pl_apply, "_ENV_HF_TOKEN", None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv(pl_apply.SERVICE_HF_TOKEN_ENV, "")
    captured = {}
    monkeypatch.setattr(restart_service.os, "execve",
                        lambda path, argv, env: captured.update(env=env))
    restart_service.trigger_self_restart(delay_sec=0.1)
    _FakeTimer.instances[0].function()
    assert pl_apply.SERVICE_HF_TOKEN_ENV not in captured["env"]


def test_reexec_drops_the_hf_token_promoted_from_config(monkeypatch):
    """main.py copies cfg.HF_TOKEN into the bare HF_TOKEN that huggingface_hub
    reads. An exec that inherited the copy kept sending a token the operator
    had withdrawn (emptied secret file, field taken out of config): the new
    process promotes nothing and adopts the leftover as the environment's."""
    from faster_whisper_backend.pipeline import apply as pl_apply
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    monkeypatch.setattr(pl_apply, "_ENV_HF_TOKEN", None)
    monkeypatch.setenv("HF_TOKEN", "hf_promoted")
    captured = {}
    monkeypatch.setattr(restart_service.os, "execve",
                        lambda path, argv, env: captured.update(env=env))
    restart_service.trigger_self_restart(delay_sec=0.1)
    _FakeTimer.instances[0].function()
    assert "HF_TOKEN" not in captured["env"]


def test_reexec_restores_the_service_environments_hf_token(monkeypatch):
    """A token the service environment set (systemd Environment=) survives
    the restart even when a config token was promoted over it."""
    from faster_whisper_backend.pipeline import apply as pl_apply
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    monkeypatch.setattr(pl_apply, "_ENV_HF_TOKEN", "hf_service")
    monkeypatch.setenv("HF_TOKEN", "hf_promoted")
    captured = {}
    monkeypatch.setattr(restart_service.os, "execve",
                        lambda path, argv, env: captured.update(env=env))
    restart_service.trigger_self_restart(delay_sec=0.1)
    _FakeTimer.instances[0].function()
    assert captured["env"]["HF_TOKEN"] == "hf_service"


def test_reexec_callback_falls_back_to_exit_on_execv_failure(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "linux")

    def failing_execv(*a, **k):
        raise OSError("exec failed")

    exited = {}
    monkeypatch.setattr(restart_service.os, "execve", failing_execv)
    monkeypatch.setattr(restart_service.os, "_exit",
                        lambda code: exited.__setitem__("code", code))
    restart_service.trigger_self_restart()
    _FakeTimer.instances[0].function()
    assert exited["code"] == 0


def test_win32_winsw_missing_fallback(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "win32")
    # Force "WhisperAPI.exe not present".
    monkeypatch.setattr(restart_service.os.path, "isfile", lambda p: False)
    label = restart_service.trigger_self_restart(delay_sec=1.0)
    assert label == "winsw-missing-fallback"
    assert len(_FakeTimer.instances) == 1
    assert _FakeTimer.instances[0].interval == 1.0
    exited = {}
    monkeypatch.setattr(restart_service.os, "_exit",
                        lambda code: exited.__setitem__("code", code))
    _FakeTimer.instances[0].function()
    assert exited["code"] == 0


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

    # The recorder selects the WinSW spawn by name so an extra Popen elsewhere
    # in the callback cannot mask a missing spawn.
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


# The restart bypasses ASGI shutdown, so every exec/exit path must run
# _flush_before_exit (held receipts, stats samples, NVML) first.
def _order_probe(monkeypatch):
    order = []
    monkeypatch.setattr(restart_service, "_flush_before_exit",
                        lambda: order.append("flush"))
    return order


def test_reexec_flushes_before_execv(monkeypatch):
    monkeypatch.setattr(restart_service.sys, "platform", "linux")
    order = _order_probe(monkeypatch)
    monkeypatch.setattr(restart_service.os, "execve",
                        lambda path, argv, env: order.append("execv"))
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
    """A broken flush must never block the restart itself: every duty is
    attempted in order (held receipts, stats samples, NVML), and a failing
    one must not skip the ones after it."""
    from faster_whisper_backend.runtime import system_stats
    from faster_whisper_backend.stats import sampler
    from faster_whisper_backend.transcription import receipt_hold

    reached = []

    def _boom_flush(*a, **k):
        reached.append("flush")
        raise RuntimeError("receipt flush exploded")

    def _boom_sampler():
        reached.append("sampler")
        raise RuntimeError("sampler flush exploded")

    def _boom_shutdown():
        reached.append("shutdown")
        raise RuntimeError("nvml exploded")

    monkeypatch.setattr(receipt_hold, "flush_all", _boom_flush)
    monkeypatch.setattr(sampler, "flush", _boom_sampler)
    monkeypatch.setattr(system_stats, "shutdown", _boom_shutdown)
    _REAL_FLUSH()  # must not raise
    assert reached == ["flush", "sampler", "shutdown"]


def test_reclaim_hard_restart_orphans(tmp_path, monkeypatch):
    """The startup sweep reclaims what an admin restart (os.execve/os._exit,
    no ASGI shutdown) orphaned — urldl- job dirs, sepsrc-/vocals- WAVs —
    while leaving fresh entries (a just-overlapping process) and unrelated
    names alone. (No app_module fixture: that fixture stubs the sweep out
    so TestClient startups never touch the real tempdir.)"""
    import time as _time

    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(restart_service.tempfile, "gettempdir",
                        lambda: str(fake_tmp))

    old = _time.time() - 3600
    old_dir = fake_tmp / "urldl-dead"
    old_dir.mkdir()
    (old_dir / "media.m4a.part").write_bytes(b"x")
    os.utime(old_dir, (old, old))
    for name in ("sepsrc-dead.wav", "vocals-dead.wav"):
        p = fake_tmp / name
        p.write_bytes(b"x")
        os.utime(p, (old, old))
    fresh_dir = fake_tmp / "urldl-live"
    fresh_dir.mkdir()
    (fake_tmp / "sepsrc-live.wav").write_bytes(b"x")
    (fake_tmp / "unrelated.txt").write_bytes(b"x")
    # /captures merge preview + merge_wavs' own `.tmp` sibling.
    for name in ("preview_merge_x.wav", "preview_merge_x.wav.abc.tmp"):
        p = fake_tmp / name
        p.write_bytes(b"x")
        os.utime(p, (old, old))

    restart_service.reclaim_hard_restart_orphans()

    assert not old_dir.exists()
    assert not (fake_tmp / "sepsrc-dead.wav").exists()
    assert not (fake_tmp / "vocals-dead.wav").exists()
    assert not (fake_tmp / "preview_merge_x.wav").exists()
    assert not (fake_tmp / "preview_merge_x.wav.abc.tmp").exists()
    assert fresh_dir.exists()
    assert (fake_tmp / "sepsrc-live.wav").exists()
    assert (fake_tmp / "unrelated.txt").exists()


def test_reclaim_hard_restart_orphans_spares_foreign_pkg_dirs(tmp_path, monkeypatch):
    """The subtitle-package workdir carries the app-specific `fwb-pkg-`
    prefix: a bare `pkg-` dir is a generic name another same-user program
    may own in the shared TMPDIR, so the sweep must not rmtree it."""
    import time as _time

    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(restart_service.tempfile, "gettempdir",
                        lambda: str(fake_tmp))
    old = _time.time() - 3600
    ours = fake_tmp / "fwb-pkg-x"
    foreign = fake_tmp / "pkg-foreign"
    for d in (ours, foreign):
        d.mkdir()
        (d / "out.mkv").write_bytes(b"x")
        os.utime(d, (old, old))

    restart_service.reclaim_hard_restart_orphans()

    assert not ours.exists()
    assert (foreign / "out.mkv").exists()


def test_reclaim_hard_restart_orphans_covers_pipeline_copies_and_uploads(
        tmp_path, monkeypatch):
    """The sweep also reclaims the `urlmedia-` pipeline copies
    (url_media_store.make_pipeline_copy) and the `whisperup-` batch upload
    spools, under the same 60 s age guard."""
    import time as _time

    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(restart_service.tempfile, "gettempdir",
                        lambda: str(fake_tmp))
    old = _time.time() - 3600
    for name in ("urlmedia-dead.m4a", "whisperup-dead.wav"):
        p = fake_tmp / name
        p.write_bytes(b"x")
        os.utime(p, (old, old))
    (fake_tmp / "urlmedia-live.m4a").write_bytes(b"x")
    (fake_tmp / "whisperup-live.wav").write_bytes(b"x")
    (fake_tmp / "tmpabc123.wav").write_bytes(b"x")   # someone else's tempfile
    os.utime(fake_tmp / "tmpabc123.wav", (old, old))

    restart_service.reclaim_hard_restart_orphans()

    assert not (fake_tmp / "urlmedia-dead.m4a").exists()
    assert not (fake_tmp / "whisperup-dead.wav").exists()
    assert (fake_tmp / "urlmedia-live.m4a").exists()
    assert (fake_tmp / "whisperup-live.wav").exists()
    assert (fake_tmp / "tmpabc123.wav").exists()
