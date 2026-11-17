"""core/proc.python_exe / python_env: the interpreter a Python child is
started with. In a Windows venv sys.executable is the venv's launcher, which
starts the base interpreter as its own child: a kill reached only the
launcher, and the rule, decode and link helpers went on running (the Windows
CI leg, run 1860: killed helpers kept answering)."""

import os
import re
import sys

import pytest

from faster_whisper_backend.core import proc as core_proc
from faster_whisper_backend.paths import REPO_ROOT


def _as(monkeypatch, platform, executable, base):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "executable", executable)
    monkeypatch.setattr(sys, "_base_executable", base, raising=False)


def test_a_windows_venv_starts_the_base_interpreter_as_the_venv(monkeypatch):
    venv = os.path.abspath(os.path.join("venv", "Scripts", "python.exe"))
    base = os.path.abspath(os.path.join("Python314", "python.exe"))
    _as(monkeypatch, "win32", venv, base)
    monkeypatch.setenv("KEEP", "1")
    assert core_proc.python_exe() == base
    env = core_proc.python_env()
    assert env["__PYVENV_LAUNCHER__"] == venv and env["KEEP"] == "1"
    # A caller's own environment is kept as given, not os.environ.
    own = core_proc.python_env({"ONLY": "x"})
    assert own == {"ONLY": "x", "__PYVENV_LAUNCHER__": venv}


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_outside_a_windows_venv_the_child_is_this_interpreter(monkeypatch, platform):
    exe = os.path.abspath("python")
    _as(monkeypatch, platform, exe, exe)
    assert core_proc.python_exe() == exe
    assert core_proc.python_env() is None
    given = {"A": "1"}
    assert core_proc.python_env(given) == given and core_proc.python_env(given) is not given


def test_a_linux_venv_keeps_sys_executable(monkeypatch):
    # A POSIX venv's python is a symlink, not a launcher: nothing to bypass.
    _as(monkeypatch, "linux", "/srv/venv/bin/python", "/usr/bin/python3.14")
    assert core_proc.python_exe() == "/srv/venv/bin/python"
    assert core_proc.python_env() is None


def test_every_python_child_is_started_through_python_exe():
    # A new sys.executable child would bring the launcher back on Windows.
    allowed = {
        # os.execve replaces this process; there is no child to kill.
        os.path.join("admin", "restart_service.py"),
        # Locates the DLL directories, starts nothing.
        "main.py",
        "core" + os.sep + "proc.py",
    }
    pkg = os.path.join(REPO_ROOT, "faster_whisper_backend")
    hits = []
    for root, _dirs, files in os.walk(pkg):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, pkg)
            if rel in allowed:
                continue
            with open(path, encoding="utf-8") as fh:
                for no, line in enumerate(fh, 1):
                    code = line.split("#", 1)[0]
                    if re.search(r"\bsys\.executable\b", code) and "``" not in line:
                        hits.append(f"{rel}:{no}: {line.strip()}")
    assert not hits, hits
