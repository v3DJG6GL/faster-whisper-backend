"""scripts/sync_venv.py: a stale test venv is synced and the run restarted."""

from __future__ import annotations

import importlib.util
import os
import sys
from importlib import metadata

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sv(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "sync_venv", os.path.join(REPO_ROOT, "scripts", "sync_venv.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k in ("CI", "GITHUB_ACTIONS", "FORGEJO_ACTIONS", mod.SYNCED_ENV, mod.OPT_OUT_ENV):
        monkeypatch.delenv(k, raising=False)
    return mod


def _versions(monkeypatch, installed, requires=None):
    def version(name):
        if name not in installed:
            raise metadata.PackageNotFoundError(name)
        return installed[name]
    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(metadata, "requires", lambda name: (requires or {}).get(name))


def _files(tmp_path, **files):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")


def test_requirements_follow_includes_and_skip_options(sv, tmp_path):
    _files(tmp_path,
           **{"a.txt": "# c\n-r b.txt\n--extra-index-url https://x\nfoo>=1  # why\n",
              "b.txt": "bar<2\n-r a.txt\n"})
    assert [str(r) for r in sv.requirements(("a.txt",), str(tmp_path))] == ["bar<2", "foo>=1"]


def test_the_real_files_name_every_floor(sv):
    names = {r.name for r in sv.requirements()}
    # requirements-heavy.txt comes in through requirements.txt's -r.
    assert {"imageio-ffmpeg", "faster-whisper", "fastapi", "pytest", "ruff"} <= names


def test_stale_reports_low_missing_and_extra_deps(sv, tmp_path, monkeypatch):
    _files(tmp_path, **{"r.txt": "low>=2\ngone\nok>=1\nwin; sys_platform == 'nope'\nsrv[std]>=1\n"})
    _versions(monkeypatch, {"low": "1.5", "ok": "1.0", "srv": "1.0", "loop": "0.1"},
              {"srv": ["loop>=0.2; extra == 'std'", "plain>=9", "other; extra == 'x'"]})
    assert sv.stale(("r.txt",), str(tmp_path)) == [
        "low 1.5: wants >=2",
        "gone: not installed (wants any version)",
        "loop 0.1: wants >=0.2 (via srv[std])",
    ]


def test_a_matching_venv_runs_on(sv, monkeypatch):
    monkeypatch.setattr(sv, "stale", lambda: [])
    monkeypatch.setattr(sv.subprocess, "call", pytest.fail)
    assert sv.ensure() is None


def _stale_venv(sv, monkeypatch):
    calls = []
    monkeypatch.setattr(sv, "stale", lambda: ["pkg 1: wants >=2"])
    monkeypatch.setattr(sv, "install_cmd", lambda: ["uv", "pip", "install"])
    monkeypatch.setattr(sv.sys, "prefix", "/venv")
    monkeypatch.setattr(sv.sys, "base_prefix", "/usr")
    monkeypatch.setattr(sv.sys, "platform", "linux")
    monkeypatch.setattr(sv.subprocess, "call", lambda cmd, cwd: calls.append(("install", cmd)) or 0)
    monkeypatch.setattr(sv.os, "execv", lambda exe, argv: calls.append(("exec", exe, argv)))
    monkeypatch.setattr(sv.sys, "orig_argv", ["/venv/bin/python", "-m", "pytest", "-q"])
    monkeypatch.setattr(sv.sys, "executable", "/venv/bin/python")
    return calls


def test_a_stale_venv_is_synced_and_the_run_restarted(sv, monkeypatch):
    calls = _stale_venv(sv, monkeypatch)
    sv.ensure()
    assert calls == [("install", ["uv", "pip", "install"]),
                     ("exec", "/venv/bin/python", ["/venv/bin/python", "-m", "pytest", "-q"])]
    assert os.environ[sv.SYNCED_ENV] == "1"
    monkeypatch.delenv(sv.SYNCED_ENV)


@pytest.mark.parametrize("why", ["ci", "no-venv", "restarted"])
def test_ci_a_system_python_or_a_second_try_only_report(sv, monkeypatch, why):
    calls = _stale_venv(sv, monkeypatch)
    if why == "ci":
        monkeypatch.setenv("CI", "true")
    elif why == "no-venv":
        monkeypatch.setattr(sv.sys, "base_prefix", "/venv")
    else:
        monkeypatch.setenv(sv.SYNCED_ENV, "1")
    error = sv.ensure()
    assert calls == []
    assert "pkg 1: wants >=2" in error and "uv pip install" in error


def test_a_failed_install_ends_the_run(sv, monkeypatch):
    calls = _stale_venv(sv, monkeypatch)
    monkeypatch.setattr(sv.subprocess, "call", lambda cmd, cwd: calls.append("install") or 1)
    assert "install failed" in sv.ensure()
    assert calls == ["install"]


def test_windows_syncs_but_asks_for_a_rerun(sv, monkeypatch):
    calls = _stale_venv(sv, monkeypatch)
    monkeypatch.setattr(sv.sys, "platform", "win32")
    assert "run the command again" in sv.ensure()
    assert [c[0] for c in calls] == ["install"]
    monkeypatch.delenv(sv.SYNCED_ENV)


def test_the_opt_out_reports_and_runs_on(sv, monkeypatch, capsys):
    calls = _stale_venv(sv, monkeypatch)
    monkeypatch.setenv(sv.OPT_OUT_ENV, "1")
    assert sv.ensure() is None
    assert calls == [] and "pkg 1: wants >=2" in capsys.readouterr().err


def test_install_prefers_uv_and_targets_this_interpreter(sv, monkeypatch):
    monkeypatch.setattr(sv.shutil, "which", lambda name: "/bin/uv")
    assert sv.install_cmd() == ["/bin/uv", "pip", "install", "--python", sys.executable,
                                "--upgrade", "-r", "requirements.txt", "-r", "requirements-dev.txt"]
    monkeypatch.setattr(sv.shutil, "which", lambda name: None)
    assert sv.install_cmd()[:3] == [sys.executable, "-m", "pip"]


def test_conftest_checks_before_the_workers_start():
    import tests.conftest as conftest

    class Worker:
        workerinput = {}

    conftest.pytest_configure(Worker())  # a worker returns before any check
    src = open(os.path.join(REPO_ROOT, "tests", "conftest.py"), encoding="utf-8").read()
    assert "_load_sync_venv().ensure()" in src
