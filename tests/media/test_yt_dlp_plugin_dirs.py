"""yt-dlp's default plugin dirs stay closed for the link check.

A YoutubeDL built through the API imports whatever sits in yt-dlp's default
plugin dirs: several under the home dir, /etc, and every sys.path entry. The
download child is started with --no-plugin-dirs; the link check (in a link
helper, or in-process when none was started) closes them through the API
before it builds its YoutubeDL. The SSRF guard does not come through plugin
loading (download.guard_self_check loads it by path), so it stays installed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from asyncio import run as _run

import pytest

from faster_whisper_backend.media import download as udl
from faster_whisper_backend.paths import REPO_ROOT

yt_dlp = pytest.importorskip("yt_dlp")

# Under the home dir, where yt-dlp looks for plugin packages by default.
_HOME_PLUGIN_ROOTS = ("yt-dlp-plugins", ".yt-dlp/plugins", ".config/yt-dlp/plugins",
                      ".config/yt-dlp-plugins")

# The link check as a link helper runs it (python -P, the repo root at the
# end of the import path), offline: the policy check and the extraction are
# stand-ins, the YoutubeDL and its plugin loading are yt-dlp's own.
_CHILD = r"""
import asyncio, json, sys, time
sys.path.append(sys.argv[1])
sys.path.append(sys.argv[2])
import yt_dlp
from faster_whisper_backend.media import download as udl

seen = {}

def _extract_info(self, url, download=False, **kw):
    seen["handlers"] = sorted(
        h.RH_NAME for h in self._request_director.handlers.values())
    return {"title": "t", "extractor_key": "Youtube"}

async def _policy(url, *, deadline=None):
    return "Youtube"

yt_dlp.YoutubeDL.extract_info = _extract_info
udl.check_url_policy = _policy
key, info, _target = asyncio.run(udl._probe_info(
    "https://www.youtube.com/watch?v=x", time.monotonic() + 60))
print(json.dumps({"key": key, "title": info.get("title"),
                  "handlers": seen["handlers"],
                  "installed": udl._guard_module.is_installed()}))
"""


def _plant(package_root, markers, name):
    """A plugin package whose import leaves a marker file."""
    plugin = package_root / "yt_dlp_plugins" / "extractor"
    plugin.mkdir(parents=True)
    (plugin / f"fwb_test_{name}.py").write_text(
        "import os\n"
        f"open(os.path.join({str(markers)!r}, {name!r}), 'w').close()\n",
        encoding="utf-8")


def test_the_link_check_imports_no_plugin_from_the_default_dirs(tmp_path):
    home, markers, on_path = tmp_path / "home", tmp_path / "markers", tmp_path / "path"
    markers.mkdir()
    for i, root in enumerate(_HOME_PLUGIN_ROOTS):
        _plant(home / root / "p", markers, f"home{i}")
    _plant(on_path, markers, "syspath")
    env = {k: v for k, v in os.environ.items() if k != "XDG_CONFIG_HOME"}
    env.update(HOME=str(home), USERPROFILE=str(home), APPDATA=str(home / ".config"),
               PYTHONNOUSERSITE="1")
    proc = subprocess.run(
        [sys.executable, "-P", "-c", _CHILD, REPO_ROOT, str(on_path)],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    said = json.loads(proc.stdout.strip().splitlines()[-1])
    assert sorted(os.listdir(markers)) == []
    # The check itself ran, behind the guard and nothing else.
    assert said == {"key": "Youtube", "title": "t", "installed": True,
                    "handlers": ["fwb-guarded-urllib"]}


def test_the_guard_check_closes_the_plugin_dirs_and_keeps_the_guard(monkeypatch):
    monkeypatch.setattr(yt_dlp.globals.plugin_dirs, "value", ["default"])
    udl.guard_self_check(force=True)
    assert yt_dlp.globals.plugin_dirs.value == []
    assert udl._guard_module.is_installed()


def test_the_probe_closes_the_plugin_dirs_before_it_builds_its_yt_dlp(monkeypatch):
    """The guard check answers from its cache after the first link; dirs
    opened since (another user of yt-dlp in this process) are closed again."""
    udl.guard_self_check(force=True)
    monkeypatch.setattr(yt_dlp.globals.plugin_dirs, "value", ["default"])
    built = []
    init = yt_dlp.YoutubeDL.__init__

    def _init(self, *a, **kw):
        built.append(list(yt_dlp.globals.plugin_dirs.value))
        init(self, *a, **kw)

    async def _policy(url, *, deadline=None):
        return "Youtube"
    monkeypatch.setattr(yt_dlp.YoutubeDL, "__init__", _init)
    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info",
                        lambda self, url, download=False, **kw: {"title": "t"})
    monkeypatch.setattr(udl, "check_url_policy", _policy)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    info = _run(udl.probe("https://www.youtube.com/watch?v=x", timeout=30.0))
    assert info.title == "t"
    assert built == [[]]


def test_a_yt_dlp_whose_plugin_dirs_cannot_be_closed_stops_link_downloads(
        monkeypatch, caplog):
    """Fail closed, like the guard: a yt-dlp that moved the switch would
    load the default dirs again."""
    udl.guard_self_check(force=True)
    monkeypatch.setattr(udl, "_guard_ok", True)
    monkeypatch.delattr(yt_dlp.globals, "plugin_dirs")
    built = []

    async def _policy(url, *, deadline=None):
        return "Youtube"
    monkeypatch.setattr(yt_dlp.YoutubeDL, "__init__",
                        lambda self, *a, **kw: built.append(a))
    monkeypatch.setattr(udl, "check_url_policy", _policy)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    # The probe, with the guard check answering from its cache.
    with pytest.raises(udl.UrlDownloadError,
                       match="link downloads are unavailable on this server"):
        _run(udl.probe("https://www.youtube.com/watch?v=x", timeout=30.0))
    assert built == []
    assert udl.GUARD_UNAVAILABLE_MARKER in caplog.text
    # The guard check itself (the app's start, a link helper's start).
    with pytest.raises(udl.UrlDownloadError,
                       match="link downloads are unavailable on this server"):
        udl.guard_self_check(force=True)
    assert udl._guard_ok is False
    assert "REFUSING link downloads" in caplog.text
