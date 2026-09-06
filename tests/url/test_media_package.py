"""url/package.py — subtitle packaging: the language table, the MP4 rule,
the ffmpeg argv, the runner against a fake ffmpeg, and the capability probe."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from faster_whisper_backend.url import package as pk


def _run(coro):
    return asyncio.run(coro)


def test_iso639_2t_terminology_codes():
    assert pk.iso639_2t("en") == "eng"
    assert pk.iso639_2t("de") == "deu"
    assert pk.iso639_2t("fr") == "fra"
    assert pk.iso639_2t("zh") == "zho"
    assert pk.iso639_2t("nl") == "nld"
    assert pk.iso639_2t("pt-BR") == "por"
    assert pk.iso639_2t("yue") == "yue"
    assert pk.iso639_2t("xx") == "und"
    assert pk.iso639_2t("") == "und"
    assert pk.iso639_2t(None) == "und"


def test_lang_name_keeps_region_in_the_title():
    assert pk.lang_name("de") == "German"
    assert pk.lang_name("pt-BR") == "Portuguese (BR)"
    assert pk.lang_name("") == "Unknown"


def test_mp4_compatibility_allowlist():
    assert pk.mp4_compatibility("h264", "aac") == (True, None)
    assert pk.mp4_compatibility("hevc", None) == (True, None)
    ok, why = pk.mp4_compatibility("vp9", "aac")
    assert not ok and "VP9" in why
    ok, why = pk.mp4_compatibility("h264", "opus")
    assert not ok and "OPUS" in why
    ok, why = pk.mp4_compatibility("av1", "opus")
    assert not ok and "AV1" in why


def _tracks():
    return [pk.SubtitleTrack("en", "English · original", "1\n00:00:00,000 --> 00:00:01,000\nHi\n"),
            pk.SubtitleTrack("de", "German", "1\n00:00:00,000 --> 00:00:01,000\nHallo\n")]


def test_build_package_argv_mkv():
    argv = pk.build_package_argv("/m/src.mkv", ["/w/sub_0.srt", "/w/sub_1.srt"], _tracks(),
                                 container="mkv", out_path="/w/out.mkv", default_track=0)
    # Every input runs under the file-only protocol whitelist.
    assert argv.count("-protocol_whitelist") == 3
    assert argv.index("-protocol_whitelist") < argv.index("-i")
    assert argv[argv.index("-map") + 1] == "0:v:0"
    assert "0:a?" in argv and "1:0" in argv and "2:0" in argv
    assert argv[argv.index("-c:s") + 1] == "srt"
    assert "language=eng" in argv and "language=deu" in argv
    assert "title=German" in argv
    d = [argv[i + 1] for i, a in enumerate(argv) if a.startswith("-disposition:s:")]
    assert d == ["default", "0"]
    assert argv[-3:] == ["-f", "matroska", "/w/out.mkv"]


def test_build_package_argv_original_flag_and_audio_language():
    # The original-language track carries the flag, NOT a "· original" name;
    # both flags combine with "+"; every audio stream gets the spoken language.
    argv = pk.build_package_argv("/m/src.mkv", ["/w/sub_0.srt", "/w/sub_1.srt"], _tracks(),
                                 container="mkv", out_path="/w/out.mkv", default_track=0,
                                 original_track=0, audio_lang="de", audio_label="German")
    d = [argv[i + 1] for i, a in enumerate(argv) if a.startswith("-disposition:s:")]
    assert d == ["default+original", "0"]
    a = [argv[i + 1] for i, x in enumerate(argv) if x == "-metadata:s:a"]
    assert a == ["language=deu", "title=German"]
    # original without default; audio label defaults to the language name.
    argv = pk.build_package_argv("/m/src.mkv", ["/w/sub_0.srt", "/w/sub_1.srt"], _tracks(),
                                 container="mkv", out_path="/w/out.mkv", default_track=1,
                                 original_track=0, audio_lang="pt-BR")
    d = [argv[i + 1] for i, a in enumerate(argv) if a.startswith("-disposition:s:")]
    assert d == ["original", "default"]
    a = [argv[i + 1] for i, x in enumerate(argv) if x == "-metadata:s:a"]
    assert a == ["language=por", "title=Portuguese (BR)"]
    # No audio language → no audio metadata at all.
    argv = pk.build_package_argv("/m/src.mkv", [], [], container="mkv",
                                 out_path="/w/out.mkv", default_track=None)
    assert "-metadata:s:a" not in argv


def test_build_package_argv_mp4_mov_text_faststart_and_no_default():
    argv = pk.build_package_argv("/m/src.mp4", ["/w/sub_0.srt"], _tracks()[:1],
                                 container="mp4", out_path="/w/out.mp4", default_track=None)
    assert argv[argv.index("-c:s") + 1] == "mov_text"
    assert "+faststart" in argv
    assert argv[-3:] == ["-f", "mp4", "/w/out.mp4"]
    assert argv[argv.index("-disposition:s:0") + 1] == "0"
    # An unknown container falls back to mkv rather than reaching ffmpeg.
    argv = pk.build_package_argv("/m/src.mp4", [], [], container="webm",
                                 out_path="/w/out.webm", default_track=None)
    assert argv[-2] == "matroska" and "-c:s" not in argv


def _patch_argv(monkeypatch, script: str):
    """Swap the ffmpeg argv for a python script (same idiom as the yt-dlp
    tests). __OUT__ / __SRT__ are replaced with the real paths."""
    def _fake(src, srt_paths, tracks, *, container, out_path, default_track, **kw):
        s = script.replace("__OUT__", out_path).replace("__SRT__", srt_paths[0] if srt_paths else "")
        return [sys.executable, "-c", s]
    monkeypatch.setattr(pk, "build_package_argv", _fake)


_OK = """
import os
open(r"__OUT__", "wb").write(b"x" * 64)
"""


def test_package_success_returns_output_in_a_pkg_workdir(monkeypatch):
    _patch_argv(monkeypatch, _OK)
    tracks = [pk.SubtitleTrack("en", "English", "1\r\n00:00:00,000 --> 00:00:01,000\r\nHi\r\n")]
    out = _run(pk.package("/nonexistent/src.mkv", tracks, container="mkv",
                          default_track=0, timeout=10))
    try:
        assert os.path.basename(out) == "out.mkv"
        workdir = os.path.dirname(out)
        assert os.path.basename(workdir).startswith("pkg-")
        with open(os.path.join(workdir, "sub_0.srt"), "rb") as f:
            assert b"\r" not in f.read()   # normalised line endings
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_package_timeout_kills_the_child_and_cleans_the_workdir(monkeypatch):
    _patch_argv(monkeypatch, "import time; time.sleep(30)")
    before = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("pkg-")}
    with pytest.raises(pk.PackageTimeout):
        _run(pk.package("/x", [], container="mkv", default_track=None, timeout=0.5))
    after = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("pkg-")}
    assert after <= before


def test_package_nonzero_exit_never_echoes_stderr_and_classifies_srt(monkeypatch):
    _patch_argv(monkeypatch, 'import sys; sys.stderr.write("/secret/path sub_1.srt: Invalid data found\\n"); sys.exit(1)')
    with pytest.raises(pk.SubtitleParseError) as ei:
        _run(pk.package("/x", _tracks(), container="mkv", default_track=None, timeout=10))
    assert "track 2" in str(ei.value)
    assert "/secret" not in str(ei.value)
    _patch_argv(monkeypatch, 'import sys; sys.stderr.write("/secret boom\\n"); sys.exit(1)')
    with pytest.raises(pk.PackageError) as ei:
        _run(pk.package("/x", [], container="mkv", default_track=None, timeout=10))
    assert str(ei.value) == "packaging failed"


def test_package_no_output_is_an_error(monkeypatch):
    _patch_argv(monkeypatch, "pass")
    with pytest.raises(pk.PackageError, match="no output"):
        _run(pk.package("/x", [], container="mkv", default_track=None, timeout=10))


def test_ffmpeg_capabilities_parses_listings(monkeypatch):
    pk._reset_for_tests()
    outs = {"-muxers": "  E  matroska        Matroska\n  E  mp4             MP4\n",
            "-encoders": " S..... srt   SubRip\n S..... mov_text  3GPP\n",
            "-version": "ffmpeg version 7.0.2-static Copyright\n"}

    class _R:
        def __init__(self, stdout):
            self.stdout = stdout

    def _fake_run(argv, **kw):
        return _R(outs.get(argv[-1], ""))
    monkeypatch.setattr(subprocess, "run", _fake_run)
    caps = pk.ffmpeg_capabilities()
    assert caps == pk.FfmpegCaps(True, True, True, None, "7.0.2-static")
    pk._reset_for_tests()
    outs["-encoders"] = " S..... srt   SubRip\n"
    caps = pk.ffmpeg_capabilities()
    assert caps.mkv and not caps.mp4 and caps.available
    pk._reset_for_tests()
    outs["-muxers"] = "  E  mp4             MP4\n"
    caps = pk.ffmpeg_capabilities()
    assert not caps.available and "matroska" in caps.reason
    pk._reset_for_tests()

    def _missing(argv, **kw):
        raise FileNotFoundError(argv[0])
    monkeypatch.setattr(subprocess, "run", _missing)
    caps = pk.ffmpeg_capabilities()
    assert not caps.available and "no ffmpeg" in caps.reason
    pk._reset_for_tests()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a system ffmpeg")
def test_package_real_ffmpeg_muxes_language_tags(tmp_path):
    """End to end with the real binary: a 1 s test clip, two SRTs, PyAV
    reads two subtitle streams back with the 639-2/T tags and titles."""
    av = pytest.importorskip("av")
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src],
                   check=True, timeout=60)
    out = _run(pk.package(src, _tracks(), container="mkv", default_track=1, timeout=60))
    try:
        with av.open(out) as c:
            subs = [s for s in c.streams if s.type == "subtitle"]
            assert [s.metadata.get("language") for s in subs] == ["eng", "deu"]
            assert [s.metadata.get("title") for s in subs] == ["English · original", "German"]
        streams = pk.probe_streams(out)
        assert streams.video_codec == "h264" and streams.audio_codec == "aac"
        assert streams.mp4_ok and streams.width == 64
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)
    out = _run(pk.package(src, _tracks()[:1], container="mp4", default_track=0, timeout=60))
    try:
        with av.open(out) as c:
            subs = [s for s in c.streams if s.type == "subtitle"]
            assert len(subs) == 1 and subs[0].metadata.get("language") == "eng"
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)
