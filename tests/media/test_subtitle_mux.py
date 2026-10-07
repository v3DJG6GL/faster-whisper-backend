"""media/subtitle_mux.py — subtitle packaging: the MP4 rule,
the ffmpeg argv, the runner against a fake ffmpeg, and the capability probe."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from faster_whisper_backend.media import subtitle_mux as pk


def _run(coro):
    return asyncio.run(coro)


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


def _disp(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a.startswith("-disposition:s:")]


def _flagged_tracks():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nHi\n"
    return [pk.SubtitleTrack("de", "Whisper", srt, default=True, original=True),
            pk.SubtitleTrack("de", "Site", srt, original=True, hearing_impaired=True),
            pk.SubtitleTrack("en", "Translation", srt, default=True),
            pk.SubtitleTrack("en", "Auto", srt)]


def test_build_package_argv_per_track_flags():
    # Several originals, one default per language, a hearing-impaired track.
    paths = [f"/w/sub_{i}.srt" for i in range(4)]
    argv = pk.build_package_argv("/m/src.mkv", paths, _flagged_tracks(), container="mkv",
                                 out_path="/w/out.mkv", default_track=None)
    assert _disp(argv) == ["default+original", "original+hearing_impaired", "default", "0"]
    assert not any(a.startswith("handler_name=") for a in argv)
    # The legacy indices OR into the per-track flags.
    argv = pk.build_package_argv("/m/src.mkv", paths, _flagged_tracks(), container="mkv",
                                 out_path="/w/out.mkv", default_track=1, original_track=3)
    assert _disp(argv) == ["default+original", "default+original+hearing_impaired",
                           "default", "original"]


def test_build_package_argv_mp4_handler_name_and_no_original():
    # MP4 has no FlagOriginal: it is left out; default + HI stay. The hdlr
    # name carries the title (movenc writes no `title` for mov_text).
    paths = [f"/w/sub_{i}.srt" for i in range(4)]
    argv = pk.build_package_argv("/m/src.mp4", paths, _flagged_tracks(), container="mp4",
                                 out_path="/w/out.mp4", default_track=None, original_track=3)
    assert _disp(argv) == ["default", "hearing_impaired", "default", "0"]
    h = [argv[i + 1] for i, a in enumerate(argv)
         if a.startswith("-metadata:s:s:") and argv[i + 1].startswith("handler_name=")]
    assert h == ["handler_name=Whisper", "handler_name=Site",
                 "handler_name=Translation", "handler_name=Auto"]


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
        assert os.path.basename(workdir).startswith("fwb-pkg-")
        with open(os.path.join(workdir, "sub_0.srt"), "rb") as f:
            assert b"\r" not in f.read()   # normalised line endings
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_package_timeout_kills_the_child_and_cleans_the_workdir(monkeypatch):
    _patch_argv(monkeypatch, "import time; time.sleep(30)")
    before = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("fwb-pkg-")}
    with pytest.raises(pk.PackageTimeout):
        _run(pk.package("/x", [], container="mkv", default_track=None, timeout=0.5))
    after = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("fwb-pkg-")}
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
    # The bare marker with NO subtitle track is the source, not "track 1".
    _patch_argv(monkeypatch, 'import sys; sys.stderr.write("src.mp4: Invalid data found when processing input\\n"); sys.exit(1)')
    with pytest.raises(pk.PackageError) as ei:
        _run(pk.package("/x", [], container="mkv", default_track=None, timeout=10))
    assert not isinstance(ei.value, pk.SubtitleParseError)
    assert str(ei.value) == "packaging failed"
    # With tracks, a line naming the source still blames the source.
    for line in ("clip.mkv: Invalid data found when processing input",
                 "[in#0/matroska,webm @ 0x1] Error during demuxing: Invalid data found"):
        _patch_argv(monkeypatch, f'import sys; sys.stderr.write({line!r}); sys.exit(1)')
        with pytest.raises(pk.PackageError) as ei:
            _run(pk.package("/m/clip.mkv", _tracks(), container="mkv",
                            default_track=None, timeout=10))
        assert not isinstance(ei.value, pk.SubtitleParseError), line


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


def test_a_timed_out_ffmpeg_probe_is_not_cached(monkeypatch):
    """One slow first call (cold disk, loaded host at startup) must not
    disable packaging — or the empty-CC strip — until a restart."""
    pk._reset_for_tests()
    outs = {"-muxers": "  E  matroska        Matroska\n",
            "-encoders": " S..... srt   SubRip\n",
            "-version": "ffmpeg version 7.0.2\n",
            "-bsfs": "filter_units\nnull\n"}
    slow = {"left": 1}

    class _R:
        def __init__(self, stdout):
            self.stdout = stdout

    def _run(argv, **kw):
        if slow["left"]:
            slow["left"] -= 1
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        return _R(outs.get(argv[-1], ""))
    monkeypatch.setattr(subprocess, "run", _run)
    caps = pk.ffmpeg_capabilities()
    assert not caps.available and "no ffmpeg" not in caps.reason
    # Asked again — on a background thread, so this call still answers
    # "not in time"; the answer is cached once that probe lands.
    assert not pk.ffmpeg_capabilities().available
    pk._caps_state["thread"].join(timeout=5)
    assert pk.ffmpeg_capabilities().available       # cached now
    slow["left"] = 1
    assert pk.ffmpeg_capabilities().available       # the cache held
    slow["left"] = 1
    assert pk.ffmpeg_has_bsf("filter_units") is False
    assert pk.ffmpeg_has_bsf("filter_units") is True
    pk._reset_for_tests()


def test_a_retried_ffmpeg_probe_is_single_flight_and_off_the_caller(monkeypatch):
    """After a timed-out first probe, request handlers call this on the event
    loop: a re-probe there froze the server for up to 3 x 15 s per call. The
    retry runs on its own thread, one at a time; callers get the fallback."""
    import threading
    pk._reset_for_tests()
    monkeypatch.setattr(pk, "_probe_ffmpeg_capabilities", lambda: None)
    assert not pk.ffmpeg_capabilities().available     # the first, timed out
    gate = threading.Event()
    probes = []
    ok = pk.FfmpegCaps(True, True, True, None, "7")

    def _blocking():
        probes.append(threading.current_thread())
        gate.wait(5)
        return ok
    monkeypatch.setattr(pk, "_probe_ffmpeg_capabilities", _blocking)
    try:
        first = pk.ffmpeg_capabilities()
        second = pk.ffmpeg_capabilities()
        assert not first.available and not second.available
        assert "did not answer in time" in second.reason
        for _ in range(100):
            if probes:
                break
            threading.Event().wait(0.01)
        assert len(probes) == 1                       # single-flight
        assert probes[0] is not threading.current_thread()
    finally:
        gate.set()
    pk._caps_state["thread"].join(timeout=5)
    assert pk.ffmpeg_capabilities() == ok
    assert len(probes) == 1
    pk._reset_for_tests()


def test_a_spawn_failure_is_not_cached_but_a_missing_binary_is(monkeypatch):
    import errno
    pk._reset_for_tests()

    def _eagain(argv, **kw):
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
    monkeypatch.setattr(subprocess, "run", _eagain)
    assert pk._probe_ffmpeg_capabilities() is None
    assert pk.ffmpeg_has_bsf("filter_units") is False
    assert pk._bsf_cache == {}

    def _missing(argv, **kw):
        raise FileNotFoundError(argv[0])
    monkeypatch.setattr(subprocess, "run", _missing)
    caps = pk.ffmpeg_capabilities()
    assert not caps.available and "no ffmpeg" in caps.reason
    assert pk._caps_cache == [caps]
    assert pk.ffmpeg_has_bsf("filter_units") is False
    assert pk._bsf_cache == {"filter_units": False}
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


@pytest.mark.skipif(shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None,
                    reason="needs a system ffmpeg + ffprobe")
def test_package_real_ffmpeg_writes_per_track_dispositions(tmp_path):
    """The flags survive the real mux: ffprobe reads them back (MKV keeps
    FlagOriginal; MP4 keeps default + hearing_impaired and the hdlr name)."""
    import json
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", src], check=True, timeout=60)

    def _subs(path):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s",
                            "-show_streams", "-of", "json", path],
                           capture_output=True, text=True, check=True, timeout=60)
        return json.loads(r.stdout)["streams"]

    for container in ("mkv", "mp4"):
        out = _run(pk.package(src, _flagged_tracks(), container=container,
                              default_track=None, timeout=60))
        try:
            subs = _subs(out)
            d = [(s["disposition"]["default"], s["disposition"]["original"],
                  s["disposition"]["hearing_impaired"]) for s in subs]
            if container == "mkv":
                assert d == [(1, 1, 0), (0, 1, 1), (1, 0, 0), (0, 0, 0)]
            else:
                assert d == [(1, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, 0)]
                assert [s["tags"].get("handler_name") for s in subs] == [
                    "Whisper", "Site", "Translation", "Auto"]
        finally:
            shutil.rmtree(os.path.dirname(out), ignore_errors=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a system ffmpeg")
def test_probe_streams_reads_a_non_utf8_title(tmp_path):
    """An AVI whose RIFF INFO title is cp1252 ("Müller"), passed through raw
    by ffmpeg: PyAV's default metadata_errors="strict" raised
    UnicodeDecodeError and the route cached a playable video as unreadable."""
    pytest.importorskip("av")
    src = str(tmp_path / "src.avi")
    subprocess.run([b"ffmpeg", b"-hide_banner", b"-loglevel", b"error", b"-y",
                    b"-f", b"lavfi", b"-i", b"testsrc=size=64x64:rate=10:duration=1",
                    b"-c:v", b"mpeg4", b"-metadata", b"title=M\xfcller",
                    src.encode()], check=True, timeout=60)
    streams = pk.probe_streams(src)
    assert streams.video_codec == "mpeg4" and streams.width == 64


def test_build_package_argv_maps_the_probed_video_index():
    argv = pk.build_package_argv("/m/src.mkv", [], [], container="mkv",
                                 out_path="/w/out.mkv", default_track=None,
                                 video_index=1)
    assert argv[argv.index("-map") + 1] == "0:v:1"


def test_unreadable_streams_is_flagged_not_no_video():
    facts = pk.unreadable_streams().as_dict()
    assert facts["unreadable"] is True and facts["cover_art_only"] is False
    assert facts["video_codec"] is None and facts["mp4_ok"] is False
    assert facts["mp4_reason"] == pk.UNREADABLE_REASON
    # The positional shape (the seven original fields) keeps working; the
    # flags default off.
    legacy = pk.MediaStreams(None, None, None, None, None, False, "x")
    assert legacy.unreadable is False and legacy.video_index == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a system ffmpeg")
def test_probe_cover_art_only_file_has_no_video(tmp_path):
    """An m4a with embedded artwork: the attached picture is not a video
    stream, so the route's no_video guard fires instead of an MJPEG verdict."""
    pytest.importorskip("av")
    cover = str(tmp_path / "cover.jpg")
    src = str(tmp_path / "song.m4a")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=64x64:rate=1:duration=1",
                    "-frames:v", "1", cover], check=True, timeout=60)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-i", cover, "-map", "0:a", "-map", "1:v", "-c:a", "aac",
                    "-c:v", "mjpeg", "-disposition:v:0", "attached_pic", src],
                   check=True, timeout=60)
    streams = pk.probe_streams(src)
    assert streams.video_codec is None and streams.width is None
    assert streams.cover_art_only is True and streams.unreadable is False
    assert streams.audio_codec == "aac"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a system ffmpeg")
def test_probe_mp4_verdict_covers_every_audio_stream(tmp_path):
    """`-map 0:a?` muxes every audio track, so a second track MP4 can't
    carry blocks MP4 even when the first one is AAC."""
    pytest.importorskip("av")
    src = str(tmp_path / "dual.mkv")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
                    "-map", "0:v", "-map", "1:a", "-map", "2:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a:0", "aac", "-c:a:1", "flac", "-shortest", src],
                   check=True, timeout=60)
    streams = pk.probe_streams(src)
    assert streams.video_codec == "h264" and streams.audio_codec == "aac"
    assert streams.mp4_ok is False and "FLAC" in streams.mp4_reason
    assert streams.video_index == 0 and streams.cover_art_only is False


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a system ffmpeg")
def test_probe_names_mp3_by_codec_not_decoder(tmp_path):
    """PyAV decodes MP3 with "mp3float": the probe must report the codec
    ("mp3", on the MP4 allow-list), or every H.264+MP3 source greys out MP4."""
    pytest.importorskip("av")
    src = str(tmp_path / "t.mp4")
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "libmp3lame", "-shortest", src],
                       check=True, timeout=60)
    except subprocess.CalledProcessError:
        pytest.skip("this ffmpeg has no libx264/libmp3lame")
    streams = pk.probe_streams(src)
    assert streams.video_codec == "h264" and streams.audio_codec == "mp3"
    assert streams.mp4_ok is True and streams.mp4_reason is None


def test_package_forwards_a_non_zero_video_index(monkeypatch):
    seen = {}

    def _fake(src, srt_paths, tracks, *, container, out_path, default_track, **kw):
        seen.update(kw)
        return [sys.executable, "-c", _OK.replace("__OUT__", out_path)]
    monkeypatch.setattr(pk, "build_package_argv", _fake)
    out = _run(pk.package("/x", [], container="mkv", default_track=None,
                          timeout=10, video_index=1))
    shutil.rmtree(os.path.dirname(out), ignore_errors=True)
    assert seen["video_index"] == 1
