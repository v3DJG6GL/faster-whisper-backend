"""media/captions.py — empty broadcast captions (A/53 in H.264 SEI): the
bitstream scanner on synthetic bytes, and the whole path on a generated
video that carries empty or real captions."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess

import pytest

from faster_whisper_backend.media import captions as cc
from faster_whisper_backend.media import subtitle_mux as pk

PAD = (0xFC, 0x80, 0x80)          # CEA-608 field 1, valid, padding
HI = (0xFC, 0xC8, 0xE9)           # CEA-608 "Hi" with odd parity
DTVCC = (0xFF, 0x02, 0x21)        # CEA-708 packet start with data


def _sei(*triples) -> bytes:
    """One SEI NAL (Annex B) holding an ATSC cc_data() block."""
    body = bytes([0xB5, 0x00, 0x31]) + b"GA94\x03" + bytes([0x40 | len(triples), 0xFF])
    body += b"".join(bytes(t) for t in triples) + b"\xff"
    return b"\x00\x00\x00\x01\x06" + bytes([4, len(body)]) + body + b"\x80"


def test_padding_only_is_empty():
    s = cc.scan_a53([_sei(PAD, PAD, PAD) * 50])
    assert s.present and s.empty and s.pairs == 150


def test_cea608_text_is_content():
    s = cc.scan_a53([_sei(PAD) * 10 + _sei(PAD, HI) + _sei(PAD) * 10])
    assert s.present and not s.empty and s.content_pairs == 1


def test_cea708_data_is_content():
    assert not cc.scan_a53([_sei(PAD) * 5 + _sei(DTVCC)]).empty


def test_invalid_pairs_are_ignored():
    # cc_valid = 0: whatever the bytes say, nothing is shown.
    s = cc.scan_a53([_sei((0xF8, 0xC8, 0xE9), PAD)])
    assert s.empty and s.pairs == 1


def test_no_captions_is_not_empty():
    s = cc.scan_a53([b"\x00\x00\x00\x01\x65" + os.urandom(4096)])
    assert not s.present and not s.empty


def test_markers_split_across_chunks():
    data = _sei(PAD) * 20 + _sei(HI) + _sei(PAD) * 20
    one_byte = [data[i:i + 1] for i in range(len(data))]
    assert cc.scan_a53(one_byte).content_pairs == 1
    for size in (3, 7, 64):
        chunks = [data[i:i + size] for i in range(0, len(data), size)]
        assert cc.scan_a53(chunks).content_pairs == 1, size


def test_emulation_prevention_bytes_are_removed():
    # An encoder escapes 00 00 0x (x <= 3) as 00 00 03 0x; left in, the 03
    # would be read as the next pair's marker and the "Hi" pair lost.
    raw = _sei((0xFC, 0x00, 0x00), HI)
    escaped = raw.replace(b"\xfc\x00\x00\xfc", b"\xfc\x00\x00\x03\xfc")
    assert escaped != raw
    s = cc.scan_a53([escaped])
    assert s.pairs == 2 and s.content_pairs == 1


def test_argv_strips_sei_only_when_asked():
    tracks = [pk.SubtitleTrack("en", "English", "1\n00:00:00,000 --> 00:00:01,000\nHi\n")]
    base = pk.build_package_argv("/m/src.mkv", ["/w/sub_0.srt"], tracks, container="mkv",
                                 out_path="/w/out.mkv", default_track=0)
    assert "filter_units=remove_types=6" not in base
    strip = pk.build_package_argv("/m/src.mkv", ["/w/sub_0.srt"], tracks, container="mkv",
                                  out_path="/w/out.mkv", default_track=0,
                                  strip_empty_captions=True)
    i = strip.index("-bsf:v")
    assert strip[i + 1] == "filter_units=remove_types=6"
    assert strip.index("-c:v") < i


def test_only_h264_is_scanned(monkeypatch):
    calls = []
    monkeypatch.setattr(cc, "embedded_captions_empty", lambda *a, **k: calls.append(a) or True)
    assert asyncio.run(pk._empty_captions_to_strip("/m/x.mkv", "hevc", 0, 60)) is False
    assert asyncio.run(pk._empty_captions_to_strip("/m/x.mkv", None, 0, 60)) is False
    assert calls == []


# ── the whole path on a generated video ────────────────────────────────────

def _ffmpeg():
    from faster_whisper_backend.audio.ffmpeg import ffmpeg_exe
    exe = ffmpeg_exe()
    try:
        enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, check=False, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("no ffmpeg")
    if "libx264" not in enc or not pk.ffmpeg_has_bsf("filter_units"):
        pytest.skip("ffmpeg without libx264 or filter_units")
    return exe


def _video_with(tmp_path, sei: bytes) -> str:
    """A 1 s H.264 clip whose every slice is preceded by `sei`, in MKV."""
    exe = _ffmpeg()
    raw = tmp_path / "clip.h264"
    subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                    "-t", "1", "-c:v", "libx264", "-bf", "0", "-x264-params", "slices=1",
                    "-f", "h264", str(raw)], check=True, timeout=60)
    data = raw.read_bytes()
    # Before each slice NAL (types 1 and 5) of every access unit.
    data = re.sub(rb"(\x00\x00\x00\x01|\x00\x00\x01)(?=[\x01\x21\x41\x61\x05\x25\x45\x65])",
                  lambda m: sei + m.group(1), data)
    raw.write_bytes(data)
    out = tmp_path / "clip.mkv"
    subprocess.run([exe, "-v", "error", "-f", "h264", "-r", "10", "-i", str(raw),
                    "-c", "copy", str(out)], check=True, timeout=60)
    return str(out)


def _ga94_count(path: str) -> int:
    from faster_whisper_backend.audio.ffmpeg import ffmpeg_exe
    out = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", path, "-map", "0:v:0", "-c", "copy",
                          "-bsf:v", "h264_mp4toannexb", "-f", "h264", "-"],
                         capture_output=True, check=False, timeout=60).stdout
    return out.count(b"GA94")


def test_generated_clip_with_empty_captions(tmp_path):
    src = _video_with(tmp_path, _sei(PAD, PAD))
    assert _ga94_count(src) >= 5
    assert cc.embedded_captions_empty(src) is True


def test_generated_clip_with_real_captions_keeps_them(tmp_path):
    src = _video_with(tmp_path, _sei(PAD, HI))
    assert cc.embedded_captions_empty(src) is False


def test_generated_clip_without_captions(tmp_path):
    src = _video_with(tmp_path, b"")
    assert cc.embedded_captions_empty(src) is False


def test_package_strips_empty_captions_end_to_end(tmp_path):
    src = _video_with(tmp_path, _sei(PAD, PAD))
    tracks = [pk.SubtitleTrack("en", "English", "1\n00:00:00,000 --> 00:00:00,500\nHi\n")]
    out = asyncio.run(pk.package(src, tracks, container="mkv", default_track=0,
                                 timeout=120, video_codec="h264"))
    try:
        assert _ga94_count(out) == 0
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_package_keeps_real_captions_end_to_end(tmp_path):
    src = _video_with(tmp_path, _sei(PAD, HI))
    tracks = [pk.SubtitleTrack("en", "English", "1\n00:00:00,000 --> 00:00:00,500\nHi\n")]
    out = asyncio.run(pk.package(src, tracks, container="mkv", default_track=0,
                                 timeout=120, video_codec="h264"))
    try:
        assert _ga94_count(out) >= 5
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)
