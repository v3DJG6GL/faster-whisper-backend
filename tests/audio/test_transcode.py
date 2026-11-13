"""Tests for audio/transcode.py (PyAV in-process).

Real PyAV is installed (no stub), so the happy paths are true round-trips.
"""

import fractions
import os
import sys
import wave

import av
import numpy as np
import pytest

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from faster_whisper_backend.audio import transcode as audio_transcode
from tests.conftest import write_tracks, write_wav

RATE = 16000


def _write_src_wav(path, *, rate, nchannels, freq=440.0, dur_s=0.5):
    t = np.linspace(0, dur_s, int(rate * dur_s), endpoint=False)
    tone = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)
    if nchannels == 2:
        # interleave L/R identical
        frames = np.repeat(tone, 2).tobytes()
    else:
        frames = tone.tobytes()
    return write_wav(path, frames, nchannels=nchannels, rate=rate)


def _spy_av_open(monkeypatch, wrap=lambda container, kwargs: container):
    """Route av.open through a spy; return the list of (file, kwargs) it saw.
    `wrap` may replace the container handed back to the caller."""
    opens, real_open = [], av.open

    def spy_open(file, *args, **kwargs):
        opens.append((file, kwargs))
        return wrap(real_open(file, *args, **kwargs), kwargs)
    monkeypatch.setattr(av, "open", spy_open)
    return opens


@pytest.mark.parametrize("in_rate,in_ch", [
    (16000, 1),
    (44100, 1),
    (44100, 2),
    (8000, 1),
])
def test_transcode_roundtrip(in_rate, in_ch, tmp_path):
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=in_rate, nchannels=in_ch)
    dst = str(tmp_path / "out.wav")
    n_bytes = audio_transcode.transcode_to_wav_16k_mono(src, dst)

    assert n_bytes > 0
    assert n_bytes == os.path.getsize(dst)
    with wave.open(dst, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == RATE
        assert w.getnframes() > 0


def test_transcode_pyav_missing(tmp_path, monkeypatch):
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    dst = str(tmp_path / "out.wav")
    # `import av` resolves None in sys.modules -> ImportError -> RuntimeError.
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(RuntimeError, match="PyAV"):
        audio_transcode.transcode_to_wav_16k_mono(src, dst)


def _write_video_only(path):
    c = av.open(path, "w")
    st = c.add_stream("mpeg4", rate=5)
    st.width = 32
    st.height = 32
    st.pix_fmt = "yuv420p"
    for _ in range(5):
        frame = av.VideoFrame.from_ndarray(
            np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
        for pkt in st.encode(frame):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    return path


def test_transcode_no_audio_stream(tmp_path):
    src = _write_video_only(str(tmp_path / "videoonly.mp4"))
    dst = str(tmp_path / "out.wav")
    with pytest.raises(ValueError, match="no audio stream"):
        audio_transcode.transcode_to_wav_16k_mono(src, dst)
    assert not os.path.exists(dst)


def test_transcode_unreadable_input_cleans_up(tmp_path):
    bad = tmp_path / "garbage.bin"
    bad.write_bytes(b"this is definitely not audio data" * 100)
    dst = str(tmp_path / "out.wav")
    with pytest.raises(Exception):
        audio_transcode.transcode_to_wav_16k_mono(str(bad), dst)
    assert not os.path.exists(dst)


def test_transcode_mid_stream_failure_unlinks_dst(tmp_path, monkeypatch):
    """A failure AFTER the output container is open (the tests above all
    fail before dst exists) closes the output and unlinks the partial dst."""
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    dst = str(tmp_path / "out.wav")
    real = audio_transcode._decoded_frames

    def _one_then_boom(container, stream):
        for frame in real(container, stream):
            yield frame
            break
        assert os.path.exists(dst)
        raise RuntimeError("boom")
    monkeypatch.setattr(audio_transcode, "_decoded_frames", _one_then_boom)
    with pytest.raises(RuntimeError, match="boom"):
        audio_transcode.transcode_to_wav_16k_mono(src, dst)
    assert not os.path.exists(dst)


def _with_info_title(path, title: bytes):
    """Append a RIFF LIST/INFO/INAM chunk carrying raw `title` bytes (a
    Windows-made WAV with a cp1252 title looks like this)."""
    import struct
    data = bytearray(open(path, "rb").read())
    sub = b"INAM" + struct.pack("<I", len(title)) + title
    if len(title) % 2:
        sub += b"\x00"
    chunk = b"LIST" + struct.pack("<I", 4 + len(sub)) + b"INFO" + sub
    data += chunk
    data[4:8] = struct.pack("<I", len(data) - 8)
    with open(path, "wb") as f:
        f.write(bytes(data))
    return path


def test_non_utf8_riff_info_tag_decodes(tmp_path):
    """PyAV's default metadata_errors="strict" raises UnicodeDecodeError (a
    ValueError, which the upload route reads as a refusal) on a non-UTF-8
    tag; faster-whisper decodes the same file fine."""
    src = _with_info_title(
        _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1),
        b"\xff\xfe\xfa\x00")
    assert audio_transcode.refuse_multi_input(src) is None
    dst = str(tmp_path / "out.wav")
    assert audio_transcode.transcode_to_wav_16k_mono(src, dst) > 44
    assert len(audio_transcode.decode_pieces_16k(src, [0.0], 0.25)[0]) > 0


def test_transcode_close_failure_raises_and_unlinks_dst(tmp_path, monkeypatch):
    """Closing the output writes the WAV trailer and flushes the buffered
    data (all of it for a short clip); a write error there (ENOSPC) must fail
    the transcode and unlink dst, not return a truncated file as success."""
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    dst = str(tmp_path / "out.wav")

    class _FailingClose:
        def __init__(self, container):
            self._c = container

        def __getattr__(self, name):
            return getattr(self._c, name)

        def close(self):
            self._c.close()
            raise OSError(28, "No space left on device")

    _spy_av_open(monkeypatch, lambda c, kw: _FailingClose(c) if kw.get("mode") == "w" else c)

    with pytest.raises(OSError, match="No space left"):
        audio_transcode.transcode_to_wav_16k_mono(src, dst)
    assert not os.path.exists(dst)


def test_transcode_input_open_pins_file_protocol_whitelist(tmp_path, monkeypatch):
    """A PyAV upgrade or an av.open refactor must not silently drop the
    ``protocol_whitelist: file`` option — it is what stops a crafted
    ffconcat/HLS/SDP input from following external file:// or http://
    references (test_ssrf_guard.py is the URL-side twin)."""
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    dst = str(tmp_path / "out.wav")
    opens = _spy_av_open(monkeypatch)

    audio_transcode.transcode_to_wav_16k_mono(src, dst)
    in_opens = [kw for f, kw in opens if f == src]
    assert len(in_opens) == 1
    assert in_opens[0].get("options") == {
        "protocol_whitelist": "file",
        "format_whitelist": audio_ffmpeg.single_input_demuxers(),
        "max_streams": "20",
        "codec_whitelist": audio_ffmpeg.pyav_audio_decoders()}
    assert in_opens[0].get("metadata_errors") == "ignore"


def test_transcode_rejects_ffconcat_playlist_referencing_a_file(tmp_path):
    """The whitelist alone does NOT stop this one: concat is picked by
    content, its nested open uses the whitelisted file protocol, and safe=1
    admits a bare same-directory name — so an extensionless upload naming a
    sibling decodes that sibling. The demuxer refusal is what fails it (and
    leaves no dst)."""
    _write_src_wav(str(tmp_path / "real.wav"), rate=RATE, nchannels=1)
    playlist = tmp_path / "upload"
    playlist.write_text("ffconcat version 1.0\nfile 'real.wav'\n")
    dst = str(tmp_path / "out.wav")
    with pytest.raises(ValueError, match="unsupported container"):
        audio_transcode.transcode_to_wav_16k_mono(str(playlist), dst)
    assert not os.path.exists(dst)


# decode_pieces_16k feeds the link language check.

def _two_tone(path, *, rate=44100, pts_offset_s=0):
    """60 s stereo: 440 Hz for the first 30 s, 880 Hz after — a piece's
    dominant frequency tells where it was cut from. `pts_offset_s` starts
    the stream's timestamps there (a non-WAV stream with start_time != 0)."""
    t = np.arange(rate * 60) / rate
    tone = np.where(t < 30, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t))
    pcm = np.repeat((tone * 8000).astype(np.int16), 2)
    if path.endswith(".wav"):
        return write_wav(path, pcm.tobytes(), nchannels=2, rate=rate)
    mp3 = path.endswith(".mp3")
    with av.open(path, "w") as out:
        st = out.add_stream("libmp3lame" if mp3 else "aac", rate=rate,
                            layout="stereo")
        step = 1152 if mp3 else 1024
        frames = pcm.reshape(-1, 2)
        for i in range(0, len(frames), step):
            fr = av.AudioFrame.from_ndarray(
                frames[i:i + step].reshape(1, -1), format="s16", layout="stereo")
            fr.sample_rate = rate
            if pts_offset_s:
                fr.pts = pts_offset_s * rate + i
                fr.time_base = fractions.Fraction(1, rate)
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    return path


def _peak_hz(piece):
    spec = np.abs(np.fft.rfft(piece))
    return np.argmax(spec) * RATE / len(piece)


@pytest.mark.parametrize("name", ["two.wav", "two.m4a"])
def test_decode_pieces_seeks_to_each_start(tmp_path, name):
    src = _two_tone(str(tmp_path / name))
    pieces = audio_transcode.decode_pieces_16k(src, [5.0, 40.0, 10.0, 70.0], 4.0)
    assert [p.dtype for p in pieces[:3]] == [np.float32] * 3
    assert [len(p) for p in pieces] == [4 * RATE] * 3 + [0]   # past the end
    assert [round(_peak_hz(p)) for p in pieces[:3]] == [440, 880, 440]


@pytest.mark.parametrize("name", ["off.ts", "off.m4a"])
def test_decode_pieces_measures_starts_from_the_stream_start(tmp_path, name):
    """`starts` are media seconds; a stream whose pts begin at ~10 s (an
    MPEG-TS download, a remux keeping the offset) must not be cut 10 s
    early — that would take the 35 s piece from media 25 s (440 Hz)."""
    src = _two_tone(str(tmp_path / name), pts_offset_s=10)
    pieces = audio_transcode.decode_pieces_16k(src, [5.0, 35.0, 55.0], 4.0)
    assert [len(p) for p in pieces] == [4 * RATE] * 3
    assert [round(_peak_hz(p)) for p in pieces] == [440, 880, 880]


def test_decode_pieces_refuses_non_file_protocols(tmp_path, monkeypatch):
    """decode_pieces_16k opens through the same pinned whitelist as the
    transcode path (an HLS playlist fails whatever the whitelist, so only
    a spy on av.open can pin it)."""
    src = _two_tone(str(tmp_path / "two.wav"))
    opens = _spy_av_open(monkeypatch)

    audio_transcode.decode_pieces_16k(src, [0.0], 1.0)
    assert [kw.get("options") for f, kw in opens if f == src] == [
        {"protocol_whitelist": "file",
         "format_whitelist": audio_ffmpeg.single_input_demuxers(),
         "max_streams": "20",
         "codec_whitelist": audio_ffmpeg.pyav_audio_decoders()}]


def test_decode_pieces_refuses_an_ffconcat_sibling(tmp_path):
    _two_tone(str(tmp_path / "two.wav"))
    playlist = tmp_path / "upload"
    playlist.write_text("ffconcat version 1.0\nfile 'two.wav'\n")
    with pytest.raises(ValueError, match="unsupported container"):
        audio_transcode.decode_pieces_16k(str(playlist), [0.0], 1.0)


# refuse_multi_input: the raw-upload check before a non-PyAV decoder.

def test_refuse_multi_input_refuses_an_ffconcat_sibling(tmp_path):
    _write_src_wav(str(tmp_path / "real.wav"), rate=RATE, nchannels=1)
    playlist = tmp_path / "whisperup-x.wav"
    playlist.write_text("ffconcat version 1.0\nfile 'real.wav'\n")
    with pytest.raises(ValueError, match="unsupported container"):
        audio_transcode.refuse_multi_input(str(playlist))


def test_refuse_multi_input_passes_a_clip_and_leaves_av_errors_to_the_decoder(
        tmp_path):
    clip = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    assert audio_transcode.refuse_multi_input(clip) is None
    # No audio stream is required either.
    assert audio_transcode.refuse_multi_input(
        _write_video_only(str(tmp_path / "v.mp4"))) is None
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not media at all\x00\x01")
    assert audio_transcode.refuse_multi_input(str(junk)) is None


def _sibling_states(tmp_path):
    """ffconcat uploads naming a readable, an absent and (unless root) an
    unreadable sibling. A playlist demuxer opens its first entry inside
    av.open, so the error would tell the three apart."""
    _write_src_wav(str(tmp_path / "present.wav"), rate=RATE, nchannels=1)
    names = ["present.wav", "absent.wav"]
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        locked = tmp_path / "locked.wav"
        _write_src_wav(str(locked), rate=RATE, nchannels=1)
        locked.chmod(0)
        names.append("locked.wav")
    out = []
    for i, name in enumerate(names):
        p = tmp_path / f"whisperup-{i}"
        p.write_text(f"ffconcat version 1.0\nfile '{name}'\n")
        out.append(str(p))
    return out


def test_open_audio_refuses_an_ffconcat_before_it_opens_the_sibling(tmp_path):
    """The demuxer allowlist refuses the playlist at probe time: the same
    "unsupported container" whether its entry exists, is missing or cannot
    be read — no existence oracle, no sibling open."""
    for playlist in _sibling_states(tmp_path):
        with pytest.raises(ValueError, match="unsupported container") as ei:
            audio_transcode.declared_shape(playlist)
        assert not isinstance(ei.value, av.FFmpegError)
        with pytest.raises(ValueError, match="unsupported container"):
            audio_transcode.refuse_multi_input(playlist)


def test_open_audio_refuses_a_vobsub_index_before_it_opens_the_sibling(
        tmp_path):
    """A VobSub .idx makes its demuxer open the sibling <name>.sub inside
    av.open; the allowlist refuses it at probe time, with or without that
    sibling."""
    assert "vobsub" not in audio_ffmpeg.single_input_demuxers().split(",")
    idx = tmp_path / "up.wav"
    idx.write_bytes(b"# VobSub index file, v7 (Do not modify this line!)\n"
                    b"size: 720x576\n")
    for sibling_present in (False, True):
        if sibling_present:
            (tmp_path / "up.sub").write_bytes(b"\x00" * 2048)
        with pytest.raises(ValueError, match="unsupported container") as ei:
            audio_transcode.declared_shape(str(idx))
        assert not isinstance(ei.value, av.FFmpegError)


def test_open_audio_pins_the_single_input_demuxer_allowlist(tmp_path,
                                                            monkeypatch):
    allowed = audio_ffmpeg.single_input_demuxers().split(",")
    assert {"wav", "mp3", "flac", "ogg", "matroska", "mov", "aac",
            "mpegts", "flv"} <= set(allowed)
    assert not audio_ffmpeg.MULTI_INPUT_DEMUXERS & set(allowed)
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    opens = _spy_av_open(monkeypatch)
    audio_transcode.declared_shape(src)
    audio_transcode.refuse_multi_input(src)
    assert [kw.get("options") for f, kw in opens if f == src] == [
        {"protocol_whitelist": "file",
         "format_whitelist": audio_ffmpeg.single_input_demuxers(),
         "max_streams": "20",
         "codec_whitelist": audio_ffmpeg.pyav_audio_decoders()}] * 2


def _damaged_mp3(path, *, seconds=8):
    """An 8 s 440 Hz mp3 with 400 random bytes in the middle — still
    decodable by faster-whisper, but one packet makes the decoder raise
    InvalidDataError (an av.FFmpegError, which main.py treats as a decode
    failure, never as a refused input)."""
    import random

    with av.open(path, "w", format="mp3") as out:
        st = out.add_stream("libmp3lame", rate=RATE)
        st.layout = "mono"
        t = np.arange(RATE * seconds) / RATE
        x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for i in range(0, len(x), 1152):
            fr = av.AudioFrame.from_ndarray(
                x[i:i + 1152].reshape(1, -1), format="flt", layout="mono")
            fr.sample_rate = RATE
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    data = bytearray(open(path, "rb").read())
    rnd = random.Random(0)
    mid = len(data) // 2
    for i in range(mid, mid + 400):
        data[i] = rnd.randrange(256)
    with open(path, "wb") as f:
        f.write(data)
    return path


def test_transcode_skips_a_damaged_frame(tmp_path):
    src = _damaged_mp3(str(tmp_path / "damaged.mp3"))
    dst = str(tmp_path / "out.wav")
    assert audio_transcode.transcode_to_wav_16k_mono(src, dst) > 44
    with wave.open(dst, "rb") as w:
        # Most of the 8 s survive: only the damaged packets are dropped.
        assert w.getnframes() > 6 * RATE


def test_decode_pieces_skip_a_damaged_frame(tmp_path):
    src = _damaged_mp3(str(tmp_path / "damaged.mp3"))
    pieces = audio_transcode.decode_pieces_16k(src, [0.0, 3.5, 6.0], 1.0)
    assert [len(p) for p in pieces] == [RATE] * 3


class _RejectedPacket:
    """A demuxed packet the decoder rejects as invalid data."""

    def __init__(self, size):
        self.size = size

    def decode(self):
        raise av.error.InvalidDataError(1094995529, "Invalid data found")


class _RejectingContainer:
    """Stands in for an opened container whose demuxer hands out `packets`
    rejected packets of `packet_size` bytes; `pulled` counts them."""

    def __init__(self, *, size, packets, packet_size, name="",
                 kind=_RejectedPacket):
        self.size, self.name, self.pulled = size, name, 0
        self._packets, self._packet_size = packets, packet_size
        self._kind = kind

    def demux(self, _stream):
        for _ in range(self._packets):
            self.pulled += 1
            yield self._kind(self._packet_size)


def test_decoded_frames_stop_once_the_rejected_packets_outweigh_the_file(
        tmp_path, monkeypatch):
    """Packets the decoder rejects are skipped only up to the file's own
    size plus a margin: the packets of a real file are disjoint parts of
    it, so more than that means a sample table that names one byte range
    again and again, each read in full and charged to no decode budget."""
    monkeypatch.setattr(audio_transcode, "_SKIPPED_PACKETS_MARGIN", 4096)
    container = _RejectingContainer(size=10_000, packets=5000,
                                    packet_size=1000)
    state = {}
    assert list(audio_transcode._decoded_frames(container, None, state)) == []
    # 10 000 + 4096 bytes are skipped; the packet that passes them ends it.
    assert container.pulled == 15
    assert state == {"cut": True}

    # Rejected packets that fit in the file are all skipped, as before.
    container = _RejectingContainer(size=10_000, packets=14, packet_size=1000)
    state = {}
    assert list(audio_transcode._decoded_frames(container, None, state)) == []
    assert container.pulled == 14 and state == {}

    # A container that reports no size is measured on disk.
    src = tmp_path / "ten-thousand.bin"
    src.write_bytes(b"\x00" * 10_000)
    container = _RejectingContainer(size=0, packets=5000, packet_size=1000,
                                    name=str(src))
    assert list(audio_transcode._decoded_frames(container, None)) == []
    assert container.pulled == 15


class _FramelessPacket:
    """A demuxed packet the decoder accepts and turns into no frame."""

    def __init__(self, size):
        self.size = size

    def decode(self):
        return []


def test_decoded_frames_charge_a_packet_that_decodes_to_no_frame(monkeypatch):
    """A packet the decoder takes without an error and without a frame is
    demuxed in full like a rejected one, and charged like one."""
    monkeypatch.setattr(audio_transcode, "_SKIPPED_PACKETS_MARGIN", 4096)
    container = _RejectingContainer(size=10_000, packets=5000,
                                    packet_size=1000, kind=_FramelessPacket)
    state = {}
    assert list(audio_transcode._decoded_frames(container, None, state)) == []
    assert container.pulled == 15
    assert state == {"cut": True}

    container = _RejectingContainer(size=10_000, packets=14, packet_size=1000,
                                    kind=_FramelessPacket)
    state = {}
    assert list(audio_transcode._decoded_frames(container, None, state)) == []
    assert container.pulled == 14 and state == {}


def _mp4_boxes(data):
    i = 0
    while i + 8 <= len(data):
        size = int.from_bytes(data[i:i + 4], "big")
        yield data[i + 4:i + 8], data[i + 8:i + size]
        i += size


def _mp4_box(kind, payload):
    return (len(payload) + 8).to_bytes(4, "big") + kind + payload


def _mp4_with_tables(data, tables):
    """The MP4 boxes of `data` with each sample-table box named in `tables`
    rewritten by its function, and without the edit list (the demuxer would
    trim the sample table to the edit's duration)."""
    out = b""
    for kind, payload in _mp4_boxes(data):
        if kind == b"edts":
            continue
        if kind in tables:
            payload = tables[kind](payload)
        elif kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
            payload = _mp4_with_tables(payload, tables)
        out += _mp4_box(kind, payload)
    return out


def _aliased_m4a(path, *, real, repeats, sample_size):
    """A mono AAC .m4a whose sample table keeps its first `real` samples and
    then names one byte range (inside the first frame, so no AAC frame)
    `repeats` times as a sample of `sample_size` bytes: four bytes of chunk
    offset per repeat, each one demuxed in full and rejected by the
    decoder."""
    rate = 44100
    with av.open(path, "w", format="mp4") as out:
        st = out.add_stream("aac", rate=rate, layout="mono")
        t = np.arange(rate * 2) / rate
        x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for i in range(0, len(x), 1024):
            fr = av.AudioFrame.from_ndarray(
                x[i:i + 1024].reshape(1, -1), format="fltp", layout="mono")
            fr.sample_rate, fr.pts = rate, i
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    data = open(path, "rb").read()
    found = {}

    def u32(*values):
        return b"".join(v.to_bytes(4, "big") for v in values)

    def stsz(p):
        # One track: its samples lie back to back from the first chunk on.
        sizes = [int.from_bytes(p[12 + 4 * i:16 + 4 * i], "big")
                 for i in range(real)]
        found["sizes"] = sizes
        return p[:4] + u32(0, real + repeats, *sizes,
                           *[sample_size] * repeats)

    def stco(p):
        first = int.from_bytes(p[8:12], "big")
        offsets = [first + sum(found["sizes"][:i]) for i in range(real)]
        return p[:4] + u32(real + repeats, *offsets, *[first + 3] * repeats)

    tables = {b"stsz": stsz, b"stco": stco,
              b"stsc": lambda p: p[:4] + u32(1, 1, 1, 1),
              b"stts": lambda p: p[:4] + u32(1, real + repeats, 1024)}
    boxes = list(_mp4_boxes(data))
    assert boxes[-1][0] == b"moov"      # so no chunk offset moves
    # stsz stands ahead of stco in the muxer's stbl, so `found` is filled.
    moov = _mp4_with_tables(boxes[-1][1], tables)
    with open(path, "wb") as f:
        f.write(b"".join(_mp4_box(k, p) for k, p in boxes[:-1])
                + _mp4_box(b"moov", moov))
    return path


class _CountingContainer:
    """An opened container that adds up the bytes its demuxer hands out."""

    def __init__(self, container):
        self._container, self.demuxed = container, 0

    def __getattr__(self, name):
        return getattr(self._container, name)

    def demux(self, *args, **kwargs):
        for packet in self._container.demux(*args, **kwargs):
            self.demuxed += packet.size
            yield packet

    def decode(self, *args, **kwargs):
        # As PyAV's own InputContainer.decode, over the counted demux.
        for packet in self.demux(*args, **kwargs):
            yield from packet.decode()


def _count_demuxed(monkeypatch):
    """Every container _open_audio opens from here on, wrapped to count."""
    opened, real = [], audio_transcode._open_audio

    def counting(path):
        container, stream = real(path)
        opened.append(_CountingContainer(container))
        return opened[-1], stream

    monkeypatch.setattr(audio_transcode, "_open_audio", counting)
    return opened


def test_a_sample_table_naming_one_range_many_times_is_not_read_in_full(
        tmp_path, monkeypatch):
    """Behind 0.9 s of real audio the file names one 8000-byte range 3000
    times: 24 MB of demuxing for a 43 KB file, quadratic in the upload
    size. Every reader that skips rejected packets stops after about one
    file size (plus the margin) of them and keeps what decoded; the cost
    measurement, which then has not seen the file's end, answers no."""
    real, repeats, sample_size = 40, 3000, 8000
    src = _aliased_m4a(str(tmp_path / "aliased.m4a"), real=real,
                       repeats=repeats, sample_size=sample_size)
    size = os.path.getsize(src)
    assert size < 50_000
    # The honest samples, then the rejected ones up to the one that passes
    # the limit.
    per_read = (2 * size + audio_transcode._SKIPPED_PACKETS_MARGIN
                + sample_size)
    assert per_read < repeats * sample_size // 10
    head = real * 1024 * RATE // 44100
    opened = _count_demuxed(monkeypatch)

    span = audio_transcode.decode_span_16k(src, 0.0, 20.0)
    assert abs(span.size - head) < 2048
    assert opened.pop().demuxed <= per_read

    # 0.0 lies in the real audio, 20.0 and 40.0 in the repeated range.
    pieces = audio_transcode.decode_pieces_16k(src, [0.0, 20.0, 40.0], 0.5)
    assert [p.size for p in pieces] == [RATE // 2, 0, 0]
    assert opened.pop().demuxed <= 3 * per_read

    dst = str(tmp_path / "out.wav")
    audio_transcode.transcode_to_wav_16k_mono(src, dst)
    with wave.open(dst, "rb") as w:
        assert abs(w.getnframes() - head) < 2048
    assert opened.pop().demuxed <= per_read

    assert audio_transcode.decoded_cost_within(
        src, cost_per_s=lambda rate, channels: 1.0, max_cost=1e9) is False
    assert opened.pop().demuxed <= per_read


def frameless_flac_mp4(path, *, packets, packet_size):
    """A FLAC-in-MP4 whose sample table names one `packet_size` range
    `packets` times, four bytes of chunk offset each. The range starts with
    the byte of a Vorbis-comment block, which FFmpeg's flac decoder takes as
    metadata: no error and no frame."""
    with av.open(path, "w", format="mp4") as out:
        st = out.add_stream("flac", rate=RATE, layout="mono")
        fr = av.AudioFrame.from_ndarray(np.zeros((1, 4096), np.int16),
                                        format="s16", layout="mono")
        fr.sample_rate, fr.pts = RATE, 0
        for pkt in [*st.encode(fr), *st.encode(None)]:
            out.mux(pkt)
    boxes = dict(_mp4_boxes(open(path, "rb").read()))
    ftyp = _mp4_box(b"ftyp", boxes[b"ftyp"])

    def u32(*values):
        return b"".join(v.to_bytes(4, "big") for v in values)

    moov = _mp4_with_tables(boxes[b"moov"], {
        b"stts": lambda p: p[:4] + u32(1, packets, 4096),
        b"stsc": lambda p: p[:4] + u32(1, 1, 1, 1),
        b"stsz": lambda p: p[:4] + u32(packet_size, packets),
        b"stco": lambda p: p[:4] + u32(packets, *[len(ftyp) + 8] * packets)})
    with open(path, "wb") as f:
        f.write(ftyp + _mp4_box(b"mdat", b"\x04" + bytes(packet_size - 1))
                + _mp4_box(b"moov", moov))
    return path


def test_packets_that_decode_to_no_frame_are_not_read_in_full(tmp_path,
                                                              monkeypatch):
    """200 sample-table entries naming one 64 KiB range the flac decoder
    reads as metadata: 13 MB of demuxing for a 67 KB file, quadratic in the
    upload size, with no frame for any budget to count and no error. The
    readers stop after about one file size (plus the margin) of them, like
    they do for rejected packets."""
    packets, packet_size = 200, 1 << 16
    monkeypatch.setattr(audio_transcode, "_SKIPPED_PACKETS_MARGIN", 4096)
    src = frameless_flac_mp4(str(tmp_path / "frameless.mp4"), packets=packets,
                             packet_size=packet_size)
    size = os.path.getsize(src)
    assert size < 70_000
    per_read = size + 4096 + packet_size
    assert per_read < packets * packet_size // 50
    opened = _count_demuxed(monkeypatch)

    container, stream = audio_transcode._open_audio(src)
    try:
        state = {}
        assert list(audio_transcode._decoded_frames(container, stream,
                                                    state)) == []
        assert state == {"cut": True}
    finally:
        container.close()
    assert opened.pop().demuxed <= per_read

    assert audio_transcode.decode_span_16k(src, 0.0, 20.0).size == 0
    assert opened.pop().demuxed <= per_read

    assert audio_transcode.decoded_cost_within(
        src, cost_per_s=lambda rate, channels: 1.0, max_cost=1e9) is False
    assert opened.pop().demuxed <= per_read


def _decoded_frames_skipping_everything(container, stream, rejected):
    """The loop without a limit on what it skips: the reference a damaged
    real file must still match frame for frame. `rejected` collects the
    sizes of the packets it skips."""
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            rejected.append(packet.size)
            continue
        yield from frames


def test_a_damaged_real_file_decodes_as_before(tmp_path, monkeypatch):
    """Whatever a real file's rejected packets add up to fits in the file,
    so the limit never ends its decode: with no margin at all, an mp3
    damaged in several places still gives every frame the unlimited loop
    gives, and its cost is still measured to the end."""
    monkeypatch.setattr(audio_transcode, "_SKIPPED_PACKETS_MARGIN", 0)
    src = _damaged_mp3(str(tmp_path / "damaged.mp3"))
    data = bytearray(open(src, "rb").read())
    for at in (len(data) // 5, len(data) // 3, 4 * len(data) // 5):
        data[at:at + 300] = bytes(range(256)) + bytes(44)
    with open(src, "wb") as f:
        f.write(data)

    def frames(reader, *args):
        container, stream = audio_transcode._open_audio(src)
        try:
            return [f.to_ndarray().copy() for f in reader(container, stream,
                                                          *args)]
        finally:
            container.close()

    state, rejected = {}, []
    got = frames(audio_transcode._decoded_frames, state)
    want = frames(_decoded_frames_skipping_everything, rejected)
    assert len(rejected) >= 4 and sum(rejected) < len(data)
    assert state == {} and sum(f.shape[1] for f in got) > 6 * RATE
    assert len(got) == len(want)
    assert all(np.array_equal(a, b) for a, b in zip(got, want))
    assert audio_transcode.decoded_cost_within(
        src, cost_per_s=lambda rate, channels: 1.0, max_cost=1e9) is True


def _concat_mp3(tmp_path):
    """2 s of 44.1 kHz stereo mp3 followed by 2 s of 22.05 kHz mono, as
    `cat a.mp3 b.mp3` makes it: the decoder's frames change rate and layout
    halfway, which one AudioResampler refuses with a plain ValueError."""
    parts = []
    for rate, layout in ((44100, "stereo"), (22050, "mono")):
        path = str(tmp_path / f"{rate}.mp3")
        with av.open(path, "w", format="mp3") as out:
            st = out.add_stream("libmp3lame", rate=rate)
            st.layout = layout
            ch = 2 if layout == "stereo" else 1
            t = np.arange(rate * 2) / rate
            x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            for i in range(0, len(x), 1152):
                fr = av.AudioFrame.from_ndarray(
                    np.tile(x[i:i + 1152], (ch, 1)), format="fltp",
                    layout=layout)
                fr.sample_rate = rate
                for pkt in st.encode(fr):
                    out.mux(pkt)
            for pkt in st.encode(None):
                out.mux(pkt)
        parts.append(open(path, "rb").read())
    src = tmp_path / "ab.mp3"
    src.write_bytes(b"".join(parts))
    return str(src)


def test_transcode_survives_a_mid_stream_format_change(tmp_path):
    src = _concat_mp3(tmp_path)
    dst = str(tmp_path / "out.wav")
    audio_transcode.transcode_to_wav(src, dst, rate=44100, layout="stereo")
    with wave.open(dst, "rb") as w:
        assert w.getnchannels() == 2
        # Both halves, not just the first one.
        assert w.getnframes() > 3.5 * 44100


def test_decode_span_survives_a_mid_stream_format_change(tmp_path):
    src = _concat_mp3(tmp_path)
    assert len(audio_transcode.decode_span_16k(src, 1.0, 2.0)) == 2 * RATE


def test_transcode_to_wav_max_bytes_aborts_and_unlinks(tmp_path):
    """A budget the output would pass aborts mid-decode (the BGM path hands
    in the free space of the temp dir) and leaves no partial WAV behind."""
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=16000, nchannels=1,
                         dur_s=10.0)
    dst = str(tmp_path / "out.wav")
    with pytest.raises(RuntimeError, match="budget"):
        audio_transcode.transcode_to_wav(src, dst, rate=44100,
                                         layout="stereo", max_bytes=100_000)
    assert not os.path.exists(dst)


def test_transcode_to_wav_max_bytes_none_or_roomy_is_unchanged(tmp_path):
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=16000, nchannels=1,
                         dur_s=2.0)
    plain = str(tmp_path / "plain.wav")
    n = audio_transcode.transcode_to_wav(src, plain, rate=44100,
                                         layout="stereo")
    for budget in (None, n):
        dst = str(tmp_path / f"out-{budget}.wav")
        assert audio_transcode.transcode_to_wav(
            src, dst, rate=44100, layout="stereo", max_bytes=budget) == n
        with open(dst, "rb") as a, open(plain, "rb") as b:
            assert a.read() == b.read()


def _silent_flac(path, *, seconds, rate=8000):
    with av.open(path, "w", format="flac") as out:
        st = out.add_stream("flac", rate=rate)
        st.layout = "mono"
        n, step = 0, 4000
        while n < seconds * rate:
            fr = av.AudioFrame.from_ndarray(
                np.zeros((1, step), dtype=np.int16), format="s16",
                layout="mono")
            fr.sample_rate, fr.pts = rate, n
            for pkt in st.encode(fr):
                out.mux(pkt)
            n += step
        for pkt in st.encode(None):
            out.mux(pkt)
    return path


def two_rate_flac(tmp_path):
    """A second of 16 kHz FLAC followed by a second of 8 kHz, as `cat a.flac
    b.flac` makes it: the decoder's frames change rate halfway."""
    parts = []
    for rate in (16000, 8000):
        with open(_silent_flac(str(tmp_path / f"{rate}.flac"), seconds=1,
                               rate=rate), "rb") as f:
            parts.append(f.read())
    src = tmp_path / "ab.flac"
    src.write_bytes(b"".join(parts))
    return str(src)


def test_decode_span_does_not_hold_the_skipped_audio(tmp_path):
    """The skip is dropped while decoding: the piece holds `want` samples
    (plus at most one resampled frame), not everything before it. A link's
    language check takes the skip from a remote playlist."""
    src = _silent_flac(str(tmp_path / "long.flac"), seconds=600)
    a = audio_transcode.decode_span_16k(src, 580.0, 1.0)
    assert a.dtype == np.float32 and a.size == RATE
    assert a.base is None or a.base.nbytes <= 2 * RATE * 4


def _odd_rate_silent_flac(path, *, layout, seconds, rate=1100):
    """Silent FLAC at an odd rate in 65535-sample blocks: a few KB for
    hours of audio, which resamples ~50x slower than it decodes."""
    with av.open(path, "w", format="flac") as out:
        st = out.add_stream("flac", rate=rate)
        st.layout = layout
        st.codec_context.options = {"frame_size": "65535"}
        channels = 2 if layout == "stereo" else 1
        n, step = 0, rate * 600
        while n < seconds * rate:
            m = min(step, int(seconds * rate) - n)
            fr = av.AudioFrame.from_ndarray(
                np.zeros((1, m * channels), dtype=np.int16), format="s16",
                layout=layout)
            fr.sample_rate, fr.pts = rate, n
            for pkt in st.encode(fr):
                out.mux(pkt)
            n += m
        for pkt in st.encode(None):
            out.mux(pkt)
    return path


def _flac_header_and_frames(path):
    """(fLaC marker + metadata blocks, the frames after them)."""
    data = open(path, "rb").read()
    i = 4
    while True:
        last = data[i] & 0x80
        i += 4 + int.from_bytes(data[i + 1:i + 4], "big")
        if last:
            return data[:i], data[i:]


def _count_resampled(monkeypatch):
    """Input samples that reach _resampled, as a list to sum."""
    fed = []
    real = audio_transcode._resampled

    def counting(frames, **kw):
        def tally():
            for f in frames:
                fed.append(f.samples)
                yield f
        return real(tally(), **kw)

    monkeypatch.setattr(audio_transcode, "_resampled", counting)
    return fed


def test_decode_span_does_not_resample_the_skipped_audio(tmp_path,
                                                         monkeypatch):
    """A skip far into a long, tiny file costs a decode, not a resample: a
    link's language check takes the skip from a remote playlist, and a
    few-KB FLAC of hours of silence at an odd rate resamples ~50x slower
    than it decodes."""
    rate = 1100
    src = _odd_rate_silent_flac(str(tmp_path / "long.flac"), layout="mono",
                                seconds=3 * 3600, rate=rate)
    fed = _count_resampled(monkeypatch)
    a = audio_transcode.decode_span_16k(src, 3 * 3600 - 10.0, 1.0)
    assert a.size == RATE
    assert sum(fed) < 120 * rate


def test_decode_span_does_not_resample_a_skip_behind_a_format_change(
        tmp_path, monkeypatch):
    """One foreign frame in front of the skip (a stereo FLAC frame spliced
    before hours of mono ones) ends the drop early: what is left of the
    skip would be resampled, so past a budget the piece is refused instead
    (the language check then downloads the whole file)."""
    rate = 1100
    mono = _odd_rate_silent_flac(str(tmp_path / "mono.flac"), layout="mono",
                                 seconds=3 * 3600, rate=rate)
    stereo = _odd_rate_silent_flac(str(tmp_path / "stereo.flac"),
                                   layout="stereo", seconds=1, rate=rate)
    stereo_head, stereo_frames = _flac_header_and_frames(stereo)
    _mono_head, mono_frames = _flac_header_and_frames(mono)
    src = tmp_path / "mixed.flac"
    src.write_bytes(stereo_head + stereo_frames + mono_frames)
    fed = _count_resampled(monkeypatch)
    with pytest.raises(ValueError, match="changes format"):
        audio_transcode.decode_span_16k(str(src), 3 * 3600 - 10.0, 1.0)
    assert sum(fed) < 120 * rate * 40


def _count_decoded(monkeypatch):
    """Frames every _decoded_frames call hands out from here on, as a list
    of one count per call."""
    counts, real = [], audio_transcode._decoded_frames

    def counting(container, stream, *args):
        counts.append(0)
        for frame in real(container, stream, *args):
            counts[-1] += 1
            yield frame

    monkeypatch.setattr(audio_transcode, "_decoded_frames", counting)
    return counts


def _m4a_with_one_sync_sample(path, *, seconds):
    """An honest mono AAC .m4a whose sample table marks only its first
    sample as a sync sample: every seek lands on the start of the file."""
    rate = 44100
    with av.open(path, "w", format="mp4") as out:
        st = out.add_stream("aac", rate=rate, layout="mono")
        x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(rate * seconds) / rate)
             ).astype(np.float32)
        for i in range(0, len(x), 1024):
            fr = av.AudioFrame.from_ndarray(
                x[i:i + 1024].reshape(1, -1), format="fltp", layout="mono")
            fr.sample_rate, fr.pts = rate, i
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    boxes = list(_mp4_boxes(open(path, "rb").read()))
    assert boxes[-1][0] == b"moov"      # so no chunk offset moves
    stss = _mp4_box(b"stss", bytes(4) + (1).to_bytes(4, "big") * 2)
    moov = _mp4_with_tables(boxes[-1][1], {b"stbl": lambda p: p + stss})
    with open(path, "wb") as f:
        f.write(b"".join(_mp4_box(k, p) for k, p in boxes[:-1])
                + _mp4_box(b"moov", moov))
    return path


def test_decode_pieces_refuses_a_seek_that_lands_far_ahead_of_the_piece(
        tmp_path, monkeypatch):
    """Frames between where a seek lands and the piece are decoded and
    thrown away, and the file decides how many there are: with one sync
    sample every seek lands on the start, so each piece decodes the file up
    to itself — the whole table of a file that names one frame a million
    times at four bytes each. Past a budget of discarded frames, or of
    discarded samples, the piece is refused."""
    src = _m4a_with_one_sync_sample(str(tmp_path / "one-sync.m4a"),
                                    seconds=60)
    counts = _count_decoded(monkeypatch)
    # Within the budget the piece is found, as before: 50 s is 2153 frames
    # behind the start.
    [piece] = audio_transcode.decode_pieces_16k(src, [50.0], 1.0)
    assert piece.size == RATE and counts.pop() > 2153

    monkeypatch.setattr(audio_transcode, "_DISCARD_MAX_FRAMES", 100)
    with pytest.raises(ValueError, match="ahead of the piece"):
        audio_transcode.decode_pieces_16k(src, [50.0, 55.0], 1.0)
    assert counts == [101]              # and the second piece is not tried
    [piece] = audio_transcode.decode_pieces_16k(src, [2.0], 1.0)
    assert piece.size == RATE           # 86 frames behind the start

    counts.clear()
    monkeypatch.setattr(audio_transcode, "_DISCARD_MAX_FRAMES", 1 << 18)
    monkeypatch.setattr(audio_transcode, "_DISCARD_MAX_SAMPLES", 100 * 1024)
    with pytest.raises(ValueError, match="ahead of the piece"):
        audio_transcode.decode_pieces_16k(src, [50.0], 1.0)
    assert counts == [101]


def test_decode_span_refuses_a_skip_past_the_discard_budget(tmp_path,
                                                            monkeypatch):
    """The skip of a link's language-check piece comes from a remote
    playlist: one far past the file decodes every frame of it only to drop
    it. The budget the seek's lead has ends that too."""
    src = _silent_flac(str(tmp_path / "long.flac"), seconds=600)
    counts = _count_decoded(monkeypatch)
    assert audio_transcode.decode_span_16k(src, 1e6, 1.0).size == 0
    assert counts.pop() > 1000          # the whole file, within the budget
    assert audio_transcode.decode_span_16k(src, 580.0, 1.0).size == RATE

    counts.clear()
    monkeypatch.setattr(audio_transcode, "_DISCARD_MAX_FRAMES", 100)
    with pytest.raises(ValueError, match="ahead of the piece"):
        audio_transcode.decode_span_16k(src, 1e6, 1.0)
    # The first frame always passes; the 101st dropped one ends it.
    assert counts == [102]


def test_the_seek_lead_and_the_skip_share_one_discard_budget(monkeypatch):
    """One _read_16k call has one budget: what it drops ahead of `after`
    and what it drops of the skip add up."""
    def frames(_container, _stream):
        for i in range(1000):           # 10 s in 10 ms frames
            fr = av.AudioFrame.from_ndarray(
                np.zeros((1, 80), dtype=np.int16), format="s16",
                layout="mono")
            fr.sample_rate, fr.pts = 8000, i * 80
            fr.time_base = fractions.Fraction(1, 8000)
            yield fr

    monkeypatch.setattr(audio_transcode, "_decoded_frames", frames)

    def read(budget, **kw):
        monkeypatch.setattr(audio_transcode, "_DISCARD_MAX_FRAMES", budget)
        return audio_transcode._read_16k(None, None, 1600, **kw)

    # 59 frames end before 0.6 s.
    assert read(59, after=0.6).size == 1600
    # Of a 3 s skip the frames more than a second ahead of the cut are
    # dropped: 198 behind the first one, which always passes.
    assert read(198, skip=3 * RATE).size == 1600
    assert read(257, skip=3 * RATE, after=0.6).size == 1600
    for budget, kw in ((58, {"after": 0.6}), (197, {"skip": 3 * RATE}),
                       (256, {"skip": 3 * RATE, "after": 0.6})):
        with pytest.raises(ValueError, match="ahead of the piece"):
            read(budget, **kw)


class _TinyFrame:
    """Stands in for a decoded av.AudioFrame in _drop_before (weakref-able,
    so the test can count the live ones)."""

    class format:
        name, bytes = "s16", 2

    class layout:
        name, channels = "mono", (None,)

    sample_rate, samples = 1_000_003, 1


def test_drop_before_holds_a_bounded_number_of_frames():
    """While looking for a 16 kHz grid point _drop_before holds a bounded
    number of frames, not just samples: a decoded frame object costs ~1 KB
    whatever its size, and a crafted stream of one-sample frames at an odd
    rate meets no grid point and no 60 s / 64 MiB cap within 24 MiB."""
    import weakref

    live, peak = weakref.WeakSet(), [0]

    def frames():
        for _ in range(100_000):
            peak[0] = max(peak[0], len(live))
            f = _TinyFrame()
            live.add(f)
            yield f

    for f in audio_transcode._drop_before(frames(), 16000 * 1000,
                                          {"n": 0, "rate": None}):
        peak[0] = max(peak[0], len(live))
        del f
    assert peak[0] <= audio_transcode._SKIP_HOLD_MAX_FRAMES + 2


def test_drop_before_charges_each_format_change():
    """A skip whose frames alternate their rate every frame is refused early:
    each change makes _resampled flush and build a new resampler, which
    costs far more than the frame's samples, so a 2 MiB MP3 alternating
    8000 / 11025 Hz would otherwise burn ~30 s of CPU under the budget."""

    class _Frame(_TinyFrame):
        samples = 576

    class _Rate8k(_Frame):
        sample_rate = 8000

    class _Rate11k(_Frame):
        sample_rate = 11025

    def frames():
        for i in range(40_000):  # ~2400 s of audio
            yield (_Rate8k if i % 2 == 0 else _Rate11k)()

    yielded = 0
    with pytest.raises(ValueError, match="changes format"):
        for _f in audio_transcode._drop_before(frames(), 16000 * 1000,
                                               {"n": 0, "rate": None}):
            yielded += 1
    assert yielded <= 1100


def test_drop_before_passes_a_skip_with_a_few_format_changes():
    """A real stream changes setup a handful of times in the skip (an ad
    insert switching rate at a segment boundary): the per-change charge
    leaves it well inside the budget."""

    class _Frame(_TinyFrame):
        samples = 1152

    class _Rate44k(_Frame):
        sample_rate = 44100

    class _Rate48k(_Frame):
        sample_rate = 48000

    def frames():
        for i in range(30_000):  # ~750 s, a rate change every 1000 frames
            yield (_Rate44k if (i // 1000) % 2 == 0 else _Rate48k)()

    out = list(audio_transcode._drop_before(frames(), 16000 * 300,
                                            {"n": 0, "rate": None}))
    assert out


def _crc(data, *, bits, poly):
    """MSB-first CRC from a zero start value: FLAC's CRC-8 and CRC-16."""
    top, mask, crc = 1 << (bits - 1), (1 << bits) - 1, 0
    for byte in data:
        crc ^= byte << (bits - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & mask if crc & top else (crc << 1) & mask
    return crc


_FLAC_RATE_CODES = {176400: 2, 192000: 3}


def _rate_switching_flac(path, rates, *, frames):
    """A hand-written silent mono FLAC of 16-sample constant blocks whose
    frame headers name `rates` in turn (13-15 bytes a frame): the decoder
    hands each frame on at its own rate, so every frame is a format change.
    An encoder writes one rate per file; a rate off the code table is
    written in tens of Hz."""
    def frame(n, rate):
        code, tail = _FLAC_RATE_CODES.get(rate), b""
        if code is None:
            code, tail = 14, (rate // 10).to_bytes(2, "big")
        number = (bytes([n]) if n < 0x80
                  else bytes([0xC0 | n >> 6, 0x80 | n & 0x3F]))
        # Sync, block size and rate codes, mono 16-bit, the frame number,
        # the block size (minus one) and the rate the codes point at.
        head = bytes([0xFF, 0xF8, 0x60 | code, 0x08]) + number + b"\x0f" + tail
        head += bytes([_crc(head, bits=8, poly=0x07)])
        body = head + b"\x00\x00\x00"       # a constant subframe of zeros
        return body + _crc(body, bits=16, poly=0x8005).to_bytes(2, "big")

    assert frames < 0x800
    info = (b"\x00\x10\x00\x10" + b"\x00" * 6
            + ((rates[0] << 44) | (15 << 36)).to_bytes(8, "big") + b"\x00" * 16)
    with open(path, "wb") as f:
        f.write(b"fLaC" + bytes([0x80, 0, 0, len(info)]) + info)
        f.write(b"".join(frame(n, rates[n % len(rates)])
                         for n in range(frames)))
    return path


def _count_resamplers(monkeypatch):
    """The av.AudioResampler objects built from here on, as a list to
    count."""
    built, real = [], av.AudioResampler

    def counting(*args, **kwargs):
        built.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(av, "AudioResampler", counting)
    return built


@pytest.mark.parametrize("rates,frames,most", [
    ((192000, 176400), 600, 40),
    # A rebuild costs ~10x more up here, so a change weighs 13.
    ((655350, 655340), 100, 12),
])
def test_resampled_refuses_a_stream_that_changes_format_every_frame(
        tmp_path, monkeypatch, rates, frames, most):
    """A stream whose rate changes on every frame is refused after a bounded
    number of resampler rebuilds, wherever it is read: each rebuild costs
    ~2 ms (more at a high rate) for a ~14-byte frame that yields no output
    sample, so none of the output-side budgets ever ends the decode."""
    src = _rate_switching_flac(str(tmp_path / "switching.flac"), rates,
                               frames=frames)
    dst = str(tmp_path / "out.wav")
    built = _count_resamplers(monkeypatch)
    for read in (
            lambda: audio_transcode.decode_span_16k(src, 0.0, 20.0),
            lambda: audio_transcode.decode_span_16k(src, 0.5, 20.0),
            lambda: audio_transcode.decode_pieces_16k(src, [0.0], 20.0),
            lambda: audio_transcode.transcode_to_wav(
                src, dst, rate=44100, layout="stereo")):
        del built[:]
        with pytest.raises(ValueError, match="changes format too often") as e:
            read()
        # A refusal, not a decode error: the callers tell the two apart.
        assert not isinstance(e.value, av.FFmpegError)
        assert len(built) <= most
    assert not os.path.exists(dst)


def _pcm_frames(rates, *, per_rate, samples=160):
    """s16 mono frames of a ramp, `per_rate` of them at each of `rates` in
    turn."""
    for i, rate in enumerate(rates):
        for j in range(per_rate):
            ramp = (np.arange(samples) + (i * per_rate + j) * samples) % 2000
            fr = av.AudioFrame.from_ndarray(
                ramp.astype(np.int16).reshape(1, -1), format="s16",
                layout="mono")
            fr.sample_rate = rate
            yield fr


def _resampled_16k(frames):
    out = [o.to_ndarray().reshape(-1) for o in audio_transcode._resampled(
        frames, format="flt", layout="mono", rate=16000)]
    return np.concatenate(out)


def test_resampled_passes_a_stream_with_a_few_format_changes():
    """Far more changes than a real file has (a concatenated mp3 has one, a
    TS recording a handful) stay inside the allowance, and the output is
    each run resampled on its own, in order."""
    rates = [44100, 48000] * 60 + [44100]           # 120 changes
    got = _resampled_16k(_pcm_frames(rates, per_rate=10))
    runs = list(_pcm_frames(rates, per_rate=10))
    want = np.concatenate([_resampled_16k(runs[i:i + 10])
                           for i in range(0, len(runs), 10)])
    assert got.size > 0 and np.array_equal(got, want)


def test_resampled_allowance_grows_with_the_frames_decoded(monkeypatch):
    """The allowance is refilled by decoded frames, so a long recording that
    keeps changing at a slow pace is never refused, however long it runs,
    while the same number of changes at twice the pace is."""
    monkeypatch.setattr(audio_transcode, "_MAX_SETUP_CHANGES", 4)
    monkeypatch.setattr(audio_transcode, "_FRAMES_PER_SETUP_CHANGE", 16)
    rates = [44100, 48000] * 30
    assert _resampled_16k(_pcm_frames(rates, per_rate=16)).size > 0
    with pytest.raises(ValueError, match="changes format too often"):
        _resampled_16k(_pcm_frames(rates, per_rate=8))


def _read_16k_holding_everything(container, stream, want, *, skip=0):
    """The implementation that kept every skipped sample: the reference
    the drop-while-decoding one must match sample for sample."""
    chunks, have = [], 0
    for out in audio_transcode._resampled(
            audio_transcode._decoded_frames(container, stream),
            format="flt", layout="mono", rate=16000):
        chunk = out.to_ndarray().reshape(-1)
        chunks.append(chunk)
        have += chunk.size
        if have >= skip + want:
            break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[skip:skip + want]


@pytest.mark.parametrize("fixture", ["two.m4a", "two.ts", "two.mp3",
                                     "concat.mp3"])
def test_decode_span_matches_the_full_decode_reference(tmp_path, fixture):
    src = (_concat_mp3(tmp_path) if fixture == "concat.mp3"
           else _two_tone(str(tmp_path / fixture)))
    # 12.5 / 25.0 land past whole runs of frames the skip never resamples.
    for skip in (0.0, 0.001, 1.0, 3.9999, 7.3, 12.5, 25.0, 59.999, 61.2,
                 1000.0):
        for seconds in (0.0, 0.01, 1.0, 20.0):
            got = audio_transcode.decode_span_16k(src, skip, seconds)
            container, stream = audio_transcode._open_audio(src)
            try:
                want = _read_16k_holding_everything(
                    container, stream, int(seconds * 16000),
                    skip=int(skip * 16000))
            finally:
                container.close()
            assert got.dtype == want.dtype and got.shape == want.shape, (
                skip, seconds)
            assert np.array_equal(got, want), (skip, seconds)


def test_is_pcm_wav_or_flac_reads_the_bytes_not_the_name(tmp_path):
    """A PCM WAV and a FLAC are what they say, whatever their name; an MP3
    named .wav, or one muxed into a WAV container, is not."""
    wav = _write_src_wav(str(tmp_path / "pcm.mp3"), rate=RATE, nchannels=2)
    flac = _silent_flac(str(tmp_path / "s.wav"), seconds=1)
    mp3 = _damaged_mp3(str(tmp_path / "m.wav"), seconds=1)
    assert audio_transcode.is_pcm_wav_or_flac(wav) is True
    assert audio_transcode.is_pcm_wav_or_flac(flac) is True
    assert audio_transcode.is_pcm_wav_or_flac(mp3) is False
    mp3_in_wav = str(tmp_path / "mp3in.wav")
    with av.open(mp3_in_wav, "w", format="wav") as out:
        st = out.add_stream("libmp3lame", rate=RATE)
        st.layout = "mono"
        fr = av.AudioFrame.from_ndarray(
            np.zeros((1, 1152), dtype=np.int16), format="s16", layout="mono")
        fr.sample_rate = RATE
        for pkt in [*st.encode(fr), *st.encode(None)]:
            out.mux(pkt)
    assert audio_transcode.is_pcm_wav_or_flac(mp3_in_wav) is False


def _riff_chunks(data):
    """{chunk id: offset of its 8-byte header} of a RIFF/WAVE body."""
    import struct
    chunks, pos = {}, 12
    while pos + 8 <= len(data):
        cid, size = bytes(data[pos:pos + 4]), struct.unpack_from("<I", data, pos + 4)[0]
        chunks.setdefault(cid, pos)
        if cid == b"data":
            break
        pos += 8 + size + (size & 1)
    return chunks


def _forge_wav_length(path, out_path):
    """`path` with a data size far past the end of the file and a byte rate
    1000 times its own: FFmpeg then times the file by the byte rate, at a
    thousandth of what libsndfile reads (it counts the real samples)."""
    import struct
    data = bytearray(open(path, "rb").read())
    chunks = _riff_chunks(data)
    byte_rate_at = chunks[b"fmt "] + 8 + 8
    struct.pack_into("<I", data, byte_rate_at,
                     struct.unpack_from("<I", data, byte_rate_at)[0] * 1000)
    struct.pack_into("<I", data, chunks[b"data"] + 4, 0xFFFFFFF0)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def _av_wav(path, codec, fmt, layout, *, rate=22050, seconds=1.0):
    with av.open(path, "w", format="wav") as out:
        st = out.add_stream(codec, rate=rate)
        st.layout = layout
        ch = len(st.layout.channels)
        n = int(rate * seconds)
        dtype = {"s16": np.int16, "s32": np.int32, "flt": np.float32,
                 "dbl": np.float64, "u8": np.uint8}[fmt.rstrip("p")]
        arr = np.zeros((1, n * ch), dtype=dtype)
        if fmt == "u8":
            arr += 128
        fr = av.AudioFrame.from_ndarray(arr, format=fmt, layout=layout)
        fr.sample_rate = rate
        for pkt in [*st.encode(fr), *st.encode(None)]:
            out.mux(pkt)
    return path


@pytest.mark.parametrize("codec,fmt,layout", [
    ("pcm_s16le", "s16", "mono"),
    ("pcm_s24le", "s32", "stereo"),
    ("pcm_s32le", "s32", "2.1"),
    ("pcm_f32le", "flt", "2.1"),
    ("pcm_f64le", "dbl", "stereo"),
    ("pcm_u8", "u8", "2.1"),
    ("pcm_alaw", "s16", "mono"),
    ("pcm_mulaw", "s16", "stereo"),
])
def test_is_pcm_wav_or_flac_keeps_an_honest_pcm_wav(tmp_path, codec, fmt, layout):
    src = _av_wav(str(tmp_path / f"{codec}.wav"), codec, fmt, layout)
    assert audio_transcode.is_pcm_wav_or_flac(src) is True


def test_is_pcm_wav_or_flac_refuses_a_forged_byte_rate(tmp_path):
    """A PCM WAV whose data size overruns the file and whose byte rate is
    forged is timed by that byte rate (a thousandth of its samples here),
    while the separator's libsndfile reads every sample: its header does
    not bound the decode, so it is not trusted as a native PCM WAV."""
    honest = _write_src_wav(str(tmp_path / "honest.wav"), rate=RATE,
                            nchannels=1, dur_s=1.0)
    assert audio_transcode.is_pcm_wav_or_flac(honest) is True
    forged = _forge_wav_length(honest, str(tmp_path / "forged.wav"))
    assert audio_transcode.declared_shape(forged)[0] < 0.01
    assert audio_transcode.is_pcm_wav_or_flac(forged) is False



_KSDATAFORMAT_TAIL = bytes([0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71])


def _hand_wav(path, *, ch=2, bits=16, valid=None, tag=1, ext=False,
              block_align=None, byte_rate=None, list_first=False,
              rate=44100, seconds=1.0, magic=b"RIFF"):
    """A WAV written byte by byte, for fmt shapes no muxer writes: `ext`
    makes it WAVE_FORMAT_EXTENSIBLE with SubFormat `tag` and
    wValidBitsPerSample `valid`. The data chunk holds `seconds` of samples
    framed by the container width (ch x bits/8, rounded up to whole bytes),
    whatever block_align and byte_rate claim."""
    import struct
    e = ">" if magic == b"RIFX" else "<"
    ba = block_align if block_align is not None else ch * ((bits + 7) // 8)
    br = byte_rate if byte_rate is not None else rate * ba
    if ext:
        fmt = (struct.pack(e + "HHIIHH", 0xFFFE, ch, rate, br, ba, bits)
               + struct.pack(e + "HHI", 22, valid or bits, 0)
               + struct.pack("<IHH", tag, 0, 0x10) + _KSDATAFORMAT_TAIL)
    else:
        fmt = struct.pack(e + "HHIIHH", tag, ch, rate, br, ba, bits)
    data = bytes(int(rate * seconds) * ch * ((bits + 7) // 8))
    chunks = b""
    if list_first:
        info = b"INFOINAM" + struct.pack(e + "I", 5) + b"titl\0" + b"\0"
        chunks += b"LIST" + struct.pack(e + "I", len(info)) + info
    chunks += b"fmt " + struct.pack(e + "I", len(fmt)) + fmt
    chunks += b"data" + struct.pack(e + "I", len(data)) + data
    body = b"WAVE" + chunks
    with open(path, "wb") as f:
        f.write(magic + struct.pack(e + "I", len(body)) + body)
    return path


@pytest.mark.parametrize("bits,valid,block_align", [
    (8, 64, None), (8, 32, None), (16, 64, None), (8, 64, 16)])
def test_is_pcm_wav_or_flac_refuses_a_container_narrower_than_valid_bits(
        tmp_path, bits, valid, block_align):
    """An extensible WAV whose container (wBitsPerSample) is narrower than
    its valid bits: FFmpeg picks its codec from the valid bits and times
    the data that wide, while libsndfile frames it by the container and
    reads every sample — a fraction of its samples is declared, so it is
    not handed over as a native PCM WAV."""
    path = _hand_wav(str(tmp_path / "narrow.wav"), bits=bits, valid=valid,
                     ext=True, block_align=block_align,
                     byte_rate=44100 * 2 * valid // 8)
    assert audio_transcode.declared_shape(path)[0] <= 0.5
    assert audio_transcode.is_pcm_wav_or_flac(path) is False


@pytest.mark.parametrize("kw", [
    {"bits": 8}, {"bits": 16}, {"bits": 24}, {"bits": 32},
    {"bits": 20}, {"bits": 32, "tag": 3}, {"bits": 64, "tag": 3},
    {"bits": 16, "ext": True}, {"bits": 24, "ext": True},
    {"bits": 32, "ext": True}, {"bits": 32, "tag": 3, "ext": True},
    {"bits": 24, "valid": 20, "ext": True},
    {"bits": 24, "valid": 18, "ext": True},
    {"bits": 16, "valid": 12, "ext": True},
    {"bits": 24, "ch": 6, "ext": True}, {"bits": 16, "ch": 1},
    {"bits": 16, "list_first": True}, {"bits": 24, "rate": 48000},
    {"bits": 16, "magic": b"RIFX"},
], ids=repr)
def test_is_pcm_wav_or_flac_keeps_an_honest_hand_written_wav(tmp_path, kw):
    """Every honest fmt shape — valid bits narrower than the container
    (20-in-24) included, and a chunk ahead of fmt — stays a native PCM
    WAV."""
    path = _hand_wav(str(tmp_path / "honest.wav"), **kw)
    assert audio_transcode.is_pcm_wav_or_flac(path) is True


def test_wav_fmt_shape_reads_only_one_fmt_ahead_of_data(tmp_path):
    """The RIFF walk answers for one fmt chunk ahead of data; RF64, a second
    fmt and a data chunk before any fmt read as None."""
    import struct
    honest = _hand_wav(str(tmp_path / "h.wav"), bits=24, valid=20, ext=True)
    assert audio_transcode._wav_fmt_shape(honest) == (24, 6)
    raw = open(honest, "rb").read()
    rf64 = tmp_path / "rf64.wav"
    rf64.write_bytes(b"RF64" + raw[4:])
    assert audio_transcode._wav_fmt_shape(str(rf64)) is None
    fmt_at = raw.index(b"fmt ")
    fmt_size = struct.unpack_from("<I", raw, fmt_at + 4)[0]
    fmt_chunk = raw[fmt_at:fmt_at + 8 + fmt_size]
    twice = tmp_path / "twice.wav"
    twice.write_bytes(raw[:fmt_at] + fmt_chunk + raw[fmt_at:])
    assert audio_transcode._wav_fmt_shape(str(twice)) is None
    no_fmt = tmp_path / "nofmt.wav"
    no_fmt.write_bytes(raw[:fmt_at] + raw[fmt_at + 8 + fmt_size:])
    assert audio_transcode._wav_fmt_shape(str(no_fmt)) is None


def mp3_whisper_stops_early(path, *, tone_s=5, silence_s=120):
    """`tone_s` of 440 Hz then `silence_s` of silence as a 32 kbps mp3 with
    400 bytes overwritten 5% in: faster-whisper's decode (and
    decode_16k_bounded) ends at the first invalid packet, a few seconds in,
    while _decoded_frames skips it and carries on to the end."""
    with av.open(path, "w", format="mp3") as out:
        st = out.add_stream("libmp3lame", rate=RATE)
        st.layout = "mono"
        st.bit_rate = 32000
        t = np.arange(RATE * tone_s) / RATE
        x = np.concatenate([
            (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
            np.zeros(RATE * silence_s, dtype=np.float32)])
        for i in range(0, len(x), 1152):
            fr = av.AudioFrame.from_ndarray(
                x[i:i + 1152].reshape(1, -1), format="flt", layout="mono")
            fr.sample_rate = RATE
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    data = bytearray(open(path, "rb").read())
    at = len(data) // 20
    for i in range(at, at + 400):
        data[i] = 0x00 if i % 2 else 0xFF
    with open(path, "wb") as f:
        f.write(data)
    return path


def test_stop_at_bytes_ends_the_wav_where_the_caller_stops_reading(tmp_path):
    """A damaged mp3 the transcode reads past the point Whisper stopped at:
    stop_at_bytes keeps the copy to what the caller decoded, as a valid,
    shorter WAV rather than a failure."""
    src = mp3_whisper_stops_early(str(tmp_path / "d.mp3"))
    decoded_s = audio_transcode.decode_16k_bounded(src, max_seconds=None).size / RATE
    assert decoded_s < 20
    stop = int((decoded_s + 5.0) * 32000)
    dst = str(tmp_path / "out.wav")
    n = audio_transcode.transcode_to_wav_16k_mono(src, dst, stop_at_bytes=stop)
    assert 44 < n <= stop + 44
    with wave.open(dst, "rb") as w:
        assert w.getnframes() * 2 <= stop
        assert w.getnframes() >= decoded_s * RATE


def test_stop_at_bytes_past_the_audio_changes_nothing(tmp_path):
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=44100, nchannels=2,
                         dur_s=3.0)
    plain = str(tmp_path / "plain.wav")
    n = audio_transcode.transcode_to_wav_16k_mono(src, plain)
    dst = str(tmp_path / "out.wav")
    assert audio_transcode.transcode_to_wav_16k_mono(
        src, dst, stop_at_bytes=int((3.0 + 5.0) * 32000)) == n
    with open(dst, "rb") as a, open(plain, "rb") as b:
        assert a.read() == b.read()


def one_hertz_flac(tmp_path, *, frame_size, samples=65535, name="low.flac"):
    """A FLAC declaring a 1 Hz rate, `frame_size` samples per block: a few
    KB that a resampler turns into 16 000 output samples per input one."""
    import subprocess

    from faster_whisper_backend.audio.ffmpeg import ffmpeg_exe
    raw = tmp_path / "z.raw"
    raw.write_bytes((bytes(range(7, 256, 7)) * (samples // 36 * 2 + 2))[:samples * 2])
    out = tmp_path / name
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "s16le", "-ar", "1", "-ac", "1", "-i", str(raw),
             "-c:a", "flac", "-frame_size", str(frame_size), str(out)],
            capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg unavailable")
    if r.returncode != 0:
        pytest.skip("ffmpeg cannot write a 1 Hz flac")
    return str(out)


def run_under_ram_ceiling(code: str, *, headroom: int = 256 << 20) -> str:
    """Run `code` in a fresh interpreter whose address space may grow by at
    most `headroom` past what its imports took; its stdout, stripped. A
    regression then fails as a MemoryError instead of taking gigabytes from
    the machine running the suite."""
    import subprocess
    prelude = (
        "import resource\n"
        "import av, numpy\n"
        "from faster_whisper_backend.audio import transcode\n"
        "vm = [int(l.split()[1]) * 1024 for l in open('/proc/self/status')"
        " if l.startswith('VmSize:')][0]\n"
        f"resource.setrlimit(resource.RLIMIT_AS, (vm + {headroom}, vm + {headroom}))\n")
    r = subprocess.run([sys.executable, "-c", prelude + code],
                       capture_output=True, text=True, timeout=120,
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.dirname(os.path.abspath(__file__)))))
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip()


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="RLIMIT_AS and /proc/self/status are Linux's")
def test_a_frame_longer_than_60_s_is_refused_before_it_is_resampled(tmp_path):
    """One 8192-sample block at 1 Hz resamples to ~10^8 samples (2+ GB) in a
    single resampler call, ahead of every output-side budget. The refusal is
    a plain ValueError — a refused input upstream — not an av error."""
    src = one_hertz_flac(tmp_path, frame_size=8192, samples=8192 * 2)
    dst = str(tmp_path / "out.wav")
    got = run_under_ram_ceiling(
        "def kind(fn):\n"
        "    try:\n"
        "        fn()\n"
        "    except av.FFmpegError as e:\n"
        "        return 'ffmpeg:' + type(e).__name__\n"
        "    except Exception as e:\n"
        "        return type(e).__name__\n"
        "    return 'ok'\n"
        f"print(kind(lambda: transcode.transcode_to_wav({src!r}, {dst!r},"
        " rate=44100, layout='stereo', max_bytes=10_000_000)),"
        f" kind(lambda: transcode.decode_pieces_16k({src!r}, [0.0], 30.0)),"
        f" kind(lambda: transcode.decode_span_16k({src!r}, 0.0, 30.0)))\n")
    assert got == "ValueError ValueError ValueError"
    assert not os.path.exists(dst)


def test_an_8k_flac_with_the_longest_legal_frames_still_transcodes(tmp_path):
    """65535 samples at 8 kHz (8.2 s) is the longest block a real encoder
    writes; it stays well under the per-frame refusal."""
    src = _silent_flac(str(tmp_path / "big.flac"), seconds=20)
    with av.open(src, "w", format="flac") as out:
        st = out.add_stream("flac", rate=8000)
        st.layout = "mono"
        st.codec_context.options = {"frame_size": "65535"}
        x = (np.sin(2 * np.pi * 440 * np.arange(8000 * 20) / 8000)
             * 8000).astype(np.int16)
        fr = av.AudioFrame.from_ndarray(x.reshape(1, -1), format="s16",
                                        layout="mono")
        fr.sample_rate, fr.pts = 8000, 0
        for pkt in st.encode(fr):
            out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    container, stream = audio_transcode._open_audio(src)
    try:
        longest = max(f.samples for f in audio_transcode._decoded_frames(
            container, stream))
    finally:
        container.close()
    assert longest > 8000 * 8
    dst = str(tmp_path / "out.wav")
    audio_transcode.transcode_to_wav_16k_mono(src, dst)
    with wave.open(dst, "rb") as w:
        assert w.getnframes() == 20 * RATE


# transcode._MAX_FRAME_SAMPLES, spelled out: the tests below pin the value.
FRAME_SAMPLES_BOUND = 1 << 22


def one_frame_alac_caf(path, *, frame_samples, rate=192000):
    """A CAF of one mono ALAC packet of digital silence that decodes to a
    single frame of `frame_samples` samples: a few hundred bytes whatever
    the count (a run of 65 536 zero samples is 26 bits), and seconds, not
    minutes, at `rate`."""
    import struct
    bits = "0" * 23                 # one mono element, compressed, no own size
    bits += "0" * 16                # no channel mixing
    bits += "0000" "0001" "000" "00000"  # predictor: none, shift 1, order 0
    done = 0
    while done < frame_samples:
        bits += "0"                 # one zero sample
        done += 1
        if done < frame_samples:    # then a run of up to 65 535 more
            bits += "1" * 25
            done += min(65535, frame_samples - done)
    bits += "111"                   # end of the packet
    bits += "0" * (-len(bits) % 8)
    packet = int(bits, 2).to_bytes(len(bits) // 8, "big")
    cookie = struct.pack(">IBBBBBBHIII", frame_samples, 0, 24, 0, 0, 14, 1,
                         255, 0, 0, rate)
    desc = struct.pack(">d4sIIIII", float(rate), b"alac", 0, len(packet),
                       frame_samples, 1, 0)
    out = b"caff" + struct.pack(">HH", 1, 0)
    for kind, body in ((b"desc", desc), (b"kuki", cookie),
                       (b"data", struct.pack(">I", 0) + packet)):
        out += kind + struct.pack(">q", len(body)) + body
    with open(path, "wb") as f:
        f.write(out)
    return path


def test_a_frame_of_too_many_samples_is_refused_before_it_is_resampled(
        tmp_path):
    """A frame goes through the resampler whole, and _MAX_FRAME_S only reads
    its length in seconds at the rate the file declares: a frame past
    _MAX_FRAME_SAMPLES samples x channels is refused whatever that rate is,
    with the plain ValueError of a refused input, and the cost check counts
    the file as over."""
    src = one_frame_alac_caf(str(tmp_path / "over.caf"),
                             frame_samples=FRAME_SAMPLES_BOUND + 1)
    dst = str(tmp_path / "out.wav")
    for read in (
            lambda: audio_transcode.transcode_to_wav(
                src, dst, rate=44100, layout="stereo"),
            lambda: audio_transcode.decode_pieces_16k(src, [0.0], 1.0),
            lambda: audio_transcode.decode_span_16k(src, 0.0, 1.0)):
        with pytest.raises(ValueError, match="unsupported audio") as refused:
            read()
        assert not isinstance(refused.value, av.FFmpegError)
    assert not os.path.exists(dst)
    assert audio_transcode.decoded_cost_within(
        src, cost_per_s=lambda rate, channels: 1.0, max_cost=1e9) is False


def test_a_frame_at_the_sample_bound_still_transcodes(tmp_path):
    assert audio_transcode._MAX_FRAME_SAMPLES == FRAME_SAMPLES_BOUND
    src = one_frame_alac_caf(str(tmp_path / "at.caf"),
                             frame_samples=FRAME_SAMPLES_BOUND)
    dst = str(tmp_path / "out.wav")
    audio_transcode.transcode_to_wav_16k_mono(src, dst)
    with wave.open(dst, "rb") as w:
        assert abs(w.getnframes() - FRAME_SAMPLES_BOUND * RATE // 192000) <= 1
    assert audio_transcode.decoded_cost_within(
        src, cost_per_s=lambda rate, channels: 1.0, max_cost=1e9) is True


def test_a_skip_hands_a_frame_of_too_many_samples_on_to_the_refusal():
    """In a skip, such a frame is neither held while looking for a grid
    point nor charged as resample work: it goes on to _resampled, which
    refuses it, like a frame past _MAX_FRAME_S."""

    class _Frame(_TinyFrame):
        # Every frame ends on the 16 kHz grid, so a held one is dropped.
        sample_rate, samples = 192000, 1920

    class _Wide(_Frame):
        samples = FRAME_SAMPLES_BOUND + 8

    frames = [_Frame(), _Frame(), _Wide(), *(_Frame() for _ in range(8))]
    out = list(audio_transcode._drop_before(iter(frames), 16000 * 1000,
                                            {"n": 0, "rate": None}))
    assert out == [frames[0], *frames[2:]]

    # Charged, the frames behind it add up to more than _SKIP_RESAMPLE_MAX.
    rest = [_Frame() for _ in range(audio_transcode._SKIP_RESAMPLE_MAX // 1920)]
    wide = _Wide()
    out = list(audio_transcode._charged_until_cut(wide, iter(rest), 0.0, 1e9))
    assert out == [wide, *rest]


def _write_clip(path, audio_codec, *, rate=44100, video_codec=None, fmt=None):
    """A 1 s 440 Hz tone in `audio_codec`, plus a few frames of
    `video_codec` when given, muxed into `path` (format from its suffix
    unless `fmt`)."""
    c = av.open(path, "w", format=fmt)
    a = c.add_stream(audio_codec, rate=rate)
    a.layout = "mono"
    v = None
    if video_codec:
        v = c.add_stream(video_codec, rate=5)
        v.width = v.height = 32
        v.pix_fmt = "yuv420p"
    ctx = a.codec_context
    res = av.AudioResampler(format=ctx.format.name, layout="mono", rate=rate)
    t = np.arange(rate) / rate
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(tone.reshape(1, -1), format="s16",
                                       layout="mono")
    frame.sample_rate = rate
    chunk = ctx.frame_size or 1024
    pts = 0
    for out in res.resample(frame) + res.resample(None):
        arr = out.to_ndarray()
        for i in range(0, arr.shape[1], chunk):
            piece = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(arr[:, i:i + chunk]),
                format=out.format.name, layout="mono")
            piece.sample_rate = rate
            piece.pts = pts
            pts += piece.samples
            for pkt in a.encode(piece):
                c.mux(pkt)
    for pkt in a.encode():
        c.mux(pkt)
    if v is not None:
        for _ in range(5):
            vf = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
            for pkt in v.encode(vf):
                c.mux(pkt)
        for pkt in v.encode():
            c.mux(pkt)
    c.close()
    return path


_CLIPS = [
    ("a.wav", "pcm_s16le", 44100, None),
    ("a.flac", "flac", 44100, None),
    ("a.mp3", "libmp3lame", 44100, None),
    ("a.m4a", "aac", 44100, None),
    ("a.ogg", "libopus", 48000, None),
    ("a.opus", "libopus", 48000, None),
    ("av.webm", "libopus", 48000, "libvpx"),
    ("av.mkv", "flac", 44100, "mpeg4"),
    ("av264.mkv", "libopus", 48000, "libx264"),
    ("av.mp4", "aac", 44100, "mpeg4"),
]


def test_pyav_audio_decoders_lists_audio_decoders_only():
    names = audio_ffmpeg.pyav_audio_decoders().split(",")
    assert {"flac", "aac", "opus", "mp3", "pcm_s16le", "vorbis"} <= set(names)
    assert not {"mpeg4", "h264", "vp8", "png", "subrip"} & set(names)


@pytest.mark.parametrize("name,acodec,rate,vcodec", _CLIPS,
                         ids=[c[0] for c in _CLIPS])
def test_open_audio_opens_audio_decoders_only_and_pcm_is_unchanged(
        name, acodec, rate, vcodec, tmp_path, monkeypatch):
    """A client file is opened with PyAV's own audio decoders as the codec
    whitelist, so stream probing never runs a video decoder on a crafted
    video track; the decoded audio is byte-identical to an open without it."""
    import hashlib
    src = _write_clip(str(tmp_path / name), acodec, rate=rate,
                      video_codec=vcodec)
    opens = _spy_av_open(monkeypatch)
    got = audio_transcode.decode_16k_bounded(src, max_seconds=None)
    dst = str(tmp_path / "got.wav")
    audio_transcode.transcode_to_wav_16k_mono(src, dst)
    src_opens = [kw["options"] for f, kw in opens if f == src]
    assert len(src_opens) == 2 and all(
        o["codec_whitelist"] == audio_ffmpeg.pyav_audio_decoders()
        for o in src_opens)

    opens.clear()
    monkeypatch.setattr(audio_ffmpeg, "pyav_audio_decoders", lambda: "")
    ref = audio_transcode.decode_16k_bounded(src, max_seconds=None)
    ref_dst = str(tmp_path / "ref.wav")
    audio_transcode.transcode_to_wav_16k_mono(src, ref_dst)
    src_opens = [kw["options"] for f, kw in opens if f == src]
    assert len(src_opens) == 2
    assert not any("codec_whitelist" in o for o in src_opens)
    assert got.size > 8000
    assert hashlib.sha1(got.tobytes()).digest() == hashlib.sha1(ref.tobytes()).digest()
    with open(dst, "rb") as f, open(ref_dst, "rb") as g:
        assert f.read() == g.read()


def test_open_audio_refuses_to_open_the_video_decoder(tmp_path, caplog):
    """The video track is still listed, but libavformat's stream probe is
    refused its decoder ("not on whitelist") instead of decoding it."""
    import logging
    src = _write_clip(str(tmp_path / "av.mkv"), "flac", video_codec="mpeg4")
    av.logging.set_level(av.logging.ERROR)
    try:
        with caplog.at_level(logging.ERROR):
            container, stream = audio_transcode._open_audio(src)
            try:
                assert stream.type == "audio"
                assert [s.type for s in container.streams].count("video") == 1
            finally:
                container.close()
    finally:
        av.logging.set_level(None)
    assert any("not on whitelist" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


_MP4_UNITY = (b"\x00\x01\x00\x00" + bytes(12) + b"\x00\x01\x00\x00"
              + bytes(12) + b"\x40\x00\x00\x00")


def _mp4_full_box(kind, payload, flags=0):
    return _mp4_box(kind, flags.to_bytes(4, "big") + payload)


def _mp4_u32(*values):
    return b"".join(v.to_bytes(4, "big") for v in values)


def _pcm_movie(tables, *, ticks, extra=b"", tracks=1):
    """A 'moov' box of `tracks` 16-bit PCM tracks (44.1 kHz mono), each with
    the sample table of its sample description plus `tables`."""
    full, u32 = _mp4_full_box, _mp4_u32
    mvhd = full(b"mvhd", u32(0, 0, 1000, 1000, 0x10000) + b"\x01\x00"
                + bytes(10) + _MP4_UNITY + bytes(24) + u32(tracks + 1))
    return _mp4_box(b"moov", mvhd + b"".join(
        _pcm_trak(tables, ticks, n + 1) for n in range(tracks)) + extra)


def _pcm_trak(tables, ticks, track_id):
    full, u32 = _mp4_full_box, _mp4_u32
    tkhd = full(b"tkhd", u32(0, 0, track_id, 0, ticks) + bytes(8)
                + b"\x00\x00\x00\x00\x01\x00\x00\x00" + _MP4_UNITY + bytes(8),
                flags=7)
    mdhd = full(b"mdhd", u32(0, 0, 44100, ticks) + b"\x55\xc4\x00\x00")
    hdlr = full(b"hdlr", bytes(4) + b"soun" + bytes(12) + b"snd\x00")
    dinf = _mp4_box(b"dinf", full(b"dref", u32(1) + full(b"url ", b"", 1)))
    sowt = _mp4_box(b"sowt", bytes(6) + b"\x00\x01" + bytes(8)
                    + b"\x00\x01\x00\x10" + bytes(4) + u32(44100 << 16))
    stbl = _mp4_box(b"stbl", full(b"stsd", u32(1) + sowt) + tables)
    minf = _mp4_box(b"minf", full(b"smhd", bytes(4)) + dinf + stbl)
    return _mp4_box(b"trak", tkhd + _mp4_box(b"mdia", mdhd + hdlr + minf))


def _mov_declaring_samples(path, n, tracks=1):
    """A 4.6 KB QuickTime file whose sample table DECLARES `n` two-byte
    samples in one chunk (a uniform stsz, one stts run, one stsc and one
    stco entry) over 4 KB of data; `tracks` of them at ~390 bytes each."""
    full, u32 = _mp4_full_box, _mp4_u32
    ftyp = _mp4_box(b"ftyp", b"qt  " + bytes(4) + b"qt  ")

    def moov(offset):
        return _pcm_movie(
            full(b"stts", u32(1, n, 2)) + full(b"stsc", u32(1, 1, n, 1))
            + full(b"stsz", u32(2, n)) + full(b"stco", u32(1, offset)),
            ticks=2 * n, tracks=tracks)

    offset = len(ftyp) + len(moov(0)) + 8
    with open(path, "wb") as f:
        f.write(ftyp + moov(offset) + _mp4_box(b"mdat", bytes(4096)))
    return path


def _fragmented_mp4_declaring_samples(path, n):
    """The fragmented sibling: an empty sample table and one movie fragment
    whose track run declares `n` samples, none of them listed."""
    full, u32 = _mp4_full_box, _mp4_u32
    empty = (full(b"stts", u32(0)) + full(b"stsc", u32(0))
             + full(b"stsz", u32(0, 0)) + full(b"stco", u32(0)))
    moov = _pcm_movie(empty, ticks=0, extra=_mp4_box(
        b"mvex", full(b"trex", u32(1, 1, 2, 2, 0))))
    # tfhd: default-base-is-moof, with a default sample duration and size.
    traf = _mp4_box(b"traf", full(b"tfhd", u32(1, 2, 2), 0x020018)
                    + full(b"trun", u32(n)))
    with open(path, "wb") as f:
        f.write(_mp4_box(b"ftyp", b"isom" + bytes(4) + b"isomiso5") + moov
                + _mp4_box(b"moof", full(b"mfhd", u32(1)) + traf)
                + _mp4_box(b"mdat", bytes(4096)))
    return path


def test_cap_libav_alloc_finds_the_libavutil_pyav_loaded():
    """On every platform the suite runs on: a PyAV wheel that lays its
    libraries out differently would leave the cap off without an error."""
    assert audio_ffmpeg.cap_libav_alloc() is True
    assert audio_ffmpeg._LIBAV_MAX_ALLOC == 256 * 1024 * 1024


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="reads the child's peak RSS in Linux's unit")
@pytest.mark.parametrize("build", [_mov_declaring_samples,
                                   _fragmented_mp4_declaring_samples])
def test_open_audio_bounds_a_declared_sample_count(build, tmp_path):
    """A file of under 5 KB that declares 16 million samples makes the mov
    demuxer build its index for all of them inside av.open, ahead of every
    decode budget: ~36 bytes of RAM per declared sample (600 MB here, 3 GB
    at the most one track can name). Under the allocation cap the index is
    refused; the file opens without one or is refused."""
    src = build(str(tmp_path / "declared.mov"), 16_000_000)
    assert os.path.getsize(src) < 5000
    how, peak_kb = _open_audio_in_a_child(src)
    assert how in ("ok", "refused")
    assert peak_kb < 300 * 1024


def _open_audio_in_a_child(src):
    """("ok" | "refused", peak resident kB) of transcode._open_audio(src)
    in a fresh interpreter, so the peak is that open's own."""
    import subprocess
    # VmHWM, not ru_maxrss: the latter survives the exec, so it is never
    # below what this test's own (large) process held when it forked.
    code = ("import re, sys\n"
            "from faster_whisper_backend.audio import transcode\n"
            "try:\n"
            "    transcode._open_audio(sys.argv[1])[0].close()\n"
            "    how = 'ok'\n"
            "except ValueError as e:\n"
            "    assert 'tracks' in str(e), e\n"
            "    how = 'refused'\n"
            "status = open('/proc/self/status').read()\n"
            "print(how, re.search(r'VmHWM:\\s+(\\d+) kB', status).group(1))\n")
    r = subprocess.run([sys.executable, "-c", code, src],
                       capture_output=True, text=True, timeout=120,
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.dirname(os.path.abspath(__file__)))))
    assert r.returncode == 0, r.stderr[-2000:]
    how, peak_kb = r.stdout.split()
    return how, int(peak_kb)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="reads the child's peak RSS in Linux's unit")
def test_open_audio_bounds_the_tracks_that_declare_a_sample_count(tmp_path):
    """The allocation cap bounds one track's index, and tracks add up: forty
    of them, 500 000 declared samples each, are 20 KB of file and 750 MB of
    index. The open takes PYAV_MAX_STREAMS tracks and refuses the file at
    the one past them."""
    src = _mov_declaring_samples(str(tmp_path / "tracks.mov"), 500_000,
                                 tracks=40)
    assert os.path.getsize(src) < 25_000
    how, peak_kb = _open_audio_in_a_child(src)
    assert how == "refused"
    assert peak_kb < 500 * 1024


@pytest.mark.parametrize("ext", ["mov", "mp4"])
def test_open_audio_refuses_a_file_with_more_tracks_than_the_limit(
        ext, tmp_path):
    limit = audio_ffmpeg.PYAV_MAX_STREAMS
    assert limit == 20
    at_limit = write_tracks(str(tmp_path / f"at.{ext}"), limit)
    container, stream = audio_transcode._open_audio(at_limit)
    try:
        assert len(container.streams) == limit and stream.index == 0
    finally:
        container.close()
    over = write_tracks(str(tmp_path / f"over.{ext}"), limit + 1)
    with pytest.raises(ValueError, match="more than 20 tracks") as refused:
        audio_transcode._open_audio(over)
    # The plain refusal callers answer as a bad input, not PyAV's
    # MemoryError (a ValueError-free av error class).
    assert type(refused.value) is ValueError
    assert str(refused.value) == audio_ffmpeg.TOO_MUCH_TO_OPEN


def test_a_matroska_file_past_the_stream_limit_opens_with_its_first_streams(
        tmp_path):
    """Matroska does not fail at the stream past the limit: the file opens
    with its first PYAV_MAX_STREAMS streams and decodes as before (a film
    rip lists its audio ahead of its many subtitle and font streams)."""
    src = write_tracks(str(tmp_path / "rip.mkv"),
                       audio_ffmpeg.PYAV_MAX_STREAMS + 4)
    container, stream = audio_transcode._open_audio(src)
    try:
        assert len(container.streams) == audio_ffmpeg.PYAV_MAX_STREAMS
        assert stream.index == 0
    finally:
        container.close()
    audio = audio_transcode.decode_16k_bounded(src, max_seconds=None)
    assert audio.size == RATE // 10


def test_refuse_input_refuses_a_file_with_more_tracks_than_the_limit(
        app_module, tmp_path, monkeypatch):
    """Ahead of every later consumer, some of which open the file without
    the limit: from the guarded open, and from the default-whitelist reopen
    of a file the guarded open could not read."""
    over = write_tracks(str(tmp_path / "over.mov"),
                        audio_ffmpeg.PYAV_MAX_STREAMS + 1)
    with pytest.raises(ValueError, match="more than 20 tracks"):
        app_module._refuse_input(over)

    def no_open(path):
        raise av.error.InvalidDataError(1094995529, "Invalid data found")
    monkeypatch.setattr(audio_transcode, "_open_audio", no_open)
    with pytest.raises(ValueError, match="more than 20 tracks"):
        app_module._refuse_input(over)
    at_limit = write_tracks(str(tmp_path / "at.mov"),
                            audio_ffmpeg.PYAV_MAX_STREAMS)
    assert app_module._refuse_input(at_limit) is None


class _FakeLibavutil:
    """Stands in for the ctypes handle of a libavutil; records the cap."""

    def __init__(self):
        self.caps = []
        self.av_max_alloc = lambda size: self.caps.append(size)


def test_cap_libav_alloc_sets_the_cap_once(monkeypatch):
    lib = _FakeLibavutil()
    loaded = []
    monkeypatch.setattr(audio_ffmpeg, "_libav_alloc_capped", None)
    monkeypatch.setattr(audio_ffmpeg, "_libavutil_paths",
                        lambda: ["/wheel/av.libs/libavutil-1.so.60"])
    monkeypatch.setattr(audio_ffmpeg.ctypes, "CDLL",
                        lambda path: loaded.append(path) or lib)
    assert audio_ffmpeg.cap_libav_alloc() is True
    assert audio_ffmpeg.cap_libav_alloc() is True
    assert loaded == ["/wheel/av.libs/libavutil-1.so.60"]
    assert lib.caps == [256 * 1024 * 1024]


@pytest.mark.parametrize("paths", [
    [],                                                   # another layout
    ["/wheel/av.libs/libavutil-1.so.60",                  # which one is it?
     "/wheel/av.libs/libavutil-2.so.61"],
    ["/nonexistent/av.libs/libavutil-1.so.60"],           # does not load
], ids=["none", "several", "unloadable"])
def test_cap_libav_alloc_fails_open_with_one_warning(paths, monkeypatch,
                                                     caplog):
    """Without exactly one loadable libavutil nothing is capped — decoding
    goes on as before — and the log says so once, not once per open."""
    import logging
    monkeypatch.setattr(audio_ffmpeg, "_libav_alloc_capped", None)
    monkeypatch.setattr(audio_ffmpeg, "_libavutil_paths", lambda: paths)
    with caplog.at_level(logging.WARNING):
        assert audio_ffmpeg.cap_libav_alloc() is False
        assert audio_ffmpeg.cap_libav_alloc() is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "allocation cap" in warnings[0].getMessage()


def test_cap_libav_alloc_fails_open_on_a_library_without_the_symbol(
        monkeypatch, caplog):
    import logging
    monkeypatch.setattr(audio_ffmpeg, "_libav_alloc_capped", None)
    monkeypatch.setattr(audio_ffmpeg, "_libavutil_paths",
                        lambda: ["/wheel/av.libs/libavutil-1.so.60"])
    monkeypatch.setattr(audio_ffmpeg.ctypes, "CDLL", lambda path: object())
    with caplog.at_level(logging.WARNING):
        assert audio_ffmpeg.cap_libav_alloc() is False
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def _spy_alloc_cap(monkeypatch):
    """The order of cap_libav_alloc calls and av.open calls from here on."""
    events, real_open = [], av.open
    monkeypatch.setattr(audio_ffmpeg, "cap_libav_alloc",
                        lambda: events.append("cap") or True)

    def spy_open(*args, **kwargs):
        events.append("open")
        return real_open(*args, **kwargs)
    monkeypatch.setattr(av, "open", spy_open)
    return events


def test_every_open_of_a_client_file_caps_the_allocation_first(
        tmp_path, monkeypatch):
    src = _write_src_wav(str(tmp_path / "a.wav"), rate=RATE, nchannels=1)
    events = _spy_alloc_cap(monkeypatch)
    audio_transcode._open_audio(src)[0].close()
    assert events[:2] == ["cap", "open"]

    events.clear()
    audio_transcode.refuse_multi_input(src)
    assert events[:2] == ["cap", "open"]


def test_refuse_input_caps_the_allocation_ahead_of_its_own_reopen(
        app_module, tmp_path, monkeypatch):
    """_refuse_input reopens a file _open_audio could not open with FFmpeg's
    default whitelist, and faster-whisper's decode_audio opens it after
    that: both go through the same libavutil, so the cap is set before
    either, whatever the first open did."""
    src = _write_src_wav(str(tmp_path / "a.wav"), rate=RATE, nchannels=1)
    events = _spy_alloc_cap(monkeypatch)

    def no_open(path):
        raise av.error.InvalidDataError(1094995529, "Invalid data found")
    monkeypatch.setattr(audio_transcode, "_open_audio", no_open)
    app_module._refuse_input(src)
    assert events == ["cap", "open"]
