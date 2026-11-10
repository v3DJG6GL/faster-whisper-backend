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

from faster_whisper_backend.audio import transcode as audio_transcode
from tests.conftest import write_wav

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
    assert in_opens[0].get("options") == {"protocol_whitelist": "file"}
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
    with av.open(path, "w") as out:
        st = out.add_stream("aac", rate=rate, layout="stereo")
        step = 1024
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
        {"protocol_whitelist": "file"}]


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


def test_decode_span_does_not_hold_the_skipped_audio(tmp_path):
    """The skip is dropped while decoding: the piece holds `want` samples
    (plus at most one resampled frame), not everything before it. A link's
    language check takes the skip from a remote playlist."""
    src = _silent_flac(str(tmp_path / "long.flac"), seconds=600)
    a = audio_transcode.decode_span_16k(src, 580.0, 1.0)
    assert a.dtype == np.float32 and a.size == RATE
    assert a.base is None or a.base.nbytes <= 2 * RATE * 4


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


@pytest.mark.parametrize("fixture", ["two.m4a", "concat.mp3"])
def test_decode_span_matches_the_full_decode_reference(tmp_path, fixture):
    src = (_two_tone(str(tmp_path / fixture)) if fixture == "two.m4a"
           else _concat_mp3(tmp_path))
    for skip in (0.0, 0.001, 1.0, 3.9999, 7.3, 59.999, 61.2, 1000.0):
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
