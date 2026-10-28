"""Tests for audio/transcode.py (PyAV in-process).

Real PyAV is installed, so the happy path is a true round-trip. Also here:
the error paths (PyAV missing, no audio stream, dst cleanup after a
mid-stream failure), the demuxer SSRF guard (protocol_whitelist=file pin,
ffconcat/multi-input refusal, refuse_multi_input), non-UTF-8 tags,
decode_pieces_16k seeking / start_time offset and damaged-frame skipping.
"""

import fractions
import os
import sys
import wave

import numpy as np
import pytest

from faster_whisper_backend.audio import transcode as audio_transcode

RATE = 16000


def _write_src_wav(path, *, rate, nchannels, freq=440.0, dur_s=0.5):
    """Write a simple sine-tone WAV at an arbitrary rate/channel count."""
    t = np.linspace(0, dur_s, int(rate * dur_s), endpoint=False)
    tone = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)
    if nchannels == 2:
        # interleave L/R identical
        frames = np.repeat(tone, 2).tobytes()
    else:
        frames = tone.tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return path


# ---------------------------------------------------------------------------
# Happy path round-trips
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Error: PyAV missing
# ---------------------------------------------------------------------------

def test_transcode_pyav_missing(tmp_path, monkeypatch):
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    dst = str(tmp_path / "out.wav")
    # `import av` resolves None in sys.modules -> ImportError -> RuntimeError.
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(RuntimeError, match="PyAV"):
        audio_transcode.transcode_to_wav_16k_mono(src, dst)


# ---------------------------------------------------------------------------
# Error: no audio stream
# ---------------------------------------------------------------------------

def _write_video_only(path):
    """Encode a tiny video-only mp4 (no audio stream)."""
    import av
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
    # A valid container with only a video stream -> the explicit
    # ValueError("source has no audio stream") branch.
    src = _write_video_only(str(tmp_path / "videoonly.mp4"))
    dst = str(tmp_path / "out.wav")
    with pytest.raises(ValueError, match="no audio stream"):
        audio_transcode.transcode_to_wav_16k_mono(src, dst)
    # Destination must not linger on failure.
    assert not os.path.exists(dst)


def test_transcode_unreadable_input_cleans_up(tmp_path):
    # Garbage that PyAV can't even open: error propagates, dst not left behind.
    bad = tmp_path / "garbage.bin"
    bad.write_bytes(b"this is definitely not audio data" * 100)
    dst = str(tmp_path / "out.wav")
    with pytest.raises(Exception):
        audio_transcode.transcode_to_wav_16k_mono(str(bad), dst)
    assert not os.path.exists(dst)


def test_transcode_mid_stream_failure_unlinks_dst(tmp_path, monkeypatch):
    """A failure AFTER the output container is open (the tests above all
    fail inside _open_audio, before dst exists) closes the output and
    unlinks the partial dst."""
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
    import av
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    dst = str(tmp_path / "out.wav")
    real_open = av.open

    class _FailingClose:
        def __init__(self, container):
            self._c = container

        def __getattr__(self, name):
            return getattr(self._c, name)

        def close(self):
            self._c.close()
            raise OSError(28, "No space left on device")

    def spy_open(file, *args, **kwargs):
        c = real_open(file, *args, **kwargs)
        return _FailingClose(c) if kwargs.get("mode") == "w" else c
    monkeypatch.setattr(av, "open", spy_open)

    with pytest.raises(OSError, match="No space left"):
        audio_transcode.transcode_to_wav_16k_mono(src, dst)
    assert not os.path.exists(dst)


# ---------------------------------------------------------------------------
# The input open pins protocol_whitelist=file (concat/HLS/SDP SSRF guard)
# ---------------------------------------------------------------------------

def test_transcode_input_open_pins_file_protocol_whitelist(tmp_path, monkeypatch):
    """A PyAV upgrade or an av.open refactor must not silently drop the
    ``protocol_whitelist: file`` option — it is what stops a crafted
    ffconcat/HLS/SDP input from following external file:// or http://
    references (test_ssrf_guard.py is the URL-side twin)."""
    import av
    src = _write_src_wav(str(tmp_path / "in.wav"), rate=RATE, nchannels=1)
    dst = str(tmp_path / "out.wav")
    opens = []
    real_open = av.open

    def spy_open(file, *args, **kwargs):
        opens.append((file, kwargs))
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(av, "open", spy_open)

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


# ---------------------------------------------------------------------------
# decode_pieces_16k (the link language check)
# ---------------------------------------------------------------------------

def _two_tone(path, *, rate=44100, pts_offset_s=0):
    """60 s stereo: 440 Hz for the first 30 s, 880 Hz after — a piece's
    dominant frequency tells where it was cut from. `pts_offset_s` starts
    the stream's timestamps there (a non-WAV stream with start_time != 0)."""
    t = np.arange(rate * 60) / rate
    tone = np.where(t < 30, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t))
    pcm = np.repeat((tone * 8000).astype(np.int16), 2)
    if path.endswith(".wav"):
        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        return path
    import av
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
    early — the 35 s piece used to come from media 25 s (440 Hz)."""
    src = _two_tone(str(tmp_path / name), pts_offset_s=10)
    pieces = audio_transcode.decode_pieces_16k(src, [5.0, 35.0, 55.0], 4.0)
    assert [len(p) for p in pieces] == [4 * RATE] * 3
    assert [round(_peak_hz(p)) for p in pieces] == [440, 880, 880]


def test_decode_pieces_refuses_non_file_protocols(tmp_path, monkeypatch):
    """decode_pieces_16k opens through the same pinned whitelist as the
    transcode path (an HLS playlist fails whatever the whitelist, so only
    a spy on av.open can pin it)."""
    import av
    src = _two_tone(str(tmp_path / "two.wav"))
    opens = []
    real_open = av.open

    def spy_open(file, *args, **kwargs):
        opens.append((file, kwargs))
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(av, "open", spy_open)

    audio_transcode.decode_pieces_16k(src, [0.0], 1.0)
    assert [kw.get("options") for f, kw in opens if f == src] == [
        {"protocol_whitelist": "file"}]


def test_decode_pieces_refuses_an_ffconcat_sibling(tmp_path):
    _two_tone(str(tmp_path / "two.wav"))
    playlist = tmp_path / "upload"
    playlist.write_text("ffconcat version 1.0\nfile 'two.wav'\n")
    with pytest.raises(ValueError, match="unsupported container"):
        audio_transcode.decode_pieces_16k(str(playlist), [0.0], 1.0)


# ---------------------------------------------------------------------------
# refuse_multi_input (the raw-upload check before a non-PyAV decoder)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# A damaged frame mid-file is skipped, not a refusal
# ---------------------------------------------------------------------------

def _damaged_mp3(path, *, seconds=8):
    """An 8 s 440 Hz mp3 with 400 random bytes in the middle — still
    decodable by faster-whisper, but one packet makes the decoder raise
    InvalidDataError (an av.FFmpegError, which main.py treats as a decode
    failure, never as a refused input)."""
    import random

    import av
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


# ---------------------------------------------------------------------------
# Sample rate / channel layout changing mid-stream
# ---------------------------------------------------------------------------

def _concat_mp3(tmp_path):
    """2 s of 44.1 kHz stereo mp3 followed by 2 s of 22.05 kHz mono, as
    `cat a.mp3 b.mp3` makes it: the decoder's frames change rate and layout
    halfway, which one AudioResampler refuses with a plain ValueError."""
    import av
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
