"""Tests for audio_transcode.transcode_to_wav_16k_mono (PyAV in-process).

Real PyAV is installed, so the happy path is a true round-trip. Error paths:
PyAV missing -> RuntimeError; no audio stream -> ValueError (with dst cleanup).
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
