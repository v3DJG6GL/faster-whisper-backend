"""The transcription handler decodes an upload itself, under a ceiling
derived from free RAM, before Whisper sees it: only the byte size was capped
before, and a small file of compressed silence decodes to gigabytes."""

import numpy as np
import psutil
import pytest

from tests._transcription_helpers import post_transcription, wav_bytes


def test_audio_decoding_past_the_ceiling_is_a_413(client, app_module,
                                                  fake_model, monkeypatch):
    monkeypatch.setattr(app_module, "_upload_decode_cap_s", lambda: 1.0)
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=2.0), "audio/wav")})
    assert r.status_code == 413, r.text
    assert "too long to decode" in r.json()["detail"]
    assert fake_model.last_audio is None


def test_audio_within_the_ceiling_reaches_the_model_decoded(
        client, app_module, fake_model, monkeypatch):
    monkeypatch.setattr(app_module, "_upload_decode_cap_s", lambda: 3.0)
    app_module.cfg.LEADING_SILENCE_PAD_MS = 0
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=2.0), "audio/wav")})
    assert r.status_code == 200, r.text
    assert isinstance(fake_model.last_audio, np.ndarray)
    assert fake_model.last_audio.shape == (32000,)


def test_the_ceiling_is_half_the_free_ram_at_the_decode_peak(app_module,
                                                              monkeypatch):
    real = psutil.virtual_memory
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: real()._replace(available=16_000_000_000))
    # 8 GB at 16 kHz x 12 B per sample: ~11.6 h of audio.
    assert app_module._upload_decode_cap_s() == 16_000_000_000 / 2 / (16000 * 12)


def test_the_ceiling_fails_open_when_ram_cannot_be_read(app_module, monkeypatch):
    def _broken():
        raise OSError("no /proc/meminfo")
    monkeypatch.setattr(psutil, "virtual_memory", _broken)
    assert app_module._upload_decode_cap_s() is None


def test_a_decode_out_of_memory_is_a_413_not_a_second_decode(
        client, app_module, fake_model, monkeypatch):
    """A MemoryError is no property of a frame transcribe() would meet the
    same way: handing it the path would decode the whole file again, with
    no ceiling, while still holding the inference slot."""
    def _oom(path, *, max_seconds):
        raise MemoryError
    monkeypatch.setattr(app_module._transcode, "decode_16k_bounded", _oom)
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=2.0), "audio/wav")})
    assert r.status_code == 413, r.text
    assert "too long to decode" in r.json()["detail"]
    assert fake_model.last_audio is None


def _aes_hls_playlist(tmp_path):
    """An HLS playlist whose AES-128 key and segment are sibling files: the
    file-only open refuses the crypto protocol, FFmpeg's default nested
    whitelist (what faster-whisper, pyannote and the separator open with)
    plays it."""
    import subprocess

    import av
    seg = tmp_path / "seg.ts"
    with av.open(str(seg), "w", format="mpegts") as out:
        st = out.add_stream("mp2", rate=16000)
        st.layout = "mono"
        x = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
             * 8000).astype(np.int16)
        for i in range(0, len(x), 1152):
            fr = av.AudioFrame.from_ndarray(x[i:i + 1152].reshape(1, -1),
                                            format="s16", layout="mono")
            fr.sample_rate = 16000
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    key = bytes(range(0, 256, 17))
    (tmp_path / "key.ts").write_bytes(key)
    iv = "000102030405060708090a0b0c0d0e0f"
    try:
        r = subprocess.run(["openssl", "enc", "-aes-128-cbc", "-K", key.hex(),
                            "-iv", iv, "-in", str(seg),
                            "-out", str(tmp_path / "enc.ts")],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("openssl unavailable")
    if r.returncode != 0:
        pytest.skip("openssl cannot encrypt the segment")
    playlist = tmp_path / "a.m3u8"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:10\n"
        f'#EXT-X-KEY:METHOD=AES-128,URI="key.ts",IV=0x{iv}\n'
        "#EXTINF:1.0,\nenc.ts\n#EXT-X-ENDLIST\n")
    return str(playlist)


def test_a_playlist_only_the_fallback_decoders_open_is_refused(app_module,
                                                                tmp_path):
    from faster_whisper.audio import decode_audio
    path = _aes_hls_playlist(tmp_path)
    assert decode_audio(path, sampling_rate=16000).size > 0
    with pytest.raises(ValueError, match="unsupported container"):
        app_module._refuse_input(path)


def test_a_file_no_open_reads_is_left_to_the_decoder(app_module, tmp_path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"RIFFxxxxWAVE")
    assert app_module._refuse_input(str(junk)) is None
