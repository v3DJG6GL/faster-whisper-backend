"""The transcription handler decodes an upload itself, under a ceiling
derived from free RAM, before Whisper sees it: only the byte size was capped
before, and a small file of compressed silence decodes to gigabytes."""

import asyncio
import threading
import time

import av
import numpy as np
import psutil
import pytest

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from tests._transcription_helpers import post_transcription, wav_bytes
from tests.audio.test_decode_bounded import out_of_memory_after_one_frame
from tests.audio.test_transcode import FRAME_SAMPLES_BOUND, one_frame_alac_caf
from tests.conftest import write_tracks


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


def test_a_frame_of_too_many_samples_is_a_413(client, fake_model, tmp_path):
    """The ceiling counts seconds; a single frame of more samples than the
    decode buffers at once is refused the same way, whatever its length."""
    src = one_frame_alac_caf(str(tmp_path / "over.caf"),
                             frame_samples=FRAME_SAMPLES_BOUND + 1)
    with open(src, "rb") as f:
        r = post_transcription(client, files={
            "file": ("a.caf", f.read(), "audio/x-caf")})
    assert r.status_code == 413, r.text
    assert fake_model.last_audio is None


def test_the_ceiling_is_half_the_free_ram_at_the_decode_peak(app_module,
                                                              monkeypatch):
    real = psutil.virtual_memory
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: real()._replace(available=16_000_000_000))
    # Whatever memory cgroup the suite runs in stays out of the figure.
    monkeypatch.setattr(app_module.system_stats, "_cgroup_headroom",
                        lambda total: None)
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


def test_a_libav_out_of_memory_mid_decode_is_a_413_too(
        client, app_module, fake_model, monkeypatch):
    """libav's own ENOMEM, raised while the real decode runs: PyAV types it
    as an av error and a MemoryError, and it gets the MemoryError's answer."""
    monkeypatch.setattr(
        app_module._transcode, "_fw_decoded_frames",
        out_of_memory_after_one_frame(app_module._transcode._fw_decoded_frames))
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


class _CountsClose:
    """An opened container that notes its close() in `closed`."""

    def __init__(self, container, closed):
        self._container, self._closed = container, closed

    def close(self):
        self._closed.append(self._container)
        return self._container.close()

    def __getattr__(self, name):
        return getattr(self._container, name)


def _spy_opens(monkeypatch):
    """(options of each av.open that succeeded from here on, the containers
    closed)."""
    opened, closed, real_open = [], [], av.open

    def spy_open(file, *args, **kwargs):
        container = real_open(file, *args, **kwargs)
        opened.append(kwargs.get("options"))
        return _CountsClose(container, closed)
    monkeypatch.setattr(av, "open", spy_open)
    return opened, closed


def test_a_file_only_the_stream_limit_keeps_shut_is_refused(
        app_module, tmp_path, monkeypatch):
    """A demuxer may answer the stream limit with an error of its own (WTV:
    invalid data) where MP4 answers out of memory. The fallback decoders
    set no limit and open such a file, so it is refused here like an MP4
    of as many tracks; the open that tells it from a file nothing reads has
    no limit either, and is closed again."""
    from faster_whisper.audio import decode_audio
    limit = audio_ffmpeg.PYAV_MAX_STREAMS
    at_limit = write_tracks(str(tmp_path / "at.wtv"), limit)
    assert app_module._refuse_input(at_limit) is None
    over = write_tracks(str(tmp_path / "over.wtv"), limit + 1)
    assert decode_audio(over, sampling_rate=16000).size > 0
    opened, closed = _spy_opens(monkeypatch)
    with pytest.raises(ValueError) as refused:
        app_module._refuse_input(over)
    assert str(refused.value) == audio_ffmpeg.TOO_MUCH_TO_OPEN
    assert opened == [None] and len(closed) == 1


def test_an_upload_only_the_stream_limit_keeps_shut_is_a_400(
        client, fake_model, tmp_path):
    over = write_tracks(str(tmp_path / "over.wtv"),
                        audio_ffmpeg.PYAV_MAX_STREAMS + 1)
    with open(over, "rb") as f:
        r = post_transcription(client, files={
            "file": ("a.wtv", f.read(), "video/x-ms-wtv")})
    assert r.status_code == 400, r.text
    assert "more than 20 tracks" in r.json()["detail"]
    assert fake_model.last_audio is None


def _overlap_spy(monkeypatch, app_module):
    """Stand in for _refuse_input; ([most calls running at once], the names
    of the threads they ran on)."""
    lock, running, most, threads = threading.Lock(), [0], [0], []

    def _refuse(path):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
            threads.append(threading.current_thread().name)
        time.sleep(0.02)
        with lock:
            running[0] -= 1
        if path == "refused":
            raise ValueError("unsupported container")
    monkeypatch.setattr(app_module, "_refuse_input", _refuse)
    return most, threads


def test_input_checks_of_parallel_uploads_do_not_overlap(app_module,
                                                         monkeypatch):
    """The check opens the file, and one open may hold gigabytes while it
    lasts: uploads arriving together are checked one after the other, on
    the worker the other header opens of client files share, and a refusal
    still reaches the caller."""
    most, threads = _overlap_spy(monkeypatch, app_module)

    async def _six():
        return await asyncio.gather(
            *(app_module._refuse_input_gated(p)
              for p in ["a", "b", "refused", "c", "d", "e"]),
            return_exceptions=True)
    results = asyncio.run(_six())
    assert most == [1] and len(threads) == 6
    assert set(threads) == {threads[0]}
    assert threads[0].startswith("client-open")
    assert [type(r) for r in results] == [
        type(None), type(None), ValueError, type(None), type(None),
        type(None)]


def test_the_upload_route_checks_its_input_on_the_shared_worker(
        client, app_module, fake_model, monkeypatch):
    most, threads = _overlap_spy(monkeypatch, app_module)
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=1.0), "audio/wav")})
    assert r.status_code == 200, r.text
    assert len(threads) == 1 and threads[0].startswith("client-open")
