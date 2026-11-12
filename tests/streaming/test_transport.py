"""Tests for the streaming audio transports (raw PCM passthrough + ffmpeg decode).

The WebM tests generate a real WebM/Opus clip with the system ffmpeg and decode
it back to PCM, so they exercise the actual MediaRecorder-style path; they skip
if ffmpeg (or its libopus encoder) is unavailable.
"""

import asyncio
import logging
import socket
import struct
import subprocess
import sys
import threading

import pytest

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from faster_whisper_backend.audio.ffmpeg import ffmpeg_exe
from faster_whisper_backend.streaming import transport as stream_transport
from faster_whisper_backend.streaming.transport import (
    FfmpegTransport,
    RawPcmTransport,
    make_transport,
)
from tests._streaming_helpers import (
    discard as _nosink, drain_frames, handshake, ready_ws, stop_and_drain, stream_ws)


def test_ffmpeg_exe_resolves_to_runnable_binary():
    exe = ffmpeg_exe()
    assert exe
    r = subprocess.run([exe, "-version"], capture_output=True, timeout=15)
    assert r.returncode == 0
    assert b"ffmpeg version" in r.stdout


def _gen_webm(seconds: float) -> bytes:
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
             "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-f", "webm", "pipe:1"],
            capture_output=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg unavailable")
    if p.returncode != 0 or not p.stdout:
        pytest.skip("ffmpeg webm/opus encode unavailable")
    return p.stdout


def test_raw_transport_is_passthrough():
    got = bytearray()

    async def sink(b):
        got.extend(b)

    async def run():
        t = make_transport("pcm_s16le", sink)
        assert isinstance(t, RawPcmTransport)
        await t.start()
        await t.feed(b"\x01\x02\x03\x04")
        assert not t.dead
        await t.aclose()

    asyncio.run(run())
    assert bytes(got) == b"\x01\x02\x03\x04"


def test_ffmpeg_transport_decodes_webm_to_pcm():
    webm = _gen_webm(1.0)
    got = bytearray()

    async def sink(b):
        got.extend(b)

    async def run():
        t = make_transport("webm", sink)
        assert isinstance(t, FfmpegTransport)
        await t.start()
        await t.feed(webm)
        await t.aclose()

    asyncio.run(run())
    # ~1 s of 16 kHz mono s16le ≈ 32000 bytes; allow generous tolerance.
    assert len(got) > 16000


class _StuckProc:
    """An ffmpeg stand-in whose stdout never yields and that never exits."""
    returncode = None
    killed = False

    class stdin:
        @staticmethod
        def is_closing():
            return False

        @staticmethod
        def close():
            pass

    class stdout:
        @staticmethod
        async def read(_n):
            await asyncio.Event().wait()

    async def wait(self):
        await asyncio.Event().wait()

    def kill(self):
        self.killed = True


def test_ffmpeg_transport_dead_reader_warns_once(caplog):
    """Once ffmpeg has exited, feed() is still called once per inbound frame at
    the client's pace — it must log the dead decoder once, not per frame."""
    caplog.set_level(logging.WARNING, logger="faster_whisper_backend.streaming.transport")

    async def run():
        t = FfmpegTransport(_nosink)
        await t.start()
        t._proc.kill()
        for _ in range(100):
            if t._reader_dead:
                break
            await asyncio.sleep(0.05)
        assert t._reader_dead
        assert t.dead                  # the route ends the session on this
        for _ in range(50):
            await t.feed(b"\x00")
        await t.aclose()
        assert not t.dead              # a closed transport is finished, not dead

    asyncio.run(run())
    assert sum("ffmpeg exited" in r.getMessage() for r in caplog.records) == 1


def test_ffmpeg_transport_cancelled_aclose_still_kills_the_process():
    """A cancel while aclose() waits (server shutdown) must still kill ffmpeg:
    _closed is already True, so the routes backstop aclose() returns at once
    without killing it. The cancel itself must still propagate."""
    async def run():
        t = FfmpegTransport(_nosink)
        t._proc = _StuckProc()
        task = asyncio.create_task(t.aclose())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return t._proc.killed

    assert asyncio.run(run())


def test_ffmpeg_transport_cancel_during_reader_join_propagates():
    """A cancel while aclose() joins the stdout reader lands on the reader
    task; aclose() must still raise it (not return normally inside a handler
    meant to be cancelled), and the process must still be killed."""
    async def run():
        t = FfmpegTransport(_nosink)
        t._proc = _StuckProc()
        t._reader = asyncio.create_task(t._drain_stdout())
        task = asyncio.create_task(t.aclose())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return t._proc.killed, t._reader_dead

    assert asyncio.run(run()) == (True, True)


def test_ffmpeg_transport_reader_join_timeout_still_returns(monkeypatch):
    """The other side: a reader that outlives the join timeout is cancelled by
    wait_for, which reports a TimeoutError — aclose() returns normally."""
    real_wait_for = asyncio.wait_for

    def _short(aw, timeout):
        return real_wait_for(aw, timeout=0.05)

    monkeypatch.setattr(asyncio, "wait_for", _short)

    async def run():
        t = FfmpegTransport(_nosink)
        t._proc = _StuckProc()
        t._reader = asyncio.create_task(t._drain_stdout())
        await t.aclose()
        return t._proc.killed, t._reader.cancelled()

    assert asyncio.run(run()) == (True, True)


def _tiny_rate_wav(data_bytes: int, sample_rate: int = 1) -> bytes:
    """A pcm_u8 mono WAV at a near-zero sample rate: each byte of it resamples
    to 32 000 bytes of 16 kHz PCM at 1 Hz."""
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate, 1, 8)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", data_bytes) + b"\x80" * data_bytes)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_ffmpeg_transport_stops_a_decode_amplification_bomb():
    """A 128 KiB WAV at 1 Hz decodes to 4 GB of 16 kHz PCM. Output far beyond
    what any real codec yields per input byte must kill the decoder (dead →
    the route's decoder_failed close) instead of feeding the session. The
    input is above the ~64 KiB ffmpeg buffers before it decodes anything
    while stdin stays open."""
    wav = _tiny_rate_wav(128 * 1024)
    total = 0

    async def sink(b):
        nonlocal total
        total += len(b)

    async def run():
        t = FfmpegTransport(sink)
        await t.start()
        await t.feed(wav)
        for _ in range(100):
            if t.dead:
                break
            await asyncio.sleep(0.05)
        dead = t.dead
        await t.aclose()
        return dead

    assert asyncio.run(run())
    assert total <= len(wav) * 1024 + 120 * 16000 * 2


def _tiny_rate_flac() -> bytes:
    """A FLAC of 1 Hz noise in maximal 65535-sample blocks: one block
    resamples to ~1e9 samples at 16 kHz in a single frame. Noise keeps the
    blocks verbatim (~128 KiB each), past ffmpeg's ~64 KiB input probe, so it
    decodes while stdin stays open as it does in a live session."""
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", "1", "-ac", "1", "-i", "pipe:0",
             "-c:a", "flac", "-frame_size", "65535", "-f", "flac", "pipe:1"],
            input=bytes(range(7, 256, 7)) * 7282, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg unavailable")
    if r.returncode != 0 or not r.stdout:
        pytest.skip("ffmpeg flac encode unavailable")
    return r.stdout


def _peak_rss_kib(pid: int) -> int:
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_ffmpeg_transport_bounds_a_one_frame_allocation_bomb():
    """The output ratio never fires on a 1 Hz FLAC: ffmpeg allocates the whole
    resampled frame (~2 GB) before writing a byte — at the latest when the
    session ends and aclose() hands it EOF to flush. The per-allocation
    ceiling refuses that frame, so the child stays small and decodes nothing."""
    flac = _tiny_rate_flac()
    total = 0

    async def sink(b):
        nonlocal total
        total += len(b)

    async def run():
        t = FfmpegTransport(sink)
        await t.start()
        await t.feed(flac)
        pid = t._proc.pid
        peak = 0

        async def sample():
            nonlocal peak
            while True:
                try:
                    peak = max(peak, _peak_rss_kib(pid))
                except OSError:
                    return    # ffmpeg exited
                await asyncio.sleep(0.02)

        sampler = asyncio.create_task(sample())
        await t.aclose()      # EOF: ffmpeg flushes the buffered block
        sampler.cancel()
        return peak

    assert asyncio.run(run()) < 512 * 1024
    assert total == 0


def _transport_argv(monkeypatch) -> list:
    """The argv FfmpegTransport.start() hands to the subprocess, unrun."""
    seen = []

    async def fake_exec(*args, **_kw):
        seen.append(list(args))
        raise OSError("not started")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def run():
        with pytest.raises(OSError):
            await FfmpegTransport(_nosink).start()

    asyncio.run(run())
    return seen[0]


def test_ffmpeg_transport_restricts_the_input_demuxer(monkeypatch):
    """The demuxer is picked by content, so the argv must name the containers
    a dictation client sends: a manifest/playlist demuxer (dash, hls, concat)
    fetches what it references, and ffmpeg 7.0's dash demuxer does so past
    -protocol_whitelist."""
    argv = _transport_argv(monkeypatch)
    assert "-format_whitelist" in argv
    at = argv.index("-format_whitelist")
    assert at < argv.index("-i")
    allowed = set(argv[at + 1].split(","))
    assert {"matroska", "webm", "ogg", "mov", "mp4", "wav", "mp3", "aac",
            "flac"} <= allowed
    assert not allowed & {"dash", "hls", "concat", "webm_dash_manifest",
                          "sdp", "rtsp", "image2", "tee"}


def test_ffmpeg_transport_caps_the_declared_stream_count(monkeypatch):
    """Probing opens a decoder for every stream the container declares, and
    libavformat's own ceiling is 1000 of them: a few hundred KB of container
    then holds hundreds of MB in the child. -max_alloc bounds one allocation
    and the output ratio bounds PCM; neither counts streams, so the argv
    carries the count as an input option beside them."""
    argv = _transport_argv(monkeypatch)
    assert "-max_alloc" in argv
    assert "-max_streams" in argv
    at = argv.index("-max_streams")
    assert at < argv.index("-i")
    assert argv[at + 1] == str(stream_transport._MAX_INPUT_STREAMS)
    # Room for a recorder's audio (+ video) track and a file's cover art.
    assert 4 <= stream_transport._MAX_INPUT_STREAMS <= 16


def _ffmpeg_binary(which: str) -> str:
    if which == "default":
        return ffmpeg_exe()
    try:
        import imageio_ffmpeg  # noqa: PLC0415 — optional bundled binary
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pytest.skip("imageio-ffmpeg binary unavailable")


_CLIENT_CONTAINERS = {
    "webm_opus": ["-c:a", "libopus", "-f", "webm"],
    "ogg_opus": ["-c:a", "libopus", "-f", "ogg"],
    "mp4_aac_fragmented": ["-c:a", "aac", "-f", "mp4",
                           "-movflags", "frag_keyframe+empty_moov+default_base_moof"],
    "m4a_aac": ["-c:a", "aac", "-f", "ipod"],
    "mp3": ["-c:a", "libmp3lame", "-f", "mp3"],
    "flac": ["-c:a", "flac", "-f", "flac"],
    "adts_aac": ["-c:a", "aac", "-f", "adts"],
    "wav": ["-c:a", "pcm_s16le", "-f", "wav"],
    "wav_adpcm_ms": ["-c:a", "adpcm_ms", "-f", "wav"],
    "m4a_alac": ["-c:a", "alac", "-f", "ipod"],
}


async def _decode(data: bytes) -> int:
    total = 0

    async def sink(b):
        nonlocal total
        total += len(b)

    t = FfmpegTransport(sink)
    await t.start()
    await t.feed(data)
    await t.aclose()
    return total


@pytest.mark.parametrize("which", ["default", "imageio"])
@pytest.mark.parametrize("container", sorted(_CLIENT_CONTAINERS))
def test_ffmpeg_transport_decodes_every_client_container(container, which, monkeypatch, tmp_path):
    """Each container a recorder produces still opens under the demuxer
    allowlist, with the system ffmpeg and with the bundled fallback."""
    exe = _ffmpeg_binary(which)
    out = tmp_path / "clip"
    r = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1:sample_rate=48000", "-ac", "2",
         *_CLIENT_CONTAINERS[container], str(out)],
        capture_output=True, timeout=30)
    if r.returncode != 0 or not out.exists():
        pytest.skip(f"{container} encode unavailable")
    monkeypatch.setattr("faster_whisper_backend.audio.ffmpeg.ffmpeg_exe", lambda: exe)
    # ~1 s of 16 kHz mono s16le = 32000 bytes (codec priming adds a little).
    assert asyncio.run(_decode(out.read_bytes())) >= 30000


@pytest.mark.parametrize("which", ["default", "imageio"])
def test_ffmpeg_transport_refuses_a_container_with_too_many_streams(
        which, monkeypatch, tmp_path):
    """Both binaries take the option: a container at the stream cap decodes
    as before, one over it is refused at open and leaves a dead transport
    (the route then ends the session)."""
    exe = _ffmpeg_binary(which)
    cap = stream_transport._MAX_INPUT_STREAMS
    one = tmp_path / "one.mkv"
    r = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=0.5:sample_rate=48000",
         "-c:a", "flac", "-f", "matroska", str(one)],
        capture_output=True, timeout=30)
    if r.returncode != 0 or not one.exists():
        pytest.skip("mkv flac encode unavailable")

    def _tracks(n):
        out = tmp_path / f"t{n}.mkv"
        subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-i", str(one),
             *(["-map", "0:a"] * n), "-c", "copy", "-f", "matroska", str(out)],
            capture_output=True, timeout=30, check=True)
        return out.read_bytes()

    monkeypatch.setattr("faster_whisper_backend.audio.ffmpeg.ffmpeg_exe", lambda: exe)
    # ~0.5 s of 16 kHz mono s16le = 16000 bytes.
    assert asyncio.run(_decode(_tracks(cap))) >= 15000

    total = 0

    async def sink(b):
        nonlocal total
        total += len(b)

    async def run():
        t = FfmpegTransport(sink)
        await t.start()
        try:
            await t.feed(_tracks(cap + 1))
            # ffmpeg exits at the refusal, which ends the reader.
            await asyncio.wait_for(asyncio.shield(t._reader), 10)
            return t.dead
        finally:
            await t.aclose()

    assert asyncio.run(run()) is True
    assert total == 0


@pytest.mark.parametrize("which", ["default", "imageio"])
def test_ffmpeg_transport_never_follows_a_dash_manifest(which, monkeypatch):
    """A streamed MPD must not make the server connect to its BaseURL. ffmpeg
    7.0.2 (the imageio-ffmpeg fallback) dials it despite
    -protocol_whitelist pipe; the demuxer allowlist refuses the manifest."""
    exe = _ffmpeg_binary(which)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    srv.settimeout(0.1)
    port = srv.getsockname()[1]
    hits = []
    stop = threading.Event()

    def accept():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except (socket.timeout, OSError):
                continue
            hits.append(1)
            conn.close()

    th = threading.Thread(target=accept, daemon=True)
    th.start()
    mpd = (
        '<?xml version="1.0"?>\n'
        '<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" '
        'mediaPresentationDuration="PT1S" minBufferTime="PT1S" '
        'profiles="urn:mpeg:dash:profile:isoff-on-demand:2011">\n'
        f'<BaseURL>http://127.0.0.1:{port}/</BaseURL>\n'
        '<Period><AdaptationSet mimeType="audio/mp4">'
        '<Representation id="a" bandwidth="64000" codecs="mp4a.40.2">'
        '<BaseURL>seg.mp4</BaseURL></Representation></AdaptationSet></Period>'
        '</MPD>\n').encode()
    monkeypatch.setattr("faster_whisper_backend.audio.ffmpeg.ffmpeg_exe", lambda: exe)
    try:
        decoded = asyncio.run(_decode(mpd))
    finally:
        stop.set()
        th.join(5)
        srv.close()
    assert hits == []
    assert decoded == 0


def test_ffmpeg_transport_whitelists_only_audio_decoders(monkeypatch):
    """ffmpeg opens and decodes a frame of EVERY stream while probing, so the
    argv must name the decoders a dictation stream can need: all audio ones
    of the binary, no video / image / subtitle decoder."""
    argv = _transport_argv(monkeypatch)
    assert "-codec_whitelist" in argv
    at = argv.index("-codec_whitelist")
    assert at < argv.index("-i")
    allowed = set(argv[at + 1].split(","))
    assert {"opus", "vorbis", "aac", "mp3", "flac", "alac", "pcm_s16le",
            "adpcm_ms"} <= allowed
    assert not allowed & {"mpeg4", "h264", "hevc", "vp9", "mjpeg", "png",
                          "rawvideo", "ass", "subrip"}


def test_ffmpeg_transport_omits_the_decoder_list_when_the_listing_fails(monkeypatch):
    """No listing (missing binary, timeout) keeps the previous argv instead of
    passing an empty whitelist that would refuse every decoder."""
    monkeypatch.setattr("faster_whisper_backend.audio.ffmpeg.ffmpeg_exe",
                        lambda: "/nonexistent/ffmpeg-for-test")
    argv = _transport_argv(monkeypatch)
    assert "-codec_whitelist" not in argv
    assert argv[argv.index("-i") + 1] == "pipe:0"


def test_audio_decoder_whitelist_retries_after_a_failed_listing(monkeypatch):
    """A timed-out listing is not an answer: the next session asks again
    rather than running without the whitelist until a restart."""
    exe = ffmpeg_exe()
    audio_ffmpeg._listed_audio_decoders.cache_clear()
    real_run = subprocess.run

    def _timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired("ffmpeg", 15)

    monkeypatch.setattr(audio_ffmpeg.subprocess, "run", _timeout)
    assert audio_ffmpeg.audio_decoder_whitelist(exe) == ""
    monkeypatch.setattr(audio_ffmpeg.subprocess, "run", real_run)
    assert "opus" in audio_ffmpeg.audio_decoder_whitelist(exe).split(",")


def test_audio_decoder_whitelist_skips_the_legend():
    """The legend above the ------ rule has " A..... = Audio" lines too."""
    listing = ("Decoders:\n V..... = Video\n A..... = Audio\n S..... = Subtitle\n"
               " ------\n V....D h264                 H.264\n"
               " A....D opus                 Opus\n AF...D flac                 FLAC\n"
               " A..X.D sonic                Sonic\n S..... ass                  ASS\n")
    assert audio_ffmpeg.audio_decoder_names(listing) == ["opus", "flac", "sonic"]


@pytest.mark.parametrize("which", ["default", "imageio"])
def test_ffmpeg_transport_never_decodes_a_video_track(which, monkeypatch, tmp_path):
    """An mkv carrying an mpeg4 video track next to Opus: the video decoder
    is refused while probing (ffmpeg logs it off the whitelist) and the
    audio decodes byte for byte as it did without the whitelist."""
    exe = _ffmpeg_binary(which)
    clip = tmp_path / "av.mkv"
    r = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000",
         "-c:v", "mpeg4", "-c:a", "libopus", "-f", "matroska", str(clip)],
        capture_output=True, timeout=30)
    if r.returncode != 0 or not clip.exists():
        pytest.skip("mkv mpeg4+opus encode unavailable")
    data = clip.read_bytes()
    reference = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error",
         "-protocol_whitelist", "pipe",
         "-format_whitelist", stream_transport._INPUT_DEMUXERS,
         "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=data, capture_output=True, timeout=30).stdout
    assert len(reference) >= 30000

    monkeypatch.setattr("faster_whisper_backend.audio.ffmpeg.ffmpeg_exe", lambda: exe)
    real_exec = asyncio.create_subprocess_exec
    log_path = tmp_path / "ffmpeg.log"
    got = bytearray()

    async def sink(b):
        got.extend(b)

    with open(log_path, "wb") as log_fh:
        async def exec_with_debug_log(*args, **kw):
            args = list(args)
            args[args.index("-loglevel") + 1] = "debug"
            kw["stderr"] = log_fh
            return await real_exec(*args, **kw)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_with_debug_log)

        async def run():
            t = FfmpegTransport(sink)
            await t.start()
            await t.feed(data)
            await t.aclose()

        asyncio.run(run())
    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert "Codec (mpeg4) not on whitelist" in log
    assert "get_format" not in log
    assert bytes(got) == reference


def test_stream_route_accepts_webm_via_ffmpeg(client, energy_vad):
    webm = _gen_webm(2.0)
    with stream_ws(client) as ws:
        ready = handshake(ws, audio={"format": "webm"})
        assert ready["type"] == "ready"
        assert ready["audio_format"] == "webm"
        ws.send_bytes(webm)
        msgs = stop_and_drain(ws)[0]
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals, "expected a final from the decoded WebM"
    assert "welt" in "".join(m["committed"] + m.get("tail", "") for m in finals)


def test_stream_route_ends_the_session_when_the_decoder_dies(client, monkeypatch, energy_vad):
    """A dead ffmpeg discards every later frame while the audio keeps re-arming
    the idle deadline — the producer must say so and end the session instead
    of holding a STREAMING_MAX_SESSIONS slot until the client gives up."""
    from faster_whisper_backend.streaming import routes as stream_routes
    feeds = []

    class _DyingTransport:
        dead = False

        def __init__(self, sink, **_kw):
            pass

        async def start(self):
            pass

        async def feed(self, data):
            feeds.append(data)
            self.dead = True

        async def aclose(self):
            pass

    monkeypatch.setattr(stream_routes, "make_transport",
                        lambda fmt, sink, **kw: _DyingTransport(sink, **kw))
    with ready_ws(client, audio={"format": "webm"}) as ws:
        ws.send_bytes(b"\x1a\x45\xdf\xa3 corrupt")
        msgs = drain_frames(ws, 50)
    errors = [m for m in msgs if m.get("type") == "error"]
    assert [m["code"] for m in errors] == ["decoder_failed"], msgs
    assert len(feeds) == 1
