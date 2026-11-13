"""Client audio is opened and decoded in helper processes
(audio/decode_helper.py).

What a file costs libav in memory is the file's choice: one packet can hold
hundreds of frames that all decode before the first is handed out, a
container can list elements by the million, and libav bounds one allocation,
never their sum. In the server process the only end to that is the kernel
ending the server. A helper has a memory budget instead; the file that
passes it is refused and the server holds nothing of it. The rest of the
suite opens and decodes in-process (tests/conftest.py turns the helpers
off); the tests here turn them on.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from fractions import Fraction

import av
import numpy as np
import psutil
import pytest

from faster_whisper_backend.audio import decode_helper as dh
from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from faster_whisper_backend.audio import transcode as t
from faster_whisper_backend.media import subtitle_mux as pk
from tests._transcription_helpers import post_transcription, wav_bytes
from tests.conftest import write_tracks

LINUX = sys.platform.startswith("linux")
# The limit a helper holds itself to: the kernel's on Linux, and the
# watchdog thread every other platform has (run here on Linux too).
MODES = (["rlimit"] if LINUX else []) + ["watchdog"]
MIB = 1 << 20

# A stand-in helper: speaks the transport, then does `{body}` with the
# request (a dict) in `request` and the connection in `conn`.
_FAKE_BOOT = (
    "import json, os, sys, time\n"
    "from multiprocessing import connection as c\n"
    "conn = (c.PipeConnection if os.name == 'nt' else c.Connection)(int(sys.argv[2]))\n"
    "conn.send_bytes(b'ready' + json.dumps({{'limit': 'none', 'boot': ''}}).encode())\n"
    "while True:\n"
    "    try:\n"
    "        request = json.loads(conn.recv_bytes())\n"
    "    except EOFError:\n"
    "        os._exit(0)\n"
    "{body}\n")


def _fake(body: str) -> str:
    return _FAKE_BOOT.format(body="\n".join("    " + line for line in body.splitlines()))


# Answers every request with its own operation name — one with a "gate" only
# once that file exists, so a test decides how long a helper stays busy.
_ECHO = _fake("gate = request['args'].get('gate')\n"
              "while gate and not os.path.exists(gate):\n"
              "    time.sleep(0.01)\n"
              "conn.send_bytes(b'R' + json.dumps(request['op']).encode())")


def _error_of(call, *, value=False):
    """The exception `call` raised, else None (its result with `value`)."""
    try:
        result = call()
    except BaseException as exc:  # noqa: BLE001 - the error is the result
        return exc
    return result if value else None


def _in_threads(n, call):
    """Start `call` in `n` threads: (threads, the list their results go to)."""
    done: list = []
    threads = [threading.Thread(target=lambda: done.append(_error_of(call, value=True)),
                                daemon=True) for _ in range(n)]
    for thread in threads:
        thread.start()
    return threads, done


def _settled(read, expected, timeout=30.0):
    """Whether read() came to equal `expected` within `timeout` seconds."""
    end = time.monotonic() + timeout
    while read() != expected and time.monotonic() < end:
        time.sleep(0.02)
    return read() == expected


@pytest.fixture
def helpers(monkeypatch):
    """helpers(boot=None, mode=None, budget=None, spare=True): turn the
    decode helpers on for this test and return the running pool. `boot`: a
    stand-in helper program; `mode`: the limit the helpers use; `budget`:
    bytes one operation may add to its helper."""
    monkeypatch.setattr(dh, "ISOLATE", True)

    def _start(boot=None, *, mode=None, budget=None, spare=True):
        if boot is not None:
            monkeypatch.setattr(dh, "_BOOT", boot)
        if mode is not None:
            monkeypatch.setattr(dh, "LIMIT_MODE", mode)
        if budget is not None:
            monkeypatch.setattr(dh, "_budget", lambda: budget)
        dh.start(spare=spare)
        return dh._POOL
    try:
        yield _start
    finally:
        pool = dh._POOL
        dh.reset()
        assert pool is None or pool.pids() == []


def _tone(path, seconds=2.0, *, rate=44100, layout="stereo", codec="flac"):
    """A real audio file (the container is `path`'s extension)."""
    channels = len(av.AudioLayout(layout).channels)
    n = int(seconds * rate)
    wave = (np.sin(np.arange(n) * 2 * np.pi * 440 / rate) * 9000).astype(np.int16)
    with av.open(path, "w") as out:
        stream = out.add_stream(codec, rate=rate)
        stream.layout = layout
        frame = av.AudioFrame.from_ndarray(
            np.repeat(wave, channels).reshape(1, -1), format="s16", layout=layout)
        frame.sample_rate = rate
        for packet in (*stream.encode(frame), *stream.encode(None)):
            out.mux(packet)
    return path


def many_frames_in_one_packet(path, frames):
    """A Matroska file whose one packet holds `frames` FLAC frames of 7.1 /
    32-bit / 65 535 samples: about 40 bytes each in the file, 2 MB each once
    decoded — and a packet's frames are all decoded before the first is
    handed out, ahead of every per-frame budget."""
    size, rate, layout = 65535, 192000, "7.1"
    with av.open(path, "w", format="matroska") as out:
        stream = out.add_stream("flac", rate=rate)
        stream.layout, stream.format = layout, "s32"
        stream.codec_context.options = {"frame_size": str(size)}
        silence = np.zeros((1, size * 8), np.int32)
        packets = []
        for i in range(frames):
            frame = av.AudioFrame.from_ndarray(silence, format="s32", layout=layout)
            frame.sample_rate, frame.pts = rate, i * size
            frame.time_base = Fraction(1, rate)
            packets += stream.encode(frame)
        packets += stream.encode(None)
        one = av.Packet(b"".join(bytes(p) for p in packets))
        one.stream, one.pts, one.dts = stream, 0, 0
        one.time_base, one.duration = packets[0].time_base, packets[0].duration
        out.mux(one)
    return path


def _element(element_id: bytes, payload: bytes) -> bytes:
    return element_id + b"\x01" + len(payload).to_bytes(7, "big") + payload


def tag_lists(path, tags):
    """A Matroska file of one Tag holding `tags` SimpleTags of 7 bytes: the
    demuxer keeps about 200 bytes for each at the open, in allocations far
    below libav's cap on a single one."""
    simple = b"\x67\xc8\x84\x45\xa3\x81a"
    with open(path, "wb") as f:
        f.write(_element(b"\x1a\x45\xdf\xa3", _element(b"\x42\x82", b"matroska")))
        f.write(_element(b"\x18\x53\x80\x67", _element(
            b"\x12\x54\xc3\x67", _element(b"\x73\x73", simple * tags))))
    return path


class _Memory:
    """The most memory this process, and each of its helpers, held while
    the block ran (sampled every 5 ms), in bytes over what this process
    held at the start."""

    def __enter__(self):
        self._me = psutil.Process()
        self._base = self._me.memory_info().rss
        self.server = 0
        self.helper = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(0.005):
            self.server = max(self.server, self._me.memory_info().rss - self._base)
            for child in self._me.children():
                try:
                    self.helper = max(self.helper, child.memory_info().rss)
                except psutil.Error:
                    pass

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# --- the point: a file that asks for gigabytes is refused in the helper ---

@pytest.mark.parametrize("mode", MODES)
def test_a_packet_of_many_frames_ends_in_the_helper_not_in_the_server(
        helpers, tmp_path, mode):
    """150 frames in one 6 KB packet decode to 315 MB before the decode loop
    sees the first of them. In-process that is the server's memory (and
    linear: 1 MB of upload asks for tens of GB); in a helper the decode is a
    MemoryError at the budget, for every reader of the file."""
    src = many_frames_in_one_packet(str(tmp_path / "many.mkv"), 150)
    assert os.path.getsize(src) < 16 * 1024
    good = _tone(str(tmp_path / "good.flac"))
    dst = str(tmp_path / "out.wav")
    pool = helpers(mode=mode, budget=96 * MIB)
    assert t.declared_shape(good)[1:] == (44100, 2)       # a helper is up
    with _Memory() as memory:
        with pytest.raises(MemoryError):
            t.decode_16k_bounded(src, max_seconds=36000.0)
        with pytest.raises(MemoryError):
            t.transcode_to_wav(src, dst, rate=16000, layout="mono")
        with pytest.raises(MemoryError):
            t.decoded_cost_within(src, cost_per_s=t.separator_bytes_per_s,
                                  max_cost=1e15)
    assert not os.path.exists(dst)
    assert memory.server < 48 * MIB
    if mode == "rlimit":
        assert memory.helper < (96 + 160) * MIB
    # Each of the three was answered by a helper that is gone; the next
    # file meets a fresh one.
    assert len(t.decode_16k_bounded(good, max_seconds=60.0)) == 32000
    assert pool.pids()


def test_an_upload_past_the_helpers_budget_is_a_413_and_the_next_one_runs(
        client, app_module, fake_model, helpers, tmp_path):
    src = many_frames_in_one_packet(str(tmp_path / "many.mkv"), 150)
    helpers(budget=96 * MIB)
    app_module.cfg.LEADING_SILENCE_PAD_MS = 0
    with open(src, "rb") as f:
        r = post_transcription(client, files={
            "file": ("a.mkv", f.read(), "video/x-matroska")})
    assert r.status_code == 413, r.text
    assert "too long to decode" in r.json()["detail"]
    assert fake_model.last_audio is None
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=2.0), "audio/wav")})
    assert r.status_code == 200, r.text
    assert isinstance(fake_model.last_audio, np.ndarray)
    assert fake_model.last_audio.shape == (32000,)


@pytest.mark.parametrize("mode", MODES)
def test_a_file_of_tag_lists_is_held_to_the_budget_at_the_open(
        helpers, tmp_path, mode):
    """1.5 million 7-byte tags are 10 MiB of file and over 300 MB at the
    open, each list under libav's own cap. In a helper the open ends at the
    budget: refused as too much to open, or — where the demuxer drops what
    it could not allocate and reads on — answered as that demuxer answers a
    file without tracks."""
    src = tag_lists(str(tmp_path / "tags.mkv"), 1_500_000)
    helpers(mode=mode, budget=96 * MIB)
    with _Memory() as memory:
        for check in (t.refuse_input, pk.probe_streams):
            try:
                check(src)
            except av.FFmpegError:
                assert mode == "rlimit"
            except ValueError as refused:
                assert str(refused) == audio_ffmpeg.TOO_MUCH_TO_OPEN
    assert memory.server < 48 * MIB
    if mode == "rlimit":
        assert memory.helper < (96 + 160) * MIB


# --- a helper answers as the in-process call does ---------------------------------

def test_every_operation_answers_through_a_helper_as_it_does_in_process(
        helpers, tmp_path):
    flac = _tone(str(tmp_path / "a.flac"), 3.0)
    wav = str(tmp_path / "a.wav")
    with open(wav, "wb") as f:
        f.write(wav_bytes(seconds=2.0))
    tracks = write_tracks(str(tmp_path / "two.mkv"), 2)

    def answers():
        out = {}
        for name, src in (("flac", flac), ("wav", wav), ("mkv", tracks)):
            dst = str(tmp_path / f"{name}.out.wav")
            out[name] = {
                "refuse": t.refuse_input(src),
                "shape": t.declared_shape(src),
                "native": t.is_pcm_wav_or_flac(src),
                "cost": [t.decoded_cost_within(
                    src, cost_per_s=t.separator_bytes_per_s, max_cost=cost)
                    for cost in (1.0, 1e12)],
                "decode": t.decode_16k_bounded(src, max_seconds=60.0),
                "pieces": t.decode_pieces_16k(src, [0.0, 0.05, 99.0], 0.05),
                "span": t.decode_span_16k(src, 0.02, 0.05),
                "wav": (t.transcode_to_wav(src, dst, rate=22050, layout="stereo"),
                        open(dst, "rb").read()),
                "streams": pk.probe_streams(src),
            }
        return out

    def same(a, b):
        if isinstance(a, np.ndarray):
            return a.dtype == b.dtype and np.array_equal(a, b)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        return type(a) is type(b) and a == b

    local = answers()
    assert local["flac"]["decode"].dtype == np.float32
    assert len(local["flac"]["decode"]) == 48000
    pool = helpers()
    assert dh.routed() and pool.wait_idle(1, 60)
    first = set(pool.pids())
    helped = answers()
    for name in local:
        for key in local[name]:
            assert same(local[name][key], helped[name][key]), (name, key)
    # 27 operations and no helper lost or replaced on the way: the first
    # one and its spare.
    assert pool.wait_idle(2, 60)
    assert first <= set(pool.pids()) and len(pool.pids()) == 2


def test_a_helper_raises_what_the_in_process_call_raises(helpers, tmp_path):
    wav = str(tmp_path / "a.wav")
    with open(wav, "wb") as f:
        f.write(wav_bytes(seconds=2.0))
    junk = str(tmp_path / "junk.bin")
    with open(junk, "wb") as f:
        f.write(os.urandom(4096))
    playlist = str(tmp_path / "list.ffconcat")
    with open(playlist, "w") as f:
        f.write("ffconcat version 1.0\nfile 'a.wav'\n")
    missing = str(tmp_path / "missing.wav")
    dst = str(tmp_path / "out.wav")
    calls = {
        "junk decode": lambda: t.decode_16k_bounded(junk, max_seconds=60.0),
        "missing decode": lambda: t.decode_16k_bounded(missing, max_seconds=60.0),
        "missing shape": lambda: t.declared_shape(missing),
        "playlist check": lambda: t.refuse_input(playlist),
        "playlist probe": lambda: pk.probe_streams(playlist),
        "too long": lambda: t.decode_16k_bounded(wav, max_seconds=1.0),
        "over budget": lambda: t.transcode_to_wav(
            wav, dst, rate=44100, layout="stereo", max_bytes=10),
        "junk pieces": lambda: t.decode_pieces_16k(junk, [0.0], 1.0),
    }
    local = {name: _error_of(call) for name, call in calls.items()}
    assert isinstance(local["junk decode"], av.error.InvalidDataError)
    assert type(local["missing decode"]) is av.error.FileNotFoundError
    assert type(local["playlist check"]) is ValueError
    assert type(local["too long"]) is t.DecodeTooLong
    assert type(local["over budget"]) is t.TranscodeBudgetExceeded
    helpers()
    for name, call in calls.items():
        helped = _error_of(call)
        assert type(helped) is type(local[name]), name
        assert str(helped) == str(local[name]), name
        assert getattr(helped, "errno", None) == getattr(local[name], "errno", None), name
    assert not os.path.exists(dst)


def test_a_failed_decode_comes_back_with_its_cause(helpers, tmp_path, monkeypatch):
    """main answers a DecodeFailed with the error under it (an av error is
    an undecodable upload, anything else a 500), so the cause crosses too."""
    said = dh._describe(_error_of(lambda: _raise_from(
        t.DecodeFailed("decoder stopped"),
        av.error.InvalidDataError(1094995529, "Invalid data", "x.flac"))))
    rebuilt = dh._rebuild(json.loads(json.dumps(said)), t._HELPER_ERRORS)
    assert type(rebuilt) is t.DecodeFailed and str(rebuilt) == "decoder stopped"
    assert type(rebuilt.__cause__) is av.error.InvalidDataError
    assert rebuilt.__cause__.errno == 1094995529


def _raise_from(exc, cause):
    raise exc from cause


def test_only_pyavs_ffmpeg_errors_come_back_as_av_errors():
    """Callers read a plain ValueError as a refusal of the file and an av
    error as a failed decode; PyAV's unknown-codec error is the former."""
    from av.codec.codec import UnknownCodecError
    assert not issubclass(UnknownCodecError, av.FFmpegError)
    rebuilt = dh._rebuild(dh._describe(UnknownCodecError("nope")), {})
    assert type(rebuilt) is ValueError and str(rebuilt) == "nope"


@pytest.mark.parametrize("said, kind, text", [
    ({"name": "UnicodeDecodeError", "base": "ValueError", "text": "bad tag"},
     ValueError, "bad tag"),
    ({"name": "FileNotFoundError", "base": "OSError", "errno": 2,
      "text": "No such file or directory"}, FileNotFoundError,
     "[Errno 2] No such file or directory"),
    ({"name": "Weird", "base": "Weird", "text": "x"}, RuntimeError, "Weird: x"),
    ({"name": "DecodeTooLong", "local": False, "base": "RuntimeError", "text": "x"},
     RuntimeError, "x"),
    ({"name": "SystemExit", "av": [1, "x", None]}, av.error.FFmpegError, None),
    ("not a description", RuntimeError, None),
])
def test_a_helpers_error_is_rebuilt_from_a_fixed_set_of_classes(said, kind, text):
    """What a helper sends is data: the class comes from PyAV's errors, the
    caller's own table or a short list of builtins, never from a name the
    helper chose freely."""
    rebuilt = dh._rebuild(said, t._HELPER_ERRORS)
    assert type(rebuilt) is kind
    if text is not None:
        assert str(rebuilt) == text


# --- a helper that ends, lies or stalls -------------------------------------------

def test_a_helper_that_ends_on_the_file_refuses_it(helpers, tmp_path):
    """A crash of the decoder (in-process: of the server) is a refusal of
    the file at an open, DecoderEnded in a decode — never a second try in
    this process."""
    pool = helpers(_fake("os._exit(3)"))
    with pytest.raises(ValueError) as refused:
        t.refuse_input("a.bin")
    assert str(refused.value) == audio_ffmpeg.STOPS_THE_DECODER
    with pytest.raises(ValueError) as refused:
        pk.probe_streams("a.bin")
    assert str(refused.value) == audio_ffmpeg.STOPS_THE_DECODER
    with pytest.raises(t.DecoderEnded) as ended:
        t.decode_16k_bounded("a.bin", max_seconds=1.0)
    assert ended.value.returncode == 3
    assert ended.value.error_class == "decode_failed"
    dst = str(tmp_path / "out.wav")
    open(dst, "wb").close()
    with pytest.raises(t.DecoderEnded):
        t.transcode_to_wav("a.bin", dst, rate=16000, layout="mono")
    assert not os.path.exists(dst)
    assert pool.wait_idle(1, 30)          # replaced


def test_a_helper_the_watchdog_ended_is_out_of_memory(helpers):
    helpers(_fake(f"os._exit({dh._EXIT_MEMORY})"))
    with pytest.raises(MemoryError):
        t.decode_16k_bounded("a.bin", max_seconds=1.0)
    with pytest.raises(ValueError) as refused:
        t.declared_shape("a.bin")
    assert str(refused.value) == audio_ffmpeg.TOO_MUCH_TO_OPEN


def test_a_helper_cannot_send_more_samples_than_the_ceiling_allows(helpers):
    """The helper read a client's file; it is trusted with the server's
    memory no more than the file is."""
    pool = helpers(_fake("for _ in range(64):\n"
                         "    conn.send_bytes(b'D' + bytes(1 << 20))\n"
                         "conn.send_bytes(b'R' + b'null')"))
    assert pool.wait_idle(1, 30)
    before = pool.pids()
    with pytest.raises(t.DecodeTooLong):
        t.decode_16k_bounded("a.bin", max_seconds=10.0)      # 320 KB of samples
    assert pool.wait_idle(1, 30)
    assert not set(before) & set(pool.pids())                # killed mid-send
    audio = t.decode_16k_bounded("a.bin", max_seconds=None)  # no ceiling known
    assert audio.shape == (32 << 20,)


@pytest.mark.parametrize("answer", [
    "conn.send_bytes(b'R' + b'{not json')",
    "conn.send_bytes(b'?' + b'null')",
    "conn.send_bytes(b'R' + json.dumps([1, 2]).encode())",
    "conn.send_bytes(b'D' + bytes(6)); conn.send_bytes(b'R' + b'null')",
    "conn.send_bytes(b'D' + bytes(8)); conn.send_bytes(b'D' + bytes(8)); "
    "conn.send_bytes(b'R' + b'null')",
    "conn.send_bytes(b'R' + b'[' * 200000)",
    "conn.send_bytes(b'R' + b'[1, Infinity, 2]')",
    "conn.send_bytes(b'R' + b'[NaN, 1, 2]')",
    "conn.send_bytes(b'R' + b'[1e400, 1, 2]')",
    "conn.send_bytes(b'R' + b'[' + b'9' * 4000 + b', 1, 2]')",
])
def test_an_answer_of_another_shape_is_a_helper_that_ended(helpers, answer):
    helpers(_fake(answer))
    for call in (lambda: t.declared_shape("a.bin"),
                 lambda: t.decode_span_16k("a.bin", 0.0, 1.0)):
        error = _error_of(call)
        if error is None:
            # A number no file declares reads as "not declared", no more.
            assert "1e400" in answer or "9' * 4000" in answer
            assert t.declared_shape("a.bin") == (None, 1, 2)
            continue
        assert isinstance(error, (t.DecoderEnded, ValueError)), error
        assert not isinstance(error, av.FFmpegError)


def test_a_helper_whose_answer_cannot_be_read_does_not_serve_again(helpers):
    pool = helpers(_fake("conn.send_bytes(b'R' + b'{not json')"))
    assert pool.wait_idle(1, 30)
    served = pool.pids()
    with pytest.raises(t.DecoderEnded):
        dh.run("x", {})
    assert pool.wait_idle(1, 30)
    assert not set(served) & set(pool.pids())


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_a_helper_killed_from_outside_is_the_servers_fault_not_the_files(helpers):
    """The kernel ends a helper first when the machine runs out of memory
    (see test_the_kernel_would_end_a_helper_before_the_server): whatever
    file it was reading then is not thereby undecodable. A 500 the client
    may retry, and a probe the media store repeats."""
    from faster_whisper_backend.media import routes as media_routes
    helpers(_fake("os.kill(os.getpid(), 9)"))
    for call in (lambda: t.refuse_input("a.bin"),
                 lambda: t.decode_16k_bounded("a.bin", max_seconds=1.0),
                 lambda: pk.probe_streams("a.bin")):
        error = _error_of(call)
        assert type(error) is dh.HelperFailed, error
        assert media_routes._is_server_fault(error)


@pytest.mark.skipif(os.name == "nt", reason="writes the pair's raw descriptor")
def test_an_answer_that_starts_and_never_ends_is_cut_at_the_deadline(
        helpers, monkeypatch):
    """The header check runs on the one worker every upload waits for: a
    helper that sends the length of a message and then nothing must not
    hold it past the deadline."""
    monkeypatch.setattr(t, "_OPEN_OP_TIMEOUT_S", 0.5)
    pool = helpers(_fake("import struct\n"
                         "os.write(conn.fileno(), struct.pack('!i', 100) + b'R')\n"
                         "time.sleep(600)"))
    assert pool.wait_idle(1, 30)
    stuck = pool.pids()
    began = time.monotonic()
    with pytest.raises(ValueError) as refused:
        t.refuse_input("a.bin")
    assert str(refused.value) == audio_ffmpeg.TOO_SLOW_TO_OPEN
    assert time.monotonic() - began < 20
    assert pool.wait_idle(1, 30)
    assert not set(stuck) & set(pool.pids())


@pytest.mark.parametrize("said", [
    {"video_codec": ["h264"]}, {"width": -1}, {"width": True}, {"duration": "1"},
    {"duration": float("nan")}, {"duration": float("inf")}, {"width": 10 ** 4000},
    {"format_name": "concat"}, {"format_name": "matroska,webm;x"},
    {"audio_codec": "a" * 300}, [], None,
])
def test_stream_facts_from_a_helper_are_checked_field_by_field(said):
    with pytest.raises(ValueError):
        pk._streams_said(said)


def test_stream_facts_from_a_helper_keep_what_the_probe_found(tmp_path):
    found = pk._probe_streams(write_tracks(str(tmp_path / "two.mkv"), 2))
    assert pk._streams_said(json.loads(json.dumps(found.as_dict()))) == found


def test_an_open_that_does_not_end_is_stopped_and_refused(helpers, monkeypatch):
    monkeypatch.setattr(t, "_OPEN_OP_TIMEOUT_S", 0.3)
    monkeypatch.setattr(pk, "_PROBE_OP_TIMEOUT_S", 0.3)
    pool = helpers(_fake("time.sleep(600)"))
    assert pool.wait_idle(1, 30)
    stuck = pool.pids()
    for check in (t.refuse_input, t.declared_shape, pk.probe_streams):
        began = time.monotonic()
        with pytest.raises(ValueError) as refused:
            check("a.bin")
        assert str(refused.value) == audio_ffmpeg.TOO_SLOW_TO_OPEN
        assert time.monotonic() - began < 20
    assert pool.wait_idle(1, 30)
    assert not set(stuck) & set(pool.pids())                 # killed, not left


# --- the pool ---------------------------------------------------------------------

def test_one_helper_serves_one_request_after_another(helpers):
    pool = helpers(_ECHO)
    assert pool.wait_idle(1, 30)
    first = pool.pids()
    assert [dh.run(op, {}) for op in ("a", "b", "c")] == ["a", "b", "c"]
    assert pool.wait_idle(2, 30)          # the one that served, and the spare
    assert set(first) <= set(pool.pids()) and len(pool.pids()) == 2


def test_no_more_helpers_than_the_pool_allows_run_at_once(helpers, monkeypatch,
                                                           tmp_path):
    monkeypatch.setattr(dh, "MAX_LIVE", 3)
    gate = str(tmp_path / "gate")
    pool = helpers(_ECHO)
    threads, done = _in_threads(8, lambda: dh.run("x", {"gate": gate}))
    assert _settled(lambda: len(pool.pids()), 3)
    time.sleep(0.5)
    assert len(pool.pids()) == 3 and done == []
    open(gate, "wb").close()
    for thread in threads:
        thread.join(60)
    assert done == ["x"] * 8
    assert len(pool.pids()) <= 3


def test_whole_file_decodes_leave_helpers_for_the_quick_checks(helpers, monkeypatch,
                                                              tmp_path):
    """Long transcodes take at most _LONG_SLOTS helpers: an upload's input
    check never waits behind a run of them."""
    monkeypatch.setattr(dh, "MAX_LIVE", 2)
    monkeypatch.setattr(dh, "_LONG_SLOTS", 1)
    gate = str(tmp_path / "gate")
    helpers(_ECHO)
    threads, done = _in_threads(3, lambda: dh.run("long", {"gate": gate}, long=True))
    time.sleep(0.3)
    quick, answered = _in_threads(1, lambda: dh.run("quick", {}))
    quick[0].join(30)
    assert answered == ["quick"]
    assert done == []                     # all three still held or queued
    open(gate, "wb").close()
    for thread in threads:
        thread.join(60)
    assert done == ["long"] * 3


def test_idle_helpers_past_the_kept_ones_retire(helpers, monkeypatch, tmp_path):
    monkeypatch.setattr(dh, "_IDLE_RETIRE_S", 0.3)
    gate = str(tmp_path / "gate")
    pool = helpers(_ECHO)
    threads, done = _in_threads(5, lambda: dh.run("x", {"gate": gate}))
    assert _settled(lambda: len(pool.pids()) >= 5, True)
    open(gate, "wb").close()
    for thread in threads:
        thread.join(60)
    assert done == ["x"] * 5
    assert _settled(lambda: (len(pool.pids()), pool.idle_count()),
                    (dh.KEEP_IDLE, dh.KEEP_IDLE))
    assert dh.run("y", {}) == "y"


def test_after_shutdown_a_file_is_refused_not_opened_in_process(helpers, monkeypatch):
    pool = helpers(_ECHO)
    assert pool.wait_idle(1, 30)
    dh.shutdown()
    assert pool.pids() == []
    monkeypatch.setattr(t, "_refuse_input", lambda path: pytest.fail("in-process"))
    monkeypatch.setattr(t, "_decode_16k_frames",
                        lambda *a, **k: pytest.fail("in-process"))
    with pytest.raises(dh.HelperFailed):
        t.refuse_input("a.bin")
    with pytest.raises(dh.HelperFailed):
        t.decode_16k_bounded("a.bin", max_seconds=1.0)


def test_a_shutdown_ends_a_decode_in_flight(helpers, tmp_path):
    pool = helpers(_ECHO)
    assert pool.wait_idle(1, 30)
    threads, done = _in_threads(1, lambda: dh.run(
        "x", {"gate": str(tmp_path / "never")}, long=True))
    time.sleep(0.5)
    dh.shutdown()
    threads[0].join(30)
    assert not threads[0].is_alive()
    assert isinstance(done[0], dh.HelperFailed)
    assert pool.pids() == []


def test_where_no_helper_ever_starts_the_work_runs_in_process_and_says_so(
        helpers, monkeypatch, caplog, tmp_path):
    """A platform that starts no helper keeps transcribing as it did before
    helpers, with the gap named in the log once."""
    wav = str(tmp_path / "a.wav")
    with open(wav, "wb") as f:
        f.write(wav_bytes(seconds=1.0))
    monkeypatch.setattr(dh, "_spawn", lambda register: _raise_from(
        OSError("no processes here"), None))
    helpers(spare=False)
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        assert len(t.decode_16k_bounded(wav, max_seconds=60.0)) == 16000
        assert t.refuse_input(wav) is None
        assert pk.probe_streams(wav).audio_codec == "pcm_s16le"
    said = [r.getMessage() for r in caplog.records if "[decode]" in r.getMessage()]
    assert len(said) == 1
    assert "inside the server process" in said[0] and "no processes here" in said[0]
    assert not dh.routed()                # until the next try


def test_a_helper_that_cannot_load_the_decoder_is_no_helper(helpers, caplog, tmp_path):
    wav = str(tmp_path / "a.wav")
    with open(wav, "wb") as f:
        f.write(wav_bytes(seconds=1.0))
    helpers("import json, os, sys\n"
            "from multiprocessing import connection as c\n"
            "conn = (c.PipeConnection if os.name == 'nt' else c.Connection)(int(sys.argv[2]))\n"
            "conn.send_bytes(b'ready' + json.dumps("
            "{'limit': 'none', 'boot': 'ImportError: no av'}).encode())\n",
            spare=False)
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        assert t.declared_shape(wav) == (1.0, 16000, 1)
    assert any("ImportError: no av" in r.getMessage() for r in caplog.records)


def test_once_helpers_have_run_a_failed_start_refuses_the_file(helpers, monkeypatch):
    pool = helpers(_ECHO, spare=False)
    assert dh.run("x", {}) == "x"
    assert pool.proven
    assert pool.wait_idle(2, 30)          # the one that served, and its spare
    for pid in pool.pids():
        psutil.Process(pid).kill()
    monkeypatch.setattr(dh, "_spawn", lambda register: _raise_from(
        OSError("out of processes"), None))
    monkeypatch.setattr(t, "_refuse_input", lambda path: pytest.fail("in-process"))
    end = time.monotonic() + 30
    while pool.pids() and time.monotonic() < end:
        time.sleep(0.02)
    with pytest.raises(dh.HelperFailed) as failed:
        t.refuse_input("a.bin")
    assert "out of processes" in str(failed.value)


def test_run_without_started_helpers_runs_nothing():
    assert not dh.routed()
    with pytest.raises(dh.NotStarted):
        dh.run("declared_shape", {"src": "a.wav"})


def test_the_suite_opens_and_decodes_in_process():
    """tests/conftest.py: no test starts a helper by accident."""
    assert dh.ISOLATE is False
    dh.start()
    assert dh._POOL is None and not dh.routed()


def test_decoded_cost_within_takes_a_named_cost_in_a_helper(helpers, tmp_path):
    flac = _tone(str(tmp_path / "a.flac"))
    assert t.decoded_cost_within(flac, cost_per_s=lambda rate, ch: 1.0, max_cost=10)
    helpers()
    with pytest.raises(TypeError):
        t.decoded_cost_within(flac, cost_per_s=lambda rate, ch: 1.0, max_cost=10)


# --- the helper process -----------------------------------------------------------

def test_the_real_helper_reports_its_limit_and_serves_again(helpers, tmp_path):
    flac = _tone(str(tmp_path / "a.flac"))
    pool = helpers()
    assert pool.wait_idle(1, 60)
    assert [h.limit for h, _since in pool._idle] == ["rlimit" if LINUX else "watchdog"]
    before = pool.pids()
    for _ in range(3):
        assert t.declared_shape(flac)[1:] == (44100, 2)
    assert set(before) <= set(pool.pids())


@pytest.mark.skipif(not LINUX, reason="/proc")
def test_the_kernel_would_end_a_helper_before_the_server(helpers):
    pool = helpers()
    assert pool.wait_idle(1, 60)
    for pid in pool.pids():
        with open(f"/proc/{pid}/oom_score_adj") as f:
            assert f.read().strip() == "1000"


@pytest.mark.parametrize("answer", ["b'r' + json.dumps(request['op']).encode()", "b'M'"])
def test_a_helper_that_says_it_is_done_serving_is_replaced(helpers, answer):
    """A helper that grew, or met its budget, answers and ends itself: what
    its allocator kept would count against the next file."""
    pool = helpers(_fake(f"conn.send_bytes({answer})\nos._exit(0)"))
    assert pool.wait_idle(1, 30)
    served = pool.pids()
    if answer == "b'M'":
        with pytest.raises(MemoryError):
            dh.run("x", {})
    else:
        assert dh.run("x", {}) == "x"
    assert pool.wait_idle(1, 30)
    assert not set(served) & set(pool.pids())


@pytest.mark.parametrize("mode", MODES)
def test_the_limit_knows_when_its_process_grew(mode, monkeypatch):
    monkeypatch.setattr(dh, "_RECYCLE_BYTES", 64 * MIB)
    limit = dh._Limit(mode)
    assert limit.mode == mode
    assert not limit.grown()
    held = b"\x01" * (96 * MIB)
    assert limit.grown()
    del held


@pytest.mark.skipif(os.name == "nt", reason="POSIX reparenting")
def test_a_helper_ends_once_its_server_is_gone(tmp_path):
    """A helper inside a decode does not read its connection; it still must
    not outlive the server that would have collected its answer."""
    script = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', '''\n"
        "import sys, time\n"
        "sys.path.append(sys.argv[1])\n"
        "from faster_whisper_backend.audio import decode_helper as dh\n"
        "dh._PARENT_POLL_S = 0.1\n"
        "dh._end_with_parent()\n"
        "print(\\\"watching\\\", flush=True)\n"
        "time.sleep(600)\n"
        "''', sys.argv[1]], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)\n"
        "assert child.stdout.readline().strip() == b'watching'\n"
        "print(child.pid, flush=True)\n")
    from faster_whisper_backend import paths
    out = subprocess.run([sys.executable, "-c", script, paths.REPO_ROOT],
                         capture_output=True, text=True, timeout=60)
    pid = int(out.stdout.strip())
    end = time.monotonic() + 30
    while psutil.pid_exists(pid) and time.monotonic() < end:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.Error:
            break
        time.sleep(0.05)
    else:
        if psutil.pid_exists(pid):
            psutil.Process(pid).kill()
            pytest.fail("the helper outlived its parent")


# --- the app ----------------------------------------------------------------------

def test_the_lifespan_starts_and_stops_the_decode_helpers(app_module, monkeypatch):
    from starlette.testclient import TestClient

    events: list = []
    monkeypatch.setattr(dh, "start", lambda **k: events.append(("start", k)))
    monkeypatch.setattr(dh, "shutdown", lambda: events.append("shutdown"))
    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        assert events == [("start", {})]
        events.clear()
    assert events == ["shutdown"]


def test_the_running_app_decodes_an_upload_in_a_helper(app_module, fake_model,
                                                      helpers, monkeypatch):
    """The lifespan's own start, not the fixture's: after it an upload is
    checked and decoded by helpers, and the shutdown leaves none behind."""
    from starlette.testclient import TestClient

    app_module.cfg.LEADING_SILENCE_PAD_MS = 0
    local = [monkeypatch.setattr(t, name, lambda *a, **k: pytest.fail("in-process"))
             for name in ("_refuse_input", "_decode_16k_frames")]
    assert len(local) == 2
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        pool = dh._POOL
        assert pool is not None and dh.routed()
        r = post_transcription(client, files={
            "file": ("a.wav", wav_bytes(seconds=2.0), "audio/wav")})
        assert r.status_code == 200, r.text
        assert fake_model.last_audio.shape == (32000,)
        assert pool.proven and pool.pids()
    assert pool.pids() == []


def test_an_upload_that_ends_the_helper_is_a_400_not_a_second_decode(
        client, app_module, fake_model, monkeypatch, caplog):
    def _ended(path, *, max_seconds):
        raise t.DecoderEnded(-11)
    monkeypatch.setattr(app_module._transcode, "decode_16k_bounded", _ended)
    with caplog.at_level(logging.WARNING):
        r = post_transcription(client, files={
            "file": ("a.wav", wav_bytes(seconds=1.0), "audio/wav")})
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "the file could not be decoded as audio"
    assert fake_model.last_audio is None
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


def test_with_helpers_a_failed_open_is_answered_not_handed_to_the_model(
        client, app_module, fake_model, monkeypatch):
    """transcribe(path) decodes the file in this process, with no bound on
    its memory. Without helpers that is still how an undecodable upload gets
    its error (and how the suite's stub upload reaches the model); with
    them the helper's error is the answer."""
    def _unopened(path, *, max_seconds):
        raise av.error.InvalidDataError(1094995529, "Invalid data", path)
    monkeypatch.setattr(app_module._transcode, "decode_16k_bounded", _unopened)
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=1.0), "audio/wav")})
    assert r.status_code == 200, r.text
    assert isinstance(fake_model.last_audio, str)            # the path
    fake_model.last_audio = None
    # start() never ran here, so the input check still runs in-process;
    # only the handler is told that helpers are on.
    monkeypatch.setattr(app_module._decode_helper, "routed", lambda: True)
    monkeypatch.setattr(dh, "run", lambda *a, **k: _raise_from(dh.NotStarted(), None))
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=1.0), "audio/wav")})
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "the file could not be decoded as audio"
    assert fake_model.last_audio is None
