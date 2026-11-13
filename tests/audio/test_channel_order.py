"""A file whose audio stream lists its channels in a CUSTOM order (an
8-channel AIFF, a QuickTime `chan` atom) is refused at the open on the PyAV
versions that free such a layout's channel map twice.

PyAV 17 and 18 wrap libav's channel layout as a bitwise copy and release it
when the wrapper dies; a custom-order layout owns a heap map, so dropping any
decoded frame (or reading the stream's channel count) frees the map libav
still holds — glibc aborts the whole process. Nothing here ever lets such a
wrapper die: the tests that would do so run in a child process.
"""

import struct
import subprocess
import sys

import pytest

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from faster_whisper_backend.audio import transcode
from faster_whisper_backend.paths import REPO_ROOT

pytestmark = pytest.mark.skipif(
    not audio_ffmpeg.pyav_frees_custom_maps(),
    reason="this PyAV keeps a custom channel map intact")


def _aiff(path, channels: int, *, chan: "bytes | None" = None):
    """80 frames of 8 kHz s16 silence; `chan` is the body of a CHAN chunk."""
    comm = struct.pack(">hIh", channels, 80, 16) + bytes.fromhex("400bfa00000000000000")
    chunks = [(b"COMM", comm), (b"SSND", bytes(8 + 80 * channels * 2))]
    if chan is not None:
        chunks.insert(0, (b"CHAN", chan))
    body = b"AIFF" + b"".join(
        cid + struct.pack(">I", len(data)) + data for cid, data in chunks)
    path.write_bytes(b"FORM" + struct.pack(">I", len(body)) + body)
    return str(path)


def _custom_order_aiff(path):
    """What `ffmpeg -f lavfi -i anullsrc=cl=7.1 -c:a pcm_s16be x.aiff` writes:
    the AIFF channel order of 7.1 is not libav's native one."""
    return _aiff(path, 8, chan=bytes.fromhex("008000080000000000000000"))


def test_a_file_in_custom_channel_order_is_refused_at_the_open(tmp_path):
    src = _custom_order_aiff(tmp_path / "a.aiff")
    with pytest.raises(ValueError, match="channel order") as refused:
        transcode._open_audio(src)
    # A plain ValueError is the refusal main._refuse_input answers 400; an
    # av error would read as "does not open" and reach the fallback decoders.
    assert type(refused.value) is ValueError


def test_the_upload_check_refuses_custom_channel_order(tmp_path, app_module):
    src = _custom_order_aiff(tmp_path / "a.aiff")
    with pytest.raises(ValueError, match="channel order"):
        app_module._refuse_input(src)


def test_the_upload_check_refuses_it_on_the_default_reopen_too(tmp_path, app_module, monkeypatch):
    """A file the guarded open cannot read is reopened with FFmpeg's defaults,
    the way the fallback decoders would open it; that arm checks the order
    of every audio stream as well."""
    import av

    src = _custom_order_aiff(tmp_path / "a.aiff")

    def no_guarded_open(path, **kwargs):
        raise av.error.InvalidDataError(1094995529, "Invalid data", path)

    monkeypatch.setattr(app_module._transcode, "_open_audio", no_guarded_open)
    with pytest.raises(ValueError, match="channel order") as refused:
        app_module._refuse_input(src)
    assert type(refused.value) is ValueError
    # Native order on that arm is still "not refused".
    assert app_module._refuse_input(_aiff(tmp_path / "stereo.aiff", 2)) is None


def test_native_channel_order_still_opens_and_decodes(tmp_path):
    src = _aiff(tmp_path / "stereo.aiff", 2)
    container, stream = transcode._open_audio(src)
    try:
        assert not audio_ffmpeg.holds_custom_channel_map(stream.codec_context.layout)
    finally:
        container.close()
    assert transcode.decode_16k_bounded(src, max_seconds=None).size == 160


def test_refusing_custom_channel_order_leaves_the_process_alive(tmp_path):
    """The whole point: every reader ends in the refusal and the interpreter
    exits cleanly afterwards (rc 0, not SIGABRT at a later free or at exit)."""
    src = _custom_order_aiff(tmp_path / "a.aiff")
    code = (
        "import sys\n"
        "from faster_whisper_backend.audio import transcode\n"
        "for read in (lambda: transcode.decode_16k_bounded(sys.argv[1], max_seconds=None),\n"
        "             lambda: transcode.declared_shape(sys.argv[1]),\n"
        "             lambda: transcode.decoded_cost_within(sys.argv[1], cost_per_s=1, max_cost=1)):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            read()\n"
        "        except ValueError as e:\n"
        "            assert 'channel order' in str(e), e\n"
        "        else:\n"
        "            raise SystemExit('not refused')\n"
        "print('refused')\n")
    done = subprocess.run([sys.executable, "-c", code, src], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert (done.returncode, done.stdout.strip()) == (0, "refused"), done.stderr[-400:]
