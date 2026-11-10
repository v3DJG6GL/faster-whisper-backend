"""Spies and stubs shared by the tests that drive a transcription through its
stages. Import as ``from tests.main._stage_helpers import ...``."""

from faster_whisper_backend.audio import transcode as audio_transcode
from faster_whisper_backend.transcription import receipt as tx_receipt


def capture_receipt(monkeypatch, seen=None):
    """Record every _format_request_block kwarg (and the rendered block, as
    "rendered") into `seen`, a new dict by default; returns it."""
    seen = {} if seen is None else seen
    real_block = tx_receipt._format_request_block

    def _capture(**kw):
        seen.update(kw)
        seen["rendered"] = real_block(**kw)
        return seen["rendered"]
    monkeypatch.setattr(tx_receipt, "_format_request_block", _capture)
    return seen


def stage_raises(monkeypatch, module, name, exc):
    """Replace the async stage function `module.name` with one that raises
    `exc` when awaited."""
    async def _raises(*a, **k):
        raise exc
    monkeypatch.setattr(module, name, _raises)


def must_not_run(monkeypatch, module, name, msg):
    """Replace the async stage function `module.name` with one that raises
    AssertionError(msg) when awaited."""
    stage_raises(monkeypatch, module, name, AssertionError(msg))


def transcode_raises(monkeypatch, exc):
    """Replace the synchronous audio_transcode.transcode_to_wav with one that
    raises `exc`. Keeps the real signature: the route's fallback arm would
    swallow the TypeError of a stale one."""
    def _raises(src, dst, *, rate=44100, layout="stereo", max_bytes=None):
        raise exc
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _raises)
