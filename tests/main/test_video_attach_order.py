"""The handler's progress-entry close vs. its keep_video attach.

When the video fetch finishes BEFORE the transcription, the handler itself
attaches the video to the job row (after its own job finish). /result keeps
`source_video_pending` only while the progress entry is open, so closing
the entry before that attach left a poll in between with neither the flag
nor the video keys. The keep_video stubs are the ones the url-route tests
use (fixtures imported, not copied).
"""

import time

from faster_whisper_backend.media import video as media_video
from faster_whisper_backend.transcription import progress as tx_progress
from tests.media.test_routes_url import (  # noqa: F401 — fixtures
    _PID, _post_url, url_enabled, video_enabled)


def test_handler_side_attach_runs_while_the_progress_entry_is_open(
        client, video_enabled, fake_model, monkeypatch):  # noqa: F811
    open_at_attach: list = []
    real_attach = media_video._jobs_attach_video_sync

    def _attach(pid, state):
        open_at_attach.append(pid in tx_progress._BATCH_PROGRESS)
        real_attach(pid, state)
    monkeypatch.setattr(media_video, "_jobs_attach_video_sync", _attach)

    # The stub video download finishes at once; a slow decode (it runs on
    # a worker thread) makes sure it lands before the transcription ends.
    real_transcribe = fake_model.transcribe

    def _slow(path, **kw):
        time.sleep(0.3)
        return real_transcribe(path, **kw)
    monkeypatch.setattr(fake_model, "transcribe", _slow)

    r = _post_url(client, keep_video="true", progress_id=_PID)
    assert r.status_code == 200, r.text
    assert r.json().get("source_video_media_id")
    assert open_at_attach == [True]
    assert _PID not in tx_progress._BATCH_PROGRESS   # closed after it


def test_handler_closes_the_entry_when_the_video_attach_gave_up(
        client, video_enabled, fake_model, monkeypatch):  # noqa: F811
    """The run finishes first, then the video lands while the handler's job
    finish is still on its thread: the task's attach sees the row "running",
    gives up and leaves the close to the handler, which attaches after its
    finish and must then close the entry itself (not leave it to the sweep)."""
    import asyncio

    release = video_enabled._video_gate["make"]()
    video_enabled._video_gate["release"] = release
    attach_calls: list = []
    real_attach = media_video._jobs_attach_video_sync

    def _attach(pid, state):
        attach_calls.append(pid)
        if len(attach_calls) == 1:
            return False        # the task's side: row still "running"
        return real_attach(pid, state)
    monkeypatch.setattr(media_video, "_jobs_attach_video_sync", _attach)

    real_finish = tx_progress._jobs_finish

    async def _slow_finish(pid, **kw):
        # The video lands (and its attach gives up) while the finish is
        # still pending.
        release.set()
        for _ in range(500):
            if attach_calls:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)   # let the video task return
        await real_finish(pid, **kw)
    monkeypatch.setattr(tx_progress, "_jobs_finish", _slow_finish)

    r = _post_url(client, keep_video="true", progress_id=_PID)
    assert r.status_code == 200, r.text
    assert r.json().get("source_video_pending") is True
    assert len(attach_calls) == 2
    assert _PID not in tx_progress._BATCH_PROGRESS
    res = client.get(f"/v1/jobs/{_PID}/result")
    assert res.status_code == 200, res.text
    assert res.json().get("source_video_media_id")


def test_task_closes_the_entry_when_the_handler_passed_its_check_first(
        client, video_enabled, fake_model, monkeypatch):  # noqa: F811
    """The mirror race: the task's attach reads the row still "running" and
    gives up, but the handler's finish commits and its continuation runs the
    done() check BEFORE the task resumes — so the handler skips its fallback
    attach and close. The `finish_landed` flag tells the task the fallback
    is its own: it retries the (idempotent) attach and closes the entry."""
    import asyncio
    import threading

    release = video_enabled._video_gate["make"]()
    video_enabled._video_gate["release"] = release
    attach_calls: list = []
    finish_returned = threading.Event()
    real_attach = media_video._jobs_attach_video_sync

    def _attach(pid, state):
        attach_calls.append(pid)
        if len(attach_calls) == 1:
            # The task's first attach read "running"; it only returns once
            # the handler's finish has returned, so the task resumes after
            # the handler's continuation (same loop step) passed its check.
            finish_returned.wait(10)
            return False
        return real_attach(pid, state)
    monkeypatch.setattr(media_video, "_jobs_attach_video_sync", _attach)

    real_finish = tx_progress._jobs_finish

    async def _finish_then_return_first(pid, **kw):
        release.set()
        for _ in range(500):
            if attach_calls:
                break
            await asyncio.sleep(0.01)
        await real_finish(pid, **kw)
        finish_returned.set()
    monkeypatch.setattr(tx_progress, "_jobs_finish", _finish_then_return_first)

    r = _post_url(client, keep_video="true", progress_id=_PID)
    assert r.status_code == 200, r.text
    assert r.json().get("source_video_pending") is True
    deadline = time.monotonic() + 10
    while _PID in tx_progress._BATCH_PROGRESS and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(attach_calls) == 2           # the task retried the attach
    assert _PID not in tx_progress._BATCH_PROGRESS
    res = client.get(f"/v1/jobs/{_PID}/result")
    assert res.status_code == 200, res.text
    assert res.json().get("source_video_media_id")   # pending flag swapped
