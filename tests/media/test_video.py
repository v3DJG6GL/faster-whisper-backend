"""media/video.py's keep_video task: when its staging job dir exists, and
the job-row attach / progress-close hand-off with the handler. The
download itself and the ledger are stubbed; the media store runs real."""

import asyncio
import os

from faster_whisper_backend.media import download as udl
from faster_whisper_backend.media import media_store as ums
from faster_whisper_backend.media import video as media_video
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import progress as tx_progress

_RUNG = {"height": 720, "container": "mkv", "approx_bytes": 10}


def _staging_jobs() -> "list[str]":
    return [n for n in os.listdir(ums.staging_dir()) if n.startswith("vid-")]


def _stub_download(monkeypatch, seen: "list[str] | None" = None):
    async def _download_video(url, *, dest_dir, **kw):
        if seen is not None:
            seen.append(dest_dir)
        path = os.path.join(dest_dir, "media.mkv")
        with open(path, "wb") as f:
            f.write(b"v" * 10)
        return path
    monkeypatch.setattr(udl, "download_video", _download_video)


def test_the_staging_job_is_made_only_once_the_download_slot_is_ours(
        monkeypatch):
    """A job dir made while queued kept its creation mtime through the whole
    wait, and the stale-staging reaper (video wall clock + margin) deleted
    a live job queued behind full-timeout downloads."""
    seen: "list[str]" = []
    _stub_download(monkeypatch, seen)

    async def _main():
        sem = asyncio.Semaphore(1)
        monkeypatch.setattr(tx_models, "_url_download_semaphore", sem)
        await sem.acquire()                 # another download holds the slot
        task = asyncio.create_task(media_video._download_video_for_run(
            None, "https://e.test/v", _RUNG, capped=False, user_id=None,
            protect=None, run_finished=[False]))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
        assert _staging_jobs() == []        # queued: nothing to reap
        sem.release()
        return await task
    state = asyncio.run(_main())
    assert state["state"] == "done", state
    assert seen and os.path.basename(seen[0]).startswith("vid-")
    assert _staging_jobs() == []            # removed after register()


def _attach_hand_off(monkeypatch, *, finish_landed):
    _stub_download(monkeypatch)
    attach_calls: "list[str]" = []
    closed: "list[str]" = []

    def _attach(pid, state):
        attach_calls.append(pid)
        # First read: the handler's finish has not committed yet.
        return len(attach_calls) > 1
    monkeypatch.setattr(media_video, "_jobs_attach_video_sync", _attach)
    monkeypatch.setattr(tx_progress, "_progress_close", closed.append)

    async def _main():
        monkeypatch.setattr(tx_models, "_url_download_semaphore",
                            asyncio.Semaphore(1))
        return await media_video._download_video_for_run(
            "pid-1", "https://e.test/v", _RUNG, capped=False, user_id=None,
            protect=None, run_finished=[True], job_row=True,
            finish_landed=finish_landed)
    asyncio.run(_main())
    return attach_calls, closed


def test_an_attach_that_gave_up_after_the_handler_passed_its_check_closes(
        monkeypatch):
    """The finish committed just after the attach's last read and the
    handler's continuation ran first: it saw this task still running and
    skipped its fallback. The task must retry the swap and close."""
    attach_calls, closed = _attach_hand_off(monkeypatch, finish_landed=[True])
    assert attach_calls == ["pid-1", "pid-1"]
    assert closed == ["pid-1"]


def test_an_attach_that_gave_up_before_the_finish_leaves_the_close_to_the_handler(
        monkeypatch):
    attach_calls, closed = _attach_hand_off(monkeypatch, finish_landed=[False])
    assert attach_calls == ["pid-1"]
    assert closed == []


def test_the_video_download_is_capped_at_what_the_store_can_keep(monkeypatch):
    """register() drops a file over RETAINED_MEDIA_MAX_BYTES: the fetch is
    held to it up front instead of downloading the whole file for nothing."""
    monkeypatch.setattr(ums.cfg, "MEDIA_MAX_BYTES", 10_000, raising=False)
    monkeypatch.setattr(ums.cfg, "RETAINED_MEDIA_MAX_BYTES", 1000, raising=False)
    caps: "list[int]" = []

    async def _download_video(url, *, dest_dir, max_bytes, **kw):
        caps.append(max_bytes)
        raise udl.UrlPolicyError("this media exceeds the server's size limit")
    monkeypatch.setattr(udl, "download_video", _download_video)

    async def _main():
        monkeypatch.setattr(tx_models, "_url_download_semaphore",
                            asyncio.Semaphore(1))
        return await media_video._download_video_for_run(
            None, "https://e.test/v", _RUNG, capped=False, user_id=None,
            protect=None, run_finished=[False])
    state = asyncio.run(_main())
    assert caps == [1000]
    assert state["state"] == "failed" and "size limit" in state["error"]
