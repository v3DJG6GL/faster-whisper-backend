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
