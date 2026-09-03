"""transcribe-from-URL records a `downloading` stage timing so the receipt's
Pipeline table (and the response's `stages`) account for the fetch."""

from __future__ import annotations

import os

import pytest

import url_download
import url_media_store
from url_download import UrlMediaInfo

_URL = "https://www.youtube.com/watch?v=abc123xyz"


@pytest.fixture
def url_enabled(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "URL_DOWNLOAD_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "URL_MEDIA_DIR",
                        str(tmp_path / "url_media"), raising=False)
    url_media_store.startup_reset()

    async def _probe(url, *, timeout):
        return UrlMediaInfo(
            url=url, extractor_key="Youtube", title="A talk", duration=90.0,
            uploader="chan", filesize_approx=4096, is_live=False,
            thumbnail_url=None, ext="m4a", abr=128.0)

    async def _download(url, *, dest_dir, max_bytes=None, timeout=None,
                        progress_cb=None, cancel_check=None):
        path = os.path.join(dest_dir, "media.m4a")
        with open(path, "wb") as f:
            f.write(b"m4a-bytes" * 8)
        return path

    monkeypatch.setattr(url_download, "probe", _probe)
    monkeypatch.setattr(url_download, "download", _download)
    return app_module


def test_url_download_appears_in_response_stages(client, url_enabled):
    r = client.post("/v1/audio/transcriptions",
                    data={"model": "whisper-1", "source_url": _URL,
                          "response_format": "verbose_json"})
    assert r.status_code == 200
    stages = r.json().get("stages") or []
    dl = [s for s in stages if s.get("name") == "downloading"]
    assert len(dl) == 1
    assert dl[0]["detail"] == "Youtube"
    assert isinstance(dl[0]["secs"], float)
    assert stages[0]["name"] == "downloading"


def test_url_policy_rejection_lands_policy_blocked_on_the_ledger(client, url_enabled,
                                                                  monkeypatch):
    """The policy refusal becomes a 400 for the caller; the ledger keeps WHY
    (policy_blocked in the downloading stage), not just "error"."""
    import recent_transcriptions_store
    import usage_store

    async def _refuse(url, *, timeout):
        raise url_download.UrlPolicyError(
            "this site isn't allowed by the server's URL policy")
    monkeypatch.setattr(url_download, "probe", _refuse)
    r = client.post("/v1/audio/transcriptions",
                    data={"source_url": _URL, "model": "whisper-1"})
    assert r.status_code == 400, r.text
    row = recent_transcriptions_store.list_recent(limit=1)[0]
    assert (row["error_class"], row["error_stage"]) == ("policy_blocked", "downloading")
    job = usage_store._require_conn().execute(
        "SELECT error_class, error_stage FROM usage_jobs ORDER BY created_ts DESC LIMIT 1"
    ).fetchone()
    assert tuple(job) == ("policy_blocked", "downloading")


def test_failed_stage_row_carries_its_error_class(url_enabled):
    """A soft-failed stage (the job goes on without it) gets a receipt row
    with the failure class the usage ledger counts; without it a failed
    stage left no row anywhere."""
    import time
    app_module = url_enabled
    row = app_module._failed_stage(
        "diarizing", time.perf_counter() - 1.0, "pyannote/x",
        RuntimeError("CUDA failed with error out of memory"))
    assert row["name"] == "diarizing" and row["model"] == "pyannote/x"
    assert row["error"] == "cuda_oom" and row["detail"] == "failed"
    assert 0.9 <= row["secs"] <= 5.0
    assert app_module._failed_stage("translating", time.perf_counter(), None,
                                    TimeoutError())["error"] == "timeout"
