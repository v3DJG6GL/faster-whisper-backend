"""Integration tests for the /stats router (host-gated dashboard)."""

from starlette.testclient import TestClient

import jobs
import translation


def test_stats_page_loopback_ok(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_stats_snapshot_open_mode_ok(client):
    r = client.get("/stats/snapshot")
    assert r.status_code == 200
    # The snapshot is a JSON object payload built by _build_payload.
    assert isinstance(r.json(), dict)


def test_stats_usage_ok(client):
    r = client.get("/stats/usage")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"days", "metric", "by", "bucket", "lines", "leaderboard"}


# /stats is user-tier (USER_WEBUI_ALLOWED_HOSTS, OPEN by default). Narrow the
# list to loopback so a non-loopback host exercises the host gate (403 before
# the page-permission check).
def test_stats_snapshot_host_gate_rejects_non_loopback(app_module, monkeypatch):
    import config as cfg
    monkeypatch.setattr(
        cfg, "USER_WEBUI_ALLOWED_HOSTS", ["127.0.0.1", "::1"], raising=False
    )
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        r = c.get("/stats/snapshot")
        assert r.status_code == 403


def test_snapshot_carries_running_jobs(client):
    jid = jobs.job_start("download", model="gguf:org/m", user="u1")
    try:
        snap = client.get("/stats/snapshot").json()
        rows = [j for j in snap.get("jobs", []) if j["id"] == jid]
        assert rows and rows[0]["kind"] == "download"
        # Open mode resolves to the synthetic admin → identity included.
        assert rows[0].get("user") == "u1"
    finally:
        jobs.job_end(jid)
    snap = client.get("/stats/snapshot").json()
    assert all(j["id"] != jid for j in snap.get("jobs", []))


def test_snapshot_lite_shape(client):
    snap = client.get("/stats/snapshot?lite=1").json()
    assert set(snap) >= {"ts", "jobs", "gpu", "host", "models",
                         "in_flight_transcriptions", "severity"}
    # The diet payload skips the heavy metrics keys.
    assert "recent_transcriptions" not in snap
    assert "requests" not in snap
    assert set(snap["host"]) == {"cpu_pct", "ram_used_mb", "ram_total_mb",
                                 "ram_pct"}


def test_translate_run_registers_a_job(client, app_module, monkeypatch):
    """The text-translations handler holds a 'translate' job for the run's
    duration — observed from inside a stubbed translate_segments."""
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    seen = {}

    async def _fake(segments, targets, *, progress_cb=None, **kwargs):
        seen["jobs"] = jobs.jobs_snapshot(include_identity=True)
        progress_cb(0.5, "en 1/1", None)
        seen["after_progress"] = jobs.jobs_snapshot()
        return ([{t: s["text"] for t in targets} for s in segments], [],
                {"model": "org/m", "source": "", "mode": "fluent"})
    monkeypatch.setattr(translation, "translate_segments", _fake)

    r = client.post("/v1/text/translations",
                    json={"segments": [{"id": 0, "text": "hi"}],
                          "targets": ["en"]})
    assert r.status_code == 200, r.text
    tr_jobs = [j for j in seen["jobs"] if j["kind"] == "translate"]
    assert tr_jobs and tr_jobs[0]["detail"] == "1 segs → en"
    upd = [j for j in seen["after_progress"] if j["kind"] == "translate"]
    assert upd[0]["progress"] == 0.5 and upd[0]["step"] == "en 1/1"
    assert jobs.jobs_snapshot() == []          # ended by the finally


def test_stats_page_host_gate_rejects_non_loopback(app_module, monkeypatch):
    import config as cfg
    monkeypatch.setattr(
        cfg, "USER_WEBUI_ALLOWED_HOSTS", ["127.0.0.1", "::1"], raising=False
    )
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        assert c.get("/stats").status_code == 403
