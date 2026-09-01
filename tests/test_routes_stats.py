"""Integration tests for the /stats router (host-gated dashboard)."""

import json

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


def test_header_activity_cluster_shell_on_every_page(client):
    """The cluster shell rides {{SEV_PILLS}}, so every template gets it;
    it renders as an empty, default-hidden shell (no live values baked into
    the cached page shell) and the JS that fills it ships alongside."""
    import web_common

    frag = web_common.sev_pills_html()
    assert frag.index('id="hact"') < frag.index('class="sevpills"')
    for path in ("/stats", "/logs"):
        html = client.get(path).text
        assert 'id="hact"' in html, path
        assert 'id="hact-pop"' in html, path
        assert '/stats/stream?lite=1' in html, path
        assert '<span id="hact-jobs" class="v">0</span>' in html, path
        # GPU/VRAM values render as placeholders, never live numbers.
        assert 'id="hact-gpuv"' in html, path


def test_header_activity_cluster_inert_on_headerless_hub(client):
    """The hub's status strip is a plain <div> — no <header> — so
    syncAllowed()'s header-scoped selector can never reveal the button. The
    JS must bail before installing its timer/listeners, and the hub CSS hides
    the empty flex item the shared fragment still ships."""
    import web_common

    assert "document.querySelector('header')" in web_common.ACTIVITY_CLUSTER_JS
    html = client.get("/").text
    # a CSS comment mentions "<header>", so assert on the closing tag
    assert "</header>" not in html
    assert ".hub-sev .hact-wrap { display: none; }" in html


def test_stats_page_colours_the_vad_stage(client):
    """main.py emits a "vad" row into stage timings; both /stats renderers of
    the stage vocabulary (the .pipe glyph CSS and stageColor's map) must know
    it, with the same hue /quick-config assigns (.seg-vad #93b76f)."""
    html = client.get("/stats").text
    assert ".pipe i.vad" in html
    assert "vad: '#93b76f'" in html


def test_header_activity_cluster_js_contract():
    """The cluster's inline JS has no unit harness, so pin the load-bearing
    strings: the cancel POST must carry the CSRF header (the cookie-auth
    middleware 403s it otherwise), the VRAM chip must read the `vram_mb`
    field the lite stream actually emits, a fatal EventSource close must be
    retried, and staleness is signalled via `title` (the data-tip CSS
    tooltip is scoped to the .vtag chip)."""
    import web_common

    js = web_common.ACTIVITY_CLUSTER_JS
    cancel = js[js.index("transcriptions/cancel/"):]
    cancel = cancel[:cancel.index(".then(")]
    assert "X-CSRF-Token" in cancel and "_csrfToken()" in cancel
    assert "c.disabled = false" in js
    assert "vram_mb" in js and "vram_bytes" not in js
    assert "readyState === 2" in js
    assert "activity feed stale" in js
    assert "setAttribute('data-tip', 'activity feed stale" not in js
    assert "preload:'pl'" in js
    # The 1 Hz popover rebuild wipes DOM state, so the cancel button's
    # disabled flag must live in the module-scope `cancelling` map.
    assert "cancelling[" in js


def test_nav_css_defines_the_magenta_token():
    """NAV_CSS's own activity cluster paints with var(--magenta) (.hact-ring
    border, .hact-bar.vram fill). Pages that define no --magenta of their own
    (/dictate, /settings/api-keys) rely on NAV_CSS's :root carrying it --
    without it the ring is invisible and the VRAM bar always reads empty."""
    import web_common

    assert "--magenta: #d2a8ff;" in web_common.NAV_CSS


def test_stats_stream_frame_is_built_off_the_loop(client):
    """Each /stats/stream frame is built via asyncio.to_thread (the builder
    does blocking psutil/NVML/SQLite work), so a slow host snapshot cannot
    stall the loop that serves every other request. Driving the endpoint
    itself is not an option — the generator never ends, and TestClient has
    no way to cancel it — so pin the offload at the source and check the
    payload shape the frame carries survives being built in a worker
    thread."""
    import asyncio
    import inspect

    import stats_routes

    src = inspect.getsource(stats_routes.stats_stream)
    assert "await asyncio.to_thread(" in src
    assert "_build_payload" in src.split("await asyncio.to_thread(")[1][:80]

    snap = asyncio.run(asyncio.to_thread(
        stats_routes._build_payload, lite=True, include_identity=False))
    assert set(snap) >= {"ts", "jobs", "gpu", "host", "models",
                         "in_flight_transcriptions", "severity"}


def test_stats_stage_vocabulary_is_covered_by_both_renderers(client):
    """Every stage name main.py emits (plus the preload job kind that reaches
    the compact glyph via pipeGlyph's kind fallback) needs both a
    `.pipe i.<name>` CSS rule and a stageColor() map entry, or one of the two
    renderers falls back to unstyled grey."""
    html = client.get("/stats").text
    expect = {
        "vad": "'#93b76f'", "separating": "'var(--magenta)'",
        "transcribing": "'var(--cyan)'", "diarizing": "'var(--yellow)'",
        "translating": "'var(--green)'", "downloading": "'var(--cyan)'",
        "preload": "'var(--help)'",
    }
    for name, colour in expect.items():
        assert f".pipe i.{name}" in html, name
        assert f"{name}: {colour}" in html, name


def test_recent_jobs_counter_counts_rendered_rows(client):
    """The heading counter must describe the table it heads: running rows
    included and the kind/warnings/user filters applied (all.length), not the
    raw unfiltered finished list (rt.length). The inline JS has no unit
    harness, so pin the strings."""
    html = client.get("/stats").text
    assert 'Recent jobs (<span id="rt-n">0</span> shown)' in html
    assert "$('rt-n').textContent = all.length;" in html
    assert "textContent = rt.length" not in html


def test_stats_page_renders_every_job_kind(client):
    """jobs.KINDS is the single source of truth; each kind needs a kindchip
    colour and a Recent-jobs filter button or it renders as an unstyled,
    unfilterable grey chip."""
    html = client.get("/stats").text
    for kind in jobs.KINDS:
        assert f".kindchip.{kind}" in html, kind
        assert f'data-v="{kind}"' in html, kind


def test_log_stage_colors_flag_reaches_logs_page(client, app_module,
                                                monkeypatch):
    """LOG_STAGE_COLORS is hot-mutable via the settings save path, so it
    must enter the render_page memo key and land in the viewer JS."""
    import web_common

    monkeypatch.setattr(app_module.cfg, "LOG_STAGE_COLORS", True,
                        raising=False)
    assert "const _STAGE_COLORS = true;" in client.get("/logs").text
    monkeypatch.setattr(app_module.cfg, "LOG_STAGE_COLORS", False,
                        raising=False)
    html = client.get("/logs").text
    assert "const _STAGE_COLORS = false;" in html
    assert "{{LOG_STAGE_COLORS}}" not in html
    assert web_common.cfg is app_module.cfg


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


def test_snapshot_running_row_carries_the_username(client, app_module,
                                                    make_user_key, monkeypatch):
    """The running rows and the finished rows share one user column, so a
    job started from a request must register the caller's display name —
    the same value the finished rows show — not the opaque user_id."""
    from conftest import bearer
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    _, raw_admin = make_user_key("root", is_admin=True)
    seen = {}

    async def _fake(segments, targets, *, progress_cb=None, **kwargs):
        seen["jobs"] = jobs.jobs_snapshot(include_identity=True)
        return ([{t: s["text"] for t in targets} for s in segments], [],
                {"model": "org/m", "source": "", "mode": "fluent"})
    monkeypatch.setattr(translation, "translate_segments", _fake)

    r = client.post("/v1/text/translations", headers=bearer(raw_admin),
                    json={"segments": [{"id": 0, "text": "hi"}],
                          "targets": ["en"]})
    assert r.status_code == 200, r.text
    row = next(j for j in seen["jobs"] if j["kind"] == "translate")
    assert row["user"] == "root"
    # /stats/snapshot echoes the registry rows verbatim (pinned above by
    # test_snapshot_carries_running_jobs), so a row parked with a username
    # surfaces as that username to an admin viewer.
    jid = jobs.job_start("transcribe", model="m", user="root")
    try:
        snap = client.get("/stats/snapshot", headers=bearer(raw_admin)).json()
        assert next(j for j in snap["jobs"] if j["id"] == jid)["user"] == "root"
    finally:
        jobs.job_end(jid)
