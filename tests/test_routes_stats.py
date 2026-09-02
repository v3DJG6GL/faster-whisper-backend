"""Integration tests for the /stats router (host-gated dashboard)."""

import json

from starlette.testclient import TestClient

import jobs
import re
import metrics
import pytest
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
    # v2 superset
    assert body["v"] == 2
    assert set(body) >= {"tz", "range", "filter", "totals", "today", "stages",
                         "hours", "breakdown", "models", "compare",
                         "time_saved_s", "scope"}
    assert body["range"]["days"] == 30 and body["bucket"] == "day"
    assert len(body["days"]) == 30                      # dense axis
    assert body["compare"] is None
    # v1 query shapes keep working.
    old = client.get("/stats/usage?days=30&bucket=week&by=key&metric=words").json()
    assert old["bucket"] == "week" and old["by"] == "key" and old["metric"] == "words"
    life = client.get("/stats/usage?days=0").json()
    assert life["range"]["days"] == 1 and life["range"]["first_day"] is None


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
    it, through the shared --stage-vad token /quick-config's .seg-vad also
    uses."""
    html = client.get("/stats").text
    assert re.search(r"\.pipe i\.vad\s+\{ background: var\(--stage-vad\); \}", html)
    assert "'vad'" in html and "'var(--stage-' + key + ')'" in html


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
    # Own-scope lite payloads carry a coarse gpu dict {busy, mem_*} with no
    # util_pct; the cluster must read busy/idle instead of printing "–".
    assert "gpu.busy" in js


def test_stage_hues_have_one_definition():
    """A pipeline stage is the same colour on every backend page and in the
    desktop app: NAV_CSS emits web_common.STAGE_COLORS as --stage-* tokens
    (the app.css values), and the consumers (stats page + stats.js, the
    /logs receipt, /quick-config traces, captures) reference the tokens
    rather than carrying their own hexes."""
    import pathlib
    import main as app_main
    import quick_config_routes
    import stats_routes
    import web_common

    assert set(web_common.STAGE_COLORS) == {
        "downloading", "separating", "vad", "transcribing", "diarizing", "translating"}
    for stage, hexv in web_common.STAGE_COLORS.items():
        assert f"--stage-{stage}: {hexv};" in web_common.NAV_CSS
    # The desktop app's tokens (faster-whisper-frontend/src/app.css, dark).
    assert web_common.STAGE_COLORS["downloading"] == "#d9a45b"
    assert web_common.STAGE_COLORS["separating"] == "#6faed9"
    assert web_common.STAGE_COLORS["transcribing"] == "#93b76f"
    assert web_common.STAGE_COLORS["diarizing"] == "#c68fb4"
    assert web_common.STAGE_COLORS["translating"] == "#4dd0c4"
    js = pathlib.Path(stats_routes.__file__).with_name("static").joinpath("stats.js").read_text(encoding="utf-8")
    for stage in web_common.STAGE_COLORS:
        assert f"var(--stage-{stage})" in stats_routes._STATS_VIEWER_HTML
        assert f"var(--stage-{stage})" in quick_config_routes._QUICK_CONFIG_HTML
    assert "getPropertyValue('--stage-' + k)" in js
    # Job kinds likewise: the desktop app's --c-chart-* palette as --kind-*.
    assert web_common.KIND_COLORS == {"dictation": "#cf7b00", "file": "#3e96ea",
                                      "url": "#d76797", "text": "#6f675c"}
    for kind, hexv in web_common.KIND_COLORS.items():
        assert f"--kind-{kind}: {hexv};" in web_common.NAV_CSS
        assert f"var(--kind-{kind})" in stats_routes._STATS_VIEWER_HTML
    assert "getPropertyValue('--kind-' + k)" in js
    hexes = set(web_common.STAGE_COLORS.values())
    for src in (stats_routes._STATS_VIEWER_HTML, quick_config_routes._QUICK_CONFIG_HTML,
                app_main._LOG_VIEWER_HTML):
        assert not any(h in src for h in hexes), "stage hex copied instead of var(--stage-*)"


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
        stats_routes._build_payload, stats_routes.ADMIN_SCOPE, lite=True))
    assert set(snap) >= {"ts", "jobs", "gpu", "host", "models",
                         "in_flight_transcriptions", "severity"}


def test_stats_stage_vocabulary_is_covered_by_both_renderers(client):
    """Every stage name main.py emits (plus the preload job kind that reaches
    the compact glyph via pipeGlyph's kind fallback) needs both a
    `.pipe i.<name>` CSS rule and a stageColor() map entry, or one of the two
    renderers falls back to unstyled grey."""
    html = client.get("/stats").text
    for name in ("vad", "separating", "transcribing", "diarizing", "translating", "downloading"):
        assert f".pipe i.{name}" in html, name
        assert f"var(--stage-{name})" in html, name
    assert ".pipe i.preload" in html and "'preload' ? 'var(--help)'" in html
    # stageColor() resolves every stage to its shared token, aliases included.
    assert "['vad', 'separating', 'transcribing', 'diarizing', 'translating', 'downloading'].includes(key)" in html
    assert "translate: 'translating', download: 'downloading'" in html


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
    colour, and the Recent-jobs table (which follows the sub-bar's usage
    kinds) must map every job kind from some usage kind — or preload, which
    always shows — or it becomes unfilterable."""
    html = client.get("/stats").text
    m = re.search(r"const RJ_KIND_OF = \{(.*?)\};", html, re.S)
    assert m, "RJ_KIND_OF map missing"
    mapped = set(re.findall(r"'([a-z]+)'", m.group(1)))
    for kind in jobs.KINDS:
        assert f".kindchip.{kind}" in html, kind
        assert kind in mapped or kind == "preload", kind
    assert ".concat(['preload'])" in html


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


# ---------------------------------------------------------------------------
# StatsScope: the "own" page scope (decisions 1-3, 2026-09-02)
# ---------------------------------------------------------------------------

_MACHINE_KEYS = {"gpu", "gpu_error", "host", "process", "models", "model_loads",
                 "preload", "uptime_sec", "requests", "errors_total",
                 "errors_window", "latency_ms", "in_flight_transcriptions"}


def _user(scope, uid="ua", is_admin=False):
    import auth
    return {"user_id": uid, "username": "alice", "key_id": "ka",
            "is_admin": is_admin,
            "permissions": auth.Permissions({"pages": {"stats": scope}},
                                            is_admin)}


def _seed_jobs():
    a = jobs.job_start("transcribe", model="m", user="alice", user_id="ua",
                       detail="alice.wav")
    b = jobs.job_start("transcribe", model="m", user="bob", user_id="ub",
                       detail="bob.wav")
    return a, b


def test_stats_scope_rules(client, app_module, monkeypatch):
    import stats_routes
    admin = stats_routes.stats_scope_for(_user("all", is_admin=True))
    assert admin == stats_routes.ADMIN_SCOPE
    preview = stats_routes.stats_scope_for(_user("all", is_admin=True),
                                           preview_user_id="ub")
    assert (preview.scope, preview.user_id, preview.include_identity,
            preview.sees_machine) == ("own", "ub", True, True)
    all_ = stats_routes.stats_scope_for(_user("all"))
    assert (all_.scope, all_.user_id, all_.include_identity,
            all_.sees_machine) == ("all", None, False, True)
    assert all_.viewer_user_id == "ua"
    own = stats_routes.stats_scope_for(_user("own"))
    assert (own.scope, own.user_id, own.include_identity,
            own.sees_machine) == ("own", "ua", True, False)
    monkeypatch.setattr(app_module.cfg, "STATS_OWN_SHOWS_MACHINE", True)
    assert stats_routes.stats_scope_for(_user("own")).sees_machine is True


def test_own_scope_full_payload_is_coarse(client, tx_store):
    """Own scope: only the caller's jobs and recent rows, identities on
    (they are all theirs), the machine keys replaced by the `server`
    block."""
    import stats_routes
    metrics.record_transcription("m", 1.0, 0.5, "ok", 3, request_id="a1",
                                 user_id="ua")
    metrics.record_transcription("m", 2.0, 0.5, "ok", 3, request_id="b1",
                                 user_id="ub")
    a, b = _seed_jobs()
    try:
        snap = stats_routes._build_payload(
            stats_routes.stats_scope_for(_user("own")))
    finally:
        jobs.job_end(a); jobs.job_end(b)
    assert snap["scope"] == "own" and snap["machine"] is False
    assert not (_MACHINE_KEYS & set(snap))
    assert set(snap["server"]) == {"gpu", "models_loaded"}
    assert set(snap["server"]["gpu"]) == {"present", "busy", "mem_used_mb",
                                          "mem_total_mb"}
    assert snap["server"]["gpu"]["busy"] is True      # a job was running
    assert [j["id"] for j in snap["jobs"]] == [a]
    assert snap["jobs"][0]["user"] == "alice"          # own rows keep identity
    assert [r["audio_dur"] for r in snap["recent_transcriptions"]] == [1.0]


def test_own_scope_lite_payload_scoped(client):
    import stats_routes
    a, b = _seed_jobs()
    try:
        snap = stats_routes._build_payload(
            stats_routes.stats_scope_for(_user("own")), lite=True)
    finally:
        jobs.job_end(a); jobs.job_end(b)
    assert set(snap) == {"ts", "scope", "machine", "jobs", "severity", "gpu",
                         "models", "server"}
    assert set(snap["gpu"]) == {"busy", "mem_used_mb", "mem_total_mb"}
    assert snap["models"] == []
    assert [j["id"] for j in snap["jobs"]] == [a]


def test_own_scope_toggle_restores_machine(client, app_module, monkeypatch):
    """The /settings switch: machine keys come back for own-scope viewers,
    the job filter stays."""
    import stats_routes
    monkeypatch.setattr(app_module.cfg, "STATS_OWN_SHOWS_MACHINE", True)
    a, b = _seed_jobs()
    try:
        snap = stats_routes._build_payload(
            stats_routes.stats_scope_for(_user("own")))
    finally:
        jobs.job_end(a); jobs.job_end(b)
    assert snap["machine"] is True and "server" not in snap
    assert {"gpu", "host", "process", "latency_ms", "preload"} <= set(snap)
    assert [j["id"] for j in snap["jobs"]] == [a]


def test_nonadmin_all_scope_unchanged(client):
    """stats="all" for a non-admin is today's behaviour: every job, machine
    visible, identities scrubbed — plus the cancel handle on their own row."""
    import stats_routes
    a, b = _seed_jobs()
    jobs.job_update(a, progress_id="pid-a")
    jobs.job_update(b, progress_id="pid-b")
    try:
        snap = stats_routes._build_payload(
            stats_routes.stats_scope_for(_user("all")))
    finally:
        jobs.job_end(a); jobs.job_end(b)
    assert snap["scope"] == "all" and snap["machine"] is True
    assert {"gpu", "host", "latency_ms"} <= set(snap)
    rows = {j["id"]: j for j in snap["jobs"]}
    assert set(rows) == {a, b}
    assert "user" not in rows[a] and "detail" not in rows[b]
    assert rows[a]["progress_id"] == "pid-a"
    assert "progress_id" not in rows[b]


def test_snapshot_route_own_user_over_http(client, make_user_key):
    from conftest import bearer
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"stats": "own"})
    a = jobs.job_start("transcribe", model="m", user="alice", user_id=uid)
    b = jobs.job_start("transcribe", model="m", user="bob", user_id="other")
    try:
        snap = client.get("/stats/snapshot", headers=bearer(raw)).json()
    finally:
        jobs.job_end(a); jobs.job_end(b)
    assert snap["scope"] == "own" and snap["machine"] is False
    assert [j["id"] for j in snap["jobs"]] == [a]
    assert "gpu" not in snap and "server" in snap
    lite = client.get("/stats/snapshot?lite=1", headers=bearer(raw)).json()
    assert set(lite["gpu"]) == {"busy", "mem_used_mb", "mem_total_mb"}


def test_stream_rechecks_version(client, make_user_key, app_module):
    """The stream re-resolves its StatsScope when the config version moves
    (a permission edit) and ends only when the caller lost access."""
    import inspect
    import config_store
    import stats_routes
    from fastapi import HTTPException
    from test_sse_auth_shared import _fake_request
    from conftest import bearer

    src = inspect.getsource(stats_routes.stats_stream)
    assert "_rescope_on_version_change(" in src
    assert "config_store.config_version()" in src

    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"stats": "all"})
    req = _fake_request(headers=bearer(raw))
    seen = config_store.config_version()
    assert stats_routes._rescope_on_version_change(req, seen) is None
    import api_keys_store
    api_keys_store.set_user_permissions(uid, {"pages": {"stats": "own"}})
    res = stats_routes._rescope_on_version_change(req, seen)
    assert res is not None
    scope, seen2 = res
    assert scope.scope == "own" and scope.user_id == uid
    assert seen2 != seen
    api_keys_store.set_user_permissions(uid, {"pages": {"stats": "none"}})
    with pytest.raises(HTTPException):
        stats_routes._rescope_on_version_change(req, seen2)


# ---------------------------------------------------------------------------
# /stats/usage under StatsScope (own → own rows; all → scrubbed; admin → named)
# ---------------------------------------------------------------------------

def _seed_usage(app_module, alice, bob, alice_key, bob_key):
    import usage_store as us
    h = us.now_hour()
    us.record_usage(key_id=alice_key, user_id=alice, audio_s=10.0, words=5,
                    status="ok", hour=h)
    us.record_usage(key_id=bob_key, user_id=bob, audio_s=90.0, words=5,
                    status="ok", hour=h)


def _two_users(client, make_user_key):
    import api_keys_store
    _, raw_admin = make_user_key("root", is_admin=True)
    alice, raw_alice = make_user_key("alice", pages={"stats": "own"})
    bob, raw_bob = make_user_key("bob", pages={"stats": "all"})
    keys = {}
    for uid in (alice, bob):
        keys[uid] = api_keys_store.list_keys(uid)[0]["id"]
    return raw_admin, alice, raw_alice, bob, raw_bob, keys


def test_usage_own_scope_by_user_is_403(client, app_module, make_user_key):
    from conftest import bearer
    _, alice, raw_alice, bob, _, keys = _two_users(client, make_user_key)
    _seed_usage(app_module, alice, bob, keys[alice], keys[bob])
    r = client.get("/stats/usage?by=user", headers=bearer(raw_alice))
    assert r.status_code == 403
    # An unknown `by` normalises to "user" BEFORE the scope check.
    assert client.get("/stats/usage?by=zzz",
                      headers=bearer(raw_alice)).status_code == 403
    # ?user= is never honoured for a non-admin.
    assert client.get(f"/stats/usage?by=key&user={bob}",
                      headers=bearer(raw_alice)).status_code == 403


def test_usage_own_scope_by_key_only_own_keys(client, app_module,
                                               make_user_key):
    from conftest import bearer
    _, alice, raw_alice, bob, _, keys = _two_users(client, make_user_key)
    _seed_usage(app_module, alice, bob, keys[alice], keys[bob])
    body = client.get("/stats/usage?by=key", headers=bearer(raw_alice)).json()
    assert body["scope"] == "own"
    assert [r["id"] for r in body["leaderboard"]] == [keys[alice]]
    assert body["leaderboard"][0]["me"] is True
    assert body["leaderboard"][0]["user_label"] == "alice"
    # The axis/series is the caller's own total, not the server's.
    assert sum(v for line in body["lines"] for v in line["values"]) == 10.0


def test_usage_all_scope_nonadmin_scrubs_names_and_marks_me(
        client, app_module, make_user_key):
    from conftest import bearer
    _, alice, _, bob, raw_bob, keys = _two_users(client, make_user_key)
    _seed_usage(app_module, alice, bob, keys[alice], keys[bob])
    body = client.get("/stats/usage?by=user", headers=bearer(raw_bob)).json()
    assert body["scope"] == "all"
    rows = {r["id"]: r for r in body["leaderboard"]}
    assert set(rows) == {alice, bob}
    assert rows[bob]["label"] == "bob" and rows[bob]["me"] is True
    assert re.fullmatch(r"user-[0-9a-f]{8}", rows[alice]["label"])
    assert "me" not in rows[alice]
    again = client.get("/stats/usage?by=user", headers=bearer(raw_bob)).json()
    assert {r["id"]: r["label"] for r in again["leaderboard"]}[alice] == \
        rows[alice]["label"]
    by_key = client.get("/stats/usage?by=key", headers=bearer(raw_bob)).json()
    krows = {r["id"]: r for r in by_key["leaderboard"]}
    assert re.fullmatch(r"key-[0-9a-f]{8}", krows[keys[alice]]["label"])
    assert re.fullmatch(r"user-[0-9a-f]{8}", krows[keys[alice]]["user_label"])
    assert krows[keys[bob]]["user_label"] == "bob"
    me_lines = [ln for ln in by_key["lines"] if ln.get("me")]
    assert [ln["id"] for ln in me_lines] == [keys[bob]]


def test_usage_admin_sees_names_and_can_preview_user(client, app_module,
                                                      make_user_key):
    from conftest import bearer
    raw_admin, alice, _, bob, _, keys = _two_users(client, make_user_key)
    _seed_usage(app_module, alice, bob, keys[alice], keys[bob])
    body = client.get("/stats/usage?by=user", headers=bearer(raw_admin)).json()
    assert {r["label"] for r in body["leaderboard"]} == {"alice", "bob"}
    assert body["scope"] == "all"
    prev = client.get(f"/stats/usage?by=key&user={alice}",
                      headers=bearer(raw_admin)).json()
    assert prev["scope"] == "own"
    assert [r["id"] for r in prev["leaderboard"]] == [keys[alice]]
    assert prev["leaderboard"][0]["label"]   # named: the admin is looking


# ---------------------------------------------------------------------------
# Page chrome for the own scope (string pins: the inline JS has no harness)
# ---------------------------------------------------------------------------

def test_stats_page_ships_own_scope_chrome(client):
    html = client.get("/stats").text
    assert 'id="own-server"' in html
    assert 'id="scope-pill"' in html and 'your usage' in html
    assert "GS_LAYOUT_KEY += '-own'" in html
    assert "if (snap.machine === false) {" in html
    js = open("static/stats.js", encoding="utf-8").read()
    assert "not available for your scope" in js


def test_stats_page_removes_machine_tiles_for_own(client):
    """Every machine tile id must be in the removal list, or an own-scope
    viewer keeps an empty card the payload no longer feeds."""
    html = client.get("/stats").text
    assert "grid.removeWidget(el, true)" in html
    lst = html.split("const MACHINE_TILES = [")[1].split("];")[0]
    for gs_id in ("gpu", "cpu", "ram", "process", "activity", "errors",
                  "latency", "endpoints", "models"):
        assert f"'{gs_id}'" in lst, gs_id
        assert f'gs-id="{gs_id}"' in html, gs_id


def test_stats_usage_v2_params(client, app_module):
    import usage_store as us
    h = us.now_hour()
    us.record_usage(key_id="k1", user_id="alice", audio_s=10.0, words=5,
                    status="ok", hour=h, proc_s=2.0, job_id="j1", kind="file",
                    stages=[{"name": "diarizing", "secs": 1.0, "speakers": 2}],
                    model="large-v3")
    us.record_usage(key_id="k2", user_id="bob", audio_s=4.0, words=5,
                    status="ok", hour=h, proc_s=1.0, job_id="j2", kind="dictation")
    kinds = client.get("/stats/usage?by=kind&metric=proc_s&compare=prev").json()
    assert kinds["by"] == "kind" and kinds["metric"] == "proc_s"
    # Every kind is a line (stable series identity); only two carry data.
    assert {ln["id"] for ln in kinds["lines"]} == {"dictation", "file", "url", "text"}
    assert {ln["id"] for ln in kinds["lines"] if sum(ln["values"])} == {"file", "dictation"}
    assert kinds["compare"]["mode"] == "prev" and kinds["compare"]["range"]["days"] == 30
    assert kinds["totals"]["all"]["proc_s"] == 3.0
    assert kinds["leaderboard"][0]["proc_s"] == 2.0     # flat v1 metrics too
    sessions = client.get("/stats/usage?by=model&metric=sessions").json()
    assert sessions["breakdown"]["source"] == "jobs"
    assert {r["id"] for r in sessions["leaderboard"]} == {"large-v3", "(unknown)"}
    assert [m["model"] for m in sessions["models"]][0] == "large-v3"
    stage = client.get("/stats/usage?by=stage&with=diarizing").json()
    assert [ln["id"] for ln in stage["lines"]] == ["diarizing"]
    assert stage["range"]["source"] == "jobs"
    keyed = client.get("/stats/usage?by=key&key=k1").json()
    assert [r["id"] for r in keyed["leaderboard"]] == ["k1"]
    assert keyed["filter"]["key_id"] == "k1"
    span = client.get("/stats/usage?from=20000&to=20006&bucket=month").json()
    assert {k: span["range"][k] for k in ("from", "to", "days", "source")} == {
        "from": 20000, "to": 20006, "days": 7, "source": "rollups"}
    assert span["bucket"] == "month" and len(span["days"]) == 1
    assert client.get("/stats/usage?with=decoding").status_code == 422
    assert client.get("/stats/usage?from=20007&to=20006").status_code == 422
    assert client.get("/stats/usage?from=x").status_code == 422
    zone = client.get("/stats/usage?tz=Europe/Zurich").json()
    assert zone["tz"] == "Europe/Zurich"
    assert client.get("/stats/usage?tz=Mars/Olympus").json()["tz"] == "local"


def test_snapshot_models_carry_size_meta(client, monkeypatch):
    """Loaded-model rows gain the ledger size with its provenance and the
    on-disk weight; the lookup is held for a minute because the stream
    rebuilds the payload every second and disk_size walks a directory."""
    import model_sizes
    import stats_routes
    import system_stats
    calls = {"lookup": 0, "disk": 0}

    def _lookup(name, device, compute_type):
        calls["lookup"] += 1
        return {"bytes": 3_000_000_000, "src": "proxy", "n": 2, "ts": 1.0}

    def _disk(name):
        calls["disk"] += 1
        return 1_500_000_000
    monkeypatch.setattr(model_sizes, "lookup", _lookup)
    monkeypatch.setattr(model_sizes, "disk_size", _disk)
    stats_routes._size_meta_cache.clear()
    system_stats.register_loaded_model("large-v3", 2_000_000_000, "cuda", "float16")
    try:
        snap = client.get("/stats/snapshot").json()
        row = next(m for m in snap["models"] if m["name"] == "large-v3")
        assert row["size_bytes"] == 3_000_000_000 and row["size_src"] == "proxy"
        assert row["disk_bytes"] == 1_500_000_000
        client.get("/stats/snapshot")
        assert calls == {"lookup": 1, "disk": 1}
        assert "size_src" not in client.get("/stats/snapshot?lite=1").json()["models"][0]
    finally:
        system_stats.unregister_loaded_model("large-v3")
        stats_routes._size_meta_cache.clear()


def test_stats_page_loads_static_stats_js(client):
    """The GridStack layout and the usage section live in static/stats.js,
    linked with the build version so the cacheable /static mount serves a
    fresh copy per build, and placed BEFORE the inline dashboard IIFE that
    reads its globals."""
    import stats_routes
    html = client.get("/stats").text
    tag = f'<script src="/static/stats.js?v={stats_routes.ASSET_VERSION}"></script>'
    assert tag in html
    assert "__ASSET_V__" not in html
    assert html.index(tag) < html.index("function applyScope(")
    r = client.get("/static/stats.js")
    assert r.status_code == 200
    js = r.text
    assert "const GS_KEY_BASE = 'whisper-stats-layout-v11'" in js
    assert "const grid = GridStack.init({" in js
    assert "function load()" in js
    # Nothing moved twice: the inline page no longer carries either block.
    assert "GridStack.init({" not in html
    assert "function load()" not in html


def test_stats_page_usage_cards_and_scope_bar(client):
    """The usage half: scope bar rows in the header, the three new tiles,
    the usage card's v2 controls, and the models table's usage columns."""
    html = client.get("/stats").text
    for el in ('id="sb-range"', 'id="sb-compare"', 'id="sb-custom"', 'id="sb-kind"',
               'id="sb-with"', 'id="sb-filters"', 'id="sb-summary"'):
        assert el in html, el
    for gs_id in ("headline", "usage", "stages", "hours", "recent"):
        assert f'gs-id="{gs_id}"' in html, gs_id
    assert 'id="usage-view"' in html and 'id="usage-legend"' in html
    assert 'aria-live="polite"' in html
    for v in ("kind", "user", "key", "model", "stage"):
        assert f'<button data-v="{v}"' in html.split('id="usage-by"')[1].split("</div>")[0], v
    for v in ("proc_s", "sessions"):
        assert f'data-v="{v}"' in html, v
    assert "renderModels(snap)" in html and "colspan=\"11\"" in html
    assert "window._fwRerenderModels" in html


def test_stats_js_contract(client):
    """static/stats.js has no unit harness; pin the behaviours the design
    promises: URL-mirrored state, stacked bars via uPlot's bars path, a
    keyboard-scrubbable chart with an aria-live readout, quartile levels for
    the hour grid, the v6 layout key, and the own-scope reload hook."""
    js = open("static/stats.js", encoding="utf-8").read()
    for s in ("history.replaceState", "uPlot.paths.bars(", "function quantileBreaks",
              "function parsePageQuery", "function pageQueryParams",
              "'ArrowLeft', 'ArrowRight', 'Home', 'End'", "usage-live",
              "window.__statsUsage", "window._fwUsageReload",
              "whisper-stats-layout-v11", "compare", "renderStages", "renderHours",
              "not available for your scope"):
        assert s in js, s
    # The stacked series draw top-of-stack first so lower segments paint over.
    assert "series = rows.slice().reverse()" in js
    # Colours follow the entity, never its rank.
    assert "getPropertyValue('--kind-' + k)" in js


def test_stats_layout_presets_edit_mode_and_ring_freeze(client):
    """Layout presets (positions saved per preset), an explicit edit mode
    with keyboard move/resize (GridStack has none), and the live rings'
    shared crosshair that freezes the readouts at the hovered second."""
    html = client.get("/stats").text
    js = open("static/stats.js", encoding="utf-8").read()
    assert 'id="layout-preset"' in html and 'id="edit-layout-btn"' in html
    assert 'id="layout-live"' in html
    for s in ("const GS_PRESETS = {", "staticGrid: true", "function setPreset(",
              "window._fwSetPreset", "'whisper-stats-preset'", "grid.setStatic(!layoutEditing)",
              "grid.update(item, { x, y })", "grid.update(item, { w, h })",
              "GS_KEY_BASE + ':' + name"):
        assert s in js, s
    # Every preset names real tiles.
    lst = js.split("const GS_PRESETS = {")[1].split("};")[0]
    for gs_id in ("gpu", "cpu", "ram", "process", "activity", "errors", "latency",
                  "endpoints", "models", "recent", "headline", "usage", "stages", "hours"):
        assert f"'{gs_id}'" in lst, gs_id
        assert f'gs-id="{gs_id}"' in html, gs_id
    # Own scope forces the full preset before the machine tiles leave.
    assert "window._fwSetPreset('both', false)" in html
    # Rings: one sync key, a freeze hook, timestamp-based re-find each tick.
    assert "sync: { key: 'live'" in html
    assert "hooks: { setCursor: [onSparkHover], draw: [drawInsetLabels] }" in html
    assert "histX.indexOf(frozenTs)" in html
    assert "if (frozenTs != null) { applyFreeze(); refreshStatusPill(); }" in html


def test_lite_payload_and_activity_card_carry_the_gpu_gate(client):
    import stats_routes
    lite = stats_routes._build_payload(stats_routes.ADMIN_SCOPE, lite=True)
    assert set(lite["gpu_gate"]) == {"capacity", "held", "queue_depth", "oldest_wait_s"}
    html = client.get("/stats").text
    assert 'id="gate-meta"' in html and "snap.gpu_gate" in html


def test_transcription_records_the_gate_wait(client):
    """A real request through the handler goes through the timed gate and
    lands a wait_s on its ledger rows (0 with a free slot, never None)."""
    import transcriptions_store
    import usage_store
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")},
                    data={"model": "whisper-1"})
    assert r.status_code == 200, r.text
    row = transcriptions_store.list_recent(limit=1)[0]
    assert row["wait_s"] is not None and row["wait_s"] >= 0.0
    job = usage_store._require_conn().execute(
        "SELECT wait_s FROM usage_jobs ORDER BY created_ts DESC LIMIT 1").fetchone()
    assert job is not None and job["wait_s"] >= 0.0
    import metrics
    assert metrics.gpu_gate is not None and metrics.gpu_gate.held == 0


def test_stats_history_shape_step_cap_and_gate(client, make_user_key):
    import time
    import transcriptions_store
    from conftest import bearer
    now = int(time.time()) // 10 * 10
    transcriptions_store.record_sys_samples(
        [{"ts": now - 600 + i * 10, "gpu_util": float(i), "slot_busy": 0.25}
         for i in range(60)])
    body = client.get(f"/stats/history?metric=gpu_util&from={now - 600}&to={now}").json()
    assert set(body) == {"metric", "from", "to", "step", "t", "avg", "max"}
    assert body["step"] == 10 and len(body["t"]) == 60
    week = client.get(f"/stats/history?metric=slot_busy&from={now - 7 * 86400}&to={now}").json()
    assert week["step"] >= 7 * 86400 // 2000          # ~2 000 points at most
    assert client.get("/stats/history?metric=nope").status_code == 422
    assert client.get(f"/stats/history?metric=gpu_util&from={now}&to={now - 1}").status_code == 422
    # Own scope without the machine cards: no curves.
    make_user_key("root", is_admin=True)
    _, raw = make_user_key("alice", pages={"stats": "own"})
    assert client.get("/stats/history?metric=gpu_util",
                      headers=bearer(raw)).status_code == 403
    _, raw_all = make_user_key("bob", pages={"stats": "all"})
    assert client.get("/stats/history?metric=gpu_util",
                      headers=bearer(raw_all)).status_code == 200


def test_snapshot_carries_slot_busy(client):
    snap = client.get("/stats/snapshot").json()
    assert set(snap["slot_busy"]) == {"pct_1m", "pct_5m", "pct_15m", "samples"}


def test_stats_tail_shape_and_scope(client, app_module, make_user_key):
    import usage_store as us
    from conftest import bearer
    import api_keys_store
    _, raw_admin = make_user_key("root", is_admin=True)
    alice, raw_alice = make_user_key("alice", pages={"stats": "own"})
    bob, raw_bob = make_user_key("bob", pages={"stats": "all"})
    ka = api_keys_store.list_keys(alice)[0]["id"]
    h = us.now_hour()
    us.record_usage(key_id=ka, user_id=alice, audio_s=10.0, words=5, status="error",
                    hour=h, proc_s=2.0, job_id="a1", kind="file", wait_s=3.0,
                    error_class="cuda_oom", error_stage="transcribing")
    us.record_usage(key_id="kb", user_id=bob, audio_s=4.0, words=5, status="ok",
                    hour=h, proc_s=1.0, job_id="b1", kind="dictation", wait_s=0.5)
    body = client.get("/stats/tail", headers=bearer(raw_admin)).json()
    assert set(body) >= {"range", "wait", "turnaround", "failures", "models", "compare", "scope"}
    assert body["scope"] == "all" and body["wait"]["n"] == 2
    assert body["failures"]["by_class"] == {"cuda_oom": 1}
    own = client.get("/stats/tail", headers=bearer(raw_alice)).json()
    assert own["scope"] == "own" and own["wait"]["n"] == 1 and own["wait"]["max"] == 3.0
    assert client.get("/stats/tail?kind=dictation",
                      headers=bearer(raw_bob)).json()["wait"]["n"] == 1
    assert client.get("/stats/tail?kind=zzz", headers=bearer(raw_bob)).status_code == 422
    assert client.get(f"/stats/tail?user={alice}", headers=bearer(raw_bob)).status_code == 403
    prev = client.get(f"/stats/tail?user={alice}", headers=bearer(raw_admin)).json()
    assert prev["scope"] == "own" and prev["wait"]["n"] == 1
    assert client.get("/stats/tail?from=20007&to=20006",
                      headers=bearer(raw_admin)).status_code == 422


def _seed_jobs_pages(n, user_id="ua", **kw):
    import transcriptions_store
    for i in range(n):
        transcriptions_store.record_timing(
            request_id=f"{user_id}-{i}", model="m", audio_dur_s=10.0,
            proc_dur_s=(9.0 if i % 5 == 0 else 1.0), user_id=user_id,
            status=("error" if i % 4 == 0 else "ok"), words_count=1,
            kind="transcribe", created_ts=2000.0 + i, key_label="lbl", **kw)


def test_stats_jobs_pages_with_cursor_and_filters(client):
    _seed_jobs_pages(7)
    first = client.get("/stats/jobs?limit=3").json()
    assert set(first) == {"jobs", "next_cursor", "scope", "running"}
    assert [j["request_id"] for j in first["jobs"]] == ["ua-6", "ua-5", "ua-4"]
    assert first["next_cursor"] == 2004.0
    second = client.get(f"/stats/jobs?limit=3&cursor={first['next_cursor']}").json()
    assert "running" not in second
    assert [j["request_id"] for j in second["jobs"]] == ["ua-3", "ua-2", "ua-1"]
    third = client.get(f"/stats/jobs?limit=3&cursor={second['next_cursor']}").json()
    assert [j["request_id"] for j in third["jobs"]] == ["ua-0"]
    assert third["next_cursor"] is None
    failed = client.get("/stats/jobs?status=failed").json()["jobs"]
    assert [j["request_id"] for j in failed] == ["ua-4", "ua-0"]
    slow = client.get("/stats/jobs?slow_rtf=0.5").json()["jobs"]
    assert [j["request_id"] for j in slow] == ["ua-5", "ua-0"]
    assert client.get("/stats/jobs?kind=zzz").status_code == 422
    assert client.get("/stats/jobs?status=zzz").status_code == 422
    assert client.get("/stats/jobs?limit=999").json()["jobs"].__len__() == 7
    row = first["jobs"][0]
    assert {"request_id", "language", "wait_s", "error_class", "username", "key_label"} <= set(row)
    assert row["username"] == "" and row["key_label"] == "lbl"     # open mode = admin


def test_stats_jobs_scope_rules(client, make_user_key):
    from conftest import bearer
    _, raw_admin = make_user_key("root", is_admin=True)
    alice, raw_alice = make_user_key("alice", pages={"stats": "own"})
    bob, raw_bob = make_user_key("bob", pages={"stats": "all"})
    _seed_jobs_pages(2, user_id=alice)
    _seed_jobs_pages(3, user_id=bob)
    a = jobs.job_start("transcribe", model="m", user="alice", user_id=alice)
    jobs.job_update(a, progress_id="pid-a")
    try:
        own = client.get("/stats/jobs", headers=bearer(raw_alice)).json()
        assert own["scope"] == "own"
        assert {j["request_id"][:len(alice)] for j in own["jobs"]} == {alice}
        assert own["jobs"][0]["key_label"] == "lbl"                # own rows keep identity
        assert [r["id"] for r in own["running"]] == [a]
        assert own["running"][0]["progress_id"] == "pid-a"       # own cancel handle
        all_ = client.get("/stats/jobs", headers=bearer(raw_bob)).json()
        assert all_["scope"] == "all" and len(all_["jobs"]) == 5
        assert all(j["key_label"] == "" for j in all_["jobs"])   # scrubbed
        assert "progress_id" not in all_["running"][0]           # not bob's job
        assert client.get(f"/stats/jobs?user={alice}",
                          headers=bearer(raw_bob)).status_code == 403
        prev = client.get(f"/stats/jobs?user={alice}", headers=bearer(raw_admin)).json()
        assert prev["scope"] == "own" and len(prev["jobs"]) == 2
    finally:
        jobs.job_end(a)


def test_stats_page_jobs_table_v2(client):
    """Quick views, the wait column, the cancel cell on running rows, the
    load-older button and the timeline expansion; the popover and the
    table share one cancel path."""
    html = client.get("/stats").text
    assert 'id="rj-view"' in html and 'id="rj-more"' in html
    assert 'data-label="wait"' in html
    assert "window._fwCancelJob(c.dataset.pid, c)" in html
    assert "fetch('/stats/jobs?' + p.toString()" in html
    assert "rj-stage-row wait" in html and "end to end" in html
    assert "colspan=\"11\"" in html
    import web_common
    js = web_common.ACTIVITY_CLUSTER_JS
    assert "window._fwCancelJob = function(pid, c)" in js


def test_stats_page_ring_scrubber_range_mode_and_tail_cards(client):
    """Phase 3.2: the ring scrubber and live/range seg in the header, the
    sparks fed from /stats/history in range mode, the turnaround and
    failures tiles fed from /stats/tail, turnaround in the headline."""
    html = client.get("/stats").text
    js = open("static/stats.js", encoding="utf-8").read()
    assert 'id="live-range"' in html and 'id="ring-scrub"' in html
    assert "fetch('/stats/history?metric='" in html
    assert "if (!u || liveMode !== 'live') return;" in html
    assert "function scrubTo(" in html and "function backToLive(" in html
    assert "paused · ' + behind + ' s behind" in html
    assert "busy.pct_15m" in html
    for gs_id in ("turnaround", "failures"):
        assert f'gs-id="{gs_id}"' in html, gs_id
    for s in ("fetch('/stats/tail' + tailQuery()", "function renderTurnaround",
              "function renderFailures", "'turnaround', 'failures'", "wait_share",
              "turnaround p50"):
        assert s in js, s


def test_stats_usage_list_filters_kinds_users_keys(client, usage_store_db):
    """The page's filter bar sends comma lists: `kinds` keeps only those
    job kinds (client-side splits AND the server breakdowns), `users` /
    `keys` keep only those owners / keys (an IN clause), every filter is
    echoed under `filter`, and /stats/pick lists the pickable users ranked
    by the measure with the same labels the leaderboard uses."""
    import time
    us = usage_store_db
    h = int(time.time() // 3600)
    us.record_usage(key_id="ka", user_id="alice", audio_s=100.0, words=10,
                    status="ok", hour=h, proc_s=2.0, job_id="a1", kind="file")
    us.record_usage(key_id="ka", user_id="alice", audio_s=50.0, words=5,
                    status="ok", hour=h, proc_s=1.0, job_id="a2", kind="dictation")
    us.record_usage(key_id="kb", user_id="bob", audio_s=30.0, words=3,
                    status="ok", hour=h, proc_s=1.0, job_id="b1", kind="url")
    us.record_usage(key_id="kc", user_id="carla", audio_s=20.0, words=2,
                    status="ok", hour=h, proc_s=1.0, job_id="c1", kind="file")
    everything = client.get("/stats/usage?by=user").json()
    assert everything["totals"]["all"]["audio_s"] == 200.0
    assert everything["filter"]["kinds"] == [] and everything["filter"]["users"] == []

    kinds = client.get("/stats/usage?by=user&kinds=file,url").json()
    assert kinds["totals"]["all"]["audio_s"] == 150.0
    assert kinds["totals"]["dictation"]["audio_s"] == 0.0
    assert kinds["filter"]["kinds"] == ["file", "url"]
    assert {r["id"]: r["audio_s"] for r in kinds["leaderboard"]} == {"alice": 100.0, "bob": 30.0, "carla": 20.0}
    assert client.get("/stats/usage?kinds=file,bogus").status_code == 422

    users = client.get("/stats/usage?by=kind&users=alice,carla").json()
    assert users["totals"]["all"]["audio_s"] == 170.0
    assert users["filter"]["users"] == ["alice", "carla"]
    both = client.get("/stats/usage?by=kind&users=alice,carla&kinds=file").json()
    assert both["totals"]["all"]["audio_s"] == 120.0
    keys = client.get("/stats/usage?by=kind&keys=kb,kc").json()
    assert keys["totals"]["all"]["audio_s"] == 50.0
    one = client.get("/stats/usage?by=kind&keys=kb").json()
    assert one["filter"]["key_id"] == "kb"          # v2 shape for one key

    tail = client.get("/stats/tail?kind=file,url&users=alice,bob").json()
    assert tail["turnaround"]["n"] == 2
    assert client.get("/stats/tail?kind=nope").status_code == 422

    pick = client.get("/stats/pick?dim=user&metric=audio_s&kinds=file").json()
    assert pick["dim"] == "user"
    assert [(r["id"], r["value"]) for r in pick["rows"]] == [("alice", 100.0), ("carla", 20.0)]
    pk = client.get("/stats/pick?dim=key&users=alice").json()
    assert [r["id"] for r in pk["rows"]] == ["ka"] and pk["rows"][0]["user_label"] == "alice"
    assert client.get("/stats/pick?dim=model").status_code == 422


def test_stats_usage_list_filters_respect_scope(client, usage_store_db, make_user_key):
    """Own-scope users cannot widen to other users (users= is 403, and so is
    the user picker); a non-admin 'all' viewer sends the opaque labels it
    was shown and gets the matching rows back."""
    import time
    from conftest import bearer
    us = usage_store_db
    h = int(time.time() // 3600)
    make_user_key("root", is_admin=True)
    alice_uid, alice_raw = make_user_key("alice", pages={"stats": "own"})
    _, viewer_raw = make_user_key("viewer", pages={"stats": "all"})
    us.record_usage(key_id="ka", user_id=alice_uid, audio_s=100.0, words=10,
                    status="ok", hour=h, proc_s=2.0, job_id="a1", kind="file")
    us.record_usage(key_id="kb", user_id="bob", audio_s=30.0, words=3,
                    status="ok", hour=h, proc_s=1.0, job_id="b1", kind="url")
    hdr = bearer(alice_raw)
    assert client.get("/stats/usage?users=bob", headers=hdr).status_code == 403
    assert client.get("/stats/pick?dim=user", headers=hdr).status_code == 403
    r = client.get("/stats/pick?dim=key", headers=hdr).json()
    assert [x["id"] for x in r["rows"]] == ["ka"]
    hdr = bearer(viewer_raw)
    pick = client.get("/stats/pick?dim=user", headers=hdr).json()
    labels = {x["id"]: x["label"] for x in pick["rows"]}
    assert all(re.fullmatch(r"user-[0-9a-f]{8}", v) for v in labels.values())
    bob_label = labels["bob"]
    doc = client.get("/stats/usage?by=kind&users=" + bob_label, headers=hdr).json()
    assert doc["totals"]["all"]["audio_s"] == 30.0
    assert doc["filter"]["users"] == [bob_label]
