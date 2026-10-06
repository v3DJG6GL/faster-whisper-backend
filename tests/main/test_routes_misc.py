"""Misc app-level routes: /v1/models, /logs, /sev, /auth/whoami."""

from tests.conftest import bearer
from faster_whisper_backend.admin import logs_routes
from faster_whisper_backend.transcription import catalog_routes as tx_catalog_routes


def test_v1_models_requires_a_user(client, make_user_key):
    # User-tier auth like its /v1 siblings: the payload carries the build
    # version, the per-process boot_id and the whole ALLOWED_MODELS list.
    make_user_key("root", is_admin=True)   # locks the server down
    assert client.get("/v1/models").status_code == 401
    _uid, raw = make_user_key("alice")
    assert client.get("/v1/models", headers=bearer(raw)).status_code == 200


def test_v1_models_shape(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert "boot_id" in body and isinstance(body["boot_id"], str)
    # Build identity travels with the model list so clients can display
    # "faster-whisper-backend · <version>" (exact version varies per build).
    assert body["server_name"] == "faster-whisper-backend"
    assert isinstance(body["server_version"], str) and body["server_version"]
    assert isinstance(body["data"], list)
    for entry in body["data"]:
        assert entry["object"] == "model"
        assert "id" in entry
        assert "loaded" in entry  # bool flag
        assert isinstance(entry["loaded"], bool)
    # No model is loaded in the harness (preload neutralised), so every
    # listed model reports loaded=False.
    assert all(e["loaded"] is False for e in body["data"])


def test_v1_models_name_each_device(client, app_module, monkeypatch):
    from faster_whisper_backend.runtime import model_registry
    monkeypatch.setattr(app_module.cfg, "DEFAULT_MODEL", "tiny")
    monkeypatch.setattr(app_module.cfg, "ALLOWED_MODELS", {"tiny", "small"})
    monkeypatch.setattr(app_module.cfg, "MODEL_DEVICE", "cuda")
    monkeypatch.setattr(app_module.cfg, "MODEL_OVERRIDES",
                        {"small": {"MODEL_DEVICE": "cpu"}})
    # Not loaded: where a load would put it (per-model override > global).
    devices = {e["id"]: e["device"] for e in client.get("/v1/models").json()["data"]}
    assert devices == {"tiny": "cuda", "small": "cpu"}
    # Loaded: where it actually sits — a CUDA load that fell back to CPU
    # registers the fallback device.
    monkeypatch.setattr(model_registry, "_loaded_models", {})
    model_registry.register_loaded_model("tiny", None, device="cpu",
                                         compute_type="int8")
    devices = {e["id"]: e["device"] for e in client.get("/v1/models").json()["data"]}
    assert devices["tiny"] == "cpu"


def test_model_device_resolves_auto(app_module, monkeypatch):
    import sys
    import types
    fake = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert tx_catalog_routes._model_device("tiny", "AUTO") == "cuda"
    fake.get_cuda_device_count = lambda: 0
    assert tx_catalog_routes._model_device("tiny", "auto") == "cpu"


def test_logs_page_open_no_auth(client):
    r = client.get("/logs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_sev_shape(client):
    r = client.get("/sev")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"warn", "err", "crit"}
    assert all(isinstance(v, int) for v in body.values())


def test_whoami_open_mode_admin(client):
    r = client.get("/auth/whoami")
    assert r.status_code == 200
    body = r.json()
    assert body["open_mode"] is True
    assert body["is_admin"] is True
    assert "permissions" in body and "pages" in body["permissions"]


def test_logs_older_open(client):
    # /logs/older needs the 'logs' scope='all'. In open mode the synthetic
    # admin bypasses the page gate, so it returns the pagination envelope.
    r = client.get("/logs/older")
    assert r.status_code == 200
    body = r.json()
    assert "lines" in body and "next_skip" in body


# ---------------------------------------------------------------------------
# _security_headers_mw — the outermost response-header layer
# ---------------------------------------------------------------------------

def test_security_headers_on_every_response(client):
    r = client.get("/")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    # No script-src / default-src: every page relies on inline <script>, an
    # inline onclick= in the shared header, blob: AudioWorklets and data: SVGs.
    csp = r.headers["Content-Security-Policy"]
    assert "script-src" not in csp and "default-src" not in csp


def test_data_responses_default_to_no_store(client):
    assert client.get("/").headers["Cache-Control"] == "no-store"
    # ...including the early returns from the inner middlewares: the header
    # layer is registered last (outermost) so it wraps _max_body_mw's 413...
    r = client.put("/v1/synced-client-settings", content=b"x" * (8 * 1024 * 1024),
                   headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}
    # ...and _csrf_mw's cross-origin 403.
    r403 = client.post("/auth/logout", headers={"Origin": "http://evil.example"})
    assert r403.status_code == 403
    assert r403.json() == {"detail": "Origin not allowed for this host"}
    for early in (r, r403):
        assert early.headers["Cache-Control"] == "no-store"
        assert early.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in early.headers["Content-Security-Policy"]


def test_static_assets_stay_cacheable(client):
    r = client.get("/static/favicon.svg")
    assert r.status_code == 200
    assert r.headers.get("Cache-Control") is None


def test_an_explicit_cache_control_is_not_overridden(client):
    # /settings sends a stronger value of its own; the middleware defaults,
    # it does not stamp over a handler's deliberate choice.
    cc = client.get("/settings").headers["Cache-Control"]
    assert "must-revalidate" in cc


def test_logs_older_past_the_page_cap_terminates_paging(client, app_module,
                                                        monkeypatch):
    # The _LOG_OLDER_MAX_PAGES cap ends paging (empty page, next_skip=null)
    # instead of clamping skip — a clamped skip re-served the same window on
    # every further "Load older" click while the viewer's own cursor grew.
    monkeypatch.setattr(app_module.cfg, "LOG_VIEWER_INITIAL_LINES", 10,
                        raising=False)
    cap = 10 * logs_routes._LOG_OLDER_MAX_PAGES
    # The real test log holds a handful of lines, which would end paging on
    # its own: a chain that always has more makes the cap the only stop.
    calls: list = []

    def _endless_chain(path, *, skip, want):
        calls.append(skip)
        return ["x"] * want, skip + want
    monkeypatch.setattr(logs_routes, "_read_chain_window", _endless_chain)
    for skip in (cap + 10, cap):
        r = client.get(f"/logs/older?skip={skip}")
        assert r.status_code == 200
        assert r.json() == {"lines": [], "next_skip": None}
    assert calls == []          # the cap path does no disk I/O
    # One page below the cap is served, but its next_skip lands on the cap.
    r = client.get(f"/logs/older?skip={cap - 10}")
    assert r.json() == {"lines": ["x"] * 10, "next_skip": None}
    assert calls == [cap - 10]


def test_logs_page_onerror_closes_eventsource(client):
    # The onerror handler must close the EventSource before arming the probe:
    # otherwise the browser's native retry reconnects behind openLogStream()
    # and replays the backlog into an un-cleared DOM.
    body = client.get("/logs").text
    i = body.index("es.onerror")
    j = body.index("es.close()", i)
    k = body.index("setTimeout(probe", i)
    assert j < k


def test_logs_page_live_trim_steps_the_load_older_cursor_back(client):
    # The live-tail DOM cap trims lines off the top; the "Load older" cursor
    # counts lines from the chain head that are in the DOM, so it must step
    # back per trimmed line or the first click skips the trimmed window.
    html = client.get("/logs").text
    trim = html[html.index("while (log.childElementCount > _LOG_DOM_MAX)"):]
    trim = trim[:trim.index("if (!paused) window.scrollTo")]
    assert "_logsSkip--" in trim
    assert "contains('line')" in trim


def test_logs_page_copy_handler_keeps_folded_rows_drops_controls(client):
    # Copying a selection must carry the folded / clamped rows it spans; the
    # handler removes only the controls and the rows the search filter hides.
    html = client.get("/logs").text
    h = html[html.index("document.addEventListener('copy'"):]
    h = h[:h.index("e.preventDefault()")]
    assert "cloneContents()" in h
    assert "'.line.hidden, .fold-ctl, button'" in h
    assert ".folded" not in h

