"""
/stats system overview dashboard.

Routes (all gated by web_common.require_user_webui_host):

  GET /stats           HTML page (single-file inline-HTML+CSS+JS, mirrors /logs)
  GET /stats/snapshot  one-shot JSON: ts + metrics_snapshot() + system_snapshot() + severity_counts()
  GET /stats/stream    SSE: same JSON, ~1 Hz (1 s data cadence defeats idle-proxy timeouts; no separate keepalive frame)

Access control (user tier): the shell is gated only by the host allowlist
cfg.USER_WEBUI_ALLOWED_HOSTS (loopback always allowed); the data endpoints
stack require_page("stats") so the API key is the inner gate. The dependency
reads cfg at request time so the admin WebUI can broaden access without a
restart.

Live updates: SSE rather than polling so we get free auto-reconnect on
service-restart, matching the /logs page UX.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse

import hmac
import hashlib
import secrets

import build_info
import config as cfg
import config_store
import jobs
import metrics
import model_sizes
import preload
import system_stats
import web_common
import auth
from auth import require_page

router = APIRouter()

_require_stats_host = web_common.require_user_webui_host


def _require_stats_page_sse(request: Request) -> dict[str, Any]:
    """SSE-aware `require_page("stats")` — one line over the shared resolver
    in auth, so this gate can never drift from the Depends path (bearer
    header, then the HttpOnly session cookie EventSource sends by itself;
    open mode only on the admin host allowlist)."""
    return auth.resolve_user_for_page_sse(request, "stats")


@dataclass(frozen=True)
class StatsScope:
    """What one viewer of /stats may see — resolved ONCE per request (or per
    stream connection) from the authenticated record and handed to every
    builder, so the snapshot, the stream and the usage endpoint can never
    disagree about a caller's scope.

      scope            "own" (the caller's own jobs + usage) or "all"
      user_id          owner filter for jobs / recent rows / usage; None = no filter
      viewer_user_id   the caller's own id (cancel handles on their rows)
      include_identity user/key/detail on job rows, names on recent rows —
                       admins, and own-scope viewers (every row is theirs)
      sees_machine     the machine cards (gpu/host/process/latency/endpoints/
                       5xx/models/preload); False replaces them with a
                       coarse `server` block (decision 2, 2026-09-02)
    """
    scope: str
    user_id: str | None
    viewer_user_id: str | None
    include_identity: bool
    sees_machine: bool


ADMIN_SCOPE = StatsScope("all", None, None, True, True)


def stats_scope_for(user: dict[str, Any], *,
                    preview_user_id: str | None = None) -> StatsScope:
    """Resolve a viewer's StatsScope from the record auth._resolve_user
    returns (`user_id`, `is_admin`, `permissions`).

    admin                       → all, every identity, machine visible
    admin + preview_user_id     → "own" for THAT user (the api-keys page's
                                  preview link); still sees the machine —
                                  it is the admin looking, by design
    non-admin, stats="all"      → all, identities scrubbed, machine visible
                                  (today's behaviour)
    non-admin, stats="own"      → own rows only, identities on (they are
                                  all the caller's), machine only when
                                  cfg.STATS_OWN_SHOWS_MACHINE (read at call
                                  time: the /settings switch hot-applies)
    A client-supplied user/scope is never trusted; only the admin preview
    reaches this function as `preview_user_id`."""
    is_admin = bool(user.get("is_admin"))
    caller_uid = user.get("user_id") or None
    if is_admin:
        if preview_user_id:
            return StatsScope("own", preview_user_id, caller_uid, True, True)
        return ADMIN_SCOPE
    perms = user.get("permissions")
    effective = (perms.effective_user_id_for("stats", caller_uid or "")
                 if perms is not None else None)
    if effective:
        return StatsScope(
            "own", effective, caller_uid, True,
            bool(getattr(cfg, "STATS_OWN_SHOWS_MACHINE", False)))
    return StatsScope("all", None, caller_uid, False, True)


def _coarse_server(sysnap: dict[str, Any], any_job_running: bool
                   ) -> dict[str, Any]:
    """The own-scope replacement for the machine cards: enough to act on
    ("is the GPU busy, is there VRAM headroom, is a model loaded"), nothing
    that lets a viewer reconstruct other people's activity — no utilisation
    curve, no per-model list, no request counters."""
    gpu = sysnap.get("gpu") or None
    if gpu:
        util = gpu.get("util_pct")
        busy = any_job_running or (util is not None and util >= 5)
        g = {"present": True, "busy": bool(busy),
             "mem_used_mb": gpu.get("mem_used_mb"),
             "mem_total_mb": gpu.get("mem_total_mb")}
    else:
        g = {"present": False, "busy": bool(any_job_running),
             "mem_used_mb": None, "mem_total_mb": None}
    return {"gpu": g, "models_loaded": len(sysnap.get("models") or [])}


# (name, device, compute_type) → (expires_at, meta). model_sizes.lookup() may
# walk a model directory on disk and the stream builds a payload every
# second, so the answer is held for a minute.
_SIZE_META_TTL_S = 60.0
_size_meta_cache: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}


def _model_size_meta(name: str, device: str, compute_type: str) -> dict[str, Any]:
    """`{size_bytes, size_src, disk_bytes}` for a loaded-models row: the
    ledger's best size with its provenance (measured / proxy / disk) and the
    weight on disk, both None when unknown."""
    key = (name or "", device or "", compute_type or "")
    now = time.monotonic()
    hit = _size_meta_cache.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    rec = model_sizes.lookup(*key)
    meta = {
        "size_bytes": None if rec is None else rec["bytes"],
        "size_src": None if rec is None else rec["src"],
        "disk_bytes": model_sizes.disk_size(name),
    }
    _size_meta_cache[key] = (now + _SIZE_META_TTL_S, meta)
    return meta


def _build_payload(scope: StatsScope = ADMIN_SCOPE, *,
                   lite: bool = False) -> dict[str, Any]:
    """Combine request metrics + system snapshot into one payload for the
    given viewer scope.

    `lite=True` is the header activity cluster's diet: ts, running jobs,
    gpu, host, loaded models, in-flight count and severity — skipping
    metrics_snapshot() and with it the recent-transcriptions SQLite query
    (the cluster polls/streams from EVERY WebUI page, so the full payload
    would multiply that query by the open-tab count).

    Every payload carries `scope` ("own"|"all") and `machine` (bool) so the
    page can shape itself on the first frame. When `scope.sees_machine` is
    False the machine keys are absent and a `server` block (see
    _coarse_server) stands in; the lite variant keeps a coarse `gpu` dict
    {busy, mem_used_mb, mem_total_mb} so the header cluster keeps working."""
    sysnap = system_stats.system_snapshot()
    base = {
        "ts": time.time(),
        "scope": scope.scope,
        "machine": scope.sees_machine,
        "jobs": jobs.jobs_snapshot(include_identity=scope.include_identity,
                                   user_id=scope.user_id,
                                   viewer_user_id=scope.viewer_user_id),
        "severity": web_common.severity_counts(),
    }
    if not scope.sees_machine:
        any_running = bool(jobs.jobs_snapshot())
        server = _coarse_server(sysnap, any_running)
        if lite:
            g = server["gpu"]
            return {
                **base,
                "gpu": {"busy": g["busy"], "mem_used_mb": g["mem_used_mb"],
                        "mem_total_mb": g["mem_total_mb"]},
                "models": [],
                "server": server,
            }
        recent = metrics.metrics_snapshot(
            include_identity=scope.include_identity,
            user_id=scope.user_id)["recent_transcriptions"]
        return {**base, "server": server, "recent_transcriptions": recent}
    # Beside the loaded-model list, and in the lite payload too: the most
    # likely failure of model preloading is SILENCE (no worker, no plans,
    # no loads), which is invisible everywhere else. Five cheap scalars,
    # no identities — assembled here rather than in system_stats so that
    # import-light module needn't reach preload.
    base["preload"] = preload.diagnostics()
    if lite:
        host = sysnap.get("host") or {}
        return {
            **base,
            "gpu": sysnap.get("gpu"),
            "host": {k: host.get(k) for k in
                     ("cpu_pct", "ram_used_mb", "ram_total_mb", "ram_pct")},
            "models": sysnap.get("models"),
            "in_flight_transcriptions": metrics.in_flight_transcriptions,
            "gpu_gate": metrics.gpu_gate_snapshot(),
        }
    models = [
        {**m, **_model_size_meta(m.get("name"), m.get("device"),
                                 m.get("compute_type"))}
        for m in (sysnap.get("models") or [])
    ]
    return {
        **base,
        **metrics.metrics_snapshot(include_identity=scope.include_identity,
                                   user_id=scope.user_id),
        **sysnap,
        "models": models,
    }


def _rescope_on_version_change(request: Request, seen_version: int
                               ) -> tuple[StatsScope, int] | None:
    """Stream helper: when config_store.config_version() moved since
    `seen_version` (a permission edit bumps it), re-resolve the caller and
    return the fresh (scope, version); None when nothing changed. Raises
    HTTPException when the caller lost access, which ends the stream (the
    page reconnects and gets the 401/403).

    Re-resolving rather than ending the stream matters with several workers:
    every sibling commit — including a key's debounced last_used_ts touch —
    bumps the version, and an ended stream makes the page discard its
    two-minute sparkline history on reconnect."""
    current = config_store.config_version()
    if current == seen_version:
        return None
    fresh = auth.resolve_user_for_page_sse(request, "stats")
    return stats_scope_for(fresh), current


@router.get(
    "/stats",
    response_class=HTMLResponse,
    # HTML page is host-only — the bearer isn't available on initial
    # navigation. API endpoints below gate by `require_page("stats")`;
    # the page's first snapshot fetch 403s for non-permitted users.
    dependencies=[Depends(_require_stats_host)],
)
async def stats_page() -> HTMLResponse:
    """Single-file inline HTML page. `no-store` so a browser never serves a
    stale build after a service restart."""
    return HTMLResponse(
        web_common.render_page(_STATS_VIEWER_HTML, current="stats"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get(
    "/stats/snapshot",
    dependencies=[Depends(_require_stats_host)],
)
async def stats_snapshot(
    lite: int = 0,
    user: dict[str, Any] = Depends(require_page("stats")),
) -> dict[str, Any]:
    """One-shot JSON. Useful for scripts and for the page's initial render.
    `?lite=1` returns the header activity cluster's diet payload."""
    return _build_payload(stats_scope_for(user), lite=bool(lite))


# Per-process salt for the opaque labels a non-admin "all" viewer sees on the
# leaderboard: stable within a process (a chart line keeps its name across
# refreshes), meaningless across restarts, and never a plain hash of the id
# (usernames/ids would otherwise be recoverable by table lookup).
_SCRUB_SALT = secrets.token_bytes(16)


def _opaque_user_label(user_id: str) -> str:
    return "user-" + hmac.new(_SCRUB_SALT, (user_id or "").encode(),
                              hashlib.sha256).hexdigest()[:8]


def _opaque_key_label(key_id: str) -> str:
    return "key-" + hmac.new(_SCRUB_SALT, ("k:" + (key_id or "")).encode(),
                             hashlib.sha256).hexdigest()[:8]


@router.get(
    "/stats/usage",
    dependencies=[Depends(_require_stats_host)],
)
async def stats_usage(
    days: int | None = None,
    bucket: str = "auto",
    by: str = "user",
    metric: str = "audio_s",
    tz: str | None = None,
    from_: int | None = Query(default=None, alias="from"),
    to: int | None = None,
    all: bool = False,
    with_: str | None = Query(default=None, alias="with"),
    compare: str = "off",
    key: str | None = None,
    user_q: str | None = Query(None, alias="user"),
    user: dict[str, Any] = Depends(require_page("stats")),
) -> dict[str, Any]:
    """Historical usage, v2: usage_store.overview() — totals, today, stages,
    the hour grid, a dense-axis breakdown of `metric` by `by`, a leaderboard
    over the same entities, an optional comparison window and a per-model
    table. Served once per page load / selector change — NOT part of the
    1 Hz SSE payload.

    Window: `days` (default 30; <=0 = lifetime, the v1 spelling of `all=1`),
    or an explicit inclusive `from`/`to` (days-since-epoch in `tz`), or
    `all=1`; `tz` is an IANA name (server-local when absent). `bucket` ∈
    {auto, day, week, month}; `by` ∈ {user, key, kind, model, stage};
    `metric` ∈ {audio_s, words, requests, errors, proc_s, sessions};
    `compare` ∈ {off, prev, yoy}; `with` narrows to jobs that ran every
    listed stage; `key` narrows the key-bearing tables to one API key.
    422 on an unknown stage or from > to. v1 queries keep working: the v1
    keys (days, metric, by, bucket, lines, leaderboard) keep their meaning
    and the board rows carry their metrics flat as before.

    Scope (see StatsScope): an admin sees every user, named, and may pass
    `?user=<id>` to preview exactly what that user's own scope shows. A
    non-admin with stats="all" sees every user's numbers but opaque
    `user-xxxxxxxx` / `key-xxxxxxxx` labels — except their own row, which
    keeps its name and carries `me: true`. A non-admin with stats="own"
    gets only their own rows; `by=user` is refused (403) because the only
    row would be themselves and the page ranks their keys instead. A
    non-admin passing `?user=` gets 403 — it is never trusted."""
    import api_keys_store
    import usage_store

    # Normalise BEFORE the scope check: an unknown `by` collapses to "user"
    # and must not slip past the own-scope refusal below.
    by = by if by in usage_store.BREAKDOWNS else "user"
    is_admin = bool(user.get("is_admin"))
    if user_q and not is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            detail="user= is admin-only")
    scope = stats_scope_for(user, preview_user_id=(user_q or None)
                            if is_admin else None)
    if scope.scope == "own" and by == "user":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="per-user leaderboard needs stats scope 'all'")
    caller_uid = user.get("user_id") or None
    scrub = not scope.include_identity
    # v1 spelling: days<=0 meant lifetime.
    if days is not None and days <= 0:
        days, all = None, True
    try:
        w = usage_store.parse_window_params(
            days=days, from_day=from_, to_day=to, all_time=all, with_=with_,
            tz=tz)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    jobs_retention = int(getattr(cfg, "USAGE_JOBS_RETENTION_DAYS", 365) or 0)

    # Everything below is store work — a handful of aggregate scans over the
    # rollups (twice with a compare window) plus name lookups. Run the whole
    # gather off the event loop, like the reports/quick-config siblings.
    def _gather() -> dict[str, Any]:
        out = usage_store.overview(
            user_id=scope.user_id, key_id=key or None, tz=w.tz,
            tz_name=w.tz_name, days=w.days, from_day=w.from_day,
            to_day=w.to_day, all_time=w.all_time, with_stages=w.with_stages,
            by=by, metric=metric, bucket=bucket, compare=compare,
            jobs_retention_days=jobs_retention)

        # Resolve display names server-side (the /stats client has no
        # api-keys data). Revoked users/keys still resolve; sentinels stay
        # literal. Non-admin "all" viewers get opaque labels instead — only
        # their own row keeps its name (and is flagged `me`).
        rows = out["leaderboard"]
        names = api_keys_store.get_usernames(
            [r["user_id"] for r in rows if r.get("user_id")])

        def _user_label(uid: str) -> str:
            if scrub and uid != caller_uid:
                return _opaque_user_label(uid)
            return names.get(uid) or uid

        labels: dict[str, dict[str, Any]] = {}
        for r in rows:
            mine = bool(caller_uid) and r.get("user_id") == caller_uid
            if by == "user":
                r["label"] = _user_label(r["id"])
            elif by == "key":
                kid = r["id"]
                if scrub and not mine:
                    r["label"] = _opaque_key_label(kid)
                else:
                    krec = (api_keys_store.get_key(kid)
                            if kid and not kid.startswith("(") else None)
                    lbl = (krec or {}).get("label") or ""
                    disp = (krec or {}).get("key_prefix")
                    r["label"] = (lbl or (disp + "…" if disp else kid))
                r["user_label"] = _user_label(r["user_id"])
            if mine:
                r["me"] = True
            # v1 shape: the metrics flat on the row as well.
            r.update(r["totals"])
            labels[r["id"]] = {k: r[k] for k in ("label", "user_label", "me")
                               if k in r}
        for ln in out["lines"]:
            ln.update(labels.get(ln["id"], {}))
        out["scope"] = scope.scope
        return out

    return await asyncio.to_thread(_gather)


@router.get(
    "/stats/history",
    dependencies=[Depends(_require_stats_host)],
)
async def stats_history(
    metric: str = "gpu_util",
    from_: float | None = Query(default=None, alias="from"),
    to: float | None = None,
    step: int | None = None,
    user: dict[str, Any] = Depends(require_page("stats")),
) -> dict[str, Any]:
    """Range-mode machine history for a live card's "history ↗": one metric
    (gpu_util | gpu_mem_mb | gpu_temp | cpu_pct | ram_pct | slot_busy) from
    the sampler's sys_samples, downsampled to `step` seconds (default: the
    smallest step that keeps the window under ~2 000 points, never below the
    sample cadence). `from`/`to` are epoch seconds; default the last hour.
    Own-scope viewers get it only when they see the machine cards (the
    same rule as the live payload: utilisation curves reveal other
    people's jobs)."""
    import transcriptions_store

    scope = stats_scope_for(user)
    if not scope.sees_machine:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            detail="machine history needs the machine cards")
    if metric not in transcriptions_store.SYS_SAMPLE_METRICS:
        raise HTTPException(422,
                            detail=f"unknown metric: {metric!r}")
    now = time.time()
    t1 = float(to) if to is not None else now
    t0 = float(from_) if from_ is not None else t1 - 3600
    if t0 >= t1:
        raise HTTPException(422,
                            detail="'from' is not before 'to'")
    t0 = max(t0, t1 - 3650 * 86400)
    cadence = max(1, int(getattr(cfg, "STATS_HISTORY_SAMPLE_S", 10) or 10))
    auto = max(cadence, int(-(-(t1 - t0) // 2000)))
    step_s = max(cadence, int(step)) if step else auto
    series = await asyncio.to_thread(
        transcriptions_store.list_sys_samples, metric=metric, from_ts=t0,
        to_ts=t1, step_s=step_s)
    return {"metric": metric, "from": int(t0), "to": int(t1), "step": step_s,
            **series}


@router.get(
    "/stats/stream",
    dependencies=[Depends(_require_stats_host)],
)
async def stats_stream(
    request: Request,
    lite: int = 0,
    user: dict[str, Any] = Depends(_require_stats_page_sse),
) -> StreamingResponse:
    """1 Hz SSE stream of the snapshot payload. The 1-second data cadence
    already counts as traffic for idle-proxy timeout purposes — no separate
    keepalive frame needed. `?lite=1` streams the activity-cluster diet
    payload (see _build_payload).

    The viewer's StatsScope is resolved once here and re-resolved whenever
    the config version moves (a permission edit), so an admin narrowing a
    user's stats scope takes effect on that user's open tab within a tick —
    without the reconnect churn of ending the stream on every bump."""
    _lite = bool(lite)
    scope = stats_scope_for(user)
    seen = config_store.config_version()

    async def gen():
        nonlocal scope, seen
        while True:
            payload = await asyncio.to_thread(
                _build_payload, scope, lite=_lite)
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)
            try:
                fresh = _rescope_on_version_change(request, seen)
            except HTTPException:
                return
            if fresh is not None:
                scope, seen = fresh

    return web_common.sse_response(gen())


# --- HTML template -----------------------------------------------------------
# Single-file, no build step. Mirrors the /logs and /settings style. uPlot is
# loaded from the local /static mount — no CDN, works offline.

_STATS_VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
{{SCALE_BOOTSTRAP_HEAD}}
<link rel="stylesheet" href="/static/uplot.min.css">
<link rel="stylesheet" href="/static/gridstack.min.css">
<script src="/static/uplot.iife.min.js"></script>
<script src="/static/gridstack.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d;
  }
  /* Font tokens, --font-sans, --font-mono and html font-size live in
     NAV_CSS (injected further down). Important: never embed the NAV_CSS
     template placeholder inside another comment block — render_page() does
     a naive string replace and would inject NAV_CSS into this comment,
     prematurely closing it (NAV_CSS contains its own internal comments)
     and silently dropping every CSS rule that follows. Chrome (titles,
     buttons, badges, card headers) uses --font-sans; uPlot's axis labels
     and the spark-head numeric readouts stay in --font-mono so digits
     align (font-variant-numeric: tabular-nums hint relies on the mono
     stack for crisp tabular alignment). */
  html, body { background: var(--bg); color: var(--fg);
    font: 1rem/1.5 var(--font-sans);
    margin: 0; padding: 0; min-height: 100%; }
  input, textarea, select, kbd, code, pre { font-family: var(--font-mono); }
  /* header / .header-inner / .title / page-toolbar controls (buttons,
     pills) are all centralized in NAV_CSS. */
  {{NAV_CSS}}
  .grid { padding: 0.875rem; max-width: 68.75rem; margin: 0 auto;
    box-sizing: border-box; min-height: 60vh; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.625rem 0.75rem; min-width: 0; height: 100%; box-sizing: border-box;
    overflow: auto; display: flex; flex-direction: column; min-height: 0; }
  .card h3 { font-size: var(--fs-xs); color: var(--dim); margin: 0 0 0.375rem;
    text-transform: uppercase; letter-spacing: .05em; font-weight: 500; }
  .card .val { color: var(--bold); font-size: var(--fs-xxl); font-weight: 600; line-height: 1.1; }
  .card .val .sub { color: var(--dim); font-size: var(--fs-sm); font-weight: normal; margin-left: 0.375rem; }
  .card .meta { color: var(--dim); font-size: var(--fs-xs); margin-top: 0.25rem; }
  .card .meta b { color: var(--fg); font-weight: 500; }
  .card .meta .warn { color: var(--yellow); font-weight: 600; }
  .bar { height: 6px; background: #21262d; border-radius: 3px; margin-top: 0.375rem; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--cyan);
    transition: width .3s ease; }
  .bar.warn > i { background: var(--yellow); }
  .bar.crit > i { background: var(--red); }
  .spark-wrap  { margin-top: 0.625rem; min-width: 0;
                 flex: 1 1 0; min-height: 0;
                 display: flex; flex-direction: column; }
  .spark-head  { display: flex; justify-content: space-between; align-items: baseline;
                 font: var(--fs-xs) var(--font-mono);
                 color: var(--dim); margin-bottom: 2px; flex: 0 0 auto; }
  .spark-label { letter-spacing: .03em; text-transform: uppercase; }
  .spark-now.frozen { color: var(--yellow); }
  .u-cursor-x { border-right: 1px dashed var(--yellow) !important; }
  /* Edit-layout mode: dashed tile borders, grab cursor on titles, corners
     visible. Off = tiles are static (see staticGrid) and titles are text. */
  body.layout-edit .grid-stack-item .card { border-style: dashed; border-color: var(--cyan); }
  body.layout-edit .grid-stack-item .card h3 { cursor: grab; }
  body.layout-edit .grid-stack-item .card h3:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }
  body.layout-edit .grid-stack > .grid-stack-item > .ui-resizable-handle { opacity: 0.6; }
  body:not(.layout-edit) .grid-stack-item .card h3 { cursor: default; }
  header #edit-layout-btn.active { color: var(--cyan); border-color: #1f3a5a; background: var(--hover, #1f2630); }
  .spark-now   { color: var(--bold); font-weight: 600;
                 font-variant-numeric: tabular-nums; }
  .spark       { flex: 1 1 0; min-height: 4rem; width: 100%; }
  .uplot, .u-wrap { background: transparent !important; }
  .u-legend { display: none; }
  .u-axis { color: var(--dim); }
  table.tbl { width: 100%; border-collapse: collapse; font-size: var(--fs-sm); }
  table.tbl th, table.tbl td { padding: 0.25rem 0.375rem; text-align: left;
    border-bottom: 1px solid #21262d; }
  table.tbl th { color: var(--dim); font-weight: 500; font-size: var(--fs-xs);
    text-transform: uppercase; }
  table.tbl td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.tbl th.num { text-align: right; }
  .badge { display: inline-block; font-size: 0.667rem; padding: 0.0625rem 0.375rem;
    border-radius: 999px; border: 1px solid var(--border); color: var(--dim); }
  .badge.warm { color: var(--green); border-color: #1f4d2a; }
  .badge.cold { color: var(--yellow); border-color: #4d3e1f; }
  .badge.ok { color: var(--green); border-color: #1f4d2a; }
  .badge.err { color: var(--red); border-color: #5a2424; }
  .ts { color: var(--dim); font-variant-numeric: tabular-nums; }
  .core-strip { display: flex; gap: 2px; margin-top: 0.375rem; height: 1.5rem;
    align-items: flex-end; }
  .core-strip > div { flex: 1; background: var(--cyan); border-radius: 1px;
    min-height: 2px; transition: height .3s ease; }
  .err-strip { display: flex; gap: 0.25rem; margin-top: 0.375rem; }
  .err-strip .seg { flex: 1; text-align: center; padding: 0.375rem;
    background: #21262d; border-radius: 4px; }
  .err-strip .seg b { color: var(--bold); display: block; font-size: var(--fs-xl); }
  .err-strip .seg.hot { background: #2d1414; }
  .err-strip .seg.hot b { color: var(--red); }
  .empty { color: var(--dim); font-style: italic; }
  .hidden { display: none !important; }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip: rect(0 0 0 0); white-space: nowrap; }
  /* --- Scope bar (header rows): range / compare on the first subbar row,
     kind / ran / filter chips + the resolved window on the second. --- */
  header .subbar .seg-label { color: var(--dim); font-size: var(--fs-xs);
    text-transform: uppercase; letter-spacing: .04em; margin-left: 0.4rem; }
  header .subbar-usage { padding-top: 0.25rem; }
  .chips { display: inline-flex; gap: 0.3rem; flex-wrap: wrap; align-items: center; }
  .chip { font: inherit; font-size: var(--fs-xs); border: 1px dashed var(--border);
    border-radius: 999px; padding: 0.05rem 0.55rem; color: var(--dim);
    background: transparent; cursor: pointer; display: inline-flex;
    align-items: center; gap: 0.35rem; white-space: nowrap; }
  .chip .sw { width: 8px; height: 8px; border-radius: 2px; display: inline-block; opacity: .35; }
  .chip.on { border-style: solid; color: var(--fg); background: var(--panel); }
  .chip.on .sw { opacity: 1; }
  .chip.filter { border-style: solid; border-color: #1f3a5a; color: var(--cyan);
    background: #1f2630; }
  .chip.filter .x { color: var(--dim); }
  .chip:focus-visible { outline: 2px solid var(--cyan); outline-offset: 1px; }
  .sb-none { font-size: var(--fs-xs); color: var(--dim); }
  .sb-summary { margin-left: auto; font: var(--fs-xs) var(--font-mono); color: var(--dim);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 60%; }
  .sb-custom { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.5rem; border: 1px solid var(--border); border-radius: 6px;
    background: var(--panel); font-size: var(--fs-sm); }
  .sb-custom label { color: var(--dim); display: inline-flex; gap: 0.3rem; align-items: center; }
  .sb-custom input[type=date] { background: var(--bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px; padding: 0.1rem 0.3rem;
    font: inherit; font-size: var(--fs-xs); }
  .sb-custom .spans { display: inline-flex; gap: 0.25rem; flex-wrap: wrap; }
  .sb-custom button { font: inherit; font-size: var(--fs-xs); background: var(--bg);
    color: var(--dim); border: 1px solid var(--border); border-radius: 4px;
    padding: 0.1rem 0.45rem; cursor: pointer; }
  .sb-custom button:hover { color: var(--fg); }
  .sb-custom button.primary { color: var(--cyan); border-color: #1f3a5a; }
  .sb-custom button:disabled { opacity: .4; cursor: not-allowed; }
  .sb-custom .note { font: var(--fs-xs) var(--font-mono); color: var(--dim); }
  /* --- Usage-fed cards: dim while a refetch is in flight (the previous
     render stays; no skeleton, no jump). --- */
  .usage-fed.updating { opacity: .6; transition: opacity .15s ease; }
  .card h3 .tag { font-family: var(--font-mono); color: var(--cyan); text-transform: none;
    letter-spacing: 0; font-weight: 400; margin-left: 0.3rem; }
  .usage-error { font-size: var(--fs-xs); color: var(--yellow); }
  .usage-error button { font: inherit; font-size: var(--fs-xs); color: var(--fg);
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    padding: 0 0.4rem; cursor: pointer; margin-left: 0.3rem; }
  .usage-chart:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }
  .usage-empty { position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; text-align: center; padding: 1rem; color: var(--dim);
    font-size: var(--fs-sm); }
  .usage-table { flex: 1 1 auto; min-height: 9rem; overflow: auto; }
  .usage-table table { font-family: var(--font-mono); font-size: var(--fs-xs); }
  .usage-table td.dim { color: var(--dim); }
  .usage-legend { display: flex; gap: 0.7rem; flex-wrap: wrap; align-items: center;
    font-size: var(--fs-xs); color: var(--dim); margin: 0.15rem 0 0.4rem; }
  .usage-legend .what { color: var(--dim); }
  .usage-legend button { font: inherit; font-size: var(--fs-xs); background: transparent;
    border: 0; color: var(--fg); display: inline-flex; align-items: center; gap: 0.3rem;
    cursor: pointer; padding: 0; }
  .usage-legend button.off { opacity: .4; text-decoration: line-through; }
  .usage-legend .cmp i { display: inline-block; width: 14px; border-top: 1.5px dashed #8b949e;
    vertical-align: middle; margin-right: 0.3rem; }
  .usage-legend .kb { margin-left: auto; font-family: var(--font-mono); color: var(--dim); }
  .usage-tip .tip-row.tot { border-top: 1px solid var(--border); margin-top: 0.15rem;
    padding-top: 0.15rem; color: var(--bold); font-weight: 600; }
  .usage-tip .tip-row.cmp { color: var(--dim); }
  table.usage-board tr.pick { cursor: pointer; }
  table.usage-board tr.pick:hover td { background: rgba(121, 192, 255, 0.04); }
  table.usage-board tr.pick:focus-visible { outline: 2px solid var(--cyan); outline-offset: -2px; }
  table.usage-board td .share { display: inline-block; height: 5px; border-radius: 3px;
    vertical-align: middle; margin-right: 0.4rem; }
  table.tbl td.err { color: var(--red); }
  .rtf { display: inline-block; padding: 0 0.35rem; border-radius: 999px;
    font: 0.667rem var(--font-mono); border: 1px solid var(--border); color: var(--dim); }
  .rtf.slow { color: var(--yellow); border-color: #4d3e1f; }
  .badge.est { color: var(--magenta); border-color: #3d2a5a; }
  /* --- Headline strip --- */
  .hl-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.4rem; }
  .hl { background: #21262d; border-radius: 4px; padding: 0.3rem 0.5rem; min-width: 0; }
  .hl .l { font: var(--fs-xs) var(--font-mono); color: var(--dim); text-transform: uppercase;
    letter-spacing: .03em; }
  .hl .v { color: var(--bold); font-size: var(--fs-xl); font-weight: 600;
    font-variant-numeric: tabular-nums; line-height: 1.15; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
  .hl .v small { color: var(--dim); font-size: var(--fs-xs); font-weight: 400; margin-left: 0.25rem; }
  .hl .v.warn { color: var(--yellow); }
  .delta { display: block; font: var(--fs-xs) var(--font-mono); color: var(--dim); }
  .delta.good { color: var(--green); } .delta.bad { color: var(--red); }
  @media (max-width: 60em) { .hl-strip { grid-template-columns: repeat(3, 1fr); } }
  /* --- Pipeline stages --- */
  .stages-bar { display: flex; height: 12px; border-radius: 4px; overflow: hidden; gap: 2px;
    background: #21262d; margin: 0.2rem 0 0.4rem; }
  .stages-bar > span { min-width: 2px; }
  table.stages td .sub { display: block; color: var(--dim); font-size: var(--fs-xs); }
  table.stages tr.dim td { color: var(--dim); }
  table.stages tr.pinned td:first-child { color: var(--bold); }
  .stage-sw { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 2px;
    margin-right: 0.35rem; vertical-align: -1px; }
  .meter { display: inline-block; width: 3rem; height: 5px; border-radius: 3px;
    background: #21262d; vertical-align: middle; margin-right: 0.35rem; overflow: hidden; }
  .meter i { display: block; height: 100%; background: var(--cyan); }
  /* --- Busy hours grid (one blue ramp toward the accent) --- */
  .hours { display: grid; grid-template-columns: 2rem repeat(24, 1fr); gap: 2px;
    font: 0.62rem var(--font-mono); color: var(--dim); flex: 1; align-content: start; }
  .hours .hl { text-align: center; background: none; padding: 0; border-radius: 0; }
  .hours .dl { align-self: center; }
  .hours i, .hours-legend i { display: block; aspect-ratio: 1.3; border-radius: 2px;
    background: #161b22; }
  .hours i { cursor: default; }
  .hours i:focus-visible { outline: 2px solid var(--cyan); outline-offset: 1px; }
  .hours i[data-l="1"], .hours-legend i[data-l="1"] { background: #0e2a3f; }
  .hours i[data-l="2"], .hours-legend i[data-l="2"] { background: #124b73; }
  .hours i[data-l="3"], .hours-legend i[data-l="3"] { background: #1f6fa8; }
  .hours i[data-l="4"], .hours-legend i[data-l="4"] { background: #58a6ff; }
  .hours i.peak { outline: 1.5px solid var(--bold); outline-offset: -1.5px; }
  .hours-legend { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap;
    font: var(--fs-xs) var(--font-mono); color: var(--dim); margin-top: 0.3rem; }
  .hours-legend span { display: inline-flex; align-items: center; gap: 0.25rem; }
  .hours-legend i { width: 9px; height: 9px; aspect-ratio: auto; }
  .hours-legend .what { margin-left: auto; }
  /* Own-scope "server" strip: stands in for the machine tiles (which are
     removed from the grid for scope=own unless STATS_OWN_SHOWS_MACHINE).
     A plain block above the grid, outside GridStack, so it never takes
     part in the saved layout. */
  .own-server { max-width: 68.75rem; margin: 0.875rem auto 0; padding: 0 0.875rem;
    box-sizing: border-box; }
  .own-server .card { flex-direction: row; flex-wrap: wrap; align-items: baseline;
    gap: 0.4rem 1.4rem; height: auto; }
  .own-server .card h3 { margin: 0; cursor: default; flex-basis: 100%; }
  .own-server .stat { display: inline-flex; align-items: baseline; gap: 0.4rem;
    font-size: var(--fs-sm); color: var(--dim); }
  .own-server .stat b { color: var(--bold); font-size: var(--fs-lg);
    font-variant-numeric: tabular-nums; }
  .own-server .stat .badge.warm { color: var(--green); border-color: #1f4d2a; }
  .own-server .stat .badge.busy { color: var(--yellow); border-color: #4d3e1f; }
  header .pill.scope { color: #fff; background: #1f6feb; border: 1px solid #1f6feb; }
  /* Usage-over-time tile — a full-width GridStack item. The card fills the
     whole tile (height:100%) and the chart flexes to absorb any slack, so the
     tile is never taller than the card (no dead clickable space below) and
     resizing the tile grows/shrinks the chart. */
  .usage-card { height: 100%; }
  .usage-toolbar { display: flex; flex-wrap: wrap; align-items: baseline;
    gap: 0.4rem 0.9rem; margin-bottom: 0.5rem; }
  .usage-toolbar h3 { margin: 0; }
  .usage-toolbar .spacer { flex: 1 1 auto; }
  .usage-seg { display: inline-flex; align-items: center; gap: 0.4rem; }
  .usage-seg .seg-label { color: var(--dim); font-size: var(--fs-xs);
    text-transform: uppercase; letter-spacing: .04em; }
  .seg-ctrl { display: inline-flex; border: 1px solid var(--border);
    border-radius: 6px; overflow: hidden; }
  .seg-ctrl button { background: var(--bg); color: var(--dim);
    border: none; border-left: 1px solid var(--border);
    padding: 0.15rem 0.55rem; font: inherit; font-size: var(--fs-sm);
    line-height: 1.3; cursor: pointer; }
  .seg-ctrl button:first-child { border-left: none; }
  .seg-ctrl button:hover { color: var(--fg); }
  .seg-ctrl button.active { background: var(--panel); color: var(--cyan);
    font-weight: 600; }
  .usage-chart { width: 100%; flex: 1 1 0; min-height: 9rem; min-width: 0; overflow: hidden;
    position: relative; }
  .usage-plot { width: 100%; height: 100%; min-width: 0; }
  .usage-note { color: var(--dim); font-size: var(--fs-xs);
    margin: 0.15rem 0 0.5rem; }
  /* Floating cursor tooltip over the multi-line usage chart. */
  .usage-tip { position: fixed; z-index: 5; pointer-events: none;
    background: var(--panel); border: 1px solid var(--border); border-radius: 5px;
    padding: 0.3rem 0.45rem; font: var(--fs-xs)/1.35 var(--font-mono);
    color: var(--fg); white-space: nowrap; display: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
  .usage-tip .tip-date { color: var(--dim); margin-bottom: 0.15rem; }
  .usage-tip .tip-row { display: flex; align-items: center; gap: 0.35rem; }
  .usage-tip .tip-row.focus { color: var(--bold); font-weight: 600; }
  .usage-tip .tip-row .tip-val { margin-left: auto;
    font-variant-numeric: tabular-nums; }
  .usage-swatch { display: inline-block; width: 0.6rem; height: 0.6rem;
    border-radius: 2px; flex: 0 0 auto; vertical-align: baseline; }
  table.usage-board td.rank { color: var(--dim);
    font-variant-numeric: tabular-nums; width: 2rem; }
  table.usage-board td.name { color: var(--fg); }
  table.usage-board td.name .usage-swatch { margin-right: 0.4rem; }
  table.usage-board td.name .sub { color: var(--dim);
    font-size: var(--fs-xs); margin-left: 0.4rem; }
  /* Recent-jobs table: kind chips, pipeline glyph strip, expandable
     per-stage bars, pinned running rows. Colors reuse the page tokens. */
  .rj-flag { color: var(--dim); font-size: var(--fs-xs); white-space: nowrap;
    display: inline-flex; align-items: center; gap: 0.3rem; }
  #rj-user { background: var(--bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 0.15rem 0.35rem; font: inherit; font-size: var(--fs-sm); }
  .rj-x { width: 1.4rem; }
  .rj-tbl tr.rj-main { cursor: pointer; }
  .rj-tbl tr.rj-main:hover td { background: rgba(121, 192, 255, 0.04); }
  .rj-caret { color: var(--dim); display: inline-block;
    transition: transform .15s ease; }
  tr.rj-main.open .rj-caret { transform: rotate(90deg); }
  .kindchip { display: inline-block; font-size: 0.667rem;
    padding: 0.0625rem 0.375rem; border-radius: 999px;
    border: 1px solid var(--border); color: var(--dim); white-space: nowrap; }
  .kindchip.transcribe { color: var(--cyan);    border-color: #1f3a5a; }
  .kindchip.dictate    { color: var(--green);   border-color: #1f4d2a; }
  .kindchip.translate  { color: var(--magenta); border-color: #3d2a5a; }
  .kindchip.download   { color: var(--yellow);  border-color: #4d3e1f; }
  .kindchip.preload    { color: var(--help);    border-color: var(--border); }
  .pipe { display: inline-flex; gap: 2px; }
  .pipe i { display: inline-block; width: 0.55rem; height: 0.55rem;
    border-radius: 2px; background: #21262d; }
  .pipe i.vad          { background: #93b76f; }  /* matches /quick-config's .seg-vad */
  .pipe i.separating   { background: var(--magenta); }
  .pipe i.transcribing { background: var(--cyan); }
  .pipe i.diarizing    { background: var(--yellow); }
  .pipe i.translating,
  .pipe i.translate    { background: var(--green); }
  .pipe i.download,
  .pipe i.downloading  { background: var(--cyan); }
  .pipe i.preload      { background: var(--help); }
  tr.rj-expand td { background: rgba(110, 118, 129, 0.06); }
  .rj-stages { padding: 0.25rem 0.25rem 0.35rem; }
  .rj-stage-row { display: flex; align-items: center; gap: 0.5rem;
    font-size: var(--fs-xs); margin: 0.15rem 0; }
  .rj-stage-row .nm { flex: 0 0 6.5rem; color: var(--dim);
    text-transform: uppercase; letter-spacing: .03em; }
  .rj-stage-row .stage-bar { flex: 1 1 auto; height: 8px;
    background: #21262d; border-radius: 3px; overflow: hidden; }
  .rj-stage-row .stage-bar i { display: block; height: 100%; }
  .rj-stage-row .secs { flex: 0 0 4.5rem; text-align: right;
    font-variant-numeric: tabular-nums; color: var(--fg); }
  .rj-stage-row .det { flex: 0 1 auto; color: var(--dim);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  tr.rj-run td { background: rgba(121, 192, 255, 0.05); }
  .rj-runbar { display: inline-block; width: 5rem; height: 6px;
    background: #21262d; border-radius: 3px; overflow: hidden;
    vertical-align: middle; }
  .rj-runbar i { display: block; height: 100%; background: var(--cyan); }
  .rj-spin { display: inline-block; width: 0.7rem; height: 0.7rem;
    border: 2px solid #21262d; border-top-color: var(--cyan);
    border-radius: 50%; animation: rj-spin 0.9s linear infinite;
    vertical-align: middle; }
  @keyframes rj-spin { to { transform: rotate(360deg); } }
  /* GridStack integration — drag-to-reorder + click-to-resize tiles. */
  .grid-stack { background: transparent; }
  .grid-stack-item-content { background: transparent; padding: 0; overflow: visible; }
  .grid-stack-item .card { cursor: default; }
  .grid-stack-item .card h3 { cursor: grab; user-select: none; }
  .grid-stack-item .card h3:active { cursor: grabbing; }
  .grid-stack-placeholder > .placeholder-content {
    background: rgba(56, 189, 248, 0.08);
    border: 1px dashed var(--cyan);
    border-radius: 6px;
  }
  .grid-stack > .grid-stack-item > .ui-resizable-handle {
    background-image: none;
    color: var(--dim);
    opacity: 0;
    transition: opacity 120ms ease;
  }
  .grid-stack > .grid-stack-item:hover > .ui-resizable-handle { opacity: 0.6; }
  .grid-stack > .grid-stack-item > .ui-resizable-se {
    width: 12px; height: 12px;
    border-right: 2px solid var(--dim);
    border-bottom: 2px solid var(--dim);
    transform: none;
  }
</style></head>
<body>
<header>
  <div class="header-inner">
    <span class="title">{{HEADER_BRAND}}</span>{{HEADER_VTAG}}
    <span class="brand-sep" aria-hidden="true"></span>
    {{NAV}}
    <span class="spacer"></span>
    <span class="hdr-right">{{SEV_PILLS}}{{SCALE_PICKER}}{{RELOAD}}{{LOGOUT}}</span>
  </div>
  <div class="subbar">
    <span class="subbar-title">Stats</span>
    <span class="seg-label">range</span>
    <div class="seg-ctrl" id="sb-range">
      <button data-v="7">7d</button><button data-v="30" class="active">30d</button><button data-v="90">90d</button><button data-v="180">180d</button><button data-v="365">1y</button><button data-v="all">all</button><button data-v="custom">custom…</button>
    </div>
    <div id="sb-custom" class="sb-custom hidden" role="dialog" aria-label="custom range">
      <label>from <input type="date" id="sb-from"></label>
      <label>to <input type="date" id="sb-to"></label>
      <span class="spans"><button type="button" data-span="month">this month</button><button type="button" data-span="lastmonth">last month</button><button type="button" data-span="quarter">this quarter</button><button type="button" data-span="year">this year</button><button type="button" data-span="lastyear">last year</button></span>
      <span id="sb-custom-note" class="note"></span>
      <button type="button" id="sb-custom-cancel">Cancel</button><button type="button" id="sb-custom-apply" class="primary">Apply</button>
    </div>
    <span class="seg-label">compare</span>
    <div class="seg-ctrl" id="sb-compare"><button data-v="off" class="active">off</button><button data-v="prev">previous</button><button data-v="yoy">last year</button></div>
    <div class="subbar-right">
      <span id="scope-pill" class="pill scope hidden" title="Your /stats scope is “own”: only jobs and usage from your own keys; machine cards replaced by a coarse server status">your usage</span>
      <span class="seg-label">layout</span>
      <div class="seg-ctrl" id="layout-preset" title="which tiles are on the grid; positions are remembered per preset">
        <button data-v="ops">ops</button><button data-v="usage">usage</button><button data-v="both" class="active">both</button>
      </div>
      <button id="edit-layout-btn" aria-pressed="false" title="drag tile titles and resize corners (E); Alt+arrows move, Alt+Shift+arrows resize the focused tile">✎ edit layout</button>
      <span id="layout-live" class="sr-only" aria-live="polite"></span>
      <button id="reset-layout-btn" title="reset this preset's tile layout to defaults">↺ layout</button>
      <span id="status" class="pill live">live</span>
    </div>
  </div>
  <!-- Usage scope, second row: kind (client-side), "ran" stage chips (server
       narrows to jobs that ran every chosen stage), click-to-filter chips
       from the leaderboard, and the resolved window. -->
  <div class="subbar subbar-usage">
    <span class="seg-label">kind</span>
    <span class="chips" id="sb-kind">
      <button type="button" class="chip on" data-v="all">all</button>
      <button type="button" class="chip" data-v="dictation"><i class="sw" style="background:#2ea043"></i>dictation</button>
      <button type="button" class="chip" data-v="file"><i class="sw" style="background:#388bfd"></i>files</button>
      <button type="button" class="chip" data-v="url"><i class="sw" style="background:#bb8009"></i>links</button>
      <button type="button" class="chip" data-v="text"><i class="sw" style="background:#8957e5"></i>text</button>
    </span>
    <span class="seg-label">ran</span>
    <span class="chips" id="sb-with">
      <button type="button" class="chip" data-v="translating">translated</button>
      <button type="button" class="chip" data-v="diarizing">diarized</button>
      <button type="button" class="chip" data-v="separating">music separated</button>
      <button type="button" class="chip" data-v="vad">silence skipped</button>
    </span>
    <span class="seg-label">filters</span>
    <span class="chips" id="sb-filters"></span>
    <span class="sb-summary" id="sb-summary"></span>
  </div>
</header>

<!-- Own-scope server strip (scope=own without STATS_OWN_SHOWS_MACHINE): the
     coarse block the payload's `server` key carries. Hidden until the first
     snapshot says machine=false. -->
<div id="own-server" class="own-server hidden">
  <div class="card">
    <h3>Server</h3>
    <span class="stat">GPU <b id="own-gpu">—</b></span>
    <span class="stat">VRAM <b id="own-vram">—</b></span>
    <span class="stat">models loaded <b id="own-models">—</b></span>
    <span class="stat" style="color:var(--dim)">machine cards are hidden for your scope — an admin can enable “Own-scope users see machine cards” in /settings</span>
  </div>
</div>

<div id="grid" class="grid">
 <div class="grid-stack">
  <!-- GPU (one GridStack item; inner content swaps between "live" and "no NVML"
       — a second hidden grid-stack-item would still occupy a cell and, under
       float: true, shift to a free slot that overlaps the row below.) -->
  <div class="grid-stack-item" gs-id="gpu" gs-x="0" gs-y="0" gs-w="6" gs-h="9">
   <div class="grid-stack-item-content"><div id="card-gpu" class="card">
    <h3>GPU</h3>
    <div id="gpu-content">
     <div id="gpu-name" class="val">—</div>
     <div id="gpu-meta" class="meta"></div>
     <div id="gpu-mem-bar" class="bar"><i style="width:0"></i></div>
     <div id="gpu-meta2" class="meta"></div>
     <div class="spark-wrap">
       <div class="spark-head"><span class="spark-label">GPU util %</span>
         <span id="gpu-util-now" class="spark-now">—</span></div>
       <div id="gpu-spark-util" class="spark"></div>
     </div>
     <div class="spark-wrap">
       <div class="spark-head"><span class="spark-label">VRAM used %</span>
         <span id="gpu-mem-now" class="spark-now">—</span></div>
       <div id="gpu-spark-mem" class="spark"></div>
     </div>
     <div class="spark-wrap">
       <div class="spark-head"><span class="spark-label">GPU temp °C</span>
         <span id="gpu-temp-now" class="spark-now">—</span></div>
       <div id="gpu-spark-temp" class="spark"></div>
     </div>
    </div>
    <div id="gpu-empty" class="hidden">
     <div class="empty">NVML unavailable — running on CPU or pynvml not installed.</div>
     <div id="gpu-error" class="meta"></div>
    </div>
   </div></div>
  </div>

  <!-- Host CPU -->
  <div class="grid-stack-item" gs-id="cpu" gs-x="6" gs-y="0" gs-w="6" gs-h="5">
   <div class="grid-stack-item-content"><div class="card">
    <h3>CPU (host)</h3>
    <div id="cpu-pct" class="val">—<span class="sub">%</span></div>
    <div id="cpu-cores" class="core-strip"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">CPU %</span>
        <span id="cpu-now" class="spark-now">—</span></div>
      <div id="cpu-spark" class="spark"></div>
    </div>
   </div></div>
  </div>

  <!-- Host RAM -->
  <div class="grid-stack-item" gs-id="ram" gs-x="6" gs-y="5" gs-w="6" gs-h="4">
   <div class="grid-stack-item-content"><div class="card">
    <h3>RAM</h3>
    <div id="ram-val" class="val">— <span class="sub">/ — GB</span></div>
    <div id="ram-bar" class="bar"><i style="width:0"></i></div>
    <div id="ram-meta" class="meta"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">RAM used %</span>
        <span id="ram-now" class="spark-now">—</span></div>
      <div id="ram-spark" class="spark"></div>
    </div>
   </div></div>
  </div>

  <!-- Process -->
  <div class="grid-stack-item" gs-id="process" gs-x="0" gs-y="9" gs-w="4" gs-h="3">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Process</h3>
    <div id="proc-rss" class="val">—<span class="sub">MB RSS</span></div>
    <div id="proc-meta" class="meta"></div>
   </div></div>
  </div>

  <!-- In-flight + uptime -->
  <div class="grid-stack-item" gs-id="activity" gs-x="4" gs-y="9" gs-w="4" gs-h="3">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Activity</h3>
    <div id="inflight-val" class="val">0<span class="sub">in flight</span></div>
    <div id="gate-meta" class="meta"></div>
    <div id="activity-meta" class="meta"></div>
   </div></div>
  </div>

  <!-- Errors window -->
  <div class="grid-stack-item" gs-id="errors" gs-x="8" gs-y="9" gs-w="4" gs-h="3">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Errors (5xx)</h3>
    <div class="err-strip">
      <div id="err-1m" class="seg"><b>0</b>1 min</div>
      <div id="err-5m" class="seg"><b>0</b>5 min</div>
      <div id="err-15m" class="seg"><b>0</b>15 min</div>
    </div>
    <div id="err-meta" class="meta"></div>
   </div></div>
  </div>

  <!-- Latency -->
  <div class="grid-stack-item" gs-id="latency" gs-x="0" gs-y="12" gs-w="6" gs-h="5">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Request latency (last <span id="lat-n">0</span>)</h3>
    <div id="lat-val" class="val">— <span class="sub">ms p50</span></div>
    <div id="lat-meta" class="meta"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">p50 latency (ms)</span>
        <span id="lat-now" class="spark-now">—</span></div>
      <div id="lat-spark" class="spark"></div>
    </div>
   </div></div>
  </div>

  <!-- Endpoint counters -->
  <div class="grid-stack-item" gs-id="endpoints" gs-x="6" gs-y="12" gs-w="6" gs-h="5">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Endpoint counters</h3>
    <table class="tbl rcards"><thead><tr><th>path</th><th class="num">requests</th><th class="num">5xx</th></tr></thead>
    <tbody id="endpoints-rows"></tbody></table>
   </div></div>
  </div>

  <!-- Loaded models -->
  <div class="grid-stack-item" gs-id="models" gs-x="0" gs-y="17" gs-w="12" gs-h="4">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Loaded models <span class="tag">· audio and RTF over the usage window</span></h3>
    <table class="tbl rcards"><thead><tr>
      <th>name</th><th>device</th><th>compute</th>
      <th class="num">audio</th><th class="num">RTF</th>
      <th class="num">VRAM (MB)</th><th class="num">disk</th><th>state</th>
      <th class="num">age</th><th class="num">idle</th>
      <th class="num">cold-load</th>
    </tr></thead><tbody id="models-rows"></tbody></table>
    <div class="meta" id="preload-line"></div>
   </div></div>
  </div>

  <!-- Usage headline: five numbers for the scope bar's window, deltas vs
       the compare window. Fed by the usage document (static/stats.js). -->
  <div class="grid-stack-item" gs-id="headline" gs-x="0" gs-y="21" gs-w="12" gs-h="2">
   <div class="grid-stack-item-content"><div class="card usage-fed">
    <h3>Usage <span class="tag" id="headline-tag"></span></h3>
    <div class="hl-strip" id="headline-strip"><span class="empty">— loading —</span></div>
   </div></div>
  </div>

  <!-- Usage over time: stacked bars by kind (lines for user/key/model/
       stage), dashed compare line, legend, table twin, leaderboard. -->
  <div class="grid-stack-item" gs-id="usage" gs-x="0" gs-y="23" gs-w="12" gs-h="10">
   <div class="grid-stack-item-content"><div class="card usage-card usage-fed">
    <div class="usage-toolbar">
      <h3>Usage over time</h3>
      <span id="usage-error" class="usage-error hidden"></span>
      <span class="spacer"></span>
      <div class="usage-seg"><span class="seg-label">bucket</span>
        <div class="seg-ctrl" id="usage-bucket">
          <button data-v="auto" class="active">auto</button>
          <button data-v="day">day</button>
          <button data-v="week">week</button>
          <button data-v="month">month</button>
        </div>
      </div>
      <div class="usage-seg"><span class="seg-label">metric</span>
        <div class="seg-ctrl" id="usage-metric">
          <button data-v="audio_s" class="active">audio</button>
          <button data-v="words">words</button>
          <button data-v="sessions">sessions</button>
          <button data-v="requests">requests</button>
          <button data-v="errors">errors</button>
          <button data-v="proc_s">GPU s</button>
        </div>
      </div>
      <div class="usage-seg"><span class="seg-label">by</span>
        <div class="seg-ctrl" id="usage-by">
          <button data-v="kind" class="active">kind</button>
          <button data-v="user">user</button>
          <button data-v="key">key</button>
          <button data-v="model">model</button>
          <button data-v="stage">stage</button>
        </div>
      </div>
      <div class="usage-seg">
        <div class="seg-ctrl"><button id="usage-table-btn" data-v="table" title="show the chart's numbers as a table (T)">table</button></div>
      </div>
    </div>
    <div class="usage-chart" id="usage-chart-wrap" tabindex="0" aria-label="usage chart — arrow keys scrub the buckets, T toggles the table">
      <div id="usage-plot" class="usage-plot"></div>
      <div id="usage-tip" class="usage-tip"></div>
      <div id="usage-empty" class="usage-empty hidden"></div>
    </div>
    <div id="usage-table" class="usage-table hidden"></div>
    <span id="usage-live" class="sr-only" aria-live="polite"></span>
    <div class="usage-legend" id="usage-legend"></div>
    <table class="tbl usage-board rcards"><thead id="usage-board-head"><tr>
      <th class="rank">#</th><th>name</th>
      <th class="num">audio</th><th class="num">sessions</th><th class="num">requests</th>
      <th class="num">audio</th><th class="num">GPU s</th><th class="num">RTF</th><th class="num">err</th>
    </tr></thead><tbody id="usage-board-rows">
      <tr><td colspan="9" class="empty">— loading —</td></tr>
    </tbody></table>
   </div></div>
  </div>

  <!-- Pipeline stages: share of eligible runs + speed per optional stage. -->
  <div class="grid-stack-item" gs-id="stages" gs-x="0" gs-y="33" gs-w="6" gs-h="6">
   <div class="grid-stack-item-content"><div class="card usage-fed">
    <h3>Pipeline stages <span class="tag" id="stages-tag"></span></h3>
    <div class="stages-bar" id="stages-bar"></div>
    <table class="tbl stages"><thead><tr>
      <th>stage</th><th class="num">runs</th><th class="num">of eligible</th>
      <th class="num">audio</th><th class="num">GPU s</th><th class="num">RTF</th>
    </tr></thead><tbody id="stages-rows"><tr><td colspan="6" class="empty">— loading —</td></tr></tbody></table>
   </div></div>
  </div>

  <!-- Busy hours: weekday × hour of GPU seconds, quartile-levelled. -->
  <div class="grid-stack-item" gs-id="hours" gs-x="6" gs-y="33" gs-w="6" gs-h="6">
   <div class="grid-stack-item-content"><div class="card usage-fed">
    <h3>Busy hours <span class="tag" id="hours-tag"></span></h3>
    <div class="hours" id="hours-grid"></div>
    <div class="hours-legend" id="hours-legend"></div>
   </div></div>
  </div>

  <!-- Recent jobs (unified: transcribe / dictate / translate / download;
       running jobs from snap.jobs pinned on top) -->
  <div class="grid-stack-item" gs-id="recent" gs-x="0" gs-y="39" gs-w="12" gs-h="6">
   <div class="grid-stack-item-content"><div class="card">
    <div class="usage-toolbar">
      <h3>Recent jobs (<span id="rt-n">0</span> shown)</h3>
      <span class="spacer"></span>
      <div class="usage-seg"><span class="seg-label">kind</span>
        <div class="seg-ctrl" id="rj-kind">
          <button data-v="" class="active">all</button>
          <button data-v="transcribe">transcribe</button>
          <button data-v="dictate">dictate</button>
          <button data-v="translate">translate</button>
          <button data-v="download">download</button>
          <button data-v="preload">preload</button>
        </div>
      </div>
      <label class="rj-flag"><input type="checkbox" id="rj-warnonly"> warnings only</label>
      <select id="rj-user" aria-label="filter by user"><option value="">all users</option></select>
    </div>
    <table class="tbl rcards rj-tbl"><thead><tr>
      <th class="rj-x"></th><th>when</th><th>type</th><th>pipeline</th><th>model</th>
      <th>user·key</th><th class="num">input</th><th class="num">wall</th>
      <th class="num">speed</th><th>status</th>
    </tr></thead><tbody id="rt-rows"><tr><td colspan="10" class="empty">— no jobs yet —</td></tr></tbody></table>
   </div></div>
  </div>
 </div>
</div>

<!-- First-party page script (GridStack layout + the usage section); the
     build version busts the cacheable /static mount. -->
<script src="/static/stats.js?v=__ASSET_V__"></script>
<script>
(() => {
'use strict';

// --- per-metric history rings ----------------------------------------------
const HISTORY_LEN = 120;     // 2 min @ 1 Hz
const histX = [];            // shared time axis (epoch seconds)
const hist = {
  gpu_util: [], gpu_mem_pct: [], gpu_temp: [],
  cpu: [], ram_pct: [], lat_p50: [],
};

function pushHistory(snap) {
  const now = Math.floor(snap.ts || (Date.now() / 1000));
  histX.push(now);
  hist.gpu_util.push(snap.gpu ? snap.gpu.util_pct ?? null : null);
  hist.gpu_mem_pct.push(snap.gpu && snap.gpu.mem_total_mb
    ? (snap.gpu.mem_used_mb / snap.gpu.mem_total_mb * 100) : null);
  hist.gpu_temp.push(snap.gpu ? snap.gpu.temp_c ?? null : null);
  hist.cpu.push(snap.host ? snap.host.cpu_pct ?? null : null);
  hist.ram_pct.push(snap.host ? snap.host.ram_pct ?? null : null);
  hist.lat_p50.push(snap.latency_ms && snap.latency_ms.n > 0 ? snap.latency_ms.p50 : null);
  if (histX.length > HISTORY_LEN) {
    histX.shift();
    for (const k in hist) hist[k].shift();
  }
}

// --- uPlot factory ---------------------------------------------------------
// Each spark gets:
//   - explicit `splits` to force readable y-axis ticks (uPlot auto-picks one
//     tick on flat/idle data, which renders as a lonely "0").
//   - `unit` suffix on those ticks.
//   - auto-padding (10% top/bottom) when no fixed range — keeps unbounded
//     metrics like temperature / latency from pinning to the bottom.
//
// uPlot's canvas rendering needs px (not rem). These helpers read the
// current --fs-base via getComputedStyle so axis sizing tracks the scale
// picker. `--fs-base` is set by SCALE_BOOTSTRAP_HEAD BEFORE this script
// runs, so on first load the axes match the saved scale. Live picker
// changes don't refit the canvas — switching scale visibly updates HTML
// chrome but axis labels stay at construction-time size until the next
// page load. Acceptable trade-off vs destroying/rebuilding sparks (which
// would blank the chart until the next SSE tick).
function _remPx(n) {
  const base = parseFloat(getComputedStyle(document.documentElement).fontSize) || 15;
  return Math.round(n * base);
}
function _axisFontPx() { return _remPx(0.733); }   // matches --fs-xs
const _MONO_STACK = 'Consolas, "Cascadia Code", "JetBrains Mono", Menlo, ui-monospace, monospace';
const sparks = {};   // name -> uPlot instance
// --- hover freeze -----------------------------------------------------------
// While the pointer is on a ring the readouts show THAT sample with its
// clock time, in yellow, so a frozen number cannot be mistaken for live.
// The ring shifts left every second, so the frozen TIMESTAMP is kept (not
// the index) and re-found each tick; when it falls off the ring, unfreeze.
let frozenTs = null;
const READOUTS = [
  ['gpu-util-now', 'gpu_util', v => v.toFixed(0) + '%'],
  ['gpu-mem-now', 'gpu_mem_pct', v => v.toFixed(0) + '%'],
  ['gpu-temp-now', 'gpu_temp', v => v.toFixed(0) + '°C'],
  ['cpu-now', 'cpu', v => v.toFixed(0) + '%'],
  ['ram-now', 'ram_pct', v => v.toFixed(0) + '%'],
  ['lat-now', 'lat_p50', v => v.toFixed(0) + ' ms'],
];
function onSparkHover(u) {
  const idx = u.cursor.idx;
  const ts = (idx == null || idx < 0 || idx >= histX.length) ? null : histX[idx];
  if (ts === frozenTs) return;
  frozenTs = ts;
  applyFreeze();
}
function applyFreeze() {
  const idx = frozenTs == null ? -1 : histX.indexOf(frozenTs);
  if (idx < 0) {
    frozenTs = null;
    for (const [id] of READOUTS) { const el = $(id); if (el) el.classList.remove('frozen'); }
    return;   // render() has just written the live values
  }
  const t = new Date(frozenTs * 1000);
  const p2 = n => ('0' + n).slice(-2);
  const clock = p2(t.getHours()) + ':' + p2(t.getMinutes()) + ':' + p2(t.getSeconds());
  for (const [id, key, fmt] of READOUTS) {
    const el = $(id); if (!el) continue;
    const v = hist[key][idx];
    el.textContent = (v == null ? '—' : fmt(v)) + ' @ ' + clock;
    el.classList.add('frozen');
  }
}

function makeSpark(elId, color, opts={}) {
  const el = document.getElementById(elId);
  if (!el) return null;
  const w = el.clientWidth || 240;
  const h = el.clientHeight || 72;
  const yScale = opts.range
    ? { range: opts.range }
    : { range: { min: { pad: 0.1, mode: 1 }, max: { pad: 0.1, mode: 1 } } };
  // uPlot's canvas API needs px values, not rem. Read them from --fs-base
  // so axis labels track the scale picker — see _axisFontPx below.
  const axisFontPx = _axisFontPx();
  const inst = new uPlot({
    width: w, height: h,
    // [top, right, bottom, left] in px. Top AND bottom both ≥ ½ axis-font
    // height + a small breathing margin so the highest split label
    // ("100%" / "60°") and the lowest ("0%" / "30°") render their full
    // glyph height inside the canvas instead of being clipped by the
    // canvas edges (uPlot draws tick labels centered on the data-area
    // edge — half the glyph extends past the edge, so padding must
    // exceed font-size/2). Left padding plus axis size gives uPlot room
    // to draw "100%" without GridStack's overflow-x clipping the "1".
    padding: [_remPx(0.55), 6, _remPx(0.4), _remPx(0.25)],
    // One crosshair for every live ring: the sparks share histX, so a
    // cursor on one lands on the same second in all of them (sync key
    // 'live'; the usage chart is a different key and never follows).
    // Hovering freezes the card readouts at that sample (onSparkHover).
    cursor: { show: true, x: true, y: false, points: { show: false },
              drag: { x: false, y: false },
              sync: { key: 'live', setSeries: false, scales: ['x', null] } },
    hooks: { setCursor: [onSparkHover] },
    legend: { show: false },
    select: { show: false },
    scales: { x: { time: false }, y: yScale },
    axes: [
      { show: false },
      { show: true, size: _remPx(2.6), gap: 4,
        font: axisFontPx + 'px ' + _MONO_STACK,
        stroke: '#6e7681',
        grid:  { stroke: '#21262d', width: 1 },
        ticks: { stroke: '#30363d', width: 1, size: 3 },
        splits: opts.splits,
        values: opts.splits ? (u, splits) => splits.map(v => v + (opts.unit || '')) : null,
      },
    ],
    series: [
      {},
      { stroke: color, width: 1.5, fill: color + '22', spanGaps: true,
        points: { show: false } },
    ],
  }, [[], []], el);
  // Responsive sizing. ResizeObserver on the spark element fires for any
  // size source — GridStack drag-resize, window resize, scale-picker rem
  // changes, .hidden toggle reflow. rAF coalescing avoids thrashing
  // setSize() during a drag (it's a relatively expensive canvas rebuild).
  let raf = 0;
  const ro = new ResizeObserver(() => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const cw = el.clientWidth, ch = el.clientHeight;
      if (cw < 1 || ch < 1) return;
      inst.setSize({ width: cw, height: ch });
    });
  });
  ro.observe(el);
  return inst;
}

function ensureSparks() {
  if (sparks.cpu) return;     // already built
  // Percentage sparks: fixed [0, 100] range with 0/50/100 ticks.
  sparks.cpu      = makeSpark('cpu-spark',      '#79c0ff', { range: [0, 100], splits: [0, 50, 100], unit: '%' });
  sparks.ram      = makeSpark('ram-spark',      '#7ee787', { range: [0, 100], splits: [0, 50, 100], unit: '%' });
  sparks.gpu_util = makeSpark('gpu-spark-util', '#79c0ff', { range: [0, 100], splits: [0, 50, 100], unit: '%' });
  sparks.gpu_mem  = makeSpark('gpu-spark-mem',  '#d2a8ff', { range: [0, 100], splits: [0, 50, 100], unit: '%' });
  // Temperature: fixed coarse splits at 30/60/90 °C cover idle through hot.
  sparks.gpu_temp = makeSpark('gpu-spark-temp', '#f2cc60', { splits: [30, 60, 90], unit: '°' });
  // Latency: unbounded, auto-range with 10% padding.
  sparks.lat      = makeSpark('lat-spark',      '#7ee787');
}

function setData(u, ys) {
  if (!u) return;
  // uPlot wants nulls preserved for spanGaps; convert undefined -> null.
  const xs = histX.slice();
  const yClean = ys.map(v => (v == null ? null : v));
  u.setData([xs, yClean], true);
}

// --- DOM helpers -----------------------------------------------------------
const $ = id => document.getElementById(id);
function fmtBytes(mb) {
  if (mb == null) return '—';
  return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb.toFixed(0) + ' MB';
}
function fmtSec(s) {
  if (s == null) return '—';
  if (s < 60) return s.toFixed(0) + ' s';
  if (s < 3600) return (s / 60).toFixed(1) + ' min';
  if (s < 86400) return (s / 3600).toFixed(1) + ' h';
  return (s / 86400).toFixed(1) + ' d';
}
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function setBar(barEl, pct) {
  const bar = barEl.querySelector('i');
  bar.style.width = Math.max(0, Math.min(100, pct)).toFixed(1) + '%';
  barEl.classList.toggle('warn', pct >= 75 && pct < 90);
  barEl.classList.toggle('crit', pct >= 90);
}

// --- Render ----------------------------------------------------------------
// --- Scope-aware chrome (first snapshot only) --------------------------------
// The payload's `scope` ("own"|"all") and `machine` (bool) are decided
// server-side (StatsScope). scope=own hides the per-user controls — the only
// user is the viewer — and machine=false swaps the machine tiles for the
// #own-server strip. Applied once: the scope cannot change mid-stream
// without the server re-resolving it, and a re-resolve that narrows the
// scope reloads the page (below).
const MACHINE_TILES = ['gpu', 'cpu', 'ram', 'process', 'activity', 'errors',
                       'latency', 'endpoints', 'models'];
let scopeApplied = null;
function applyScope(snap) {
  const sig = (snap.scope || 'all') + ':' + (snap.machine === false ? 0 : 1);
  if (scopeApplied === sig) return;
  if (scopeApplied !== null) { location.reload(); return; }   // scope moved
  scopeApplied = sig;
  if (snap.scope === 'own') {
    document.body.classList.add('scope-own');
    $('scope-pill').classList.remove('hidden');
    // The per-user filter and the by-user leaderboard would list exactly
    // one person; the server refuses by=user for own scope anyway.
    const userSel = $('rj-user');
    if (userSel) { userSel.value = ''; userSel.classList.add('hidden'); }
    const byUser = document.querySelector('#usage-by button[data-v="user"]');
    if (byUser) byUser.classList.add('hidden');
    if (typeof window._fwUsageReload === 'function') window._fwUsageReload();
  }
  if (snap.machine === false) {
    // Presets pick machine tiles that are gone for this scope: force the
    // full set (without overwriting the remembered choice) and hide the
    // control; the '-own' key keeps this layout apart from an admin's.
    if (typeof window._fwSetPreset === 'function') window._fwSetPreset('both', false);
    const presetSeg = document.getElementById('layout-preset');
    if (presetSeg) { presetSeg.classList.add('hidden'); if (presetSeg.previousElementSibling) presetSeg.previousElementSibling.classList.add('hidden'); }
    GS_LAYOUT_KEY += '-own';
    grid.batchUpdate();
    MACHINE_TILES.forEach(id => {
      const el = document.querySelector(`.grid-stack-item[gs-id="${id}"]`);
      if (el) grid.removeWidget(el, true);
    });
    grid.batchUpdate(false);
    try {
      const saved = localStorage.getItem(GS_LAYOUT_KEY);
      if (saved) grid.load(JSON.parse(saved), false);
    } catch (_) {}
    $('own-server').classList.remove('hidden');
  }
}

function renderServer(server) {
  const s = server || {};
  const g = s.gpu || {};
  const gpuEl = $('own-gpu');
  if (!g.present) {
    gpuEl.innerHTML = '<span class="badge">none</span>';
  } else {
    gpuEl.innerHTML = g.busy
      ? '<span class="badge busy">busy</span>'
      : '<span class="badge warm">idle</span>';
  }
  $('own-vram').textContent = g.mem_total_mb
    ? `${fmtBytes(g.mem_used_mb)} / ${fmtBytes(g.mem_total_mb)}` : '—';
  $('own-models').textContent = s.models_loaded ?? '—';
}

// Loaded-models table. Joins the usage document's per-model totals
// (window.__statsUsage, published by static/stats.js after every usage load)
// for the audio / RTF columns, and the snapshot's size provenance
// (size_bytes / size_src / disk_bytes) for VRAM + disk.
let lastModelsSnap = null;
function renderModels(snap) {
  lastModelsSnap = snap;
  const modelLoads = snap.model_loads || {};
  const usage = (window.__statsUsage && window.__statsUsage.models) || {};
  const mrows = (snap.models || []).map(m => {
    const warm = m.idle_sec < 60;
    const cold = modelLoads[m.name];
    const coldStr = cold
      ? `${cold.first}s / ~${cold.last5_avg}s avg (${cold.count})`
      : '—';
    const u = usage[m.name];
    const audio = u ? fmtSec(u.audio_s) : '—';
    const rtf = u && u.rtf != null
      ? `<span class="rtf${u.rtf > 0.35 ? ' slow' : ''}">${u.rtf.toFixed(2)}×</span>` : '—';
    const srcBadge = m.size_src && m.size_src !== 'measured'
      ? ` <span class="badge est" title="${m.size_src === 'disk' ? 'on-disk prior, never measured on this placement' : 'measured on another device / compute type'}">est</span>`
      : '';
    const disk = m.disk_bytes != null ? fmtBytes(m.disk_bytes / 1048576) : '—';
    return `<tr>
      <td data-label="name">${esc(m.name)}</td>
      <td data-label="device">${esc(m.device || '—')}</td>
      <td data-label="compute">${esc(m.compute_type || '—')}</td>
      <td class="num" data-label="audio">${audio}</td>
      <td class="num" data-label="RTF">${rtf}</td>
      <td class="num" data-label="VRAM (MB)">${m.vram_mb != null ? m.vram_mb.toFixed(0) : '—'}${srcBadge}</td>
      <td class="num" data-label="disk">${disk}</td>
      <td data-label="state"><span class="badge ${warm ? 'warm' : 'cold'}">${warm ? 'warm' : 'cold'}</span></td>
      <td class="num" data-label="age">${fmtSec(m.age_sec)}</td>
      <td class="num" data-label="idle">${fmtSec(m.idle_sec)}</td>
      <td class="num" data-label="cold-load">${coldStr}</td>
    </tr>`;
  });
  $('models-rows').innerHTML = mrows.length
    ? mrows.join('') : '<tr><td colspan="11" class="empty">— no models loaded —</td></tr>';
}
window._fwRerenderModels = () => { if (lastModelsSnap) renderModels(lastModelsSnap); };

function render(snap) {
  applyScope(snap);
  if (snap.machine === false) {
    // Own scope without the machine cards: the strip, the jobs table and
    // the header cluster are all there is to draw.
    renderServer(snap.server);
    renderJobs(snap);
    if (typeof window._fwFeedActivity === 'function') {
      try { window._fwFeedActivity(snap); } catch (_) {}
    }
    return;
  }
  ensureSparks();

  // --- GPU ---
  if (snap.gpu) {
    // Swap inner content of the single GPU GridStack item (not the wrapper
    // — a hidden grid-stack-item still occupies a cell and breaks the row
    // below it under float: true).
    $('gpu-content').classList.remove('hidden');
    $('gpu-empty').classList.add('hidden');
    $('gpu-name').textContent = snap.gpu.name || 'GPU';
    $('gpu-meta').innerHTML =
      `<b>util</b> ${snap.gpu.util_pct ?? '—'}% &nbsp; ` +
      `<b>temp</b> ${snap.gpu.temp_c ?? '—'}°C &nbsp; ` +
      `<b>power</b> ${snap.gpu.power_w ?? '—'} / ${snap.gpu.power_limit_w ?? '—'} W &nbsp; ` +
      `<b>state</b> ${snap.gpu.p_state || '—'}`;
    const memPct = snap.gpu.mem_total_mb
      ? snap.gpu.mem_used_mb / snap.gpu.mem_total_mb * 100 : 0;
    setBar($('gpu-mem-bar'), memPct);
    $('gpu-meta2').innerHTML =
      `<b>VRAM</b> ${fmtBytes(snap.gpu.mem_used_mb)} / ${fmtBytes(snap.gpu.mem_total_mb)} ` +
      `(${memPct.toFixed(0)}%) &nbsp; ` +
      `<b>SM clock</b> ${snap.gpu.sm_clock_mhz ?? '—'} MHz &nbsp; ` +
      `<b>driver</b> ${snap.gpu.driver || '—'} &nbsp; <b>CUDA</b> ${snap.gpu.cuda || '—'}`;
    setData(sparks.gpu_util, hist.gpu_util);
    setData(sparks.gpu_mem,  hist.gpu_mem_pct);
    setData(sparks.gpu_temp, hist.gpu_temp);
    $('gpu-util-now').textContent = (snap.gpu.util_pct ?? 0).toFixed(0) + '%';
    $('gpu-mem-now').textContent  = memPct.toFixed(0) + '%';
    $('gpu-temp-now').textContent = (snap.gpu.temp_c ?? 0).toFixed(0) + '°C';
  } else {
    $('gpu-content').classList.add('hidden');
    $('gpu-empty').classList.remove('hidden');
    $('gpu-error').textContent = snap.gpu_error || '';
  }

  // --- Host CPU ---
  $('cpu-pct').innerHTML = (snap.host.cpu_pct ?? 0).toFixed(1) + '<span class="sub">%</span>';
  const stripEl = $('cpu-cores');
  const cores = snap.host.cpu_per_core || [];
  if (stripEl.children.length !== cores.length) {
    stripEl.innerHTML = '';
    for (let i = 0; i < cores.length; i++) stripEl.appendChild(document.createElement('div'));
  }
  for (let i = 0; i < cores.length; i++) {
    stripEl.children[i].style.height = Math.max(2, cores[i]) + '%';
  }
  setData(sparks.cpu, hist.cpu);
  $('cpu-now').textContent = (snap.host.cpu_pct ?? 0).toFixed(0) + '%';

  // --- Host RAM ---
  $('ram-val').innerHTML = `${fmtBytes(snap.host.ram_used_mb)} ` +
    `<span class="sub">/ ${fmtBytes(snap.host.ram_total_mb)}</span>`;
  setBar($('ram-bar'), snap.host.ram_pct);
  $('ram-meta').innerHTML = `<b>${snap.host.ram_pct.toFixed(1)}%</b> used &nbsp; ` +
    `<b>disk free</b> ${snap.host.disk_free_gb ?? '—'} GB (model cache)`;
  setData(sparks.ram, hist.ram_pct);
  $('ram-now').textContent = (snap.host.ram_pct ?? 0).toFixed(0) + '%';

  // --- Process ---
  $('proc-rss').innerHTML = (snap.process.rss_mb ?? 0).toFixed(0) +
    '<span class="sub">MB RSS</span>';
  $('proc-meta').innerHTML =
    `<b>PID</b> ${snap.process.pid} &nbsp; ` +
    `<b>CPU</b> ${(snap.process.cpu_pct ?? 0).toFixed(1)}% &nbsp; ` +
    `<b>threads</b> ${snap.process.threads ?? '—'} &nbsp; ` +
    `<b>uptime</b> ${fmtSec(snap.process.uptime_sec)}`;

  // --- Activity / in-flight ---
  $('inflight-val').innerHTML = `${snap.in_flight_transcriptions}` +
    `<span class="sub">in flight</span>`;
  const gate = snap.gpu_gate || {};
  const gateEl = $('gate-meta');
  if (gateEl) {
    if (gate.capacity == null) {
      gateEl.innerHTML = '<span class="empty">GPU gate not built yet — no inference so far</span>';
    } else {
      const q = gate.queue_depth || 0;
      gateEl.innerHTML =
        `<b>GPU slots</b> ${gate.held} / ${gate.capacity} held &nbsp; ` +
        `<b>queue</b> <span class="${q ? 'warn' : ''}">${q}</span>` +
        (q ? ` &nbsp; <b>oldest wait</b> ${fmtSec(gate.oldest_wait_s)}` : '');
    }
  }
  const totalReq = Object.values(snap.requests || {}).reduce((a, b) => a + b, 0);
  $('activity-meta').innerHTML =
    `<b>uptime</b> ${fmtSec(snap.uptime_sec)} &nbsp; ` +
    `<b>total req</b> ${totalReq}`;

  // --- Latency ---
  const lat = snap.latency_ms || { n: 0, p50: 0, p95: 0, p99: 0 };
  $('lat-n').textContent = lat.n;
  if (lat.n > 0) {
    $('lat-val').innerHTML = lat.p50.toFixed(0) + '<span class="sub">ms p50</span>';
    $('lat-meta').innerHTML =
      `<b>p95</b> ${lat.p95.toFixed(0)} ms &nbsp; ` +
      `<b>p99</b> ${lat.p99.toFixed(0)} ms`;
    $('lat-now').textContent = lat.p50.toFixed(0) + ' ms';
  } else {
    $('lat-val').innerHTML = '—';
    $('lat-meta').innerHTML = '<span class="empty">no requests yet</span>';
    $('lat-now').textContent = '—';
  }
  setData(sparks.lat, hist.lat_p50);

  // --- Errors window ---
  const ew = snap.errors_window || { '1m': 0, '5m': 0, '15m': 0 };
  for (const k of ['1m', '5m', '15m']) {
    const seg = $('err-' + k);
    seg.firstElementChild.textContent = ew[k];
    seg.classList.toggle('hot', ew[k] > 0);
  }
  const errTotal = Object.values(snap.errors_total || {}).reduce((a, b) => a + b, 0);
  $('err-meta').innerHTML = `<b>total</b> ${errTotal} since startup`;

  // --- Endpoint counters ---
  const rows = [];
  const paths = Array.from(new Set([
    ...Object.keys(snap.requests || {}),
    ...Object.keys(snap.errors_total || {}),
  ])).sort(new Intl.Collator('de', { sensitivity: 'base', numeric: true }).compare);
  for (const p of paths) {
    const n = snap.requests[p] || 0;
    const errs = snap.errors_total[p] || 0;
    rows.push(`<tr><td data-label="path">${esc(p)}</td><td class="num" data-label="requests">${n}</td>` +
      `<td class="num" data-label="5xx" style="${errs ? 'color:var(--red)' : ''}">${errs}</td></tr>`);
  }
  $('endpoints-rows').innerHTML = rows.length
    ? rows.join('') : '<tr><td colspan="3" class="empty">— none yet —</td></tr>';

  // --- Loaded models ---
  renderModels(snap);
  // Preload diagnostics: the most likely failure of preloading is silence,
  // and "enabled but no worker" is exactly that — so it gets the red badge.
  const pl = snap.preload;
  if (pl) {
    const dead = pl.enabled && !pl.worker_alive;
    $('preload-line').innerHTML =
      `preload <span class="badge ${pl.enabled ? 'warm' : 'cold'}">${pl.enabled ? 'enabled' : 'off'}</span> · ` +
      `worker <span class="badge ${dead ? 'err' : (pl.worker_alive ? 'ok' : 'cold')}">${pl.worker_alive ? 'ok' : 'down'}</span> · ` +
      `plans <b>${pl.plans}</b> · warm <b>${pl.warm}</b> · queue <b>${pl.queue_depth}</b>`;
  }

  // Frozen readouts win over the live values just written.
  if (frozenTs != null) applyFreeze();

  // --- Recent jobs (unified) ---
  renderJobs(snap);

  // Feed the header activity cluster from THIS page's stream — on /stats
  // the cluster opens no second EventSource (window._fwFeedActivity hook,
  // defined by ACTIVITY_CLUSTER_JS at body-end; guard for load order).
  if (typeof window._fwFeedActivity === 'function') {
    try { window._fwFeedActivity(snap); } catch (_) {}
  }

  // Severity pills are driven by SEV_POLLER_JS injected at body-end
  // (5-s poll of /sev), so no per-tick update needed here.
}

// --- Recent jobs table -------------------------------------------------------
// One table for every job kind: running jobs (snap.jobs) pinned on top,
// finished rows (snap.recent_transcriptions — the store keeps its historic
// name) below. Filters re-render from the last snapshot without waiting for
// the next SSE tick. Expanded rows survive re-renders via a ts-keyed set.
let lastJobsSnap = null;
const rjExpanded = new Set();

function segValRJ() {
  const b = document.querySelector('#rj-kind button.active');
  return b ? b.dataset.v : '';
}

function jobSpeed(r) {
  const wall = r.proc_dur || 0;
  if (r.kind === 'download') {
    const st = (r.stages || []).find(s => s.bytes);
    if (st && st.secs > 0) return ((st.bytes / 1048576) / st.secs).toFixed(1) + ' MB/s';
    return '—';
  }
  if (r.kind === 'translate') {
    const st = (r.stages || [])[0];
    const m = st && st.detail && String(st.detail).match(/^(\d+) segs/);
    if (m && wall > 0) return (Number(m[1]) / wall).toFixed(1) + ' seg/s';
    return '—';
  }
  return r.rtf != null ? r.rtf.toFixed(2) + '×' : '—';
}

function jobInput(r) {
  if (r.kind === 'download') {
    const st = (r.stages || []).find(s => s.bytes);
    return st ? (st.bytes / 1073741824).toFixed(2) + ' GB' : '—';
  }
  if (r.kind === 'translate') {
    const st = (r.stages || [])[0];
    const m = st && st.detail && String(st.detail).match(/^(\d+) segs/);
    return m ? m[1] + ' segs' : '—';
  }
  return (r.audio_dur || 0).toFixed(1) + ' s';
}

function pipeGlyph(r) {
  const stages = (r.stages || []).length ? r.stages
    : [{ name: r.kind === 'dictate' ? 'transcribing' : r.kind }];
  return '<span class="pipe">' + stages.map(s =>
    `<i class="${esc(s.name)}" title="${esc(s.name)} ${s.secs != null ? s.secs + 's' : ''}"></i>`
  ).join('') + '</span>';
}

function stageRows(r) {
  const stages = r.stages || [];
  if (!stages.length) {
    return '<div class="rj-stages"><span class="empty">no per-stage timings recorded</span></div>';
  }
  const max = Math.max(...stages.map(s => s.secs || 0), 0.001);
  return '<div class="rj-stages">' + stages.map(s => `
    <div class="rj-stage-row">
      <span class="nm">${esc(s.name)}</span>
      <span class="stage-bar"><i class="pipe-fill ${esc(s.name)}"
        style="width:${Math.max(2, (s.secs || 0) / max * 100).toFixed(1)}%;
               background:${stageColor(s.name)}"></i></span>
      <span class="secs">${(s.secs || 0).toFixed(2)} s</span>
      <span class="det">${esc([s.model, s.detail].filter(Boolean).join(' · '))}</span>
    </div>`).join('') + '</div>';
}

function stageColor(name) {
  // Keep in step with the .pipe i.<name> CSS rules above and with the
  // stage vocabulary main.py emits (vad hue matches /quick-config's .seg-vad).
  return ({ vad: '#93b76f', separating: 'var(--magenta)',
            transcribing: 'var(--cyan)',
            diarizing: 'var(--yellow)', translating: 'var(--green)',
            translate: 'var(--green)', download: 'var(--cyan)',
            downloading: 'var(--cyan)', preload: 'var(--help)' })[name]
    || 'var(--dim)';
}

function renderJobs(snap) {
  lastJobsSnap = snap;
  const kindF = segValRJ();
  const warnOnly = $('rj-warnonly').checked;
  const userSel = $('rj-user');
  const userF = userSel.value;

  const rt = (snap.recent_transcriptions || []);
  const running = (snap.jobs || []);

  // Keep the user select populated (preserving the current choice).
  const users = Array.from(new Set(rt.map(r => r.username).filter(Boolean))).sort();
  const want = ['', ...users];
  const have = Array.from(userSel.options).map(o => o.value);
  if (want.join(',') !== have.join(',')) {
    userSel.innerHTML = '<option value="">all users</option>'
      + users.map(u => `<option value="${esc(u)}">${esc(u)}</option>`).join('');
    userSel.value = want.includes(userF) ? userF : '';
  }

  const runRows = running
    .filter(j => !kindF || j.kind === kindF)
    .filter(j => !userF || j.user === userF)   // j.user is the username, like r.username
    .filter(() => !warnOnly)
    .map(j => {
      const pct = j.progress != null ? Math.round(j.progress * 100) : null;
      return `<tr class="rj-run">
      <td><span class="rj-spin" title="running"></span></td>
      <td class="ts">running · ${fmtSec(j.elapsed_s)}</td>
      <td><span class="kindchip ${esc(j.kind)}">${esc(j.kind)}</span></td>
      <td>${esc(j.stage || '')}</td>
      <td>${esc(j.model || '—')}</td>
      <td>${esc(j.user || '')}</td>
      <td class="num">${esc(j.detail || '—')}</td>
      <td class="num">—</td>
      <td class="num">${pct != null
        ? `<span class="rj-runbar"><i style="width:${pct}%"></i></span> ${pct}%`
        : (esc(j.step || '') || '—')}</td>
      <td><span class="badge warm">running</span></td>
    </tr>`;
    });

  const doneRows = rt
    .filter(r => !kindF || r.kind === kindF)
    .filter(r => !warnOnly || r.status !== 'ok')
    .filter(r => !userF || r.username === userF)
    .map(r => {
      const key = String(r.ts || 0);
      const open = rjExpanded.has(key);
      const who = [r.username, r.key_label].filter(Boolean).join(' · ');
      const main = `<tr class="rj-main${open ? ' open' : ''}" data-key="${esc(key)}">
      <td><span class="rj-caret">▸</span></td>
      <td class="ts" data-label="when" data-ts="${r.ts || 0}" title="${absTime(r.ts)}">${fmtWhen(r.ts)}</td>
      <td data-label="type"><span class="kindchip ${esc(r.kind)}">${esc(r.kind)}</span></td>
      <td data-label="pipeline">${pipeGlyph(r)}</td>
      <td data-label="model">${esc(r.model)}</td>
      <td data-label="user·key">${esc(who || '—')}</td>
      <td class="num" data-label="input">${jobInput(r)}</td>
      <td class="num" data-label="wall">${r.proc_dur.toFixed(2)} s</td>
      <td class="num" data-label="speed">${jobSpeed(r)}</td>
      <td data-label="status"><span class="badge ${r.status === 'ok' ? 'ok' : 'err'}">${esc(r.status)}</span></td>
    </tr>`;
      const detail = open
        ? `<tr class="rj-expand"><td colspan="10">${stageRows(r)}</td></tr>`
        : '';
      return main + detail;
    });

  const all = runRows.concat(doneRows);
  // Count what the table actually shows: running rows included, kind /
  // warnings-only / user filters applied -- not the raw finished list.
  $('rt-n').textContent = all.length;
  $('rt-rows').innerHTML = all.length
    ? all.join('')
    : '<tr><td colspan="10" class="empty">— no jobs yet —</td></tr>';
}

// Filter wiring: kind segments, warnings-only, user select — all re-render
// from the last snapshot immediately.
(() => {
  const kindCtl = $('rj-kind');
  if (kindCtl) {
    kindCtl.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b || b.classList.contains('active')) return;
      kindCtl.querySelectorAll('button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      if (lastJobsSnap) renderJobs(lastJobsSnap);
    });
  }
  const warnCtl = $('rj-warnonly');
  if (warnCtl) warnCtl.addEventListener('change', () => {
    if (lastJobsSnap) renderJobs(lastJobsSnap);
  });
  const userCtl = $('rj-user');
  if (userCtl) userCtl.addEventListener('change', () => {
    if (lastJobsSnap) renderJobs(lastJobsSnap);
  });
  // Row expansion (event delegation — rows are re-rendered every tick).
  const body = $('rt-rows');
  if (body) body.addEventListener('click', (e) => {
    const tr = e.target.closest('tr.rj-main');
    if (!tr) return;
    const key = tr.dataset.key;
    if (rjExpanded.has(key)) rjExpanded.delete(key); else rjExpanded.add(key);
    if (lastJobsSnap) renderJobs(lastJobsSnap);
  });
})();

// --- SSE consumer ----------------------------------------------------------
let es = null;
let recoveryTimer = null;
const statusEl = $('status');

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = 'pill ' + cls;
}

function openStream() {
  if (es) { try { es.close(); } catch {} }
  // EventSource sends the session cookie automatically (same-origin).
  es = new EventSource('/stats/stream');
  es.onmessage = (e) => {
    try {
      const snap = JSON.parse(e.data);
      pushHistory(snap);
      render(snap);
      setStatus('live', 'live');
    } catch (err) {
      console.warn('[stats] parse error', err);
    }
  };
  es.onerror = () => {
    setStatus('reconnecting…', 'paused');
    // Service may have restarted. Mirror /settings: poll a cheap idempotent
    // endpoint until it 200s, then force-reopen the SSE. Back off on repeated
    // failures (1.5s → ×1.7 → cap 30s) so a genuine outage doesn't hammer.
    if (recoveryTimer) return;
    let delay = 1500;
    const probe = async () => {
      try {
        const r = await fetch('/v1/models', { cache: 'no-store' });
        if (r.ok) {
          clearTimeout(recoveryTimer);
          recoveryTimer = null;
          // Drop history — server uptime jumped, the gap would be misleading.
          histX.length = 0;
          for (const k in hist) hist[k].length = 0;
          openStream();
          return;
        }
      } catch {}
      delay = Math.min(delay * 1.7, 30000);
      recoveryTimer = setTimeout(probe, delay);
    };
    recoveryTimer = setTimeout(probe, delay);
  };
}

// Visibility handler: closes the SSE on hidden tabs to defeat the browser's
// 6-connection-per-origin cap. Reopens on visible. Also cancels any in-flight
// recovery poll — otherwise a poll that succeeds in the background would
// openStream() concurrently with the visibility re-open, racing two
// EventSources for the same gid until one was orphaned.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    if (es) { try { es.close(); } catch {} es = null; }
    if (recoveryTimer) { clearInterval(recoveryTimer); recoveryTimer = null; }
    setStatus('paused (hidden)', 'paused');
  } else {
    openStream();
  }
});

// Auth rides the HttpOnly session cookie, sent automatically on both the
// fetch and the EventSource (same-origin) — no manual header or ?key= param.

// Initial fetch so the page renders before the first SSE tick arrives.
// role-admin used to be added here unconditionally — that leaked admin
// chrome to non-admins. OPEN_MODE_BANNER_JS is now the single source
// of truth (it sets role-admin iff whoami.is_admin=true).
fetch('/stats/snapshot', { cache: 'no-store' })
  .then(r => r.ok ? r.json() : null)
  .then(snap => {
    if (!snap) return;
    pushHistory(snap); render(snap);
  })
  .catch(err => console.warn('[stats] initial fetch failed', err))
  .finally(openStream);

})();
</script>

{{TIME_HELPERS_JS}}
{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
<script>
// Runs AFTER TIME_HELPERS_JS defines timeTick. Ages the relative suffix on the
// recent-transcriptions WHEN cells between SSE snapshots; re-queries [data-ts]
// each tick so it also catches rows added by the next snapshot render.
timeTick('#rt-rows [data-ts]');
</script>
</body></html>"""
# /static is cacheable (ETag), the page is not: the version in the query
# string is what makes a new build fetch a new stats.js. Substituted once
# at import — render_page() has a fixed placeholder list and caches by
# template string.
_STATS_VIEWER_HTML = _STATS_VIEWER_HTML.replace(
    "__ASSET_V__", build_info.APP_VERSION.replace("+", "."))

