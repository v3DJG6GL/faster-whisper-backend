"""
/stats system overview dashboard.

Routes (all gated by web_common.require_user_webui_host):

  GET /stats           HTML page (single-file inline-HTML+CSS+JS, mirrors /logs)
  GET /stats/snapshot  one-shot JSON (_build_payload): ts + metrics + system +
                       severity counts, cut to the viewer's StatsScope (scope /
                       machine / server for own scope); `lite=1` is the
                       header activity cluster's diet payload
  GET /stats/stream    SSE: same JSON, ~1 Hz (1 s data cadence defeats idle-proxy timeouts; no separate keepalive frame)
  GET /stats/usage     the usage document (usage_store.overview) for a window
  GET /stats/pick      the who / keys pickers' ranked rows
  GET /stats/jobs      one page of the recent-jobs table (cursor paging)
  GET /stats/tail      wait / turnaround / failures / per-model (usage_store.tail)
  GET /stats/history   range-mode machine history from system_metrics_store

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
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse

import hmac
import hashlib

from faster_whisper_backend import build_info
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import version as settings_version
from faster_whisper_backend.core import jobs
from faster_whisper_backend.stats import metrics
from faster_whisper_backend.runtime import model_sizes
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.stats import system_metrics_store
from faster_whisper_backend.runtime import system_stats
from faster_whisper_backend.core import web_common
from faster_whisper_backend.auth import dependencies as auth

log = logging.getLogger(__name__)
from faster_whisper_backend.auth.dependencies import require_page
from faster_whisper_backend.paths import REPO_ROOT
from faster_whisper_backend.core import templates
from faster_whisper_backend.auth import api_keys_store
from faster_whisper_backend.stats import recent_transcriptions_store
from faster_whisper_backend.stats import usage_store

router = APIRouter(prefix="/stats")

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
                                  cfg.STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS (read at call
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
    if effective is not None:
        return StatsScope(
            "own", effective, caller_uid, True,
            bool(getattr(cfg, "STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS", False)))
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
    {busy, mem_used_mb, mem_total_mb} (None on a GPU-less server, like the
    machine payload's) so the header cluster keeps working."""
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
                "gpu": ({"busy": g["busy"], "mem_used_mb": g["mem_used_mb"],
                         "mem_total_mb": g["mem_total_mb"]}
                        if g["present"] else None),
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


# Wall-clock ceiling between re-resolves when the config version does not
# move: a cookie session reaching SESSION_TTL_S bumps nothing (expiry is only
# checked at lookup), so a version-only check streamed past it forever.
_REAUTH_INTERVAL_S = 60.0


def _rescope_on_version_change(request: Request, seen_version: int,
                               resolved_at: float
                               ) -> tuple[StatsScope, int, float] | None:
    """Stream helper: when settings_version.config_version() moved since
    `seen_version` (a permission edit bumps it), or _REAUTH_INTERVAL_S passed
    since `resolved_at` (time.monotonic(); a session expiring bumps nothing),
    re-resolve the caller and return the fresh (scope, version, resolved_at);
    None when nothing is due. Raises HTTPException when the caller lost
    access, which ends the stream (the page reconnects and gets the 401/403).

    Re-resolving rather than ending the stream matters with several workers:
    every sibling commit — including a key's debounced last_used_ts touch —
    bumps the version, and an ended stream makes the page discard its
    two-minute sparkline history on reconnect."""
    current = settings_version.config_version()
    now = time.monotonic()
    if current == seen_version and now - resolved_at < _REAUTH_INTERVAL_S:
        return None
    fresh = auth.resolve_user_for_page_sse(request, "stats")
    return stats_scope_for(fresh), current, now


@router.get(
    "",
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
    "/snapshot",
    dependencies=[Depends(_require_stats_host)],
)
async def stats_snapshot(
    lite: int = 0,
    user: dict[str, Any] = Depends(require_page("stats")),
) -> dict[str, Any]:
    """One-shot JSON. Useful for scripts and for the page's initial render.
    `?lite=1` returns the header activity cluster's diet payload."""
    return await asyncio.to_thread(_build_payload, stats_scope_for(user), lite=bool(lite))


# Salt for the opaque labels a non-admin "all" viewer sees on the leaderboard.
# Derived from the data directory so every worker in a multi-process deploy
# produces the same labels — a picker selection from one worker resolves in
# another. Changes across installs; never a plain hash of the id.
_SCRUB_SALT = hashlib.sha256(b"stats-scrub:" + cfg._DATA_DIR.encode()).digest()[:16]


def _opaque_user_label(user_id: str) -> str:
    return "user-" + hmac.new(_SCRUB_SALT, (user_id or "").encode(),
                              hashlib.sha256).hexdigest()[:8]


def _csv(v: str | None) -> list[str]:
    """A comma-separated query value → de-duplicated non-empty items."""
    return list(dict.fromkeys(x.strip() for x in (v or "").split(",") if x.strip()))


def _unscrub(dim: str, labels: list[str], scrub: bool, caller_uid: str | None) -> list[str]:
    """The ids behind the picked labels. Admin (and own) viewers send ids.
    A non-admin "all" viewer sends the row `id` the picker / leaderboard
    gave it (or, for older links, the opaque label it was shown), so each
    value is matched against every id the rollups know and against its
    opaque label. Unknown values are dropped rather than refused."""
    if not scrub:
        return labels
    want = set(labels)
    fn = _opaque_user_label if dim == "user" else _opaque_key_label
    out = [i for i in usage_store.distinct_ids(dim + "_id") if i in want or fn(i) in want]
    if caller_uid and dim == "user" and caller_uid in want:
        out.append(caller_uid)
    resolved = list(dict.fromkeys(out))
    if want and not resolved:
        log.debug("_unscrub: none of %d %s labels matched a known id", len(want), dim)
        return ["__no_match__"]
    return resolved


def _one_or_many(ids: list[str]) -> str | list[str]:
    """One picked id keeps the v2 shape (`filter.key_id == "k1"`); several
    go as a list (an IN clause in the store)."""
    return ids[0] if len(ids) == 1 else ids


def _label_rows(rows: list[dict[str, Any]], by: str, *, scrub: bool,
                caller_uid: str | None) -> dict[str, dict[str, Any]]:
    """Resolve display names on leaderboard / picker rows in place and
    return {id: {label, user_label, me}} for the chart lines. Revoked
    users/keys still resolve; sentinels stay literal. Non-admin "all"
    viewers get opaque labels instead — only their own rows keep a name
    (and are flagged `me`)."""
    names = api_keys_store.get_usernames(
        [r["user_id"] for r in rows if r.get("user_id")])

    # A sentinel id ("(open-mode)") names nobody: scrubbing it only made it
    # read like a real person on the board.
    def _user_label(uid: str) -> str:
        if scrub and uid != caller_uid and not (uid or "").startswith("("):
            return _opaque_user_label(uid)
        return names.get(uid) or uid

    labels: dict[str, dict[str, Any]] = {}
    for r in rows:
        mine = bool(caller_uid) and r.get("user_id") == caller_uid
        if by == "user":
            r["label"] = _user_label(r["id"])
        elif by == "key":
            kid = r["id"]
            if scrub and not mine and not (kid or "").startswith("("):
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
        labels[r["id"]] = {k: r[k] for k in ("label", "user_label", "me") if k in r}
    return labels


def _opaque_key_label(key_id: str) -> str:
    return "key-" + hmac.new(_SCRUB_SALT, ("k:" + (key_id or "")).encode(),
                             hashlib.sha256).hexdigest()[:8]


@router.get(
    "/usage",
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
    kinds: str | None = None,
    users: str | None = None,
    keys: str | None = None,
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
    `metric` ∈ {audio_s, words, requests, errors, processing_s, sessions};
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
    non-admin passing `?user=` gets 403 — it is never trusted.

    Filters (comma lists, echoed under `filter`): `kinds` keeps only those
    job kinds (422 on an unknown one); `users` / `keys` keep only those
    owners / keys — as picked in the page's who / keys pickers, so a
    non-admin "all" viewer sends the opaque labels it was shown and they
    are mapped back here. `users` needs the "all" scope (403 for own)."""

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
    kind_list = _csv(kinds)
    for k in kind_list:
        if k not in usage_store.KINDS:
            raise HTTPException(422, detail=f"unknown kind: {k!r}")
    user_list = _csv(users)
    if user_list and scope.scope == "own":
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            detail="users= needs stats scope 'all'")
    key_list = _csv(keys) or ([key] if key else [])
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
        uid_filter = _one_or_many(_unscrub("user", user_list, scrub, caller_uid)) if user_list else None
        kid_filter = _one_or_many(_unscrub("key", key_list, scrub, caller_uid)) if key_list else None
        out = usage_store.overview(
            user_id=uid_filter if uid_filter is not None else scope.user_id,
            key_id=kid_filter, tz=w.tz,
            tz_name=w.tz_name, days=w.days, from_day=w.from_day,
            to_day=w.to_day, all_time=w.all_time, with_stages=w.with_stages,
            by=by, metric=metric, bucket=bucket, compare=compare,
            jobs_retention_days=jobs_retention, kinds=kind_list)

        # Resolve display names server-side (the /stats client has no
        # api-keys data).
        rows = out["leaderboard"]
        labels = _label_rows(rows, by, scrub=scrub, caller_uid=caller_uid)
        for r in rows:
            # v1 shape: the metrics flat on the row as well.
            r.update(r["totals"])
        for ln in out["lines"]:
            ln.update(labels.get(ln["id"], {}))
        out["scope"] = scope.scope
        # What the page asked for, as it asked (labels for scrubbed viewers).
        out["filter"].update({"kinds": kind_list, "users": user_list, "keys": key_list})
        # Display names for the picked user ids. The page otherwise learns
        # them only from the who picker or a by=user board, so a pasted or
        # reloaded `users=` link matched the jobs table's running rows (keyed
        # by username) against raw ids. Names only where the viewer already
        # sees identities, or for their own id.
        if user_list:
            names = api_keys_store.get_usernames(user_list)
            out["filter"]["user_labels"] = {
                u: names[u] for u in user_list
                if names.get(u) and (not scrub or u == caller_uid)}
        return out

    return await asyncio.to_thread(_gather)


@router.get(
    "/pick",
    dependencies=[Depends(_require_stats_host)],
)
async def stats_pick(
    dim: str = "user",
    days: int | None = None,
    tz: str | None = None,
    from_: int | None = Query(default=None, alias="from"),
    to: int | None = None,
    all: bool = False,
    with_: str | None = Query(default=None, alias="with"),
    metric: str = "audio_s",
    kinds: str | None = None,
    users: str | None = None,
    keys: str | None = None,
    user: dict[str, Any] = Depends(require_page("stats")),
) -> dict[str, Any]:
    """Options for the page's who / keys pickers: every user (or key) with
    usage in the window, ranked by `metric`, labelled like the leaderboard
    (opaque for non-admin "all" viewers, `me` on the caller's own rows).
    `dim=key` with `users` lists only those users' keys; `dim=user` with
    `keys` ranks the users by those keys only (the page sends its filter
    slice minus the picker's own dimension). Own scope may list its keys
    but not users (403)."""

    if dim not in ("user", "key"):
        raise HTTPException(422, detail="dim must be user or key")
    scope = stats_scope_for(user)
    if dim == "user" and scope.scope == "own":
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            detail="per-user picker needs stats scope 'all'")
    caller_uid = user.get("user_id") or None
    scrub = not scope.include_identity
    kind_list = _csv(kinds)
    for k in kind_list:
        if k not in usage_store.KINDS:
            raise HTTPException(422, detail=f"unknown kind: {k!r}")
    user_list = _csv(users) if scope.scope != "own" else []
    key_list = _csv(keys)
    if days is not None and days <= 0:
        days, all = None, True
    try:
        w = usage_store.parse_window_params(
            days=days, from_day=from_, to_day=to, all_time=all, with_=with_, tz=tz)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))

    def _gather() -> dict[str, Any]:
        uid_filter = _one_or_many(_unscrub("user", user_list, scrub, caller_uid)) if user_list else None
        kid_filter = _one_or_many(_unscrub("key", key_list, scrub, caller_uid)) if key_list else None
        out = usage_store.overview(
            user_id=uid_filter if uid_filter is not None else scope.user_id,
            key_id=kid_filter,
            tz=w.tz, tz_name=w.tz_name, days=w.days, from_day=w.from_day,
            to_day=w.to_day, all_time=w.all_time, with_stages=w.with_stages,
            by=dim, metric=metric, top_k=1, limit=500, kinds=kind_list)
        rows = out["leaderboard"]
        _label_rows(rows, dim, scrub=scrub, caller_uid=caller_uid)
        return {
            "dim": dim, "metric": out["metric"], "range": out["range"],
            "rows": [{"id": r["id"], "label": r["label"],
                      **({"user_label": r["user_label"]} if "user_label" in r else {}),
                      **({"me": True} if r.get("me") else {}),
                      "value": float(r["totals"].get(out["metric"], 0) or 0)}
                     for r in rows],
        }

    return await asyncio.to_thread(_gather)


JOB_KINDS: frozenset[str] = frozenset(jobs.KINDS)
JOB_STATUSES: frozenset[str] = frozenset(("ok", "error", "cancelled", "failed"))


@router.get(
    "/jobs",
    dependencies=[Depends(_require_stats_host)],
)
async def stats_jobs(
    cursor: float | None = None,
    limit: int = 50,
    kind: str | None = None,
    status_q: str | None = Query(default=None, alias="status"),
    slow_rtf: float | None = None,
    user_q: str | None = Query(None, alias="user"),
    users: str | None = None,
    user: dict[str, Any] = Depends(require_page("stats")),
) -> dict[str, Any]:
    """The jobs table beyond the snapshot's last few rows: finished jobs
    newest-first, `limit` (1..200) per page, paged with `cursor` = the
    previous page's `next_cursor` (a created_ts; null when exhausted).
    Filters: `kind` (a recent-jobs kind, or a comma list: any of them —
    the page's kind chips map onto several), `status` (ok | error |
    cancelled | failed), `slow_rtf` (processing longer than that fraction
    of the audio), `users` (comma list of ids, or the opaque labels a
    non-admin "all" viewer was shown — mapped back like /stats/usage; 403
    for own scope). `running` — the live registry rows with their cancel
    handles for the caller's own jobs (admins: all) — comes only with the
    first page. Scoped like the snapshot: own rows for "own", every user
    with identities scrubbed for non-admin "all", `?user=` preview for
    admins (403 for anyone else)."""

    is_admin = bool(user.get("is_admin"))
    if user_q and not is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="user= is admin-only")
    scope = stats_scope_for(user, preview_user_id=(user_q or None) if is_admin else None)
    kind_list = _csv(kind)
    for k in kind_list:
        if k not in JOB_KINDS:
            raise HTTPException(422, detail=f"unknown kind: {k!r}")
    if status_q and status_q not in JOB_STATUSES:
        raise HTTPException(422, detail=f"unknown status: {status_q!r}")
    user_list = _csv(users)
    if user_list and scope.scope == "own":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="users= needs stats scope 'all'")
    caller_uid = user.get("user_id") or None
    scrub = not scope.include_identity
    n = max(1, min(int(limit), 200))

    def _page() -> dict[str, Any]:
        uid_filter = _one_or_many(_unscrub("user", user_list, scrub, caller_uid)) if user_list else None
        rows = recent_transcriptions_store.list_recent(
            before_ts=cursor, limit=n,
            user_id_filter=uid_filter if uid_filter is not None else scope.user_id,
            kind=kind_list or None, status=status_q or None, slow_rtf=slow_rtf)
        out = {
            "jobs": [metrics.project_recent_row(
                r, include_identity=scope.include_identity) for r in rows],
            "next_cursor": rows[-1]["ts"] if len(rows) >= n else None,
            "scope": scope.scope,
        }
        if not cursor:
            out["running"] = jobs.jobs_snapshot(
                include_identity=scope.include_identity, user_id=scope.user_id,
                viewer_user_id=scope.viewer_user_id)
        return out

    return await asyncio.to_thread(_page)


@router.get(
    "/tail",
    dependencies=[Depends(_require_stats_host)],
)
async def stats_tail(
    days: int | None = None,
    tz: str | None = None,
    from_: int | None = Query(default=None, alias="from"),
    to: int | None = None,
    all: bool = False,
    kind: str | None = None,
    key: str | None = None,
    user_q: str | None = Query(None, alias="user"),
    users: str | None = None,
    keys: str | None = None,
    user: dict[str, Any] = Depends(require_page("stats")),
) -> dict[str, Any]:
    """The tail of the distribution, from the per-job rows: queue wait
    (p50 / p95 / max, by day), the turnaround histogram with the queue-wait
    share per bucket, failures by stage and class, per-model runs / RTF /
    wait, and deltas against the immediately preceding window. Same window
    vocabulary as /stats/usage (days | from/to | all, tz); `kind` narrows to
    one job kind or a comma list of them, `key` to one API key, `users` /
    `keys` to the page's picked owners / keys (as in /stats/usage). Scoped like /stats/usage: own rows
    for "own", every user for "all", `?user=` preview for admins (403 for
    anyone else). The per-job rows keep USAGE_JOBS_RETENTION_DAYS; a window
    that starts earlier says so in range.truncated_to_days."""

    is_admin = bool(user.get("is_admin"))
    if user_q and not is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="user= is admin-only")
    scope = stats_scope_for(user, preview_user_id=(user_q or None) if is_admin else None)
    if days is not None and days <= 0:
        days, all = None, True
    try:
        w = usage_store.parse_window_params(days=days, from_day=from_, to_day=to,
                                            all_time=all, tz=tz)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    kind_list = _csv(kind)
    for k in kind_list:
        if k not in usage_store.KINDS:
            raise HTTPException(422, detail=f"unknown kind: {k!r}")
    user_list = _csv(users)
    if user_list and scope.scope == "own":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="users= needs stats scope 'all'")
    key_list = _csv(keys) or ([key] if key else [])
    caller_uid = user.get("user_id") or None
    scrub = not scope.include_identity
    jobs_retention = int(getattr(cfg, "USAGE_JOBS_RETENTION_DAYS", 365) or 0)

    def _gather() -> dict[str, Any]:
        uid_filter = _one_or_many(_unscrub("user", user_list, scrub, caller_uid)) if user_list else None
        kid_filter = _one_or_many(_unscrub("key", key_list, scrub, caller_uid)) if key_list else None
        return usage_store.tail(
            user_id=uid_filter if uid_filter is not None else scope.user_id,
            key_id=kid_filter, kind=kind_list or None, tz=w.tz, tz_name=w.tz_name,
            days=w.days, from_day=w.from_day, to_day=w.to_day, all_time=w.all_time,
            jobs_retention_days=jobs_retention)

    doc = await asyncio.to_thread(_gather)
    doc["scope"] = scope.scope
    return doc


@router.get(
    "/history",
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
    the sampler's system_metrics rows, downsampled to `step` seconds (default: the
    smallest step that keeps the window under ~2 000 points, never below the
    sample cadence). `from`/`to` are epoch seconds; default the last hour.
    Own-scope viewers get it only when they see the machine cards (the
    same rule as the live payload: utilisation curves reveal other
    people's jobs)."""

    scope = stats_scope_for(user)
    if not scope.sees_machine:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            detail="system metrics history needs the system metrics cards")
    if metric not in system_metrics_store.METRICS:
        raise HTTPException(422,
                            detail=f"unknown metric: {metric!r}")
    now = time.time()
    t1 = float(to) if to is not None else now
    t0 = float(from_) if from_ is not None else t1 - 3600
    if not (math.isfinite(t0) and math.isfinite(t1)):
        raise HTTPException(422, detail="'from'/'to' must be finite numbers")
    # Finite is not enough: int() of 1e300 (or of a huge `step`) overflows
    # SQLite's 64-bit INTEGER in list_series, a 500 instead of a 422.
    if t0 < 0 or t1 > now + 86400:
        raise HTTPException(422, detail="'from'/'to' out of range")
    if t0 >= t1:
        raise HTTPException(422,
                            detail="'from' is not before 'to'")
    t0 = max(t0, t1 - 3650 * 86400)
    cadence = max(1, int(getattr(cfg, "STATS_SYSTEM_METRICS_SAMPLE_S", 10) or 10))
    auto = max(cadence, int(-(-(t1 - t0) // 2000)))
    # A step longer than the window is one bucket anyway; capping it keeps
    # an absurd value inside SQLite's integer range.
    step_s = (min(max(cadence, int(step)), max(cadence, int(t1 - t0)))
              if step else auto)
    # A store whose init_db failed at startup (logged, non-fatal) raised from
    # _require_conn() here: a bare 500 with a traceback on every spark fetch.
    if not system_metrics_store.is_open():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="system metrics history unavailable — see the startup log "
                   "for the system-metrics store error (STATS_SYSTEM_METRICS_DB "
                   "path/permissions)")
    series = await asyncio.to_thread(
        system_metrics_store.list_series, metric=metric, from_ts=t0,
        to_ts=t1, step_s=step_s)
    return {"metric": metric, "from": int(t0), "to": int(t1), "step": step_s,
            **series}


@router.get(
    "/stream",
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
    without the reconnect churn of ending the stream on every bump — and
    at least every _REAUTH_INTERVAL_S, so an expired session ends it."""
    _lite = bool(lite)
    scope = stats_scope_for(user)
    seen = settings_version.config_version()
    resolved_at = time.monotonic()

    async def gen():
        nonlocal scope, seen, resolved_at
        while True:
            payload = await asyncio.to_thread(
                _build_payload, scope, lite=_lite)
            yield f"data: {json.dumps(payload, allow_nan=False, default=str)}\n\n"
            await asyncio.sleep(1.0)
            try:
                fresh = await asyncio.to_thread(
                    _rescope_on_version_change, request, seen, resolved_at)
            except HTTPException:
                return
            if fresh is not None:
                scope, seen, resolved_at = fresh

    return web_common.sse_response(gen())


# --- HTML template -----------------------------------------------------------
# Single-file, no build step. Mirrors the /logs and /settings style. uPlot is
# loaded from the local /static mount — no CDN, works offline.

_STATS_VIEWER_HTML = templates.load(__file__, "stats.html")
# /static is cacheable (ETag), the page is not: the version in the query
# string is what makes a new build fetch a new stats.js. It is a content
# hash of the file (APP_VERSION alone is unchanged between dev restarts,
# so browsers kept serving a stale copy). Substituted once at import —
# render_page() has a fixed placeholder list and caches by template string.
def _asset_version() -> str:
    from pathlib import Path
    try:
        digest = hashlib.sha1((Path(REPO_ROOT) / "static" / "stats.js").read_bytes()).hexdigest()[:10]
    except OSError:
        return build_info.APP_VERSION.replace("+", ".")
    return digest


ASSET_VERSION = _asset_version()
_STATS_VIEWER_HTML = _STATS_VIEWER_HTML.replace("__ASSET_V__", ASSET_VERSION)

