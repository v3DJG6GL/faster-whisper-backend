"""Admin /captures page + JSON APIs for Whisper fine-tuning data review.

Layout:

  GET    /captures                       HTML page (admin host + token)
  GET    /captures/api/list              paginated metadata (no audio)
  GET    /captures/api/{cid}             single row incl. word timestamps
  GET    /captures/api/by-request/{rid}  cross-link from /reports
  GET    /captures/api/{cid}/audio       streams the raw audio (Range OK)
  PATCH  /captures/api/{cid}             corrections / corrected_text /
                                         admin_notes / status
  DELETE /captures/api/{cid}             single delete
  POST   /captures/api/clear             typed-confirmation wipe
  GET    /captures/api/export            tar.gz (manifest.jsonl + audio/)

Mutating routes use a HEADER-ONLY admin-token guard (no ?token= fallback).
The audio endpoint is GET and also header-only — browsers consume it via
fetch() + URL.createObjectURL(blob), not <audio src=...>, so no token in
URL is needed.

Word-timestamp + chip schema matches /reports' (text_corrections), so a
future "promote a capture into a report" flow needs no translation.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import io
import json
import logging
import os
import tarfile
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from pydantic import BaseModel, Field

from faster_whisper_backend.auth import api_keys_store
from faster_whisper_backend.captures import merge_proposer as captures_merge_proposer
from faster_whisper_backend.captures import reapply as captures_reapply
from faster_whisper_backend.captures import samples as capture_samples
from faster_whisper_backend.captures import store as captures_store
from faster_whisper_backend.captures import vad_reprocess as captures_vad_reprocess
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.auth import rate_limit
from faster_whisper_backend.core import store_common
from faster_whisper_backend.core import text_corrections
from faster_whisper_backend.core import web_common
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.core.web_common import require_user_webui_host
from faster_whisper_backend.auth.dependencies import get_current_user, require_admin, require_page
from faster_whisper_backend.core import templates
from faster_whisper_backend.captures import merge as audio_merge
from faster_whisper_backend.captures import samples_store as capture_samples_store
from faster_whisper_backend.captures import vad_trim as audio_vad_trim

logger = logging.getLogger("whisper-api")

# Router-level dependency: only the IP gate. The page-permission gate
# (`require_page("captures")`) must NOT live at router level because it
# transitively requires a bearer (via get_current_user), and the HTML
# page is fetched by browser navigation which can't pass Authorization
# headers — the login modal runs in the page's own JS. Page-perm gates
# therefore live per-API-route, where fetch() already attaches the
# bearer. Mutation routes additionally `Depends(require_admin)` for
# system-wide writes (clear, reprocess-all, export).
# One dedicated worker for the merge proposer. NOT the default executor:
# that is the pool main.transcribe runs CT2 inference in, and a proposer sweep
# is seconds of pure CPU that any captures-page holder can trigger without a
# rate limit. Built lazily so importing this module costs nothing, and
# single-worker so the sweeps serialise against each other exactly as
# _SWEEP_LOCK already makes them.
_PROPOSER_POOL: "concurrent.futures.ThreadPoolExecutor | None" = None


def _proposer_pool() -> "concurrent.futures.ThreadPoolExecutor":
    global _PROPOSER_POOL
    if _PROPOSER_POOL is None:
        _PROPOSER_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="captures-proposer")
    return _PROPOSER_POOL


# Serialises every read-merge-write of member `corrections`: the single
# capture PATCH, the sample PATCH fan-out and preview-save-chips all run in
# worker threads now, so without it one save can land between another's read
# and its writes, and the three-way merge would run against a stale `current`
# (a lost chip edit). A threading.Lock, taken INSIDE the to_thread callables,
# so a waiter parks a worker thread, never the event loop.
_corrections_write_lock = threading.Lock()


router = APIRouter(
    prefix="/captures",
    dependencies=[Depends(require_user_webui_host)],
)


# ---------------------------------------------------------------------
# Rate limit per identity on the audio-streaming endpoints
# ---------------------------------------------------------------------
# Sized for how the review UI actually behaves, not for a steady cadence:
# working a captures backlog means auditioning card after card, plus the
# merge modal's preview and the sample players, so a reviewer produces bursts
# of dozens of fetches within a few seconds. The budget is per IDENTITY now
# rather than per host, so a shared office egress IP no longer pools every
# reviewer's clicks into one bucket.

_audio_rate = rate_limit.FixedWindow(
    config_field="CAPTURES_AUDIO_RATE_PER_MIN",
    window_s=60.0,
    default_max=240,
    message="Too many audio requests ({limit}/min) — retry in "
            "{retry_after}s.",
)


def _audit_cross_user_read(
    user: dict[str, Any], row: dict[str, Any] | None,
    kind: str, row_id: str,
) -> None:
    """Emit an INFO line when a non-admin viewer reads a row owned by a
    different user (scope=all path). Self-reads + admin-host requests
    that already have access don't audit — only the data-leaving-the-
    user-pool case is interesting. Cheap; makes DSGVO Art. 9 data-
    subject access requests answerable from the standard log stream.
    Silent for admin users (they bypass scope and would otherwise
    audit every read on their own dashboard)."""
    if user.get("is_admin"):
        return
    caller_uid = user.get("user_id") or ""
    owner_uid = (row or {}).get("user_id") or ""
    if not owner_uid or owner_uid == caller_uid:
        return
    logger.info(
        "[audit] cross-user-read user=%s(uid=%s) read %s id=%s owner=%s",
        # Usernames are length-capped but not character-screened, so a bare
        # CR/LF would split this audit record into extra attacker-written
        # lines in the /logs viewer. Same treatment as api_keys_store's calls.
        store_common.log_safe(user.get("username") or "?"),
        caller_uid[:8] if caller_uid else "?",
        kind,
        (row_id or "?")[:8],
        owner_uid[:8],
    )


def _assert_member_sample_not_locked(
    row: dict[str, Any], user: dict[str, Any],
) -> None:
    """Reject a non-admin mutate/delete of a capture that is a member of a
    LOCKED sample. Mirrors dissolve_sample_api's `is_locked and not is_admin
    → 409` guard so the member surface can't bypass the sample lock: deleting
    a member auto-dissolves (and destroys the merged WAV of) its parent
    sample, and editing/reprocessing a member rewrites the text the locked
    sample derives at read time. Admins are exempt; captures with no parent
    sample pass through unchanged."""
    sid = row.get("sample_id")
    if not sid or user.get("is_admin"):
        return
    sample = capture_samples_store.get_sample(sid)
    if sample is not None and sample.get("is_locked"):
        raise HTTPException(status.HTTP_409_CONFLICT, "sample is locked")


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

class CorrectionIn(BaseModel):
    model_config = {"extra": "forbid"}
    wrong: str = ""
    correct: str = ""
    idx: int | None = None
    idx_end: int | None = None


class PatchCaptureIn(BaseModel):
    model_config = {"extra": "forbid"}
    status: Literal["new", "reviewed", "ready", "dismissed"] | None = None
    # Bounds mirror what captures_store.update_capture truncates to
    # (_CAP_CORRECTED / _CAP_ADMIN_NOTES), so nothing a client may
    # legitimately save is rejected here.
    corrected_text: str | None = Field(default=None, max_length=100_000)
    corrections: list[CorrectionIn] | None = Field(
        default=None, max_length=text_corrections.CAP_CORRECTIONS)
    # Snapshot of `corrections` the client loaded with this capture.
    # When provided alongside `corrections`, the server applies a
    # three-way merge against the current DB state so a concurrent
    # write (another admin in another tab, or a group save touching this
    # member) doesn't get clobbered by the user's save. Omitted → legacy
    # replace.
    baseline_corrections: list[CorrectionIn] | None = Field(
        default=None, max_length=text_corrections.CAP_CORRECTIONS)
    admin_notes: str | None = Field(default=None, max_length=8000)


class BulkStatusIn(BaseModel):
    """PATCH /captures/api/bulk — one status for many ids. The cap equals
    the list endpoint's page cap, so a "select all loaded" is always one
    request. `audio_missing` is system-set and never a target."""
    model_config = {"extra": "forbid"}
    ids: list[str] = Field(min_length=1, max_length=1000)
    status: Literal["new", "reviewed", "ready", "dismissed"]


class BulkIdsIn(BaseModel):
    model_config = {"extra": "forbid"}
    ids: list[str] = Field(min_length=1, max_length=1000)


class ClearIn(BaseModel):
    model_config = {"extra": "forbid"}
    # Typed confirmation — the literal string "CAPTURES" must be sent.
    # Training data is irrecoverable; the modal asks the admin to type it.
    confirm: str = Field(default="", max_length=32)


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

@router.get(
    "",
    response_class=HTMLResponse,
)
async def captures_page() -> HTMLResponse:
    if not getattr(cfg, "ADMIN_UI_ENABLED", False):
        return HTMLResponse("Admin UI disabled.", status_code=404)
    cap_s = float(getattr(cfg, "CAPTURES_SAMPLE_MAX_DURATION_S", 29.9))
    html = web_common.render_page(_CAPTURES_HTML, current="captures")
    html = html.replace("{{SAMPLE_CAP_S}}", f"{cap_s:g}")
    return HTMLResponse(html, media_type="text/html")


# ---------------------------------------------------------------------
# JSON APIs
# ---------------------------------------------------------------------

# Cap on the admin-only `?user_id=a,b,...` list (one bind variable each).
_MAX_OWNER_FILTER_IDS = 200


def _effective_owner_filter(
    user: dict[str, Any], user_filter: str | None,
) -> "str | list[str] | None":
    """Owner scope for the list / stats queries. `scope=own` callers are
    pinned to themselves by effective_user_id_for; `scope=all` callers
    (incl. admins) see everyone and may narrow with the admin-only
    `?user_id=a,b` (comma-separated, the speaker picker) — a non-admin's
    query is ignored. More than _MAX_OWNER_FILTER_IDS ids → 422: every id
    is one SQL bind variable in each of the list / counts / stats queries,
    and past SQLite's variable limit that is an unhandled 500."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    effective: "str | list[str] | None" = perms.effective_user_id_for(
        "captures", caller_uid)
    if user.get("is_admin") and user_filter:
        ids = [u.strip() for u in user_filter.split(",") if u.strip()]
        if len(ids) > _MAX_OWNER_FILTER_IDS:
            raise HTTPException(
                422, f"user_id: at most {_MAX_OWNER_FILTER_IDS} ids")
        if ids:
            effective = ids[0] if len(ids) == 1 else ids
    return effective


@router.get(
    "/api/list",
    dependencies=[Depends(require_page("captures"))],
)
async def list_captures_api(
    status_filter: str = Query("all", alias="status"),
    limit: int = Query(200, ge=1, le=1000),
    before_ts: float | None = Query(None),
    user_filter: str | None = Query(None, alias="user_id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    """Scope-aware list. `scope=own` users see only their own captures;
    `scope=all` users (incl. admins) see every capture and may narrow
    via the admin-only `?user_id=...` query for the per-user dropdown."""
    effective_user = _effective_owner_filter(user, user_filter)

    def _render() -> str:
        rows = captures_store.list_captures(
            status=status_filter, limit=limit, before_ts=before_ts,
            user_id=effective_user,
        )
        # Per-row pipeline self-heal happens in get_capture_api (expand) only.
        # Running it here would be 2 _postprocess_text calls × `limit` rows per
        # list render, which dominates response time on /captures with
        # limit=500.
        usernames = api_keys_store.get_usernames(
            [r.get("user_id") for r in rows])
        for r in rows:
            _apply_trim_to_capture_row(r)
            r["username"] = usernames.get(r.get("user_id"))
        return json.dumps({
            "captures": rows,
            "counts": captures_store.counts_by_status(user_id=effective_user),
            "enabled": bool(getattr(cfg, "CAPTURES_RECORDING_ENABLED", False)),
            "retention_days": int(getattr(cfg, "CAPTURES_RETENTION_DAYS", 0)),
            "total_count": captures_store.count(user_id=effective_user),
            "is_admin": bool(user.get("is_admin")),
            "user_id": user.get("user_id"),
        }, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    # Query AND serialization off the loop, exactly like
    # reports_routes.list_reports_api: up to 1000 rows each carrying four
    # 50k-char text columns plus a per-row json.loads, then two full-table
    # aggregates, then JSONResponse's own json.dumps of the whole body — of
    # which the serialization was the larger half there (1.8 s of 2.17 s).
    # Rows are plain types, so these are the same bytes JSONResponse emits.
    body = await asyncio.to_thread(_render)
    return Response(content=body, media_type="application/json")


@router.get(
    "/api/stats",
    dependencies=[Depends(require_page("captures"))],
)
async def captures_stats_api(
    user_filter: str | None = Query(None, alias="user_id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    """Queue totals for the summary strip: hours + counts by status, review
    progress, ready-this-week, hours per speaker. Scoped exactly like the
    list (own-scope callers see only themselves). The page calls it WITHOUT
    a speaker filter so the strip describes the whole queue while the list
    is narrowed; `?user_id=` exists for API callers."""
    effective_user = _effective_owner_filter(user, user_filter)

    def _render() -> str:
        st = captures_store.stats(user_id=effective_user)
        users = st.get("by_user") or []
        names = api_keys_store.get_usernames([u["user_id"] for u in users])
        top: list[dict[str, Any]] = []
        rest_n, rest_s = 0, 0.0
        for i, u in enumerate(users):
            if i < 8:
                top.append({**u, "username": names.get(u["user_id"])})
            else:
                rest_n += int(u["n"]); rest_s += float(u["s"])
        if rest_n:
            top.append({"user_id": None, "username": "others",
                        "n": rest_n, "s": rest_s})
        st["by_user"] = top
        # The fold above is presentation for the strip only. The speaker
        # picker reads this full ranking — fed from `by_user` it could never
        # offer speaker #9 and beyond.
        st["by_user_all"] = [{**u, "username": names.get(u["user_id"])}
                             for u in users]
        st["is_admin"] = bool(user.get("is_admin"))
        st["user_id"] = user.get("user_id")
        return json.dumps(st, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"))

    body = await asyncio.to_thread(_render)
    return Response(content=body, media_type="application/json")


@router.get(
    "/api/propose-merges",
    dependencies=[Depends(require_page("captures"))],
)
async def propose_merges_api(
    user_filter: str | None = Query(None, alias="user_id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Ranked auto-merge proposals for the /captures fine-tuning data UI.

    Scope-aware: `scope=own` users see proposals built from their own
    captures only (the `user_id` query is ignored — that's an admin-
    only override). `scope=all` users (incl. admins) see cross-user
    proposals; admins may further narrow via `?user_id=...`. Results
    are cached per scope with a TTL (cfg.CAPTURES_PROPOSER_CACHE_TTL_S),
    invalidated on any capture/group write."""
    perms = user["permissions"]
    caller_uid = str(user.get("user_id") or "")
    sees_all = perms.scope("captures") == "all"
    # Off the event loop AND off the default executor. On a cold
    # _TRIM_DUR_CACHE (every restart) this walks up to
    # the hardcoded 500-row proposer window
    # (captures_merge_proposer._propose_merges_locked) doing a PCM read plus a
    # full VAD pass per row,
    # and the candidate walk itself is O(N*M^2) — measured 25-144 s on a
    # 500-row bucket. asyncio.to_thread would put that on the DEFAULT executor,
    # which is the pool main.transcribe runs CT2 inference in, so concurrent
    # sweeps could starve transcription outright. The proposer gets one
    # dedicated worker instead: same proposals, same caller latency, but it can
    # never hold more than one thread no matter how many callers ask.
    proposals, cached = await asyncio.get_running_loop().run_in_executor(
        _proposer_pool(),
        functools.partial(
            captures_merge_proposer.propose_merges,
            # Only scope=all callers can narrow via ?user_id=; the proposer
            # ignores user_id_filter when is_admin=False (caller scoped to
            # caller_user_id partition).
            user_id_filter=user_filter if sees_all else None,
            is_admin=sees_all,
            caller_user_id=caller_uid,
        ),
    )
    if proposals:
        all_uids: set[str] = set()
        for p in proposals:
            if p.get("user_id"):
                all_uids.add(p["user_id"])
            for m in p.get("member_previews", []):
                if m.get("user_id"):
                    all_uids.add(m["user_id"])
        usernames = api_keys_store.get_usernames(list(all_uids)) if all_uids else {}
        for p in proposals:
            # A scope=all non-admin gets other users' capture previews + the
            # owner's resolved username here, same as the read-by-id siblings
            # that audit. One line per proposal, keyed on its first member so
            # a DSAR can trace which rows left the owner's pool.
            _audit_cross_user_read(
                user, p, "merge-proposal",
                (p.get("member_ids") or [""])[0],
            )
            p["username"] = usernames.get(p.get("user_id"))
            for m in p.get("member_previews", []):
                m["username"] = usernames.get(m.get("user_id"))
    return JSONResponse({
        "proposals": proposals,
        "generated_ts": time.time(),
        "cached": cached,
    })


@router.get(
    "/api/by-request/{request_id}",
    dependencies=[Depends(require_page("captures"))],
)
async def by_request_id_api(
    request_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Cross-link from /reports → captures sharing a request_id. Non-admin
    callers (scope=own) see only rows they own; admin-equivalent (scope=
    all) sees every match. The endpoint backs the reports-page "show
    capture" jump, so the same scope rules that gate /captures itself
    apply here."""
    rows = captures_store.find_by_request_id(request_id)
    perms = user["permissions"]
    if perms.scope("captures") == "own":
        caller_uid = user.get("user_id")
        rows = [r for r in rows if r.get("user_id") == caller_uid]
    usernames = api_keys_store.get_usernames([r.get("user_id") for r in rows])
    for r in rows:
        # This route hands a scope=all non-admin the FULL capture row —
        # raw/final text and the owner's resolved username — for someone
        # else's capture, exactly the case the eleven read-by-id siblings
        # audit. It was the only cross-user read-by-key path with no log line.
        _audit_cross_user_read(user, r, "capture-by-request", r.get("id") or "")
        _apply_trim_to_capture_row(r)
        r["username"] = usernames.get(r.get("user_id"))
    return JSONResponse({"captures": rows})


# Literal-path GET routes (export, groups) MUST be declared BEFORE the
# parameterized /captures/api/{cid} route — FastAPI/Starlette match in
# declaration order, and the `{cid}` placeholder would otherwise swallow
# any literal-named GET like /captures/api/export with cid="export" or
# /captures/api/samples with cid="samples" (which silently 404s the
# group-list fetch and hides newly created groups from the UI).
@router.get(
    "/api/export",
    dependencies=[
        Depends(require_page("captures")),
        Depends(require_admin),
    ],
)
async def export_captures_api(
    only_status: str = Query("ready"),
    include_audio: int = Query(1, ge=0, le=1),
) -> Response:
    """Streaming tar.gz of (manifest.jsonl, audio/<id>.<ext>...). The
    `only_status` filter defaults to 'ready' — admins should mark their
    triaged training samples ready before exporting. Pass 'all' to dump
    everything (typically only useful for one-off backup)."""
    status_filter: str | None = None if only_status == "all" else only_status
    fname = f"whisper-captures-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
    return StreamingResponse(
        _build_export_stream(status_filter, bool(include_audio)),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get(
    "/api/samples",
    dependencies=[Depends(require_page("captures"))],
)
async def list_samples_api(
    user_filter: str | None = Query(None, alias="user_id"),
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(200, ge=1, le=1000),
    before_ts: float | None = Query(None),
    before_id: str | None = Query(None, max_length=64),
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """List packed training samples. `scope=own` users see only
    their own samples; `scope=all` users (incl. admins) see every sample
    and may narrow via the admin-only `?user_id=...` query. Optional
    `?status=` filter accepts the same enum as PatchSampleIn
    (new/reviewed/ready/dismissed); unknown values fall through to no
    filter, matching list_captures_api's tolerance.

    Paged newest-first: pass the `next` object from the previous response
    back as `?before_ts=&before_id=` to fetch the next page. `next` is null on
    the last page. Each group costs a member query, so returning the whole
    table on first paint was both slow and unbounded.

    Declared above `/captures/api/{cid}` because GET with cid="samples"
    would otherwise resolve to the single-capture handler and 404 — the
    UI's `load()` then silently swallows the failure and renders no
    groups, making merged groups invisible after creation."""
    # Same owner scope as the list: `?user_id=a,b` narrows to several
    # speakers (the raw string bound as `user_id = ?` matched nothing).
    scope = _effective_owner_filter(user, user_filter)

    def _gather() -> tuple[list[dict[str, Any]], bool]:
        # limit + 1: one row past the page tells us whether another page
        # exists without a second COUNT query. Read `has_more` BEFORE
        # truncating — a page that happens to be exactly `limit` long with
        # nothing after it must not advertise a next cursor.
        groups = capture_samples_store.list_samples(
            user_id=scope, status=status_filter,
            limit=limit + 1, before_ts=before_ts, before_id=before_id,
        )
        has_more = len(groups) > limit
        del groups[limit:]
        usernames = api_keys_store.get_usernames(
            [g.get("user_id") for g in groups])
        for g in groups:
            # Re-derive transcript + corrections per group so the collapsed
            # card preview reflects chip-applied final text (matches the
            # expanded card + export). Members fetched once per group via the
            # light get_members projection (word_count instead of the words
            # blob) — no per-member get_capture and no merged_words on the
            # list path; both are expand-only.
            members = capture_samples_store.get_members(g["id"])
            g["transcript"] = capture_samples._build_default_transcript(
                members, g.get("transcript_join_strategy") or "space",
            )
            g["corrections"] = _project_member_corrections(members)
            g["username"] = usernames.get(g.get("user_id"))
            # What the page's model filter and search box match a group by:
            # a group has no model / request of its own, its members do.
            g["models"] = sorted({m["model"] for m in members if m.get("model")})
            g["member_ids"] = [m["id"] for m in members]
            g["member_request_ids"] = [m["request_id"] for m in members
                                       if m.get("request_id")]
        return groups, has_more

    # OFF the loop, like propose_merges_api / get_sample_audio_api /
    # regenerate_sample_api above. Was one full get_capture per member
    # (measured 3.0 s of frozen event loop for 300 groups x 4 members at
    # 234 KB of words each); now one light query per group, still offloaded
    # because a 200-group page is hundreds of queries.
    groups, has_more = await asyncio.to_thread(_gather)
    # Cursor for the next page, or null when this was the last one. The page
    # loads more on demand rather than rendering the whole table at once —
    # each group costs a member query, so an
    # unbounded first paint was both slow and unbounded in memory.
    nxt = None
    if has_more and groups:
        last = groups[-1]
        nxt = {"before_ts": last.get("created_ts"), "before_id": last.get("id")}
    return JSONResponse({"samples": groups, "next": nxt})


@router.get(
    "/api/{cid}",
    dependencies=[Depends(require_page("captures"))],
)
async def get_capture_api(
    cid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    # Scope guard. 404 (not 403) on cross-user access — a 403 would
    # confirm the row exists (OWASP IDOR cheatsheet).
    user["permissions"].assert_can_read_row(
        row, "captures", user.get("user_id") or "",
        detail="capture not found",
    )
    _audit_cross_user_read(user, row, "capture", cid)

    # OFF the loop, like list_samples_api. `_align_member_words` runs
    # `_align_words_to_final` twice, and that is a real O(n*m) DP — at the
    # column caps (_CAP_WORDS_JSON ~10k words, _CAP_FINAL 50k chars) it
    # measures 129 ms at 1000x1000 and 1.87 s at 4000x4000. Plus
    # `_refresh_final_if_stale`, which re-runs the text pipeline.
    def _detail() -> dict[str, Any]:
        _refresh_final_if_stale(row)
        # Attach BOTH the runtime-`final` token (`word`) and the EXCLUDE-aware
        # training token (`train_word`/`train_removed`) per raw word — the same
        # shape the merge/proposal path produces. The Corrections strip + chips
        # display the training token so they match what the Final result and the
        # export actually emit (CAPTURES_PIPELINE_RULES_EXCLUDE respected); the
        # runtime form stays visible on the "runtime (dictation-map applied)"
        # line.
        row["words"] = _align_member_words(row)
        # Shift word/segment timestamps onto the trimmed-audio timeline
        # when the capture has been VAD-trimmed; the karaoke band plays the
        # trimmed WAV so time math has to match. Stored words stay in
        # original-audio time in the DB — this is read-time projection
        # only.
        _apply_trim_to_capture_row(row)
        row["username"] = api_keys_store.get_username(row.get("user_id"))
        return row

    return JSONResponse({"capture": await asyncio.to_thread(_detail)})


# Audio sniff signatures — first few bytes -> MIME. Used by the audio
# endpoint to set Content-Type without trusting the on-row audio_format
# (filename-derived extensions lie). Browsers are picky: Safari refuses
# m4a/mp4 unless Content-Type is exactly `audio/mp4`.
_AUDIO_SNIFFS: tuple[tuple[bytes, int, str], ...] = (
    # offset 0
    (b"RIFF", 0, "audio/wav"),
    (b"OggS", 0, "audio/ogg"),
    (b"ID3",  0, "audio/mpeg"),
    (b"\xFF\xFB", 0, "audio/mpeg"),   # MP3 frame
    (b"\xFF\xF3", 0, "audio/mpeg"),
    (b"\xFF\xF2", 0, "audio/mpeg"),
    (b"fLaC", 0, "audio/flac"),
    (b"\x1A\x45\xDF\xA3", 0, "audio/webm"),  # EBML — webm/matroska
    # offset 4
    (b"ftyp", 4, "audio/mp4"),  # m4a / mp4 audio
)


def _sniff_audio_mime(abs_path: str, fallback_ext: str) -> str:
    try:
        with open(abs_path, "rb") as f:
            head = f.read(16)
    except OSError:
        head = b""
    for sig, off, mime in _AUDIO_SNIFFS:
        if head[off:off + len(sig)] == sig:
            return mime
    # Fallback to extension-based guess
    ext = (fallback_ext or "").lower().lstrip(".")
    return {
        "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
        "oga": "audio/ogg", "opus": "audio/ogg", "flac": "audio/flac",
        "m4a": "audio/mp4", "mp4": "audio/mp4", "aac": "audio/aac",
        "webm": "audio/webm",
    }.get(ext, "application/octet-stream")


def _capture_audio_file(row: dict[str, Any], original: bool) -> tuple[str, str]:
    """(abs path, mime) of the file get_audio_api serves for `row`.

    Prefer the trimmed WAV when one exists — that's what the export
    uses, so reviewers should hear the same thing. Falls back to the
    original if the trimmed file is missing on disk for any reason.
    `original=True` serves the untrimmed utterance instead: that is the audio
    the decode actually received, which a latency / hallucination replay
    needs (the trim removes exactly the silence such bugs live in)."""
    trimmed_rel = None if original else row.get("audio_trimmed_relpath")
    abs_path: str | None = None
    if trimmed_rel:
        try:
            cand = captures_store.abs_audio_path(trimmed_rel)
            if os.path.isfile(cand):
                abs_path = cand
        except ValueError:
            abs_path = None
    if abs_path is None:
        try:
            abs_path = captures_store.abs_audio_path(row["audio_relpath"])
        except ValueError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "audio path invalid")
        if not os.path.isfile(abs_path):
            raise HTTPException(status.HTTP_410_GONE, "audio file is gone")
    return abs_path, _sniff_audio_mime(abs_path, row.get("audio_format", ""))


@router.get(
    "/api/{cid}/audio",
    dependencies=[Depends(require_page("captures"))],
)
async def get_audio_api(
    cid: str,
    request: Request,
    original: bool = Query(False),
    user: dict[str, Any] = Depends(get_current_user),
) -> FileResponse:
    _audio_rate.hit(rate_limit.identity_key(user, request))
    # Off the loop: this is the busiest captures endpoint (reviewers audition
    # card after card), and get_capture json.loads the words + segments blobs
    # on the shared connection that clear_all VACUUMs.
    row = await asyncio.to_thread(captures_store.get_capture, cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    user["permissions"].assert_can_read_row(
        row, "captures", user.get("user_id") or "",
        detail="capture not found",
    )
    _audit_cross_user_read(user, row, "audio", cid)
    abs_path, mime = await asyncio.to_thread(_capture_audio_file, row, original)
    # FileResponse handles Range automatically — seeking in the karaoke
    # player won't re-download the whole file.
    return FileResponse(
        path=abs_path,
        media_type=mime,
        filename=(f"{cid}{'.original' if original else ''}"
                  f".{row.get('audio_format','bin')}"),
        # Raw dictation audio behind a per-row owner check. FileResponse sends
        # ETag/Last-Modified and no Cache-Control, which makes it heuristically
        # cacheable — a shared cache in front of the app would answer the next,
        # differently-authenticated caller before the app's check ever runs.
        # The grouped-sample audio route already sends this.
        headers={"Cache-Control": "no-store"},
    )


def _audio_still_missing(row: dict[str, Any]) -> bool:
    """True while an audio_missing row's original WAV is really absent. The
    status is only set at boot (reconcile_on_startup) and never cleared, so
    a file restored after a late mount must not keep the row un-triageable
    forever. A relpath that escapes the audio root counts as missing."""
    try:
        return not os.path.isfile(
            captures_store.abs_audio_path(row.get("audio_relpath") or ""))
    except ValueError:
        return True


def _bulk_guard(
    ids: list[str], user: dict[str, Any], kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Per-row admission for the bulk endpoints — the same three checks the
    single-id PATCH / DELETE apply, but collected instead of raised:
    unknown OR cross-user → `not_found` (uniform, no existence oracle),
    member of a locked sample (non-admin) → `locked`. Accepted cross-user
    rows are audit-logged like their single-id counterparts. Ids are
    de-duplicated preserving order."""
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    wanted = list(dict.fromkeys(i for i in ids if isinstance(i, str) and i))
    # One light lookup (id / user_id / status / sample_id — all the checks
    # below read) instead of a get_capture per id, which json.loads the
    # words + segments blobs of up to 1000 rows just to discard them.
    light = captures_store.get_captures_light(wanted)
    # Sample-lock verdict per sample_id: members of one group share it.
    lock_memo: dict[str, bool] = {}
    for cid in wanted:
        row = light.get(cid)
        if row is None:
            skipped.append({"id": cid, "reason": "not_found"})
            continue
        try:
            user["permissions"].assert_can_read_row(
                row, "captures", user.get("user_id") or "",
                detail="capture not found",
            )
        except HTTPException:
            skipped.append({"id": cid, "reason": "not_found"})
            continue
        sid = row.get("sample_id") or ""
        if sid not in lock_memo:
            try:
                _assert_member_sample_not_locked(row, user)
                lock_memo[sid] = False
            except HTTPException:
                lock_memo[sid] = True
        if lock_memo[sid]:
            skipped.append({"id": cid, "reason": "locked"})
            continue
        _audit_cross_user_read(user, row, kind, cid)
        rows.append(row)
    return rows, skipped


@router.patch(
    "/api/bulk",
    dependencies=[Depends(require_page("captures"))],
)
async def bulk_status_api(
    payload: BulkStatusIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Set one status on many captures. Partial success by design: rows
    the caller may not touch are reported under `skipped` with a reason
    (`not_found` / `locked` / `audio_missing`) and the rest are updated.
    `updated[].prev_status` lets the page offer an 8 s undo."""
    def _run() -> dict[str, Any]:
        rows, skipped = _bulk_guard(payload.ids, user, "capture-bulk-status")
        ok_ids: list[str] = []
        for r in rows:
            if r.get("status") == "audio_missing" and _audio_still_missing(r):
                # system status — the file is gone; nothing to review
                skipped.append({"id": r["id"], "reason": "audio_missing"})
            else:
                ok_ids.append(r["id"])
        updated = captures_store.bulk_update_status(ok_ids, payload.status) \
            if ok_ids else []
        # The store returns rows in index order; report them in request order.
        pos = {cid: i for i, cid in enumerate(ok_ids)}
        updated.sort(key=lambda u: pos.get(u["id"], len(pos)))
        # A row deleted between the guard and the write (another tab, the
        # retention evictor) is absent from `updated`; report it, as
        # bulk-delete does, so updated + skipped covers every submitted id.
        done = {u["id"] for u in updated}
        skipped.extend({"id": i, "reason": "not_found"}
                       for i in ok_ids if i not in done)
        return {"ok": True, "status": payload.status,
                "updated": [{"id": u["id"], "prev_status": u["prev_status"]}
                            for u in updated],
                "skipped": skipped}
    return JSONResponse(await asyncio.to_thread(_run))


@router.post(
    "/api/bulk-delete",
    dependencies=[Depends(require_page("captures"))],
)
async def bulk_delete_api(
    payload: BulkIdsIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Delete many captures (rows + audio; a member's parent sample is
    auto-dissolved, as with the single-id DELETE). Same admission rules
    and partial-success shape as the bulk status endpoint. No undo."""
    def _run() -> dict[str, Any]:
        rows, skipped = _bulk_guard(payload.ids, user, "capture-bulk-delete")
        deleted: list[str] = []
        for r in rows:
            if captures_store.delete_capture(r["id"]):
                deleted.append(r["id"])
            else:
                skipped.append({"id": r["id"], "reason": "not_found"})
        return {"ok": True, "deleted": deleted, "skipped": skipped}
    return JSONResponse(await asyncio.to_thread(_run))


@router.patch(
    "/api/{cid}",
    dependencies=[Depends(require_page("captures"))],
)
async def patch_capture_api(
    cid: str,
    payload: PatchCaptureIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Edit a single capture (corrections, status, notes). `scope=own`
    users can edit only their own; `scope=all` users (incl. admins)
    can edit any capture. 404 (not 403) on cross-user access."""
    # Off the loop, like reprocess_capture_api: update_capture takes
    # captures_store._lock, which clear_all holds across DELETE + VACUUM and
    # other workers hold too — acquiring it inline would park the event loop
    # (every request and WebSocket) for that span. The HTTPExceptions raised
    # inside propagate out of to_thread unchanged.
    def _run() -> dict[str, Any] | None:
        if payload.corrections is None:
            return _patch()
        # Read-merge-write of the chips, serialised against the sample-level
        # fan-out (see _corrections_write_lock); the row is read under it.
        with _corrections_write_lock:
            return _patch()

    def _patch() -> dict[str, Any] | None:
        row = captures_store.get_capture(cid)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
        user["permissions"].assert_can_read_row(
            row, "captures", user.get("user_id") or "",
            detail="capture not found",
        )
        _audit_cross_user_read(user, row, "capture-patch", cid)
        _assert_member_sample_not_locked(row, user)
        if (payload.status is not None
                and row.get("status") == "audio_missing"
                and _audio_still_missing(row)):
            # Same rule as bulk_status_api: a system status — the file is
            # gone; nothing to review, and a ready row would leave eviction
            # tier 2 for the protected last tier. A restored file lifts it.
            raise HTTPException(status.HTTP_409_CONFLICT, "audio missing")
        patch: dict[str, Any] = {}
        if payload.status is not None:
            patch["status"] = payload.status
        if payload.corrected_text is not None:
            patch["corrected_text"] = payload.corrected_text
        if payload.corrections is not None:
            edited = [c.model_dump() for c in payload.corrections]
            if payload.baseline_corrections is not None:
                # Three-way merge: apply the user's deltas to the current
                # DB state, not just replace. Protects against concurrent
                # cross-tab admin saves (and the same member edited from its
                # group view).
                current = row.get("corrections") or []
                baseline = [c.model_dump() for c in payload.baseline_corrections]
                edited = text_corrections.three_way_merge_corrections(
                    baseline, edited, current,
                )
            if text_corrections.over_cap(edited):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"a capture holds at most"
                    f" {text_corrections.CAP_CORRECTIONS} corrections",
                )
            patch["corrections"] = edited
        if payload.admin_notes is not None:
            patch["admin_notes"] = payload.admin_notes
        try:
            return captures_store.update_capture(cid, patch)
        except ValueError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    updated = await asyncio.to_thread(_run)
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    return JSONResponse({"ok": True, "capture": updated})


@router.delete(
    "/api/{cid}",
    dependencies=[Depends(require_page("captures"))],
)
async def delete_capture_api(
    cid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Delete a single capture. `scope=own` users can delete only their
    own; `scope=all` users (incl. admins) can delete any. Bulk wipe is
    via /clear which stays admin-only."""
    row = await asyncio.to_thread(captures_store.get_capture, cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    user["permissions"].assert_can_read_row(
        row, "captures", user.get("user_id") or "",
        detail="capture not found",
    )
    _audit_cross_user_read(user, row, "capture-delete", cid)
    _assert_member_sample_not_locked(row, user)
    # Off the loop: deleting a member auto-dissolves its sample, which waits
    # on that sample's rebuild lock (see samples_store.dissolve_sample).
    if not await asyncio.to_thread(captures_store.delete_capture, cid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    return JSONResponse({"ok": True})


@router.post(
    "/api/clear",
    dependencies=[
        Depends(require_page("captures")),
        Depends(require_admin),
    ],
)
async def clear_captures_api(payload: ClearIn, request: Request) -> JSONResponse:
    if payload.confirm != "CAPTURES":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Confirm by sending {\"confirm\": \"CAPTURES\"}.",
        )
    host = request.client.host if request.client else ""
    # Off the loop: clear_all runs a DELETE plus a full VACUUM on a database
    # that can reach ~1 GB, then rmtree's up to CAPTURES_MAX_MB of audio. The
    # store takes its own lock and the route holds none.
    n = await asyncio.to_thread(captures_store.clear_all, host)
    return JSONResponse({"ok": True, "deleted": n})


# ---------------------------------------------------------------------
# Per-capture pipeline reprocess (re-run rules on `raw`)
# ---------------------------------------------------------------------

@router.post(
    "/api/{cid}/reprocess",
    dependencies=[Depends(require_page("captures"))],
)
async def reprocess_capture_api(
    cid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Re-run the post-processing pipeline on the stored `raw` text and
    update both `final` and `text_for_training` to reflect the current
    PIPELINE_RULES (and the captures-specific exclude set).

    Use case: after editing PIPELINE_RULES (e.g. adding a typo-fix or
    a new de-dictation-map entry), a reviewer wants this specific capture
    re-derived without waiting for the bulk reapply job. The bulk job
    /quick-config/reapply-rules also handles this row eventually, but
    the per-row trigger gives immediate feedback in the UI. `scope=own`
    users can reprocess only their own captures.
    """
    # Off the loop, like get_capture_api's _detail(): one or two full
    # pipeline passes over raw text of up to 50k chars (owner regex rules
    # included) plus the SQLite read and write would otherwise stall every
    # other request and WebSocket for the length of the run. The HTTPExceptions
    # raised inside propagate out of to_thread unchanged.
    def _run() -> tuple[dict[str, Any], list[str]]:
        row = captures_store.get_capture(cid)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
        user["permissions"].assert_can_read_row(
            row, "captures", user.get("user_id") or "",
            detail="capture not found",
        )
        _audit_cross_user_read(user, row, "capture-reprocess", cid)
        _assert_member_sample_not_locked(row, user)
        raw = row.get("raw") or ""
        captures_excludes = getattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None)
        # Resolve the CAPTURE OWNER's effective pipeline (not the caller's — an admin
        # may reprocess another user's row; the result must reflect that user's
        # rules). Pipeline-only: no key / no per-request layer on reprocess.
        ident = effective_config.build_ident({"user_id": row.get("user_id")}, row.get("model"))
        try:
            new_final = pl_engine._postprocess_text(raw, model_name=row.get("model"), ident=ident, language=captures_store.text_language(row))
        except Exception as e:
            logger.error("[captures] reprocess pipeline failed on `final`: %s", e)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "pipeline reprocessing failed",
            )
        # When no captures-specific excludes are configured, the training-text
        # pass would produce byte-identical output to `final` — skip the
        # second full pipeline pass and reuse.
        if captures_excludes:
            try:
                new_training = pl_engine._postprocess_text(
                    raw,
                    model_name=row.get("model"),
                    extra_excludes=captures_excludes,
                    ident=ident,
                    language=captures_store.text_language(row),
                )
            except Exception as e:
                logger.error(
                    "[captures] reprocess pipeline failed on `text_for_training`: %s", e)
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "pipeline reprocessing failed",
                )
        else:
            new_training = new_final
        patch: dict[str, Any] = {}
        if new_final != (row.get("final") or ""):
            patch["final"] = new_final
        if new_training != (row.get("text_for_training") or ""):
            patch["text_for_training"] = new_training
        updated = captures_store.update_capture(cid, patch) if patch else row
        return updated or row, list(patch.keys())

    capture, changed = await asyncio.to_thread(_run)
    return JSONResponse({"capture": capture, "changed": changed})


@router.post(
    "/api/reprocess-all",
    dependencies=[
        Depends(require_page("captures")),
        Depends(require_admin),
    ],
)
async def reprocess_all_captures_api() -> JSONResponse:
    """Trigger the bulk pipeline-reapply job — same worker used by
    /quick-config/reapply-rules. Idempotent: a second call while the
    job is running returns the current state instead of spawning a
    duplicate worker. Use after PIPELINE_RULES edits to bring every
    capture's `final` + `text_for_training` (and downstream group
    `transcript`) in line with the current rules."""
    return JSONResponse(captures_reapply.start())


@router.get(
    "/api/reprocess-all/status",
    dependencies=[Depends(require_page("captures")), Depends(require_admin)],
)
async def reprocess_all_status_api() -> JSONResponse:
    """Live progress of the pipeline-reapply job (for the Advanced menu's
    progress line)."""
    return JSONResponse(captures_reapply.status())


@router.post(
    "/api/reprocess-vad",
    dependencies=[Depends(require_page("captures")), Depends(require_admin)],
)
async def reprocess_vad_api() -> JSONResponse:
    """Trigger the bulk VAD/silence re-merge job: rebuild every sample's
    merged WAV with the CURRENT global silence settings (skips locked
    samples; over-cap samples are flagged stale, never truncated). Idempotent:
    a second call while running returns the current state. Use after editing
    the global Sample-sizing / Silence-trim settings."""
    return JSONResponse(captures_vad_reprocess.start())


@router.get(
    "/api/reprocess-vad/status",
    dependencies=[Depends(require_page("captures")), Depends(require_admin)],
)
async def reprocess_vad_status_api() -> JSONResponse:
    """Live progress of the VAD/silence re-merge job."""
    return JSONResponse(captures_vad_reprocess.status())


# ---------------------------------------------------------------------
# Capture samples (packed training clips within the configured duration cap)
# ---------------------------------------------------------------------

# Join strategy + inter-member silence are GLOBAL admin settings now
# (cfg.CAPTURES_SAMPLE_JOIN_STRATEGY / cfg.CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS),
# not per-request — so the merge/preview/patch payloads no longer carry them.
_MAX_GROUP_MEMBERS = 30
# A group's chips are the projection of ALL its members' chips, so the
# group-level schemas bound the whole projection; the binding limit is per
# member (CAP_CORRECTIONS), checked with over_cap after the split. Bounding
# the group list by the per-member cap 422'd every save of a group whose
# members together held more than 200 chips.
_CAP_GROUP_CORRECTIONS = text_corrections.CAP_CORRECTIONS * _MAX_GROUP_MEMBERS


class CreateSampleIn(BaseModel):
    model_config = {"extra": "forbid"}
    member_ids: list[str] = Field(min_length=1, max_length=_MAX_GROUP_MEMBERS)


class PreviewMergeIn(BaseModel):
    """Preview the merged audio without creating a sample."""
    model_config = {"extra": "forbid"}
    member_ids: list[str] = Field(min_length=1, max_length=_MAX_GROUP_MEMBERS)


class PreviewSaveChipsIn(BaseModel):
    """Save chip corrections from a not-yet-merged proposal. Chips carry
    GLOBAL word indices into the merged karaoke strip; server fans them
    out to per-member captures via _split_corrections_to_members."""
    model_config = {"extra": "forbid"}
    member_ids: list[str] = Field(min_length=1, max_length=_MAX_GROUP_MEMBERS)
    corrections: list[CorrectionIn] = Field(
        default_factory=list, max_length=_CAP_GROUP_CORRECTIONS)


class PatchSampleIn(BaseModel):
    model_config = {"extra": "forbid"}
    is_locked: bool | None = None
    corrections: list[CorrectionIn] | None = Field(
        default=None, max_length=_CAP_GROUP_CORRECTIONS)
    # Snapshot of the group-derived chips at GET time. When provided
    # alongside `corrections`, the server applies a three-way merge
    # against the current member-projected chips so concurrent reports
    # / cross-tab admin saves survive. Omitted → legacy replace.
    baseline_corrections: list[CorrectionIn] | None = Field(
        default=None, max_length=_CAP_GROUP_CORRECTIONS)
    status: Literal["new", "reviewed", "ready", "dismissed"] | None = None
    admin_notes: str | None = Field(default=None, max_length=8000)


def _global_join_strategy() -> str:
    """Transcript join strategy, sourced from the global setting."""
    j = getattr(cfg, "CAPTURES_SAMPLE_JOIN_STRATEGY", "space")
    return j if j in ("space", "period_space") else "space"


def _shift_word_times(
    items: list[dict[str, Any]] | None,
    lead_ms: int,
    eff_duration_s: float | None,
) -> list[dict[str, Any]]:
    """Return a NEW list with each item's start/end shifted by -lead_ms/1000
    and clamped to [0, eff_duration_s] (when given).

    Used after a VAD trim where the served audio is shorter than the
    original: stored words/segments live in original-audio time so the
    DB stays canonical; this helper rebases them onto the trimmed
    audio's timeline so audio.currentTime alignment is correct.

    Items whose interval lies entirely outside [0, eff_duration_s] are
    dropped — they map to audio that was cut away. All other fields
    (word, raw_word, removed, member_idx, …) are preserved verbatim.
    Returns a deep-enough copy (dicts re-built so the originals are
    untouched).

    Returns an empty list when items is empty / None. Returns items
    unchanged (as a fresh list) when both lead_ms and eff_duration_s
    indicate no work (no shift, no clamp)."""
    if not items:
        return []
    shift_s = float(lead_ms or 0) / 1000.0
    if shift_s <= 0 and eff_duration_s is None:
        return list(items)
    out: list[dict[str, Any]] = []
    for it in items:
        try:
            s_old = float(it.get("start") or 0.0)
            e_old = float(it.get("end", s_old) or 0.0)
        except (TypeError, ValueError):
            continue
        s_new = s_old - shift_s
        e_new = e_old - shift_s
        if e_new <= 0:
            continue
        if eff_duration_s is not None and s_new >= eff_duration_s:
            continue
        s_clamped = max(0.0, s_new)
        e_clamped = e_new
        if eff_duration_s is not None:
            e_clamped = min(eff_duration_s, e_clamped)
        new_it = dict(it)
        new_it["start"] = s_clamped
        new_it["end"] = max(s_clamped, e_clamped)
        out.append(new_it)
    return out


def _apply_trim_to_capture_row(row: dict[str, Any]) -> None:
    """In-place: if `row` carries trim offsets, shift its `words` and
    `segments` onto the trimmed-audio timeline. Always sets
    `effective_audio_s` (= original `audio_s` when
    lead/trail are None or 0) so consumers can read one field
    uniformly without branching on trim presence."""
    if not row:
        return
    lead = row.get("audio_trim_lead_ms")
    trail = row.get("audio_trim_trail_ms")
    if not lead and not trail:
        # Still expose effective_audio_s equal to audio_s
        # so consumers can use a single field uniformly.
        row["effective_audio_s"] = float(row.get("audio_s") or 0.0)
        return
    lead_ms = int(lead or 0)
    trail_ms = int(trail or 0)
    orig_s = float(row.get("audio_s") or 0.0)
    eff = max(0.0, orig_s - (lead_ms + trail_ms) / 1000.0)
    row["effective_audio_s"] = eff
    if "words" in row:
        row["words"] = _shift_word_times(row.get("words"), lead_ms, eff)
    if "segments" in row:
        row["segments"] = _shift_word_times(row.get("segments"), lead_ms, eff)


def _validate_merge_payload(
    member_ids: list[str],
    silence_ms: int,
    user: dict[str, Any],
    *,
    enforce_cap: bool = True,
) -> tuple[list[dict[str, Any]], str, list[str], int]:
    """Shared validation for create_sample_api and the preview-audio endpoint.

    Validates: deduped member_ids, every capture exists, none is already in
    a group, all members belong to the same user (and the caller is either
    that user or admin), audio files are present on disk, and — when
    `enforce_cap` — the TRIMMED merged duration (per-member VAD trim + inter-
    segment silence) is within the configured cap. The cap is measured on trimmed audio because
    that's what the merged WAV actually is (and what the proposer packs to);
    measuring raw would reject groups that comfortably fit after trimming.

    `enforce_cap=False` skips only the cap (used by the merge-estimate
    endpoint, which needs the totals even when they exceed the cap).

    Returns (captures, owner_user_id, member_paths, total_trimmed_ms) so
    downstream callers don't re-fetch the same rows."""
    if len(member_ids) != len(set(member_ids)):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "duplicate capture in member_ids",
        )

    captures: list[dict[str, Any]] = []
    member_paths: list[str] = []
    user_ids: set[str] = set()
    tasks: set[str] = set()
    languages: set[str] = set()
    for mid in member_ids:
        cap = captures_store.get_capture(mid)
        if cap is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"capture {mid} not found",
            )
        # Enforce the scope guard per-member BEFORE revealing any state
        # (already-in-sample / audio-missing) — otherwise a scope=own caller
        # could probe another user's capture id for existence + state. scope=all
        # (incl. admin) bypasses; scope=own requires the caller to BE the owner.
        # 404 (not 403) matches the captures detail endpoints — don't leak
        # existence. detail matches the missing branch above byte-for-byte so
        # the body doesn't become the existence oracle the status closes.
        user["permissions"].assert_can_read_row(
            cap, "captures", user.get("user_id") or "",
            detail=f"capture {mid} not found",
        )
        if cap.get("sample_id"):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"capture {mid} is already in a sample",
            )
        abs_p = captures_store.abs_audio_path(cap["audio_relpath"])
        if not os.path.exists(abs_p):
            raise HTTPException(
                status.HTTP_410_GONE, f"capture {mid} audio is missing",
            )
        member_paths.append(abs_p)
        user_ids.add(cap.get("user_id") or "")
        tasks.add(cap.get("task") or "transcribe")
        _lang = captures_merge_proposer._bcp47_primary(cap.get("language"))
        if _lang:
            languages.add(_lang)
        captures.append(cap)
    if len(user_ids) != 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "members must all belong to the same user",
        )
    # Same partition the proposer buckets by. A translate capture carries the
    # SOURCE language but English text, and the sample takes one language and
    # one task label for the whole WAV, so a mixed group would export
    # wrong-language training data (see _group_task).
    if len(tasks) != 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "members must all share one task",
        )
    if len(languages) > 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "members must all share one language",
        )
    owner_user_id = next(iter(user_ids))
    # Re-assert at the resolved owner (defense-in-depth; the per-member guard in
    # the loop above already enforced this for every member). scope=all (incl.
    # admin) bypasses; scope=own requires the caller to BE the owner.
    user["permissions"].assert_can_read_row(
        {"user_id": owner_user_id}, "captures", user.get("user_id") or "",
    )
    # Audit the cross-user merge surface so a non-admin scope=all caller
    # reading another user's captures via /groups, /preview-audio,
    # /preview-words or /preview-save-chips is recorded — matches the
    # per-row audit calls at every other capture/group endpoint.
    _audit_cross_user_read(
        user, {"user_id": owner_user_id}, "merge",
        ",".join(member_ids[:3]) + ("+" if len(member_ids) > 3 else ""),
    )

    # Cap on TRIMMED audio — what the merged WAV actually is. Reuses the
    # proposer's cached per-capture trim so the batch flow (already warm) pays
    # nothing here; a cold manual merge trims each member once (then cached).
    total_trimmed_ms = sum(
        int(round(captures_merge_proposer.trimmed_duration_s(c) * 1000))
        for c in captures
    )
    # Real merged length under the uniform layout: 2×outer-edge +
    # Σ trimmed bodies + (N-1)×join silence.
    n_members = len(member_ids)
    total_gap_ms = int(silence_ms) * max(0, n_members - 1)
    edge_ms = capture_samples._global_edge_ms()
    # The outer edge margin exists only on merge_wavs' trim path; its legacy
    # (trim-disabled) layout adds no outer margin, so counting 2×edge here when
    # trimming is off over-rejects in-cap merges by ~2×edge_ms.
    trim_samples = bool(getattr(cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES", False))
    total_edge_ms = 2 * edge_ms if (trim_samples and n_members >= 1) else 0
    cap_ms = int(float(getattr(cfg, "CAPTURES_SAMPLE_MAX_DURATION_S", 29.9)) * 1000)
    total_ms = total_trimmed_ms + total_gap_ms + total_edge_ms
    if enforce_cap and total_ms > cap_ms:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"merged duration would exceed {cap_ms / 1000:.1f} s "
            f"({total_ms / 1000:.2f}s)",
        )
    return captures, owner_user_id, member_paths, total_trimmed_ms


def _preview_member_trims(
    member_ids: list[str], member_paths: list[str],
) -> dict[str, Any]:
    """Compute the same per-member trim map merge_wavs would produce, without
    writing a merged WAV. Used by /preview-words so the karaoke overlay lines
    up with the audio /preview-audio streams for the same payload. Returns {}
    when group trimming is disabled, or when any member cannot be read/trimmed
    (then _build_merged_words uses the legacy full-duration timeline for every
    member — a partial map would mix absolute and legacy offsets per member)."""
    if not getattr(cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES", False):
        return {}
    edge = capture_samples._global_edge_ms()
    max_gap = int(getattr(cfg, "CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS", 300))
    join_ms = int(capture_samples._global_silence_ms())
    trims: dict[str, Any] = {}
    # Mirror merge_wavs' uniform layout: leading edge, then bodies joined by
    # `join_ms`, stamping each member's absolute offset so the preview karaoke
    # lines up with the audio /preview-audio streams.
    cursor_ms = edge
    first = True
    for mid, p in zip(member_ids, member_paths):
        try:
            pcm, n = audio_merge.read_pcm(p)
            res = audio_vad_trim.trim_pcm_for_merge(
                pcm, n, edge_pad_ms=edge, max_internal_gap_ms=max_gap,
            )
        except Exception:
            # merge_wavs raises on the same failure, so /preview-audio fails too;
            # abandon the trim map entirely rather than emit a partial one that
            # would desync this member's karaoke from the rest.
            return {}
        if not first:
            cursor_ms += join_ms
        first = False
        trims[mid] = {
            "lead_ms": int(res["lead_ms"]),
            "new_duration_ms": int(res["new_duration_ms"]),
            "segments": res["segments"],
            "offset_ms": int(cursor_ms),
        }
        cursor_ms += int(res["new_duration_ms"])
    return trims


@router.post(
    "/api/samples",
    dependencies=[Depends(require_page("captures"))],
)
async def create_sample_api(
    payload: CreateSampleIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Pack one or more same-user captures into a training sample within the configured duration cap.

    Server-enforced invariants:
      - all members exist, are not yet in a sample, are all owned by the
        same user (and either the caller is that user OR is admin)
      - total audio + gap silence within the configured duration cap
      - members' audio files match (1 ch, 16 bit, 16 kHz)
    """
    import uuid as _uuid

    member_ids = payload.member_ids
    # OFF the loop: _validate_merge_payload runs a Silero VAD pass per member
    # and _build_merged_wav runs another plus a WAV write with fsync — up to 30
    # members per call, with no rate limit. The sibling routes that touch the
    # same helpers (get_sample_audio_api, regenerate_sample_api) were already
    # offloaded for exactly this reason; this one was missed.
    captures, owner_user_id, member_paths, _total_audio_ms = (
        await asyncio.to_thread(
            _validate_merge_payload, member_ids, capture_samples._global_silence_ms(), user)
    )

    # Build merged WAV — sid generated upfront so the build path is
    # known before the DB insert (mirrors captures_store).
    sid = _uuid.uuid4().hex
    transcript = capture_samples._build_default_transcript(captures, _global_join_strategy())
    duration_ms, hashes, member_trims = await asyncio.to_thread(
        functools.partial(
            capture_samples._build_merged_wav,
            sid=sid,
            member_paths=member_paths,
            member_ids=member_ids,
            silence_ms=capture_samples._global_silence_ms(),
        )
    )
    # Derive language from the first member with a populated value —
    # Whisper detects language per-clip; members of the same group should
    # all share it, but if a member somehow has an empty language we
    # tolerate that and fall through to the next one rather than
    # emitting an empty `language` in the export manifest.
    group_language = ""
    for _c in captures:
        _lang = (_c.get("language") or "").strip()
        if _lang:
            group_language = _lang
            break
    # Off the loop too: the insert blocks on both store locks, which
    # captures_store.clear_all holds across a full VACUUM (see
    # patch_capture_api) — inline, a Merge during a clear parks every request.
    try:
        await asyncio.to_thread(functools.partial(
            _insert_sample_with_sid,
            sid=sid,
            user_id=owner_user_id,
            member_ids=member_ids,
            transcript=transcript,
            join_strategy=_global_join_strategy(),
            silence_ms=capture_samples._global_silence_ms(),
            member_hash_map=hashes,
            duration_ms=duration_ms,
            language=group_language,
            member_trims=member_trims,
        ))
    except Exception:
        # Insert failed — roll back the WAV we just wrote so the
        # next merge attempt for the same captures starts clean.
        try:
            os.unlink(capture_samples_store.abs_path_for(
                capture_samples_store._relpath_for(sid)))
        except OSError:
            pass
        raise
    return JSONResponse({"sample_id": sid})


@router.post(
    "/api/samples/preview-audio",
    dependencies=[Depends(require_page("captures"))],
)
async def preview_merge_audio_api(
    payload: PreviewMergeIn,
    request: Request,
    background: BackgroundTasks,
    user: dict[str, Any] = Depends(get_current_user),
) -> FileResponse:
    """Build the merged WAV exactly as create_sample_api would, stream it
    back to the caller as audio/wav, and delete the temp file after the
    response completes. Does NOT persist a capture_samples row.

    Used by the /captures Auto-propose merges modal + the manual merge-
    modal to let users preview the merged audio before committing."""

    _audio_rate.hit(rate_limit.identity_key(user, request))

    _captures, _owner, member_paths, _total_audio_ms = (
        await asyncio.to_thread(
            _validate_merge_payload, payload.member_ids,
            capture_samples._global_silence_ms(), user)
    )

    # tempfile.NamedTemporaryFile(delete=False) so FileResponse can stream
    # the closed file; background unlink fires after the response finishes.
    fd, tmp_path = tempfile.mkstemp(prefix="preview_merge_", suffix=".wav")
    os.close(fd)
    try:
        # Off the loop: a VAD pass per member plus the WAV write with fsync.
        await asyncio.to_thread(
            functools.partial(
                audio_merge.merge_wavs,
                member_paths, tmp_path, gap_ms=capture_samples._global_silence_ms(),
                trim=bool(getattr(
                    cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES", False)),
                edge_pad_ms=capture_samples._global_edge_ms(),
                max_internal_gap_ms=int(
                    getattr(cfg, "CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS", 300)),
            )
        )
    except audio_merge.WavFormatError as e:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except ValueError as e:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise

    def _cleanup():
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    background.add_task(_cleanup)
    return FileResponse(
        path=tmp_path,
        media_type="audio/wav",
        filename="preview.wav",
        # Merged dictation audio. FileResponse otherwise emits only
        # ETag/Last-Modified, which makes the body heuristically cacheable by
        # any shared cache in front of the app — the same reason get_audio_api
        # and get_sample_audio_api send this.
        headers={"Cache-Control": "no-store"},
    )


@router.post(
    "/api/samples/preview-words",
    dependencies=[Depends(require_page("captures"))],
)
async def preview_merge_words_api(
    payload: PreviewMergeIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Return the projected merged words + projected corrections + joined
    transcript for a hypothetical merge — timestamps aligned to the audio
    that POST /preview-audio would stream for the same payload. Lets the
    UI overlay karaoke highlighting AND seed the chip-correction box on
    the preview panel without persisting a group row.

    Pure CPU; no rate-limit (bounded by ≤30 members × few hundred words +
    memoized per-word _postprocess_text). Same validation gates as the
    audio endpoint."""
    captures, _owner, member_paths, _total_audio_ms = (
        await asyncio.to_thread(
            _validate_merge_payload, payload.member_ids,
            capture_samples._global_silence_ms(), user)
    )
    # `_preview_member_trims` must be called INSIDE the thread. As a
    # functools.partial argument it was evaluated eagerly on the event loop,
    # which is where its blocking PCM read + uncached Silero-VAD pass per
    # member (up to 30) then ran — ~2 ms of loop block per second of member
    # audio, and _validate_merge_payload's cap is on the TRIMMED total so
    # silence-heavy members are not bounded by it. The sibling
    # preview_merge_audio_api already does its VAD inside merge_wavs.
    def _build() -> list[dict[str, Any]]:
        return _build_merged_words(
            captures, capture_samples._global_silence_ms(),
            member_trims=_preview_member_trims(
                payload.member_ids, member_paths),
        )

    words = await asyncio.to_thread(_build)
    return JSONResponse({
        "words": words,
        "corrections": _project_member_corrections(captures),
        "transcript": capture_samples._build_default_transcript(captures, _global_join_strategy()),
    })


@router.post(
    "/api/samples/merge-estimate",
    dependencies=[Depends(require_page("captures"))],
)
async def merge_estimate_api(
    payload: PreviewMergeIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Return the raw + TRIMMED merged-duration totals for a hypothetical
    merge so the manual-selection meter can show (and gate on) the real
    post-trim length instead of the raw sum. Skips the cap so the UI can
    display an over-cap value and disable Merge itself. Same ownership gates
    as the other merge endpoints; reuses the proposer's cached per-capture
    trim."""
    # Off the loop with its siblings: a VAD pass per member, no rate limit.
    captures, _owner, _paths, trimmed_ms = await asyncio.to_thread(
        functools.partial(
            _validate_merge_payload,
            payload.member_ids, capture_samples._global_silence_ms(), user, enforce_cap=False,
        )
    )
    raw_ms = sum(
        int(round(float(c.get("audio_s") or 0.0) * 1000))
        for c in captures
    )
    n = len(payload.member_ids)
    gap_ms = int(capture_samples._global_silence_ms()) * max(0, n - 1)
    edge_ms = capture_samples._global_edge_ms()
    # Mirror merge_wavs: the outer edge margin only exists on the trim path.
    trim_samples = bool(getattr(cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES", False))
    total_edge_ms = 2 * edge_ms if (trim_samples and n >= 1) else 0
    cap_ms = int(float(getattr(cfg, "CAPTURES_SAMPLE_MAX_DURATION_S", 29.9)) * 1000)
    trimmed_total = trimmed_ms + gap_ms + total_edge_ms
    return JSONResponse({
        "raw_total_s": (raw_ms + gap_ms + total_edge_ms) / 1000.0,
        "trimmed_total_s": trimmed_total / 1000.0,
        "hard_cap_s": cap_ms / 1000.0,
        "fits": trimmed_total <= cap_ms,
    })


@router.post(
    "/api/samples/preview-save-chips",
    dependencies=[Depends(require_page("captures"))],
)
async def preview_save_chips_api(
    payload: PreviewSaveChipsIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Persist chip corrections from a not-yet-merged proposal. Fans the
    global-indexed chips out to each member capture's local indices via
    _split_corrections_to_members, then REPLACES each member's
    `corrections` field. Same fan-out semantics as the group-level chip
    save (patch_sample_api below).

    Re-fetches every touched member and returns the canonical chips so
    the client can reproject (via _project_member_corrections) to refresh
    its baseline without a full /preview-words round-trip."""
    captures, owner_user_id, _member_paths, _total_audio_ms = (
        await asyncio.to_thread(
            _validate_merge_payload, payload.member_ids,
            capture_samples._global_silence_ms(), user)
    )
    chips_in = [c.model_dump(exclude_none=True) for c in payload.corrections]
    per_member = _split_corrections_to_members(chips_in, captures)
    # Before any write, as in patch_sample_api: the schema bounds the whole
    # projection, the store caps each member.
    if any(text_corrections.over_cap(c) for c in per_member.values()):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"a member capture holds at most"
            f" {text_corrections.CAP_CORRECTIONS} corrections",
        )

    saved: dict[str, int] = {}
    members_corrections: dict[str, list[dict[str, Any]]] = {}

    # Off the loop: up to 30 update_capture calls, each taking
    # captures_store._lock (see patch_capture_api). Under the corrections
    # lock, and re-checked: a concurrent create_sample may have grouped a
    # member since the validation above — a sample member's chips are the
    # sample's to write (and may be locked), so it is skipped.
    def _save() -> None:
        with _corrections_write_lock:
            light = captures_store.get_captures_light([c["id"] for c in captures])
            for cap in captures:
                mid = cap["id"]
                now_row = light.get(mid)
                if now_row is None or now_row.get("sample_id"):
                    continue
                member_chips = per_member.get(mid, [])
                updated = captures_store.update_capture(mid, {"corrections": member_chips})
                canonical = (updated or {}).get("corrections") or []
                members_corrections[mid] = canonical
                saved[mid] = len(canonical)

    await asyncio.to_thread(_save)
    captures_merge_proposer.invalidate(owner_user_id)
    return JSONResponse({
        "saved": saved,
        "members_corrections": members_corrections,
    })


def _insert_sample_with_sid(
    *,
    sid: str,
    user_id: str,
    member_ids: list[str],
    transcript: str,
    join_strategy: str,
    silence_ms: int,
    member_hash_map: dict[str, str],
    duration_ms: int,
    language: str | None = None,
    member_trims: dict[str, Any] | None = None,
) -> None:
    """Direct insert that honours a pre-allocated sid (needed because the
    audio file is written at the sid path before this call).

    Group chip state lives on the member captures, not on the group
    row — every read re-projects from members — so no chip plumbing
    appears here.

    The member UPDATE's rowcount is the concurrency gate: create_sample_api
    awaits between _validate_merge_payload and this insert, so a
    double-submitted Merge can pass validation twice. The `sample_id IS
    NULL` predicate makes the loser's UPDATE match nothing; raising here
    rolls the sample row back (explicit BEGIN/ROLLBACK — the shared
    connection is autocommit, so `with conn:` alone would not) and lets
    the caller's except-branch unlink the merged WAV. Both store locks are
    held for the BEGIN..COMMIT span: the two stores share one autocommit
    connection, so a bare captures_store statement from another thread
    would otherwise join (and be rolled back with) this transaction. No
    path takes captures_store._lock before samples_store._lock (see
    store.py sweep_retention/delete_capture), so this order is acyclic."""

    relpath = capture_samples_store._relpath_for(sid)
    now = time.time()
    conn = capture_samples_store._require_conn()
    with capture_samples_store._lock, captures_store._lock:
        conn.execute("BEGIN")
        try:
            conn.execute(
                "INSERT INTO capture_samples"
                " (id, user_id, created_ts, merged_wav_relpath,"
                "  merged_duration_ms, transcript,"
                "  transcript_join_strategy, member_hashes,"
                "  inter_segment_silence_ms, is_stale, is_locked,"
                "  language, merged_lead_trim_ms, merged_trail_trim_ms,"
                "  member_trims)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,0,?,0,0,?)",
                (
                    sid, user_id, now, relpath, int(duration_ms),
                    transcript, join_strategy,
                    json.dumps(member_hash_map, sort_keys=True),
                    int(silence_ms),
                    language or None,
                    json.dumps(member_trims or {}, sort_keys=True),
                ),
            )
            for order, mid in enumerate(member_ids):
                cur = conn.execute(
                    "UPDATE captures SET sample_id = ?, sample_order = ?"
                    " WHERE id = ? AND sample_id IS NULL",
                    (sid, order, mid),
                )
                if cur.rowcount != 1:
                    raise HTTPException(
                        status.HTTP_409_CONFLICT,
                        "capture already belongs to a sample")
            # Inside the try, as in dissolve_sample: a COMMIT that fails
            # (SQLITE_FULL / IOERR / BUSY) can leave the transaction open on
            # the shared autocommit connection, and every later BEGIN then
            # fails until a restart.
            conn.execute("COMMIT")
        except BaseException:
            # Guarded like samples_store.dissolve_sample: keep the real error
            # when SQLite already rolled back by itself.
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    captures_merge_proposer.invalidate(user_id)
    logger.info(
        "[samples] created sid=%s user=%s n=%d dur=%.1fs",
        sid[:8], (user_id or "?")[:8], len(member_ids), duration_ms / 1000.0,
    )


@router.get(
    "/api/samples/{sid}",
    dependencies=[Depends(require_page("captures"))],
)
async def get_sample_api(
    sid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    # 404 (not 403) on cross-user — leaking existence violates OWASP IDOR.
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
        detail="sample not found",
    )
    _audit_cross_user_read(user, g, "sample", sid)
    # OFF the loop, like list_samples_api: `_enrich_sample` runs
    # `_align_member_words` per member, and each of those runs the O(n*m)
    # `_align_words_to_final` DP twice — measured 129 ms at 1000x1000 words
    # and 1.87 s at 4000x4000, doubled per member.
    return JSONResponse({"sample": await asyncio.to_thread(_enrich_sample, g)})


def _hydrate_members(members: list[dict[str, Any]]) -> None:
    """Populate `words` (decoded) and `model` on each member dict in
    place by fetching the full capture row once. `capture_samples_store.
    get_members` drops words/model to keep the projection light;
    the chip/karaoke helpers below need both. Idempotent — skips members
    that already carry the fields."""
    for m in members:
        if "words" in m and "model" in m:
            continue
        cap = captures_store.get_capture(m["id"]) or {}
        m["words"] = cap.get("words") or []
        m["model"] = cap.get("model")


def _project_member_corrections(
    members: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project each member's chip corrections into a group-level chip
    list with global word indices. Member chips reference indices into
    that member's words (immutable raw STT). Group chips need
    indices into the flattened merged_words array. The offset for
    member m is Σ_{j<m} len(words_j) — silence gaps contribute no
    words, so they don't shift the index.

    Members must either be hydrated (`_hydrate_members`, carrying `words`)
    or come straight from `samples_store.get_members`, which carries
    `word_count`; only the count is needed here."""
    out: list[dict[str, Any]] = []
    offset = 0
    for m in members:
        n = (len(m.get("words") or []) if "words" in m
             else int(m.get("word_count") or 0))
        # A wordless member has no global anchor to project chips onto.
        # Without this skip the clamp below would collapse to `offset`,
        # which is also the first word index of the NEXT member — the
        # round-trip through _split_corrections_to_members would silently
        # re-attribute the chip to that next member.
        if n <= 0:
            continue
        for c in (m.get("corrections") or []):
            try:
                idx = int(c["idx"]) + offset
            except (TypeError, ValueError, KeyError):
                continue
            c2 = dict(c)
            c2["idx"] = min(idx, offset + max(0, n - 1))
            if c.get("idx_end") is not None:
                try:
                    end = int(c["idx_end"]) + offset
                    c2["idx_end"] = min(end, offset + max(0, n - 1))
                except (TypeError, ValueError):
                    c2.pop("idx_end", None)
            out.append(c2)
        offset += n
    return out


def _split_corrections_to_members(
    corrections: list[dict[str, Any]],
    members: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Inverse of `_project_member_corrections`. Given a list of
    group-level chips with global word indices, return a dict mapping
    each member_id → list of chips with member-local indices.

    Every member is represented in the result (with `[]` if it owns no
    chips after the split) so the caller can fan out via
    `update_capture(member_id, {"corrections": chips})` and reliably
    REPLACE each member's chip list. Chips whose `idx` is out of range
    are silently dropped; `idx_end` is clipped to the same member's
    last word."""
    word_counts: list[int] = []
    for m in members:
        word_counts.append(len(m.get("words") or []))
    offsets = [0]
    for n in word_counts[:-1]:
        offsets.append(offsets[-1] + n)
    out: dict[str, list[dict[str, Any]]] = {m["id"]: [] for m in members}
    for c in (corrections or []):
        if not isinstance(c, dict):
            continue
        try:
            gidx = int(c["idx"])
        except (TypeError, ValueError, KeyError):
            continue
        target: int | None = None
        for i in range(len(members)):
            start = offsets[i]
            end = start + word_counts[i]
            if start <= gidx < end:
                target = i
                break
        if target is None:
            continue
        c2 = dict(c)
        c2["idx"] = gidx - offsets[target]
        if c.get("idx_end") is not None:
            try:
                end_local = int(c["idx_end"]) - offsets[target]
                max_local = word_counts[target] - 1
                c2["idx_end"] = min(max(end_local, c2["idx"]), max_local)
            except (TypeError, ValueError):
                c2.pop("idx_end", None)
        out[members[target]["id"]].append(c2)
    return out


def _enrich_sample(g: dict[str, Any]) -> dict[str, Any]:
    """Add `members` + `merged_words` to a group dict and re-derive the
    chip-dependent fields (transcript + corrections) from current
    member state.

    Source of truth for chips is each MEMBER's `corrections` list. With
    every read going through this function, member-chip edits on /captures
    (the member's own singleton card, or another admin tab) flow through
    to the group's Corrections section automatically — no in-DB chip
    storage needed at the group level."""
    members = capture_samples_store.get_members(g["id"])
    _hydrate_members(members)
    usernames = api_keys_store.get_usernames(
        [m.get("user_id") for m in members] + [g.get("user_id")]
    )
    member_trims = g.get("member_trims") or {}
    for m in members:
        _refresh_final_if_stale(m, parent_locked=bool(g.get("is_locked")))
        m["username"] = usernames.get(m.get("user_id"))
        # Per-member trimmed duration so the expanded member list shows the
        # length each clip actually contributes to the merged WAV (mirrors the
        # singleton effective_audio_s field). Falls back to raw for
        # legacy groups with no stored per-member trim map.
        info = member_trims.get(m.get("id"))
        if info and info.get("new_duration_ms") is not None:
            m["effective_audio_s"] = float(info["new_duration_ms"]) / 1000.0
        else:
            m["effective_audio_s"] = float(m.get("audio_s") or 0.0)
    g["members"] = members
    g["username"] = usernames.get(g.get("user_id"))
    g["transcript"] = capture_samples._build_default_transcript(
        members, g.get("transcript_join_strategy") or "space",
    )
    g["corrections"] = _project_member_corrections(members)
    g["merged_words"] = _build_merged_words(
        members, int(g["inter_segment_silence_ms"]),
        member_trims=g.get("member_trims") or {},
        merged_lead_trim_ms=int(g.get("merged_lead_trim_ms") or 0),
        merged_duration_ms=int(g.get("merged_duration_ms") or 0),
    )
    # Effective duration mirrors the singleton field: equals
    # merged_duration_ms / 1000 since merged_duration_ms is already the
    # post-trim value. Exposed for a uniform display shape with
    # singletons.
    g["effective_audio_s"] = float(g.get("merged_duration_ms") or 0) / 1000.0
    return g


def _refresh_final_if_stale(
    row: dict[str, Any], *, parent_locked: bool | None = None,
) -> None:
    """Recompute `final` AND `text_for_training` from `raw` via the
    current pipeline. If either differs from what's stored, write it
    back and update the row in place.

    This is the per-row self-heal that keeps fetched captures
    rule-current without requiring the user to click "Re-apply rules"
    first. The bulk reapply job is still useful for unfetched captures
    (export, retention sweep).

    `text_for_training` is the canonical /captures display text — it
    must reflect current PIPELINE_RULES (minus the captures-specific
    excludes) so reviewers see what the export will emit and chips
    apply against the same text the trainer will consume.

    Members of a LOCKED sample are left untouched: the lock guard
    (_assert_member_sample_not_locked) blocks explicit member rewrites
    and the admin-only bulk reapply preserves a locked sample's
    transcript snapshot — a mere page view must not rewrite the text
    both of those freeze. `parent_locked` skips the per-row sample
    lookup when the caller (e.g. _enrich_sample) already has the parent."""
    if parent_locked is None:
        sid = row.get("sample_id")
        if sid:
            s = capture_samples_store.get_sample(sid)
            parent_locked = bool(s and s.get("is_locked"))
    if parent_locked:
        return
    raw = row.get("raw") or ""
    stored_training = row.get("text_for_training") or ""
    if not raw:
        return
    stored_final = row.get("final") or ""
    try:
        # Resolve the capture OWNER's effective pipeline (mirrors the explicit
        # /reprocess endpoint). Without ident the self-heal would recompute with
        # GLOBAL rules and write the result back, silently reverting any
        # per-identity reprocess and producing wrong text for owners with
        # per-identity pipeline rules.
        ident = effective_config.build_ident({"user_id": row.get("user_id")}, row.get("model"))
        fresh_final = pl_engine._postprocess_text(raw, model_name=row.get("model"), ident=ident, language=captures_store.text_language(row))
    except Exception:
        return
    patch: dict[str, Any] = {}
    if fresh_final != stored_final:
        patch["final"] = fresh_final
        row["final"] = fresh_final
    # Self-heal text_for_training even when `final` is up-to-date — the
    # captures-excludes set could have changed (admin tweaked
    # CAPTURES_PIPELINE_RULES_EXCLUDE) without an underlying PIPELINE_RULES
    # change, and stale training text would mislead reviewers and
    # the export.
    captures_excludes = getattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None)
    if captures_excludes:
        try:
            fresh_training = pl_engine._postprocess_text(
                raw,
                model_name=row.get("model"),
                extra_excludes=captures_excludes,
                ident=ident,
                language=captures_store.text_language(row),
            )
        except Exception:
            fresh_training = None
    else:
        # No captures-specific excludes — training text is identical to
        # final, skip the second pipeline pass.
        fresh_training = fresh_final
    if fresh_training is not None and fresh_training != stored_training:
        patch["text_for_training"] = fresh_training
        row["text_for_training"] = fresh_training
    if patch:
        try:
            captures_store.update_capture(row["id"], patch)
        except Exception as e:
            logger.warning(
                "[captures] self-heal write-back failed for %s: %s",
                str(row.get("id"))[:8], e,
            )


def _remap_time_ms(tm: float, segments: list[list[int]]) -> float:
    """Map an original member-time (ms) onto trimmed member-local time (ms)
    via the per-member kept-speech `segments` list
    ([orig_start_ms, orig_end_ms, new_start_ms], ascending). A time inside a
    kept span maps linearly; a time in a dropped/collapsed silence region
    snaps to the nearest span boundary (words live in speech, so this only
    fires on the rare straddle/rounding case)."""
    if not segments:
        return tm
    if tm <= segments[0][0]:
        return float(segments[0][2])
    for os_ms, oe_ms, ns_ms in segments:
        if tm <= oe_ms:
            if tm >= os_ms:
                return float(ns_ms + (tm - os_ms))
            return float(ns_ms)  # in collapsed gap before this span
    last = segments[-1]
    return float(last[2] + (last[1] - last[0]))  # past the last span


def _emit_member_words(
    merged: list[dict[str, Any]],
    ws: list[dict[str, Any]],
    *,
    i: int,
    member_offset_s: float,
    eff_dur_s: float | None,
    segments: list[list[int]] | None,
) -> None:
    """Append member i's aligned words to `merged`, placed on the merged
    timeline. When `segments` is given, each word's original time is remapped
    through the per-member trim map first (per-member trimming); otherwise the
    legacy flat offset is used."""
    for w in ws:
        start = w.get("start")
        end = w.get("end", start)
        word = w.get("word", "")
        if start is None or end is None:
            continue
        if segments is None:
            s_new = float(start) + member_offset_s
            e_new = float(end) + member_offset_s
        else:
            s_new = _remap_time_ms(float(start) * 1000.0, segments) / 1000.0 \
                + member_offset_s
            e_new = _remap_time_ms(float(end) * 1000.0, segments) / 1000.0 \
                + member_offset_s
        if e_new <= 0:
            continue
        if eff_dur_s is not None and s_new >= eff_dur_s:
            continue
        s_clamped = max(0.0, s_new)
        e_clamped = e_new
        if eff_dur_s is not None:
            e_clamped = min(eff_dur_s, e_clamped)
        entry = {
            "word":       word,
            "start":      s_clamped,
            "end":        max(s_clamped, e_clamped),
            "member_idx": i,
        }
        if w.get("raw_word"):
            entry["raw_word"] = w["raw_word"]
        if w.get("removed"):
            entry["removed"] = True
        # Training-text token for this raw word (CAPTURES_PIPELINE_RULES_EXCLUDE
        # respected). The Corrections strip shows `word` (runtime `final`); the
        # Final-result karaoke uses `train_word` so it matches what the export
        # emits. Absent when training text == final (no excluded rule differs).
        if w.get("train_word") is not None:
            entry["train_word"] = w["train_word"]
            if w.get("train_removed"):
                entry["train_removed"] = True
        merged.append(entry)


def _align_member_words(
    m: dict[str, Any],
    ident_cache: dict | None = None,
) -> list[dict[str, Any]]:
    """Align a member's raw words to its runtime `final` (for the Corrections
    strip), and — when the training text differs (CAPTURES_PIPELINE_RULES_EXCLUDE
    drops some rules) — also align to `text_for_training` and attach the
    per-word `train_word`/`train_removed`. One entry per raw word, so both
    alignments are index-parallel and chip word-indices stay valid.

    `ident_cache` (optional) memoises the owner-identity resolve by
    (user_id, model) so a merge of N same-owner members does one
    api_keys_store lookup instead of one per member."""
    words = m.get("words") or []
    final = m.get("final") or ""
    training = m.get("text_for_training") or final
    uid, mdl = m.get("user_id"), m.get("model")
    if ident_cache is None:
        ident = effective_config.build_ident({"user_id": uid}, mdl)
    else:
        ckey = (uid, mdl)
        if ckey not in ident_cache:
            ident_cache[ckey] = effective_config.build_ident({"user_id": uid}, mdl)
        ident = ident_cache[ckey]
    # Pipeline scope = the TEXT language ("en" for task=translate).
    _lang = captures_store.text_language(m)
    ws = capture_samples._align_words_to_final(words, final, model_name=m.get("model"), ident=ident, language=_lang)
    if training != final:
        wt = capture_samples._align_words_to_final(words, training, model_name=m.get("model"), ident=ident, language=_lang)
        for i, w in enumerate(ws):
            tw = wt[i] if i < len(wt) else None
            if tw is None:
                continue
            # Preserve the raw word's leading whitespace in front of the
            # training token. `_align_words_to_final` strips the lead off
            # MATCHED tokens, so without this a multi-word Corrections range
            # would join into a run with no inter-word spaces ("134Schrägstrich92")
            # and fail to match the training text. The Corrections strip's
            # `.replace(/^\s+/,' ')` normalizes the display lead, and
            # `_renderGroundSpans` .trim()s, so neither is affected.
            raw_w = (words[i].get("word") if i < len(words) else "") or ""
            lead = raw_w[: len(raw_w) - len(raw_w.lstrip())]
            core = (tw.get("word") or "").lstrip()
            w["train_word"] = (lead + core) if core else ""
            if tw.get("removed"):
                w["train_removed"] = True
    return ws


def _build_merged_words(
    members: list[dict[str, Any]],
    silence_ms: int,
    *,
    member_trims: dict[str, Any] | None = None,
    merged_lead_trim_ms: int = 0,
    merged_duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Project each member's per-word timestamps onto the merged-audio
    timeline. start/end are returned in seconds (matches audio.currentTime
    and the single-capture karaoke band's expectation).

    Two timelines, picked by whether `member_trims` is populated:

    - Per-member trimming (new groups): each member was silence-trimmed
      before concatenation, so member i starts at (Σ_{j<i} new_dur_j) +
      i × silence_s, and each word's original time is remapped through that
      member's kept-speech `segments` map. This is what keeps karaoke aligned
      after the dead-air at member joins is removed.

    - Legacy groups (member_trims empty): member i starts at (Σ_{j<i} dur_j) +
      i × silence_s − merged_lead_trim_ms, using full member durations and the
      single merged-WAV outer-edge offset — i.e. the original behaviour, so
      pre-existing groups render exactly as before.

    `get_members` strips heavy fields for the list view, so callers hydrate
    `words` first. Each member's words go through `_align_words_to_final`
    (LCS-align raw→final). Cost is bounded — ≤30 members, ≤a few hundred words
    per packed sample — and only runs on expand."""
    silence_s = max(0, int(silence_ms)) / 1000.0
    eff_dur_s: float | None = None
    if merged_duration_ms is not None:
        eff_dur_s = max(0.0, float(merged_duration_ms) / 1000.0)
    merged: list[dict[str, Any]] = []
    # Members of a merge share an owner+model, so memoise the per-member
    # identity resolve to avoid one api_keys_store lookup per member.
    ident_cache: dict = {}

    use_per_member = bool(member_trims)
    # New uniform-silence layout stamps each member with an absolute
    # `offset_ms`; legacy groups don't, and fall back to the cum+i*silence
    # formula so their karaoke renders exactly as before (until reprocessed).
    _first = (member_trims or {}).get(members[0]["id"]) if (use_per_member and members) else None
    new_layout = bool(_first and _first.get("offset_ms") is not None)

    if use_per_member and eff_dur_s is None:
        # Preview path: no stored merged duration — derive it.
        n = len(members)
        if new_layout:
            edge_ms = float(capture_samples._global_edge_ms())
            last_end = 0.0
            for m in members:
                info = (member_trims or {}).get(m["id"]) or {}
                last_end = max(last_end, float(info.get("offset_ms") or 0.0)
                               + float(info.get("new_duration_ms") or 0.0))
            eff_dur_s = (last_end + edge_ms) / 1000.0
        else:
            total_new_ms = 0.0
            for m in members:
                info = member_trims.get(m["id"]) if member_trims else None
                if info and info.get("new_duration_ms") is not None:
                    total_new_ms += float(info["new_duration_ms"])
                else:
                    total_new_ms += float(m.get("audio_s") or 0.0) * 1000.0
            eff_dur_s = (total_new_ms + max(0, n - 1) * float(silence_ms)) / 1000.0

    if use_per_member:
        cum_ms = 0.0
        for i, m in enumerate(members):
            info = (member_trims or {}).get(m["id"])
            off_ms = None
            if info and info.get("segments"):
                segments = info["segments"]
                new_dur_ms = float(info.get("new_duration_ms") or 0.0)
                off_ms = info.get("offset_ms")
            else:
                # Member not in the trim map (shouldn't happen) → identity.
                dur_ms = float(m.get("audio_s") or 0.0) * 1000.0
                segments = [[0, int(dur_ms), 0]]
                new_dur_ms = dur_ms
            if off_ms is not None:
                member_offset_s = float(off_ms) / 1000.0     # new uniform layout
            else:
                member_offset_s = cum_ms / 1000.0 + i * silence_s  # legacy
            ws = _align_member_words(m, ident_cache)
            _emit_member_words(
                merged, ws, i=i, member_offset_s=member_offset_s,
                eff_dur_s=eff_dur_s, segments=segments,
            )
            cum_ms += new_dur_ms
        return merged

    # Legacy path (no per-member trims): original flat-offset behaviour.
    lead_s = max(0, int(merged_lead_trim_ms or 0)) / 1000.0
    cum = 0.0
    for i, m in enumerate(members):
        member_offset_s = cum + i * silence_s - lead_s
        ws = _align_member_words(m, ident_cache)
        _emit_member_words(
            merged, ws, i=i, member_offset_s=member_offset_s,
            eff_dur_s=eff_dur_s, segments=None,
        )
        cum += float(m.get("audio_s") or 0.0)
    return merged


@router.patch(
    "/api/samples/{sid}",
    dependencies=[Depends(require_page("captures"))],
)
async def patch_sample_api(
    sid: str,
    payload: PatchSampleIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
        detail="sample not found",
    )
    _audit_cross_user_read(user, g, "sample-patch", sid)
    if g["is_locked"] and not user.get("is_admin"):
        # `is_locked` is writable by any captures-scoped caller (below), so the
        # lock must also be RELEASABLE by them — otherwise setting it is a
        # one-way switch: the guard here would block the very patch that clears
        # it, and only an admin could undo it. A non-admin may therefore send
        # exactly one thing while the sample is locked, the unlock itself.
        # Every other edit stays frozen, which is what the lock is for.
        _unlock_only = (
            payload.is_locked is False
            and payload.model_dump(exclude_none=True).keys() == {"is_locked"}
        )
        if not _unlock_only:
            raise HTTPException(status.HTTP_409_CONFLICT, "sample is locked")

    # Off the loop. With corrections set this hydrates every member (one full
    # get_capture each, json.loads of the words/segments blobs — the pattern
    # list_samples_api measured at ~2.5 ms/member), fans the chips out with up
    # to 30 update_capture writes and rebuilds the transcript.
    def _apply() -> dict[str, Any] | None:
        patch: dict[str, Any] = {}
        # Lazily-fetched hydrated members; up to three branches below need
        # this list and used to issue independent get_members calls each.
        _members_cache: list[dict[str, Any]] | None = None
        def _members() -> list[dict[str, Any]]:
            nonlocal _members_cache
            if _members_cache is None:
                _members_cache = capture_samples_store.get_members(sid)
                _hydrate_members(_members_cache)
            return _members_cache

        # Join strategy + inter-member silence are GLOBAL settings now; they're no
        # longer patchable per-sample. Changing them and rebuilding audio for
        # existing samples is done via the bulk VAD reprocess action / regenerate.
        if payload.is_locked is not None:
            patch["is_locked"] = 1 if payload.is_locked else 0
        if payload.status is not None:
            patch["status"] = payload.status
        if payload.admin_notes is not None:
            patch["admin_notes"] = payload.admin_notes
        if payload.corrections is not None:
            # Fan group-level chip edits DOWN to the owning members. Group
            # corrections are derived from members on every read (see
            # `_enrich_sample`); writing to a group-level chip column would
            # be discarded by the next GET.
            #
            # When the client also sends `baseline_corrections` (a snapshot
            # of what it loaded), apply a three-way merge against the
            # current member-projected chips BEFORE the split — that way a
            # concurrent cross-tab admin save (or a member edited from its
            # singleton /captures card) isn't clobbered by the user's payload,
            # and the user's deltas (additions, removals, edits) apply on top.
            members_now = _members()
            edited = [c.model_dump() for c in payload.corrections]
            if payload.baseline_corrections is not None:
                baseline = [c.model_dump() for c in payload.baseline_corrections]
                current = _project_member_corrections(members_now)
                edited = text_corrections.three_way_merge_corrections(
                    baseline, edited, current,
                )
            by_member = _split_corrections_to_members(edited, members_now)
            # Before any write: a member past the store's cap would lose
            # chips silently.
            if any(text_corrections.over_cap(c) for c in by_member.values()):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"a member capture holds at most"
                    f" {text_corrections.CAP_CORRECTIONS} corrections",
                )
            # Skip members whose chip set didn't change — a 30-member group
            # with one edited chip otherwise fires 30 UPDATEs where 29 are
            # idempotent rewrites of the same JSON column.
            current_by_id = {m["id"]: (m.get("corrections") or []) for m in members_now}
            for member_id, chips in by_member.items():
                if json.dumps(current_by_id.get(member_id) or [], sort_keys=True) == \
                        json.dumps(chips, sort_keys=True):
                    continue
                captures_store.update_capture(member_id, {"corrections": chips})

        # Re-derive `transcript` from current members + chips ONLY when the
        # corrections changed. The common status/admin_notes/is_locked auto-save
        # click would otherwise trigger a get_members + transcript rebuild + DB
        # write on every click. Join strategy uses the sample's stored value
        # (the global only re-applies on regenerate / bulk reprocess).
        if payload.corrections is not None:
            join_for_derive = g["transcript_join_strategy"] or "space"
            patch["transcript"] = capture_samples._build_default_transcript(_members(), join_for_derive)

        return capture_samples_store.update_sample(sid, patch)

    def _apply_serialised() -> dict[str, Any] | None:
        if payload.corrections is None:
            return _apply()
        # The member read, three-way merge and per-member writes must not
        # interleave with a member's own PATCH or another group save.
        with _corrections_write_lock:
            return _apply()

    updated = await asyncio.to_thread(_apply_serialised)
    if updated is None:
        # Dissolved while we waited (dissolve / member delete run in threads
        # too): update_sample matched no row. A 404, not a 500 from
        # _enrich_sample(None).
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    # Off the loop — see get_sample_api: the enrich pass is a quadratic LCS
    # per member.
    return JSONResponse(
        {"sample": await asyncio.to_thread(_enrich_sample, updated)})


@router.post(
    "/api/samples/{sid}/regenerate",
    dependencies=[Depends(require_page("captures"))],
)
async def regenerate_sample_api(
    sid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Rebuild the merged WAV from current member content using the CURRENT
    global silence setting (so regenerate is how an existing sample adopts a
    changed global), refresh hashes, clear `is_stale`. Transcript is preserved
    (admin's edits stay)."""
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
        detail="sample not found",
    )
    _audit_cross_user_read(user, g, "sample-regenerate", sid)
    if g["is_locked"] and not user.get("is_admin"):
        raise HTTPException(status.HTTP_409_CONFLICT, "sample is locked")
    members = capture_samples_store.get_members(sid)
    silence_ms = capture_samples._global_silence_ms()

    def _regenerate() -> dict[str, Any]:
        with capture_samples._rebuild_lock(sid):
            # Re-read under the lock dissolve_sample also takes: a sample
            # dissolved while this request waited must not get its merged WAV
            # rebuilt onto disk with no row (nor a 500 from _enrich_sample).
            if capture_samples_store.get_sample(sid) is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
            duration_ms, hashes, member_trims = capture_samples._build_merged_wav(
                sid=sid,
                member_ids=[m["id"] for m in members],
                silence_ms=silence_ms,
            )
            regen_patch = capture_samples._merged_wav_patch(duration_ms, hashes, member_trims)
            regen_patch["inter_segment_silence_ms"] = silence_ms
            return capture_samples_store.update_sample(sid, regen_patch)

    # Off the loop for the same reason as the sample-audio route: the lock is
    # shared with the VAD reprocess worker thread, and acquiring it inline
    # blocks the whole server for the length of that rebuild.
    updated = await asyncio.to_thread(_regenerate)
    # The enrich pass belongs off the loop too — see get_sample_api. It was
    # left inline while the rebuild beside it was already offloaded.
    return JSONResponse(
        {"sample": await asyncio.to_thread(_enrich_sample, updated)})


@router.delete(
    "/api/samples/{sid}",
    dependencies=[Depends(require_page("captures"))],
)
async def dissolve_sample_api(
    sid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
        detail="sample not found",
    )
    _audit_cross_user_read(user, g, "sample-delete", sid)
    if g["is_locked"] and not user.get("is_admin"):
        raise HTTPException(status.HTTP_409_CONFLICT, "sample is locked")
    # Off the loop: dissolve waits on the per-sid rebuild lock, which an
    # in-flight regenerate / VAD rebuild holds for seconds.
    await asyncio.to_thread(capture_samples_store.dissolve_sample, sid)
    return JSONResponse({"ok": True})

def _ensure_sample_wav(g: dict[str, Any]) -> str:
    """Resolve the merged-WAV abs path. If the file is missing on
    disk but every member capture still has its row + audio, rebuild
    the WAV in place and return its abs path. If unrecoverable, raise
    HTTPException(410).

    Merged WAVs are deterministic functions of (members, silence_ms,
    join_strategy) — treating them as cached derived data instead of
    a precious one-shot artifact means no deletion path (clear_all,
    legacy reconcile, crash mid-write, manual cleanup, etc.) surfaces
    as a hard 404 to the user. The "Regenerate" button still exists
    for the legitimate force-rebuild case (user edited silence/join).
    """
    try:
        abs_p = capture_samples_store.abs_path_for(g["merged_wav_relpath"])
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "merged audio missing")
    if os.path.exists(abs_p):
        return abs_p

    members = capture_samples_store.get_members(g["id"])
    if not members:
        raise HTTPException(
            status.HTTP_410_GONE,
            "members deleted — sample is unrecoverable",
        )
    for m in members:
        cap = captures_store.get_capture(m["id"])
        if cap is None:
            raise HTTPException(
                status.HTTP_410_GONE,
                f"member {m['id'][:8]} row deleted — sample is unrecoverable",
            )
        member_abs = captures_store.abs_audio_path(cap["audio_relpath"])
        if not os.path.exists(member_abs):
            raise HTTPException(
                status.HTTP_410_GONE,
                f"member {m['id'][:8]} audio is gone — sample is unrecoverable",
            )

    member_ids = [m["id"] for m in members]
    with capture_samples._rebuild_lock(g["id"]):
        if os.path.exists(abs_p):
            return abs_p
        # Dissolved while we waited (dissolve_sample holds this lock): do not
        # resurrect its merged WAV as an orphan.
        if capture_samples_store.get_sample(g["id"]) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
        logger.warning(
            "[samples] sid=%s auto-rebuilding missing WAV from %d members",
            g["id"][:8], len(member_ids),
        )
        duration_ms, hashes, member_trims = capture_samples._build_merged_wav(
            sid=g["id"],
            member_ids=member_ids,
            silence_ms=int(g["inter_segment_silence_ms"]),
        )
        capture_samples_store.update_sample(
            g["id"], capture_samples._merged_wav_patch(duration_ms, hashes, member_trims),
        )
    return abs_p


@router.get(
    "/api/samples/{sid}/audio",
    dependencies=[Depends(require_page("captures"))],
)
async def get_sample_audio_api(
    sid: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Stream the merged WAV, self-healing if it's missing on disk
    but reconstructable from member captures."""
    _audio_rate.hit(rate_limit.identity_key(user, request))
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
        detail="sample not found",
    )
    _audit_cross_user_read(user, g, "sample-audio", sid)
    # Off the loop: this takes a cross-thread threading.Lock that the VAD
    # reprocess worker holds for a whole rebuild, and then runs the merge
    # (a Silero-VAD pass per member plus a WAV write) synchronously. Inline
    # in an `async def` that parked the single event-loop thread inside
    # lock.acquire(), stalling every other request and WebSocket.
    abs_p = await asyncio.to_thread(_ensure_sample_wav, g)
    return FileResponse(
        abs_p,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------
# Export: streamed tar.gz
# ---------------------------------------------------------------------

def _build_manifest_row(
    *,
    audio_filepath: str,
    text: str,
    duration: float,
    language: str,
    source: str,
    user_id: str,
    status_value: str,
    created_ts: float,
    model: str,
    request_id: str,
    member_count: int,
    admin_notes: str,
    corrections: list,
    task: str = "transcribe",
) -> dict[str, Any]:
    """Build a single manifest line dict with the unified 14-key schema.

    Same keys for singletons and groups — defaults populate the keys
    that don't apply (e.g. groups → request_id=""; singletons →
    member_count=1). Heterogeneous-field-set was a documented pain
    point for cross-corpus filtering during fine-tuning.

    `task` is Whisper's own multitask label: "transcribe" pairs audio with
    text in the SAME language, "translate" pairs non-English audio with
    English text. A finetuning run has to condition on it, and a row that
    does not say which it is cannot be used for either.
    """
    return {
        "audio_filepath": audio_filepath,
        "text": text,
        "duration": float(duration or 0.0),
        "language": language or "",
        "source": source,
        "user_id": user_id or "",
        "status": status_value or "",
        "created_ts": float(created_ts or 0.0),
        "model": model or "",
        "request_id": request_id or "",
        "member_count": int(member_count or 0),
        "admin_notes": admin_notes or "",
        "corrections": list(corrections or []),
        "task": task or "transcribe",
    }


def _lang_base(code: str) -> str:
    """`en-US` → `en`: the base subtag is what "English track" means."""
    return (str(code or "").split("-", 1)[0] or "").strip().lower()


def _group_task(members: list) -> str:
    """The task label for a merged group.

    A group is one audio file built from several members, so it only has a
    task if every member agrees on one. Anything mixed falls back to
    "transcribe" — the conservative answer, since mislabelling a row as
    translate would feed a finetuning run text in the wrong language."""
    tasks = {(m.get("task") or "transcribe") for m in (members or [])}
    return tasks.pop() if len(tasks) == 1 else "transcribe"


def _build_export_stream(only_status: str | None, include_audio: bool):
    """Generator that yields tar.gz bytes containing manifest.jsonl and,
    optionally, audio/<id>.wav entries. One manifest entry per training
    unit — a "unit" is either a capture group (packed sample) OR
    an ungrouped singleton capture. Group members never appear as
    singletons (no double-counting; the group transcript covers them).

    Hard filters applied regardless of `only_status`:
      - `is_stale=true` (group audio/text drift) → skipped
      - `is_locked=true` (admin lock; mirrors reapply skip) → skipped
      - `status=audio_missing` (file is gone) → skipped
      - missing WAV on disk → both manifest entry AND tar entry skipped
        (prevents the manifest from referencing files that don't exist
        in the tarball)
    `only_status` (default "ready" from the route) further narrows.
    `audio_missing` rows leak only when `only_status='all'`, and even
    then the hard filter above drops them.
    """

    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6)
    # Spooled, not a list: the manifest holds one JSON line per row carrying
    # the full transcript, admin notes and correction chips, and every line was
    # kept resident until the b"".join at the end materialised a second copy.
    # Measured ~104 MB peak at the shipped 5000-row cap, and grouped rows are
    # exempt from that cap. Rolls to disk past 8 MB; the tarball is unchanged.
    manifest_lines = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    # Closed even if the client disconnects mid-stream and the
    # generator is abandoned — otherwise every aborted export leaks
    # a spooled temp file.
    try:

        # 1. Capture groups (the packed-for-fine-tune training samples).
        user_filter_scope: str | None = None  # admin-only path; no per-user scope
        for g in capture_samples_store.list_samples(user_id=user_filter_scope):
            # Status gate — groups have the same status field as captures.
            # The caller has already mapped "all" → None upstream, so a
            # truthy only_status here is a concrete status to match.
            if only_status:
                if (g.get("status") or "new") != only_status:
                    continue
            # Hard filters (apply even on `only_status=all`). Stale = audio/
            # text drift; locked = admin lock (same skip as captures_reapply,
            # so this stays consistent with what the trainer last saw).
            if g.get("is_stale") or g.get("is_locked"):
                continue
            # Always rebuild the transcript at export time from members +
            # chips, so the exported text reflects current corrections even
            # if the stored snapshot is stale. Source from the training-form
            # column so reviewers see — and the trainer learns from — the
            # same text.
            sid = g["id"]
            members = capture_samples_store.get_members(sid)
            text = capture_samples._build_default_transcript(
                members, g.get("transcript_join_strategy") or "space",
            ).strip()
            if not text:
                continue
            # Audio existence gate: skip the manifest entry entirely if the
            # WAV isn't on disk, to avoid manifest pointing at missing files.
            try:
                abs_p = capture_samples_store.abs_path_for(g["merged_wav_relpath"])
            except ValueError:
                continue
            if not os.path.isfile(abs_p):
                continue

            audio_name = f"audio/{sid}.wav"
            # Group `model` and `request_id` are intentionally empty — a
            # group has multiple members each with their own model id. Per-
            # member audit is reachable via the group's GET /members endpoint.
            manifest_lines.write(json.dumps(_build_manifest_row(
                audio_filepath=audio_name,
                text=text,
                duration=float(g.get("merged_duration_ms") or 0) / 1000.0,
                language=g.get("language") or "",
                source="sample",
                user_id=g.get("user_id") or "",
                status_value=g.get("status") or "new",
                created_ts=float(g.get("created_ts") or 0.0),
                model="",
                request_id="",
                member_count=len(members),
                admin_notes=g.get("admin_notes") or "",
                corrections=[],
                task=_group_task(members),
            ), ensure_ascii=False).encode("utf-8") + b"\n")

            if include_audio:
                info = tarfile.TarInfo(audio_name)
                info.size = os.path.getsize(abs_p)
                info.mtime = int(g.get("created_ts") or time.time())
                with open(abs_p, "rb") as af:
                    tar.addfile(info, af)
                chunk = buf.getvalue()
                buf.seek(0); buf.truncate()
                if chunk:
                    yield chunk

        # 2. Ungrouped captures (no sample_id).
        for row in captures_store.iter_captures_for_export(status=only_status):
            if row.get("sample_id"):
                continue
            # Hard filter: `audio_missing` rows have no WAV; never valid
            # training data. (Caught here even when only_status='all'.)
            if (row.get("status") or "") == "audio_missing":
                continue
            cid = row["id"]
            # Source training-form text first so the export matches what
            # reviewers see on /captures. Chip-applied on top. `final` and
            # `raw` fall-backs cover captures from before the
            # text_for_training column existed.
            base = (row.get("text_for_training")
                    or row.get("final")
                    or row.get("raw") or "")
            text = capture_samples._apply_chips_to_text(base, row.get("corrections") or [])
            if not text.strip():
                continue
            # Audio path: prefer the trimmed companion if one was produced and
            # is still on disk, else the original (as get_audio_api plays it:
            # a stale trimmed relpath must not drop a row the reviewer heard).
            # The manifest line is skipped only if neither file is on disk
            # (defense against the audio_missing leak path).
            rel = abs_p = None
            trimmed = False
            for cand in (row.get("audio_trimmed_relpath"), row.get("audio_relpath")):
                if not cand:
                    continue
                try:
                    cand_abs = captures_store.abs_audio_path(cand)
                except ValueError:
                    continue
                if os.path.isfile(cand_abs):
                    rel, abs_p = cand, cand_abs
                    trimmed = cand == row.get("audio_trimmed_relpath")
                    break
            if rel is None:
                continue
            # The manifest duration must describe the file actually packed:
            # the trimmed companion is shorter by the cut lead + trail.
            dur = float(row.get("audio_s") or 0.0)
            if trimmed:
                dur = max(0.0, dur - (int(row.get("audio_trim_lead_ms") or 0)
                                      + int(row.get("audio_trim_trail_ms") or 0)) / 1000.0)
            ext = os.path.splitext(rel)[1].lstrip(".").lower() or "wav"
            audio_name = f"audio/{cid}.{ext}"
            manifest_lines.write(json.dumps(_build_manifest_row(
                audio_filepath=audio_name,
                text=text,
                duration=dur,
                language=row.get("language") or "",
                source="singleton",
                user_id=row.get("user_id") or "",
                status_value=row.get("status") or "",
                created_ts=float(row.get("created_ts") or 0.0),
                model=row.get("model") or "",
                request_id=row.get("request_id") or "",
                member_count=1,
                admin_notes=row.get("admin_notes") or "",
                corrections=row.get("corrections") or [],
                task=row.get("task") or "transcribe",
            ), ensure_ascii=False).encode("utf-8") + b"\n")

            # A second manifest line for the English translation, when one
            # exists AND a human has reviewed it. Whisper's translate task
            # targets English only, so no other track is ever eligible; and
            # an unreviewed line is a cascade pseudo-label (Whisper → HY-MT)
            # whose errors compound, which is exactly the material a
            # finetuning run should not be fed silently. The review workflow
            # is what promotes it, so `ready` is the gate. Match the track by
            # BASE subtag (`en-US` is accepted as a target and stored under
            # that key), and never emit one for English-language audio: the
            # "translation" of an English source is its own transcript.
            # Nor for a capture that IS a translate request: its singleton
            # line above already carries the English target as task=translate,
            # so an MT track would be a contradictory duplicate (other text,
            # other model) for the same audio_filepath.
            _tr = row.get("translations") or {}
            _en_key = next((k for k in _tr if _lang_base(k) == "en"), None)
            _en = (_tr.get(_en_key) or "").strip() if _en_key else ""
            _src_is_en = _lang_base(row.get("language") or "") == "en"
            _row_is_translate = (row.get("task") or "transcribe") == "translate"
            if (_en and not _src_is_en and not _row_is_translate
                    and (row.get("status") or "") == "ready"):
                manifest_lines.write(json.dumps(_build_manifest_row(
                    audio_filepath=audio_name,
                    text=_en,
                    duration=dur,
                    language=row.get("language") or "",
                    source="singleton",
                    user_id=row.get("user_id") or "",
                    status_value=row.get("status") or "",
                    created_ts=float(row.get("created_ts") or 0.0),
                    model=row.get("translation_model") or "",
                    request_id=row.get("request_id") or "",
                    member_count=1,
                    admin_notes=row.get("admin_notes") or "",
                    corrections=[],
                    task="translate",
                ), ensure_ascii=False).encode("utf-8") + b"\n")

            if include_audio:
                info = tarfile.TarInfo(audio_name)
                info.size = os.path.getsize(abs_p)
                info.mtime = int(row.get("created_ts") or time.time())
                with open(abs_p, "rb") as af:
                    tar.addfile(info, af)
                chunk = buf.getvalue()
                buf.seek(0); buf.truncate()
                if chunk:
                    yield chunk

        # Manifest last so it's written in row order matching the audio.
        info = tarfile.TarInfo("manifest.jsonl")
        info.size = manifest_lines.tell()
        info.mtime = int(time.time())
        manifest_lines.seek(0)
        tar.addfile(info, manifest_lines)

        tar.close()
        final_chunk = buf.getvalue()
        if final_chunk:
            yield final_chunk
    finally:
        manifest_lines.close()


# ---------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------
# Single-page admin: list view + per-row expand. The expanded row plays
# the audio karaoke-style (active word highlights with playback), lets
# the admin shift-click to mark wrong words, and edits the ground-truth
# via per-span correction chips. Status flow: new → reviewed → ready
# (export-eligible) | dismissed (omitted).
#
# IMPORTANT (CLAUDE memory note): never place a `{{...}}` placeholder
# inside a /* */, //, or <!-- --> comment — render_page() does a literal
# string replace and the substitution corrupts the surrounding context.

_CAPTURES_HTML = templates.load(__file__, "captures.html")
