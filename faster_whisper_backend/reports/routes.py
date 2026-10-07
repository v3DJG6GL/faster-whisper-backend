"""Transcription error report submission + admin triage.

Two surfaces:

  /quick-config/reports/api/*       — end-user routes. Gated by
    require_user_webui_host + get_current_user. The form + delete control
    live inline on each .trace-item in /quick-config:
    POST   /quick-config/reports/api/submit                — receiver
    DELETE /quick-config/reports/api/by-request/{request_id} — caller
                                                              undoes
                                                              own row

  /reports                          — triage page + APIs:
    GET   /reports                  HTML triage page
    GET   /reports/api/list         reports in scope (newest first)
    PATCH /reports/api/{rid}        status + admin_notes
    DELETE /reports/api/{rid}       single delete
    POST  /reports/api/clear        wipe all (confirm dialog)
    GET   /reports/api/export       full JSON dump (envelope-wrapped)
  list / PATCH / DELETE gate on require_page("reports") plus the per-row
  scope check, so a scope=own user reads, edits and deletes their own
  rows; only /clear and /export use Depends(require_admin).

Reports are an independent store: submitting or deleting a report does
NOT touch any capture's chip corrections. End users edit PIPELINE_RULES
at /quick-config for single-word fixes; the captures-side merge-proposal
+ batch-review flow at /captures handles bulk corrections on stored
training data.

Per-identity rate limit on submission: the shared rate_limit.FixedWindow,
keyed on user_id, else the API key id, else request.client.host.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from faster_whisper_backend.auth import api_keys_store
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.auth import rate_limit
from faster_whisper_backend.reports import store as reports_store
from faster_whisper_backend.core import web_common
from faster_whisper_backend.core.web_common import require_user_webui_host
from faster_whisper_backend.auth.dependencies import get_current_user, require_admin, require_page
from faster_whisper_backend.core import templates

router = APIRouter()


# ---------------------------------------------------------------------
# Submission payload
# ---------------------------------------------------------------------

class CorrectionIn(BaseModel):
    model_config = {"extra": "forbid"}
    # Hygiene only: bounds what can be STORED per correction. The store
    # truncates both to text_corrections.CAP_CORRECTION_FIELD (200) anyway,
    # so 4096 (~20x that) is a generous edge bound that cannot reject any
    # submission which succeeds today. It is NOT a memory guard — the whole
    # body is json.loads'd before pydantic sees it; that ceiling lives in
    # main.py's request-size handling.
    wrong: str = Field(default="", max_length=4096)
    correct: str = Field(default="", max_length=4096)
    idx: int | None = None
    # Inclusive end-of-range for multi-word selections. Omitted (or equal
    # to idx) for single-word corrections. Validation happens server-side
    # in reports_store._clean_corrections.
    idx_end: int | None = None


# Bounds for the nested walk in ReportSubmitIn._bound_nested_strings.
_NESTED_STR_MAX = 65_536
_NESTED_BUDGET = {"steps": 1_048_576, "stages": 262_144}
_NESTED_MAX_NODES = 10_000


class ReportSubmitIn(BaseModel):
    # allow_inf_nan=False: a bare float otherwise accepts Infinity/NaN, which
    # sqlite stores intact and Starlette's allow_nan=False renderer then chokes
    # on for every subsequent list read. reports_store re-bounds it as well.
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    trace_ts: float = Field(default=0.0, ge=0, le=4_102_444_800)
    request_id: str | None = Field(default=None, max_length=128)
    model: str = Field(default="", max_length=256)
    # Bounded at the edge instead of silently truncated in the store. The values
    # mirror reports_store's own caps (_CAP_RAW / _CAP_FINAL / _CAP_STEPS_ROWS),
    # so nothing that is stored intact today starts failing — but a submission
    # far past them no longer gets parsed and re-serialised on the way in.
    raw: str = Field(default="", max_length=50_000)
    final: str = Field(default="", max_length=50_000)
    steps: list[Any] = Field(default=[], max_length=500)
    corrections: list[CorrectionIn] = Field(default=[], max_length=500)
    # 64 KB, deliberately NOT the store's own 2 000 / 4 000 caps. The store
    # slices these AFTER pydantic has parsed the whole body, and the only other
    # ceiling is MAX_REQUEST_BYTES (256 MB) — a submit carrying two 100 MB
    # strings parsed fine and peaked at 627 MB of RSS to store 6 KB. A ceiling
    # 16x above the store's functional cap kills that while staying far above
    # anything a human types: the report textarea carries no maxlength, so
    # bounding at 4 000 would turn a long comment that succeeds today (silently
    # truncated) into a 422 with the text lost.
    intended_text: str = Field(default="", max_length=65_536)
    user_comment: str = Field(default="", max_length=65_536)
    # Job provenance. extra="forbid" above means the client cannot send
    # these until they are declared here -- a report about a translation
    # used to identify neither the language nor the translation nor the
    # model that produced it.
    language: str = Field(default="", max_length=32)
    stages: list[Any] = Field(default=[], max_length=32)

    # Parity with the intended_text/user_comment ceilings above: max_length
    # on a list[Any] bounds the ITEM COUNT only, so one 100 MB string nested
    # in steps[0] (or inside a stages dict) reproduces exactly the case the
    # 64 KB ceilings were added to kill. Any contained str is bounded at the
    # same 64 KB, and the walked total at a budget far above the store's
    # _CAP_STEPS_JSON / _CAP_STAGES_JSON, so nothing stored intact today
    # starts 422-ing. The types stay list[Any]: tightening them would turn
    # shapes _truncate_steps silently drops today into 422s.
    @field_validator("steps", "stages")
    @classmethod
    def _bound_nested_strings(cls, v: list[Any], info: Any) -> list[Any]:
        budget = _NESTED_BUDGET[info.field_name]
        total = 0
        stack: list[Any] = [v]
        visited = 0
        while stack:
            item = stack.pop()
            visited += 1
            if visited > _NESTED_MAX_NODES:
                raise ValueError("field too large")
            if isinstance(item, str):
                if len(item) > _NESTED_STR_MAX:
                    raise ValueError("field too large")
                total += len(item)
                if total > budget:
                    raise ValueError("field too large")
            elif isinstance(item, dict):
                stack.extend(item.keys())
                stack.extend(item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        return v


# ---------------------------------------------------------------------
# Rate limit (per reporter — identity, with a host fallback)
# ---------------------------------------------------------------------
# The threat model is "accidental double-click / runaway script", not a
# motivated attacker — for that the LAN box is already locked down by
# require_user_webui_host. Each accepted submission writes a durable row and
# runs an eviction sweep, so the window is long (10 min) and the budget small.

_rate = rate_limit.FixedWindow(
    config_field="REPORTS_SUBMIT_RATE_PER_10MIN",
    window_s=600.0,
    default_max=20,
    message="Too many reports ({limit} per 10 minutes). Try again in "
            "{retry_after}s.",
)


# ---------------------------------------------------------------------
# Submission endpoint (under /quick-config)
# ---------------------------------------------------------------------

@router.post(
    "/quick-config/reports/api/submit",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def submit_report(
    payload: ReportSubmitIn,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    is_admin = bool(user.get("is_admin"))
    if not getattr(cfg, "REPORTS_ALLOW_USER_SUBMIT", True) and not is_admin:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Report submission is disabled by the admin.",
        )

    _rate.hit(rate_limit.identity_key(user, request))

    intended = (payload.intended_text or "").strip()
    comment = (payload.user_comment or "").strip()
    corrections = reports_store._clean_corrections(
        [c.model_dump() for c in (payload.corrections or [])]
    )

    if not corrections and not intended and not comment:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Mark a wrong word, write what you meant to say, or leave a comment.",
        )

    host = request.client.host if request.client else ""
    # Off the loop: upsert_report runs _evict_to_cap (COUNT(*) + DELETE) on
    # every call, which measured a median 71.5 ms with the table at REPORTS_MAX
    # versus 7.5 ms on a partly-filled one. reports_store is thread-safe
    # (threading.Lock + check_same_thread=False), same as the list/export paths.
    rid, was_updated = await asyncio.to_thread(
        reports_store.upsert_report,
        user_id=user.get("user_id"),
        request_id=payload.request_id,
        trace_ts=float(payload.trace_ts or 0.0),
        model=payload.model or "",
        raw=payload.raw or "",
        final=payload.final or "",
        steps=list(payload.steps or []),
        corrections=corrections,
        intended_text=intended,
        user_comment=comment,
        reporter_role="admin" if is_admin else "user",
        reporter_host=host,
        language=payload.language or None,
        stages=list(payload.stages or []),
    )
    return JSONResponse({
        "ok": True,
        "id": rid,
        "was_updated": was_updated,
    })


# ---------------------------------------------------------------------
# Admin page + APIs
# ---------------------------------------------------------------------

@router.get(
    "/reports",
    # HTML page is host-only — the login modal runs in this page's
    # own JS, so the bearer isn't available on the initial navigation.
    # API endpoints below gate by `require_page("reports")`; if the
    # user lacks access, the first list-fetch 403s and the JS renders
    # a "no access" landing.
    dependencies=[Depends(require_user_webui_host)],
    response_class=HTMLResponse,
)
async def reports_page() -> HTMLResponse:
    if not getattr(cfg, "ADMIN_UI_ENABLED", False):
        return HTMLResponse("Admin UI disabled.", status_code=404)
    return HTMLResponse(
        web_common.render_page(_REPORTS_HTML, current="reports"),
        media_type="text/html",
    )


class PatchReportIn(BaseModel):
    model_config = {"extra": "forbid"}
    status: Literal["open", "resolved", "dismissed"] | None = None
    admin_notes: str | None = Field(default=None, max_length=8000)


@router.get(
    "/reports/api/list",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("reports")),
    ],
)
async def list_reports_api(
    user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    """Scope-aware report list. `scope=own` users see only their own
    reports; `scope=all` users (incl. admins) see every report. Closes
    the previous "list_reports returns ALL rows" leak the moment non-
    admins can reach the page."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    effective_user = perms.effective_user_id_for("reports", caller_uid)
    def _render() -> str:
        limit = reports_store.LIST_LIMIT
        # One row past the ceiling tells "older rows exist" apart from "the
        # scope holds exactly LIST_LIMIT" (a full store sits at REPORTS_MAX,
        # which defaults to LIST_LIMIT, forever).
        rows = reports_store.list_reports(user_id=effective_user,
                                          limit=limit + 1)
        truncated = len(rows) > limit
        rows = rows[:limit]
        usernames = api_keys_store.get_usernames(
            [r.get("user_id") for r in rows],
        )
        for r in rows:
            r["username"] = usernames.get(r.get("user_id"))
        return json.dumps({
            "reports": rows,
            # counts are UNCAPPED (the toolbar states scope totals) while
            # `reports` stops at LIST_LIMIT; `truncated` tells the page that
            # older rows exist past the ceiling.
            "truncated": truncated,
            "counts": reports_store.counts_by_status(user_id=effective_user),
            "retention_days": int(getattr(cfg, "REPORTS_RETENTION_DAYS", 0)),
            "is_admin": bool(user.get("is_admin")),
            "scope": perms.scope("reports"),
        }, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    # The SERIALIZATION has to happen in the thread too, not just the query:
    # at the store's 1000-row LIMIT with max-size rows this measured 2.17 s of
    # frozen event loop, of which the DB read plus per-row json.loads was only
    # 0.36 s — the remaining 1.8 s was JSONResponse rendering the body. Rows
    # are plain types out of _row_to_dict, and the dumps kwargs mirror
    # Starlette's JSONResponse.render (allow_nan=False in particular: a
    # non-finite value must raise here, not emit bare `NaN` that breaks the
    # browser's response.json(); _row_to_dict already maps a legacy
    # non-finite trace_ts to created_ts). Matches captures_routes.
    body = await asyncio.to_thread(_render)
    return Response(content=body, media_type="application/json")


@router.patch(
    "/reports/api/{rid}",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("reports")),
    ],
)
async def patch_report_api(
    rid: str, payload: PatchReportIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Mark a report status/notes. `scope=own` users can edit only their
    own; `scope=all` users (incl. admins) can edit any."""
    # Store calls off the loop, like submit/clear: they wait on
    # reports_store._lock, which a submit's eviction or the retention sweep
    # holds for a full-table pass — inline, that wait froze the event loop.
    existing = await asyncio.to_thread(reports_store.get_report, rid)
    if existing is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "report not found")
    user["permissions"].assert_can_read_row(
        existing, "reports", user.get("user_id") or "",
        detail="report not found",
    )
    patch: dict[str, Any] = {}
    if payload.status is not None:
        patch["status"] = payload.status
    if payload.admin_notes is not None:
        patch["admin_notes"] = payload.admin_notes
    try:
        updated = await asyncio.to_thread(reports_store.update_report, rid, patch)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "report not found")
    return JSONResponse({"ok": True, "report": updated})


@router.delete(
    "/quick-config/reports/api/by-request/{request_id}",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def delete_my_report_api(
    request_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Delete the caller's report for a given request_id. One report
    per (user_id, request_id) is enforced by upsert_report, so this
    targets exactly the caller's row. Does NOT touch capture chips —
    reports and captures are independent stores."""
    # find_by_request_user returns None when user_id is falsy (e.g. an
    # unauthenticated caller); the 404 path below covers both that and
    # an authenticated caller whose user_id has no matching report row.
    # In open mode user_id is the literal "(open-mode)" sentinel — a
    # real value — so the query runs and matches the admin's own row.
    # Off the loop (see patch_report_api): both wait on reports_store._lock.
    existing = await asyncio.to_thread(
        reports_store.find_by_request_user, request_id, user.get("user_id"),
    )
    if not existing:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no report to delete",
        )
    await asyncio.to_thread(reports_store.delete_report, existing.get("id"))
    return JSONResponse({"ok": True})


@router.delete(
    "/reports/api/{rid}",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("reports")),
    ],
)
async def delete_report_api(
    rid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Delete a single report. `scope=own` users can delete only their
    own; `scope=all` users (incl. admins) can delete any. Bulk wipe is
    via /clear which stays admin-only. Does NOT touch capture chips."""
    # Off the loop (see patch_report_api): both wait on reports_store._lock.
    report = await asyncio.to_thread(reports_store.get_report, rid)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "report not found")
    user["permissions"].assert_can_read_row(
        report, "reports", user.get("user_id") or "",
        detail="report not found",
    )
    await asyncio.to_thread(reports_store.delete_report, rid)
    return JSONResponse({"ok": True})


@router.post(
    "/reports/api/clear",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_admin),
    ],
)
async def clear_reports_api(request: Request) -> JSONResponse:
    host = request.client.host if request.client else ""
    # clear_all does DELETE FROM reports + a full VACUUM — 0.416 s of frozen
    # loop at the default REPORTS_MAX=1000 worst case, and REPORTS_MAX is
    # settable to 100_000. Offloaded like the captures twin.
    n = await asyncio.to_thread(reports_store.clear_all, reporter_host=host)
    return JSONResponse({"ok": True, "deleted": n})


@router.get(
    "/reports/api/export",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_admin),
    ],
)
async def export_reports_api() -> Response:
    def _render() -> str:
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "app": "faster-whisper-backend",
            # limit=None: this is the documented FULL dump — the browser
            # list's read ceiling must not truncate a backup when
            # REPORTS_MAX is raised above _LIST_LIMIT.
            "reports": reports_store.list_reports(limit=None),
        }
        # allow_nan=False for the same reason as list_reports_api: a
        # non-finite value must fail loudly here, not ship a "backup"
        # carrying bare Infinity/NaN that no strict parser reads.
        return json.dumps(payload, ensure_ascii=False, indent=2,
                          allow_nan=False)

    # Same shape as list_reports_api: query AND serialize off the loop.
    blob = await asyncio.to_thread(_render)
    fname = f"whisper-reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    return Response(
        content=blob,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ---------------------------------------------------------------------
# /reports HTML page
# ---------------------------------------------------------------------
# Card list with status/model filter + free-text search. Word-correction
# chips render the wrong word red-strike + the correct word green.
# Sentence rewrite computes a word-level LCS diff between `final` and
# `intended_text` and renders deletions / insertions inline.
#
# IMPORTANT (CLAUDE memory note): never place a `{{...}}` placeholder
# inside a /* */, //, or <!-- --> comment — render_page() does a literal
# string replace and corrupting context kills the page. The placeholders
# below are all at HTML-element scope or inside <style>/<script> as bare
# tokens, never inside a comment.

_REPORTS_HTML = templates.load(__file__, "reports.html")
