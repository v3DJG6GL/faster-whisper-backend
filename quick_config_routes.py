"""
End-user simple-config WebUI for faster-whisper-backend.

Mounted at /quick-config. Endpoints:

  GET  /quick-config                      HTML page (loopback / USER_WEBUI_ALLOWED_HOSTS)
  GET  /quick-config/state                Returns ONLY the rules an admin marked exposed
  POST /quick-config/state                Patch enabled / body fields on exposed rules
  POST /quick-config/reapply-rules        Kick off bulk-reapply job over existing captures
  GET  /quick-config/reapply-rules/status Poll bulk-reapply job state
  GET  /quick-config/recent               Snapshot of the recent-traces ring buffer
  GET  /quick-config/stream               SSE stream of recent traces (live updates)

Security model:
  1. IP gate:           require_user_webui_host (loopback always permitted)
  2. API key:           Depends(get_current_user) — bearer must resolve to
                        an active key. Admin = is_admin=True.
  3. Rule allow-list:   POST enforces `exposed == True` AND a per-type
                        field allow-list, regardless of caller role. Defends
                        against `{"locked": false}`-style bypass attempts —
                        the filter runs BEFORE the merge into PIPELINE_RULES.

The recent-traces ring buffer holds literal dictation snippets, which
can be sensitive. RAM-only, capped, lost on restart. Don't log
buffer contents and don't persist them.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

import config as cfg
import config_store
import store_common
import quick_config_state
import transcriptions_store
import web_common
from admin_routes import (
    _apply_hot_changes,
    _canon_rules,
)
from web_common import require_user_webui_host
from auth import (
    Permissions, get_current_user, open_mode_host_ok, require_page,
    user_from_session_cookie,
)

logger = logging.getLogger("whisper-api")

router = APIRouter(prefix="/quick-config")


# Per-type allow-list of fields an end-user is allowed to patch on an
# already-existing exposed rule. Anything else in a patch dict triggers a
# 400. `name`, `label`, `type`, `exposed`, `locked`, `seeded` are NEVER
# editable from /quick-config (admin-only). Adding a new rule, deleting a
# rule, or reordering is also admin-only.

# Dedicated, deliberately tiny pool for the one blocking call that can hold a
# thread for seconds: config_store.save_overrides runs the ReDoS guard, which
# forks a `sys.executable` child and waits up to _GUARD_TIMEOUT (2 s) per
# catastrophic pattern. Running that on asyncio's DEFAULT executor (which
# to_thread uses) lets a caller with only the `quick_config` page scope — no
# admin, no host gate, no rate limit — occupy every one of its
# min(32, cpu_count+4) threads and add seconds of scheduling latency to every
# unrelated to_thread call in the app. Bounding it at 2 confines the damage to
# this endpoint. Legitimate saves take ~25 ms, so queueing at 2 is not
# observable. Created once at import; never per-request (a per-request executor
# would leak threads and defeat the bound).
_GUARDED_SAVE_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="regex-guard",
)

# Ingress cap for a callback:map patch, read off the schema so the two can
# never drift. The same bound is enforced by Pydantic inside save_overrides,
# but only AFTER the stamping loop below has already walked the caller's dict
# twice on the event loop.
_MAP_MAX_ENTRIES: int = next(
    (m.max_length
     for m in config_store.MapRule.model_fields["map"].metadata
     if getattr(m, "max_length", None) is not None),
    500,
)


_PATCH_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "regex-list":                  frozenset({"enabled", "entries"}),
    "callback:map":                frozenset({"enabled", "map"}),
    "callback:lowercase-wordlist": frozenset({"enabled", "pattern", "wordlist"}),
    "callback:dedup":              frozenset({"enabled", "pattern"}),
    "callback:upper":              frozenset({"enabled", "pattern"}),
}


# SSE endpoint compatibility: EventSource has no way to attach an
# Authorization header, so the /stream endpoint rides the session cookie
# the browser sends automatically on a same-origin stream.

def require_user_or_admin_sse(request: Request) -> dict[str, Any]:
    """SSE-aware variant of get_current_user + require_page("quick_config").
    Resolves auth from (in order) the bearer header and the HttpOnly session
    cookie (EventSource sends it automatically). Attaches the Permissions
    policy object and rejects callers whose quick_config scope is "none".

    In OPEN mode the synthetic admin is confined to the admin host
    allowlist (auth.open_mode_host_ok), like every other gate."""
    import api_keys_store
    if not api_keys_store.is_locked_down() and open_mode_host_ok(request):
        rec = dict(api_keys_store.OPEN_MODE_USER)
    else:
        auth_header = request.headers.get("authorization") or ""
        raw = ""
        if auth_header.lower().startswith("bearer "):
            raw = auth_header.split(" ", 1)[1].strip()
        rec = api_keys_store.lookup_by_raw_key(raw) if raw else None
        if rec is None:
            rec = user_from_session_cookie(request)
        if rec is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    rec["permissions"] = Permissions(
        rec.get("permissions_raw") or {}, bool(rec.get("is_admin")),
    )
    if not rec["permissions"].can("quick_config"):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "no access to /quick-config",
        )
    return rec


class QuickPatchPayload(BaseModel):
    """POST body shape:
        {"rules_patch": {slug: {field: value, ...}, ...},
         "fingerprints": {slug: <hex>, ...}}      # optional, recommended

    `fingerprints` carries the per-rule hash each slug had when the
    client loaded /state. Server compares against the current rule's
    fingerprint and reports a conflict for any mismatch. Without it,
    last-writer-wins (legacy behavior; clients are expected to send it
    now). See _rule_fingerprint()."""
    model_config = {"extra": "forbid"}
    rules_patch: dict[str, dict[str, Any]]
    fingerprints: dict[str, str] | None = None


def _rule_fingerprint(rule: dict[str, Any]) -> str:
    """Stable short hash of a rule dict for optimistic concurrency
    control (HTTP-ETag style). Order-insensitive on dict fields. Uses
    sha1 because it's fast and we don't need cryptographic strength —
    the worst case of a collision is "two different rule states hash
    the same" which is statistically irrelevant at our buffer cap. Cut
    to 12 hex chars (~48 bits) to keep the wire payload small."""
    canonical = json.dumps(
        rule, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Shared view/edit helpers — single home for the pipeline-rule policy, reused
# by the /quick-config WebUI routes below AND the /v1/pipeline-rules client API
# (v1_router, further down). Keeping these in one place means the desktop
# client and the browser behave identically.
# ---------------------------------------------------------------------------

def build_visible_rules(user: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """The pipeline rules `user` may see (canonical + fingerprinted) and their
    role. The visibility policy (Permissions.can_see_rule: rule must be
    `exposed`, then admin OR untagged rule OR rule.tags ∩ caller.tags) lives in
    exactly one place. The terminal sentinel is always excluded. Each returned
    rule carries a `_fp` (computed AFTER _canon_rules, so client + server hash
    the same canonical bytes) for optimistic-concurrency on the patch path."""
    perms = user["permissions"]
    canonical = [
        r for r in _canon_rules(list(cfg.PIPELINE_RULES))
        if isinstance(r, dict)
        and r.get("type") != "terminal"
        and perms.can_see_rule(r)
    ]
    for rd in canonical:
        rd["_fp"] = _rule_fingerprint(rd)
    role = "admin" if user.get("is_admin") else "user"
    return canonical, role


def editable_fields_map() -> dict[str, list[str]]:
    """The per-rule-type field allow-list as JSON-serialisable sorted lists, so
    a client can render exactly the editable fields without hardcoding the
    policy. Mirrors _PATCH_ALLOWED_FIELDS (the server-side enforcement)."""
    return {rtype: sorted(fields) for rtype, fields in _PATCH_ALLOWED_FIELDS.items()}


def build_word_suggestions(user: dict[str, Any], *, max_words: int) -> list[str]:
    """Recently-transcribed word + phrase suggestions for `user`, scoped exactly
    like GET /quick-config/recent (own rows unless quick_config scope == "all";
    admins / open-mode see everything). Aggregates each row's tokens ∪ bigrams
    newest-first with a case-insensitive dedup (first-seen = newest casing wins),
    capped at `max_words`. This is the single-source-of-truth Python port of the
    web page's rebuildDatalist() JS, reused by GET /v1/recent-words so the
    desktop Dictionary and the browser autocomplete agree. max_words <= 0 → [].
    Resilient: any store error (e.g. store not yet initialised) → []."""
    if max_words <= 0:
        return []
    perms = user["permissions"]
    sees_all = perms.scope("quick_config") == "all"
    user_filter = None if sees_all else (user.get("user_id") or "")
    scan = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
    try:
        rows = transcriptions_store.list_recent(limit=scan, user_id_filter=user_filter)
    except Exception:
        return []
    seen: dict[str, str] = {}
    for row in rows:
        for word in list(row.get("tokens") or []) + list(row.get("bigrams") or []):
            ws = str(word)
            key = ws.lower()
            if key and key not in seen:
                seen[key] = ws
                if len(seen) >= max_words:
                    return list(seen.values())
    return list(seen.values())


def _redact_invisible_slugs(
    errors: list[dict[str, str]],
    user: dict[str, Any],
    rules: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Blank out rule slugs the caller is not allowed to see.

    save_overrides re-validates the ENTIRE merged rule list, and config_store
    builds its messages as `rule {idx} ({slug!r}) ...` regardless of who asked.
    Rule visibility is a real authorization boundary — the GET path filters on
    `can_see_rule` and the PATCH path 403s an invisible slug — so returning
    those messages verbatim to a non-admin leaks the names (and list positions)
    of rules they cannot otherwise learn exist. Two ways that fires: a rule that
    compiles but fails the guard (the load path validates without `guard_regex`,
    so it can sit on disk), or the shared guard budget expiring while the child
    is on a later hidden rule.

    The caller's OWN rules are always visible to them — the PATCH path rejected
    anything else before we got here — so their real errors stay legible.
    Admins see everything unchanged.
    """
    if user.get("is_admin"):
        return errors
    perms = user.get("permissions")
    if perms is None:
        return errors
    # The slug lives under "name" on a rule dict (see by_slug below); _RuleBase
    # forbids extras, so no rule ever carries a "slug" key and keying on one
    # made this a silent no-op.
    hidden = [
        str(r.get("name") or "") for r in rules
        if r.get("name") and not perms.can_see_rule(r)
    ]
    if not hidden:
        return errors
    # format_validation_errors returns {"loc": ..., "msg": ...} dicts, not
    # bare strings — redact each field rather than the mapping.
    out: list[dict[str, str]] = []
    for entry in errors:
        red = dict(entry)
        for key in ("loc", "msg"):
            val = str(red.get(key) or "")
            for slug in hidden:
                # config_store formats the slug with !r, hence the quotes.
                val = val.replace(f"'{slug}'", "'<hidden rule>'")
            red[key] = val
        out.append(red)
    return out


async def apply_rules_patch(
    user: dict[str, Any],
    rules_patch: dict[str, dict[str, Any]],
    fingerprints: dict[str, str] | None = None,
    *,
    client_host: str = "?",
) -> tuple[int, dict[str, Any]]:
    """Validate + apply a per-rule patch to PIPELINE_RULES. Shared by
    POST /quick-config/state and PATCH /v1/pipeline-rules so the edit policy
    lives in one home: per-type field allow-list, the same can_see_rule
    re-check the GET uses, terminal + locked guards, optimistic-concurrency by
    fingerprint, server-owned map_meta stamping, then the full Pydantic
    re-validation (incl. the 2 s ReDoS guard) via save_overrides.

    Returns (status_code, body): 200 on success or conflict-only; 422
    {"errors": [...]} when save validation fails (an error may name a rule the
    user didn't touch — the admin's pipeline is invalid). Raises HTTPException
    for 400 (malformed patch / unknown slug / disallowed field), 403 (rule not
    visible to this user / terminal rule / rule locked by an admin) or 500
    (config write failure)."""
    fingerprints = fingerprints or {}
    if not rules_patch:
        return 200, {
            "saved": [], "conflicts": [],
            "hot_applied": [], "cold_pending": [],
            "env_pinned_ignored": [], "evicted": [],
            "requires_restart": False,
        }

    # Snapshot the current PIPELINE_RULES as plain dicts so we can overlay
    # patches deterministically; the merged list replaces cfg.PIPELINE_RULES
    # verbatim (count + order preserved → terminal rule stays last).
    current_rules: list[dict[str, Any]] = []
    for r in cfg.PIPELINE_RULES:
        if hasattr(r, "model_dump"):
            current_rules.append(r.model_dump())
        else:
            current_rules.append(dict(r))
    by_slug = {r.get("name"): i for i, r in enumerate(current_rules)}
    # Canonicalize for fingerprint comparison — must hash the same shape /state
    # served. _canon_rules drops None fields and sorts dict keys.
    canonical_now = {r["name"]: r for r in _canon_rules(current_rules)
                     if isinstance(r, dict) and r.get("name")}

    saved: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for slug, patch in rules_patch.items():
        if not isinstance(patch, dict):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"rules_patch['{slug}'] must be an object",
            )
        # Must pass the same can_see_rule check the GET uses, so a user can't
        # PATCH a rule their tag set forbids them from seeing. The exposed +
        # terminal + admin checks all live inside can_see_rule.
        #
        # `by_slug` is built from the FULL rule list, so answering "no such
        # rule" and "a rule you may not see" differently was an existence
        # oracle: a non-admin could enumerate the slugs the admin curated out
        # of their view, one guess per request, with no rate limit. For a
        # non-admin the two collapse into the same 400 — the doctrine
        # auth.Permissions.assert_can_read_row states in its own comment
        # ("404, not 403 — a 403 would confirm the row exists").
        #
        # Rules that ARE exposed to the caller are unaffected: they resolve
        # normally here and keep their precise errors. Admins keep both.
        idx = by_slug.get(slug)
        visible = (
            idx is not None
            and user["permissions"].can_see_rule(current_rules[idx])
        )
        if not visible and not user.get("is_admin"):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"unknown rule slug: '{slug}' (adding rules is not allowed)",
            )
        if idx is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"unknown rule slug: '{slug}' (adding rules is not allowed)",
            )
        target = current_rules[idx]
        if not visible:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"rule '{slug}' is not visible to your user",
            )
        rtype = target.get("type", "?")
        if rtype == "terminal":
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "the terminal rule cannot be edited",
            )
        # `locked` is the admin's freeze switch on /settings — enforced here,
        # not just greyed out in the editor. Admins can still patch (they own
        # the flag and can lift it on /settings).
        if target.get("locked") and not user.get("is_admin"):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"rule '{slug}' is locked and cannot be edited",
            )
        allowed = _PATCH_ALLOWED_FIELDS.get(rtype, frozenset())
        for field in patch.keys():
            if field not in allowed:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"field '{field}' is not editable on rule '{slug}' "
                    f"(type={rtype})",
                )
        # Optimistic-concurrency: if the client sent a fingerprint, it must
        # match what's on disk now. Mismatch → another writer changed this rule
        # between load and save; skip + report rather than clobber. Patches
        # without a fingerprint fall through to legacy last-writer-wins.
        client_fp = fingerprints.get(slug)
        if client_fp:
            current = canonical_now.get(slug, {})
            current_fp = _rule_fingerprint(current) if current else None
            if current_fp != client_fp:
                conflicts.append({"slug": slug, "current_fp": current_fp})
                continue
        # Server-owned map_meta: stamp added/value-changed cb:map entries with
        # the current epoch, drop meta for removed keys. Done after the conflict
        # check, before the overlay — so the client can't forge timestamps.
        if rtype == "callback:map" and "map" in patch:
            old_map = target.get("map") or {}
            new_map = patch["map"] or {}
            # rules_patch is typed dict[str, dict[str, Any]], so the per-FIELD
            # value is unconstrained and Pydantic only sees it later, inside
            # save_overrides. A non-dict here reached .items() below as an
            # unhandled AttributeError -> bare 500.
            if not isinstance(new_map, dict):
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"rules_patch['{slug}']['map'] must be an object",
                )
            # Bound the dict BEFORE walking it: the stamping loop and the
            # comprehension below both iterate a caller-supplied dict on the
            # event loop, and the only other cap (MapRule.map's max_length) is
            # not reached until save_overrides, which runs after. Anything over
            # the cap was already destined for a 422 there, so only the shape
            # of the rejection changes.
            if len(new_map) > _MAP_MAX_ENTRIES:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"rules_patch['{slug}']['map'] may hold at most "
                    f"{_MAP_MAX_ENTRIES} entries",
                )
            if not isinstance(old_map, dict):
                old_map = {}
            meta = dict(target.get("map_meta") or {})
            now = int(time.time())
            for k, v in new_map.items():
                if k not in old_map or old_map[k] != v:
                    meta[k] = now
            target["map_meta"] = {k: meta[k] for k in new_map if k in meta}
        target.update(patch)
        saved.append(slug)

    # If every patch conflicted, skip the save+rebuild — nothing to write.
    if not saved:
        return 200, {
            "saved": [],
            "conflicts": conflicts,
            "hot_applied": [], "cold_pending": [],
            "env_pinned_ignored": [], "evicted": [],
            "requires_restart": False,
        }

    # Hand the merged list to the same save path the admin uses: full Pydantic
    # re-validation, with the 2 s ReDoS guard scoped (guard_slugs) to the rules
    # THIS patch changed. Unscoped, every user save re-probed every admin rule:
    # one pre-existing rule the current guard refuses — saved before a guard
    # tightening, loading fine ever since — 422'd every save of any rule, and a
    # large rule set could burn the shared guard budget on its own. An error
    # may still name an untouched rule (non-guard validation covers the whole
    # merged list) — the client surfaces this gracefully.
    # Off the event loop. The guard runs the candidate patterns in a child
    # process and waits up to _GUARD_TIMEOUT for it, so calling save_overrides
    # inline freezes the whole worker — every HTTP request, SSE stream and
    # WebSocket on it — for the duration, and this endpoint is reachable by a
    # non-admin with no rate limit. admin_routes' rule dry-run already offloads
    # the same hazard the same way.
    # It rides a dedicated 2-thread pool rather than asyncio's default executor
    # so a flood of guard-tripping saves here cannot starve every other
    # to_thread call in the app — see _GUARDED_SAVE_EXECUTOR.
    try:
        written = await asyncio.get_running_loop().run_in_executor(
            _GUARDED_SAVE_EXECUTOR,
            functools.partial(
                config_store.save_overrides, {"PIPELINE_RULES": current_rules},
                guard_slugs=frozenset(saved),
            ),
        )
    except ValidationError as e:
        errs = config_store.format_validation_errors(e)
        # Unlike the success line below, this branch used to return with NO
        # log at all — a save could fail for every user while the log showed
        # only interleaved successes. Full (unredacted) detail is fine here:
        # the log is admin-eyes-only. log_safe for the same CR/LF-forgery
        # reason as the success line.
        logger.warning(
            "[pipeline-rules] save validation failed from=%s user=%s admin=%s "
            "patched=%s errors=%s",
            client_host, store_common.log_safe(user.get("username") or "?"),
            user.get("is_admin"), saved, store_common.log_safe(str(errs)),
        )
        return status.HTTP_422_UNPROCESSABLE_CONTENT, {
            "errors": _redact_invisible_slugs(errs, user, current_rules),
        }
    except OSError as e:
        logger.error("[pipeline-rules] save failed: %s", e)
        # Generic client detail — the OSError text embeds the absolute
        # config.local.json path, and this patch endpoint is reachable by a
        # non-admin user (exposed-rule edits). Full detail is logged above.
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "could not save configuration",
        )

    applied = await _apply_hot_changes(written)

    logger.info(
        "[pipeline-rules] patch from=%s user=%s admin=%s saved=%s conflicts=%s",
        # Usernames are length-capped but never character-screened, so a bare
        # CR/LF here would forge extra records in the /logs viewer.
        client_host, store_common.log_safe(user.get("username") or "?"),
        user.get("is_admin"),
        saved, [c["slug"] for c in conflicts],
    )

    # Admin-only: captures_store.count() with no argument is the unfiltered
    # "admin / scope=all" query, and a caller holding only the quick_config
    # page has captures scope "none" — the global total is not theirs to see.
    # It is also useless to them: the page fires the silent re-apply job on
    # captures_count > 0, and POST /quick-config/reapply-rules is admin-only,
    # so a non-admin save on a server with any capture ended in a permanent
    # red "Re-apply failed: HTTP 403" strip. Reporting 0 keeps that branch shut
    # and takes the blocking SQLite COUNT(*) off the non-admin path entirely.
    captures_count = 0
    if user.get("is_admin") and getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False):
        try:
            import captures_store
            captures_count = captures_store.count()
        except Exception as _e:
            logger.warning("[pipeline-rules] capture count lookup failed: %s", _e)

    return 200, {
        "saved": saved,
        "conflicts": conflicts,
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
        "captures_count": captures_count,
    }


# ---------------------------------------------------------------------------
# /v1/pipeline-rules — client API (desktop app). NOT host-gated.
# ---------------------------------------------------------------------------
# Same tag/exposed gating + per-type field allow-list + validation as the
# /quick-config WebUI below (shared build_visible_rules / apply_rules_patch),
# but in the /v1 namespace with NO host allowlist — auth is the per-user API
# key (bearer) plus the quick_config page permission. Mounted always-on in
# main.py alongside /v1/me, so the desktop client can manage rules regardless
# of the ADMIN_UI_ENABLED WebUI switch.
# Gate on the router constructor so every present + future sub-route inherits
# the quick_config page check (the auth.require_page convention — "closes the
# 'forgot to gate this endpoint' hole"). Both routes below are user-tier.
v1_router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(require_page("quick_config"))],
)


@v1_router.get("/pipeline-rules")
async def v1_get_pipeline_rules(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The post-processing (pipeline) rules THIS caller may view + edit, with
    the same `exposed` + tag gating as /quick-config. Returns
    `{rules, role, editable_fields}`; each rule carries an `_fp` to echo back on
    PATCH. User-tier auth (any valid key); 403 if the caller's quick_config page
    scope is "none". No host allowlist (unlike the /quick-config WebUI)."""
    rules, role = build_visible_rules(user)
    return {
        "rules": rules,
        "role": role,
        "editable_fields": editable_fields_map(),
        "map_collapse_after": int(getattr(cfg, "QUICK_CONFIG_MAP_COLLAPSE_AFTER", 15)),
        "map_max_entries": _MAP_MAX_ENTRIES,
    }


@v1_router.patch("/pipeline-rules")
async def v1_patch_pipeline_rules(
    payload: QuickPatchPayload,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Apply a per-rule patch — `enabled` + the per-type body fields only
    (regex-list→entries, callback:map→map, lowercase-wordlist→pattern+wordlist,
    dedup/upper→pattern). name/label/type/tags/colour/exposed/locked are never
    editable here, and rules can't be added/removed/reordered (admin-only, on
    the /settings WebUI). Body `{rules_patch:{slug:{field:val}}, fingerprints:
    {slug:fp}}`. 200 `{saved, conflicts, requires_restart, …}`; 422
    `{errors:[…]}` on validation; 400/403 on malformed/forbidden patches.
    Identical semantics to POST /quick-config/state (shared apply_rules_patch)."""
    client_host = request.client.host if request.client else "?"
    code, body = await apply_rules_patch(
        user, payload.rules_patch, payload.fingerprints, client_host=client_host,
    )
    return JSONResponse(body, status_code=code)


@v1_router.get("/recent-words")
async def v1_get_recent_words(
    user: dict[str, Any] = Depends(get_current_user),
    limit: int | None = None,
) -> dict[str, Any]:
    """Recently-transcribed word + phrase suggestions for filling a spoken-symbol
    (callback:map) key in the desktop Dictionary — the /v1 analogue of the
    /quick-config autocomplete datalist. Scoped to the caller (own rows unless
    their quick_config scope is "all"). Returns `{words, max}` where `max` is the
    server cap (QUICK_CONFIG_WORD_SUGGESTIONS_MAX; 0 = disabled). Optional
    `?limit=` lets the client request fewer (clamped to the cap). User-tier; 403
    if the caller's quick_config scope is "none". No host allowlist (unlike the
    host-gated /quick-config/recent)."""
    cap = int(getattr(cfg, "QUICK_CONFIG_WORD_SUGGESTIONS_MAX", 200))
    max_words = cap
    if limit is not None:
        try:
            max_words = max(0, min(cap, int(limit)))
        except (TypeError, ValueError):
            max_words = cap
    words = await asyncio.to_thread(
        functools.partial(build_word_suggestions, user, max_words=max_words))
    return {"words": words, "max": cap}


@v1_router.get("/usage")
async def v1_get_my_usage(
    tz_midnight: float | None = None,
    days: int = 30,
    bucket: str = "day",
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The caller's OWN transcription usage for the desktop app's Home stats +
    chip readout: today + lifetime totals, PLUS a self-scoped daily/weekly
    SERIES for a trend chart — the /v1 analogue of the host-gated
    /quick-config 'usage' banner. Lives in the /v1 namespace with NO host
    allowlist, so a remote desktop client isn't 403'd by USER_WEBUI_ALLOWED_HOSTS
    (unlike the browser /quick-config/usage). User-tier auth (any valid key);
    403 if the caller's quick_config page scope is 'none'.

    STRICTLY self-scoped: reads only the authenticated user's user_id, so no
    admin scope is needed and no other user's numbers are ever returned — even
    an admin sees only their own here (the global view lives on /stats).
    Best-effort: any failure yields zeros + an empty series so the client just
    hides the widget instead of erroring.

    'today' resets at the VIEWER's local midnight: the client passes
    `tz_midnight` (epoch seconds of its local 00:00); absent/invalid falls back
    to the server's local day. `days` bounds the trend window (clamped 1..366;
    <= 0 = lifetime/all). The series buckets into SERVER-local days like the
    admin /stats chart; `bucket` is 'day' or 'week'."""
    zero = {"requests": 0, "errors": 0, "words": 0, "audio_s": 0.0}
    uid = user.get("user_id") or ""
    bucket = "week" if str(bucket or "").lower() == "week" else "day"
    try:
        eff_days = int(days)
    except (TypeError, ValueError):
        eff_days = 30
    today = dict(zero)
    total = dict(zero)
    series: list[dict[str, Any]] = []
    if uid:
        try:
            import usage_store
            if tz_midnight and tz_midnight > 0:
                today_start = usage_store.hour_for_ts(float(tz_midnight))
            else:
                today_start = usage_store.local_day_start_hour()
            total = usage_store.totals_for_user(uid)
            today = usage_store.totals_for_user(uid, start_hour=today_start)
            series_start: int | None = None
            if eff_days > 0:
                eff_days = min(eff_days, 366)
                series_start = usage_store.local_day_start_hour(days_ago=eff_days - 1)
            series = usage_store.series(
                start_hour=series_start, user_id=uid, bucket=bucket,
            )
        except Exception:
            today, total, series = dict(zero), dict(zero), []
    return {
        "username": user.get("username") or "",
        "today": today,
        "total": total,
        "range": {"days": max(0, eff_days), "bucket": bucket},
        "series": series,
    }


@router.get(
    "",
    # HTML page is host-only — the login modal runs in this page's
    # own JS, so the bearer isn't available on the initial navigation.
    # The per-page permission check lives on each API route below; if
    # the user lacks /quick-config access, the page's first state fetch
    # 403s and the JS renders a "no access" landing.
    dependencies=[Depends(require_user_webui_host)],
)
async def get_quick_config_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_QUICK_CONFIG_HTML, current="quick-config"),
        media_type="text/html",
    )


@router.get(
    "/state",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_state(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return rules the caller can see on /quick-config.

    Visibility is a triple-AND: (a) the rule must not be the terminal
    sentinel, (b) it must be `exposed=True`, (c) the caller's
    `quick_config_tags` must intersect `rule.tags` (or `rule.tags` must
    be empty = visible-to-all). Admins bypass (b)+(c).

    All three checks live behind `perms.can_see_rule()` (auth.py) so the
    policy lives in exactly one place — same pattern as the existing
    page-permission gates."""
    # Visibility + fingerprinting lives in build_visible_rules (shared with
    # GET /v1/pipeline-rules) so the policy has exactly one home.
    canonical, role = build_visible_rules(user)
    # Server-authoritative "this transcription has been reported" set
    # + the user's previously-submitted chip corrections per request_id.
    # The /quick-config page reads both: badges sync from the id list;
    # the chip map seeds form._corrections so re-opening a reported
    # trace shows what was submitted, instead of an empty form. Capped
    # at 100 newest per user. Failure (init_db never ran, etc.) is
    # non-fatal — the client falls back to its localStorage hint and
    # an empty chip map.
    reported_chips: dict[str, dict[str, Any]] = {}
    try:
        import reports_store
        uid = user.get("user_id") or ""
        if uid:
            # Off the loop like every sibling read in this file (_recent_page,
            # the SSE replay, /v1/recent-words): this is a SELECT * returning
            # full transcript rows plus a json.loads per row, and /state is the
            # page's auth probe — re-issued on every save and every
            # optimistic-concurrency conflict.
            my_reports = await asyncio.to_thread(
                reports_store.recent_reports_for_user, uid, limit=100,
            )
            for rep in my_reports:
                rid = rep.get("request_id")
                if not rid:
                    continue
                # Newest-first iteration above; first occurrence wins on
                # duplicate request_id (the upsert keeps a single row
                # per (request_id, user_id), so duplicates are rare).
                if rid not in reported_chips:
                    reported_chips[rid] = {
                        "corrections": rep.get("corrections") or [],
                    }
        # No fallback when uid is empty: reports are per-user; open-mode
        # callers see no reported-badge state at all.
    except Exception:
        pass
    return {
        "rules": canonical,
        "role": role,
        "reported_chips": reported_chips,
        "map_collapse_after": int(getattr(cfg, "QUICK_CONFIG_MAP_COLLAPSE_AFTER", 15)),
        "word_suggestions_max": int(getattr(cfg, "QUICK_CONFIG_WORD_SUGGESTIONS_MAX", 200)),
        # Schema cap on a callback:map's entry count — the page shows a
        # "n / cap" readout so a full dictionary is visible BEFORE a save
        # bounces off the Pydantic max_length.
        "map_max_entries": _MAP_MAX_ENTRIES,
    }


@router.get(
    "/usage",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_my_usage(
    tz_midnight: float | None = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The caller's OWN transcription usage (today + lifetime) for the
    personal banner in the /quick-config subbar. Self-scoped: reads only the
    authenticated user's user_id, so no admin scope is needed and no other
    user's numbers are ever returned. Best-effort — any failure yields zeros
    so the page never breaks.

    'today' resets at the VIEWER's local midnight: the browser passes
    `tz_midnight` (epoch seconds of its local 00:00), and we sum the UTC hours
    since then. When the param is absent/invalid we fall back to the server's
    local day."""
    zero = {"requests": 0, "errors": 0, "words": 0, "audio_s": 0.0}
    uid = user.get("user_id") or ""
    today = dict(zero)
    total = dict(zero)
    if uid:
        try:
            import usage_store
            if tz_midnight and tz_midnight > 0:
                start_hour = usage_store.hour_for_ts(float(tz_midnight))
            else:
                start_hour = usage_store.local_day_start_hour()
            total = usage_store.totals_for_user(uid)
            today = usage_store.totals_for_user(uid, start_hour=start_hour)
        except Exception:
            today, total = dict(zero), dict(zero)
    return {
        "username": user.get("username") or "",
        "today": today,
        "total": total,
    }


@router.post(
    "/state",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def post_state(
    payload: QuickPatchPayload,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Apply a per-rule patch. Rejects:
       - patches against rules that aren't currently exposed
       - patches against terminal rules
       - non-admin patches against rules the admin marked `locked`
       - any field outside the per-type allow-list (including admin-only
         fields like `locked`, `name`, `label`, `exposed`, `seeded`)

    Defense order matters: validate + filter the patch BEFORE merging into
    PIPELINE_RULES, so an end-user can't sneak `locked: false` past the
    Pydantic schema (which accepts `locked` as a real field) by routing it
    through the merge step.
    """
    client_host = request.client.host if request.client else "?"
    code, body = await apply_rules_patch(
        user, payload.rules_patch, payload.fingerprints, client_host=client_host,
    )
    return JSONResponse(body, status_code=code)


# --- Re-apply current pipeline rules to existing captures ------------
#
# Quick-config rules are only baked into a capture's `final` at
# transcription time. After a rule edit, historical captures are
# frozen against the rule set at their time of capture. These two
# endpoints drive a background backfill job that re-runs the current
# pipeline over every capture's raw text, updates `final`, and
# rebuilds affected (unlocked) group transcript snapshots from the
# now-refreshed member text. `corrected_text` and chip corrections
# are admin-authoritative — preserved untouched.

@router.post(
    "/reapply-rules",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def post_reapply_rules(
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    if not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    if not getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False):
        return JSONResponse({"status": "idle", "note": "captures disabled"})
    import captures_reapply
    return JSONResponse(captures_reapply.start())


@router.get(
    "/reapply-rules/status",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_reapply_rules_status(
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    if not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    import captures_reapply
    return JSONResponse(captures_reapply.status())


# --- Recent transcription traces (panel + autocomplete source) -------------
#
# The ring buffer lives in quick_config_state. main.py's transcribe handler
# appends an entry per completed transcription. Both endpoints below are
# token-gated so end-users without a valid token can't enumerate recent
# dictation snippets.

@router.get(
    "/recent",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_recent(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Newest-first slice of the durable recent-transcriptions store.
    The /stream endpoint replays the same slice on connect; /recent
    additionally supports the "Load older" pagination cursor.

    Query params:
      before_ts (float, optional) — cursor: return rows STRICTLY older
        than this created_ts. Omit / 0 to fetch the freshest slice.
      limit (int, optional)       — clamped to cfg.RECENT_TRANSCRIPTIONS_PAGE_SIZE.

    Response:
      {recent: [...], next_before_ts: <float|null>}
        next_before_ts is the oldest entry's created_ts when the slice
        was fully filled (caller can re-query with that as before_ts);
        null when this is the last batch.

    Scope-aware: scope=own users see only their own rows; scope=all
    users (admins) see every row. The username column is materialized
    at write so the read path doesn't hit api_keys_store.

    Free-text search lives on POST /quick-config/recent/search, NOT here:
    the term is by construction words out of a dictation, and a query string
    is copied verbatim into uvicorn's access log, the reverse proxy's access
    log and the browser's history — none of which are the 0600 log file the
    transcript text is otherwise confined to."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    sees_all = perms.scope("quick_config") == "all"

    page_size = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
    try:
        q_before = float(request.query_params.get("before_ts", "") or 0.0)
    except (TypeError, ValueError):
        q_before = 0.0
    try:
        q_limit = int(request.query_params.get("limit", "") or page_size)
    except (TypeError, ValueError):
        q_limit = page_size
    q_limit = max(1, min(q_limit, page_size))

    return await _recent_page(
        before_ts=q_before, limit=q_limit,
        query=None, sees_all=sees_all, caller_uid=caller_uid,
    )


async def _recent_page(
    *,
    before_ts: float,
    limit: int,
    query: str | None,
    sees_all: bool,
    caller_uid: str,
) -> dict[str, Any]:
    """Shared body of GET /recent and POST /recent/search — identical scoping
    and pagination, the two differ only in where the search term travels.

    Off the loop: a search is an unindexable LIKE '%needle%' over two TEXT
    columns capped at 50 000 chars each, so SQLite scans every retained row.
    Measured ~30 ms per search at the shipped RECENT_TRANSCRIPTIONS_MAX of 500,
    unthrottled, and .env.example documents 0 = unbounded."""
    traces = await asyncio.to_thread(
        functools.partial(
            transcriptions_store.list_recent,
            before_ts=before_ts if before_ts > 0 else None,
            limit=limit,
            user_id_filter=None if sees_all else caller_uid,
            query=query or None,
        )
    )
    next_before = traces[-1]["created_ts"] if len(traces) >= limit else None
    return {"recent": traces, "next_before_ts": next_before}


class RecentSearchIn(BaseModel):
    model_config = {"extra": "forbid"}
    q: str = Field(default="", max_length=512)
    before_ts: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    limit: int | None = Field(default=None, ge=1, le=1000)


@router.post(
    "/recent/search",
    dependencies=[
        Depends(require_user_webui_host),
        Depends(require_page("quick_config")),
    ],
)
async def post_recent_search(
    payload: RecentSearchIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Same slice as GET /recent, with the free-text filter supplied in the
    BODY. A POST purely to keep dictation words out of every access log and
    out of browser history — the scoping, bounds and response shape are the
    GET's, unchanged."""
    perms = user["permissions"]
    page_size = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
    limit = max(1, min(payload.limit or page_size, page_size))
    return await _recent_page(
        before_ts=payload.before_ts,
        limit=limit,
        query=(payload.q or "").strip() or None,
        sees_all=perms.scope("quick_config") == "all",
        caller_uid=user.get("user_id") or "",
    )


@router.get(
    "/stream",
    dependencies=[Depends(require_user_webui_host)],
)
async def stream_recent(
    request: Request,
    user: dict[str, Any] = Depends(require_user_or_admin_sse),
) -> StreamingResponse:
    """Server-sent events stream of recent transcriptions.

    On connect, replays the current buffer (`event: trace` for each entry,
    oldest first). After the replay, pushes any new transcription as
    another `event: trace`. Sends a `: keepalive` SSE comment line every
    15 s so reverse proxies don't kill an idle connection.

    Scope-aware: `scope=own` filters replay AND live items to the
    caller's user_id; `scope=all` lets everything through. Live items
    flow through quick_config_state.subscribe()'s shared queue — every
    subscriber gets every event — so filtering happens here per
    subscriber rather than at the publisher (no extra queue infra)."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    sees_all = perms.scope("quick_config") == "all"

    def _visible(entry: dict[str, Any] | None) -> bool:
        if sees_all:
            return True
        # caller_uid is already coerced from None to "" above; coerce the
        # entry side too so a persisted row with user_id=NULL doesn't get
        # silently excluded for a caller whose own user_id is missing.
        return bool(entry) and (entry.get("user_id") or "") == caller_uid

    async def gen():
        q = quick_config_state.subscribe()
        try:
            # Replay the freshest page from the durable store (oldest-
            # first so the client receives them in chronological order,
            # matching the prior in-memory deque iteration semantics).
            page_size = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
            # Off the loop with its siblings — this runs once per SSE connect.
            replay = await asyncio.to_thread(
                functools.partial(
                    transcriptions_store.list_recent,
                    limit=page_size,
                    user_id_filter=None if sees_all else caller_uid,
                )
            )
            for entry in reversed(replay):
                if _visible(entry):
                    yield f"event: trace\ndata: {json.dumps(entry)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                    ev = item.get("event", "trace")
                    payload = item.get("data") or {}
                    if ev == "trace" and not _visible(payload):
                        continue
                    yield f"event: {ev}\ndata: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            quick_config_state.unsubscribe(q)

    return web_common.sse_response(gen())


_QUICK_CONFIG_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
{{SCALE_BOOTSTRAP_HEAD}}
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d; --input-bg: #0d1117;
  }
  html, body { background: var(--bg); color: var(--fg);
    font-family: var(--font-sans); font-size: var(--fs-base); margin: 0; }
  a { color: var(--cyan); }
  /* header / .header-inner / .title / page-toolbar controls (buttons,
     pills, the #status text) are all centralized in NAV_CSS. */
  main { padding: 1rem; max-width: 60rem; margin: 0 auto; }
  /* Personal self-usage banner in the subbar's empty middle. Grows to fill,
     truncates on narrow screens; the subbar's own flex-wrap drops it to its
     own line when space is tight. */
  header .subbar .qc-usage { flex: 1 1 auto; min-width: 0; text-align: center;
    font-size: var(--fs-sm); color: var(--dim); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
  header .subbar .qc-usage:empty { display: none; }
  .qc-usage .u-name { color: var(--bold); font-weight: 600; }
  .qc-usage .u-sep { color: var(--dim); margin: 0 0.5rem; }
  .qc-usage .u-num { font-family: var(--font-mono); }
  .qc-usage .u-today { color: var(--cyan); }
  .qc-usage .u-total { color: var(--fg); }
  .card { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; }
  /* Per-rule colour token mirrored from /settings/pipeline — a 3px coloured
     left border + a faint background wash (admin uses border + rail gradient;
     quick-config cards have no rail, so the wash stands in). */
  .card[data-color="red"]    { border-left: 3px solid rgba(248,81,73,0.7);   background: linear-gradient(90deg, rgba(248,81,73,0.10),   rgba(248,81,73,0.02) 55%),   var(--panel); }
  .card[data-color="amber"]  { border-left: 3px solid rgba(210,153,34,0.75); background: linear-gradient(90deg, rgba(210,153,34,0.10),  rgba(210,153,34,0.02) 55%),  var(--panel); }
  .card[data-color="green"]  { border-left: 3px solid rgba(63,185,80,0.7);   background: linear-gradient(90deg, rgba(63,185,80,0.10),   rgba(63,185,80,0.02) 55%),   var(--panel); }
  .card[data-color="teal"]   { border-left: 3px solid rgba(57,197,207,0.7);  background: linear-gradient(90deg, rgba(57,197,207,0.10),  rgba(57,197,207,0.02) 55%),  var(--panel); }
  .card[data-color="blue"]   { border-left: 3px solid rgba(121,192,255,0.7); background: linear-gradient(90deg, rgba(121,192,255,0.10), rgba(121,192,255,0.02) 55%), var(--panel); }
  .card[data-color="purple"] { border-left: 3px solid rgba(188,140,255,0.7); background: linear-gradient(90deg, rgba(188,140,255,0.10), rgba(188,140,255,0.02) 55%), var(--panel); }
  .card[data-color="pink"]   { border-left: 3px solid rgba(255,123,156,0.7); background: linear-gradient(90deg, rgba(255,123,156,0.10), rgba(255,123,156,0.02) 55%), var(--panel); }
  .card h3 { margin: 0 0 0.25rem 0; font-size: var(--fs-lg); color: var(--bold);
    display: flex; align-items: baseline; gap: 0.5rem; }
  .card .type-pill { display: inline-block; padding: 0 0.375rem;
    border-radius: 3px; font-size: var(--fs-xs); background: #21262d;
    color: var(--cyan); font-weight: normal; }
  /* Admin-locked rule (non-admin view): a yellow lock chip in the heading —
     same pill geometry as the type pill, same yellow the admin pipeline page
     uses for its padlock — plus not-allowed cursors on the frozen controls.
     Contrast is left untouched: locked means protected, not deactivated. */
  .card .type-pill.lock-pill { background: rgba(242, 204, 96, 0.14);
    color: var(--yellow); }
  .card.locked .enabled-row input:disabled,
  .card.locked .rule-editor input:disabled,
  .card.locked .rule-editor textarea:disabled { cursor: not-allowed; }
  .card.locked .rule-editor .rl-grip { cursor: not-allowed; opacity: 0.4; }
  .card .enabled-row { display: flex; align-items: center; gap: 0.4rem;
    margin: 0.4rem 0; font-size: var(--fs-sm); color: var(--dim); }
  .card .enabled-row input { margin: 0; }
  .card .help { color: var(--help); font-size: var(--fs-sm);
    margin-top: 0.375rem; }
  .card .rule-editor { margin-top: 0.4rem; display: flex;
    flex-direction: column; gap: 0.25rem; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  .card .rule-editor input, .card .rule-editor textarea {
    box-sizing: border-box;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.25rem 0.4rem; font: inherit; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  /* Browser-default focus outline draws OUTSIDE the input's box, so when
     the input fills its td (width:100%) the blue ring overflows the
     table cell on the right. Negative outline-offset overlays the
     existing 1px border instead. */
  .card .rule-editor input:focus-visible,
  .card .rule-editor textarea:focus-visible {
    outline: 2px solid var(--cyan); outline-offset: -2px;
  }
  .card .rule-editor textarea { width: 100%; resize: vertical; }
  .card .rule-editor .map-table { width: 100%; }
  .card .rule-editor .map-table input { width: 100%; }
  /* Inline "added / last-updated" date column (last cell). Dim + mono, fixed
     width so the key/value inputs keep their space. */
  .card .rule-editor .map-table td.map-date-cell {
    width: 11rem; padding-left: 0.4rem; text-align: right;
    white-space: nowrap; vertical-align: middle; }
  /* Phone: a fixed 11rem date column eats ~half a 360px screen, crushing the
     key/value inputs — let it shrink to its content and wrap under them. */
  @media (max-width: 40em) {
    .card .rule-editor .map-table td.map-date-cell {
      width: auto; white-space: normal; }
  }
  .card .rule-editor .map-date {
    color: var(--dim); font-size: var(--fs-xs); font-family: var(--font-mono); }
  /* Collapse the oldest entries behind the toggle. _readMap still reads them
     (display:none keeps them in the DOM). */
  .card .rule-editor .map-table:not(.show-all) tr.map-row-collapsed {
    display: none; }
  .card .rule-editor button.map-toggle {
    align-self: flex-start; background: none; border: none; padding: 0.2rem 0;
    color: var(--dim); font-size: var(--fs-sm); cursor: pointer; }
  .card .rule-editor button.map-toggle:hover:not(:disabled) {
    background: none; color: var(--cyan); }
  .card .rule-editor button { background: var(--panel);
    border: 1px solid var(--border); color: var(--fg);
    padding: 0.4rem 0.9rem; border-radius: 3px; cursor: pointer;
    font: inherit; font-size: var(--fs-md); line-height: 1.4; }
  .card .rule-editor button:hover:not(:disabled) {
    background: #21262d; color: var(--bold); }
  .card .rule-editor button:disabled {
    opacity: 0.4; cursor: not-allowed; }
  .card .rule-editor button.primary:not(:disabled) {
    color: var(--green); border-color: var(--green); }
  .card .rule-editor button.primary:hover:not(:disabled) {
    background: rgba(46, 160, 67, 0.12); }
  .empty-state { text-align: center; color: var(--dim);
    padding: 4rem 1rem; font-size: var(--fs-md); }
  .empty-state h2 { color: var(--fg); font-size: var(--fs-lg);
    margin: 0 0 0.5rem 0; }
  #toast { position: fixed; bottom: 1rem; right: 1rem; padding: 0.5rem 1rem;
    background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
    color: var(--fg); font-size: var(--fs-sm); display: none; z-index: 10; }
  #toast.err { border-color: var(--red); color: var(--red); }
  #toast.ok { border-color: var(--green); color: var(--green); }

  /* Recent transcriptions panel */
  #recent-panel { margin-top: 1.5rem; border-top: 1px solid var(--border);
    padding-top: 1rem; }
  .recent-header { display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.5rem; flex-wrap: wrap; }
  .recent-header h2 { font-size: var(--fs-lg); margin: 0; color: var(--bold); }
  .recent-header .recent-label { font-size: var(--fs-xs); color: var(--dim);
    font-style: italic; }
  .recent-header .spacer { flex: 1; }
  /* Free-text search for the recent list — mirrors the /captures subbar
     search. margin-left:auto pushes it to the right edge of the header;
     the magnifier replaces a "search" label to save width. Sized in
     em/rem so it tracks the scale picker. */
  .recent-header .subbar-search { flex: 1 1 auto; min-width: 8rem;
    max-width: 28rem; margin-left: auto; display: inline-flex;
    align-items: center; gap: 0.35rem; }
  .recent-header .subbar-search .search-ico { flex: 0 0 auto;
    width: 0.95em; height: 0.95em; stroke: var(--help); fill: none;
    stroke-width: 2; }
  .recent-header .subbar-search input[type="text"] { flex: 1 1 auto;
    min-width: 4rem; max-width: none; }
  .recent-pager { display: flex; justify-content: center; padding: 0.5rem 0 1rem; }
  .recent-pager button { background: transparent; color: var(--cyan);
    border: 1px solid var(--border); padding: 0.4rem 0.9rem;
    border-radius: 4px; font: inherit; cursor: pointer; }
  .recent-pager button:hover { background: var(--panel); }
  .recent-pager button[disabled] { opacity: 0.5; cursor: default; }
  .empty-recent { color: var(--dim); font-style: italic; padding: 1rem 0;
    font-size: var(--fs-sm); }
  .trace-item { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; }
  .trace-meta { display: flex; gap: 0.5rem; color: var(--dim);
    font-size: var(--fs-xs); margin-bottom: 0.375rem; align-items: center; }
  .trace-meta .pill { background: var(--panel-alt, rgba(255,255,255,0.04));
    border: 1px solid var(--border); border-radius: 999rem;
    padding: 0.05rem 0.5rem; color: var(--fg); font-family: var(--font-sans);
    font-size: var(--fs-xs); }
  /* source chip: live (streaming) vs file (batch upload) */
  .trace-meta .src-chip { text-transform: uppercase; letter-spacing: 0.04em; }
  .trace-meta .src-chip.src-stream { background: rgba(79,140,255,0.18);
    border-color: rgba(79,140,255,0.55); }
  .trace-meta .src-chip.src-file { background: rgba(255,255,255,0.03);
    color: var(--dim); }
  .trace-text { font-family: var(--font-mono); font-size: var(--fs-sm);
    word-wrap: break-word; }
  .trace-raw { color: var(--dim); margin-bottom: 0.25rem; }
  .trace-tag { display: inline-block; min-width: 3rem; color: var(--dim);
    font-family: var(--font-sans); font-size: var(--fs-xs);
    margin-right: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;
    /* Keep double-click word-selection off the tag so selecting the first
       word of the adjacent text doesn't also grab "RAW"/"FINAL". */
    -webkit-user-select: none; user-select: none; }
  .trace-final { color: var(--bold); }
  .trace-final .trace-tag { color: var(--green); }
  details.trace-steps { margin-top: 0.375rem; }
  /* The disclosure is a REAL button: border, hover, rotating chevron chip,
     verb-first label. Still a native <details>/<summary> underneath. */
  details.trace-steps > summary { list-style: none; display: flex;
    align-items: center; gap: 0.5rem; background: var(--input-bg);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 0.3rem 0.6rem; font-size: var(--fs-sm); color: var(--fg);
    cursor: pointer; }
  details.trace-steps > summary:hover { background: #21262d;
    border-color: #8b949e; }
  details.trace-steps > summary::-webkit-details-marker { display: none; }
  details.trace-steps > summary::marker { content: ''; }
  .steps-chev { display: inline-grid; place-items: center; width: 18px;
    height: 18px; border-radius: 4px; background: rgba(126,231,135,0.12);
    color: var(--green); font-size: 11px; flex-shrink: 0;
    transition: transform 0.12s; }
  details[open].trace-steps > summary .steps-chev { transform: rotate(90deg); }
  .steps-cnt { background: rgba(242,204,96,0.12); color: var(--yellow);
    border-radius: 999px; padding: 0 0.5rem; font-size: var(--fs-xs); }
  .steps-cnt.dim { background: rgba(255,255,255,0.04); color: var(--dim); }
  .steps-lbl-hide { display: none; }
  details[open].trace-steps > summary .steps-lbl-show { display: none; }
  details[open].trace-steps > summary .steps-lbl-hide { display: inline; }
  .step-excerpt.steps-teaser { padding-left: 0; margin-top: 0.35rem; }
  .trace-step { padding: 0.25rem 0 0.25rem 0.75rem; font-size: var(--fs-xs);
    border-left: 2px solid var(--border); margin-top: 0.25rem; }
  .trace-step.skipped { opacity: 0.55; }
  .trace-step .step-label { color: var(--dim); display: block;
    font-family: var(--font-sans); margin-bottom: 0.125rem;
    -webkit-user-select: none; user-select: none; }
  .trace-step .step-label .skipped-tag { color: var(--yellow);
    margin-left: 0.25rem; }
  .trace-step .step-before, .trace-step .step-after {
    font-family: var(--font-mono); display: block; word-wrap: break-word; }
  .trace-step .step-before { color: var(--dim); }
  .trace-step .step-after { color: var(--fg); }
  .trace-step .step-arrow { color: var(--green); margin-right: 0.25rem; }

  /* Long-text folding: file transcriptions clamp raw/final at 3 lines with a
     size-honest "Show all" button; steps on long texts render word-diff
     excerpts instead of two full copies. */
  .ws-region.trace-clip { display: -webkit-box; -webkit-box-orient: vertical;
    -webkit-line-clamp: 3; overflow: hidden; }
  .trace-text.has-clip { display: flex; align-items: flex-start; }
  .trace-text.has-clip .trace-tag { flex-shrink: 0; }
  .trace-text.has-clip .ws-region { min-width: 0; }
  .trace-more { display: inline-flex; align-items: center; gap: 0.3rem;
    margin: 0.2rem 0 0 3rem; background: var(--input-bg);
    border: 1px solid var(--border); border-radius: 5px;
    padding: 0.1rem 0.5rem; font-size: var(--fs-xs); color: var(--fg);
    cursor: pointer; font-family: inherit; }
  .trace-more:hover { background: #21262d; color: var(--bold); }
  .step-excerpt { font-family: var(--font-mono); font-size: var(--fs-xs);
    padding: 0.15rem 0 0.15rem 1.5rem; color: var(--dim);
    overflow-wrap: anywhere; }
  .step-excerpt .dx-del { background: rgba(255,123,114,0.15);
    color: var(--red); border-radius: 3px; padding: 0 3px;
    text-decoration: line-through; }
  .step-excerpt .dx-ins { background: rgba(126,231,135,0.15);
    color: var(--green); border-radius: 3px; padding: 0 3px; }
  .step-excerpt .dx-count { color: var(--yellow); }
  .step-excerpt .dx-full { color: var(--dim); text-decoration: underline;
    cursor: pointer; }

  /* Per-trace report row + inline form */
  .trace-actions { display: flex; align-items: center; gap: 0.5rem;
    margin-top: 0.5rem; flex-wrap: wrap; }
  .trace-report-btn { background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.125rem 0.5rem; font: inherit; font-size: var(--fs-xs);
    cursor: pointer; }
  .trace-report-btn:hover { background: #21262d; color: var(--bold); }
  .trace-reported-badge { color: var(--green); font-size: var(--fs-xs);
    border: 1px solid var(--green); border-radius: 3px;
    padding: 0.05rem 0.4rem; font-family: var(--font-sans); }
  .trace-report-form { display: none; margin-top: 0.5rem;
    border-top: 1px dashed var(--border); padding-top: 0.5rem; }
  .trace-report-form.open { display: block; }
  .trace-report-form .form-label { font-size: var(--fs-sm);
    color: var(--fg); margin: 0.5rem 0 0.25rem; display: block;
    font-family: var(--font-sans); }
  .trace-report-form .form-label .help { color: var(--help);
    font-size: var(--fs-xs); font-weight: normal; margin-left: 0.25rem; }
  .rep-final-words { font-family: var(--font-mono); font-size: var(--fs-sm);
    line-height: 1.6; padding: 0.375rem 0.5rem; background: var(--input-bg);
    border: 1px solid var(--border); border-radius: 3px;
    word-wrap: break-word; }
  .rep-final-words .word { cursor: pointer; padding: 0 0.05rem;
    border-radius: 2px; }
  .rep-final-words .word:hover { background: #21262d; }
  .rep-final-words .word.selected { color: var(--red);
    text-decoration: line-through; background: rgba(255, 123, 114, 0.12); }
  /* Inline replacement next to a struck-through word — mirrors the
     /captures word strip and /reports' .diff-ins green look so the
     correction reads as track-changes inline, not just in the chip
     panel below. */
  .rep-final-words .word-replacement {
    color: var(--green); font-weight: 600;
    background: rgba(126, 231, 135, 0.10);
    padding: 0 0.25rem; margin-left: 0.25rem;
    border-radius: 2px;
    font-family: var(--font-mono);
  }
  .rep-corrections { display: flex; flex-wrap: wrap; gap: 0.375rem;
    margin-top: 0.375rem; }
  .rep-corrections:empty { display: none; }
  .rep-correction { display: inline-flex; align-items: center;
    gap: 0.25rem; background: var(--panel); border: 1px solid var(--border);
    border-radius: 3px; padding: 0.125rem 0.4rem; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  .rep-correction .rep-wrong { color: var(--red);
    text-decoration: line-through; }
  .rep-correction .rep-arrow { color: var(--dim); }
  .rep-correction .rep-correct { background: var(--input-bg);
    color: var(--green); border: 1px solid var(--border); border-radius: 3px;
    padding: 0 0.25rem; font: inherit; font-family: var(--font-mono);
    font-size: var(--fs-sm); min-width: 6rem; }
  .rep-correction .rep-correct:focus-visible { outline: 1px solid var(--cyan);
    outline-offset: 0; }
  .rep-correction .rep-correction-remove { background: transparent;
    border: none; color: var(--dim); cursor: pointer;
    font-size: var(--fs-md); padding: 0 0.25rem; line-height: 1; }
  .rep-correction .rep-correction-remove:hover { color: var(--red); }
  .trace-report-form textarea { box-sizing: border-box; width: 100%;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.25rem 0.4rem; font-family: var(--font-sans);
    font-size: var(--fs-sm); resize: vertical; }
  .trace-report-form textarea:focus-visible {
    outline: 2px solid var(--cyan); outline-offset: -2px; }
  .rep-actions { display: flex; gap: 0.5rem; margin-top: 0.5rem;
    justify-content: flex-end; }
  .rep-actions button { background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.125rem 0.625rem; font: inherit; font-size: var(--fs-sm);
    cursor: pointer; }
  .rep-actions button.primary { color: var(--green); border-color: var(--green); }
  .rep-actions button:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Non-modal silent-reapply strip — shown under the header bar after
     Save kicks off the reapply job automatically. Fades out 3s after
     completion. */
  #reapply-strip {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.35rem 0.75rem;
    background: var(--input-bg); border-bottom: 1px solid var(--border);
    font-family: var(--font-sans); font-size: var(--fs-sm);
    color: var(--fg);
    transition: opacity 0.4s ease-out;
  }
  #reapply-strip[hidden] { display: none; }
  #reapply-strip.fade-out { opacity: 0; }
  #reapply-strip.err { background: rgba(220, 80, 80, 0.12);
    border-bottom-color: var(--red, #c84343); }
  #reapply-strip .r-label { color: var(--dim); flex-shrink: 0; }
  #reapply-strip .r-bar {
    display: inline-block; height: 0.5rem; flex: 1; min-width: 4rem;
    background: var(--border); border-radius: 2px; overflow: hidden;
  }
  #reapply-strip .r-fill {
    display: block; height: 100%; width: 0%; background: var(--green);
    transition: width 0.3s ease-out;
  }
  #reapply-strip .r-stats { font-family: var(--font-mono);
    font-size: var(--fs-xs); color: var(--dim); flex-shrink: 0; }

  {{NAV_CSS}}
</style>
</head>
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
    <span class="subbar-title">Quick config</span>
    <div class="qc-usage" id="qc-usage"></div>
    <div class="subbar-right">
      <button id="discard-btn" disabled>discard</button>
      <button id="save-btn" class="primary" disabled>save</button>
      <span id="status">loading…</span>
    </div>
  </div>
</header>

<div id="reapply-strip" hidden>
  <span class="r-label">Re-applying…</span>
  <span class="r-bar"><span class="r-fill"></span></span>
  <span class="r-stats">0 / 0</span>
</div>

<datalist id="recent-words"></datalist>

<main>
  <section id="cards"></section>
  <section id="recent-panel">
    <div class="recent-header">
      <h2>Recent transcriptions</h2>
      <span class="recent-label" title="Persisted in SQLite (WAL). Caps and TTL are configurable at /settings.">persistent · paginated</span>
      <label class="subbar-search" title="search recent transcriptions">
        <svg class="search-ico" viewBox="0 0 24 24" aria-hidden="true" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>
        <input id="recent-search" type="text" aria-label="search recent transcriptions" placeholder="text in raw / final">
      </label>
    </div>
    <div id="recent-list">
      <div class="empty-recent">No transcriptions yet — they'll appear here as you dictate.</div>
    </div>
    <div id="recent-pager" class="recent-pager">
      <button type="button" id="btn-load-older" class="ghost" style="display:none;">Load older</button>
    </div>
  </section>
</main>

<div id="toast"></div>

{{RULE_EDITOR_JS}}

<script>
(() => {
'use strict';

let initialRules = [];      // last-loaded rules from server (deep-copy snapshot)
let liveRules = [];         // editable rules — diffed against initialRules to build patch
let dirty = new Set();      // slugs with changes

async function api(method, path, body) {
  // Session cookie sent automatically; mutations carry the CSRF token.
  const h = { 'Accept': 'application/json' };
  if (method !== 'GET' && method !== 'HEAD') {
    h['X-CSRF-Token'] = window._csrfToken ? window._csrfToken() : '';
  }
  const opts = { method, headers: h };
  if (body !== undefined) {
    h['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  return fetch(path, opts);
}

function showToast(msg, kind) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = kind || '';
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 3500);
}

function setStatus(s) {
  document.getElementById('status').textContent = s;
}

async function ensureToken() {
  // Probe /quick-config/state. On 401 the shared login gate (web_common)
  // prompts for a key and reloads on success. On 403 (valid bearer, no
  // quick_config scope) render the shared no-access landing and keep the
  // bearer so the user can navigate elsewhere without re-logging in.
  let r = await api('GET', '/quick-config/state');
  if (r.status === 401) {
    if (window._showLoginGate) window._showLoginGate();
    return null;
  }
  if (r.status === 403 && typeof _renderNoAccessLanding === 'function') {
    // Refresh whoami so the landing can list reachable pages.
    try {
      const w = await fetch('/auth/whoami');
      if (w.ok) window.__whoami = await w.json();
    } catch (_) {}
    _renderNoAccessLanding({ page: 'quick_config' });
    return null;
  }
  return r;
}

function commitData(slug) {
  dirty.add(slug);
  updateButtons();
}
// Per-card Save buttons, keyed by rule slug. Rebuilt on every renderCards()
// so an entry only ever refers to a button currently in the DOM.
let _perCardSaveBtns = new Map();
// Slugs rendered read-only because the admin locked them (non-admin view).
// Rebuilt alongside _perCardSaveBtns on every renderCards().
let _lockedSlugs = new Set();
// role: "admin" (set on <body> by load()) short-circuits the server-side
// lock guard, so an admin keeps fully editable controls on a locked rule.
function _isAdmin() { return document.body.classList.contains('role-admin'); }
// Enforced read-only for a locked rule: disable every control that could
// mutate it (the enabled switch, the per-type editor's inputs, its
// "+ add entry" / delete buttons and the per-card Save) and kill the
// regex-list drag grips. View-only affordances stay live — the cb:map
// "show N older" toggle and the regex-entry note toggle reveal content,
// they don't edit it, so the rule stays fully readable.
function _applyLockState(card) {
  card.classList.add('locked');
  card.querySelectorAll('input, select, textarea, button').forEach((el) => {
    if (el.classList.contains('map-toggle')) return;
    if (el.classList.contains('rl-notebtn')) return;
    el.disabled = true;
  });
  card.querySelectorAll('.rl-grip').forEach((grip) => {
    grip.draggable = false;
    grip.title = 'locked by an administrator';
  });
}
function _buildPerCardSaveBtn(slug) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'primary';
  btn.textContent = 'save';
  btn.disabled = !dirty.has(slug);
  btn.addEventListener('click', doSave);
  _perCardSaveBtns.set(slug, btn);
  return btn;
}
function updateButtons() {
  const has = dirty.size > 0;
  document.getElementById('save-btn').disabled = !has;
  document.getElementById('discard-btn').disabled = !has;
  setStatus(has ? (dirty.size + ' rule' + (dirty.size === 1 ? '' : 's') + ' modified')
                : (initialRules.length + ' rule' + (initialRules.length === 1 ? '' : 's') + ' loaded'));
  for (const [slug, btn] of _perCardSaveBtns) {
    btn.disabled = _lockedSlugs.has(slug) || !dirty.has(slug);
  }
}

function renderCards() {
  const root = document.getElementById('cards');
  root.innerHTML = '';
  _perCardSaveBtns = new Map();  // detach stale refs; GC takes the old DOM
  _lockedSlugs = new Set();
  if (!liveRules.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.innerHTML = '<h2>Nothing to configure here yet</h2>'
      + '<p>No rules have been exposed by your admin yet.</p>';
    root.appendChild(empty);
    return;
  }
  for (const rule of liveRules) {
    const card = document.createElement('div');
    card.className = 'card';
    // Mirror the per-rule colour token set on /settings/pipeline (admin).
    if (rule.color) card.dataset.color = rule.color;
    const h3 = document.createElement('h3');
    const labelText = document.createTextNode(rule.label || rule.name);
    h3.appendChild(labelText);
    const pill = document.createElement('span');
    pill.className = 'type-pill';
    pill.textContent = _typePill(rule.type);
    h3.appendChild(pill);
    // A rule the admin froze is read-only for everyone but an admin — the
    // server refuses the patch (403), so say so up front rather than let the
    // edit be composed and rejected on Save.
    const isLocked = !!rule.locked && !_isAdmin();
    if (isLocked) {
      const lockPill = document.createElement('span');
      lockPill.className = 'type-pill lock-pill';
      lockPill.textContent = '\u{1F512} locked';
      lockPill.title = 'locked by an administrator — read-only';
      h3.appendChild(lockPill);
    }
    card.appendChild(h3);

    const enRow = document.createElement('label');
    enRow.className = 'enabled-row';
    const enCb = document.createElement('input');
    enCb.type = 'checkbox'; enCb.className = 'switch'; enCb.setAttribute('role', 'switch');
    enCb.checked = !!rule.enabled;
    enCb.addEventListener('change', () => {
      rule.enabled = enCb.checked;
      commitData(rule.name);
    });
    enRow.appendChild(enCb);
    enRow.appendChild(document.createTextNode(' enabled'));
    card.appendChild(enRow);

    const slug = rule.name;
    const editor = renderTypeEditor(rule, () => commitData(slug), {
      datalistId: 'recent-words',
      makeSaveBtn: () => _buildPerCardSaveBtn(slug),
      commitOnEnter: doSave,
      showMapDates: true,
      collapseMapAfter: (window.__mca ?? 15),
    });
    card.appendChild(editor);

    // After the whole card exists, so every per-type widget (regex-list
    // entries, cb:map rows, the cb:wordlist textarea, the bare pattern
    // inputs) is covered by one sweep.
    if (isLocked) {
      _lockedSlugs.add(slug);
      _applyLockState(card);
    }

    root.appendChild(card);
  }
}

// ---------- Recent transcriptions panel + autocomplete -------------------
//
// SSE-driven for the freshest page; durable store backs the "Load older"
// pagination cursor. /quick-config/stream replays the freshest server-
// configured page on connect (oldest-first), then pushes new traces as
// `event: trace`. EventSource auto-reconnects with a 3 s backoff per
// the WHATWG spec — no custom retry logic needed. _oldestLoadedTs tracks
// the bottom edge of the in-DOM list so "Load older" can issue a
// before_ts cursor.
const _MAX_DATALIST = 200;
let _es = null;
let _oldestLoadedTs = null;
let _loadOlderBusy = false;
// request_ids already rendered, so the SSE on-connect replay (which always
// re-sends the freshest page) doesn't duplicate rows already placed by the
// initial /quick-config/recent fetch or by "Load older".
let _seenReqIds = new Set();
// Active free-text filter (server-side substring over raw/final). Empty =
// unfiltered newest-first view.
let _searchQuery = '';
// Active stream-recovery poller (mirrors /stats); non-null while reconnecting.
let _recoveryTimer = null;

function _entryKey(entry) {
  if (!entry) return '';
  if (entry.request_id) return 'r:' + entry.request_id;
  if (entry.id != null) return 'i:' + entry.id;
  return '';
}
// Returns true if this entry was already rendered (and should be skipped);
// otherwise records it as seen and returns false. Entries without any id are
// never deduped (always allowed through).
function _seenTrace(entry) {
  const k = _entryKey(entry);
  if (!k) return false;
  if (_seenReqIds.has(k)) return true;
  _seenReqIds.add(k);
  return false;
}
function _matchesSearch(entry) {
  if (!_searchQuery) return true;
  const q = _searchQuery.toLowerCase();
  return (((entry && entry.raw) || '') + ' ' + ((entry && entry.final) || ''))
    .toLowerCase().indexOf(q) !== -1;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s == null ? '' : String(s);
  // textContent->innerHTML escapes & < > but NOT quotes; escape those too so
  // the result stays safe if it is ever interpolated into an attribute.
  return div.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
// absTime is injected via TIME_HELPERS_JS.

// ── Long-text folding + step diffs (file transcriptions) ────────────────────
// A file transcript used to render in FULL three-plus times per trace (raw,
// final, and once per changed step's before AND after). Fold it instead:
// clamp raw/final with a size-honest "Show all", and render changed steps as
// word-diff excerpts. Live dictation snippets are short and stay as-is.
var TRACE_CLAMP_CHARS = 400;   // raw/final fold threshold
var TRACE_DIFF_CHARS = 300;    // steps switch to excerpt diffs past this
var TRACE_DIFF_SAMPLE = 3;     // excerpts shown per step
var TRACE_DIFF_CONTEXT = 8;    // context words around each change
var TRACE_DIFF_RESYNC = 30;    // resync lookahead (words)

function fmtCount(n) {
  return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, '\u202f');
}

// One raw/final text block; clamps + "Show all · N chars" when long.
function traceTextBlock(tagLabel, text, extraClass) {
  const wrap = document.createElement('div');
  const div = document.createElement('div');
  div.className = 'trace-text ' + extraClass;
  const tag = document.createElement('span');
  tag.className = 'trace-tag';
  tag.textContent = tagLabel;
  div.appendChild(tag);
  const span = document.createElement('span');
  span.className = 'ws-region';
  span.textContent = text;
  div.appendChild(span);
  wrap.appendChild(div);
  if (text.length > TRACE_CLAMP_CHARS) {
    div.classList.add('has-clip');
    span.classList.add('trace-clip');
    const label = 'Show all \u00b7 ' + fmtCount(text.length) + ' chars \u25be';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'trace-more';
    btn.textContent = label;
    btn.addEventListener('click', () => {
      const clipped = span.classList.toggle('trace-clip');
      btn.textContent = clipped ? label : 'Show less \u25b4';
    });
    wrap.appendChild(btn);
  }
  return wrap;
}

// Linear resync word diff — regions are [aStart, aEnd, bStart, bEnd). Good
// for pipeline edits (local changes) and cheap on hour-long transcripts
// (no LCS). A failed resync collapses the rest into one tail region.
function wordDiffRegions(A, B) {
  const regions = [];
  let i = 0, j = 0;
  while (i < A.length && j < B.length) {
    if (A[i] === B[j]) { i++; j++; continue; }
    let found = null;
    for (let d = 1; d <= TRACE_DIFF_RESYNC && !found; d++) {
      for (let di = 0; di <= d && !found; di++) {
        const dj = d - di, ai = i + di, bj = j + dj;
        if (ai < A.length && bj < B.length && A[ai] === B[bj]
            && (ai + 1 >= A.length || bj + 1 >= B.length || A[ai + 1] === B[bj + 1])) {
          found = [di, dj];
        }
      }
    }
    if (!found || regions.length >= 500) {
      regions.push([i, A.length, j, B.length]);
      return regions;
    }
    regions.push([i, i + found[0], j, j + found[1]]);
    i += found[0]; j += found[1];
  }
  if (i < A.length || j < B.length) regions.push([i, A.length, j, B.length]);
  return regions;
}

// "…context DELETED INSERTED context…" for one change region. Built with
// textContent throughout — trace text is untrusted.
function diffExcerpt(A, B, region) {
  const as = region[0], ae = region[1], bs = region[2], be = region[3];
  const el = document.createElement('div');
  el.className = 'step-excerpt';
  const push = (cls, text) => {
    if (!text) return;
    const sp = document.createElement('span');
    if (cls) sp.className = cls;
    sp.textContent = text;
    el.appendChild(sp);
  };
  const pre = A.slice(Math.max(0, as - TRACE_DIFF_CONTEXT), as).join(' ');
  const post = A.slice(ae, ae + TRACE_DIFF_CONTEXT).join(' ');
  push('', (as > TRACE_DIFF_CONTEXT ? '\u2026' : '') + pre + (pre ? ' ' : ''));
  const del = A.slice(as, Math.min(ae, as + 24)).join(' ');
  push('dx-del', del + (ae - as > 24 ? ' \u2026' : ''));
  if (del) push('', ' ');
  const ins = B.slice(bs, Math.min(be, bs + 24)).join(' ');
  push('dx-ins', ins + (be - bs > 24 ? ' \u2026' : ''));
  push('', (post ? ' ' : '') + post + (ae + TRACE_DIFF_CONTEXT < A.length ? '\u2026' : ''));
  return el;
}

function renderTrace(entry) {
  const item = document.createElement('div');
  item.className = 'trace-item';
  // Stash the entry so rebuildDatalist() can read tokens/bigrams without
  // a separate cache. JSON encoding is fine — tokens/bigrams are small.
  try { item.dataset.entry = JSON.stringify(entry); } catch (_) {}

  const meta = document.createElement('div');
  meta.className = 'trace-meta';
  const ts = document.createElement('span');
  ts.className = 'trace-ts';
  const _ets = entry.ts || 0;
  ts.textContent = fmtWhen(_ets);
  if (_ets) { ts.dataset.ts = String(_ets); ts.title = absTime(_ets); }
  meta.appendChild(ts);
  if (entry.model) {
    const mdl = document.createElement('span');
    mdl.textContent = entry.model;
    meta.appendChild(mdl);
  }
  // Source chip: live streaming dictation vs file-upload (batch) transcription.
  {
    const isStream = entry.source === 'stream';
    const src = document.createElement('span');
    src.className = 'pill src-chip ' + (isStream ? 'src-stream' : 'src-file');
    src.title = isStream ? 'live streaming dictation (WebSocket)'
                         : 'file upload (batch /v1/audio/transcriptions)';
    src.textContent = isStream ? 'live' : 'file';
    meta.appendChild(src);
  }
  if (entry.username) {
    const spk = document.createElement('span');
    spk.className = 'pill';
    spk.title = 'speaker';
    spk.textContent = entry.username;
    meta.appendChild(spk);
  } else if (entry.user_id) {
    const spk = document.createElement('span');
    spk.className = 'pill';
    spk.title = 'speaker (unknown user)';
    spk.textContent = String(entry.user_id).slice(0, 6);
    meta.appendChild(spk);
  }
  // Size readout for long texts — the fold never hides how much is folded.
  {
    const t = entry.final || entry.raw || '';
    if (t.length > TRACE_CLAMP_CHARS) {
      const size = document.createElement('span');
      size.className = 'pill';
      size.style.fontFamily = 'var(--font-mono)';
      const words = t.trim() ? t.trim().split(/\s+/).length : 0;
      const secs = Number(entry.audio_dur);
      const dur = Number.isFinite(secs) && secs > 0
        ? (secs >= 3600
            ? Math.floor(secs / 3600) + ' h ' + Math.round((secs % 3600) / 60) + ' min'
            : secs >= 60 ? Math.round(secs / 60) + ' min' : Math.round(secs) + ' s')
        : '';
      size.textContent = (dur ? dur + ' \u00b7 ' : '') + fmtCount(words) + ' words';
      meta.appendChild(size);
    }
  }
  item.appendChild(meta);

  item.appendChild(traceTextBlock('raw', entry.raw || '', 'trace-raw'));

  const steps = entry.steps || [];
  const changed = steps.filter(s =>
    Array.isArray(s) && s.length >= 3 && s[1] !== s[2]);
  if (steps.length) {
    // Open by default only when the content is glanceable: live dictation
    // snippets, or a file trace whose final text is short. Long file traces
    // arrive folded — behind the teaser + button below.
    const startOpen = !!changed.length && (entry.source === 'stream'
        || (entry.final || '').length <= TRACE_CLAMP_CHARS);
    // Teaser: the first diff excerpt renders ABOVE the button even while
    // collapsed, so the feature advertises itself with content, not copy.
    if (!startOpen && changed.length) {
      const first = changed.find(c =>
        (c[1] || '').length > TRACE_DIFF_CHARS
        || (c[2] || '').length > TRACE_DIFF_CHARS) || changed[0];
      const A = (first[1] || '').split(/\s+/), B = (first[2] || '').split(/\s+/);
      const regions = wordDiffRegions(A, B);
      if (regions.length) {
        const teaser = diffExcerpt(A, B, regions[0]);
        teaser.classList.add('steps-teaser');
        item.appendChild(teaser);
      }
    }
    const det = document.createElement('details');
    det.className = 'trace-steps';
    if (startOpen) det.open = true;
    const sum = document.createElement('summary');
    const chev = document.createElement('span');
    chev.className = 'steps-chev';
    chev.textContent = '\u25b8';
    sum.appendChild(chev);
    const show = document.createElement('span');
    show.className = 'steps-lbl-show';
    show.textContent = changed.length ? 'Show all pipeline diffs' : 'Show pipeline steps';
    sum.appendChild(show);
    const hide = document.createElement('span');
    hide.className = 'steps-lbl-hide';
    hide.textContent = changed.length ? 'Hide pipeline diffs' : 'Hide pipeline steps';
    sum.appendChild(hide);
    if (changed.length) {
      const c1 = document.createElement('span');
      c1.className = 'steps-cnt';
      c1.textContent = changed.length + ' changed';
      sum.appendChild(c1);
    }
    if (steps.length > changed.length) {
      const c2 = document.createElement('span');
      c2.className = 'steps-cnt dim';
      c2.textContent = (steps.length - changed.length) + ' unchanged';
      sum.appendChild(c2);
    }
    det.appendChild(sum);
    for (const s of steps) {
      if (!Array.isArray(s) || s.length < 3) continue;
      const [label, before, after] = s;
      const lblHtml = escapeHtml(label || '?');
      const stepEl = document.createElement('div');
      if (before === after) {
        // Unchanged: the label line is the whole story — repeating the full
        // text twice said nothing.
        stepEl.className = 'trace-step skipped';
        stepEl.innerHTML = '<span class="step-label">\u25b8 ' + lblHtml
          + ' <span class="skipped-tag">[unchanged]</span></span>';
        det.appendChild(stepEl);
        continue;
      }
      stepEl.className = 'trace-step';
      const long = (before || '').length > TRACE_DIFF_CHARS
        || (after || '').length > TRACE_DIFF_CHARS;
      if (!long) {
        stepEl.innerHTML =
          '<span class="step-label">\u25b8 ' + lblHtml + '</span>'
          + '<span class="step-before"><span class="ws-region">'
          + escapeHtml(before || '') + '</span></span>'
          + '<span class="step-after"><span class="step-arrow">\u2192</span>'
          + '<span class="ws-region">' + escapeHtml(after || '') + '</span></span>';
        det.appendChild(stepEl);
        continue;
      }
      // Long text: word-diff excerpts instead of two full copies.
      const A = (before || '').split(/\s+/), B = (after || '').split(/\s+/);
      const regions = wordDiffRegions(A, B);
      const head = document.createElement('span');
      head.className = 'step-label';
      head.innerHTML = '\u25b8 ' + lblHtml
        + ' <span class="skipped-tag">changed ' + fmtCount(regions.length)
        + ' spot' + (regions.length === 1 ? '' : 's') + '</span>';
      stepEl.appendChild(head);
      for (const r of regions.slice(0, TRACE_DIFF_SAMPLE)) {
        stepEl.appendChild(diffExcerpt(A, B, r));
      }
      const foot = document.createElement('div');
      foot.className = 'step-excerpt';
      if (regions.length > TRACE_DIFF_SAMPLE) {
        const more = document.createElement('span');
        more.className = 'dx-count';
        more.textContent = '+ ' + fmtCount(regions.length - TRACE_DIFF_SAMPLE) + ' more \u00b7 ';
        foot.appendChild(more);
      }
      const full = document.createElement('span');
      full.className = 'dx-full';
      full.textContent = 'show full before/after';
      full.addEventListener('click', () => {
        const fullEl = document.createElement('div');
        fullEl.innerHTML =
          '<span class="step-before"><span class="ws-region">'
          + escapeHtml(before || '') + '</span></span>'
          + '<span class="step-after"><span class="step-arrow">\u2192</span>'
          + '<span class="ws-region">' + escapeHtml(after || '') + '</span></span>';
        stepEl.appendChild(fullEl);
        foot.remove();
      });
      foot.appendChild(full);
      stepEl.appendChild(foot);
      det.appendChild(stepEl);
    }
    item.appendChild(det);
  }

  item.appendChild(traceTextBlock('final', entry.final || '', 'trace-final'));

  // Action row: "Report" button + (conditional) "✓ reported" badge.
  // The inline form is lazy — built on first click, kept in DOM after
  // submit so subsequent visits see the badge.
  const actions = document.createElement('div');
  actions.className = 'trace-actions';
  const reportBtn = document.createElement('button');
  reportBtn.type = 'button';
  reportBtn.className = 'trace-report-btn';
  reportBtn.textContent = 'Report a problem';
  reportBtn.title = 'For single-word fixes, edit the pipeline rules above. '
                  + 'Use this form for issues larger than a single word.';
  reportBtn.addEventListener('click', () => toggleReportForm(item, entry));
  actions.appendChild(reportBtn);
  const badge = document.createElement('span');
  badge.className = 'trace-reported-badge';
  badge.textContent = '✓ reported';
  badge.style.display = isReported(entry.request_id) ? 'inline-flex' : 'none';
  actions.appendChild(badge);
  item.appendChild(actions);

  return item;
}

// ---------- "Report a problem" inline form -------------------------------
//
// Each trace gets a lazy form: built on the first "Report" click and
// kept in DOM thereafter. The form supports three complementary inputs:
//   (a) per-word corrections — click a word chip in `final` to mark it
//       wrong, then type the replacement. Multiple corrections allowed.
//   (b) "what you meant to say" — free-text rewrite for full rephrases.
//   (c) free-text comment — anything else for the admin.
// At least one of (a-with-non-empty-correct), (b), or (c) must be
// non-empty for the server to accept the submission.

const _REPORTED_KEY = 'whisper-reported-ids';
const _REPORTED_MAX = 200;
// Server-authoritative set, refreshed on every load() of /state.
// Survives across browsers/devices. localStorage is a per-tab UX hint
// that lets a freshly submitted badge show instantly without waiting
// for the next /state poll.
let _serverReportedSet = new Set();
// Per-request_id chip corrections previously submitted by THIS user,
// keyed on request_id. Populated from /state.reported_chips. Lets the
// report form re-render chips on form-open instead of starting empty.
// Filtered server-side to caller's user_id; never contains other
// users' chips.
let _serverReportedChips = {};

function getReportedIds() {
  try { return JSON.parse(localStorage.getItem(_REPORTED_KEY) || '[]'); }
  catch (_) { return []; }
}
function isReported(rid) {
  if (!rid) return false;
  if (_serverReportedSet.has(rid)) return true;
  const ids = getReportedIds();
  return ids.indexOf(rid) !== -1;
}
function markReported(rid) {
  if (!rid) return;
  const ids = getReportedIds();
  if (ids.indexOf(rid) !== -1) return;
  ids.unshift(rid);
  if (ids.length > _REPORTED_MAX) ids.length = _REPORTED_MAX;
  try { localStorage.setItem(_REPORTED_KEY, JSON.stringify(ids)); }
  catch (_) { /* quota — accept that the badge won't survive a reload */ }
}
function unmarkReported(rid) {
  // Called when the user explicitly removes their report. Drops the rid
  // from the localStorage hint AND the in-memory server set so the
  // badge clears immediately, without waiting for the next /state poll.
  if (!rid) return;
  const ids = getReportedIds().filter((x) => x !== rid);
  try { localStorage.setItem(_REPORTED_KEY, JSON.stringify(ids)); }
  catch (_) {}
  _serverReportedSet.delete(rid);
}

function syncReportedBadges() {
  // Called after /state load to flip badges visible/hidden on every
  // trace item currently in the DOM, using the freshly fetched server
  // list. Cheap; the recent panel holds at most 20 items. Also flips
  // the "Remove report" button visibility per open report form.
  document.querySelectorAll('.trace-item').forEach((item) => {
    let rid = '';
    try { rid = (JSON.parse(item.dataset.entry || '{}') || {}).request_id || ''; }
    catch (_) {}
    const reported = isReported(rid);
    const badge = item.querySelector('.trace-reported-badge');
    if (badge) badge.style.display = reported ? 'inline-flex' : 'none';
    const removeBtn = item.querySelector('.rep-remove');
    if (removeBtn) removeBtn.style.display = reported ? 'inline-flex' : 'none';
  });
}

// Tokenization for the clickable-chip layer. We anchor on any word-char
// (`\w` = [A-Za-z0-9_]) plus the Latin-Extended range À-ɏ so numeric
// tokens like "123" and digit-leading mixed tokens ("3D", "5mg") are
// also selectable — matching /captures' "every Whisper token is
// clickable" behaviour. Punctuation runs are emitted as plain-text
// nodes between word spans.
//
// We deliberately diverge from quick_config_state.py:_TOKEN_RE here.
// That server-side regex extracts autocomplete candidates and
// intentionally rejects digit-only tokens (rule candidates shouldn't
// be raw numbers). The clickable-chip layer has a different purpose:
// the user must be able to point at any visible token to correct it.
const _WORD_RE = /[\wÀ-ɏ][\wÀ-ɏ\-]*/g;

function _wordifyFinal(form, finalText) {
  // Tokenize `final` into clickable word spans + track char offsets so a
  // multi-word chip can slice the original string and preserve any
  // inter-word punctuation/whitespace.
  const container = form.querySelector('.rep-final-words');
  container.innerHTML = '';
  form._finalText = finalText || '';
  form._wordPositions = [];   // [{idx, start, end, text}]
  if (!finalText) {
    container.innerHTML = '<span class="empty-recent">(no final text)</span>';
    return;
  }
  let last = 0;
  let wordIdx = 0;
  _WORD_RE.lastIndex = 0;
  let m;
  while ((m = _WORD_RE.exec(finalText)) !== null) {
    if (m.index > last) {
      container.appendChild(
        document.createTextNode(finalText.slice(last, m.index))
      );
    }
    const span = document.createElement('span');
    span.className = 'word';
    span.dataset.idx = String(wordIdx);
    span.textContent = m[0];
    span.addEventListener('click', (e) => {
      toggleCorrection(form, parseInt(span.dataset.idx, 10),
                       span.textContent, !!e.shiftKey);
    });
    container.appendChild(span);
    form._wordPositions.push({
      idx: wordIdx, start: m.index, end: m.index + m[0].length, text: m[0],
    });
    last = m.index + m[0].length;
    wordIdx++;
  }
  if (last < finalText.length) {
    container.appendChild(document.createTextNode(finalText.slice(last)));
  }
}

function _buildReportForm(entry) {
  const form = document.createElement('div');
  form.className = 'trace-report-form';
  form.innerHTML =
    '<div class="help rep-scope-hint">'
    + 'For a single wrong word, prefer editing the pipeline rules above. '
    + 'Use this report when the issue is larger than a single word.'
    + '</div>'
    + '<div class="form-label">Mark wrong words'
    + ' <span class="help">(click a word; shift-click another to extend'
    + ' the range; type the correction; press Enter to add another)</span>'
    + '</div>'
    + '<div class="rep-final-words"></div>'
    + '<div class="rep-corrections"></div>'
    + '<label class="form-label">Comment'
    + ' <span class="help">(optional)</span>'
    + '</label>'
    + '<textarea class="rep-comment" rows="3"'
    + ' placeholder="What went wrong? Anything else the admin should know?"></textarea>'
    + '<div class="rep-actions">'
    + '  <button type="button" class="rep-cancel">Cancel</button>'
    + '  <button type="button" class="rep-remove">Remove report</button>'
    + '  <button type="button" class="rep-submit primary">Submit report</button>'
    + '</div>';
  // Seed chips from the user's previously-submitted report (if any).
  // Deep-clone so user edits don't mutate the shared server snapshot;
  // _serverReportedChips is rewritten on every /state load.
  const _seed = _serverReportedChips[entry.request_id || ''];
  form._corrections = (_seed && Array.isArray(_seed.corrections))
    ? JSON.parse(JSON.stringify(_seed.corrections))
    : [];
  // idx_end is the inclusive range end; single-word chips set
  // idx_end === idx so range-aware code doesn't need to special-case.
  form._entry = entry;
  _wordifyFinal(form, entry.final || '');
  // Render chips + struck-through + inline replacements immediately so
  // the user sees what they previously submitted. Skipped when there
  // are no seeded chips — empty render is a no-op but a needless
  // reflow.
  if (form._corrections.length) {
    _renderCorrections(form);
  }
  form.querySelector('.rep-cancel').addEventListener('click', () => {
    form.classList.remove('open');
  });
  form.querySelector('.rep-submit').addEventListener('click', () => {
    submitReport(form);
  });
  const removeBtn = form.querySelector('.rep-remove');
  // Visible only when the trace has already been reported. Lives in
  // the actions row so its placement matches Cancel/Submit.
  removeBtn.style.display =
    isReported(entry.request_id) ? 'inline-flex' : 'none';
  removeBtn.addEventListener('click', () => {
    removeReport(form);
  });
  return form;
}

function toggleReportForm(item, entry) {
  let form = item.querySelector('.trace-report-form');
  if (!form) {
    form = _buildReportForm(entry);
    item.appendChild(form);
  }
  form.classList.toggle('open');
  if (form.classList.contains('open')) {
    // Focus the first correction input if any chips are present (e.g.
    // rehydrated from a previous submission or freshly added). With
    // no chips yet the user's next action is clicking a word in the
    // strip, so leave focus where it is.
    const firstCorr = form.querySelector('.rep-correct');
    if (firstCorr) firstCorr.focus();
  }
}

// Range-aware chip helpers ------------------------------------------------
//
// A chip spans [chip.idx … chip.idx_end] inclusive. Single-word chips set
// idx_end === idx so every helper can iterate the range uniformly.
function _chipCovers(chip, idx) {
  return chip.idx <= idx && idx <= chip.idx_end;
}

function _selectWord(form, idx, on) {
  const span = form.querySelector(
    '.rep-final-words .word[data-idx="' + idx + '"]'
  );
  if (span) span.classList.toggle('selected', !!on);
}

// Inline replacement next to a struck-through word — mirrors the
// .diff-ins green look from /reports' renderDiff so the user sees
// the correction inline in the words strip, not just in the chip
// panel below.
function _setReplacementInline(form, chip) {
  if (typeof chip.idx !== 'number') return;
  const lastIdx = (typeof chip.idx_end === 'number') ? chip.idx_end : chip.idx;
  const anchor = form.querySelector(
    '.rep-final-words .word[data-idx="' + lastIdx + '"]'
  );
  if (!anchor) return;
  let existing = anchor.nextSibling;
  if (!existing || !existing.classList ||
      !existing.classList.contains('word-replacement')) {
    existing = null;
  }
  const text = (chip.correct || '').trim();
  if (!text) {
    if (existing) existing.parentNode.removeChild(existing);
    return;
  }
  if (!existing) {
    existing = document.createElement('span');
    existing.className = 'word-replacement';
    anchor.parentNode.insertBefore(existing, anchor.nextSibling);
  }
  existing.textContent = text;
}

function _clearReplacementInline(form, chip) {
  if (typeof chip.idx !== 'number') return;
  const lastIdx = (typeof chip.idx_end === 'number') ? chip.idx_end : chip.idx;
  const anchor = form.querySelector(
    '.rep-final-words .word[data-idx="' + lastIdx + '"]'
  );
  if (!anchor) return;
  const nxt = anchor.nextSibling;
  if (nxt && nxt.classList && nxt.classList.contains('word-replacement')) {
    nxt.parentNode.removeChild(nxt);
  }
}

function _spanText(form, a, b) {
  // Slice the original final string so inter-word punctuation/whitespace
  // is preserved (e.g. "A, B C" stays "A, B C", not "A B C").
  const pos = form._wordPositions || [];
  const finalText = form._finalText || '';
  if (!pos[a] || !pos[b]) {
    // Fallback: join textContent of the matching spans.
    const parts = [];
    for (let i = a; i <= b; i++) {
      const s = form.querySelector(
        '.rep-final-words .word[data-idx="' + i + '"]'
      );
      if (s) parts.push(s.textContent);
    }
    return parts.join(' ');
  }
  return finalText.slice(pos[a].start, pos[b].end);
}

function _focusLastInput(form) {
  const inputs = form.querySelectorAll('.rep-correction .rep-correct');
  const last = inputs[inputs.length - 1];
  if (last) last.focus();
}

function _removeChip(form, chipIdx) {
  const chip = form._corrections[chipIdx];
  if (!chip) return;
  for (let j = chip.idx; j <= chip.idx_end; j++) _selectWord(form, j, false);
  _clearReplacementInline(form, chip);
  form._corrections.splice(chipIdx, 1);
  _renderCorrections(form);
}

function _extendLastChip(form, idx) {
  // Grow the most-recent chip to include `idx`, then absorb any earlier
  // chip whose range overlaps the new one. The user-typed `correct`
  // value on the last chip is preserved; absorbed chips lose theirs.
  const lastI = form._corrections.length - 1;
  const last = form._corrections[lastI];
  if (!last) return;
  const newStart = Math.min(last.idx, idx);
  const newEnd   = Math.max(last.idx_end, idx);
  form._corrections = form._corrections.filter((c, i) => {
    if (i === lastI) return true;
    const overlaps = !(c.idx_end < newStart || c.idx > newEnd);
    if (overlaps) {
      for (let j = c.idx; j <= c.idx_end; j++) _selectWord(form, j, false);
    }
    return !overlaps;
  });
  last.idx = newStart;
  last.idx_end = newEnd;
  last.wrong = _spanText(form, newStart, newEnd);
  for (let j = newStart; j <= newEnd; j++) _selectWord(form, j, true);
  _renderCorrections(form);
  _focusLastInput(form);
}

function toggleCorrection(form, idx, word, shiftKey) {
  const last = form._corrections[form._corrections.length - 1];
  if (shiftKey && last) {
    _extendLastChip(form, idx);
    return;
  }
  // Click on a word that's already inside any chip removes that chip.
  // Splitting a multi-word chip in two is intentionally out of scope —
  // simpler to nuke and re-create.
  const existingI = form._corrections.findIndex(c => _chipCovers(c, idx));
  if (existingI >= 0) {
    _removeChip(form, existingI);
    return;
  }
  form._corrections.push({ wrong: word, correct: '', idx: idx, idx_end: idx });
  _selectWord(form, idx, true);
  _renderCorrections(form);
  _focusLastInput(form);
}

function _renderCorrections(form) {
  const box = form.querySelector('.rep-corrections');
  box.innerHTML = '';
  // Refresh inline replacement siblings for all current chips so
  // re-renders keep the strip in sync with the chip-panel state.
  form._corrections.forEach(c => _setReplacementInline(form, c));
  form._corrections.forEach((c, i) => {
    const chip = document.createElement('div');
    chip.className = 'rep-correction';
    chip.dataset.idx = String(c.idx);
    const w = document.createElement('span');
    w.className = 'rep-wrong';
    w.textContent = c.wrong;
    const a = document.createElement('span');
    a.className = 'rep-arrow';
    a.textContent = '→';
    const inp = document.createElement('input');
    inp.type = 'text';
    inp.className = 'rep-correct';
    inp.placeholder = 'correct word';
    inp.value = c.correct;
    inp.addEventListener('input', () => {
      c.correct = inp.value;
      _setReplacementInline(form, c);
    });
    inp.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        // Move focus to the next chip input if any, else to the
        // comment textarea so the user can add context. No more
        // chips + no comment yet = a deliberate done signal; we
        // don't auto-submit here, the user still clicks the button.
        const inputs = Array.from(form.querySelectorAll('.rep-correct'));
        const next = inputs[i + 1];
        if (next) next.focus();
        else {
          const c = form.querySelector('.rep-comment');
          if (c) c.focus();
        }
      }
    });
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'rep-correction-remove';
    x.textContent = '×';
    x.setAttribute('aria-label', 'remove correction');
    x.addEventListener('click', () => {
      // Find the chip's current index — _renderCorrections rebuilds
      // the list, so the closure-captured `i` may be stale.
      const idx = form._corrections.indexOf(c);
      if (idx >= 0) _removeChip(form, idx);
    });
    chip.appendChild(w);
    chip.appendChild(a);
    chip.appendChild(inp);
    chip.appendChild(x);
    box.appendChild(chip);
  });
}

async function submitReport(form) {
  const entry = form._entry || {};
  const comment = form.querySelector('.rep-comment').value.trim();
  const corrections = form._corrections
    .map(c => {
      const out = {
        wrong: c.wrong,
        correct: (c.correct || '').trim(),
        idx: c.idx,
      };
      if (typeof c.idx_end === 'number' && c.idx_end !== c.idx) {
        out.idx_end = c.idx_end;
      }
      return out;
    })
    .filter(c => c.correct.length > 0);
  if (!corrections.length && !comment) {
    showToast('Mark a wrong word or leave a comment.', 'err');
    return;
  }
  const submitBtn = form.querySelector('.rep-submit');
  submitBtn.disabled = true;
  try {
    const r = await api('POST', '/quick-config/reports/api/submit', {
      trace_ts: entry.ts || 0,
      request_id: entry.request_id || null,
      model: entry.model || '',
      raw: entry.raw || '',
      final: entry.final || '',
      steps: entry.steps || [],
      corrections: corrections,
      user_comment: comment,
    });
    if (!r.ok) {
      let msg = 'Report failed (' + r.status + ')';
      try { const j = await r.json(); if (j && j.detail) msg = j.detail; } catch (_) {}
      showToast(msg, 'err');
      return;
    }
    if (entry.request_id) {
      markReported(entry.request_id);
      _serverReportedSet.add(entry.request_id);
      // Snapshot the user's just-submitted chips so any re-render that
      // calls _buildReportForm(entry) again (e.g. a future SSE-driven
      // refresh) re-seeds correctly. Slight skew if the server merged
      // with pre-existing chips; the next /state load reconciles.
      _serverReportedChips[entry.request_id] = {
        corrections: JSON.parse(JSON.stringify(form._corrections || [])),
      };
    }
    // Update the badge + collapse the form. Keep the form's filled-in
    // values intact so the user can re-open it if they want to add more.
    const item = form.closest('.trace-item');
    if (item) {
      const badge = item.querySelector('.trace-reported-badge');
      if (badge) badge.style.display = 'inline-flex';
    }
    form.classList.remove('open');
    let body = {};
    try { body = await r.json(); } catch (_) {}
    const wasUpdated = !!body.was_updated;
    showToast(wasUpdated ? 'Report updated — thank you.' : 'Report sent — thank you.', 'ok');
  } catch (e) {
    showToast('Report failed: ' + e, 'err');
  } finally {
    submitBtn.disabled = false;
  }
}

async function removeReport(form) {
  // Withdraw the user's report for this trace. Reports are an
  // independent store: removing one does NOT touch any capture's chip
  // corrections (those are managed at /captures via batch-review).
  const entry = form._entry || {};
  const rid = entry.request_id || '';
  if (!rid) {
    showToast('No request_id for this trace — nothing to remove.', 'err');
    return;
  }
  if (!confirm('Remove your report for this trace?')) return;
  const removeBtn = form.querySelector('.rep-remove');
  removeBtn.disabled = true;
  try {
    const r = await api(
      'DELETE',
      '/quick-config/reports/api/by-request/' + encodeURIComponent(rid),
    );
    if (!r.ok) {
      let msg = 'Remove failed (' + r.status + ')';
      try { const j = await r.json(); if (j && j.detail) msg = j.detail; } catch (_) {}
      showToast(msg, 'err');
      return;
    }
    unmarkReported(rid);
    // Drop the seed entry so a fresh _buildReportForm call after this
    // remove starts empty (matches the server-side state). The form
    // we hold open here keeps form._corrections in-memory; that's
    // fine — closing the form below means the next open re-uses this
    // form node, which still has its in-memory chips. The user can
    // re-submit those by re-clicking Submit. They can also click
    // each chip's × to clear them.
    delete _serverReportedChips[rid];
    syncReportedBadges();
    form.classList.remove('open');
    showToast('Report removed.', 'ok');
  } catch (e) {
    showToast('Remove failed: ' + e, 'err');
  } finally {
    removeBtn.disabled = false;
  }
}

function pushTrace(entry) {
  const list = document.getElementById('recent-list');
  if (!list) return;
  // Skip rows already on screen (the SSE replay re-sends the freshest page
  // that the initial fetch already rendered) and, while a search is active,
  // live rows that don't match the filter (so they don't pollute the view).
  if (_seenTrace(entry)) return;
  if (!_matchesSearch(entry)) return;
  const empty = list.querySelector('.empty-recent');
  if (empty) empty.remove();
  list.insertBefore(renderTrace(entry), list.firstChild);
  // Server caps row count via RECENT_TRANSCRIPTIONS_MAX + TTL prune; no
  // client-side trim. The "Load older" cursor walks back through the
  // store from whatever the user has on screen.
  if (_oldestLoadedTs == null || (entry.ts && entry.ts < _oldestLoadedTs)) {
    // First SSE event after a fresh connect seeds the cursor; subsequent
    // pushes are newer than the current bottom and don't move it.
    if (_oldestLoadedTs == null) _oldestLoadedTs = entry.ts || null;
  }
  rebuildDatalist();
}

function appendTraceAtBottom(entry) {
  const list = document.getElementById('recent-list');
  if (!list) return;
  if (_seenTrace(entry)) return;
  // #recent-pager is a SIBLING of #recent-list (it sits below the list as a
  // separate footer bar), not a child — so insertBefore(node, pager) throws
  // NotFoundError. Older entries belong at the end of the list, so just
  // append them there (they land above the pager, which follows the list).
  list.appendChild(renderTrace(entry));
}

// Clear the list and (re)load the freshest page from the durable store for
// the current _searchQuery. Used for the initial population (so the panel no
// longer depends on the SSE replay succeeding), on every search change, and
// to catch up after a stream reconnect. Resets the dedupe set + pagination
// cursor so "Load older" walks back through the (optionally filtered) set.
async function reloadRecent() {
  const list = document.getElementById('recent-list');
  const btn = document.getElementById('btn-load-older');
  if (!list) return;
  _seenReqIds = new Set();
  _oldestLoadedTs = null;
  const q = _searchQuery;
  let j = null;
  try {
    // A search term is dictation text, so it goes in a POST body — a query
    // string lands in every access log and in browser history. No term =
    // plain GET, which stays cacheable/bookmarkable as before.
    const r = q
      ? await api('POST', '/quick-config/recent/search', { q: q })
      : await api('GET', '/quick-config/recent');
    if (r.ok) j = await r.json();
    else showToast('Recent load failed (' + r.status + ')', 'err');
  } catch (e) {
    showToast('Recent load failed: ' + e, 'err');
  }
  const rows = (j && j.recent) || [];
  // Server returns newest-first; appending in order leaves newest at the top.
  list.innerHTML = '';
  for (const entry of rows) appendTraceAtBottom(entry);
  _oldestLoadedTs = j && j.next_before_ts ? j.next_before_ts : null;
  if (!rows.length) {
    const d = document.createElement('div');
    d.className = 'empty-recent';
    d.textContent = q
      ? 'No transcriptions match your search.'
      : "No transcriptions yet — they'll appear here as you dictate.";
    list.appendChild(d);
  }
  if (btn) btn.style.display = _oldestLoadedTs ? '' : 'none';
  rebuildDatalist();
}

async function loadOlder() {
  if (_loadOlderBusy) return;
  const list = document.getElementById('recent-list');
  const btn = document.getElementById('btn-load-older');
  if (!list || !btn) return;
  if (_oldestLoadedTs == null) {
    // Bootstrap: derive cursor from the oldest .trace-item currently rendered.
    const items = list.querySelectorAll('.trace-item');
    if (items.length) {
      try {
        const data = JSON.parse(items[items.length - 1].dataset.entry || '{}');
        _oldestLoadedTs = data.ts || null;
      } catch (_) {}
    }
  }
  if (_oldestLoadedTs == null) { btn.style.display = 'none'; return; }
  _loadOlderBusy = true;
  btn.disabled = true;
  const prevLabel = btn.textContent;
  btn.textContent = 'Loading…';
  try {
    // Same split as reloadRecent: the term rides in the body when there is one.
    const r = _searchQuery
      ? await api('POST', '/quick-config/recent/search',
                  { q: _searchQuery, before_ts: _oldestLoadedTs })
      : await api('GET', '/quick-config/recent?before_ts='
                  + encodeURIComponent(_oldestLoadedTs));
    if (!r.ok) { showToast('Load older failed (' + r.status + ')', 'err'); return; }
    const j = await r.json();
    const rows = (j && j.recent) || [];
    for (const entry of rows) appendTraceAtBottom(entry);
    _oldestLoadedTs = j && j.next_before_ts ? j.next_before_ts : null;
    if (rows.length) rebuildDatalist();
    btn.style.display = _oldestLoadedTs ? '' : 'none';
  } catch (e) {
    showToast('Load older failed: ' + e, 'err');
  } finally {
    _loadOlderBusy = false;
    btn.disabled = false;
    btn.textContent = prevLabel;
  }
}

function rebuildDatalist() {
  const dl = document.getElementById('recent-words');
  if (!dl) return;
  // Iterate trace items newest → oldest. First-seen wins (= newest casing).
  const seen = new Map();
  document.querySelectorAll('.trace-item').forEach(item => {
    let data;
    try { data = JSON.parse(item.dataset.entry || '{}'); }
    catch (_) { return; }
    for (const t of data.tokens || []) {
      const k = String(t).toLowerCase();
      if (!seen.has(k)) seen.set(k, t);
    }
    for (const b of data.bigrams || []) {
      const k = String(b).toLowerCase();
      if (!seen.has(k)) seen.set(k, b);
    }
  });
  dl.innerHTML = '';
  // Cap sourced from the backend config (QUICK_CONFIG_WORD_SUGGESTIONS_MAX) so
  // the page + desktop client agree; _MAX_DATALIST is the fallback default.
  // 0 = suggestions disabled (datalist already cleared above).
  const cap = (typeof window.__wsm === 'number') ? window.__wsm : _MAX_DATALIST;
  if (cap <= 0) return;
  let n = 0;
  for (const v of seen.values()) {
    const opt = document.createElement('option');
    opt.value = v;
    dl.appendChild(opt);
    if (++n >= cap) break;
  }
}

function _setRecentLabel(text) {
  const el = document.querySelector('.recent-header .recent-label');
  if (el) el.textContent = text;
}

function startStream() {
  if (_es) { try { _es.close(); } catch (_) {} _es = null; }
  // EventSource sends the HttpOnly session cookie automatically (same-origin),
  // so the SSE auth dep resolves it without the legacy ?key= query param.
  _es = new EventSource('/quick-config/stream');
  _es.addEventListener('trace', (e) => {
    try { pushTrace(JSON.parse(e.data)); } catch (_) {}
  });
  _es.onopen = () => {
    // A successful (re)connect ends any recovery polling and restores the label.
    if (_recoveryTimer) { clearTimeout(_recoveryTimer); _recoveryTimer = null; }
    _setRecentLabel('persistent · paginated');
  };
  _es.onerror = () => {
    // EventSource does NOT auto-reconnect after an HTTP error (e.g. an
    // intermittent 401 where the browser didn't attach the session cookie to
    // the SSE handshake) — it fails the connection permanently. Mirror the
    // /stats recovery: poll a cheap cookie-authed endpoint until it 200s,
    // then catch up via reloadRecent() (deduped) and reopen the stream.
    // Back off on repeated failures (3s → ×1.7 → cap 30s).
    _setRecentLabel('reconnecting…');
    if (_recoveryTimer) return;
    let delay = 3000;
    const probe = async () => {
      try {
        const r = await api('GET', '/quick-config/recent?limit=1');
        if (r.ok) {
          clearTimeout(_recoveryTimer);
          _recoveryTimer = null;
          await reloadRecent();
          startStream();
          return;
        }
      } catch (_) { /* keep polling */ }
      delay = Math.min(delay * 1.7, 30000);
      _recoveryTimer = setTimeout(probe, delay);
    };
    _recoveryTimer = setTimeout(probe, delay);
  };
}

// --- Field-level diff helpers (used for both build-patch and conflict re-merge) ---
//
// For each editable field we represent the user's change as a typed diff
// vs. their original baseline (initialRules at load time):
//   dict   (cb:map.map)                    → {kind:'dict', added, changed, removed[]}
//   list   (cb:lowercase-wordlist.wordlist)→ {kind:'list', added[], removed[]}
//   scalar (everything else)               → {kind:'scalar', value}
// On a conflict we re-apply the diff to the FRESH server state, so the
// other user's entries survive — only the cells the user actually
// touched get overlaid.
const _DICT_FIELDS = new Set(['map']);
const _LIST_FIELDS = new Set(['wordlist']);

function _stringify(x) {
  // Stable order for objects so two equal-content objects compare equal.
  if (x !== null && typeof x === 'object' && !Array.isArray(x)) {
    const out = {};
    for (const k of Object.keys(x).sort()) out[k] = x[k];
    return JSON.stringify(out);
  }
  return JSON.stringify(x);
}

function _fieldDiff(baseline, edited, kind) {
  if (kind === 'dict') {
    const base = baseline || {};
    const cur = edited || {};
    const added = {};
    const changed = {};
    const removed = [];
    for (const k of Object.keys(cur)) {
      if (!(k in base)) added[k] = cur[k];
      else if (_stringify(base[k]) !== _stringify(cur[k])) changed[k] = cur[k];
    }
    for (const k of Object.keys(base)) if (!(k in cur)) removed.push(k);
    if (!Object.keys(added).length && !Object.keys(changed).length && !removed.length) {
      return null;
    }
    return { kind: 'dict', added, changed, removed };
  }
  if (kind === 'list') {
    const base = (baseline || []).map(_stringify);
    const baseSet = new Set(base);
    const cur = (edited || []).map(_stringify);
    const curSet = new Set(cur);
    const added = (edited || []).filter(x => !baseSet.has(_stringify(x)));
    const removed = (baseline || []).filter(x => !curSet.has(_stringify(x)));
    if (!added.length && !removed.length) return null;
    return { kind: 'list', added, removed };
  }
  // scalar
  if (_stringify(baseline) === _stringify(edited)) return null;
  return { kind: 'scalar', value: edited };
}

function _applyFieldDiff(serverValue, diff) {
  if (!diff) return serverValue;
  if (diff.kind === 'dict') {
    const out = { ...(serverValue || {}) };
    for (const k of diff.removed) delete out[k];
    Object.assign(out, diff.added, diff.changed);
    return out;
  }
  if (diff.kind === 'list') {
    const removedKeys = new Set(diff.removed.map(_stringify));
    const out = (serverValue || []).filter(x => !removedKeys.has(_stringify(x)));
    for (const item of diff.added) out.push(item);
    return out;
  }
  return diff.value;
}

function _diffKindFor(field) {
  if (_DICT_FIELDS.has(field)) return 'dict';
  if (_LIST_FIELDS.has(field)) return 'list';
  return 'scalar';
}

// Build the rules_patch dict + per-rule fingerprints by diffing live vs initial.
// The fingerprint is the `_fp` field the server stamped on each rule at /state
// load — sending it back lets the server detect that another writer changed
// this rule since we loaded, instead of silently overwriting their edit.
function buildPatchAndFingerprints() {
  const patch = {};
  const fingerprints = {};
  const byName = new Map(initialRules.map(r => [r.name, r]));
  for (const slug of dirty) {
    const live = liveRules.find(r => r.name === slug);
    const orig = byName.get(slug);
    if (!live || !orig) continue;
    const allowed = _ALLOWED_BY_TYPE[live.type] || new Set(['enabled']);
    const delta = {};
    for (const k of allowed) {
      const a = live[k], b = orig[k];
      if (JSON.stringify(a) !== JSON.stringify(b)) {
        delta[k] = a;
      }
    }
    if (Object.keys(delta).length) {
      patch[slug] = delta;
      if (orig._fp) fingerprints[slug] = orig._fp;
    }
  }
  return { patch, fingerprints };
}

const _ALLOWED_BY_TYPE = {
  'regex-list':                  new Set(['enabled', 'entries']),
  'callback:map':                new Set(['enabled', 'map']),
  'callback:lowercase-wordlist': new Set(['enabled', 'pattern', 'wordlist']),
  'callback:dedup':              new Set(['enabled', 'pattern']),
  'callback:upper':              new Set(['enabled', 'pattern']),
};

async function load() {
  const r = await ensureToken();
  if (!r) { setStatus('not authenticated'); return; }
  if (!r.ok) {
    setStatus('load failed (' + r.status + ')');
    showToast('load failed: ' + r.status, 'err');
    return;
  }
  const j = await r.json();
  initialRules = j.rules || [];
  // Newest-cb:map-entries-shown threshold, sourced from the backend config
  // (QUICK_CONFIG_MAP_COLLAPSE_AFTER) so the page + desktop client agree.
  window.__mca = (typeof j.map_collapse_after === 'number') ? j.map_collapse_after : 15;
  // Recent-word autocomplete cap, sourced from the backend config
  // (QUICK_CONFIG_WORD_SUGGESTIONS_MAX) so the page + desktop client agree.
  window.__wsm = (typeof j.word_suggestions_max === 'number') ? j.word_suggestions_max : 200;
  // Schema cap on map entries — feeds the "n / cap" readout in the map
  // editor (web_common) so a full dictionary is visible before Save bounces.
  window.__mme = (typeof j.map_max_entries === 'number') ? j.map_max_entries : 0;
  liveRules = JSON.parse(JSON.stringify(initialRules));
  dirty = new Set();
  // role: "admin" reveals .admin-only nav links + sev pills. Non-admin keys
  // sessions never get the class so admin chrome stays hidden.
  if (j.role === 'admin') document.body.classList.add('role-admin');
  else document.body.classList.remove('role-admin');
  // Server-authoritative "reported" set + chip-corrections map. The
  // ids feed the badge visibility; the chips feed form rehydration on
  // form-open. Badges in already-rendered .trace-item nodes update
  // in-place via syncReportedBadges; chip rehydration is lazy (only
  // applied when the user opens a report form).
  _serverReportedChips = j.reported_chips || {};
  _serverReportedSet = new Set(Object.keys(_serverReportedChips));
  syncReportedBadges();
  renderCards();
  updateButtons();
  loadMyUsage();   // fire-and-forget; populates the subbar self-usage banner
}

// --- personal self-usage banner (subbar middle) ---------------------------
function _uCount(n) {
  n = Number(n || 0);
  if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(Math.round(n));
}
function _uDur(sec) {
  sec = Number(sec || 0);
  if (sec < 60) return sec.toFixed(sec < 10 ? 1 : 0) + 's';
  if (sec < 3600) return (sec / 60).toFixed(1).replace(/\.0$/, '') + 'm';
  if (sec < 86400) return (sec / 3600).toFixed(1).replace(/\.0$/, '') + 'h';
  return (sec / 86400).toFixed(1).replace(/\.0$/, '') + 'd';
}
function _uEsc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
async function loadMyUsage() {
  const el = document.getElementById('qc-usage');
  if (!el) return;
  try {
    // Pass this browser's local midnight (epoch seconds) so "today" resets at
    // the viewer's own midnight, not the server's.
    var _mid = new Date(); _mid.setHours(0, 0, 0, 0);
    var tzMidnight = Math.floor(_mid.getTime() / 1000);
    const r = await api('GET', '/quick-config/usage?tz_midnight=' + tzMidnight);
    if (!r.ok) { el.innerHTML = ''; return; }
    const j = await r.json();
    const name = '<span class="u-name">' + _uEsc(j.username || 'there') + '</span>';
    const today = j.today || {}, total = j.total || {};
    if (!total.requests) {
      el.innerHTML = 'Hi ' + name + '<span class="u-sep">·</span>no transcriptions yet';
      return;
    }
    const seg = (cls, n) =>
      '<span class="u-num ' + cls + '">' + _uCount(n.words) + '</span> words '
      + '<span class="u-num ' + cls + '">' + _uDur(n.audio_s) + '</span> audio';
    el.innerHTML = 'Hi ' + name
      + '<span class="u-sep">·</span>today ' + seg('u-today', today)
      + '<span class="u-sep">·</span>total ' + seg('u-total', total);
  } catch (_) {
    el.innerHTML = '';
  }
}

async function doSave() {
  const { patch, fingerprints } = buildPatchAndFingerprints();
  if (!Object.keys(patch).length) return;
  setStatus('saving…');
  const r = await api('POST', '/quick-config/state',
                      { rules_patch: patch, fingerprints });
  if (r.status === 422) {
    // Validation error. Errors naming rules this user can't see arrive
    // redacted ('<hidden rule>') and stay behind the generic "contact admin"
    // message. But an error on one of the USER'S OWN rules — their dictionary
    // hitting the server's entry cap, a key the schema refuses — is legible
    // and actionable, so surface it instead of blaming the admin pipeline.
    let errs = [];
    try { const j = await r.json(); errs = (j && j.errors) || []; } catch (_) {}
    let msg = '';
    for (const e of errs) {
      const loc = String((e && e.loc) || ''), m = String((e && e.msg) || '');
      if (m.indexOf('<hidden rule>') !== -1) continue;
      const cap = /at most (\d+) items/.exec(m);
      msg = (cap && loc.indexOf('callback:map') !== -1)
        ? 'this dictionary is full (server cap: ' + cap[1]
          + ' entries) — delete some entries before adding new ones'
        : (loc ? loc + ': ' : '') + m.slice(0, 200);
      break;
    }
    showToast(msg || 'admin pipeline has a validation error — please contact admin', 'err');
    setStatus('save failed');
    return;
  }
  if (!r.ok) {
    let msg = 'save failed (' + r.status + ')';
    try { const j = await r.json(); if (j && j.detail) msg = j.detail; } catch (_) {}
    showToast(msg, 'err');
    setStatus('save failed');
    return;
  }
  const result = await r.json();
  const conflicts = result.conflicts || [];
  const saved = result.saved || [];
  if (conflicts.length) {
    // Another writer changed one or more rules between our load and save.
    // We preserve the user's intent by computing their per-field DIFF
    // against the baseline they saw at load time, refetching the fresh
    // server state, and re-applying that diff on top. For cb:map this
    // means only the keys the user added/changed/removed get overlaid —
    // the other user's keys survive. For scalars (regex pattern,
    // replacement, enabled) the user's new value still wins (no way to
    // merge two scalar edits), but the user reviews before re-saving.
    const conflictSlugs = conflicts.map(c => c.slug);

    // Snapshot diffs BEFORE await load() — load() resets initialRules.
    const diffsBySlug = new Map();
    for (const slug of conflictSlugs) {
      const live = liveRules.find(r => r.name === slug);
      const orig = initialRules.find(r => r.name === slug);
      if (!live || !orig) continue;
      const allowed = _ALLOWED_BY_TYPE[live.type] || new Set(['enabled']);
      const fieldDiffs = {};
      for (const k of allowed) {
        const d = _fieldDiff(orig[k], live[k], _diffKindFor(k));
        if (d) fieldDiffs[k] = d;
      }
      if (Object.keys(fieldDiffs).length) diffsBySlug.set(slug, fieldDiffs);
    }

    await load();   // refetches; clears dirty; renders fresh server state

    // Re-apply each user diff onto the fresh server state. dirty gets
    // re-populated so the next Save sends an updated patch with the
    // FRESH fingerprint (which now matches the server) plus the
    // user's intent overlaid.
    let merged = 0;
    for (const [slug, fieldDiffs] of diffsBySlug) {
      const idx = liveRules.findIndex(r => r.name === slug);
      if (idx < 0) continue;  // rule was deleted / un-exposed server-side
      for (const [field, diff] of Object.entries(fieldDiffs)) {
        liveRules[idx][field] = _applyFieldDiff(liveRules[idx][field], diff);
      }
      dirty.add(slug);
      merged++;
    }

    if (merged) {
      renderCards();
      updateButtons();
      const prefix = saved.length ? 'Saved ' + saved.length + '. ' : '';
      showToast(prefix + 'Auto-merged your changes for '
                + conflictSlugs.join(', ')
                + ' on top of the latest version — review and Save again.',
                'err');
    } else {
      // The conflicted rule was deleted server-side or no diff survived.
      const prefix = saved.length ? 'Saved ' + saved.length + '. ' : '';
      showToast(prefix + 'Conflict on ' + conflictSlugs.join(', ')
                + ' could not be auto-merged (rule changed or removed).',
                'err');
    }
    return;
  }
  showToast(saved.length
    ? ('saved ' + saved.length + ' rule' + (saved.length === 1 ? '' : 's'))
    : 'nothing to save', 'ok');
  if (result.captures_count > 0) {
    // Auto-trigger the reapply job silently — admins shouldn't have to
    // click twice. The header strip shows progress; there's no manual
    // re-apply button anymore.
    startReapplyJobSilent(result.captures_count);
  }
  await load();
}


// --- Silent reapply strip (auto-triggered after Save) ---
//
// Same /reapply-rules POST + status-polling as the manual modal, but
// rendered as an inline header strip so the admin keeps editing.
// Fades out 3s after completion; sticks (red-tinted) on error so the
// failure isn't missed.
let _stripPoll = null;
let _stripFadeTimer = null;

function _showStrip(n) {
  const strip = document.getElementById('reapply-strip');
  strip.hidden = false;
  strip.classList.remove('fade-out');
  strip.classList.remove('err');
  strip.querySelector('.r-label').textContent =
    'Re-applying to ' + (n || '?') + ' captures…';
  strip.querySelector('.r-fill').style.width = '0%';
  strip.querySelector('.r-stats').textContent = '0 / ' + (n || 0);
  if (_stripFadeTimer) { clearTimeout(_stripFadeTimer); _stripFadeTimer = null; }
}
function _hideStripSoon() {
  const strip = document.getElementById('reapply-strip');
  strip.classList.add('fade-out');
  if (_stripFadeTimer) clearTimeout(_stripFadeTimer);
  _stripFadeTimer = setTimeout(() => {
    strip.hidden = true;
    strip.classList.remove('fade-out');
  }, 3000);
}
async function _pollStripOnce() {
  let r;
  try { r = await api('GET', '/quick-config/reapply-rules/status'); }
  catch (_) { return null; }
  if (!r.ok) return null;
  const s = await r.json();
  const strip = document.getElementById('reapply-strip');
  const fill = strip.querySelector('.r-fill');
  const stats = strip.querySelector('.r-stats');
  const label = strip.querySelector('.r-label');
  const pct = s.total > 0
    ? Math.round((s.processed / s.total) * 100)
    : (s.status === 'done' ? 100 : 0);
  fill.style.width = pct + '%';
  stats.textContent = (s.processed || 0) + ' / ' + (s.total || 0)
    + ' · ' + (s.captures_updated || 0) + ' updated · '
    + (s.groups_updated || 0) + ' groups';
  if (s.status === 'running') {
    label.textContent = 'Re-applying rules…';
  } else if (s.status === 'done') {
    label.textContent = 'Re-applied ' + (s.captures_updated || 0)
      + ' captures · ' + (s.groups_updated || 0) + ' groups';
    if (_stripPoll) { clearInterval(_stripPoll); _stripPoll = null; }
    _hideStripSoon();
  } else if (s.status === 'error') {
    label.textContent = 'Re-apply failed: ' + (s.error || 'unknown');
    strip.classList.add('err');
    if (_stripPoll) { clearInterval(_stripPoll); _stripPoll = null; }
    // Don't auto-hide on error — the user needs to see it.
  }
  return s.status || null;
}
async function startReapplyJobSilent(n) {
  _showStrip(n);
  const r = await api('POST', '/quick-config/reapply-rules');
  if (!r.ok) {
    const strip = document.getElementById('reapply-strip');
    strip.classList.add('err');
    strip.querySelector('.r-label').textContent =
      'Re-apply failed: HTTP ' + r.status;
    return;
  }
  // Reset any prior interval before scheduling — guards against a double
  // click triggering two concurrent polls.
  if (_stripPoll) { clearInterval(_stripPoll); _stripPoll = null; }
  const status = await _pollStripOnce();
  // Only schedule the recurring poll while the job is actually running;
  // status==='done'|'error' already cleared the strip via _pollStripOnce.
  if (status === 'running') {
    _stripPoll = setInterval(_pollStripOnce, 1500);
  }
}

function doDiscard() {
  liveRules = JSON.parse(JSON.stringify(initialRules));
  dirty = new Set();
  renderCards();
  updateButtons();
}

document.getElementById('save-btn').addEventListener('click', doSave);
document.getElementById('discard-btn').addEventListener('click', doDiscard);
const _loadOlderBtn = document.getElementById('btn-load-older');
if (_loadOlderBtn) _loadOlderBtn.addEventListener('click', loadOlder);

// Recent-transcriptions search: debounced server-side substring filter over
// raw/final. Mirrors the /captures search (150 ms debounce); whole-DB so
// matches are found regardless of how far the list has been paginated.
const _recentSearchInput = document.getElementById('recent-search');
let _recentSearchTimer = null;
if (_recentSearchInput) {
  _recentSearchInput.addEventListener('input', () => {
    if (_recentSearchTimer) clearTimeout(_recentSearchTimer);
    _recentSearchTimer = setTimeout(() => {
      _recentSearchTimer = null;
      const next = (_recentSearchInput.value || '').trim();
      if (next === _searchQuery) return;
      _searchQuery = next;
      reloadRecent();
    }, 150);
  });
}

load().then(() => {
  // Populate the recent list via a plain fetch FIRST so the panel no longer
  // depends on the SSE replay succeeding (a single failed /stream handshake
  // used to leave it empty). reloadRecent() also seeds the dedupe set + the
  // "Load older" cursor before the stream opens, so the on-connect replay is
  // skipped as already-seen rather than duplicated.
  return reloadRecent();
}).then(() => {
  startStream();
});
})();
</script>

{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
{{TIME_HELPERS_JS}}

<script>
// Must run AFTER TIME_HELPERS_JS defines timeTick. Default [data-ts] selector
// refreshes both the cb:map date cells created by later renderCards() calls and
// the Recent-transcriptions .trace-ts nodes; all carry static fmtWhen() text at
// creation, this just ages the relative suffix in place.
timeTick();
</script>

</body>
</html>
"""
